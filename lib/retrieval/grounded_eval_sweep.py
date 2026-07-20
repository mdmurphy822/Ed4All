"""Flag A/B sweep driver for the grounded-answer eval (E1).

Seven ``ED4ALL_ANSWER_*`` retrieval/answer-augmentation flags ship with ZERO
measured evidence — they were never run through the gold set. This driver closes
that gap: it runs the SAME gold set once per config arm (a baseline arm with all
seven off, plus one arm per flag flipped on), then diffs every arm against the
baseline via the existing :mod:`lib.retrieval.grounded_eval_diff` machinery and
rolls the per-metric deltas into a matrix summary — the adopt/reject evidence
each flag lacks.

The seven swept flags (arm name → env var):

  * ``decompose``               → ``ED4ALL_ANSWER_DECOMPOSE``
  * ``hyde``                    → ``ED4ALL_ANSWER_HYDE``
  * ``graph_expand``            → ``ED4ALL_ANSWER_GRAPH_EXPAND``
  * ``hedge_tier``              → ``ED4ALL_ANSWER_HEDGE_TIER``
  * ``intent_route``            → ``ED4ALL_ANSWER_INTENT_ROUTE``
  * ``rerank``                  → ``ED4ALL_RERANK_PROVIDER``
  * ``completeness_reretrieve`` → ``ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE``

Each arm's env overlay sets EVERY swept flag to a known baseline (unset) and then
turns the arm's own flag on, so an ambient flag in the operator's shell can never
contaminate a measurement. The environment is snapshotted and RESTORED after
every arm (and after the whole sweep) — the driver is a strict no-op on
``os.environ`` once it returns. The eval's own ``flag_config`` stamp records the
resolved config each arm actually ran under.

Offline-testable: ``answer_fn`` is injected straight through to
``run_grounded_eval`` (no real pipeline, no model, no network). A spy ``answer_fn``
can read ``os.environ`` to confirm the arm's flag was live during its run.

This driver makes NO LLM calls of its own (it only sets env + delegates), so it
wires no ``DecisionCapture`` — the decision-capture contract is scoped to LLM
call sites, and every LLM call happens inside ``run_grounded_eval`` / the injected
pipeline, which own their own captures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from lib.retrieval.grounded_eval import run_grounded_eval
from lib.retrieval.grounded_eval_diff import diff_reports

#: Additive sweep-summary schema; tracked independently of EVAL_SCHEMA_VERSION.
SWEEP_SCHEMA_VERSION = "1.0"

#: The baseline arm name (all swept flags off).
BASELINE_ARM = "baseline"


@dataclass(frozen=True)
class SweepFlag:
    """One swept flag: its env var and the value that turns it on."""

    env: str
    on_value: str = "1"


#: The seven unevaluated answer-path flags, keyed by canonical arm name. Order is
#: the sweep order (deterministic). ``rerank`` maps to a PROVIDER-name env var
#: (any non-empty string enables the hook; an unknown provider fails OPEN in the
#: pipeline), so its on-value is a placeholder the operator overrides with a real
#: seat when actually exercising reranking.
SWEEP_FLAGS: Dict[str, SweepFlag] = {
    "decompose": SweepFlag("ED4ALL_ANSWER_DECOMPOSE"),
    "hyde": SweepFlag("ED4ALL_ANSWER_HYDE"),
    "graph_expand": SweepFlag("ED4ALL_ANSWER_GRAPH_EXPAND"),
    "hedge_tier": SweepFlag("ED4ALL_ANSWER_HEDGE_TIER"),
    "intent_route": SweepFlag("ED4ALL_ANSWER_INTENT_ROUTE"),
    "rerank": SweepFlag("ED4ALL_RERANK_PROVIDER"),
    "completeness_reretrieve": SweepFlag("ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE"),
}

#: All env vars the sweep manages (cleared to baseline before every arm).
_MANAGED_ENV: Tuple[str, ...] = tuple(f.env for f in SWEEP_FLAGS.values())

#: Pinned headline metrics surfaced in the matrix summary (label → dotted path).
#: Mirrors the pinned specs in grounded_eval_diff so the matrix reads the same
#: numbers the regression gate does.
_MATRIX_METRICS: Tuple[Tuple[str, str], ...] = (
    ("answer_rate", "headline.answer_rate"),
    ("citation_precision", "headline.citation_precision"),
    ("citation_precision_primary", "headline.citation_precision_primary"),
    ("groundedness_rate_mean", "headline.groundedness_rate_mean"),
    ("unsupported_claim_rate", "headline.unsupported_claim_rate"),
    ("refusal_recall", "headline.refusal.refusal_recall"),
    ("refusal_precision", "headline.refusal.refusal_precision"),
)


@contextmanager
def _env_overlay(overlay: Dict[str, Optional[str]]) -> Iterator[None]:
    """Apply an env overlay (``None`` value ⇒ unset), restore on exit.

    Snapshots exactly the keys in ``overlay`` and restores their prior values
    (including prior-absence) afterwards, so the driver leaves ``os.environ``
    byte-identical to how it found it.
    """
    saved: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in overlay}
    try:
        for key, val in overlay.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _arm_overlay(
    arm: str, on_value_overrides: Optional[Dict[str, str]]
) -> Dict[str, Optional[str]]:
    """Env overlay for one arm: every swept flag off, then this arm's flag on."""
    overlay: Dict[str, Optional[str]] = {env: None for env in _MANAGED_ENV}
    if arm == BASELINE_ARM:
        return overlay
    flag = SWEEP_FLAGS[arm]
    value = (on_value_overrides or {}).get(arm, flag.on_value)
    overlay[flag.env] = value
    return overlay


def _dotted_get(report: Dict[str, Any], path: str) -> Any:
    node: Any = report
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def resolve_arms(arms: Optional[Sequence[str]]) -> List[str]:
    """Resolve the arm list: default = baseline + all seven flags.

    An explicit list is validated against the known arm names (an unknown arm is
    a loud ``ValueError`` — never a silently-dropped no-op). The baseline arm is
    always included and always first (the diff anchor).
    """
    if arms is None:
        return [BASELINE_ARM, *SWEEP_FLAGS.keys()]
    unknown = [a for a in arms if a != BASELINE_ARM and a not in SWEEP_FLAGS]
    if unknown:
        raise ValueError(
            f"unknown sweep arm(s) {unknown!r}; valid arms: "
            f"{[BASELINE_ARM, *SWEEP_FLAGS.keys()]!r}"
        )
    ordered = [BASELINE_ARM] + [a for a in SWEEP_FLAGS if a in arms]
    return ordered


def run_flag_sweep(
    repo_root: Path,
    course_slug: str,
    *,
    arms: Optional[Sequence[str]] = None,
    engine: str = "semantic",
    answer_fn: Optional[Any] = None,
    tolerance_pp: float = 5.0,
    on_value_overrides: Optional[Dict[str, str]] = None,
    write: bool = False,
    output_dir: Optional[Path] = None,
    **eval_kwargs: Any,
) -> Dict[str, Any]:
    """Run the gold set once per config arm, diff each arm against baseline.

    For each arm: overlay the env (all swept flags off, this arm's flag on),
    run :func:`run_grounded_eval`, then restore the env. The baseline arm is the
    diff anchor; every other arm carries a ``diff_vs_baseline`` block plus its
    delta contribution to the matrix summary.

    ``answer_fn`` is passed straight through (offline tests). ``**eval_kwargs``
    forward to ``run_grounded_eval`` (e.g. ``with_groundedness``, ``limit``,
    ``refusal_probes_path``). When ``write`` is True each arm's report is written
    under ``output_dir`` (default: the course ``retrieval_eval/`` dir) with an
    arm-tagged filename.

    Returns the sweep summary dict (arms + matrix + regression/improvement
    lists). The environment is byte-identical after the call.
    """
    repo_root = Path(repo_root)
    arm_names = resolve_arms(arms)

    arm_reports: Dict[str, Dict[str, Any]] = {}
    arm_records: Dict[str, Dict[str, Any]] = {}

    for arm in arm_names:
        overlay = _arm_overlay(arm, on_value_overrides)
        out_path: Optional[Path] = None
        review_path: Optional[Path] = None
        if write:
            base_dir = (
                Path(output_dir)
                if output_dir is not None
                else repo_root / "LibV2" / "courses" / course_slug
                / "retrieval_eval"
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / f"grounded_answer_eval_sweep_{arm}.json"
            # Per-arm review sample so arms never clobber one shared file.
            review_path = base_dir / f"groundedness_review_sample_sweep_{arm}.json"
        with _env_overlay(overlay):
            report = run_grounded_eval(
                repo_root,
                course_slug,
                engine=engine,
                answer_fn=answer_fn,
                write=write,
                output_path=out_path,
                review_sample_path=review_path,
                **eval_kwargs,
            )
        arm_reports[arm] = report
        arm_records[arm] = {
            "flags_on": {
                k: v for k, v in overlay.items() if v is not None
            },
            "report_path": (
                report.get("_written", {}).get("report_path")
                if isinstance(report.get("_written"), dict)
                else None
            ),
        }

    baseline_report = arm_reports[BASELINE_ARM]

    # Matrix: per pinned metric, the per-arm delta vs baseline (None-safe).
    matrix: Dict[str, Dict[str, Any]] = {}
    for label, path in _MATRIX_METRICS:
        base_val = _as_number(_dotted_get(baseline_report, path))
        row: Dict[str, Any] = {"baseline": base_val, "arms": {}}
        for arm in arm_names:
            if arm == BASELINE_ARM:
                continue
            arm_val = _as_number(_dotted_get(arm_reports[arm], path))
            delta = (
                (arm_val - base_val)
                if (arm_val is not None and base_val is not None)
                else None
            )
            row["arms"][arm] = {
                "value": arm_val,
                "delta": delta,
                "delta_pp": (delta * 100.0) if delta is not None else None,
            }
        matrix[label] = row

    # Per-arm full diff vs baseline (reuses the regression-diff machinery).
    regressions: List[Dict[str, str]] = []
    improvements: List[Dict[str, str]] = []
    for arm in arm_names:
        if arm == BASELINE_ARM:
            arm_records[arm]["diff_vs_baseline"] = None
            continue
        diff = diff_reports(
            baseline_report, arm_reports[arm], tolerance_pp=tolerance_pp
        )
        diff_dict = diff.to_dict()
        arm_records[arm]["diff_vs_baseline"] = diff_dict
        for r in diff.regressions:
            regressions.append({"arm": arm, "metric": r.label})
        for r in diff.metrics:
            if r.status == "improved":
                improvements.append({"arm": arm, "metric": r.label})

    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "course_slug": course_slug,
        "engine": engine,
        "baseline_arm": BASELINE_ARM,
        "tolerance_pp": tolerance_pp,
        "arm_order": arm_names,
        "arms": arm_records,
        "matrix": matrix,
        "regressions": regressions,
        "improvements": improvements,
        "_note": (
            "Flag A/B sweep (E1): one arm per unevaluated ED4ALL_ANSWER_* flag, "
            "each diffed against the all-off baseline. delta_pp is the arm's "
            "pinned-metric change vs baseline; a regression beyond tolerance_pp "
            "is surfaced for adopt/reject. Reranker arm sets a placeholder "
            "provider unless on_value_overrides pins a real seat."
        ),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_matrix(summary: Dict[str, Any]) -> str:
    """Human-readable matrix: rows = metrics, columns = arm deltas (pp)."""
    arms = [a for a in summary.get("arm_order", []) if a != BASELINE_ARM]
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(
        f"flag sweep: {summary.get('course_slug')} "
        f"(engine {summary.get('engine')}, tol {summary.get('tolerance_pp')} pp)"
    )
    lines.append("=" * 78)
    header = f"  {'metric':<28}{'base':>8}" + "".join(f"{a[:10]:>11}" for a in arms)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for label, row in summary.get("matrix", {}).items():
        base = row.get("baseline")
        base_s = "  n/a " if base is None else f"{base:6.4f}"
        cells = []
        for a in arms:
            d = (row.get("arms", {}).get(a) or {}).get("delta_pp")
            cells.append("    n/a" if d is None else f"{d:+7.2f}")
        lines.append(f"  {label:<28}{base_s:>8}" + "".join(f"{c:>11}" for c in cells))
    lines.append("")
    regs = summary.get("regressions", [])
    if regs:
        lines.append("REGRESSIONS (arm: metric):")
        for r in regs:
            lines.append(f"  {r['arm']}: {r['metric']}")
    else:
        lines.append("no pinned regressions beyond tolerance on any arm")
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.grounded_eval_sweep",
        description=(
            "Flag A/B sweep: run a course's gold set once per unevaluated "
            "ED4ALL_ANSWER_* flag (+ an all-off baseline) and emit per-arm "
            "reports + a matrix of per-metric deltas vs baseline."
        ),
    )
    parser.add_argument("--course", required=True, help="course slug")
    parser.add_argument("--engine", default="semantic")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "comma-separated arm subset (default: all). Valid: "
            + ",".join([BASELINE_ARM, *SWEEP_FLAGS.keys()])
        ),
    )
    parser.add_argument("--tolerance-pp", type=float, default=5.0)
    parser.add_argument(
        "--no-groundedness", action="store_true",
        help="skip the per-answer NLI groundedness pass in every arm",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="repo root (default: auto-detect from this file)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="emit the machine-readable sweep summary instead of the matrix",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    args = _build_arg_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root)
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    arms = (
        [a.strip() for a in args.arms.split(",") if a.strip()]
        if args.arms
        else None
    )
    summary = run_flag_sweep(
        repo_root,
        args.course,
        arms=arms,
        engine=args.engine,
        limit=args.limit,
        tolerance_pp=args.tolerance_pp,
        with_groundedness=not args.no_groundedness,
        write=True,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_matrix(summary))
    return 1 if summary["regressions"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SWEEP_SCHEMA_VERSION",
    "BASELINE_ARM",
    "SweepFlag",
    "SWEEP_FLAGS",
    "resolve_arms",
    "run_flag_sweep",
    "render_matrix",
    "main",
]
