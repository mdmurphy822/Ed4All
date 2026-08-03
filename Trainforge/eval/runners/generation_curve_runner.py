"""C0..C5 generation-quality curve runner (W5 §6).

Models the arm-comparison pattern of ``Trainforge/eval/runners/ablation_runner.py`` +
``headline_delta.py``: a list of technique modes → per-arm
``generation_quality_eval`` report → a per-technique delta table answering "how
much does each lever buy on 7B." The ``ceiling`` block reports the C5 numbers
(the 7B-alone ceiling) and reserves a Spark ``C5-70b`` row so the
70B-single-sample-vs-7B-best-of-N comparison is a one-row add later.

For each mode the runner: ``apply_mode_to_env(resolve_technique_mode(mode), env)``
→ ``synthesize_fn(course_slug)`` (re-synthesize) OR reuse a pre-synthesized
per-mode artifact pool (re-select) → ``run_generation_quality_eval(...,
technique_config=mode)`` → collect the headline. ``arm_basis`` records
``resynthesized`` vs ``reselected`` so each delta is honestly attributed.

CI never runs the real-model curve (the report-shape test runs it on the fixture
with fakes). Boundary: this runner consumes W5's own harness + technique-mode
resolver; it does NOT author synthesis (W2 owns ``synthesize_fn``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: Curve report schema version (§6.2).
CURVE_SCHEMA_VERSION = "1.0"

#: Default mode sequence (lowest→highest lever).
DEFAULT_MODES = ["C0", "C1", "C2", "C3", "C4", "C5"]

#: Per-edge technique label (the lever each delta isolates).
_TECHNIQUE_LABELS = {
    ("C0", "C1"): "chunk_scoped",
    ("C1", "C2"): "free_text_envelope",
    ("C2", "C3"): "self_verify",
    ("C3", "C4"): "refine",
    ("C4", "C5"): "best_of_n_nli",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _delta(to_headline: Dict[str, Any], from_headline: Dict[str, Any], key: str):
    """Numeric delta ``to[key] - from[key]`` (None when either side is null)."""
    a = to_headline.get(key)
    b = from_headline.get(key)
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def run_generation_curve(
    repo_root: Path,
    course_slug: str,
    *,
    modes: Optional[List[str]] = None,
    synthesize_fn: Optional[Callable[[str], Any]] = None,
    eval_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    nli: Optional[Any] = None,
    embedder: Optional[Any] = None,
    best_of_n: int = 5,
    reselect_from_c3: bool = False,
    write: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the C0..C5 curve and emit the per-technique delta report (§6.1).

    For each mode the technique env is projected, synthesis re-runs (or the
    selector re-applies to a fixed pool for C3+ when ``reselect_from_c3``), and
    the generation-quality harness scores the artifacts. Returns the curve dict.

    ``synthesize_fn`` is injected by W2 (re-runs synthesis under the active env);
    when absent the runner reuses pre-synthesized per-mode artifacts via
    ``eval_fn`` alone (the default eval_fn loads on-disk artifacts). ``eval_fn``
    defaults to ``run_generation_quality_eval`` and is injectable for tests.
    """
    from lib.generation.technique_modes import (
        apply_mode_to_env,
        resolve_technique_mode,
    )

    mode_list = list(modes or DEFAULT_MODES)
    if eval_fn is None:
        from lib.objectives.generation_quality_eval import run_generation_quality_eval

        eval_fn = run_generation_quality_eval

    arms: List[Dict[str, Any]] = []
    for mode_name in mode_list:
        mode = resolve_technique_mode(mode_name, best_of_n=best_of_n)
        # Project the mode onto a copy of the process env so the synthesis +
        # router run under the right knobs. We mutate os.environ for the duration
        # of the arm (synthesis reads it), then restore.
        saved_env = dict(os.environ)
        try:
            apply_mode_to_env(mode, os.environ)
            # Selection-only arms (C3+) can re-apply the selector to a fixed pool
            # instead of re-synthesizing — the runner records which path ran.
            is_reselect = reselect_from_c3 and mode_name in {"C3", "C4", "C5"}
            arm_basis = "reselected" if is_reselect else "resynthesized"
            if synthesize_fn is not None and not is_reselect:
                synthesize_fn(course_slug)
            report = eval_fn(
                repo_root,
                course_slug,
                nli=nli,
                embedder=embedder,
                technique_config=mode_name,
                write=False,
            )
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
        arms.append(
            {
                "mode": mode_name,
                "arm_basis": arm_basis,
                "headline": report.get("headline", {}),
                "success_condition": report.get("success_condition"),
            }
        )

    # Build the per-technique delta table.
    deltas: List[Dict[str, Any]] = []
    for prev, cur in zip(arms, arms[1:]):
        from_h = prev["headline"]
        to_h = cur["headline"]
        edge = (prev["mode"], cur["mode"])
        deltas.append(
            {
                "from": prev["mode"],
                "to": cur["mode"],
                "technique": _TECHNIQUE_LABELS.get(edge, "unknown"),
                "block_prose_grounding_rate": _delta(
                    to_h, from_h, "block_prose_grounding_rate"
                ),
                "contradicted_block_count": (
                    (to_h.get("contradicted_block_count", 0) or 0)
                    - (from_h.get("contradicted_block_count", 0) or 0)
                ),
                "objective_grounding_rate": _delta(
                    to_h, from_h, "objective_grounding_rate"
                ),
            }
        )

    c5_arm = next((a for a in arms if a["mode"] == "C5"), None)
    ceiling = {
        "mode": "C5",
        "block_prose_grounding_rate": (
            c5_arm["headline"].get("block_prose_grounding_rate") if c5_arm else None
        ),
        "note": (
            "C5 numbers are the 7B-alone ceiling; the 70B/120B single-sample "
            "comparison is one config-row away when the Spark arrives "
            "(add a C5-70b arm)."
        ),
    }

    curve: Dict[str, Any] = {
        "schema_version": CURVE_SCHEMA_VERSION,
        "course_slug": course_slug,
        "generated_at": _utcnow_iso(),
        "provenance": {
            "nli_model_revision": getattr(nli, "_revision", None),
            "embedding_provider": getattr(embedder, "provider_name", None),
            "best_of_n": best_of_n,
            "single_corpus_caveat": True,
        },
        "arms": arms,
        "deltas": deltas,
        "ceiling": ceiling,
    }

    if write:
        libv2_root = os.environ.get("ED4ALL_LIBV2_ROOT")
        base = Path(libv2_root) if libv2_root else (Path(repo_root) / "LibV2")
        eval_dir = base / "courses" / course_slug / "generation_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(output_path) if output_path is not None else (
            eval_dir / f"generation_quality_curve_{ts}.json"
        )
        out.write_text(json.dumps(curve, indent=2, sort_keys=True), encoding="utf-8")
        curve["_written"] = {"curve_path": str(out)}

    return curve


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the C0..C5 generation-quality technique curve for one course."
    )
    parser.add_argument("--course", required=True, help="Course slug.")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated mode list (default C0,C1,C2,C3,C4,C5).",
    )
    parser.add_argument("--best-of-n", type=int, default=5, help="C5 best-of-N count.")
    parser.add_argument(
        "--no-resynthesize",
        action="store_true",
        help="Re-apply the selector to a fixed pool for C3+ (no N× synthesis).",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("ED4ALL_ROOT", "."),
        help="Repo root (defaults to ED4ALL_ROOT / cwd).",
    )
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    curve = run_generation_curve(
        Path(args.repo_root),
        args.course,
        modes=modes,
        best_of_n=args.best_of_n,
        reselect_from_c3=args.no_resynthesize,
        write=True,
    )
    written = curve.get("_written", {}).get("curve_path")
    print(json.dumps({"course": args.course, "curve_path": written}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["run_generation_curve", "main", "CURVE_SCHEMA_VERSION", "DEFAULT_MODES"]
