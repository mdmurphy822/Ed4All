"""Regression-diff for two ``grounded_answer_eval_<ts>.json`` reports.

Compares an OLD and a NEW grounded-answer eval report (emitted by
``lib.retrieval.grounded_eval.run_grounded_eval``) and reports the per-metric
deltas: headline rates (answer_rate, groundedness macro/micro, citation
metrics, refusal/abstention rates) plus the per-phrasing and per-category
diagnostic buckets. A delta worse than ``--tolerance-pp`` on a PINNED metric is
a REGRESSION and drives exit code 1; everything else exits 0.

Design contract (why this module reads JSON only, never grounded_eval internals):
  * It is a pure comparator — it makes NO LLM calls, so it wires no
    DecisionCapture (the decision-capture contract is scoped to LLM call sites).
  * It reads only the published report JSON SHAPE, tolerating additive unknown
    fields so it works across ``EVAL_SCHEMA_VERSION`` 1.6 and 1.7 (and beyond):
    every metric is pulled by a dotted path that resolves to ``None`` when
    absent, and a metric missing on either side is reported ``n/a`` (never a
    fabricated 0.0, never an exit-1). New top-level keys a future schema adds
    are simply ignored.

Basis semantics:
  * ``pinned`` metrics (the milestone-pinned rates) drive the exit code — a
    regression beyond tolerance on any pinned metric exits 1.
  * ``diagnostic`` metrics + ALL bucket families (per-phrasing, per-category)
    are reported INFORMATIONALLY — a diagnostic regression is shown (so an
    operator sees it) but NEVER triggers exit-1, matching the report's own
    ``_diagnostic`` annotations on those blocks.

Config-diff attribution: if either report carries an ``ED4ALL_ANSWER_*`` flag
stamp in its header, a changed flag value is surfaced in a config-diff section —
a changed config means the deltas are attributable to the config change, not a
regression, and the output says so.

Usage::

    python -m lib.retrieval.grounded_eval_diff OLD.json NEW.json [--tolerance-pp 5.0] [--json]

Exit codes: 0 = no pinned regression beyond tolerance, 1 = at least one.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Metric specs
# --------------------------------------------------------------------------- #

#: Direction of "better" for a metric.
_HIGHER = "higher"  # a larger value is better (rates like answer_rate)
_LOWER = "lower"  # a smaller value is better (rates like unsupported_claim_rate)

#: Basis of a metric — whether a beyond-tolerance regression drives exit-1.
_PINNED = "pinned"  # milestone-pinned; a regression exits 1
_DIAGNOSTIC = "diagnostic"  # informational; a regression NEVER exits 1


@dataclass(frozen=True)
class MetricSpec:
    """One comparable scalar metric pulled from the report by dotted path."""

    label: str
    path: str  # dotted path into the report dict (e.g. "headline.answer_rate")
    direction: str  # _HIGHER | _LOWER
    basis: str  # _PINNED | _DIAGNOSTIC


#: Ordered registry of the headline metrics diffed on every run. Only PINNED
#: rate metrics in [0,1] drive the exit code; count-based signals
#: (false_refusals, blocked counts) are DIAGNOSTIC because they scale with gold
#: size and are not comparable as raw counts across runs.
_METRIC_SPECS: Tuple[MetricSpec, ...] = (
    # --- pinned headline rates (drive exit-1) ---
    MetricSpec("answer_rate", "headline.answer_rate", _HIGHER, _PINNED),
    MetricSpec(
        "citation_resolution_rate",
        "headline.citation_resolution_rate",
        _HIGHER,
        _PINNED,
    ),
    MetricSpec(
        "citation_precision", "headline.citation_precision", _HIGHER, _PINNED
    ),
    MetricSpec(
        "citation_precision_primary",
        "headline.citation_precision_primary",
        _HIGHER,
        _PINNED,
    ),
    MetricSpec(
        "groundedness_rate_mean",
        "headline.groundedness_rate_mean",
        _HIGHER,
        _PINNED,
    ),
    MetricSpec(
        "unsupported_claim_rate",
        "headline.unsupported_claim_rate",
        _LOWER,
        _PINNED,
    ),
    MetricSpec(
        "refusal_recall", "headline.refusal.refusal_recall", _HIGHER, _PINNED
    ),
    MetricSpec(
        "refusal_precision",
        "headline.refusal.refusal_precision",
        _HIGHER,
        _PINNED,
    ),
    # --- diagnostic rates / counts (informational; never exit-1) ---
    MetricSpec(
        "groundedness_macro",
        "headline.groundedness_breakdown.macro_groundedness",
        _HIGHER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "groundedness_micro",
        "headline.groundedness_breakdown.micro_groundedness",
        _HIGHER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "key_point_coverage_rate",
        "headline.key_point_coverage.coverage_rate",
        _HIGHER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "unsupported_on_answered_probes",
        "headline.refusal.unsupported_answer_rate_on_answered_probes",
        _LOWER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "false_refusals_on_gold",
        "headline.refusal.false_refusals_on_gold",
        _LOWER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "blocked_invalid_citation",
        "blocked.invalid_citation",
        _LOWER,
        _DIAGNOSTIC,
    ),
    MetricSpec(
        "blocked_citation_gate", "blocked.citation_gate", _LOWER, _DIAGNOSTIC
    ),
    MetricSpec(
        "composer_exhausted", "composer_exhausted", _LOWER, _DIAGNOSTIC
    ),
)

#: Bucket families — dicts keyed by a bucket name (phrasing value / category)
#: whose per-bucket scalar we compare. ALL bucket families are DIAGNOSTIC (never
#: exit-1), matching the report's own diagnostic annotation on these blocks.
#: (family_label, dotted path to the dict, per-bucket value key, direction).
_BUCKET_FAMILIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("phrasing", "headline.phrasing_breakdown", "answered_rate", _HIGHER),
    (
        "question_type",
        "headline.groundedness_breakdown.by_question_type",
        "macro_groundedness",
        _HIGHER,
    ),
    (
        "difficulty",
        "headline.groundedness_breakdown.by_stratum",
        "macro_groundedness",
        _HIGHER,
    ),
)

#: Header fields surfaced in the meta line of the human + JSON output.
_META_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "course_slug",
    "engine",
    "model_id",
    "prompt_version",
    "refusal_policy_version",
    "generated_at",
)

#: Candidate header locations an ``ED4ALL_ANSWER_*`` flag stamp may live under.
#: The current schema (1.6) carries no stamp; a future schema may add one under
#: any of these. We scan them all and harvest ED4ALL_ANSWER_* keys, tolerating
#: absence (empty dict).
_CONFIG_STAMP_CANDIDATES: Tuple[str, ...] = (
    "config",
    "flag_config",
    "answer_config",
    "flags",
    "env",
)

_ANSWER_FLAG_PREFIX = "ED4ALL_ANSWER_"


# --------------------------------------------------------------------------- #
# Report access helpers (tolerant of additive / unknown fields)
# --------------------------------------------------------------------------- #

def _dotted_get(report: Dict[str, Any], path: str) -> Any:
    """Resolve a dotted path into ``report``; ``None`` if any hop is absent."""
    node: Any = report
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _as_number(value: Any) -> Optional[float]:
    """Coerce a report value to float; ``None`` for null / non-numeric."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_answer_config(report: Dict[str, Any]) -> Dict[str, str]:
    """Harvest any ``ED4ALL_ANSWER_*`` flag stamp from the report header.

    Scans the candidate stamp locations (and top-level keys) for keys that name
    an answer-backend flag; returns ``{flag: str(value)}``. Empty when the
    report carries no stamp (the current 1.6 shape) — the diff simply skips the
    config-diff section then.
    """
    found: Dict[str, str] = {}

    def _harvest(d: Any) -> None:
        if not isinstance(d, dict):
            return
        for key, val in d.items():
            if isinstance(key, str) and key.startswith(_ANSWER_FLAG_PREFIX):
                found[key] = "" if val is None else str(val)

    _harvest(report)
    for candidate in _CONFIG_STAMP_CANDIDATES:
        _harvest(report.get(candidate))
        # one level deeper (e.g. report["config"]["flags"])
        nested = report.get(candidate)
        if isinstance(nested, dict):
            for sub in _CONFIG_STAMP_CANDIDATES:
                _harvest(nested.get(sub))
    return found


def _meta(report: Dict[str, Any]) -> Dict[str, Any]:
    return {f: report.get(f) for f in _META_FIELDS}


# --------------------------------------------------------------------------- #
# Diff computation
# --------------------------------------------------------------------------- #

#: Row status values.
_IMPROVED = "improved"
_WITHIN = "within_tolerance"
_REGRESSED = "regressed"
_NA = "n/a"


@dataclass
class DiffRow:
    label: str
    basis: str
    direction: str
    old: Optional[float]
    new: Optional[float]
    delta: Optional[float]  # raw new - old
    delta_pp: Optional[float]  # (new - old) * 100 for rate metrics
    status: str
    counts_toward_exit: bool
    bucket_family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "basis": self.basis,
            "direction": self.direction,
            "old": self.old,
            "new": self.new,
            "delta": self.delta,
            "delta_pp": self.delta_pp,
            "status": self.status,
            "counts_toward_exit": self.counts_toward_exit,
            **(
                {"bucket_family": self.bucket_family}
                if self.bucket_family is not None
                else {}
            ),
        }


def _classify(
    old: Optional[float],
    new: Optional[float],
    direction: str,
    basis: str,
    tolerance_pp: float,
) -> Tuple[str, Optional[float], Optional[float], bool]:
    """Return (status, delta, delta_pp, counts_toward_exit)."""
    if old is None or new is None:
        return _NA, None, None, False
    delta = new - old
    delta_pp = delta * 100.0
    # "worse" magnitude in pp, sign-corrected for direction.
    worse_pp = -delta_pp if direction == _HIGHER else delta_pp
    if worse_pp > tolerance_pp:
        status = _REGRESSED
    elif worse_pp < -tolerance_pp:
        status = _IMPROVED
    else:
        status = _WITHIN
    counts = status == _REGRESSED and basis == _PINNED
    return status, delta, delta_pp, counts


def _diff_metrics(
    old: Dict[str, Any], new: Dict[str, Any], tolerance_pp: float
) -> List[DiffRow]:
    rows: List[DiffRow] = []
    for spec in _METRIC_SPECS:
        o = _as_number(_dotted_get(old, spec.path))
        n = _as_number(_dotted_get(new, spec.path))
        status, delta, delta_pp, counts = _classify(
            o, n, spec.direction, spec.basis, tolerance_pp
        )
        rows.append(
            DiffRow(
                label=spec.label,
                basis=spec.basis,
                direction=spec.direction,
                old=o,
                new=n,
                delta=delta,
                delta_pp=delta_pp,
                status=status,
                counts_toward_exit=counts,
            )
        )
    return rows


def _diff_buckets(
    old: Dict[str, Any], new: Dict[str, Any], tolerance_pp: float
) -> Dict[str, List[DiffRow]]:
    families: Dict[str, List[DiffRow]] = {}
    for family, path, value_key, direction in _BUCKET_FAMILIES:
        old_dict = _dotted_get(old, path)
        new_dict = _dotted_get(new, path)
        old_dict = old_dict if isinstance(old_dict, dict) else {}
        new_dict = new_dict if isinstance(new_dict, dict) else {}
        keys = sorted(set(old_dict) | set(new_dict))
        rows: List[DiffRow] = []
        for key in keys:
            o = _as_number(
                (old_dict.get(key) or {}).get(value_key)
                if isinstance(old_dict.get(key), dict)
                else None
            )
            n = _as_number(
                (new_dict.get(key) or {}).get(value_key)
                if isinstance(new_dict.get(key), dict)
                else None
            )
            # Bucket families are ALWAYS diagnostic → never count toward exit.
            status, delta, delta_pp, _ = _classify(
                o, n, direction, _DIAGNOSTIC, tolerance_pp
            )
            rows.append(
                DiffRow(
                    label=key,
                    basis=_DIAGNOSTIC,
                    direction=direction,
                    old=o,
                    new=n,
                    delta=delta,
                    delta_pp=delta_pp,
                    status=status,
                    counts_toward_exit=False,
                    bucket_family=family,
                )
            )
        if rows:
            families[family] = rows
    return families


@dataclass
class DiffResult:
    tolerance_pp: float
    old_meta: Dict[str, Any]
    new_meta: Dict[str, Any]
    config_diff: Dict[str, Any]
    metrics: List[DiffRow]
    buckets: Dict[str, List[DiffRow]]

    @property
    def regressions(self) -> List[DiffRow]:
        return [r for r in self.metrics if r.counts_toward_exit]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)

    @property
    def exit_code(self) -> int:
        return 1 if self.has_regression else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_pp": self.tolerance_pp,
            "old_report": self.old_meta,
            "new_report": self.new_meta,
            "config_diff": self.config_diff,
            "metrics": [r.to_dict() for r in self.metrics],
            "buckets": {
                fam: [r.to_dict() for r in rows]
                for fam, rows in self.buckets.items()
            },
            "regression": self.has_regression,
            "regressed_metrics": [r.label for r in self.regressions],
            "exit_code": self.exit_code,
        }


def _config_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    old_cfg = _extract_answer_config(old)
    new_cfg = _extract_answer_config(new)
    changes: Dict[str, Dict[str, Optional[str]]] = {}
    for flag in sorted(set(old_cfg) | set(new_cfg)):
        ov = old_cfg.get(flag)
        nv = new_cfg.get(flag)
        if ov != nv:
            changes[flag] = {"old": ov, "new": nv}
    return {
        "present": bool(old_cfg or new_cfg),
        "changed": bool(changes),
        "changes": changes,
    }


def diff_reports(
    old: Dict[str, Any], new: Dict[str, Any], tolerance_pp: float = 5.0
) -> DiffResult:
    """Compute the full metric + bucket + config diff between two reports."""
    return DiffResult(
        tolerance_pp=tolerance_pp,
        old_meta=_meta(old),
        new_meta=_meta(new),
        config_diff=_config_diff(old, new),
        metrics=_diff_metrics(old, new, tolerance_pp),
        buckets=_diff_buckets(old, new, tolerance_pp),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_val(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{v:6.4f}"


def _fmt_delta_pp(v: Optional[float]) -> str:
    if v is None:
        return "    n/a"
    return f"{v:+7.2f}"


_STATUS_TAG = {
    _IMPROVED: "improved",
    _WITHIN: "ok",
    _REGRESSED: "REGRESSION",
    _NA: "n/a",
}


def _row_tag(row: DiffRow) -> str:
    if row.status == _REGRESSED and not row.counts_toward_exit:
        # A diagnostic regression is shown but flagged as non-blocking.
        return "regressed(diag)"
    return _STATUS_TAG[row.status]


def render_text(result: DiffResult) -> str:
    """Human-readable delta table with regression highlighting."""
    lines: List[str] = []
    om, nm = result.old_meta, result.new_meta
    lines.append("=" * 78)
    lines.append("grounded-answer eval regression diff")
    lines.append("=" * 78)
    lines.append(
        f"  old: schema {om.get('schema_version')}  "
        f"course {om.get('course_slug')}  engine {om.get('engine')}  "
        f"generated {om.get('generated_at')}"
    )
    lines.append(
        f"  new: schema {nm.get('schema_version')}  "
        f"course {nm.get('course_slug')}  engine {nm.get('engine')}  "
        f"generated {nm.get('generated_at')}"
    )
    lines.append(f"  tolerance: {result.tolerance_pp:.2f} pp")
    lines.append("")

    # Config-diff section.
    cfg = result.config_diff
    if cfg["changed"]:
        lines.append("-- config diff (ED4ALL_ANSWER_*) -------------------------")
        for flag, chg in cfg["changes"].items():
            lines.append(f"  {flag}: {chg['old']!r} -> {chg['new']!r}")
        lines.append(
            "  NOTE: config changed between runs — deltas below are "
            "attributable to config, not (necessarily) regression."
        )
        lines.append("")
    elif cfg["present"]:
        lines.append(
            "-- config diff: ED4ALL_ANSWER_* flags unchanged between runs --"
        )
        lines.append("")

    # Metric table.
    header = f"  {'metric':<32}{'old':>8}{'new':>8}{'Δpp':>9}  {'basis':<11}status"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in result.metrics:
        lines.append(
            f"  {row.label:<32}{_fmt_val(row.old):>8}{_fmt_val(row.new):>8}"
            f"{_fmt_delta_pp(row.delta_pp):>9}  {row.basis:<11}{_row_tag(row)}"
        )

    # Bucket families.
    for family, rows in result.buckets.items():
        lines.append("")
        lines.append(f"-- bucket: {family} (diagnostic; never blocks) --")
        for row in rows:
            lines.append(
                f"  {row.label:<32}{_fmt_val(row.old):>8}{_fmt_val(row.new):>8}"
                f"{_fmt_delta_pp(row.delta_pp):>9}  {'diagnostic':<11}"
                f"{_row_tag(row)}"
            )

    # Verdict.
    lines.append("")
    lines.append("=" * 78)
    if result.has_regression:
        labels = ", ".join(r.label for r in result.regressions)
        lines.append(f"VERDICT: REGRESSION on pinned metric(s): {labels}")
        if cfg["changed"]:
            lines.append(
                "  (config changed — confirm the regression is not simply the "
                "config change before treating it as a defect.)"
            )
        lines.append("  exit 1")
    else:
        lines.append("VERDICT: no pinned regression beyond tolerance. exit 0")
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_report(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"report {path} is not a JSON object")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.grounded_eval_diff",
        description=(
            "Diff two grounded_answer_eval_<ts>.json reports: per-metric "
            "deltas with regression highlighting. Exits 1 on any pinned "
            "regression beyond --tolerance-pp, else 0. Diagnostic buckets are "
            "informational and never drive the exit code."
        ),
    )
    parser.add_argument("old_report", help="path to the OLD (baseline) report")
    parser.add_argument("new_report", help="path to the NEW (candidate) report")
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=5.0,
        help="regression tolerance in percentage points (default: 5.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the machine-readable diff shape instead of the table",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        old = _load_report(args.old_report)
        new = _load_report(args.new_report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"grounded-eval-diff: cannot read report: {exc}", file=sys.stderr)
        return 2

    tol = args.tolerance_pp if args.tolerance_pp >= 0 else 5.0
    result = diff_reports(old, new, tolerance_pp=tol)

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(result))
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MetricSpec",
    "DiffRow",
    "DiffResult",
    "diff_reports",
    "render_text",
    "main",
]
