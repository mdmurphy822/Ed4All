"""Calibration helper — propose warning → critical threshold flips post-rebuild.

This is **operator-facing tooling** that bridges the warning-day-1 contract
shipped across Waves 5 / 6 / 7 / 9 to the eventual warning-→-critical flip.
Across those waves every new gate landed at warning-severity day-1 with
calibration debt deferred to the operator's next corpus rebuild. The
operator runs the rebuild, reads fire rates against the warning thresholds,
then has to MANUALLY decide which thresholds are stable enough to flip.
This helper automates the read-fire-rates + propose-flips half of that
loop.

CRITICAL CONTRACT — purely informational
========================================
This script **DOES NOT MODIFY ANY SOURCE FILE**. It reads post-rebuild
signals from a real corpus rebuild and writes a single markdown proposal
the operator reviews before any threshold flip lands. The script never
edits ``lib/validators/*.py`` (which carry the canonical thresholds);
edits to those files are the operator's job after reviewing this proposal.

Usage::

    python -m Trainforge.scripts.harness.calibrate_pair_validation \\
        --course-code <course-slug>

    # Custom output path:
    python -m Trainforge.scripts.harness.calibrate_pair_validation \\
        --course-code <course-slug> \\
        --out /tmp/proposal.md

    # Tighter / looser percentile target (default 5):
    python -m Trainforge.scripts.harness.calibrate_pair_validation \\
        --course-code <course-slug> \\
        --target-percentile 10

Output (default):
    LibV2/courses/<slug>/quality/pair_calibration_proposal.md

Inputs read (best-effort, missing files surface as ``MISSING_INPUT_*``
warning rows in the proposal):
    LibV2/courses/<slug>/eval/eval_report.json                (W3.F + W7)
    LibV2/courses/<slug>/quality/quality_report.json          (W6)
    LibV2/courses/<slug>/training_specs/dataset_config.json   (W4 + W8)
    LibV2/courses/<slug>/training_specs/instruction_pairs.jsonl (W9)
    LibV2/courses/<slug>/training_specs/preference_pairs.jsonl (W9)

The script is dependency-light: stdlib only. No torch / transformers /
sentence-transformers imports — runs on a thin operator box.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lib.utils import bootstrap_percentile_ci, percentile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# Verdict enum + per-row dataclass
# --------------------------------------------------------------------------

VERDICT_READY = "READY_TO_FLIP"
VERDICT_NEEDS_DATA = "NEEDS_MORE_DATA"
VERDICT_KEEP = "KEEP_WARNING"
VERDICT_MISSING = "MISSING_INPUT"


@dataclass
class CalibrationRow:
    """One proposed-flip row in the markdown table.

    A row covers exactly one warning threshold across the four waves.
    """

    wave: str  # e.g. "Wave 4 W4.A"
    metric: str  # human-readable metric name
    current_threshold: float  # the day-1 warning floor / ceiling
    direction: str  # "below" (floor) or "above" (ceiling)
    fire_rate: Optional[float]  # fraction of corpus that triggered the warning
    sample_n: int  # number of observations used
    recommended: Optional[float]  # proposed critical threshold
    ci_low: Optional[float]
    ci_high: Optional[float]
    verdict: str
    file_path: str  # operator-edit hint
    notes: str = ""


# --------------------------------------------------------------------------
# Defensive JSON / JSONL loading
# --------------------------------------------------------------------------

def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_load_jsonl(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines — best-effort load.
                    continue
        return rows
    except OSError:
        return None


# --------------------------------------------------------------------------
# Statistics helpers (pure-stdlib bootstrap CI)
# --------------------------------------------------------------------------

# Floor under which the bootstrap CI is too wide to trust the
# recommendation. Observations below this count → NEEDS_MORE_DATA.
_MIN_SAMPLE_FOR_FLIP: int = 20

# Bootstrap iteration count. Pinned at 100 per the spec; sufficient for
# a 95% CI on a single-percentile estimator.
_BOOTSTRAP_ITERS: int = 100


# Module-private aliases preserve the legacy ``_percentile`` /
# ``_bootstrap_percentile_ci`` symbols for any internal helpers that still
# reference them by underscore-prefix; canonical home is now
# :mod:`lib.utils.stats` (see W-D6 plan §3.4).
_percentile = percentile


def _bootstrap_percentile_ci(
    values: List[float],
    pct: float,
    iterations: int = _BOOTSTRAP_ITERS,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Thin wrapper around :func:`lib.utils.bootstrap_percentile_ci`."""
    return bootstrap_percentile_ci(values, pct, iterations=iterations, seed=seed)


def _verdict_from_ci(
    current_threshold: float,
    direction: str,
    recommended: Optional[float],
    ci_low: Optional[float],
    ci_high: Optional[float],
    sample_n: int,
    fire_rate: Optional[float],
) -> str:
    """Decide READY / NEEDS_MORE_DATA / KEEP per row.

    - NEEDS_MORE_DATA: too few observations to trust the recommendation
      (sample size below the floor) OR the CI is None.
    - KEEP_WARNING: fire rate already at acceptable level (≤ ~5%) at the
      current warning threshold; promoting to critical adds no signal.
    - READY_TO_FLIP: CI doesn't cross the current warning threshold in
      the wrong direction (a "floor" recommendation must be ≥ current
      to be tighter; a "ceiling" recommendation must be ≤ current).
    """
    if recommended is None or ci_low is None or ci_high is None:
        return VERDICT_NEEDS_DATA
    if sample_n < _MIN_SAMPLE_FOR_FLIP:
        return VERDICT_NEEDS_DATA
    # Fire rate at the current warning threshold is already in spec —
    # no need to crank harder.
    if fire_rate is not None and fire_rate <= 0.05:
        # If the fire rate is exactly at the gate's acceptance line,
        # warning is the right severity. Promoting to critical would
        # block on noise.
        return VERDICT_KEEP
    # CI crossing the current threshold means the recommendation is
    # not robustly tighter than the warning floor / ceiling. Defer.
    if direction == "below":
        # Floor: recommended threshold should be ≥ current to be a
        # tighter critical floor. CI must not extend below current.
        if ci_low < current_threshold:
            return VERDICT_NEEDS_DATA
    else:  # "above"
        # Ceiling: recommended threshold should be ≤ current to be a
        # tighter critical ceiling. CI must not extend above current.
        if ci_high > current_threshold:
            return VERDICT_NEEDS_DATA
    return VERDICT_READY


# --------------------------------------------------------------------------
# Per-wave row builders
# --------------------------------------------------------------------------

def _wave4_w4a_rows(
    pairs: List[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """W4.A — claim-support thresholds.

    Per-pair signals: ``per_claim_support[].outcome`` (entailed /
    contradicted / unsupported), ``per_claim_support[].max_entailment``,
    ``per_claim_support[].max_contradiction``.
    """

    # Aggregate per-pair: unsupported_rate (fraction of claims marked
    # ``unsupported`` per pair) and contradicted_rate (fraction marked
    # ``contradicted``). These are the per-pair scalars W4.A's gate
    # ceilings target.
    unsupported_rates: List[float] = []
    contradicted_rates: List[float] = []
    max_entailments: List[float] = []
    max_contradictions: List[float] = []

    for p in pairs:
        claims = p.get("per_claim_support") or []
        if not claims:
            continue
        n_total = len(claims)
        n_unsup = sum(1 for c in claims if c.get("outcome") == "unsupported")
        n_contra = sum(1 for c in claims if c.get("outcome") == "contradicted")
        unsupported_rates.append(n_unsup / n_total)
        contradicted_rates.append(n_contra / n_total)
        for c in claims:
            me = c.get("max_entailment")
            mc = c.get("max_contradiction")
            if isinstance(me, (int, float)):
                max_entailments.append(float(me))
            if isinstance(mc, (int, float)):
                max_contradictions.append(float(mc))

    rows: List[CalibrationRow] = []

    # max_unsupported_claim_rate: ceiling, default 0.20
    rows.append(_build_ceiling_row(
        wave="Wave 4 W4.A",
        metric="max_unsupported_claim_rate",
        current=0.20,
        observations=unsupported_rates,
        target_percentile=100.0 - target_percentile,
        file_path="lib/validators/claim_support.py::_DEFAULT_MAX_UNSUPPORTED_RATE",
        notes="Per-pair fraction of claims marked unsupported.",
    ))

    # max_contradicted_claim_rate: ceiling, default 0.05
    rows.append(_build_ceiling_row(
        wave="Wave 4 W4.A",
        metric="max_contradicted_claim_rate",
        current=0.05,
        observations=contradicted_rates,
        target_percentile=100.0 - target_percentile,
        file_path="lib/validators/claim_support.py::_DEFAULT_MAX_CONTRADICTED_RATE",
        notes="Per-pair fraction of claims marked contradicted.",
    ))

    # entailment_floor: floor, default 0.70
    rows.append(_build_floor_row(
        wave="Wave 4 W4.A",
        metric="entailment_floor",
        current=0.70,
        observations=max_entailments,
        target_percentile=target_percentile,
        file_path="lib/validators/claim_support.py::_DEFAULT_ENTAILMENT_FLOOR",
        notes="Per-claim NLI entailment floor.",
    ))

    # contradiction_floor: floor, default 0.50
    rows.append(_build_floor_row(
        wave="Wave 4 W4.A",
        metric="contradiction_floor",
        current=0.50,
        observations=max_contradictions,
        target_percentile=target_percentile,
        file_path="lib/validators/claim_support.py::_DEFAULT_CONTRADICTION_FLOOR",
        notes="Per-claim NLI contradiction floor.",
    ))

    return rows


def _wave5_w5d_rows(
    pairs: List[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """W5.D — per-claim attribution cosine match floor (default 0.50)."""
    cosines: List[float] = []
    for p in pairs:
        claims = p.get("per_claim_support") or []
        for c in claims:
            cm = c.get("claim_match_cosine") or c.get("per_claim_match_cosine")
            if isinstance(cm, (int, float)):
                cosines.append(float(cm))
    return [_build_floor_row(
        wave="Wave 5 W5.D",
        metric="per_claim_match_cosine_floor",
        current=0.50,
        observations=cosines,
        target_percentile=target_percentile,
        file_path="lib/validators/pair_claim_support.py::_PER_CLAIM_MATCH_COSINE_FLOOR",
        notes="Per-claim attribution: sentence ↔ claim cosine match.",
    )]


def _wave6_w6a_rows(
    quality_report: Optional[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """W6.A — per-question-type quality thresholds (5 types × 4 axes)."""

    # Source of truth for the day-1 warning floors. Mirrors
    # ``lib/validators/assessment.py::_PER_QUESTION_TYPE_THRESHOLDS``.
    per_type_thresholds: Dict[str, Dict[str, float]] = {
        "multiple_choice": {
            "stem_diversity": 0.75,
            "correct_answer_diversity": 0.65,
            "distractor_template_max_ratio": 0.25,
            "min_stem_chars": 12,
        },
        "true_false": {
            "stem_diversity": 0.50,
            "correct_answer_diversity": 0.40,
            "distractor_template_max_ratio": 1.0,
            "min_stem_chars": 10,
        },
        "short_answer": {
            "stem_diversity": 0.65,
            "correct_answer_diversity": 0.55,
            "distractor_template_max_ratio": 1.0,
            "min_stem_chars": 12,
        },
        "essay": {
            "stem_diversity": 0.55,
            "correct_answer_diversity": 0.50,
            "distractor_template_max_ratio": 1.0,
            "min_stem_chars": 15,
        },
        "fill_in_blank": {
            "stem_diversity": 0.65,
            "correct_answer_diversity": 0.55,
            "distractor_template_max_ratio": 0.30,
            "min_stem_chars": 10,
        },
    }

    summary = (
        quality_report.get("assessments", {}).get("per_question_type_summary", {})
        if quality_report else {}
    )

    rows: List[CalibrationRow] = []
    for qt, axes in per_type_thresholds.items():
        bucket = summary.get(qt) or {}
        # Observations: stem_diversity / correct_answer_diversity come
        # straight from the bucket dict. We treat each bucket as a
        # single-observation source for the purpose of the per-type
        # gate; a multi-bucket corpus aggregates across buckets.
        for axis, current in axes.items():
            value = bucket.get(axis)
            obs: List[float] = []
            if isinstance(value, (int, float)):
                obs.append(float(value))
            if axis == "distractor_template_max_ratio":
                # Ceiling axis. Direction: above.
                rows.append(_build_ceiling_row(
                    wave="Wave 6 W6.A",
                    metric=f"{qt}.{axis}",
                    current=float(current),
                    observations=obs,
                    target_percentile=100.0 - target_percentile,
                    file_path=(
                        f"lib/validators/assessment.py::_PER_QUESTION_TYPE_THRESHOLDS"
                        f"[\"{qt}\"][\"{axis}\"]"
                    ),
                ))
            else:
                # Floor axis. Direction: below.
                rows.append(_build_floor_row(
                    wave="Wave 6 W6.A",
                    metric=f"{qt}.{axis}",
                    current=float(current),
                    observations=obs,
                    target_percentile=target_percentile,
                    file_path=(
                        f"lib/validators/assessment.py::_PER_QUESTION_TYPE_THRESHOLDS"
                        f"[\"{qt}\"][\"{axis}\"]"
                    ),
                ))
    return rows


def _wave7_w7d_rows(
    eval_report: Optional[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """W7.D — per-type eval thresholds (6 metrics × per-type buckets)."""

    # Mirrors ``lib/validators/eval_gating.py::PER_TYPE_METRIC_CONFIG``.
    per_type_config = [
        ("answerable_rate", "answerable_rate", 0.30, "below"),
        ("single_correct_rate", "single_correct_rate", 0.85, "below"),
        ("distractor_entropy", "distractor_entropy_mean", 0.50, "below"),
        ("bloom_alignment_rate", "bloom_alignment_rate", 0.50, "below"),
        ("placeholder_rate", "placeholder_rate", 0.05, "above"),
        ("source_support_rate", "source_support_rate", 0.40, "below"),
    ]

    rows: List[CalibrationRow] = []
    if not eval_report:
        for metric_name, _, current, direction in per_type_config:
            rows.append(CalibrationRow(
                wave="Wave 7 W7.D",
                metric=f"min_per_type_{metric_name}"
                if direction == "below"
                else f"max_per_type_{metric_name}",
                current_threshold=float(current),
                direction=direction,
                fire_rate=None,
                sample_n=0,
                recommended=None,
                ci_low=None,
                ci_high=None,
                verdict=VERDICT_MISSING,
                file_path="lib/validators/eval_gating.py::_DEFAULT_THRESHOLDS",
                notes="MISSING_INPUT_eval_report.json",
            ))
        return rows

    for metric_name, value_field, current, direction in per_type_config:
        block = eval_report.get(metric_name)
        per_type = (
            block.get("per_question_type")
            if isinstance(block, dict) else None
        )
        observations: List[float] = []
        if isinstance(per_type, dict):
            for qt, bucket in per_type.items():
                if not isinstance(bucket, dict):
                    continue
                # Skip buckets marked irrelevant (e.g. distractor_entropy
                # on essay) — W7.D's gate skips them too.
                if bucket.get("relevant") is False:
                    continue
                v = bucket.get(value_field)
                if isinstance(v, (int, float)):
                    observations.append(float(v))
        if direction == "below":
            rows.append(_build_floor_row(
                wave="Wave 7 W7.D",
                metric=f"min_per_type_{metric_name}",
                current=float(current),
                observations=observations,
                target_percentile=target_percentile,
                file_path="lib/validators/eval_gating.py::_DEFAULT_THRESHOLDS",
            ))
        else:
            rows.append(_build_ceiling_row(
                wave="Wave 7 W7.D",
                metric=f"max_per_type_{metric_name}",
                current=float(current),
                observations=observations,
                target_percentile=100.0 - target_percentile,
                file_path="lib/validators/eval_gating.py::_DEFAULT_THRESHOLDS",
            ))
    return rows


def _wave9_tight_rows(
    pairs: List[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """Wave 9 TIGHT — DART contradiction floor + disagreement-rate ceiling."""

    # DART contradiction floor: per-claim NLI contradiction scores from
    # the dual-source DART re-score path.
    dart_contradictions: List[float] = []
    # Disagreement: count pairs where ANY per_claim entry has
    # outcome=="dart_disagreement".
    n_audited = 0
    n_disagreement = 0
    for p in pairs:
        claims = p.get("per_claim_support") or []
        if not claims:
            continue
        n_audited += 1
        had_disagreement = False
        for c in claims:
            outcome = c.get("dart_source_check") or c.get("outcome")
            if outcome == "dart_disagreement":
                had_disagreement = True
            ds_max_contra = c.get("dart_max_contradiction")
            if isinstance(ds_max_contra, (int, float)):
                dart_contradictions.append(float(ds_max_contra))
        if had_disagreement:
            n_disagreement += 1

    rows: List[CalibrationRow] = []
    rows.append(_build_floor_row(
        wave="Wave 9 TIGHT",
        metric="dart_contradiction_floor",
        current=0.50,
        observations=dart_contradictions,
        target_percentile=target_percentile,
        file_path="lib/validators/pair_claim_support.py::_DEFAULT_DART_CONTRADICTION_FLOOR",
        notes="DART NLI contradiction floor for dual-source re-score.",
    ))

    # dart_disagreement_rate: ceiling, default 0.05.
    fire_rate = (n_disagreement / n_audited) if n_audited > 0 else None
    # The "observation set" here is binary per-pair (1 if disagreement,
    # else 0); recommendation = the rate at which the corpus operates,
    # nudged tighter via the same target-percentile mechanic.
    binary_obs = [
        1.0 if (p.get("per_claim_support") and any(
            (c.get("dart_source_check") or c.get("outcome")) == "dart_disagreement"
            for c in p.get("per_claim_support") or []
        )) else 0.0
        for p in pairs if p.get("per_claim_support")
    ]
    if binary_obs:
        recommended = sum(binary_obs) / len(binary_obs)
        ci_low, ci_high = _bootstrap_percentile_ci(binary_obs, 50.0)
        verdict = _verdict_from_ci(
            current_threshold=0.05,
            direction="above",
            recommended=recommended,
            ci_low=ci_low,
            ci_high=ci_high,
            sample_n=len(binary_obs),
            fire_rate=fire_rate,
        )
    else:
        recommended, ci_low, ci_high = None, None, None
        verdict = VERDICT_NEEDS_DATA

    rows.append(CalibrationRow(
        wave="Wave 9 TIGHT",
        metric="dart_disagreement_rate_ceiling",
        current_threshold=0.05,
        direction="above",
        fire_rate=fire_rate,
        sample_n=len(binary_obs),
        recommended=recommended,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict,
        file_path="lib/validators/pair_claim_support.py::_DART_DISAGREEMENT_RATE_WARN_CEILING",
        notes="Aggregate-rate ceiling: % of pairs with any dart_disagreement.",
    ))
    return rows


def _wave17_block_objective_rows(
    eval_report: Optional[Dict[str, Any]],
    quality_report: Optional[Dict[str, Any]],
    target_percentile: float,
) -> List[CalibrationRow]:
    """Wave 1.7 — per-block-type entailment floors (extended coverage).

    Beyond the brief: ``BlockObjectiveDeliveryValidator._PER_BLOCK_TYPE_
    ENTAILMENT_FLOOR`` carries 6 per-block-type floors that are also
    warning-day-1. Surface them in the proposal so the operator gets
    the full inventory.
    """
    per_block_floors = {
        "concept": 0.30,
        "example": 0.40,
        "explanation": 0.45,
        "activity": 0.40,
        "self_check_question": 0.50,
        "assessment_item": 0.55,
    }
    rows: List[CalibrationRow] = []
    for block_type, current in per_block_floors.items():
        # No per-block-type entailment surface in the rebuild reports
        # today; emit the row with no observations so the operator
        # gets the inventory + the canonical file path.
        rows.append(CalibrationRow(
            wave="Wave 1.7 W1.7.C",
            metric=f"per_block_type_entailment_floor[{block_type}]",
            current_threshold=float(current),
            direction="below",
            fire_rate=None,
            sample_n=0,
            recommended=None,
            ci_low=None,
            ci_high=None,
            verdict=VERDICT_NEEDS_DATA,
            file_path=(
                "lib/validators/block_objective_delivery.py::"
                "_PER_BLOCK_TYPE_ENTAILMENT_FLOOR"
            ),
            notes="No per-block entailment surface in the rebuild reports.",
        ))
    return rows


# --------------------------------------------------------------------------
# Floor / ceiling row builders
# --------------------------------------------------------------------------

def _build_floor_row(
    wave: str,
    metric: str,
    current: float,
    observations: List[float],
    target_percentile: float,
    file_path: str,
    notes: str = "",
) -> CalibrationRow:
    """Build a calibration row for a FLOOR (direction=below)."""
    if not observations:
        return CalibrationRow(
            wave=wave,
            metric=metric,
            current_threshold=current,
            direction="below",
            fire_rate=None,
            sample_n=0,
            recommended=None,
            ci_low=None,
            ci_high=None,
            verdict=VERDICT_NEEDS_DATA,
            file_path=file_path,
            notes=notes or "No observations.",
        )
    # Fire rate at current floor: % of observations BELOW the floor.
    n_below = sum(1 for v in observations if v < current)
    fire_rate = n_below / len(observations)
    # Recommendation: the target_percentile-th value (e.g. p5 for the
    # worst-5% target) defines the floor that catches the worst 5%.
    recommended = _percentile(observations, target_percentile)
    ci_low, ci_high = _bootstrap_percentile_ci(observations, target_percentile)
    verdict = _verdict_from_ci(
        current_threshold=current,
        direction="below",
        recommended=recommended,
        ci_low=ci_low,
        ci_high=ci_high,
        sample_n=len(observations),
        fire_rate=fire_rate,
    )
    return CalibrationRow(
        wave=wave,
        metric=metric,
        current_threshold=current,
        direction="below",
        fire_rate=fire_rate,
        sample_n=len(observations),
        recommended=recommended,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict,
        file_path=file_path,
        notes=notes,
    )


def _build_ceiling_row(
    wave: str,
    metric: str,
    current: float,
    observations: List[float],
    target_percentile: float,
    file_path: str,
    notes: str = "",
) -> CalibrationRow:
    """Build a calibration row for a CEILING (direction=above).

    ``target_percentile`` is the upper percentile (e.g. 95 for the
    worst-5% target). Recommendation = the value at that percentile;
    fire rate at current = % of observations ABOVE the ceiling.
    """
    if not observations:
        return CalibrationRow(
            wave=wave,
            metric=metric,
            current_threshold=current,
            direction="above",
            fire_rate=None,
            sample_n=0,
            recommended=None,
            ci_low=None,
            ci_high=None,
            verdict=VERDICT_NEEDS_DATA,
            file_path=file_path,
            notes=notes or "No observations.",
        )
    n_above = sum(1 for v in observations if v > current)
    fire_rate = n_above / len(observations)
    recommended = _percentile(observations, target_percentile)
    ci_low, ci_high = _bootstrap_percentile_ci(observations, target_percentile)
    verdict = _verdict_from_ci(
        current_threshold=current,
        direction="above",
        recommended=recommended,
        ci_low=ci_low,
        ci_high=ci_high,
        sample_n=len(observations),
        fire_rate=fire_rate,
    )
    return CalibrationRow(
        wave=wave,
        metric=metric,
        current_threshold=current,
        direction="above",
        fire_rate=fire_rate,
        sample_n=len(observations),
        recommended=recommended,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict,
        file_path=file_path,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Markdown formatting
# --------------------------------------------------------------------------

def _fmt_optional_float(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _fmt_row(row: CalibrationRow) -> str:
    return (
        f"| `{row.metric}` "
        f"| {row.direction} "
        f"| {_fmt_optional_float(row.current_threshold)} "
        f"| {_fmt_optional_float(row.fire_rate)} "
        f"| {row.sample_n} "
        f"| {_fmt_optional_float(row.recommended)} "
        f"| {_fmt_optional_float(row.ci_low)}–{_fmt_optional_float(row.ci_high)} "
        f"| **{row.verdict}** "
        f"| `{row.file_path}` "
        f"| {row.notes} |"
    )


_TABLE_HEADER = (
    "| Metric | Dir | Current | Fire Rate | N | Recommended | 95% CI | "
    "Verdict | Edit Path | Notes |\n"
    "|--------|-----|---------|-----------|---|-------------|--------|"
    "---------|-----------|-------|"
)


def render_proposal(
    course_code: str,
    target_percentile: float,
    rows_by_wave: Dict[str, List[CalibrationRow]],
    missing_inputs: List[str],
) -> str:
    """Render the full markdown proposal."""
    out: List[str] = []
    out.append(f"# Pair Validation Threshold Calibration Proposal — `{course_code}`")
    out.append("")
    out.append(
        "**This proposal is read-only.** The script that produced it "
        "does not modify any threshold source file. Review each "
        "`READY_TO_FLIP` row, then manually edit the listed `Edit Path` "
        "to promote the warning-severity floor / ceiling to a "
        "critical-severity gate."
    )
    out.append("")
    out.append(f"- Target percentile: **{target_percentile:.1f}%** (worst-{target_percentile:.0f}% catch)")
    out.append(f"- Bootstrap iterations: {_BOOTSTRAP_ITERS}")
    out.append(f"- Min sample for flip: {_MIN_SAMPLE_FOR_FLIP}")

    # Tally verdicts.
    all_rows: List[CalibrationRow] = []
    for rows in rows_by_wave.values():
        all_rows.extend(rows)
    n_ready = sum(1 for r in all_rows if r.verdict == VERDICT_READY)
    n_keep = sum(1 for r in all_rows if r.verdict == VERDICT_KEEP)
    n_data = sum(1 for r in all_rows if r.verdict == VERDICT_NEEDS_DATA)
    n_missing = sum(1 for r in all_rows if r.verdict == VERDICT_MISSING)

    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- **READY_TO_FLIP**: {n_ready}")
    out.append(f"- **KEEP_WARNING**: {n_keep}")
    out.append(f"- **NEEDS_MORE_DATA**: {n_data}")
    out.append(f"- **MISSING_INPUT**: {n_missing}")
    out.append(f"- **Total thresholds covered**: {len(all_rows)}")
    out.append("")

    if missing_inputs:
        out.append("### Missing Inputs")
        out.append("")
        for path in missing_inputs:
            out.append(f"- `{path}` — **MISSING_INPUT_{Path(path).name}**")
        out.append("")

    # One section per wave.
    for wave_label, rows in rows_by_wave.items():
        out.append(f"## {wave_label}")
        out.append("")
        out.append(_TABLE_HEADER)
        for row in rows:
            out.append(_fmt_row(row))
        out.append("")

    out.append("---")
    out.append("")
    out.append(
        "Generated by `Trainforge.scripts.harness.calibrate_pair_validation`. "
        "This script reads post-rebuild signals from a real corpus rebuild "
        "and proposes (does NOT apply) warning → critical threshold flips."
    )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------

def run_calibration(
    course_root: Path,
    course_code: str,
    target_percentile: float,
) -> Tuple[Dict[str, List[CalibrationRow]], List[str]]:
    """Read every input + build per-wave row lists.

    Returns ``(rows_by_wave, missing_inputs)``.
    """
    eval_report_path = course_root / "eval" / "eval_report.json"
    quality_report_path = course_root / "quality" / "quality_report.json"
    dataset_config_path = course_root / "training_specs" / "dataset_config.json"
    instruction_pairs_path = course_root / "training_specs" / "instruction_pairs.jsonl"
    preference_pairs_path = course_root / "training_specs" / "preference_pairs.jsonl"

    eval_report = _safe_load_json(eval_report_path)
    quality_report = _safe_load_json(quality_report_path)
    dataset_config = _safe_load_json(dataset_config_path)
    instruction_pairs = _safe_load_jsonl(instruction_pairs_path) or []
    preference_pairs = _safe_load_jsonl(preference_pairs_path) or []

    missing: List[str] = []
    for path, obj in [
        (eval_report_path, eval_report),
        (quality_report_path, quality_report),
        (dataset_config_path, dataset_config),
    ]:
        if obj is None:
            missing.append(str(path.relative_to(course_root.parent.parent.parent))
                           if course_root.parent.parent.parent in path.parents
                           else str(path))
    if not instruction_pairs_path.exists():
        missing.append(str(instruction_pairs_path))
    if not preference_pairs_path.exists():
        missing.append(str(preference_pairs_path))

    # Aggregate pairs (W4.A / W5.D / W9 read across both pair types).
    all_pairs: List[Dict[str, Any]] = []
    all_pairs.extend(instruction_pairs)
    all_pairs.extend(preference_pairs)

    rows_by_wave: Dict[str, List[CalibrationRow]] = {}
    rows_by_wave["Wave 4 W4.A — Claim-Support Thresholds"] = _wave4_w4a_rows(
        all_pairs, target_percentile,
    )
    rows_by_wave["Wave 5 W5.D — Per-Claim Attribution"] = _wave5_w5d_rows(
        all_pairs, target_percentile,
    )
    rows_by_wave["Wave 6 W6.A — Per-Question-Type Quality"] = _wave6_w6a_rows(
        quality_report, target_percentile,
    )
    rows_by_wave["Wave 7 W7.D — Per-Type Eval Thresholds"] = _wave7_w7d_rows(
        eval_report, target_percentile,
    )
    rows_by_wave["Wave 9 TIGHT — Dual-Source DART Pair Validation"] = _wave9_tight_rows(
        all_pairs, target_percentile,
    )
    rows_by_wave["Wave 1.7 W1.7.C — Per-Block-Type Entailment Floors"] = (
        _wave17_block_objective_rows(eval_report, quality_report, target_percentile)
    )

    return rows_by_wave, missing


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Propose warning → critical threshold flips post corpus rebuild. "
            "READ-ONLY: this script never modifies any source file."
        ),
    )
    parser.add_argument(
        "--course-code",
        required=True,
        help="LibV2 course slug (e.g. <course-slug>).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output markdown path. Default: "
            "LibV2/courses/<slug>/quality/pair_calibration_proposal.md"
        ),
    )
    parser.add_argument(
        "--target-percentile",
        type=float,
        default=5.0,
        help=(
            "Worst-N%% target for the recommended threshold. "
            "Default 5 (a critical floor that catches the worst 5%% of "
            "the corpus)."
        ),
    )
    parser.add_argument(
        "--course-root",
        type=Path,
        default=None,
        help=(
            "Override the course root path. Default: "
            "<repo>/LibV2/courses/<course-code>. Used by the test "
            "harness to point at a tmp_path fixture."
        ),
    )
    args = parser.parse_args(argv)

    course_root = args.course_root or (
        PROJECT_ROOT / "LibV2" / "courses" / args.course_code
    )
    if not course_root.exists():
        print(
            f"ERROR: course root not found: {course_root}",
            file=sys.stderr,
        )
        return 2

    rows_by_wave, missing = run_calibration(
        course_root=course_root,
        course_code=args.course_code,
        target_percentile=args.target_percentile,
    )

    out_path = args.out or (course_root / "quality" / "pair_calibration_proposal.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_proposal(
        course_code=args.course_code,
        target_percentile=args.target_percentile,
        rows_by_wave=rows_by_wave,
        missing_inputs=missing,
    )
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Proposal written: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
