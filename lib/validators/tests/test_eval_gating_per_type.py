"""Wave 7 W7.D — per-question-type warning surface on EvalGatingValidator.

Pins the day-1 warning emit on each of the 6 W3.F per-metric per-type
threshold checks. Reads the W7.A (`answerable_rate`, `placeholder_rate`)
and W7.B (`single_correct_rate`, `distractor_entropy`,
`bloom_alignment_rate`, `source_support_rate`) ``per_question_type``
blocks emitted by every W3.F evaluator and asserts:

  * a below-threshold bucket fires a warning-severity issue;
  * the warning never flips ``passed=False`` (calibration-deferred
    per plan §5);
  * legacy reports (no per-type blocks) skip silently;
  * non-relevant buckets skip silently;
  * deps-missing branch (per_question_type=None) skips silently;
  * the ``eval_gating_decision`` rationale interpolates the per-type
    below-threshold values;
  * existing aggregate critical thresholds remain unchanged.

Mirrors the test pattern at ``lib/tests/test_eval_gating_validator.py``
(the legacy `_write_report` helper) but pins the per-type surface on
each of the 6 W7.D codes individually.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.eval_gating import EvalGatingValidator  # noqa: E402


def _write_report(model_dir: Path, **fields: Any) -> Path:
    """Build a baseline-passing eval_report.json under model_dir, with
    overrides applied via fields. Mirrors the helper in
    ``lib/tests/test_eval_gating_validator.py``."""
    report: Dict[str, Any] = {
        "faithfulness": 0.80,
        "coverage": 0.90,
        "profile": "rdf_shacl",
        "per_tier": {},
        "per_invariant": {},
        "baseline_delta": 0.10,
        "source_match": 0.65,
        "negative_grounding_accuracy": 0.70,
        "yes_rate": 0.55,
        "metrics": {"hallucination_rate": 0.20},
    }
    report.update(fields)
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / "eval_report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
# 1. answerable_rate per-type below-threshold fires warning.
# --------------------------------------------------------------------- #


def test_per_type_answerable_rate_below_threshold_warns(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-w7d-1"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        answerable_rate={
            "answerable_rate": 0.80,
            "per_question_type": {
                "essay": {
                    "answerable_rate": 0.20,  # below 0.30 floor
                    "total_questions": 5,
                    "answerable_count": 1,
                    "relevant": True,
                },
                "multiple_choice": {
                    "answerable_rate": 0.95,
                    "total_questions": 20,
                    "answerable_count": 19,
                    "relevant": True,
                },
            },
        },
    )
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    warning_codes = [i.code for i in result.issues if i.severity == "warning"]
    assert "EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD" in warning_codes
    # Only the essay bucket fires (not the MC one).
    msgs = [
        i.message
        for i in result.issues
        if i.code == "EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert len(msgs) == 1
    assert "essay" in msgs[0]
    assert "0.2" in msgs[0]


# --------------------------------------------------------------------- #
# 2. Per-type warning never blocks promotion.
# --------------------------------------------------------------------- #


def test_per_type_warning_does_not_block_promotion(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-w7d-2"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        answerable_rate={
            "answerable_rate": 0.80,
            "per_question_type": {
                "essay": {
                    "answerable_rate": 0.20,
                    "total_questions": 5,
                    "answerable_count": 1,
                    "relevant": True,
                },
            },
        },
    )
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    # Passed=True even with a per-type warning fired (warning-severity
    # never flips passed=False).
    assert result.passed is True
    # The warning IS in the issues array.
    assert any(
        i.severity == "warning"
        and i.code == "EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD"
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# 3. Legacy report (no per-type blocks) skips silently.
# --------------------------------------------------------------------- #


def test_per_type_skipped_when_per_question_type_block_absent(
    tmp_path: Path,
) -> None:
    """Legacy report with no `per_question_type` field on any metric
    must not fire any per-type warnings."""
    model_dir = tmp_path / "models" / "test-w7d-3"
    model_dir.mkdir(parents=True)
    # Baseline report has none of the per-type-emitting metric blocks.
    _write_report(model_dir)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    per_type_codes = {
        "EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD",
        "EVAL_SINGLE_CORRECT_PER_TYPE_BELOW_THRESHOLD",
        "EVAL_DISTRACTOR_ENTROPY_PER_TYPE_BELOW_THRESHOLD",
        "EVAL_BLOOM_ALIGNMENT_PER_TYPE_BELOW_THRESHOLD",
        "EVAL_PLACEHOLDER_PER_TYPE_ABOVE_THRESHOLD",
        "EVAL_SOURCE_SUPPORT_PER_TYPE_BELOW_THRESHOLD",
    }
    fired = [i.code for i in result.issues if i.code in per_type_codes]
    assert fired == []


# --------------------------------------------------------------------- #
# 4. Non-relevant bucket skips silently.
# --------------------------------------------------------------------- #


def test_per_type_skipped_when_relevant_false(tmp_path: Path) -> None:
    """``single_correct_rate`` is MC + TF only; an essay bucket
    carrying ``relevant=False`` must not fire a warning even if the
    score is below the floor."""
    model_dir = tmp_path / "models" / "test-w7d-4"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        single_correct_rate={
            "single_correct_rate": 0.90,
            "per_question_type": {
                "essay": {
                    "single_correct_rate": 0.0,  # would be below 0.85
                    "total_questions": 3,
                    "single_correct_count": 0,
                    "relevant": False,  # skip flag
                },
                "multiple_choice": {
                    "single_correct_rate": 0.95,
                    "total_questions": 20,
                    "single_correct_count": 19,
                    "relevant": True,
                },
            },
        },
    )
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    fired = [
        i.code
        for i in result.issues
        if i.code == "EVAL_SINGLE_CORRECT_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert fired == []


# --------------------------------------------------------------------- #
# 5. Deps-missing branch skips silently.
# --------------------------------------------------------------------- #


def test_per_type_skipped_when_deps_missing(tmp_path: Path) -> None:
    """``bloom_alignment_rate``'s BERT ensemble may be absent; the
    evaluator emits ``per_question_type=None, deps_missing=True`` and
    the gate must skip cleanly without crashing."""
    model_dir = tmp_path / "models" / "test-w7d-5"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        bloom_alignment_rate={
            "bloom_alignment_rate": None,
            "per_question_type": None,
            "deps_missing": True,
        },
    )
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    fired = [
        i.code
        for i in result.issues
        if i.code == "EVAL_BLOOM_ALIGNMENT_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert fired == []
    # And the gate result is still valid (no crash, no MISSING_*).
    assert result.passed is True


# --------------------------------------------------------------------- #
# 6. Decision-capture rationale interpolates per-type below values.
# --------------------------------------------------------------------- #


def test_per_type_below_threshold_in_decision_rationale(tmp_path: Path) -> None:
    """Multiple per-type warnings must surface in the captured
    ``eval_gating_decision`` rationale so a post-hoc audit can replay
    which buckets fell below their floor."""
    class _Capture:
        def __init__(self) -> None:
            self.events: list = []

        def log_decision(self, **kwargs: Any) -> None:
            self.events.append(dict(kwargs))

    model_dir = tmp_path / "models" / "test-w7d-6"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        answerable_rate={
            "answerable_rate": 0.50,
            "per_question_type": {
                "essay": {
                    "answerable_rate": 0.10,
                    "total_questions": 10,
                    "answerable_count": 1,
                    "relevant": True,
                },
            },
        },
        placeholder_rate={
            "placeholder_rate": 0.10,
            "per_question_type": {
                "multiple_choice": {
                    "placeholder_rate": 0.50,  # above 0.05 ceiling
                    "total_questions": 4,
                    "placeholder_count": 2,
                    "relevant": True,
                },
            },
        },
    )
    capture = _Capture()
    EvalGatingValidator().validate({
        "model_dir": str(model_dir),
        "capture": capture,
    })
    decisions = [
        e for e in capture.events
        if e.get("decision_type") == "eval_gating_decision"
    ]
    assert decisions, "eval_gating_decision must be captured"
    rationale = decisions[0]["rationale"]
    # Both below-threshold buckets surfaced in the rationale.
    assert "answerable_rate.essay.answerable_rate" in rationale
    assert "placeholder_rate.multiple_choice.placeholder_rate" in rationale
    assert "below" in rationale or "above" in rationale


# --------------------------------------------------------------------- #
# 7. Aggregate critical thresholds unchanged (back-compat).
# --------------------------------------------------------------------- #


def test_existing_aggregate_critical_thresholds_unchanged(tmp_path: Path) -> None:
    """Adding the per-type warning surface must not alter the corpus-
    wide critical-severity behavior. A faithfulness=0.20 corpus-wide
    score still fails closed."""
    model_dir = tmp_path / "models" / "test-w7d-7"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, faithfulness=0.20)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    critical_codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_FAITHFULNESS_BELOW_THRESHOLD" in critical_codes


# --------------------------------------------------------------------- #
# Bonus: distractor_entropy uses the `distractor_entropy_mean` value
# field — pin the field-name binding so a typo in PER_TYPE_METRIC_CONFIG
# would surface as a regression here rather than silently never-firing.
# --------------------------------------------------------------------- #


def test_per_type_distractor_entropy_uses_mean_field(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-w7d-bonus"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        distractor_entropy={
            "distractor_entropy_mean": 0.80,
            "per_question_type": {
                "multiple_choice": {
                    "distractor_entropy_mean": 0.10,  # below 0.50 floor
                    "total_questions": 8,
                    "relevant": True,
                },
            },
        },
    )
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    fired = [
        i.code
        for i in result.issues
        if i.code == "EVAL_DISTRACTOR_ENTROPY_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert fired == ["EVAL_DISTRACTOR_ENTROPY_PER_TYPE_BELOW_THRESHOLD"]
