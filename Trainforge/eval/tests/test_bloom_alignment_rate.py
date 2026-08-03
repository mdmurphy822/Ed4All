"""Tests for ``BloomAlignmentRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.metrics.bloom_alignment_rate import BloomAlignmentRateEvaluator


def _make_stub_ensemble(verdicts):
    """Build a stub ensemble whose classify() returns canned verdicts.

    ``verdicts`` is a list of strings (winner levels) consumed in order.
    ``_load_members`` returns a non-empty stub list so the evaluator
    proceeds past the deps-missing short-circuit.
    """
    stub = MagicMock()
    stub._load_members.return_value = ["member-stub-1", "member-stub-2"]
    iterator = iter(verdicts)

    def classify(text):
        try:
            level = next(iterator)
        except StopIteration:
            level = "unknown"
        return {
            "winner_level": level,
            "winner_score": 0.85,
            "dispersion": 0.1,
            "per_member": [],
        }

    stub.classify.side_effect = classify
    return stub


def test_aligned_question_counted() -> None:
    """Injected classifier winner == declared bloom_level → aligned."""
    stub = _make_stub_ensemble(["understand"])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {"question_id": "q-1", "stem": "Explain the concept.", "bloom_level": "understand"}
    ]
    result = evaluator.evaluate(prompts)
    assert result["bloom_alignment_rate"] == 1.0
    assert result["aligned_count"] == 1
    assert result["mismatched_count"] == 0
    assert result["deps_missing"] is False


def test_mismatched_question_counted() -> None:
    """Injected classifier winner != declared bloom_level → mismatched."""
    stub = _make_stub_ensemble(["create"])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {"question_id": "q-1", "stem": "Recall the definition.", "bloom_level": "remember"}
    ]
    result = evaluator.evaluate(prompts)
    assert result["bloom_alignment_rate"] == 0.0
    assert result["aligned_count"] == 0
    assert result["mismatched_count"] == 1


def test_mixed_alignment_three_questions() -> None:
    """2 aligned, 1 mismatched → rate = 2/3."""
    stub = _make_stub_ensemble(["understand", "apply", "create"])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {"question_id": "q-1", "stem": "Explain X.", "bloom_level": "understand"},
        {"question_id": "q-2", "stem": "Apply X.", "bloom_level": "apply"},
        {"question_id": "q-3", "stem": "Identify X.", "bloom_level": "remember"},
    ]
    result = evaluator.evaluate(prompts)
    assert result["aligned_count"] == 2
    assert result["mismatched_count"] == 1
    assert abs(result["bloom_alignment_rate"] - (2 / 3)) < 1e-3


def test_deps_missing_graceful_degrade() -> None:
    """No usable classifier members returns the compatibility sentinel."""
    stub = MagicMock()
    stub._load_members.return_value = []
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {"question_id": "q-1", "stem": "anything", "bloom_level": "understand"}
    ]
    result = evaluator.evaluate(prompts)
    assert result["bloom_alignment_rate"] is None
    assert result["deps_missing"] is True


def test_no_declared_bloom_skipped() -> None:
    """Question missing bloom_level → skipped, not aligned/mismatched."""
    stub = _make_stub_ensemble([])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {"question_id": "q-1", "stem": "anything"},
    ]
    result = evaluator.evaluate(prompts)
    assert result["aligned_count"] == 0
    assert result["mismatched_count"] == 0
    assert result["skipped_count"] == 1
    # 0/0 → safe 0.0 fallback
    assert result["bloom_alignment_rate"] == 0.0


# ---------------------------------------------------------------------------
# Per-question-type segmentation.
# ---------------------------------------------------------------------------


def test_bloom_alignment_rate_emits_per_question_type_block() -> None:
    """3 prompts across 2 types — assert per-bucket alignment rates."""
    # mc-1 aligned, mc-2 mismatched, essay-1 aligned.
    stub = _make_stub_ensemble(["understand", "create", "evaluate"])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "Explain X.",
            "bloom_level": "understand",
        },
        {
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "stem": "Recall X.",
            "bloom_level": "remember",
        },
        {
            "question_id": "essay-1",
            "question_type": "essay",
            "stem": "Evaluate X.",
            "bloom_level": "evaluate",
        },
    ]
    result = evaluator.evaluate(prompts)
    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "essay"}

    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["aligned_count"] == 1
    assert pqt["multiple_choice"]["bloom_alignment_rate"] == 0.5
    # bloom_alignment_rate is relevant across all 5 question types.
    assert pqt["multiple_choice"]["relevant"] is True

    assert pqt["essay"]["total_questions"] == 1
    assert pqt["essay"]["aligned_count"] == 1
    assert pqt["essay"]["bloom_alignment_rate"] == 1.0
    assert pqt["essay"]["relevant"] is True


def test_bloom_alignment_rate_deps_missing_returns_per_question_type_none() -> None:
    """Unavailable classifiers emit the compatible per-type sentinel shape."""
    stub = MagicMock()
    stub._load_members.return_value = []
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {
            "question_id": "q-1",
            "question_type": "multiple_choice",
            "stem": "anything",
            "bloom_level": "understand",
        }
    ]
    result = evaluator.evaluate(prompts)
    assert result["bloom_alignment_rate"] is None
    assert result["deps_missing"] is True
    assert "per_question_type" in result
    assert result["per_question_type"] is None


def test_bloom_alignment_rate_skipped_records_excluded_from_bucket() -> None:
    """Records with no declared bloom_level (skipped) should not pollute
    per-bucket rates."""
    # Only aligned/mismatched records are counted; skipped records carry
    # no declared level so they aren't bucketed.
    stub = _make_stub_ensemble(["understand"])
    evaluator = BloomAlignmentRateEvaluator(ensemble=stub)
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "Explain X.",
            "bloom_level": "understand",
        },
        {
            # No bloom_level → skipped.
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "stem": "Anything",
        },
    ]
    result = evaluator.evaluate(prompts)
    pqt = result["per_question_type"]
    # MC bucket only counts the scored record.
    assert pqt["multiple_choice"]["total_questions"] == 1
    assert pqt["multiple_choice"]["aligned_count"] == 1
    assert pqt["multiple_choice"]["bloom_alignment_rate"] == 1.0
