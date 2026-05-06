"""Tests for ``BloomAlignmentRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.bloom_alignment_rate import BloomAlignmentRateEvaluator


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
    """Ensemble winner == declared bloom_level → aligned."""
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
    """Ensemble winner != declared bloom_level → mismatched."""
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
    """No loaded members → bloom_alignment_rate is None, deps_missing=True."""
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
