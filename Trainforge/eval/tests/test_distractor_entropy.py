"""Tests for ``DistractorEntropyEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.distractor_entropy import DistractorEntropyEvaluator


def test_high_entropy_distractors_well_spread() -> None:
    """Three varied-length distractors → entropy well above 0.5."""
    prompts = [
        {
            "question_id": "q-1",
            "distractors": [
                "RDFS describes vocabulary terms using classes and properties",
                "SHACL validates constraints",
                "OWL extends RDF Schema with rich axioms for ontologies",
            ],
        }
    ]
    result = DistractorEntropyEvaluator().evaluate(prompts)
    # Three distractors with different sizes should have entropy > 0.5
    assert result["mean_distractor_entropy"] > 0.5
    assert result["low_entropy_count"] == 0


def test_low_entropy_collapsed_distractors() -> None:
    """Single distractor → entropy 0 → low-entropy."""
    prompts = [
        {
            "question_id": "q-1",
            "distractors": ["only one distractor here"],
        }
    ]
    result = DistractorEntropyEvaluator().evaluate(prompts)
    # Single bucket → entropy 0
    assert result["mean_distractor_entropy"] == 0.0
    assert result["low_entropy_count"] == 1
    assert result["per_question"][0]["low_entropy"] is True


def test_zero_distractors_yields_zero_entropy() -> None:
    """Missing / empty distractors list → entropy 0, counted as low."""
    prompts = [{"question_id": "q-1", "distractors": []}]
    result = DistractorEntropyEvaluator().evaluate(prompts)
    assert result["mean_distractor_entropy"] == 0.0
    assert result["low_entropy_count"] == 1
    assert result["per_question"][0]["distractor_count"] == 0


def test_single_distractor_low_entropy_counted() -> None:
    """Three uniform-length distractors → entropy ~= log(3) ≈ 1.10 (high)."""
    prompts = [
        {
            "question_id": "q-1",
            "distractors": ["alpha bravo", "charlie delta", "echo foxtrot"],
        }
    ]
    result = DistractorEntropyEvaluator().evaluate(prompts)
    # All three buckets equal-size 2 → entropy = log(3) ≈ 1.0986
    import math
    assert abs(result["mean_distractor_entropy"] - math.log(3)) < 1e-3
    assert result["low_entropy_count"] == 0
