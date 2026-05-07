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


# ---------------------------------------------------------------------------
# Wave 7 W7.B — per-question-type segmentation tests.
# ---------------------------------------------------------------------------


def test_distractor_entropy_emits_per_question_type_block() -> None:
    """4 prompts (2 MC w/ varied distractors + 2 essay w/ none) → bucket
    rates carry the per-type entropy mean."""
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "distractors": [
                "RDFS describes vocabulary terms classes properties",
                "SHACL validates constraints",
                "OWL extends RDF Schema with rich axioms",
            ],
        },
        {
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "distractors": [
                "alpha beta gamma delta",
                "epsilon zeta",
                "eta theta iota",
            ],
        },
        {
            "question_id": "essay-1",
            "question_type": "essay",
            "distractors": [],  # essays have no distractors
        },
        {
            "question_id": "essay-2",
            "question_type": "essay",
            "distractors": [],
        },
    ]
    result = DistractorEntropyEvaluator().evaluate(prompts)

    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "essay"}

    # MC bucket has non-zero entropy mean.
    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["distractor_entropy_mean"] > 0.5

    # Essay bucket has zero entropy (no distractors).
    assert pqt["essay"]["total_questions"] == 2
    assert pqt["essay"]["distractor_entropy_mean"] == 0.0


def test_distractor_entropy_marks_irrelevant_buckets_as_not_relevant() -> None:
    """distractor_entropy is structurally MC-only — non-MC buckets get
    relevant=False so the W7.D gate consumer can skip them cleanly."""
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "distractors": ["alpha", "beta gamma", "delta epsilon zeta"],
        },
        {
            "question_id": "essay-1",
            "question_type": "essay",
            "distractors": [],
        },
        {
            "question_id": "tf-1",
            "question_type": "true_false",
            "distractors": ["False"],
        },
    ]
    result = DistractorEntropyEvaluator().evaluate(prompts)
    pqt = result["per_question_type"]

    assert pqt["multiple_choice"]["relevant"] is True
    assert pqt["essay"]["relevant"] is False
    assert pqt["true_false"]["relevant"] is False
