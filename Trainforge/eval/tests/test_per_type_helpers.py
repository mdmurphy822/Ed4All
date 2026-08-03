"""Wave 7 W7.A tests — shared per-question-type segmentation helper +
prompt-loader threading + answerable_rate / placeholder_rate per-type
extensions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.metrics._per_type_helpers import (
    RELEVANT_QUESTION_TYPES,
    attach_relevance,
    bucket_per_question_records,
    normalize_question_type,
)
from Trainforge.eval.metrics.answerable_rate import AnswerableRateEvaluator
from Trainforge.eval.metrics.placeholder_rate import PlaceholderRateEvaluator


# ---------------------------------------------------------------------------
# 1. normalize_question_type — generator's canonical question_type field.
# ---------------------------------------------------------------------------
def test_normalize_question_type_resolves_canonical_field() -> None:
    """Generator emit shape with ``question_type`` resolves to lower."""
    result = normalize_question_type({"question_type": "Multiple_Choice"})
    assert result == "multiple_choice"


# ---------------------------------------------------------------------------
# 2. normalize_question_type — defensive QTI ``type`` fallback.
# ---------------------------------------------------------------------------
def test_normalize_question_type_resolves_qti_field() -> None:
    """QTI-style dict missing ``question_type`` falls back to ``type``."""
    result = normalize_question_type({"type": "Essay"})
    assert result == "essay"


# ---------------------------------------------------------------------------
# 3. bucket_per_question_records — parallel-indexed bucketing over a 5-item
#    pair of records / types reduces to 3 distinct buckets.
# ---------------------------------------------------------------------------
def test_bucket_per_question_records_aligns_by_index() -> None:
    """5 records across 3 types lands in 3 correctly-populated buckets."""
    records = [
        {"question_id": "q-1"},
        {"question_id": "q-2"},
        {"question_id": "q-3"},
        {"question_id": "q-4"},
        {"question_id": "q-5"},
    ]
    types = ["multiple_choice", "essay", "multiple_choice", "true_false", "essay"]
    buckets = bucket_per_question_records(records, question_types=types)

    assert set(buckets.keys()) == {"multiple_choice", "essay", "true_false"}
    assert [r["question_id"] for r in buckets["multiple_choice"]] == ["q-1", "q-3"]
    assert [r["question_id"] for r in buckets["essay"]] == ["q-2", "q-5"]
    assert [r["question_id"] for r in buckets["true_false"]] == ["q-4"]


# ---------------------------------------------------------------------------
# 4. bucket_per_question_records — defensive length-mismatch contract.
# ---------------------------------------------------------------------------
def test_bucket_raises_on_length_mismatch() -> None:
    """Records and question_types disagreeing on length raises."""
    with pytest.raises(ValueError, match="length mismatch"):
        bucket_per_question_records(
            [{"a": 1}, {"a": 2}],
            question_types=["multiple_choice"],
        )


# ---------------------------------------------------------------------------
# 5. attach_relevance — distractor_entropy is structurally MC-only.
# ---------------------------------------------------------------------------
def test_attach_relevance_marks_distractor_entropy_mc_only() -> None:
    """``distractor_entropy`` flags MC bucket relevant, others not."""
    per_question_type = {
        "multiple_choice": {"distractor_entropy": 0.8},
        "essay": {"distractor_entropy": 0.0},
        "true_false": {"distractor_entropy": 0.5},
    }
    out = attach_relevance(per_question_type, metric_name="distractor_entropy")

    assert out["multiple_choice"]["relevant"] is True
    assert out["essay"]["relevant"] is False
    assert out["true_false"]["relevant"] is False
    # Sanity-check the canonical relevance table for the same metric.
    assert RELEVANT_QUESTION_TYPES["distractor_entropy"] == {"multiple_choice"}


# ---------------------------------------------------------------------------
# 6. _load_prompts_for_metrics threads question_type through the projection.
# ---------------------------------------------------------------------------
def test_load_prompts_for_metrics_threads_question_type(tmp_path: Path) -> None:
    """A 3-question fixture round-trips question_type values via the
    harness prompt-loader projection."""
    from Trainforge.eval.runners.slm_eval_harness import SLMEvalHarness

    course_dir = tmp_path / "course"
    course_dir.mkdir()
    assessments = {
        "questions": [
            {
                "question_id": "q-mc",
                "question_type": "multiple_choice",
                "stem": "MC stem",
                "correct_answer": "alpha",
                "distractors": ["beta"],
            },
            {
                "question_id": "q-essay",
                # QTI-flavored: ``type`` rather than ``question_type``.
                "type": "essay",
                "stem": "Essay stem",
            },
            {
                "question_id": "q-untyped",
                "stem": "Untyped stem",
            },
        ],
    }
    (course_dir / "assessments.json").write_text(
        json.dumps(assessments), encoding="utf-8"
    )

    harness = SLMEvalHarness.__new__(SLMEvalHarness)
    harness.course_path = course_dir  # type: ignore[attr-defined]
    prompts = harness._load_prompts_for_metrics()

    assert len(prompts) == 3
    by_id = {p["question_id"]: p for p in prompts}
    assert by_id["q-mc"]["question_type"] == "multiple_choice"
    assert by_id["q-essay"]["question_type"] == "essay"
    assert by_id["q-untyped"]["question_type"] == ""


# ---------------------------------------------------------------------------
# 7. AnswerableRateEvaluator emits per_question_type with correct rates.
# ---------------------------------------------------------------------------
def test_answerable_rate_emits_per_question_type_block() -> None:
    """4 prompts (2 MC fully answerable + 2 essay unanswerable) →
    per_question_type carries both buckets with correct per-bucket rates."""
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "correct_answer": "RDFS describes vocabulary terms classes properties resources",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "correct_answer": "RDFS describes vocabulary terms classes properties resources",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "essay-1",
            "question_type": "essay",
            "correct_answer": "alpha beta gamma delta epsilon",
            "source_chunks": ["c2"],
        },
        {
            "question_id": "essay-2",
            "question_type": "essay",
            "correct_answer": "alpha beta gamma delta epsilon",
            "source_chunks": ["c2"],
        },
    ]
    chunks = {
        "c1": "RDFS describes vocabulary terms using classes properties resources knowledge",
        "c2": "completely unrelated octopus mountain pizza canyon",
    }
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)

    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "essay"}

    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["answerable_count"] == 2
    assert pqt["multiple_choice"]["answerable_rate"] == 1.0
    assert pqt["multiple_choice"]["relevant"] is True

    assert pqt["essay"]["total_questions"] == 2
    assert pqt["essay"]["answerable_count"] == 0
    assert pqt["essay"]["answerable_rate"] == 0.0
    assert pqt["essay"]["relevant"] is True


# ---------------------------------------------------------------------------
# 8. PlaceholderRateEvaluator emits per_question_type with correct rates.
# ---------------------------------------------------------------------------
def test_placeholder_rate_emits_per_question_type_block() -> None:
    """4 prompts (2 MC clean, 2 essay with one carrying a placeholder
    pattern) → per_question_type[essay].placeholder_rate == 0.5."""
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "What does RDFS define?",
            "correct_answer": "Classes and properties.",
            "distractors": ["A constraint language.", "A query language.", "A vocabulary."],
            "feedback": "RDFS introduces class and property semantics.",
        },
        {
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "stem": "Which standard validates RDF graphs?",
            "correct_answer": "SHACL.",
            "distractors": ["RDFS.", "OWL.", "SPARQL."],
            "feedback": "SHACL validates structural constraints.",
        },
        {
            "question_id": "essay-clean",
            "question_type": "essay",
            "stem": "Explain how RDFS extends RDF.",
            "correct_answer": "RDFS adds class hierarchy and property semantics on top of RDF triples.",
            "feedback": "Look for class / property axioms in the answer.",
        },
        {
            "question_id": "essay-dirty",
            "question_type": "essay",
            "stem": "Discuss SHACL constraints.",
            # Placeholder pattern: matches the canonical
            # "Correct answer based on content" regex.
            "correct_answer": "Correct answer based on content from the lecture.",
            "feedback": "Address each constraint kind.",
        },
    ]
    result = PlaceholderRateEvaluator().evaluate(prompts)

    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "essay"}

    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["placeholder_count"] == 0
    assert pqt["multiple_choice"]["placeholder_rate"] == 0.0
    assert pqt["multiple_choice"]["relevant"] is True

    assert pqt["essay"]["total_questions"] == 2
    assert pqt["essay"]["placeholder_count"] == 1
    assert pqt["essay"]["placeholder_rate"] == 0.5
    assert pqt["essay"]["relevant"] is True
