"""Tests for ``AnswerableRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.metrics.answerable_rate import AnswerableRateEvaluator


def test_answerable_case_passes_threshold() -> None:
    """Answer text shares enough tokens with cited chunks → answerable."""
    prompts = [
        {
            "question_id": "q-1",
            "correct_answer": (
                "RDFS describes vocabulary terms with classes and "
                "properties for resources."
            ),
            "source_chunks": ["c1"],
        }
    ]
    chunks = {
        "c1": (
            "RDFS describes vocabulary terms using classes properties "
            "and resources to model knowledge."
        )
    }
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)
    assert result["total_questions"] == 1
    assert result["answerable_count"] == 1
    assert result["answerable_rate"] == 1.0
    assert result["per_question"][0]["answerable"] is True


def test_unanswerable_below_threshold() -> None:
    """Answer text and chunk text barely overlap → unanswerable."""
    prompts = [
        {
            "question_id": "q-1",
            "correct_answer": "alpha beta gamma delta epsilon",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "completely unrelated octopus mountain pizza"}
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)
    assert result["answerable_count"] == 0
    assert result["answerable_rate"] == 0.0
    assert result["per_question"][0]["answerable"] is False


def test_missing_chunks_recorded() -> None:
    """Cited chunk_ids absent from corpus → recorded but doesn't crash."""
    prompts = [
        {
            "question_id": "q-1",
            "correct_answer": "describes vocabulary classes properties",
            "source_chunks": ["c1", "missing-chunk"],
        }
    ]
    chunks = {"c1": "describes vocabulary classes properties resources"}
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)
    assert "missing-chunk" in result["per_question"][0]["missing_chunks"]
    # The present chunk is enough to push the question over threshold.
    assert result["answerable_count"] == 1


def test_empty_answer_returns_zero_overlap() -> None:
    """Empty correct_answer → overlap is 0 → unanswerable."""
    prompts = [
        {
            "question_id": "q-1",
            "correct_answer": "",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "any chunk text whatsoever"}
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)
    assert result["answerable_count"] == 0
    assert result["per_question"][0]["overlap"] == 0.0


def test_fraction_calc_invariant_three_questions() -> None:
    """2 of 3 answerable → answerable_rate ~= 2/3."""
    prompts = [
        {
            "question_id": "q-1",
            "correct_answer": "RDFS describes classes properties",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "q-2",
            "correct_answer": "SHACL validates RDF graphs constraints",
            "source_chunks": ["c2"],
        },
        {
            "question_id": "q-3",
            "correct_answer": "completely irrelevant tokens nowhere near chunk",
            "source_chunks": ["c3"],
        },
    ]
    chunks = {
        "c1": "RDFS describes classes properties resources vocabulary",
        "c2": "SHACL validates RDF graphs constraints shapes",
        "c3": "an entirely different topic about cats and dogs eating",
    }
    result = AnswerableRateEvaluator().evaluate(prompts, chunks)
    assert result["total_questions"] == 3
    assert result["answerable_count"] == 2
    # 2/3 ≈ 0.6667 rounded to 4 places
    assert abs(result["answerable_rate"] - (2 / 3)) < 1e-3
