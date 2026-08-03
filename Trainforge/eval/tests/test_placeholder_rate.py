"""Tests for ``PlaceholderRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.metrics.placeholder_rate import PlaceholderRateEvaluator


def test_clean_question_passes() -> None:
    """No placeholder patterns → contaminated=0, rate=0."""
    prompts = [
        {
            "question_id": "q-1",
            "stem": "What does RDFS describe?",
            "correct_answer": "Vocabulary terms via classes and properties.",
            "distractors": [
                "Constraint shapes for graph validation.",
                "An ontology language with rich axioms.",
                "An RDF query language.",
            ],
            "feedback": "RDFS adds class and property semantics to RDF.",
        }
    ]
    result = PlaceholderRateEvaluator().evaluate(prompts)
    assert result["placeholder_rate"] == 0.0
    assert result["contaminated_count"] == 0
    assert result["hit_pattern_histogram"] == {}


def test_contaminated_question_correct_answer_placeholder() -> None:
    """``Correct answer based on content`` → contaminated."""
    prompts = [
        {
            "question_id": "q-1",
            "stem": "What is X?",
            "correct_answer": "Correct answer based on content",
        }
    ]
    result = PlaceholderRateEvaluator().evaluate(prompts)
    assert result["placeholder_rate"] == 1.0
    assert result["contaminated_count"] == 1
    assert any(
        "Correct answer based on content" in pat
        for pat in result["hit_pattern_histogram"]
    )


def test_multi_pattern_hit_aggregates() -> None:
    """Single question hit by two distinct patterns → both in histogram."""
    prompts = [
        {
            "question_id": "q-1",
            "stem": "Statement about RDFS content.",
            "correct_answer": "Correct answer based on content",
            "distractors": ["Plausible distractor A"],
        }
    ]
    result = PlaceholderRateEvaluator().evaluate(prompts)
    assert result["contaminated_count"] == 1
    # Each of the three offenders should land in the histogram.
    assert len(result["hit_pattern_histogram"]) >= 2


def test_pattern_histogram_aggregates_across_corpus() -> None:
    """Two questions hitting same pattern → histogram count = 2."""
    prompts = [
        {
            "question_id": "q-1",
            "stem": "What is X?",
            "correct_answer": "Correct answer based on content",
        },
        {
            "question_id": "q-2",
            "stem": "What is Y?",
            "correct_answer": "Correct answer based on content",
        },
        {
            "question_id": "q-3",
            "stem": "Clean question.",
            "correct_answer": "Clean answer.",
        },
    ]
    result = PlaceholderRateEvaluator().evaluate(prompts)
    assert result["total_questions"] == 3
    assert result["contaminated_count"] == 2
    # 2/3 ≈ 0.6667
    assert abs(result["placeholder_rate"] - (2 / 3)) < 1e-3
    # The "Correct answer based on content" pattern fired twice in total.
    correct_answer_hits = sum(
        v for k, v in result["hit_pattern_histogram"].items()
        if "Correct answer based on content" in k
    )
    assert correct_answer_hits == 2
