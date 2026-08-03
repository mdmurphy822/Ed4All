"""Tests for ``SourceSupportRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.metrics.source_support_rate import SourceSupportRateEvaluator


class _StubScore:
    def __init__(self, entailment: float, neutral: float, contradiction: float) -> None:
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


def _make_stub_classifier(verdicts):
    """Stub NLI classifier returning canned (ent, neu, con) scores."""
    iterator = iter(verdicts)

    def score_pair(*, premise, hypothesis):
        ent, neu, con = next(iterator)
        return _StubScore(ent, neu, con)

    stub = MagicMock()
    stub.score_pair.side_effect = score_pair
    return stub


def test_supported_question_counted() -> None:
    """High entailment + low contradiction → supported."""
    stub = _make_stub_classifier([(0.95, 0.04, 0.01)])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [
        {
            "question_id": "q-1",
            "stem": "What does RDFS describe?",
            "correct_answer": "Vocabulary terms.",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "RDFS describes vocabulary terms via classes and properties."}
    result = evaluator.evaluate(prompts, chunks)
    assert result["source_support_rate"] == 1.0
    assert result["supported_count"] == 1
    assert result["contradicted_count"] == 0
    assert result["per_question"][0]["outcome"] == "supported"


def test_contradicted_question_counted() -> None:
    """High contradiction → contradicted."""
    stub = _make_stub_classifier([(0.05, 0.05, 0.90)])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [
        {
            "question_id": "q-1",
            "stem": "Is X true?",
            "correct_answer": "Yes",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "X is definitely false."}
    result = evaluator.evaluate(prompts, chunks)
    assert result["supported_count"] == 0
    assert result["contradicted_count"] == 1
    assert result["per_question"][0]["outcome"] == "contradicted"


def test_unsupported_in_between() -> None:
    """Neutral score (low ent + low con) → unsupported but not contradicted."""
    stub = _make_stub_classifier([(0.30, 0.65, 0.05)])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [
        {
            "question_id": "q-1",
            "stem": "Tell me about Y",
            "correct_answer": "Something tangential",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "Unrelated topic about Z."}
    result = evaluator.evaluate(prompts, chunks)
    assert result["supported_count"] == 0
    assert result["contradicted_count"] == 0
    assert result["per_question"][0]["outcome"] == "unsupported"


def test_deps_missing_graceful_degrade(monkeypatch) -> None:
    """When NliClassifier.get_or_load returns None → deps_missing=True."""
    from lib.classifiers import nli_classifier as nc_mod

    monkeypatch.setattr(
        nc_mod.NliClassifier,
        "get_or_load",
        classmethod(lambda cls: None),
    )
    evaluator = SourceSupportRateEvaluator()  # classifier=None → loader path
    prompts = [
        {
            "question_id": "q-1",
            "stem": "Anything",
            "correct_answer": "Anything",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "Some text."}
    result = evaluator.evaluate(prompts, chunks)
    assert result["source_support_rate"] is None
    assert result["deps_missing"] is True


def test_no_source_chunks_skipped() -> None:
    """Question with no source_chunks → skipped, not counted."""
    stub = _make_stub_classifier([])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [{"question_id": "q-1", "stem": "Hi", "correct_answer": "Hello"}]
    chunks: dict = {}
    result = evaluator.evaluate(prompts, chunks)
    assert result["skipped_count"] == 1
    assert result["supported_count"] == 0
    # Skipping all → safe 0.0 fallback
    assert result["source_support_rate"] == 0.0


# ---------------------------------------------------------------------------
# Wave 7 W7.B — per-question-type segmentation tests.
# ---------------------------------------------------------------------------


def test_source_support_rate_emits_per_question_type_block() -> None:
    """3 prompts (2 MC supported + 1 essay contradicted) → per-bucket rates."""
    # mc-1 supported, mc-2 supported, essay-1 contradicted.
    stub = _make_stub_classifier([
        (0.95, 0.04, 0.01),  # mc-1 supported
        (0.90, 0.05, 0.05),  # mc-2 supported
        (0.05, 0.05, 0.90),  # essay-1 contradicted
    ])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "What does RDFS describe?",
            "correct_answer": "Vocabulary terms.",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "stem": "What validates RDF?",
            "correct_answer": "SHACL.",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "essay-1",
            "question_type": "essay",
            "stem": "Is X true?",
            "correct_answer": "Yes",
            "source_chunks": ["c1"],
        },
    ]
    chunks = {"c1": "RDFS describes vocabulary terms via classes and properties."}
    result = evaluator.evaluate(prompts, chunks)

    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "essay"}

    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["supported_count"] == 2
    assert pqt["multiple_choice"]["source_support_rate"] == 1.0
    # source_support_rate is relevant across all 5 question types.
    assert pqt["multiple_choice"]["relevant"] is True

    assert pqt["essay"]["total_questions"] == 1
    assert pqt["essay"]["supported_count"] == 0
    assert pqt["essay"]["source_support_rate"] == 0.0
    assert pqt["essay"]["relevant"] is True


def test_source_support_rate_deps_missing_returns_per_question_type_none(
    monkeypatch,
) -> None:
    """Deps-missing branch must emit per_question_type=None + deps_missing=True."""
    from lib.classifiers import nli_classifier as nc_mod

    monkeypatch.setattr(
        nc_mod.NliClassifier,
        "get_or_load",
        classmethod(lambda cls: None),
    )
    evaluator = SourceSupportRateEvaluator()  # classifier=None → loader path
    prompts = [
        {
            "question_id": "q-1",
            "question_type": "multiple_choice",
            "stem": "Anything",
            "correct_answer": "Anything",
            "source_chunks": ["c1"],
        }
    ]
    chunks = {"c1": "Some text."}
    result = evaluator.evaluate(prompts, chunks)
    assert result["source_support_rate"] is None
    assert result["deps_missing"] is True
    assert "per_question_type" in result
    assert result["per_question_type"] is None


def test_source_support_rate_skipped_records_excluded_from_bucket() -> None:
    """Records skipped (no chunks / empty hypothesis) should not contribute
    to per-bucket rates."""
    stub = _make_stub_classifier([
        (0.95, 0.04, 0.01),  # only mc-1 gets scored
    ])
    evaluator = SourceSupportRateEvaluator(classifier=stub)
    prompts = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "What does RDFS describe?",
            "correct_answer": "Vocabulary terms.",
            "source_chunks": ["c1"],
        },
        {
            # No source_chunks → skipped.
            "question_id": "mc-2",
            "question_type": "multiple_choice",
            "stem": "Hi",
            "correct_answer": "Hello",
        },
    ]
    chunks = {"c1": "RDFS describes vocabulary terms."}
    result = evaluator.evaluate(prompts, chunks)
    pqt = result["per_question_type"]
    # MC bucket only counts the scored record (not the skipped one).
    assert pqt["multiple_choice"]["total_questions"] == 1
    assert pqt["multiple_choice"]["supported_count"] == 1
    assert pqt["multiple_choice"]["source_support_rate"] == 1.0
