"""QuestionData observed-Bloom serialization contract tests.

Pins three contracts on
``Trainforge/generators/assessment/generator.py::QuestionData``:

  1. The optional fields (``observed_bloom_level``, ``bloom_alignment``)
     default to ``None`` when not supplied, so existing constructor calls
     keep working.
  2. ``to_dict()`` round-trips both fields unconditionally (None is an
     acceptable shape for a not-yet-classified question; the consumer
     decides on filter semantics).
  3. The two fields accept canonical Bloom levels and boolean alignment
     values respectively.
"""

from __future__ import annotations

import pytest

from Trainforge.generators.assessment.generator import QuestionData


def _legacy_question(**overrides) -> QuestionData:
    kwargs = dict(
        question_id="q-001",
        question_type="multiple_choice",
        stem="What does sh:datatype constrain?",
        bloom_level="understand",
        objective_id="TO-01",
    )
    kwargs.update(overrides)
    return QuestionData(**kwargs)


def test_legacy_constructor_call_still_works():
    """Omitted optional fields default to None."""
    q = _legacy_question()
    assert q.observed_bloom_level is None
    assert q.bloom_alignment is None


def test_to_dict_round_trips_none_defaults():
    """to_dict() emits the two new keys unconditionally — None is an
    acceptable shape for a not-yet-classified question."""
    q = _legacy_question()
    d = q.to_dict()
    assert "observed_bloom_level" in d
    assert "bloom_alignment" in d
    assert d["observed_bloom_level"] is None
    assert d["bloom_alignment"] is None


@pytest.mark.parametrize(
    "level",
    ["remember", "understand", "apply", "analyze", "evaluate", "create"],
)
def test_to_dict_round_trips_canonical_observed_bloom_level(level):
    q = _legacy_question(observed_bloom_level=level, bloom_alignment=True)
    d = q.to_dict()
    assert d["observed_bloom_level"] == level
    assert d["bloom_alignment"] is True


@pytest.mark.parametrize("alignment", [True, False])
def test_to_dict_round_trips_bloom_alignment(alignment):
    q = _legacy_question(observed_bloom_level="apply", bloom_alignment=alignment)
    d = q.to_dict()
    assert d["bloom_alignment"] is alignment


def test_to_dict_preserves_existing_keys():
    """Observed-Bloom keys are additive to the established serialized shape."""
    q = _legacy_question(
        observed_bloom_level="apply",
        bloom_alignment=True,
        feedback="Good job",
    )
    d = q.to_dict()
    expected_keys = {
        "question_id",
        "question_type",
        "stem",
        "bloom_level",
        "objective_id",
        "choices",
        "correct_answer",
        "points",
        "feedback",
        "source_chunks",
        "generation_rationale",
        "observed_bloom_level",
        "bloom_alignment",
    }
    assert set(d.keys()) == expected_keys
