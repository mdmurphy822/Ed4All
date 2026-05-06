"""W3.H sub-task H4 — ``AssessmentData.to_dict`` ``source_coverage`` emit.

Pins the canonical W3.H ``source_coverage`` block on the
``Trainforge/generators/assessment_generator.py::AssessmentData.to_dict``
serialisation. Four tests:

* canonical shape (5 fields present, coverage_pct calc, dropped_count
  == sum invariant);
* drop reasons aggregate from ``SkippedItem.reason``;
* ``INTERNAL_DROP_REASON_MISSING`` fires via the helper when the
  histogram is forced out of balance;
* back-compat — every legacy ``to_dict`` field (assessment_id,
  questions, etc.) still surfaces unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentData,
    QuestionData,
    SkippedItem,
)
from lib.governance.source_coverage import (  # noqa: E402
    INTERNAL_DROP_REASON_MISSING,
    build_source_coverage,
)


def _make_question(qid: str, *, bloom: str = "remember") -> QuestionData:
    return QuestionData(
        question_id=qid,
        question_type="multiple_choice",
        stem=f"Question {qid}",
        bloom_level=bloom,
        objective_id="TO-01",
    )


def _make_skipped(qid: str, reason: str) -> SkippedItem:
    return SkippedItem(
        question_id=qid,
        question_type="multiple_choice",
        bloom_level="remember",
        objective_id="TO-01",
        reason=reason,
    )


@pytest.mark.unit
def test_h4_source_coverage_canonical_shape() -> None:
    """3 questions emitted + 2 skipped => 5 attempted, coverage 0.6."""
    a = AssessmentData(
        assessment_id="ASM-H4-001",
        title="H4 Test",
        course_code="H4_COURSE",
        questions=[_make_question(f"q{i}") for i in range(3)],
        skipped_items=[
            _make_skipped("s1", "no_source_chunks"),
            _make_skipped("s2", "no_extractable_content"),
        ],
    )
    doc = a.to_dict()
    assert "source_coverage" in doc
    block = doc["source_coverage"]
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    assert block["consumed_count"] == 5
    assert block["emitted_count"] == 3
    assert block["dropped_count"] == 2
    assert block["coverage_pct"] == pytest.approx(0.6, rel=1e-3)
    assert block["drop_reasons"]["no_source_chunks"] == 1
    assert block["drop_reasons"]["no_extractable_content"] == 1
    assert block["dropped_count"] == sum(block["drop_reasons"].values())


@pytest.mark.unit
def test_h4_drop_reasons_histogram_aggregates() -> None:
    """Multiple skips of the same reason aggregate into one bucket."""
    a = AssessmentData(
        assessment_id="ASM-H4-002",
        title="H4 Histogram",
        course_code="H4_COURSE",
        questions=[_make_question("q1")],
        skipped_items=[
            _make_skipped("s1", "no_source_chunks"),
            _make_skipped("s2", "no_source_chunks"),
            _make_skipped("s3", "no_source_chunks"),
            _make_skipped("s4", "padded_distractor_fallback_eliminated"),
        ],
    )
    doc = a.to_dict()
    block = doc["source_coverage"]
    assert block["drop_reasons"]["no_source_chunks"] == 3
    assert block["drop_reasons"]["padded_distractor_fallback_eliminated"] == 1
    assert block["dropped_count"] == 4


@pytest.mark.unit
def test_h4_internal_drop_reason_missing_fires() -> None:
    """The silent-gaming check fires via the helper when histogram total < dropped."""
    import logging

    rec: list[logging.LogRecord] = []
    h = logging.Handler()
    h.emit = lambda r: rec.append(r)
    lg = logging.getLogger("lib.governance.source_coverage")
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        block = build_source_coverage(
            consumed_count=10,
            emitted_count=4,
            drop_reasons={"no_source_chunks": 2},
            dropped_count=6,
            label="assessment_data:ASM-H4-FORCED",
        )
    finally:
        lg.removeHandler(h)
    assert block["drop_reasons"][INTERNAL_DROP_REASON_MISSING] == 4
    assert block["dropped_count"] == sum(block["drop_reasons"].values())
    assert any(
        "assessment_data:ASM-H4-FORCED" in (r.getMessage() or "")
        for r in rec
    )


@pytest.mark.unit
def test_h4_back_compat_legacy_fields_preserved() -> None:
    """Every pre-W3.H ``to_dict`` field is still emitted (no breaking change)."""
    a = AssessmentData(
        assessment_id="ASM-LEGACY",
        title="Legacy Shape",
        course_code="LEGACY",
        questions=[_make_question("q1"), _make_question("q2")],
        objectives_targeted=["TO-01"],
        bloom_levels=["remember"],
    )
    doc = a.to_dict()
    for required in (
        "assessment_id",
        "title",
        "course_code",
        "questions",
        "objectives_targeted",
        "bloom_levels",
        "created_at",
        "status",
        "question_count",
        "total_points",
        "skipped_items_count",
        "skipped_items_summary",
    ):
        assert required in doc, f"legacy field {required!r} missing from to_dict"
    # And the new W3.H field is additive (legacy callers ignore it).
    assert "source_coverage" in doc


@pytest.mark.unit
def test_h4_zero_attempts_zero_coverage() -> None:
    """Assessment with no questions or skips (degenerate) → coverage 0.0."""
    a = AssessmentData(
        assessment_id="ASM-EMPTY",
        title="Empty",
        course_code="EMPTY",
    )
    doc = a.to_dict()
    block = doc["source_coverage"]
    assert block["consumed_count"] == 0
    assert block["emitted_count"] == 0
    assert block["coverage_pct"] == 0.0
