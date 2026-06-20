"""Tests for the pure answer-completeness analysis (PIECE B).

No network, no model load, no embedding call — every assertion is over the
deterministic lexical core in :mod:`lib.retrieval.answer_completeness`.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.retrieval.answer_completeness import (
    CompletenessResult,
    analyze_completeness,
    part_addressed_by_answer,
    part_grounded_in_passages,
    split_question_parts,
)


@dataclass(frozen=True)
class _Passage:
    """Minimal duck-typed passage (``.text`` + ``.chunk_id``)."""

    chunk_id: str
    text: str


# Synthetic passages: one covers rectangle perimeter, one covers circle
# circumference. Built so token-coverage of the matching question half clears
# the 0.80 grounding floor.
_RECT_PASSAGE = _Passage(
    chunk_id="chunk-rect",
    text=(
        "To find the perimeter of a rectangle, add the lengths of all four "
        "sides: perimeter equals two times length plus two times width."
    ),
)
_CIRCLE_PASSAGE = _Passage(
    chunk_id="chunk-circle",
    text=(
        "To find the circumference of a circle, multiply the diameter by pi, "
        "or use two times pi times the radius. The circumference is the "
        "distance around the circle."
    ),
)

_MOTIVATING_QUERY = (
    "how do i find the perimeter of a rectangle? "
    "the circumference of a circle?"
)


# --------------------------------------------------------------------------- #
# (a) two-part split of the motivating question
# --------------------------------------------------------------------------- #


def test_split_two_part_motivating_question():
    parts = split_question_parts(_MOTIVATING_QUERY)
    assert len(parts) == 2
    # First part keeps the rectangle subject.
    assert "perimeter" in parts[0].lower()
    assert "rectangle" in parts[0].lower()
    # Second part carries the circle subject (stem carry is cosmetic; the
    # content tokens are what matter).
    assert "circumference" in parts[1].lower()
    assert "circle" in parts[1].lower()


# --------------------------------------------------------------------------- #
# (b) single-part question → 1 part / is_multipart False
# --------------------------------------------------------------------------- #


def test_split_single_part_question():
    parts = split_question_parts("What is the area of a triangle?")
    assert len(parts) == 1
    assert "area" in parts[0].lower()


def test_single_part_result_is_not_multipart():
    result = analyze_completeness(
        "What is the area of a triangle?",
        "The area of a triangle is one half base times height.",
        [_RECT_PASSAGE, _CIRCLE_PASSAGE],
    )
    assert isinstance(result, CompletenessResult)
    assert result.is_multipart is False
    assert len(result.parts) == 1


def test_split_does_not_over_split_plain_and():
    # A single coordinated clause that is NOT two interrogatives should stay
    # whole (under-split bias) — "rectangle and a square" is one subject list.
    parts = split_question_parts(
        "How do I find the perimeter of a rectangle and a square?"
    )
    assert len(parts) == 1


# --------------------------------------------------------------------------- #
# (c) part_addressed_by_answer
# --------------------------------------------------------------------------- #


def test_part_addressed_true_when_answer_contains_part():
    part = "how do i find the perimeter of a rectangle"
    answer = (
        "To find the perimeter of a rectangle you add up the lengths of all "
        "four sides, which is two times the length plus two times the width."
    )
    addressed, signals = part_addressed_by_answer(part, answer)
    assert addressed is True
    assert signals["token_coverage"] >= signals["min_token_coverage"]


def test_part_addressed_false_when_answer_omits_part():
    part = "how do i find the circumference of a circle"
    # Answer is entirely about rectangles — circumference/circle absent.
    answer = (
        "To find the perimeter of a rectangle you add up the lengths of all "
        "four sides, which is two times the length plus two times the width."
    )
    addressed, signals = part_addressed_by_answer(part, answer)
    assert addressed is False
    assert signals["token_coverage"] < signals["min_token_coverage"]


# --------------------------------------------------------------------------- #
# (d) part_grounded_in_passages
# --------------------------------------------------------------------------- #


def test_part_grounded_true_when_a_passage_covers_it():
    part = "how do i find the circumference of a circle"
    grounded, signals = part_grounded_in_passages(part, [_RECT_PASSAGE, _CIRCLE_PASSAGE])
    assert grounded is True
    assert signals["best_chunk_id"] == "chunk-circle"
    assert signals["best_token_coverage"] >= signals["min_token_coverage"]


def test_part_grounded_false_when_no_passage_covers_it():
    part = "how do i compute the standard deviation of a sample"
    grounded, signals = part_grounded_in_passages(part, [_RECT_PASSAGE, _CIRCLE_PASSAGE])
    assert grounded is False
    assert signals["best_token_coverage"] < signals["min_token_coverage"]


# --------------------------------------------------------------------------- #
# (e) end-to-end analyze_completeness
# --------------------------------------------------------------------------- #


def test_analyze_flags_circumference_uncovered_but_grounded():
    # Answer covers ONLY the rectangle half; a passage covers circumference.
    answer = (
        "To find the perimeter of a rectangle, add the lengths of all four "
        "sides: that is two times the length plus two times the width."
    )
    result = analyze_completeness(
        _MOTIVATING_QUERY,
        answer,
        [_RECT_PASSAGE, _CIRCLE_PASSAGE],
    )
    assert result.is_multipart is True
    flagged = result.uncovered_but_grounded
    assert len(flagged) == 1
    assert "circumference" in flagged[0].lower()
    assert "circle" in flagged[0].lower()


def test_analyze_flags_nothing_when_answer_covers_both():
    answer = (
        "To find the perimeter of a rectangle, add the lengths of all four "
        "sides: two times length plus two times width. "
        "To find the circumference of a circle, multiply the diameter by pi, "
        "or use two times pi times the radius."
    )
    result = analyze_completeness(
        _MOTIVATING_QUERY,
        answer,
        [_RECT_PASSAGE, _CIRCLE_PASSAGE],
    )
    assert result.is_multipart is True
    assert result.uncovered_but_grounded == []


def test_analyze_flags_nothing_when_uncovered_part_is_ungrounded():
    # Answer covers ONLY the rectangle half, and NO passage covers the
    # circumference half → ungrounded → NOT a re-ask trigger.
    answer = (
        "To find the perimeter of a rectangle, add the lengths of all four "
        "sides: two times length plus two times width."
    )
    result = analyze_completeness(
        _MOTIVATING_QUERY,
        answer,
        [_RECT_PASSAGE],  # only the rectangle passage — circle is ungrounded
    )
    assert result.is_multipart is True
    assert result.uncovered_but_grounded == []
