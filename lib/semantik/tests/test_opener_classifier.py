"""A7 — pedagogical-opener classifier (standalone + leading-split).

Synthetic strings only (no corpus files).
"""
from __future__ import annotations

from lib.semantik.opener_classifier import (
    ROLE_OBJECTIVES,
    ROLE_READINESS,
    ROLE_SOLUTION,
    ROLE_TRY_IT,
    ROLE_WORKED_EXAMPLE,
    classify_opener_label,
    split_interior_openers,
    split_leading_opener,
)


# ---------------------------------------------------------------------------
# classify_opener_label — standalone labels.
# ---------------------------------------------------------------------------


def test_standalone_example_numbered():
    assert classify_opener_label("EXAMPLE 9.1") == ("Example 9.1", ROLE_WORKED_EXAMPLE)


def test_standalone_try_it_numbered():
    assert classify_opener_label("TRY IT 9.7") == ("Try It 9.7", ROLE_TRY_IT)


def test_standalone_be_prepared():
    assert classify_opener_label("BE PREPARED 9.1") == (
        "Be Prepared 9.1",
        ROLE_READINESS,
    )


def test_standalone_learning_objectives():
    assert classify_opener_label("Learning Objectives") == (
        "Learning Objectives",
        ROLE_OBJECTIVES,
    )


def test_standalone_solution_and_gutter_glyph():
    assert classify_opener_label("Solution") == ("Solution", ROLE_SOLUTION)
    # Up to 3 leading OCR gutter glyphs tolerated.
    assert classify_opener_label(") Solution") == ("Solution", ROLE_SOLUTION)


def test_standalone_rejects_prose():
    # A sentence merely containing the word is NOT a standalone label.
    assert classify_opener_label("Examples are common in algebra.") is None
    assert classify_opener_label("Find the solution set of the inequality") is None
    assert classify_opener_label("") is None
    assert classify_opener_label(None) is None


# ---------------------------------------------------------------------------
# split_leading_opener — fused opener + trailing prose.
# ---------------------------------------------------------------------------


def test_split_leading_example():
    got = split_leading_opener("EXAMPLE 9.1 Simplify: (a) sqrt 36 (b) sqrt 196")
    assert got == (
        "Example 9.1",
        ROLE_WORKED_EXAMPLE,
        "Simplify: (a) sqrt 36 (b) sqrt 196",
    )


def test_split_leading_objectives():
    got = split_leading_opener(
        "Learning Objectives By the end of this section you will be able to"
    )
    assert got is not None
    assert got[0] == "Learning Objectives"
    assert got[1] == ROLE_OBJECTIVES
    assert got[2].startswith("By the end")


def test_split_leading_solution():
    got = split_leading_opener("Solution (a) Since 6 squared is 36 the root is 6")
    assert got is not None
    assert got[0] == "Solution"
    assert got[1] == ROLE_SOLUTION


def test_split_leading_requires_number_for_numbered_opener():
    # "Example shows ..." (no number, lowercase remainder) must NOT split.
    assert split_leading_opener("Example shows that squares are positive") is None


def test_split_leading_rejects_lowercase_label():
    # A lowercase mid-sentence usage is never a heading.
    assert split_leading_opener("try it again later when ready to") is None


def test_split_leading_rejects_trivial_remainder():
    # Too few remainder words (1 < _MIN_REMAINDER_WORDS) → the standalone case.
    assert split_leading_opener("TRY IT 9.1 now") is None


# ---------------------------------------------------------------------------
# split_interior_openers — ITEM 4 (round-2 audit): interior TRY-IT/EXAMPLE
# fusion split.
# ---------------------------------------------------------------------------


def test_interior_split_multi_marker_fusion():
    parts = split_interior_openers(
        "Combine like radicals. TRY IT 9.201 Simplify: some prose here. "
        "TRY IT 9.202 Simplify: more prose. EXAMPLE 9.102 Simplify: final prose."
    )
    assert parts is not None
    assert parts[0] == ("text", "Combine like radicals.")
    openers = [p for p in parts if p[0] == "opener"]
    assert [(p[1], p[2]) for p in openers] == [
        ("Try It 9.201", ROLE_TRY_IT),
        ("Try It 9.202", ROLE_TRY_IT),
        ("Example 9.102", ROLE_WORKED_EXAMPLE),
    ]


def test_interior_split_leading_marker_with_second_fused():
    # A block STARTING with an opener still splits when >= 2 markers are fused.
    parts = split_interior_openers(
        "TRY IT 9.199 Simplify this expression. TRY IT 9.200 Simplify that one."
    )
    assert parts is not None
    assert parts[0] == ("opener", "Try It 9.199", ROLE_TRY_IT)
    assert ("opener", "Try It 9.200", ROLE_TRY_IT) in parts


def test_interior_split_single_leading_marker_owned_by_leading_path():
    # A lone LEADING opener (no preceding prose) belongs to split_leading_opener.
    assert split_interior_openers("TRY IT 9.1 Simplify the expression now.") is None


def test_interior_split_rejects_lowercase_for_example():
    # "for example 9.2" mid-sentence must never be a split point.
    assert (
        split_interior_openers(
            "We compute this, for example 9.2 shows the ratio clearly here."
        )
        is None
    )


def test_interior_split_requires_decimal_number():
    assert (
        split_interior_openers(
            "Combine like radicals. TRY IT Simplify this expression now."
        )
        is None
    )


def test_interior_split_masks_math_spans():
    # A marker-shaped token INSIDE $…$ math is never a split point.
    assert (
        split_interior_openers(
            "The result equals $x + EXAMPLE 9.1$ in the masked span here."
        )
        is None
    )


def test_interior_split_noop_on_plain_prose():
    assert split_interior_openers("Just ordinary prose with no markers.") is None
    assert split_interior_openers("") is None
    assert split_interior_openers(None) is None


# ---------------------------------------------------------------------------
# Round-3 Defect 3 — pipe-cell masking (marker inside a cell is not a split
# point; a marker OUTSIDE the pipe run splits off).
# ---------------------------------------------------------------------------
def test_interior_split_masks_pipe_cell_marker():
    # A TRY IT marker fused INSIDE a pipe-table cell is NOT a split point.
    assert (
        split_interior_openers("| Check: | value | TRY IT 2.35 | Solve: x | more")
        is None
    )


def test_interior_split_surfaces_marker_outside_pipe_run():
    # The ch02 s320 shape: a trailing TRY IT / EXAMPLE spilled PAST the last
    # pipe splits off, while the pipe run (incl. an in-cell marker) is preserved
    # verbatim in the leading text part.
    parts = split_interior_openers(
        "| Check: | 14 - 23 | TRY IT 2.35 | Solve: a. "
        "TRY IT 2.36 Solve: b. EXAMPLE 2.19 Solve: c."
    )
    assert parts is not None
    kinds = [p[0] for p in parts]
    assert kinds[0] == "text"  # the whole pipe run stays as the lead text part
    assert "| TRY IT 2.35 |" in parts[0][1]  # in-cell marker NOT split
    openers = [p[1] for p in parts if p[0] == "opener"]
    assert openers == ["Try It 2.36", "Example 2.19"]  # only OUTSIDE-pipe markers
