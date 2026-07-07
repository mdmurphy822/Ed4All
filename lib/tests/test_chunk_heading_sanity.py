"""Unit tests for the defensive chunk heading-sanity filter.

Locks in the conservative ``is_suspect_section_heading`` predicate (NEVER
demote a real heading) + the repair logic (repair to nearest CLEAN ancestor,
never blank, never repair noise → noise) + the ``TRAINFORGE_HEADING_SANITY_FILTER``
gate parse-with-fallback. All fixtures inline — no course-data references.

The positive (noise) / negative (real-title) lists are calibrated against the
real full-book corpus headings observed at the live ``_run_dart_chunking``
path: the 8 noise families the task names + the real section titles that MUST
survive untouched.
"""

from __future__ import annotations

import pytest

from lib.chunk_heading_sanity import (
    FLAG_ENV,
    is_heading_sanity_filter_enabled,
    is_suspect_section_heading,
    repair_section_heading,
)


# ---------------------------------------------------------------------------
# Detector — NOISE headings must be flagged (the 8 families from the task)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # 1. Front-matter (canonical detector).
        "Preface",
        # 2. Answer-key numeric run with embedded markers (canonical detector).
        "146,023 13. 5,846,103 14. 1,458,398",
        # 3. Circled-answer markers (canonical detector).
        "51,493 ⓐ 1, ⓑ 4, ⓒ 9 2. 87,210 ⓐ 2 ⓑ 8",
        # 4. Bracket-glyph math run dense with operators (canonical numeric ratio).
        "2⎡⎣1 + 3(10 − 2)⎤⎦ 126. 5⎡⎣2 + 4(3 − 2)⎤⎦",
        # 5. Compact numeric/operator noise (canonical numeric ratio).
        "24a 32b 2",
        # 6. OpenStax exercise banner + following-exercises prose (Layer 2a).
        "EXERCISES Practice Makes Perfect Use Place Value with Whole Numbers "
        "In the following exercises, find the place value of each digit in "
        "the given numbers.",
        "EXERCISES Practice Makes Perfect Multiply Integers In the following "
        "exercises, multiply.",
        "EXERCISES Practice Makes Perfect Simplify Expressions with Square "
        "Roots In the following exercises, simplify.",
        # 7. Math-prose mixed with >=2 embedded exercise number markers (Layer 2b).
        "3u 2 − 4u + 5 when u = −3 313. 9a − 2b − 8 when 314. 7m − 4n − 2 "
        "when a = −6 and b = −3 m = −4 and n = −9",
        # 8a. Full-sentence imperative prose ending in a period (Layer 3a).
        "Write the numbers one under the other, lining up the decimal points "
        "carefully before you begin adding each column.",
        # 8b. The REAL corpus instance: prose with an internal sentence
        # period + a numeric measurement tail "0.6" (Layer 3b).
        "Write the numbers one under the other, lining up the decimal "
        "points. 0.6",
    ],
)
def test_noise_headings_are_flagged(text):
    assert is_suspect_section_heading(text) is True, (
        f"expected {text[:60]!r}... to be flagged as noise"
    )


# ---------------------------------------------------------------------------
# Detector — REAL headings must survive (no false positive / demotion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Real OpenStax section titles observed on the live path.
        "1.1 Introduction to Whole Numbers",
        "1.1 Introduction to Whole Numbers (part 1)",
        "1.3 Add and Subtract Integers",
        "1.6 Add and Subtract Fractions",
        "1.9 Properties of Real Numbers",
        "Multiple of a Number",
        "Prime Factorization",
        "Least Common Multiple",
        "Properties of Zero",
        "Properties of Zero (part 14)",
        "Inequality",
        "Equation English Sentence",
        "Operation Phrase Expression",
        "Simplify an Expression",
        "Evaluate an Expression",
        "Like Terms",
        "Opposite",
        "Property of Absolute Value",
        "Equivalent Fractions Property",
        "Reciprocal",
        "Rational Number",
        "Irrational Number",
        "Real Number",
        "Percent",
        "Fraction",
        "Self Check",
        "Solution",
        "Solution (part 2)",
        "Inverse Property",
        # The math chapter title (all-caps single word).
        "FOUNDATIONS",
        "Foundations",
        # A real title that merely contains a number.
        "Section 3.5",
        "Chapter 2",
        # Short abbreviation titles with internal periods — must survive
        # (they sit under the Layer-3b word floor).
        "U.S. Customary Units",
        "Mr. Smith Example",
    ],
)
def test_real_headings_survive(text):
    assert is_suspect_section_heading(text) is False, (
        f"expected {text!r} to survive (false positive on a real title)"
    )


def test_empty_and_whitespace_not_flagged():
    assert is_suspect_section_heading("") is False
    assert is_suspect_section_heading("   ") is False
    assert is_suspect_section_heading(None) is False


# ---------------------------------------------------------------------------
# Repair — substitute nearest clean ancestor; never blank; never noise→noise
# ---------------------------------------------------------------------------


def test_repair_uses_first_clean_merged_heading():
    res = repair_section_heading(
        "EXERCISES Practice Makes Perfect Multiply Integers In the following "
        "exercises, multiply.",
        item={"title": "Elementary Algebra", "module_title": "Elementary Algebra"},
        merged_headings=["1.3 Add and Subtract Integers", "Solution"],
    )
    assert res["suspect"] is True
    assert res["repaired"] is True
    assert res["heading"] == "1.3 Add and Subtract Integers"
    assert res["original_heading"].startswith("EXERCISES")


def test_repair_skips_noisy_ancestor_and_uses_next_clean_one():
    # First merged heading is itself noise → must skip to the clean one.
    res = repair_section_heading(
        "24a 32b 2",
        item={"title": "Elementary Algebra"},
        merged_headings=[
            "146,023 13. 5,846,103 14. 1,458,398",  # noise — must be skipped
            "Prime Factorization",  # clean — must win
        ],
    )
    assert res["repaired"] is True
    assert res["heading"] == "Prime Factorization"


def test_repair_falls_back_to_page_title_when_no_clean_merged():
    res = repair_section_heading(
        "24a 32b 2",
        item={"title": "Elementary Algebra Foundations"},
        merged_headings=[],
    )
    assert res["repaired"] is True
    assert res["heading"] == "Elementary Algebra Foundations"


def test_repair_no_clean_ancestor_leaves_as_is_and_flags_suspect():
    # Every candidate is noise → never blank, keep original, mark suspect.
    res = repair_section_heading(
        "24a 32b 2",
        item={"title": "Preface", "module_title": "146,023 13. 5,846,103"},
        merged_headings=["Preface"],
    )
    assert res["suspect"] is True
    assert res["repaired"] is False
    assert res["heading"] == "24a 32b 2"  # left as-is, NOT blanked


def test_repair_real_heading_is_passthrough():
    res = repair_section_heading(
        "Prime Factorization",
        item={"title": "Elementary Algebra"},
        merged_headings=["Multiple of a Number"],
    )
    assert res["suspect"] is False
    assert res["repaired"] is False
    assert res["heading"] == "Prime Factorization"


def test_repair_strips_part_suffix_from_chosen_ancestor():
    res = repair_section_heading(
        "EXERCISES Practice Makes Perfect In the following exercises, add.",
        item={"title": "Elementary Algebra"},
        merged_headings=["1.6 Add and Subtract Fractions (part 2)"],
    )
    assert res["repaired"] is True
    assert res["heading"] == "1.6 Add and Subtract Fractions"


# ---------------------------------------------------------------------------
# Flag parse-with-fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("garbage", False),
        ("", False),
    ],
)
def test_flag_parse_with_fallback(monkeypatch, value, expected):
    monkeypatch.setenv(FLAG_ENV, value)
    assert is_heading_sanity_filter_enabled() is expected


def test_flag_unset_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    assert is_heading_sanity_filter_enabled() is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
