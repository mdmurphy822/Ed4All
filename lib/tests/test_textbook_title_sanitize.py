"""Unit tests for the running-header / page-number title sanitizer.

Fixtures model a real 3-chapter OpenStax Elementary Algebra SCAN whose
converted ``<title>``/``<h1>`` fused the chapter title with the running header
+ page number + first content word:

    "Chapter 1 Foundations 83 ✓ Solution"  -> "Chapter 1 Foundations"
    "Chapter 3 Math Models 401 Question"   -> "Chapter 3 Math Models"

while a clean chapter title is returned verbatim.
"""

from __future__ import annotations

from lib.textbook_title_sanitize import (
    FLAG_ENV,
    sanitize_running_header_title,
    title_sanitize_enabled,
)


def test_strips_page_number_and_trailing_furniture():
    assert (
        sanitize_running_header_title("Chapter 1 Foundations 83 ✓ Solution")
        == "Chapter 1 Foundations"
    )
    assert (
        sanitize_running_header_title("Chapter 3 Math Models 401 Question")
        == "Chapter 3 Math Models"
    )


def test_clean_title_is_unchanged():
    clean = "Chapter 2 Solving Linear Equations and Inequalities"
    assert sanitize_running_header_title(clean) == clean


def test_section_decimal_not_treated_as_page_number():
    # "1.2" is a decimal section marker, not a 2-4 digit page number.
    assert sanitize_running_header_title("1.2 Add Whole Numbers") == "1.2 Add Whole Numbers"


def test_single_digit_chapter_number_not_stripped():
    # Leading "Chapter 1" carries a 1-digit number — never a page-number seam.
    assert sanitize_running_header_title("Chapter 1 Foundations") == "Chapter 1 Foundations"


def test_thin_prefix_keeps_original():
    # Fewer than 2 alphabetic words precede the number → not confident, keep.
    assert sanitize_running_header_title("Unit 42 Overview") == "Unit 42 Overview"


def test_long_tail_after_number_keeps_original():
    # A long tail signals the number is real content, not furniture.
    original = "Chapter 4 Systems 1984 A Very Long Subtitle That Continues On"
    assert sanitize_running_header_title(original) == original


def test_five_digit_number_not_a_page_number():
    original = "Population Study of 27493 Residents"
    assert sanitize_running_header_title(original) == original


def test_empty_and_none():
    assert sanitize_running_header_title("") == ""
    assert sanitize_running_header_title(None) == ""


def test_flag_parse_with_fallback():
    assert title_sanitize_enabled({}) is False
    assert title_sanitize_enabled({FLAG_ENV: "true"}) is True
    assert title_sanitize_enabled({FLAG_ENV: "1"}) is True
    assert title_sanitize_enabled({FLAG_ENV: "on"}) is True
    assert title_sanitize_enabled({FLAG_ENV: "junk"}) is False
    assert title_sanitize_enabled({FLAG_ENV: "0"}) is False
