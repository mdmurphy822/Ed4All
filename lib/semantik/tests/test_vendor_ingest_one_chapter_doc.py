"""§3.4a — a vendor file that NAMES one chapter is one chapter.

Regression for the 2026-08-01 CHAPTER_EXPLOSION block. ``vendor_ingest`` opened
a chapter at every ``h1``/``h2``, on the assumption that one file = one SECTION
of a larger work. A publisher shipping one file PER CHAPTER inverts that, and
this publisher's markup is worse than inverted — it is flat: ``<h1>`` carries
the chapter title, the numbered sections ("1.1 Whole Numbers"), the
subsections, AND repeated end-of-section apparatus ("Answers", "Glossary",
"Key Concepts"). 57 h1s, 37 distinct, no usable hierarchy.

Measured on the 9-file corpus: h1+h2 -> 843 chapters; h1-only with
running-header dedup -> 485; file-is-chapter -> 9. Hence the rule under test:
when exactly one chapter TITLE is named, no heading opens another chapter.

The multi-chapter path must stay byte-identical — that is what the second
group pins.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.semantik.vendor_ingest import (  # noqa: E402
    _CHAPTER_TITLE_RE,
    _document_opens_one_chapter,
    build_chapters_ir_from_html,
)


def _roots(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [soup.body or soup]


# --- the chapter-title signal --------------------------------------------

def test_chapter_title_regex_accepts_real_chapter_headings():
    for text in (
        "CHAPTER 1 Whole Numbers, Integers, and Introduction to Algebra",
        "CHAPTER 9 Trigonometry",
        "Unit 2 — Kinematics",
        "Module 4: Ratios",
        "Chapter IV Foundations",
        "Lesson Three",
    ):
        assert _CHAPTER_TITLE_RE.match(text), text


def test_chapter_title_regex_rejects_titles_sections_and_apparatus():
    """A book TITLE must not match, or one-file-per-book corpora would flip."""
    for text in (
        "Introductory Algebra",       # book title
        "1.1 Whole Numbers",          # numbered SECTION
        "Use Place Value with Whole Numbers",
        "Key Concepts",
        "Practice Makes Perfect",
        "Answers",
        "Glossary",
        "Chapters of Interest",
    ):
        assert not _CHAPTER_TITLE_RE.match(text), text


# --- document-level detection --------------------------------------------

def test_repeated_running_header_still_detects_one_chapter():
    """The chapter h1 repeats as a running header — 57x in the real corpus."""
    html = "".join(
        "<h1>CHAPTER 1 Whole Numbers</h1><p>body</p>" for _ in range(20)
    )
    assert _document_opens_one_chapter(_roots(html)) == "CHAPTER 1 Whole Numbers"


def test_two_distinct_chapter_titles_is_not_a_one_chapter_doc():
    html = "<h1>CHAPTER 1 Whole Numbers</h1><p>a</p><h1>CHAPTER 2 Fractions</h1><p>b</p>"
    assert _document_opens_one_chapter(_roots(html)) is None


def test_book_title_h1_is_not_a_one_chapter_doc():
    html = "<h1>Introductory Algebra</h1><h2>Chapter One Stuff</h2><p>a</p>"
    assert _document_opens_one_chapter(_roots(html)) is None


# --- end-to-end boundary behaviour ---------------------------------------

def test_flat_h1_markup_yields_exactly_one_chapter():
    """The real shape: chapter title + sections + apparatus, all as h1."""
    html = (
        "<h1>CHAPTER 1 Whole Numbers</h1>"
        "<h1>1.1 Whole Numbers</h1><p>alpha</p>"
        "<h2>Use Place Value</h2><p>beta</p>"
        "<h1>Key Concepts</h1><p>gamma</p>"
        "<h1>Answers</h1><p>delta</p>"
        "<h1>CHAPTER 1 Whole Numbers</h1>"   # running header repeat
        "<h1>1.2 Language of Algebra</h1><p>epsilon</p>"
    )
    chapters = build_chapters_ir_from_html(html, doc_title="ch01.html")
    assert len(chapters) == 1
    assert chapters[0].title == "CHAPTER 1 Whole Numbers"
    # every paragraph survives as content, none lost to chapter splitting
    text = " ".join(b.raw_text or "" for b in chapters[0].blocks)
    for word in ("alpha", "beta", "gamma", "delta", "epsilon"):
        assert word in text


def test_chapter_titled_from_heading_not_filename():
    html = "<h1>CHAPTER 3 Measurement</h1><p>x</p>"
    chapters = build_chapters_ir_from_html(html, doc_title="03-measurement.html")
    assert chapters[0].title == "CHAPTER 3 Measurement"


def test_multi_chapter_file_keeps_the_h1_h2_boundary_rule():
    """The pre-existing path must not move."""
    html = (
        "<h1>Some Book</h1>"
        "<h2>First Section</h2><p>a</p>"
        "<h2>Second Section</h2><p>b</p>"
        "<h2>Third Section</h2><p>c</p>"
    )
    chapters = build_chapters_ir_from_html(html, doc_title="book.html")
    assert len(chapters) > 1, "multi-chapter grouping must still split at h2"


def test_no_content_is_dropped_by_the_one_chapter_rule():
    html = (
        "<h1>CHAPTER 2 Fractions</h1>"
        + "".join(f"<h1>2.{i} Section</h1><p>para{i}</p>" for i in range(30))
    )
    chapters = build_chapters_ir_from_html(html, doc_title="ch02.html")
    assert len(chapters) == 1
    text = " ".join(b.raw_text or "" for b in chapters[0].blocks)
    for i in range(30):
        assert f"para{i}" in text
