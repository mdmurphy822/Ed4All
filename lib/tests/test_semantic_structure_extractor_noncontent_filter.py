"""Content-heading filter tests for SemanticStructureExtractor.

A live full-textbook (ch1-3) pipeline run emitted 15
"chapters" because the heading-hierarchy fallback promoted non-content
headings to chapter/section nodes:

* answer-key numeric rows (``"146,023 13. 5,846,103 14. 1,458,398"``) and
  circled-answer markers (ⓐⓑⓒⓓⓔ),
* front-matter headings ("Preface", "Acknowledgments"),
* donor/foundation list lines ("Bill & Melinda Gates Foundation National
  Science Foundation").

Those slugs ended up as ``objective_charles-koch-foundation-...`` and
inflated per-chapter LLM work ~15x until an 1800s phase timeout fired.

These tests lock in ``_is_noncontent_heading`` filtering on the
heading-hierarchy fallback path AND the empty-after-filter fail-safe.
All fixtures are inline HTML — no real course-data references.
"""

from __future__ import annotations

import logging

import pytest

from lib.semantic_structure_extractor import SemanticStructureExtractor
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _is_noncontent_heading,
)


# ---------------------------------------------------------------------------
# Unit tests for the predicate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Answer-key numeric rows.
        "146,023 13. 5,846,103 14. 1,458,398",
        "78 41. 900 42. 800",
        "51,493 1. 87,210 2. 146,023 3.",
        # Circled-answer markers.
        "51,493 ⓐ 1, ⓑ 4, ⓒ 9, ⓓ 5, ⓔ 3 2. 87,210 ⓐ 2 ⓑ 8",
        "ⓐ 1 ⓑ 2 ⓒ 3",
        # Front-matter / acknowledgments.
        "Preface",
        "preface",
        "Acknowledgments",
        "Acknowledgements",
        "About the Authors",
        "Dedication",
        "Preface:",
        # Donor / foundation list lines.
        "Bill & Melinda Gates Foundation National Science Foundation",
        "Charles Koch Foundation The Stuart Family Foundation",
        "Laura and John Arnold Foundation The Maxfield Foundation",
        # End-of-chapter exercise / review / drill section headings
        # (a scanned algebra textbook EOC family). Exact matches.
        "Practice Makes Perfect",
        "Section Exercises",
        "Review Exercises",
        "Chapter Review Exercises",
        "Practice Test",
        "Additional Practice",
        "Everyday Math",
        "Writing Exercises",
        # Case / whitespace / trailing-colon variants normalize the same.
        "practice makes perfect",
        "PRACTICE MAKES PERFECT",
        "  Review   Exercises  ",
        "Everyday Math:",
        # Prefix form — a scanned algebra textbook appends the section topic
        # to the exercise-block phrases. Distinctive enough that a real title
        # never starts with them.
        "Practice Makes Perfect: Order Sample Widgets",
        "Review Exercises: Solve Linear Equations",
        "Chapter Review Exercises for Chapter 3",
        "Writing Exercises for This Section",
    ],
)
def test_noncontent_headings_are_rejected(text):
    assert _is_noncontent_heading(text) is True, (
        f"expected {text!r} to be filtered as non-content"
    )


@pytest.mark.parametrize(
    "text",
    [
        # The math chapter title — must survive (single word, all-caps).
        "FOUNDATIONS",
        "Foundations",
        # Legitimate section titles, some with embedded numbers.
        "Multiple of a Number",
        "Divisibility Tests",
        "Prime Factorization",
        "Use the Language of Algebra",
        "Add Whole Numbers",
        "1.2 Add Whole Numbers",
        "Section 3.5",
        "Introduction to Whole Numbers",
        "Chapter 2",
        # Real content titles from the prompt — must NOT be caught by the
        # EOC exercise/summary filter.
        "Solve Equations Using the Subtraction Property",
        "Decimals",
        "Multiply and Divide Mixed Numbers",
        # A chapter title that merely shares a leading word with an EOC
        # phrase — prefix matching must be word-boundary safe, not loose
        # substring or bare-word.
        "Practice Problems in Real Analysis",
        "Practice Makes Perfection a Reality",  # "perfect" is not a prefix
        "Practice Testing as a Study Strategy",  # "practice test" is EXACT-only
        "Reviewing Exercises for the Exam",     # not the "review exercises" prefix
        # DELIBERATELY EXCLUDED glossary/summary/metacognition family — this
        # predicate is shared with lib/chunk_heading_sanity.py, which treats
        # these as REAL chunk headings (its regression suite locks in
        # "Self Check"). They MUST survive here too.
        "Self Check",
        "Key Terms",
        "Key Concepts",
        "Chapter Summary",
        "Key Terms in Set Theory",
    ],
)
def test_content_headings_survive(text):
    assert _is_noncontent_heading(text) is False, (
        f"expected {text!r} to survive (false positive on a real title)"
    )


# ---------------------------------------------------------------------------
# End-to-end fallback path: contaminated headings dropped, real ones kept
# ---------------------------------------------------------------------------


# Heading-only HTML (no <section>/doc-chapter wrappers) so extraction
# routes through the heading-hierarchy fallback. Mixes the all-caps math
# chapter title FOUNDATIONS with answer-key rows, donor lines, and
# Preface — exactly the scanned-algebra-textbook contamination shape.
_CONTAMINATED_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Acme Algebra Shape</title></head>
<body>
  <main>
    <h1>Acme Algebra</h1>
    <h2>Preface</h2>
    <p>Front matter prose.</p>
    <h3>Laura and John Arnold Foundation The Maxfield Foundation</h3>
    <p>Donor list prose.</p>
    <h3>Bill &amp; Melinda Gates Foundation National Science Foundation</h3>
    <p>More donor list prose.</p>
    <h2>FOUNDATIONS</h2>
    <p>The Foundations chapter introduction.</p>
    <h3>Multiple of a Number</h3>
    <p>Multiples prose with substantive content.</p>
    <h3>Divisibility Tests</h3>
    <p>Divisibility prose with substantive content.</p>
    <h3>Prime Factorization</h3>
    <p>Prime factorization prose with substantive content.</p>
    <h2>146,023 13. 5,846,103 14. 1,458,398</h2>
    <p>Answer key numbers.</p>
    <h2>78 41. 900 42. 800</h2>
    <p>More answer key numbers.</p>
    <h3>51,493 ⓐ 1, ⓑ 4, ⓒ 9, ⓓ 5, ⓔ 3 2. 87,210 ⓐ 2 ⓑ 8</h3>
    <p>Circled answer key.</p>
    <h2>Use the Language of Algebra</h2>
    <p>Language of algebra prose with substantive content.</p>
    <h3>Add Whole Numbers</h3>
    <p>Add whole numbers prose with substantive content.</p>
  </main>
</body>
</html>
"""


def _all_heading_texts(chapters):
    texts = []

    def _walk_sections(sections):
        for sec in sections:
            texts.append(sec["headingText"])
            _walk_sections(sec.get("subsections", []))

    for ch in chapters:
        texts.append(ch["headingText"])
        _walk_sections(ch.get("sections", []))
    return texts


def test_contaminated_headings_filtered_real_titles_kept():
    """The contaminated heading set yields ONLY real titles as
    chapter/section nodes; no answer-key / donor / front-matter node."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_CONTAMINATED_HTML)
    chapters = result["chapters"]
    all_titles = _all_heading_texts(chapters)

    # Real content titles must all be present somewhere in the tree.
    for real in [
        "FOUNDATIONS",
        "Multiple of a Number",
        "Divisibility Tests",
        "Prime Factorization",
        "Use the Language of Algebra",
        "Add Whole Numbers",
    ]:
        assert real in all_titles, (
            f"real title {real!r} missing from extracted structure: "
            f"{all_titles}"
        )

    # None of the contaminated headings may appear anywhere.
    for bad in [
        "Preface",
        "Laura and John Arnold Foundation The Maxfield Foundation",
        "Bill & Melinda Gates Foundation National Science Foundation",
        "146,023 13. 5,846,103 14. 1,458,398",
        "78 41. 900 42. 800",
        "51,493 ⓐ 1, ⓑ 4, ⓒ 9, ⓓ 5, ⓔ 3 2. 87,210 ⓐ 2 ⓑ 8",
    ]:
        assert bad not in all_titles, (
            f"contaminated heading {bad!r} leaked into structure: "
            f"{all_titles}"
        )

    # Chapter count should collapse from the contaminated ~6 h2s to the
    # ~2 real chapter-level titles (FOUNDATIONS, Use the Language ...).
    assert len(chapters) <= 3, (
        f"expected <= 3 chapters after filtering, got {len(chapters)}: "
        f"{[c['headingText'] for c in chapters]}"
    )


def test_all_caps_foundations_chapter_title_is_kept():
    """The all-caps math chapter title FOUNDATIONS must survive — it must
    NOT be confused with a donor 'Foundation' list line."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_CONTAMINATED_HTML)
    titles = _all_heading_texts(result["chapters"])
    assert "FOUNDATIONS" in titles


# ---------------------------------------------------------------------------
# Empty-after-filter fail-safe: revert to unfiltered, never emit empty
# ---------------------------------------------------------------------------


# Every heading here is non-content (front matter only). The filter would
# remove all of them; the fail-safe must revert to the unfiltered set so
# we never emit a zero-chapter structure.
_ALL_NONCONTENT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Front Matter Only</title></head>
<body>
  <main>
    <h2>Preface</h2>
    <p>Preface prose.</p>
    <h2>Acknowledgments</h2>
    <p>Acknowledgments prose.</p>
    <h2>About the Authors</h2>
    <p>Authors prose.</p>
  </main>
</body>
</html>
"""


def test_empty_after_filter_falls_back_to_unfiltered():
    """If filtering removes every heading, the extractor must revert to
    the unfiltered structure (fail-safe), never emit zero chapters."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_ALL_NONCONTENT_HTML)
    chapters = result["chapters"]
    assert len(chapters) >= 1, (
        "empty-after-filter fail-safe failed: extractor emitted an empty "
        "structure instead of reverting to the unfiltered heading set"
    )


def test_empty_after_filter_logs_warning(caplog):
    """The fail-safe revert must log a WARNING so an over-aggressive
    predicate on a real document is observable."""
    caplog.set_level(
        logging.WARNING,
        logger="lib.semantic_structure_extractor.semantic_structure_extractor",
    )
    extractor = SemanticStructureExtractor()
    extractor.extract(_ALL_NONCONTENT_HTML, source_path="frontmatter.html")
    revert_warnings = [
        rec for rec in caplog.records
        if "reverting to unfiltered heading set" in rec.message
    ]
    assert revert_warnings, (
        "expected a fail-safe revert warning, got: "
        f"{[r.message for r in caplog.records]}"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
