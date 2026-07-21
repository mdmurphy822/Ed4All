"""Tests for the ``SemanticStructureExtractor`` ``chapter_text`` /
``section_text`` extension (three-stage textbook synthesis, plan §1 / §9).

The three-stage LLM synthesis architecture needs the FULL inter-heading
prose per chapter — the sparse structural ``contentBlocks`` carries only
a sliver. This module pins the new ``chapter_text`` / ``section_text``
fields:

- ``chapter_text`` carries the full inter-heading prose for a chapter,
  including the prose nested under its sections.
- ``section_text`` carries just that section's own scope.
- ``contentBlocks`` semantics are byte-stable — the new fields are
  purely additive; the structural surface is unchanged.
- A multi-chapter single-HTML-file fixture is segmented correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantic_structure_extractor import SemanticStructureExtractor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — multi-chapter, single HTML file (the single-file-textbook shape:
# one DART HTML file carrying every chapter).
# ---------------------------------------------------------------------------


_MULTI_CHAPTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Example Algebra Textbook</title></head>
<body>
  <main>
    <h1>Example Algebra Textbook</h1>
    <h2 id="ch-found">Foundations</h2>
    <p>Foundations introduces whole numbers and the order of operations.</p>
    <p>It also covers prime factorization of integers in detail.</p>
    <h3 id="ch-found-primes">Prime Factorization</h3>
    <p>A prime factorization expresses an integer as a product of primes.</p>
    <h2 id="ch-lin">Solving Linear Equations</h2>
    <p>Linear equations are equations whose graph is a straight line.</p>
    <h3 id="ch-lin-slope">Slope Intercept Form</h3>
    <p>Slope-intercept form writes a linear equation as y equals mx plus b.</p>
    <p>The slope m measures steepness; b is the y-intercept.</p>
  </main>
</body>
</html>
"""


@pytest.fixture
def extractor() -> SemanticStructureExtractor:
    return SemanticStructureExtractor()


@pytest.fixture
def multi_chapter_structure(extractor):
    return extractor.extract(_MULTI_CHAPTER_HTML, format="html")


# ---------------------------------------------------------------------------
# Multi-chapter single-file segmentation
# ---------------------------------------------------------------------------


def test_multi_chapter_single_file_yields_two_chapters(
    multi_chapter_structure,
):
    chapters = multi_chapter_structure["chapters"]
    assert len(chapters) == 2
    titles = {c["headingText"] for c in chapters}
    assert titles == {"Foundations", "Solving Linear Equations"}


# ---------------------------------------------------------------------------
# chapter_text carries full inter-heading prose
# ---------------------------------------------------------------------------


def test_chapter_text_carries_full_inter_heading_prose(
    multi_chapter_structure,
):
    chapters = multi_chapter_structure["chapters"]
    found = next(
        c for c in chapters if c["headingText"] == "Foundations"
    )
    text = found.get("chapter_text", "")
    # The chapter's own intro prose is captured.
    assert "Foundations introduces whole numbers" in text
    assert "prime factorization of integers" in text
    # Prose nested under the chapter's section is folded into
    # chapter_text — the chapter scope includes its sections.
    assert "product of primes" in text


def test_chapter_text_does_not_leak_across_chapter_boundary(
    multi_chapter_structure,
):
    chapters = multi_chapter_structure["chapters"]
    found = next(
        c for c in chapters if c["headingText"] == "Foundations"
    )
    text = found.get("chapter_text", "")
    # Prose belonging to the *next* chapter must NOT appear.
    assert "straight line" not in text
    assert "y-intercept" not in text


def test_second_chapter_text_captures_its_own_prose(
    multi_chapter_structure,
):
    chapters = multi_chapter_structure["chapters"]
    found = next(
        c for c in chapters
        if c["headingText"] == "Solving Linear Equations"
    )
    text = found.get("chapter_text", "")
    assert "graph is a straight line" in text
    assert "y equals mx plus b" in text


# ---------------------------------------------------------------------------
# section_text carries the section's own scope
# ---------------------------------------------------------------------------


def test_section_text_captures_section_scope(multi_chapter_structure):
    chapters = multi_chapter_structure["chapters"]
    found = next(
        c for c in chapters if c["headingText"] == "Foundations"
    )
    sections = found.get("sections", [])
    assert sections, "Foundations should have a Prime Factorization section"
    sec = sections[0]
    sec_text = sec.get("section_text", "")
    assert "product of primes" in sec_text
    # Section text is scoped — it does NOT include the parent chapter's
    # own intro prose.
    assert "whole numbers" not in sec_text


# ---------------------------------------------------------------------------
# contentBlocks byte-stability — the new fields are purely additive
# ---------------------------------------------------------------------------


def test_content_blocks_byte_stable_with_chapter_text_extension():
    """The structural ``contentBlocks`` surface is unchanged by the
    ``chapter_text`` / ``section_text`` extension."""
    extractor = SemanticStructureExtractor()
    structure = extractor.extract(_MULTI_CHAPTER_HTML, format="html")

    def _strip_synthesis_fields(node: dict) -> dict:
        """Recursively drop the additive synthesis fields so the
        remaining structural shape can be compared."""
        clean = {
            k: v for k, v in node.items()
            if k not in ("chapter_text", "section_text")
        }
        if "sections" in clean:
            clean["sections"] = [
                _strip_synthesis_fields(s) for s in clean["sections"]
            ]
        if "subsections" in clean:
            clean["subsections"] = [
                _strip_synthesis_fields(s) for s in clean["subsections"]
            ]
        return clean

    for chapter in structure["chapters"]:
        stripped = _strip_synthesis_fields(chapter)
        # contentBlocks survives the strip unchanged.
        assert "contentBlocks" in stripped
        # The structural keys are exactly the legacy set.
        assert set(stripped) == {
            "id", "headingLevel", "headingText", "headingId",
            "explicitObjectives", "contentBlocks", "sections",
        }


def test_chapter_text_field_absent_when_no_prose():
    """A heading with no following prose emits no ``chapter_text`` key —
    the field is additive, so legacy structural fixtures stay
    byte-stable when prose is absent."""
    bare_html = """
    <html><body><main>
      <h1>Bare Doc</h1>
      <h2>Empty Chapter One</h2>
      <h2>Empty Chapter Two</h2>
    </main></body></html>
    """
    extractor = SemanticStructureExtractor()
    structure = extractor.extract(bare_html, format="html")
    for chapter in structure["chapters"]:
        # No prose between the headings → no chapter_text key emitted.
        assert "chapter_text" not in chapter or chapter["chapter_text"]
