"""Wave 15 tests: cross-reference resolver behaviour.

Exercises :func:`DART.converter.cross_refs.resolve_cross_references` in
isolation (by feeding it a hand-built HTML string + block list) so the
unit contract is decoupled from the rest of the assembler plumbing.

Covers:

* ``See Chapter N`` / ``Chapter N`` rewrite
* Orphan chapter refs left as plain text
* ``Figure N.M`` rewrite; orphan figure untouched
* ``Section N.M`` rewrite; orphan section untouched
* ``[N]`` citation rewrite; orphan citation untouched
* Mixed references in a single paragraph all resolve
* Already-linked ``<a>See Chapter 1</a>`` not double-wrapped
* ``<head>`` block never rewritten even when references appear there
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.cross_refs import resolve_cross_references


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chap(number: int) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=f"Chapter {number}", block_id=f"c{number:03d}"),
        role=BlockRole.CHAPTER_OPENER,
        confidence=0.9,
        attributes={"heading_text": f"Chapter {number}", "number": str(number)},
    )


def _fig(n: int, m: int) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=f"Figure {n}.{m}", block_id=f"f{n}{m}"),
        role=BlockRole.FIGURE,
        confidence=0.8,
        attributes={"number": f"{n}.{m}"},
    )


def _sec(n: int, m: int) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=f"Section {n}.{m}", block_id=f"s{n}{m}"),
        role=BlockRole.SECTION_HEADING,
        confidence=0.9,
        attributes={"heading_text": f"{n}.{m} Heading", "number": f"{n}.{m}"},
    )


def _cite(number: int) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=f"[{number}] Doe (2024).", block_id=f"r{number:03d}"),
        role=BlockRole.BIBLIOGRAPHY_ENTRY,
        confidence=0.8,
        attributes={"number": str(number)},
    )


def _wrap_body(body: str) -> str:
    """Wrap ``body`` in a minimal document shell so head-splitting works."""
    return (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        f"<body>{body}</body></html>"
    )


# ---------------------------------------------------------------------------
# Chapter rewrites
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestChapterRewrites:
    def test_see_chapter_rewrites_to_anchor(self):
        doc = _wrap_body("<p>For details, See Chapter 2 below.</p>")
        out = resolve_cross_references(doc, [_chap(2)])
        assert '<a href="#chap-2">Chapter 2</a>' in out
        # "See " prefix is preserved outside the anchor text per spec.
        assert "See <a" in out

    def test_bare_chapter_rewrites_to_anchor(self):
        doc = _wrap_body("<p>Chapter 3 introduces the key idea.</p>")
        out = resolve_cross_references(doc, [_chap(3)])
        assert '<a href="#chap-3">Chapter 3</a>' in out

    def test_orphan_chapter_left_as_plain_text(self):
        doc = _wrap_body("<p>See Chapter 99 which does not exist.</p>")
        out = resolve_cross_references(doc, [_chap(1), _chap(2)])
        assert '<a href="#chap-99"' not in out
        assert "See Chapter 99" in out

    def test_chapter_number_scraped_from_heading_text(self):
        """Blocks without explicit number attribute still register when the
        heading text encodes the number."""
        block = ClassifiedBlock(
            raw=RawBlock(text="Chapter 4: Advanced Topics", block_id="c4"),
            role=BlockRole.CHAPTER_OPENER,
            confidence=0.9,
            attributes={"heading_text": "Chapter 4: Advanced Topics"},
        )
        doc = _wrap_body("<p>Recall Chapter 4 for the full treatment.</p>")
        out = resolve_cross_references(doc, [block])
        assert '<a href="#chap-4">Chapter 4</a>' in out


# ---------------------------------------------------------------------------
# Figure rewrites
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestFigureRewrites:
    def test_figure_dot_numbered_rewrites(self):
        doc = _wrap_body("<p>Figure 2.1 shows the schema.</p>")
        out = resolve_cross_references(doc, [_fig(2, 1)])
        assert '<a href="#fig-2-1">Figure 2.1</a>' in out

    def test_orphan_figure_left_as_plain_text(self):
        doc = _wrap_body("<p>Figure 9.9 is not present.</p>")
        out = resolve_cross_references(doc, [_fig(1, 1)])
        assert '<a href="#fig-9-9"' not in out
        assert "Figure 9.9" in out


# ---------------------------------------------------------------------------
# Section rewrites
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSectionRewrites:
    def test_section_rewrites_to_anchor(self):
        doc = _wrap_body("<p>Per Section 3.2, the answer is obvious.</p>")
        out = resolve_cross_references(doc, [_sec(3, 2)])
        assert '<a href="#sec-3-2">Section 3.2</a>' in out

    def test_orphan_section_left_as_plain_text(self):
        doc = _wrap_body("<p>Section 8.8 does not exist here.</p>")
        out = resolve_cross_references(doc, [_sec(1, 1)])
        assert '<a href="#sec-8-8"' not in out
        assert "Section 8.8" in out


# ---------------------------------------------------------------------------
# Citation rewrites
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestCitationRewrites:
    def test_citation_marker_rewrites(self):
        doc = _wrap_body("<p>See [3] for the full proof.</p>")
        out = resolve_cross_references(doc, [_cite(3)])
        assert '<a href="#ref-3">[3]</a>' in out

    def test_orphan_citation_left_as_plain_text(self):
        doc = _wrap_body("<p>Ignore [77] for now.</p>")
        out = resolve_cross_references(doc, [_cite(1), _cite(2)])
        assert '<a href="#ref-77"' not in out
        assert "[77]" in out


# ---------------------------------------------------------------------------
# Mixed + edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestMixedAndEdgeCases:
    def test_mixed_references_all_rewrite(self):
        doc = _wrap_body(
            "<p>See Chapter 1, Figure 2.1, and Section 3.2. Also [4].</p>"
        )
        out = resolve_cross_references(
            doc,
            [_chap(1), _fig(2, 1), _sec(3, 2), _cite(4)],
        )
        assert '<a href="#chap-1">Chapter 1</a>' in out
        assert '<a href="#fig-2-1">Figure 2.1</a>' in out
        assert '<a href="#sec-3-2">Section 3.2</a>' in out
        assert '<a href="#ref-4">[4]</a>' in out

    def test_existing_anchor_not_double_wrapped(self):
        """``<a>See Chapter 1</a>`` must survive untouched even if the
        target chapter exists in the block list."""
        doc = _wrap_body(
            '<p>For background, <a href="#existing">See Chapter 1</a> directly.</p>'
        )
        out = resolve_cross_references(doc, [_chap(1)])
        # No nested <a> inside the existing one.
        assert '<a href="#chap-1"' not in out
        # Original anchor preserved verbatim.
        assert '<a href="#existing">See Chapter 1</a>' in out

    def test_head_block_passes_through_unchanged(self):
        """References mentioned in <title> / <meta> must not be rewritten."""
        doc = (
            '<!DOCTYPE html><html><head>'
            '<title>See Chapter 1 and Figure 2.1</title>'
            '<meta name="description" content="Cites [5] in passing">'
            '</head><body><p>See Chapter 1.</p></body></html>'
        )
        out = resolve_cross_references(doc, [_chap(1), _fig(2, 1), _cite(5)])
        # Head block still has the raw text.
        assert "<title>See Chapter 1 and Figure 2.1</title>" in out
        assert 'content="Cites [5] in passing"' in out
        # Body got rewritten.
        assert '<a href="#chap-1">Chapter 1</a>' in out

    def test_empty_block_list_is_noop(self):
        doc = _wrap_body("<p>See Chapter 1 and Figure 3.2.</p>")
        out = resolve_cross_references(doc, [])
        # No rewrites when block list is empty.
        assert "<a" not in out.split("<body>")[1].split("</body>")[0]

    def test_multiple_occurrences_all_rewrite(self):
        doc = _wrap_body(
            "<p>Chapter 1 introduces it. Chapter 1 again wraps up.</p>"
        )
        out = resolve_cross_references(doc, [_chap(1)])
        assert out.count('<a href="#chap-1">Chapter 1</a>') == 2
