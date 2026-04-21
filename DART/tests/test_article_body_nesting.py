"""Wave 29 article-body nesting tests (Defect 1).

Pre-Wave-29 the document assembler closed every ``<article
role="doc-chapter">`` wrapper immediately after its ``<header>``,
leaving chapter body paragraphs stranded as siblings of an empty
article element. That broke both:

* WCAG 1.3.1 — ``role="doc-chapter"`` promises a container that
  contains the chapter's content, not an empty shell.
* ``MCP/tools/_content_gen_helpers.py::parse_dart_html_files``,
  which reads chapter prose from INSIDE the article; zero paragraphs
  yielded zero real content on real corpora (Wave 28 synthetic
  tests still passed because they used a single paragraph, which
  produced an article-free fallback section).

Wave 29 moves to chapter-scoped nesting: subsequent body blocks
render INSIDE the open article until the next chapter opener, a
bibliography / TOC terminator, or end-of-body is reached.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from DART.converter import convert_pdftotext_to_html
from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.document_assembler import _render_body, _split_chapter_opener


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _mk(role: BlockRole, text: str, block_id: str = None, **attrs) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=text, block_id=block_id or f"blk{hash(text) & 0xFFFF:04x}"),
        role=role,
        confidence=0.9,
        attributes=attrs,
    )


def _chapter(number: str, title: str, block_id: str = None) -> ClassifiedBlock:
    return _mk(
        BlockRole.CHAPTER_OPENER,
        f"Chapter {number}: {title}",
        block_id=block_id,
        heading_text=title,
        chapter_number=number,
    )


def _paragraph(text: str, block_id: str = None) -> ClassifiedBlock:
    return _mk(BlockRole.PARAGRAPH, text, block_id=block_id)


def _find_article_blocks(html: str) -> List[str]:
    """Return the text INSIDE each ``<article role="doc-chapter">`` wrapper."""
    pat = re.compile(
        r'(?is)<article\b[^>]*?role\s*=\s*["\']doc-chapter["\'][^>]*>(.*?)</article>'
    )
    return [m.group(1) for m in pat.finditer(html)]


# --------------------------------------------------------------------- #
# Core nesting behaviour
# --------------------------------------------------------------------- #


def test_chapter_opener_followed_by_paragraphs_nests_them():
    """Chapter opener + 3 paragraphs + chapter opener should produce
    two articles; the first carries all 3 paragraphs inside."""
    blocks = [
        _chapter("1", "Foundations", block_id="ch1"),
        _paragraph("First paragraph of chapter 1 body text.", block_id="p1"),
        _paragraph("Second paragraph with more detail about foundations.", block_id="p2"),
        _paragraph("Third paragraph wrapping up the introduction.", block_id="p3"),
        _chapter("2", "Advanced Topics", block_id="ch2"),
        _paragraph("First paragraph of chapter 2 body text.", block_id="p4"),
    ]
    html = _render_body(blocks)

    articles = _find_article_blocks(html)
    assert len(articles) == 2, f"Expected 2 articles, got {len(articles)}: {html}"

    # First article contains all 3 paragraphs.
    first = articles[0]
    assert "First paragraph of chapter 1" in first
    assert "Second paragraph" in first
    assert "Third paragraph" in first
    assert "First paragraph of chapter 2" not in first

    # Second article contains its own paragraph.
    second = articles[1]
    assert "First paragraph of chapter 2" in second
    assert "First paragraph of chapter 1" not in second


def test_chapter_with_no_body_still_emits_valid_article():
    """A chapter opener followed by another chapter opener (no body)
    should still produce valid back-to-back articles — back-compat
    with short test fixtures."""
    blocks = [
        _chapter("1", "Alpha", block_id="ca"),
        _chapter("2", "Beta", block_id="cb"),
    ]
    html = _render_body(blocks)

    articles = _find_article_blocks(html)
    assert len(articles) == 2
    # Both should be valid (just their <header>s inside).
    for art in articles:
        assert "<h2" in art


def test_bibliography_after_chapter_closes_the_chapter():
    """Bibliography entries sit at document scope, not inside a
    chapter — they should terminate any open chapter."""
    biblio = _mk(BlockRole.BIBLIOGRAPHY_ENTRY, "Smith, J. (2020). Some reference.")
    blocks = [
        _chapter("1", "Main Chapter", block_id="ch1"),
        _paragraph("Body paragraph inside the chapter.", block_id="p1"),
        biblio,
    ]
    html = _render_body(blocks)

    articles = _find_article_blocks(html)
    assert len(articles) == 1
    assert "Body paragraph" in articles[0]
    # Bibliography entry is NOT inside the chapter article.
    assert "Smith, J." not in articles[0]
    # Bibliography wrapper is at document scope.
    assert "<ol role=\"doc-bibliography\">" in html
    # Order check: the </article> closing tag comes before the <ol ...>.
    art_close_idx = html.rfind("</article>")
    ol_open_idx = html.rfind('<ol role="doc-bibliography">')
    assert art_close_idx < ol_open_idx


def test_duplicate_chapter_opener_demoted_to_paragraph_nests_inside():
    """The Wave 25 dedup demotes duplicate CHAPTER_OPENER blocks to
    PARAGRAPH. Those demoted paragraphs should nest inside the
    currently-open chapter, not sit outside it."""
    from DART.converter.document_assembler import _dedup_chapter_openers

    dup1 = _chapter("1", "Foundations", block_id="c1a")
    dup2 = _chapter("1", "Foundations", block_id="c1b")  # same number = duplicate
    mid = _paragraph("Real chapter body.", block_id="p1")
    deduped = _dedup_chapter_openers([dup1, mid, dup2])

    # Sanity: second chapter opener was demoted.
    roles = [b.role for b in deduped]
    assert roles.count(BlockRole.CHAPTER_OPENER) == 1
    assert roles.count(BlockRole.PARAGRAPH) == 2

    html = _render_body(deduped)
    articles = _find_article_blocks(html)
    assert len(articles) == 1
    # Both the real body paragraph AND the demoted duplicate should
    # be inside the one article.
    assert "Real chapter body" in articles[0]
    assert "Foundations" in articles[0]


def test_end_of_body_closes_open_chapter():
    """If the body ends with an open chapter + body blocks, we must
    still emit the closing ``</article>`` tag."""
    blocks = [
        _chapter("1", "Only Chapter", block_id="ch1"),
        _paragraph("Last paragraph of the document.", block_id="p1"),
    ]
    html = _render_body(blocks)
    assert html.count("<article") == 1
    assert html.count("</article>") == 1
    articles = _find_article_blocks(html)
    assert "Last paragraph" in articles[0]


def test_split_chapter_opener_helper():
    """_split_chapter_opener correctly splits rendered article
    templates into ``(opener_without_closer, closer)``."""
    rendered = '<article class="dart-section" role="doc-chapter" id="chap-1"><header><h2>T</h2></header></article>'
    opener, closer = _split_chapter_opener(rendered)
    assert closer == "</article>"
    assert opener.endswith("</header>")
    assert "</article>" not in opener


def test_split_chapter_opener_missing_close_is_defensive():
    """If the rendered string somehow has no closing article tag,
    the helper returns the input unchanged as opener + empty closer."""
    opener, closer = _split_chapter_opener("<article><header></header>")
    # Returns full string (no match) with empty closer.
    assert closer == ""
    assert opener == "<article><header></header>"


def test_parse_dart_html_files_sees_paragraphs_inside_article():
    """End-to-end: after Wave 29, ``parse_dart_html_files`` extracts
    real paragraph topics from inside the chapter article. Pre-Wave-29
    the paragraphs were siblings OUTSIDE the article, producing empty
    chapter content on real corpora."""
    from MCP.tools._content_gen_helpers import parse_dart_html_files

    # Build a multi-chapter HTML with real-corpus-like prose.
    raw = (
        "Chapter 1: Fundamentals of Pedagogy\n\n"
        "Pedagogy is the method and practice of teaching. This field "
        "encompasses instructional strategies, curriculum design, and "
        "assessment approaches that educators use to facilitate learning. "
        "Understanding pedagogical principles requires examining both "
        "historical context and contemporary research on how students "
        "acquire knowledge and develop skills in educational settings.\n\n"
        "Effective teaching demands deliberate planning and thoughtful "
        "reflection on practice. Teachers must consider learner diversity, "
        "subject-matter expertise, and the sociocultural context of "
        "their classrooms when designing meaningful learning experiences "
        "that engage every student.\n\n"
        "Chapter 2: Curriculum Design\n\n"
        "Curriculum design is the systematic process of creating coherent "
        "educational experiences aligned with learning objectives and "
        "outcomes. It requires balancing content depth with breadth, "
        "scaffolding complexity for learner development, and integrating "
        "multiple modalities to support diverse access needs.\n"
    )
    html = convert_pdftotext_to_html(raw, title="Wave 29 Nesting Smoke")

    # Write to a temporary file so we can feed parse_dart_html_files.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        path = Path(f.name)

    try:
        topics = parse_dart_html_files([path])
    finally:
        path.unlink(missing_ok=True)

    # At least one topic should have real paragraph content now.
    # Pre-Wave-29 this returned zero topics on real corpora.
    topic_paragraph_counts = [len(t.get("paragraphs", [])) for t in topics]
    assert any(c > 0 for c in topic_paragraph_counts), (
        "Expected at least one topic with paragraphs; Wave 29 should "
        "surface prose from inside doc-chapter articles."
    )


def test_wave25_chapter_dedup_preserved():
    """Wave 25 chapter-id dedup must still fire under Wave 29 nesting —
    duplicate opener gets demoted to paragraph and flows into the
    currently-open article rather than spawning a duplicate chap-N
    anchor."""
    from DART.converter.document_assembler import _dedup_chapter_openers

    blocks = [
        _chapter("6", "Chapter Six Title", block_id="c6a"),
        _paragraph("Real body.", block_id="p1"),
        # Back-matter recap entry (same title + number).
        _chapter("6", "Chapter Six Title", block_id="c6b"),
    ]
    deduped = _dedup_chapter_openers(blocks)
    html = _render_body(deduped)

    # Only one id="chap-6" anchor in the output.
    assert html.count('id="chap-6"') == 1
