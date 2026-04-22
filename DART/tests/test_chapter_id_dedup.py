"""Wave 25 Fix 3: duplicate CHAPTER_OPENER → single emission.

Audit evidence: on textbooks whose back matter repeats chapter
titles, ``id="chap-1"`` can appear 2×, ``chap-6`` 3×, ``chap-12`` 3×.
The classifier promotes both the real chapter opener and a back-of-
book recap/index entry referencing the same chapter. The assembler's
dedup pass suppresses second-and-later occurrences, and the JSON-LD
``hasPart`` list stays unique.
"""

from __future__ import annotations

import re

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.document_assembler import assemble_html


def _chapter(text: str, block_id: str, *, page: int = 1) -> ClassifiedBlock:
    raw = RawBlock(text=text, block_id=block_id, page=page, extractor="pdftotext")
    return ClassifiedBlock(
        raw=raw,
        role=BlockRole.CHAPTER_OPENER,
        confidence=0.9,
        attributes={"heading_text": text},
        classifier_source="heuristic",
    )


@pytest.mark.unit
@pytest.mark.dart
class TestChapterIdDedup:
    def test_duplicate_openers_emit_once(self):
        # Two blocks with identical first-line text → only the first
        # emits a chap-N anchor; the second is demoted to paragraph.
        blocks = [
            _chapter("Chapter 6: Digital Learning", "blk-a", page=60),
            _chapter("Chapter 6: Digital Learning", "blk-b", page=500),
        ]
        html = assemble_html(blocks, title="T", metadata={})
        # Only one <article id="chap-6"> emission.
        assert html.count('id="chap-6"') == 1

    def test_haspart_deduped(self):
        blocks = [
            _chapter("Chapter 1 Introduction", "b1", page=1),
            _chapter("Chapter 1 Introduction", "b2", page=200),
            _chapter("Chapter 2 Learning Theories", "b3", page=30),
        ]
        html = assemble_html(blocks, title="T", metadata={})
        # hasPart list should carry exactly 2 entries (one per unique
        # chapter).
        match = re.search(
            r'"hasPart":\s*(\[[^\]]*\])', html, re.DOTALL
        )
        assert match is not None, "hasPart not emitted"
        # Count Chapter items in the hasPart list.
        has_part = match.group(1)
        # Each entry carries "url": "#chap-N"; count distinct urls.
        urls = set(re.findall(r'"url":\s*"#chap-(\d+)"', has_part))
        assert urls == {"1", "2"}

    def test_unique_chapters_all_emit_anchors(self):
        blocks = [
            _chapter("Chapter 1 Beginnings", "b1", page=1),
            _chapter("Chapter 2 Middles", "b2", page=50),
            _chapter("Chapter 3 Ends", "b3", page=100),
        ]
        html = assemble_html(blocks, title="T", metadata={})
        assert 'id="chap-1"' in html
        assert 'id="chap-2"' in html
        assert 'id="chap-3"' in html

    def test_surviving_emission_keeps_data_dart_pages(self):
        # The first of a duplicate pair keeps its page provenance.
        # Simulate the common case where the body opener is on an
        # early page and a back-of-book recap repeats the same
        # chapter title on a later page.
        blocks = [
            _chapter("Chapter 8 Curriculum Design", "b1", page=200),
            _chapter("Chapter 8 Curriculum Design", "b2", page=480),
        ]
        html = assemble_html(blocks, title="T", metadata={})
        # The first opener's data-dart-pages="200" survives (appears
        # within the single <article role="doc-chapter"> emission).
        article_match = re.search(
            r'<article[^>]*role="doc-chapter"[^>]*id="chap-8"[^>]*>',
            html,
        )
        assert article_match is not None
        assert 'data-dart-pages="200"' in article_match.group(0)
