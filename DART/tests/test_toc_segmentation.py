"""Wave 18 tests for TOC navigation segmentation + template.

* ``segment_extracted_document`` prepends a ``TOC_NAV`` block when
  ``ExtractedDocument.toc`` is non-empty.
* The ``TOC_NAV`` template renders a nested ``<ol>`` keyed by level
  with ``<a href="#chap-N">`` / ``<a href="#sec-N-M">`` anchors when the
  TOC titles follow the canonical numbering conventions.
* When ``doc.toc`` is empty, no TOC_NAV block is emitted (behaviour is
  unchanged from pre-Wave-18 segmentation).
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock
from DART.converter.block_segmenter import segment_extracted_document
from DART.converter.block_templates import render_block
from DART.converter.extractor import (
    ExtractedDocument,
    ExtractedTOCEntry,
)
from DART.converter.heuristic_classifier import HeuristicClassifier


def _doc_with_toc(toc_entries):
    return ExtractedDocument(
        raw_text="Intro prose.\n\nMore prose.",
        source_pdf="/tmp/x.pdf",
        pages_count=1,
        toc=list(toc_entries),
    )


@pytest.mark.unit
@pytest.mark.dart
class TestTocSegmentation:
    def test_non_empty_toc_prepends_toc_nav_block(self):
        doc = _doc_with_toc(
            [
                ExtractedTOCEntry(level=1, title="Chapter 1: Intro", page=3),
                ExtractedTOCEntry(level=2, title="1.1 Background", page=4),
            ]
        )
        blocks = segment_extracted_document(doc)
        assert blocks[0].extractor_hint == BlockRole.TOC_NAV
        entries = blocks[0].extra.get("entries", [])
        assert len(entries) == 2
        assert entries[0]["title"] == "Chapter 1: Intro"
        assert entries[0]["level"] == 1
        assert entries[1]["level"] == 2

    def test_empty_toc_emits_no_toc_nav(self):
        doc = _doc_with_toc([])
        blocks = segment_extracted_document(doc)
        for block in blocks:
            assert block.extractor_hint != BlockRole.TOC_NAV

    def test_toc_entries_have_level_page_title(self):
        doc = _doc_with_toc(
            [ExtractedTOCEntry(level=3, title="A.1.1 Sub-sub", page=12)]
        )
        blocks = segment_extracted_document(doc)
        entries = blocks[0].extra["entries"]
        assert entries[0] == {"level": 3, "title": "A.1.1 Sub-sub", "page": 12}

    def test_empty_titles_excluded(self):
        """Titleless TOC rows are filtered out before emission."""
        doc = _doc_with_toc(
            [
                ExtractedTOCEntry(level=1, title="", page=1),
                ExtractedTOCEntry(level=1, title="Real Chapter", page=2),
            ]
        )
        blocks = segment_extracted_document(doc)
        entries = blocks[0].extra["entries"]
        assert len(entries) == 1
        assert entries[0]["title"] == "Real Chapter"


@pytest.mark.unit
@pytest.mark.dart
class TestTocNavTemplate:
    def _classified(self, entries):
        from DART.converter.block_roles import RawBlock

        raw = RawBlock(
            text="toc",
            block_id="tocblk",
            page=1,
            extractor="pymupdf",
            extractor_hint=BlockRole.TOC_NAV,
            extra={"entries": entries},
        )
        classifier = HeuristicClassifier()
        return classifier.classify_sync([raw])[0]

    def test_template_emits_doc_toc_role_and_nested_ol(self):
        classified = self._classified(
            [
                {"level": 1, "title": "Chapter 1: Foo", "page": 3},
                {"level": 2, "title": "1.1 Background", "page": 4},
                {"level": 2, "title": "1.2 Related Work", "page": 6},
                {"level": 1, "title": "Chapter 2: Bar", "page": 10},
            ]
        )
        html = render_block(classified)
        assert '<nav role="doc-toc"' in html
        assert "<h2" in html and "Contents" in html
        # The first level-1 entry opens the top-level ol.
        assert html.startswith('<nav role="doc-toc"') or '<nav role="doc-toc"' in html[:200]
        # Nested ol contains level-2 entries.
        assert html.count("<ol>") >= 2

    def test_chapter_anchors_resolve_to_chap_N(self):
        classified = self._classified(
            [{"level": 1, "title": "Chapter 3: Dynamics", "page": 42}]
        )
        html = render_block(classified)
        assert 'href="#chap-3"' in html
        assert "Chapter 3: Dynamics" in html

    def test_section_anchors_resolve_to_sec_N_M(self):
        classified = self._classified(
            [{"level": 2, "title": "2.1 Background", "page": 8}]
        )
        html = render_block(classified)
        assert 'href="#sec-2-1"' in html

    def test_fallback_anchor_uses_page_when_no_chapter_or_section(self):
        classified = self._classified(
            [{"level": 1, "title": "Preface", "page": 12}]
        )
        html = render_block(classified)
        assert 'href="#page-12"' in html
