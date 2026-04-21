"""Wave 25 Fix 6: doc-pagebreak anchors + TOC target validation.

Audit: Bates ``<nav role="doc-toc">`` emits 190 ``<li><a href="#page-N">``
entries; the body contains 0 ``id="page-N"`` anchors. 178 of the 190
TOC links are dead. Root cause: the TOC template fallback writes
``#page-N`` when the title doesn't match a chapter/section pattern,
but no ``doc-pagebreak`` anchors are emitted.

Wave 25:
1. Segmenter emits synthetic PAGE_BREAK blocks for TOC-referenced
   pages (only — avoids ~600-anchor bloat).
2. Template emits ``<span role="doc-pagebreak" id="page-N" aria-label>``.
3. Post-assembly pass rewrites orphan ``#page-N`` TOC links to the
   nearest in-body anchor on that page (or demotes them to
   aria-disabled spans).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.block_segmenter import segment_extracted_document
from DART.converter.block_templates import render_block
from DART.converter.cross_refs import resolve_cross_references


class _FakeTOCEntry:
    def __init__(self, level, title, page):
        self.level = level
        self.title = title
        self.page = page


def _make_doc(
    raw_text: str,
    *,
    toc=None,
    tables=None,
    figures=None,
):
    return SimpleNamespace(
        raw_text=raw_text,
        toc=toc or [],
        tables=tables or [],
        figures=figures or [],
        page_chrome=None,
    )


@pytest.mark.unit
@pytest.mark.dart
class TestPageBreakBlockEmission:
    def test_form_feed_with_toc_reference_emits_pagebreak(self):
        # 3 pages, TOC references page 2. Expect one PAGE_BREAK
        # block for page 2.
        raw = (
            "Body of page one content.\n"
            "\x0c"
            "Body of page two content.\n"
            "\x0c"
            "Body of page three content.\n"
        )
        doc = _make_doc(
            raw,
            toc=[_FakeTOCEntry(1, "Introduction", 2)],
        )
        blocks = segment_extracted_document(doc)
        pagebreaks = [b for b in blocks if b.extractor_hint == BlockRole.PAGE_BREAK]
        assert len(pagebreaks) == 1
        assert pagebreaks[0].page == 2

    def test_no_toc_reference_no_pagebreak(self):
        raw = (
            "Body of page one.\n"
            "\x0c"
            "Body of page two.\n"
            "\x0c"
            "Body of page three.\n"
        )
        doc = _make_doc(raw, toc=[])
        blocks = segment_extracted_document(doc)
        pagebreaks = [b for b in blocks if b.extractor_hint == BlockRole.PAGE_BREAK]
        assert len(pagebreaks) == 0

    def test_multiple_toc_pages_multiple_pagebreaks(self):
        raw = (
            "Body page 1.\n"
            "\x0c"
            "Body page 2.\n"
            "\x0c"
            "Body page 3.\n"
            "\x0c"
            "Body page 4.\n"
        )
        doc = _make_doc(
            raw,
            toc=[
                _FakeTOCEntry(1, "Chapter 1", 1),
                _FakeTOCEntry(1, "Chapter 2", 3),
            ],
        )
        blocks = segment_extracted_document(doc)
        pagebreak_pages = sorted(
            b.page for b in blocks if b.extractor_hint == BlockRole.PAGE_BREAK
        )
        assert pagebreak_pages == [1, 3]


@pytest.mark.unit
@pytest.mark.dart
class TestPageBreakTemplate:
    def test_template_emits_id_and_aria_label(self):
        raw = RawBlock(
            text="page-5",
            block_id="pb5",
            page=5,
            extractor="pdftotext",
            extra={"page": 5, "toc_target": True},
        )
        cb = ClassifiedBlock(
            raw=raw,
            role=BlockRole.PAGE_BREAK,
            confidence=1.0,
            attributes={"page": 5, "toc_target": True},
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert 'id="page-5"' in html
        assert 'role="doc-pagebreak"' in html
        assert 'aria-label="page 5"' in html


@pytest.mark.unit
@pytest.mark.dart
class TestTOCPageAnchorValidation:
    def _make_chapter(self, number: int, page: int):
        raw = RawBlock(
            text=f"Chapter {number} Heading",
            block_id=f"ch{number}",
            page=page,
            extractor="pdftotext",
        )
        return ClassifiedBlock(
            raw=raw,
            role=BlockRole.CHAPTER_OPENER,
            confidence=0.9,
            attributes={"heading_text": f"Chapter {number} Heading"},
            classifier_source="heuristic",
        )

    def test_valid_page_link_left_alone(self):
        # Body has <span id="page-5">; TOC link #page-5 is valid.
        html = (
            '<html><body>'
            '<nav><a href="#page-5">Intro</a></nav>'
            '<span id="page-5" role="doc-pagebreak"></span>'
            '<p>Body</p>'
            '</body></html>'
        )
        blocks = [self._make_chapter(1, page=5)]
        out = resolve_cross_references(html, blocks)
        assert '<a href="#page-5">Intro</a>' in out

    def test_orphan_page_link_rewritten_to_chap_anchor(self):
        # #page-5 has no emitted span, but chapter 1 lives on page 5
        # → rewrite to #chap-1.
        html = (
            '<html><body>'
            '<nav><a href="#page-5">Scenario</a></nav>'
            '<article id="chap-1">...</article>'
            '</body></html>'
        )
        blocks = [self._make_chapter(1, page=5)]
        out = resolve_cross_references(html, blocks)
        assert 'href="#chap-1"' in out
        assert 'href="#page-5"' not in out

    def test_orphan_page_link_no_anchor_demoted(self):
        # No anchor on page 99 → demote to aria-disabled span.
        html = (
            '<html><body>'
            '<nav><a href="#page-99">Scenario</a></nav>'
            '<p>Body</p>'
            '</body></html>'
        )
        blocks = [self._make_chapter(1, page=5)]
        out = resolve_cross_references(html, blocks)
        assert 'aria-disabled="true"' in out
        assert 'href="#page-99"' not in out

    def test_mixed_valid_and_orphan_handled_independently(self):
        # #page-5 valid (has id); #page-99 orphan (no chapter / anchor).
        html = (
            '<html><body>'
            '<nav>'
            '<a href="#page-5">V</a>'
            '<a href="#page-99">O</a>'
            '</nav>'
            '<span id="page-5" role="doc-pagebreak"></span>'
            '</body></html>'
        )
        blocks = []
        out = resolve_cross_references(html, blocks)
        assert '<a href="#page-5">V</a>' in out
        assert 'href="#page-99"' not in out
        assert 'aria-disabled="true"' in out


@pytest.mark.unit
@pytest.mark.dart
class TestDartMarkersStaysClean:
    def test_pagebreak_emission_does_not_trip_validator(self):
        # End-to-end: build a doc via segmenter + classifier +
        # assembler; assert the dart_markers gate still passes.
        from DART.converter import (
            default_classifier,
            segment_extracted_document,
        )
        from DART.converter.document_assembler import assemble_html
        from DART.converter.heuristic_classifier import HeuristicClassifier

        raw = (
            "First line paragraph page one body prose content.\n"
            "\x0c"
            "Chapter 1\n\nFirst section prose text.\n"
            "\x0c"
            "Second page prose text content here with meaning.\n"
        )
        doc = _make_doc(
            raw,
            toc=[
                _FakeTOCEntry(1, "Chapter 1", 2),
                _FakeTOCEntry(1, "Scenario", 3),
            ],
        )
        blocks = segment_extracted_document(doc)
        clf: HeuristicClassifier = default_classifier()
        classified = clf.classify_sync(blocks)
        html = assemble_html(classified, title="Book", metadata={})
        # The output must contain at least one PAGE_BREAK anchor for
        # the TOC-referenced page 2 (chapter 1) or 3 (scenario).
        assert 'role="doc-pagebreak"' in html
        # And the validator critical checks should still pass.
        from lib.validators.dart_markers import DartMarkersValidator

        validator = DartMarkersValidator()
        result = validator.validate({"html_content": html})
        # "critical issues" count must be zero.
        critical = [
            issue
            for issue in result.issues
            if getattr(issue, "severity", None) == "critical"
        ]
        assert critical == [], f"Expected no critical issues, got: {critical}"
