"""Wave 16 tests for structured-block segmentation + templating.

Covers the full dual-extraction flow:

* ``segment_extracted_document`` emits table + figure blocks with
  ``extractor_hint`` + ``extra`` populated.
* ``HeuristicClassifier`` honours ``extractor_hint`` at confidence 1.0
  and forwards ``extra`` into ``attributes``.
* ``LLMClassifier`` skips hinted blocks entirely — the injected
  backend is never called for them.
* ``TABLE`` template renders ``<thead>`` / ``<tbody>`` with
  ``scope="col"`` / ``scope="row"``.
* ``FIGURE`` template emits ``<figure>`` + ``<figcaption>`` + ``<img
  alt>`` when alt-text is populated.
* End-to-end: extracted doc -> segment -> classify -> assemble yields
  HTML carrying both the structured table and the figure.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from DART.converter import extractor as extractor_module
from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.block_segmenter import (
    segment_extracted_document,
    segment_pdftotext_output,
)
from DART.converter.block_templates import render_block
from DART.converter.document_assembler import assemble_html
from DART.converter.extractor import (
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    extract_document,
)
from DART.converter.heuristic_classifier import HeuristicClassifier
from DART.converter.llm_classifier import LLMClassifier


def _make_doc():
    return ExtractedDocument(
        raw_text="Intro paragraph one.\n\nIntro paragraph two.",
        source_pdf="/tmp/x.pdf",
        pages_count=1,
        tables=[
            ExtractedTable(
                page=1,
                bbox=(0.0, 0.0, 100.0, 100.0),
                header_rows=[["Name", "Age", "City"]],
                body_rows=[
                    ["Alice", "30", "Seattle"],
                    ["Bob", "42", "Portland"],
                ],
                caption="Table 1: Demographics",
            )
        ],
        figures=[
            ExtractedFigure(
                page=2,
                bbox=(1.0, 2.0, 3.0, 4.0),
                image_path="/tmp/fig1.png",
                alt_text="A histogram of sales by quarter",
                caption="Figure 1: Quarterly sales",
            )
        ],
        ocr_text=None,
    )


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSegmentExtractedDocument:
    def test_emits_table_block_with_structured_extra(self):
        doc = _make_doc()
        blocks = segment_extracted_document(doc)

        table_blocks = [b for b in blocks if b.extractor_hint == BlockRole.TABLE]
        assert len(table_blocks) == 1
        tb = table_blocks[0]
        assert tb.extractor == "pdfplumber"
        assert tb.extra["header_rows"] == [["Name", "Age", "City"]]
        assert tb.extra["body_rows"][0] == ["Alice", "30", "Seattle"]
        assert tb.extra["caption"] == "Table 1: Demographics"
        assert tb.page == 1

    def test_emits_figure_block_with_alt_text(self):
        doc = _make_doc()
        blocks = segment_extracted_document(doc)

        fig_blocks = [b for b in blocks if b.extractor_hint == BlockRole.FIGURE]
        assert len(fig_blocks) == 1
        fb = fig_blocks[0]
        assert fb.extractor == "pymupdf"
        assert fb.extra["image_path"] == "/tmp/fig1.png"
        assert fb.extra["alt"] == "A histogram of sales by quarter"
        assert fb.extra["caption"] == "Figure 1: Quarterly sales"

    def test_prose_blocks_still_emitted_alongside_structured(self):
        doc = _make_doc()
        blocks = segment_extracted_document(doc)
        prose = [b for b in blocks if b.extractor_hint is None]
        assert len(prose) >= 2  # Two intro paragraphs.

    def test_block_ids_are_unique_across_combined_sequence(self):
        doc = _make_doc()
        blocks = segment_extracted_document(doc)
        ids = [b.block_id for b in blocks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Classifier honours extractor_hint
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestHeuristicClassifierHonoursHint:
    def test_hinted_table_block_classified_at_full_confidence(self):
        raw = RawBlock(
            text="row1 | row2",
            block_id="abc123",
            extractor_hint=BlockRole.TABLE,
            extra={
                "header_rows": [["A", "B"]],
                "body_rows": [["1", "2"]],
                "caption": "Tbl",
            },
        )
        classified = HeuristicClassifier().classify_sync([raw])
        assert len(classified) == 1
        assert classified[0].role == BlockRole.TABLE
        assert classified[0].confidence == 1.0
        assert classified[0].attributes["body_rows"] == [["1", "2"]]

    def test_hinted_figure_block_classified_at_full_confidence(self):
        raw = RawBlock(
            text="caption",
            block_id="def456",
            extractor_hint=BlockRole.FIGURE,
            extra={"alt": "chart", "image_path": "/tmp/x.png", "caption": "c"},
        )
        classified = HeuristicClassifier().classify_sync([raw])
        assert classified[0].role == BlockRole.FIGURE
        assert classified[0].confidence == 1.0
        assert classified[0].attributes["alt"] == "chart"


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierSkipsHintedBlocks:
    def test_llm_never_called_for_hinted_blocks(self):
        from MCP.orchestrator.llm_backend import MockBackend

        # An empty-response backend would normally send everything to
        # the heuristic fallback. Here we use a permissive response
        # that would mislabel tables if the LLM saw them — proving
        # the hinted path bypasses the LLM entirely.
        permissive = MockBackend(
            response_fn=lambda sys, usr: '[{"block_id":"HINTED","role":"paragraph"}]'
        )

        hinted = RawBlock(
            text="table contents",
            block_id="HINTED",
            extractor_hint=BlockRole.TABLE,
            extra={"header_rows": [["H"]], "body_rows": [["v"]]},
        )
        unhinted = RawBlock(text="some prose", block_id="PROSE")

        classifier = LLMClassifier(llm=permissive)
        import asyncio

        results = asyncio.run(classifier.classify([hinted, unhinted]))

        # Verify the hinted block stayed as TABLE at confidence 1.0, not
        # downgraded to paragraph via the permissive LLM response.
        hinted_result = next(r for r in results if r.raw.block_id == "HINTED")
        assert hinted_result.role == BlockRole.TABLE
        assert hinted_result.confidence == 1.0
        assert hinted_result.classifier_source == "extractor_hint"

        # The only LLM call made should have been for the unhinted block.
        # permissive.calls reflects actual backend invocations.
        assert len(permissive.calls) == 1
        assert "PROSE" in permissive.calls[0].user
        assert "HINTED" not in permissive.calls[0].user


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestTableTemplateStructured:
    def test_thead_scope_col_on_every_header_cell(self):
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="caption",
                block_id="t1",
                extractor_hint=BlockRole.TABLE,
                extra={},
            ),
            role=BlockRole.TABLE,
            confidence=1.0,
            attributes={
                "header_rows": [["Name", "Age"]],
                "body_rows": [["Alice", "30"], ["Bob", "42"]],
                "caption": "Demographics",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert '<th scope="col">Name</th>' in html
        assert '<th scope="col">Age</th>' in html
        # First body cell is a row header.
        assert '<th scope="row">Alice</th>' in html
        assert "<td>30</td>" in html
        assert "Demographics" in html

    def test_table_template_falls_back_to_legacy_shape(self):
        cb = ClassifiedBlock(
            raw=RawBlock(text="caption", block_id="t2"),
            role=BlockRole.TABLE,
            confidence=0.9,
            attributes={},
            classifier_source="heuristic",
        )
        html = render_block(cb)
        # Never regresses below the Wave 13 minimal shape.
        assert '<table role="grid"' in html
        assert "<caption" in html
        assert "<thead>" in html
        assert "<tbody>" in html


@pytest.mark.unit
@pytest.mark.dart
class TestFigureTemplateStructured:
    def test_figure_template_emits_img_with_alt_when_populated(self):
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="caption",
                block_id="f1",
                extractor_hint=BlockRole.FIGURE,
            ),
            role=BlockRole.FIGURE,
            confidence=1.0,
            attributes={
                "image_path": "/static/fig1.png",
                "alt": "Histogram of quarterly sales",
                "caption": "Figure 1: Quarterly sales",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert '<img src="/static/fig1.png"' in html
        assert 'alt="Histogram of quarterly sales"' in html
        assert "<figcaption" in html
        assert "Figure 1: Quarterly sales" in html

    def test_figure_template_alt_missing_emits_role_presentation(self):
        """Wave 17: populated src + missing alt -> alt="" role="presentation"."""
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="caption",
                block_id="f2",
                extractor_hint=BlockRole.FIGURE,
            ),
            role=BlockRole.FIGURE,
            confidence=1.0,
            attributes={
                "image_path": "/static/fig2.png",
                # No "alt" key at all
                "caption": "Figure 2: Something",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert '<img src="/static/fig2.png"' in html
        assert 'alt=""' in html
        assert 'role="presentation"' in html

    def test_figure_template_never_emits_literal_figure_placeholder(self):
        """Wave 17: the string "(figure)" must never leak into HTML."""
        # Simulate what the segmenter produces when no caption / alt is
        # available: empty raw.text, empty caption / alt attributes.
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="",  # Wave 17 empty descriptor
                block_id="f3",
                extractor_hint=BlockRole.FIGURE,
            ),
            role=BlockRole.FIGURE,
            confidence=1.0,
            attributes={
                "image_path": "/static/fig3.png",
                "alt": "",
                "caption": "",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert "(figure)" not in html
        # Still emits a <figure> wrapper so schema.org microdata survives.
        assert "<figure" in html
        # No stray <figcaption> with placeholder content.
        assert "<figcaption" not in html

    def test_figure_template_empty_src_still_emits_figure_wrapper(self):
        """Wave 17: missing image_path still emits graceful <figure>."""
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="",
                block_id="f4",
                extractor_hint=BlockRole.FIGURE,
            ),
            role=BlockRole.FIGURE,
            confidence=1.0,
            attributes={
                "image_path": "",
                "alt": "",
                "caption": "Figure 4: Caption only.",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        # No <img> when src is empty.
        assert "<img" not in html
        # Caption present.
        assert "Figure 4: Caption only." in html
        # Never the placeholder.
        assert "(figure)" not in html

    def test_figure_template_llm_backed_alt_text_renders(self):
        """Wave 17: MockBackend alt-text surfaces through the template."""
        cb = ClassifiedBlock(
            raw=RawBlock(
                text="",
                block_id="f5",
                extractor_hint=BlockRole.FIGURE,
            ),
            role=BlockRole.FIGURE,
            confidence=1.0,
            attributes={
                "image_path": "/static/fig5.png",
                # Alt populated as if AltTextGenerator(llm=MockBackend) ran.
                "alt": "A chart showing quarterly revenue growth",
                "caption": "Figure 5: Revenue.",
            },
            classifier_source="extractor_hint",
        )
        html = render_block(cb)
        assert 'alt="A chart showing quarterly revenue growth"' in html
        assert 'role="presentation"' not in html


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestExtractorToHtmlEndToEnd:
    def test_extract_document_to_html_contains_structured_markup(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        def _fake_run(cmd, *args, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout=b"Chapter 1: Overview.\n\nIntro paragraph.",
                stderr=b"",
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        monkeypatch.setattr(
            extractor_module,
            "_extract_tables_pdfplumber",
            lambda p: [
                ExtractedTable(
                    page=1,
                    bbox=(0.0, 0.0, 100.0, 100.0),
                    header_rows=[["Metric", "Value"]],
                    body_rows=[["Accuracy", "0.95"]],
                    caption="Table 1: Results",
                )
            ],
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            extractor_module,
            "_extract_figures",
            lambda pdf_path, *, llm=None, **_: [
                ExtractedFigure(
                    page=2,
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    image_path="/tmp/fig.png",
                    alt_text="Sales chart",
                    caption="Figure 1: Sales",
                )
            ],
        )

        doc = extract_document(str(pdf))
        blocks = segment_extracted_document(doc)
        classified = HeuristicClassifier().classify_sync(blocks)
        html = assemble_html(classified, "Test Doc", {})

        # Table made it through with scoped headers.
        assert '<th scope="col">Metric</th>' in html
        assert '<th scope="row">Accuracy</th>' in html
        assert "Table 1: Results" in html

        # Figure made it through with alt-text and image src.
        assert '<img src="/tmp/fig.png"' in html
        assert 'alt="Sales chart"' in html
        assert "Figure 1: Sales" in html

    def test_raw_text_path_still_works_without_pdf(self):
        # A pdftotext-only segmenter call (no ExtractedDocument) should
        # still produce the baseline prose blocks with empty extractor_hint.
        blocks = segment_pdftotext_output("Hello.\n\nWorld.")
        assert all(b.extractor_hint is None for b in blocks)
        assert all(b.extra == {} for b in blocks)
