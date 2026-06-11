"""B2 regression: when DART content classifies entirely into leaf blocks
(prose paragraphs + subheadings — the common textbook layout), the page
numbers tracked on each ``RawBlock`` (from pdftotext form-feed boundaries /
page-chrome printed-page labels) must survive to the emitted HTML via a
``data-dart-pages`` attribute on the assembler's fallback ``<section>``
wrapper, so the downstream chunker can harvest real per-chunk page
provenance.

Before this fix the fallback wrapper carried only
``data-dart-block-id="main-content"`` and ``data-dart-source`` — the page
numbers died at the HTML boundary because leaf templates never carry
provenance attributes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock  # noqa: E402
from DART.converter.document_assembler import (  # noqa: E402
    _aggregate_block_pages,
    assemble_html,
)


def _para(text: str, page: int | None, page_label: str | None = None) -> ClassifiedBlock:
    extra = {"page_label": page_label} if page_label else {}
    raw = RawBlock(text=text, block_id=f"b{page}", page=page, extra=extra)
    return ClassifiedBlock(
        raw=raw,
        role=BlockRole.PARAGRAPH,
        confidence=0.8,
        attributes={},
        classifier_source="heuristic",
    )


# ---------------------------------------------------------------------------
# _aggregate_block_pages — page-set formatting
# ---------------------------------------------------------------------------


def test_aggregate_contiguous_pages_to_range() -> None:
    blocks = [_para("a", 1), _para("b", 2), _para("c", 3)]
    assert _aggregate_block_pages(blocks) == "1-3"


def test_aggregate_single_page() -> None:
    assert _aggregate_block_pages([_para("a", 4)]) == "4"


def test_aggregate_non_contiguous_pages_to_list() -> None:
    blocks = [_para("a", 1), _para("b", 3), _para("c", 7)]
    assert _aggregate_block_pages(blocks) == "1,3,7"


def test_aggregate_prefers_printed_page_label() -> None:
    blocks = [_para("a", 1, page_label="164"), _para("b", 2, page_label="165")]
    assert _aggregate_block_pages(blocks) == "164-165"


def test_aggregate_empty_when_no_pages() -> None:
    assert _aggregate_block_pages([_para("a", None)]) == ""


# ---------------------------------------------------------------------------
# assemble_html — fallback-section page stamping
# ---------------------------------------------------------------------------


def test_fallback_section_carries_aggregated_pages() -> None:
    """A document that classifies as pure paragraphs across pages 1-3 emits
    the fallback wrapper with ``data-dart-pages="1-3"`` alongside the
    existing ``data-dart-block-id="main-content"``."""
    blocks = [_para("Intro prose.", 1), _para("Stage prose.", 2), _para("Recap.", 3)]
    html = assemble_html(blocks, title="Photosynthesis", metadata={})

    # Fallback wrapper present with both block-id and pages.
    assert 'data-dart-block-id="main-content"' in html
    m = re.search(
        r'<section class="dart-section"[^>]*data-dart-block-id="main-content"'
        r'[^>]*data-dart-pages="([^"]+)"',
        html,
    )
    assert m is not None, "fallback section missing data-dart-pages"
    assert m.group(1) == "1-3"


def test_fallback_section_omits_pages_when_unknown() -> None:
    """No page info on any block -> the fallback wrapper omits the
    attribute entirely (never an empty / lying value)."""
    blocks = [_para("Prose with no page.", None)]
    html = assemble_html(blocks, title="Doc", metadata={})
    assert 'data-dart-block-id="main-content"' in html
    # The fallback wrapper line carries no pages attribute.
    fallback = re.search(r'<section class="dart-section"[^>]*>', html)
    assert fallback is not None
    assert "data-dart-pages" not in fallback.group(0)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "tests/fixtures/pipeline/fixture_corpus.pdf").exists(),
    reason="pipeline fixture PDF not present",
)
def test_end_to_end_pdf_to_html_carries_pages() -> None:
    """Full converter path on the multi-page fixture PDF: form-feed page
    tracking flows through extraction -> segmentation -> classification ->
    assembly so the emitted HTML carries real page provenance."""
    import shutil

    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext binary not available")

    from DART.converter import (
        default_classifier,
        extract_document,
        segment_extracted_document,
    )

    pdf = PROJECT_ROOT / "tests/fixtures/pipeline/fixture_corpus.pdf"
    doc = extract_document(str(pdf))
    assert "\x0c" in doc.raw_text, "fixture should be multi-page (form feeds)"
    blocks = segment_extracted_document(doc)
    classified = default_classifier().classify_sync(blocks)
    html = assemble_html(classified, title="Photosynthesis", metadata={})

    assert "data-dart-block-id=" in html
    pages = re.findall(r'data-dart-pages="([^"]+)"', html)
    assert pages, "emitted HTML carries no data-dart-pages"
    # The fixture spans pages 1-3.
    assert any("1" in p for p in pages)
