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
    _aggregate_block_page_kind,
    _aggregate_block_pages,
    assemble_html,
)


def _para(
    text: str,
    page: int | None,
    page_label: str | None = None,
    page_kind: str | None = None,
) -> ClassifiedBlock:
    extra: dict = {}
    if page_label:
        extra["page_label"] = page_label
    if page_kind:
        extra["page_kind"] = page_kind
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
# _aggregate_block_page_kind — page-kind provenance ladder
# ---------------------------------------------------------------------------


def test_aggregate_kind_all_printed() -> None:
    blocks = [
        _para("a", 1, page_label="47", page_kind="printed"),
        _para("b", 2, page_label="48", page_kind="printed"),
    ]
    assert _aggregate_block_page_kind(blocks) == "printed"


def test_aggregate_kind_all_interpolated() -> None:
    blocks = [
        _para("a", 1, page_label="47", page_kind="interpolated"),
        _para("b", 2, page_label="48", page_kind="interpolated"),
    ]
    assert _aggregate_block_page_kind(blocks) == "interpolated"


def test_aggregate_kind_mixed_falls_to_weakest_physical() -> None:
    """One printed + one physical (no kind) -> weakest floor ``physical`` ->
    emitted as the absent-attribute default (empty string)."""
    blocks = [
        _para("a", 1, page_label="47", page_kind="printed"),
        _para("b", 2),  # physical (no page_kind)
    ]
    assert _aggregate_block_page_kind(blocks) == ""


def test_aggregate_kind_printed_plus_interpolated_falls_to_interpolated() -> None:
    """Mixed printed + interpolated -> weakest present is ``interpolated``."""
    blocks = [
        _para("a", 1, page_label="47", page_kind="printed"),
        _para("b", 2, page_label="48", page_kind="interpolated"),
    ]
    assert _aggregate_block_page_kind(blocks) == "interpolated"


def test_aggregate_kind_none_when_no_kind() -> None:
    """No constituent carries a kind -> emit nothing (absent == physical)."""
    blocks = [_para("a", 1), _para("b", 2)]
    assert _aggregate_block_page_kind(blocks) == ""


def test_aggregate_kind_ignores_pageless_blocks() -> None:
    """A block with a printed kind but NO page number never contributed a
    page, so it doesn't lift the wrapper's aggregate kind."""
    blocks = [
        _para("a", 1),  # physical, contributes page 1
        _para("b", None, page_kind="printed"),  # no page -> excluded
    ]
    assert _aggregate_block_page_kind(blocks) == ""


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


def test_fallback_section_emits_page_kind_when_all_printed() -> None:
    """All constituent pages share kind ``printed`` -> wrapper emits
    ``data-dart-page-kind="printed"`` alongside ``data-dart-pages``."""
    blocks = [
        _para("Intro.", 1, page_label="47", page_kind="printed"),
        _para("Recap.", 2, page_label="48", page_kind="printed"),
    ]
    html = assemble_html(blocks, title="Doc", metadata={})
    m = re.search(
        r'<section class="dart-section"[^>]*data-dart-block-id="main-content"'
        r'[^>]*data-dart-pages="47-48"[^>]*data-dart-page-kind="([^"]+)"',
        html,
    )
    assert m is not None, "fallback section missing data-dart-page-kind"
    assert m.group(1) == "printed"


def test_fallback_section_page_kind_all_interpolated() -> None:
    blocks = [
        _para("Intro.", 1, page_label="47", page_kind="interpolated"),
        _para("Recap.", 2, page_label="48", page_kind="interpolated"),
    ]
    html = assemble_html(blocks, title="Doc", metadata={})
    assert 'data-dart-page-kind="interpolated"' in html


def test_fallback_section_page_kind_mixed_falls_to_physical() -> None:
    """One printed + one physical -> the honest floor is ``physical``, which is
    the absent-attribute default: NO ``data-dart-page-kind`` is emitted (never
    an invented stronger label)."""
    blocks = [
        _para("Intro.", 1, page_label="47", page_kind="printed"),
        _para("Recap.", 2),  # physical
    ]
    html = assemble_html(blocks, title="Doc", metadata={})
    fallback = re.search(r'<section class="dart-section"[^>]*>', html)
    assert fallback is not None
    assert "data-dart-page-kind" not in fallback.group(0)
    # data-dart-pages still aggregates both contributing pages (sorted).
    assert 'data-dart-pages="2,47"' in fallback.group(0)


def test_fallback_section_no_page_kind_is_byte_identical() -> None:
    """No-regression: when NO constituent carries a page_kind (the whole
    existing corpus today), the wrapper output is BYTE-IDENTICAL to a run
    whose helper would have to add a kind attr — i.e. no
    ``data-dart-page-kind`` token appears anywhere in the document."""
    blocks = [_para("Intro prose.", 1), _para("Stage prose.", 2), _para("Recap.", 3)]
    html = assemble_html(blocks, title="Photosynthesis", metadata={})
    # Pages still flow through (unchanged behaviour).
    assert 'data-dart-pages="1-3"' in html
    # The new attribute never appears for kind-less constituents.
    assert "data-dart-page-kind" not in html


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
