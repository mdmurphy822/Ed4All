"""Tests for the vendor-ingest path (publisher-supplied accessible HTML).

Exercises ``lib.semantik.vendor_ingest`` against a small synthetic vendor
HTML document and the REAL Ed4All contract validators (``DartMarkersValidator``,
``source_refs``, ``SemanticStructureExtractor``). Mirrors the SemantiK adapter
test assertions (the vendor path REUSES the adapter), with the added
``data-dart-source="vendor"`` provenance-discriminator checks.

NO models / NO GPU.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_vendor_ingest.py -q
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.vendor_ingest import (
    build_chapters_ir_from_html,
    ingest_vendor_html,
)


# A minimal but realistic vendor accessible-HTML page (publisher-shaped):
# h2 section boundaries, h3 in-section headings, paragraphs, a list, a table,
# and a figure with a figcaption.
_PAGE_1 = """
<main>
<div class="page" data-type="page">
  <h2>1.1 Whole Numbers</h2>
  <h3>Learning Objectives</h3>
  <p>By the end of this section you can use place value.</p>
  <ul><li>Identify place value</li><li>Round whole numbers</li></ul>
  <h3>Prime Numbers</h3>
  <p>A prime number is a counting number greater than 1 whose only
     factors are 1 and itself.</p>
  <figure><img src="line.png" alt="A number line from 0 to 10">
    <figcaption>Figure 1.1 A number line</figcaption></figure>
</div>
</main>
"""

_PAGE_2 = """
<main>
<div class="page" data-type="page">
  <h2>1.2 The Language of Algebra</h2>
  <h3>Variables and Constants</h3>
  <p>In algebra a variable is a letter that represents a number whose
     value can change.</p>
  <table><tr><th>Symbol</th><th>Meaning</th></tr>
    <tr><td>x</td><td>variable</td></tr></table>
</div>
</main>
"""


@pytest.fixture
def vendor_out():
    pages = [
        {"output_html": _PAGE_1, "title": "1.1 Whole Numbers", "page": "1-1"},
        {"output_html": _PAGE_2, "title": "1.2 Language of Algebra", "page": "1-2"},
    ]
    return ingest_vendor_html(
        pages,
        pdf_stem="sample-algebra-ch1",
        doc_title="Elementary Algebra (OpenStax, CC-BY)",
    )


def test_vendor_source_discriminator(vendor_out):
    """Every <section> carries data-dart-source="vendor" (authoritative),
    NOT the SemantiK "synthesized" default."""
    html = vendor_out["html"]
    assert vendor_out["data_dart_source"] == "vendor"
    assert 'data-dart-source="vendor"' in html
    assert 'data-dart-source="synthesized"' not in html
    assert 'data-dart-source=""' not in html
    # Every section open tag has the vendor value.
    sections = re.findall(r"<section\b[^>]*>", html)
    assert sections
    for tag in sections:
        assert 'data-dart-source="vendor"' in tag

    # Sidecar provenance agrees with the markup.
    for sec in vendor_out["synthesized_sidecar"]["sections"]:
        assert sec["provenance"]["sources"] == ["vendor"]
        assert sec["provenance"]["strategy"] == "vendor_ingest"
    assert vendor_out["synthesized_sidecar"]["document_provenance"][
        "extractors_used"
    ] == ["vendor"]


def test_vendor_dart_markers_pass(vendor_out):
    """The four critical DART markers are present; gate passes with no
    critical issues."""
    from lib.validators.dart_markers import DartMarkersValidator

    res = DartMarkersValidator().validate({"html_content": vendor_out["html"]})
    critical = [i for i in res.issues if i.severity == "critical"]
    assert res.passed, f"dart_markers failed: {[i.code for i in critical]}"
    assert not critical


def test_vendor_source_ids_resolve(vendor_out):
    """Every emitted data-dart-block-id forms a valid dart: sourceId AND
    resolves against the sidecar id universe (source_refs gate parity)."""
    from lib.validators.source_refs import SOURCE_ID_RE

    out = vendor_out
    slug = out["slug"]
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    html_ids = set(re.findall(r'data-dart-block-id="([^"]+)"', out["html"]))
    assert html_ids
    for bid in html_ids:
        source_id = f"dart:{slug}#{bid}"
        assert SOURCE_ID_RE.match(source_id), f"bad sourceId: {source_id}"
        assert bid in sidecar_ids, f"{bid} unresolved against sidecar"


def test_vendor_structure_extractor_no_collapse(vendor_out):
    """The structure extractor finds the per-page chapters and no chapter
    trips the >40-section collapse."""
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
        _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    )

    structure = SemanticStructureExtractor().extract(vendor_out["html"])
    chapters = structure["chapters"]
    assert len(chapters) >= 2
    for ch in chapters:
        assert len(ch.get("sections", [])) <= _STRUCTURE_COLLAPSE_SECTION_THRESHOLD


def test_vendor_region_kinds_and_figure_alt(vendor_out):
    """region_kind is derived from the source tag; figure_alt recovered from
    <figcaption>."""
    chapters = build_chapters_ir_from_html(
        [
            {"output_html": _PAGE_1, "title": "1.1", "page": "1-1"},
        ],
        doc_title="doc",
    )
    kinds = {b.region_kind for ch in chapters for b in ch.blocks}
    # paragraph + list + figure (+ heading blocks for h3) present.
    assert "paragraph" in kinds
    assert "list" in kinds
    assert "figure" in kinds
    # figure_alt recovered from figcaption.
    alts = [
        b.figure_alt for ch in chapters for b in ch.blocks if b.figure_alt
    ]
    assert any("number line" in (a or "").lower() for a in alts)
    # block_role carries the vendor discriminator down to the IR.
    roles = {b.block_role for ch in chapters for b in ch.blocks}
    assert any(r and r.startswith("vendor_") for r in roles)


def test_vendor_success_and_method(vendor_out):
    """Already-accessible vendor HTML ships with confidence."""
    assert vendor_out["success"] is True
    assert vendor_out["certification_status"] == "certified"
    assert vendor_out["method"] == "vendor_ingest"
    assert vendor_out["html_length"] == len(vendor_out["html"])
    assert vendor_out["word_count"] > 0


def test_adapter_default_still_synthesized():
    """The adapter source override defaults to 'synthesized' — the vendor
    value is opt-in only, so the SemantiK path is byte-unaffected."""
    from lib.semantik.adapter import (
        _AdapterBlock,
        _AdapterChapter,
        normalize_cascade_to_ed4all,
    )

    class _R:
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        chapters = [
            _AdapterChapter(
                title="Ch1",
                blocks=[
                    _AdapterBlock(
                        html="<p>x</p>",
                        region_kind="paragraph",
                        raw_block_index=0,
                        raw_text="x",
                        heading_text="Intro",
                    )
                ],
            )
        ]

    out = normalize_cascade_to_ed4all(_R(), pdf_stem="t")
    assert out["data_dart_source"] == "synthesized"
    assert 'data-dart-source="synthesized"' in out["html"]
    assert 'data-dart-source="vendor"' not in out["html"]
