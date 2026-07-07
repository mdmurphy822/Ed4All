"""P3d tests — input-type detection + the vendor-ingest conversion seam.

The DUAL-SOURCE companion to ``test_semantik_dispatch_flip.py``. SemantiK
converts a PDF; a publisher's already-accessible HTML (e.g. a CC-BY publisher)
routes through ``_run_vendor_ingest_conversion`` instead. These tests prove
the input-type detection routes HTML→vendor / PDF→cascade and that the vendor
seam writes the SAME ``{stem}_accessible.html`` + sidecars contract the PDF
seam writes.

Run with NO models / GPU. The SemantiK cascade seam is MOCKED so the PDF arm
is exercised without touching SemantiK's heavy runtime deps; the vendor arm
runs the REAL ``ingest_vendor_html`` over a tiny inline accessible-HTML fixture
(pure BeautifulSoup + the model-free adapter — no GPU).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu CUDA_VISIBLE_DEVICES="" \
    python -m pytest MCP/tests/test_vendor_ingest_seam.py -q

Covers (SemantiK migration P3d):
  * ``_detect_conversion_input_type`` rule: PDF file / dir → "pdf"; HTML file /
    dir → "vendor"; missing path classified by suffix; mixed dir → "pdf"
    (PDF wins); unknown → "unknown".
  * ``dart_conversion`` phase routes an HTML-dir input to the vendor seam, NOT
    the SemantiK cascade seam.
  * ``dart_conversion`` phase routes a PDF input to the SemantiK cascade seam,
    NOT the vendor seam.
  * ``dart_conversion`` phase fails closed clear on an unknown input.
  * ``_run_vendor_ingest_conversion`` writes {stem}_accessible.html + both
    sidecars, returns the required contract keys + data-dart-source="vendor".
  * Fail-closed-clear on an empty / unreadable input.
"""

from __future__ import annotations

import json

import pytest


# A tiny but realistic publisher-shaped accessible HTML page (CC-BY style).
_VENDOR_PAGE_HTML = """<!DOCTYPE html><html lang="en"><head><title>1-1</title>
</head><body><main>
<h2>1.1 Introduction to Whole Numbers</h2>
<p>By the end of this section, you will be able to use place value with whole
numbers and identify multiples.</p>
<h3>Use Place Value with Whole Numbers</h3>
<p>A whole number is one of the numbers 0, 1, 2, 3, and so on.</p>
<ul><li>ones</li><li>tens</li><li>hundreds</li></ul>
</main></body></html>"""


def _registry_tool():
    from MCP.tools.pipeline_tools import _build_tool_registry

    registry = _build_tool_registry()
    return registry["extract_and_convert_pdf"]


# ---------------------------------------------------------------------------
# (a) Input-type detection rule.
# ---------------------------------------------------------------------------


def test_detect_input_type_rules(tmp_path):
    from pathlib import Path

    import MCP.tools.pipeline_tools as pt

    # HTML file → vendor.
    html_file = tmp_path / "page.html"
    html_file.write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    assert pt._detect_conversion_input_type(html_file) == "vendor"

    htm_file = tmp_path / "page.htm"
    htm_file.write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    assert pt._detect_conversion_input_type(htm_file) == "vendor"

    # Directory of HTML pages (no PDF) → vendor.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "p1.html").write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    (corpus / "p2.html").write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    assert pt._detect_conversion_input_type(corpus) == "vendor"

    # Our own {stem}_accessible.html output is NOT counted as vendor input.
    out_only = tmp_path / "outdir"
    out_only.mkdir()
    (out_only / "x_accessible.html").write_text("<html></html>", encoding="utf-8")
    assert pt._detect_conversion_input_type(out_only) == "unknown"

    # Existing PDF file → pdf.
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    assert pt._detect_conversion_input_type(pdf_file) == "pdf"

    # Mixed dir (PDF + HTML) → pdf wins.
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "a.html").write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    (mixed / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    assert pt._detect_conversion_input_type(mixed) == "pdf"

    # Non-existent path classified by suffix (back-compat with the P3c
    # bare-pdf_path dispatch-flip contract).
    assert pt._detect_conversion_input_type(Path("missing.pdf")) == "pdf"
    assert pt._detect_conversion_input_type(Path("missing.html")) == "vendor"
    assert pt._detect_conversion_input_type(Path("missing.txt")) == "unknown"


# ---------------------------------------------------------------------------
# (b) dart_conversion phase → routes HTML to the vendor seam (not cascade).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dart_conversion_html_dir_routes_to_vendor(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    cascade_calls: list = []

    def _fake_cascade(pdf_path, output_path, **kwargs):
        cascade_calls.append((pdf_path, output_path, kwargs))
        return {"success": True, "method": "semantik_v2", "html_path": output_path}

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_cascade)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1-1.html").write_text(_VENDOR_PAGE_HTML, encoding="utf-8")

    tool = _registry_tool()
    out = await tool(
        pdf_path=str(corpus),
        course_code="EA_1",
        phase="dart_conversion",
        output_dir=str(tmp_path / "out"),
    )
    payload = json.loads(out)

    assert cascade_calls == [], "SemantiK cascade was called for an HTML input"
    assert payload["success"] is True
    assert payload["method"] == "vendor_ingest"
    assert payload["data_dart_source"] == "vendor"
    # The accessible-HTML + sidecars contract is written.
    from pathlib import Path

    html_path = Path(payload["html_path"])
    assert html_path.exists() and html_path.name.endswith("_accessible.html")
    assert Path(payload["synthesized_sidecar_path"]).exists()
    assert Path(payload["quality_sidecar_path"]).exists()
    # Provenance survives into the emitted HTML.
    assert 'data-dart-source="vendor"' in html_path.read_text(encoding="utf-8")
    # Required Ed4All tool JSON keys present (parity with the PDF seam).
    for key in (
        "output_path",
        "output_paths",
        "html_path",
        "html_paths",
        "html_length",
        "word_count",
    ):
        assert key in payload, f"missing contract key {key}"


# ---------------------------------------------------------------------------
# (c) dart_conversion phase → routes a PDF to the cascade (not vendor).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dart_conversion_pdf_routes_to_cascade(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    cascade_calls: list = []
    vendor_calls: list = []

    def _fake_cascade(pdf_path, output_path, **kwargs):
        cascade_calls.append((pdf_path, output_path, kwargs))
        return {
            "success": True,
            "method": "semantik_v2",
            "html_path": output_path,
            "output_path": output_path,
        }

    def _fake_vendor(html_path_or_dir, output_path, **kwargs):
        vendor_calls.append((html_path_or_dir, output_path, kwargs))
        return {"success": True, "method": "vendor_ingest"}

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_cascade)
    monkeypatch.setattr(pt, "_run_vendor_ingest_conversion", _fake_vendor)

    tool = _registry_tool()
    out = await tool(
        pdf_path="sample_text.pdf",  # bare path; classified pdf by suffix
        course_code="ALG_9",
        phase="dart_conversion",
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    assert len(cascade_calls) == 1, "PDF input did not reach the SemantiK seam"
    assert vendor_calls == [], "PDF input wrongly reached the vendor seam"
    assert payload["method"] == "semantik_v2"


# ---------------------------------------------------------------------------
# (d) dart_conversion phase → fails closed clear on an unknown input.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dart_conversion_unknown_input_fails_closed(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    monkeypatch.setattr(
        pt,
        "_run_semantik_v2_conversion",
        lambda *a, **k: pytest.fail("cascade must not run on unknown input"),
    )

    txt = tmp_path / "notes.txt"
    txt.write_text("some plain text", encoding="utf-8")

    tool = _registry_tool()
    out = await tool(
        pdf_path=str(txt),
        course_code="X",
        phase="dart_conversion",
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)
    assert payload["success"] is False
    assert "unrecognized input" in payload["error"]


# ---------------------------------------------------------------------------
# (e) The vendor seam directly: file input, multi-page input, fail-closed.
# ---------------------------------------------------------------------------


def test_vendor_seam_single_file(tmp_path):
    import MCP.tools.pipeline_tools as pt

    html_file = tmp_path / "1-1.html"
    html_file.write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    out_dir = tmp_path / "out"

    res = pt._run_vendor_ingest_conversion(
        str(html_file), str(out_dir / "1-1_accessible.html")
    )
    assert res["success"] is True
    assert res["method"] == "vendor_ingest"
    assert res["runtime_mode"] == "real"
    assert res["data_dart_source"] == "vendor"
    assert res["vendor_html_file_count"] == 1
    assert res["html_length"] > 0

    from pathlib import Path

    assert Path(res["html_path"]).exists()
    assert Path(res["synthesized_sidecar_path"]).exists()
    assert Path(res["quality_sidecar_path"]).exists()


def test_vendor_seam_multi_page_dir(tmp_path):
    import MCP.tools.pipeline_tools as pt

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1-1.html").write_text(_VENDOR_PAGE_HTML, encoding="utf-8")
    (corpus / "1-2.html").write_text(
        _VENDOR_PAGE_HTML.replace("1.1 Introduction", "1.2 Language of Algebra"),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    res = pt._run_vendor_ingest_conversion(str(corpus), str(out_dir))
    assert res["success"] is True
    assert res["vendor_html_file_count"] == 2
    # The dir name drives the {stem}_accessible.html.
    assert res["html_path"].endswith("corpus_accessible.html")
    assert res["data_dart_source"] == "vendor"


def test_vendor_seam_empty_input_fails_closed(tmp_path):
    import MCP.tools.pipeline_tools as pt

    empty = tmp_path / "empty"
    empty.mkdir()

    res = pt._run_vendor_ingest_conversion(str(empty), str(tmp_path / "out"))
    assert res["success"] is False
    assert res["method"] == "vendor_ingest"
    assert "no readable HTML" in res["error"]


def test_vendor_seam_whitespace_only_html_fails_closed(tmp_path):
    import MCP.tools.pipeline_tools as pt

    blank = tmp_path / "blank.html"
    blank.write_text("   \n  ", encoding="utf-8")

    res = pt._run_vendor_ingest_conversion(
        str(blank), str(tmp_path / "blank_accessible.html")
    )
    assert res["success"] is False
    assert res["method"] == "vendor_ingest"
