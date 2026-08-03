"""GLM-OCR lane orchestration tests — STUBBED SDK + alt-text, NO GPU / seat.

Covers the lane wire-up, sidecar emission, the alt-text pass with a stub
client, and (when ``lib/semantik`` is importable) the end-to-end render of the
accessible HTML + ``data-semantik-*`` / ``semantik:`` provenance
contract from the lane's ``region_provenance``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from semantik_structure.glmocr import lane as lane_mod
from semantik_structure.glmocr.lane import (
    build_cascade_result_dict,
    run_glmocr_lane,
)
from semantik_structure.glmocr.transform import GlmPage

# repo root (for the optional lib.semantik adapter integration)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _StubSdkClient:
    """Returns fixed GlmPages regardless of the image paths."""

    def __init__(self, pages):
        self._pages = pages
        self.seen = None

    def parse_pages(self, image_paths):
        self.seen = list(image_paths)
        return self._pages


class _StubAltTextClient:
    def __init__(self):
        self.calls = 0

    def describe(self, image_b64, existing_caption):
        self.calls += 1
        return {"alt": "A labelled widget schematic.",
                "long_desc": None,
                "caption_suggestion": "Widget schematic."}


def _fixture_pages():
    return [
        GlmPage(page_no=1, width=1224, height=1584, image_path="p1.png", regions=[
            {"index": 0, "native_label": "doc_title", "content": "# Widgets",
             "bbox_2d": [100, 50, 900, 90]},
            {"index": 1, "native_label": "paragraph_title",
             "content": "1.1 Introduce Widgets", "bbox_2d": [100, 120, 900, 150]},
            {"index": 2, "native_label": "text",
             "content": "Widgets are a foundational concept in this unit.",
             "bbox_2d": [100, 180, 900, 220]},
            {"index": 3, "native_label": "image", "content": "",
             "bbox_2d": [100, 260, 500, 560]},
        ]),
    ]


def _patch_render(monkeypatch, tmp_path):
    def _fake_render(pdf_path, out_dir, *, dpi=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "page-1.png"
        png.write_bytes(b"\x89PNG\r\n")
        return [png]
    monkeypatch.setattr(lane_mod, "render_pdf_to_pngs", _fake_render)


def test_lane_emits_wire_contract_and_sidecars(monkeypatch, tmp_path):
    _patch_render(monkeypatch, tmp_path)
    client = _StubSdkClient(_fixture_pages())
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    result = run_glmocr_lane(pdf, tmp_path, sdk_client=client)

    # wire contract present + reasonable
    kinds = [p["region_kind"] for p in result.region_provenance]
    assert "heading" in kinds and "figure" in kinds
    assert client.seen  # the render output reached the SDK client

    # layout sidecar
    layout = json.loads(Path(result.layout_sidecar).read_text())
    assert layout["schema"] == "glmocr-layout/1.0"
    assert layout["pages"][0]["regions"][0]["native_label"] == "doc_title"

    # escalations sidecar (JSONL) — the caption-less figure is flagged
    lines = Path(result.escalations_sidecar).read_text().splitlines()
    recs = [json.loads(ln) for ln in lines if ln.strip()]
    assert any(r["reason"] == "caption_less_figure" for r in recs)
    assert all(r["schema"] == "glmocr-escalation/1.0" for r in recs)


def test_alt_text_pass_off_by_default(monkeypatch, tmp_path):
    _patch_render(monkeypatch, tmp_path)
    monkeypatch.delenv("SEMANTIK_ALTTEXT_PROVIDER", raising=False)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    alt = _StubAltTextClient()
    result = run_glmocr_lane(
        pdf, tmp_path, sdk_client=_StubSdkClient(_fixture_pages()),
        alttext_client=alt,
    )
    assert result.alt_text_generated == 0
    assert alt.calls == 0  # provider off → seat never called


def test_alt_text_pass_on_fills_figure_alt(monkeypatch, tmp_path):
    _patch_render(monkeypatch, tmp_path)
    monkeypatch.setenv("SEMANTIK_ALTTEXT_PROVIDER", "qwen30")
    # avoid the real Pillow crop path by stubbing the b64 crop
    monkeypatch.setattr(
        "semantik_structure.glmocr.alttext._crop_b64",
        lambda image_path, bbox: "ZmFrZQ==",
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    alt = _StubAltTextClient()
    result = run_glmocr_lane(
        pdf, tmp_path, sdk_client=_StubSdkClient(_fixture_pages()),
        alttext_client=alt,
    )
    fig = [p for p in result.region_provenance if p["region_kind"] == "figure"][0]
    assert fig["figure_alt"] == "A labelled widget schematic."
    assert fig["alt_source"] == "generated"
    assert result.alt_text_generated == 1


def test_build_cascade_result_dict_shape(monkeypatch, tmp_path):
    _patch_render(monkeypatch, tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = run_glmocr_lane(pdf, tmp_path, sdk_client=_StubSdkClient(_fixture_pages()))
    cascade = build_cascade_result_dict(result, pdf_path=str(pdf), return_html=True)
    assert cascade["lane_used"] == "glmocr"
    assert cascade["region_provenance"] is result.region_provenance
    assert "html" in cascade
    assert cascade["theta"]["action"] == "ship_with_flag"


def test_render_accessible_html_emits_provenance_contract(monkeypatch, tmp_path):
    pytest.importorskip("lib.semantik.adapter")
    _patch_render(monkeypatch, tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = run_glmocr_lane(
        pdf, tmp_path, sdk_client=_StubSdkClient(_fixture_pages()),
        render_html=True, pdf_stem="doc",
    )
    assert result.html_path and Path(result.html_path).exists()
    html = Path(result.html_path).read_text()
    # the data-semantik-* provenance contract is present
    assert "data-semantik-block-id" in html
    assert "data-semantik-source" in html
    # figure surfaced
    assert "<figure" in html


def test_parser_construction_serialized(monkeypatch):
    """Concurrent SDK parser construction must serialize on the module lock.

    transformers/accelerate model loading is not thread-safe (meta-tensor
    race seen live on the 4-worker fan-out, 2026-07-16); _make_parser must
    hold _PARSER_CONSTRUCT_LOCK around GlmOcr() so two workers never
    construct simultaneously.
    """
    import sys
    import threading
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor

    from semantik_structure.glmocr import sdk_client

    in_flight = 0
    overlap = []
    guard = threading.Lock()

    class _FakeGlmOcr:
        def __init__(self, **kwargs):
            nonlocal in_flight
            with guard:
                in_flight += 1
                if in_flight > 1:
                    overlap.append(in_flight)
            time.sleep(0.05)
            with guard:
                in_flight -= 1

        def close(self):
            pass

    fake_api = types.ModuleType("glmocr.api")
    fake_api.GlmOcr = _FakeGlmOcr
    fake_pkg = types.ModuleType("glmocr")
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "glmocr", fake_pkg)
    monkeypatch.setitem(sys.modules, "glmocr.api", fake_api)

    client = sdk_client.SdkGlmOcrClient(
        base_url="http://127.0.0.1:8002/v1", model="glm-ocr",
        workers=4, layout_device="cpu",
    )
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: client._make_parser(), range(4)))
    assert not overlap, f"concurrent constructions observed: {overlap}"


def test_render_dir_isolated_per_document(monkeypatch, tmp_path):
    """A prior document's leftover page renders must never leak into a later
    document's page list through shared-directory glob poisoning."""
    import subprocess as sp

    from semantik_structure.glmocr import sdk_client as sc

    monkeypatch.setattr(sc.shutil, "which", lambda _: "/usr/bin/pdftoppm")

    def fake_run(cmd, **kw):
        # emulate pdftoppm: render 3 pages for docA, 2 for docB
        prefix = Path(cmd[-1])
        n = 3 if "docA" in cmd[-2] else 2
        for i in range(1, n + 1):
            (prefix.parent / f"page-{i}.png").write_bytes(b"png")
        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    (tmp_path / "docA.pdf").write_bytes(b"%PDF")
    (tmp_path / "docB.pdf").write_bytes(b"%PDF")
    render = tmp_path / "_render"
    a = sc.render_pdf_to_pngs(tmp_path / "docA.pdf", render)
    b = sc.render_pdf_to_pngs(tmp_path / "docB.pdf", render)
    assert len(a) == 3
    assert len(b) == 2, f"docB page list poisoned: {[p.name for p in b]}"
    assert all("docB" in str(p) for p in b)
    # re-render of docA is idempotent (pre-clean)
    a2 = sc.render_pdf_to_pngs(tmp_path / "docA.pdf", render)
    assert len(a2) == 3


def test_alt_text_breaker_short_circuits_dead_seat(monkeypatch):
    """After N consecutive seat failures the remaining figures must not touch
    the client (a wedged seat otherwise costs one full HTTP timeout per
    figure — ~an hour per document; seen live 2026-07-17)."""
    from semantik_structure.glmocr import alttext as at

    monkeypatch.setenv("SEMANTIK_ALTTEXT_PROVIDER", "qwen30")
    monkeypatch.setattr(at, "_crop_b64", lambda *a, **k: "Zm9v")

    calls = {"n": 0}

    class _DeadSeat:
        def describe(self, b64, caption):
            calls["n"] += 1
            raise TimeoutError("read timed out")

    figures = [
        {"region_kind": "figure", "figure_alt": None, "caption_text": None,
         "source_page": 1, "pages": [1], "bbox": [0, 0, 10, 10],
         "first_raw_block_index": i, "native_label": "image"}
        for i in range(40)
    ]
    esc = []
    monkeypatch.setenv("SEMANTIK_ALTTEXT_CONCURRENCY", "1")  # deterministic order
    n = at.apply_alt_text(figures, {1: "/tmp/fake.png"}, client=_DeadSeat(),
                          escalations=esc)
    assert n == 0
    assert calls["n"] == 5, f"breaker did not trip at 5 (calls={calls['n']})"
    assert len(esc) == 40  # every figure escalated
    assert sum(1 for e in esc if "breaker open" in str(e.get("detail"))) == 35
