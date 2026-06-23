"""Unit tests — Part F Stage F sidecar write + ``image_src`` wiring +
bytes-not-in-JSON, plus the cascade_ir emitter ``<img src>`` fill.

``cascade.py`` transitively imports heavy GPU/browser deps (axe / playwright
/ torch / llama_cpp) that are absent from a CI box, so we stub those modules
in ``sys.modules`` BEFORE importing the module — the sidecar-write function
itself is pure (Region + dataclasses.replace + Path IO only). No GPU, no PDF.
"""

from __future__ import annotations

import sys
import types

import pytest

# --- Stub the heavy import chain so ``dart_semantic.cascade`` imports here. ---
_STUBS = [
    "axe_playwright_python",
    "axe_playwright_python.sync_playwright",
    "playwright",
    "playwright.sync_api",
    "torch",
    "transformers",
    "peft",
    "sentence_transformers",
    "llama_cpp",
]
for _name in _STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["axe_playwright_python.sync_playwright"].Axe = object
sys.modules["playwright.sync_api"].sync_playwright = lambda *a, **k: None

from dart_semantic.cascade import (  # noqa: E402
    _build_region_provenance,
    _write_figure_sidecars,
    resolve_detect_figures,
    resolve_figure_caption_mode,
)
from dart_semantic.structure_graph import Region  # noqa: E402
from dart_semantic.types import FeatureBlock, RawBlock  # noqa: E402


def _figure_region(first_fb: int, png: bytes | None) -> Region:
    payload = {"alt_text": "A widget."}
    if png is not None:
        payload["image_png_bytes"] = png
    return Region(
        kind="figure",
        feature_block_indices=(first_fb,),
        payload=payload,
    )


def _image_fb(page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text="",
        page=page,
        bbox=(0.0, 0.0, 100.0, 100.0),
        page_width=612.0,
        page_height=792.0,
        source="pypdfium2:image",
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
        is_image=True,
    )


def test_sidecar_write_stamps_src_and_strips_bytes(tmp_path):
    pdf = tmp_path / "mybook.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    png = b"\x89PNG\r\n\x1a\nFAKE"
    regions = [_figure_region(7, png)]

    out = _write_figure_sidecars(regions, pdf)
    payload = out[0].payload

    # Sidecar written at the deterministic path.
    sidecar = tmp_path / "mybook_figures" / "fig-7.png"
    assert sidecar.exists()
    assert sidecar.read_bytes() == png

    # image_src is the RELATIVE path; raw bytes stripped from the payload.
    assert payload["image_src"] == "./mybook_figures/fig-7.png"
    assert "image_png_bytes" not in payload


def test_sidecar_no_bytes_is_text_only(tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF")
    out = _write_figure_sidecars([_figure_region(3, None)], pdf)
    assert "image_src" not in out[0].payload
    assert not (tmp_path / "b_figures").exists()


def test_sidecar_nonfigure_passthrough(tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF")
    para = Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "hi"})
    out = _write_figure_sidecars([para], pdf)
    assert out[0] is para or out[0].kind == "paragraph"


def test_bytes_never_reach_region_provenance(tmp_path):
    """The JSON-bridge source (region_provenance) carries image_src + alt,
    never the raw PNG bytes."""
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF")
    png = b"\x89PNGbytes"
    regions = _write_figure_sidecars([_figure_region(2, png)], pdf)
    fbs = [_image_fb()] * 3  # index 2 resolvable
    prov = _build_region_provenance(
        region_order=[0],
        regions=regions,
        feature_blocks=fbs,
        stage7_results={},
    )
    entry = prov[0]
    assert entry["region_kind"] == "figure"
    assert entry["image_src"] == "./b_figures/fig-2.png"
    assert entry["figure_alt"] == "A widget."
    # No bytes anywhere in the JSON-safe provenance dict.
    assert all("image_png_bytes" != k for k in entry)
    assert b"PNG" not in repr(entry).encode()


def test_region_provenance_no_image_src_when_absent(tmp_path):
    """Byte-stable: a figure with no src omits the image_src key entirely."""
    fbs = [_image_fb()]
    region = Region(kind="figure", feature_block_indices=(0,), payload={"alt_text": "x"})
    prov = _build_region_provenance([0], [region], fbs, {})
    assert "image_src" not in prov[0]


def test_resolve_detect_figures_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    assert resolve_detect_figures() is False
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "on")
    assert resolve_detect_figures() is True


def test_resolve_figure_caption_mode(monkeypatch):
    monkeypatch.delenv("SEMANTIK_FIGURE_CAPTION", raising=False)
    assert resolve_figure_caption_mode() == "auto"
    monkeypatch.setenv("SEMANTIK_FIGURE_CAPTION", "0")
    assert resolve_figure_caption_mode() == "off"
    monkeypatch.setenv("SEMANTIK_FIGURE_CAPTION", "1")
    assert resolve_figure_caption_mode() == "on"
    monkeypatch.setenv("SEMANTIK_FIGURE_CAPTION", "weird")
    assert resolve_figure_caption_mode() == "auto"
