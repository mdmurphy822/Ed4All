"""Regression: SEMANTIK_OCR_RENDER_SCALE tunes the Tesseract OCR raster DPI.

``_tesseract_page_blocks`` renders the page for OCR via pypdfium2 at a scale
that used to be hardcoded to 2.0 (~144 DPI on a 612x792pt page — marginal for
~10pt print on a rasterized/image-only scan). ``resolve_ocr_render_scale``
makes the scale an operator knob, read at call time with parse-with-fallback:
unset / garbage / non-positive -> 2.0; a valid positive float -> that float.

The render + pytesseract boundary is mocked, so no pypdfium2 render and no real
OCR run — the test asserts only the resolved scale that reaches the renderer.
CPU-only, no model load, no PDF IO.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from dart_semantic import extract_shared
from dart_semantic.extract_shared import resolve_ocr_render_scale


# ---- resolver: parse-with-fallback --------------------------------------


def test_default_scale_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_OCR_RENDER_SCALE", raising=False)
    assert resolve_ocr_render_scale() == 2.0


def test_env_set_overrides(monkeypatch):
    monkeypatch.setenv("SEMANTIK_OCR_RENDER_SCALE", "4.0")
    assert resolve_ocr_render_scale() == 4.0


@pytest.mark.parametrize("bad", ["garbage", "", "0", "-1.5", "nan", "inf"])
def test_garbage_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("SEMANTIK_OCR_RENDER_SCALE", bad)
    assert resolve_ocr_render_scale() == 2.0


# ---- call site: the resolved scale reaches the renderer -----------------


def _fake_pytesseract():
    """A stand-in pytesseract exposing the minimal image_to_data surface."""
    mod = types.ModuleType("pytesseract")
    mod.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(image, output_type=None):
        # No words -> _tesseract_page_blocks emits zero blocks (OCR bypassed).
        return {
            "text": [],
            "block_num": [],
            "par_num": [],
            "line_num": [],
            "height": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
        }

    mod.image_to_data = image_to_data
    return mod


def _run_capturing_scale(monkeypatch):
    captured = {}

    class _Img:
        width = 100
        height = 100

    def fake_render(pdf_path, page_num, scale=2.0):
        captured["scale"] = scale
        return _Img()

    monkeypatch.setattr(
        extract_shared, "_pypdfium2_render_page_to_image", fake_render
    )
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract())
    out = extract_shared._tesseract_page_blocks(Path("/nonexistent.pdf"), 1)
    assert out["text_blocks"] == []
    return captured["scale"]


def test_call_site_uses_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_OCR_RENDER_SCALE", raising=False)
    assert _run_capturing_scale(monkeypatch) == 2.0


def test_call_site_uses_env_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_OCR_RENDER_SCALE", "4.0")
    assert _run_capturing_scale(monkeypatch) == 4.0


def test_call_site_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("SEMANTIK_OCR_RENDER_SCALE", "garbage")
    assert _run_capturing_scale(monkeypatch) == 2.0
