"""Unit tests — Part F Stage A Y-flip + flag gating (no pypdfium2, no PDF IO).

The PDF-object enumeration needs pypdfium2 (SemantiK runtime venv only), but
the load-bearing Y-flip transform is factored into a pure helper that is
testable here, and the flag/cache-key behaviour is pure env logic.
"""

from __future__ import annotations

from semantik_structure.extract_shared import (
    _detect_figures_enabled,
    _image_bbox_top_left,
)


def test_yflip_bottom_left_to_top_left():
    # Page 800pt tall. An image from bottom-left (L=100,B=200,R=300,T=600)
    # becomes top-left [100, 800-600=200, 300, 800-200=600].
    out = _image_bbox_top_left((100.0, 200.0, 300.0, 600.0), 800.0)
    assert out == [100.0, 200.0, 300.0, 600.0]


def test_yflip_inverts_y_axis():
    # An image hugging the TOP of the page (high T/B in bottom-left coords)
    # gets a SMALL y0 in top-left coords.
    out = _image_bbox_top_left((0.0, 780.0, 50.0, 795.0), 800.0)
    assert out[1] == 800.0 - 795.0  # y0 = 5
    assert out[3] == 800.0 - 780.0  # y1 = 20
    assert out[1] < out[3]


def test_detect_figures_flag_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    assert _detect_figures_enabled() is False
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "1")
    assert _detect_figures_enabled() is True
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "off")
    assert _detect_figures_enabled() is False
    monkeypatch.setenv("SEMANTIK_DETECT_FIGURES", "garbage")
    assert _detect_figures_enabled() is False
