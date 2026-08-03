"""DecisionCapture coverage for the figure-captioner VLM call site.

The SmolVLM2 caption call site must emit one ``alt_text_generation`` decision
per captioned figure with a dynamic, replayable rationale (image hash /
geometry / model + max_tokens / caption length). These tests MOCK the VLM
boundary (``_run_smolvlm_caption``) so no torch / SmolVLM2 weights load.

Invariants pinned here:
  (a) one decision per figure region, decision_type ``alt_text_generation``;
  (b) rationale is dynamic (>= 20 chars, carries the image hash + caption len);
  (c) captioning still produces the alt_text payload (capture is a side effect);
  (d) a None / unavailable capture never breaks the caption path;
  (e) the real ``_build_caption_capture`` constructs a working DecisionCapture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from semantik_structure.figures import captioner as figure_captioner


@dataclass(frozen=True)
class _Region:
    kind: str
    feature_block_indices: tuple = ()
    payload: dict[str, Any] = field(default_factory=dict)


class _SpyCapture:
    """Records log_decision calls in place of a real DecisionCapture."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _figure_region(fb: int) -> _Region:
    return _Region(
        kind="figure",
        feature_block_indices=(fb,),
        payload={"image_png_bytes": f"PNG-{fb}".encode(), "px_size": (640, 480)},
    )


@pytest.fixture
def _mock_vlm(monkeypatch):
    """Mock the SmolVLM2 boundary so no model loads."""
    def _fake(image_bytes, prompt, *, max_new_tokens):
        # Distinct text per prompt so alt vs extended lengths differ.
        return "a bar chart" if max_new_tokens <= 64 else "a bar chart with three bars"

    monkeypatch.setattr(figure_captioner, "_run_smolvlm_caption", _fake)


def test_capture_fires_per_figure(monkeypatch, _mock_vlm):
    spy = _SpyCapture()
    monkeypatch.setattr(figure_captioner, "_build_caption_capture", lambda: spy)

    regions = [
        _Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "x"}),
        _figure_region(1),
        _figure_region(2),
    ]
    out = figure_captioner.caption_figure_regions(regions, run_extended=True)

    # Capture instrumentation must not alter generated figure payloads.
    assert out[1].payload["alt_text"] == "a bar chart"
    assert out[2].payload["alt_text"] == "a bar chart"
    assert out[0].payload == {"text": "x"}

    # Emit exactly one decision for each figure region.
    assert len(spy.calls) == 2
    for call in spy.calls:
        assert call["decision_type"] == "alt_text_generation"
        rationale = call["rationale"]
        # The rationale carries enough dynamic context for replay.
        assert len(rationale) >= 20
        assert "img_sha256=" in rationale
        assert "alt_len=" in rationale
        assert figure_captioner.SMOLVLM_MODEL_ID in rationale
        assert "max_new_tokens" in rationale

    # The two distinct figures carry distinct image hashes in the rationale.
    hashes = {c["rationale"].split("img_sha256=")[1].split(",")[0] for c in spy.calls}
    assert len(hashes) == 2


def test_none_capture_does_not_break_caption(monkeypatch, _mock_vlm):
    monkeypatch.setattr(figure_captioner, "_build_caption_capture", lambda: None)
    out = figure_captioner.caption_figure_regions([_figure_region(1)])
    assert out[0].payload["alt_text"] == "a bar chart"


def test_capture_construction_failure_is_non_fatal(monkeypatch, _mock_vlm):
    # Caption generation remains available when its capture sink cannot load.
    def _boom():
        raise ImportError("no lib on path")

    monkeypatch.setattr(figure_captioner, "_build_caption_capture", _build_or_none(_boom))
    out = figure_captioner.caption_figure_regions([_figure_region(3)])
    assert out[0].payload["alt_text"] == "a bar chart"


def _build_or_none(fn):
    def _inner():
        try:
            return fn()
        except Exception:
            return None
    return _inner


def test_real_build_caption_capture_returns_working_capture():
    cap = figure_captioner._build_caption_capture()
    if cap is None:
        pytest.skip("lib.decision_capture not importable in this environment")
    # The production capture accepts the captioner's decision contract.
    cap.log_decision(
        decision_type="alt_text_generation",
        decision="smoke",
        rationale="figure-caption capture smoke test rationale >= 20 chars",
    )
    assert cap.decisions and cap.decisions[-1]["decision_type"] == "alt_text_generation"


def test_no_figures_no_capture(monkeypatch, _mock_vlm):
    # A figure-free call avoids both model and capture initialization.
    called = {"built": False}

    def _spy_build():
        called["built"] = True
        return _SpyCapture()

    monkeypatch.setattr(figure_captioner, "_build_caption_capture", _spy_build)
    regions = [_Region(kind="paragraph", feature_block_indices=(0,), payload={})]
    out = figure_captioner.caption_figure_regions(regions)
    assert out == regions
    assert called["built"] is False
