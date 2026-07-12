"""DecisionCapture regression for the reasoning-QC VLM call site.

Mirrors ``tests/test_figure_captioner_capture.py``: a ``_SpyCapture`` stub
monkeypatched over ``reasoning_qc._build_reasoning_qc_capture`` asserts the QC
pass emits ONE ``structure_detection`` decision per QC unit (window) with a
dynamic, replayable rationale carrying the page number, model id, region-index
set, and order divergence. The VLM boundary is mocked (no torch/ollama/weights).

Invariants pinned here:
  (a) one decision per QC WINDOW, decision_type ``structure_detection``;
  (b) rationale >= 20 chars, carries page number + model id + region set +
      divergence/confidence;
  (c) the QC output (qc_audit) is still produced when capture is None;
  (d) a build-failure (lib not importable) is non-fatal;
  (e) the real ``_build_reasoning_qc_capture`` returns a working capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from semantik_structure import reasoning_qc


# ---------------------------------------------------------------------------
# Minimal Region / FeatureBlock / RawBlock stand-ins (no torch stack).
# ---------------------------------------------------------------------------
@dataclass
class _Raw:
    text: str
    page: int


@dataclass
class _FB:
    raw: _Raw


@dataclass(frozen=True)
class _Region:
    kind: str
    feature_block_indices: tuple = ()
    payload: dict = field(default_factory=dict)


class _SpyCapture:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Assembled:
    def __init__(self, region_provenance):
        self.region_provenance = region_provenance


def _two_page_doc():
    """Two regions on page 1, one on page 2 → two QC windows."""
    feature_blocks = [
        _FB(_Raw("1.1 Real Section Heading", 1)),
        _FB(_Raw("Body prose for the section.", 1)),
        _FB(_Raw("Chapter 2 running header", 2)),
    ]
    capped = [
        _Region(kind="heading", feature_block_indices=(0,)),
        _Region(kind="paragraph", feature_block_indices=(1,)),
        _Region(kind="heading", feature_block_indices=(2,)),
    ]
    region_order = [0, 1, 2]
    return capped, feature_blocks, _Assembled(region_order), region_order


@pytest.fixture
def _mock_vlm(monkeypatch):
    """Mock the seat + per-window judgment so no model loads."""
    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", lambda: object())
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)

    def _judge(seat, pdf_path, page_num, block_texts):
        if page_num == 2:
            # Flag the running-header block on page 2 as a phantom heading.
            return {"reading_order": [], "phantom_headings": [{"index": 0, "reason": "running header"}], "confidence": 0.9}
        return {"reading_order": [1, 0], "phantom_headings": [], "confidence": 0.8}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)
    monkeypatch.setattr(reasoning_qc, "resolve_reasoning_qc_model", lambda default_model=None: "qc-model:test")


# ---------------------------------------------------------------------------
# (a)+(b) capture fires per window with a dynamic rationale.
# ---------------------------------------------------------------------------
def test_capture_fires_per_window(monkeypatch, _mock_vlm):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    spy = _SpyCapture()
    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", lambda: spy)

    capped, fbs, assembled, order = _two_page_doc()
    new_capped, _verdicts, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )

    # Shadow → capped byte-identical (same object list).
    assert new_capped is capped
    # (a) one decision per WINDOW (two pages → two decisions).
    assert len(spy.calls) == 2
    for call in spy.calls:
        assert call["decision_type"] == "structure_detection"
        rationale = call["rationale"]
        # (b) dynamic + replayable.
        assert len(rationale) >= 20
        assert "page " in rationale
        assert "model=qc-model:test" in rationale
        assert "regions=" in rationale
        assert "divergence" in rationale
    # The audit records both windows + the phantom flag.
    assert audit["ran"] is True and audit["mode"] == "shadow"
    assert audit["capture_fired"] == 2
    assert any(f["failure_mode"] == "example_as_heading" for f in audit["flagged"])


# ---------------------------------------------------------------------------
# (c) None capture → QC still produces the audit.
# ---------------------------------------------------------------------------
def test_none_capture_still_produces_audit(monkeypatch, _mock_vlm):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", lambda: None)
    capped, fbs, assembled, order = _two_page_doc()
    _new_capped, _v, audit = reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert audit["ran"] is True
    assert audit["capture_fired"] == 0


# ---------------------------------------------------------------------------
# (d) build-failure is non-fatal.
# ---------------------------------------------------------------------------
def test_capture_build_failure_non_fatal(monkeypatch, _mock_vlm):
    # Even a builder that RAISES (a broken lib) must not break the QC pass —
    # run_reasoning_qc guards the _build call and degrades to no capture.
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")

    def _boom():
        raise ImportError("no lib on path")

    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", _boom)
    capped, fbs, assembled, order = _two_page_doc()
    _new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert audit["ran"] is True
    assert audit["capture_fired"] == 0


def test_real_builder_swallows_import_error(monkeypatch):
    # Force the lib import to fail inside the real builder → returns None.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "lib.decision_capture" or name.startswith("lib.decision_capture"):
            raise ImportError("simulated missing lib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert reasoning_qc._build_reasoning_qc_capture() is None


# ---------------------------------------------------------------------------
# (e) real builder returns a working capture.
# ---------------------------------------------------------------------------
def test_real_build_capture_returns_working_capture():
    cap = reasoning_qc._build_reasoning_qc_capture()
    if cap is None:
        pytest.skip("lib.decision_capture not importable in this environment")
    cap.log_decision(
        decision_type="structure_detection",
        decision="smoke",
        rationale="reasoning-QC capture smoke test rationale >= 20 chars",
    )
    assert cap.decisions and cap.decisions[-1]["decision_type"] == "structure_detection"
