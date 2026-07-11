"""DecisionCapture on the P0 VLM page-transcription call site.

The Qwen2.5-VL per-page transcription POST (``vlm_extract.extract_page_markdown``
/ ``_post_chat_completion``) must emit one ``structure_detection`` decision per
LIVE page transcription with a dynamic, replayable rationale (pdf sha / page /
model+provider / render pixels / prompt version / thinking-off flag /
max_tokens / produced markdown length + line count). These tests MOCK the VLM
HTTP boundary (an injected ``requests`` module) and the page render so no live
model call / pypdfium2 render / real PDF IO happens — mirroring
``test_vlm_extract.py`` conventions and the ``test_figure_captioner_capture.py``
spy pattern.

Invariants pinned here:
  (a) one decision per LIVE page transcription, decision_type
      ``structure_detection``;
  (b) rationale is dynamic (>= 20 chars, carries the pdf sha + page number +
      produced markdown length);
  (c) transcription still produces the markdown (capture is a side effect);
  (d) a None / unavailable capture never breaks extraction;
  (e) a CACHE HIT makes no POST → no LLM call → NO new decision;
  (f) the real ``_build_vlm_extract_capture`` constructs a working
      DecisionCapture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from semantik_structure import vlm_extract


# ---------------------------------------------------------------------------
# Fakes (mirror test_vlm_extract.py) + a decision-capture spy.
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, response=None):
        self.calls = []
        self._response = response

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self._response


def _ok_response(markdown="# Title\n\nBody prose $x^2$."):
    return _FakeResp(payload={"choices": [{"message": {"content": markdown}}]})


def _render_fn(w=1600, h=2000):
    return lambda pdf_path, page_num, scale: Image.new("RGB", (w, h), (255, 255, 255))


def _local_seat():
    from semantik_structure import extract_shared

    return extract_shared.resolve_vlm_seat()


class _SpyCapture:
    """Records log_decision calls in place of a real DecisionCapture."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _extract(page_num, *, requests_obj, cache_dir, pdf_sha="shaCAP", markdown=None):
    resp = _ok_response(markdown) if markdown is not None else _ok_response()
    requests_obj._response = resp
    return vlm_extract.extract_page_markdown(
        Path("x.pdf"),
        page_num,
        seat=_local_seat(),
        render_scale=2.0,
        timeout=180.0,
        cache_dir=cache_dir,
        render_fn=_render_fn(),
        requests_module=requests_obj,
        pdf_sha_override=pdf_sha,
    )


# ---------------------------------------------------------------------------
# (a)+(b)+(c) — capture fires per LIVE transcription with a dynamic rationale;
# transcription still works.
# ---------------------------------------------------------------------------
def test_capture_fires_per_live_page(monkeypatch, tmp_path):
    spy = _SpyCapture()
    monkeypatch.setattr(vlm_extract, "_build_vlm_extract_capture", lambda: spy)
    vlm_extract.begin_document_session()

    fake = _FakeRequests()
    out1 = _extract(1, requests_obj=fake, cache_dir=tmp_path,
                    markdown="# Page one\nprose line")
    out2 = _extract(2, requests_obj=fake, cache_dir=tmp_path,
                    markdown="# Page two\nmore prose\n\ntable row")

    # (c) transcription still produced the markdown on both pages.
    assert out1["markdown"] == "# Page one\nprose line"
    assert out2["markdown"] == "# Page two\nmore prose\n\ntable row"

    # (a) exactly one decision per LIVE page, decision_type structure_detection.
    assert len(spy.calls) == 2
    for call in spy.calls:
        assert call["decision_type"] == "structure_detection"
        rationale = call["rationale"]
        # (b) dynamic + replayable rationale (contract: >= 20 chars).
        assert len(rationale) >= 20
        assert "pdf_sha256=" in rationale
        assert "page=" in rationale
        assert "markdown_len=" in rationale
        assert "qwen2.5vl:7b" in rationale
        assert "prompt_version=" in rationale
        assert "max_tokens=" in rationale

    # The two pages carry DISTINCT page numbers + markdown lengths in the
    # rationale (genuinely dynamic, not boilerplate).
    pages = {c["rationale"].split("page=")[1].split(" ")[0] for c in spy.calls}
    assert pages == {"1", "2"}
    lens = {c["rationale"].split("markdown_len=")[1].split(" ")[0] for c in spy.calls}
    assert len(lens) == 2


# ---------------------------------------------------------------------------
# (e) — a CACHE HIT makes no POST → no LLM call → NO new decision.
# ---------------------------------------------------------------------------
def test_cache_hit_logs_no_decision(monkeypatch, tmp_path):
    spy = _SpyCapture()
    monkeypatch.setattr(vlm_extract, "_build_vlm_extract_capture", lambda: spy)
    vlm_extract.begin_document_session()

    fake = _FakeRequests()
    _extract(7, requests_obj=fake, cache_dir=tmp_path, markdown="# Cached")
    assert len(fake.calls) == 1
    assert len(spy.calls) == 1

    # Second extraction of the same page is a cache hit → no POST, no decision.
    fake2 = _FakeRequests()
    out = _extract(7, requests_obj=fake2, cache_dir=tmp_path, markdown="# Cached")
    assert fake2.calls == []
    assert out["markdown"] == "# Cached"
    assert len(spy.calls) == 1  # unchanged — no LLM call, no new decision


# ---------------------------------------------------------------------------
# (d) — a None / unavailable capture never breaks extraction.
# ---------------------------------------------------------------------------
def test_none_capture_does_not_break_extraction(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm_extract, "_build_vlm_extract_capture", lambda: None)
    vlm_extract.begin_document_session()
    out = _extract(1, requests_obj=_FakeRequests(), cache_dir=tmp_path,
                   markdown="# Body")
    assert out["markdown"] == "# Body"


def test_capture_construction_failure_is_non_fatal(monkeypatch, tmp_path):
    # Simulate lib.decision_capture unavailable — the REAL _build_* swallows the
    # ImportError and returns None, and extraction must still run.
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "lib.decision_capture":
            raise ImportError("no lib on path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    vlm_extract.begin_document_session()
    assert vlm_extract._build_vlm_extract_capture() is None
    out = _extract(1, requests_obj=_FakeRequests(), cache_dir=tmp_path,
                   markdown="# Body")
    assert out["markdown"] == "# Body"


def test_log_failure_is_non_fatal(monkeypatch, tmp_path):
    class _RaisingCapture:
        def log_decision(self, **kwargs):
            raise RuntimeError("disk full")

    monkeypatch.setattr(
        vlm_extract, "_build_vlm_extract_capture", lambda: _RaisingCapture()
    )
    vlm_extract.begin_document_session()
    out = _extract(1, requests_obj=_FakeRequests(), cache_dir=tmp_path,
                   markdown="# Body")
    assert out["markdown"] == "# Body"  # a logging crash never breaks extraction


# ---------------------------------------------------------------------------
# One capture per DOCUMENT (session), reused across pages.
# ---------------------------------------------------------------------------
def test_one_capture_built_per_document(monkeypatch, tmp_path):
    built = {"n": 0}

    def _spy_build():
        built["n"] += 1
        return _SpyCapture()

    monkeypatch.setattr(vlm_extract, "_build_vlm_extract_capture", _spy_build)
    vlm_extract.begin_document_session()
    fake = _FakeRequests()
    _extract(1, requests_obj=fake, cache_dir=tmp_path, markdown="# a")
    _extract(2, requests_obj=fake, cache_dir=tmp_path, markdown="# b")
    assert built["n"] == 1  # ONE capture for the whole document

    # A new document session rebuilds it.
    vlm_extract.begin_document_session()
    _extract(3, requests_obj=fake, cache_dir=tmp_path, markdown="# c")
    assert built["n"] == 2


# ---------------------------------------------------------------------------
# (f) — the real builder constructs a working DecisionCapture (isolated tmp via
# the repo-root autouse ED4ALL_TRAINING_CAPTURES_DIR fixture).
# ---------------------------------------------------------------------------
def test_real_build_vlm_extract_capture_returns_working_capture():
    cap = vlm_extract._build_vlm_extract_capture()
    if cap is None:
        pytest.skip("lib.decision_capture not importable in this environment")
    cap.log_decision(
        decision_type="structure_detection",
        decision="smoke",
        rationale="vlm-extract capture smoke test rationale >= 20 chars",
    )
    assert cap.decisions and cap.decisions[-1]["decision_type"] == "structure_detection"
