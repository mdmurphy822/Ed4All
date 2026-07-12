"""Unit tests for the reasoning-QC VLM client (``reasoning_qc_vlm``).

The endpoint is MOCKED at the module-level ``_post_with_retry`` seam (and the
seat at ``resolve_vlm_seat``) so NO torch / ollama / weights load. These tests
pin the transport contract:

  (a) seat resolution honors ``SEMANTIK_REASONING_QC_MODEL`` + fail-loud
      ``require_ready`` on a non-loopback keyless hosted seat;
  (b) ``render_page`` produces a base64 JPEG (via a fake render primitive);
  (c) the composed body carries a text part + an image_url data URI,
      ``temperature=0``, the JSON-mode fields, and ``chat_template_kwargs``
      iff ``SEMANTIK_VLM_DISABLE_THINKING`` is set;
  (d) the JSON parse tolerates fenced / embedded output;
  (e) a ``VlmExtractError`` degrades to an EMPTY verdict (fail-soft, no raise).
"""
from __future__ import annotations

import os

import pytest

from semantik_structure import extract_shared, reasoning_qc_vlm, vlm_extract
from semantik_structure.extract_shared import VLMSeat, VLMSeatError


# ---------------------------------------------------------------------------
# (a) seat resolution + fail-loud require_ready.
# ---------------------------------------------------------------------------
def test_qc_model_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "nemotron-omni:120b")
    assert reasoning_qc_vlm.resolve_reasoning_qc_model(default_model="qwen2.5vl:7b") == "nemotron-omni:120b"


def test_qc_model_defaults_to_extraction_model(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MODEL", raising=False)
    assert reasoning_qc_vlm.resolve_reasoning_qc_model(default_model="qwen2.5vl:7b") == "qwen2.5vl:7b"


def test_resolve_seat_uses_override_model(monkeypatch):
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "resolve_vlm_seat",
        lambda: VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="qwen2.5vl:7b"),
    )
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "big-reasoner")
    seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    assert seat.model == "big-reasoner"
    assert seat.provider == "local"


def test_resolve_seat_fail_loud_on_keyless_hosted(monkeypatch):
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "resolve_vlm_seat",
        lambda: VLMSeat(provider="nvidia", base_url="https://api.nvidia.example/v1", api_key=None, model="m"),
    )
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MODEL", raising=False)
    with pytest.raises(VLMSeatError):
        reasoning_qc_vlm.resolve_reasoning_qc_seat()


# ---------------------------------------------------------------------------
# (b) render_page → base64 JPEG (fake render primitive; no pypdfium2 needed).
# ---------------------------------------------------------------------------
class _FakeImage:
    def __init__(self, w=800, h=1000):
        self.width = w
        self.height = h
        self.mode = "RGB"

    def resize(self, size):
        self.width, self.height = size
        return self

    def convert(self, mode):
        return self

    def save(self, buf, format="JPEG", quality=90):
        buf.write(b"\xff\xd8\xff\xe0JFIF-FAKE-JPEG\xff\xd9")


def test_render_page_returns_b64(monkeypatch):
    monkeypatch.setattr(
        extract_shared, "_pypdfium2_render_page_to_image", lambda p, n, scale=2.0: _FakeImage()
    )
    b64 = reasoning_qc_vlm.render_page("/tmp/x.pdf", 3, scale=2.0)
    assert isinstance(b64, str) and b64
    import base64

    raw = base64.b64decode(b64)
    assert raw.startswith(b"\xff\xd8")  # JPEG SOI


# ---------------------------------------------------------------------------
# (c) composed body shape.
# ---------------------------------------------------------------------------
class _CapturingResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _CapturingRequests:
    """Records the last POST body; returns a canned QC JSON completion."""

    def __init__(self, content='{"reading_order": [], "phantom_headings": []}'):
        self.last_body = None
        self._content = content

    def post(self, url, json=None, headers=None, timeout=None):
        self.last_body = json
        return _CapturingResp({"choices": [{"message": {"content": self._content}}]})


def test_post_body_shape(monkeypatch):
    monkeypatch.delenv("SEMANTIK_VLM_DISABLE_THINKING", raising=False)
    req = _CapturingRequests()
    out = reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="qwen2.5vl:7b",
        b64_jpeg="QUJD",
        block_texts=["First block", "Second block"],
        timeout=30.0,
        requests_module=req,
    )
    assert '"reading_order"' in out
    body = req.last_body
    assert body["temperature"] == 0
    assert body["model"] == "qwen2.5vl:7b"
    # JSON-mode fields present (transcription client has neither).
    assert body["response_format"] == {"type": "json_object"}
    assert body["format"] == "json"
    # Two-part user content: text + image_url data URI.
    user = body["messages"][1]["content"]
    kinds = {part["type"] for part in user}
    assert kinds == {"text", "image_url"}
    img = next(p for p in user if p["type"] == "image_url")
    assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # The ordered block list is in the text part.
    text = next(p for p in user if p["type"] == "text")["text"]
    assert "[0]" in text and "First block" in text
    # No chat_template_kwargs when thinking-off is unset.
    assert "chat_template_kwargs" not in body


def test_post_body_thinking_off(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_DISABLE_THINKING", "1")
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        b64_jpeg="QUJD",
        block_texts=["b"],
        timeout=30.0,
        requests_module=req,
    )
    assert req.last_body["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}


# ---------------------------------------------------------------------------
# (d) tolerant JSON parse.
# ---------------------------------------------------------------------------
def test_parse_qc_json_variants():
    assert reasoning_qc_vlm.parse_qc_json('{"a": 1}') == {"a": 1}
    assert reasoning_qc_vlm.parse_qc_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert reasoning_qc_vlm.parse_qc_json('noise {"a": 3} tail') == {"a": 3}
    assert reasoning_qc_vlm.parse_qc_json("") == {}
    assert reasoning_qc_vlm.parse_qc_json("not json") == {}
    assert reasoning_qc_vlm.parse_qc_json("[1,2,3]") == {}  # a list is not a verdict dict


# ---------------------------------------------------------------------------
# (e) run_qc_judgment fail-soft.
# ---------------------------------------------------------------------------
def test_run_qc_judgment_happy(monkeypatch):
    monkeypatch.setattr(reasoning_qc_vlm, "render_page", lambda *a, **k: "QUJD")
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "_post_with_retry",
        lambda **k: '{"reading_order": [1, 0], "phantom_headings": []}',
    )
    seat = VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="m")
    out = reasoning_qc_vlm.run_qc_judgment(seat, "/tmp/x.pdf", 1, ["a", "b"], requests_module=object())
    assert out == {"reading_order": [1, 0], "phantom_headings": []}


def test_run_qc_judgment_failsoft_on_vlm_error(monkeypatch):
    monkeypatch.setattr(reasoning_qc_vlm, "render_page", lambda *a, **k: "QUJD")

    def _boom(**k):
        raise vlm_extract.VlmExtractError("permanent boom", transient=False)

    monkeypatch.setattr(reasoning_qc_vlm, "_post_with_retry", _boom)
    seat = VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="m")
    out = reasoning_qc_vlm.run_qc_judgment(seat, "/tmp/x.pdf", 1, ["a"], requests_module=object())
    assert out == {}  # fail-soft empty verdict, no raise


def test_run_qc_judgment_failsoft_on_render_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("render blew up")

    monkeypatch.setattr(reasoning_qc_vlm, "render_page", _boom)
    seat = VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="m")
    out = reasoning_qc_vlm.run_qc_judgment(seat, "/tmp/x.pdf", 1, ["a"], requests_module=object())
    assert out == {}


def test_post_with_retry_is_mockable_module_level():
    # The seam the higher-level tests monkeypatch must exist as a module attr.
    assert callable(reasoning_qc_vlm._post_with_retry)
