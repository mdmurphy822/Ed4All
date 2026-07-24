"""Model-agnosticism net: Nemotron nano is a DEPLOYMENT DEFAULT only.

With ED4ALL_ASSISTANT_MODEL / _BASE_URL / _SEAT overridden to a non-nano
stack, the client/engine must build requests against exactly those values,
the seat autostart path must target the overridden seat name, and no
"nemotron"/"nano" string may appear in any outgoing request or seat
resolution."""

from __future__ import annotations

import json

import pytest

from lib.assistant.client import (
    AssistantClient,
    autostart_seat,
    reset_seat_cache,
    resolve_active_seat,
    resolve_assistant_seat,
    seat_start_hint,
)
from lib.assistant.engine import AssistantEngine


@pytest.fixture(autouse=True)
def _hermetic_seat_resolution(monkeypatch):
    """Dynamic clients re-resolve the active seat per chat(); stub the default
    probe OFF (no network) + reset the TTL cache between cases."""
    monkeypatch.setattr("lib.assistant.client._default_probe", lambda base_url: False)
    reset_seat_cache()
    yield
    reset_seat_cache()


class _RecordingCapture:
    def log_decision(self, **kwargs):
        pass


def _final_body(content):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_overridden_stack_carries_no_nano_assumptions(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_BASE_URL", "http://127.0.0.1:9123/v1")
    monkeypatch.setenv("ED4ALL_ASSISTANT_MODEL", "qwen-chat")
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT", "my-seat")
    monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)

    wire = {}

    def transport(url, payload, timeout):
        wire["url"] = url
        wire["payload"] = payload
        return _final_body("hi")

    client = AssistantClient(transport=transport)
    engine = AssistantEngine(client=client, capture=_RecordingCapture())
    engine.run_turn("hello")

    # Exactly the overridden values on the wire…
    assert wire["url"] == "http://127.0.0.1:9123/v1/chat/completions"
    assert wire["payload"]["model"] == "qwen-chat"
    # …and no nano/nemotron string anywhere in the outgoing request.
    request_text = json.dumps(wire["payload"]).lower() + wire["url"].lower()
    assert "nemotron" not in request_text
    assert "nano" not in request_text
    # Thinking-off surgery is CONDITIONAL (flag off → no chat_template_kwargs).
    assert "chat_template_kwargs" not in wire["payload"]


def test_seat_resolution_honors_override(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT", "my-seat")
    assert resolve_assistant_seat() == "my-seat"

    hint = seat_start_hint("http://127.0.0.1:9123/v1")
    assert "my-seat" in hint
    assert "nano" not in hint.lower()

    started = {}

    def fake_start(seat_name, run_dir=None):
        started["seat"] = seat_name

        class _R:
            ok = True

        return _R()

    monkeypatch.setattr("lib.assistant.client.start_seat_coherent", fake_start)
    autostart_seat()
    assert started["seat"] == "my-seat"


def test_seat_default_is_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT", "   ")
    assert resolve_assistant_seat() == "spark-nano"  # blank → default
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    assert resolve_assistant_seat() == "spark-nano"
    assert resolve_assistant_seat("explicit-seat") == "explicit-seat"


def test_resolve_active_seat_carries_no_model_name_assumptions(monkeypatch):
    """With registry + priority + served model id all overridden to a non-nano
    stack, resolve_active_seat carries no nano/nemotron/super assumption: the
    served model id is READ from the injected reader, and the chosen seat name +
    priority come straight from the overriding env — never a hardcoded name."""
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "alpha-seat=http://127.0.0.1:9001/v1,beta-seat=http://127.0.0.1:9002/v1",
    )
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT_PRIORITY", "alpha-seat,beta-seat")
    monkeypatch.delenv("ED4ALL_ASSISTANT_MODEL", raising=False)
    reset_seat_cache()

    # Only beta-seat is live; its served model id is read from the seat.
    probe = lambda url: url == "http://127.0.0.1:9002/v1"
    model_reader = lambda url: "custom-served-model" if url == "http://127.0.0.1:9002/v1" else None

    seat = resolve_active_seat(force=True, probe=probe, model_reader=model_reader)

    assert seat.seat_name == "beta-seat"
    assert seat.base_url == "http://127.0.0.1:9002/v1"
    assert seat.model == "custom-served-model"
    assert seat.live is True
    assert seat.source == "priority"

    blob = json.dumps(
        {"seat": seat.seat_name, "url": seat.base_url, "model": seat.model}
    ).lower()
    for banned in ("nemotron", "nano", "super"):
        assert banned not in blob
