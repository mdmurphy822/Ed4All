"""Client tests: loopback guard, parse-with-fallback env knobs, Nemotron
thinking-off injection, transport-failure surfacing. No network — a fake
transport stands in for the vLLM seat."""

from __future__ import annotations

import urllib.error

import pytest

from lib.assistant.client import (
    ENV_ASSISTANT_AUTOSTART,
    ENV_ASSISTANT_BASE_URL,
    ENV_ASSISTANT_MAX_TOKENS,
    ENV_ASSISTANT_MODEL,
    ENV_ASSISTANT_TIMEOUT,
    AssistantClient,
    AssistantClientError,
    AssistantProviderNotLocal,
    AssistantSeatUnavailable,
    reset_seat_cache,
    resolve_assistant_autostart,
    resolve_assistant_base_url,
    resolve_assistant_max_tokens,
    resolve_assistant_model,
    resolve_assistant_timeout,
)


@pytest.fixture(autouse=True)
def _hermetic_seat_resolution(monkeypatch):
    """A transport-only client is DYNAMIC (no explicit base_url/model), so its
    chat() now re-resolves the active seat. Stub the default probe OFF (no
    network) and reset the TTL cache so every case starts clean and the dynamic
    fallback lands on today's static defaults (localhost:8004 / env model)."""
    monkeypatch.setattr("lib.assistant.client._default_probe", lambda base_url: False)
    reset_seat_cache()
    yield
    reset_seat_cache()


def _body(content="hello", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


# --------------------------------------------------------------------------- #
# Loopback guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://spark:8004/v1",
        "http://203.0.113.5:8004/v1",
        "https://integrate.api.nvidia.com/v1",
        "http://evil.example.com/v1",
    ],
)
def test_non_loopback_base_url_raises(monkeypatch, url):
    monkeypatch.setenv(ENV_ASSISTANT_BASE_URL, url)
    with pytest.raises(AssistantProviderNotLocal):
        resolve_assistant_base_url()


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8004/v1", "http://127.0.0.1:8004/v1", "http://[::1]:8004/v1"],
)
def test_loopback_base_url_accepted(monkeypatch, url):
    monkeypatch.setenv(ENV_ASSISTANT_BASE_URL, url)
    assert resolve_assistant_base_url() == url.rstrip("/")


def test_default_base_url(monkeypatch):
    monkeypatch.delenv(ENV_ASSISTANT_BASE_URL, raising=False)
    assert resolve_assistant_base_url() == "http://localhost:8004/v1"


def test_client_constructor_enforces_loopback(monkeypatch):
    monkeypatch.setenv(ENV_ASSISTANT_BASE_URL, "http://spark:8004/v1")
    with pytest.raises(AssistantProviderNotLocal):
        AssistantClient()


# --------------------------------------------------------------------------- #
# Parse-with-fallback knobs
# --------------------------------------------------------------------------- #


def test_model_default_and_override(monkeypatch):
    monkeypatch.delenv(ENV_ASSISTANT_MODEL, raising=False)
    assert resolve_assistant_model() == "nemotron-3-nano"
    monkeypatch.setenv(ENV_ASSISTANT_MODEL, "nemotron-3-nano-30b")
    assert resolve_assistant_model() == "nemotron-3-nano-30b"


@pytest.mark.parametrize("raw,expected", [("512", 512), ("garbage", 1024), ("-3", 1024), ("", 1024)])
def test_max_tokens_parse_with_fallback(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ASSISTANT_MAX_TOKENS, raw)
    assert resolve_assistant_max_tokens() == expected


@pytest.mark.parametrize("raw,expected", [("30", 30.0), ("nope", 120.0), ("0", 120.0), ("", 120.0)])
def test_timeout_parse_with_fallback(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ASSISTANT_TIMEOUT, raw)
    assert resolve_assistant_timeout() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("ON", True), ("0", False), ("garbage", False), ("", False)],
)
def test_autostart_parse_with_fallback(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ASSISTANT_AUTOSTART, raw)
    assert resolve_assistant_autostart() is expected


# --------------------------------------------------------------------------- #
# Thinking-off injection (Nemotron reasoning control)
# --------------------------------------------------------------------------- #


def test_thinking_off_injects_directive_and_template_kwargs(monkeypatch):
    monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", "1")
    captured = {}

    def transport(url, payload, timeout):
        captured.update(payload)
        return _body()

    client = AssistantClient(transport=transport)
    client.chat([{"role": "user", "content": "hi"}])

    first = captured["messages"][0]
    assert first == {"role": "system", "content": "detailed thinking off"}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_off_absent_by_default(monkeypatch):
    monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)
    captured = {}

    def transport(url, payload, timeout):
        captured.update(payload)
        return _body()

    client = AssistantClient(transport=transport)
    client.chat([{"role": "user", "content": "hi"}])

    assert "chat_template_kwargs" not in captured
    assert all(
        m.get("content") != "detailed thinking off" for m in captured["messages"]
    )


# --------------------------------------------------------------------------- #
# Failure surfacing — loud, never a canned fallback
# --------------------------------------------------------------------------- #


def test_transport_failure_raises_seat_unavailable():
    def transport(url, payload, timeout):
        raise urllib.error.URLError("connection refused")

    client = AssistantClient(transport=transport)
    with pytest.raises(AssistantSeatUnavailable) as exc_info:
        client.chat([{"role": "user", "content": "hi"}])
    assert "not serving" in str(exc_info.value) or "Start the nano" in str(exc_info.value)


def test_malformed_body_raises_client_error():
    client = AssistantClient(transport=lambda u, p, t: {"error": "boom"})
    with pytest.raises(AssistantClientError):
        client.chat([{"role": "user", "content": "hi"}])


def test_tools_are_forwarded_on_the_wire():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(payload)
        return _body()

    client = AssistantClient(transport=transport)
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    client.chat([{"role": "user", "content": "hi"}], tools=tools)
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"
