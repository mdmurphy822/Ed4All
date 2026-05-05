"""Regression tests for the provider-agnostic ``llm_chat_call`` capture.

Wires ``DecisionCapture``-shaped events at the ``LLMBackend`` base
class so every concrete backend (``AnthropicBackend``, ``LocalBackend``,
``MailboxBrokeredBackend``, ``OpenAIBackend``, ``MockBackend``) emits
one ``decision_type="llm_chat_call"`` event per dispatch when a
capture is wired. Mirrors the canonical pattern at
``Trainforge/generators/_openai_compatible_client.py``.

LLM-agnostic intent: no hardcoded provider names appear in event
field names — only the ``provider`` audit value identifies the
backend. A future ``OpenAIBackend`` / ``TogetherBackend`` /
``OllamaBackend`` plugged in via the same mixin must surface its
own ``provider_label`` in the rationale without re-implementing
the capture pattern.

Coverage:

- Each backend that has a callable ``complete_sync`` body fires
  exactly one ``llm_chat_call`` event per call when capture wired.
- Rationale ≥20 chars and references dynamic signals (model,
  max_tokens, latency_ms, response_text_len, messages_count).
- ``provider_label`` flows into rationale + decision strings.
- ``capture=None`` is a clean no-op (no exception, no event).
- Capture failures inside ``log_decision`` do NOT crash the LLM
  dispatch (defensive try/except contract).
- ``build_backend(capture=...)`` threads the capture through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from MCP.orchestrator.llm_backend import (
    AnthropicBackend,
    BackendSpec,
    LocalBackend,
    MailboxBrokeredBackend,
    MockBackend,
    OpenAIBackend,
    build_backend,
)
from MCP.orchestrator.task_mailbox import TaskMailbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCapture:
    """DecisionCapture-shaped stub that records ``log_decision`` calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _RaisingCapture:
    """Capture that always raises — used to assert defensive try/except."""

    def __init__(self) -> None:
        self.call_count = 0

    def log_decision(self, **kwargs: Any) -> None:
        self.call_count += 1
        raise RuntimeError("simulated capture explosion")


def _assert_one_chat_call(cap: _FakeCapture, *, expected_provider: str) -> Dict[str, Any]:
    assert len(cap.events) == 1, f"expected 1 event, got {len(cap.events)}"
    event = cap.events[0]
    assert event["decision_type"] == "llm_chat_call"
    rationale = event["rationale"]
    decision = event["decision"]
    # Rationale ≥ 20 chars per the project contract.
    assert len(rationale) >= 20, rationale
    # Provider label flows into both surfaces.
    assert expected_provider in rationale
    assert expected_provider in decision
    # Dynamic signals interpolated.
    assert "model=" in rationale
    assert "max_tokens=" in rationale
    assert "latency_ms=" in rationale
    assert "messages_count=" in rationale
    assert "response_text_len=" in rationale
    return event


# ---------------------------------------------------------------------------
# MockBackend — simplest exerciser of the mixin contract
# ---------------------------------------------------------------------------


def test_mock_backend_emits_one_llm_chat_call():
    cap = _FakeCapture()
    backend = MockBackend(responses=["mock response"], capture=cap)
    out = backend.complete_sync(
        "system prompt",
        "user prompt",
        model="mock-model-id",
        max_tokens=128,
        temperature=0.3,
    )
    assert out == "mock response"
    event = _assert_one_chat_call(cap, expected_provider="mock")
    assert "mock-model-id" in event["rationale"]
    assert "max_tokens=128" in event["rationale"]
    # response_text_len reflects actual output length.
    assert "response_text_len=13" in event["rationale"]


def test_mock_backend_no_capture_is_clean_noop():
    """``capture=None`` must not raise and must not emit any event."""
    backend = MockBackend(responses=["ok"])
    out = backend.complete_sync("sys", "user", model="m", max_tokens=64)
    assert out == "ok"
    # Sanity: backend has no recorded capture state to leak.
    assert getattr(backend, "_capture", None) is None


@pytest.mark.asyncio
async def test_mock_backend_async_complete_emits_capture():
    cap = _FakeCapture()
    backend = MockBackend(responses=["async response"], capture=cap)
    out = await backend.complete(
        "sys", "user", model="m", max_tokens=32, temperature=0.0
    )
    assert out == "async response"
    _assert_one_chat_call(cap, expected_provider="mock")


# ---------------------------------------------------------------------------
# AnthropicBackend — patched SDK; no real network call
# ---------------------------------------------------------------------------


def _stub_anthropic_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def test_anthropic_backend_emits_one_llm_chat_call():
    cap = _FakeCapture()
    backend = AnthropicBackend(
        api_key="test-key",
        default_model="claude-test-model",
        capture=cap,
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("hi there")
    backend._client = fake_client

    out = backend.complete_sync(
        "system", "user", model="claude-test-model",
        max_tokens=256, temperature=0.5,
    )
    assert out == "hi there"
    fake_client.messages.create.assert_called_once()
    event = _assert_one_chat_call(cap, expected_provider="anthropic")
    assert "claude-test-model" in event["rationale"]
    assert "max_tokens=256" in event["rationale"]


def test_anthropic_backend_no_capture_is_clean_noop():
    backend = AnthropicBackend(api_key="test-key", default_model="claude-test")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("ok")
    backend._client = fake_client

    out = backend.complete_sync("s", "u", max_tokens=32)
    assert out == "ok"


def test_anthropic_backend_capture_failure_does_not_crash_dispatch():
    """Capture exceptions must be swallowed — LLM dispatch keeps working."""
    cap = _RaisingCapture()
    backend = AnthropicBackend(
        api_key="test-key", default_model="m", capture=cap,
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_anthropic_response("ok")
    backend._client = fake_client
    out = backend.complete_sync("s", "u", max_tokens=32)
    # Dispatch returned successfully despite capture explosion.
    assert out == "ok"
    assert cap.call_count == 1


# ---------------------------------------------------------------------------
# MailboxBrokeredBackend — real TaskMailbox with stubbed completion
# ---------------------------------------------------------------------------


def test_mailbox_brokered_backend_emits_one_llm_chat_call(tmp_path: Path):
    cap = _FakeCapture()
    mailbox = TaskMailbox(run_id="test-run", base_dir=tmp_path)
    backend = MailboxBrokeredBackend(
        mailbox,
        timeout_seconds=5.0,
        poll_interval=0.01,
        default_model="qwen2.5-7b-instruct",
        capture=cap,
    )

    # Stub mailbox to simulate a successful operator completion.
    mailbox.put_pending = MagicMock()
    mailbox.wait_for_completion = MagicMock(
        return_value={"success": True, "result": {"response_text": "hello from operator"}}
    )
    mailbox.cleanup = MagicMock()

    out = backend.complete_sync(
        "system", "user", model="qwen2.5-7b-instruct",
        max_tokens=512, temperature=0.2,
    )
    assert out == "hello from operator"
    event = _assert_one_chat_call(cap, expected_provider="mailbox")
    assert "qwen2.5-7b-instruct" in event["rationale"]
    assert "task_id=" in event["decision"]


def test_mailbox_backend_emits_capture_even_on_failure(tmp_path: Path):
    """Capture fires in ``finally`` so failed dispatches still emit events."""
    cap = _FakeCapture()
    mailbox = TaskMailbox(run_id="test-run", base_dir=tmp_path)
    backend = MailboxBrokeredBackend(
        mailbox, timeout_seconds=1.0, poll_interval=0.01,
        default_model="m", capture=cap,
    )
    mailbox.put_pending = MagicMock()
    mailbox.wait_for_completion = MagicMock(
        return_value={"success": False, "error": "operator failed"}
    )
    mailbox.cleanup = MagicMock()

    with pytest.raises(RuntimeError):
        backend.complete_sync("s", "u", max_tokens=32)
    # Capture still fired exactly once.
    assert len(cap.events) == 1
    assert cap.events[0]["decision_type"] == "llm_chat_call"


# ---------------------------------------------------------------------------
# LocalBackend / OpenAIBackend — constructor accepts capture; never emit
# (their .complete_sync raises NotImplementedError so capture is moot)
# ---------------------------------------------------------------------------


def test_local_backend_accepts_capture_kwarg():
    cap = _FakeCapture()
    backend = LocalBackend(capture=cap)
    # Calling complete_sync raises by design — no capture emit expected.
    with pytest.raises(NotImplementedError):
        backend.complete_sync("s", "u")
    assert cap.events == []


def test_openai_backend_accepts_capture_kwarg():
    cap = _FakeCapture()
    backend = OpenAIBackend(api_key="ignored", capture=cap)
    with pytest.raises(NotImplementedError):
        backend.complete_sync("s", "u")
    assert cap.events == []


# ---------------------------------------------------------------------------
# build_backend factory threading
# ---------------------------------------------------------------------------


def test_build_backend_threads_capture_through_anthropic_path():
    cap = _FakeCapture()
    spec = BackendSpec(mode="api", provider="anthropic", api_key="k")
    backend = build_backend(spec, capture=cap)
    assert isinstance(backend, AnthropicBackend)
    assert backend._capture is cap


def test_build_backend_threads_capture_through_mock_path():
    cap = _FakeCapture()
    spec = BackendSpec(mode="api", provider="mock")
    backend = build_backend(spec, capture=cap)
    assert isinstance(backend, MockBackend)
    assert backend._capture is cap


def test_build_backend_capture_default_none_preserves_legacy_path():
    """Omitting ``capture`` must keep every backend's ``_capture`` as None."""
    spec = BackendSpec(mode="api", provider="mock")
    backend = build_backend(spec)
    assert backend._capture is None


# ---------------------------------------------------------------------------
# LLM-agnostic provider_label contract
# ---------------------------------------------------------------------------


def test_provider_labels_are_distinct_per_backend():
    """No two backends share a provider_label — audit trail must disambiguate.

    This is the regression net for the "LLM-agnostic" intent: a future
    ``TogetherBackend`` plugged into the same mixin must declare its own
    label, not shadow an existing one.
    """
    labels = {
        AnthropicBackend.provider_label,
        LocalBackend.provider_label,
        MailboxBrokeredBackend.provider_label,
        OpenAIBackend.provider_label,
        MockBackend.provider_label,
    }
    assert len(labels) == 5, f"provider labels collided: {labels}"
    # No provider-name strings leak into the mixin's generic field
    # contract — only the value carries the brand.
    assert "anthropic" in labels
    assert "openai" in labels
    assert "local" in labels
    assert "mailbox" in labels
    assert "mock" in labels
