"""Unit tests for ``_BaseLLMProvider`` (Phase 3 Subtask 11).

The base class is abstract — subclasses (``ContentGeneratorProvider``,
the upcoming ``OutlineProvider`` / ``RewriteProvider``) compose it via
``super().__init__(...)`` and override the page-authoring surface.
This suite exercises the dispatch / decision-capture plumbing the base
owns, independent of any specific tier's task semantics.

Coverage:

- ``_dispatch_call`` routes to ``_call_anthropic`` for the Anthropic
  backend and to the embedded ``OpenAICompatibleClient`` for ``local``
  / ``together``.
- ``_emit_decision`` forwards ``decision_type`` / ``decision`` /
  ``rationale`` straight to the injected capture and swallows
  exceptions raised by the capture so a flaky audit trail never
  bricks an LLM call.
- ``_last_capture_id`` returns ``in-memory:{id(self)}`` when no
  capture is wired and ``{file_basename}:{event_index}`` when a
  streaming capture is present (Wave 112 audit-trail format).
- Unknown provider raises ``ValueError`` with the subclass's name in
  the message.

Mirrors the ``httpx.MockTransport`` fixture pattern from
``Courseforge/tests/test_content_generator_provider.py`` so the two
LLM call-site test surfaces stay parallel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._base import _BaseLLMProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-model") -> dict:
    return {
        "id": "cmpl-base-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _capturing_client(sink: List[Dict[str, Any]]) -> httpx.Client:
    """A mock client that records each POST's JSON body into ``sink``."""

    def handler(request: httpx.Request) -> httpx.Response:
        sink.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_success_body("ok"))

    return _make_client(handler)


class _FakeCapture:
    """Lightweight stand-in for :class:`DecisionCapture`.

    Exposes the ``events`` attribute the base's ``_last_capture_id``
    walks, plus a ``log_decision(**kwargs)`` shim that records every
    emit.
    """

    def __init__(self, *, raises: bool = False) -> None:
        self.events: List[Dict[str, Any]] = []
        self._raises = raises

    def log_decision(self, **kwargs: Any) -> None:
        if self._raises:
            raise RuntimeError("capture flaky")
        self.events.append(kwargs)


class _StreamingFakeCapture(_FakeCapture):
    """Adds a ``_stream_path`` so ``_last_capture_id`` returns the
    Wave 112 ``{file_basename}:{event_index}`` form."""

    def __init__(self, stream_path: Path) -> None:
        super().__init__()
        self._stream_path = stream_path


class _MinimalProvider(_BaseLLMProvider):
    """Concrete subclass used to exercise the base's plumbing.

    Implements the abstract methods with the simplest possible
    behavior — the goal is to test the base's dispatch / capture
    plumbing, not the subclass's task semantics.
    """

    def _render_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        return "test prompt"

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        self._emit_decision(
            decision_type="content_generator_call",
            decision=f"output chars={len(raw_text or '')}",
            rationale=(
                f"Test rationale routing through provider={self._provider}, "
                f"model={self._model}, retry_count={retry_count}."
            ),
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_with_subclass_name(monkeypatch):
    """``ValueError`` message names the concrete subclass, not the
    abstract base — debugging correctness."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    with pytest.raises(ValueError) as excinfo:
        _MinimalProvider(provider="bogus")
    assert "_MinimalProvider" in str(excinfo.value)
    assert "bogus" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------


def test_dispatch_call_routes_to_anthropic_for_anthropic_provider(
    monkeypatch,
):
    """``_dispatch_call`` routes through the Anthropic SDK path
    (``_call_anthropic``) when ``provider='anthropic'``; the injected
    fake client's ``messages.create`` is what gets invoked."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    create_calls: List[Dict[str, Any]] = []

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            create_calls.append(kwargs)
            return {"content": [{"type": "text", "text": "hello"}]}

    class _FakeClient:
        messages = _FakeMessages()

    p = _MinimalProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
        system_prompt="SYS",
    )
    text, retries = p._dispatch_call("user prompt")
    assert text == "hello"
    assert retries == 0
    assert len(create_calls) == 1
    # The base passes the system prompt through verbatim.
    assert create_calls[0]["system"] == "SYS"


def test_dispatch_call_routes_to_oa_client_for_local_provider(monkeypatch):
    """``_dispatch_call`` routes through ``OpenAICompatibleClient`` for
    the ``local`` backend; the embedded handler observes a single POST
    to the resolved base URL and the response body is unwrapped."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("oa-reply"))

    p = _MinimalProvider(
        provider="local",
        client=_make_client(handler),
        system_prompt="SYS",
    )
    text, retries = p._dispatch_call("user prompt")
    assert text == "oa-reply"
    assert retries == 0
    assert len(seen) == 1
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"


# ---------------------------------------------------------------------------
# W9.2 — unified endpoint-registry migration
# ---------------------------------------------------------------------------


def test_local_seat_resolves_byte_identical_client_config(monkeypatch):
    """The local/default seat, now built via the unified registry
    (``build_openai_compatible_client``), resolves to the SAME client
    config (base_url / model / api_key / provider_label) as the legacy
    per-vendor branch — and threads NO extra_body into the payload
    (byte-identical dispatch for the already-reachable seat)."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)

    bodies: List[Dict[str, Any]] = []
    p = _MinimalProvider(
        provider="local",
        client=_capturing_client(bodies),
        system_prompt="SYS",
    )

    # Registry-resolved identity matches the pre-migration local defaults.
    assert p._oa_client.base_url == "http://localhost:11434/v1"
    assert p._oa_client.model == "qwen2.5:7b-instruct-q4_K_M"
    assert p._model == "qwen2.5:7b-instruct-q4_K_M"
    assert p._base_url == "http://localhost:11434/v1"
    assert p._api_key == "local"
    assert p._oa_client.provider_label == "local"
    # The ``local`` row declares no extra_body → nothing threaded.
    assert p._extra_body is None

    p._dispatch_call("hello")
    assert bodies, "handler observed a POST"
    body = bodies[0]
    assert body["model"] == "qwen2.5:7b-instruct-q4_K_M"
    for key in ("model", "messages", "temperature", "max_tokens"):
        assert key in body
    # No vendor-specific request-body extras leaked onto the local seat.
    assert "chat_template_kwargs" not in body


def test_previously_unreachable_seat_resolves_and_threads_extra_body(
    monkeypatch,
):
    """A registry endpoint the legacy 4-branch tuple never reached
    (``nvidia-deepseek`` — the NVIDIA-hosted large reasoning seat) now
    resolves through the base, AND its row-declared ``extra_body``
    (``chat_template_kwargs:{thinking:false}`` — reasoning suppression)
    is threaded into the dispatched request body."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)

    bodies: List[Dict[str, Any]] = []
    p = _MinimalProvider(
        provider="nvidia-deepseek",
        client=_capturing_client(bodies),
        system_prompt="SYS",
    )

    # The seat resolves entirely from the endpoint row.
    assert p._provider == "nvidia-deepseek"
    assert p._oa_client.base_url == "https://integrate.api.nvidia.com/v1"
    assert p._oa_client.model == "deepseek-ai/deepseek-v4-pro"
    # The row's extra_body is captured and threaded on dispatch.
    assert p._extra_body == {"chat_template_kwargs": {"thinking": False}}

    p._dispatch_call("hello")
    assert bodies, "handler observed a POST"
    assert bodies[0]["chat_template_kwargs"] == {"thinking": False}


def test_caller_extra_payload_wins_over_registry_extra_body(monkeypatch):
    """A caller-supplied ``extra_payload`` overrides the registry
    ``extra_body`` for the same key (per-call override precedence)."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)

    bodies: List[Dict[str, Any]] = []
    p = _MinimalProvider(
        provider="nvidia-deepseek",
        client=_capturing_client(bodies),
        system_prompt="SYS",
    )
    p._dispatch_call(
        "hello",
        extra_payload={"chat_template_kwargs": {"thinking": True}},
    )
    assert bodies[-1]["chat_template_kwargs"] == {"thinking": True}


def test_default_supported_providers_reaches_all_registry_endpoints():
    """The base's default provider allow-list now spans EVERY
    ``openai_compatible`` registry endpoint (reach-all), plus the
    ``anthropic`` SDK seat, while excluding the ``claude_session``
    dispatcher (not an HTTP endpoint)."""
    from Courseforge.generators._base import _DEFAULT_SUPPORTED_PROVIDERS

    for name in (
        "anthropic",
        "local",
        "together",
        "nvidia",
        "nvidia-deepseek",
        "together-vision",
        "groq",
        "fireworks",
        "deepseek",
    ):
        assert name in _DEFAULT_SUPPORTED_PROVIDERS
    assert "claude_session" not in _DEFAULT_SUPPORTED_PROVIDERS


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------


def test_emit_decision_includes_required_fields(monkeypatch):
    """``_emit_decision`` forwards ``decision_type`` / ``decision`` /
    ``rationale`` verbatim to ``capture.log_decision`` so the audit
    trail records exactly what the subclass interpolated."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture()
    p = _MinimalProvider(
        provider="local",
        capture=cap,
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("x"))
        ),
        system_prompt="SYS",
    )
    p._emit_decision(
        decision_type="content_generator_call",
        decision="dummy decision",
        rationale="Unit-test rationale exceeds twenty characters.",
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "content_generator_call"
    assert event["decision"] == "dummy decision"
    assert event["rationale"].startswith("Unit-test rationale")


def test_emit_decision_swallows_capture_exceptions(monkeypatch):
    """A flaky capture must NEVER brick an LLM call — the base's
    ``_emit_decision`` catches exceptions and logs them. This
    matches the pre-Phase-3 behaviour of ``ContentGeneratorProvider``
    that the Wave 112 audit-trail contract relies on."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture(raises=True)
    p = _MinimalProvider(
        provider="local",
        capture=cap,
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("x"))
        ),
        system_prompt="SYS",
    )
    # Must NOT raise even though the capture's log_decision throws.
    p._emit_decision(
        decision_type="content_generator_call",
        decision="d",
        rationale="r" * 25,
    )


# ---------------------------------------------------------------------------
# _last_capture_id semantics
# ---------------------------------------------------------------------------


def test_last_capture_id_falls_back_to_in_memory_when_capture_none(
    monkeypatch,
):
    """When no capture is wired, ``_last_capture_id`` returns
    ``in-memory:{id(self)}`` so the Wave 112 invariant
    (``decision_capture_id`` ≥ 1 char) holds without forcing tests to
    wire up a full capture surface."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    p = _MinimalProvider(
        provider="local",
        capture=None,
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("x"))
        ),
        system_prompt="SYS",
    )
    cap_id = p._last_capture_id()
    assert cap_id.startswith("in-memory:")
    assert len(cap_id) > len("in-memory:")


def test_last_capture_id_format_when_streaming_capture_present(
    monkeypatch, tmp_path,
):
    """When a streaming capture's ``_stream_path`` is set, the base
    formats the ID as ``{file_basename}:{event_index}`` so a
    ``Touch.decision_capture_id`` can resolve to the exact JSONL line
    that explained the LLM call (Wave 112 audit-trail format)."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    stream_path = tmp_path / "decisions_test_20260502.jsonl"
    cap = _StreamingFakeCapture(stream_path)
    p = _MinimalProvider(
        provider="local",
        capture=cap,
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("x"))
        ),
        system_prompt="SYS",
    )

    # No events emitted yet — index falls back to 0.
    cap_id = p._last_capture_id()
    assert cap_id == "decisions_test_20260502.jsonl:0"

    # Emit two events; index becomes len(events)-1 = 1.
    p._emit_decision(
        decision_type="content_generator_call",
        decision="d1",
        rationale="r" * 25,
    )
    p._emit_decision(
        decision_type="content_generator_call",
        decision="d2",
        rationale="r" * 25,
    )
    cap_id = p._last_capture_id()
    assert cap_id == "decisions_test_20260502.jsonl:1"
