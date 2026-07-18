"""Tests for the LLM-agnostic OpenAICompatibleClient.

Exercises the wire-format machinery shared across every
OpenAI-compatible backend (Together, local Ollama / vLLM, Fireworks,
Groq, hosted Mistral, ...). The client must stay backend-agnostic:
the same class, configured with a different base_url + model + label,
must work for every such provider without code changes.

Coverage:

- Happy path: assistant content from a 200 response.
- Decision-capture fires once per call with provider_label, base_url,
  model, and caller-injected ``decision_metadata`` interpolated into
  the rationale.
- 429 retry-then-success.
- 500 -> 503 -> 200 across three attempts.
- Transport-error retries exhausted raise with
  ``code="max_retries_exceeded"``.
- 4xx outside the retry list (e.g. 400) fails immediately with
  ``code=str(status)``.
- Malformed response (missing ``choices``) raises
  ``code="malformed_response"``.
- ``provider_label`` flows into rationale (regression test for the
  LLM-agnostic intent — same client serves any backend, audit trail
  reflects which one).
- ``decision_metadata`` keys flow into rationale (lets task providers
  inject task-specific signals like ``chunk_id`` / ``classification_target``
  without the client growing surface).
- Authorization header omitted when ``api_key`` is ``None`` (local
  servers that ignore auth).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._openai_compatible_client import (  # noqa: E402
    OpenAICompatibleClient,
    SynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-model") -> dict:
    """Build an OpenAI-shaped 200 response payload."""
    return {
        "id": "cmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 35,
            "total_tokens": 155,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _client_yielding(*responses: httpx.Response) -> httpx.Client:
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    return _make_client(handler)


def _build(
    *,
    api_key: str | None = "test-key",
    capture: Any | None = None,
    transport_handler: Callable[[httpx.Request], httpx.Response] | None = None,
    transport: httpx.BaseTransport | None = None,
    base_url: str = "https://api.example.com/v1",
    model: str = "test-model",
    provider_label: str = "test_provider",
    max_retries: int = 3,
) -> OpenAICompatibleClient:
    if transport is None:
        if transport_handler is None:
            raise ValueError("either transport or transport_handler required")
        transport = httpx.MockTransport(transport_handler)
    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        capture=capture,
        max_retries=max_retries,
        provider_label=provider_label,
        client=httpx.Client(transport=transport),
        # Patch sleep_fn so retries don't actually wait in tests.
        sleep_fn=lambda _s: None,
    )


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_empty_base_url_raises_value_error():
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="", model="m")


def test_empty_model_raises_value_error():
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="https://x/v1", model="")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_assistant_content():
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("hello world"))

    client = _build(transport_handler=handler)
    out = client.chat_completion(
        [{"role": "user", "content": "hi"}], max_tokens=64, temperature=0.0
    )
    assert out == "hello world"
    # Endpoint built off base_url with /chat/completions appended.
    assert str(seen[0].url) == "https://api.example.com/v1/chat/completions"
    # Authorization carried.
    assert seen[0].headers["Authorization"] == "Bearer test-key"


def test_no_api_key_omits_authorization_header():
    """Local servers that ignore auth get no Authorization header at all."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler, api_key=None)
    client.chat_completion(
        [{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.0,
    )
    assert "Authorization" not in seen[0].headers


# ---------------------------------------------------------------------------
# Decision capture wiring
# ---------------------------------------------------------------------------


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def test_decision_capture_fires_once_with_dynamic_signals():
    cap = _FakeCapture()
    client = _build(
        transport_handler=lambda r: httpx.Response(
            200, json=_success_body("stuff", model="provider-model-id")
        ),
        capture=cap,
        model="provider-model-id",
        provider_label="custom_label",
    )
    client.chat_completion(
        [{"role": "user", "content": "Q"}],
        max_tokens=99,
        temperature=0.4,
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "llm_chat_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    # Dynamic signals interpolated.
    assert "provider-model-id" in rationale
    assert "custom_label" in rationale
    assert "prompt_tokens=120" in rationale
    assert "completion_tokens=35" in rationale
    assert "http_retries=0" in rationale
    assert "max_tokens=99" in rationale


def test_decision_metadata_keys_flow_into_rationale():
    """Caller-injected task signals flow into the rationale verbatim.

    This is the seam that lets task providers (Together, Local,
    Curriculum, future Fireworks, ...) inject ``chunk_id`` /
    ``template_id`` / ``classification_target`` without the client
    needing to know what they are.
    """
    cap = _FakeCapture()
    client = _build(
        transport_handler=lambda r: httpx.Response(
            200, json=_success_body("stuff")
        ),
        capture=cap,
    )
    client.chat_completion(
        [{"role": "user", "content": "Q"}],
        decision_metadata={
            "chunk_id": "chunk_42",
            "template_id": "remember.explanation",
            "classification_target": "teaching_role",
        },
    )
    rationale = cap.events[0]["rationale"]
    assert "chunk_id=chunk_42" in rationale
    assert "template_id=remember.explanation" in rationale
    assert "classification_target=teaching_role" in rationale
    decision = cap.events[0]["decision"]
    # Task signals appear in the decision string too.
    assert "chunk_id=chunk_42" in decision


def test_provider_label_is_authentically_swappable():
    """Regression: same client class, two provider labels → two distinct rationales.

    The client must not hard-code Together / Local / any-specific
    branding. A future Fireworks / Groq / hosted-Mistral provider
    using this same client class must surface its own label in audit
    output.
    """
    cap1 = _FakeCapture()
    cap2 = _FakeCapture()
    handler = lambda r: httpx.Response(200, json=_success_body("ok"))
    c1 = _build(
        transport_handler=handler,
        capture=cap1,
        provider_label="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )
    c2 = _build(
        transport_handler=handler,
        capture=cap2,
        provider_label="groq",
        base_url="https://api.groq.com/openai/v1",
    )
    c1.chat_completion([{"role": "user", "content": "x"}])
    c2.chat_completion([{"role": "user", "content": "x"}])
    assert "fireworks" in cap1.events[0]["rationale"]
    assert "https://api.fireworks.ai/inference/v1" in cap1.events[0]["rationale"]
    assert "groq" in cap2.events[0]["rationale"]
    assert "https://api.groq.com/openai/v1" in cap2.events[0]["rationale"]
    # Cross-pollination check.
    assert "groq" not in cap1.events[0]["rationale"]
    assert "fireworks" not in cap2.events[0]["rationale"]


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_429_retry_then_success():
    transport = httpx.MockTransport(
        _client_yielding(
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=_success_body("after retry")),
        ).__getattribute__("_transport")
        if False
        else None
    )
    # Simpler: build a queue handler.
    state = {"i": 0}
    responses = [
        httpx.Response(429, json={"error": "rate limited"}),
        httpx.Response(200, json=_success_body("after retry")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client = _build(transport_handler=handler)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "after retry"
    assert state["i"] == 2


def test_500_then_503_then_200_succeeds_across_three_attempts():
    state = {"i": 0}
    responses = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(503, json={"error": "still booming"}),
        httpx.Response(200, json=_success_body("third time lucky")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client = _build(transport_handler=handler, max_retries=3)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "third time lucky"


def test_retries_exhausted_on_transient_status_raises_with_status_code():
    """When all retries fail on the same transient status, surface that status.

    Matches the legacy Together-provider behavior so the existing
    fixture suite (which composing tests inherit) keeps passing
    unmodified.
    """
    handler = lambda r: httpx.Response(500, json={"error": "boom"})
    client = _build(transport_handler=handler, max_retries=3)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "500"


def test_transport_errors_exhausted_raise_max_retries_exceeded():
    """Transport-level failures (no HTTP status) → max_retries_exceeded.

    Distinct from the retries-exhausted-on-transient-status case: when
    the connection itself never succeeds, there's no status code to
    surface, so the typed code is the dedicated sentinel.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _build(transport_handler=handler, max_retries=3)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "max_retries_exceeded"


def test_4xx_outside_retry_list_fails_immediately():
    """Non-retryable 4xx status surfaces with code=str(status), no retry."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": "bad"})

    client = _build(transport_handler=handler, max_retries=3)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "400"
    # Exactly one POST — no retry.
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Malformed response
# ---------------------------------------------------------------------------


def test_response_missing_choices_raises_malformed_response():
    handler = lambda r: httpx.Response(
        200, json={"id": "cmpl", "usage": {}}  # no "choices"
    )
    client = _build(transport_handler=handler)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "malformed_response"


def test_non_json_200_body_raises_malformed_response():
    handler = lambda r: httpx.Response(200, content=b"not json")
    client = _build(transport_handler=handler)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "malformed_response"


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def test_extra_payload_merges_into_request_body():
    """Caller-supplied extra knobs (top_p, stop, ...) flow through."""
    import json as _json

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler)
    client.chat_completion(
        [{"role": "user", "content": "x"}],
        max_tokens=42,
        temperature=0.1,
        extra_payload={"top_p": 0.9, "stop": ["\n\n"]},
    )
    body = _json.loads(seen[0].content.decode("utf-8"))
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.1
    assert body["top_p"] == 0.9
    assert body["stop"] == ["\n\n"]
    # Caller cannot clobber model / messages via extra_payload.
    assert body["model"] == "test-model"


def test_empty_messages_raises_value_error():
    client = _build(transport_handler=lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.chat_completion([])


# ---------------------------------------------------------------------------
# Wave 113: json_mode + lenient JSON extraction
# ---------------------------------------------------------------------------


def test_json_mode_includes_format_field():
    """When ``json_mode=True``, the request payload carries BOTH the
    Ollama-style ``format: "json"`` field AND the OpenAI-spec
    ``response_format: {"type": "json_object"}`` field. Servers that
    don't recognize one or the other ignore it silently — defense in
    depth across hosted-OSS providers + local servers."""
    import json as _json

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b-instruct-q4_K_M",
        api_key=None,
        client=httpx.Client(transport=transport),
        sleep_fn=lambda _s: None,
        json_mode=True,
    )
    client.chat_completion([{"role": "user", "content": "hi"}])
    body = _json.loads(seen[0].content.decode("utf-8"))
    assert body.get("format") == "json"
    assert body.get("response_format") == {"type": "json_object"}


def test_json_mode_off_does_not_inject_format_fields():
    """Default ``json_mode=False`` leaves the payload untouched."""
    import json as _json

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler)
    client.chat_completion([{"role": "user", "content": "hi"}])
    body = _json.loads(seen[0].content.decode("utf-8"))
    assert "format" not in body
    assert "response_format" not in body


def test_extract_json_lenient_plain():
    """Strategy 1: direct ``json.loads`` on a clean JSON object."""
    out = OpenAICompatibleClient._extract_json_lenient('{"a": 1}')
    assert out == {"a": 1}


def test_extract_json_lenient_markdown_fence():
    """Strategy 2: strip enclosing markdown code fences (``json...``)."""
    out = OpenAICompatibleClient._extract_json_lenient(
        '```json\n{"a": 1}\n```'
    )
    assert out == {"a": 1}
    # Also without the language hint.
    out2 = OpenAICompatibleClient._extract_json_lenient(
        '```\n{"b": 2}\n```'
    )
    assert out2 == {"b": 2}


def test_extract_json_lenient_with_prose():
    """Strategy 3: scan the first balanced JSON object out of surrounding
    prose. Handles the natural-language drift Wave 113 Task 10 saw."""
    out = OpenAICompatibleClient._extract_json_lenient(
        'Sure! {"a": 1} hope this helps'
    )
    assert out == {"a": 1}


def test_extract_json_lenient_unrecoverable():
    """Genuinely unrecoverable response: caller decides retry strategy
    on ``None``. No exception thrown — caller is responsible for the
    decision."""
    out = OpenAICompatibleClient._extract_json_lenient(
        "I cannot help with that."
    )
    assert out is None


def test_extract_json_lenient_empty_returns_none():
    assert OpenAICompatibleClient._extract_json_lenient("") is None
    assert OpenAICompatibleClient._extract_json_lenient("   \n") is None


def test_extract_json_lenient_non_object_returns_none():
    """Non-object JSON (top-level array, string, number) returns None —
    the synthesis contract requires an object."""
    assert OpenAICompatibleClient._extract_json_lenient("[1, 2, 3]") is None
    assert OpenAICompatibleClient._extract_json_lenient("42") is None


# ---------------------------------------------------------------------------
# Stage-0 429 fix: Retry-After header honoring
# ---------------------------------------------------------------------------


def _build_capturing_sleep(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 3,
    initial_backoff_seconds: float = 1.0,
) -> tuple[OpenAICompatibleClient, List[float]]:
    """Build a client whose ``sleep_fn`` records every requested wait.

    Lets a test assert the exact duration the retry path WOULD have
    slept without actually waiting (no real time burned, no network).
    """
    slept: List[float] = []
    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="test-key",
        max_retries=max_retries,
        initial_backoff_seconds=initial_backoff_seconds,
        provider_label="test_provider",
        client=httpx.Client(transport=transport),
        sleep_fn=lambda s: slept.append(s),
    )
    return client, slept


def test_retry_after_integer_seconds_is_honored():
    """A 429 carrying ``Retry-After: 5`` → the client sleeps ~5s.

    The server-supplied wait is honored in preference to the computed
    exponential backoff (which on attempt 1 would be ~1s).
    """
    state = {"i": 0}
    responses = [
        httpx.Response(
            429, json={"error": "rate limited"}, headers={"Retry-After": "5"}
        ),
        httpx.Response(200, json=_success_body("after retry")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client, slept = _build_capturing_sleep(handler)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "after retry"
    assert slept == [5.0]


def test_retry_after_http_date_is_parsed_to_seconds():
    """``Retry-After`` as an HTTP-date → parsed to ~seconds-from-now."""
    from email.utils import format_datetime
    from datetime import datetime, timedelta, timezone

    when = datetime.now(timezone.utc) + timedelta(seconds=12)
    http_date = format_datetime(when)

    state = {"i": 0}
    responses = [
        httpx.Response(
            503,
            json={"error": "unavailable"},
            headers={"Retry-After": http_date},
        ),
        httpx.Response(200, json=_success_body("recovered")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client, slept = _build_capturing_sleep(handler)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "recovered"
    assert len(slept) == 1
    # ~12s from now; allow a generous window for clock skew + test runtime.
    assert 8.0 <= slept[0] <= 12.5


def test_retry_after_above_ceiling_is_clamped():
    """A pathological ``Retry-After`` is clamped to the max ceiling."""
    from Trainforge.generators._openai_compatible_client import (
        _RETRY_AFTER_MAX_SECONDS,
    )

    state = {"i": 0}
    responses = [
        httpx.Response(
            429,
            json={"error": "rate limited"},
            headers={"Retry-After": "86400"},  # one day — pathological
        ),
        httpx.Response(200, json=_success_body("ok")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client, slept = _build_capturing_sleep(handler)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "ok"
    assert slept == [_RETRY_AFTER_MAX_SECONDS]


def test_x_ratelimit_reset_used_when_retry_after_absent():
    """``X-RateLimit-Reset`` (bare delta seconds) is the secondary source."""
    state = {"i": 0}
    responses = [
        httpx.Response(
            429,
            json={"error": "rate limited"},
            headers={"X-RateLimit-Reset": "7"},
        ),
        httpx.Response(200, json=_success_body("ok")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client, slept = _build_capturing_sleep(handler)
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "ok"
    assert slept == [7.0]


def test_no_retry_after_header_uses_exponential_backoff():
    """No header → exponential-backoff fallback (byte-stable contract).

    With full jitter the sleep lies in ``[0, initial * 2**(attempt-1)]``.
    The legacy fixed schedule is the upper bound — the client never
    sleeps LONGER than the historical exponential value, and a server
    that sends no ``Retry-After`` keeps the legacy fallback path.
    """
    state = {"i": 0}
    responses = [
        httpx.Response(500, json={"error": "boom"}),  # no Retry-After
        httpx.Response(503, json={"error": "still boom"}),  # no Retry-After
        httpx.Response(200, json=_success_body("third time lucky")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    client, slept = _build_capturing_sleep(
        handler, max_retries=3, initial_backoff_seconds=1.0
    )
    out = client.chat_completion([{"role": "user", "content": "x"}])
    assert out == "third time lucky"
    assert len(slept) == 2
    # Attempt 1 base = 1.0 → [0, 1.0]; attempt 2 base = 2.0 → [0, 2.0].
    assert 0.0 <= slept[0] <= 1.0
    assert 0.0 <= slept[1] <= 2.0


# ---------------------------------------------------------------------------
# Nemotron "detailed thinking off" toggle (ED4ALL_REASONING_THINKING_OFF)
# ---------------------------------------------------------------------------


def _capture_request_body(
    monkeypatch: Any,
    *,
    env: str | None,
    messages: List[Dict[str, Any]],
    extra_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run one chat_completion under a given env and return the request JSON.

    Toggles ``ED4ALL_REASONING_THINKING_OFF`` via monkeypatch (with cleanup),
    captures the outgoing request body in the MockTransport handler, and
    returns the parsed JSON so a test can assert on messages + payload keys.
    """
    import json as _json

    if env is None:
        monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", env)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler)
    client.chat_completion(
        messages,
        max_tokens=64,
        temperature=0.0,
        extra_payload=extra_payload,
    )
    return _json.loads(seen[0].content.decode("utf-8"))


def test_thinking_off_unset_is_byte_identical(monkeypatch):
    """Flag unset → NO ``chat_template_kwargs`` key AND messages unchanged.

    This is the hard backward-compat property: an existing Qwen / dev-box
    deployment must see a byte-identical payload.
    """
    input_messages = [{"role": "user", "content": "hi"}]
    body = _capture_request_body(
        monkeypatch, env=None, messages=input_messages
    )
    assert "chat_template_kwargs" not in body
    # Messages equal the input verbatim — no injected system directive.
    assert body["messages"] == input_messages


def test_thinking_off_on_no_existing_system_inserts_directive(monkeypatch):
    """Flag on + no pre-existing system message → a system directive is
    inserted at index 0 AND ``chat_template_kwargs`` is set."""
    body = _capture_request_body(
        monkeypatch,
        env="1",
        messages=[{"role": "user", "content": "hi"}],
    )
    first = body["messages"][0]
    assert first["role"] == "system"
    assert "detailed thinking off" in first["content"]
    # Original user message preserved after the inserted system message.
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_off_on_prepends_to_existing_system(monkeypatch):
    """Flag on + pre-existing system message → the directive is PREPENDED
    (not replaced); still exactly one system message."""
    body = _capture_request_body(
        monkeypatch,
        env="true",
        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "hi"},
        ],
    )
    system_msgs = [m for m in body["messages"] if m.get("role") == "system"]
    assert len(system_msgs) == 1
    content = system_msgs[0]["content"]
    assert content.startswith("detailed thinking off")
    # Original system text retained.
    assert "You are a helpful tutor." in content
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_off_caller_chat_template_kwargs_wins(monkeypatch):
    """Flag on + caller-supplied ``chat_template_kwargs`` → caller wins.

    ``setdefault`` must not clobber an explicit ``extra_payload`` override.
    """
    body = _capture_request_body(
        monkeypatch,
        env="on",
        messages=[{"role": "user", "content": "hi"}],
        extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_thinking_off_does_not_mutate_callers_messages(monkeypatch):
    """The caller's messages list is never mutated in place (non-mutation
    contract, same as ``_attach_vision_blocks``)."""
    monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", "yes")
    input_messages = [{"role": "user", "content": "hi"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler)
    client.chat_completion(input_messages)
    # Caller's list still holds exactly the original single user message.
    assert input_messages == [{"role": "user", "content": "hi"}]


def test_resolve_reasoning_thinking_off_parse_with_fallback(monkeypatch):
    """Truthy parse: ``1/true/yes/on`` (case-insensitive) → True; else False."""
    from Trainforge.generators._openai_compatible_client import (
        resolve_reasoning_thinking_off,
    )

    for truthy in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", truthy)
        assert resolve_reasoning_thinking_off() is True
    for falsey in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv("ED4ALL_REASONING_THINKING_OFF", falsey)
        assert resolve_reasoning_thinking_off() is False
    monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)
    assert resolve_reasoning_thinking_off() is False


def test_compute_backoff_upper_bound_matches_legacy_schedule():
    """``_compute_backoff`` never exceeds the legacy fixed exponential.

    Pins the byte-stability guarantee at the unit level: across many
    samples the jittered value stays within ``[0, initial*2**(n-1)]`` —
    the exact ceiling of the pre-jitter schedule.
    """
    client, _ = _build_capturing_sleep(
        lambda r: httpx.Response(200, json=_success_body("ok")),
        initial_backoff_seconds=1.0,
    )
    for attempt in range(1, 5):
        ceiling = 1.0 * (2 ** (attempt - 1))
        for _ in range(200):
            val = client._compute_backoff(attempt)
            assert 0.0 <= val <= ceiling


# ---------------------------------------------------------------------------
# Task #10 — TTFT (time-to-first-token) metering (ED4ALL_LLM_TTFT_METER)
# ---------------------------------------------------------------------------


import json as _json_ttft  # noqa: E402


class _FakeClock:
    """Deterministic monotonic clock: pops successive values, then latches
    on the last so an unexpected extra call never raises."""

    def __init__(self, values: List[float]) -> None:
        self._values = list(values)
        self._i = 0

    def __call__(self) -> float:
        v = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return v


def _sse_bytes(
    *chunks: Dict[str, Any],
    usage: Dict[str, Any] | None = None,
    done: bool = True,
) -> bytes:
    """Serialise OpenAI-style SSE ``data:`` lines from chunk dicts."""
    lines: List[str] = []
    for c in chunks:
        lines.append("data: " + _json_ttft.dumps(c))
        lines.append("")
    if usage is not None:
        lines.append(
            "data: " + _json_ttft.dumps({"choices": [], "usage": usage})
        )
        lines.append("")
    if done:
        lines.append("data: [DONE]")
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _delta(content: str | None = None, finish: str | None = None) -> Dict[str, Any]:
    return {
        "choices": [
            {"index": 0, "delta": ({"content": content} if content is not None else {}),
             "finish_reason": finish}
        ]
    }


def _build_streaming(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    monotonic_fn: Callable[[], float] | None = None,
    api_key: str | None = "test-key",
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key=api_key,
        provider_label="test_provider",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=lambda _s: None,
        monotonic_fn=monotonic_fn,
    )


def test_ttft_flag_off_no_stream_field_in_body(monkeypatch):
    """Regression: flag OFF → request body carries NO ``stream`` key and the
    legacy non-streaming JSON path is used verbatim."""
    monkeypatch.delenv("ED4ALL_LLM_TTFT_METER", raising=False)
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("legacy"))

    client = _build(transport_handler=handler)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    assert out == "legacy"
    body = _json_ttft.loads(seen[0].content.decode("utf-8"))
    assert "stream" not in body
    assert "stream_options" not in body
    assert client.last_ttft_ms is None


def test_ttft_flag_on_sets_stream_and_stream_options(monkeypatch):
    """Flag ON → request body carries ``stream: true`` + ``stream_options``."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "1")
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=_sse_bytes(
                _delta("hi", None),
                _delta(None, "stop"),
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = _build_streaming(handler)
    client.chat_completion([{"role": "user", "content": "hi"}])
    body = _json_ttft.loads(seen[0].content.decode("utf-8"))
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_ttft_sse_reconstruction_matches_non_streaming(monkeypatch):
    """SSE accumulation reconstructs content + finish_reason + usage so the
    call is behaviorally identical to the non-streaming fixture."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "on")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_bytes(
                _delta("hello", None),
                _delta(" world", None),
                _delta(None, "stop"),
                usage={
                    "prompt_tokens": 120,
                    "completion_tokens": 35,
                    "total_tokens": 155,
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = _build_streaming(handler)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    # Content reconstructed identical to the non-streaming _success_body path.
    assert out == "hello world"
    # Usage reconstructed identically → last_usage carries server counts.
    assert client.last_usage == {
        "prompt_tokens": 120,
        "completion_tokens": 35,
        "total_tokens": 155,
    }


def test_ttft_finish_reason_length_triggers_truncation_guard(monkeypatch):
    """finish_reason reconstruction is faithful: a streamed
    ``finish_reason='length'`` fires the same ``output_truncated`` guard as the
    non-streaming path (no behavioral divergence for downstream callers)."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "true")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_bytes(
                _delta("partial", None),
                _delta(None, "length"),
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = _build_streaming(handler)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "output_truncated"


def test_ttft_measured_with_monkeypatched_clock(monkeypatch, tmp_path):
    """TTFT is measured off the injected clock and written to the usage row.

    Clock call order: (1) _call_start, (2) stream req_start, (3) first delta
    → ttft, (4) _duration_ms. ttft = (call3 - call2) * 1000.
    """
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "yes")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-TTFT")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_bytes(
                _delta("a", None),
                _delta(None, "stop"),
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            ),
            headers={"content-type": "text/event-stream"},
        )

    clock = _FakeClock([1000.0, 1000.0, 1000.5, 1002.0])
    client = _build_streaming(handler, monotonic_fn=clock)
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert client.last_ttft_ms == 500.0

    usage_path = tmp_path / "runs" / "RUN-TTFT" / "llm_usage.jsonl"
    rows = [
        _json_ttft.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["ttft_ms"] == 500.0
    assert row["duration_ms"] == 2000.0
    assert row["prompt_tokens"] == 5
    assert row["completion_tokens"] == 3
    # Usage present → no stream_usage_present note.
    assert "stream_usage_present" not in row


def test_ttft_off_row_has_no_ttft_field(monkeypatch, tmp_path):
    """Flag OFF → the OP2 usage row is byte-identical to the legacy shape
    (no ttft_ms / stream_usage_present keys)."""
    monkeypatch.delenv("ED4ALL_LLM_TTFT_METER", raising=False)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-NOTTFT")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(transport_handler=handler)
    client.chat_completion([{"role": "user", "content": "hi"}])
    usage_path = tmp_path / "runs" / "RUN-NOTTFT" / "llm_usage.jsonl"
    row = _json_ttft.loads(usage_path.read_text(encoding="utf-8").splitlines()[0])
    assert "ttft_ms" not in row
    assert "stream_usage_present" not in row
    # perf-doc P4: finish_reason is stamped whenever the server reports it
    # (``_success_body`` returns ``"stop"``); it is additive and never carries
    # the TTFT keys.
    assert row["finish_reason"] == "stop"
    assert set(row.keys()) == {
        "ts", "provider", "model", "prompt_tokens", "completion_tokens",
        "duration_ms", "finish_reason",
    }


# ---------------------------------------------------------------------------
# perf-doc P4 — finish_reason in the OP2 usage row
# ---------------------------------------------------------------------------


def _length_body(content: str = "partial") -> dict:
    """OpenAI-shaped body whose first choice was cut off at max_tokens."""
    return {
        "id": "cmpl-len",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 90,
            "completion_tokens": 800,
            "total_tokens": 890,
        },
    }


def test_finish_reason_stamped_on_usage_row(monkeypatch, tmp_path):
    """A normal ``finish_reason='stop'`` is recorded on the OP2 usage row."""
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-FR-STOP")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("hi"))

    client = _build(transport_handler=handler)
    client.chat_completion([{"role": "user", "content": "hi"}])
    row = _json_ttft.loads(
        (tmp_path / "runs" / "RUN-FR-STOP" / "llm_usage.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["finish_reason"] == "stop"


def test_finish_reason_length_row_written_before_truncation_raise(
    monkeypatch, tmp_path
):
    """The whole point of perf-doc P4: a ``finish_reason='length'`` response
    RAISES ``output_truncated`` from ``_extract_text`` — but the metering row
    is written FIRST and names the truncation, so truncation diagnosis is a
    fact on the row rather than guesswork."""
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-FR-LEN")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_length_body())

    client = _build(transport_handler=handler)
    with pytest.raises(SynthesisProviderError) as excinfo:
        client.chat_completion([{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "output_truncated"
    # Despite the raise, the usage row exists and records the truncation.
    usage_path = tmp_path / "runs" / "RUN-FR-LEN" / "llm_usage.jsonl"
    assert usage_path.exists()
    row = _json_ttft.loads(usage_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["finish_reason"] == "length"
    assert row["prompt_tokens"] == 90
    assert row["completion_tokens"] == 800


def test_finish_reason_omitted_when_absent(monkeypatch, tmp_path):
    """A body missing ``finish_reason`` → the field is OMITTED so the row stays
    byte-identical to the legacy OP2 shape (additive-only contract)."""
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-FR-NONE")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = _build(transport_handler=handler)
    client.chat_completion([{"role": "user", "content": "hi"}])
    row = _json_ttft.loads(
        (tmp_path / "runs" / "RUN-FR-NONE" / "llm_usage.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert "finish_reason" not in row


def test_extract_finish_reason_helper():
    """Unit: the extractor is total — malformed shapes yield None, never raise."""
    _ex = OpenAICompatibleClient._extract_finish_reason
    assert _ex({"choices": [{"finish_reason": "stop"}]}) == "stop"
    assert _ex({"choices": [{"finish_reason": "length"}]}) == "length"
    assert _ex({"choices": [{"finish_reason": None}]}) is None
    assert _ex({"choices": [{"finish_reason": ""}]}) is None
    assert _ex({"choices": []}) is None
    assert _ex({}) is None
    assert _ex({"choices": ["not-a-dict"]}) is None


def test_ttft_missing_stream_usage_noted(monkeypatch, tmp_path):
    """When the streaming server omits usage despite include_usage, the row
    notes ``stream_usage_present: false`` and token counts default to 0."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "1")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-NOUSAGE")

    def handler(request: httpx.Request) -> httpx.Response:
        # No usage chunk at all.
        return httpx.Response(
            200,
            content=_sse_bytes(_delta("x", None), _delta(None, "stop")),
            headers={"content-type": "text/event-stream"},
        )

    client = _build_streaming(handler)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    assert out == "x"
    row = _json_ttft.loads(
        (tmp_path / "runs" / "RUN-NOUSAGE" / "llm_usage.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["stream_usage_present"] is False
    assert row["prompt_tokens"] == 0
    assert row["completion_tokens"] == 0
    assert "ttft_ms" in row  # ttft was still measured off the content delta


def test_ttft_streaming_rejected_falls_back_to_non_streaming(monkeypatch, tmp_path):
    """Server rejects the streaming request (non-200) → the client falls back
    to a non-streaming call that succeeds, with NO ttft recorded."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "1")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-FALLBACK")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json_ttft.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            return httpx.Response(400, json={"error": "streaming not supported"})
        return httpx.Response(200, json=_success_body("fallback ok"))

    client = _build_streaming(handler)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    assert out == "fallback ok"
    assert client.last_ttft_ms is None
    row = _json_ttft.loads(
        (tmp_path / "runs" / "RUN-FALLBACK" / "llm_usage.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    assert "ttft_ms" not in row


def test_ttft_malformed_sse_falls_back_to_non_streaming(monkeypatch):
    """Malformed SSE JSON → fall back to non-streaming (never fail the call)."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json_ttft.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            return httpx.Response(
                200,
                content=b"data: {not valid json\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=_success_body("recovered"))

    client = _build_streaming(handler)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    assert out == "recovered"
    assert client.last_ttft_ms is None


def test_ttft_reasoning_delta_counts_for_first_token(monkeypatch):
    """A reasoning-only first delta (nemotron/deepseek reasoning_content) still
    marks TTFT; reasoning is reconstructed into message.reasoning_content and
    kept OUT of the returned content (parity with non-streaming _extract_text)."""
    monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        reasoning_chunk = {
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "thinking..."},
                 "finish_reason": None}
            ]
        }
        return httpx.Response(
            200,
            content=_sse_bytes(
                reasoning_chunk,
                _delta("answer", None),
                _delta(None, "stop"),
            ),
            headers={"content-type": "text/event-stream"},
        )

    clock = _FakeClock([0.0, 0.0, 0.25, 1.0])
    client = _build_streaming(handler, monotonic_fn=clock)
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    # Only the content delta reaches the caller; reasoning is isolated.
    assert out == "answer"
    # TTFT measured off the FIRST (reasoning) delta = 0.25s → 250ms.
    assert client.last_ttft_ms == 250.0


def test_resolve_ttft_meter_parse_with_fallback(monkeypatch):
    """Truthy parse: ``1/true/yes/on`` (case-insensitive) → True; else False."""
    from Trainforge.generators._openai_compatible_client import (
        resolve_ttft_meter,
    )

    for truthy in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", truthy)
        assert resolve_ttft_meter() is True
    for falsey in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv("ED4ALL_LLM_TTFT_METER", falsey)
        assert resolve_ttft_meter() is False
    monkeypatch.delenv("ED4ALL_LLM_TTFT_METER", raising=False)
    assert resolve_ttft_meter() is False
