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
