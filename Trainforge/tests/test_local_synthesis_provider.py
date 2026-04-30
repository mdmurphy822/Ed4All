"""Wave 113: tests for LocalSynthesisProvider.

Mirrors :mod:`Trainforge.tests.test_together_synthesis_provider` but
exercises the local-server HTTP path. Covers:

- Default base URL is the Ollama default (no env set).
- ``LOCAL_SYNTHESIS_BASE_URL`` env override is picked up.
- ``LOCAL_SYNTHESIS_MODEL`` env override is picked up.
- ``LOCAL_SYNTHESIS_API_KEY`` is OPTIONAL — instantiation succeeds
  with no env / no kwarg / no client (the local server typically
  ignores the auth header).
- Happy path instruction + preference paraphrase set
  ``provider="local"`` (NOT ``"together"``).
- Length-clamp invariant (Wave 112): too-short paraphrase raises
  ``SynthesisProviderError``; over-max output gets truncated.
- Transient HTTP 503 is retried once and succeeds.
- Decision-capture fires once per call with ``base_url``, ``model``,
  ``chunk_id`` interpolated in the rationale.

All tests use ``httpx.MockTransport`` so no real network calls fire —
no real Ollama / vLLM / llama.cpp server is required to run them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import patch

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._local_provider import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_SYNTHESIS_MODEL,
    LocalSynthesisProvider,
    SynthesisProviderError,
)
from Trainforge.generators._together_provider import (  # noqa: E402
    MAX_HTTP_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str) -> dict:
    """Build an Ollama-shaped (= OpenAI-compatible) successful response."""
    return {
        "id": "cmpl-local-test",
        "model": DEFAULT_SYNTHESIS_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 180,
            "completion_tokens": 90,
            "total_tokens": 270,
        },
    }


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def _client_yielding(*responses: httpx.Response) -> httpx.Client:
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    return _make_client(handler)


def _instruction_draft() -> dict:
    return {
        "prompt": "Define the central concept behind topic X.",
        "completion": (
            "Topic X is a central idea in this material. Learners should be "
            "able to recall and restate it."
        ),
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "",
        "template_id": "remember.explanation",
        "provider": "mock",
        "schema_version": "v1",
    }


def _preference_draft() -> dict:
    return {
        "prompt": "Explain topic X clearly enough to avoid the misconception.",
        "chosen": (
            "Topic X is a foundational idea in the course material; the "
            "correct framing emphasises its grounding rules."
        ),
        "rejected": (
            "Topic X is mostly a theoretical curiosity; you can safely "
            "ignore the formal definition for everyday work."
        ),
        "misconception_id": "mc_abc",
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "seed": 17,
        "decision_capture_id": "",
        "rejected_source": "misconception",
        "provider": "mock",
        "schema_version": "v1",
    }


def _chunk() -> dict:
    return {
        "id": "chunk_001",
        "text": (
            "Topic X is the foundational concept in chapter one. Learners "
            "encounter it in every subsequent chapter, and the correct "
            "framing anchors all later examples."
        ),
        "learning_outcome_refs": ["TO-01"],
    }


# ---------------------------------------------------------------------------
# Auth-not-required + env-var resolution
# ---------------------------------------------------------------------------


def test_no_api_key_does_not_raise(monkeypatch):
    """LOCAL_SYNTHESIS_API_KEY is optional — no env / no kwarg must work."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    # No client injected — would raise on Together. Must not raise here.
    p = LocalSynthesisProvider()
    # Authorization header still gets a placeholder so reverse proxies
    # that DO check auth see a stable value.
    assert p._api_key == "local"


def test_default_base_url_is_ollama_when_no_env(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = LocalSynthesisProvider()
    assert p._base_url == DEFAULT_BASE_URL.rstrip("/")
    assert p.api_url == DEFAULT_BASE_URL.rstrip("/") + "/chat/completions"


def test_local_synthesis_base_url_env_override(monkeypatch):
    """LOCAL_SYNTHESIS_BASE_URL env var must be picked up at construction."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    p = LocalSynthesisProvider(client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    # vLLM-style base URL flowed through to the request.
    assert str(seen[0].url) == "http://localhost:8000/v1/chat/completions"


def test_local_synthesis_model_env_override(monkeypatch):
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    p = LocalSynthesisProvider(client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    body = json.loads(seen[0].content.decode("utf-8"))
    assert body["model"] == "Qwen/Qwen2.5-32B-Instruct"


def test_explicit_base_url_kwarg_beats_env(monkeypatch):
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://env-set:9999/v1")
    p = LocalSynthesisProvider(base_url="http://kwarg-set:1234/v1")
    assert p._base_url == "http://kwarg-set:1234/v1"


# ---------------------------------------------------------------------------
# Happy paths — provider tag flips to "local"
# ---------------------------------------------------------------------------


def test_instruction_paraphrase_sets_provider_local(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the central idea introduced for topic X in this material.",
        "completion": (
            "Topic X is the foundational concept of the chapter; recalling "
            "its formal definition is the first step toward applying it in "
            "subsequent material."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())

    # Critical: provider tag must be "local", not "together".
    assert out["provider"] == "local"
    # Metadata preserved.
    assert out["chunk_id"] == "chunk_001"
    assert out["bloom_level"] == "remember"
    assert out["template_id"] == "remember.explanation"
    # Endpoint hit was the Ollama default.
    assert str(seen[0].url) == DEFAULT_BASE_URL + "/chat/completions"
    # Authorization header always present (placeholder when no key).
    assert seen[0].headers["Authorization"] == "Bearer local"
    body = json.loads(seen[0].content.decode("utf-8"))
    assert body["model"] == DEFAULT_SYNTHESIS_MODEL


def test_preference_paraphrase_sets_provider_local(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": (
            "Briefly explain topic X to a learner about to encounter the "
            "common misconception."
        ),
        "chosen": (
            "Topic X is the foundational concept of the chapter; the "
            "course material grounds every later example in its formal "
            "definition."
        ),
        "rejected": (
            "Topic X is essentially optional; my experience says you can "
            "skip the formal definition without losing much."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_preference(_preference_draft(), _chunk())

    # Provider tag flipped to local, NOT together.
    assert out["provider"] == "local"
    assert "foundational concept" in out["chosen"]
    assert out["chunk_id"] == "chunk_001"
    assert out["misconception_id"] == "mc_abc"
    assert out["rejected_source"] == "misconception"


def test_explicit_api_key_kwarg_used_in_authorization_header(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    # Explicit api_key must override the placeholder.
    p = LocalSynthesisProvider(client=client, api_key="proxy-secret-123")
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert seen[0].headers["Authorization"] == "Bearer proxy-secret-123"


# ---------------------------------------------------------------------------
# Length-clamp / Wave 112 invariant
# ---------------------------------------------------------------------------


def test_too_short_completion_retries_then_raises(monkeypatch):
    """Wave 114: below-min completion now triggers a length-retry
    inside ``_call_with_parse``. After ``MAX_PARSE_RETRIES`` exhausted
    short responses the provider raises ``paraphrase_invalid_after_retry``
    (replacing the prior single-shot ``completion_below_minimum``
    raise). No sentinel filler is ever injected."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        # Far below COMPLETION_MIN (50 chars). Repeated 3x to exhaust
        # the length-retry budget.
        "completion": "too short",
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
        httpx.Response(200, json=_success_body(paraphrased)),
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        with pytest.raises(SynthesisProviderError) as excinfo:
            p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "paraphrase_invalid_after_retry"
    assert "completion length 9 below minimum 50" in str(excinfo.value)


def test_too_short_prompt_retries_then_raises(monkeypatch):
    """Wave 114 + Wave 122 follow-up: below-min prompt triggers a
    length-retry. Local floor is now realigned to the schema's 40-char
    PROMPT_MIN; a 9-char prompt trips it. Three exhausted retries ->
    paraphrase_invalid_after_retry."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "too short",  # below DEFAULT_LOCAL_KIND_BOUNDS prompt floor (40).
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
        httpx.Response(200, json=_success_body(paraphrased)),
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        with pytest.raises(SynthesisProviderError) as excinfo:
            p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "paraphrase_invalid_after_retry"
    assert "prompt length 9 below minimum 40" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP retry behavior — 503 (more likely on a local server than 429)
# ---------------------------------------------------------------------------


def test_http_503_then_200_succeeds_after_retry(monkeypatch):
    """Local servers are more prone to 5xx (OOM, model load) than 429."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(503, json={"error": "model loading"}),
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    # Patch sleep so the test doesn't actually wait. The base-class
    # together-provider module owns the retry loop.
    with patch("Trainforge.generators._together_provider.time.sleep"):
        out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "local"


def test_http_500_three_times_raises_with_status_code(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    client = _client_yielding(
        *[httpx.Response(500, json={"error": "boom"})] * MAX_HTTP_RETRIES
    )
    p = LocalSynthesisProvider(client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        with pytest.raises(SynthesisProviderError) as excinfo:
            p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "500"


# ---------------------------------------------------------------------------
# DecisionCapture wiring — must include base_url + model + chunk_id
# ---------------------------------------------------------------------------


def test_decision_capture_fires_with_base_url_and_chunk_id_in_rationale(
    monkeypatch,
):
    """Crucial: rationale must include base_url so post-hoc audit can
    tell which local server (Ollama / vLLM / llama.cpp / LM Studio,
    on which workstation) produced each pair."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    p = LocalSynthesisProvider(
        client=client, capture=_FakeCapture(), base_url="http://my-rig:11434/v1",
    )
    p.paraphrase_instruction(_instruction_draft(), _chunk())

    assert len(captured) == 1
    event = captured[0]
    assert event["decision_type"] == "synthesis_provider_call"
    rationale = event["rationale"]
    # Rationale length contract.
    assert len(rationale) >= 20
    # Three required dynamic signals.
    assert "chunk_id=chunk_001" in rationale
    assert DEFAULT_SYNTHESIS_MODEL in rationale
    assert "http://my-rig:11434/v1" in rationale
    # Decision string also interpolates per-call signals.
    decision = event["decision"]
    assert "chunk_001" in decision
    assert "http_retries=" in decision
    assert "http://my-rig:11434/v1" in decision
    # Wave 115: rationale interpolates per-chunk pedagogical signals
    assert "bloom_level=" in rationale
    assert "concept_tags=" in rationale


# ---------------------------------------------------------------------------
# Wave 113 hardening: json_mode + lenient JSON parse + strict directive
# ---------------------------------------------------------------------------


def test_local_provider_passes_json_mode_to_client(monkeypatch):
    """Critical for 7B-class models: ``json_mode=True`` must flow into
    the embedded client's request payload so Ollama's
    JSON-grammar-constrained decoding fires on every call."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": (
            "Recall the foundational concept introduced for topic X "
            "in chapter one."
        ),
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    p = LocalSynthesisProvider(client=client)
    # Direct check of the embedded client config.
    assert p._oa_client._json_mode is True
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    body = json.loads(seen[0].content.decode("utf-8"))
    # Both fields land in the request: Ollama's top-level ``format``
    # and OpenAI's ``response_format``.
    assert body.get("format") == "json"
    assert body.get("response_format") == {"type": "json_object"}


def test_local_provider_recovers_from_markdown_fence_response(monkeypatch):
    """Happy-path drift recovery: 7B model wraps its JSON in
    ```json ... ``` despite the strict directive. Lenient extractor
    strips the fence and the paraphrase succeeds."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    inner = json.dumps({
        "prompt": (
            "Recall the foundational concept introduced for topic X "
            "in chapter one."
        ),
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    drifted_response = f"```json\n{inner}\n```"
    client = _client_yielding(
        httpx.Response(200, json=_success_body(drifted_response)),
    )
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "local"
    assert "Recall the foundational" in out["prompt"]


def test_local_provider_recovers_from_prose_drift(monkeypatch):
    """Drift recovery: 7B model adds prose around the JSON despite
    the directive. Lenient extractor finds the first balanced object."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    inner = json.dumps({
        "prompt": (
            "Recall the foundational concept introduced for topic X "
            "in chapter one."
        ),
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    drifted_response = f"Sure! Here is the JSON you asked for: {inner}\n\nHope this helps."
    client = _client_yielding(
        httpx.Response(200, json=_success_body(drifted_response)),
    )
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "local"


def test_local_provider_raises_after_lenient_retry_exhaustion(monkeypatch):
    """After 3 unrecoverable responses, raise SynthesisProviderError
    with ``code='paraphrase_invalid_after_retry'`` (Wave 114 unified
    the parse-failure and length-failure exhaustion paths under one
    error code) and a truncated tail of the last response in the
    message — postmortem visibility."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    bad_response = (
        "I cannot help with that request, but here is some other "
        "text that drifts off into natural language and does not "
        "contain a JSON object at all. " * 4
    ) + "DISTINCTIVE_TAIL_MARKER"
    client = _client_yielding(
        httpx.Response(200, json=_success_body(bad_response)),
        httpx.Response(200, json=_success_body(bad_response)),
        httpx.Response(200, json=_success_body(bad_response)),
    )
    p = LocalSynthesisProvider(client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        with pytest.raises(SynthesisProviderError) as excinfo:
            p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "paraphrase_invalid_after_retry"
    # Truncated tail of the last response surfaces in the message.
    assert "DISTINCTIVE_TAIL_MARKER" in str(excinfo.value)


def test_local_provider_prompt_contains_strict_json_directive():
    """The strict-JSON directive must be appended to the user message
    (NOT the system message — keeping the system message unchanged
    preserves provider parity with Together / Anthropic)."""
    user_text = LocalSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001"
    )
    # End-of-prompt directive, verbatim from the constant.
    assert "RESPOND ONLY WITH A JSON OBJECT" in user_text
    assert "Do not wrap in markdown" in user_text
    assert "Do not add commentary" in user_text
    # Both required keys explicitly enumerated.
    assert '"prompt"' in user_text
    assert '"completion"' in user_text

    pref_user_text = LocalSynthesisProvider._render_preference_user(
        _preference_draft(), "chunk_001"
    )
    assert "RESPOND ONLY WITH A JSON OBJECT" in pref_user_text
    # All three preference keys explicitly enumerated.
    assert '"prompt"' in pref_user_text
    assert '"chosen"' in pref_user_text
    assert '"rejected"' in pref_user_text


def test_decision_capture_fires_on_preference_call(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": (
            "Briefly explain topic X to a learner about to encounter the "
            "common misconception."
        ),
        "chosen": (
            "Topic X is the foundational concept of the chapter; the "
            "course material grounds every later example in its formal "
            "definition."
        ),
        "rejected": (
            "Topic X is essentially optional; in my experience you can "
            "skip the formal definition without consequence."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    p = LocalSynthesisProvider(client=client, capture=_FakeCapture())
    p.paraphrase_preference(_preference_draft(), _chunk())

    assert len(captured) == 1
    assert captured[0]["decision_type"] == "synthesis_provider_call"
    assert "chunk_id=chunk_001" in captured[0]["rationale"]
    # Default base URL surfaces in the rationale when none supplied.
    assert DEFAULT_BASE_URL in captured[0]["rationale"]


# ---------------------------------------------------------------------------
# Wave 114: per-provider kind_bounds + length-retry + slim system prompts
# ---------------------------------------------------------------------------


def test_local_provider_default_prompt_floor_matches_schema(monkeypatch):
    """Wave 122 follow-up: ``DEFAULT_LOCAL_KIND_BOUNDS["prompt"]`` is
    realigned to the schema floor (``PROMPT_MIN`` = 40) after the
    14B-Q4 uncapped run on rdf-shacl-551-2 emitted 5/263 schema-invalid
    short prompts the previous 25-char floor admitted. 7B-Q4 callers
    that need a lower floor can still pass ``kind_bounds={"prompt":
    (25, PROMPT_MAX), ...}`` to the constructor."""
    from Trainforge.generators._local_provider import (  # noqa: E402
        DEFAULT_LOCAL_KIND_BOUNDS,
    )
    from Trainforge.generators._anthropic_provider import (  # noqa: E402
        PROMPT_MIN,
    )

    assert DEFAULT_LOCAL_KIND_BOUNDS["prompt"] == (PROMPT_MIN, 400)
    assert PROMPT_MIN == 40  # canonical schema floor

    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    provider = LocalSynthesisProvider()
    assert provider._kind_bounds["prompt"] == (40, 400)
    assert provider._kind_bounds["completion"] == (50, 600)


def test_local_provider_kind_bounds_constructor_override(monkeypatch):
    """Wave 114: ``kind_bounds=`` constructor kwarg overrides the
    module default. Constructor wraps the input in ``dict(kind_bounds)``
    so post-construction mutation of the caller's dict does not leak
    into the provider's internal state."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    custom_bounds = {
        "prompt": (15, 200),
        "completion": (40, 500),
        "chosen": (40, 500),
        "rejected": (40, 500),
    }
    expected = dict(custom_bounds)
    provider = LocalSynthesisProvider(kind_bounds=custom_bounds)
    assert provider._kind_bounds == expected

    # Mutate the caller's dict AFTER construction; the provider's
    # internal copy must remain unchanged.
    custom_bounds["prompt"] = (1, 1)
    custom_bounds["completion"] = (1, 1)
    assert provider._kind_bounds == expected
    assert provider._kind_bounds["prompt"] == (15, 200)


def test_local_provider_retries_on_short_field_then_succeeds(monkeypatch):
    """Wave 114: a short ``prompt`` field triggers a length-retry
    (parallel to the JSON-parse retry path). On retry success, the
    final paraphrase carries the LONG prompt — proving the retry path
    accepted the second response and discarded the first."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    short_response = json.dumps({
        "prompt": "short",  # 5 chars — far below the 40-char floor.
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    long_prompt = (
        "Recall the foundational shacl concept introduced for topic X "
        "in chapter one of the source material."
    )
    long_response = json.dumps({
        "prompt": long_prompt,
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(short_response)),
        httpx.Response(200, json=_success_body(long_response)),
    )
    p = LocalSynthesisProvider(client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        result = p.paraphrase_instruction(_instruction_draft(), _chunk())

    # Second response was accepted, first was discarded.
    assert result["prompt"] == long_prompt
    assert len(result["prompt"]) >= 40
    assert "shacl" in result["prompt"]
    assert result["provider"] == "local"


def test_local_provider_uses_slim_local_system_prompts():
    """Wave 114: the local path uses module-level slim system prompts
    (<50 words each) instead of the verbose Together / Anthropic
    prompts. 7B-Q4 instruction models attend less reliably to long
    behavioral preambles; the inlined JSON shape directive in the user
    message is the most-respected part of the prompt."""
    from Trainforge.generators._local_provider import (  # noqa: E402
        _LOCAL_INSTRUCTION_SYSTEM_PROMPT,
        _LOCAL_PREFERENCE_SYSTEM_PROMPT,
    )

    instruction_word_count = len(_LOCAL_INSTRUCTION_SYSTEM_PROMPT.split())
    preference_word_count = len(_LOCAL_PREFERENCE_SYSTEM_PROMPT.split())
    assert instruction_word_count < 50, (
        f"_LOCAL_INSTRUCTION_SYSTEM_PROMPT must stay under 50 words; "
        f"got {instruction_word_count}"
    )
    assert preference_word_count < 50, (
        f"_LOCAL_PREFERENCE_SYSTEM_PROMPT must stay under 50 words; "
        f"got {preference_word_count}"
    )

    # Behavioral anchors — task framing per surface.
    assert "You paraphrase" in _LOCAL_INSTRUCTION_SYSTEM_PROMPT
    assert "DPO" in _LOCAL_PREFERENCE_SYSTEM_PROMPT
    # Inlined JSON shape directive — instruction surface enumerates
    # the ``prompt`` key explicitly so the model sees the required
    # shape end-to-end.
    assert '{"prompt"' in _LOCAL_INSTRUCTION_SYSTEM_PROMPT


def test_rationale_interpolates_chunk_bloom_and_concept_tags(monkeypatch):
    """Wave 115: rationale string varies per-chunk so the decision
    capture validator scores it 'proficient' rather than 'developing'."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })

    captured: List[dict] = []

    class _Capture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client, capture=_Capture())

    chunk_with_signals = {
        "id": "chunk_001",
        "text": "Topic X is the foundational concept.",
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "analyze",
        "concept_tags": ["sh-datatype", "sh-class", "rdfs-subclassof", "owl-sameas"],
    }
    p.paraphrase_instruction(_instruction_draft(), chunk_with_signals)

    assert len(captured) == 1
    rationale = captured[0]["rationale"]
    assert "bloom_level=analyze" in rationale
    # Only the first 3 concept_tags interpolate (cap at 3 to bound length).
    assert "sh-datatype" in rationale
    assert "sh-class" in rationale
    assert "rdfs-subclassof" in rationale
    assert "owl-sameas" not in rationale


# ---------------------------------------------------------------------------
# Wave 120: surface-form preservation
# ---------------------------------------------------------------------------


def test_paraphrase_instruction_includes_preserve_directive_in_user_prompt(
    monkeypatch,
):
    """When ``preserve_tokens`` is non-empty, the user prompt sent to the
    model carries an explicit 'PRESERVE THESE TOKENS VERBATIM' directive
    naming each token. 14B-class local models silently rewrite
    ``sh:NodeShape`` to 'node shape' without this directive."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Explain how sh:NodeShape applies to topic X.",
        "completion": (
            "sh:NodeShape constrains node-typed nodes; learners apply it "
            "by validating instances against the declared shape and "
            "checking sh:datatype on each property."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    p = LocalSynthesisProvider(client=_make_client(handler))
    out = p.paraphrase_instruction(
        _instruction_draft(), _chunk(),
        preserve_tokens=["sh:NodeShape", "sh:datatype"],
    )
    body = json.loads(seen[0].content.decode("utf-8"))
    user_msg = body["messages"][-1]["content"]
    assert "PRESERVE THESE TOKENS VERBATIM" in user_msg
    assert "'sh:NodeShape'" in user_msg
    assert "'sh:datatype'" in user_msg
    # And the model's output is preserved through the clamp.
    assert "sh:NodeShape" in out["prompt"] or "sh:NodeShape" in out["completion"]


def test_paraphrase_instruction_retries_on_preserve_miss(monkeypatch):
    """When the first response drops a preserve_token, the provider
    appends a remediation message and retries. A subsequent good response
    is accepted; ``preserve-retry`` warning is logged."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    bad_paraphrase = json.dumps({
        "prompt": "Explain how the node shape concept applies to topic X.",
        "completion": (
            "The node-shape idea constrains node-typed entities; a learner "
            "applies it by validating instances against the declared "
            "constraint and checking the datatype on each property."
        ),
    })
    good_paraphrase = json.dumps({
        "prompt": "Explain how sh:NodeShape applies to topic X.",
        "completion": (
            "sh:NodeShape constrains node-typed nodes; learners apply it "
            "by validating instances against the declared shape constraint "
            "and the sh:datatype rule on each property."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(bad_paraphrase)),
        httpx.Response(200, json=_success_body(good_paraphrase)),
    )
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_instruction(
        _instruction_draft(), _chunk(),
        preserve_tokens=["sh:NodeShape", "sh:datatype"],
    )
    assert "sh:NodeShape" in out["prompt"] + out["completion"]
    assert "sh:datatype" in out["prompt"] + out["completion"]


def test_paraphrase_instruction_raises_preservation_failed_after_retry(
    monkeypatch,
):
    """When all retries return a paraphrase that drops a required token,
    the provider raises ``SynthesisProviderError`` with code
    ``surface_form_preservation_failed`` so the caller can fall back to
    the deterministic draft."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    bad_paraphrase = json.dumps({
        "prompt": "Explain how the node shape applies to topic X.",
        "completion": (
            "The node-shape concept constrains node-typed entities; a "
            "learner applies it by validating instances against the "
            "declared constraint and the per-property type rule."
        ),
    })
    # Yield bad responses for all 3 retries. ``MAX_PARSE_RETRIES`` is 3
    # so we provide 4 to guard against off-by-one.
    client = _client_yielding(*[
        httpx.Response(200, json=_success_body(bad_paraphrase)) for _ in range(5)
    ])
    p = LocalSynthesisProvider(client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(
            _instruction_draft(), _chunk(),
            preserve_tokens=["sh:NodeShape"],
        )
    assert excinfo.value.code == "surface_form_preservation_failed"


def test_max_parse_retries_constructor_kwarg_caps_retry_budget(monkeypatch):
    """Wave 120 smoke-mode fix: ``max_parse_retries=1`` caps the parse-
    retry loop at a single attempt so a property-heavy stratified
    sample doesn't compound retry cost. With cap=1 a single bad
    response immediately raises (no retry), and the error message
    reports the runtime budget — not the module default."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    bad_paraphrase = json.dumps({
        "prompt": "Explain how the node shape applies to topic X.",
        "completion": (
            "The node-shape concept constrains node-typed entities; a "
            "learner applies it by validating instances against the "
            "declared constraint."
        ),
    })
    client = _client_yielding(*[
        httpx.Response(200, json=_success_body(bad_paraphrase)) for _ in range(5)
    ])
    p = LocalSynthesisProvider(client=client, max_parse_retries=1)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(
            _instruction_draft(), _chunk(),
            preserve_tokens=["sh:NodeShape"],
        )
    assert excinfo.value.code == "surface_form_preservation_failed"
    # Error message reports the runtime budget (1), not the module default (3).
    assert "after 1 attempts" in str(excinfo.value)


def test_paraphrase_preference_only_checks_chosen_field(monkeypatch):
    """Preference pairs check ``chosen`` only — the rule-synthesized
    rejection legitimately may not contain the literal CURIE. A
    paraphrase that preserves the token in chosen but drops it in
    rejected is accepted."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Explain sh:NodeShape clearly enough to avoid the misconception.",
        "chosen": (
            "sh:NodeShape is a foundational SHACL construct; the correct "
            "framing emphasises its grounding rules and how it constrains "
            "node-typed instances."
        ),
        "rejected": (
            "Node shapes are mostly a theoretical curiosity; you can "
            "safely ignore the formal definition for everyday work."
        ),
    })
    client = _client_yielding(httpx.Response(200, json=_success_body(paraphrased)))
    p = LocalSynthesisProvider(client=client)
    out = p.paraphrase_preference(
        _preference_draft(), _chunk(),
        preserve_tokens=["sh:NodeShape"],
    )
    assert "sh:NodeShape" in out["chosen"]
    assert "sh:NodeShape" not in out["rejected"]  # legitimate
