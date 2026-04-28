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


def test_too_short_completion_raises_synthesis_provider_error(monkeypatch):
    """Below-min paraphrase must raise (no sentinel filler)."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        # Far below COMPLETION_MIN (50 chars).
        "completion": "too short",
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "completion_below_minimum"


def test_too_short_prompt_raises_synthesis_provider_error(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    paraphrased = json.dumps({
        "prompt": "too short",  # below PROMPT_MIN (40 chars).
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = LocalSynthesisProvider(client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "prompt_below_minimum"


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
    with ``code='json_parse_failed_after_lenient_retry'`` and a
    truncated tail of the last response in the message — postmortem
    visibility."""
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
    assert excinfo.value.code == "json_parse_failed_after_lenient_retry"
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
