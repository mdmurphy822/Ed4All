"""Wave 113 prep: tests for TogetherSynthesisProvider.

Mirrors :mod:`Trainforge.tests.test_anthropic_synthesis_provider` but
exercises the Together AI HTTP path. Covers:

- Missing TOGETHER_API_KEY raises a clear RuntimeError naming the env var.
- Happy path instruction + preference paraphrase preserve metadata and
  flip ``provider`` to ``"together"``.
- Length-clamp invariant (Wave 112): too-short paraphrase raises
  ``SynthesisProviderError``; over-max output gets truncated.
- Transient HTTP 429 is retried; persistent 5xx exhausts retries with
  ``code`` set to the HTTP status string.
- 4xx other than 429 surfaces immediately with the HTTP status code.
- DecisionCapture fires once per call with chunk_id + model in the
  rationale (LLM call-site instrumentation contract).

All tests use ``httpx.MockTransport`` so no real network calls fire.
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

from Trainforge.generators._together_provider import (  # noqa: E402
    DEFAULT_SYNTHESIS_MODEL,
    MAX_HTTP_RETRIES,
    TOGETHER_API_URL,
    SynthesisProviderError,
    TogetherSynthesisProvider,
)
from Trainforge.generators._anthropic_provider import (  # noqa: E402
    COMPLETION_MIN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str) -> dict:
    """Build an OpenAI-shaped successful response payload."""
    return {
        "id": "cmpl-test",
        "model": DEFAULT_SYNTHESIS_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
        },
    }


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Wrap a per-request handler in an httpx.MockTransport client."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def _client_yielding(*responses: httpx.Response) -> httpx.Client:
    """Return a client whose handler walks the response list one call at a time.

    Mirrors the Anthropic-test ``_client_returning`` helper. Excess calls
    raise ``IndexError`` so an unintended extra request fails the test
    loud rather than silently re-using the last response.
    """
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
            "encounter it in every subsequent chapter, and the correct framing "
            "anchors all later examples."
        ),
        "learning_outcome_refs": ["TO-01"],
    }


# ---------------------------------------------------------------------------
# Missing-API-key path
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        TogetherSynthesisProvider()
    msg = str(excinfo.value)
    assert "TOGETHER_API_KEY" in msg
    assert "together" in msg.lower()


def test_injected_client_bypasses_api_key_check(monkeypatch):
    """Tests can inject a mock client without setting the env var."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    client = _client_yielding(
        httpx.Response(200, json=_success_body("{}"))
    )
    p = TogetherSynthesisProvider(client=client)
    assert p.client is client


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_instruction_paraphrase_replaces_prompt_and_completion():
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
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())

    # Prompt + completion replaced.
    assert "Recall the central idea" in out["prompt"]
    assert "foundational concept" in out["completion"]
    # Metadata preserved.
    assert out["chunk_id"] == "chunk_001"
    assert out["bloom_level"] == "remember"
    assert out["template_id"] == "remember.explanation"
    assert out["lo_refs"] == ["TO-01"]
    assert out["seed"] == 17
    # Provider tag flipped to together.
    assert out["provider"] == "together"
    # Endpoint + auth header asserted exactly once.
    assert len(seen) == 1
    req = seen[0]
    assert str(req.url) == TOGETHER_API_URL
    assert req.headers["Authorization"] == "Bearer tg-test"
    body = json.loads(req.content.decode("utf-8"))
    assert body["model"] == DEFAULT_SYNTHESIS_MODEL
    # OpenAI-compatible payload shape.
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "Source chunk text" in body["messages"][0]["content"]


def test_preference_paraphrase_replaces_prompt_chosen_rejected():
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
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    draft = _preference_draft()
    out = p.paraphrase_preference(draft, _chunk())

    assert "foundational concept" in out["chosen"]
    assert out["rejected"] != draft["rejected"]
    assert out["chunk_id"] == "chunk_001"
    assert out["misconception_id"] == "mc_abc"
    assert out["rejected_source"] == "misconception"
    assert out["provider"] == "together"


def test_together_synthesis_model_env_override(monkeypatch):
    monkeypatch.setenv("TOGETHER_SYNTHESIS_MODEL", "Qwen/Qwen2.5-72B-Instruct-Turbo")
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrased))

    client = _make_client(handler)
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())

    body = json.loads(seen[0].content.decode("utf-8"))
    assert body["model"] == "Qwen/Qwen2.5-72B-Instruct-Turbo"


# ---------------------------------------------------------------------------
# Length-clamp / Wave 112 invariant
# ---------------------------------------------------------------------------


def test_too_short_completion_raises_synthesis_provider_error():
    """Below-min paraphrase must raise (no sentinel filler)."""
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        # Far below COMPLETION_MIN (50 chars).
        "completion": "too short",
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "completion_below_minimum"
    assert "below minimum" in str(excinfo.value)


def test_too_short_prompt_raises_synthesis_provider_error():
    paraphrased = json.dumps({
        "prompt": "too short",  # below PROMPT_MIN (40 chars).
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    })
    client = _client_yielding(
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "prompt_below_minimum"


def test_clamp_truncates_over_max_completion():
    """Direct unit test of the clamp helper's truncation branch."""
    p = TogetherSynthesisProvider.__new__(TogetherSynthesisProvider)
    long_text = ("This sentence is grammatically complete. " * 40).strip()
    out = p._clamp(long_text, kind="completion")
    assert len(out) <= 600  # COMPLETION_MAX
    assert out  # non-empty


# ---------------------------------------------------------------------------
# HTTP retry behavior
# ---------------------------------------------------------------------------


def test_http_429_then_200_succeeds_after_retry():
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    client = _client_yielding(
        httpx.Response(429, json={"error": "rate limited"}),
        httpx.Response(200, json=_success_body(paraphrased)),
    )
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    # Patch sleep so the test doesn't actually wait.
    with patch("Trainforge.generators._together_provider.time.sleep"):
        out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "foundational concept" in out["prompt"]


def test_http_500_three_times_raises_with_status_code():
    client = _client_yielding(
        *[httpx.Response(500, json={"error": "boom"})] * MAX_HTTP_RETRIES
    )
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    with patch("Trainforge.generators._together_provider.time.sleep"):
        with pytest.raises(SynthesisProviderError) as excinfo:
            p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "500"


def test_http_400_surfaces_immediately_without_retry():
    """4xx other than 429 must NOT be retried."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    client = _make_client(handler)
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert excinfo.value.code == "400"
    # Exactly one POST — no retry on a permanent 4xx.
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# DecisionCapture wiring
# ---------------------------------------------------------------------------


def test_decision_capture_fires_per_call_with_chunk_id_in_rationale():
    """Regression test for the LLM call-site instrumentation contract.

    Every Together paraphrase call must emit one
    ``synthesis_provider_call`` decision-capture event whose rationale
    interpolates dynamic signals (chunk_id, model, retry_count, token
    counts) — at least 20 chars per the project contract.
    """
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

    p = TogetherSynthesisProvider(
        api_key="tg-test", client=client, capture=_FakeCapture()
    )
    p.paraphrase_instruction(_instruction_draft(), _chunk())

    assert len(captured) == 1
    event = captured[0]
    assert event["decision_type"] == "synthesis_provider_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    # Dynamic signals interpolated.
    assert "chunk_id=chunk_001" in rationale
    assert DEFAULT_SYNTHESIS_MODEL in rationale
    assert "template_id=remember.explanation" in rationale
    # Decision string interpolates token counts + retry count.
    decision = event["decision"]
    assert "chunk_001" in decision
    assert "retry_count=" in decision


def test_decision_capture_fires_on_preference_call():
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

    p = TogetherSynthesisProvider(
        api_key="tg-test", client=client, capture=_FakeCapture()
    )
    p.paraphrase_preference(_preference_draft(), _chunk())

    assert len(captured) == 1
    assert captured[0]["decision_type"] == "synthesis_provider_call"
    assert "chunk_id=chunk_001" in captured[0]["rationale"]


# ---------------------------------------------------------------------------
# JSON-fence tolerance + parse-retry
# ---------------------------------------------------------------------------


def test_json_fence_response_parsed_correctly():
    """Some OSS models wrap their JSON output in ```json fences."""
    fenced = (
        "```json\n"
        + json.dumps({
            "prompt": "Recall the foundational concept introduced for topic X in chapter one.",
            "completion": (
                "Topic X anchors every later chapter; recall its formal "
                "definition before attempting application questions."
            ),
        })
        + "\n```"
    )
    client = _client_yielding(
        httpx.Response(200, json=_success_body(fenced)),
    )
    p = TogetherSynthesisProvider(api_key="tg-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "foundational concept" in out["prompt"]


def test_clamp_unknown_kind_raises_value_error():
    """Defensive: feeding an unknown kind into _clamp is a programmer error."""
    p = TogetherSynthesisProvider.__new__(TogetherSynthesisProvider)
    with pytest.raises(ValueError):
        p._clamp("x" * 100, kind="not_a_real_kind")
