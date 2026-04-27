"""Wave 91 Action A: tests for AnthropicSynthesisProvider.

Covers:
- Missing API key raises a clear RuntimeError naming the env var.
- Valid response is parsed and shape-validated; output preserves draft
  metadata and replaces only ``prompt`` / ``completion`` (instruction)
  or ``prompt`` / ``chosen`` / ``rejected`` (preference).
- Malformed JSON retries up to MAX_PARSE_RETRIES then raises RuntimeError.
- ``cache_control: ephemeral`` is set on the chunk-text system block.
- DecisionCapture fires once per call when injected.

All tests mock the Anthropic ``client.messages.create`` call — no real
network access. The fixture chunks are minimal hand-rolled dicts so the
tests don't depend on the Trainforge fixture corpus shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._anthropic_provider import (  # noqa: E402
    DEFAULT_SYNTHESIS_MODEL,
    MAX_PARSE_RETRIES,
    AnthropicSynthesisProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(text: str, *, input_tokens: int = 100, output_tokens: int = 50) -> Any:
    """Build a MagicMock that mimics the Anthropic ``Message`` shape."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = input_tokens
    response.usage = usage
    return response


def _client_returning(*texts: str) -> Any:
    """Build a mock Anthropic client whose ``messages.create`` walks the
    supplied text list one call at a time."""
    client = MagicMock()
    responses = [_mock_response(t) for t in texts]
    client.messages.create.side_effect = responses
    return client


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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        AnthropicSynthesisProvider()
    msg = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "anthropic" in msg.lower()


def test_injected_client_bypasses_api_key_check(monkeypatch):
    """Tests can inject a mock client without setting the env var."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = MagicMock()
    p = AnthropicSynthesisProvider(client=client)
    assert p.client is client


# ---------------------------------------------------------------------------
# Instruction paraphrase happy path
# ---------------------------------------------------------------------------


def test_instruction_paraphrase_parses_and_preserves_metadata():
    paraphrased = json.dumps({
        "prompt": "Recall the central idea introduced for topic X in this material.",
        "completion": (
            "Topic X is the foundational concept of the chapter; recalling "
            "its formal definition is the first step toward applying it in "
            "subsequent material."
        ),
    })
    client = _client_returning(paraphrased)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    draft = _instruction_draft()
    chunk = _chunk()

    out = p.paraphrase_instruction(draft, chunk)

    # Prompt + completion replaced.
    assert out["prompt"] != draft["prompt"]
    assert out["completion"] != draft["completion"]
    assert "Recall the central idea" in out["prompt"]
    # Metadata preserved.
    assert out["chunk_id"] == "chunk_001"
    assert out["bloom_level"] == "remember"
    assert out["template_id"] == "remember.explanation"
    assert out["lo_refs"] == ["TO-01"]
    assert out["seed"] == 17
    # Provider tag flipped to anthropic.
    assert out["provider"] == "anthropic"
    # Exactly one SDK call.
    assert client.messages.create.call_count == 1


def test_instruction_paraphrase_uses_default_model():
    client = _client_returning(json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    }))
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_SYNTHESIS_MODEL


def test_anthropic_synthesis_model_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_SYNTHESIS_MODEL", "claude-test-override")
    client = _client_returning(json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    }))
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert client.messages.create.call_args.kwargs["model"] == "claude-test-override"


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------


def test_cache_control_block_is_set_on_system_prompt():
    client = _client_returning(json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    }))
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    p.paraphrase_instruction(_instruction_draft(), _chunk())

    call_kwargs = client.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert isinstance(system_blocks, list)
    # Last block (chunk-text block) carries cache_control: ephemeral.
    last = system_blocks[-1]
    assert last["type"] == "text"
    assert last.get("cache_control") == {"type": "ephemeral"}
    # First block (instructions) is NOT cached — chunk text is the prefix
    # we want to amortise across multiple paraphrase calls per chunk.
    first = system_blocks[0]
    assert first["type"] == "text"
    assert "cache_control" not in first


# ---------------------------------------------------------------------------
# Retry / failure modes
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_succeeds():
    """Two malformed responses, then a valid one — expect a parsed result."""
    bad = "this is not JSON at all"
    bad2 = "{still not valid"
    good = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    })
    client = _client_returning(bad, bad2, good)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "foundational concept" in out["prompt"]
    assert client.messages.create.call_count == 3


def test_malformed_json_exhausts_retries_then_raises():
    bad_responses = ["nope"] * MAX_PARSE_RETRIES
    client = _client_returning(*bad_responses)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    with pytest.raises(RuntimeError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "MAX_PARSE_RETRIES" in str(excinfo.value) or str(MAX_PARSE_RETRIES) in str(excinfo.value)
    assert client.messages.create.call_count == MAX_PARSE_RETRIES


def test_response_missing_required_keys_retries():
    """Response is valid JSON but missing 'completion' — retry path."""
    missing = json.dumps({"prompt": "only prompt, no completion"})
    good = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    })
    client = _client_returning(missing, good)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert client.messages.create.call_count == 2
    assert "foundational concept" in out["prompt"]


# ---------------------------------------------------------------------------
# Preference paraphrase
# ---------------------------------------------------------------------------


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
    client = _client_returning(paraphrased)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    draft = _preference_draft()
    chunk = _chunk()

    out = p.paraphrase_preference(draft, chunk)

    assert out["prompt"] != draft["prompt"]
    assert out["chosen"] != draft["chosen"]
    assert out["rejected"] != draft["rejected"]
    assert "foundational concept" in out["chosen"]
    # Metadata preserved.
    assert out["chunk_id"] == "chunk_001"
    assert out["misconception_id"] == "mc_abc"
    assert out["rejected_source"] == "misconception"
    assert out["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# DecisionCapture wiring
# ---------------------------------------------------------------------------


def test_decision_capture_fires_per_call():
    paraphrased = json.dumps({
        "prompt": "Recall the foundational concept introduced for topic X.",
        "completion": (
            "Topic X anchors every later chapter; recall its definition "
            "before attempting application questions in this course."
        ),
    })
    client = _client_returning(paraphrased)
    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    p = AnthropicSynthesisProvider(
        api_key="sk-test", client=client, capture=_FakeCapture()
    )
    p.paraphrase_instruction(_instruction_draft(), _chunk())

    assert len(captured) == 1
    event = captured[0]
    assert event["decision_type"] == "synthesis_provider_call"
    # Rationale must reference dynamic signals.
    rationale = event["rationale"]
    assert len(rationale) >= 20
    assert "chunk_id=chunk_001" in rationale
    assert "template_id=remember.explanation" in rationale
    assert "draft_prompt_len=" in rationale


def test_decision_capture_fires_on_preference_call():
    paraphrased = json.dumps({
        "prompt": "Briefly explain topic X to a learner.",
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
    client = _client_returning(paraphrased)
    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    p = AnthropicSynthesisProvider(
        api_key="sk-test", client=client, capture=_FakeCapture()
    )
    p.paraphrase_preference(_preference_draft(), _chunk())

    assert len(captured) == 1
    assert captured[0]["decision_type"] == "synthesis_provider_call"


# ---------------------------------------------------------------------------
# JSON-fence tolerance
# ---------------------------------------------------------------------------


def test_json_fence_response_parsed_correctly():
    """Some Claude responses arrive wrapped in ```json fences."""
    fenced = (
        "```json\n"
        + json.dumps({
            "prompt": "Recall the foundational concept introduced for topic X.",
            "completion": (
                "Topic X anchors every later chapter; recall its formal "
                "definition before attempting application questions."
            ),
        })
        + "\n```"
    )
    client = _client_returning(fenced)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "foundational concept" in out["prompt"]


def test_json_with_preamble_recovered_via_brace_scan():
    """Model accidentally adds a preamble — first {...} is scanned."""
    text = (
        "Sure! Here you go:\n"
        + json.dumps({
            "prompt": "Recall the foundational concept introduced for topic X.",
            "completion": (
                "Topic X anchors every later chapter; recall its formal "
                "definition before attempting application questions."
            ),
        })
        + "\nhope that helps."
    )
    client = _client_returning(text)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "foundational concept" in out["prompt"]
