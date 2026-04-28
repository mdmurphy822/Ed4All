"""Unit tests for ClaudeSessionProvider — Wave 107 Phase A.

The provider is the third synthesis backend (alongside mock + anthropic).
It dispatches paraphrase requests to the running Claude Code session via
LocalDispatcher's mailbox bridge so users on Claude Max (no API key) can
produce real LLM-paraphrased training corpora.
"""

from __future__ import annotations

import pytest

from Trainforge.generators._claude_session_provider import ClaudeSessionProvider
from Trainforge.tests._synthesis_fakes import (
    FakeLocalDispatcher,
    make_failure_response,
    make_instruction_response,
    make_preference_response,
)


def test_constructor_requires_dispatcher() -> None:
    """No dispatcher means no Claude Code session — fail loud, do not silently
    fall back to mock or anthropic."""
    with pytest.raises(RuntimeError, match="requires a LocalDispatcher"):
        ClaudeSessionProvider(dispatcher=None)


def test_paraphrase_instruction_returns_rewritten_pair_with_metadata_preserved() -> None:
    async def agent_tool(**_kwargs: object) -> dict:
        return make_instruction_response(
            prompt="Paraphrased: explain RDFS domain",
            completion="RDFS describes; SHACL validates. [chunk_00054]",
        )

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    provider = ClaudeSessionProvider(dispatcher=dispatcher, run_id="run-test-1")

    draft = {
        "prompt": "Original prompt",
        "completion": "Original completion",
        "chunk_id": "rdf_shacl_551_chunk_00054",
        "lo_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type": "explanation",
        "seed": 42,
        "template_id": "understand.explanation",
        "schema_version": "chunk_v4",
        "provider": "mock",
    }
    chunk = {"id": "rdf_shacl_551_chunk_00054", "text": "RDFS allows..."}

    out = provider.paraphrase_instruction(draft, chunk)

    assert out["prompt"] == "Paraphrased: explain RDFS domain"
    assert out["completion"] == "RDFS describes; SHACL validates. [chunk_00054]"
    assert out["provider"] == "claude_session"
    # Metadata preserved verbatim:
    assert out["chunk_id"] == "rdf_shacl_551_chunk_00054"
    assert out["lo_refs"] == ["TO-01"]
    assert out["bloom_level"] == "understand"
    assert out["content_type"] == "explanation"
    assert out["seed"] == 42
    assert out["template_id"] == "understand.explanation"
    assert out["schema_version"] == "chunk_v4"
    # Dispatcher actually got called once:
    assert len(dispatcher.calls) == 1
    agent_type, params = dispatcher.calls[0]
    assert agent_type == "training-synthesizer"
    assert params["kind"] == "instruction"
    assert params["chunk_id"] == "rdf_shacl_551_chunk_00054"
    assert params["chunk_text"] == "RDFS allows..."
    assert params["expected_keys"] == ["prompt", "completion"]


def test_paraphrase_preference_rewrites_chosen_and_rejected() -> None:
    async def agent_tool(**_kwargs: object) -> dict:
        return make_preference_response(
            prompt="Which statement about RDFS is correct?",
            chosen="RDFS describes vocabulary semantics. [chunk_00054]",
            rejected="RDFS validates data graphs against shapes.",
        )

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    provider = ClaudeSessionProvider(dispatcher=dispatcher, run_id="run-test-2")

    draft = {
        "prompt": "Original Q",
        "chosen": "Original chosen",
        "rejected": "Original rejected",
        "chunk_id": "rdf_shacl_551_chunk_00054",
        "misconception_id": "mc_abcd1234efgh5678",
        "seed": 7,
        "provider": "mock",
        "rejected_source": "rule_synthesized",
    }
    chunk = {"id": "rdf_shacl_551_chunk_00054", "text": "RDFS allows..."}

    out = provider.paraphrase_preference(draft, chunk)

    assert out["prompt"] == "Which statement about RDFS is correct?"
    assert out["chosen"] == "RDFS describes vocabulary semantics. [chunk_00054]"
    assert out["rejected"] == "RDFS validates data graphs against shapes."
    assert out["provider"] == "claude_session"
    # Metadata preserved:
    assert out["misconception_id"] == "mc_abcd1234efgh5678"
    assert out["seed"] == 7
    assert out["rejected_source"] == "rule_synthesized"
    # Dispatcher payload:
    assert dispatcher.calls[0][1]["kind"] == "preference"
    assert dispatcher.calls[0][1]["expected_keys"] == ["prompt", "chosen", "rejected"]


def test_dispatcher_failure_raises_runtime_error() -> None:
    async def agent_tool(**_kwargs: object) -> dict:
        return make_failure_response(error="agent timed out", error_code="MAILBOX_TIMEOUT")

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-fail",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(RuntimeError, match="MAILBOX_TIMEOUT"):
        provider.paraphrase_instruction(draft, chunk)


def test_missing_required_output_key_raises_runtime_error() -> None:
    async def agent_tool(**_kwargs: object) -> dict:
        # 'completion' missing from outputs:
        return {"success": True, "outputs": {"prompt": "ok"}, "artifacts": []}

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-malformed",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(RuntimeError, match="missing key 'completion'"):
        provider.paraphrase_instruction(draft, chunk)


class _RecordingCapture:
    """Minimal DecisionCapture stand-in — records ``log_decision`` calls."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def log_decision(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))


def test_paraphrase_instruction_emits_synthesis_provider_call_capture() -> None:
    async def agent_tool(**_kwargs: object) -> dict:
        return make_instruction_response(prompt="p2", completion="c2")

    capture = _RecordingCapture()
    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-cap",
        capture=capture,
    )

    provider.paraphrase_instruction(
        draft={"prompt": "p1", "completion": "c1", "template_id": "understand._default"},
        chunk={"id": "rdf_shacl_551_chunk_00054", "text": "..."},
    )

    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "synthesis_provider_call"
    assert event["decision"] == "claude_session::instruction"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    # Rationale must reference dynamic signals per CLAUDE.md mandate:
    assert "rdf_shacl_551_chunk_00054" in rationale
    assert "understand._default" in rationale


import json
from pathlib import Path


def test_cache_hit_skips_dispatcher_and_returns_cached_output(tmp_path: Path) -> None:
    call_count = 0

    async def agent_tool(**_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return make_instruction_response(prompt=f"p{call_count}", completion=f"c{call_count}")

    cache_path = tmp_path / "synthesis_cache.jsonl"
    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    provider = ClaudeSessionProvider(
        dispatcher=dispatcher,
        run_id="run-cache",
        cache_path=cache_path,
    )

    draft = {
        "prompt": "P", "completion": "C", "template_id": "apply._default",
        "chunk_id": "rdf_shacl_551_chunk_00054",
    }
    chunk = {"id": "rdf_shacl_551_chunk_00054", "text": "..."}

    first = provider.paraphrase_instruction(draft, chunk)
    second = provider.paraphrase_instruction(draft, chunk)

    assert first == second
    assert call_count == 1, "Second call should hit cache, not dispatcher"
    # Cache file persisted:
    assert cache_path.exists()
    lines = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["kind"] == "instruction"
    assert lines[0]["chunk_id"] == "rdf_shacl_551_chunk_00054"
    assert lines[0]["provider_version"] == "v1"
    assert lines[0]["outputs"]["prompt"] == "p1"


def test_cache_invalidates_on_provider_version_bump(tmp_path: Path) -> None:
    call_count = 0

    async def agent_tool(**_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return make_instruction_response(prompt=f"p{call_count}", completion=f"c{call_count}")

    cache_path = tmp_path / "cache.jsonl"
    draft = {"prompt": "P", "completion": "C", "template_id": "x", "chunk_id": "c1"}
    chunk = {"id": "c1", "text": "t"}

    p1 = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        cache_path=cache_path, provider_version="v1",
    )
    p1.paraphrase_instruction(draft, chunk)

    # Same chunk + same draft, but bumped version — should NOT hit cache:
    p2 = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        cache_path=cache_path, provider_version="v2",
    )
    p2.paraphrase_instruction(draft, chunk)

    assert call_count == 2
