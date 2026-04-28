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
