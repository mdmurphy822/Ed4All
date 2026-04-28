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


# Wave 112 Task 4: helpers that produce strings comfortably above the
# PROMPT_MIN=40 / COMPLETION_MIN=50 floors enforced by
# _claude_session_provider._validate_lengths. The marker token lets each
# test still uniquely identify a paraphrase output.
def _ok_prompt(marker: str = "p") -> str:
    return f"Paraphrased prompt {marker} explaining RDFS in detail for the learner."


def _ok_completion(marker: str = "c") -> str:
    return (
        f"Paraphrased completion {marker} grounded in the source chunk text "
        f"covering RDFS semantics and SHACL validation contracts."
    )


def test_constructor_requires_dispatcher() -> None:
    """No dispatcher means no Claude Code session — fail loud, do not silently
    fall back to mock or anthropic."""
    with pytest.raises(RuntimeError, match="requires a LocalDispatcher"):
        ClaudeSessionProvider(dispatcher=None)


def test_paraphrase_instruction_returns_rewritten_pair_with_metadata_preserved() -> None:
    # Wave 112 Task 4 floors: PROMPT_MIN=40, COMPLETION_MIN=50.
    paraphrased_prompt = (
        "Paraphrased: explain how RDFS defines the domain "
        "and range of properties in vocabularies."
    )
    paraphrased_completion = (
        "RDFS describes vocabulary semantics with rdfs:domain and "
        "rdfs:range, while SHACL validates instance graphs against "
        "node and property shapes. [chunk_00054]"
    )

    async def agent_tool(**_kwargs: object) -> dict:
        return make_instruction_response(
            prompt=paraphrased_prompt,
            completion=paraphrased_completion,
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

    assert out["prompt"] == paraphrased_prompt
    assert out["completion"] == paraphrased_completion
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
    # Wave 112 Task 4 floors: PROMPT_MIN=40, COMPLETION_MIN=50.
    paraphrased_prompt = (
        "Which statement about RDFS describes its purpose accurately "
        "with respect to SHACL validation?"
    )
    paraphrased_chosen = (
        "RDFS describes vocabulary semantics; it does not validate "
        "instance data against shapes the way SHACL does. [chunk_00054]"
    )
    paraphrased_rejected = (
        "RDFS validates data graphs against shapes by emitting conformance "
        "reports the same way that SHACL conformance does."
    )

    async def agent_tool(**_kwargs: object) -> dict:
        return make_preference_response(
            prompt=paraphrased_prompt,
            chosen=paraphrased_chosen,
            rejected=paraphrased_rejected,
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

    assert out["prompt"] == paraphrased_prompt
    assert out["chosen"] == paraphrased_chosen
    assert out["rejected"] == paraphrased_rejected
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
        return make_instruction_response(
            prompt=_ok_prompt("p2"), completion=_ok_completion("c2"),
        )

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
        return make_instruction_response(
            prompt=_ok_prompt(f"p{call_count}"),
            completion=_ok_completion(f"c{call_count}"),
        )

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
    assert lines[0]["outputs"]["prompt"] == _ok_prompt("p1")


def test_cache_invalidates_on_provider_version_bump(tmp_path: Path) -> None:
    call_count = 0

    async def agent_tool(**_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return make_instruction_response(
            prompt=_ok_prompt(f"p{call_count}"),
            completion=_ok_completion(f"c{call_count}"),
        )

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


def test_max_dispatches_cap_raises_synthesis_budget_exceeded(tmp_path: Path) -> None:
    """When the cap is hit, the next dispatch raises before contacting
    the dispatcher — partial work in the cache is preserved."""
    from Trainforge.generators._session_budget import SynthesisBudgetExceeded
    call_count = 0

    async def agent_tool(**_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return make_instruction_response(
            prompt=_ok_prompt(f"p{call_count}"),
            completion=_ok_completion(f"c{call_count}"),
        )

    cache = tmp_path / "cache.jsonl"
    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        cache_path=cache,
        max_dispatches=2,
        telemetry_path=tmp_path / "telemetry.jsonl",
    )
    chunk = lambda i: {"id": f"chunk_{i}", "text": "x"}
    draft = lambda i: {"prompt": f"P{i}", "completion": f"C{i}", "template_id": "t",
                       "chunk_id": f"chunk_{i}"}
    provider.paraphrase_instruction(draft(1), chunk(1))
    provider.paraphrase_instruction(draft(2), chunk(2))
    with pytest.raises(SynthesisBudgetExceeded) as ei:
        provider.paraphrase_instruction(draft(3), chunk(3))
    assert ei.value.dispatched == 2
    assert call_count == 2  # 3rd call NEVER reached the dispatcher


def test_cache_hits_do_not_tick_dispatch_counter(tmp_path: Path) -> None:
    """Re-running against a populated cache costs zero dispatches even
    when max_dispatches is 1."""
    async def agent_tool(**_kwargs: object) -> dict:
        return make_instruction_response(
            prompt=_ok_prompt("p1"), completion=_ok_completion("c1"),
        )

    cache = tmp_path / "cache.jsonl"
    p1 = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        cache_path=cache,
        max_dispatches=1,
    )
    chunk = {"id": "chunk_1", "text": "x"}
    draft = {"prompt": "P", "completion": "C", "template_id": "t", "chunk_id": "chunk_1"}
    p1.paraphrase_instruction(draft, chunk)

    p2 = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        cache_path=cache,
        max_dispatches=1,
    )
    out = p2.paraphrase_instruction(draft, chunk)
    assert out["prompt"] == _ok_prompt("p1")
    p2.paraphrase_instruction(
        {"prompt": "P2", "completion": "C2", "template_id": "t", "chunk_id": "chunk_2"},
        {"id": "chunk_2", "text": "x"},
    )


def test_telemetry_jsonl_written_once_per_call(tmp_path: Path) -> None:
    async def agent_tool(*, task_params, **_kwargs: object) -> dict:
        if task_params["kind"] == "instruction":
            return make_instruction_response(
                prompt=_ok_prompt(), completion=_ok_completion(),
            )
        return make_preference_response(
            prompt=_ok_prompt(),
            chosen=_ok_completion("chosen"),
            rejected=_ok_completion("rejected"),
        )

    tel = tmp_path / "telemetry.jsonl"
    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        telemetry_path=tel,
    )
    provider.paraphrase_instruction(
        {"prompt": "P", "completion": "C", "template_id": "t", "chunk_id": "c1"},
        {"id": "c1", "text": "x"},
    )
    provider.paraphrase_preference(
        {"prompt": "P", "chosen": "C", "rejected": "R", "template_id": "t",
         "chunk_id": "c1"},
        {"id": "c1", "text": "x"},
    )
    rows = [json.loads(l) for l in tel.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["instruction", "preference"]


def test_dispatch_rejects_empty_string_value() -> None:
    """Wave 112 Task 3: empty-string output values must fail loud rather
    than silently passing the key-presence check."""
    from Trainforge.generators._anthropic_provider import SynthesisProviderError

    async def agent_tool(**_kwargs: object) -> dict:
        return {
            "success": True,
            "outputs": {
                "prompt": "",
                "completion": "ok-prompt-text-of-sufficient-length",
            },
            "artifacts": [],
        }

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-empty-string",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(SynthesisProviderError) as ei:
        provider.paraphrase_instruction(draft, chunk)
    assert ei.value.code == "empty_field"
    assert "prompt" in str(ei.value)


def test_dispatch_rejects_whitespace_only_value() -> None:
    """Whitespace-only output values are equivalent to empty for our purposes."""
    from Trainforge.generators._anthropic_provider import SynthesisProviderError

    async def agent_tool(**_kwargs: object) -> dict:
        return {
            "success": True,
            "outputs": {
                "prompt": "ok-prompt-text-of-sufficient-length",
                "completion": "  ",
            },
            "artifacts": [],
        }

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-whitespace",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(SynthesisProviderError) as ei:
        provider.paraphrase_instruction(draft, chunk)
    assert ei.value.code == "empty_field"
    assert "completion" in str(ei.value)


def test_dispatch_rejects_none_value() -> None:
    """A None value (json null) must fail loud as well — non-str sentinel."""
    from Trainforge.generators._anthropic_provider import SynthesisProviderError

    async def agent_tool(**_kwargs: object) -> dict:
        return {
            "success": True,
            "outputs": {
                "prompt": None,
                "completion": "ok-prompt-text-of-sufficient-length",
            },
            "artifacts": [],
        }

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-none",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(SynthesisProviderError) as ei:
        provider.paraphrase_instruction(draft, chunk)
    assert ei.value.code == "empty_field"
    assert "prompt" in str(ei.value)


def test_paraphrase_instruction_rejects_short_prompt() -> None:
    """Wave 112 Task 4: a paraphrased prompt below PROMPT_MIN must fail
    loud rather than silently shipping a sub-floor pair into the cache."""
    from Trainforge.generators._anthropic_provider import (
        SynthesisProviderError, PROMPT_MIN,
    )

    short_prompt = "x" * 10  # well below PROMPT_MIN (40)
    long_completion = "yyy yyyy yyyy yyyy yyyy yyyy yyyy yyyy yyyy yyyy yyyy yyyy"
    assert len(short_prompt) < PROMPT_MIN

    async def agent_tool(**_kwargs: object) -> dict:
        return {
            "success": True,
            "outputs": {"prompt": short_prompt, "completion": long_completion},
            "artifacts": [],
        }

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-short-prompt",
    )
    draft = {"prompt": "p", "completion": "c"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(SynthesisProviderError) as ei:
        provider.paraphrase_instruction(draft, chunk)
    assert ei.value.code == "prompt_below_minimum"


def test_paraphrase_preference_rejects_short_chosen() -> None:
    """Analogous floor-check on the preference path's `chosen` arm."""
    from Trainforge.generators._anthropic_provider import (
        SynthesisProviderError, COMPLETION_MIN,
    )

    long_prompt = "z" * 60  # above PROMPT_MIN (40)
    short_chosen = "yy"  # below COMPLETION_MIN (50)
    long_rejected = "rrr rrrr rrrr rrrr rrrr rrrr rrrr rrrr rrrr rrrr rrrr rrrr"
    assert len(short_chosen) < COMPLETION_MIN

    async def agent_tool(**_kwargs: object) -> dict:
        return {
            "success": True,
            "outputs": {
                "prompt": long_prompt,
                "chosen": short_chosen,
                "rejected": long_rejected,
            },
            "artifacts": [],
        }

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        run_id="run-short-chosen",
    )
    draft = {"prompt": "p", "chosen": "c", "rejected": "r"}
    chunk = {"id": "c1", "text": "t"}

    with pytest.raises(SynthesisProviderError) as ei:
        provider.paraphrase_preference(draft, chunk)
    assert ei.value.code == "chosen_below_minimum"


def test_load_cache_rejects_poisoned_outputs(tmp_path: Path) -> None:
    """Wave 112 Task 7: a JSONL cache row with null/empty `outputs` must
    be rejected at load time rather than silently surviving and serving
    a poisoned string on the next cache hit."""
    from Trainforge.generators._anthropic_provider import SynthesisProviderError

    cache_path = tmp_path / "synthesis_cache.jsonl"
    valid_entry = {
        "key": "k_valid",
        "kind": "instruction",
        "chunk_id": "chunk_a",
        "provider_version": "v1",
        "outputs": {
            "prompt": _ok_prompt("valid"),
            "completion": _ok_completion("valid"),
        },
    }
    poisoned_entry = {
        "key": "k_poisoned",
        "kind": "instruction",
        "chunk_id": "chunk_b",
        "provider_version": "v1",
        "outputs": {
            "prompt": None,
            "completion": _ok_completion("ok"),
        },
    }
    cache_path.write_text(
        json.dumps(valid_entry, sort_keys=True) + "\n"
        + json.dumps(poisoned_entry, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    async def agent_tool(**_kwargs: object) -> dict:  # pragma: no cover
        raise AssertionError("dispatcher must not be hit during load")

    with pytest.raises(SynthesisProviderError) as ei:
        ClaudeSessionProvider(
            dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
            cache_path=cache_path,
        )
    # The error code should be the prompt-empty discriminator (None
    # value tripped the empty-field rail in _validate_lengths).
    assert ei.value.code in ("empty_field", "prompt_below_minimum")


def test_circuit_opens_after_repeated_dispatcher_failures(tmp_path: Path) -> None:
    """Three MAILBOX_TIMEOUT in a row trips the breaker; the 4th call
    raises SynthesisCircuitOpen WITHOUT contacting the dispatcher."""
    from Trainforge.generators._session_budget import SynthesisCircuitOpen

    async def agent_tool(**_kwargs: object) -> dict:
        return make_failure_response(error="timed out", error_code="MAILBOX_TIMEOUT")

    provider = ClaudeSessionProvider(
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        failures_to_open=3,
        failure_window_seconds=60.0,
    )
    chunk = {"id": "c1", "text": "x"}
    draft = {"prompt": "P", "completion": "C"}
    for _ in range(3):
        with pytest.raises(RuntimeError, match="MAILBOX_TIMEOUT"):
            provider.paraphrase_instruction(draft, chunk)
    with pytest.raises(SynthesisCircuitOpen):
        provider.paraphrase_instruction(draft, chunk)
