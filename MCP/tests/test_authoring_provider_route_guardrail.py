"""Tests for the Marketable-v1 A3 blessed authoring-provider route guardrail.

``WorkflowRunner._enforce_authoring_provider_route`` fails fast (instead of
hanging on an unserviced mailbox or silently degrading to a templated stub)
when an LLM-needing phase in the workflow would not resolve its generation
through the in-process provider lattice. These tests drive the guardrail
directly with synthetic ``WorkflowPhase`` lists — no orchestrator, no LLM,
no live providers.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from MCP.core.config import WorkflowPhase
from MCP.core.executor import AGENT_AUTHORING_PROVIDER_ENV_MAP
from MCP.core.workflow_runner import (
    AuthoringProviderRouteError,
    WorkflowRunner,
)

# Every env var that can flip a dispatch decision — cleared per test so the
# guardrail sees a known baseline regardless of the ambient environment.
_ALL_DECISION_ENVS = (
    "ED4ALL_AGENT_DISPATCH",
    "ED4ALL_MAILBOX_SERVICED",
    "LOCAL_DISPATCHER_ALLOW_STUB",
    "LLM_PROVIDER",
    "COURSEFORGE_TWO_PASS",
    "COURSEFORGE_BLOCK_ROUTING_PATH",
    *AGENT_AUTHORING_PROVIDER_ENV_MAP.values(),
)


def _make_runner() -> WorkflowRunner:
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _clear_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _ALL_DECISION_ENVS:
        monkeypatch.delenv(env_var, raising=False)


def _phase(name: str, agents: List[str], **kw) -> WorkflowPhase:
    return WorkflowPhase(name=name, agents=list(agents), **kw)


# A representative slice of the textbook_to_course LLM-needing phases.
_LLM_PHASES = [
    _phase("course_planning", ["course-outliner"]),
    _phase("content_generation", ["content-generator"]),
]
# Deterministic phases (no LLM) — must never trip the guardrail.
_DETERMINISTIC_PHASES = [
    _phase("staging", ["textbook-stager"]),
    _phase("source_mapping", ["source-router"]),
    _phase("packaging", ["brightspace-packager"]),
]


def test_guardrail_blocks_unserviced_run(monkeypatch: pytest.MonkeyPatch):
    """No provider env + no session/stub opt-in -> fail fast, not hang."""
    _clear_envs(monkeypatch)
    runner = _make_runner()
    with pytest.raises(AuthoringProviderRouteError) as exc:
        runner._enforce_authoring_provider_route(_LLM_PHASES, {})
    msg = str(exc.value)
    # Names the offending agents AND the actionable fix.
    assert "course-outliner" in msg
    assert "content-generator" in msg
    assert "COURSEPLANNER_PROVIDER" in msg
    assert "COURSEFORGE_PROVIDER" in msg


def test_guardrail_blocks_agent_dispatch_without_servicer(
    monkeypatch: pytest.MonkeyPatch,
):
    """ED4ALL_AGENT_DISPATCH=true but no servicer -> still fails (would hang)."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    runner = _make_runner()
    with pytest.raises(AuthoringProviderRouteError):
        runner._enforce_authoring_provider_route(_LLM_PHASES, {})


def test_guardrail_passes_with_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Each LLM agent's provider env set -> blessed lattice route, no raise."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "local")
    runner = _make_runner()
    # Does not raise.
    runner._enforce_authoring_provider_route(_LLM_PHASES, {})


def test_guardrail_passes_with_session_servicer(monkeypatch: pytest.MonkeyPatch):
    """ED4ALL_AGENT_DISPATCH=true + servicer opt-in -> session route allowed."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("ED4ALL_MAILBOX_SERVICED", "1")
    runner = _make_runner()
    runner._enforce_authoring_provider_route(_LLM_PHASES, {})


def test_guardrail_passes_with_stub_opt_in(monkeypatch: pytest.MonkeyPatch):
    """LOCAL_DISPATCHER_ALLOW_STUB=1 -> templated-stub opt-in allowed."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    runner = _make_runner()
    runner._enforce_authoring_provider_route(_LLM_PHASES, {})


def test_guardrail_ignores_deterministic_phases(monkeypatch: pytest.MonkeyPatch):
    """A workflow with no LLM-needing agent never trips the guardrail."""
    _clear_envs(monkeypatch)
    runner = _make_runner()
    runner._enforce_authoring_provider_route(_DETERMINISTIC_PHASES, {})


def test_guardrail_partial_provider_env_still_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """One provider env set, the other missing -> still blocks on the gap."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "local")
    # COURSEPLANNER_PROVIDER deliberately unset.
    runner = _make_runner()
    with pytest.raises(AuthoringProviderRouteError) as exc:
        runner._enforce_authoring_provider_route(_LLM_PHASES, {})
    msg = str(exc.value)
    assert "course-outliner" in msg
    # The satisfied agent is not listed as an offender.
    assert "content-generator" not in msg


def test_guardrail_skips_optional_skipped_phase(monkeypatch: pytest.MonkeyPatch):
    """An optional LLM phase that will be skipped is not an offender.

    ``trainforge_assessment`` is optional and skipped via
    ``generate_assessments=False``; the guardrail must use the same
    skip predicate the run loop uses so it never blocks on a phase the
    workflow won't run.
    """
    _clear_envs(monkeypatch)
    phases = [
        _phase("staging", ["textbook-stager"]),
        _phase(
            "trainforge_assessment",
            ["assessment-generator"],
            optional=True,
        ),
    ]
    runner = _make_runner()
    # generate_assessments=False -> _should_skip_phase returns True for the
    # optional assessment phase, so its LLM agent is excluded.
    runner._enforce_authoring_provider_route(
        phases, {"generate_assessments": False}
    )


# ---------------------------------------------------------------------------
# T2 (W1) — end-to-end single-switch parity with the guardrail
# ---------------------------------------------------------------------------


def test_single_switch_fill_satisfies_guardrail(monkeypatch: pytest.MonkeyPatch):
    """The full run path: corpus-generalization defaults THEN the W1
    authoring-route fill THEN the guardrail -> NO AuthoringProviderRouteError.

    Proves a bare ``--provider local`` (resolved here via ``LLM_PROVIDER=local``)
    satisfies the blessed-route guardrail with no by-hand env exports.
    """
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    runner = _make_runner()

    runner._apply_corpus_generalization_defaults("textbook_to_course")
    runner._apply_authoring_route_env("textbook_to_course", "")

    # Does not raise — every LLM-needing agent now resolves the lattice.
    runner._enforce_authoring_provider_route(_LLM_PHASES, {})


def test_without_authoring_route_fill_guardrail_still_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    """Negative control: skip the W1 fill and the guardrail DOES raise.

    Proves the new fill is load-bearing — without it, a bare
    ``--provider local`` run hard-fails exactly as it does today, because the
    corpus-generalization defaults fill only TRAINFORGE_SYNTHESIS_PROVIDER, not
    COURSEFORGE_PROVIDER / COURSEPLANNER_PROVIDER.
    """
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    runner = _make_runner()

    # Corpus-generalization defaults alone do NOT fill the content/outliner
    # authoring envs.
    runner._apply_corpus_generalization_defaults("textbook_to_course")

    with pytest.raises(AuthoringProviderRouteError) as exc:
        runner._enforce_authoring_provider_route(_LLM_PHASES, {})
    msg = str(exc.value)
    assert "course-outliner" in msg
    assert "content-generator" in msg
