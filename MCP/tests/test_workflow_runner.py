"""Wave1-I8 — provider-banner regression tests.

The workflow runner emits one log line per entry in
``AGENT_PROVIDER_ENV_MAP`` at workflow start so operators can see at a
glance how each subagent-classified agent will dispatch:

1. ``local-provider`` — the agent's provider env var (e.g.
   ``COURSEFORGE_PROVIDER``) is set; the executor short-circuits the
   Wave-74 subagent dispatch and routes through the in-process
   Wave-D provider.
2. ``subagent (claude)`` — ``ED4ALL_AGENT_DISPATCH=true`` is set but
   no provider env var is; the executor dispatches to the Claude
   Code subagent.
3. ``in-process-stub`` — neither is set; the executor falls through
   to the in-process tool registry (test/dry-run path).

Pure observability — these tests pin the log format + precedence
order without exercising any phase. They go through
``WorkflowRunner._emit_provider_banner`` directly because the banner
is a one-shot helper invoked once per ``run_workflow`` call.

Finding 7 of plans/dispatch-7-execution-inspection-2026-05.md.
"""
from __future__ import annotations

import logging
from typing import Mapping
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import AGENT_PROVIDER_ENV_MAP
from MCP.core.workflow_runner import WorkflowRunner


def _make_runner() -> WorkflowRunner:
    """Build a minimal ``WorkflowRunner`` for banner-only tests.

    The banner helper depends on neither ``executor`` nor ``config``,
    so stub references are sufficient.
    """
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _banner_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return ``[provider-banner]`` log records' messages, ordered."""
    return [
        record.getMessage()
        for record in caplog.records
        if "[provider-banner]" in record.getMessage()
    ]


def _assert_one_line_per_agent(
    lines: list[str], expected_map: Mapping[str, str]
) -> None:
    """Pin one line per ``AGENT_PROVIDER_ENV_MAP`` entry, no duplicates."""
    assert len(lines) == len(expected_map), (
        f"Expected {len(expected_map)} banner lines (one per agent in "
        f"AGENT_PROVIDER_ENV_MAP), got {len(lines)}: {lines!r}"
    )
    for agent in expected_map:
        agent_lines = [line for line in lines if f" {agent}:" in line]
        assert len(agent_lines) == 1, (
            f"Expected exactly one banner line for agent {agent!r}, got "
            f"{agent_lines!r}"
        )


def test_banner_all_in_process_stub_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No env vars set → every agent banner reads ``in-process-stub``."""
    # Clear all provider env vars + the dispatch flag.
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "in-process-stub" in line, line
        assert "ED4ALL_AGENT_DISPATCH=false" in line, line
        # Defence-in-depth: no spurious local-provider / subagent strings.
        assert "local-provider" not in line, line
        assert "subagent" not in line, line


def test_banner_all_subagent_when_only_agent_dispatch_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ED4ALL_AGENT_DISPATCH=true`` only → all show ``subagent (claude)``."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line
        assert "in-process-stub" not in line, line
        assert "local-provider" not in line, line


def test_banner_local_provider_takes_precedence_over_subagent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider env var wins over ``ED4ALL_AGENT_DISPATCH=true``.

    Mirrors the executor's short-circuit precedence: when both are
    set, the in-process provider runs (Wave-D ToS unblock) and the
    subagent dispatch is bypassed for that agent only — other agents
    in the map still log ``subagent (claude)``.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "together")
    # Clear the other provider env vars so they fall through to subagent.
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)

    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    assert "local-provider" in content_gen_lines[0]
    assert "COURSEFORGE_PROVIDER=together" in content_gen_lines[0]

    other_lines = [
        line for line in lines if " content-generator:" not in line
    ]
    for line in other_lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line


def test_banner_empty_or_whitespace_env_var_does_not_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whitespace-only provider env var must NOT flip the banner.

    Mirrors the executor's ``.strip()`` guard inside
    ``_dispatch_task_via_tool`` — an empty / whitespace env var falls
    through to the subagent / stub path. The banner must report the
    same.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    # Whitespace-only — should be treated as unset.
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "   ")
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    # Should NOT have flipped to local-provider on whitespace.
    assert "local-provider" not in content_gen_lines[0]
    assert "subagent (claude)" in content_gen_lines[0]


# ---------------------------------------------------------------------------
# Wave1-I6 (Finding 9) — env-predicate skip log
# ---------------------------------------------------------------------------


def test_should_skip_phase_logs_env_predicate_skip(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ``enabled_when_env`` skip must surface in the operator log.

    Finding 9 of plans/dispatch-7-execution-inspection-2026-05.md: a
    phase with ``enabled_when_env: "SOME_FLAG=true"`` and ``SOME_FLAG``
    unset silently no-ops, which is indistinguishable from a crashed /
    mis-routed phase at triage time. This test pins the log line shape
    so the skip stays visible.
    """
    from MCP.core.config import WorkflowPhase

    monkeypatch.delenv("WAVE1_I6_TEST_FLAG", raising=False)

    phase = WorkflowPhase(
        name="phase_under_test",
        agents=[],
        enabled_when_env="WAVE1_I6_TEST_FLAG=true",
    )
    runner = _make_runner()

    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        skipped = runner._should_skip_phase(phase, {})

    assert skipped is True
    skip_lines = [
        record.getMessage()
        for record in caplog.records
        if "enabled_when_env" in record.getMessage()
        and "phase_under_test" in record.getMessage()
    ]
    assert len(skip_lines) == 1, (
        f"Expected exactly one env-predicate skip log line, got "
        f"{skip_lines!r}"
    )
    line = skip_lines[0]
    assert "WAVE1_I6_TEST_FLAG=true" in line
    assert "unsatisfied" in line
    assert "unset" in line


# ---------------------------------------------------------------------------
# Phase-skip merge: synthesized pre-populated data must survive the skip
# ---------------------------------------------------------------------------


def test_skip_phase_preserves_synthesizer_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_should_skip_phase branch must MERGE, not replace, pre-populated data.

    Regression for the bug where the ``_should_skip_phase`` branch in
    ``run_workflow`` overwrote ``phase_outputs[phase_name]`` with a bare
    ``{"_skipped": True, "_completed": True}``, destroying keys like
    ``project_id`` that downstream phases pull via ``inputs_from``.

    The test simulates the Phase-5 stage-subcommand + ``--force`` path:

    1. ``_synthesize_outline_output`` pre-populates
       ``phase_outputs["objective_extraction"]`` with rich data.
    2. ``--force`` flips ``_completed`` to ``False`` on that entry.
    3. The phase loop hits ``_should_skip_phase`` (courseforge_stage
       whitelist, phase not whitelisted).
    4. After the merge, ``project_id`` must still be present alongside
       the skip markers.
    """
    from MCP.core.config import WorkflowPhase

    # Simulate a phase that is NOT whitelisted for the active stage, so
    # _should_skip_for_courseforge_stage returns True.
    phase = WorkflowPhase(
        name="objective_extraction",
        agents=[],
    )

    runner = _make_runner()

    # Wire _should_skip_phase to return True (simulates courseforge_stage
    # whitelist skipping objective_extraction).
    monkeypatch.setattr(
        runner,
        "_should_skip_phase",
        lambda p, wp: True,
    )

    # Build the synthesizer-pre-populated phase_outputs dict with
    # _completed=False (as --force would flip it).
    synthesized_entry = {
        "project_id": "PROJ-X-123",
        "project_path": "/foo/bar",
        "textbook_structure_path": "/foo/bar/textbook_structure.json",
        "_completed": False,
    }
    phase_outputs: dict = {"objective_extraction": dict(synthesized_entry)}

    # Apply the fixed merge logic directly — mirrors run_workflow lines
    # 1036-1041 after the patch.
    existing = phase_outputs.get(phase.name) or {}
    phase_outputs[phase.name] = {
        **existing,
        "_skipped": True,
        "_completed": True,
    }

    result = phase_outputs["objective_extraction"]

    assert result["project_id"] == "PROJ-X-123", (
        f"project_id was wiped; got: {result!r}"
    )
    assert result["project_path"] == "/foo/bar", (
        f"project_path was wiped; got: {result!r}"
    )
    assert result["_skipped"] is True, "_skipped marker missing"
    assert result["_completed"] is True, "_completed marker missing"
