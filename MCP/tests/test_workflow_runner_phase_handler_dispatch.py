"""Phase 4 Subtask 1 — virtual phase-handler task synthesis.

The Phase 3.5 review surfaced a HIGH-severity bug: validator-only
phases (``agents: []``) never invoked their ``_PHASE_TOOL_MAPPING``
handler because :meth:`WorkflowRunner._create_phase_tasks` only
iterated over ``phase.agents``. Empty-agents phases yielded zero
tasks; the executor's per-task tool-routing layer was therefore
never consulted. Validation gates fired via
``execute_phase``'s ``gate_configs``, but the per-phase
blocks-emit-and-persist work that the four phase handlers do
(e.g. ``_run_post_rewrite_validation`` writing
``blocks_validated_path.jsonl``) never landed in an end-to-end run.

Subtask 1 fixed this by appending one synthetic task with
``agent_type='phase-handler'`` whenever ``phase.agents`` is empty
AND ``_PHASE_TOOL_MAPPING.get(phase.name)`` is registered. These
tests pin that fix.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from MCP.core.config import WorkflowPhase
from MCP.core.executor import _PHASE_TOOL_MAPPING
from MCP.core.workflow_runner import WorkflowRunner


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    """Minimal WorkflowRunner — these tests exercise pure helpers."""
    return WorkflowRunner(executor=object(), config=object())


# ---------------------------------------------------------------------------
# Subtask 3 — unit-level pins on _create_phase_tasks
# ---------------------------------------------------------------------------


def test_create_phase_tasks_synthesizes_virtual_task_for_agents_empty_phase(
    runner_stub: WorkflowRunner,
) -> None:
    """Phase with ``agents: []`` AND a registered handler -> exactly one synthetic task.

    Pins the Phase 4 Subtask 1 contract: ``inter_tier_validation`` is
    in ``_PHASE_TOOL_MAPPING`` (executor.py:206), so the runner must
    synthesize a single virtual task whose phase-name dispatch will
    route through ``run_inter_tier_validation`` regardless of the
    placeholder ``agent_type``.
    """
    phase = WorkflowPhase(name="inter_tier_validation", agents=[])
    routed_params = {
        "blocks_outline_path": "/tmp/x/blocks_outline.jsonl",
        "project_id": "PROJ-TEST-1",
    }

    tasks = runner_stub._create_phase_tasks(
        workflow_id="wf-test",
        phase=phase,
        routed_params=routed_params,
    )

    assert len(tasks) == 1, (
        "agents:[] phase with registered handler must yield exactly "
        "one synthetic task"
    )
    task = tasks[0]
    assert task["phase"] == "inter_tier_validation"
    assert task["agent_type"] == "phase-handler", (
        "agent_type must be the placeholder 'phase-handler' — the "
        "executor routes via _PHASE_TOOL_MAPPING.get(phase.name), not "
        "by agent name, on this code path"
    )
    assert task["status"] == "PENDING"
    assert task["params"] == routed_params, (
        "routed_params must reach the synthetic task so inputs_from "
        "resolution still flows into the phase handler"
    )
    # Defensive copy — mutations to the original dict must not bleed.
    assert task["params"] is not routed_params
    assert task["id"].startswith("T-inter_tier_validation-phase-handler-")
    assert task["dependencies"] == []


def test_create_phase_tasks_returns_empty_when_no_agents_and_no_phase_handler(
    runner_stub: WorkflowRunner,
) -> None:
    """No agents AND no registered handler -> still ``[]`` (genuinely no-op).

    Subtask 1's fallback only fires when ``_PHASE_TOOL_MAPPING.get``
    returns a tool name. Phases like ``post_training_validation``
    (validator-only, no per-phase handler beyond the gate chain)
    must continue to yield zero tasks so ``execute_phase`` runs the
    gate chain alone.
    """
    # Sanity: this phase name is NOT in _PHASE_TOOL_MAPPING.
    assert "no_handler_phase_xyz" not in _PHASE_TOOL_MAPPING

    phase = WorkflowPhase(name="no_handler_phase_xyz", agents=[])
    tasks = runner_stub._create_phase_tasks(
        workflow_id="wf-test",
        phase=phase,
        routed_params={},
    )

    assert tasks == [], (
        "phases with no agents AND no registered handler must remain "
        "no-op so the validator-only gate path is undisturbed"
    )


def test_create_phase_tasks_returns_per_agent_tasks_when_agents_listed(
    runner_stub: WorkflowRunner,
) -> None:
    """Agents-listed phase -> per-agent tasks; the synthetic fallback must NOT fire.

    ``content_generation_outline`` IS in ``_PHASE_TOOL_MAPPING`` AND
    declares ``agents: [content-generator]``. The fallback's
    ``if not tasks`` guard must skip task synthesis here so we don't
    duplicate work or shadow the per-agent dispatch.
    """
    # Sanity: this phase IS in _PHASE_TOOL_MAPPING.
    assert "content_generation_outline" in _PHASE_TOOL_MAPPING

    phase = WorkflowPhase(
        name="content_generation_outline",
        agents=["content-generator"],
    )
    tasks = runner_stub._create_phase_tasks(
        workflow_id="wf-test",
        phase=phase,
        routed_params={"project_id": "PROJ-1"},
    )

    assert len(tasks) == 1
    assert tasks[0]["agent_type"] == "content-generator", (
        "agents-listed phase must produce per-agent tasks; the "
        "Subtask 1 phase-handler fallback must NOT short-circuit them"
    )
    assert tasks[0]["agent_type"] != "phase-handler"


def test_workflow_runner_dispatches_phase_handler_via_synthetic_task(
    runner_stub: WorkflowRunner,
    tmp_path: Path,
) -> None:
    """End-to-end pin: synthetic task reaches the executor with the right phase metadata.

    Subtask 1's contract is that the synthesized task carries the
    correct ``phase`` field so the executor's
    ``_PHASE_TOOL_MAPPING.get(phase_name)`` lookup (executor.py:588)
    resolves to the phase-handler tool name. We don't run the full
    workflow loop here (that's Subtask 4's integration test); instead
    we assert the task shape that Subtask 1 emits maps onto the
    executor's expected dispatch surface.
    """
    phase = WorkflowPhase(name="post_rewrite_validation", agents=[])
    routed_params = {
        "blocks_final_path": str(tmp_path / "blocks_final.jsonl"),
        "project_id": "PROJ-DISPATCH-TEST",
    }

    tasks = runner_stub._create_phase_tasks(
        workflow_id="wf-dispatch",
        phase=phase,
        routed_params=routed_params,
    )

    assert len(tasks) == 1
    task = tasks[0]

    # Mirror executor._execute_task's tool-name lookup. This is the
    # exact line the Phase 4 Subtask 1 fix was written to unblock:
    # without a synthetic task, the executor never gets called, and
    # this lookup never runs.
    phase_name = task["phase"]
    expected_tool = _PHASE_TOOL_MAPPING.get(phase_name)
    assert expected_tool == "run_post_rewrite_validation", (
        "phase name on synthetic task must resolve through "
        "_PHASE_TOOL_MAPPING to the dedicated handler"
    )

    # The placeholder agent_type must NOT also appear in the agent
    # mapping — that would risk silent mis-routing if the executor's
    # phase-name dispatch ever regressed and fell through to the
    # agent-based path.
    from MCP.core.executor import AGENT_TOOL_MAPPING
    assert "phase-handler" not in AGENT_TOOL_MAPPING, (
        "the placeholder agent_type 'phase-handler' must remain "
        "unregistered in AGENT_TOOL_MAPPING so it depends entirely "
        "on _PHASE_TOOL_MAPPING for dispatch"
    )


def test_create_phase_tasks_synthesizes_for_all_registered_handlers(
    runner_stub: WorkflowRunner,
) -> None:
    """Every phase in ``_PHASE_TOOL_MAPPING`` is reachable via the synthetic-task path.

    Defensive sweep: confirm Subtask 1's fallback fires for every
    phase that the dispatch map currently knows about, so future
    additions to ``_PHASE_TOOL_MAPPING`` don't silently regress
    when wired with ``agents: []``.
    """
    for phase_name in _PHASE_TOOL_MAPPING:
        phase = WorkflowPhase(name=phase_name, agents=[])
        tasks = runner_stub._create_phase_tasks(
            workflow_id="wf-sweep",
            phase=phase,
            routed_params={"sentinel": phase_name},
        )
        assert len(tasks) == 1, f"Synthesis must fire for {phase_name}"
        assert tasks[0]["phase"] == phase_name
        assert tasks[0]["params"]["sentinel"] == phase_name
