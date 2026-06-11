"""Anti-zombie phase guard regression tests.

A phase that dispatched at least one task, had ZERO of those tasks
return a COMPLETE (success) result, AND produced no canonical output
keys must NOT be stamped ``_completed=True``. Pre-guard, such a phase
was persisted as ``_completed=True, keys=[]`` (the real motivating run:
a ``content_generation`` checkpoint with ``tasks_completed: []`` + failed
tasks). On ``--resume`` those empty-completed phases are skipped, which
starves every downstream phase that needs a canonical key (content
gates, ``imscc_chunking`` needing ``packaging.package_path``).

The guard lives in ``WorkflowRunner.run_workflow`` where ``_completed``
is stamped (right after ``_extract_phase_outputs``). It must be carefully
scoped so it does NOT fire on legitimate no-output phases:

  * Validator-only phases (``agents: []`` -> synthesised ``phase-handler``
    task) that PASS their gates emit a COMPLETE result with no canonical
    keys; keyed on a COMPLETE result existing, those still complete.
  * Optional phases (``optional: true``) are excluded outright.
  * A normal phase with COMPLETE task results completes.
  * A phase with successful tasks but a genuinely empty canonical-key set
    still completes (don't over-trigger).

These tests drive ``run_workflow`` end-to-end against a tmp on-disk
workflow-state file with a mocked ``executor.execute_phase`` so the only
thing under test is the guard's branch.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import WorkflowRunner


def _write_workflow_state(
    state_root, workflow_id: str, workflow_type: str = "test_wf"
) -> None:
    """Materialise the minimal on-disk workflow-state JSON run_workflow reads."""
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": workflow_type,
                "params": {},
                "phase_outputs": {},
                "tasks": [],
            }
        )
    )


def _result(task_id: str, status: str, payload: Dict[str, Any] | None = None) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=payload)


def _run(
    tmp_path,
    monkeypatch,
    *,
    phases: List[WorkflowPhase],
    execute_phase_side_effect,
) -> Dict[str, Any]:
    """Drive ``run_workflow`` with a mocked executor and a tmp STATE_PATH.

    ``execute_phase_side_effect`` is an async function (workflow_id,
    phase_name, ...) -> (results, gates_passed, gate_results).
    """
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)

    # These tests drive run_workflow with a MOCKED executor (no real
    # dispatch), so opt into the templated-stub path the Marketable-v1 A3
    # authoring-route guardrail honors for tests / dry runs.
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")

    workflow_id = "WF-TEST-0001"
    _write_workflow_state(state_root, workflow_id)

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=execute_phase_side_effect)

    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test", phases=phases
    )

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(workflow_id))


def _load_persisted_phase_outputs(tmp_path) -> Dict[str, Any]:
    state_path = tmp_path / "state" / "workflows" / "WF-TEST-0001.json"
    return json.loads(state_path.read_text()).get("phase_outputs", {})


# --------------------------------------------------------------------------
# (a) dispatched-but-all-failed + empty outputs -> NOT completed, workflow stops
# --------------------------------------------------------------------------
def test_all_tasks_failed_empty_outputs_not_completed_and_stops(
    tmp_path, monkeypatch, caplog
) -> None:
    # Use a real phase name with declared output keys, but return only
    # ERROR results carrying none of those keys -> extracted is empty.
    phases = [
        WorkflowPhase(name="content_generation", agents=["content-generator"]),
    ]

    async def side_effect(*_a, **kw):  # noqa: ANN001
        # One dispatched task, all failed, no canonical keys.
        results = {"T-1": _result("T-1", "ERROR", {"oops": "no canonical keys"})}
        return results, True, []

    with caplog.at_level("ERROR"):
        out = _run(
            tmp_path, monkeypatch, phases=phases,
            execute_phase_side_effect=side_effect,
        )

    assert out["status"] == "FAILED"
    persisted = _load_persisted_phase_outputs(tmp_path)
    assert persisted["content_generation"]["_completed"] is False
    # No canonical key leaked through.
    assert not [k for k in persisted["content_generation"] if not k.startswith("_")]
    assert any(
        "zero successful tasks and no" in rec.getMessage()
        for rec in caplog.records
    )


# --------------------------------------------------------------------------
# (b) validator-only phase passing gates with no output keys STILL completes
# --------------------------------------------------------------------------
def test_validator_only_phase_completes_with_no_output_keys(
    tmp_path, monkeypatch
) -> None:
    # agents: [] -> _create_phase_tasks synthesises a single phase-handler
    # task only for phases registered in _PHASE_TOOL_MAPPING.
    # inter_tier_validation IS so registered. Its phase-handler task
    # returns COMPLETE but emits no canonical keys (validator-only).
    phases = [
        WorkflowPhase(name="inter_tier_validation", agents=[]),
    ]

    async def side_effect(*_a, **kw):  # noqa: ANN001
        # The synthesised phase-handler task ran and succeeded but
        # surfaces no canonical output keys.
        results = {"T-ph": _result("T-ph", "COMPLETE", {"unrelated": 1})}
        return results, True, []

    out = _run(
        tmp_path, monkeypatch, phases=phases,
        execute_phase_side_effect=side_effect,
    )

    assert out["status"] == "COMPLETE"
    persisted = _load_persisted_phase_outputs(tmp_path)
    assert persisted["inter_tier_validation"]["_completed"] is True


# --------------------------------------------------------------------------
# (c) normal phase with COMPLETE task results completes
# --------------------------------------------------------------------------
def test_normal_phase_with_complete_results_completes(
    tmp_path, monkeypatch
) -> None:
    phases = [
        WorkflowPhase(name="packaging", agents=["brightspace-packager"]),
    ]

    async def side_effect(*_a, **kw):  # noqa: ANN001
        results = {
            "T-1": _result(
                "T-1", "COMPLETE",
                {"package_path": "/tmp/course.imscc", "project_id": "P1"},
            )
        }
        return results, True, []

    out = _run(
        tmp_path, monkeypatch, phases=phases,
        execute_phase_side_effect=side_effect,
    )

    assert out["status"] == "COMPLETE"
    persisted = _load_persisted_phase_outputs(tmp_path)
    assert persisted["packaging"]["_completed"] is True
    assert persisted["packaging"]["package_path"] == "/tmp/course.imscc"


# --------------------------------------------------------------------------
# (d) optional phase unaffected (all-failed + empty -> still NOT a hard stop)
# --------------------------------------------------------------------------
def test_optional_phase_unaffected_by_guard(tmp_path, monkeypatch) -> None:
    phases = [
        WorkflowPhase(
            name="training_synthesis",
            agents=["training-synthesizer"],
            optional=True,
        ),
    ]

    async def side_effect(*_a, **kw):  # noqa: ANN001
        # All tasks failed, no outputs -- but the phase is optional, so
        # the guard must NOT trip and must NOT stop the workflow. The
        # existing optional-phase fail path also lets the workflow finish.
        results = {"T-1": _result("T-1", "ERROR", {"oops": 1})}
        return results, True, []

    out = _run(
        tmp_path, monkeypatch, phases=phases,
        execute_phase_side_effect=side_effect,
    )

    # Optional phase failure does not fail the workflow.
    assert out["status"] == "COMPLETE"
    persisted = _load_persisted_phase_outputs(tmp_path)
    # The guard did NOT force _completed=False; the normal path stamped True.
    assert persisted["training_synthesis"]["_completed"] is True


# --------------------------------------------------------------------------
# (e) successful tasks but genuinely empty canonical keys still completes
# --------------------------------------------------------------------------
def test_successful_task_empty_canonical_keys_still_completes(
    tmp_path, monkeypatch
) -> None:
    # Real phase with declared output keys, COMPLETE result that simply
    # does not carry any of them -> extracted is empty BUT a task
    # succeeded, so the guard must NOT trip (keyed on success, not keys).
    phases = [
        WorkflowPhase(name="content_generation", agents=["content-generator"]),
    ]

    async def side_effect(*_a, **kw):  # noqa: ANN001
        results = {"T-1": _result("T-1", "COMPLETE", {"unrelated": "value"})}
        return results, True, []

    out = _run(
        tmp_path, monkeypatch, phases=phases,
        execute_phase_side_effect=side_effect,
    )

    assert out["status"] == "COMPLETE"
    persisted = _load_persisted_phase_outputs(tmp_path)
    assert persisted["content_generation"]["_completed"] is True
