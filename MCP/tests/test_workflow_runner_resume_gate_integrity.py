"""Resume gate-integrity regression tests (Trap 1).

A phase whose validation gates FAILED must NOT be treated as
resumable-complete. A prior (buggy) run could persist a gate-failed
phase as ``_completed=True`` WITH ``_gates_passed=False`` (the stamp at
the bottom of the phase loop sets ``_completed=True`` unconditionally,
then the workflow breaks on the gate failure AFTER the state was already
saved). The resume skip-check must therefore require BOTH ``_completed``
AND gates-not-failed to skip — else ``--resume`` marches the workflow
downstream on the phase's bad output.

Backward-compat with old + benign persisted state:

  * ``_gates_passed`` absent  -> default True -> skip (pre-``_gates_passed``
    completed phases; optional-phase skip markers).
  * ``_gates_passed is True`` -> skip (normal happy resume).
  * ``_gates_passed is False`` -> DO NOT skip -> re-run the phase.

These tests drive ``run_workflow`` against a tmp on-disk workflow-state
file pre-populated with a completed phase, with a mocked
``executor.execute_phase`` so the only thing under test is the resume
skip-check branch.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import WorkflowRunner


def _result(task_id: str, status: str, payload: Dict[str, Any] | None = None) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=payload)


def _run_with_state(
    tmp_path,
    monkeypatch,
    *,
    phases: List[WorkflowPhase],
    initial_phase_outputs: Dict[str, Any],
    executed: List[str],
) -> Dict[str, Any]:
    """Drive ``run_workflow`` with pre-populated resume phase_outputs.

    ``executed`` is appended with each phase name the mocked executor is
    invoked for, so the test can assert re-run vs skip.
    """
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")

    workflow_id = "WF-TEST-RESUME"
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": "test_wf",
                "params": {},
                "phase_outputs": initial_phase_outputs,
                "tasks": [],
            }
        )
    )

    async def side_effect(*_a, **kw):  # noqa: ANN001
        executed.append(kw.get("phase_name") or (_a[1] if len(_a) > 1 else ""))
        results = {"T-1": _result("T-1", "COMPLETE", {"ok_key": "value"})}
        return results, True, []

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=side_effect)

    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test", phases=phases
    )

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(workflow_id))


def _load_persisted(tmp_path) -> Dict[str, Any]:
    state_path = tmp_path / "state" / "workflows" / "WF-TEST-RESUME.json"
    return json.loads(state_path.read_text()).get("phase_outputs", {})


_PHASE = WorkflowPhase(name="gated_phase", agents=["content-generator"])


def test_gate_failed_completed_phase_reruns_on_resume(tmp_path, monkeypatch) -> None:
    """_completed=True + _gates_passed=False -> phase RE-RUNS on resume."""
    executed: List[str] = []
    initial = {
        "gated_phase": {
            "_completed": True,
            "_gates_passed": False,
            "stale_key": "bad_output",
        }
    }
    out = _run_with_state(
        tmp_path, monkeypatch,
        phases=[_PHASE],
        initial_phase_outputs=initial,
        executed=executed,
    )
    assert "gated_phase" in executed, "gate-failed phase must re-run, not skip"
    persisted = _load_persisted(tmp_path)
    # Re-run refreshed the gate verdict.
    assert persisted["gated_phase"]["_gates_passed"] is True
    assert persisted["gated_phase"]["_completed"] is True


def test_gates_passed_completed_phase_skips_on_resume(tmp_path, monkeypatch) -> None:
    """_completed=True + _gates_passed=True -> phase is SKIPPED on resume."""
    executed: List[str] = []
    initial = {
        "gated_phase": {
            "_completed": True,
            "_gates_passed": True,
            "ok_key": "prior_value",
        }
    }
    _run_with_state(
        tmp_path, monkeypatch,
        phases=[_PHASE],
        initial_phase_outputs=initial,
        executed=executed,
    )
    assert "gated_phase" not in executed, "gates-passed phase must skip"


def test_legacy_completed_phase_without_gate_flag_skips(tmp_path, monkeypatch) -> None:
    """_completed=True, no _gates_passed key (legacy good state) -> SKIP."""
    executed: List[str] = []
    initial = {
        "gated_phase": {
            "_completed": True,
            "ok_key": "prior_value",
        }
    }
    _run_with_state(
        tmp_path, monkeypatch,
        phases=[_PHASE],
        initial_phase_outputs=initial,
        executed=executed,
    )
    assert "gated_phase" not in executed, (
        "legacy completed phase (no _gates_passed) must still skip"
    )


# ---------------------------------------------------------------------------
# Bug A — partial-artifact multi-task phase must NOT be stamped _completed.
#
# A multi-task phase (e.g. dart_conversion, one task per corpus PDF) where SOME
# tasks COMPLETE and others TIME OUT must persist ``_completed=False`` so a
# subsequent ``--resume`` re-runs it instead of skipping the whole phase (the
# live-fire trap: 1 of 10 PDFs converted, 9 timed out, phase was stamped
# _completed=True + _gates_passed=True and skipped whole on resume).
# ---------------------------------------------------------------------------
def _run_multitask_partial(
    tmp_path,
    monkeypatch,
    *,
    phase: WorkflowPhase,
    results: Dict[str, ExecutionResult],
    gates_passed: bool = True,
) -> Dict[str, Any]:
    """Drive ``run_workflow`` for a single multi-task phase with a fixed
    per-task result map (keyed arbitrarily; the runner only reads statuses)."""
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")

    workflow_id = "WF-TEST-RESUME"
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": "test_wf",
                "params": {},
                "phase_outputs": {},
                "tasks": [],
            }
        )
    )

    async def side_effect(*_a, **_kw):  # noqa: ANN001
        return results, gates_passed, []

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=side_effect)

    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test", phases=[phase]
    )

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(workflow_id))


def test_partial_multitask_phase_not_completed(tmp_path, monkeypatch) -> None:
    """1 COMPLETE + 1 TIMEOUT (of 2 dispatched) -> _completed=False + FAILED."""
    # Two agents => _create_phase_tasks dispatches two tasks (default path).
    phase = WorkflowPhase(name="multi_phase", agents=["a", "b"])
    results = {
        "T-a": _result("T-a", "COMPLETE", {"output_path": "/x/ch09.html"}),
        "T-b": _result("T-b", "TIMEOUT", None),
    }
    out = _run_multitask_partial(
        tmp_path, monkeypatch, phase=phase, results=results,
    )
    assert out["status"] == "FAILED"
    assert out["failed_phase"] == "multi_phase"
    persisted = _load_persisted(tmp_path)
    # Persisted as explicitly NOT completed so a later --resume re-runs it.
    assert persisted["multi_phase"]["_completed"] is False, (
        "a partially-completed multi-task phase must persist _completed=False "
        "so --resume re-runs it (Bug A partial-artifact resume trap)"
    )


def test_partial_completed_phase_reruns_on_resume(tmp_path, monkeypatch) -> None:
    """A phase persisted _completed=False (partial) RE-RUNS on resume."""
    executed: List[str] = []
    initial = {
        "gated_phase": {
            "_completed": False,
            "_gates_passed": True,
            "output_path": "/x/ch09.html",  # the one legit artifact
        }
    }
    _run_with_state(
        tmp_path, monkeypatch,
        phases=[_PHASE],
        initial_phase_outputs=initial,
        executed=executed,
    )
    assert "gated_phase" in executed, (
        "a phase persisted _completed=False must re-run on resume, not skip"
    )


def test_all_complete_multitask_phase_is_completed(tmp_path, monkeypatch) -> None:
    """Every dispatched task COMPLETE -> _completed=True (happy-path unchanged)."""
    phase = WorkflowPhase(name="multi_phase", agents=["a", "b"])
    results = {
        "T-a": _result("T-a", "COMPLETE", {"output_path": "/x/ch01.html"}),
        "T-b": _result("T-b", "COMPLETE", {"output_path": "/x/ch02.html"}),
    }
    out = _run_multitask_partial(
        tmp_path, monkeypatch, phase=phase, results=results,
    )
    assert out["status"] == "COMPLETE"
    persisted = _load_persisted(tmp_path)
    assert persisted["multi_phase"]["_completed"] is True
