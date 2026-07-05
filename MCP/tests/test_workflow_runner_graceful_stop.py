"""Graceful-stop ("checkpoint on command") run-loop regression tests.

The workflow run loop cooperates with the ``lib.generation.stop_control``
filesystem sentinel so any run can be halted to a PAUSED status (never FAILED)
at a phase boundary, then resumed. This suite pins the runner-owned half of the
contract (P1/P2 of the graceful-stop plan):

  * (a) between-phase probe: a stop requested mid-run pauses BEFORE the next
    phase dispatches — downstream phases never run, status = PAUSED, an explicit
    GPU lease sweep fires.
  * (b) mid-phase pause: a phase that comes back with PAUSED task results is
    persisted ``_completed=False, _paused=True`` (NOT stamped FAILED by the
    anti-zombie / Bug-A guards) and halts the workflow.
  * ``--resume`` re-enters the paused phase (the ``_completed=False`` phase is
    not skipped by the completed-phase guard).
  * (e) a STALE run-scoped sentinel is cleared at launch, so a fresh/resume run
    does not immediately pause on the first phase boundary.
  * (f) a global STOP_ALL sentinel refuses the run outright with a loud error.
  * the PipelineOrchestrator status collapse maps PAUSED -> "paused".

Harness mirrors ``test_workflow_runner_gpu_lifecycle.py``: a tmp on-disk
workflow-state file + a mocked ``executor.execute_phase``. Sentinels are
isolated via ``ED4ALL_STATE_RUNS_DIR`` (read at call time by
``lib.paths.get_state_runs_dir``); no course slugs/paths, stdlib-only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.generation import stop_control
from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import WorkflowRunner


WORKFLOW_ID = "WF-STOP-0001"
RUN_ID = "RUN-STOP-0001"
PHASE_ONE = "inter_tier_validation"
PHASE_TWO = "post_rewrite_validation"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _write_workflow_state(
    state_root: Path,
    *,
    workflow_id: str = WORKFLOW_ID,
    phase_outputs: Optional[Dict[str, Any]] = None,
) -> None:
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": "test_wf",
                "params": {},
                "phase_outputs": phase_outputs or {},
                "tasks": [],
            }
        )
    )


def _result(task_id: str, status: str) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=None)


def _two_phase_config() -> WorkflowConfig:
    phases: List[WorkflowPhase] = [
        WorkflowPhase(name=PHASE_ONE, agents=[]),
        WorkflowPhase(name=PHASE_TWO, agents=[], depends_on=[PHASE_ONE]),
    ]
    return WorkflowConfig(description="test", phases=phases)


def _make_executor(*, run_id: str = RUN_ID) -> MagicMock:
    """Executor whose every phase COMPLETEs (overridable per test)."""
    executor = MagicMock()
    executor.run_id = run_id

    async def side_effect(*_a, **kw):
        results = {"T-ph": _result("T-ph", "COMPLETE")}
        return results, True, []

    executor.execute_phase = AsyncMock(side_effect=side_effect)
    return executor


def _isolate(tmp_path: Path, monkeypatch) -> Path:
    """Point STATE_PATH + the sentinel root at a tmp dir; return state_root."""
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(state_root / "runs"))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    monkeypatch.delenv("ED4ALL_VRAM_DOCTOR", raising=False)
    return state_root


def _run(executor: MagicMock, *, workflow_id: str = WORKFLOW_ID) -> Dict[str, Any]:
    config = MagicMock()
    config.get_workflow.return_value = _two_phase_config()
    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(workflow_id))


def _load_state(state_root: Path, workflow_id: str = WORKFLOW_ID) -> Dict[str, Any]:
    return json.loads(
        (state_root / "workflows" / f"{workflow_id}.json").read_text()
    )


# --------------------------------------------------------------------------
# (b) mid-phase pause: PAUSED task result -> paused workflow, not FAILED
# --------------------------------------------------------------------------
def test_mid_phase_pause_result_pauses_workflow(tmp_path, monkeypatch) -> None:
    state_root = _isolate(tmp_path, monkeypatch)
    _write_workflow_state(state_root)

    sweep = AsyncMock(name="_gpu_lifecycle_sweep")
    monkeypatch.setattr(WorkflowRunner, "_gpu_lifecycle_sweep", sweep)

    executor = _make_executor()

    async def paused_first_phase(*_a, **_kw):
        # The first (and only reached) phase comes back with a PAUSED task
        # result — the executor's GracefulStopRequested -> PAUSED mapping.
        return {"T-ph": _result("T-ph", "PAUSED")}, True, []

    executor.execute_phase = AsyncMock(side_effect=paused_first_phase)

    out = _run(executor)

    assert out["status"] == "PAUSED"
    assert out["paused_phase"] == PHASE_ONE
    # Downstream phase never dispatched (single execute_phase call).
    assert executor.execute_phase.await_count == 1
    # Persisted resumable: _completed False, _paused True.
    state = _load_state(state_root)
    assert state["status"] == "PAUSED"
    assert state["paused_phase"] == PHASE_ONE
    ph1 = state["phase_outputs"][PHASE_ONE]
    assert ph1["_completed"] is False
    assert ph1["_paused"] is True
    # Explicit GPU lease sweep fired for the paused phase.
    assert PHASE_ONE in [c.args[0] for c in sweep.await_args_list]


# --------------------------------------------------------------------------
# (a) between-phase sentinel: pause BEFORE the next phase dispatches
# --------------------------------------------------------------------------
def test_between_phase_sentinel_pauses_before_next(tmp_path, monkeypatch) -> None:
    state_root = _isolate(tmp_path, monkeypatch)
    _write_workflow_state(state_root)

    sweep = AsyncMock(name="_gpu_lifecycle_sweep")
    monkeypatch.setattr(WorkflowRunner, "_gpu_lifecycle_sweep", sweep)

    executor = _make_executor()
    dispatched: List[str] = []

    async def arm_after_first(*_a, **kw):
        pname = kw.get("phase_name")
        dispatched.append(pname)
        # After the first phase completes, an operator requests a stop; the
        # loop-top probe must trip before phase two dispatches.
        if pname == PHASE_ONE:
            stop_control.request_stop(RUN_ID, scope="run", source="test")
        return {"T-ph": _result("T-ph", "COMPLETE")}, True, []

    executor.execute_phase = AsyncMock(side_effect=arm_after_first)

    out = _run(executor)

    assert out["status"] == "PAUSED"
    assert out["paused_phase"] == PHASE_TWO
    # Only phase one ran; phase two was never dispatched.
    assert dispatched == [PHASE_ONE]
    assert executor.execute_phase.await_count == 1
    state = _load_state(state_root)
    assert PHASE_TWO not in state["phase_outputs"]
    # Pause-path sweep fired for the not-dispatched downstream phase.
    assert PHASE_TWO in [c.args[0] for c in sweep.await_args_list]


# --------------------------------------------------------------------------
# --resume re-enters the paused phase (not skipped by the _completed guard)
# --------------------------------------------------------------------------
def test_resume_reenters_paused_phase(tmp_path, monkeypatch) -> None:
    state_root = _isolate(tmp_path, monkeypatch)
    # Prior run left phase one paused: _completed False, _paused True.
    _write_workflow_state(
        state_root,
        phase_outputs={PHASE_ONE: {"_completed": False, "_paused": True}},
    )

    monkeypatch.setattr(
        WorkflowRunner, "_gpu_lifecycle_sweep", AsyncMock(name="sweep")
    )

    executor = _make_executor()
    dispatched: List[str] = []

    async def record(*_a, **kw):
        dispatched.append(kw.get("phase_name"))
        return {"T-ph": _result("T-ph", "COMPLETE")}, True, []

    executor.execute_phase = AsyncMock(side_effect=record)

    out = _run(executor)

    assert out["status"] == "COMPLETE"
    # The paused (_completed=False) phase is re-dispatched, not skipped.
    assert dispatched == [PHASE_ONE, PHASE_TWO]


# --------------------------------------------------------------------------
# (e) a stale run-scoped sentinel is cleared at launch
# --------------------------------------------------------------------------
def test_stale_run_sentinel_cleared_at_launch(tmp_path, monkeypatch) -> None:
    state_root = _isolate(tmp_path, monkeypatch)
    _write_workflow_state(state_root)

    monkeypatch.setattr(
        WorkflowRunner, "_gpu_lifecycle_sweep", AsyncMock(name="sweep")
    )

    # A leftover run-scoped sentinel from a prior attempt under the same run_id.
    sentinel = stop_control.request_stop(RUN_ID, scope="run", source="stale")
    assert sentinel is not None and sentinel.exists()

    out = _run(_make_executor())

    # Launch cleared the stale sentinel: the run completes both phases.
    assert out["status"] == "COMPLETE"
    assert not sentinel.exists()


# --------------------------------------------------------------------------
# (f) a global STOP_ALL sentinel refuses the run
# --------------------------------------------------------------------------
def test_global_stop_all_refuses_start(tmp_path, monkeypatch) -> None:
    state_root = _isolate(tmp_path, monkeypatch)
    _write_workflow_state(state_root)

    executor = _make_executor()
    stop_control.request_stop(scope="all", source="operator")

    out = _run(executor)

    assert "error" in out
    assert "--clear-all" in out["error"]
    # No phase ever dispatched.
    executor.execute_phase.assert_not_awaited()


def test_global_stop_all_refuses_even_after_run_clear(tmp_path, monkeypatch) -> None:
    # The launch handshake clears the RUN sentinel but must NEVER clear the
    # operator-owned global one; the global sentinel survives and refuses.
    state_root = _isolate(tmp_path, monkeypatch)
    _write_workflow_state(state_root)
    stop_control.request_stop(RUN_ID, scope="run", source="stale")
    global_sentinel = stop_control.request_stop(scope="all", source="operator")

    out = _run(_make_executor())

    assert "error" in out
    assert global_sentinel is not None and global_sentinel.exists()


# --------------------------------------------------------------------------
# PipelineOrchestrator status collapse: PAUSED -> "paused"
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestrator_collapses_paused_status(tmp_path, monkeypatch) -> None:
    import MCP.orchestrator.pipeline_orchestrator as po
    from MCP.core.config import OrchestratorConfig
    from MCP.orchestrator.pipeline_orchestrator import (
        OrchestratorResult,
        PipelineOrchestrator,
    )

    config = OrchestratorConfig()
    config.workflows["test_wf"] = _two_phase_config()

    state_dir = tmp_path / "state" / "workflows"
    state_dir.mkdir(parents=True)
    (state_dir / "WF-P.json").write_text(
        json.dumps({"id": "WF-P", "type": "test_wf", "params": {}})
    )
    monkeypatch.setattr(po, "STATE_PATH", tmp_path / "state")
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "state" / "runs"))

    orch = PipelineOrchestrator(config=config, project_root=tmp_path)

    async def fake_run(self, workflow_id: str):
        return {
            "workflow_id": workflow_id,
            "status": "PAUSED",
            "paused_phase": PHASE_ONE,
            "phase_results": {},
            "phase_outputs": {},
        }

    with patch(
        "MCP.core.workflow_runner.WorkflowRunner.run_workflow", new=fake_run
    ):
        result = await orch.run("WF-P")

    assert isinstance(result, OrchestratorResult)
    assert result.status == "paused"
