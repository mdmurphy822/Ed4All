"""Graceful-stop ("checkpoint on command") threading in the executor.

Covers plan §P1 for the executor / schemas / checkpoint owner:

- ``TaskStatus.PAUSED`` exists (schemas.py).
- ``_execute_with_retries`` catches ``GracefulStopRequested`` FIRST → a PAUSED
  ExecutionResult with retry_count 0, error_class "paused", and NO poison
  classification / no ErrorClassifier pass.
- ``execute_phase``: a PAUSED task result → checkpoint stamped ``paused`` (not
  failed), ``fail_phase`` NOT called, validation gates skipped, the paused
  signal surfaced via PAUSED results, and the completed sibling recorded in the
  per-task ledger (``tasks_completed``) while the paused task stays in
  ``tasks_pending`` (resumable).
- A pre-armed sentinel → ``_execute_parallel`` dispatches nothing new (0 tool
  calls) and marks the un-run tasks PAUSED. Both run-scoped and global
  (``STOP_ALL``) sentinels trip it.
- ``CheckpointManager.pause_phase`` + ``PhaseCheckpoint.can_resume`` /
  ``is_paused`` for the ``paused`` status.

Hermetic: ``ED4ALL_STATE_RUNS_DIR`` is redirected into ``tmp_path`` so the stop
sentinels + run checkpoints never touch the real ``runtime/state/runs/`` (no course
slugs / paths anywhere). No new env flag is introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MCP.core.executor import ExecutionResult, TaskExecutor  # noqa: E402
from MCP.core.schemas import TaskStatus  # noqa: E402
from MCP.hardening.checkpoint import CheckpointManager  # noqa: E402
from lib.generation.stop_control import (  # noqa: E402
    GracefulStopRequested,
    clear_stop,
    request_stop,
)

RUN_ID = "WF-STOP-TEST"


@pytest.fixture()
def isolated_state_runs(tmp_path, monkeypatch):
    """Redirect the runtime/state/runs parent into tmp; clear any inherited run id/home."""
    runs = tmp_path / "state_runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    # Belt-and-braces: no stale sentinel from a prior test in this parent.
    clear_stop(RUN_ID, include_global=True)
    yield runs
    clear_stop(RUN_ID, include_global=True)


def _pending(task_id: str, deps=None):
    return {"id": task_id, "status": "PENDING", "dependencies": deps or []}


def _make_executor(runs: Path) -> TaskExecutor:
    return TaskExecutor(
        tool_registry={},
        run_id=RUN_ID,
        run_path=runs / RUN_ID,
    )


# ---------------------------------------------------------------------------
# schemas.py
# ---------------------------------------------------------------------------
def test_task_status_paused_exists():
    assert TaskStatus.PAUSED.value == "PAUSED"
    assert TaskStatus("PAUSED") is TaskStatus.PAUSED


# ---------------------------------------------------------------------------
# _execute_with_retries — GracefulStopRequested → PAUSED, no retry, no poison
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retries_maps_graceful_stop_to_paused_no_retry(isolated_state_runs):
    executor = _make_executor(isolated_state_runs)

    calls = {"n": 0}

    async def boom(tool_name, task_params):
        calls["n"] += 1
        raise GracefulStopRequested("test-site", units_completed=3)

    executor._invoke_tool = boom  # type: ignore[assignment]

    # Spy the poison detector so we can prove the graceful stop never reaches it.
    poison_calls = []
    if executor.poison_detector is not None:
        orig = executor.poison_detector.record_failure
        executor.poison_detector.record_failure = (  # type: ignore[assignment]
            lambda classified: poison_calls.append(classified) or orig(classified)
        )

    res = await executor._execute_with_retries(
        task_id="T1", tool_name="sometool", task_params={"id": "T1"}
    )

    assert res.status == "PAUSED"
    assert res.retry_count == 0
    assert res.error_class == "paused"
    # Raised exactly once — no retry loop on a graceful stop.
    assert calls["n"] == 1
    # The graceful stop was NOT run through the poison detector / classifier.
    assert poison_calls == []


# ---------------------------------------------------------------------------
# execute_phase — paused checkpoint, fail_phase not called, ledger accurate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_execute_phase_pauses_without_failing(isolated_state_runs):
    executor = _make_executor(isolated_state_runs)

    fail_calls = []
    orig_fail = executor.checkpoint_manager.fail_phase
    executor.checkpoint_manager.fail_phase = (  # type: ignore[assignment]
        lambda *a, **k: fail_calls.append((a, k)) or orig_fail(*a, **k)
    )

    async def fake_execute_task(workflow_id, task_id):
        if task_id == "T-stop":
            return ExecutionResult(
                task_id="T-stop", status="PAUSED", error="stop", error_class="paused"
            )
        return ExecutionResult(task_id=task_id, status="COMPLETE", result={"ok": task_id})

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("T-ok"), _pending("T-stop")]
    results, gates_passed, gate_results = await executor.execute_phase(
        workflow_id="WF",
        phase_name="p_stop",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    # Paused signal surfaced via a PAUSED result; a pause is not a gate failure.
    assert results["T-stop"].status == "PAUSED"
    assert results["T-ok"].status == "COMPLETE"
    assert gates_passed is True
    assert gate_results is None
    # fail_phase must NOT run on a graceful stop.
    assert fail_calls == []

    # Checkpoint stamped paused + resumable; completed sibling in the ledger,
    # the paused task left pending for --resume.
    cp = executor.checkpoint_manager.load_checkpoint("p_stop")
    assert cp is not None
    assert cp.status == "paused"
    assert cp.is_paused is True
    assert cp.can_resume is True
    assert "T-ok" in cp.tasks_completed
    assert "T-stop" in cp.tasks_pending
    assert "T-stop" not in cp.tasks_failed


# ---------------------------------------------------------------------------
# Pre-armed sentinel → _execute_parallel dispatches nothing new
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prearmed_run_sentinel_dispatches_nothing(isolated_state_runs):
    executor = _make_executor(isolated_state_runs)

    dispatched = {"n": 0}

    async def counting_execute_task(workflow_id, task_id):
        dispatched["n"] += 1
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = counting_execute_task  # type: ignore[assignment]

    # Arm the run-scoped sentinel BEFORE the loop starts.
    assert request_stop(RUN_ID, scope="run", reason="test", source="unit") is not None

    tasks = [_pending("A"), _pending("B")]
    results = await executor._execute_parallel(
        workflow_id="WF", tasks=tasks, max_concurrent=5
    )

    # Nothing dispatched; both tasks marked PAUSED.
    assert dispatched["n"] == 0
    assert results["A"].status == "PAUSED"
    assert results["B"].status == "PAUSED"


@pytest.mark.asyncio
async def test_prearmed_global_sentinel_dispatches_nothing(isolated_state_runs):
    executor = _make_executor(isolated_state_runs)

    dispatched = {"n": 0}

    async def counting_execute_task(workflow_id, task_id):
        dispatched["n"] += 1
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = counting_execute_task  # type: ignore[assignment]

    # Global STOP_ALL trips every run regardless of run_id.
    assert request_stop(scope="all", reason="test", source="unit") is not None

    tasks = [_pending("A")]
    results = await executor._execute_parallel(
        workflow_id="WF", tasks=tasks, max_concurrent=5
    )

    assert dispatched["n"] == 0
    assert results["A"].status == "PAUSED"


@pytest.mark.asyncio
async def test_no_sentinel_runs_normally(isolated_state_runs):
    """Sanity: with no sentinel, the loop dispatches + completes as before."""
    executor = _make_executor(isolated_state_runs)

    async def ok(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = ok  # type: ignore[assignment]

    tasks = [_pending("A"), _pending("B")]
    results = await executor._execute_parallel(
        workflow_id="WF", tasks=tasks, max_concurrent=5
    )
    assert results["A"].status == "COMPLETE"
    assert results["B"].status == "COMPLETE"


# ---------------------------------------------------------------------------
# checkpoint.py — pause_phase / can_resume / is_paused
# ---------------------------------------------------------------------------
def test_checkpoint_pause_phase_is_resumable(tmp_path):
    mgr = CheckpointManager(tmp_path / "run")
    mgr.start_phase(
        run_id="R", workflow_id="W", phase_name="phase", phase_index=0,
        task_ids=["t1", "t2"],
    )
    mgr.complete_task("phase", "t1", success=True)

    cp = mgr.pause_phase("phase", reason="graceful stop")
    assert cp.status == "paused"
    assert cp.is_paused is True
    assert cp.is_failed is False
    assert cp.is_complete is False
    # t2 was never completed → still pending → resumable.
    assert cp.can_resume is True
    assert "t2" in cp.tasks_pending

    # And it is discoverable as the resumable phase.
    resumable = mgr.get_resumable_phase()
    assert resumable is not None
    assert resumable.phase_name == "phase"


def test_checkpoint_pause_phase_missing_raises(tmp_path):
    mgr = CheckpointManager(tmp_path / "run")
    with pytest.raises(ValueError):
        mgr.pause_phase("never-started")
