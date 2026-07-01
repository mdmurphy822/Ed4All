"""W2.2 regression: a phase batch timeout preserves ALREADY-completed tasks.

Bug: ``execute_phase`` wraps ``_execute_parallel`` in ``asyncio.wait_for``.
On a whole-phase batch timeout the coroutine is CANCELLED and its local
``results`` dict (holding every task that finished before the deadline) was
thrown away — the ``except asyncio.TimeoutError`` rebuilt a fresh dict marking
every still-PENDING task ``TIMEOUT``. So a phase that ran 9 of 10 tasks to
completion, then blew the deadline on the 10th, discarded all 9 real results
AND their checkpoints and reported the whole phase timed out.

Fix: ``_execute_parallel`` writes each task's ``ExecutionResult`` into a
caller-owned ``results_sink`` the instant it finishes; on timeout
``execute_phase`` preserves the sink's completed results (and their
checkpoints) and marks ONLY the unfinished tasks ``TIMEOUT``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import ExecutionResult, TaskExecutor


def _pending(task_id: str):
    return {"id": task_id, "status": "PENDING", "dependencies": []}


@pytest.mark.asyncio
async def test_batch_timeout_preserves_completed_task_results(tmp_path):
    """One fast task completes, one hangs past the deadline: the fast task's
    real COMPLETE result (with payload) survives; only the slow task is
    TIMEOUT."""
    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    # Force a sub-second batch deadline so the hang trips it quickly.
    executor.batch_timeout_seconds = 0.3

    async def fake_execute_task(workflow_id, task_id):
        if task_id == "T-fast":
            return ExecutionResult(
                task_id="T-fast", status="COMPLETE", result={"payload": 99}
            )
        # T-slow hangs well past the 0.3s batch deadline.
        await asyncio.sleep(5)
        return ExecutionResult(task_id="T-slow", status="COMPLETE")

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("T-fast"), _pending("T-slow")]

    results, gates_passed, _ = await executor.execute_phase(
        workflow_id="WF-W22",
        phase_name="p",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    # The completed task's REAL result is preserved, not overwritten with a
    # TIMEOUT placeholder.
    assert results["T-fast"].status == "COMPLETE"
    assert results["T-fast"].result == {"payload": 99}
    # Only the unfinished task is abandoned as TIMEOUT.
    assert results["T-slow"].status == "TIMEOUT"
    # In-place task status was updated for the completed task.
    assert tasks[0]["status"] == "COMPLETE"
    # No gates configured → gates_passed stays True.
    assert gates_passed is True


@pytest.mark.asyncio
async def test_batch_timeout_preserves_completed_checkpoints(tmp_path):
    """The completed task's checkpoint is written (success=True); pre-fix the
    completed result never reached the checkpoint-update loop at all."""
    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    executor.batch_timeout_seconds = 0.3

    # Spy the checkpoint manager so we can observe complete_task calls.
    cm = MagicMock(name="checkpoint_manager")
    executor.checkpoint_manager = cm

    async def fake_execute_task(workflow_id, task_id):
        if task_id == "T-done":
            return ExecutionResult(task_id="T-done", status="COMPLETE")
        await asyncio.sleep(5)
        return ExecutionResult(task_id="T-hang", status="COMPLETE")

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("T-done"), _pending("T-hang")]

    await executor.execute_phase(
        workflow_id="WF-W22-CKPT",
        phase_name="p",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    per_task_success = {
        c.kwargs.get("task_id"): c.kwargs.get("success")
        for c in cm.complete_task.call_args_list
    }
    # The completed task reached the checkpoint loop with success=True.
    assert per_task_success.get("T-done") is True
    # The abandoned task is checkpointed as not-successful (TIMEOUT).
    assert per_task_success.get("T-hang") is False


@pytest.mark.asyncio
async def test_all_completing_batch_has_no_timeout_placeholder(tmp_path):
    """Sanity: when nothing hangs, no TIMEOUT is injected and all real results
    flow through (byte-stable happy path)."""
    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    executor.batch_timeout_seconds = 5

    async def fake_execute_task(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE", result={"id": task_id})

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("A"), _pending("B"), _pending("C")]
    results, gates_passed, _ = await executor.execute_phase(
        workflow_id="WF-W22-OK",
        phase_name="p",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    assert {tid: r.status for tid, r in results.items()} == {
        "A": "COMPLETE",
        "B": "COMPLETE",
        "C": "COMPLETE",
    }
    assert all(r.status != "TIMEOUT" for r in results.values())
    assert gates_passed is True
