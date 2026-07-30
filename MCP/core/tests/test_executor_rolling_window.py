"""Rolling-concurrency-window contracts for ``TaskExecutor._execute_parallel``.

``_execute_parallel`` used to be a BATCH BARRIER: it sliced the dependency-
satisfied frontier to ``max_concurrent`` and blocked on ``asyncio.gather`` over
the WHOLE slice before dispatching the next slice, so concurrency drained toward
1 as the fast tasks in a slice finished while gather waited on the slowest. It is
now a ROLLING WINDOW that keeps up to ``max_concurrent`` tasks in flight and
refills a freed slot the instant any single task completes.

These tests pin the eight contracts the refactor had to preserve while removing
the drain (all hermetic — fake ``execute_task`` stubs with event-driven,
sleep-free ordering; no real LLM / seat / docker / registry):

1. Concurrency STAYS PINNED at ``max_concurrent`` under a slow+fast mix (the win)
   — and, discriminatingly, fast tasks keep completing WHILE a slow task holds a
   slot, which a barrier could not do.
2. Dependency DAG still gates — a dependent never starts before its dep completes.
3. Poison-pill still halts the phase — dispatch stops on the poisoned completion,
   in-flight tasks finish (not cancelled), un-run tasks stay PENDING.
4. Stop-sentinel halts cleanly — no new dispatch after stop, in-flight drain +
   checkpoint, un-run PENDING marked PAUSED.
5. Started tasks are NOT cancelled on a stop/poison halt (artifact integrity).
6. Per-task checkpoint written for every completed task.
7. The whole-phase batch-timeout wall bound still fires (a wedged task cannot
   hang the phase forever) — via the unchanged ``execute_phase`` wrapper.
8. completed/failed accounting + return value identical to the barrier for a
   no-failure run.

Hermetic: ``ED4ALL_STATE_RUNS_DIR`` is redirected into ``tmp_path`` so stop
sentinels + checkpoints never touch the real ``runtime/state/runs/``. No course slugs /
paths / campaign names anywhere.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MCP.core.executor import ExecutionResult, TaskExecutor  # noqa: E402
from lib.generation.stop_control import (  # noqa: E402
    clear_stop,
    request_stop,
)

RUN_ID = "WF-ROLLING-TEST"


@pytest.fixture()
def isolated_state_runs(tmp_path, monkeypatch):
    runs = tmp_path / "state_runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    clear_stop(RUN_ID, include_global=True)
    yield runs
    clear_stop(RUN_ID, include_global=True)


def _pending(task_id: str, deps=None):
    return {"id": task_id, "status": "PENDING", "dependencies": deps or []}


def _make_executor(runs: Path) -> TaskExecutor:
    return TaskExecutor(tool_registry={}, run_id=RUN_ID, run_path=runs / RUN_ID)


async def _wait_for(predicate, *, timeout=5.0, tick=0.005):
    """Await until ``predicate()`` is truthy, or fail the test on timeout.

    A BARRIER regression (fast tasks blocked behind a held slow task) makes the
    predicate never come true, so this converts a hang into a fast failure.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(tick)
    raise AssertionError("condition not met before timeout (barrier regression?)")


# ---------------------------------------------------------------------------
# 1. Concurrency stays pinned at max_concurrent (the core win)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrency_stays_pinned_with_slow_and_fast_mix(isolated_state_runs):
    """One slow task holds a slot while a stream of fast tasks refills the rest.

    With a barrier, the fast tasks in the slow task's slice would finish and
    ``gather`` would wait on the slow one, draining concurrency to 1 and blocking
    all remaining fast tasks until the slow one finished. The rolling window
    refills on each completion, so:
      * peak in-flight reaches ``max_concurrent``, and
      * ALL fast tasks complete WHILE the slow task is still held (the barrier
        could not do this — it is the discriminating assertion).
    """
    executor = _make_executor(isolated_state_runs)
    max_concurrent = 4
    n_fast = 12

    release_slow = asyncio.Event()
    slow_started = asyncio.Event()
    live: set = set()
    peak = {"n": 0}
    fast_done = {"n": 0}

    async def fake_execute_task(workflow_id, task_id):
        live.add(task_id)
        peak["n"] = max(peak["n"], len(live))
        try:
            if task_id == "SLOW":
                slow_started.set()
                await release_slow.wait()
            else:
                # Yield a few times so co-dispatched siblings are simultaneously
                # "live", making the peak-concurrency measurement meaningful.
                for _ in range(3):
                    await asyncio.sleep(0)
                fast_done["n"] += 1
            return ExecutionResult(task_id=task_id, status="COMPLETE")
        finally:
            live.discard(task_id)

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("SLOW")] + [_pending(f"F{i}") for i in range(n_fast)]
    run = asyncio.ensure_future(
        executor._execute_parallel("WF", tasks, max_concurrent=max_concurrent)
    )

    # The slow task must start and all fast tasks must complete WHILE it is held.
    await _wait_for(slow_started.is_set)
    await _wait_for(lambda: fast_done["n"] == n_fast)
    assert not run.done(), "phase finished before the slow task was released"

    # Now release the slow task and let the phase finish.
    release_slow.set()
    results = await asyncio.wait_for(run, timeout=5.0)

    # Peak in-flight hit the full width (slow + 3 fast) and never exceeded it.
    assert peak["n"] == max_concurrent
    assert all(r.status == "COMPLETE" for r in results.values())
    assert len(results) == n_fast + 1


# ---------------------------------------------------------------------------
# 2. Dependency DAG gating
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dependency_gating_dependent_waits_for_dep(isolated_state_runs):
    """A task with an unmet dep never starts before its dep completes."""
    executor = _make_executor(isolated_state_runs)

    release_dep = asyncio.Event()
    dep_started = asyncio.Event()
    started_order: list = []

    async def fake_execute_task(workflow_id, task_id):
        started_order.append(task_id)
        if task_id == "DEP":
            dep_started.set()
            await release_dep.wait()
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    tasks = [_pending("DEP"), _pending("CHILD", deps=["DEP"])]
    run = asyncio.ensure_future(
        executor._execute_parallel("WF", tasks, max_concurrent=5)
    )

    await _wait_for(dep_started.is_set)
    # DEP is running (held); CHILD must NOT have started — its dep isn't COMPLETE.
    await asyncio.sleep(0)
    assert "CHILD" not in started_order, "dependent started before its dep completed"

    release_dep.set()
    results = await asyncio.wait_for(run, timeout=5.0)

    assert started_order.index("DEP") < started_order.index("CHILD")
    assert results["DEP"].status == "COMPLETE"
    assert results["CHILD"].status == "COMPLETE"


@pytest.mark.asyncio
async def test_completion_unlocks_new_frontier_tasks(isolated_state_runs):
    """A completing task unlocks its dependents into the refill frontier."""
    executor = _make_executor(isolated_state_runs)

    async def ok(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = ok  # type: ignore[assignment]

    # A -> B -> C chain plus an independent D; max_concurrent=1 forces the
    # refill path to unlock each successor exactly when its predecessor lands.
    tasks = [
        _pending("A"),
        _pending("B", deps=["A"]),
        _pending("C", deps=["B"]),
        _pending("D"),
    ]
    results = await executor._execute_parallel("WF", tasks, max_concurrent=1)
    assert {tid: r.status for tid, r in results.items()} == {
        "A": "COMPLETE",
        "B": "COMPLETE",
        "C": "COMPLETE",
        "D": "COMPLETE",
    }


# ---------------------------------------------------------------------------
# 3 + 5. Poison-pill halts dispatch; in-flight finish (not cancelled)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_poison_pill_halts_dispatch_inflight_finishes(isolated_state_runs):
    """A poisoned completion stops further dispatch; the sibling in flight is
    allowed to finish (never cancelled) and un-run tasks stay PENDING."""
    executor = _make_executor(isolated_state_runs)

    release_slow = asyncio.Event()
    slow_started = asyncio.Event()
    dispatched: list = []
    cancelled_slow = {"hit": False}

    async def fake_execute_task(workflow_id, task_id):
        dispatched.append(task_id)
        if task_id == "POISON":
            return ExecutionResult(task_id=task_id, status="POISON_PILL", error="boom")
        if task_id == "SLOW":
            slow_started.set()
            try:
                await release_slow.wait()
            except asyncio.CancelledError:
                cancelled_slow["hit"] = True
                raise
            return ExecutionResult(task_id=task_id, status="COMPLETE")
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    # SLOW + POISON dispatched together (width 2); LATER1/LATER2 must NOT be
    # dispatched once POISON trips.
    tasks = [
        _pending("SLOW"),
        _pending("POISON"),
        _pending("LATER1"),
        _pending("LATER2"),
    ]
    run = asyncio.ensure_future(
        executor._execute_parallel("WF", tasks, max_concurrent=2)
    )

    await _wait_for(slow_started.is_set)
    # Give the loop a chance to process the poisoned completion + attempt refill.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not run.done(), "phase halted before draining the in-flight SLOW task"
    assert "LATER1" not in dispatched and "LATER2" not in dispatched

    # Release SLOW: it drains to COMPLETE (was never cancelled).
    release_slow.set()
    results = await asyncio.wait_for(run, timeout=5.0)

    assert cancelled_slow["hit"] is False
    assert results["SLOW"].status == "COMPLETE"
    assert results["POISON"].status == "POISON_PILL"
    # Un-run tasks were never dispatched and are absent from results (PENDING,
    # not PAUSED — poison writes no stop sentinel).
    assert "LATER1" not in results and "LATER2" not in results
    assert "LATER1" not in dispatched and "LATER2" not in dispatched


# ---------------------------------------------------------------------------
# 4 + 5 + 6. Stop-sentinel: clean drain, no new dispatch, PAUSED marking,
# checkpoint written, in-flight not cancelled
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stop_sentinel_midflight_drains_and_pauses(isolated_state_runs):
    """A stop armed while a task is in flight: the in-flight task finishes +
    checkpoints, no new task is dispatched, and un-run PENDING tasks are PAUSED."""
    executor = _make_executor(isolated_state_runs)

    # Publish an active phase so per-task checkpoints are written.
    executor._active_phase_name = "p_stop_mid"
    executor.checkpoint_manager.start_phase(
        run_id=RUN_ID, workflow_id="WF", phase_name="p_stop_mid", phase_index=0,
        task_ids=["INFLIGHT", "LATER"],
    )
    checkpointed: list = []
    orig_cp = executor._checkpoint_task_result
    executor._checkpoint_task_result = (  # type: ignore[assignment]
        lambda phase, tid, success: checkpointed.append((tid, success))
        or orig_cp(phase, tid, success)
    )

    release = asyncio.Event()
    inflight_started = asyncio.Event()
    dispatched: list = []
    cancelled = {"hit": False}

    async def fake_execute_task(workflow_id, task_id):
        dispatched.append(task_id)
        inflight_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled["hit"] = True
            raise
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    # max_concurrent=1 so only INFLIGHT dispatches; LATER must stay un-dispatched
    # once the stop is armed.
    tasks = [_pending("INFLIGHT"), _pending("LATER")]
    run = asyncio.ensure_future(
        executor._execute_parallel("WF", tasks, max_concurrent=1)
    )

    await _wait_for(inflight_started.is_set)
    request_stop(RUN_ID, scope="run", reason="test", source="unit")
    release.set()  # let the in-flight task drain to its unit boundary
    results = await asyncio.wait_for(run, timeout=5.0)

    # In-flight task completed (drained, never cancelled) + checkpointed.
    assert cancelled["hit"] is False
    assert results["INFLIGHT"].status == "COMPLETE"
    assert ("INFLIGHT", True) in checkpointed
    # LATER never dispatched; marked PAUSED for --resume.
    assert "LATER" not in dispatched
    assert results["LATER"].status == "PAUSED"
    assert results["LATER"].error_class == "paused"


# ---------------------------------------------------------------------------
# 6. Per-task checkpoint written for every completed task
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_per_task_checkpoint_written_for_all_completions(isolated_state_runs):
    executor = _make_executor(isolated_state_runs)
    executor._active_phase_name = "p_cp"
    executor.checkpoint_manager.start_phase(
        run_id=RUN_ID, workflow_id="WF", phase_name="p_cp", phase_index=0,
        task_ids=["A", "B", "C"],
    )
    checkpointed: list = []
    orig_cp = executor._checkpoint_task_result
    executor._checkpoint_task_result = (  # type: ignore[assignment]
        lambda phase, tid, success: checkpointed.append((tid, success))
        or orig_cp(phase, tid, success)
    )

    async def mixed(workflow_id, task_id):
        if task_id == "B":
            return ExecutionResult(task_id=task_id, status="ERROR", error="x")
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = mixed  # type: ignore[assignment]

    tasks = [_pending("A"), _pending("B"), _pending("C")]
    await executor._execute_parallel("WF", tasks, max_concurrent=3)

    by_tid = dict(checkpointed)
    assert by_tid == {"A": True, "B": False, "C": True}


# ---------------------------------------------------------------------------
# 7. Whole-phase batch-timeout wall bound still fires (wedged task ≠ hang)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_wedged_task_does_not_hang_phase(isolated_state_runs):
    """A task that never consults the stop sentinel cannot hang the phase past
    the batch deadline + grace — the unchanged ``execute_phase`` wrapper wraps
    the whole rolling coroutine and hard-cancels it, preserving completed tasks
    and marking the wedged one TIMEOUT."""
    executor = _make_executor(isolated_state_runs)
    executor.batch_timeout_seconds = 0.2  # sub-second whole-phase deadline

    async def maybe_wedge(workflow_id, task_id):
        if task_id == "WEDGED":
            await asyncio.sleep(30)  # never checks the sentinel
        return ExecutionResult(task_id=task_id, status="COMPLETE", result={"id": task_id})

    executor.execute_task = maybe_wedge  # type: ignore[assignment]

    tasks = [_pending("FAST"), _pending("WEDGED")]
    results, gates_passed, _ = await asyncio.wait_for(
        executor.execute_phase(
            workflow_id="WF", phase_name="p_wedge", phase_index=0,
            tasks=tasks, gate_configs=None, max_concurrent=5,
        ),
        timeout=10.0,
    )

    assert results["FAST"].status == "COMPLETE"
    assert results["FAST"].result == {"id": "FAST"}
    assert results["WEDGED"].status == "TIMEOUT"
    assert gates_passed is True


# ---------------------------------------------------------------------------
# 8. Accounting + return value identical to the barrier for a no-failure run
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_failure_accounting_matches_barrier(isolated_state_runs):
    """A no-failure run returns exactly one COMPLETE result per task, keyed by
    task id — the same shape the barrier produced (only completion ORDER differs,
    which was already non-deterministic under gather)."""
    executor = _make_executor(isolated_state_runs)

    async def ok(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE", result={"id": task_id})

    executor.execute_task = ok  # type: ignore[assignment]

    tasks = [_pending(f"T{i}") for i in range(23)]  # > max_concurrent
    results = await executor._execute_parallel("WF", tasks, max_concurrent=5)

    assert set(results) == {f"T{i}" for i in range(23)}
    assert all(r.status == "COMPLETE" for r in results.values())
    assert all(results[f"T{i}"].result == {"id": f"T{i}"} for i in range(23))
    # Every task ran exactly once (no double-dispatch across refills).
    assert len(results) == 23


@pytest.mark.asyncio
async def test_results_written_into_caller_sink(isolated_state_runs):
    """The ``results_sink`` contract is preserved: completed results land in the
    caller-owned dict as they finish (what the batch-timeout path relies on)."""
    executor = _make_executor(isolated_state_runs)

    async def ok(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = ok  # type: ignore[assignment]

    sink: dict = {}
    tasks = [_pending("A"), _pending("B")]
    returned = await executor._execute_parallel(
        "WF", tasks, max_concurrent=5, results_sink=sink
    )
    assert returned is sink
    assert sink["A"].status == "COMPLETE"
    assert sink["B"].status == "COMPLETE"
