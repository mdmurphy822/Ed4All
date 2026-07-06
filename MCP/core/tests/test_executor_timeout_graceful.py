"""Timeout → graceful conversion in the executor (plan §P5, Wave D).

Covers the two owner-assigned P5 legs with the BINDING AMENDMENT #5 / D10
mechanics:

1. **BATCH timeout** (``execute_phase`` / ``ED4ALL_BATCH_TIMEOUT_MINUTES`` /
   per-phase YAML). ``asyncio.wait_for`` CANCELS at the deadline — the hard kill
   the grace period must prevent. The two-stage restructure instead:
   - stage 1: non-cancelling ``asyncio.wait`` to the deadline;
   - on expiry: ``stop_control.request_stop`` (RUN-SCOPED → whole-phase pause,
     "timeouts become pauses");
   - stage 2: bounded grace window for in-flight workers to drain to a unit
     boundary and return PAUSED;
   - only on grace expiry: hard-cancel → the pre-existing TIMEOUT marking.
   A grace-drained batch surfaces PAUSED, NOT TIMEOUT; the sentinel stays set
   (the run pauses). An unresponsive worker falls through to TIMEOUT and the
   timeout-authored sentinel is cleared (the run does NOT pause).

2. **TASK timeout** (``_invoke_tool`` / ``ED4ALL_TASK_TIMEOUT_MINUTES``).
   Resolved Decision D10: a single slow task must NOT pause the whole run, so
   the run-scoped sentinel is NEVER written. Instead a per-TASK in-process stop
   Event is signalled (``current_task_stop_event``), a grace window granted, and
   after grace the existing ``asyncio.TimeoutError`` classification + transient
   retry ladder stands unchanged (retry replays the sidecar → no work lost).

Hermetic: ``ED4ALL_STATE_RUNS_DIR`` is redirected into ``tmp_path`` so the stop
sentinels never touch the real ``state/runs/``. CPU-only; no models loaded; no
course slugs / paths; no new env flag.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MCP.core import executor as executor_mod  # noqa: E402
from MCP.core.executor import (  # noqa: E402
    ExecutionResult,
    TaskExecutor,
    _grace_seconds,
    current_task_stop_event,
)
from lib.generation.stop_control import (  # noqa: E402
    clear_stop,
    stop_requested,
)

RUN_ID = "WF-TIMEOUT-GRACE"


@pytest.fixture()
def isolated_state_runs(tmp_path, monkeypatch):
    """Redirect the state/runs parent into tmp; clear any inherited run id/home."""
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


def _make_executor(runs: Path, **kwargs) -> TaskExecutor:
    return TaskExecutor(
        tool_registry=kwargs.pop("tool_registry", {}),
        run_id=RUN_ID,
        run_path=runs / RUN_ID,
        **kwargs,
    )


def _passthrough_mapper(executor: TaskExecutor) -> None:
    """Make the param mapper a no-op passthrough for synthetic tool names.

    The real ``TaskParameterMapper`` rejects unregistered tool names; these
    tests exercise the ``_invoke_tool`` TIMEOUT mechanics, not param mapping, so
    they register a synthetic tool and route its params straight through.
    """
    executor.param_mapper.map_task_to_tool_params = (  # type: ignore[assignment]
        lambda task_params, tool_name: {
            k: v for k, v in (task_params or {}).items() if k != "agent_type"
        }
    )


# ---------------------------------------------------------------------------
# _grace_seconds — the module-constant grace formula
# ---------------------------------------------------------------------------
def test_grace_seconds_fraction_and_cap(monkeypatch):
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 0.10)
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_CAP_SECONDS", 600.0)
    # 10% of the deadline while under the cap.
    assert _grace_seconds(1800) == pytest.approx(180.0)
    # Capped for very large deadlines (10% of 2h = 720 > 600 → 600).
    assert _grace_seconds(7200) == pytest.approx(600.0)
    # Degrades to 0 on garbage rather than raising.
    assert _grace_seconds("nope") == 0.0


# ---------------------------------------------------------------------------
# BATCH timeout — stop-aware workers drain to PAUSED within grace (no cancel)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_grace_drains_to_paused_not_timeout(
    isolated_state_runs, monkeypatch
):
    """Deadline fires → run-scoped stop requested → stop-aware workers drain to
    PAUSED within grace. Result is PAUSED (not TIMEOUT), sentinel stays set, no
    hard cancel."""
    executor = _make_executor(isolated_state_runs)
    # Tiny deadline; generous grace so the pollers notice the sentinel.
    executor.batch_timeout_seconds = 0.15
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 4.0)

    async def stop_aware(workflow_id, task_id):
        # A stop-aware worker: drains at its next unit boundary once the
        # run-scoped sentinel appears (models the Wave-A/C in-loop check).
        for _ in range(4000):
            if stop_requested(executor.run_id):
                return ExecutionResult(
                    task_id=task_id, status="PAUSED", error_class="paused"
                )
            await asyncio.sleep(0.005)
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = stop_aware  # type: ignore[assignment]

    tasks = [_pending("A"), _pending("B")]
    results, gates_passed, gate_results = await executor.execute_phase(
        workflow_id="WF",
        phase_name="p_grace",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    # Drained to PAUSED — NOT hard-cancelled to TIMEOUT.
    assert results["A"].status == "PAUSED"
    assert results["B"].status == "PAUSED"
    assert all(r.status != "TIMEOUT" for r in results.values())
    # A pause is not a gate failure.
    assert gates_passed is True
    assert gate_results is None
    # The run-scoped sentinel stays set (the run is paused, cleared only at the
    # next run_workflow start / --resume).
    assert stop_requested(executor.run_id) is True
    # Phase checkpoint stamped paused (resumable), not failed.
    cp = executor.checkpoint_manager.load_checkpoint("p_grace")
    assert cp is not None and cp.status == "paused"


# ---------------------------------------------------------------------------
# BATCH timeout — unresponsive worker → grace expiry → existing TIMEOUT path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_unresponsive_worker_grace_expiry_timeout(
    isolated_state_runs, monkeypatch
):
    """A worker that never checks the sentinel: deadline + grace both elapse →
    hard cancel → the pre-existing TIMEOUT marking fires exactly as before, and
    the timeout-authored sentinel is CLEARED so the run does not spuriously
    pause a later phase."""
    executor = _make_executor(isolated_state_runs)
    executor.batch_timeout_seconds = 0.1
    # Small grace so the test doesn't linger.
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 0.5)

    async def unresponsive(workflow_id, task_id):
        if task_id == "T-fast":
            return ExecutionResult(
                task_id="T-fast", status="COMPLETE", result={"payload": 7}
            )
        await asyncio.sleep(30)  # ignores the sentinel entirely
        return ExecutionResult(task_id="T-slow", status="COMPLETE")

    executor.execute_task = unresponsive  # type: ignore[assignment]

    tasks = [_pending("T-fast"), _pending("T-slow")]
    results, gates_passed, _ = await executor.execute_phase(
        workflow_id="WF",
        phase_name="p_hard",
        phase_index=0,
        tasks=tasks,
        gate_configs=None,
        max_concurrent=5,
    )

    # Pre-graceful behaviour preserved: completed task kept, unfinished TIMEOUT.
    assert results["T-fast"].status == "COMPLETE"
    assert results["T-fast"].result == {"payload": 7}
    assert results["T-slow"].status == "TIMEOUT"
    # No PAUSED task → not a pause; gates run (none configured → True).
    assert all(r.status != "PAUSED" for r in results.values())
    assert gates_passed is True
    # The timeout-authored run-scoped sentinel was cleared on the hard path.
    assert stop_requested(executor.run_id) is False


# ---------------------------------------------------------------------------
# BATCH timeout — the two-stage deadline + grace timeouts are what we expect
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_two_stage_wait_timeouts(isolated_state_runs, monkeypatch):
    """The stage-1 wait gets the batch deadline; the stage-2 wait gets exactly
    ``_grace_seconds(deadline)``. Monkeypatched grace constants are respected."""
    executor = _make_executor(isolated_state_runs)
    executor.batch_timeout_seconds = 0.1
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 0.5)
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_CAP_SECONDS", 600.0)

    recorded: list = []
    _real_wait = asyncio.wait

    async def recording_wait(aws, *args, timeout=None, **kwargs):
        recorded.append(timeout)
        return await _real_wait(aws, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(executor_mod.asyncio, "wait", recording_wait)

    async def unresponsive(workflow_id, task_id):
        await asyncio.sleep(30)
        return ExecutionResult(task_id=task_id, status="COMPLETE")

    executor.execute_task = unresponsive  # type: ignore[assignment]

    await executor.execute_phase(
        workflow_id="WF",
        phase_name="p_two_stage",
        phase_index=0,
        tasks=[_pending("A")],
        gate_configs=None,
        max_concurrent=5,
    )

    # Exactly two waits: stage-1 deadline, then stage-2 grace.
    assert len(recorded) == 2
    assert recorded[0] == pytest.approx(0.1)
    assert recorded[1] == pytest.approx(_grace_seconds(0.1))
    assert recorded[1] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_batch_all_complete_before_deadline_no_stop(isolated_state_runs):
    """Happy path: everything finishes before the deadline → no stop requested,
    no sentinel written, all COMPLETE (byte-stable)."""
    executor = _make_executor(isolated_state_runs)
    executor.batch_timeout_seconds = 5

    async def ok(workflow_id, task_id):
        return ExecutionResult(task_id=task_id, status="COMPLETE", result={"id": task_id})

    executor.execute_task = ok  # type: ignore[assignment]

    results, gates_passed, _ = await executor.execute_phase(
        workflow_id="WF",
        phase_name="p_ok",
        phase_index=0,
        tasks=[_pending("A"), _pending("B")],
        gate_configs=None,
        max_concurrent=5,
    )

    assert {tid: r.status for tid, r in results.items()} == {
        "A": "COMPLETE",
        "B": "COMPLETE",
    }
    assert gates_passed is True
    # No deadline hit → no sentinel authored.
    assert stop_requested(executor.run_id) is False


# ---------------------------------------------------------------------------
# TASK timeout — run-scoped sentinel NEVER written (D10); TimeoutError raised
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_task_timeout_raises_without_run_sentinel(
    isolated_state_runs, monkeypatch
):
    """An in-process tool that ignores its per-task deadline: after deadline +
    grace it is hard-cancelled and ``asyncio.TimeoutError`` is raised — and the
    RUN-scoped sentinel is NEVER written (one slow task must not pause the
    run)."""
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 0.3)

    async def slow_tool(**kwargs):
        await asyncio.sleep(30)
        return json.dumps({"ok": True})

    executor = _make_executor(
        isolated_state_runs, tool_registry={"slow_tool": slow_tool}
    )
    executor.timeout_seconds = 0.1
    _passthrough_mapper(executor)

    with pytest.raises(asyncio.TimeoutError):
        await executor._invoke_tool("slow_tool", {"id": "T", "agent_type": None})

    # D10: the per-task timeout uses a TASK-scoped channel, never the run
    # sentinel — the run stays runnable.
    assert stop_requested(executor.run_id) is False


@pytest.mark.asyncio
async def test_task_stop_event_signalled_and_consultable(
    isolated_state_runs, monkeypatch
):
    """A stop-aware in-process tool can consult ``current_task_stop_event`` and
    drain when the per-task deadline signals it — proving the task-scoped
    channel is wired (event present during the call, set on timeout) — again
    with NO run-scoped sentinel."""
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 4.0)

    observed: dict = {}

    async def stop_aware_tool(**kwargs):
        ev = current_task_stop_event()
        observed["present"] = ev is not None
        for _ in range(4000):
            if ev is not None and ev.is_set():
                observed["saw_set"] = True
                return json.dumps({"drained": True})
            await asyncio.sleep(0.005)
        return json.dumps({"drained": False})

    executor = _make_executor(
        isolated_state_runs, tool_registry={"aware": stop_aware_tool}
    )
    executor.timeout_seconds = 0.1  # deadline; grace = 4.0 * 0.1 = 0.4s
    _passthrough_mapper(executor)

    result = await executor._invoke_tool("aware", {"id": "T", "agent_type": None})

    assert observed.get("present") is True
    assert observed.get("saw_set") is True
    assert result == {"drained": True}
    # Task-scoped only — no run sentinel written.
    assert stop_requested(executor.run_id) is False
    # And the channel is torn down after the call (no leak into a later tool).
    assert current_task_stop_event() is None


@pytest.mark.asyncio
async def test_task_timeout_retry_ladder_preserved(isolated_state_runs, monkeypatch):
    """The task-timeout leg keeps today's transient-retry ladder: a persistently
    slow tool is classified transient and retried ``max_retries`` times, ends
    ERROR (never PAUSED), and never writes the run sentinel."""
    monkeypatch.setattr(executor_mod, "BATCH_TIMEOUT_GRACE_FRACTION", 0.3)

    attempts = {"n": 0}

    async def slow_tool(**kwargs):
        attempts["n"] += 1
        await asyncio.sleep(30)
        return json.dumps({"ok": True})

    executor = TaskExecutor(
        tool_registry={"slow_tool": slow_tool},
        run_id=RUN_ID,
        run_path=isolated_state_runs / RUN_ID,
        max_retries=1,
        timeout_seconds=0.1,
    )
    _passthrough_mapper(executor)

    res = await executor._execute_with_retries(
        task_id="T", tool_name="slow_tool", task_params={"id": "T", "agent_type": None}
    )

    # Retried once (2 attempts total) — the transient-timeout ladder is intact.
    # ``retry_count`` counts loop iterations (one per attempt), matching the
    # pre-change behaviour byte-for-byte.
    assert attempts["n"] == 2
    assert res.retry_count == 2
    # A timeout is transient → after exhaustion it is ERROR, NOT PAUSED.
    assert res.status == "ERROR"
    assert res.status != "PAUSED"
    # Never the run-scoped pause path.
    assert stop_requested(executor.run_id) is False
