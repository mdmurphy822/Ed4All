"""Per-phase VRAM-trajectory hook regression tests.

The workflow run loop wraps every ``executor.execute_phase`` call with a
best-effort "VRAM doctor" snapshot (``WorkflowRunner._vram_doctor_snapshot``)
so a crashed / VRAM-contended run leaves a forensic free-VRAM timeline at
``state/runs/<run_id>/vram_trajectory.jsonl`` (the same run dir the executor
writes its phase checkpoints into).

Contract under test:

  * Default OFF (``ED4ALL_VRAM_DOCTOR`` unset/false) → the hook is a strict
    no-op: ``snapshot_vram`` is NEVER called (no NVML probe, no ollama HTTP
    call) and ``run_workflow`` behaves exactly as before.
  * Enabled (``ED4ALL_VRAM_DOCTOR=1``) → ``write_trajectory_row`` fires once
    BEFORE and once AFTER each phase, with the executor's ``run_id`` + the
    phase name + the right ``when`` / ``event`` args, and the AFTER row carries
    the gate verdict.
  * Best-effort isolation → a snapshot / write that RAISES does not break the
    run: it still completes with its normal ``final_status``.

The hook itself lives entirely behind ``vram_doctor_enabled()`` so the default
path is byte-identical + zero-overhead.

These tests drive ``run_workflow`` end-to-end against a tmp on-disk
workflow-state file with a mocked ``executor.execute_phase`` (mirroring
``test_workflow_runner_zombie_phase_guard.py``), and additionally exercise the
extracted ``_vram_doctor_snapshot`` helper directly.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult
from MCP.core.workflow_runner import WorkflowRunner
from lib.llm.vram_doctor import VramSnapshot


WORKFLOW_ID = "WF-VRAM-0001"


def _write_workflow_state(state_root, workflow_id: str = WORKFLOW_ID) -> None:
    """Materialise the minimal on-disk workflow-state JSON run_workflow reads."""
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


def _result(task_id: str, status: str, payload: Dict[str, Any] | None = None) -> ExecutionResult:
    return ExecutionResult(task_id=task_id, status=status, result=payload)


def _make_executor(run_id: str = "RUN-VRAM-XYZ") -> MagicMock:
    """A mocked executor whose ``execute_phase`` succeeds for every phase."""
    executor = MagicMock()
    # The trajectory hook reads ``self.executor.run_id`` to land the file in
    # the same run dir as the checkpoint manager; pin it for assertions.
    executor.run_id = run_id

    async def side_effect(*_a, **_kw):  # noqa: ANN001
        # A validator-only phase-handler task that succeeds with no canonical
        # keys still completes the phase (see the zombie-guard tests).
        results = {"T-ph": _result("T-ph", "COMPLETE", {"unrelated": 1})}
        return results, True, []

    executor.execute_phase = AsyncMock(side_effect=side_effect)
    return executor


def _two_phase_config() -> WorkflowConfig:
    # Both phases are validator-only (agents: []) AND registered in
    # _PHASE_TOOL_MAPPING, so _create_phase_tasks synthesises a phase-handler
    # task and the mocked execute_phase drives them to COMPLETE.
    phases: List[WorkflowPhase] = [
        WorkflowPhase(name="inter_tier_validation", agents=[]),
        WorkflowPhase(name="post_rewrite_validation", agents=[]),
    ]
    return WorkflowConfig(description="test", phases=phases)


def _run(tmp_path, monkeypatch, executor: MagicMock) -> Dict[str, Any]:
    """Drive ``run_workflow`` with the given executor + a tmp STATE_PATH."""
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    _write_workflow_state(state_root)

    config = MagicMock()
    config.get_workflow.return_value = _two_phase_config()

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(WORKFLOW_ID))


# --------------------------------------------------------------------------
# (a) default OFF -> the hook NEVER snapshots VRAM
# --------------------------------------------------------------------------
def test_default_off_makes_zero_vram_doctor_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_VRAM_DOCTOR", raising=False)

    snapshot_mock = MagicMock(name="snapshot_vram")
    write_mock = MagicMock(name="write_trajectory_row")
    # Patch the real symbols on the module; leave vram_doctor_enabled REAL so
    # the env-driven default-OFF gate is what suppresses the probe.
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    out = _run(tmp_path, monkeypatch, _make_executor())

    assert out["status"] == "COMPLETE"
    # The expensive probe + the writer were never touched on the default path.
    snapshot_mock.assert_not_called()
    write_mock.assert_not_called()


def test_default_off_falsey_value_is_off(tmp_path, monkeypatch) -> None:
    # A non-truthy value parses to OFF (parse-with-fallback).
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "off")
    snapshot_mock = MagicMock(name="snapshot_vram")
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    out = _run(tmp_path, monkeypatch, _make_executor())

    assert out["status"] == "COMPLETE"
    snapshot_mock.assert_not_called()
    write_mock.assert_not_called()


# --------------------------------------------------------------------------
# (b) enabled -> write_trajectory_row fires before + after each phase
# --------------------------------------------------------------------------
def test_enabled_writes_before_and_after_each_phase(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")

    snap = VramSnapshot(
        free_mib=1234, total_mib=8192, cuda_available=True, probe_source="nvml"
    )
    snapshot_mock = MagicMock(name="snapshot_vram", return_value=snap)
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    executor = _make_executor(run_id="RUN-ENABLED-1")
    out = _run(tmp_path, monkeypatch, executor)

    assert out["status"] == "COMPLETE"

    # Two phases x (before, after) = 4 snapshots + 4 trajectory rows.
    assert snapshot_mock.call_count == 4
    assert write_mock.call_count == 4

    # Reconstruct (phase, when, event, extra) for every row.
    rows = []
    for call in write_mock.call_args_list:
        args, kwargs = call
        run_id, phase, when = args[0], args[1], args[2]
        rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "when": when,
                "event": kwargs.get("event"),
                "extra": kwargs.get("extra"),
            }
        )

    # Every row lands under the executor's run_id (same dir as checkpoints).
    assert all(r["run_id"] == "RUN-ENABLED-1" for r in rows)

    # Ordered: before/after of phase 1, then before/after of phase 2.
    assert [(r["phase"], r["when"], r["event"]) for r in rows] == [
        ("inter_tier_validation", "before", "phase_start"),
        ("inter_tier_validation", "after", "phase_end"),
        ("post_rewrite_validation", "before", "phase_start"),
        ("post_rewrite_validation", "after", "phase_end"),
    ]
    # The AFTER rows carry the gate verdict; the BEFORE rows carry no extra.
    assert rows[0]["extra"] is None
    assert rows[1]["extra"] == {"phase_passed": True}
    assert rows[3]["extra"] == {"phase_passed": True}

    # The snapshot object is threaded through verbatim.
    for call in write_mock.call_args_list:
        assert call.args[3] is snap


# --------------------------------------------------------------------------
# (c) a raising snapshot/write does NOT break the run (best-effort isolation)
# --------------------------------------------------------------------------
def test_snapshot_raising_does_not_break_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")

    boom = MagicMock(name="snapshot_vram", side_effect=RuntimeError("nvml exploded"))
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", boom)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    out = _run(tmp_path, monkeypatch, _make_executor())

    # The run still completes with its normal status despite the doctor blowing
    # up on every phase boundary.
    assert out["status"] == "COMPLETE"
    assert boom.call_count == 4  # still attempted at each boundary
    # The writer never ran (snapshot raised first), and the run survived.
    write_mock.assert_not_called()


def test_write_raising_does_not_break_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")

    snap = VramSnapshot(
        free_mib=None, total_mib=None, cuda_available=False, probe_source="unavailable"
    )
    snapshot_mock = MagicMock(name="snapshot_vram", return_value=snap)
    boom = MagicMock(name="write_trajectory_row", side_effect=OSError("disk full"))
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", boom)

    out = _run(tmp_path, monkeypatch, _make_executor())

    assert out["status"] == "COMPLETE"
    assert boom.call_count == 4


# --------------------------------------------------------------------------
# (d) the extracted helper in isolation
# --------------------------------------------------------------------------
def test_helper_no_op_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_VRAM_DOCTOR", raising=False)
    snapshot_mock = MagicMock(name="snapshot_vram")
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    # The default-off path must return BEFORE any offload — no worker thread.
    to_thread_mock = AsyncMock(name="to_thread")
    monkeypatch.setattr(wr_mod.asyncio, "to_thread", to_thread_mock)

    runner = WorkflowRunner(executor=_make_executor(), config=MagicMock())
    asyncio.run(runner._vram_doctor_snapshot("some_phase", "before", "phase_start"))

    snapshot_mock.assert_not_called()
    write_mock.assert_not_called()
    to_thread_mock.assert_not_called()  # zero overhead: no thread spawned


def test_helper_offloads_to_thread_when_enabled(monkeypatch) -> None:
    # The blocking snapshot+write is offloaded off the event loop so a slow
    # ollama/NVML probe can't stall the run loop at a phase boundary.
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")
    monkeypatch.setattr(wr_mod, "snapshot_vram", MagicMock())
    monkeypatch.setattr(wr_mod, "write_trajectory_row", MagicMock())

    to_thread_mock = AsyncMock(name="to_thread")
    monkeypatch.setattr(wr_mod.asyncio, "to_thread", to_thread_mock)

    runner = WorkflowRunner(executor=_make_executor(run_id="RUN-OFF"), config=MagicMock())
    asyncio.run(
        runner._vram_doctor_snapshot("p", "before", "phase_start", extra=None)
    )

    # Offloaded via to_thread, passing the blocking body + its args.
    to_thread_mock.assert_awaited_once()
    args = to_thread_mock.await_args.args
    assert args[0] == runner._vram_doctor_snapshot_blocking
    assert args[1] == "RUN-OFF"
    assert args[2] == "p"
    assert args[3] == "before"
    assert args[4] == "phase_start"
    assert args[5] is None


def test_helper_passes_run_id_phase_and_extra_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "yes")
    snap = VramSnapshot(
        free_mib=42, total_mib=8192, cuda_available=True, probe_source="torch"
    )
    snapshot_mock = MagicMock(name="snapshot_vram", return_value=snap)
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    runner = WorkflowRunner(executor=_make_executor(run_id="RUN-H"), config=MagicMock())
    asyncio.run(
        runner._vram_doctor_snapshot(
            "imscc_chunking", "after", "phase_end", extra={"phase_passed": False}
        )
    )

    snapshot_mock.assert_called_once_with()
    write_mock.assert_called_once()
    args, kwargs = write_mock.call_args
    assert args[0] == "RUN-H"
    assert args[1] == "imscc_chunking"
    assert args[2] == "after"
    assert args[3] is snap
    assert kwargs["event"] == "phase_end"
    assert kwargs["extra"] == {"phase_passed": False}


def test_blocking_body_passes_run_id_phase_and_extra(monkeypatch) -> None:
    # The sync body (run inside the worker thread) threads args through verbatim.
    snap = VramSnapshot(
        free_mib=42, total_mib=8192, cuda_available=True, probe_source="torch"
    )
    snapshot_mock = MagicMock(name="snapshot_vram", return_value=snap)
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    WorkflowRunner._vram_doctor_snapshot_blocking(
        "RUN-H", "imscc_chunking", "after", "phase_end", {"phase_passed": False}
    )

    snapshot_mock.assert_called_once_with()
    write_mock.assert_called_once()
    args, kwargs = write_mock.call_args
    assert args[0] == "RUN-H"
    assert args[1] == "imscc_chunking"
    assert args[2] == "after"
    assert args[3] is snap
    assert kwargs["event"] == "phase_end"
    assert kwargs["extra"] == {"phase_passed": False}


def test_helper_swallows_run_id_access_error(monkeypatch) -> None:
    # Even if reading executor.run_id were to raise, the helper must not.
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")
    monkeypatch.setattr(wr_mod, "snapshot_vram", MagicMock())
    monkeypatch.setattr(wr_mod, "write_trajectory_row", MagicMock())

    class _Boom:
        @property
        def run_id(self):  # noqa: D401
            raise RuntimeError("no run id")

    runner = WorkflowRunner(executor=_Boom(), config=MagicMock())
    # Must not raise.
    asyncio.run(runner._vram_doctor_snapshot("p", "before", "phase_start"))


def test_helper_swallows_to_thread_error(monkeypatch) -> None:
    # A to_thread failure on the enabled path must not perturb control flow.
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")
    monkeypatch.setattr(wr_mod, "snapshot_vram", MagicMock())
    monkeypatch.setattr(wr_mod, "write_trajectory_row", MagicMock())
    monkeypatch.setattr(
        wr_mod.asyncio,
        "to_thread",
        AsyncMock(side_effect=RuntimeError("thread pool exhausted")),
    )

    runner = WorkflowRunner(executor=_make_executor(run_id="RUN-T"), config=MagicMock())
    # Must not raise.
    asyncio.run(runner._vram_doctor_snapshot("p", "before", "phase_start"))
