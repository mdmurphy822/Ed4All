"""Phase-boundary GPU-lifecycle sweep regression tests.

The workflow run loop fires a deterministic GPU lease hand-off
(``WorkflowRunner._gpu_lifecycle_sweep``) AFTER each phase completes
successfully (post task results + gates + checkpoint persist) and BEFORE the
next phase dispatches, so the resident local model + torch allocator cache are
handed to the next stage.

Contract under test:

  * Default ON (``ED4ALL_GPU_LIFECYCLE`` unset) → the sweep fires exactly once
    per COMPLETED phase, AFTER phase completion/gates and BEFORE the next phase
    dispatch (call-order assertion against a recording executor).
  * Explicitly off (``ED4ALL_GPU_LIFECYCLE=0``) → zero sweep calls
    (byte-identical control flow).
  * A raising sweep never changes ``final_status`` or the phase results.
  * No sweep after a FAILED phase break.
  * No sweep between tasks within a phase (the hook is a phase-boundary hook,
    exercised via the direct-helper test's single call).

Drives ``run_workflow`` end-to-end against a tmp on-disk workflow-state file
with a mocked ``executor.execute_phase`` (mirroring the VRAM-doctor tests).
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


WORKFLOW_ID = "WF-GPULC-0001"


def _write_workflow_state(state_root, workflow_id: str = WORKFLOW_ID) -> None:
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


def _make_executor(run_id: str = "RUN-GPULC", *, fail: bool = False) -> MagicMock:
    executor = MagicMock()
    executor.run_id = run_id

    async def side_effect(*_a, **_kw):
        if fail:
            results = {"T-ph": _result("T-ph", "ERROR", None)}
            return results, False, []
        results = {"T-ph": _result("T-ph", "COMPLETE", {"unrelated": 1})}
        return results, True, []

    executor.execute_phase = AsyncMock(side_effect=side_effect)
    return executor


def _two_phase_config() -> WorkflowConfig:
    phases: List[WorkflowPhase] = [
        WorkflowPhase(name="inter_tier_validation", agents=[]),
        WorkflowPhase(name="post_rewrite_validation", agents=[]),
    ]
    return WorkflowConfig(description="test", phases=phases)


def _run(tmp_path, monkeypatch, executor: MagicMock) -> Dict[str, Any]:
    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    # Keep the VRAM doctor OFF so its own snapshots don't interfere.
    monkeypatch.delenv("ED4ALL_VRAM_DOCTOR", raising=False)
    _write_workflow_state(state_root)

    config = MagicMock()
    config.get_workflow.return_value = _two_phase_config()

    runner = WorkflowRunner(executor=executor, config=config)
    return asyncio.run(runner.run_workflow(WORKFLOW_ID))


# --------------------------------------------------------------------------
# default ON — sweep fires once per completed phase
# --------------------------------------------------------------------------
def test_default_on_sweeps_each_completed_phase(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)

    sweep = AsyncMock(name="_gpu_lifecycle_sweep")
    monkeypatch.setattr(WorkflowRunner, "_gpu_lifecycle_sweep", sweep)

    out = _run(tmp_path, monkeypatch, _make_executor())

    assert out["status"] == "COMPLETE"
    # One sweep per completed phase, in phase order. (An AsyncMock patched onto
    # the class is not a descriptor, so ``self`` is not bound → phase_name is
    # args[0].)
    assert [c.args[0] for c in sweep.await_args_list] == [
        "inter_tier_validation",
        "post_rewrite_validation",
    ]


def test_sweep_fires_after_phase_execute(tmp_path, monkeypatch) -> None:
    # Call-order: the sweep for phase N must fire AFTER that phase's
    # execute_phase and BEFORE the next phase's execute_phase.
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)
    order: List[str] = []

    executor = _make_executor()
    real_execute = executor.execute_phase

    async def recording_execute(*a, **kw):
        order.append(f"execute:{kw.get('phase_name')}")
        return await real_execute(*a, **kw)

    executor.execute_phase = AsyncMock(side_effect=recording_execute)

    async def recording_sweep(self, phase_name):  # noqa: ANN001
        order.append(f"sweep:{phase_name}")

    monkeypatch.setattr(WorkflowRunner, "_gpu_lifecycle_sweep", recording_sweep)

    out = _run(tmp_path, monkeypatch, executor)
    assert out["status"] == "COMPLETE"
    assert order == [
        "execute:inter_tier_validation",
        "sweep:inter_tier_validation",
        "execute:post_rewrite_validation",
        "sweep:post_rewrite_validation",
    ]


# --------------------------------------------------------------------------
# explicitly off — zero sweep calls
# --------------------------------------------------------------------------
def test_disabled_flag_zero_release_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")

    # Leave _gpu_lifecycle_sweep REAL (it is always awaited), but patch the
    # underlying release primitives; disabled mode must make ZERO of them fire.
    import lib.gpu_lifecycle as gl

    ollama = MagicMock(name="release_ollama_models", return_value=[])
    torch = MagicMock(name="release_torch")
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    out = _run(tmp_path, monkeypatch, _make_executor())

    assert out["status"] == "COMPLETE"
    ollama.assert_not_called()
    torch.assert_not_called()


# --------------------------------------------------------------------------
# a raising sweep never changes final_status
# --------------------------------------------------------------------------
def test_sweep_exception_does_not_change_status(tmp_path, monkeypatch) -> None:
    # The REAL async wrapper wraps the blocking body best-effort, so even a
    # blocking body that raises on every phase boundary must not abort the run.
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)

    def _boom(*_a, **_k):
        raise RuntimeError("sweep body exploded")

    monkeypatch.setattr(
        WorkflowRunner, "_gpu_lifecycle_sweep_blocking", staticmethod(_boom)
    )

    out = _run(tmp_path, monkeypatch, _make_executor())
    assert out["status"] == "COMPLETE"


# --------------------------------------------------------------------------
# no sweep after a FAILED phase break
# --------------------------------------------------------------------------
def test_no_sweep_after_failed_phase(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)

    sweep = AsyncMock(name="_gpu_lifecycle_sweep")
    monkeypatch.setattr(WorkflowRunner, "_gpu_lifecycle_sweep", sweep)

    # First phase ERRORs → the loop breaks before the sweep call.
    out = _run(tmp_path, monkeypatch, _make_executor(fail=True))

    assert out["status"] == "FAILED"
    sweep.assert_not_awaited()


# --------------------------------------------------------------------------
# the helper in isolation
# --------------------------------------------------------------------------
def test_helper_no_op_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "off")

    to_thread_mock = AsyncMock(name="to_thread")
    monkeypatch.setattr(wr_mod.asyncio, "to_thread", to_thread_mock)

    runner = WorkflowRunner(executor=_make_executor(), config=MagicMock())
    asyncio.run(runner._gpu_lifecycle_sweep("some_phase"))

    # Disabled → returns before spawning a worker thread (zero overhead).
    to_thread_mock.assert_not_called()


def test_helper_offloads_to_thread_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)  # default ON

    to_thread_mock = AsyncMock(name="to_thread")
    monkeypatch.setattr(wr_mod.asyncio, "to_thread", to_thread_mock)

    runner = WorkflowRunner(executor=_make_executor(run_id="RUN-Z"), config=MagicMock())
    asyncio.run(runner._gpu_lifecycle_sweep("content_generation"))

    to_thread_mock.assert_awaited_once()
    args = to_thread_mock.await_args.args
    assert args[0] == runner._gpu_lifecycle_sweep_blocking
    assert args[1] == "RUN-Z"
    assert args[2] == "content_generation"


def test_helper_swallows_to_thread_error(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)
    monkeypatch.setattr(
        wr_mod.asyncio,
        "to_thread",
        AsyncMock(side_effect=RuntimeError("pool exhausted")),
    )
    runner = WorkflowRunner(executor=_make_executor(), config=MagicMock())
    # Must not raise.
    asyncio.run(runner._gpu_lifecycle_sweep("p"))


def test_blocking_body_releases_and_writes_trajectory_when_doctor_on(monkeypatch) -> None:
    import lib.gpu_lifecycle as gl

    ollama = MagicMock(return_value=["qwen2.5:7b"])
    torch = MagicMock()
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    # Doctor ON → a lifecycle_sweep trajectory row is appended.
    monkeypatch.setenv("ED4ALL_VRAM_DOCTOR", "1")
    snapshot_mock = MagicMock(name="snapshot_vram", return_value="SNAP")
    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "snapshot_vram", snapshot_mock)
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    WorkflowRunner._gpu_lifecycle_sweep_blocking("RUN-A", "packaging")

    ollama.assert_called_once()
    torch.assert_called_once()
    write_mock.assert_called_once()
    args, kwargs = write_mock.call_args
    assert args[0] == "RUN-A"
    assert args[1] == "packaging"
    assert args[2] == "after"
    assert kwargs["event"] == "lifecycle_sweep"
    assert kwargs["extra"] == {"evicted_models": ["qwen2.5:7b"]}


def test_blocking_body_no_trajectory_when_doctor_off(monkeypatch) -> None:
    import lib.gpu_lifecycle as gl

    monkeypatch.setattr(gl, "release_ollama_models", MagicMock(return_value=[]))
    monkeypatch.setattr(gl, "release_torch", MagicMock())
    monkeypatch.delenv("ED4ALL_VRAM_DOCTOR", raising=False)

    write_mock = MagicMock(name="write_trajectory_row")
    monkeypatch.setattr(wr_mod, "write_trajectory_row", write_mock)

    WorkflowRunner._gpu_lifecycle_sweep_blocking("RUN-B", "packaging")

    # Doctor off → releases still fire, but no trajectory row.
    write_mock.assert_not_called()
