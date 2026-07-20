"""Orchestrator → executor plumbing.

``PipelineOrchestrator._get_executor(workflow_state=...)`` must hand the
``TaskExecutor`` the workflow's own run_id, run_path, and DecisionCapture.
Constructing it with none of those is silently wrong, not loud:

* ``TaskExecutor.run_id`` falls back to a timestamp, so ``run_path`` becomes
  ``state/runs/run_{ts}/`` instead of the workflow's run directory —
  ``CheckpointManager`` writes to an orphan dir nobody reads and
  ``LockfileManager`` operates outside the workflow's namespace.
* ``self.capture is None`` makes every ``phase_start`` /
  ``phase_completion`` / ``task_retry`` / ``workflow_execution`` emit site a
  no-op, so a completed run leaves empty checkpoint + capture dirs.

This suite locks in the wire-up and the no-state back-compat semantics.
"""

from __future__ import annotations

import json

import pytest

from MCP.core.executor import TaskExecutor
from MCP.orchestrator.pipeline_orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def synthetic_workflow_state(tmp_path, monkeypatch):
    """Write a minimal workflow state + return (orchestrator, state)."""
    # Shape-preserving synthetic run id (<workflow-prefix>_<course>_<stamp>).
    run_id = "WF_TEST_100_20260420_123456"
    state = {
        "workflow_id": run_id,
        "type": "textbook_to_course",
        "params": {
            "course_name": "TEST_100",
            "run_id": run_id,
            "pdf_paths": [],
            "duration_weeks": 6,
        },
        "phase_outputs": {},
        "tasks": [],
        "status": "PENDING",
    }
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True)
    path = workflows_dir / f"{run_id}.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    # Monkeypatch STATE_PATH everywhere it's consumed in the
    # orchestrator plumbing.
    monkeypatch.setattr(
        "MCP.orchestrator.pipeline_orchestrator.STATE_PATH", tmp_path,
    )
    monkeypatch.setattr(
        "MCP.core.workflow_runner.STATE_PATH", tmp_path,
    )
    # TaskExecutor resolves run_path via ``lib.paths.get_state_runs_dir``,
    # which honors ``ED4ALL_STATE_RUNS_DIR``. Set it so this test's executor
    # checkpoints land in tmp_path instead of the project state/runs/.
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    # ``_get_executor`` also exports ``ED4ALL_RUN_ID`` into the process env
    # (so downstream pipeline tools can build a MailboxBrokeredBackend bound
    # to this run's mailbox). monkeypatch so it is restored on teardown —
    # otherwise an unrelated later test calling ``build_backend()`` reads the
    # stale id and recreates ``state/runs/<old-run-id>/``.
    monkeypatch.setenv("ED4ALL_RUN_ID", run_id)
    return run_id, state


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


def test_get_executor_with_workflow_state_sets_run_id(synthetic_workflow_state):
    """_get_executor(workflow_state=...) gives TaskExecutor the workflow run_id."""
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor(workflow_state=state)

    assert executor.run_id == run_id, (
        "Executor should use the workflow's params.run_id, not a "
        "timestamp-generated orphan ID."
    )


def test_get_executor_with_workflow_state_sets_run_path(synthetic_workflow_state, tmp_path):
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor(workflow_state=state)

    assert executor.run_path == tmp_path / "runs" / run_id, (
        "Executor run_path must match the workflow's run directory "
        "so checkpoints + lockfiles land in the right namespace."
    )


def test_get_executor_with_workflow_state_creates_capture(synthetic_workflow_state):
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor(workflow_state=state)

    assert executor.capture is not None, (
        "Executor must receive a DecisionCapture when a workflow state "
        "is known; otherwise every phase_start/phase_completion/"
        "task_retry/workflow_execution emit site silently no-ops."
    )


def test_executor_capture_uses_normalized_course_code(synthetic_workflow_state):
    """Capture must use normalize_course_code so course_id validates."""
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor(workflow_state=state)

    from lib.decision_capture import normalize_course_code
    expected = normalize_course_code("TEST_100")

    assert executor.capture.course_code == expected


def test_get_executor_without_state_still_works_for_legacy_callers(state_runs_isolated):
    """Back-compat: _get_executor() with no args (legacy signature) still works.

    Uses ``state_runs_isolated`` so the timestamp-fallback ``run_path``
    lands in tmp_path instead of polluting project ``state/runs/``.
    """
    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor()  # no workflow_state — legacy call shape

    assert isinstance(executor, TaskExecutor)
    # When there's no workflow state, capture falls back to None (old behaviour)
    # and run_id falls back to a timestamp — both acceptable for tests.
    assert executor.run_id  # set to something


def test_executor_is_cached_across_dispatcher_callbacks(synthetic_workflow_state):
    """Repeat _get_executor calls should return the same TaskExecutor."""
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    e1 = orch._get_executor(workflow_state=state)
    e2 = orch._get_executor(workflow_state=state)

    assert e1 is e2, "Executor identity must survive across calls."


def test_normalize_course_code_is_importable_from_lib_decision_capture():
    """normalize_course_code must be exported from lib.decision_capture."""
    from lib.decision_capture import normalize_course_code
    assert callable(normalize_course_code)


def test_workflow_run_emits_phase_start_capture(synthetic_workflow_state, tmp_path):
    """Running a workflow through the orchestrator must emit phase_start captures."""
    run_id, state = synthetic_workflow_state

    orch = PipelineOrchestrator(mode="local")
    executor = orch._get_executor(workflow_state=state)

    # Spy on the capture
    calls = []
    original_log = executor.capture.log_decision

    def _spy(decision_type, decision, rationale, **kwargs):
        calls.append({"type": decision_type, "decision": decision})
        return original_log(decision_type, decision, rationale, **kwargs)

    executor.capture.log_decision = _spy

    # Drive execute_phase directly — it's what the runner calls per-phase.
    import asyncio
    async def _run():
        # Minimal phase execution with no tasks — should still emit
        # phase_start / phase_completion via the capture.
        return await executor.execute_phase(
            workflow_id=run_id,
            phase_name="test_phase",
            phase_index=0,
            tasks=[],
            gate_configs=None,
            max_concurrent=1,
        )

    asyncio.run(_run())

    types = {c["type"] for c in calls}
    assert "phase_start" in types, (
        f"Expected phase_start capture to fire. Got types: {types}"
    )
    assert "phase_completion" in types, (
        f"Expected phase_completion capture to fire. Got types: {types}"
    )
