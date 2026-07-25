"""Active-phase metering context in the executor.

Metering-correctness: ``execute_phase`` publishes the currently-executing phase
name in the ``ED4ALL_ACTIVE_PHASE`` env for the duration of the phase (mirroring
the instance-level ``self._active_phase_name`` it already tracks) so an
in-process content-gen LLM usage tap — which is NOT handed the executor instance
— can stamp the SPENDING phase on its ``llm_usage.jsonl`` row. This is the same
env the Trainforge tap reads
(``Trainforge.generators._openai_compatible_client.ENV_ACTIVE_PHASE``).

Contract verified here:
- The env is SET to the phase name while a task runs inside the phase.
- The env is RESTORED to its prior value (removed when it was unset) after the
  phase returns, so a subsequent / nested phase never inherits a stale phase.

Hermetic: ``ED4ALL_STATE_RUNS_DIR`` is redirected into ``tmp_path`` and no run
state touches the real ``state/runs/``. No course slugs / paths anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MCP.core.executor import ExecutionResult, TaskExecutor  # noqa: E402
from lib.generation.stop_control import clear_stop  # noqa: E402

_ENV = "ED4ALL_ACTIVE_PHASE"
RUN_ID = "WF-ACTIVE-PHASE-TEST"


@pytest.fixture()
def isolated_state_runs(tmp_path, monkeypatch):
    runs = tmp_path / "state_runs"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    monkeypatch.delenv(_ENV, raising=False)
    clear_stop(RUN_ID, include_global=True)
    yield runs
    clear_stop(RUN_ID, include_global=True)


def _make_executor(runs: Path) -> TaskExecutor:
    return TaskExecutor(
        tool_registry={},
        run_id=RUN_ID,
        run_path=runs / RUN_ID,
    )


def _pending(task_id: str):
    return {"id": task_id, "status": "PENDING", "dependencies": []}


@pytest.mark.asyncio
async def test_active_phase_env_set_during_phase_and_cleared_after(
    isolated_state_runs,
):
    executor = _make_executor(isolated_state_runs)
    observed: dict = {}

    async def fake_execute_task(workflow_id, task_id):
        # Captured from INSIDE the phase execution window — an in-process
        # content-gen tap running here would read exactly this value.
        observed[task_id] = os.environ.get(_ENV)
        return ExecutionResult(
            task_id=task_id, status="COMPLETE", result={"ok": task_id}
        )

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    # Env is unset before the phase.
    assert _ENV not in os.environ

    await executor.execute_phase(
        workflow_id="WF",
        phase_name="course_planning",
        phase_index=0,
        tasks=[_pending("T-a"), _pending("T-b")],
        max_concurrent=2,
    )

    # Every task saw the active phase published.
    assert observed["T-a"] == "course_planning"
    assert observed["T-b"] == "course_planning"
    # Restored to its prior (unset) state — no leak into the next phase.
    assert _ENV not in os.environ


@pytest.mark.asyncio
async def test_active_phase_env_restored_to_prior_value(
    isolated_state_runs, monkeypatch
):
    executor = _make_executor(isolated_state_runs)
    monkeypatch.setenv(_ENV, "outer_phase")
    observed: dict = {}

    async def fake_execute_task(workflow_id, task_id):
        observed[task_id] = os.environ.get(_ENV)
        return ExecutionResult(
            task_id=task_id, status="COMPLETE", result={"ok": task_id}
        )

    executor.execute_task = fake_execute_task  # type: ignore[assignment]

    await executor.execute_phase(
        workflow_id="WF",
        phase_name="content_generation_rewrite",
        phase_index=1,
        tasks=[_pending("T-1")],
        max_concurrent=1,
    )

    # The inner phase's tap saw the inner phase name...
    assert observed["T-1"] == "content_generation_rewrite"
    # ...and the pre-existing outer value is restored afterwards.
    assert os.environ.get(_ENV) == "outer_phase"
