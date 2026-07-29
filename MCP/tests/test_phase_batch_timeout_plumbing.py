"""Regression: per-phase ``batch_timeout_minutes`` is plumbed end-to-end.

Bug (confirmed): ``config/workflows.yaml`` sets the
``content_generation_rewrite`` phase ``batch_timeout_minutes: 240`` (a slow
local-7B rewrite), but that per-phase value was NEVER applied — the phase was
killed at the 30-minute executor-wide default mid-run. Two layers were broken:

1. ``WorkflowPhase`` had no ``batch_timeout_minutes`` field, so the YAML value
   was dropped at config-parse time.
2. ``TaskExecutor.execute_phase`` only read the executor-wide
   ``self.batch_timeout_seconds`` (resolved once at construction from
   ``ED4ALL_BATCH_TIMEOUT_MINUTES`` env → 30), with no per-phase override.

The fix:
- ``config.py`` parses ``batch_timeout_minutes`` onto ``WorkflowPhase``.
- ``execute_phase`` accepts ``phase_batch_timeout_minutes`` and resolves the
  effective batch timeout with precedence: per-phase YAML value (if set) →
  ``ED4ALL_BATCH_TIMEOUT_MINUTES`` env (the executor-wide fallback) → 30.
- ``WorkflowRunner.run_workflow`` forwards ``phase.batch_timeout_minutes``.

These tests pin all three layers, with the priority assertion being:
``content_generation_rewrite`` declares 240 in YAML => 14400s even when the
env override is unset.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MCP.core.config import OrchestratorConfig, WorkflowConfig, WorkflowPhase
from MCP.core.executor import ExecutionResult, TaskExecutor
from MCP.core.workflow_runner import WorkflowRunner


# ---------------------------------------------------------------------------
# Layer 1 — config parse: workflows.yaml -> WorkflowPhase.batch_timeout_minutes
# ---------------------------------------------------------------------------


def test_workflows_yaml_parses_rewrite_batch_timeout() -> None:
    """The real config/workflows.yaml value flows onto WorkflowPhase.

    Pre-fix WorkflowPhase had no such field, so this assertion could not
    even be expressed — the YAML value was silently dropped.

    The assertion is against the RAW yaml rather than a hardcoded number:
    what must hold is that whatever the operator writes in the phase block
    arrives on the dataclass. Pinning the literal turns a legitimate
    retune of the ceiling into a test failure, which teaches people to
    edit the test rather than think about the value. What is worth
    pinning is that the phase declares one at all, and that it is not
    silently the executor-wide 30-minute default.
    """
    import yaml  # noqa: PLC0415 - test-local

    raw = yaml.safe_load(
        (Path(__file__).parent.parent.parent
         / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    config = OrchestratorConfig.load()

    for workflow_type in ("textbook_to_course", "course_generation"):
        wf = config.get_workflow(workflow_type)
        assert wf is not None, f"{workflow_type} workflow missing from config"
        rewrite = next(
            (p for p in wf.phases if p.name == "content_generation_rewrite"),
            None,
        )
        assert rewrite is not None, (
            f"{workflow_type} has no content_generation_rewrite phase"
        )
        raw_phase = next(
            p for p in raw["workflows"][workflow_type]["phases"]
            if p.get("name") == "content_generation_rewrite"
        )
        declared = raw_phase.get("batch_timeout_minutes")
        assert declared is not None, (
            f"{workflow_type}.content_generation_rewrite must DECLARE a "
            f"batch_timeout_minutes — without one it silently inherits the "
            f"30-minute executor default, which cannot fit the tier."
        )
        assert rewrite.batch_timeout_minutes == declared, (
            f"{workflow_type}.content_generation_rewrite declares "
            f"batch_timeout_minutes: {declared} in YAML but WorkflowPhase "
            f"carries {rewrite.batch_timeout_minutes!r} — the value is being "
            f"dropped between config and dataclass."
        )
        assert declared > 30, (
            f"{workflow_type}.content_generation_rewrite declares "
            f"{declared} min, at or below the executor-wide 30-minute "
            f"fallback the per-phase value exists to override."
        )


def test_phase_without_yaml_batch_timeout_defaults_none() -> None:
    """A phase with no YAML ``batch_timeout_minutes`` stays None (byte-stable)."""
    phase = WorkflowPhase(name="staging", agents=["textbook-stager"])
    assert phase.batch_timeout_minutes is None


# ---------------------------------------------------------------------------
# Layer 2 — executor: execute_phase honors the per-phase value
# ---------------------------------------------------------------------------


def _capture_wait_for_timeout(monkeypatch):
    """Record the batch wall-clock deadline execute_phase enforces.

    Post-graceful-stop (plan P5 / AMENDMENT #5) ``execute_phase`` no longer
    wraps ``_execute_parallel`` in ``asyncio.wait_for`` (that would hard-cancel
    the batch at the deadline). It wraps it in a Task and enforces the deadline
    with a NON-cancelling ``asyncio.wait({task}, timeout=deadline)``. This helper
    captures the STAGE-1 timeout (the resolved effective batch deadline) from the
    first ``asyncio.wait`` call, delegating to the real ``asyncio.wait`` so the
    wrapped batch (empty task list in these plumbing tests) actually completes.
    """
    captured: dict = {}
    _real_wait = asyncio.wait

    async def fake_wait(aws, *args, timeout=None, **kwargs):
        captured.setdefault("timeout", timeout)
        return await _real_wait(aws, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr("MCP.core.executor.asyncio.wait", fake_wait)
    return captured


async def _run_execute_phase(executor: TaskExecutor, **kwargs):
    return await executor.execute_phase(
        workflow_id="WF-TIMEOUT",
        phase_name="content_generation_rewrite",
        phase_index=0,
        tasks=[],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_execute_phase_honors_phase_batch_timeout_minutes(
    monkeypatch, tmp_path
) -> None:
    """YAML 240 -> 14400s used as the batch wait_for timeout, env UNSET.

    This is the priority assertion: the value flows from workflows.yaml
    through to the executor's batch wall-clock without the env override.
    """
    monkeypatch.delenv("ED4ALL_BATCH_TIMEOUT_MINUTES", raising=False)
    captured = _capture_wait_for_timeout(monkeypatch)

    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    # Sanity: executor-wide fallback is the 30-min default when env unset.
    assert executor.batch_timeout_seconds == 30 * 60

    await _run_execute_phase(executor, phase_batch_timeout_minutes=240)

    assert captured["timeout"] == 240 * 60, (
        "execute_phase must use the per-phase 240-minute timeout (14400s), "
        f"got {captured['timeout']!r}"
    )


@pytest.mark.asyncio
async def test_execute_phase_none_falls_back_to_executor_default(
    monkeypatch, tmp_path
) -> None:
    """No per-phase value (None) -> executor-wide fallback (30 min, env unset)."""
    monkeypatch.delenv("ED4ALL_BATCH_TIMEOUT_MINUTES", raising=False)
    captured = _capture_wait_for_timeout(monkeypatch)

    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    await _run_execute_phase(executor, phase_batch_timeout_minutes=None)

    assert captured["timeout"] == 30 * 60


@pytest.mark.asyncio
async def test_execute_phase_env_applies_when_no_phase_value(
    monkeypatch, tmp_path
) -> None:
    """ENV remains a useful knob for phases with NO YAML value."""
    monkeypatch.setenv("ED4ALL_BATCH_TIMEOUT_MINUTES", "90")
    captured = _capture_wait_for_timeout(monkeypatch)

    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    assert executor.batch_timeout_seconds == 90 * 60
    await _run_execute_phase(executor, phase_batch_timeout_minutes=None)

    assert captured["timeout"] == 90 * 60


@pytest.mark.asyncio
async def test_execute_phase_phase_value_wins_over_env(
    monkeypatch, tmp_path
) -> None:
    """Precedence: per-phase YAML value WINS over the env override."""
    monkeypatch.setenv("ED4ALL_BATCH_TIMEOUT_MINUTES", "60")
    captured = _capture_wait_for_timeout(monkeypatch)

    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    await _run_execute_phase(executor, phase_batch_timeout_minutes=240)

    assert captured["timeout"] == 240 * 60


@pytest.mark.asyncio
async def test_execute_phase_garbage_phase_value_falls_back(
    monkeypatch, tmp_path
) -> None:
    """A non-positive / unparseable per-phase value degrades to the fallback."""
    monkeypatch.delenv("ED4ALL_BATCH_TIMEOUT_MINUTES", raising=False)
    captured = _capture_wait_for_timeout(monkeypatch)

    executor = TaskExecutor(tool_registry={}, run_path=tmp_path / "run")
    await _run_execute_phase(executor, phase_batch_timeout_minutes=0)
    assert captured["timeout"] == 30 * 60


# ---------------------------------------------------------------------------
# Layer 3 — runner: run_workflow forwards phase.batch_timeout_minutes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_workflow_forwards_phase_batch_timeout(
    monkeypatch, tmp_path
) -> None:
    """End-to-end stage path: run_workflow passes the phase's YAML value to
    execute_phase as ``phase_batch_timeout_minutes``.

    Uses a non-special workflow_type (so the corpus-generalization /
    authoring-route preamble helpers no-op) and a recording AsyncMock
    executor, so this pins the single forwarding line without driving the
    full pipeline.
    """
    monkeypatch.delenv("ED4ALL_BATCH_TIMEOUT_MINUTES", raising=False)
    # Test/dry-run authoring-route opt-in so the LLM-needing agent doesn't
    # trip the fail-fast mailbox guard (the executor is mocked anyway).
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")

    # Build a minimal config with a single non-special workflow + one phase
    # that carries the per-phase YAML timeout.
    phase = WorkflowPhase(
        name="slow_phase",
        agents=["content-generator"],
        timeout_minutes=10080,
        batch_timeout_minutes=240,
    )
    config = OrchestratorConfig()
    config.workflows["tf_timeout_test"] = WorkflowConfig(
        description="timeout plumbing fixture",
        phases=[phase],
    )

    # Recording executor: capture execute_phase kwargs.
    recorded: dict = {}

    async def fake_execute_phase(**kwargs):
        recorded.update(kwargs)
        tasks = kwargs.get("tasks") or []
        results = {
            t.get("id"): ExecutionResult(task_id=t.get("id"), status="COMPLETE")
            for t in tasks
        }
        return results, True, None

    executor = AsyncMock()
    executor.execute_phase.side_effect = fake_execute_phase
    executor.run_id = "WF-TIMEOUT-RUNNER"

    runner = WorkflowRunner(executor=executor, config=config)

    # Write the workflow state file the runner reads.
    state_dir = tmp_path / "state" / "workflows"
    state_dir.mkdir(parents=True)
    workflow_id = "WF-TF-TIMEOUT"
    (state_dir / f"{workflow_id}.json").write_text(
        json.dumps({
            "type": "tf_timeout_test",
            "params": {},
            # A persisted incomplete phase exercises the resume path: timeout
            # policy must come from the freshly loaded registry, never stale
            # task/checkpoint state.
            "phase_outputs": {
                "slow_phase": {"_completed": False, "_paused": True}
            },
        })
    )

    with patch("MCP.core.workflow_runner.STATE_PATH", tmp_path / "state"):
        await runner.run_workflow(workflow_id)

    assert recorded.get("phase_batch_timeout_minutes") == 240, (
        "run_workflow must forward phase.batch_timeout_minutes (240) to "
        f"execute_phase, got {recorded.get('phase_batch_timeout_minutes')!r}"
    )
    assert recorded.get("phase_task_timeout_minutes") == 10080
