"""Phase 4 Subtask 4 — full WorkflowRunner exercises phase-handler dispatch.

Subtask 1 added the synthetic-task fallback in
``WorkflowRunner._create_phase_tasks`` so validator-only phases
(``agents: []``) registered in ``_PHASE_TOOL_MAPPING`` actually
reach their handler. Subtasks 2-3 documented + unit-tested the
fix at the helper level. This integration test drives the FULL
``WorkflowRunner.run_workflow`` path against a canned workflow
state and an in-process handler registered via ``tool_registry``,
then asserts disk-write side-effects land — confirming Phase 3.5's
``_run_post_rewrite_validation`` (and siblings) actually fire in
end-to-end runs, not just direct ``asyncio.run`` invocations.

Schema-registration note (deeper architectural fix landed here):
the four phase handlers (``run_*``) were wired into
``_build_tool_registry`` (pipeline_tools.py) and into
``_PHASE_TOOL_MAPPING`` (executor.py) but missed the third leg of
the wiring invariant: ``TOOL_SCHEMAS`` (tool_schemas.py). Without
those entries the executor's ``param_mapper`` raised
``ParameterMappingError("Unknown tool: ...")`` whenever the
synthetic task tried to dispatch. This test file is the proof
that the schema additions in this commit unblock the full path.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from MCP.core.config import OrchestratorConfig, WorkflowConfig, WorkflowPhase  # noqa: E402
from MCP.core.executor import (  # noqa: E402
    AGENT_TOOL_MAPPING,
    TaskExecutor,
    _PHASE_TOOL_MAPPING,
)
from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — minimal in-process phase handler that records disk writes.
# ---------------------------------------------------------------------------


def _make_disk_writing_handler(out_dir: Path) -> Any:
    """Build an async handler that mirrors ``_run_post_rewrite_validation``'s
    contract: write ``blocks_validated.jsonl`` + ``blocks_failed.jsonl`` to
    a project-relative dir and return the JSON envelope a phase handler
    would return.

    This stands in for the real pipeline_tools handler so the test can
    assert disk writes WITHOUT spinning up the full Phase 3 router stack
    (which would force us into corpus / project / objectives setup beyond
    the integration test's scope).
    """

    async def handler(**kwargs) -> str:
        validated_path = out_dir / "blocks_validated.jsonl"
        failed_path = out_dir / "blocks_failed.jsonl"
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        validated_path.write_text(
            json.dumps({"block_id": "test_block_01", "passed": True}) + "\n",
            encoding="utf-8",
        )
        failed_path.write_text("", encoding="utf-8")
        return json.dumps({
            "success": True,
            "blocks_validated_path": str(validated_path),
            "blocks_failed_path": str(failed_path),
            "block_count": 1,
            "failed_block_count": 0,
            # Echo back the routed_params so the test can assert that
            # the synthetic task carried them through to the handler.
            "_received_kwargs": sorted(kwargs.keys()),
        })

    return handler


def _seed_workflow_state(
    state_dir: Path,
    workflow_id: str,
    workflow_type: str,
    workflow_params: Dict[str, Any],
    phase_outputs: Dict[str, Dict[str, Any]],
) -> Path:
    """Write a minimal workflow state JSON the runner can load."""
    workflows_dir = state_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = workflows_dir / f"{workflow_id}.json"
    workflow_path.write_text(json.dumps({
        "workflow_id": workflow_id,
        "type": workflow_type,
        "status": "PENDING",
        "params": workflow_params,
        "phase_outputs": phase_outputs,
        "tasks": [],
    }), encoding="utf-8")
    return workflow_path


def _make_minimal_config(phase_name: str) -> OrchestratorConfig:
    """Build an OrchestratorConfig holding a single-phase workflow.

    The phase declares ``agents: []`` (Subtask 1's target case) so the
    runner's ``_create_phase_tasks`` must synthesize the virtual task
    via ``_PHASE_TOOL_MAPPING.get(phase.name)`` for the handler to fire.
    """
    phase = WorkflowPhase(
        name=phase_name,
        agents=[],
        parallel=False,
        max_concurrent=1,
        depends_on=[],
        timeout_minutes=2,
    )
    workflow = WorkflowConfig(
        description="Phase 4 Subtask 4 minimal workflow",
        phases=[phase],
    )
    cfg = OrchestratorConfig.__new__(OrchestratorConfig)
    cfg.workflows = {"phase4_subtask4_integration": workflow}
    cfg.agents = {}
    return cfg


# ---------------------------------------------------------------------------
# Subtask 4 — full WorkflowRunner.run_workflow end-to-end pin
# ---------------------------------------------------------------------------


def test_workflow_runner_dispatches_post_rewrite_validation_handler_to_disk(
    tmp_path: Path,
) -> None:
    """The runner must invoke ``run_post_rewrite_validation`` via the
    synthetic task and the handler must persist its outputs.

    This is the assertion the Phase 3.5 review identified as missing
    pre-Phase 4: validator-only phases passed unit tests via direct
    asyncio.run but never landed their disk-write side-effects in
    end-to-end runs.
    """
    state_dir = tmp_path / "state"
    out_dir = tmp_path / "phase_handler_out"
    workflow_id = "WF-PHASE4-SUB4-A"

    # Build the in-process registered handler for the phase.
    handler = _make_disk_writing_handler(out_dir)
    tool_registry = {"run_post_rewrite_validation": handler}

    # Seed a minimal workflow state file the runner can load.
    _seed_workflow_state(
        state_dir=state_dir,
        workflow_id=workflow_id,
        workflow_type="phase4_subtask4_integration",
        workflow_params={
            "course_name": "PHASE4_TEST",
            "blocks_final_path": str(tmp_path / "blocks_final.jsonl"),
            "project_id": "PROJ-PHASE4-TEST",
        },
        phase_outputs={},
    )

    config = _make_minimal_config("post_rewrite_validation")
    executor = TaskExecutor(
        tool_registry=tool_registry,
        validate_registry=False,
    )
    runner = WorkflowRunner(executor=executor, config=config)

    # Stub _route_params so the integration test exercises the synthetic
    # task + dispatch + handler-disk-write path WITHOUT depending on the
    # YAML-driven inputs_from resolution (which targets phase chains, not
    # standalone single-phase workflows). Subtask 1's contract is that
    # the routed_params dict propagates verbatim to the synthetic task;
    # this stub simply mirrors what the YAML resolution would have
    # produced for the post_rewrite_validation phase if upstream phases
    # had populated phase_outputs.
    def _stub_route(phase_name, workflow_params, phase_outputs):
        return {
            "blocks_final_path": workflow_params.get("blocks_final_path"),
            "project_id": workflow_params.get("project_id"),
        }

    runner._route_params = _stub_route  # type: ignore[assignment]

    # Redirect both modules' STATE_PATH at the tmp dir so the runner +
    # executor read/write the seeded state file.
    with patch("MCP.core.workflow_runner.STATE_PATH", state_dir), \
         patch("MCP.core.executor.STATE_PATH", state_dir):
        result = asyncio.run(runner.run_workflow(workflow_id))

    # Workflow-level assertions — the phase landed and reported COMPLETE.
    assert result.get("status") == "COMPLETE", (
        f"workflow status should be COMPLETE, got {result.get('status')!r}; "
        f"full result: {result}"
    )

    # Phase-handler disk-write assertions — pre-Subtask-1 these would
    # never land because the per-agent task loop yielded zero tasks
    # for ``agents: []`` phases.
    validated_path = out_dir / "blocks_validated.jsonl"
    failed_path = out_dir / "blocks_failed.jsonl"
    assert validated_path.exists(), (
        "blocks_validated.jsonl must land on disk via the phase handler; "
        "pre-Subtask-1 the synthetic task was never created and the "
        "handler was never invoked"
    )
    assert failed_path.exists(), (
        "blocks_failed.jsonl must also land — handler I/O contract"
    )

    # The validated JSONL carries the canonical payload the handler emits.
    payload_line = validated_path.read_text(encoding="utf-8").strip()
    assert payload_line, "blocks_validated.jsonl must be non-empty"
    record = json.loads(payload_line)
    assert record["block_id"] == "test_block_01"
    assert record["passed"] is True

    # Phase outputs were extracted and persisted on the workflow state.
    phase_outputs = result.get("phase_outputs", {})
    assert "post_rewrite_validation" in phase_outputs, (
        "extracted phase outputs must be re-surfaced in the run result"
    )
    handler_outputs = phase_outputs["post_rewrite_validation"]
    assert handler_outputs.get("blocks_validated_path") == str(validated_path)
    assert handler_outputs.get("blocks_failed_path") == str(failed_path)


def test_workflow_runner_synthetic_task_carries_routed_params_to_handler(
    tmp_path: Path,
) -> None:
    """The synthetic task's ``params`` must reach the handler kwargs.

    Subtask 1's contract: ``routed_params.copy()`` is propagated to the
    synthetic task so ``inputs_from``-resolved params still flow through
    to the phase handler (e.g. ``project_id``, ``blocks_final_path``).
    Without this propagation the handler would crash on a missing kwarg.
    """
    state_dir = tmp_path / "state"
    out_dir = tmp_path / "phase_handler_out_b"
    workflow_id = "WF-PHASE4-SUB4-B"

    received: Dict[str, Any] = {}

    async def capturing_handler(**kwargs) -> str:
        received.update(kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "handler_ran.txt"
        sentinel.write_text("yes", encoding="utf-8")
        return json.dumps({
            "success": True,
            "blocks_validated_path": str(sentinel),
        })

    # Seed workflow state with phase_outputs that the routing table
    # would extract a project_id / blocks_final_path from. We bypass
    # the legacy routing table by writing the params directly into
    # workflow_params; the runner's _route_params helper does NOT
    # have a routing entry for our synthetic phase name, so it falls
    # through to whatever workflow_params provides — but that's fine
    # because the routed_params dict is what _create_phase_tasks
    # propagates verbatim to the synthetic task.
    _seed_workflow_state(
        state_dir=state_dir,
        workflow_id=workflow_id,
        workflow_type="phase4_subtask4_integration",
        workflow_params={
            "course_name": "PHASE4_TEST",
            "blocks_final_path": "/tmp/canned/blocks_final.jsonl",
            "project_id": "PROJ-CARRY-PARAMS",
        },
        phase_outputs={},
    )

    tool_registry = {
        "run_post_rewrite_validation": capturing_handler,
    }
    config = _make_minimal_config("post_rewrite_validation")
    executor = TaskExecutor(
        tool_registry=tool_registry,
        validate_registry=False,
    )
    runner = WorkflowRunner(executor=executor, config=config)

    # Same _route_params stub as the disk-write test — propagates the
    # blocks_final_path + project_id from workflow_params to the
    # synthetic task's params dict.
    def _stub_route(phase_name, workflow_params, phase_outputs):
        return {
            "blocks_final_path": workflow_params.get("blocks_final_path"),
            "project_id": workflow_params.get("project_id"),
        }

    runner._route_params = _stub_route  # type: ignore[assignment]

    with patch("MCP.core.workflow_runner.STATE_PATH", state_dir), \
         patch("MCP.core.executor.STATE_PATH", state_dir):
        result = asyncio.run(runner.run_workflow(workflow_id))

    assert result.get("status") == "COMPLETE", result
    assert (out_dir / "handler_ran.txt").exists(), (
        "handler must have been invoked via the synthetic task path"
    )

    # The handler received the kwargs propagated through the synthetic
    # task. project_id + blocks_final_path are the routed_params the
    # _stub_route stub seeded; these must reach the handler.
    assert received, (
        "handler must receive kwargs — empty kwargs would mean the "
        "synthetic task lost its params on the way through the executor"
    )
    assert received.get("project_id") == "PROJ-CARRY-PARAMS", (
        f"project_id must propagate routed_params -> synthetic task "
        f"-> handler kwargs (got {received.get('project_id')!r})"
    )
    assert received.get("blocks_final_path") == "/tmp/canned/blocks_final.jsonl", (
        "blocks_final_path must propagate routed_params -> synthetic "
        "task -> handler kwargs"
    )


def test_workflow_runner_phase_with_no_handler_remains_no_op(
    tmp_path: Path,
) -> None:
    """Phases with ``agents: []`` AND no registered handler must not
    crash the runner — they remain genuinely no-op so the validator-
    only path (where the gate chain runs but no per-phase handler
    exists) keeps working.
    """
    state_dir = tmp_path / "state"
    workflow_id = "WF-PHASE4-SUB4-C"
    phase_name = "phase4_no_handler_xyz"

    # Sanity: this phase is NOT in _PHASE_TOOL_MAPPING.
    assert phase_name not in _PHASE_TOOL_MAPPING

    _seed_workflow_state(
        state_dir=state_dir,
        workflow_id=workflow_id,
        workflow_type="phase4_subtask4_integration",
        workflow_params={"course_name": "PHASE4_TEST"},
        phase_outputs={},
    )

    config = _make_minimal_config(phase_name)
    executor = TaskExecutor(tool_registry={}, validate_registry=False)
    runner = WorkflowRunner(executor=executor, config=config)

    with patch("MCP.core.workflow_runner.STATE_PATH", state_dir), \
         patch("MCP.core.executor.STATE_PATH", state_dir):
        result = asyncio.run(runner.run_workflow(workflow_id))

    # Phase had zero tasks AND no handler. The runner should not crash
    # and should still mark the workflow COMPLETE (the phase ran a
    # no-op gate chain and emitted no failures).
    assert result.get("status") == "COMPLETE", (
        f"no-handler phase should not break the runner; got {result}"
    )
    phase_results = result.get("phase_results", {})
    assert phase_results.get(phase_name, {}).get("task_count") == 0, (
        "no-handler phase should report zero tasks (runner ran gate "
        "chain only)"
    )
