"""Wave 33 Bug B — gates see the current phase's outputs.

Pre-Wave-33 ``TaskExecutor.execute_phase`` ran its gate router
against a stale ``phase_outputs`` dict — the current phase's results
had not yet been extracted. ``_extract_phase_outputs`` ran
post-``execute_phase`` in ``WorkflowRunner.run_workflow``, so every
per-gate builder saw only PRIOR phases' outputs, not the in-progress
phase's just-produced keys.

Live sim-03 surfaced six gates skipping with ``missing inputs: *``
for exactly this reason (keys like ``page_paths``, ``html_paths``,
``chunks_path``, ``manifest_path`` etc. were produced by the
current phase but invisible to the router).

The fix threads an ``extract_phase_outputs_fn`` callable from
WorkflowRunner down into execute_phase, and the executor injects
the current phase's extracted outputs into the ``phase_outputs``
view BEFORE dispatching the gate router.

These tests exercise the ordering contract in-process against a
synthetic TaskExecutor — no live workflow, no filesystem fixtures
beyond tmp_path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.executor import ExecutionResult, TaskExecutor
from MCP.hardening.validation_gates import (
    GateConfig,
    GateIssue,
    GateResult,
    GateSeverity,
)


# ---------------------------------------------------------------------- #
# Helpers: a gate manager that records what phase_outputs the gate saw.
# ---------------------------------------------------------------------- #


class _RecordingGateManager:
    """Minimal gate-manager shim matching the executor's contract.

    ``run_gate(gate, merged_inputs)`` records the ``merged_inputs`` blob
    per-gate so tests can assert on the inputs the router passed in.
    """

    def __init__(self):
        self.captured_inputs: Dict[str, Dict[str, Any]] = {}

    def run_gate(self, gate, merged_inputs):
        self.captured_inputs[gate.gate_id] = dict(merged_inputs)
        return GateResult(
            gate_id=gate.gate_id,
            validator_name=gate.validator_path,
            validator_version="recording-manager",
            passed=True,
            score=1.0,
            issues=[],
        )


class _RecordingGateRouter:
    """Gate router shim that records which phase_outputs it was called
    with and returns a deterministic inputs blob derived from them."""

    def __init__(self):
        # List of (validator_path, phase_outputs_snapshot, workflow_params)
        self.call_log: List = []

    def build(self, validator_path, phase_outputs, workflow_params):
        # Snapshot phase_outputs (deep enough for test asserts on per-phase
        # keys without persisting shared references).
        snapshot = {k: dict(v) if isinstance(v, dict) else v
                    for k, v in phase_outputs.items()}
        self.call_log.append((validator_path, snapshot, dict(workflow_params)))

        # Surface key phase_outputs keys into the inputs blob so the
        # gate manager sees them. If the router can't resolve a key we
        # return it as missing.
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # For test purposes we look up ``page_paths`` — a key the
        # current phase ("content_generation") produces. If the router
        # sees it in phase_outputs, it's wired correctly; if not, the
        # builder reports it as missing (which would trigger the
        # pre-Wave-33 "gate skipped" path).
        cg = snapshot.get("content_generation") or {}
        if "page_paths" in cg:
            inputs["page_paths"] = cg["page_paths"]
        else:
            missing.append("page_paths")
        return inputs, missing


def _wire_executor(
    gate_manager: _RecordingGateManager,
    gate_router: _RecordingGateRouter,
) -> TaskExecutor:
    """Build a TaskExecutor with the recording shims wired in.

    Checkpoint manager + error classifier stay default — the tests
    don't exercise those paths.
    """
    executor = TaskExecutor(tool_registry={}, max_retries=0)
    executor.gate_manager = gate_manager
    executor.gate_input_router = gate_router
    return executor


def _synthetic_extract_fn(phase_name: str, results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
    """Minimal extractor that surfaces ``page_paths`` and ``success``
    from the task results, mirroring the production
    ``WorkflowRunner._extract_phase_outputs`` shape."""
    collected_pages: List[str] = []
    for r in results.values():
        if r.status != "COMPLETE":
            continue
        if isinstance(r.result, dict) and "page_path" in r.result:
            collected_pages.append(r.result["page_path"])
    extracted: Dict[str, Any] = {}
    if collected_pages:
        extracted["page_paths"] = ",".join(collected_pages)
    return extracted


# ---------------------------------------------------------------------- #
# 1. Gate builder must see current phase's extracted outputs.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gate_router_receives_current_phase_outputs():
    """Simulate content_generation: 3 task results, each with its own
    ``page_path``. The gate router for page_objectives-style validators
    must receive a phase_outputs dict that already contains
    ``content_generation.page_paths`` — proving the extraction ran
    BEFORE the router was called (pre-Wave-33 it ran after, so the
    builder raised missing_inputs and the gate skipped)."""
    gate_manager = _RecordingGateManager()
    router = _RecordingGateRouter()
    executor = _wire_executor(gate_manager, router)

    # Synthetic task results — executor didn't run them; we craft them.
    task_results = {
        "T01": ExecutionResult(task_id="T01", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/week01.html"}),
        "T02": ExecutionResult(task_id="T02", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/week02.html"}),
        "T03": ExecutionResult(task_id="T03", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/week03.html"}),
    }

    # Bypass ``_execute_parallel`` by patching it to return our crafted
    # results — we're testing the gate-ordering seam, not parallel exec.
    async def fake_parallel(*args, **kwargs):
        return task_results
    gate_configs = [{
        "gate_id": "page_objectives_under_test",
        "validator": "lib.validators.page_objectives.PageObjectivesValidator",
        "severity": "critical",
        "threshold": {"max_critical_issues": 0},
    }]

    with patch.object(executor, "_execute_parallel", fake_parallel):
        _results, gates_passed, gate_results = await executor.execute_phase(
            workflow_id="W_gate_order_001",
            phase_name="content_generation",
            phase_index=5,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs={},  # No prior phases — only current phase's extraction matters
            workflow_params={"course_name": "TEST_101"},
            extract_phase_outputs_fn=_synthetic_extract_fn,
        )

    # Router must have been called at least once with a phase_outputs
    # dict that includes the current phase's key.
    assert len(router.call_log) == 1
    _validator, phase_outputs_seen, _wparams = router.call_log[0]
    assert "content_generation" in phase_outputs_seen, (
        "Pre-Wave-33 the gate router was called BEFORE extraction, so "
        "phase_outputs never contained a 'content_generation' block. "
        "This assertion catches a regression of that ordering bug."
    )
    assert "page_paths" in phase_outputs_seen["content_generation"]
    # All three page paths surfaced.
    assert (
        phase_outputs_seen["content_generation"]["page_paths"]
        == "/tmp/week01.html,/tmp/week02.html,/tmp/week03.html"
    )

    # Gate manager ran — no "missing inputs" skip.
    assert "page_objectives_under_test" in gate_manager.captured_inputs
    captured = gate_manager.captured_inputs["page_objectives_under_test"]
    assert captured["page_paths"] == (
        "/tmp/week01.html,/tmp/week02.html,/tmp/week03.html"
    )

    # Phase passed gates (the recording manager always returns passed=True).
    assert gates_passed is True
    assert gate_results and len(gate_results) == 1


# ---------------------------------------------------------------------- #
# 2. Router still sees prior phases' outputs alongside current.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_prior_phase_outputs_preserved_alongside_current():
    """The current phase's extraction must MERGE into phase_outputs —
    not replace the prior phases. Both must be visible to the router."""
    gate_manager = _RecordingGateManager()
    router = _RecordingGateRouter()
    executor = _wire_executor(gate_manager, router)

    prior_outputs = {
        "objective_extraction": {
            "_completed": True,
            "textbook_structure_path": "/tmp/structure.json",
            "chapters": 8,
        },
        "staging": {
            "_completed": True,
            "staged_dir": "/tmp/staged",
        },
    }

    task_results = {
        "T01": ExecutionResult(task_id="T01", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/w01.html"}),
    }

    async def fake_parallel(*args, **kwargs):
        return task_results

    gate_configs = [{
        "gate_id": "page_objectives_under_test",
        "validator": "lib.validators.page_objectives.PageObjectivesValidator",
        "severity": "critical",
        "threshold": {},
    }]

    with patch.object(executor, "_execute_parallel", fake_parallel):
        await executor.execute_phase(
            workflow_id="W_gate_order_002",
            phase_name="content_generation",
            phase_index=5,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs=prior_outputs,
            workflow_params={},
            extract_phase_outputs_fn=_synthetic_extract_fn,
        )

    _validator, phase_outputs_seen, _wparams = router.call_log[0]
    # Prior phases still visible.
    assert "objective_extraction" in phase_outputs_seen
    assert phase_outputs_seen["objective_extraction"]["chapters"] == 8
    assert "staging" in phase_outputs_seen
    # Current phase freshly injected.
    assert "content_generation" in phase_outputs_seen
    assert phase_outputs_seen["content_generation"]["page_paths"] == "/tmp/w01.html"


# ---------------------------------------------------------------------- #
# 3. Executor must NOT mutate the caller's phase_outputs dict.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_caller_phase_outputs_not_mutated():
    """``execute_phase`` only uses ``phase_outputs`` as read-only context.

    The WorkflowRunner calls ``_extract_phase_outputs`` AFTER
    ``execute_phase`` returns to persist the canonical extraction —
    the executor's in-place injection for gate routing must NOT leak
    back into the caller's dict (otherwise the next phase starts
    with a corrupt state).
    """
    gate_manager = _RecordingGateManager()
    router = _RecordingGateRouter()
    executor = _wire_executor(gate_manager, router)

    caller_phase_outputs = {
        "staging": {"_completed": True, "staged_dir": "/tmp/staged"},
    }
    caller_copy_before = {
        k: dict(v) if isinstance(v, dict) else v
        for k, v in caller_phase_outputs.items()
    }

    task_results = {
        "T01": ExecutionResult(task_id="T01", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/x.html"}),
    }

    async def fake_parallel(*args, **kwargs):
        return task_results

    gate_configs = [{
        "gate_id": "page_objectives_under_test",
        "validator": "lib.validators.page_objectives.PageObjectivesValidator",
        "severity": "critical",
        "threshold": {},
    }]

    with patch.object(executor, "_execute_parallel", fake_parallel):
        await executor.execute_phase(
            workflow_id="W_gate_order_003",
            phase_name="content_generation",
            phase_index=5,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs=caller_phase_outputs,
            workflow_params={},
            extract_phase_outputs_fn=_synthetic_extract_fn,
        )

    # Caller's dict identical to the pre-call snapshot.
    assert caller_phase_outputs == caller_copy_before, (
        "execute_phase leaked current-phase extraction back into the "
        "caller's phase_outputs dict. WorkflowRunner does its own "
        "extraction + persistence after execute_phase returns — the "
        "executor must not pre-empt that."
    )


# ---------------------------------------------------------------------- #
# 4. Backward compat: no extract_fn → router sees prior phases only.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_extract_fn_preserves_legacy_behaviour():
    """Legacy callers that don't pass ``extract_phase_outputs_fn``
    (e.g., tests / pre-existing Wave 23 code paths) must still work:
    the gate router sees only prior phases' outputs, matching the
    pre-Wave-33 default."""
    gate_manager = _RecordingGateManager()
    router = _RecordingGateRouter()
    executor = _wire_executor(gate_manager, router)

    prior_outputs = {
        "staging": {"_completed": True, "staged_dir": "/tmp/staged"},
    }
    task_results = {
        "T01": ExecutionResult(task_id="T01", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/y.html"}),
    }

    async def fake_parallel(*args, **kwargs):
        return task_results

    gate_configs = [{
        "gate_id": "page_objectives_under_test",
        "validator": "lib.validators.page_objectives.PageObjectivesValidator",
        "severity": "warning",  # warn — so the skip doesn't hard-fail
        "threshold": {},
    }]

    with patch.object(executor, "_execute_parallel", fake_parallel):
        _results, gates_passed, gate_results = await executor.execute_phase(
            workflow_id="W_gate_order_004",
            phase_name="content_generation",
            phase_index=5,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs=prior_outputs,
            workflow_params={},
            # extract_phase_outputs_fn omitted on purpose.
        )

    _validator, phase_outputs_seen, _wparams = router.call_log[0]
    # Current phase NOT in the router's view (backward-compat legacy behaviour).
    assert "content_generation" not in phase_outputs_seen
    # But staging is.
    assert "staging" in phase_outputs_seen


# ---------------------------------------------------------------------- #
# 5. Failed extraction doesn't crash the gate step.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extract_fn_exception_logs_and_continues():
    """If the extractor raises, the executor logs a warning and
    proceeds as if no extraction ran — a bug in the extractor must
    not take down the entire phase."""
    gate_manager = _RecordingGateManager()
    router = _RecordingGateRouter()
    executor = _wire_executor(gate_manager, router)

    def broken_extract(phase_name, results):
        raise RuntimeError("Deliberate extractor failure for test")

    task_results = {
        "T01": ExecutionResult(task_id="T01", status="COMPLETE",
                               result={"success": True, "page_path": "/tmp/z.html"}),
    }

    async def fake_parallel(*args, **kwargs):
        return task_results

    gate_configs = [{
        "gate_id": "page_objectives_under_test",
        "validator": "lib.validators.page_objectives.PageObjectivesValidator",
        "severity": "warning",
        "threshold": {},
    }]

    with patch.object(executor, "_execute_parallel", fake_parallel):
        _results, gates_passed, _gate_results = await executor.execute_phase(
            workflow_id="W_gate_order_005",
            phase_name="content_generation",
            phase_index=5,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs={},
            workflow_params={},
            extract_phase_outputs_fn=broken_extract,
        )

    # Router still called — gates still evaluated (they'll skip with
    # "missing inputs", same as legacy behaviour).
    assert len(router.call_log) == 1


# ---------------------------------------------------------------------- #
# 6. Regression: the three-location workflow must still validate.
# ---------------------------------------------------------------------- #


def test_execute_phase_signature_includes_extract_fn():
    """Regression guard: a future contributor who drops the
    ``extract_phase_outputs_fn`` parameter from ``execute_phase`` would
    silently revert the Wave 33 Bug B fix. Assert the parameter exists
    in the live signature."""
    import inspect

    sig = inspect.signature(TaskExecutor.execute_phase)
    assert "extract_phase_outputs_fn" in sig.parameters, (
        "execute_phase must accept an extract_phase_outputs_fn kwarg "
        "(Wave 33 Bug B). Dropping it reverts the gate-ordering fix."
    )
    # Default must be None so legacy callers still work.
    assert sig.parameters["extract_phase_outputs_fn"].default is None
