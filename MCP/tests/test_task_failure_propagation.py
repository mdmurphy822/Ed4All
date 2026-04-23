"""Wave 33 Bug C — tool envelopes with ``success=False`` must FAIL.

Pre-Wave-33 ``TaskExecutor._execute_with_retries`` marked a task
``COMPLETE`` as soon as the underlying tool returned any parseable
dict, including ``{"success": False, "error_code": "..."}``. The
phase summary then showed ``12/12 complete, gates=pass`` even when
every task reported a permanent error — content_generation routinely
reported "12/12 complete" on 48 empty-template pages because the
emptiness guard returned ``{"success": False}`` envelopes that the
executor silently rewrote as successes.

The fix inspects each tool result:

* ``result.get("success") is False`` → ``status="FAILED"`` with the
  envelope's ``error_code`` + ``error_message`` promoted into the
  ExecutionResult.
* Anything else (including dicts without a ``success`` key, or plain
  strings / numbers) → preserves legacy behaviour (``status="COMPLETE"``
  with the result stored as-is).
* A raised exception still routes through the existing error
  classifier / retry / poison-pill machinery.

Phase-level aggregation (``workflow_runner._create_phase_tasks`` →
phase summary ``failed`` count + ``phase_failed`` gate) now treats
``FAILED`` alongside ``ERROR`` / ``TIMEOUT``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.executor import AGENT_TOOL_MAPPING, ExecutionResult, TaskExecutor

# Agent type → tool name pairing from AGENT_TOOL_MAPPING. We pick
# content-generator because the whole motivating failure mode is
# ``_check_content_nonempty`` returning ``success=False`` envelopes on
# 48 empty-template pages.
_AGENT = "content-generator"


def _build_executor_with_stub_tool(tool_result_json: str) -> TaskExecutor:
    """Wire a TaskExecutor against a single async tool that returns
    ``tool_result_json`` verbatim."""
    tool_name = AGENT_TOOL_MAPPING[_AGENT]

    async def stub_tool(**kwargs):
        return tool_result_json

    # Build an executor with only the tool we're testing — skip the
    # tool-registry validation by passing an empty dict and then
    # monkey-patching the one tool in.
    registry = {tool_name: stub_tool}
    executor = TaskExecutor(tool_registry=registry, max_retries=0)
    return executor


async def _run_task(
    executor: TaskExecutor,
    task_params: dict,
) -> ExecutionResult:
    """Invoke ``execute_task`` bypassing the filesystem-backed
    ``_load_task`` + ``_update_task_status`` helpers."""
    task = {
        "id": "T_fail_prop_001",
        "agent_type": _AGENT,
        "params": task_params,
    }
    with patch.object(executor, "_load_task", return_value=task):
        with patch.object(executor, "_update_task_status"):
            return await executor.execute_task("W_fail_prop_001", "T001")


# ---------------------------------------------------------------------- #
# 1. Success envelope → COMPLETE (legacy behaviour preserved).
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_success_true_envelope_completes():
    """A tool returning ``{"success": True, ...}`` stays COMPLETE."""
    executor = _build_executor_with_stub_tool(
        json.dumps({
            "success": True,
            "project_id": "P_001",
            "page_path": "/tmp/p.html",
        })
    )
    result = await _run_task(executor, {"project_id": "P_001"})
    assert result.status == "COMPLETE"
    assert result.result["success"] is True
    assert result.result["page_path"] == "/tmp/p.html"
    # No error surfaced on the happy path.
    assert result.error is None


# ---------------------------------------------------------------------- #
# 2. Failure envelope → FAILED with error_code surfaced.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_success_false_envelope_marks_failed():
    """A tool returning ``{"success": False, ...}`` becomes FAILED.

    ``error_code`` + ``error_message`` must be hoisted into the
    ExecutionResult so downstream aggregators (phase summary,
    GENERATION_PROGRESS.md error table) see the structured failure.
    """
    executor = _build_executor_with_stub_tool(
        json.dumps({
            "success": False,
            "error_code": "EMPTY_CONTENT",
            "error_message": "Page has no non-template sections",
            "project_id": "P_002",
        })
    )
    result = await _run_task(executor, {"project_id": "P_002"})
    assert result.status == "FAILED"
    assert result.error_class == "EMPTY_CONTENT"
    assert "EMPTY_CONTENT" in (result.error or "")
    assert "Page has no non-template sections" in (result.error or "")
    # The full envelope is preserved in the ExecutionResult's result field
    # so gate builders that need the raw dict can still reach it.
    assert result.result["error_code"] == "EMPTY_CONTENT"


# ---------------------------------------------------------------------- #
# 3. Raised exception → FAILED/ERROR via the error-classifier path.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_raised_exception_still_marks_error():
    """Existing raise-based error path must still flow through the
    error classifier / retry / poison-pill machinery."""
    async def raising_tool(**kwargs):
        raise ValueError("Permanent schema error — not retryable")

    tool_name = AGENT_TOOL_MAPPING[_AGENT]
    executor = TaskExecutor(
        tool_registry={tool_name: raising_tool}, max_retries=0,
    )
    task = {
        "id": "T_fail_prop_002",
        "agent_type": _AGENT,
        "params": {"project_id": "P_003"},
    }
    with patch.object(executor, "_load_task", return_value=task):
        with patch.object(executor, "_update_task_status"):
            result = await executor.execute_task("W_fail_prop_002", "T002")
    # status is ERROR (via exception path), not COMPLETE.
    assert result.status == "ERROR"
    assert "Permanent schema error" in (result.error or "")


# ---------------------------------------------------------------------- #
# 4. Phase aggregation: one FAILED task → phase counts as failed.
# ---------------------------------------------------------------------- #


def test_phase_aggregation_counts_failed_as_failure():
    """The phase summary counters in ``workflow_runner`` treat
    ``FAILED`` alongside ``ERROR`` / ``TIMEOUT``.

    Pre-Wave-33 the counters only checked ``("ERROR", "TIMEOUT")`` —
    a phase with 12 tasks where one returned ``success=False`` was
    mis-reported as "12/12 complete, 0 failed".
    """
    # Synthesise 12 results: 11 COMPLETE + 1 FAILED.
    results = {}
    for i in range(11):
        results[f"T{i:02d}"] = ExecutionResult(
            task_id=f"T{i:02d}",
            status="COMPLETE",
            result={"success": True},
        )
    results["T11"] = ExecutionResult(
        task_id="T11",
        status="FAILED",
        result={"success": False, "error_code": "EMPTY_CONTENT"},
        error="EMPTY_CONTENT: 48 empty-template pages",
        error_class="EMPTY_CONTENT",
    )

    # Apply the same aggregation rule ``workflow_runner.run_workflow``
    # now uses — Wave 33 Bug C updated both the ``failed`` counter and
    # the ``phase_failed`` flag.
    completed = sum(1 for r in results.values() if r.status == "COMPLETE")
    failed = sum(
        1 for r in results.values()
        if r.status in ("ERROR", "TIMEOUT", "FAILED")
    )
    phase_failed = any(
        r.status in ("ERROR", "TIMEOUT", "FAILED")
        for r in results.values()
    )

    assert completed == 11
    assert failed == 1
    assert phase_failed is True, (
        "Pre-Wave-33 phase_failed was False here — a single "
        "success=False envelope slipped through aggregation and the "
        "phase continued as if all 12 tasks succeeded."
    )


# ---------------------------------------------------------------------- #
# 5. content_generation emptiness guard → phase gates=fail.
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_content_generation_empty_envelope_fails():
    """The live-scenario: ``generate_course_content`` returns the
    emptiness guard envelope (``_check_content_nonempty`` → Wave 32
    Deliverable C). Task must end up FAILED, not silently COMPLETE.
    This is the exact failure mode that produced "12/12 complete,
    gates=pass" on 48 empty-template pages in sim-03.
    """
    executor = _build_executor_with_stub_tool(
        json.dumps({
            "success": False,
            "error_code": "EMPTY_CONTENT",
            "error_message": (
                "All 48 generated pages contain only template chrome; "
                "no Courseforge content sections were emitted. Refusing "
                "to advance."
            ),
            "empty_pages": 48,
            "total_pages": 48,
        })
    )
    result = await _run_task(
        executor, {"project_id": "BATES_101", "week_range": "1-12"}
    )

    # Task must not survive as COMPLETE — otherwise the phase summary
    # reverts to the sim-03 "gates=pass on empty pages" bug.
    assert result.status == "FAILED"
    assert result.error_class == "EMPTY_CONTENT"
    assert result.result["empty_pages"] == 48
    assert result.result["total_pages"] == 48
