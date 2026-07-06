"""P3.M item 2 — LocalDispatcher marshals a GRACEFUL_STOP envelope back into
the pause channel.

In ``--mode local`` the big LLM loops run in the mailbox *servicer* process;
when one hits a stop sentinel it raises ``GracefulStopRequested``, which the
servicer marshals into a completion envelope carrying
``error_code == "GRACEFUL_STOP"``. On the orchestrator side, the dispatcher
must re-raise that as ``GracefulStopRequested`` (BEFORE the generic
success=False passthrough) so the executor stamps the task PAUSED — no retry,
no poison, not FAILED.

The envelope boundary is what's under test (same-process, tmp mailbox dir);
real processes aren't needed.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict

import pytest

from lib.generation.stop_control import GracefulStopRequested
from MCP.core.executor import AGENT_SUBAGENT_SET, TaskExecutor
from MCP.orchestrator.local_dispatcher import (
    GRACEFUL_STOP_ERROR_CODE,
    LocalDispatcher,
    _graceful_stop_layer,
    _raise_if_graceful_stop,
)
from MCP.orchestrator.task_mailbox import TaskMailbox


# ---------------------------------------------------------- envelope helpers


def test_graceful_stop_layer_detects_flat_and_nested():
    flat = {"success": False, "error_code": GRACEFUL_STOP_ERROR_CODE, "site_id": "s"}
    nested = {"success": False, "result": {"error_code": GRACEFUL_STOP_ERROR_CODE}}
    plain = {"success": False, "error_code": "TOOL_RAISED"}

    assert _graceful_stop_layer(flat) is flat
    assert _graceful_stop_layer(nested) is nested["result"]
    assert _graceful_stop_layer(plain) is None
    assert _graceful_stop_layer("not a dict") is None


def test_raise_if_graceful_stop_carries_payload():
    env = {
        "success": False,
        "error_code": GRACEFUL_STOP_ERROR_CODE,
        "site_id": "outline_tier",
        "units_completed": 7,
    }
    with pytest.raises(GracefulStopRequested) as ei:
        _raise_if_graceful_stop(env, task_id="T1")
    assert ei.value.site_id == "outline_tier"
    assert ei.value.units_completed == 7


def test_raise_if_graceful_stop_noop_on_plain_failure():
    # A normal failure envelope must NOT be converted to a pause.
    _raise_if_graceful_stop({"success": False, "error_code": "TOOL_RAISED"})
    _raise_if_graceful_stop({"success": True})


def test_tool_dict_from_envelope_raises_on_nested_stop():
    """The servicer wraps the tool dict under ``result``; the dispatcher
    still detects the nested GRACEFUL_STOP and raises."""
    envelope = {
        "success": False,
        "result": {
            "success": False,
            "error_code": GRACEFUL_STOP_ERROR_CODE,
            "site_id": "rewrite_tier",
            "units_completed": 3,
        },
    }
    with pytest.raises(GracefulStopRequested) as ei:
        LocalDispatcher._tool_dict_from_envelope(envelope, "task-abc")
    assert ei.value.units_completed == 3


# ------------------------------------------------- callable-path round trip


@pytest.mark.asyncio
async def test_dispatch_task_callable_stop_envelope_raises(tmp_path):
    """agent_tool returns a flat GRACEFUL_STOP envelope → dispatch_task
    re-raises the pause channel (not a returned failure dict)."""

    async def stop_agent_tool(request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": GRACEFUL_STOP_ERROR_CODE,
            "site_id": "concept_windows",
            "units_completed": 4,
        }

    disp = LocalDispatcher(agent_tool=stop_agent_tool, mailbox_base_dir=tmp_path)
    with pytest.raises(GracefulStopRequested) as ei:
        await disp.dispatch_task(
            task_name="run_content_generation_outline",
            agent_type="content-generator",
            task_params={"project_id": "P"},
            run_id="R",
        )
    assert ei.value.site_id == "concept_windows"


# -------------------------------------------------- mailbox-path round trip


@pytest.mark.asyncio
async def test_dispatch_task_mailbox_stop_envelope_raises(monkeypatch, tmp_path):
    """An operator (servicer) writes a nested GRACEFUL_STOP completion →
    the mailbox dispatch path re-raises GracefulStopRequested."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "5")

    disp = LocalDispatcher(mailbox_base_dir=tmp_path, mailbox_poll_interval=0.02)
    run_id = "MB_STOP"
    mb = TaskMailbox(run_id=run_id, base_dir=tmp_path)

    def operator_thread() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pending = mb.list_pending()
            if pending:
                task_id = pending[0]
                mb.claim(task_id)
                mb.complete(
                    task_id,
                    {
                        "success": False,
                        "result": {
                            "success": False,
                            "error_code": GRACEFUL_STOP_ERROR_CODE,
                            "site_id": "stage2_windows",
                            "units_completed": 5,
                        },
                    },
                )
                return
            time.sleep(0.02)

    op = threading.Thread(target=operator_thread, daemon=True)
    op.start()
    try:
        with pytest.raises(GracefulStopRequested) as ei:
            await disp.dispatch_task(
                task_name="run_content_generation_rewrite",
                agent_type="content-generator",
                task_params={"project_id": "P"},
                run_id=run_id,
            )
    finally:
        op.join(timeout=2.0)
    assert ei.value.units_completed == 5


# ------------------------------------------- executor maps the raise → PAUSED


@pytest.mark.asyncio
async def test_executor_maps_stop_envelope_to_paused(monkeypatch, state_runs_isolated):
    """End-to-end envelope → ExecutionResult PAUSED: a real LocalDispatcher
    whose agent_tool returns a GRACEFUL_STOP envelope, routed through the
    executor's dispatch fork. The Wave-A ``except GracefulStopRequested``
    handler stamps PAUSED with no retry / no poison."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    assert "content-generator" in AGENT_SUBAGENT_SET

    async def stop_agent_tool(request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "error_code": GRACEFUL_STOP_ERROR_CODE,
            "site_id": "outline_tier",
            "units_completed": 2,
        }

    disp = LocalDispatcher(agent_tool=stop_agent_tool)

    async def _dummy_tool(**kwargs) -> str:  # pragma: no cover - not reached
        return "{}"

    ex = TaskExecutor(
        tool_registry={"generate_course_content": _dummy_tool},
        dispatcher=disp,
        run_id="EXEC_STOP",
    )

    result = await ex._execute_with_retries(
        "task-1",
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "P", "course_name": "C"},
        },
    )

    assert result.status == "PAUSED"
    assert result.retry_count == 0
    assert result.error_class == "paused"
