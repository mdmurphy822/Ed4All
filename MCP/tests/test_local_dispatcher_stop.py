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

from lib.generation.stop_control import GracefulStopRequested, RUN_SENTINEL_NAME
from MCP.core.executor import AGENT_SUBAGENT_SET, TaskExecutor
from MCP.orchestrator import local_dispatcher as _ld
from MCP.orchestrator.local_dispatcher import (
    GRACEFUL_STOP_ERROR_CODE,
    LocalDispatcher,
    _graceful_stop_layer,
    _mailbox_grace_seconds,
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


# ============================================================================
# P5 item 3 — mailbox-waiter deadline: sentinel-then-grace, NOT immediate
# MAILBOX_TIMEOUT.  On deadline the waiter writes a run-scoped stop sentinel
# (``request_stop``) and drains a short grace window; a completion (normal or
# GRACEFUL_STOP) arriving within grace is handled normally / as PAUSED, and only
# past grace does it surface MAILBOX_TIMEOUT.
# ============================================================================


def _sentinel_path(state_dir, run_id):
    """Run-scoped stop sentinel path under the isolated state/runs dir."""
    return state_dir / run_id / "control" / RUN_SENTINEL_NAME


def _make_operator_after_sentinel(mb, sentinel_path, envelope, budget_s=5.0):
    """Thread body: wait until the deadline sentinel appears, THEN complete.

    Completing only after the sentinel exists guarantees the completion lands in
    the grace window (post-deadline), so the test deterministically exercises
    the sentinel-then-grace path rather than a pre-deadline normal completion.
    """

    def _run():
        end = time.monotonic() + budget_s
        while time.monotonic() < end:
            if sentinel_path.exists():
                pending = mb.list_pending()
                if pending:
                    task_id = pending[0]
                    mb.claim(task_id)
                    mb.complete(task_id, envelope)
                return
            time.sleep(0.01)

    return _run


def test_mailbox_grace_seconds_formula():
    """Grace = min(600, int(0.10 * timeout)); sub-10s deadlines truncate to 0."""
    assert _mailbox_grace_seconds(1800.0) == 180.0
    assert _mailbox_grace_seconds(10000.0) == 600.0  # capped
    assert _mailbox_grace_seconds(0.3) == 0.0        # tiny (test) deadline
    assert _mailbox_grace_seconds(10.0) == 1.0


@pytest.mark.asyncio
async def test_deadline_grace_normal_completion_not_timeout(
    monkeypatch, state_runs_isolated
):
    """Deadline expiry writes the sentinel; a NORMAL completion arriving within
    grace is handled normally — the waiter must NOT return MAILBOX_TIMEOUT."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "0.3")
    # Tiny deadline → formula grace would be 0; force a generous grace so the
    # operator has time to respond (module fn so this is honored at call time).
    monkeypatch.setattr(_ld, "_mailbox_grace_seconds", lambda _t: 3.0)

    state_dir = state_runs_isolated
    run_id = "GRACE_OK"
    disp = LocalDispatcher(mailbox_base_dir=state_dir, mailbox_poll_interval=0.02)
    mb = TaskMailbox(run_id=run_id, base_dir=state_dir)
    sentinel = _sentinel_path(state_dir, run_id)

    op = threading.Thread(
        target=_make_operator_after_sentinel(
            mb,
            sentinel,
            {
                "success": True,
                "result": {"success": True, "outputs": {"ok": 1}, "artifacts": []},
            },
        ),
        daemon=True,
    )
    op.start()
    try:
        result = await disp.dispatch_task(
            task_name="run_content_generation_outline",
            agent_type="content-generator",
            task_params={"project_id": "P"},
            run_id=run_id,
        )
    finally:
        op.join(timeout=2.0)

    # Handled normally — not the timeout envelope.
    assert result["success"] is True
    assert result.get("error_code") != "MAILBOX_TIMEOUT"
    assert result["outputs"] == {"ok": 1}
    # The deadline DID trip and write the sentinel (proves the grace path ran).
    assert sentinel.exists()


@pytest.mark.asyncio
async def test_deadline_grace_graceful_stop_pauses(
    monkeypatch, state_runs_isolated
):
    """A GRACEFUL_STOP completion arriving within grace re-raises the pause
    channel (PAUSED), never MAILBOX_TIMEOUT."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "0.3")
    monkeypatch.setattr(_ld, "_mailbox_grace_seconds", lambda _t: 3.0)

    state_dir = state_runs_isolated
    run_id = "GRACE_PAUSE"
    disp = LocalDispatcher(mailbox_base_dir=state_dir, mailbox_poll_interval=0.02)
    mb = TaskMailbox(run_id=run_id, base_dir=state_dir)
    sentinel = _sentinel_path(state_dir, run_id)

    op = threading.Thread(
        target=_make_operator_after_sentinel(
            mb,
            sentinel,
            {
                "success": False,
                "result": {
                    "success": False,
                    "error_code": GRACEFUL_STOP_ERROR_CODE,
                    "site_id": "outline_tier",
                    "units_completed": 6,
                },
            },
        ),
        daemon=True,
    )
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
    assert ei.value.units_completed == 6
    assert sentinel.exists()


@pytest.mark.asyncio
async def test_deadline_grace_expires_returns_timeout(
    monkeypatch, state_runs_isolated
):
    """No completion within deadline+grace → MAILBOX_TIMEOUT after grace, and
    the deadline still wrote the stop sentinel."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setattr(_ld, "_mailbox_grace_seconds", lambda _t: 0.2)

    state_dir = state_runs_isolated
    run_id = "GRACE_TIMEOUT"
    disp = LocalDispatcher(mailbox_base_dir=state_dir, mailbox_poll_interval=0.02)
    sentinel = _sentinel_path(state_dir, run_id)

    result = await disp.dispatch_task(
        task_name="generate_course_content",
        agent_type="content-generator",
        task_params={"project_id": "P"},
        run_id=run_id,
    )
    assert result["success"] is False
    assert result["error_code"] == "MAILBOX_TIMEOUT"
    assert result["mailbox_task_id"].startswith("content-generator-")
    # Sentinel was written on deadline even though grace also elapsed.
    assert sentinel.exists()
