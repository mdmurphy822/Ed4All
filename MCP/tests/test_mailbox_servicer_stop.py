"""P3.M item 1 — the mailbox servicer marshals a graceful stop instead of
swallowing it into a retried TOOL_RAISED failure.

Two guarantees:

1. A servicer-executed tool that raises ``GracefulStopRequested`` (a Wave-B
   pt.py loop hitting its own stop check) is completed with a
   ``error_code == "GRACEFUL_STOP"`` envelope — caught by a dedicated
   ``except`` BEFORE the catch-all that would otherwise stamp TOOL_RAISED
   (→ transient-classified → retried 3× → phase FAILED). Round-tripping that
   envelope through the dispatcher re-raises the pause channel.
2. Pre-claim check: once a stop sentinel is up the servicer claims / executes
   NO new tasks (the in-flight one, if any, still finishes first).

The servicer hardcodes its mailbox under ``ROOT/state/runs`` while the sentinel
resolves through ``get_state_runs_dir()`` — the tests pin both together
(monkeypatch ``ROOT`` + ``state_runs_isolated`` sets ``ED4ALL_STATE_RUNS_DIR``)
to also exercise risk R2 (foreign-CWD path resolution).
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

import scripts.ops.mailbox_servicer as servicer
from lib.generation import stop_control
from lib.generation.stop_control import GracefulStopRequested
from MCP.orchestrator.local_dispatcher import (
    GRACEFUL_STOP_ERROR_CODE,
    LocalDispatcher,
)
from MCP.orchestrator.task_mailbox import TaskMailbox


@pytest.fixture
def wired_servicer(monkeypatch, tmp_path, state_runs_isolated):
    """Point the servicer's hardcoded ROOT at tmp_path so its mailbox and the
    stop sentinel (ED4ALL_STATE_RUNS_DIR) resolve to the same runtime/state/runs."""
    monkeypatch.setattr(servicer, "ROOT", tmp_path)
    return tmp_path


def _mb(run_id: str) -> TaskMailbox:
    return TaskMailbox(run_id=run_id)  # reads ED4ALL_STATE_RUNS_DIR


# ------------------------------------------------ tool raises → GRACEFUL_STOP


async def test_servicer_marshals_graceful_stop_envelope(wired_servicer, monkeypatch):
    run_id = "SVC_STOP"

    async def raising_tool(**kwargs: Any) -> Dict[str, Any]:
        raise GracefulStopRequested("outline_tier", 6)

    monkeypatch.setattr(servicer, "REGISTRY", {"raising_tool": raising_tool})
    monkeypatch.setattr(servicer, "DETERMINISTIC_TOOLS", {"raising_tool"})

    mb = _mb(run_id)
    mb.put_pending(
        "svc-task-1",
        {"kind": "agent_task", "agent_type": "content-generator",
         "tool_name": "raising_tool", "task_params": {}},
    )

    serviced = await servicer.drain(run_id, 0.0)
    assert serviced == 1  # the stopped task counts as serviced (envelope written)

    envelope = mb.read_completion("svc-task-1")
    assert envelope["success"] is False
    inner = envelope["result"]
    assert inner["error_code"] == GRACEFUL_STOP_ERROR_CODE
    assert inner["units_completed"] == 6
    assert inner["site_id"] == "outline_tier"

    # Round-trip: the dispatcher re-raises the pause channel from that envelope.
    with pytest.raises(GracefulStopRequested) as ei:
        LocalDispatcher._tool_dict_from_envelope(envelope, "svc-task-1")
    assert ei.value.units_completed == 6


async def test_servicer_stop_is_not_tool_raised(wired_servicer, monkeypatch):
    """Regression guard: the pause must NOT be recorded as TOOL_RAISED (the
    old catch-all behavior that got retried + stamped FAILED)."""
    run_id = "SVC_NOTRAISED"

    async def raising_tool(**kwargs: Any) -> Dict[str, Any]:
        raise GracefulStopRequested("rewrite_tier", 1)

    monkeypatch.setattr(servicer, "REGISTRY", {"raising_tool": raising_tool})
    monkeypatch.setattr(servicer, "DETERMINISTIC_TOOLS", {"raising_tool"})

    mb = _mb(run_id)
    mb.put_pending(
        "t", {"kind": "agent_task", "agent_type": "a",
              "tool_name": "raising_tool", "task_params": {}},
    )
    await servicer.drain(run_id, 0.0)

    inner = mb.read_completion("t")["result"]
    assert inner["error_code"] != "TOOL_RAISED"
    assert inner["error_code"] == GRACEFUL_STOP_ERROR_CODE


# --------------------------------------------------------- pre-claim check


async def test_servicer_pre_armed_sentinel_claims_nothing(wired_servicer, monkeypatch):
    """Sentinel up before drain → the servicer executes 0 tasks and writes no
    completion (it refuses to claim new work)."""
    run_id = "SVC_PREARM"
    calls: list = []

    async def counting_tool(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {"success": True, "outputs": {}}

    monkeypatch.setattr(servicer, "REGISTRY", {"counting_tool": counting_tool})
    monkeypatch.setattr(servicer, "DETERMINISTIC_TOOLS", {"counting_tool"})

    mb = _mb(run_id)
    mb.put_pending(
        "pending-1",
        {"kind": "agent_task", "agent_type": "a",
         "tool_name": "counting_tool", "task_params": {}},
    )
    stop_control.request_stop(run_id=run_id, reason="test", source="test")

    serviced = await servicer.drain(run_id, 0.0)

    assert calls == [], "pre-armed sentinel must block tool execution"
    assert serviced == 0
    assert mb.list_completed() == []
    # The task is left unclaimed for a resume leg.
    assert "pending-1" in mb.list_pending()


async def test_servicer_no_sentinel_baseline_runs(wired_servicer, monkeypatch):
    """Control: with no sentinel the same task IS executed and completed —
    proving the pre-claim gate (not some other short-circuit) is what blocks."""
    run_id = "SVC_BASELINE"
    calls: list = []

    async def counting_tool(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {"success": True, "outputs": {}}

    monkeypatch.setattr(servicer, "REGISTRY", {"counting_tool": counting_tool})
    monkeypatch.setattr(servicer, "DETERMINISTIC_TOOLS", {"counting_tool"})

    mb = _mb(run_id)
    mb.put_pending(
        "pending-1",
        {"kind": "agent_task", "agent_type": "a",
         "tool_name": "counting_tool", "task_params": {}},
    )

    serviced = await servicer.drain(run_id, 0.0)

    assert len(calls) == 1
    assert serviced == 1
    assert mb.read_completion("pending-1")["success"] is True
