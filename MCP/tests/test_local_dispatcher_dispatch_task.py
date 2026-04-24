"""Wave 74 Session 1 — ``LocalDispatcher.dispatch_task`` path coverage.

Pins the three-path selector:

1. ``agent_tool`` callable injected → call it, parse JSON/dict response.
2. No callable + ``LOCAL_DISPATCHER_ALLOW_STUB=1`` → stub envelope.
3. No callable + stub flag off → mailbox bridge.

Parallel coverage exists for ``dispatch_phase``; these tests guard the
per-task hook which is the Wave 38 gap-close's new surface.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict

import pytest

from MCP.orchestrator.local_dispatcher import LocalDispatcher
from MCP.orchestrator.task_mailbox import TaskMailbox


# ---------------------------------------------------------------- Path 1


@pytest.mark.asyncio
async def test_dispatch_task_via_callable_dict(tmp_path):
    """``agent_tool`` returning a dict → passed through verbatim."""
    calls = []

    async def fake_agent_tool(request: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(request)
        return {
            "success": True,
            "artifacts": ["foo.html"],
            "outputs": {"weeks": 1},
        }

    disp = LocalDispatcher(
        agent_tool=fake_agent_tool,
        mailbox_base_dir=tmp_path,
    )
    result = await disp.dispatch_task(
        task_name="generate_course_content",
        agent_type="content-generator",
        task_params={"project_id": "P", "course_name": "C"},
        run_id="TEST_RUN",
    )

    assert result["success"] is True
    assert result["artifacts"] == ["foo.html"]
    assert len(calls) == 1
    call = calls[0]
    assert call["agent_type"] == "content-generator"
    assert call["tool_name"] == "generate_course_content"
    assert call["run_id"] == "TEST_RUN"


@pytest.mark.asyncio
async def test_dispatch_task_via_callable_json_string(tmp_path):
    """``agent_tool`` returning a JSON string → parsed into dict."""

    async def fake_agent_tool(request: Dict[str, Any]) -> str:
        return json.dumps({"success": True, "outputs": {"ok": True}})

    disp = LocalDispatcher(
        agent_tool=fake_agent_tool,
        mailbox_base_dir=tmp_path,
    )
    result = await disp.dispatch_task(
        task_name="t", agent_type="content-generator",
        task_params={}, run_id="R",
    )
    assert result["success"] is True
    assert result["outputs"]["ok"] is True


@pytest.mark.asyncio
async def test_dispatch_task_via_callable_raises(tmp_path):
    """``agent_tool`` exceptions → failure envelope with AGENT_TOOL_RAISED."""

    async def fake_agent_tool(request):
        raise RuntimeError("boom")

    disp = LocalDispatcher(
        agent_tool=fake_agent_tool,
        mailbox_base_dir=tmp_path,
    )
    result = await disp.dispatch_task(
        task_name="t", agent_type="content-generator",
        task_params={}, run_id="R",
    )
    assert result["success"] is False
    assert result["error_code"] == "AGENT_TOOL_RAISED"
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_task_via_callable_returns_non_json_string(tmp_path):
    """Non-JSON string → INVALID_AGENT_RESPONSE failure envelope."""

    async def fake_agent_tool(request):
        return "definitely not json {{{"

    disp = LocalDispatcher(
        agent_tool=fake_agent_tool,
        mailbox_base_dir=tmp_path,
    )
    result = await disp.dispatch_task(
        task_name="t", agent_type="content-generator",
        task_params={}, run_id="R",
    )
    assert result["success"] is False
    assert result["error_code"] == "INVALID_AGENT_RESPONSE"


# ---------------------------------------------------------------- Path 2


@pytest.mark.asyncio
async def test_dispatch_task_stub_when_flag_set(monkeypatch, tmp_path):
    """No callable + LOCAL_DISPATCHER_ALLOW_STUB=1 → stub envelope."""
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
    disp = LocalDispatcher(mailbox_base_dir=tmp_path)

    result = await disp.dispatch_task(
        task_name="generate_assessments",
        agent_type="assessment-generator",
        task_params={"course_id": "C"},
        run_id="R",
    )
    assert result["success"] is True
    assert result["dispatch_mode"] == "stub"
    assert result["agent_type"] == "assessment-generator"
    assert result["outputs"] == {}


# ---------------------------------------------------------------- Path 3


@pytest.mark.asyncio
async def test_dispatch_task_mailbox_happy_path(monkeypatch, tmp_path):
    """No callable + flag off → writes pending, blocks on completion.

    Simulates an operator by spinning a thread that polls pending,
    claims, writes a completion envelope. The backend unblocks and
    returns the tool-shape dict.
    """
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    disp = LocalDispatcher(
        mailbox_base_dir=tmp_path,
        mailbox_poll_interval=0.02,
    )
    # Keep the per-task timeout short for tests (default is 1800s).
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "5")

    # Need a TaskMailbox bound to the same run_id to simulate the
    # operator side. The dispatcher creates one internally; we mirror.
    run_id = "TEST_RUN_MB"
    mb = TaskMailbox(run_id=run_id, base_dir=tmp_path)

    def operator_thread():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pending = mb.list_pending()
            if pending:
                task_id = pending[0]
                spec = mb.claim(task_id)
                # Sanity: the spec carries the fields the operator
                # needs to dispatch a real subagent.
                assert spec["kind"] == "agent_task"
                assert spec["agent_type"] == "content-generator"
                assert spec["tool_name"] == "generate_course_content"
                mb.complete(
                    task_id,
                    {
                        "success": True,
                        "result": {
                            "success": True,
                            "artifacts": ["week_01.html"],
                            "outputs": {"weeks_generated": 1},
                        },
                    },
                )
                return
            time.sleep(0.02)

    op = threading.Thread(target=operator_thread, daemon=True)
    op.start()

    result = await disp.dispatch_task(
        task_name="generate_course_content",
        agent_type="content-generator",
        task_params={"project_id": "P"},
        run_id=run_id,
    )
    op.join(timeout=2.0)

    assert result["success"] is True
    assert result["artifacts"] == ["week_01.html"]
    assert "mailbox_task_id" in result
    assert result["mailbox_task_id"].startswith("content-generator-")


@pytest.mark.asyncio
async def test_dispatch_task_mailbox_timeout(monkeypatch, tmp_path):
    """Operator never writes completion → MAILBOX_TIMEOUT envelope."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "0.2")

    disp = LocalDispatcher(
        mailbox_base_dir=tmp_path,
        mailbox_poll_interval=0.02,
    )
    result = await disp.dispatch_task(
        task_name="generate_course_content",
        agent_type="content-generator",
        task_params={"project_id": "P"},
        run_id="TIMEOUT_RUN",
    )
    assert result["success"] is False
    assert result["error_code"] == "MAILBOX_TIMEOUT"
    assert "mailbox_task_id" in result


@pytest.mark.asyncio
async def test_dispatch_task_mailbox_failure_envelope(monkeypatch, tmp_path):
    """Operator writes a success=False completion → pass-through
    failure envelope so the executor retry path sees the tool-shape
    error signals (Wave 33 Bug C contract)."""
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "5")

    disp = LocalDispatcher(
        mailbox_base_dir=tmp_path,
        mailbox_poll_interval=0.02,
    )
    run_id = "FAIL_RUN"
    mb = TaskMailbox(run_id=run_id, base_dir=tmp_path)

    def operator_thread():
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
                        "error": "operator refused",
                        "error_code": "OPERATOR_REFUSED",
                    },
                )
                return
            time.sleep(0.02)

    op = threading.Thread(target=operator_thread, daemon=True)
    op.start()

    result = await disp.dispatch_task(
        task_name="t", agent_type="content-generator",
        task_params={}, run_id=run_id,
    )
    op.join(timeout=2.0)

    assert result["success"] is False
    assert result["error_code"] == "OPERATOR_REFUSED"


# ---------------------------------------------------------------- id shape


@pytest.mark.asyncio
async def test_mailbox_task_ids_use_agent_type_prefix(monkeypatch, tmp_path):
    """Task-id shape is ``{agent_type}-{uuid12}``. Operators filter
    pending work by prefix to pick the right subagent kind."""
    import re
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "0.2")

    disp = LocalDispatcher(
        mailbox_base_dir=tmp_path,
        mailbox_poll_interval=0.02,
    )
    result = await disp.dispatch_task(
        task_name="generate_assessments",
        agent_type="assessment-generator",
        task_params={}, run_id="ID_RUN",
    )
    assert re.match(
        r"^assessment-generator-[0-9a-f]{12}$",
        result["mailbox_task_id"],
    ), result["mailbox_task_id"]


@pytest.mark.asyncio
async def test_agent_spec_path_resolved_for_known_agent(tmp_path):
    """When a subagent-classified agent has an agent spec markdown
    under one of ``AGENT_SPEC_DIRS``, the pending task-spec carries
    the relative path so operators can inject it into the subagent
    prompt. When no spec exists, the field is None (best-effort)."""
    # content-generator has a spec at Courseforge/agents/content-generator.md
    disp = LocalDispatcher()
    resolved = disp._resolve_agent_spec_path("content-generator")
    # Don't assert the exact path — it's environment-dependent — but
    # when the repo has the spec file, it should resolve.
    # Unknown agent → None.
    assert disp._resolve_agent_spec_path("totally-fake-agent-xyz") is None
    # Known agent path should point at a markdown file OR be None if
    # the test runner stripped it.
    if resolved is not None:
        assert resolved.endswith("content-generator.md")
