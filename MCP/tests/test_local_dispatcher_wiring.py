"""Wave 28 / 34: LocalDispatcher dispatch paths.

Wave 28 fixed a silent-success bug: without an ``agent_tool`` callable and
without ``LOCAL_DISPATCHER_ALLOW_STUB=1``, the dispatcher used to return
``status="ok"`` with empty ``outputs``. Wave 34 changes the default path:
with no ``agent_tool`` and no stub flag, the dispatcher writes the task
spec to a ``TaskMailbox`` and blocks on completion from an outer
Claude Code watcher session. With no watcher running the wait times out
and the dispatcher surfaces a ``MAILBOX_TIMEOUT`` failure whose error
message names the three recovery paths (run ``ed4all mailbox watch``,
inject an ``agent_tool``, or rerun with ``--mode api``).

Dispatch paths exercised here:

  * No ``agent_tool`` + no stub flag + no watcher → ``status="fail"`` with
    ``error_code=MAILBOX_TIMEOUT`` and all three recovery paths mentioned.
  * ``LOCAL_DISPATCHER_ALLOW_STUB=1`` → stub ``PhaseOutput`` preserved
    for tests / dry-run.
  * Real ``agent_tool`` injected → mailbox bypassed entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.local_dispatcher import LocalDispatcher  # noqa: E402
from MCP.orchestrator.worker_contracts import (  # noqa: E402
    PhaseInput,
    PhaseOutput,
)


def _phase_input(phase_name: str = "content_generation") -> PhaseInput:
    return PhaseInput(
        run_id="RUN_W28_001",
        workflow_type="textbook_to_course",
        phase_name=phase_name,
        phase_config={"agents": ["content-generator"], "max_concurrent": 4},
        params={"course_name": "SYNTH_101"},
        mode="local",
    )


class TestDefaultFailsLoud:
    @pytest.mark.asyncio
    async def test_no_agent_tool_and_no_watcher_fails_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Without agent_tool / stub-flag / watcher, dispatch must time
        out on the mailbox and surface a MAILBOX_TIMEOUT failure — not a
        silent OK."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,
            mailbox_poll_interval=0.02,
        )
        result = await dispatcher.dispatch_phase(_phase_input())
        assert isinstance(result, PhaseOutput)
        assert result.status == "fail"
        assert result.error
        assert "mailbox_timeout" in result.error.lower()
        assert result.metrics.get("error_code") == "MAILBOX_TIMEOUT"

    @pytest.mark.asyncio
    async def test_fail_message_points_to_fix_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Error message must mention the three recovery paths so an
        operator isn't stuck guessing."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,
            mailbox_poll_interval=0.02,
        )
        result = await dispatcher.dispatch_phase(_phase_input())
        err = (result.error or "").lower()
        # All three recovery paths should be mentioned.
        assert "--mode api" in err
        assert "agent_tool" in err
        assert "mailbox watch" in err


class TestOptInStubPath:
    @pytest.mark.asyncio
    async def test_env_flag_re_enables_stub_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
        dispatcher = LocalDispatcher(project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "ok"
        assert result.outputs.get("dispatch_mode") == "stub"


class TestAgentToolOverridesEverything:
    @pytest.mark.asyncio
    async def test_real_agent_tool_bypasses_stub_and_fail_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # No stub env flag: the presence of agent_tool should take
        # precedence over the fail-loud path.
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)

        captured: dict = {}

        async def fake_agent(request):
            captured["request"] = request
            return json.dumps({
                "run_id": "RUN_W28_001",
                "phase_name": "content_generation",
                "outputs": {"emitted_pages": 5},
                "status": "ok",
            })

        dispatcher = LocalDispatcher(
            agent_tool=fake_agent, project_root=tmp_path,
        )
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "ok"
        assert result.outputs == {"emitted_pages": 5}
        assert captured["request"]["subagent_type"] == "content-generator"


class TestDispatchedListRecorded:
    @pytest.mark.asyncio
    async def test_fail_path_still_records_dispatch_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Even when dispatch fails, the attempt must appear in the
        tracked list so the orchestrator can report accurate run metrics."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,
            mailbox_poll_interval=0.02,
        )
        await dispatcher.dispatch_phase(_phase_input(phase_name="p_alpha"))
        await dispatcher.dispatch_phase(_phase_input(phase_name="p_beta"))
        dispatched = await dispatcher.after_run(workflow_id="W1", result={})
        assert dispatched == ["p_alpha", "p_beta"]
