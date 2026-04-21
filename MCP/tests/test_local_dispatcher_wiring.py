"""Wave 28: verify LocalDispatcher fails loudly when no agent_tool is wired.

Pre-Wave-28, ``LocalDispatcher.dispatch_phase`` returned ``status="ok"``
with an empty ``outputs`` dict whenever no agent_tool callable was injected.
That let production ``--mode local`` runs report success while producing no
real phase output. The fix (Wave 28):

  * Default path with no agent_tool AND no LOCAL_DISPATCHER_ALLOW_STUB env
    flag → ``status="fail"`` with a descriptive error message.
  * LOCAL_DISPATCHER_ALLOW_STUB=1 → opt-in stub path preserved for tests
    and dry-run usage.
  * Real agent_tool injected → unchanged (dispatches + parses JSON).
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
    async def test_no_agent_tool_and_no_flag_fails_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Without agent_tool and without the opt-in env flag, dispatch
        must return status=fail with a descriptive error — not silent OK."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert isinstance(result, PhaseOutput)
        assert result.status == "fail"
        assert result.error
        assert "agent_tool" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fail_message_points_to_fix_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Error message must mention the three recovery paths so an
        operator isn't stuck guessing."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        err = (result.error or "").lower()
        # Either "--mode api" or "allow_stub" environment variable name
        # must surface in the message.
        assert "--mode api" in err or "local_dispatcher_allow_stub" in err


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
        dispatcher = LocalDispatcher(project_root=tmp_path)
        await dispatcher.dispatch_phase(_phase_input(phase_name="p_alpha"))
        await dispatcher.dispatch_phase(_phase_input(phase_name="p_beta"))
        dispatched = await dispatcher.after_run(workflow_id="W1", result={})
        assert dispatched == ["p_alpha", "p_beta"]
