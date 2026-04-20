"""Tests for the LocalDispatcher (Wave 7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.orchestrator.local_dispatcher import LocalDispatcher
from MCP.orchestrator.worker_contracts import PhaseInput, PhaseOutput


def _phase_input(
    phase_name: str = "dart_conversion",
    agent: str = "dart-converter",
    project_root: Path | None = None,
) -> PhaseInput:
    return PhaseInput(
        run_id="RUN_LOCAL_001",
        workflow_type="textbook_to_course",
        phase_name=phase_name,
        phase_config={"agents": [agent], "max_concurrent": 4},
        params={"course_name": "TEST_101", "pdf_paths": "/tmp/fake.pdf"},
        mode="local",
        project_root=project_root,
    )


class TestLocalDispatcherStub:
    @pytest.mark.asyncio
    async def test_stub_dispatch_returns_ok(self, tmp_path: Path):
        dispatcher = LocalDispatcher(project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert isinstance(result, PhaseOutput)
        assert result.status == "ok"
        assert result.run_id == "RUN_LOCAL_001"
        assert result.phase_name == "dart_conversion"
        assert "dispatch_mode" in result.outputs
        assert result.outputs["dispatch_mode"] == "stub"

    @pytest.mark.asyncio
    async def test_dispatched_list_tracked(self, tmp_path: Path):
        dispatcher = LocalDispatcher(project_root=tmp_path)
        await dispatcher.dispatch_phase(_phase_input(phase_name="p1"))
        await dispatcher.dispatch_phase(_phase_input(phase_name="p2"))
        dispatched = await dispatcher.after_run(workflow_id="W1", result={})
        assert dispatched == ["p1", "p2"]


class TestLocalDispatcherWithAgentTool:
    @pytest.mark.asyncio
    async def test_uses_agent_tool_when_provided(self, tmp_path: Path):
        captured = {}

        async def fake_agent(request):
            captured["request"] = request
            return json.dumps({
                "run_id": "RUN_LOCAL_001",
                "phase_name": "dart_conversion",
                "outputs": {"ok": True},
                "status": "ok",
            })

        dispatcher = LocalDispatcher(
            agent_tool=fake_agent, project_root=tmp_path
        )
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "ok"
        assert result.outputs == {"ok": True}
        assert captured["request"]["subagent_type"] == "dart-converter"

    @pytest.mark.asyncio
    async def test_invalid_json_response_marks_fail(self, tmp_path: Path):
        async def bad_agent(request):
            return "not valid json"

        dispatcher = LocalDispatcher(agent_tool=bad_agent, project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "fail"
        assert "invalid subagent JSON" in (result.error or "")

    @pytest.mark.asyncio
    async def test_non_dict_response_marks_fail(self, tmp_path: Path):
        async def list_agent(request):
            return json.dumps(["a", "b"])

        dispatcher = LocalDispatcher(agent_tool=list_agent, project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "fail"
        assert "JSON object" in (result.error or "")

    @pytest.mark.asyncio
    async def test_agent_tool_exception_caught(self, tmp_path: Path):
        async def crashy_agent(request):
            raise RuntimeError("boom")

        dispatcher = LocalDispatcher(agent_tool=crashy_agent, project_root=tmp_path)
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "fail"
        assert "boom" in (result.error or "")


class TestAgentSpecLoading:
    @pytest.mark.asyncio
    async def test_loads_spec_when_file_exists(self, tmp_path: Path):
        # Create a fake agent spec in one of the search dirs
        agent_dir = tmp_path / "Courseforge" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "course-outliner.md").write_text("# Course Outliner\nTest spec content")

        dispatcher = LocalDispatcher(project_root=tmp_path)
        phase_input = _phase_input(
            phase_name="planning", agent="course-outliner", project_root=tmp_path
        )
        prompt = dispatcher._build_subagent_prompt(phase_input)
        assert "Test spec content" in prompt
        assert "course-outliner" in prompt

    def test_no_spec_file_uses_fallback(self, tmp_path: Path):
        dispatcher = LocalDispatcher(project_root=tmp_path)
        spec = dispatcher._load_agent_spec("nonexistent-agent")
        assert "no spec file found" in spec
