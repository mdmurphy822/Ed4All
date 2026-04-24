"""End-to-end tests for PipelineOrchestrator (Wave 7)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from MCP.core.config import OrchestratorConfig, WorkflowConfig, WorkflowPhase
from MCP.orchestrator.llm_backend import BackendSpec, MockBackend
from MCP.orchestrator.pipeline_orchestrator import (
    OrchestratorResult,
    PipelineOrchestrator,
)
from MCP.orchestrator.worker_contracts import PhaseInput


def _make_config() -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with a test workflow."""
    config = OrchestratorConfig()
    phases = [
        WorkflowPhase(name="planning", agents=["course-outliner"], depends_on=[]),
        WorkflowPhase(
            name="content_generation",
            agents=["content-generator"],
            depends_on=["planning"],
        ),
        WorkflowPhase(
            name="packaging",
            agents=["brightspace-packager"],
            depends_on=["content_generation"],
        ),
    ]
    config.workflows["test_wf"] = WorkflowConfig(
        description="Test workflow", phases=phases
    )
    return config


class TestConstruction:
    def test_default_mode_is_local(self, tmp_path: Path):
        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)
        assert orch.mode == "local"

    def test_api_mode(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
        orch = PipelineOrchestrator(
            config=_make_config(),
            mode="api",
            backend_spec=BackendSpec(mode="api", provider="anthropic"),
            project_root=tmp_path,
        )
        assert orch.mode == "api"

    def test_explicit_factory_wins(self, tmp_path: Path):
        factory = lambda: MockBackend(responses=["x"])
        orch = PipelineOrchestrator(
            config=_make_config(),
            mode="api",
            llm_factory=factory,
            project_root=tmp_path,
        )
        assert orch.llm_factory is factory
        backend = orch.llm_factory()
        assert isinstance(backend, MockBackend)

    def test_unknown_mode_raises(self, tmp_path: Path):
        orch = PipelineOrchestrator(
            config=_make_config(),
            mode="weird",  # type: ignore[arg-type]
            project_root=tmp_path,
        )
        with pytest.raises(ValueError, match="Unknown orchestrator mode"):
            orch._get_dispatcher()


class TestPlan:
    def test_plan_returns_phases_in_order(self, tmp_path: Path, monkeypatch):
        config = _make_config()
        state_dir = tmp_path / "state" / "workflows"
        state_dir.mkdir(parents=True)
        state_path = state_dir / "WF-TEST.json"
        state_path.write_text(
            json.dumps({"type": "test_wf", "id": "WF-TEST", "params": {}})
        )

        # Patch STATE_PATH used by orchestrator
        import MCP.orchestrator.pipeline_orchestrator as po

        monkeypatch.setattr(po, "STATE_PATH", tmp_path / "state")

        orch = PipelineOrchestrator(config=config, project_root=tmp_path)
        plan = orch.plan("WF-TEST")
        names = [p["name"] for p in plan]
        assert names == ["planning", "content_generation", "packaging"]

    def test_plan_missing_workflow_returns_empty(self, tmp_path: Path):
        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)
        assert orch.plan("does-not-exist") == []


class TestBuildPhaseInput:
    def test_phase_input_wired_correctly(self, tmp_path: Path):
        orch = PipelineOrchestrator(
            config=_make_config(),
            mode="api",
            llm_factory=lambda: MockBackend(responses=["r"]),
            project_root=tmp_path,
        )
        pi = orch.build_phase_input(
            run_id="RUN_1",
            workflow_type="test_wf",
            phase_name="planning",
            phase_config={"agents": ["course-outliner"]},
            params={"course_name": "C"},
            course_code="C",
            tool="courseforge",
        )
        assert isinstance(pi, PhaseInput)
        assert pi.run_id == "RUN_1"
        assert pi.phase_name == "planning"
        assert pi.mode == "api"
        backend = pi.llm_factory()
        assert isinstance(backend, MockBackend)
        # captures_dir path shape
        assert "phase_planning" in str(pi.captures_dir)
        assert "courseforge" in str(pi.captures_dir)


class TestRun:
    @pytest.mark.asyncio
    async def test_run_missing_workflow(self, tmp_path: Path, monkeypatch):
        import MCP.orchestrator.pipeline_orchestrator as po

        monkeypatch.setattr(po, "STATE_PATH", tmp_path / "state")
        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)
        result = await orch.run("nonexistent")
        assert isinstance(result, OrchestratorResult)
        assert result.status == "failed"
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_run_delegates_to_workflow_runner(self, tmp_path: Path, monkeypatch):
        """PipelineOrchestrator.run() should invoke WorkflowRunner.run_workflow
        and wrap the result into an OrchestratorResult."""
        import MCP.orchestrator.pipeline_orchestrator as po

        state_dir = tmp_path / "state" / "workflows"
        state_dir.mkdir(parents=True)
        (state_dir / "WF-T.json").write_text(
            json.dumps({"id": "WF-T", "type": "test_wf", "params": {}})
        )
        monkeypatch.setattr(po, "STATE_PATH", tmp_path / "state")

        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)

        # Stub WorkflowRunner.run_workflow to return a "complete" result
        async def fake_run(self, workflow_id: str):
            return {
                "workflow_id": workflow_id,
                "status": "COMPLETE",
                "phase_results": {
                    "planning": {"task_count": 1, "completed": 1, "gates_passed": True}
                },
                "phase_outputs": {"planning": {"_completed": True}},
            }

        with patch(
            "MCP.core.workflow_runner.WorkflowRunner.run_workflow", new=fake_run
        ):
            result = await orch.run("WF-T")

        assert result.status == "ok"
        assert "planning" in result.phase_results
        assert result.phase_outputs == {"planning": {"_completed": True}}

    @pytest.mark.asyncio
    async def test_run_handles_workflow_exception(self, tmp_path: Path, monkeypatch):
        import MCP.orchestrator.pipeline_orchestrator as po

        state_dir = tmp_path / "state" / "workflows"
        state_dir.mkdir(parents=True)
        (state_dir / "WF-T.json").write_text(
            json.dumps({"id": "WF-T", "type": "test_wf", "params": {}})
        )
        monkeypatch.setattr(po, "STATE_PATH", tmp_path / "state")

        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)

        async def crashy(self, workflow_id: str):
            raise RuntimeError("catastrophic")

        with patch(
            "MCP.core.workflow_runner.WorkflowRunner.run_workflow", new=crashy
        ):
            result = await orch.run("WF-T")

        assert result.status == "failed"
        assert "catastrophic" in (result.error or "")


class TestDescribe:
    def test_describe_contains_fields(self, tmp_path: Path):
        orch = PipelineOrchestrator(config=_make_config(), project_root=tmp_path)
        snapshot = orch.describe()
        assert snapshot["mode"] == "local"
        assert "dispatcher" in snapshot
        assert "timestamp" in snapshot


class TestWorkerContracts:
    def test_phase_output_roundtrip(self):
        from MCP.orchestrator.worker_contracts import (
            GateResult,
            PhaseOutput,
        )

        original = PhaseOutput(
            run_id="R1",
            phase_name="planning",
            outputs={"a": 1},
            artifacts=[Path("/tmp/x.txt")],
            gate_results={
                "content_structure": GateResult(
                    gate_id="content_structure",
                    severity="critical",
                    passed=True,
                )
            },
            status="ok",
        )
        data = original.to_dict()
        restored = PhaseOutput.from_dict(data)
        assert restored.run_id == "R1"
        assert restored.phase_name == "planning"
        assert restored.outputs == {"a": 1}
        assert restored.status == "ok"
        assert "content_structure" in restored.gate_results
        assert restored.gate_results["content_structure"].passed is True

    def test_phase_input_serialization_drops_factory(self):
        from MCP.orchestrator.worker_contracts import PhaseInput

        pi = PhaseInput(
            run_id="R",
            workflow_type="t",
            phase_name="p",
            phase_config={},
            params={},
            mode="local",
            llm_factory=lambda: None,
        )
        data = pi.to_dict()
        assert "llm_factory" not in data
        # Round-trip shouldn't crash
        json.dumps(data)
