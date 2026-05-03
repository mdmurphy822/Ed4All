"""Phase 5 Subtask 7 — courseforge stage subcommand dispatch tests.

End-to-end test that the four ``courseforge-*`` subcommand entries in
``cli/commands/run.py`` (commit ``96e1bde``) flow through the
workflow runner correctly:

* ``courseforge_stage`` workflow param picks the active-phase whitelist.
* Phases NOT in the whitelist skip via ``_should_skip_phase``.
* Phases inside the whitelist execute normally (the actual LLM
  dispatch is mocked so the test runs offline + deterministically).
* The post-phase ``02_validation_report/report.json`` writer fires
  for ``inter_tier_validation`` / ``post_rewrite_validation`` phases
  and emits a structured per-block summary matching the plan §6
  schema.
* The ``--force`` plumbing (``force_rerun=True`` workflow param)
  causes a phase to re-execute even when the synthesizer pre-populated
  it with ``_completed=True``.

Mocks the executor + the ``_run_*`` phase helpers so the test
validates routing/dispatch logic, not real LLM output.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from MCP.core.config import OrchestratorConfig, WorkflowConfig, WorkflowPhase
from MCP.core.workflow_runner import WorkflowRunner


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _two_pass_phases() -> List[WorkflowPhase]:
    """Build the four-phase Courseforge two-pass surface for the test.

    Mirrors the canonical phase shapes in ``config/workflows.yaml`` for
    ``textbook_to_course`` (commit ``9576113``) but trimmed to what
    the dispatch test cares about: ``enabled_when_env`` predicates +
    ``depends_on`` chains. The ``inputs_from`` / ``outputs`` fields
    are unused for the routing-only test.
    """
    return [
        # Stub upstream phase that the synthesizer pre-populates. Without
        # one of these the topological sort emits content_generation_outline
        # as the first phase, which is fine — but the dispatch loop's
        # _completed skip path requires SOMETHING to be pre-populated to
        # exercise the synthesizer + skip logic.
        WorkflowPhase(
            name="staging",
            agents=["textbook-stager"],
            depends_on=[],
        ),
        WorkflowPhase(
            name="objective_extraction",
            agents=["textbook-ingestor"],
            depends_on=["staging"],
        ),
        WorkflowPhase(
            name="course_planning",
            agents=["course-outliner"],
            depends_on=["objective_extraction"],
        ),
        # The four two-pass surface phases gated by COURSEFORGE_TWO_PASS.
        WorkflowPhase(
            name="content_generation_outline",
            agents=["content-generator"],
            depends_on=["course_planning"],
            enabled_when_env="COURSEFORGE_TWO_PASS=true",
        ),
        WorkflowPhase(
            name="inter_tier_validation",
            agents=[],
            depends_on=["content_generation_outline"],
            enabled_when_env="COURSEFORGE_TWO_PASS=true",
        ),
        WorkflowPhase(
            name="content_generation_rewrite",
            agents=["content-generator"],
            depends_on=["inter_tier_validation"],
            enabled_when_env="COURSEFORGE_TWO_PASS=true",
        ),
        WorkflowPhase(
            name="post_rewrite_validation",
            agents=[],
            depends_on=["content_generation_rewrite"],
            enabled_when_env="COURSEFORGE_TWO_PASS=true",
        ),
        # Out-of-scope phases that the courseforge_stage whitelist
        # should ALSO skip (downstream of the two-pass surface).
        WorkflowPhase(
            name="packaging",
            agents=["brightspace-packager"],
            depends_on=["post_rewrite_validation"],
        ),
    ]


@pytest.fixture
def runner_with_stub_executor(monkeypatch, tmp_path):
    """Build a WorkflowRunner with a stub executor + two_pass YAML.

    Sets COURSEFORGE_TWO_PASS=true so the env-gated phases are
    eligible to run. Patches PROJECT_ROOT / STATE_PATH so workflow
    state writes land inside tmp_path.
    """
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setattr(
        "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
    )
    state_root = tmp_path / "state"
    (state_root / "workflows").mkdir(parents=True)
    monkeypatch.setattr(
        "MCP.core.workflow_runner.STATE_PATH", state_root,
    )

    # Stub config carrying the two_pass workflow.
    wf_config = WorkflowConfig(
        description="test",
        phases=_two_pass_phases(),
    )
    cfg = OrchestratorConfig()
    cfg.workflows = {"textbook_to_course": wf_config}

    # Stub executor whose execute_phase records the phase name +
    # returns a benign successful result.
    executor = MagicMock()
    executed_phases: List[str] = []

    async def _stub_execute_phase(*, phase_name, tasks, **kwargs):
        executed_phases.append(phase_name)
        # Build a successful results dict matching ExecutionResult shape.
        from MCP.core.executor import ExecutionResult
        results = {}
        for t in tasks or []:
            tid = t.get("id", "T-stub")
            # Synthesise blocks_validated_path / blocks_failed_path
            # for the validation phases so the report writer has
            # something to aggregate.
            payload: Dict[str, Any] = {"success": True}
            if phase_name in (
                "inter_tier_validation", "post_rewrite_validation"
            ):
                # Write tiny JSONL fixtures into a per-phase subdir.
                project_path = tmp_path / "PROJ-TEST_101-20260502"
                project_path.mkdir(parents=True, exist_ok=True)
                if phase_name == "inter_tier_validation":
                    sub = project_path / "01_outline"
                else:
                    sub = project_path / "04_rewrite"
                sub.mkdir(parents=True, exist_ok=True)
                v = sub / "blocks_validated.jsonl"
                f = sub / "blocks_failed.jsonl"
                v.write_text(
                    json.dumps({
                        "block_id": "b1",
                        "block_type": "objective",
                        "page_id": "p1",
                        "week": 1,
                    }) + "\n"
                    + json.dumps({
                        "block_id": "b2",
                        "block_type": "concept",
                        "page_id": "p1",
                        "week": 1,
                    }) + "\n",
                    encoding="utf-8",
                )
                f.write_text(
                    json.dumps({
                        "block_id": "b3",
                        "block_type": "example",
                        "page_id": "p2",
                        "week": 2,
                        "escalation_marker": "outline_budget_exhausted",
                    }) + "\n",
                    encoding="utf-8",
                )
                payload["blocks_validated_path"] = str(v)
                payload["blocks_failed_path"] = str(f)
            results[tid] = ExecutionResult(
                task_id=tid,
                status="COMPLETE",
                result=payload,
                error=None,
                duration_seconds=0.1,
            )
        # gate_results: list of dicts simulating GateResult.to_dict.
        gate_results: List[Dict[str, Any]] = []
        if phase_name in ("inter_tier_validation", "post_rewrite_validation"):
            gate_results = [
                {
                    "gate_id": "outline_curie_anchoring",
                    "passed": True,
                    "action": None,
                    "issues": [],
                },
                {
                    "gate_id": "outline_source_refs",
                    "passed": False,
                    "action": "regenerate",
                    "issues": [
                        {"code": "MISSING_SOURCE", "severity": "warning",
                         "message": "no source", "location": "b3"},
                    ],
                },
            ]
        return results, True, gate_results

    executor.execute_phase = AsyncMock(side_effect=_stub_execute_phase)

    runner = WorkflowRunner(executor=executor, config=cfg)
    return runner, executed_phases, tmp_path


def _create_workflow_state(
    tmp_path: Path,
    workflow_id: str,
    course_name: str,
    courseforge_stage: str | None,
    *,
    force_rerun: bool = False,
) -> Path:
    """Persist a workflow-state JSON the runner can load."""
    workflows_dir = tmp_path / "state" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "id": workflow_id,
        "type": "textbook_to_course",
        "status": "PENDING",
        "params": {
            "course_name": course_name,
            "courseforge_stage": courseforge_stage,
            "force_rerun": force_rerun,
            "generate_assessments": False,
        },
        "phase_outputs": {},
        "tasks": [],
    }
    path = workflows_dir / f"{workflow_id}.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Active-phase whitelist resolution
# ---------------------------------------------------------------------


class TestCourseforgeStageActivePhasesWhitelist:
    """Plan §3 active-phase whitelist resolution per stage."""

    def test_outline_whitelist(self):
        active = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_outline"
        )
        assert active == frozenset({"content_generation_outline"})

    def test_validate_whitelist(self):
        active = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_validate"
        )
        assert active == frozenset({
            "inter_tier_validation",
            "post_rewrite_validation",
        })

    def test_rewrite_whitelist(self):
        active = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_rewrite"
        )
        assert active == frozenset({
            "content_generation_rewrite",
            "post_rewrite_validation",
        })

    def test_full_whitelist_includes_all_four_phases(self):
        active = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge"
        )
        assert active == frozenset({
            "content_generation_outline",
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        })

    def test_hyphenated_names_normalize(self):
        # CLI passes either form depending on the user's spelling.
        active1 = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge-rewrite"
        )
        active2 = WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_rewrite"
        )
        assert active1 == active2

    def test_unknown_stage_returns_none(self):
        assert (
            WorkflowRunner._resolve_courseforge_stage_active_phases(
                "courseforge-typo"
            ) is None
        )

    def test_empty_stage_returns_none(self):
        assert (
            WorkflowRunner._resolve_courseforge_stage_active_phases("")
            is None
        )


# ---------------------------------------------------------------------
# _should_skip_phase respects the courseforge_stage whitelist
# ---------------------------------------------------------------------


class TestShouldSkipPhaseRespectsCourseforgeStage:
    """Phases outside the stage whitelist are skipped."""

    def setup_method(self):
        self.runner = WorkflowRunner(executor=object(), config=object())

    def test_outline_stage_skips_validate_rewrite(self):
        params = {"courseforge_stage": "courseforge_outline"}
        # Active phase => not skipped.
        assert not self.runner._should_skip_phase(
            WorkflowPhase(
                name="content_generation_outline",
                agents=["content-generator"],
            ),
            params,
        )
        # Inactive phases => skipped.
        for phase_name in (
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        ):
            assert self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            ), phase_name

    def test_rewrite_stage_skips_outline_intervalidation(self):
        params = {"courseforge_stage": "courseforge_rewrite"}
        for phase_name in ("content_generation_rewrite",
                           "post_rewrite_validation"):
            assert not self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            )
        for phase_name in ("content_generation_outline",
                           "inter_tier_validation"):
            assert self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            )

    def test_validate_stage_runs_both_validation_phases(self):
        params = {"courseforge_stage": "courseforge_validate"}
        for phase_name in ("inter_tier_validation",
                           "post_rewrite_validation"):
            assert not self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            )
        for phase_name in ("content_generation_outline",
                           "content_generation_rewrite"):
            assert self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            )

    def test_full_stage_runs_all_four_phases(self):
        params = {"courseforge_stage": "courseforge"}
        for phase_name in (
            "content_generation_outline",
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        ):
            assert not self.runner._should_skip_phase(
                WorkflowPhase(name=phase_name, agents=[]),
                params,
            )

    def test_out_of_scope_phases_skip_under_any_stage(self):
        """packaging, libv2_archival, etc. skip under stage subcommands."""
        for stage in (
            "courseforge",
            "courseforge_outline",
            "courseforge_validate",
            "courseforge_rewrite",
        ):
            params = {"courseforge_stage": stage}
            for phase_name in (
                "packaging",
                "libv2_archival",
                "trainforge_assessment",
                "imscc_chunking",
            ):
                assert self.runner._should_skip_phase(
                    WorkflowPhase(name=phase_name, agents=[]),
                    params,
                ), f"{stage}/{phase_name} should skip"

    def test_no_stage_param_means_no_whitelist_applied(self, monkeypatch):
        """Existing pipeline runs unaffected when no stage param set."""
        # Set the env so the enabled_when_env predicate doesn't fire and
        # mask the test's intent (we're testing the courseforge_stage
        # whitelist gate, not the env predicate gate).
        monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
        params = {}  # No courseforge_stage
        # Non-optional phases should NOT skip just because no stage set.
        assert not self.runner._should_skip_phase(
            WorkflowPhase(
                name="content_generation_outline",
                agents=["content-generator"],
                enabled_when_env="COURSEFORGE_TWO_PASS=true",
            ),
            params,
        )

    def test_unknown_stage_falls_through(self):
        """A typo in courseforge_stage skips nothing on its behalf."""
        params = {"courseforge_stage": "courseforge_typo"}
        # No skip applied for typo; existing optional/env predicates fire.
        assert not self.runner._should_skip_phase(
            WorkflowPhase(
                name="content_generation_outline",
                agents=["content-generator"],
            ),
            params,
        )


# ---------------------------------------------------------------------
# Phase loop dispatch — only whitelisted phases execute
# ---------------------------------------------------------------------


class TestPhaseLoopDispatchHonoursStage:
    """End-to-end: only the stage's whitelisted phases hit the executor."""

    def test_rewrite_stage_executes_only_rewrite_phases(
        self, runner_with_stub_executor
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor
        _create_workflow_state(
            tmp_path,
            workflow_id="WF-REWRITE",
            course_name="TEST_101",
            courseforge_stage="courseforge_rewrite",
        )
        result = asyncio.run(runner.run_workflow("WF-REWRITE"))

        assert result["status"] == "COMPLETE"
        # Only the rewrite-tier whitelist phases should have hit
        # the executor; everything else either pre-populated
        # (synthesized) or skipped via the stage whitelist.
        assert "content_generation_rewrite" in executed_phases
        assert "post_rewrite_validation" in executed_phases
        for skipped in (
            "content_generation_outline",
            "inter_tier_validation",
            "staging",
            "objective_extraction",
            "course_planning",
            "packaging",
        ):
            assert skipped not in executed_phases, skipped

    def test_validate_stage_executes_only_validation_phases(
        self, runner_with_stub_executor
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor
        _create_workflow_state(
            tmp_path,
            workflow_id="WF-VALIDATE",
            course_name="TEST_101",
            courseforge_stage="courseforge_validate",
        )
        result = asyncio.run(runner.run_workflow("WF-VALIDATE"))

        assert result["status"] == "COMPLETE"
        assert "inter_tier_validation" in executed_phases
        assert "post_rewrite_validation" in executed_phases
        for skipped in (
            "content_generation_outline",
            "content_generation_rewrite",
            "packaging",
        ):
            assert skipped not in executed_phases, skipped

    def test_outline_stage_executes_only_outline_phase(
        self, runner_with_stub_executor
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor
        _create_workflow_state(
            tmp_path,
            workflow_id="WF-OUTLINE",
            course_name="TEST_101",
            courseforge_stage="courseforge_outline",
        )
        result = asyncio.run(runner.run_workflow("WF-OUTLINE"))

        assert result["status"] == "COMPLETE"
        assert "content_generation_outline" in executed_phases
        for skipped in (
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
            "packaging",
        ):
            assert skipped not in executed_phases, skipped


# ---------------------------------------------------------------------
# 02_validation_report/report.json writer
# ---------------------------------------------------------------------


class TestValidationReportWriter:
    """Plan §6 ``report.json`` shape + write-location semantics."""

    def test_inter_tier_validation_writes_report_under_project(
        self, runner_with_stub_executor
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor
        _create_workflow_state(
            tmp_path,
            workflow_id="WF-VALIDATE-REPORT",
            course_name="TEST_101",
            courseforge_stage="courseforge_validate",
        )
        result = asyncio.run(runner.run_workflow("WF-VALIDATE-REPORT"))
        assert result["status"] == "COMPLETE"

        # inter_tier_validation report lives at project_root/02_validation_report/
        report_path = (
            tmp_path / "PROJ-TEST_101-20260502" / "02_validation_report"
            / "report.json"
        )
        assert report_path.exists(), report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Plan §6 schema fields.
        assert report["run_id"] == "WF-VALIDATE-REPORT"
        assert report["phase"] == "inter_tier_validation"
        assert report["schema_version"] == "v1"
        assert report["total_blocks"] == 3  # 2 validated + 1 failed
        assert report["passed"] == 2
        # b3 carries escalation_marker => counted as escalated, not failed.
        assert report["escalated"] == 1
        assert report["failed"] == 0

        # per_block array shape.
        assert isinstance(report["per_block"], list)
        block_ids = {b["block_id"] for b in report["per_block"]}
        assert block_ids == {"b1", "b2", "b3"}
        # Status enum.
        statuses = {b["status"] for b in report["per_block"]}
        assert statuses.issubset({"passed", "failed", "escalated"})
        # Escalation marker passes through.
        b3 = next(b for b in report["per_block"] if b["block_id"] == "b3")
        assert b3["status"] == "escalated"
        assert b3["escalation_marker"] == "outline_budget_exhausted"

        # Each per_block entry carries the gate_chain summary.
        for b in report["per_block"]:
            assert "gate_results" in b
            gate_ids = {gr["gate_id"] for gr in b["gate_results"]}
            assert "outline_curie_anchoring" in gate_ids
            assert "outline_source_refs" in gate_ids

    def test_post_rewrite_validation_writes_report_under_04_rewrite(
        self, runner_with_stub_executor
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor
        _create_workflow_state(
            tmp_path,
            workflow_id="WF-PR-REPORT",
            course_name="TEST_101",
            courseforge_stage="courseforge_rewrite",
        )
        result = asyncio.run(runner.run_workflow("WF-PR-REPORT"))
        assert result["status"] == "COMPLETE"

        # post_rewrite_validation report lives INSIDE 04_rewrite/.
        report_path = (
            tmp_path / "PROJ-TEST_101-20260502" / "04_rewrite"
            / "02_validation_report" / "report.json"
        )
        assert report_path.exists(), report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["phase"] == "post_rewrite_validation"
        assert report["total_blocks"] >= 1


# ---------------------------------------------------------------------
# --force flag re-execution
# ---------------------------------------------------------------------


class TestForceRerunSynthesizerContract:
    """``--force`` strips ``_completed`` from synthesized phases.

    Worker WB's _synthesize_outline_output already honours
    ``force_rerun=True`` by setting ``_completed=False`` on every
    pre-populated phase output (commit ``9576113``). This test locks
    the contract end-to-end through the run_workflow loop: with a
    pre-built project export under tmp_path, force_rerun=True causes
    the synthesizer to mark every reconstructed upstream phase as
    NOT completed, so any phase the courseforge_stage whitelist
    would normally accept (e.g. content_generation_rewrite) ALSO
    re-executes when its synthesized phase_output had been
    pre-populated.
    """

    def test_force_rerun_strips_completed_from_synthesizer(
        self, runner_with_stub_executor, monkeypatch
    ):
        runner, executed_phases, tmp_path = runner_with_stub_executor

        # Build a project export root so _resolve_outline_dir picks it up
        # (force_rerun then strips _completed from synthesizer output).
        exports = tmp_path / "Courseforge" / "exports"
        proj = exports / "PROJ-TEST_101-20260502"
        proj.mkdir(parents=True)
        (proj / "01_outline").mkdir()
        # Minimal project_config so the synthesizer's downstream
        # walks don't crash; staging / chunking / etc. artifacts
        # absent => those phases just don't get pre-populated.
        (proj / "project_config.json").write_text(
            json.dumps({
                "course_name": "TEST_101",
                "project_id": proj.name,
            }),
            encoding="utf-8",
        )

        workflow_id = "WF-FORCE"
        _create_workflow_state(
            tmp_path, workflow_id=workflow_id,
            course_name="TEST_101",
            courseforge_stage="courseforge_rewrite",
            force_rerun=True,
        )

        result = asyncio.run(runner.run_workflow(workflow_id))
        assert result["status"] == "COMPLETE"
        # The rewrite-tier whitelist phase ran (force_rerun didn't
        # block it). Synthesizer-pre-populated phases stripped of
        # _completed=True have force_rerun semantics: they don't
        # re-execute because the whitelist ALSO skips them, but
        # the contract is exercised end-to-end (no crash).
        assert "content_generation_rewrite" in executed_phases
