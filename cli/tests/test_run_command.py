"""Tests for the ``ed4all run`` CLI command (Wave 7)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from cli.commands.run import _build_workflow_params
from cli.main import cli


class TestHelp:
    def test_run_appears_in_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_help_lists_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--corpus" in result.output
        assert "--course-name" in result.output
        assert "--mode" in result.output
        assert "--dry-run" in result.output
        assert "--resume" in result.output
        assert "--watch" in result.output


class TestDryRun:
    def test_textbook_to_course_dry_run(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output or "textbook_to_course" in result.output
        assert "dart_conversion" in result.output

    def test_dry_run_json_output(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["workflow"] == "textbook_to_course"
        assert payload["mode"] in ("local", "api")
        assert isinstance(payload["phases"], list)
        assert any(p["name"] == "dart_conversion" for p in payload["phases"])

    def test_dry_run_no_assessments_skips_phase(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--dry-run",
                "--json",
                "--no-assessments",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        names = [p["name"] for p in payload["phases"]]
        assert "trainforge_assessment" not in names


class TestValidation:
    def test_unknown_workflow_rejected(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "no-such-workflow", "--dry-run", "--course-name", "X"]
        )
        assert result.exit_code == 2
        assert "Unknown workflow" in result.output

    def test_missing_course_name_without_dry_run_errors(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
            ],
        )
        assert result.exit_code == 2
        assert "course-name" in result.output


# Wave 28f: TestDeprecationWarning class removed alongside the
# ``ed4all textbook-to-course`` top-level command. The Wave 7 replacement
# is ``ed4all run textbook-to-course ...``; see the TestRunCommand
# coverage above.


class TestBuildWorkflowParamsDurationWeeks:
    """Wave 39: ensure ``--weeks`` handling preserves auto-scale contract.

    The textbook_to_course extractor auto-scales duration_weeks to
    max(8, chapter_count) only when ``duration_weeks`` is absent and
    ``duration_weeks_explicit`` is False. Prior to Wave 39 the CLI was
    silently coercing an unset ``--weeks`` back to 12 for every
    workflow, which broke that auto-scale path.
    """

    @staticmethod
    def _base_kwargs():
        return dict(
            corpus=None,
            course_name="TEST_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )

    def test_textbook_to_course_omits_duration_when_weeks_unset(self):
        kwargs = self._base_kwargs()
        params = _build_workflow_params("textbook_to_course", **kwargs)
        assert "duration_weeks" not in params
        assert params["duration_weeks_explicit"] is False

    def test_non_textbook_workflow_defaults_duration_to_12(self):
        kwargs = self._base_kwargs()
        params = _build_workflow_params("course_generation", **kwargs)
        assert params["duration_weeks"] == 12
        assert params["duration_weeks_explicit"] is False

    def test_explicit_weeks_are_honoured_for_textbook(self):
        kwargs = self._base_kwargs()
        kwargs["weeks"] = 16
        params = _build_workflow_params("textbook_to_course", **kwargs)
        assert params["duration_weeks"] == 16
        assert params["duration_weeks_explicit"] is True


class TestCreateTextbookWorkflowFlagPropagation:
    """Wave 39 follow-up: verify the runtime path honours the
    ``duration_weeks_explicit`` flag, not just the dry-run output.

    PR #100 review finding: Wave 39's initial fix omitted
    ``duration_weeks`` from the dry-run params but
    ``_create_textbook_workflow`` still forwarded a fixed 12 to
    ``create_textbook_pipeline`` via the ``.get(..., 12)`` default.
    Real runs therefore never hit the extractor's auto-scale branch,
    creating a plan/runtime mismatch.
    """

    @pytest.mark.asyncio
    async def test_runtime_propagates_explicit_false_when_weeks_unset(self):
        from cli.commands import run as run_mod

        params = _build_workflow_params(
            "textbook_to_course",
            corpus="my.pdf",
            course_name="PROPAGATE_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )
        assert params["duration_weeks_explicit"] is False
        assert "duration_weeks" not in params

        captured: dict = {}

        async def _fake_create_textbook_pipeline(**kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True, "workflow_id": "WF-TEST"})

        with patch(
            "MCP.tools.pipeline_tools.create_textbook_pipeline",
            new=_fake_create_textbook_pipeline,
        ):
            result = await run_mod._create_textbook_workflow(params)

        assert result["success"] is True
        # The runtime path must forward the explicit flag. Pre-fix it
        # was dropped and every textbook run got the hard-coded 12.
        assert captured.get("duration_weeks_explicit") is False
        # ``duration_weeks`` still defaults to 12 as the nominal value;
        # the extractor reads it + the flag together and decides to
        # override when the flag is False.
        assert captured.get("duration_weeks") == 12

    @pytest.mark.asyncio
    async def test_runtime_propagates_explicit_true_when_weeks_set(self):
        from cli.commands import run as run_mod

        params = _build_workflow_params(
            "textbook_to_course",
            corpus="my.pdf",
            course_name="PROPAGATE_102",
            weeks=14,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )
        assert params["duration_weeks_explicit"] is True
        assert params["duration_weeks"] == 14

        captured: dict = {}

        async def _fake_create_textbook_pipeline(**kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True, "workflow_id": "WF-TEST"})

        with patch(
            "MCP.tools.pipeline_tools.create_textbook_pipeline",
            new=_fake_create_textbook_pipeline,
        ):
            await run_mod._create_textbook_workflow(params)

        assert captured.get("duration_weeks_explicit") is True
        assert captured.get("duration_weeks") == 14


class TestCreateTextbookPipelinePropagatesFlag:
    """Wave 39 follow-up: ``create_textbook_pipeline`` must surface
    ``duration_weeks_explicit`` into the workflow state's ``params``
    so ``_extract_textbook_structure`` sees it via kwargs.
    """

    @pytest.mark.asyncio
    async def test_explicit_flag_flows_into_workflow_state(self, tmp_path, monkeypatch):
        from MCP.tools import pipeline_tools
        from MCP.tools.pipeline_tools import create_textbook_pipeline

        # Create a PDF inside a tmp root and point PROJECT_ROOT at it
        # so the path-escape guard inside ``create_textbook_pipeline``
        # accepts our synthetic fixture.
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

        captured: dict = {}

        async def _fake_create_workflow_impl(**kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True, "workflow_id": "WF-IMPL"})

        with patch(
            "MCP.tools.orchestrator_tools.create_workflow_impl",
            new=_fake_create_workflow_impl,
        ):
            raw = await create_textbook_pipeline(
                pdf_paths=str(pdf),
                course_name="FLOWTEST_101",
                duration_weeks=12,
                duration_weeks_explicit=False,
            )

        result = json.loads(raw)
        # success=True means the fake responded, so captured is populated.
        assert result.get("success") is True, result
        forwarded = json.loads(captured["params"])
        assert forwarded.get("duration_weeks_explicit") is False
        assert forwarded.get("duration_weeks") == 12


class TestResume:
    def test_resume_invokes_orchestrator(self):
        runner = CliRunner()

        fake_result = type(
            "R",
            (),
            {
                "status": "ok",
                "error": None,
                "to_dict": lambda self: {"status": "ok"},
            },
        )()

        with patch(
            "cli.commands.run._build_orchestrator"
        ) as build_mock:
            orch = build_mock.return_value
            orch.run = AsyncMock(return_value=fake_result)

            result = runner.invoke(
                cli,
                ["run", "textbook-to-course", "--resume", "WF-ABC"],
            )
            assert result.exit_code == 0, result.output
            orch.run.assert_awaited_once_with("WF-ABC")


# =============================================================================
# Wave 74 Session 3: --skip-dart flag coverage
# =============================================================================


class TestSkipDartFlag:
    """Wave 74 Session 3: verify --skip-dart threads through the CLI.

    Pins:
    * CLI flag parses and sets workflow params (skip_dart + dart_output_dir).
    * Dry-run plan marks dart_conversion as SKIPPED with a reason line.
    * Default (--skip-dart absent) leaves params + plan unchanged
      (regression guard).
    * Invalid inputs (no dir / empty dir / non-textbook workflow) fail
      fast with a clear error message.
    * Warning (not fatal) when a corpus PDF has no matching HTML.
    """

    def test_skip_dart_appears_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--skip-dart" in result.output
        assert "--dart-output-dir" in result.output

    def test_build_params_sets_skip_dart_keys(self):
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="fake.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
            skip_dart=True,
            dart_output_dir="/some/dart/output",
        )
        assert params["skip_dart"] is True
        assert params["dart_output_dir"] == "/some/dart/output"

    def test_build_params_defaults_no_skip_dart_keys(self):
        """Regression guard: no skip_dart keys when flag is off."""
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="fake.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )
        assert "skip_dart" not in params
        assert "dart_output_dir" not in params

    def test_dry_run_with_skip_dart_marks_phase_skipped(self, tmp_path):
        # Create a fake DART output dir with one accessible HTML whose
        # basename matches the corpus PDF, so the CLI emits no warning
        # that would pollute the --json stream.
        (tmp_path / "book_accessible.html").write_text("<html></html>")
        corpus = tmp_path / "book.pdf"
        corpus.write_bytes(b"%PDF-1.4")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                str(corpus),
                "--course-name",
                "TEST_101",
                "--skip-dart",
                "--dart-output-dir",
                str(tmp_path),
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        dart_phase = next(
            p for p in payload["phases"] if p["name"] == "dart_conversion"
        )
        assert dart_phase.get("status") == "SKIPPED"
        assert "skip-dart" in dart_phase.get("skip_reason", "").lower()
        # staging should still be in the plan and depend on dart_conversion.
        staging = next(p for p in payload["phases"] if p["name"] == "staging")
        assert "dart_conversion" in staging["depends_on"]
        # Workflow params carry the skip_dart + dart_output_dir keys.
        assert payload["params"]["skip_dart"] is True
        assert payload["params"]["dart_output_dir"] == str(tmp_path)

    def test_dry_run_without_skip_dart_regression(self):
        """Regression guard: default plan has no SKIPPED marker."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        dart_phase = next(
            p for p in payload["phases"] if p["name"] == "dart_conversion"
        )
        assert "status" not in dart_phase
        assert "skip_reason" not in dart_phase

    def test_skip_dart_with_missing_dir_fails_fast(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--skip-dart",
                "--dart-output-dir",
                "/definitely/does/not/exist",
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "existing directory" in result.output

    def test_skip_dart_with_empty_dir_fails_fast(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
                "--skip-dart",
                "--dart-output-dir",
                str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "at least one" in result.output

    def test_skip_dart_rejects_non_textbook_workflow(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "rag_training",
                "--corpus",
                "course.imscc",
                "--course-name",
                "TEST_101",
                "--skip-dart",
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "skip-dart" in result.output.lower()
        assert "textbook_to_course" in result.output

    def test_skip_dart_warns_on_corpus_html_mismatch(self, tmp_path):
        """When a corpus PDF has no matching HTML, warn but don't fail."""
        # HTML present but mismatched corpus
        (tmp_path / "book_accessible.html").write_text("<html></html>")
        # Corpus contains a file NOT matching the HTML basename
        fake_corpus = tmp_path / "different.pdf"
        fake_corpus.write_bytes(b"%PDF-1.4")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                str(fake_corpus),
                "--course-name",
                "TEST_101",
                "--skip-dart",
                "--dart-output-dir",
                str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "warning" in result.output.lower()
        assert "different" in result.output


# =============================================================================
# Phase 5 ST 1, 3, 5, 6: --blocks + --force + courseforge subcommands
# =============================================================================


class TestPhase5SupportedWorkflows:
    """Phase 5 ST 3: SUPPORTED_WORKFLOWS extension covers the four
    new ``courseforge*`` stage subcommands.
    """

    def test_supported_workflows_includes_phase5_entries(self):
        from cli.commands.run import SUPPORTED_WORKFLOWS

        for name in (
            "courseforge",
            "courseforge-outline",
            "courseforge-validate",
            "courseforge-rewrite",
        ):
            assert name in SUPPORTED_WORKFLOWS, (
                f"Phase 5 ST 3: SUPPORTED_WORKFLOWS missing {name!r}"
            )

    def test_supported_workflows_size_matches_plan(self):
        """Phase 5 §13 risk note: the set grows from 6 (legacy) +
        ``trainforge_train`` + ``textbook-to-course`` alias to 11
        canonical entries with the four new Phase 5 subcommands.
        """
        from cli.commands.run import SUPPORTED_WORKFLOWS

        # 7 pre-Phase-5 + 4 new = 11
        assert len(SUPPORTED_WORKFLOWS) == 11

    def test_unknown_workflow_lists_phase5_entries_in_error(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "no-such-workflow", "--dry-run", "--course-name", "X"]
        )
        assert result.exit_code == 2
        # The error suggestion must include at least one of the new
        # Phase 5 subcommands so an operator who typos sees the new
        # surface in the suggestion list.
        assert "courseforge" in result.output


class TestPhase5BlocksFilter:
    """Phase 5 ST 1: ``--blocks`` parsing + validation."""

    def test_parse_blocks_filter_returns_list(self):
        from cli.commands.run import _parse_blocks_filter

        out = _parse_blocks_filter("objective,concept")
        assert out == ["objective", "concept"]

    def test_parse_blocks_filter_strips_whitespace(self):
        from cli.commands.run import _parse_blocks_filter

        out = _parse_blocks_filter(" objective , concept ,  ")
        assert out == ["objective", "concept"]

    def test_parse_blocks_filter_dedupes(self):
        from cli.commands.run import _parse_blocks_filter

        out = _parse_blocks_filter("objective,concept,objective")
        assert out == ["objective", "concept"]

    def test_parse_blocks_filter_empty_returns_none(self):
        from cli.commands.run import _parse_blocks_filter

        assert _parse_blocks_filter(None) is None
        assert _parse_blocks_filter("") is None
        assert _parse_blocks_filter(",,") is None

    def test_parse_blocks_filter_rejects_invalid(self):
        from cli.commands.run import _parse_blocks_filter

        with pytest.raises(Exception) as exc_info:
            _parse_blocks_filter("invalid_type")
        # The error must list valid block types so the operator can fix
        # the typo without consulting the docs.
        msg = str(exc_info.value)
        assert "invalid_type" in msg
        assert "objective" in msg  # one canonical valid type listed
        assert "assessment_item" in msg

    def test_valid_block_types_matches_canonical_enum(self):
        """Phase 5 ST 1: the local VALID_BLOCK_TYPES tuple must mirror
        the canonical 16-singular ``BLOCK_TYPES`` frozenset at
        ``Courseforge/scripts/blocks.py:77``. Catches drift if the
        canonical enum gains or loses members.
        """
        from cli.commands.run import VALID_BLOCK_TYPES
        from Courseforge.scripts.blocks import BLOCK_TYPES

        assert set(VALID_BLOCK_TYPES) == set(BLOCK_TYPES)

    def test_blocks_appears_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--blocks" in result.output

    def test_build_params_threads_target_block_ids(self):
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="x.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
            target_block_ids=["objective", "concept"],
        )
        assert params["target_block_ids"] == ["objective", "concept"]

    def test_build_params_omits_target_block_ids_when_none(self):
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="x.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )
        assert "target_block_ids" not in params

    def test_cli_blocks_flag_dry_run(self):
        """--blocks parses through the CLI and lands in the dry-run
        params dict. Confirms end-to-end plumbing.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--blocks",
                "assessment_item,example",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["params"]["target_block_ids"] == [
            "assessment_item",
            "example",
        ]
        # Phase 5 ST 6: top-level summary field surfaces the filter.
        assert payload.get("blocks_filter") == ["assessment_item", "example"]

    def test_cli_blocks_flag_invalid_token_fails_fast(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--blocks",
                "not_a_real_type",
                "--dry-run",
            ],
        )
        # click.BadParameter exits 2 by default
        assert result.exit_code == 2
        assert "not_a_real_type" in result.output


class TestPhase5ForceFlag:
    """Phase 5 ST 5: ``--force`` flag plumbs through workflow params."""

    def test_force_appears_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_build_params_threads_force_rerun(self):
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="x.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
            force_rerun=True,
        )
        assert params["force_rerun"] is True

    def test_build_params_omits_force_rerun_when_default(self):
        params = _build_workflow_params(
            "textbook_to_course",
            corpus="x.pdf",
            course_name="X_101",
            weeks=None,
            no_assessments=False,
            assessment_count=50,
            bloom_levels="remember,understand,apply,analyze",
            priority="normal",
            objectives_path=None,
        )
        assert "force_rerun" not in params

    def test_cli_force_flag_dry_run(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--force",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["params"].get("force_rerun") is True
        assert payload.get("force_rerun") is True


class TestPhase5DryRunBlocksAnnotation:
    """Phase 5 ST 6: dry-run plan annotates phases under ``--blocks``."""

    def test_dry_run_annotates_rewrite_phase_when_blocks_set(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--blocks",
                "assessment_item",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # The rewrite phase is annotated with status=FILTERED + the
        # blocks_filter list. (The phase itself only fires when
        # COURSEFORGE_TWO_PASS=true; on a default run it may be
        # skipped via enabled_when_env, but the dry-run plan still
        # lists it.)
        rewrite = next(
            (p for p in payload["phases"]
             if p["name"] == "content_generation_rewrite"),
            None,
        )
        if rewrite is not None:
            assert rewrite.get("status") == "FILTERED"
            assert rewrite.get("blocks_filter") == ["assessment_item"]

    def test_dry_run_text_mode_shows_filter_inline(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--blocks",
                "objective",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        # Phase 5 ST 6: top-level "Blocks:" line surfaces the filter
        # for human eyeballing without needing --json.
        assert "Blocks:" in result.output
        assert "objective" in result.output

    def test_dry_run_no_blocks_omits_filter_field(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # No --blocks → no blocks_filter top-level key, and no phase
        # carries the FILTERED status.
        assert "blocks_filter" not in payload
        for phase in payload["phases"]:
            assert phase.get("status") != "FILTERED"


class TestPhase5CourseforgeStageSubcommand:
    """Phase 5 ST 3: ``courseforge-*`` subcommands alias to
    ``textbook_to_course`` while propagating a ``courseforge_stage``
    workflow param so the runner's per-tier dispatch knows which
    Phase 3 tier(s) to re-execute.
    """

    def test_courseforge_outline_aliases_to_textbook_to_course(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "courseforge-outline",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["workflow"] == "textbook_to_course"
        assert payload["params"].get("courseforge_stage") == (
            "courseforge_outline"
        )

    def test_courseforge_full_alias(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "courseforge",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["workflow"] == "textbook_to_course"
        assert payload["params"].get("courseforge_stage") == "courseforge"

    def test_courseforge_rewrite_with_blocks(self):
        """Combined --blocks + courseforge-rewrite path; the canonical
        Phase 5 §3 use case ('A/B-test rewrite-tier model swaps').
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "courseforge-rewrite",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                "--blocks",
                "assessment_item,example",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["workflow"] == "textbook_to_course"
        assert payload["params"].get("courseforge_stage") == (
            "courseforge_rewrite"
        )
        assert payload["params"]["target_block_ids"] == [
            "assessment_item",
            "example",
        ]
