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
