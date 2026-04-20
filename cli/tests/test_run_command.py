"""Tests for the ``ed4all run`` CLI command (Wave 7)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

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


class TestDeprecationWarning:
    def test_legacy_textbook_to_course_shows_warning(self, tmp_path):
        runner = CliRunner()
        # Use --dry-run to avoid actually running the pipeline
        fake_pdf = tmp_path / "x.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        result = runner.invoke(
            cli,
            [
                "textbook-to-course",
                str(fake_pdf),
                "-n",
                "TEST_101",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        # Deprecation warning goes to stderr (err=True in click.secho)
        combined = result.output + (result.stderr_bytes.decode() if result.stderr_bytes else "")
        assert "DEPRECATED" in combined

    def test_legacy_command_help_still_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["textbook-to-course", "--help"])
        assert result.exit_code == 0
        assert "DEPRECATED" in result.output


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
