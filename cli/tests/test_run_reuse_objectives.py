"""Wave 80 Worker A — ``--reuse-objectives`` CLI surface coverage.

Pins the CLI flag's parse-time validation contract:

* CLI accepts the flag and threads it into workflow params.
* Both supported shapes (Courseforge synthesized + Wave 75 LibV2
  archive) are accepted.
* Missing file → fast error (exit code 2).
* Malformed JSON → fast error.
* Empty terminal list → fast error.
* Non-textbook workflow → fast error (matches the --skip-dart
  precedent — only textbook_to_course has a course_planning phase to
  reuse).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.run import (
    _build_workflow_params,
    _validate_reuse_objectives_file,
)
from cli.main import cli


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def courseforge_objectives_file(tmp_path: Path) -> Path:
    """Minimal valid Courseforge-form objectives file."""
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(
        json.dumps({
            "course_name": "TEST_101",
            "duration_weeks": 8,
            "terminal_objectives": [
                {
                    "id": "TO-01",
                    "statement": "Understand the foundations of X.",
                    "bloom_level": "understand",
                },
            ],
            "chapter_objectives": [
                {
                    "chapter": "Week 1",
                    "objectives": [
                        {
                            "id": "CO-01",
                            "statement": "Identify key terms in X.",
                            "parent_terminal": "TO-01",
                            "bloom_level": "remember",
                        },
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def libv2_objectives_file(tmp_path: Path) -> Path:
    """Minimal valid Wave 75 LibV2-archive-form objectives file."""
    p = tmp_path / "objectives.json"
    p.write_text(
        json.dumps({
            "schema_version": "v1",
            "course_code": "test_101",
            "terminal_outcomes": [
                {
                    "id": "to-01",
                    "statement": "Understand the foundations of X.",
                    "bloom_level": "understand",
                },
            ],
            "component_objectives": [
                {
                    "id": "co-01",
                    "statement": "Identify key terms in X.",
                    "parent_terminal": "to-01",
                    "bloom_level": "remember",
                },
            ],
            "objective_count": {"terminal": 1, "component": 1},
        }),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------
# Pure validator coverage
# ---------------------------------------------------------------------


class TestValidateReuseObjectivesFile:
    def test_accepts_courseforge_form(self, courseforge_objectives_file):
        err = _validate_reuse_objectives_file(str(courseforge_objectives_file))
        assert err is None

    def test_accepts_libv2_archive_form(self, libv2_objectives_file):
        err = _validate_reuse_objectives_file(str(libv2_objectives_file))
        assert err is None

    def test_missing_file_fails(self, tmp_path):
        err = _validate_reuse_objectives_file(
            str(tmp_path / "no_such_file.json")
        )
        assert err is not None
        assert "not found" in err

    def test_directory_fails(self, tmp_path):
        err = _validate_reuse_objectives_file(str(tmp_path))
        assert err is not None
        # Either "must be a file" or "not found" is acceptable per
        # platform; both reject the directory.
        assert "file" in err.lower() or "not found" in err.lower()

    def test_malformed_json_fails(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        err = _validate_reuse_objectives_file(str(p))
        assert err is not None
        assert "JSON" in err

    def test_top_level_array_fails(self, tmp_path):
        p = tmp_path / "array.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        err = _validate_reuse_objectives_file(str(p))
        assert err is not None
        assert "object" in err.lower()

    def test_unknown_shape_fails(self, tmp_path):
        p = tmp_path / "weird.json"
        p.write_text(
            json.dumps({"foo": "bar", "baz": [1, 2, 3]}), encoding="utf-8",
        )
        err = _validate_reuse_objectives_file(str(p))
        assert err is not None
        assert "recognised shape" in err

    def test_empty_terminal_courseforge_fails(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(
            json.dumps({
                "terminal_objectives": [],
                "chapter_objectives": [],
            }),
            encoding="utf-8",
        )
        err = _validate_reuse_objectives_file(str(p))
        assert err is not None
        assert "zero terminal" in err

    def test_empty_terminal_libv2_fails(self, tmp_path):
        p = tmp_path / "empty_libv2.json"
        p.write_text(
            json.dumps({
                "schema_version": "v1",
                "course_code": "x",
                "terminal_outcomes": [],
                "component_objectives": [],
                "objective_count": {"terminal": 0, "component": 0},
            }),
            encoding="utf-8",
        )
        err = _validate_reuse_objectives_file(str(p))
        assert err is not None
        assert "zero terminal" in err


# ---------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------


class TestReuseObjectivesCli:
    def test_flag_appears_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--reuse-objectives" in result.output

    def test_build_params_sets_reuse_key(self, courseforge_objectives_file):
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
            reuse_objectives=str(courseforge_objectives_file),
        )
        assert params["reuse_objectives_path"] == str(
            courseforge_objectives_file
        )

    def test_build_params_default_no_reuse_key(self):
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
        assert "reuse_objectives_path" not in params

    def test_dry_run_with_reuse_marks_phase_reused(
        self, courseforge_objectives_file
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "REUSE_101",
                "--reuse-objectives",
                str(courseforge_objectives_file),
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        cp_phase = next(
            p for p in payload["phases"] if p["name"] == "course_planning"
        )
        assert cp_phase.get("status") == "REUSED"
        assert "reuse" in cp_phase.get("reuse_reason", "").lower()
        assert (
            payload["params"]["reuse_objectives_path"]
            == str(courseforge_objectives_file)
        )

    def test_dry_run_with_libv2_form_accepted(self, libv2_objectives_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "REUSE_LIBV2",
                "--reuse-objectives",
                str(libv2_objectives_file),
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        cp_phase = next(
            p for p in payload["phases"] if p["name"] == "course_planning"
        )
        assert cp_phase.get("status") == "REUSED"

    def test_missing_file_fails_fast(self):
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
                "--reuse-objectives",
                "/definitely/does/not/exist.json",
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "not found" in result.output.lower()

    def test_malformed_file_fails_fast(self, tmp_path):
        p = tmp_path / "garbage.json"
        p.write_text("{not parseable", encoding="utf-8")
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
                "--reuse-objectives",
                str(p),
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "json" in result.output.lower()

    def test_empty_terminal_fails_fast(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(
            json.dumps({
                "terminal_objectives": [],
                "chapter_objectives": [],
            }),
            encoding="utf-8",
        )
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
                "--reuse-objectives",
                str(p),
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "zero terminal" in result.output

    def test_unknown_shape_fails_fast(self, tmp_path):
        p = tmp_path / "weird.json"
        p.write_text(
            json.dumps({"some_other_key": [1, 2, 3]}), encoding="utf-8",
        )
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
                "--reuse-objectives",
                str(p),
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "recognised shape" in result.output

    def test_rejects_non_textbook_workflow(self, courseforge_objectives_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "rag_training",
                "--corpus",
                "course.imscc",
                "--course-name",
                "X_101",
                "--reuse-objectives",
                str(courseforge_objectives_file),
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "reuse-objectives" in result.output.lower()
        assert "textbook_to_course" in result.output
