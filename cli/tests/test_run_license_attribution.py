"""B3 — ``--license-note`` / ``--attribution`` CLI surface coverage.

Pins that the two optional flags:

* are accepted by ``ed4all run`` (parse + dry-run OK);
* thread into workflow params under ``license_note`` / ``attribution`` so the
  ``libv2_archival`` phase routing carries them to the manifest emit;
* are absent from params when unset (byte-identical param dict).
"""

from __future__ import annotations

from click.testing import CliRunner

from cli.commands.run import _build_workflow_params
from cli.main import cli


def test_build_params_threads_license_and_attribution():
    params = _build_workflow_params(
        "textbook_to_course",
        corpus=None,
        course_name="DEMO",
        weeks=None,
        no_assessments=False,
        assessment_count=5,
        bloom_levels="remember",
        priority="normal",
        objectives_path=None,
        license_note="CC0-1.0",
        attribution="Access for free at example.org",
    )
    assert params["license_note"] == "CC0-1.0"
    assert params["attribution"] == "Access for free at example.org"


def test_build_params_omits_when_unset():
    params = _build_workflow_params(
        "textbook_to_course",
        corpus=None,
        course_name="DEMO",
        weeks=None,
        no_assessments=False,
        assessment_count=5,
        bloom_levels="remember",
        priority="normal",
        objectives_path=None,
    )
    assert "license_note" not in params
    assert "attribution" not in params


def test_run_dry_run_accepts_flags(tmp_path):
    pdf = tmp_path / "corpus.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "textbook-to-course",
            "--corpus",
            str(pdf),
            "--course-name",
            "DEMO",
            "--license-note",
            "CC0-1.0",
            "--attribution",
            "Access for free at example.org",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
