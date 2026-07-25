"""``--auto-name`` CLI flag on ``ed4all run`` — parse-time contracts.

Locks:

* flag threading: ``--auto-name`` -> ``params["auto_name"] = True``
  (``_build_workflow_params``) and into the dry-run params surface;
* provisional derivation from the corpus filename when ``--course-name``
  is omitted (``_derive_provisional_course_name``);
* the fail-fast validation arms (non-textbook workflow, courseforge stage
  subcommand, underivable provisional);
* off-by-default: no ``auto_name`` key on params -> byte-identical.

All fixtures are synthetic — no course-data references (project rule).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from cli.commands.run import (
    _build_workflow_params,
    _derive_provisional_course_name,
)
from cli.main import cli


def _params(**overrides):
    kwargs = dict(
        corpus="inputs/synthetic/fake.pdf",
        course_name="prov-name",
        weeks=None,
        no_assessments=False,
        assessment_count=50,
        bloom_levels="remember,understand",
        priority="normal",
        objectives_path=None,
    )
    kwargs.update(overrides)
    return _build_workflow_params("textbook_to_course", **kwargs)


class TestParamsThreading:
    def test_auto_name_on_threads_param(self):
        params = _params(auto_name=True)
        assert params["auto_name"] is True

    def test_auto_name_off_is_byte_identical(self):
        assert "auto_name" not in _params()
        assert "auto_name" not in _params(auto_name=False)


class TestProvisionalDerivation:
    def test_from_single_pdf_filename(self, tmp_path):
        assert (
            _derive_provisional_course_name("inputs/x/My Book Title.pdf")
            == "my-book-title"
        )

    def test_from_directory_name(self, tmp_path):
        d = tmp_path / "Distributed Systems Docs"
        d.mkdir()
        assert (
            _derive_provisional_course_name(str(d))
            == "distributed-systems-docs"
        )

    def test_from_comma_list_uses_first(self):
        assert (
            _derive_provisional_course_name("a/first-doc.pdf,b/second.pdf")
            == "first-doc"
        )

    def test_caps_length_at_40(self):
        name = _derive_provisional_course_name(("word-" * 20) + ".pdf")
        assert name is not None and len(name) <= 40
        assert not name.endswith("-")

    def test_none_when_underivable(self):
        assert _derive_provisional_course_name("!!!.pdf") is None


class TestCliSurface:
    def test_help_lists_auto_name(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--auto-name" in result.output

    def test_dry_run_carries_auto_name_param(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run", "textbook-to-course",
                "--corpus", "inputs/synthetic/fake.pdf",
                "--course-name", "prov-name",
                "--auto-name", "--dry-run", "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output)
        assert plan["params"]["auto_name"] is True
        # Provisional identity is preserved until post-conversion resolution.
        assert plan["params"]["course_name"] == "prov-name"

    def test_dry_run_derives_provisional_without_course_name(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run", "textbook-to-course",
                "--corpus", "inputs/synthetic/Fake Manual.pdf",
                "--auto-name", "--dry-run", "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = result.output[result.output.index("{"):]
        plan = json.loads(payload)
        assert plan["params"]["course_name"] == "fake-manual"
        assert plan["params"]["auto_name"] is True

    def test_rejected_on_non_textbook_workflow(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run", "rag_training",
                "--corpus", "inputs/synthetic/fake.imscc",
                "--course-name", "prov-name",
                "--auto-name", "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "--auto-name" in result.output

    def test_rejected_on_courseforge_stage_subcommand(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run", "courseforge-rewrite",
                "--course-name", "prov-name",
                "--auto-name", "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "--auto-name" in result.output

    def test_off_by_default_no_param(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run", "textbook-to-course",
                "--corpus", "inputs/synthetic/fake.pdf",
                "--course-name", "prov-name",
                "--dry-run", "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output)
        assert "auto_name" not in plan["params"]
