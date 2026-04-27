"""Wave 103 - CliRunner tests for the new ``libv2 eval`` subcommands.

Covers:
* ``libv2 eval init <slug>`` scaffolds the four files into
  ``courses/<slug>/eval/`` and is idempotent on a second run.
* ``libv2 eval validate <slug>`` succeeds on a well-formed dir,
  fails when a required key is missing or a placeholder is dropped.
* ``libv2 eval run <slug> <model_id>`` with the new two-arg form
  dispatches the ED4ALL-Bench branch (legacy one-arg form still
  routes to retrieval-eval).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402


def _stage_course(repo_root: Path, slug: str) -> Path:
    course_dir = repo_root / "courses" / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    # Minimal manifest so course-existence checks pass.
    (course_dir / "manifest.json").write_text(
        json.dumps({"slug": slug, "classification": {}}), encoding="utf-8"
    )
    return course_dir


def test_eval_init_scaffolds_four_files(tmp_path):
    repo_root = tmp_path / "libv2"
    course_dir = _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "eval", "init", "tst-101",
    ])
    assert result.exit_code == 0, result.output
    eval_dir = course_dir / "eval"
    for fname in ("prompt_template.txt", "rubric.md", "eval_config.yaml",
                  "holdout_split.json"):
        assert (eval_dir / fname).exists(), f"missing {fname}"


def test_eval_init_is_idempotent(tmp_path):
    repo_root = tmp_path / "libv2"
    course_dir = _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    # Edit one of the files; second init must not overwrite.
    custom_template = "MY CUSTOM TEMPLATE\n{context_section}\n{question}\n"
    (course_dir / "eval" / "prompt_template.txt").write_text(
        custom_template, encoding="utf-8"
    )
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    assert result.exit_code == 0, result.output
    after = (course_dir / "eval" / "prompt_template.txt").read_text(
        encoding="utf-8"
    )
    assert after == custom_template


def test_eval_validate_ok_on_well_formed(tmp_path):
    repo_root = tmp_path / "libv2"
    _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "validate", "tst-101",
    ])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_eval_validate_fails_when_placeholders_missing(tmp_path):
    repo_root = tmp_path / "libv2"
    course_dir = _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    # Drop the placeholders from the template
    (course_dir / "eval" / "prompt_template.txt").write_text(
        "this template has no placeholders\n", encoding="utf-8"
    )
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "validate", "tst-101",
    ])
    assert result.exit_code != 0


def test_eval_validate_fails_when_required_key_missing(tmp_path):
    repo_root = tmp_path / "libv2"
    course_dir = _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    config_path = course_dir / "eval" / "eval_config.yaml"
    text = config_path.read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("top_k:")
    ) + "\n"
    config_path.write_text(text, encoding="utf-8")
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "validate", "tst-101",
    ])
    assert result.exit_code != 0


def test_eval_validate_unknown_course_fails(tmp_path):
    repo_root = tmp_path / "libv2"
    (repo_root / "courses").mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "validate", "no-such",
    ])
    assert result.exit_code != 0


def test_eval_run_two_arg_form_dispatches_ed4all_bench(tmp_path):
    """When MODEL_ID is supplied, the runner takes the ED4ALL-Bench
    branch and prints a kickoff status (the adapter bridge wave is
    deferred but the surface is present)."""
    repo_root = tmp_path / "libv2"
    course_dir = _stage_course(repo_root, "tst-101")
    # Minimal model dir so the existence check passes.
    model_dir = course_dir / "models" / "tst-101-qwen2-5-1-5b-aaaa1111"
    model_dir.mkdir(parents=True)

    runner = CliRunner()
    runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "init", "tst-101",
    ])
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "eval", "run", "tst-101", "tst-101-qwen2-5-1-5b-aaaa1111",
        "--judge", "none",
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["course"] == "tst-101"
    assert payload["model_id"] == "tst-101-qwen2-5-1-5b-aaaa1111"
    assert payload["judge"] == "none"


def test_eval_run_two_arg_form_unknown_model_fails(tmp_path):
    repo_root = tmp_path / "libv2"
    _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "eval", "run", "tst-101", "no-such-model-id",
    ])
    assert result.exit_code != 0


def test_eval_run_legacy_one_arg_form_still_works(tmp_path):
    """Legacy retrieval-eval path requires quality/eval_set.json -
    we don't stage one, so the command must surface the documented
    fail-loud message rather than crashing."""
    repo_root = tmp_path / "libv2"
    _stage_course(repo_root, "tst-101")

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root), "eval", "run", "tst-101",
    ])
    assert result.exit_code != 0
    assert "eval_set" in result.output.lower() or "generate" in result.output.lower()


def test_eval_group_help_lists_init_and_validate():
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["eval", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "validate" in result.output
    assert "run" in result.output
