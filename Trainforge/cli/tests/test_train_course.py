"""Trainforge adapter-training command contracts.

Drives the click command via :class:`click.testing.CliRunner` against a
synthetic LibV2 course in tmp_path. The CLI test exercises the full
runner path in dry-run mode (no GPU, no heavy ML deps) and asserts the
exit code is 0 + the model card path is surfaced in stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.cli import train_course  # noqa: E402


def _build_libv2_course(tmp_path: Path, slug: str = "tst-101") -> Path:
    libv2_root = tmp_path / "courses"
    course = libv2_root / slug
    (course / "corpus").mkdir(parents=True)
    (course / "graph").mkdir(parents=True)
    (course / "training_specs").mkdir(parents=True)
    (course / "corpus" / "chunks.jsonl").write_text(
        '{"id": "c1", "learning_outcome_refs": ["TO-01"]}\n',
        encoding="utf-8",
    )
    (course / "graph" / "pedagogy_graph.json").write_text(
        '{"nodes": [], "edges": []}',
        encoding="utf-8",
    )
    (course / "graph" / "concept_graph_semantic.json").write_text(
        '{"concepts": []}',
        encoding="utf-8",
    )
    (course / "graph" / "courseforge_v1.vocabulary.ttl").write_text(
        "@prefix : <http://example.com/> .",
        encoding="utf-8",
    )
    (course / "training_specs" / "instruction_pairs.jsonl").write_text(
        '{"prompt": "Q?", "completion": "A.", "chunk_id": "c1"}\n',
        encoding="utf-8",
    )
    # Admissible (editorial_or_misconception) rows clearing the default
    # min_dpo_pairs=50 — an empty preference file describes a course the runner
    # must refuse under the shipped dpo_fail_hard=true, so even a dry-run smoke
    # test needs a corpus that could legitimately train.
    (course / "training_specs" / "preference_pairs.jsonl").write_text(
        "".join(
            json.dumps({
                "prompt": f"Which statement about the CLI is correct? ({i})",
                "chosen": "The dry run plans a plan without loading weights.",
                "rejected": "The dry run loads the weights before planning.",
                "chunk_id": "c1",
                "source": "misconception",
                "misconception_id": f"mc_{i:016x}",
            }) + "\n"
            for i in range(50)
        ),
        encoding="utf-8",
    )
    (course / "training_specs" / "dataset_config.json").write_text(
        '{"format": "instruction-following"}',
        encoding="utf-8",
    )
    return libv2_root


def test_dry_run_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    libv2_root = _build_libv2_course(tmp_path)
    output_dir = tmp_path / "models_out"

    # Patch LIBV2_COURSES so the runner reads from tmp_path.
    monkeypatch.setattr(
        "Trainforge.training.runner.LIBV2_COURSES", libv2_root,
    )

    runner = CliRunner()
    result = runner.invoke(
        train_course.train_course_command,
        [
            "--course-code", "TST_101",
            "--base-model", "qwen2.5-1.5b",
            "--dry-run",
            "--output-dir", str(output_dir),
        ],
    )
    assert result.exit_code == 0, (
        f"CLI dry-run failed; output:\n{result.output}\n"
        f"exception: {result.exception!r}"
    )
    assert "Training run complete" in result.output
    # Find the printed model_card path in stdout, parse, and confirm exists.
    card_lines = [
        ln for ln in result.output.splitlines() if "Model card:" in ln
    ]
    assert card_lines, f"Expected 'Model card:' in output, got:\n{result.output}"
    card_path = Path(card_lines[0].split("Model card:")[1].strip())
    assert card_path.exists(), f"Card path printed but not on disk: {card_path}"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["course_slug"] == "tst-101"
    assert card["base_model"]["name"] == "qwen2.5-1.5b"


def test_cli_help_exposes_only_the_supported_local_training_path():
    result = CliRunner().invoke(train_course.train_course_command, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--backend" not in result.output
    assert "runpod" not in result.output.lower()
    assert "--dry-run" in result.output


def test_cli_rejects_removed_backend_option():
    result = CliRunner().invoke(
        train_course.train_course_command,
        [
            "--course-code", "TST_101",
            "--base-model", "qwen2.5-1.5b",
            "--backend", "local",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "No such option '--backend'" in result.output


def test_cli_unknown_base_model_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Click should reject an unknown --base-model with exit_code != 0."""
    libv2_root = _build_libv2_course(tmp_path)
    monkeypatch.setattr(
        "Trainforge.training.runner.LIBV2_COURSES", libv2_root,
    )
    runner = CliRunner()
    result = runner.invoke(
        train_course.train_course_command,
        [
            "--course-code", "TST_101",
            "--base-model", "no-such-model",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
