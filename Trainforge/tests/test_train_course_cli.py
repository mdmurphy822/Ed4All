"""Wave 90 — Trainforge.train_course CLI smoke test.

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge import train_course  # noqa: E402


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
    (course / "training_specs" / "preference_pairs.jsonl").write_text(
        "",
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
