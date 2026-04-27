"""Wave 103 - tests for the per-course eval-config loader.

Covers:
* Per-course eval_config.yaml wins over the default.
* Default fallback fires + flags is_default=True when the per-course
  file is missing.
* Both hashes (config + prompt_template) compute to deterministic
  64-char SHA-256 strings that match independent re-hashes.
* Required-key enforcement fails closed when a locked variable is
  removed from the per-course config.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_course_eval_dir(course_path: Path) -> None:
    """Stage a synthetic per-course eval/ tree."""
    eval_dir = course_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "prompt_template.txt").write_text(
        "Per-course template.\n{context_section}\nQ: {question}\n",
        encoding="utf-8",
    )
    (eval_dir / "rubric.md").write_text("# rubric\n", encoding="utf-8")
    (eval_dir / "eval_config.yaml").write_text(
        (
            "benchmark: ED4ALL-Bench\n"
            "benchmark_version: '1.0'\n"
            "top_k: 5\n"
            "temperature: 0.0\n"
            "top_p: 1.0\n"
            "max_new_tokens: 256\n"
            "seed: 42\n"
            "prompt_template_file: prompt_template.txt\n"
            "rubric_file: rubric.md\n"
        ),
        encoding="utf-8",
    )


def test_per_course_config_wins_over_default(tmp_path):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "tst-101"
    course.mkdir(parents=True)
    _write_course_eval_dir(course)

    loaded = load_eval_config(course)
    assert loaded.is_default is False
    assert loaded.config_path == course / "eval" / "eval_config.yaml"
    assert loaded.config["benchmark"] == "ED4ALL-Bench"
    assert "Per-course template." in loaded.prompt_template


def test_default_fallback_when_per_course_missing(tmp_path, caplog):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "no-eval"
    course.mkdir(parents=True)

    with caplog.at_level("WARNING"):
        loaded = load_eval_config(course)
    assert loaded.is_default is True
    # Default config carries ED4ALL-Bench branding too
    assert loaded.config["benchmark"] == "ED4ALL-Bench"
    # The warning should mention the course slug
    assert any("no-eval" in rec.message for rec in caplog.records)


def test_eval_config_hash_is_deterministic(tmp_path):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "tst-101"
    course.mkdir(parents=True)
    _write_course_eval_dir(course)

    a = load_eval_config(course)
    b = load_eval_config(course)
    assert a.eval_config_hash == b.eval_config_hash
    assert len(a.eval_config_hash) == 64
    # Independent re-hash matches
    canonical = json.dumps(a.config, sort_keys=True, separators=(",", ":"))
    assert (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        == a.eval_config_hash
    )


def test_eval_prompt_template_hash_matches_file_bytes(tmp_path):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "tst-101"
    course.mkdir(parents=True)
    _write_course_eval_dir(course)

    loaded = load_eval_config(course)
    template_text = loaded.prompt_template
    assert (
        hashlib.sha256(template_text.encode("utf-8")).hexdigest()
        == loaded.eval_prompt_template_hash
    )
    assert len(loaded.eval_prompt_template_hash) == 64


def test_missing_required_key_fails_closed(tmp_path):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "tst-101"
    course.mkdir(parents=True)
    _write_course_eval_dir(course)
    # Strip a required key and reserialize
    config_path = course / "eval" / "eval_config.yaml"
    text = config_path.read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("top_k:")
    ) + "\n"
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_eval_config(course)
    assert "top_k" in str(exc_info.value)


def test_template_hash_changes_when_content_changes(tmp_path):
    from Trainforge.eval.eval_config import load_eval_config

    course = tmp_path / "courses" / "tst-101"
    course.mkdir(parents=True)
    _write_course_eval_dir(course)
    a = load_eval_config(course)

    (course / "eval" / "prompt_template.txt").write_text(
        "different template\n{context_section}\n{question}\n",
        encoding="utf-8",
    )
    b = load_eval_config(course)
    assert a.eval_prompt_template_hash != b.eval_prompt_template_hash
