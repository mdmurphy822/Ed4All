"""Tests for the ``ed4all state prune --training-captures`` extension (D5).

The training-captures age-prune is OPT-IN (never runs without the flag) and
age-based (only files older than ``--older-than`` days are eligible). All tests
run against a synthetic ``tmp_path`` captures tree — never the real
``runtime/training-captures/``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.state_prune import (
    DEFAULT_TRAINING_CAPTURES_OLDER_THAN,
    CaptureFile,
    build_capture_plan,
    state_group,
    _scan_capture_files,
)


def _capture(captures_dir: Path, rel: str, *, age_days: float, body: bytes = b"{}\n") -> Path:
    path = captures_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    # Backdate mtime by age_days.
    mtime = time.time() - (age_days * 86400)
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def captures_root(tmp_path: Path) -> Path:
    root = tmp_path / "runtime/training-captures"
    root.mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- #
# Unit: build_capture_plan age filtering
# --------------------------------------------------------------------------- #


def test_build_capture_plan_drops_only_old_files(captures_root):
    old = _capture(captures_root, "courseforge/A/phase_x/decisions_old.jsonl", age_days=45)
    recent = _capture(captures_root, "courseforge/A/phase_x/decisions_new.jsonl", age_days=5)
    captures = _scan_capture_files(captures_root)
    plan = build_capture_plan(captures, older_than_days=30)
    dropped = {c.path for c in plan.drop_captures}
    assert old in dropped
    assert recent not in dropped
    assert plan.keep_count == 1


def test_build_capture_plan_boundary_is_kept(captures_root):
    # A file exactly at the threshold is KEPT (strict < cutoff).
    boundary = _capture(captures_root, "a/decisions.jsonl", age_days=30)
    captures = _scan_capture_files(captures_root)
    fixed_now = time.time()
    # Reset mtime to exactly now - 30d using the same now reference.
    mtime = fixed_now - (30 * 86400)
    os.utime(boundary, (mtime, mtime))
    captures = _scan_capture_files(captures_root)
    plan = build_capture_plan(captures, older_than_days=30, now=fixed_now)
    assert not plan.drop_captures, "a file at exactly the cutoff must be kept"


def test_scan_skips_gitkeep(captures_root):
    (captures_root / ".gitkeep").write_text("", encoding="utf-8")
    _capture(captures_root, "a/decisions.jsonl", age_days=1)
    captures = _scan_capture_files(captures_root)
    assert all(c.path.name != ".gitkeep" for c in captures)


# --------------------------------------------------------------------------- #
# CLI: opt-in only + default untouched
# --------------------------------------------------------------------------- #


def _fake_state(tmp_path: Path) -> Path:
    (tmp_path / "state" / "workflows").mkdir(parents=True)
    (tmp_path / "state" / "runs").mkdir(parents=True)
    return tmp_path / "state"


def test_default_run_does_not_touch_captures(tmp_path, captures_root):
    state_root = _fake_state(tmp_path)
    old = _capture(captures_root, "courseforge/A/phase_x/decisions.jsonl", age_days=400)
    runner = CliRunner()
    result = runner.invoke(
        state_group,
        ["prune", "--state-root", str(state_root),
         "--training-captures-root", str(captures_root)],
    )
    assert result.exit_code == 0, result.output
    # Without --training-captures the file survives even at 400 days old.
    assert old.exists(), "default prune must NEVER touch training captures"
    assert "Training captures" not in result.output


def test_opt_in_prunes_old_captures(tmp_path, captures_root):
    state_root = _fake_state(tmp_path)
    old = _capture(captures_root, "courseforge/A/phase_x/decisions_old.jsonl", age_days=90)
    recent = _capture(captures_root, "courseforge/A/phase_x/decisions_new.jsonl", age_days=2)
    runner = CliRunner()
    result = runner.invoke(
        state_group,
        ["prune", "--state-root", str(state_root),
         "--training-captures", "--older-than", "30",
         "--training-captures-root", str(captures_root)],
    )
    assert result.exit_code == 0, result.output
    assert not old.exists(), "old capture must be pruned"
    assert recent.exists(), "recent capture must survive"
    assert "Training captures" in result.output


def test_opt_in_dry_run_keeps_everything(tmp_path, captures_root):
    state_root = _fake_state(tmp_path)
    old = _capture(captures_root, "a/decisions.jsonl", age_days=90)
    runner = CliRunner()
    result = runner.invoke(
        state_group,
        ["prune", "--state-root", str(state_root), "--dry-run",
         "--training-captures", "--older-than", "30",
         "--training-captures-root", str(captures_root)],
    )
    assert result.exit_code == 0, result.output
    assert old.exists(), "dry-run must not delete anything"
    assert "Would delete training-capture files" in result.output


def test_opt_in_prunes_emptied_subdirs(tmp_path, captures_root):
    state_root = _fake_state(tmp_path)
    _capture(captures_root, "courseforge/A/phase_x/decisions.jsonl", age_days=90)
    runner = CliRunner()
    runner.invoke(
        state_group,
        ["prune", "--state-root", str(state_root),
         "--training-captures", "--older-than", "30",
         "--training-captures-root", str(captures_root)],
    )
    # The now-empty subdir tree is swept; the root survives.
    assert captures_root.exists()
    assert not (captures_root / "courseforge" / "A" / "phase_x").exists()


def test_default_older_than_constant():
    assert DEFAULT_TRAINING_CAPTURES_OLDER_THAN == 30
