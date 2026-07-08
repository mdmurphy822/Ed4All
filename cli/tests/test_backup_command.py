"""Tests for ``ed4all backup`` (OP3).

Verifies (1) the backup resolves + captures the data dirs under a tmp
``ED4ALL_HOME`` (including ``secrets.json``), (2) the archive is written
``0600``, (3) verification passes on a clean archive, and (4) verification
catches a corrupted member.
"""

from __future__ import annotations

import json
import os
import stat
import tarfile
from pathlib import Path

from click.testing import CliRunner

from cli.commands.backup import (
    backup_command,
    create_backup,
    resolve_backup_dirs,
    verify_archive,
    _verify_extracted,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _make_home(tmp_path: Path) -> Path:
    """Build a tmp ED4ALL_HOME with state (incl. secrets.json) + libv2 + captures."""
    home = tmp_path / "home"
    state = home / "state"
    (state / "runs" / "WF-1").mkdir(parents=True)
    (state / "runs" / "WF-1" / "checkpoint.json").write_text("{}")
    (state / "gui").mkdir(parents=True)
    (state / "gui" / "secrets.json").write_text('{"LOCAL_SYNTHESIS_API_KEY": "sk-REAL"}')

    (home / "libv2" / "courses").mkdir(parents=True)
    (home / "libv2" / "courses" / "note.txt").write_text("course data")

    (home / "training-captures" / "tool").mkdir(parents=True)
    (home / "training-captures" / "tool" / "d.jsonl").write_text('{"x": 1}\n')
    return home


# ---------------------------------------------------------------------- #
# resolve_backup_dirs
# ---------------------------------------------------------------------- #


def test_resolve_honors_ed4all_home(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    # Clear per-dir overrides that other tests / the env might leave set.
    for var in (
        "ED4ALL_LIBV2_ROOT",
        "ED4ALL_TRAINING_CAPTURES_DIR",
        "ED4ALL_STATE_RUNS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    dirs = resolve_backup_dirs()
    assert dirs["state"] == home / "state"
    assert dirs["libv2"] == home / "libv2"
    assert dirs["training-captures"] == home / "training-captures"


def test_resolve_captures_scattered_runs(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    scattered = tmp_path / "elsewhere" / "runs"
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(scattered))
    dirs = resolve_backup_dirs()
    # runs relocated OUT of the state root → captured separately.
    assert dirs.get("state-runs") == scattered


# ---------------------------------------------------------------------- #
# create_backup + verify
# ---------------------------------------------------------------------- #


def test_backup_includes_resolved_dirs_and_secrets(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    dirs = {
        "state": home / "state",
        "libv2": home / "libv2",
        "training-captures": home / "training-captures",
        "exports": home / "exports",  # missing on purpose
        "dart-output": home / "dart-output",  # missing on purpose
    }
    out = tmp_path / "backup.tar.gz"
    result = create_backup(out, dirs=dirs)

    assert out.exists()
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    # secrets.json IS included in a backup (restore-able system).
    assert "state/gui/secrets.json" in names
    assert "libv2/courses/note.txt" in names
    assert "training-captures/tool/d.jsonl" in names
    assert "manifest.json" in names
    # Missing dirs are reported, not fatal.
    assert set(result.missing_dirs) == {"exports", "dart-output"}


def test_backup_archive_is_0600(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    out = tmp_path / "backup.tar.gz"
    create_backup(out, dirs={"state": home / "state"})
    mode = stat.S_IMODE(os.stat(out).st_mode)
    assert mode == 0o600


def test_verify_passes_on_clean_archive(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    out = tmp_path / "backup.tar.gz"
    create_backup(out, dirs={"state": home / "state", "libv2": home / "libv2"})
    result = verify_archive(out)
    assert result.ok, result.issues
    assert result.checked > 0


def test_verify_catches_corrupted_member(tmp_path: Path) -> None:
    home = _make_home(tmp_path)
    out = tmp_path / "backup.tar.gz"
    create_backup(out, dirs={"state": home / "state"})

    # Extract, tamper a member, and verify the extracted tree directly.
    extract = tmp_path / "extract"
    extract.mkdir()
    with tarfile.open(out, "r:gz") as tar:
        tar.extractall(extract)
    target = extract / "state" / "gui" / "secrets.json"
    target.write_text('{"LOCAL_SYNTHESIS_API_KEY": "TAMPERED"}')

    result = _verify_extracted(extract)
    assert not result.ok
    assert any("corrupted member" in i for i in result.issues)


def test_verify_missing_manifest_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _verify_extracted(empty)
    assert not result.ok
    assert any("manifest.json missing" in i for i in result.issues)


# ---------------------------------------------------------------------- #
# CLI wiring
# ---------------------------------------------------------------------- #


def test_cli_create_and_verify(tmp_path: Path, monkeypatch) -> None:
    home = _make_home(tmp_path)
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    for var in ("ED4ALL_LIBV2_ROOT", "ED4ALL_TRAINING_CAPTURES_DIR", "ED4ALL_STATE_RUNS_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = tmp_path / "backup.tar.gz"

    runner = CliRunner()
    res = runner.invoke(backup_command, ["--output", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert "0600" in res.output

    res2 = runner.invoke(backup_command, ["--verify", str(out)])
    assert res2.exit_code == 0, res2.output
    assert "verified" in res2.output.lower()


def test_cli_rejects_both_modes(tmp_path: Path) -> None:
    out = tmp_path / "b.tar.gz"
    out.write_bytes(b"x")
    runner = CliRunner()
    res = runner.invoke(backup_command, ["--output", str(out), "--verify", str(out)])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.output
