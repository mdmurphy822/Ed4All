"""Backup / restore service + CLI tests (W8.7 disaster-recovery path).

Synthetic ``tmp_path`` libraries only — no real LibV2 dir is ever touched. Covers
the metadata-only snapshot selection, tar + dir formats, checksum verification
(incl. corruption detection), idempotent/resumable restore, non-clobbering of a
divergent live file, dry-run, path-traversal refusal, and the CLI round-trips.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from LibV2.tools.libv2 import backup as bk
from LibV2.tools.libv2.cli import main as libv2_main


# --------------------------------------------------------------------------- #
# Fixture library builder
# --------------------------------------------------------------------------- #


def _make_library(root: Path) -> Path:
    """Build a minimal but realistic LibV2 metadata store under ``root``."""
    catalog = root / "catalog"
    (catalog / "by_domain").mkdir(parents=True)
    catalog.joinpath("master_catalog.json").write_text(
        json.dumps({"version": "1.0.0", "total_courses": 2, "courses": []}),
        encoding="utf-8",
    )
    catalog.joinpath("course_index.json").write_text(
        json.dumps({"demo-101": "courses/demo-101"}), encoding="utf-8"
    )
    catalog.joinpath("by_domain", "physics.json").write_text(
        json.dumps({"courses": []}), encoding="utf-8"
    )

    for slug in ("demo-101", "demo-202"):
        cdir = root / "courses" / slug
        (cdir / "imscc_chunks").mkdir(parents=True)
        # A big chunk body that must NEVER be captured by a metadata backup.
        (cdir / "imscc_chunks" / "chunks.jsonl").write_bytes(b"CHUNK" * 10_000)
        cdir.joinpath("manifest.json").write_text(
            json.dumps({"slug": slug, "classification": {"domain": "physics"}}),
            encoding="utf-8",
        )
        cdir.joinpath("course.json").write_text(
            json.dumps({"course_code": slug.upper(), "learning_outcomes": []}),
            encoding="utf-8",
        )
    return root


# --------------------------------------------------------------------------- #
# Snapshot selection (read-only, metadata-only)
# --------------------------------------------------------------------------- #


def test_snapshot_selects_catalog_and_course_metadata_only(tmp_path):
    root = _make_library(tmp_path / "libv2")
    rels = bk.iter_snapshot_files(root)

    assert "catalog/master_catalog.json" in rels
    assert "catalog/course_index.json" in rels
    assert "catalog/by_domain/physics.json" in rels
    assert "courses/demo-101/manifest.json" in rels
    assert "courses/demo-101/course.json" in rels
    assert "courses/demo-202/manifest.json" in rels
    # The multi-MB chunk body is NEVER snapshotted.
    assert not any("chunks.jsonl" in r for r in rels)
    # Deterministic ordering.
    assert rels == sorted(rels)


def test_backup_is_read_only_over_live_store(tmp_path):
    root = _make_library(tmp_path / "libv2")
    chunk = root / "courses" / "demo-101" / "imscc_chunks" / "chunks.jsonl"
    before = chunk.read_bytes()
    bk.create_backup(root, tmp_path / "b.tar.gz")
    assert chunk.read_bytes() == before, "backup must not touch course content"


# --------------------------------------------------------------------------- #
# Tar + dir backup / verify round-trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt,name", [("tar", "b.tar.gz"), ("dir", "bdir")])
def test_backup_then_verify_passes(tmp_path, fmt, name):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / name
    result = bk.create_backup(root, dest, fmt=fmt)
    assert result.manifest.course_count == 2
    assert result.manifest.file_count == len(bk.iter_snapshot_files(root))

    v = bk.verify_backup(dest)
    assert v.passed
    assert not v.mismatched and not v.missing
    assert len(v.ok) == result.manifest.file_count


def test_tar_backup_contains_manifest_and_data_prefix(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    bk.create_backup(root, dest)
    with tarfile.open(dest, "r:*") as tf:
        names = tf.getnames()
    assert bk.BACKUP_MANIFEST_NAME in names
    assert "data/catalog/master_catalog.json" in names
    assert "data/courses/demo-101/manifest.json" in names


def test_backup_refuses_overwrite_without_force(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    bk.create_backup(root, dest)
    with pytest.raises(bk.BackupError) as exc:
        bk.create_backup(root, dest)
    assert exc.value.code == "dest_exists"
    # --force overwrites.
    bk.create_backup(root, dest, force=True)


def test_backup_missing_root_errors(tmp_path):
    with pytest.raises(bk.BackupError) as exc:
        bk.create_backup(tmp_path / "nope", tmp_path / "b.tar.gz")
    assert exc.value.code == "root_not_found"


# --------------------------------------------------------------------------- #
# Corruption detection
# --------------------------------------------------------------------------- #


def test_verify_detects_corruption(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "bdir"
    bk.create_backup(root, dest, fmt="dir")
    # Tamper a data file after the manifest recorded its checksum.
    tampered = dest / "data" / "courses" / "demo-101" / "manifest.json"
    tampered.write_text('{"slug": "TAMPERED"}', encoding="utf-8")
    v = bk.verify_backup(dest)
    assert not v.passed
    assert "courses/demo-101/manifest.json" in v.mismatched


def test_verify_detects_missing_member(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "bdir"
    bk.create_backup(root, dest, fmt="dir")
    (dest / "data" / "catalog" / "course_index.json").unlink()
    v = bk.verify_backup(dest)
    assert not v.passed
    assert "catalog/course_index.json" in v.missing


def test_restore_refuses_corrupt_backup(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "bdir"
    bk.create_backup(root, dest, fmt="dir")
    (dest / "data" / "courses" / "demo-101" / "manifest.json").write_text(
        "corrupt", encoding="utf-8"
    )
    target = tmp_path / "restored"
    with pytest.raises(bk.BackupError) as exc:
        bk.restore_backup(target, dest)
    assert exc.value.code == "corrupt_member"
    # Fail-closed: nothing written (target dir never created content).
    assert not (target / "catalog").exists()


def test_bad_backup_path_errors(tmp_path):
    junk = tmp_path / "notabackup.txt"
    junk.write_text("hello", encoding="utf-8")
    with pytest.raises(bk.BackupError) as exc:
        bk.verify_backup(junk)
    assert exc.value.code == "bad_backup"


# --------------------------------------------------------------------------- #
# Restore: round-trip, idempotent/resumable, non-clobbering, dry-run
# --------------------------------------------------------------------------- #


def test_restore_round_trip(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    bk.create_backup(root, dest)

    target = tmp_path / "restored"
    result = bk.restore_backup(target, dest)
    assert len(result.restored) == bk.verify_backup(dest).ok.__len__()
    # Content matches the source metadata byte for byte.
    assert (
        (target / "courses" / "demo-101" / "manifest.json").read_bytes()
        == (root / "courses" / "demo-101" / "manifest.json").read_bytes()
    )
    assert (target / "catalog" / "master_catalog.json").is_file()
    # The chunk body was never captured, so it is never restored.
    assert not (target / "courses" / "demo-101" / "imscc_chunks").exists()


def test_restore_idempotent_and_resumable(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "bdir"
    bk.create_backup(root, dest, fmt="dir")
    target = tmp_path / "restored"

    first = bk.restore_backup(target, dest)
    assert first.restored and not first.skipped

    # Second run: everything already matches → all skipped, none rewritten.
    second = bk.restore_backup(target, dest)
    assert not second.restored
    assert len(second.skipped) == len(first.restored)

    # Resumable: drop one restored file, re-run → only that one is rewritten.
    (target / "catalog" / "course_index.json").unlink()
    third = bk.restore_backup(target, dest)
    assert third.restored == ["catalog/course_index.json"]


def test_restore_does_not_clobber_divergent_live_file(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    bk.create_backup(root, dest)
    target = _make_library(tmp_path / "live")
    # Diverge a live manifest from the backup.
    live_manifest = target / "courses" / "demo-101" / "manifest.json"
    live_manifest.write_text('{"slug": "demo-101", "edited": true}', encoding="utf-8")
    divergent_bytes = live_manifest.read_bytes()

    result = bk.restore_backup(target, dest)
    assert "courses/demo-101/manifest.json" in result.conflicted
    # Left untouched.
    assert live_manifest.read_bytes() == divergent_bytes

    # With --overwrite it IS replaced.
    result2 = bk.restore_backup(target, dest, overwrite=True)
    assert "courses/demo-101/manifest.json" in result2.restored
    assert (
        live_manifest.read_bytes()
        == (root / "courses" / "demo-101" / "manifest.json").read_bytes()
    )


def test_restore_dry_run_writes_nothing(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    bk.create_backup(root, dest)
    target = tmp_path / "restored"

    result = bk.restore_backup(target, dest, dry_run=True)
    assert result.dry_run
    assert result.planned and not result.restored
    assert not (target / "catalog").exists(), "dry-run must not write"


# --------------------------------------------------------------------------- #
# Path-traversal defence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["../evil.json", "/abs/evil.json", "a/../../evil"])
def test_restore_rejects_escaping_member_path(tmp_path, bad):
    # Hand-craft a hostile dir backup whose manifest points outside the root.
    dest = tmp_path / "hostile"
    (dest / "data").mkdir(parents=True)
    payload = b'{"slug": "evil"}'
    entry = {"path": bad, "sha256": bk._sha256_bytes(payload), "size": len(payload)}
    (dest / bk.BACKUP_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": bk.BACKUP_SCHEMA_VERSION,
                "created_at": "2026-07-01T00:00:00Z",
                "libv2_root": "x",
                "course_count": 0,
                "file_count": 1,
                "files": [entry],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(bk.BackupError) as exc:
        bk.restore_backup(tmp_path / "target", dest)
    assert exc.value.code == "bad_member_path"


# --------------------------------------------------------------------------- #
# CLI round-trips
# --------------------------------------------------------------------------- #


def test_cli_backup_then_restore_round_trip(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    runner = CliRunner()

    r = runner.invoke(
        libv2_main, ["--repo", str(root), "backup", "--out", str(dest)]
    )
    assert r.exit_code == 0, r.output
    assert "Backed up LibV2 metadata" in r.output
    assert dest.is_file()

    # verify-only against a live-repo restore target.
    target = _make_library(tmp_path / "target")
    v = runner.invoke(
        libv2_main, ["--repo", str(target), "restore", str(dest), "--verify-only"]
    )
    assert v.exit_code == 0, v.output
    assert "verification passed" in v.output.lower()

    # Fresh restore into an empty target.
    empty = tmp_path / "empty"
    (empty / "courses").mkdir(parents=True)
    (empty / "catalog").mkdir(parents=True)
    rr = runner.invoke(
        libv2_main, ["--repo", str(empty), "restore", str(dest)]
    )
    assert rr.exit_code == 0, rr.output
    assert "Restored" in rr.output
    assert (empty / "courses" / "demo-101" / "manifest.json").is_file()


def test_cli_backup_dir_format_and_dry_run(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "snapshot"
    runner = CliRunner()

    r = runner.invoke(
        libv2_main,
        ["--repo", str(root), "backup", "--format", "dir", "--out", str(dest)],
    )
    assert r.exit_code == 0, r.output
    assert (dest / bk.BACKUP_MANIFEST_NAME).is_file()

    target = tmp_path / "restored"
    (target / "courses").mkdir(parents=True)
    (target / "catalog").mkdir(parents=True)
    dr = runner.invoke(
        libv2_main, ["--repo", str(target), "restore", str(dest), "--dry-run"]
    )
    assert dr.exit_code == 0, dr.output
    assert "dry-run" in dr.output.lower()
    assert not (target / "catalog" / "master_catalog.json").exists()


def test_cli_backup_refuses_overwrite_without_force(tmp_path):
    root = _make_library(tmp_path / "libv2")
    dest = tmp_path / "b.tar.gz"
    runner = CliRunner()
    runner.invoke(libv2_main, ["--repo", str(root), "backup", "--out", str(dest)])
    r = runner.invoke(libv2_main, ["--repo", str(root), "backup", "--out", str(dest)])
    assert r.exit_code == 1
    assert "dest_exists" in r.output
    r2 = runner.invoke(
        libv2_main, ["--repo", str(root), "backup", "--out", str(dest), "--force"]
    )
    assert r2.exit_code == 0, r2.output
