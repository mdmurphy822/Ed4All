"""``ed4all backup`` — full data-dir backup + restore verification (OP3).

Unlike ``support-bundle`` (a REDACTED diagnostic slice), a backup is a
COMPLETE, restore-able snapshot of every mutable data directory. It is meant to
restore a working system, so:

* It captures the full ``_DATA_DIR_KEYS`` set resolved through ``lib.paths``
  (``state`` / ``libv2`` / ``exports`` / ``training-captures`` / ``semantik-output``),
  honoring ``ED4ALL_HOME`` and every per-dir override (``ED4ALL_LIBV2_ROOT`` …).
  When ``ED4ALL_STATE_RUNS_DIR`` scatters the runs subtree OUTSIDE the state
  root it is captured too (as ``state-runs/``).
* ``secrets.json`` IS included (a restored system needs its credentials to be
  functional). The archive is therefore sensitive — it is written ``0600`` and
  the command says so.
* Docker-only external stores (HF model cache, ollama models) are OUT of scope —
  see docs/operations/backup-restore.md.

``--verify ARCHIVE`` extracts the archive to a temp dir and (1) recomputes every
member's sha256 against the embedded ``manifest.json`` (catching a corrupted
member) and (2) runs the LibV2 fsck over the extracted ``libv2/`` tree.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import click

from lib.paths import (
    PROJECT_ROOT,
    courseforge_exports_dir,
    ed4all_home,
    get_state_runs_dir,
    get_training_captures_dir,
    is_path_within,
    libv2_path,
    semantik_output_dir,
)

#: Name of the embedded manifest (recomputed against on ``--verify``).
_MANIFEST_NAME = "manifest.json"


def resolve_backup_dirs() -> Dict[str, Path]:
    """Return the ``key -> resolved path`` map of data dirs to back up.

    Resolves EVERY dir through ``lib.paths`` helpers (never hardcodes a
    repo-relative path) so ``ED4ALL_HOME`` and the per-dir overrides
    (``ED4ALL_LIBV2_ROOT`` / ``ED4ALL_TRAINING_CAPTURES_DIR`` / …) are honored.
    The ``state`` root is resolved home-aware at call time; when
    ``ED4ALL_STATE_RUNS_DIR`` relocates the runs subtree outside that root it is
    captured separately as ``state-runs`` so a scattered layout still round-trips.
    """
    home = ed4all_home()
    state_root = (home / "state") if home is not None else (PROJECT_ROOT / "runtime" / "state")

    dirs: Dict[str, Path] = {
        "state": state_root,
        "libv2": libv2_path(),
        "exports": courseforge_exports_dir(),
        "training-captures": get_training_captures_dir(),
        "semantik-output": semantik_output_dir(),
    }

    # Per-dir override may scatter the runs subtree out of the state root.
    runs_dir = get_state_runs_dir()
    if not is_path_within(runs_dir, state_root):
        dirs["state-runs"] = runs_dir

    return dirs


@dataclass
class BackupResult:
    """Outcome of :func:`create_backup`."""

    output: Path
    size_bytes: int
    member_count: int
    included_dirs: Dict[str, Path]
    missing_dirs: List[str] = field(default_factory=list)


def _iter_dir_files(root: Path):
    """Yield ``(abs_path, relative_posix)`` for every file under ``root``."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path, path.relative_to(root).as_posix()


def create_backup(
    output: Path,
    *,
    dirs: Optional[Dict[str, Path]] = None,
    mtime: Optional[float] = None,
) -> BackupResult:
    """Tar every resolved data dir into ``output`` (gzip) and ``0600`` it.

    ``dirs`` defaults to :func:`resolve_backup_dirs`. Each dir's files land
    under ``<key>/<relative-path>`` inside the archive. An embedded
    ``manifest.json`` records every member's size + sha256 so ``--verify`` can
    catch a corrupted member. The archive contains ``secrets.json`` (a backup
    must restore a working system), so it is chmod'd to owner-only ``0600``.
    """
    if mtime is None:
        mtime = time.time()
    if dirs is None:
        dirs = resolve_backup_dirs()

    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_files: List[dict] = []
    missing: List[str] = []
    member_count = 0

    with tarfile.open(output, "w:gz") as tar:
        for key, root in dirs.items():
            if not root.exists():
                missing.append(key)
                continue
            for abs_path, rel in _iter_dir_files(root):
                try:
                    data = abs_path.read_bytes()
                except OSError:
                    continue
                arcname = f"{key}/{rel}"
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                info.mtime = int(mtime)
                tar.addfile(info, io.BytesIO(data))
                manifest_files.append(
                    {
                        "arcname": arcname,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                member_count += 1

        manifest = {
            "kind": "ed4all-backup",
            "generated_at": mtime,
            "dirs": {k: str(v) for k, v in dirs.items()},
            "missing_dirs": missing,
            "files": manifest_files,
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(name=_MANIFEST_NAME)
        info.size = len(manifest_bytes)
        info.mtime = int(mtime)
        tar.addfile(info, io.BytesIO(manifest_bytes))

    # A backup carries secrets.json — restrict to owner read/write only.
    try:
        os.chmod(output, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass

    return BackupResult(
        output=output,
        size_bytes=output.stat().st_size,
        member_count=member_count,
        included_dirs={k: v for k, v in dirs.items() if k not in missing},
        missing_dirs=missing,
    )


@dataclass
class VerifyResult:
    """Outcome of a backup verification."""

    ok: bool = True
    checked: int = 0
    issues: List[str] = field(default_factory=list)
    member_names: List[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.issues.append(message)


def _verify_extracted(extract_dir: Path) -> VerifyResult:
    """Verify an already-extracted backup tree against its manifest + fsck.

    (1) Recompute every manifest member's sha256 and compare — a mismatch or a
    missing member fails verification (this is what catches a corrupted
    member). (2) If a ``libv2/`` tree is present, run the LibV2 fsck and fold in
    any errors. Split out from :func:`verify_archive` so tests can corrupt a
    file in an extracted tree and assert detection without rebuilding a tarball.
    """
    result = VerifyResult()
    manifest_path = extract_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        result.fail(f"manifest.json missing from archive at {extract_dir}")
        return result

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.fail(f"manifest.json unreadable: {exc}")
        return result

    for entry in manifest.get("files", []):
        arcname = entry.get("arcname")
        expected = entry.get("sha256")
        result.member_names.append(arcname)
        member = extract_dir / arcname
        if not member.exists():
            result.fail(f"missing member: {arcname}")
            continue
        actual = hashlib.sha256(member.read_bytes()).hexdigest()
        result.checked += 1
        if actual != expected:
            result.fail(
                f"corrupted member: {arcname} (sha256 {actual} != {expected})"
            )

    # LibV2 fsck over the extracted libv2 tree (best-effort).
    libv2_dir = extract_dir / "libv2"
    if libv2_dir.is_dir():
        try:
            from lib.libv2_fsck import LibV2Fsck  # noqa: PLC0415

            fsck = LibV2Fsck(libv2_dir).check_all(fix=False)
            for issue in fsck.issues:
                if issue.severity == "error":
                    result.fail(f"libv2 fsck: {issue.category}: {issue.message}")
        except Exception as exc:  # noqa: BLE001 — fsck is advisory here
            result.issues.append(f"libv2 fsck skipped: {exc}")

    return result


def verify_archive(archive: Path) -> VerifyResult:
    """Extract ``archive`` to a temp dir and verify it (manifest + fsck).

    A tar/gzip stream that is itself corrupt (unreadable) fails verification
    loudly rather than raising.
    """
    result = VerifyResult()
    if not archive.exists():
        result.fail(f"archive not found: {archive}")
        return result

    with tempfile.TemporaryDirectory(prefix="ed4all-verify-") as tmp:
        extract_dir = Path(tmp)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                # Read every member's data (surfaces a corrupted gzip stream)
                # and extract to the temp tree.
                _safe_extractall(tar, extract_dir)
        except (tarfile.TarError, OSError, EOFError) as exc:
            result.fail(f"archive unreadable (corrupt tar/gzip): {exc}")
            return result
        return _verify_extracted(extract_dir)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` under ``dest``, refusing any member that escapes it."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not (target == dest or str(target).startswith(str(dest) + os.sep)):
            raise tarfile.TarError(f"unsafe member path: {member.name}")
    tar.extractall(dest)  # noqa: S202 — paths validated above


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < step or unit == "GB":
            return f"{n} B" if unit == "B" else f"{val:.1f} {unit}"
        val /= step
    return f"{n} B"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command("backup")
@click.option(
    "--output",
    "-o",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a full data-dir backup .tar.gz here (0600). Required for create mode.",
)
@click.option(
    "--verify",
    "verify",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=None,
    help="Verify an existing backup archive (manifest sha256 + libv2 fsck) instead of creating one.",
)
def backup_command(output: Optional[Path], verify: Optional[Path]) -> None:
    """Create a full data-dir backup, or verify an existing one.

    Create mode (``--output``): tars every resolved data dir (honoring
    ``ED4ALL_HOME`` + per-dir overrides), INCLUDING ``secrets.json``, and
    ``0600``\\ s the archive. Verify mode (``--verify``): recomputes each
    member's sha256 against the embedded manifest and runs the LibV2 fsck.
    """
    if verify is not None and output is not None:
        raise click.UsageError("--output and --verify are mutually exclusive.")

    if verify is not None:
        result = verify_archive(verify)
        click.echo(f"ed4all backup --verify {verify}")
        click.echo(f"  Members checked: {result.checked}")
        if result.ok and not result.issues:
            click.secho("  OK — archive integrity verified.", fg="green")
        else:
            for issue in result.issues:
                color = "red" if not result.ok else "yellow"
                click.secho(f"  - {issue}", fg=color)
            if not result.ok:
                click.secho("  FAILED — backup is not restore-safe.", fg="red")
        raise SystemExit(0 if result.ok else 1)

    if output is None:
        raise click.UsageError("provide --output PATH (create) or --verify ARCHIVE.")

    result = create_backup(output)
    click.echo(f"Backup: {result.output}")
    click.echo(f"  Size:    {_format_bytes(result.size_bytes)}")
    click.echo(f"  Members: {result.member_count}")
    click.echo("  Included dirs:")
    for key, path in result.included_dirs.items():
        click.echo(f"    {key}: {path}")
    if result.missing_dirs:
        click.echo(f"  Missing (not present, skipped): {', '.join(result.missing_dirs)}")
    click.secho(
        "  Mode 0600 — this archive CONTAINS secrets.json. Store it securely; "
        "do NOT commit or share it.",
        fg="yellow",
    )


def register_backup_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all backup`` command to the top-level CLI group."""
    cli_group.add_command(backup_command)
