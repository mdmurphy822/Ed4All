"""LibV2 metadata backup / disaster-recovery service (W8.7).

The LibV2 library is a gitignored, multi-writer, no-backup store. This module
adds a snapshot/export + verify/restore path for the library's **metadata
spine** — the derived catalog and every course's small metadata sidecars — so a
corrupted or truncated catalog / manifest can be recovered without re-running
the whole pipeline.

Scope (deliberately narrow — see the RAG restriction in ``LibV2/CLAUDE.md``):

* the whole ``catalog/`` tree (``master_catalog.json`` — the "library
  manifest" — ``course_index.json``, the ``by_division`` / ``by_domain`` /
  ``by_subdomain`` indexes, cross references, statistics), and
* per-course, ONLY the small metadata files in :data:`COURSE_METADATA_FILES`
  (``manifest.json`` + ``course.json``). The multi-MB ``chunks.jsonl`` bodies,
  vector indexes, and adapters are **never** walked or read — a metadata
  snapshot is cheap, safe, and the thing an operator actually needs to rebuild
  navigation after a catalog corruption.

Contracts:

* **Backup is read-only over the live store** — it only reads the allowlisted
  files and writes to an out-of-tree destination. It never touches ``courses/``
  content.
* **Restore verifies first** — every member's bytes are re-hashed against the
  backup manifest before anything is written; a checksum mismatch fails closed.
* **Idempotent + resumable** — restore skips any target file already present
  with the recorded checksum, so re-running after a partial restore only writes
  what's missing, and a divergent live file is never clobbered without
  ``overwrite``.

One implementation backs both the ``libv2 backup`` and ``libv2 restore`` CLI
commands (``LibV2/tools/libv2/cli.py``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Snapshot format version — bumped if the manifest shape changes.
BACKUP_SCHEMA_VERSION = "1.0"

# Name of the manifest member at the archive root (both tar + dir formats).
BACKUP_MANIFEST_NAME = "backup_manifest.json"

# Sub-directory (inside the archive) that mirrors the libv2-root-relative paths
# of the snapshotted files.
BACKUP_DATA_PREFIX = "data"

# Per-course metadata files a snapshot captures. NEVER includes chunks.jsonl /
# vector_index / models — a metadata-only DR snapshot (see module docstring).
COURSE_METADATA_FILES: Tuple[str, ...] = ("manifest.json", "course.json")

_CHUNK_SIZE = 1024 * 1024


class BackupError(Exception):
    """Typed backup/restore failure carrying a ``(status, code, detail)`` triple.

    ``status`` mirrors the HTTP-style status a future endpoint would map it to
    (422 bad input / corrupt archive, 404 missing root/member); the CLI renders
    ``detail`` and exits non-zero. Mirrors ``remove.CourseRemovalError`` so both
    destructive/recovery surfaces share one error shape.
    """

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class BackupFileEntry:
    """One snapshotted file: its libv2-root-relative POSIX path + checksum."""

    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: dict) -> "BackupFileEntry":
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size=int(payload.get("size", 0)),
        )


@dataclass
class BackupManifest:
    """The snapshot's self-describing manifest (written as backup_manifest.json)."""

    version: str
    created_at: str
    libv2_root: str
    course_count: int
    files: List[BackupFileEntry] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "libv2_root": self.libv2_root,
            "course_count": self.course_count,
            "file_count": self.file_count,
            "files": [e.to_dict() for e in self.files],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BackupManifest":
        if not isinstance(payload, dict) or "files" not in payload:
            raise BackupError(
                422, "bad_manifest", "backup manifest is malformed (no 'files')"
            )
        try:
            files = [BackupFileEntry.from_dict(e) for e in payload["files"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise BackupError(
                422, "bad_manifest", f"backup manifest has a malformed entry: {exc}"
            ) from exc
        return cls(
            version=str(payload.get("version", "")),
            created_at=str(payload.get("created_at", "")),
            libv2_root=str(payload.get("libv2_root", "")),
            course_count=int(payload.get("course_count", 0)),
            files=files,
        )


@dataclass
class BackupResult:
    """What :func:`create_backup` wrote."""

    dest: Path
    fmt: str
    manifest: BackupManifest

    def to_dict(self) -> dict:
        return {
            "dest": str(self.dest),
            "format": self.fmt,
            "file_count": self.manifest.file_count,
            "course_count": self.manifest.course_count,
        }


@dataclass
class VerifyResult:
    """Per-member checksum verdicts for :func:`verify_backup`."""

    ok: List[str] = field(default_factory=list)
    mismatched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatched and not self.missing

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "ok": self.ok,
            "mismatched": self.mismatched,
            "missing": self.missing,
        }


@dataclass
class RestoreResult:
    """What :func:`restore_backup` did (or, with ``dry_run``, would do)."""

    restored: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    conflicted: List[str] = field(default_factory=list)
    planned: List[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "restored": self.restored,
            "skipped": self.skipped,
            "conflicted": self.conflicted,
            "planned": self.planned,
        }


# --------------------------------------------------------------------------- #
# Checksum helpers
# --------------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Snapshot file selection (read-only over the live store)
# --------------------------------------------------------------------------- #


def iter_snapshot_files(libv2_root: Path) -> List[str]:
    """Return the sorted libv2-root-relative POSIX paths a snapshot captures.

    Deterministic (sorted) so re-running a backup over an unchanged store
    selects the same file set. Walks ONLY ``catalog/`` (recursively) and each
    course dir's :data:`COURSE_METADATA_FILES` — never the chunk bodies.
    """
    libv2_root = Path(libv2_root)
    rels: List[str] = []

    catalog_dir = libv2_root / "catalog"
    if catalog_dir.is_dir():
        for p in sorted(catalog_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rels.append(p.relative_to(libv2_root).as_posix())

    courses_dir = libv2_root / "courses"
    if courses_dir.is_dir():
        for cdir in sorted(courses_dir.iterdir()):
            if not cdir.is_dir() or cdir.is_symlink():
                continue
            for name in COURSE_METADATA_FILES:
                f = cdir / name
                if f.is_file() and not f.is_symlink():
                    rels.append(f.relative_to(libv2_root).as_posix())

    return sorted(rels)


def _count_courses(libv2_root: Path) -> int:
    """Count course dirs carrying at least one snapshotted metadata file."""
    courses_dir = Path(libv2_root) / "courses"
    if not courses_dir.is_dir():
        return 0
    n = 0
    for cdir in courses_dir.iterdir():
        if not cdir.is_dir() or cdir.is_symlink():
            continue
        if any((cdir / name).is_file() for name in COURSE_METADATA_FILES):
            n += 1
    return n


def build_manifest(libv2_root: Path, rels: List[str]) -> BackupManifest:
    """Hash each selected file and assemble the snapshot manifest."""
    libv2_root = Path(libv2_root)
    entries: List[BackupFileEntry] = []
    for rel in rels:
        src = libv2_root / rel
        try:
            size = src.stat().st_size
            sha = _sha256_file(src)
        except OSError as exc:  # pragma: no cover — race: file vanished mid-backup
            raise BackupError(
                404, "source_read_failed", f"could not read {rel}: {exc}"
            ) from exc
        entries.append(BackupFileEntry(path=rel, sha256=sha, size=size))
    return BackupManifest(
        version=BACKUP_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        libv2_root=str(libv2_root),
        course_count=_count_courses(libv2_root),
        files=entries,
    )


# --------------------------------------------------------------------------- #
# Backup (write)
# --------------------------------------------------------------------------- #


def create_backup(
    libv2_root: Path,
    dest: Path,
    *,
    fmt: str = "tar",
    force: bool = False,
) -> BackupResult:
    """Snapshot the LibV2 metadata spine to ``dest``.

    ``fmt`` is ``"tar"`` (a gzip tarball at ``dest``) or ``"dir"`` (a directory
    at ``dest`` holding ``backup_manifest.json`` + a ``data/`` mirror). Read-only
    over the live store. Refuses to overwrite an existing ``dest`` unless
    ``force``.

    Raises :class:`BackupError` on a missing root, a bad ``fmt``, or an
    unforced overwrite.
    """
    libv2_root = Path(libv2_root)
    dest = Path(dest)
    if fmt not in ("tar", "dir"):
        raise BackupError(422, "bad_format", f"unknown backup format: {fmt!r}")
    if not libv2_root.is_dir():
        raise BackupError(404, "root_not_found", f"LibV2 root not found: {libv2_root}")

    if dest.exists() and not force:
        raise BackupError(
            422,
            "dest_exists",
            f"backup destination already exists (use --force): {dest}",
        )

    rels = iter_snapshot_files(libv2_root)
    manifest = build_manifest(libv2_root, rels)
    manifest_bytes = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")

    if fmt == "dir":
        _write_dir_backup(libv2_root, dest, manifest, manifest_bytes, force=force)
    else:
        _write_tar_backup(libv2_root, dest, manifest, manifest_bytes, force=force)

    return BackupResult(dest=dest, fmt=fmt, manifest=manifest)


def _write_dir_backup(
    libv2_root: Path,
    dest: Path,
    manifest: BackupManifest,
    manifest_bytes: bytes,
    *,
    force: bool,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    data_root = dest / BACKUP_DATA_PREFIX
    for entry in manifest.files:
        src = libv2_root / entry.path
        target = data_root / entry.path
        # Resumable: a data file already present with the recorded checksum is
        # left untouched (re-running a `dir` backup only writes what changed).
        if target.is_file() and _sha256_file(target) == entry.sha256:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(target, src.read_bytes())
    _atomic_write_bytes(dest / BACKUP_MANIFEST_NAME, manifest_bytes)


def _write_tar_backup(
    libv2_root: Path,
    dest: Path,
    manifest: BackupManifest,
    manifest_bytes: bytes,
    *,
    force: bool,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp-backup")
    if tmp.exists():
        tmp.unlink()
    try:
        with tarfile.open(tmp, "w:gz") as tf:
            # Manifest first so a reader can stream it without scanning.
            _add_bytes_to_tar(tf, BACKUP_MANIFEST_NAME, manifest_bytes)
            for entry in manifest.files:
                src = libv2_root / entry.path
                arcname = f"{BACKUP_DATA_PREFIX}/{entry.path}"
                tf.add(str(src), arcname=arcname, recursive=False)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass


def _add_bytes_to_tar(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = 0
    tf.addfile(info, io.BytesIO(data))


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` via a temp file + ``os.replace``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp-write")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


# --------------------------------------------------------------------------- #
# Archive read access (tar or dir), used by verify + restore
# --------------------------------------------------------------------------- #


def _assert_safe_relpath(rel: str) -> None:
    """Reject a manifest entry path that is absolute or escapes via ``..``.

    Defence against a tampered/hostile backup manifest steering a restore write
    outside the target root.
    """
    if not rel or rel != rel.strip():
        raise BackupError(422, "bad_member_path", f"invalid member path: {rel!r}")
    pure = Path(rel)
    if pure.is_absolute() or rel.startswith(("/", "\\")):
        raise BackupError(422, "bad_member_path", f"absolute member path: {rel!r}")
    parts = pure.parts
    if ".." in parts:
        raise BackupError(
            422, "bad_member_path", f"member path escapes via '..': {rel!r}"
        )


class BackupArchive:
    """Read access to a backup, whether a tarball or an extracted directory.

    Used as a context manager (the tar handle is opened lazily on ``__enter__``
    and closed on exit).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.kind: str
        self._tar: Optional[tarfile.TarFile] = None
        if self.path.is_dir():
            self.kind = "dir"
        elif self.path.is_file() and tarfile.is_tarfile(self.path):
            self.kind = "tar"
        else:
            raise BackupError(
                422,
                "bad_backup",
                f"not a LibV2 backup (expected a directory or tarball): {self.path}",
            )

    def __enter__(self) -> "BackupArchive":
        if self.kind == "tar":
            self._tar = tarfile.open(self.path, "r:*")
        return self

    def __exit__(self, *exc) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def _read_member(self, arcname: str) -> bytes:
        if self.kind == "dir":
            p = self.path / arcname
            if not p.is_file():
                raise BackupError(404, "missing_member", f"missing backup member: {arcname}")
            return p.read_bytes()
        assert self._tar is not None, "BackupArchive must be used as a context manager"
        try:
            extracted = self._tar.extractfile(arcname)
        except KeyError:
            extracted = None
        if extracted is None:
            raise BackupError(404, "missing_member", f"missing backup member: {arcname}")
        with extracted:
            return extracted.read()

    def read_manifest(self) -> BackupManifest:
        try:
            raw = self._read_member(BACKUP_MANIFEST_NAME)
        except BackupError:
            raise
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackupError(422, "bad_manifest", f"unparseable manifest: {exc}") from exc
        return BackupManifest.from_dict(payload)

    def read_data(self, rel: str) -> bytes:
        return self._read_member(f"{BACKUP_DATA_PREFIX}/{rel}")


# --------------------------------------------------------------------------- #
# Verify + restore (read)
# --------------------------------------------------------------------------- #


def verify_backup(backup_path: Path) -> VerifyResult:
    """Re-hash every member against the manifest. Detects corruption/truncation.

    A member that fails :func:`_assert_safe_relpath` is reported as
    ``mismatched`` (a hostile path is a failed verification, not a crash).
    """
    result = VerifyResult()
    with BackupArchive(backup_path) as arc:
        manifest = arc.read_manifest()
        for entry in manifest.files:
            try:
                _assert_safe_relpath(entry.path)
                data = arc.read_data(entry.path)
            except BackupError as exc:
                if exc.code == "missing_member":
                    result.missing.append(entry.path)
                else:
                    result.mismatched.append(entry.path)
                continue
            if _sha256_bytes(data) == entry.sha256:
                result.ok.append(entry.path)
            else:
                result.mismatched.append(entry.path)
    return result


def restore_backup(
    libv2_root: Path,
    backup_path: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> RestoreResult:
    """Verify then restore the backup's metadata files into ``libv2_root``.

    * **Verify-first**: each member's bytes are re-hashed against the manifest
      before any write; a mismatch/missing member fails closed (nothing is
      written).
    * **Idempotent + resumable**: a target already present with the recorded
      checksum is skipped, so re-running after a partial restore writes only the
      remainder.
    * **Non-clobbering**: a target present but DIVERGENT is reported as
      ``conflicted`` and left as-is unless ``overwrite`` is set.
    * **Containment-checked**: each member path is validated to stay under the
      target root (no ``..`` / absolute escape).
    * ``dry_run`` reports what would be written without touching disk.

    Raises :class:`BackupError` on a corrupt/incomplete archive.
    """
    libv2_root = Path(libv2_root)
    result = RestoreResult(dry_run=dry_run)

    with BackupArchive(backup_path) as arc:
        manifest = arc.read_manifest()

        # Pass 1 — verify every member up front (fail closed before any write).
        loaded: List[Tuple[str, bytes]] = []
        for entry in manifest.files:
            _assert_safe_relpath(entry.path)
            data = arc.read_data(entry.path)
            if _sha256_bytes(data) != entry.sha256:
                raise BackupError(
                    422,
                    "corrupt_member",
                    f"checksum mismatch for {entry.path} — refusing to restore a "
                    "corrupt backup",
                )
            loaded.append((entry.path, data))

        # Pass 2 — write (idempotent / resumable / non-clobbering).
        for rel, data in loaded:
            target = libv2_root / rel
            if target.is_file():
                if _sha256_file(target) == _sha256_bytes(data):
                    result.skipped.append(rel)
                    continue
                if not overwrite:
                    result.conflicted.append(rel)
                    continue
            if dry_run:
                result.planned.append(rel)
                continue
            _atomic_write_bytes(target, data)
            result.restored.append(rel)

    return result


__all__ = [
    "BACKUP_SCHEMA_VERSION",
    "BACKUP_MANIFEST_NAME",
    "COURSE_METADATA_FILES",
    "BackupError",
    "BackupFileEntry",
    "BackupManifest",
    "BackupResult",
    "VerifyResult",
    "RestoreResult",
    "iter_snapshot_files",
    "build_manifest",
    "create_backup",
    "verify_backup",
    "restore_backup",
    "BackupArchive",
]
