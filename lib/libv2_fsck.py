"""
LibV2 Integrity Checker (fsck)

Validates LibV2 storage integrity and repairs issues.
Checks blob hashes, catalog consistency, run manifests, and symlinks.

Phase 0 Hardening - Requirement 6: LibV2 Storage Invariants
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .content_store import ContentStore
from .provenance import hash_file

logger = logging.getLogger(__name__)


@dataclass
class FsckIssue:
    """Single integrity issue."""
    severity: str  # "error", "warning", "info"
    category: str
    path: str
    message: str
    fixable: bool = False
    fix_action: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "category": self.category,
            "path": self.path,
            "message": self.message,
            "fixable": self.fixable,
            "fix_action": self.fix_action
        }


@dataclass
class FsckResult:
    """Result of integrity check."""
    passed: bool
    checked_files: int
    issues: List[FsckIssue] = field(default_factory=list)
    fixed_count: int = 0
    skipped_count: int = 0

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checked_files": self.checked_files,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "fixed_count": self.fixed_count,
            "skipped_count": self.skipped_count,
            "issues": [i.to_dict() for i in self.issues]
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"LibV2 fsck {status}: "
            f"{self.checked_files} files checked, "
            f"{self.error_count} errors, "
            f"{self.warning_count} warnings, "
            f"{self.fixed_count} fixed"
        )


class LibV2Fsck:
    """Integrity checker for LibV2 storage."""

    def __init__(self, libv2_root: Path, state_root: Optional[Path] = None):
        """
        Initialize integrity checker.

        Args:
            libv2_root: Path to LibV2 directory
            state_root: Path to state directory (defaults to sibling of LibV2)
        """
        self.libv2_root = libv2_root
        self.state_root = state_root or libv2_root.parent / "state"
        self.content_store = ContentStore(libv2_root)

    def check_all(self, fix: bool = False) -> FsckResult:
        """
        Run all integrity checks.

        Args:
            fix: Whether to attempt fixes for fixable issues

        Returns:
            FsckResult with all findings
        """
        result = FsckResult(passed=True, checked_files=0)

        logger.info(f"Starting LibV2 integrity check: {self.libv2_root}")

        # Check blob integrity
        self._check_blobs(result, fix)

        # Check catalog integrity
        self._check_catalog(result, fix)

        # Check run manifests
        self._check_runs(result, fix)

        # Check symlinks
        self._check_symlinks(result, fix)

        # Check for orphaned files
        self._check_orphans(result, fix)

        result.passed = result.error_count == 0

        logger.info(result.summary())
        return result

    def _check_blobs(self, result: FsckResult, fix: bool) -> None:
        """Verify all blobs match their content hash."""
        blobs_dir = self.libv2_root / "blobs"
        if not blobs_dir.exists():
            logger.debug("No blobs directory found")
            return

        for algo_dir in blobs_dir.iterdir():
            if not algo_dir.is_dir():
                continue

            algorithm = algo_dir.name

            for shard_dir in algo_dir.iterdir():
                if not shard_dir.is_dir():
                    continue

                for blob_path in shard_dir.iterdir():
                    if not blob_path.is_file():
                        continue

                    result.checked_files += 1
                    expected_hash = blob_path.name

                    try:
                        actual_hash = hash_file(blob_path, algorithm)
                        if actual_hash != expected_hash:
                            result.issues.append(FsckIssue(
                                severity="error",
                                category="blob_corruption",
                                path=str(blob_path),
                                message=f"Hash mismatch: expected {expected_hash[:12]}..., got {actual_hash[:12]}...",
                                fixable=False
                            ))
                    except Exception as e:
                        result.issues.append(FsckIssue(
                            severity="error",
                            category="blob_read_error",
                            path=str(blob_path),
                            message=str(e)
                        ))

    def _check_catalog(self, result: FsckResult, fix: bool) -> None:
        """Verify catalog index integrity."""
        catalog_dir = self.libv2_root / "catalog"
        if not catalog_dir.exists():
            return

        # Check course index
        course_index = catalog_dir / "course_index.json"
        if course_index.exists():
            result.checked_files += 1
            try:
                with open(course_index, encoding='utf-8') as f:
                    index = json.load(f)

                modified = False

                # Verify each course entry
                for course_id, entry in list(index.items()):
                    course_path = entry.get('path')
                    if course_path:
                        path = Path(course_path)
                        if not path.exists():
                            result.issues.append(FsckIssue(
                                severity="warning",
                                category="dangling_reference",
                                path=str(course_index),
                                message=f"Course {course_id} references missing path: {course_path}",
                                fixable=True,
                                fix_action=f"remove_entry:{course_id}"
                            ))

                            if fix:
                                del index[course_id]
                                modified = True
                                result.fixed_count += 1

                # Rewrite if fixed
                if fix and modified:
                    temp_path = course_index.with_suffix('.tmp')
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(index, f, indent=2)
                    temp_path.rename(course_index)
                    logger.info("Fixed course index: removed dangling entries")

            except json.JSONDecodeError as e:
                result.issues.append(FsckIssue(
                    severity="error",
                    category="catalog_corruption",
                    path=str(course_index),
                    message=f"Invalid JSON: {e}"
                ))

    def _check_runs(self, result: FsckResult, fix: bool) -> None:
        """Verify run manifest and artifact integrity."""
        runs_dir = self.state_root / "runs"
        if not runs_dir.exists():
            return

        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                result.issues.append(FsckIssue(
                    severity="warning",
                    category="missing_manifest",
                    path=str(run_dir),
                    message="Run directory missing manifest"
                ))
                continue

            result.checked_files += 1

            try:
                with open(manifest_path, encoding='utf-8') as f:
                    manifest = json.load(f)

                # Verify required fields
                required_fields = ['run_id', 'created_at']
                for field in required_fields:
                    if field not in manifest:
                        result.issues.append(FsckIssue(
                            severity="warning",
                            category="incomplete_manifest",
                            path=str(manifest_path),
                            message=f"Missing required field: {field}"
                        ))

                # Verify input hashes if present
                for input_ref in manifest.get('inputs', []):
                    path = input_ref.get('path')
                    expected_hash = input_ref.get('content_hash')
                    algorithm = input_ref.get('hash_algorithm', 'sha256')

                    if path and expected_hash:
                        input_path = Path(path)
                        if input_path.exists():
                            actual_hash = hash_file(input_path, algorithm)
                            if actual_hash != expected_hash:
                                result.issues.append(FsckIssue(
                                    severity="info",
                                    category="input_modified",
                                    path=str(manifest_path),
                                    message=f"Input {input_path.name} modified since run"
                                ))

                # Check config lockfile
                lockfile_path = run_dir / "config_lockfile.json"
                if lockfile_path.exists():
                    result.checked_files += 1
                    self._verify_lockfile(lockfile_path, run_dir, result)

            except json.JSONDecodeError as e:
                result.issues.append(FsckIssue(
                    severity="error",
                    category="manifest_corruption",
                    path=str(manifest_path),
                    message=f"Invalid JSON: {e}"
                ))

    def _verify_lockfile(
        self,
        lockfile_path: Path,
        run_dir: Path,
        result: FsckResult
    ) -> None:
        """Verify config lockfile integrity."""
        try:
            with open(lockfile_path, encoding='utf-8') as f:
                lockfile = json.load(f)

            snapshot_dir = run_dir / "config_snapshot"

            for filename, expected_hash in lockfile.get('config_hashes', {}).items():
                snapshot_path = snapshot_dir / filename
                if not snapshot_path.exists():
                    result.issues.append(FsckIssue(
                        severity="warning",
                        category="missing_snapshot",
                        path=str(lockfile_path),
                        message=f"Config snapshot missing: {filename}"
                    ))
                else:
                    actual_hash = hash_file(snapshot_path)
                    if actual_hash != expected_hash:
                        result.issues.append(FsckIssue(
                            severity="error",
                            category="snapshot_modified",
                            path=str(snapshot_path),
                            message=f"Config snapshot hash mismatch for {filename}"
                        ))

        except json.JSONDecodeError as e:
            result.issues.append(FsckIssue(
                severity="error",
                category="lockfile_corruption",
                path=str(lockfile_path),
                message=f"Invalid JSON: {e}"
            ))

    def _check_symlinks(self, result: FsckResult, fix: bool) -> None:
        """Check for broken symlinks and symlinks escaping blob directory."""
        blobs_dir = (self.libv2_root / "blobs").resolve()

        for path in self.libv2_root.rglob("*"):
            if path.is_symlink():
                result.checked_files += 1
                target = path.resolve()

                if not target.exists():
                    result.issues.append(FsckIssue(
                        severity="warning",
                        category="broken_symlink",
                        path=str(path),
                        message=f"Symlink target missing: {path.readlink()}",
                        fixable=True,
                        fix_action="remove_symlink"
                    ))

                    if fix:
                        path.unlink()
                        result.fixed_count += 1
                        logger.info(f"Removed broken symlink: {path}")

                elif blobs_dir.exists() and not str(target).startswith(str(blobs_dir)):
                    # Symlink target escapes the blob directory
                    result.issues.append(FsckIssue(
                        severity="error",
                        category="symlink_escape",
                        path=str(path),
                        message=f"Symlink target escapes blob directory: {target}",
                        fixable=True,
                        fix_action="remove_symlink"
                    ))

                    if fix:
                        path.unlink()
                        result.fixed_count += 1
                        logger.warning(f"Removed symlink escaping blob directory: {path}")

    def _check_orphans(self, result: FsckResult, fix: bool) -> None:
        """Check for orphaned files (not referenced by any manifest)."""
        # This is an expensive check, skip for now
        pass

    def quick_check(self) -> bool:
        """
        Quick integrity check (spot check, not comprehensive).

        Returns:
            True if basic checks pass
        """
        # Check LibV2 root exists
        if not self.libv2_root.exists():
            return False

        # Check catalog exists
        if not (self.libv2_root / "catalog").exists():
            return False

        # Spot check a few blobs
        blobs_dir = self.libv2_root / "blobs" / "sha256"
        if blobs_dir.exists():
            count = 0
            for shard in blobs_dir.iterdir():
                if shard.is_dir():
                    for blob in shard.iterdir():
                        if blob.is_file():
                            if not self.content_store.verify(blob.name):
                                return False
                            count += 1
                            if count >= 5:  # Check up to 5 blobs
                                break
                    if count >= 5:
                        break

        return True


def run_fsck(
    libv2_root: Optional[Path] = None,
    fix: bool = False,
    verbose: bool = False
) -> FsckResult:
    """
    Convenience function to run fsck.

    Args:
        libv2_root: Path to LibV2 (defaults to standard location)
        fix: Whether to attempt fixes
        verbose: Whether to log details

    Returns:
        FsckResult
    """
    if libv2_root is None:
        from .paths import LIBV2_PATH
        libv2_root = LIBV2_PATH

    fsck = LibV2Fsck(libv2_root)
    result = fsck.check_all(fix=fix)

    if verbose:
        for issue in result.issues:
            level = logging.ERROR if issue.severity == "error" else logging.WARNING
            logger.log(level, f"[{issue.category}] {issue.path}: {issue.message}")

    return result
