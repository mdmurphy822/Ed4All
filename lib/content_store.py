"""
Content-Addressed Blob Storage

Stores artifacts by content hash for deduplication and integrity.
Supports sharded storage for large repositories.

Phase 0 Hardening - Requirement 6: LibV2 Storage Invariants
Phase 0.5 Enhancement: Artifact Deduplication (E2)
"""

import hashlib
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StoredBlob:
    """Reference to a stored blob."""
    content_hash: str
    hash_algorithm: str
    size_bytes: int
    path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_hash": self.content_hash,
            "hash_algorithm": self.hash_algorithm,
            "size_bytes": self.size_bytes,
            "path": str(self.path)
        }


@dataclass
class DedupeArtifact:
    """Record of a deduplicated artifact."""
    original_path: str
    content_hash: str
    size_bytes: int
    replaced_with_symlink: bool
    blob_path: str


@dataclass
class DedupeResult:
    """
    Result of artifact deduplication operation.

    Phase 0.5: Artifact Deduplication (E2)
    """
    success: bool
    artifacts_scanned: int
    artifacts_deduplicated: int
    unique_blobs: int
    space_saved_bytes: int
    already_symlinks: int
    errors: List[str] = field(default_factory=list)
    deduplicated: List[DedupeArtifact] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "artifacts_scanned": self.artifacts_scanned,
            "artifacts_deduplicated": self.artifacts_deduplicated,
            "unique_blobs": self.unique_blobs,
            "space_saved_bytes": self.space_saved_bytes,
            "space_saved_mb": round(self.space_saved_bytes / (1024 * 1024), 2),
            "already_symlinks": self.already_symlinks,
            "errors": self.errors,
            "deduplicated": [asdict(d) for d in self.deduplicated],
            "duration_seconds": self.duration_seconds,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.success:
            return f"Deduplication failed: {self.errors[0] if self.errors else 'unknown error'}"
        saved_mb = round(self.space_saved_bytes / (1024 * 1024), 2)
        return (
            f"Deduplicated {self.artifacts_deduplicated}/{self.artifacts_scanned} artifacts "
            f"into {self.unique_blobs} unique blobs, saved {saved_mb} MB"
        )


class ContentStore:
    """Content-addressed blob storage with deduplication."""

    SUPPORTED_ALGORITHMS = ("sha256", "sha512", "blake3")

    def __init__(self, root_path: Path, algorithm: str = "sha256"):
        """
        Initialize content store.

        Args:
            root_path: Root directory for blob storage
            algorithm: Hash algorithm (sha256, sha512, blake3)
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.root = root_path
        self.algorithm = algorithm
        self.blobs_dir = root_path / "blobs" / algorithm
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_blob_path(self, content_hash: str) -> Path:
        """
        Get path for blob by hash (sharded by first 2 chars).

        This sharding prevents directory size issues with many blobs.
        """
        return self.blobs_dir / content_hash[:2] / content_hash

    def _compute_hash(self, content: bytes) -> str:
        """Compute hash of content."""
        if self.algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif self.algorithm == "sha512":
            return hashlib.sha512(content).hexdigest()
        elif self.algorithm == "blake3":
            try:
                import blake3
                return blake3.blake3(content).hexdigest()
            except ImportError:
                raise RuntimeError("blake3 library not installed") from None
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _compute_hash_streaming(self, file_path: Path) -> str:
        """Compute hash of file using streaming (memory efficient)."""
        if self.algorithm == "sha256":
            hasher = hashlib.sha256()
        elif self.algorithm == "sha512":
            hasher = hashlib.sha512()
        elif self.algorithm == "blake3":
            try:
                import blake3
                hasher = blake3.blake3()
            except ImportError:
                raise RuntimeError("blake3 library not installed") from None
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)

        return hasher.hexdigest()

    def store(self, content: bytes) -> StoredBlob:
        """
        Store content and return blob reference.

        Args:
            content: Bytes to store

        Returns:
            StoredBlob with hash and location
        """
        content_hash = self._compute_hash(content)
        blob_path = self._get_blob_path(content_hash)

        # Check if already exists (deduplication)
        if blob_path.exists():
            logger.debug(f"Blob already exists: {content_hash[:12]}...")
            return StoredBlob(
                content_hash=content_hash,
                hash_algorithm=self.algorithm,
                size_bytes=len(content),
                path=blob_path
            )

        # Store atomically
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = blob_path.with_suffix('.tmp')

        with open(temp_path, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        temp_path.rename(blob_path)

        logger.debug(f"Stored blob: {content_hash[:12]}... ({len(content)} bytes)")

        return StoredBlob(
            content_hash=content_hash,
            hash_algorithm=self.algorithm,
            size_bytes=len(content),
            path=blob_path
        )

    def store_file(self, file_path: Path) -> StoredBlob:
        """
        Store file and return blob reference.

        Uses streaming hash for memory efficiency.

        Args:
            file_path: Path to file to store

        Returns:
            StoredBlob with hash and location
        """
        content_hash = self._compute_hash_streaming(file_path)
        blob_path = self._get_blob_path(content_hash)
        size_bytes = file_path.stat().st_size

        # Check if already exists
        if blob_path.exists():
            logger.debug(f"Blob already exists: {content_hash[:12]}...")
            return StoredBlob(
                content_hash=content_hash,
                hash_algorithm=self.algorithm,
                size_bytes=size_bytes,
                path=blob_path
            )

        # Copy atomically
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = blob_path.with_suffix('.tmp')

        shutil.copy2(file_path, temp_path)
        temp_path.rename(blob_path)

        logger.debug(f"Stored file as blob: {content_hash[:12]}... ({size_bytes} bytes)")

        return StoredBlob(
            content_hash=content_hash,
            hash_algorithm=self.algorithm,
            size_bytes=size_bytes,
            path=blob_path
        )

    def has_content(self, content: bytes) -> bool:
        """
        Check if content already exists in the store without storing it.

        Use this for pre-generation deduplication: before processing a
        document through the pipeline, check if identical content already
        exists in the store.

        Args:
            content: Bytes to check

        Returns:
            True if content already exists in the store
        """
        content_hash = self._compute_hash(content)
        return self._get_blob_path(content_hash).exists()

    def has_file(self, file_path: Path) -> bool:
        """
        Check if a file's content already exists in the store.

        Uses streaming hash for memory efficiency.

        Args:
            file_path: Path to file to check

        Returns:
            True if identical content already exists
        """
        content_hash = self._compute_hash_streaming(file_path)
        return self._get_blob_path(content_hash).exists()

    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """
        Retrieve content by hash.

        Args:
            content_hash: Hash of content to retrieve

        Returns:
            Content bytes or None if not found
        """
        blob_path = self._get_blob_path(content_hash)
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    def retrieve_path(self, content_hash: str) -> Optional[Path]:
        """
        Get path to blob by hash.

        Args:
            content_hash: Hash of content

        Returns:
            Path to blob or None if not found
        """
        blob_path = self._get_blob_path(content_hash)
        return blob_path if blob_path.exists() else None

    def verify(self, content_hash: str) -> bool:
        """
        Verify blob integrity.

        Args:
            content_hash: Hash to verify

        Returns:
            True if blob exists and hash matches
        """
        blob_path = self._get_blob_path(content_hash)
        if not blob_path.exists():
            return False

        actual_hash = self._compute_hash_streaming(blob_path)
        return actual_hash == content_hash

    def exists(self, content_hash: str) -> bool:
        """Check if blob exists."""
        return self._get_blob_path(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """
        Delete a blob.

        Args:
            content_hash: Hash of blob to delete

        Returns:
            True if deleted, False if not found
        """
        blob_path = self._get_blob_path(content_hash)
        if not blob_path.exists():
            return False

        blob_path.unlink()
        logger.debug(f"Deleted blob: {content_hash[:12]}...")

        # Clean up empty shard directory
        shard_dir = blob_path.parent
        if shard_dir.exists() and not any(shard_dir.iterdir()):
            shard_dir.rmdir()

        return True

    def link_to_run(
        self,
        content_hash: str,
        run_path: Path,
        artifact_name: str
    ) -> Path:
        """
        Create symlink from run artifacts to blob.

        Args:
            content_hash: Hash of blob
            run_path: Path to run directory
            artifact_name: Name for the symlink

        Returns:
            Path to created symlink

        Raises:
            FileNotFoundError: If blob doesn't exist
        """
        blob_path = self._get_blob_path(content_hash)
        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found: {content_hash}")

        link_path = run_path / "artifacts" / artifact_name
        link_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing link if present
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        link_path.symlink_to(blob_path.resolve())
        logger.debug(f"Linked blob {content_hash[:12]}... to {artifact_name}")

        return link_path

    def iter_blobs(self) -> Iterator[StoredBlob]:
        """
        Iterate over all blobs in store.

        Yields:
            StoredBlob for each blob
        """
        for shard_dir in self.blobs_dir.iterdir():
            if not shard_dir.is_dir():
                continue

            for blob_path in shard_dir.iterdir():
                if blob_path.is_file():
                    yield StoredBlob(
                        content_hash=blob_path.name,
                        hash_algorithm=self.algorithm,
                        size_bytes=blob_path.stat().st_size,
                        path=blob_path
                    )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with blob count, total size, etc.
        """
        blob_count = 0
        total_size = 0
        shard_count = 0

        for shard_dir in self.blobs_dir.iterdir():
            if shard_dir.is_dir():
                shard_count += 1
                for blob_path in shard_dir.iterdir():
                    if blob_path.is_file():
                        blob_count += 1
                        total_size += blob_path.stat().st_size

        return {
            "algorithm": self.algorithm,
            "blob_count": blob_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "shard_count": shard_count,
            "root_path": str(self.root)
        }

    def verify_all(self) -> List[Dict[str, Any]]:
        """
        Verify integrity of all blobs.

        Returns:
            List of corruption issues found
        """
        issues = []

        for blob in self.iter_blobs():
            if not self.verify(blob.content_hash):
                issues.append({
                    "type": "hash_mismatch",
                    "content_hash": blob.content_hash,
                    "path": str(blob.path)
                })

        return issues

    # ========================================================================
    # Phase 0.5: Artifact Deduplication
    # ========================================================================

    def deduplicate_artifacts(
        self,
        run_path: Path,
        dry_run: bool = False,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> DedupeResult:
        """
        Deduplicate artifacts in a run by replacing duplicates with symlinks.

        Phase 0.5: Artifact Deduplication (E2)

        Scans all artifacts in a run, stores unique content in the blob store,
        and replaces duplicate files with symlinks to the blobs.

        Args:
            run_path: Path to run directory
            dry_run: If True, calculate savings without modifying files
            include_patterns: Only process files matching these patterns
            exclude_patterns: Skip files matching these patterns

        Returns:
            DedupeResult with deduplication statistics
        """
        import time
        start_time = time.time()

        result = DedupeResult(
            success=False,
            artifacts_scanned=0,
            artifacts_deduplicated=0,
            unique_blobs=0,
            space_saved_bytes=0,
            already_symlinks=0,
        )

        artifacts_path = run_path / "artifacts"
        if not artifacts_path.exists():
            result.errors.append(f"Artifacts directory not found: {artifacts_path}")
            result.duration_seconds = time.time() - start_time
            return result

        # Track content hashes to blobs
        hash_to_blob: Dict[str, StoredBlob] = {}
        hash_counts: Dict[str, int] = {}

        try:
            # First pass: scan all artifacts
            for artifact_path in artifacts_path.rglob("*"):
                if not artifact_path.is_file():
                    continue

                # Check patterns
                rel_path = str(artifact_path.relative_to(artifacts_path))

                if include_patterns:
                    if not any(self._match_pattern(rel_path, p) for p in include_patterns):
                        continue

                if exclude_patterns:
                    if any(self._match_pattern(rel_path, p) for p in exclude_patterns):
                        continue

                # Skip existing symlinks
                if artifact_path.is_symlink():
                    result.already_symlinks += 1
                    continue

                result.artifacts_scanned += 1

                # Compute hash
                content_hash = self._compute_hash_streaming(artifact_path)
                size_bytes = artifact_path.stat().st_size

                # Track for deduplication
                if content_hash not in hash_counts:
                    hash_counts[content_hash] = 0
                hash_counts[content_hash] += 1

                # Store blob if not already stored
                if content_hash not in hash_to_blob:
                    if not dry_run:
                        blob = self.store_file(artifact_path)
                        hash_to_blob[content_hash] = blob
                    else:
                        # Dry run: create synthetic blob reference
                        hash_to_blob[content_hash] = StoredBlob(
                            content_hash=content_hash,
                            hash_algorithm=self.algorithm,
                            size_bytes=size_bytes,
                            path=self._get_blob_path(content_hash),
                        )

            # Count unique blobs
            result.unique_blobs = len(hash_to_blob)

            # Second pass: replace duplicates with symlinks
            for artifact_path in artifacts_path.rglob("*"):
                if not artifact_path.is_file():
                    continue
                if artifact_path.is_symlink():
                    continue

                rel_path = str(artifact_path.relative_to(artifacts_path))

                if include_patterns:
                    if not any(self._match_pattern(rel_path, p) for p in include_patterns):
                        continue

                if exclude_patterns:
                    if any(self._match_pattern(rel_path, p) for p in exclude_patterns):
                        continue

                content_hash = self._compute_hash_streaming(artifact_path)
                size_bytes = artifact_path.stat().st_size

                # Only deduplicate if there are multiple copies
                if hash_counts.get(content_hash, 0) > 1 or self.exists(content_hash):
                    blob = hash_to_blob.get(content_hash)
                    if blob:
                        if not dry_run:
                            # Replace file with symlink
                            artifact_path.unlink()
                            artifact_path.symlink_to(blob.path.resolve())

                        result.artifacts_deduplicated += 1
                        result.space_saved_bytes += size_bytes

                        result.deduplicated.append(DedupeArtifact(
                            original_path=rel_path,
                            content_hash=content_hash,
                            size_bytes=size_bytes,
                            replaced_with_symlink=not dry_run,
                            blob_path=str(blob.path),
                        ))

            result.success = True
            logger.info(f"Deduplication complete: {result.summary}")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Deduplication failed: {e}")

        result.duration_seconds = time.time() - start_time
        return result

    def deduplicate_multiple_runs(
        self,
        run_paths: List[Path],
        dry_run: bool = False,
    ) -> Dict[str, DedupeResult]:
        """
        Deduplicate artifacts across multiple runs.

        Args:
            run_paths: List of run paths to deduplicate
            dry_run: If True, simulate without modifying files

        Returns:
            Dictionary mapping run_id to DedupeResult
        """
        results = {}
        for run_path in run_paths:
            run_id = run_path.name
            results[run_id] = self.deduplicate_artifacts(run_path, dry_run=dry_run)
        return results

    def find_duplicate_artifacts(
        self,
        run_path: Path,
    ) -> Dict[str, List[str]]:
        """
        Find duplicate artifacts without modifying files.

        Args:
            run_path: Path to run directory

        Returns:
            Dictionary mapping content_hash to list of duplicate paths
        """
        artifacts_path = run_path / "artifacts"
        if not artifacts_path.exists():
            return {}

        hash_to_paths: Dict[str, List[str]] = {}

        for artifact_path in artifacts_path.rglob("*"):
            if not artifact_path.is_file() or artifact_path.is_symlink():
                continue

            content_hash = self._compute_hash_streaming(artifact_path)
            rel_path = str(artifact_path.relative_to(artifacts_path))

            if content_hash not in hash_to_paths:
                hash_to_paths[content_hash] = []
            hash_to_paths[content_hash].append(rel_path)

        # Filter to only duplicates
        return {
            h: paths for h, paths in hash_to_paths.items()
            if len(paths) > 1
        }

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def restore_from_symlinks(
        self,
        run_path: Path,
    ) -> Tuple[int, List[str]]:
        """
        Restore files from symlinks by copying blob content.

        Useful for preparing a run for archival without blob dependencies.

        Args:
            run_path: Path to run directory

        Returns:
            Tuple of (restored_count, errors)
        """
        artifacts_path = run_path / "artifacts"
        if not artifacts_path.exists():
            return 0, ["Artifacts directory not found"]

        restored = 0
        errors = []

        for artifact_path in artifacts_path.rglob("*"):
            if not artifact_path.is_symlink():
                continue

            try:
                # Read the blob content
                target = artifact_path.resolve()
                if not target.exists():
                    errors.append(f"Broken symlink: {artifact_path}")
                    continue

                content = target.read_bytes()

                # Remove symlink and write file
                artifact_path.unlink()
                with open(artifact_path, 'wb') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())

                restored += 1

            except Exception as e:
                errors.append(f"Failed to restore {artifact_path}: {e}")

        return restored, errors


# Convenience functions

def get_content_store(root_path: Optional[Path] = None) -> ContentStore:
    """
    Get content store instance.

    Args:
        root_path: Root path (defaults to LibV2 path)

    Returns:
        ContentStore instance
    """
    if root_path is None:
        from .paths import LIBV2_PATH
        root_path = LIBV2_PATH

    return ContentStore(root_path)


def store_artifact(
    content: bytes,
    root_path: Optional[Path] = None
) -> StoredBlob:
    """
    Convenience function to store artifact.

    Args:
        content: Bytes to store
        root_path: Optional root path

    Returns:
        StoredBlob reference
    """
    store = get_content_store(root_path)
    return store.store(content)


def retrieve_artifact(
    content_hash: str,
    root_path: Optional[Path] = None
) -> Optional[bytes]:
    """
    Convenience function to retrieve artifact.

    Args:
        content_hash: Hash of artifact
        root_path: Optional root path

    Returns:
        Content bytes or None
    """
    store = get_content_store(root_path)
    return store.retrieve(content_hash)


def deduplicate_run(
    run_path: Path,
    root_path: Optional[Path] = None,
    dry_run: bool = False,
) -> DedupeResult:
    """
    Convenience function to deduplicate a run's artifacts.

    Phase 0.5: Artifact Deduplication (E2)

    Args:
        run_path: Path to run directory
        root_path: Optional root path for content store
        dry_run: If True, simulate without modifying files

    Returns:
        DedupeResult with deduplication statistics
    """
    store = get_content_store(root_path)
    return store.deduplicate_artifacts(run_path, dry_run=dry_run)


def find_duplicates(
    run_path: Path,
    root_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Find duplicate artifacts in a run without modifying files.

    Args:
        run_path: Path to run directory
        root_path: Optional root path for content store

    Returns:
        Dictionary mapping content_hash to list of duplicate paths
    """
    store = get_content_store(root_path)
    return store.find_duplicate_artifacts(run_path)
