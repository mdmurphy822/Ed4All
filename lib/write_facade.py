"""
Write Facade - Centralized Write Discipline

Provides a controlled interface for all file writes:
- Path validation within allowed directories
- Atomic write via temp+rename pattern
- Audit logging of all writes
- Transaction support for multi-file operations

Phase 0.5 Enhancement: Write Discipline Enforcement (A3)
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


# ============================================================================
# CONSTANTS
# ============================================================================

# Default max path length for security
DEFAULT_MAX_PATH_LENGTH = 4096


# ============================================================================
# EXCEPTIONS
# ============================================================================

class WriteValidationError(Exception):
    """Raised when write validation fails."""
    pass


class PathSecurityError(Exception):
    """Raised when path security check fails."""
    pass


class TransactionError(Exception):
    """Raised when a transaction fails."""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WriteResult:
    """Result of a write operation."""
    success: bool
    path: str
    bytes_written: int = 0
    content_hash: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionResult:
    """Result of a transaction with multiple writes."""
    success: bool
    writes_completed: int = 0
    writes_failed: int = 0
    results: List[WriteResult] = field(default_factory=list)
    rolled_back: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "writes_completed": self.writes_completed,
            "writes_failed": self.writes_failed,
            "results": [r.to_dict() for r in self.results],
            "rolled_back": self.rolled_back,
            "error": self.error,
        }


# ============================================================================
# WRITE FACADE CLASS
# ============================================================================

class WriteFacade:
    """
    Centralized facade for controlled file writes.

    All writes go through this facade to ensure:
    - Path is within allowed directories
    - Atomic write via temp+rename
    - Optional audit logging
    - Optional transaction semantics
    """

    def __init__(
        self,
        allowed_paths: List[Path],
        audit_callback: Optional[Callable[[WriteResult], None]] = None,
        max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
        disallow_parent_traversal: bool = True,
        enforce_allowed_paths: bool = True,
    ):
        """
        Initialize write facade.

        Args:
            allowed_paths: List of allowed base paths for writes
            audit_callback: Optional callback for auditing writes
            max_path_length: Maximum allowed path length
            disallow_parent_traversal: Reject paths with .. components
            enforce_allowed_paths: Whether to enforce path restrictions
        """
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
        self.audit_callback = audit_callback
        self.max_path_length = max_path_length
        self.disallow_parent_traversal = disallow_parent_traversal
        self.enforce_allowed_paths = enforce_allowed_paths

        # Transaction state
        self._in_transaction = False
        self._transaction_writes: List[Path] = []
        self._transaction_backups: Dict[str, Optional[bytes]] = {}

    def validate_path(self, path: Path) -> None:
        """
        Validate that a path is allowed for writing.

        Args:
            path: Path to validate

        Raises:
            PathSecurityError: If path is not allowed
        """
        path = Path(path)
        path_str = str(path)

        # Check path length
        if len(path_str) > self.max_path_length:
            raise PathSecurityError(
                f"Path exceeds maximum length ({len(path_str)} > {self.max_path_length})"
            )

        # Check for parent traversal
        if self.disallow_parent_traversal and ".." in path.parts:
            raise PathSecurityError(
                f"Parent directory traversal not allowed: {path}"
            )

        # Check if path is within allowed directories
        if self.enforce_allowed_paths:
            resolved = path.resolve()
            is_allowed = any(
                self._is_subpath(resolved, allowed)
                for allowed in self.allowed_paths
            )

            if not is_allowed:
                raise PathSecurityError(
                    f"Path not within allowed directories: {path}"
                )

    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is a subpath of parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def write(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
    ) -> WriteResult:
        """
        Write content to a file atomically.

        Args:
            path: Path to write to
            content: Content to write (str or bytes)
            encoding: Encoding for string content

        Returns:
            WriteResult with operation status
        """
        path = Path(path)

        try:
            # Validate path
            self.validate_path(path)

            # Convert content to bytes
            if isinstance(content, str):
                content_bytes = content.encode(encoding)
            else:
                content_bytes = content

            # Compute content hash
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Track for transaction rollback
            if self._in_transaction:
                self._backup_for_rollback(path)

            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write via temp file + rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp"
            )

            try:
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(content_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                os.rename(temp_path, path)

            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            result = WriteResult(
                success=True,
                path=str(path),
                bytes_written=len(content_bytes),
                content_hash=f"sha256:{content_hash}",
            )

            # Track for transaction
            if self._in_transaction:
                self._transaction_writes.append(path)

            # Audit callback
            if self.audit_callback:
                self.audit_callback(result)

            return result

        except (PathSecurityError, WriteValidationError) as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=str(e),
            )

        except Exception as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=f"Write failed: {e}",
            )

    def write_json(
        self,
        path: Union[str, Path],
        data: Any,
        indent: int = 2,
    ) -> WriteResult:
        """
        Write JSON data to a file atomically.

        Args:
            path: Path to write to
            data: Data to serialize as JSON
            indent: JSON indentation

        Returns:
            WriteResult with operation status
        """
        try:
            content = json.dumps(data, indent=indent, sort_keys=True)
            return self.write(path, content)
        except (TypeError, ValueError) as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=f"JSON serialization failed: {e}",
            )

    def _backup_for_rollback(self, path: Path) -> None:
        """Backup file content for potential rollback."""
        path_str = str(path)
        if path_str not in self._transaction_backups:
            if path.exists():
                self._transaction_backups[path_str] = path.read_bytes()
            else:
                self._transaction_backups[path_str] = None

    # ========================================================================
    # TRANSACTION SUPPORT
    # ========================================================================

    def begin_transaction(self) -> None:
        """Begin a transaction for atomic multi-file writes."""
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")

        self._in_transaction = True
        self._transaction_writes = []
        self._transaction_backups = {}

    def commit_transaction(self) -> TransactionResult:
        """
        Commit the current transaction.

        Returns:
            TransactionResult with operation summary
        """
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        result = TransactionResult(
            success=True,
            writes_completed=len(self._transaction_writes),
        )

        # Clear transaction state
        self._in_transaction = False
        self._transaction_writes = []
        self._transaction_backups = {}

        return result

    def rollback_transaction(self) -> TransactionResult:
        """
        Rollback the current transaction.

        Restores files to their pre-transaction state.

        Returns:
            TransactionResult with rollback summary
        """
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        rolled_back_count = 0
        errors = []

        # Restore backups
        for path_str, backup_content in self._transaction_backups.items():
            path = Path(path_str)
            try:
                if backup_content is None:
                    # File didn't exist before - delete it
                    if path.exists():
                        path.unlink()
                else:
                    # Restore original content
                    path.write_bytes(backup_content)
                rolled_back_count += 1
            except Exception as e:
                errors.append(f"Failed to rollback {path}: {e}")

        result = TransactionResult(
            success=len(errors) == 0,
            writes_completed=rolled_back_count,
            rolled_back=True,
            error="; ".join(errors) if errors else None,
        )

        # Clear transaction state
        self._in_transaction = False
        self._transaction_writes = []
        self._transaction_backups = {}

        return result

    def __enter__(self) -> "WriteFacade":
        """Context manager entry - begins transaction."""
        self.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - commits or rolls back."""
        if exc_type is not None:
            # Exception occurred - rollback
            self.rollback_transaction()
        else:
            # Success - commit
            self.commit_transaction()


# ============================================================================
# WRITE TRACKER
# ============================================================================

class WriteTracker:
    """
    Tracks all writes through a WriteFacade for audit purposes.
    """

    def __init__(self):
        self.writes: List[WriteResult] = []

    def track(self, result: WriteResult) -> None:
        """Track a write result."""
        self.writes.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked writes."""
        successful = [w for w in self.writes if w.success]
        failed = [w for w in self.writes if not w.success]

        return {
            "total_writes": len(self.writes),
            "successful": len(successful),
            "failed": len(failed),
            "total_bytes": sum(w.bytes_written for w in successful),
            "paths": [w.path for w in self.writes],
        }

    def to_audit_log(self) -> List[Dict[str, Any]]:
        """Convert to audit log format."""
        return [w.to_dict() for w in self.writes]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_run_write_facade(
    run_path: Path,
    audit_callback: Optional[Callable[[WriteResult], None]] = None,
) -> WriteFacade:
    """
    Create a WriteFacade for a specific run.

    Args:
        run_path: Path to run directory
        audit_callback: Optional audit callback

    Returns:
        WriteFacade configured for the run
    """
    run_path = Path(run_path)

    # Allow writes to run subdirectories
    allowed_paths = [
        run_path / "decisions",
        run_path / "audit",
        run_path / "artifacts",
        run_path / "checkpoints",
        run_path / "state",
    ]

    return WriteFacade(
        allowed_paths=allowed_paths,
        audit_callback=audit_callback,
    )


def atomic_write(
    path: Union[str, Path],
    content: Union[str, bytes],
    encoding: str = "utf-8",
) -> WriteResult:
    """
    Standalone atomic write without facade restrictions.

    Args:
        path: Path to write to
        content: Content to write
        encoding: Encoding for string content

    Returns:
        WriteResult
    """
    # Create unrestricted facade for single write
    facade = WriteFacade(
        allowed_paths=[Path(path).parent],
        enforce_allowed_paths=False,
    )
    return facade.write(path, content, encoding)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "DEFAULT_MAX_PATH_LENGTH",
    # Exceptions
    "WriteValidationError",
    "PathSecurityError",
    "TransactionError",
    # Data classes
    "WriteResult",
    "TransactionResult",
    # Main class
    "WriteFacade",
    # Tracker
    "WriteTracker",
    # Convenience functions
    "create_run_write_facade",
    "atomic_write",
]
