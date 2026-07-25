"""
State Manager - Atomic File Operations with fcntl Locking

Provides atomic JSON read/write operations with proper file locking
to prevent race conditions in concurrent access scenarios.
"""

import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# In-process locks for atomic_update_json to prevent race conditions
# between threads (fcntl only protects against other processes)
_update_locks: Dict[str, threading.Lock] = {}
_update_locks_guard = threading.Lock()


class StateFileCorruptedError(ValueError):
    """A JSON state file on disk failed to parse (torn/interleaved write).

    Raised by :func:`load_state_json` with a message naming the file, the
    corruption position, and the recovery hint — instead of a bare
    ``json.JSONDecodeError`` traceback.
    """


def atomic_write_json(
    path: Path,
    data: Dict[str, Any],
    indent: int = 2,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """
    Atomic JSON write with fcntl locking.

    Uses temp file + rename pattern for atomicity:
    1. Write to temp file with exclusive lock
    2. Flush and sync to disk
    3. Atomic rename to target path

    Args:
        path: Target file path
        data: Dictionary to serialize as JSON
        indent: JSON indentation level (default 2)
        default: Optional ``json.dump`` fallback serializer (e.g. ``str``)
            for non-JSON-native values; ``None`` keeps strict serialization

    Raises:
        OSError: If file operations fail
        TypeError: If data is not JSON serializable
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use unique temp file to avoid collision when multiple writers target the same path
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=indent, default=default)
                f.flush()
                os.fsync(f.fileno())  # Force disk write
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Atomic rename (POSIX guarantees atomicity)
        os.rename(temp_name, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_read_json(path: Path, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Read JSON file with shared lock.

    Args:
        path: File path to read
        default: Default value if file doesn't exist (default None)

    Returns:
        Parsed JSON as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist and no default provided
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(path)

    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"File not found: {path}")

    with open(path) as f:
        # Acquire shared lock (allows concurrent reads)
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_state_json(path: Path, recovery_hint: Optional[str] = None) -> Dict[str, Any]:
    """Load a JSON state file, converting parse failures into a clear error.

    A torn / interleaved write (two non-atomic writers racing the same file)
    leaves a document that fails mid-parse. Instead of a bare
    ``json.JSONDecodeError`` traceback, raise :class:`StateFileCorruptedError`
    naming the file, the corruption position, and (when given) a recovery
    hint pointing the operator at the authoritative recovery source.

    Args:
        path: State file to read
        recovery_hint: Optional operator-facing recovery guidance appended to
            the error message

    Raises:
        FileNotFoundError: If the file does not exist
        StateFileCorruptedError: If the file exists but is not valid JSON
    """
    path = Path(path)
    try:
        return atomic_read_json(path)
    except json.JSONDecodeError as exc:
        message = (
            f"State file corrupted: {path} — JSON parse failed at char {exc.pos} "
            f"(line {exc.lineno} column {exc.colno}): {exc.msg}."
        )
        if recovery_hint:
            message = f"{message} {recovery_hint}"
        raise StateFileCorruptedError(message) from exc


def atomic_update_json(
    path: Path,
    update_fn: callable,
    default: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Atomically read, update, and write JSON file.

    Args:
        path: File path to update
        update_fn: Function that takes current data and returns updated data
        default: Default value if file doesn't exist

    Returns:
        Updated dictionary

    Example:
        def increment_counter(data):
            data['counter'] = data.get('counter', 0) + 1
            return data

        atomic_update_json(path, increment_counter, default={})
    """
    path = Path(path)

    # Serialize read-update-write per path to prevent concurrent thread races
    key = str(path.resolve())
    with _update_locks_guard:
        if key not in _update_locks:
            _update_locks[key] = threading.Lock()
        lock = _update_locks[key]

    with lock:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Read current data
        if path.exists():
            current = atomic_read_json(path)
        elif default is not None:
            current = default.copy()
        else:
            current = {}

        # Apply update
        updated = update_fn(current)

        # Write atomically
        atomic_write_json(path, updated)

        return updated
