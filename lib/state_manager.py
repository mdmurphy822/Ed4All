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
from typing import Any, Dict, Optional

# In-process locks for atomic_update_json to prevent race conditions
# between threads (fcntl only protects against other processes)
_update_locks: Dict[str, threading.Lock] = {}
_update_locks_guard = threading.Lock()


def atomic_write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
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
                json.dump(data, f, indent=indent)
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
