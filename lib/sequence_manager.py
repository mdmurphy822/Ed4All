"""
Sequence Manager - Monotonic Sequence Numbers per Run

Provides thread-safe, crash-safe monotonic sequence numbers for events
within a run. Each run has its own sequence counter stored at:
    state/runs/<run_id>/state/sequence.json

Phase 0 Hardening: Requirement 4 (Decision Capture Integrity)
Phase 0.5 Enhancement: Lock timeout and retry logic (A2)
"""

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .run_manager import get_current_run, RUNS_PATH


# ============================================================================
# LOCK CONFIGURATION
# ============================================================================

# Default lock timeout in seconds
DEFAULT_LOCK_TIMEOUT_SECONDS = 30

# Default retry count
DEFAULT_LOCK_RETRY_COUNT = 3

# Exponential backoff delays in seconds
DEFAULT_LOCK_RETRY_BACKOFF = [0.1, 0.5, 2.0]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class LockTimeoutError(Exception):
    """Raised when lock acquisition times out."""
    pass


class LockAcquisitionError(Exception):
    """Raised when lock acquisition fails after retries."""
    pass


# ============================================================================
# LOCK RESULT
# ============================================================================

@dataclass
class LockResult:
    """Result of a lock acquisition attempt."""
    acquired: bool
    attempts: int
    total_wait_time: float
    error: Optional[str] = None


# ============================================================================
# SEQUENCE MANAGER CLASS
# ============================================================================

class SequenceManager:
    """
    Manages monotonic sequence numbers for a run.

    Thread-safe via fcntl file locking with timeout.
    Crash-safe via atomic write pattern.
    Includes retry logic with exponential backoff.
    """

    def __init__(
        self,
        run_id: str,
        runs_path: Path = RUNS_PATH,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        lock_retry_count: int = DEFAULT_LOCK_RETRY_COUNT,
        lock_retry_backoff: Optional[list] = None,
    ):
        """
        Initialize sequence manager for a run.

        Args:
            run_id: Run ID to manage sequences for
            runs_path: Base path for runs
            lock_timeout_seconds: Timeout for lock acquisition
            lock_retry_count: Number of retries on lock failure
            lock_retry_backoff: Backoff delays in seconds
        """
        self.run_id = run_id
        self.runs_path = runs_path
        self.sequence_path = runs_path / run_id / "state" / "sequence.json"

        # Lock configuration
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_retry_count = lock_retry_count
        self.lock_retry_backoff = lock_retry_backoff or DEFAULT_LOCK_RETRY_BACKOFF

        # Ensure directory exists
        self.sequence_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize sequence file if it doesn't exist
        if not self.sequence_path.exists():
            self._initialize_sequence_file()

    def _initialize_sequence_file(self) -> None:
        """Initialize the sequence file with starting values."""
        initial_state = {
            "run_id": self.run_id,
            "current_seq": 0,
            "last_event_id": None,
            "initialized_at": self._get_timestamp(),
        }
        self._atomic_write(initial_state)

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _atomic_write(self, data: dict) -> None:
        """Atomically write sequence state."""
        temp_path = self.sequence_path.with_suffix('.tmp')

        with open(temp_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        os.rename(temp_path, self.sequence_path)

    def _atomic_read(self) -> dict:
        """Atomically read sequence state."""
        with open(self.sequence_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _acquire_lock_with_timeout(
        self,
        lock_file,
        timeout_seconds: Optional[float] = None,
    ) -> LockResult:
        """
        Acquire an exclusive lock with timeout.

        Uses non-blocking lock attempts with polling and exponential backoff.

        Args:
            lock_file: Open file object to lock
            timeout_seconds: Timeout in seconds (uses self.lock_timeout_seconds if None)

        Returns:
            LockResult indicating success/failure
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.lock_timeout_seconds
        start_time = time.monotonic()
        attempts = 0
        total_wait = 0.0

        while True:
            attempts += 1

            try:
                # Try non-blocking lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return LockResult(
                    acquired=True,
                    attempts=attempts,
                    total_wait_time=time.monotonic() - start_time,
                )
            except BlockingIOError:
                # Lock is held by another process
                elapsed = time.monotonic() - start_time

                if elapsed >= timeout:
                    return LockResult(
                        acquired=False,
                        attempts=attempts,
                        total_wait_time=elapsed,
                        error=f"Lock timeout after {elapsed:.2f}s ({attempts} attempts)",
                    )

                # Calculate backoff delay
                backoff_index = min(attempts - 1, len(self.lock_retry_backoff) - 1)
                delay = self.lock_retry_backoff[backoff_index]

                # Don't wait longer than remaining timeout
                remaining = timeout - elapsed
                delay = min(delay, remaining)

                if delay > 0:
                    time.sleep(delay)
                    total_wait += delay

    def _acquire_lock_with_retry(
        self,
        lock_file,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """
        Acquire lock with retry and backoff, raising on failure.

        Args:
            lock_file: Open file object to lock
            timeout_seconds: Timeout in seconds

        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout
        """
        result = self._acquire_lock_with_timeout(lock_file, timeout_seconds)

        if not result.acquired:
            raise LockTimeoutError(
                f"Failed to acquire lock for {self.run_id}: {result.error}"
            )

    def next_sequence(self, timeout_seconds: Optional[float] = None) -> Tuple[int, str]:
        """
        Get the next sequence number and generate an event ID.

        This operation is atomic and thread-safe with timeout protection.

        Args:
            timeout_seconds: Optional timeout override

        Returns:
            Tuple of (sequence_number, event_id)

        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout
        """
        lock_path = self.sequence_path.with_suffix('.lock')

        with open(lock_path, 'w') as lock_file:
            # Acquire exclusive lock with timeout
            self._acquire_lock_with_retry(lock_file, timeout_seconds)

            try:
                # Read current state
                state = self._atomic_read()

                # Increment sequence
                seq = state["current_seq"]
                state["current_seq"] = seq + 1

                # Generate event ID
                event_id = generate_event_id()
                state["last_event_id"] = event_id
                state["last_updated_at"] = self._get_timestamp()

                # Write updated state
                self._atomic_write(state)

                return seq, event_id
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get_current_sequence(self) -> int:
        """
        Get the current sequence number (without incrementing).

        Returns:
            Current sequence number (next to be assigned)
        """
        state = self._atomic_read()
        return state["current_seq"]

    def get_last_event_id(self) -> Optional[str]:
        """
        Get the last assigned event ID.

        Returns:
            Last event ID, or None if no events yet
        """
        state = self._atomic_read()
        return state.get("last_event_id")


# ============================================================================
# EVENT ID GENERATION
# ============================================================================

def generate_event_id() -> str:
    """
    Generate a unique event ID.

    Format: EVT_<16-hex-chars>
    """
    return f"EVT_{uuid.uuid4().hex[:16]}"


# ============================================================================
# GLOBAL SEQUENCE ACCESS
# ============================================================================

_sequence_managers: dict[str, SequenceManager] = {}


def get_sequence_manager(run_id: Optional[str] = None) -> Optional[SequenceManager]:
    """
    Get or create a sequence manager for a run.

    Args:
        run_id: Run ID, or None to use current run

    Returns:
        SequenceManager for the run, or None if no run active
    """
    if run_id is None:
        context = get_current_run()
        if context is None:
            return None
        run_id = context.run_id

    if run_id not in _sequence_managers:
        _sequence_managers[run_id] = SequenceManager(run_id)

    return _sequence_managers[run_id]


def next_sequence(run_id: Optional[str] = None) -> Tuple[int, str]:
    """
    Get the next sequence number and event ID.

    Convenience function that uses the current run context if no run_id provided.

    Args:
        run_id: Run ID, or None to use current run

    Returns:
        Tuple of (sequence_number, event_id)

    Raises:
        RuntimeError: If no run is active and no run_id provided
    """
    manager = get_sequence_manager(run_id)
    if manager is None:
        raise RuntimeError("No active run and no run_id provided")
    return manager.next_sequence()


def get_current_seq(run_id: Optional[str] = None) -> int:
    """
    Get the current sequence number.

    Args:
        run_id: Run ID, or None to use current run

    Returns:
        Current sequence number

    Raises:
        RuntimeError: If no run is active and no run_id provided
    """
    manager = get_sequence_manager(run_id)
    if manager is None:
        raise RuntimeError("No active run and no run_id provided")
    return manager.get_current_sequence()


# ============================================================================
# LEGACY SUPPORT
# ============================================================================

def next_sequence_legacy() -> Tuple[int, str]:
    """
    Generate sequence for non-run contexts (backwards compatibility).

    For use when ED4ALL_HARDENED_MODE is not enabled and no run is active.
    Returns seq=0 and a fresh event ID.

    Returns:
        Tuple of (0, event_id)
    """
    return 0, generate_event_id()


def get_sequence_for_context(run_id: Optional[str] = None) -> Tuple[int, str]:
    """
    Get sequence number, falling back to legacy mode if no run active.

    This is the recommended function for decision capture code that needs
    to support both hardened and legacy modes.

    Args:
        run_id: Optional run ID

    Returns:
        Tuple of (sequence_number, event_id)
    """
    try:
        return next_sequence(run_id)
    except RuntimeError:
        return next_sequence_legacy()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_LOCK_RETRY_COUNT",
    "DEFAULT_LOCK_RETRY_BACKOFF",
    # Exceptions
    "LockTimeoutError",
    "LockAcquisitionError",
    # Data classes
    "LockResult",
    # Classes
    "SequenceManager",
    # Event ID generation
    "generate_event_id",
    # Global access
    "get_sequence_manager",
    "next_sequence",
    "get_current_seq",
    # Legacy support
    "next_sequence_legacy",
    "get_sequence_for_context",
]
