"""
Hash Chain - Append-Only Hash-Chained Event Logs

Provides tamper-evident logging where each event includes:
- Monotonic sequence number
- Hash of previous event (chain)
- Hash of current event content

This creates an audit trail where any modification is detectable.

Phase 0 Hardening: Requirement 10 (Security Posture - Tamper Evidence)
Phase 0.5 Enhancement: Lock timeout and auto-verify (A3)
"""

import fcntl
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .provenance import DEFAULT_ALGORITHM, hash_content

# ============================================================================
# LOCK CONFIGURATION (matches sequence_manager.py)
# ============================================================================

DEFAULT_LOCK_TIMEOUT_SECONDS = 30
DEFAULT_LOCK_RETRY_BACKOFF = [0.1, 0.5, 2.0]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ChainLockTimeoutError(Exception):
    """Raised when hash chain lock acquisition times out."""
    pass


class ChainIntegrityError(Exception):
    """Raised when hash chain integrity verification fails."""
    pass


# ============================================================================
# CONSTANTS
# ============================================================================

GENESIS_HASH = "genesis"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChainedEvent:
    """A single event in the hash chain."""
    seq: int
    prev_hash: str
    event_hash: str
    timestamp: str
    event: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "event_hash": self.event_hash,
            "timestamp": self.timestamp,
            "event": self.event,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChainedEvent":
        """Create from dictionary."""
        return cls(
            seq=data["seq"],
            prev_hash=data["prev_hash"],
            event_hash=data["event_hash"],
            timestamp=data["timestamp"],
            event=data["event"],
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'), sort_keys=True)


@dataclass
class VerificationResult:
    """Result of chain verification."""
    valid: bool
    total_events: int
    verified_events: int
    first_invalid_seq: Optional[int] = None
    error_message: Optional[str] = None
    chain_head_hash: Optional[str] = None


# ============================================================================
# HASH CHAIN CLASS
# ============================================================================

class HashChainedLog:
    """
    Append-only log with hash chaining for tamper evidence.

    Each event is linked to the previous via hash, creating an
    immutable audit trail.

    Supports lock timeout and optional auto-verification on close.
    """

    def __init__(
        self,
        log_path: Path,
        algorithm: str = DEFAULT_ALGORITHM,
        auto_create: bool = True,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        lock_retry_backoff: Optional[List[float]] = None,
        verify_on_close: bool = False,
    ):
        """
        Initialize hash-chained log.

        Args:
            log_path: Path to the JSONL log file
            algorithm: Hash algorithm for chaining
            auto_create: Create file if it doesn't exist
            lock_timeout_seconds: Timeout for lock acquisition
            lock_retry_backoff: Backoff delays for lock retries
            verify_on_close: Whether to verify chain when closing
        """
        self.log_path = Path(log_path)
        self.algorithm = algorithm
        self._seq = 0
        self._prev_hash = GENESIS_HASH

        # Lock configuration
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_retry_backoff = lock_retry_backoff or DEFAULT_LOCK_RETRY_BACKOFF
        self.verify_on_close = verify_on_close

        # Track if this instance is closed
        self._closed = False

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log_path.exists():
            # Load chain state from existing file
            self._load_chain_state()
        elif auto_create:
            # Initialize empty file
            self.log_path.touch()

    def _load_chain_state(self) -> None:
        """Load the current chain state from existing log."""
        last_event = None
        with open(self.log_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    last_event = json.loads(line)

        if last_event:
            self._seq = last_event["seq"] + 1
            self._prev_hash = last_event["event_hash"]

    def _compute_event_hash(self, prev_hash: str, event_json: str) -> str:
        """Compute hash for an event."""
        content = f"{prev_hash}{event_json}"
        return hash_content(content, self.algorithm)

    def _acquire_lock_with_timeout(
        self,
        lock_file,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Acquire an exclusive lock with timeout.

        Args:
            lock_file: Open file object to lock
            timeout_seconds: Timeout in seconds

        Returns:
            True if lock acquired

        Raises:
            ChainLockTimeoutError: If lock cannot be acquired
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.lock_timeout_seconds
        start_time = time.monotonic()
        attempts = 0

        while True:
            attempts += 1

            try:
                # Try non-blocking lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                elapsed = time.monotonic() - start_time

                if elapsed >= timeout:
                    raise ChainLockTimeoutError(
                        f"Failed to acquire lock for {self.log_path} "
                        f"after {elapsed:.2f}s ({attempts} attempts)"
                    ) from None

                # Calculate backoff delay
                backoff_index = min(attempts - 1, len(self.lock_retry_backoff) - 1)
                delay = self.lock_retry_backoff[backoff_index]

                # Don't wait longer than remaining timeout
                remaining = timeout - elapsed
                delay = min(delay, remaining)

                if delay > 0:
                    time.sleep(delay)

    def append(
        self,
        event: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> ChainedEvent:
        """
        Append an event to the chain.

        Thread-safe via file locking with timeout.

        Args:
            event: Event data to append
            timeout_seconds: Optional timeout override

        Returns:
            ChainedEvent with chain metadata

        Raises:
            ChainLockTimeoutError: If lock cannot be acquired
            RuntimeError: If chain is closed
        """
        if self._closed:
            raise RuntimeError("Cannot append to closed hash chain")

        lock_path = self.log_path.with_suffix('.lock')

        with open(lock_path, 'w') as lock_file:
            self._acquire_lock_with_timeout(lock_file, timeout_seconds)
            try:
                # Re-read chain state in case another process appended
                self._load_chain_state()

                # Create chained event
                timestamp = datetime.now().isoformat()
                event_json = json.dumps(event, separators=(',', ':'), sort_keys=True)
                event_hash = self._compute_event_hash(self._prev_hash, event_json)

                chained = ChainedEvent(
                    seq=self._seq,
                    prev_hash=self._prev_hash,
                    event_hash=event_hash,
                    timestamp=timestamp,
                    event=event,
                )

                # Append to file
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(chained.to_json() + '\n')
                    f.flush()
                    os.fsync(f.fileno())

                # Update state
                self._prev_hash = event_hash
                self._seq += 1

                return chained
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get_current_seq(self) -> int:
        """Get the next sequence number to be assigned."""
        return self._seq

    def get_chain_head(self) -> str:
        """Get the hash of the most recent event (chain head)."""
        return self._prev_hash

    def read_all(self) -> List[ChainedEvent]:
        """
        Read all events from the log.

        Returns:
            List of ChainedEvent objects
        """
        events = []
        with open(self.log_path, encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(ChainedEvent.from_dict(json.loads(line)))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return events

    def read_range(self, start_seq: int, end_seq: int) -> List[ChainedEvent]:
        """
        Read a range of events.

        Args:
            start_seq: Starting sequence number (inclusive)
            end_seq: Ending sequence number (exclusive)

        Returns:
            List of ChainedEvent objects in range
        """
        events = []
        with open(self.log_path, encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        seq = data["seq"]
                        if start_seq <= seq < end_seq:
                            events.append(ChainedEvent.from_dict(data))
                        elif seq >= end_seq:
                            break
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return events

    def iterate(self) -> Iterator[ChainedEvent]:
        """
        Iterate over events without loading all into memory.

        Yields:
            ChainedEvent objects
        """
        with open(self.log_path, encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        yield ChainedEvent.from_dict(json.loads(line))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def verify(self) -> VerificationResult:
        """
        Verify chain integrity.

        Checks that each event's hash chains correctly to the next.

        Returns:
            VerificationResult with validation status
        """
        events = self.read_all()

        if not events:
            return VerificationResult(
                valid=True,
                total_events=0,
                verified_events=0,
                chain_head_hash=GENESIS_HASH,
            )

        prev_hash = GENESIS_HASH
        verified = 0

        for event in events:
            # Check sequence is monotonic
            if event.seq != verified:
                return VerificationResult(
                    valid=False,
                    total_events=len(events),
                    verified_events=verified,
                    first_invalid_seq=event.seq,
                    error_message=f"Sequence gap: expected {verified}, got {event.seq}",
                )

            # Check prev_hash matches
            if event.prev_hash != prev_hash:
                return VerificationResult(
                    valid=False,
                    total_events=len(events),
                    verified_events=verified,
                    first_invalid_seq=event.seq,
                    error_message=f"Chain broken at seq {event.seq}: prev_hash mismatch",
                )

            # Verify event hash
            event_json = json.dumps(event.event, separators=(',', ':'), sort_keys=True)
            expected_hash = self._compute_event_hash(prev_hash, event_json)

            if event.event_hash != expected_hash:
                return VerificationResult(
                    valid=False,
                    total_events=len(events),
                    verified_events=verified,
                    first_invalid_seq=event.seq,
                    error_message=f"Event hash mismatch at seq {event.seq}: content may be tampered",
                )

            prev_hash = event.event_hash
            verified += 1

        return VerificationResult(
            valid=True,
            total_events=len(events),
            verified_events=verified,
            chain_head_hash=prev_hash,
        )

    def close(self, verify: Optional[bool] = None) -> VerificationResult:
        """
        Close the hash chain.

        Optionally verifies chain integrity before closing.

        Args:
            verify: Whether to verify (uses self.verify_on_close if None)

        Returns:
            VerificationResult (empty if not verified)

        Raises:
            ChainIntegrityError: If verification fails and verify_on_close is True
        """
        should_verify = verify if verify is not None else self.verify_on_close

        if should_verify:
            result = self.verify()
            if not result.valid:
                raise ChainIntegrityError(
                    f"Chain integrity verification failed: {result.error_message}"
                )
            self._closed = True
            return result

        self._closed = True
        return VerificationResult(
            valid=True,
            total_events=0,
            verified_events=0,
        )

    def __enter__(self) -> "HashChainedLog":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes chain."""
        if not self._closed:
            try:
                self.close()
            except ChainIntegrityError:
                # Don't suppress original exception
                if exc_type is None:
                    raise

    def find_event(self, seq: int) -> Optional[ChainedEvent]:
        """
        Find an event by sequence number.

        Args:
            seq: Sequence number to find

        Returns:
            ChainedEvent if found, None otherwise
        """
        with open(self.log_path, encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if data["seq"] == seq:
                            return ChainedEvent.from_dict(data)
                        elif data["seq"] > seq:
                            break
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return None


# ============================================================================
# INDEX MANAGEMENT
# ============================================================================

class ChainIndex:
    """
    Optional index for fast lookups in hash-chained logs.

    Maintains a separate index file mapping sequence numbers to
    byte offsets in the log file.
    """

    def __init__(self, log: HashChainedLog):
        """
        Initialize chain index.

        Args:
            log: HashChainedLog to index
        """
        self.log = log
        self.index_path = log.log_path.with_suffix('.index.json')
        self._index: Dict[int, int] = {}  # seq -> byte offset

        if self.index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load index from file."""
        with open(self.index_path, encoding='utf-8') as f:
            data = json.load(f)
            self._index = {int(k): v for k, v in data.get("offsets", {}).items()}

    def _save_index(self) -> None:
        """Save index to file."""
        data = {
            "log_path": str(self.log.log_path),
            "offsets": {str(k): v for k, v in self._index.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def rebuild(self) -> int:
        """
        Rebuild the index from the log file.

        Returns:
            Number of events indexed
        """
        self._index = {}
        offset = 0

        with open(self.log.log_path, encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._index[data["seq"]] = offset
                offset = f.tell()

        self._save_index()
        return len(self._index)

    def get_offset(self, seq: int) -> Optional[int]:
        """
        Get byte offset for a sequence number.

        Args:
            seq: Sequence number

        Returns:
            Byte offset, or None if not indexed
        """
        return self._index.get(seq)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_hash_chain(
    path: Path,
    algorithm: str = DEFAULT_ALGORITHM
) -> HashChainedLog:
    """
    Create a new hash-chained log.

    Args:
        path: Path to log file
        algorithm: Hash algorithm

    Returns:
        HashChainedLog instance
    """
    return HashChainedLog(path, algorithm, auto_create=True)


def verify_chain(path: Path) -> VerificationResult:
    """
    Verify a hash-chained log's integrity.

    Args:
        path: Path to log file

    Returns:
        VerificationResult
    """
    log = HashChainedLog(path, auto_create=False)
    return log.verify()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "GENESIS_HASH",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_LOCK_RETRY_BACKOFF",
    # Exceptions
    "ChainLockTimeoutError",
    "ChainIntegrityError",
    # Data classes
    "ChainedEvent",
    "VerificationResult",
    # Main class
    "HashChainedLog",
    # Index
    "ChainIndex",
    # Convenience functions
    "create_hash_chain",
    "verify_chain",
]
