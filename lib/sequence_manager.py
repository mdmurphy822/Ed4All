"""Thread-safe, run-scoped decision-event sequencing."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from typing import Optional, Tuple


_lock = threading.Lock()
_sequences: dict[str, int] = defaultdict(int)


def generate_event_id() -> str:
    """Return a collision-resistant canonical event identifier."""

    return f"EVT_{uuid.uuid4().hex[:16]}"


def get_sequence_for_context(run_id: Optional[str] = None) -> Tuple[int, str]:
    """Allocate the next positive sequence number within ``run_id``.

    Standalone callers must still pass their capture's stable fallback run ID;
    the deterministic sentinel is only defensive support for legacy callers.
    """

    key = str(run_id or "__standalone__")
    with _lock:
        _sequences[key] += 1
        sequence = _sequences[key]
    return sequence, generate_event_id()


__all__ = ["generate_event_id", "get_sequence_for_context"]
