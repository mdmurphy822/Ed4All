"""Wave 110 / Phase D — synthesis-session budget tracking.

Tracks dispatches vs. cache hits, persists per-call telemetry, and
fails loud when ``max_dispatches`` is exceeded so a partial Claude
Max run can resume from cache without re-paying for cached calls.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional


class SynthesisBudgetExceeded(RuntimeError):
    """Raised when ``max_dispatches`` is exceeded mid-run.

    The exception carries the current counters so callers can produce
    a "resume from cache" message without re-running everything.
    """

    def __init__(
        self,
        message: str,
        *,
        dispatched: int,
        cache_hits: int,
        max_dispatches: int,
    ) -> None:
        super().__init__(message)
        self.dispatched = dispatched
        self.cache_hits = cache_hits
        self.max_dispatches = max_dispatches


@dataclass
class _BudgetTracker:
    """Counts per-run dispatches + appends a telemetry record per call.

    ``record(...)`` is called from ``ClaudeSessionProvider`` once per
    paraphrase request (regardless of cache hit). When the running
    dispatched-count would exceed ``max_dispatches``, the call raises
    ``SynthesisBudgetExceeded`` BEFORE incrementing — so a re-run with
    a higher cap picks up cleanly from the same boundary.
    """

    telemetry_path: Optional[Path] = None
    max_dispatches: Optional[int] = None
    dispatched: int = 0
    cache_hits: int = 0
    elapsed_seconds_total: float = 0.0
    errors: int = 0

    @property
    def total_calls(self) -> int:
        return self.dispatched + self.cache_hits

    def record(
        self,
        *,
        kind: str,
        chunk_id: str,
        cached: bool,
        elapsed_seconds: float,
        error_code: Optional[str] = None,
    ) -> None:
        if not cached:
            if (
                self.max_dispatches is not None
                and self.dispatched >= self.max_dispatches
            ):
                raise SynthesisBudgetExceeded(
                    f"ClaudeSessionProvider hit max_dispatches="
                    f"{self.max_dispatches} (dispatched={self.dispatched}, "
                    f"cache_hits={self.cache_hits}). Re-run with a higher "
                    f"--max-dispatches to resume from cache, or accept the "
                    f"partial output already written to .synthesis_cache.jsonl.",
                    dispatched=self.dispatched,
                    cache_hits=self.cache_hits,
                    max_dispatches=self.max_dispatches,
                )
            self.dispatched += 1
        else:
            self.cache_hits += 1
        self.elapsed_seconds_total += float(elapsed_seconds)
        if error_code:
            self.errors += 1
        if self.telemetry_path is not None:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({
                "kind": kind,
                "chunk_id": chunk_id,
                "cached": bool(cached),
                "elapsed_seconds": float(elapsed_seconds),
                "error_code": error_code,
                "dispatched_running": self.dispatched,
                "cache_hits_running": self.cache_hits,
            }, sort_keys=True)
            with self.telemetry_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def summary(self) -> Dict[str, Any]:
        return {
            "dispatched": self.dispatched,
            "cache_hits": self.cache_hits,
            "total_calls": self.total_calls,
            "elapsed_seconds_total": round(self.elapsed_seconds_total, 4),
            "errors": self.errors,
            "max_dispatches": self.max_dispatches,
            "remaining": (
                None if self.max_dispatches is None
                else max(0, self.max_dispatches - self.dispatched)
            ),
        }


class SynthesisCircuitOpen(RuntimeError):
    """Raised when the dispatcher has produced too many failures within
    the configured window. The caller should pause + retry later
    rather than burning the dispatch cap on a likely-rate-limited
    backend."""

    def __init__(
        self,
        message: str,
        *,
        failures_in_window: int,
        window_seconds: float,
    ) -> None:
        super().__init__(message)
        self.failures_in_window = failures_in_window
        self.window_seconds = window_seconds


@dataclass
class _CircuitBreaker:
    """Sliding-window failure tracker for synthesis dispatches.

    Records each failure with a monotonic timestamp; ``before_dispatch()``
    raises ``SynthesisCircuitOpen`` when ``failures_to_open`` failures
    have occurred within the last ``window_seconds``. ``record_success()``
    drops the failure window so a healthy dispatch resets the breaker.
    """

    failures_to_open: int = 3
    window_seconds: float = 60.0
    _failure_times: Deque[float] = field(default_factory=deque)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

    def record_failure(self, *, error_code: Optional[str] = None) -> None:
        self._failure_times.append(time.monotonic())

    def record_success(self) -> None:
        self._failure_times.clear()

    def before_dispatch(self) -> None:
        self._prune()
        if len(self._failure_times) >= self.failures_to_open:
            raise SynthesisCircuitOpen(
                f"Circuit open: {len(self._failure_times)} failures in last "
                f"{self.window_seconds}s exceeds threshold "
                f"{self.failures_to_open}. Pause + retry.",
                failures_in_window=len(self._failure_times),
                window_seconds=self.window_seconds,
            )


__all__ = [
    "SynthesisBudgetExceeded",
    "SynthesisCircuitOpen",
    "_BudgetTracker",
    "_CircuitBreaker",
]
