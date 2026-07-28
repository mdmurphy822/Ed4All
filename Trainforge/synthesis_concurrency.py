"""Bounded, deterministic concurrency primitives for training synthesis.

The synthesis stage has two very different kinds of work:

* generation/validation work that may run concurrently and must not mutate
  output state; and
* ordered commits to JSONL artifacts, resume checkpoints, counters, and
  deduplication indexes that must have exactly one writer.

This module implements the first half of that contract.  Callers submit
immutable work units and consume :class:`OrderedResult` objects in input order.
At most ``max_concurrent`` futures exist at once, which supplies backpressure
without materialising an unbounded future list.  The caller remains the sole
writer and commits each result before requesting the next one.

The default resolver returns one.  Therefore importing this module or leaving
the behavior flag unset does not construct a thread pool and preserves the
legacy sequential path.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar


ENV_SYNTHESIS_MAX_CONCURRENT = "TRAINFORGE_SYNTHESIS_MAX_CONCURRENT"
DEFAULT_SYNTHESIS_MAX_CONCURRENT = 1
MAX_SYNTHESIS_MAX_CONCURRENT = 48

_T = TypeVar("_T")
_R = TypeVar("_R")


def resolve_synthesis_max_concurrent(explicit: Optional[int] = None) -> int:
    """Resolve the synthesis worker count.

    Precedence is explicit argument, then
    ``TRAINFORGE_SYNTHESIS_MAX_CONCURRENT``, then the legacy value ``1``.
    Missing, blank, non-integer, or non-positive values use the legacy value.
    Values above 48 fail loudly instead of being silently clamped to an
    unvalidated load level.
    """

    raw: object = explicit
    if raw is None:
        raw = os.environ.get(ENV_SYNTHESIS_MAX_CONCURRENT, "")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_SYNTHESIS_MAX_CONCURRENT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SYNTHESIS_MAX_CONCURRENT
    if value <= 0:
        return DEFAULT_SYNTHESIS_MAX_CONCURRENT
    if value > MAX_SYNTHESIS_MAX_CONCURRENT:
        raise ValueError(
            f"{ENV_SYNTHESIS_MAX_CONCURRENT}={value} exceeds the validated "
            f"hard ceiling {MAX_SYNTHESIS_MAX_CONCURRENT}"
        )
    return value


@dataclass(frozen=True)
class OrderedResult(Generic[_T, _R]):
    """One completed work unit restored to deterministic input order."""

    sequence: int
    item: _T
    value: _R


class BoundedOrderedMap(Generic[_T, _R]):
    """Map immutable work concurrently and yield results in input order.

    ``stop_requested`` is checked at unit-submission boundaries.  Once true,
    no more units are submitted; already-submitted units are drained and
    yielded so the caller can commit/checkpoint them.  ``stopped_early`` then
    tells the caller to raise its normal graceful-stop exception after the
    drain.  This avoids both abandoned successful calls and duplicate calls on
    resume.
    """

    def __init__(
        self,
        items: Iterable[_T],
        worker: Callable[[_T], _R],
        *,
        max_concurrent: int,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._items = iter(items)
        self._worker = worker
        self.max_concurrent = resolve_synthesis_max_concurrent(max_concurrent)
        self._stop_requested = stop_requested or (lambda: False)
        self.stopped_early = False
        self.peak_in_flight = 0
        self.submitted_count = 0
        self.yielded_count = 0
        self._metrics_lock = threading.Lock()
        self._active_count = 0
        self._started_count = 0

    @property
    def active_count(self) -> int:
        with self._metrics_lock:
            return self._active_count

    @property
    def queue_depth(self) -> int:
        """Compatibility alias for waiting executor units."""
        return self.queued_units

    @property
    def active_workers(self) -> int:
        return self.active_count

    @property
    def queued_units(self) -> int:
        with self._metrics_lock:
            return max(self.submitted_count - self._started_count, 0)

    @property
    def in_flight(self) -> int:
        return self.active_workers + self.queued_units

    def metrics_snapshot(self) -> tuple[int, int, int]:
        """Return relationally consistent active, queued, and in-flight."""
        with self._metrics_lock:
            active = self._active_count
            queued = max(self.submitted_count - self._started_count, 0)
            return active, queued, active + queued

    def _run_worker(self, item: _T) -> _R:
        with self._metrics_lock:
            self._started_count += 1
            self._active_count += 1
        try:
            return self._worker(item)
        finally:
            with self._metrics_lock:
                self._active_count -= 1

    def __iter__(self) -> Iterator[OrderedResult[_T, _R]]:
        # The exact legacy path: no executor, no lookahead, and a stop check
        # before every unit.
        if self.max_concurrent == 1:
            for sequence, item in enumerate(self._items):
                if self._stop_requested():
                    self.stopped_early = True
                    return
                with self._metrics_lock:
                    self.submitted_count += 1
                value = self._run_worker(item)
                self.yielded_count += 1
                yield OrderedResult(sequence, item, value)
            return

        executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent,
            thread_name_prefix="trainforge-synthesis",
        )
        in_flight: dict[int, tuple[_T, Future[_R]]] = {}
        next_submit = 0
        next_yield = 0
        source_exhausted = False

        def fill_window() -> None:
            nonlocal next_submit, source_exhausted
            while (
                not source_exhausted
                and len(in_flight) < self.max_concurrent
            ):
                if self._stop_requested():
                    self.stopped_early = True
                    source_exhausted = True
                    return
                try:
                    item = next(self._items)
                except StopIteration:
                    source_exhausted = True
                    return
                with self._metrics_lock:
                    self.submitted_count += 1
                in_flight[next_submit] = (
                    item, executor.submit(self._run_worker, item),
                )
                next_submit += 1
                self.peak_in_flight = max(
                    self.peak_in_flight, len(in_flight)
                )

        try:
            fill_window()
            while in_flight:
                # Waiting on the lowest sequence is deliberate: generation may
                # finish out of order, but the sole caller/writer observes and
                # commits in deterministic source order.
                item, future = in_flight.pop(next_yield)
                value = future.result()  # fail loudly; no fallback
                self.yielded_count += 1
                yield OrderedResult(next_yield, item, value)
                next_yield += 1
                fill_window()
        except BaseException:
            # Pending work has no externally-visible disposition and is safe to
            # cancel. Running calls cannot be force-cancelled; wait for them so
            # the shared HTTP client/capture is never closed underneath a call.
            for _, future in in_flight.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)


__all__ = [
    "BoundedOrderedMap",
    "DEFAULT_SYNTHESIS_MAX_CONCURRENT",
    "ENV_SYNTHESIS_MAX_CONCURRENT",
    "MAX_SYNTHESIS_MAX_CONCURRENT",
    "OrderedResult",
    "resolve_synthesis_max_concurrent",
]
