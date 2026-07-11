"""SemantiK-local cooperative graceful-stop seam (cross-venv twin).

The Ed4All orchestrator implements a "checkpoint on command" contract: an
``ed4all stop`` writes a filesystem STOP sentinel and every long-running stage
polls it at a unit boundary, finishing the in-flight unit and then halting so
the run can be resumed later (worst-case loss = one in-flight unit).

The SemantiK cascade runs in its OWN venv (or the fully out-of-process JSON
bridge) and therefore **must not import Ed4All's** ``lib/`` — including
``lib/generation/stop_control.py``. So, exactly like
``semantik_structure.gpu_lifecycle`` is a self-contained twin of Ed4All's
GPU-lifecycle lease (see ``cascade.py::_gpu_lifecycle_release``), this module is
a self-contained twin of the stop-sentinel poll: it learns which sentinel
file(s) to watch from the ``SEMANTIK_STOP_SENTINEL`` environment variable that
the Ed4All side sets before invoking the cascade (declared-env plumbing — it
carries a filesystem PATH handed in from outside, selects no provider/model,
and is a byte-identical no-op when unset, so it is NOT a behavior flag and
earns no ``docs/LICENSING.md`` row).

Contract with the Ed4All side (owned by another agent — see the module's
docstring at ``lib/generation/stop_control.py``):

- Ed4All resolves the absolute sentinel path(s) for the active run — the
  run-scoped ``<state_runs>/<run_id>/control/STOP_REQUESTED`` and the global
  ``<state_runs>/STOP_ALL`` — and exports them, ``os.pathsep``-joined, as
  ``SEMANTIK_STOP_SENTINEL`` in BOTH the in-process cascade path and the bridge
  subprocess env (mirroring how ``SEMANTIK_PYTHON`` / ``SEMANTIK_RUNTIME_DIR``
  already cross the bridge).
- A cascade seam that observes any listed path EXISTING raises
  :class:`CascadeStopRequested`. That exception propagates out of
  ``run_full_cascade``; the Ed4All-side bridge boundary catches it and
  translates it to ``lib.generation.stop_control.GracefulStopRequested`` (the
  translation is another agent's job — this module never imports Ed4All).

Design invariants (mirroring the GPU-lifecycle twin + ``stop_control``):

- **No import cached at module load; read per call.** The env var is read on
  every poll so a sentinel written after the cascade started is still seen, and
  so a process launched from a non-repo-root CWD watches exactly the path the
  CLI wrote (the seam never resolves the path itself — it is handed in whole).
- **Fail-soft on the filesystem probe.** An ``OSError`` while stat-ing a
  sentinel degrades to "no stop" — a stat failure must never be read as a stop,
  and the seam must never crash the cascade on a probe error.
- **No-op when unset.** ``SEMANTIK_STOP_SENTINEL`` unset / blank ⇒ zero paths ⇒
  every seam returns immediately, byte-identical to a cascade with no stop
  wiring at all (backward compatible).

Note the ONE deliberate difference from the fail-soft GPU-lifecycle twin: this
seam is *allowed to raise* — :class:`CascadeStopRequested` is the control
signal, not an error. "Fail-soft" here means the sentinel PROBE never raises;
observing a real sentinel raises the stop exception on purpose.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

#: Env var carrying the ``os.pathsep``-joined absolute sentinel path(s) the
#: Ed4All side hands in. Declared-env plumbing (a PATH), NOT a behavior flag.
STOP_SENTINEL_ENV = "SEMANTIK_STOP_SENTINEL"


class CascadeStopRequested(RuntimeError):
    """Raised at a cascade seam once a handed-in stop sentinel is observed.

    Carries ``site_id`` (which seam tripped) and ``sentinel_path`` (which
    sentinel file existed) for an accurate resume hint. It subclasses
    ``RuntimeError`` so a generic ``except Exception`` still catches it, but the
    Ed4All bridge boundary catches it FIRST and maps it to
    ``GracefulStopRequested`` (paused, never failed, never retried).
    """

    def __init__(self, site_id: str, sentinel_path: Optional[Path] = None) -> None:
        self.site_id = site_id
        self.sentinel_path = sentinel_path
        super().__init__(
            f"cascade stop requested at {site_id!r}"
            + (f" (sentinel: {sentinel_path})" if sentinel_path else "")
        )


def resolve_stop_sentinel_paths() -> list[Path]:
    """Parse ``SEMANTIK_STOP_SENTINEL`` into a list of paths (read per call).

    Empty / unset ⇒ ``[]`` (no wiring → every seam is a no-op). Splits on
    ``os.pathsep`` so both the run-scoped and global sentinels can ride one env
    var; blank segments are dropped.
    """
    raw = os.environ.get(STOP_SENTINEL_ENV, "").strip()
    if not raw:
        return []
    return [Path(seg) for seg in raw.split(os.pathsep) if seg.strip()]


def stop_sentinel_present(paths: Optional[Sequence[Path]] = None) -> bool:
    """True iff any handed-in sentinel path currently exists (fail-soft).

    ``paths=None`` resolves from the env per call. Any ``OSError`` while probing
    a single path is swallowed (that path is treated as absent) so a stat
    failure is never misread as a stop.
    """
    paths = resolve_stop_sentinel_paths() if paths is None else list(paths)
    for path in paths:
        try:
            if path.exists():
                return True
        except OSError as exc:  # noqa: PERF203 — probe must never crash a run
            logger.warning("stop_seam: failed to probe sentinel %s: %s", path, exc)
    return False


def check_cascade_stop(site_id: str, paths: Optional[Sequence[Path]] = None) -> None:
    """Raise :class:`CascadeStopRequested` iff a handed-in sentinel exists.

    A no-op when ``SEMANTIK_STOP_SENTINEL`` is unset (zero paths) — cheap enough
    to call at a coarse seam boundary. For a tight loop that would stat many
    times per second, use :class:`StopPoller` to bound the probe rate.
    """
    paths = resolve_stop_sentinel_paths() if paths is None else list(paths)
    if not paths:
        return
    for path in paths:
        try:
            exists = path.exists()
        except OSError as exc:
            logger.warning("stop_seam: failed to probe sentinel %s: %s", path, exc)
            continue
        if exists:
            raise CascadeStopRequested(site_id, path)


class StopPoller:
    """Monotonic-time-throttled sentinel poll for a between-batch seam.

    A single ``stat`` per adapter batch is cheap, but a poll on every inner
    iteration is not. This caches the last probe and only re-stats once
    ``min_interval_s`` of ``time.monotonic()`` has elapsed. STICKY-FALSE only:
    once a stop is seen the cache stays true, and a stop is observed within one
    interval of being written. Reimplemented locally (no Ed4All import) but
    mirrors ``lib.generation.stop_control.StopPoller``.
    """

    def __init__(self, min_interval_s: float = 2.0) -> None:
        self.min_interval_s = float(min_interval_s)
        self._paths: Optional[list[Path]] = None
        self._last_check: Optional[float] = None
        self._cached: bool = False

    def _resolve_paths(self) -> list[Path]:
        # Resolve once and cache — the env var is stable for a cascade run, and
        # the seam only needs the sentinel PATHS fixed, not their existence.
        if self._paths is None:
            self._paths = resolve_stop_sentinel_paths()
        return self._paths

    def should_stop(self) -> bool:
        """Throttled ``stop_sentinel_present`` — re-stats at most once/interval."""
        paths = self._resolve_paths()
        if not paths:
            return False
        now = time.monotonic()
        if self._last_check is None or (now - self._last_check) >= self.min_interval_s:
            self._cached = stop_sentinel_present(paths)
            self._last_check = now
        return self._cached

    def check(self, site_id: str) -> None:
        """Raise :class:`CascadeStopRequested` iff the throttled probe trips."""
        if self.should_stop():
            check_cascade_stop(site_id, self._resolve_paths())
