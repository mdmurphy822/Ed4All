"""Graceful-stop sentinel control for the "checkpoint on command" contract.

Every long-running Ed4All stage checks a filesystem sentinel at unit
boundaries (the same points where the fingerprinted resume sidecars append —
``lib/generation/llm_checkpoint.py``). When the sentinel exists the loop
finishes the in-flight unit, checkpoints it, then raises
``GracefulStopRequested`` so the runner can halt downstream work and stamp the
phase ``paused`` (never ``failed``). Worst-case loss is ONE in-flight LLM call.

Two sentinel scopes, both under the resolved ``state/runs`` parent:

- Run-scoped: ``<state_runs>/<run_id>/control/STOP_REQUESTED`` — stops one run.
- Global:    ``<state_runs>/STOP_ALL`` — stops every run (operator-owned; never
  auto-cleared, cleared only via ``clear_stop(include_global=True)``).

The file's EXISTENCE is the signal; its JSON body is diagnostic-only (reason,
source, pid, timestamp) and is never parsed for control flow.

Design invariants:

- **Stdlib-only, no env flag.** This module selects no provider/model, so it
  earns no ``docs/LICENSING.md`` row and no behavior-flags row. It is always
  on.
- **Path read per call, never CWD-relative.** The sentinel parent is resolved
  through ``lib.paths.get_state_runs_dir()`` on every call (never cached at
  import), so a servicer / bridge process launched from a non-repo-root CWD
  still sees the same sentinel the CLI wrote (risk R2 in the plan).
- **run_id resolution:** explicit ``run_id`` arg > ``os.environ["ED4ALL_RUN_ID"]``
  (exported by the pipeline orchestrator). A run-scoped operation with neither
  degrades to a no-op rather than guessing.
- **Best-effort filesystem discipline** (mirrors ``CheckpointStore.append`` at
  ``lib/generation/llm_checkpoint.py:247-251``): an ``OSError`` in
  request / clear / read is logged and degrades to a falsy result — stop
  control must NEVER crash a run.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from lib.paths import get_state_runs_dir

logger = logging.getLogger(__name__)

#: Basename of the run-scoped sentinel, under ``<run_id>/control/``.
RUN_SENTINEL_NAME = "STOP_REQUESTED"
#: Basename of the global sentinel, directly under the state/runs parent.
GLOBAL_SENTINEL_NAME = "STOP_ALL"


class GracefulStopRequested(RuntimeError):
    """Raised at a unit boundary once a stop sentinel is observed.

    Carries enough context for the runner to report an accurate resume hint:
    ``site_id`` (which loop stopped), ``units_completed`` (how many units were
    checkpointed before the stop), and ``sentinel_path`` (which sentinel
    tripped it, run-scoped or global). It subclasses ``RuntimeError`` so a
    generic ``except Exception`` still catches it, but the executor / runner
    catch it FIRST and map it to ``paused`` (never ``failed``, never retried).
    """

    def __init__(
        self,
        site_id: str,
        units_completed: int,
        sentinel_path: Optional[Path] = None,
    ) -> None:
        self.site_id = site_id
        self.units_completed = units_completed
        self.sentinel_path = sentinel_path
        super().__init__(
            f"graceful stop requested at {site_id!r} after "
            f"{units_completed} unit(s)"
            + (f" (sentinel: {sentinel_path})" if sentinel_path else "")
        )


class _StopMarker:
    """Type of the module-level :data:`STOP_MARKER` return-value singleton."""

    _repr = "<STOP_MARKER>"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self._repr


#: Return-value sentinel for coroutines gathered WITHOUT ``return_exceptions``.
#:
#: The batch-dispatch sites (course_planning / concept_extraction) call
#: ``asyncio.gather(*coros)`` with no ``return_exceptions=True``, and their
#: resume sidecar appends run ONLY in the post-gather zip loop. If a gathered
#: coroutine RAISED ``GracefulStopRequested`` mid-dispatch, gather would
#: propagate it before that append loop ran — losing up to ``batch_size - 1``
#: already-completed LLM calls. So a coroutine that observes a stop must
#: instead RETURN ``STOP_MARKER`` pre-dispatch (in-flight calls finish, new
#: ones are refused). After gather, the caller appends every real result to the
#: sidecar, treats ``STOP_MARKER`` entries as not-attempted, and only THEN
#: raises ``GracefulStopRequested``. ``STOP_MARKER`` is a dedicated object so it
#: is unambiguously distinguishable from ``None`` (the existing fail-soft
#: "unit degraded" return value) by identity (``result is STOP_MARKER``).
STOP_MARKER = _StopMarker()


# --------------------------------------------------------------------------- #
# Path resolution (read per call — never cached at import, never CWD-relative)
# --------------------------------------------------------------------------- #
def _resolve_run_id(run_id: Optional[str]) -> Optional[str]:
    """Explicit arg > ``ED4ALL_RUN_ID`` env; ``None`` when neither is set."""
    if run_id and run_id.strip():
        return run_id.strip()
    env = os.environ.get("ED4ALL_RUN_ID", "").strip()
    return env or None


def _global_sentinel_path() -> Path:
    return get_state_runs_dir() / GLOBAL_SENTINEL_NAME


def _run_sentinel_path(run_id: str) -> Path:
    return get_state_runs_dir() / run_id / "control" / RUN_SENTINEL_NAME


def _unlink_quiet(path: Path) -> None:
    """Remove ``path`` best-effort; a missing file or OSError never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("stop_control: failed to clear sentinel %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def request_stop(
    run_id: Optional[str] = None,
    scope: str = "run",
    reason: str = "",
    source: str = "",
) -> Optional[Path]:
    """Write a stop sentinel; return its path (``None`` on degrade).

    ``scope="run"`` writes the run-scoped sentinel (needs a resolvable
    ``run_id``); ``scope="all"`` writes the global ``STOP_ALL``. The small JSON
    body (reason / source / pid / timestamp) is diagnostic-only — the file's
    existence is the signal. A filesystem ``OSError`` is logged and degrades to
    ``None`` (best-effort; a lost sentinel write must not crash the caller).
    """
    if scope not in ("run", "all"):
        raise ValueError(f"stop_control.request_stop: unknown scope {scope!r}")
    if scope == "all":
        path = _global_sentinel_path()
    else:
        rid = _resolve_run_id(run_id)
        if not rid:
            logger.warning(
                "stop_control: run-scoped stop requested with no resolvable "
                "run_id (arg or ED4ALL_RUN_ID); no sentinel written."
            )
            return None
        path = _run_sentinel_path(rid)
    body = {
        "scope": scope,
        "run_id": _resolve_run_id(run_id) if scope == "run" else None,
        "reason": reason,
        "source": source,
        "pid": os.getpid(),
        "requested_at": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(body, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return path
    except OSError as exc:
        logger.warning("stop_control: failed to write sentinel %s: %s", path, exc)
        return None


def stop_requested(run_id: Optional[str] = None) -> bool:
    """True iff the global OR the resolved run-scoped sentinel exists.

    Global is checked first (it stops every run). Any ``OSError`` while probing
    degrades to ``False`` — a stat failure must not be read as "stop".
    """
    try:
        if _global_sentinel_path().exists():
            return True
        rid = _resolve_run_id(run_id)
        if rid and _run_sentinel_path(rid).exists():
            return True
        return False
    except OSError as exc:
        logger.warning("stop_control: failed to probe stop sentinel: %s", exc)
        return False


def clear_stop(run_id: Optional[str] = None, include_global: bool = False) -> None:
    """Remove the run-scoped sentinel (and the global one iff requested).

    Called at ``run_workflow`` start so a fresh/resumed run clears its own prior
    run-scoped sentinel. The global ``STOP_ALL`` is operator-owned and only
    removed when ``include_global=True`` (``ed4all stop --clear-all``).
    Best-effort throughout.
    """
    rid = _resolve_run_id(run_id)
    if rid:
        _unlink_quiet(_run_sentinel_path(rid))
    if include_global:
        _unlink_quiet(_global_sentinel_path())


def check_stop(
    site_id: str, units_completed: int, run_id: Optional[str] = None
) -> None:
    """Raise ``GracefulStopRequested`` at a unit boundary iff a stop is pending.

    A no-op when no sentinel exists — cheap enough to call at every sequential
    unit boundary. For tight inner loops use :class:`StopPoller` to bound the
    stat rate.
    """
    if not stop_requested(run_id):
        return
    rid = _resolve_run_id(run_id)
    sentinel: Optional[Path] = None
    try:
        if rid and _run_sentinel_path(rid).exists():
            sentinel = _run_sentinel_path(rid)
        elif _global_sentinel_path().exists():
            sentinel = _global_sentinel_path()
    except OSError:
        sentinel = None
    raise GracefulStopRequested(site_id, units_completed, sentinel)


class StopPoller:
    """Monotonic-time-throttled ``stop_requested()`` cache for tight loops.

    A per-unit ``stat`` is cheap but a per-inner-iteration one is not. This
    caches the last ``stop_requested()`` result and only re-probes once
    ``min_interval_s`` of ``time.monotonic()`` has elapsed. The cache is
    STICKY-FALSE only: an inner loop that checks thousands of times a second
    incurs at most one stat per interval, and a stop is observed within one
    interval of being requested.
    """

    def __init__(self, min_interval_s: float = 2.0) -> None:
        self.min_interval_s = float(min_interval_s)
        self._last_check: Optional[float] = None
        self._cached: bool = False

    def should_stop(self, run_id: Optional[str] = None) -> bool:
        """Cached ``stop_requested`` — re-probes at most once per interval."""
        now = time.monotonic()
        if (
            self._last_check is None
            or (now - self._last_check) >= self.min_interval_s
        ):
            self._cached = stop_requested(run_id)
            self._last_check = now
        return self._cached

    def check(
        self, site_id: str, units_completed: int, run_id: Optional[str] = None
    ) -> None:
        """Raise ``GracefulStopRequested`` iff the throttled probe trips."""
        if self.should_stop(run_id):
            check_stop(site_id, units_completed, run_id)
