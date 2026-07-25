"""Process-liveness + honest run-state derivation for the control-plane GUI.

The Runs listing and the progress endpoint both persist a run's LAST-WRITTEN
status (``RUNNING`` etc.), but the orchestrator has no reaper that stamps a
crashed / stopped CLI run terminal. So a ``state/workflows/WF-*.json`` still
saying ``RUNNING`` renders "Building" forever even when the ``ed4all run``
process is gone, draining, or wedged (the owner hit exactly this: an
``ed4all stop``-ed run still showing "Building"). This module derives an honest
``effective_status`` from real signals — and NEVER writes pipeline state:

  * a ``/proc`` scan for live ``ed4all run`` processes — matched on ADJACENT
    ``ed4all`` / ``run`` argv tokens (never ``pgrep -f``, which self-matches the
    wrapper/monitor shell whose whole script is one argv token); precedent scan
    is the operator campaign monitor's ``scan_pipeline_pids``. A run is
    ATTRIBUTED to such a process when any of its identifier tokens (corpus /
    project / course name / orchestrator run id / workflow ``WF-<id>``) is a
    substring of that process's argv — the workflow id is what lets a bare
    ``--resume WF-<id>`` process self-attribute,
  * the graceful-stop sentinel
    (``state/runs/<orch_run_id>/control/STOP_REQUESTED``),
  * the workflow-state file mtime (staleness fallback).

Read-only + stdlib-only + import-light (no fastapi), so it can be imported by
both ``run_service`` and ``progress_service`` without a web dependency and
without a circular import.

``effective_status`` vocabulary (all lowercase):

  * ``paused``    — resumable review-flow halt (status PAUSED). NOT building.
  * ``stopping``  — a stop sentinel is present; the run is draining to a
                    checkpoint. NOT building.
  * ``incomplete`` — a CLI run whose ``ed4all run`` process is gone without
                    reaching completion; the RUNNING record is stale — the build
                    did not make it through final completion.
  * ``stalled?``  — CLI run: live ``ed4all run`` process(es) exist but none is
                    attributable to THIS run AND its workflow file is > 10 min
                    stale. Ambiguous, flagged for the operator.
  * ``building``  — genuinely live (attributed process, or a GUI in-process
                    run, or a fresh unattributable one).
  * terminal / other statuses pass through unchanged (lowercased).

Only CLI-launched runs are backed by an external ``ed4all run`` process; a
GUI-launched run is driven in-process by the GUI server's own asyncio task
(``run_service._BACKGROUND_TASKS``) and NEVER spawns such a process. So the
``/proc``-liveness derivation (``incomplete`` / ``stalled?``) is applied ONLY to
CLI runs — a GUI run's non-terminal status is left unchanged (bar the two
signals that apply regardless: PAUSED and the stop sentinel).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Workflow-state mtime age (seconds) beyond which an unattributable-but-live
# CLI RUNNING record is flagged ``stalled?`` rather than ``building``.
STALE_MTIME_SECONDS = 600.0

# Truly-final statuses (paused handled explicitly — it is resumable, not final).
_FINAL_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "cancelled",
        "canceled",
        "interrupted",
        "timeout",
        "error",
        "incomplete",
    }
)


def _normalize_status(status: Any) -> str:
    """Lowercase + fold ``complete`` → ``completed`` (WorkflowRunner vocab)."""
    st = str(status or "").strip().lower()
    return "completed" if st == "complete" else st


def scan_pipeline_processes() -> List[Tuple[int, List[str]]]:
    """Return ``[(pid, argv), ...]`` for every live ``ed4all run`` process.

    A process qualifies only when its ``/proc/<pid>/cmdline`` carries an exact
    ``ed4all`` (or ``.../ed4all``) argv token IMMEDIATELY followed by a ``run``
    token. Token-adjacency (never a ``pgrep -f`` substring match) is what stops
    a ``bash -c '... ed4all run ...'`` wrapper / monitor shell — whose whole
    script is a single argv token — from false-positiving. Never raises: an
    unreadable ``/proc`` or a vanished pid mid-scan is skipped.
    """
    procs: List[Tuple[int, List[str]]] = []
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return procs
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue  # vanished / permission — skip
        argv = [tok.decode("utf-8", "replace") for tok in raw.split(b"\0") if tok]
        for i in range(len(argv) - 1):
            tok = argv[i]
            if (tok == "ed4all" or tok.endswith("/ed4all")) and argv[i + 1] == "run":
                procs.append((pid, argv))
                break
    procs.sort(key=lambda pa: pa[0])
    return procs


def _resolve_state_root(state_root: Optional[Path]) -> Path:
    """Resolve the ``state/`` root (honors the test ``ED4ALL_STATE_RUNS_DIR``)."""
    if state_root is not None:
        return Path(state_root)
    from gui import shared_state  # noqa: PLC0415 — lazy, avoids web deps

    return shared_state._state_root()


def stop_sentinel_present(
    orch_run_id: Optional[str], *, state_root: Optional[Path] = None
) -> bool:
    """True when ``state/runs/<orch_run_id>/control/STOP_REQUESTED`` exists.

    The per-run graceful-stop sentinel the pipeline drops on ``ed4all stop``.
    Read-only; a missing / unreadable path is simply ``False``.
    """
    if not (isinstance(orch_run_id, str) and orch_run_id.strip()):
        return False
    try:
        path = (
            _resolve_state_root(state_root)
            / "runs"
            / orch_run_id.strip()
            / "control"
            / "STOP_REQUESTED"
        )
        return path.exists()
    except OSError:
        return False


def attribution_tokens_from_params(
    params: Any,
    orch_run_id: Optional[str] = None,
    *,
    wf_id: Optional[str] = None,
) -> List[str]:
    """Collect argv-matchable identifiers for a run from its workflow params.

    The corpus path appearing in a live ``ed4all run`` argv is a decent
    attribution signal (the spec's preferred one); the orchestrator run id,
    course name and project path are additional corroborating tokens. The
    workflow id (``WF-<id>``) is included when supplied — a ``--resume
    WF-<id>`` process carries ONLY that token in its argv (the corpus / project
    / course name are loaded from persisted state, not re-passed on the
    command line), so without it a resuming build cannot self-attribute and
    false-positives as ``stalled?``. Returns a de-duplicated, order-preserving
    list of non-empty strings; never raises.
    """
    tokens: List[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            tokens.append(value.strip())
        elif isinstance(value, (list, tuple)):
            for item in value:
                _add(item)

    if isinstance(params, dict):
        for key in ("corpus", "pdf_paths", "project_path", "course_name"):
            _add(params.get(key))
    _add(orch_run_id)
    _add(wf_id)

    seen: set = set()
    out: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _attributed(
    processes: Sequence[Tuple[int, List[str]]], tokens: Iterable[str]
) -> bool:
    """True when any attribution token appears in any live process's argv."""
    token_list = [t for t in tokens if t]
    if not token_list:
        return False
    for _pid, argv in processes:
        for arg in argv:
            for tok in token_list:
                if tok in arg:
                    return True
    return False


def effective_status(
    status: Any,
    *,
    is_cli: bool,
    orch_run_id: Optional[str] = None,
    attribution_tokens: Iterable[str] = (),
    processes: Optional[Sequence[Tuple[int, List[str]]]] = None,
    wf_mtime: Optional[float] = None,
    now: Optional[float] = None,
    state_root: Optional[Path] = None,
) -> str:
    """Derive an honest ``effective_status`` (see the module docstring).

    ``status`` is the persisted run/workflow status (upper- or lowercase). The
    raw ``status`` field is never mutated — callers add this as a SEPARATE
    ``effective_status`` field. ``processes`` is the ONE-per-listing scan result
    (pass ``scan_pipeline_processes()``); ``None`` triggers a lazy scan.
    """
    st = _normalize_status(status)
    if st in _FINAL_STATUSES:
        return st
    if st == "paused":
        return "paused"

    # A stop sentinel means the run is draining to a checkpoint — honest for
    # both a CLI run and a GUI-driven run whose phases poll the same sentinel.
    if stop_sentinel_present(orch_run_id, state_root=state_root):
        return "stopping"

    # GUI-launched runs have no external process; their in-process driver
    # liveness is out of /proc's reach. Leave the non-terminal status unchanged
    # (byte-identical to the pre-change behavior for GUI records).
    if not is_cli:
        return st

    if processes is None:
        processes = scan_pipeline_processes()

    if not processes:
        # No ``ed4all run`` process anywhere — the RUNNING record is stale.
        return "incomplete"

    if _attributed(processes, attribution_tokens):
        return "building"

    # Live process(es) exist but none is attributable to this run. Prefer the
    # mtime staleness fallback: a genuinely-advancing run rewrites its workflow
    # file at every phase boundary.
    if now is None:
        now = time.time()
    if wf_mtime is not None and (now - wf_mtime) > STALE_MTIME_SECONDS:
        return "stalled?"
    return "building"


__all__ = [
    "STALE_MTIME_SECONDS",
    "attribution_tokens_from_params",
    "effective_status",
    "scan_pipeline_processes",
    "stop_sentinel_present",
]
