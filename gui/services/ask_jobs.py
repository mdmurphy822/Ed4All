"""Server-side ask-job store + single-lane runner — durable async answers.

The in-context Studio "Ask drawer" (C2) lets a learner ask a question WHILE
READING. A grounded answer over a local LLM takes 35-50s, so the request must
NOT block the content pane and MUST survive a tab refresh / uvicorn restart:
the answer is computed in a background worker and the result is persisted to
disk the moment it lands, so a poll after a reconnect still finds it.

Design (mirrors the ``gui.shared_state`` run-registry idiom):

* One job = one ``state/gui/ask_jobs/<ask_id>.json`` file, written atomically
  (tmpfile + ``os.replace``) so a reader never sees a partial record. The job
  record is the single source of truth — there is NO in-memory result cache, so
  a fresh process (uvicorn restart) serves a finished answer straight off disk.
* ``submit()`` writes a ``pending`` record and enqueues the job on a module-level
  ``queue.Queue``; a single daemon worker thread drains it (bounded concurrency
  **1** — the local LLM is single-lane). The worker calls the SAME
  ``answer_service.ask`` path the synchronous endpoint uses, then persists a
  ``done`` (or ``error``) record. Queue position is derived from the on-disk
  ``pending`` set (queue-time order), so it survives a restart too.
* ``sweep()`` opportunistically deletes job files older than the TTL (24h by
  default) so the store doesn't grow without bound. Called best-effort on each
  ``submit`` / ``list`` — no background timer needed.

The synchronous ``POST /api/learn/ask`` is untouched; the learner page keeps
using it. This module is the async arm only. Heavy imports stay lazy / behind
the answer-service seam so this module imports without the retrieval stack
(tests stub ``answer_service.ask`` exactly as the sync-path tests do).
"""

from __future__ import annotations

import inspect
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gui import shared_state

# Job lifecycle states (the on-disk + API ``status`` field).
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

_TERMINAL = frozenset({STATUS_DONE, STATUS_ERROR})

# Default time-to-live for a finished/abandoned job file (seconds).
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Single-lane worker primitives (module-level so they outlive a request but
# share the FastAPI process). Lazily started on first ``submit`` so importing
# this module never spins a thread (test collection, MCP tool imports).
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER: Optional[threading.Thread] = None
_WORKER_LOCK = threading.Lock()

# Test seam: the worker calls this to compute an answer. Defaults to the real
# service; tests monkeypatch it (or ``gui.services.answer_service.ask``) to a
# stub so no model is ever loaded. Signature mirrors ``answer_service.ask``:
# ``(slug, query, engine, *, library_wide=None, on_progress=None)``. The extra
# keyword args are passed only when the resolved callable actually accepts them
# (introspection), so a legacy 3-arg stub stays byte-identical.
_answer_fn: Optional[Callable[..., Dict[str, Any]]] = None


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def ask_jobs_dir() -> Path:
    """Return ``state/gui/ask_jobs/`` (created lazily)."""
    path = shared_state.gui_state_dir() / "ask_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(ask_id: str) -> Path:
    shared_state._validate_id(ask_id, "ask_id")  # path-separator guard
    return ask_jobs_dir() / f"{ask_id}.json"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _write_job(record: Dict[str, Any]) -> None:
    """Atomically persist a job record (keyed by ``ask_id``)."""
    record["updated_at"] = shared_state.now_iso()
    shared_state._atomic_write_json(_job_path(record["ask_id"]), record)


def read_job(ask_id: str) -> Optional[Dict[str, Any]]:
    """Return the job record for ``ask_id`` or ``None`` if unknown/corrupt."""
    path = _job_path(ask_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs() -> List[Dict[str, Any]]:
    """Return all job records, newest ``created_at`` first."""
    out: List[Dict[str, Any]] = []
    directory = ask_jobs_dir()
    for path in directory.iterdir():
        if not (path.is_file() and path.suffix == ".json" and not path.name.startswith(".")):
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Queue position (durable — derived from on-disk pending set)
# --------------------------------------------------------------------------- #


def _pending_order() -> List[str]:
    """Pending/running job ids in queue order (oldest ``created_at`` first).

    Queue position is derived from disk, not the in-memory ``queue.Queue``, so
    it stays correct across a process restart (the worker re-enqueues nothing,
    but a reloaded process can still report where a pending job sits relative to
    its siblings, and the recovery sweep re-arms abandoned pendings).
    """
    pend = [j for j in list_jobs() if j.get("status") in (STATUS_PENDING, STATUS_RUNNING)]
    pend.sort(key=lambda r: str(r.get("created_at", "")))
    return [j["ask_id"] for j in pend if j.get("ask_id")]


def _queue_position(ask_id: str) -> int:
    """0-based position of ``ask_id`` among pending/running jobs (0 == next/now).

    A ``running`` job reports 0. ``-1`` when the job isn't pending (done/error/
    unknown) so the caller can omit the field.
    """
    order = _pending_order()
    try:
        return order.index(ask_id)
    except ValueError:
        return -1


# --------------------------------------------------------------------------- #
# Worker (single lane, bounded concurrency 1)
# --------------------------------------------------------------------------- #


def _resolve_answer_fn() -> Callable[..., Dict[str, Any]]:
    """Resolve the answer callable (test seam → real service)."""
    if _answer_fn is not None:
        return _answer_fn
    from gui.services import answer_service  # noqa: PLC0415

    return answer_service.ask


def _call_answer(
    fn: Callable[..., Dict[str, Any]],
    record: Dict[str, Any],
    on_progress: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Invoke the answer callable, threading the extra kwargs it actually accepts.

    The real ``answer_service.ask`` takes keyword-only ``library_wide`` /
    ``on_progress``; a legacy 3-positional-arg test stub takes neither. We inspect
    the callable's signature and pass each extra kwarg ONLY when the parameter (or
    a ``**kwargs`` catch-all) is present, so a 3-arg stub is called byte-identically
    to before.
    """
    slug = record["slug"]
    query = record["query"]
    engine = record.get("engine", "auto")
    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(fn).parameters
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if "on_progress" in params or has_var_kw:
            kwargs["on_progress"] = on_progress
        if "library_wide" in params or has_var_kw:
            kwargs["library_wide"] = record.get("library_wide")
    except (TypeError, ValueError):
        kwargs = {}
    return fn(slug, query, engine, **kwargs)


def _process(ask_id: str) -> None:
    """Compute + persist one job's answer. Never raises (errors → ``error`` record)."""
    record = read_job(ask_id)
    if record is None or record.get("status") in _TERMINAL:
        return  # already done, swept, or cancelled
    record["status"] = STATUS_RUNNING
    record["started_at"] = shared_state.now_iso()
    _write_job(record)

    def _on_progress(payload: Dict[str, Any]) -> None:
        """L4: persist retrieved passages onto the RUNNING record (one extra write).

        Fired once from the answer pipeline after the pre-LLM refusal gate. A poll
        during the 35-50s compose window then finds the passages on the running
        job and can disclose them "passages-first" before the answer lands. Best-
        effort: a write failure never breaks the answer in flight.
        """
        try:
            if not isinstance(payload, dict):
                return
            passages = payload.get("passages")
            if passages:
                record["passages"] = passages
                record["passages_refused"] = bool(payload.get("refused"))
                _write_job(record)
        except Exception:  # noqa: BLE001 — disclosure persistence is advisory
            pass

    started = time.monotonic()
    try:
        answer = _call_answer(_resolve_answer_fn(), record, _on_progress)
        record["status"] = STATUS_DONE
        record["answer"] = answer
    except Exception as exc:  # noqa: BLE001 — map typed pipeline errors for the poller
        record["status"] = STATUS_ERROR
        record["error"] = type(exc).__name__
        record["detail"] = str(exc)
    record["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    record["finished_at"] = shared_state.now_iso()
    _write_job(record)


def _worker_loop() -> None:
    """Drain the queue forever, one job at a time (single lane)."""
    while True:
        ask_id = _QUEUE.get()
        try:
            _process(ask_id)
        except Exception:  # noqa: BLE001 — a poison job must not kill the lane
            pass
        finally:
            _QUEUE.task_done()


def _ensure_worker() -> None:
    """Start the single daemon worker thread once (idempotent)."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(
            target=_worker_loop, name="ask-jobs-worker", daemon=True
        )
        _WORKER.start()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def submit(
    slug: str,
    query: str,
    engine: str = "auto",
    library_wide: Optional[bool] = None,
) -> Dict[str, Any]:
    """Enqueue an ask job; return its ``pending`` record (with queue position).

    Writes the ``pending`` record to disk BEFORE enqueueing so a poll that races
    the worker always finds the job. Opportunistically sweeps expired jobs.

    ``library_wide`` (L3) is the request-level "search all courses" override; it
    is persisted on the record and threaded to the answer service by the worker
    (``None`` => resolve from ``ED4ALL_ANSWER_LIBRARY_WIDE``).
    """
    sweep()  # opportunistic cleanup (best-effort; never blocks a submit)
    ask_id = shared_state.new_run_id(prefix="ASK")
    record = {
        "ask_id": ask_id,
        "slug": slug,
        "query": query,
        "engine": engine or "auto",
        "status": STATUS_PENDING,
        "created_at": shared_state.now_iso(),
    }
    if library_wide is not None:
        record["library_wide"] = bool(library_wide)
    _write_job(record)
    _ensure_worker()
    _QUEUE.put(ask_id)
    record["queue_position"] = _queue_position(ask_id)
    return record


def status(ask_id: str) -> Optional[Dict[str, Any]]:
    """Return the poll view for ``ask_id`` (queue position injected), or ``None``.

    The poll shape is the on-disk record plus a live ``queue_position`` for
    pending/running jobs. ``None`` for an unknown id (the router 404s).
    """
    record = read_job(ask_id)
    if record is None:
        return None
    if record.get("status") in (STATUS_PENDING, STATUS_RUNNING):
        record["queue_position"] = _queue_position(ask_id)
    return record


def sweep(ttl_seconds: int = DEFAULT_TTL_SECONDS, now: Optional[float] = None) -> int:
    """Delete job files older than ``ttl_seconds`` (by mtime). Returns the count.

    Opportunistic + best-effort: an unlink failure is swallowed so a locked /
    racing file never breaks a submit. ``now`` is injectable for tests.
    """
    if ttl_seconds <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - ttl_seconds
    removed = 0
    directory = ask_jobs_dir()
    for path in directory.iterdir():
        if not (path.is_file() and path.suffix == ".json"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


__all__ = [
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "STATUS_ERROR",
    "DEFAULT_TTL_SECONDS",
    "ask_jobs_dir",
    "submit",
    "status",
    "read_job",
    "list_jobs",
    "sweep",
]
