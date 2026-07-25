"""Shared ``state/gui/`` store — single source of truth both the GUI backend
AND MCP tools read/write, so a Claude Code session and the GUI stay in sync.

Mirrors the ``StatusTracker`` / ``TaskMailbox`` file-IPC idiom: atomic writes
via tmpfile + ``os.replace``, append-only event log, JSON run registry. No web
deps are imported here — this module is import-safe from MCP tool code.

Layout::

    state/gui/
    ├── settings.json        # canonical settings doc (see settings_store)
    ├── .env.rendered        # rendered env file (informational)
    ├── runs/                # run registry: <run_id>.json
    ├── logs/                # <run_id>.log (appended; tailed by ws)
    ├── uploads/             # uploaded corpora (one dir per upload_id)
    └── events.jsonl         # bidirectional Claude<->GUI event log (append-only)

The state root honors ``ED4ALL_STATE_RUNS_DIR`` (so tests can redirect into a
``tmp_path``); otherwise it resolves to ``lib.paths.STATE_PATH``. Both are read
at call time, not import time, so tests can monkeypatch the env var.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Permitted ``source`` values for events (the Claude<->GUI bridge, plus the
# Studio Assistant panel — one ``assistant``-sourced event per chat exchange).
_VALID_EVENT_SOURCES = ("gui", "claude", "assistant")


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (matches the settings doc ``updated_at`` form)."""
    return datetime.now(timezone.utc).isoformat()


def _state_root() -> Path:
    """Resolve the ``state/`` root.

    Priority:
    1. ``ED4ALL_STATE_RUNS_DIR`` env var — used by tests to redirect into a
       throwaway ``tmp_path`` (mirrors ``TaskMailbox`` / ``get_state_runs_dir``).
       The override names the ``runs`` parent, so the GUI store lands as a
       sibling ``gui/`` next to it.
    2. ``lib.paths.STATE_PATH`` — the canonical project ``state/`` directory.
    """
    env_override = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_override:
        # ED4ALL_STATE_RUNS_DIR points at ``<root>/runs``; the GUI store is a
        # sibling of that runs dir under the same state root.
        return Path(env_override).parent
    # Import lazily so a missing lib.paths (unlikely) doesn't break import.
    from lib.paths import STATE_PATH  # noqa: PLC0415

    return Path(STATE_PATH)


def gui_state_dir() -> Path:
    """Return ``state/gui/`` (created lazily)."""
    path = _state_root() / "gui"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    """Return ``state/gui/settings.json`` (parent created lazily)."""
    return gui_state_dir() / "settings.json"


def env_rendered_path() -> Path:
    """Return ``state/gui/.env.rendered`` (parent created lazily)."""
    return gui_state_dir() / ".env.rendered"


def runs_dir() -> Path:
    """Return ``state/gui/runs/`` (created lazily)."""
    path = gui_state_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    """Return ``state/gui/logs/`` (created lazily)."""
    path = gui_state_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    """Return ``state/gui/uploads/`` (created lazily)."""
    path = gui_state_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def events_path() -> Path:
    """Return ``state/gui/events.jsonl`` (parent created lazily)."""
    return gui_state_dir() / "events.jsonl"


def log_path(run_id: str) -> Path:
    """Return ``state/gui/logs/<run_id>.log`` for a run (parent created lazily)."""
    _validate_id(run_id, "run_id")
    return logs_dir() / f"{run_id}.log"


# ---------------------------------------------------------------- atomic write


def _atomic_write_text(target: Path, text: str, mode: int | None = None) -> None:
    """Write ``text`` to ``target`` via tmpfile + ``os.replace``.

    Matches ``task_mailbox.py``: readers never observe a partial file, and the
    rename is atomic within a filesystem. The temp file is cleaned up on error.

    These are all local control-plane state files and some (``settings.json``,
    ``.env.rendered``) carry plaintext API keys, so the final file is chmod'd
    owner-only (0600) by default. The chmod is applied to the *temp* file before
    the atomic ``os.replace`` so the secret bytes are never world-readable even
    transiently. Callers may pass an explicit ``mode`` to override.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    effective_mode = 0o600 if mode is None else mode
    try:
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, effective_mode)
        os.replace(tmp, target)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _atomic_write_json(target: Path, payload: Any, mode: int | None = None) -> None:
    """JSON-serialize ``payload`` and atomically write it to ``target``."""
    _atomic_write_text(target, json.dumps(payload, indent=2, default=str), mode=mode)


# ----------------------------------------------------------------- event log


def append_event(source: str, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append an event to ``events.jsonl`` and return the stored record.

    Append-only; this is the bidirectional Claude<->GUI bridge. ``source`` must
    be one of ``{"gui", "claude"}``. The returned record carries a monotonically
    increasing ``seq`` (the 0-based line index it was written at) plus an ``ts``.
    """
    if source not in _VALID_EVENT_SOURCES:
        raise ValueError(
            f"source must be one of {_VALID_EVENT_SOURCES!r}, got {source!r}"
        )
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    path = events_path()
    # The seq is the count of existing lines, so consumers can poll with
    # ``since`` semantics without parsing every record's id.
    existing = _count_lines(path)
    record = {
        "seq": existing,
        "source": source,
        "kind": kind,
        "ts": _now_iso(),
        "payload": payload,
    }
    # Append is line-oriented; concurrent appends are tolerated because each
    # writer reads the current count then appends a single line. A small race
    # could reuse a seq under heavy concurrency, but the file-IPC contract here
    # is single-writer-per-side (GUI vs Claude), matching the mailbox idiom.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def read_events(since: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    """Return events with ``seq >= since`` (in append order).

    Malformed lines are skipped defensively so one bad append never blocks the
    whole feed. When ``limit`` is given, only the most-recent ``limit`` events
    (after the ``since`` filter) are returned. Records are deduped by ``seq``
    (last occurrence wins) to defend against the documented concurrent-writer
    seq-reuse race (see ``append_event``); for a clean single-writer log this is
    a no-op so default behavior is unchanged.
    """
    path = events_path()
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        seq = record.get("seq", idx)
        if isinstance(seq, int) and seq >= since:
            out.append(record)

    # Dedup by seq, keeping the last occurrence (a reused seq under concurrent
    # appends means the later line is the most recent write for that slot).
    deduped: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for record in out:
        key = record.get("seq")
        if key in deduped:
            deduped[key] = record  # last occurrence wins
        else:
            deduped[key] = record
            order.append(key)
    out = [deduped[key] for key in order]

    if limit is not None and limit >= 0:
        out = out[-limit:] if limit else []
    return out


def _count_lines(path: Path) -> int:
    """Count non-empty lines in ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------- run registry


def register_run(run_record: Dict[str, Any]) -> Path:
    """Write ``runs/<run_id>.json`` for a new run; returns the file path.

    ``run_record`` must carry a ``run_id``. ``created_at`` / ``updated_at`` are
    stamped when absent. Atomic write.
    """
    if not isinstance(run_record, dict):
        raise ValueError("run_record must be a dict")
    run_id = run_record.get("run_id")
    if not run_id:
        raise ValueError("run_record must include a non-empty 'run_id'")
    _validate_id(run_id, "run_id")

    record = dict(run_record)
    record.setdefault("created_at", _now_iso())
    record["updated_at"] = _now_iso()

    target = runs_dir() / f"{run_id}.json"
    _atomic_write_json(target, record)
    return target


def update_run(run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-shallow-merge ``patch`` into ``runs/<run_id>.json`` and return it.

    Re-stamps ``updated_at``. Raises ``FileNotFoundError`` if the run is unknown.
    """
    _validate_id(run_id, "run_id")
    if not isinstance(patch, dict):
        raise ValueError("patch must be a dict")
    target = runs_dir() / f"{run_id}.json"
    if not target.exists():
        raise FileNotFoundError(f"no run registered with id {run_id!r}")
    current = json.loads(target.read_text(encoding="utf-8"))
    current.update(patch)
    current["updated_at"] = _now_iso()
    _atomic_write_json(target, current)
    return current


def read_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return the run record for ``run_id`` or ``None`` if unknown."""
    _validate_id(run_id, "run_id")
    target = runs_dir() / f"{run_id}.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_runs() -> List[Dict[str, Any]]:
    """Return all run records, newest ``created_at`` first."""
    directory = runs_dir()
    out: List[Dict[str, Any]] = []
    for path in directory.iterdir():
        if not (path.is_file() and path.suffix == ".json" and not path.name.startswith(".")):
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    return out


# ----------------------------------------------------------------- log tail


def append_log(run_id: str, text: str) -> None:
    """Append ``text`` to ``logs/<run_id>.log`` (created lazily)."""
    path = log_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def tail_log(run_id: str, offset: int = 0) -> Tuple[str, int]:
    """Return ``(new_text, new_offset)`` from ``logs/<run_id>.log``.

    ``offset`` is a byte position. Returns ``("", offset)`` when the log is
    missing or unchanged, so a WebSocket loop can poll cheaply.
    """
    path = log_path(run_id)
    if not path.exists():
        return "", offset
    size = path.stat().st_size
    if offset >= size:
        return "", offset
    with path.open("rb") as fh:
        fh.seek(max(0, offset))
        data = fh.read()
    return data.decode("utf-8", errors="replace"), size


# --------------------------------------------------------------------- utils


def _validate_id(value: str, label: str) -> None:
    """Guard against path-separator escapes in run/upload ids."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if os.sep in value or "/" in value or ".." in value:
        raise ValueError(f"{label} may not contain path separators: {value!r}")
    if value.startswith("."):
        raise ValueError(f"{label} may not start with a dot: {value!r}")


def now_iso() -> str:
    """Public UTC ISO-8601 timestamp helper (re-exported for callers)."""
    return _now_iso()


def new_run_id(prefix: str = "GUI") -> str:
    """Mint a fresh, filesystem-safe run id: ``<prefix>-YYYYmmdd-HHMMSS-<ms hex><rand hex>``.

    The millisecond component keeps ids roughly time-sortable; the trailing
    random bytes guarantee uniqueness even for back-to-back calls within the
    same millisecond (a bare ms suffix collides when two runs register in the
    same tick).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"{int(time.time() * 1000) & 0xFFFFFF:06x}{os.urandom(3).hex()}"
    return f"{prefix}-{stamp}-{suffix}"


__all__ = [
    "gui_state_dir",
    "settings_path",
    "env_rendered_path",
    "runs_dir",
    "logs_dir",
    "uploads_dir",
    "events_path",
    "log_path",
    "append_event",
    "read_events",
    "register_run",
    "update_run",
    "read_run",
    "list_runs",
    "append_log",
    "tail_log",
    "now_iso",
    "new_run_id",
]
