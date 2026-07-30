"""Campaign self-orchestration tool set for the operator assistant.

This is the ``--campaign`` companion to :mod:`lib.assistant.tools`. It follows
the SAME 5-part registration pattern (plain ``str``-returning functions +
:data:`CAMPAIGN_TOOL_REGISTRY` + :data:`CAMPAIGN_TOOL_SCHEMAS` +
:data:`_CAMPAIGN_TOOL_ARG_WHITELIST` + :data:`_CAMPAIGN_TOOL_REQUIRED_ARGS` +
:func:`dispatch_campaign_tool`) and the same error convention (tools RETURN
strings; refusals start ``"Refused: "``; results are ``_clip``/``_redact``-ed
through the helpers imported from :mod:`lib.assistant.tools`).

Division of labor is HARD (owner-stated law):

* The assistant ORGANIZES / ARRANGES / MONITORS the multi-book campaign — it
  prepares validated JSON *data* overlays (never scripts), launches / resumes /
  stops runs through fixed argv lists, monitors run state, and files
  review-queue reports.
* Claude (dev sessions) FIXES errors and REVIEWS. Anything beyond resume/stop
  becomes a review-queue report via :func:`campaign_report`.

Sandbox rules (extend, never weaken):

* NO script/code authoring capability is reachable by the model. There is no
  generic file-write tool, no shell/exec tool, no ``eval``. Env overlays are
  validated data files only (:mod:`lib.assistant.campaign_flags`).
* ``--force`` resume does NOT exist here — :func:`campaign_resume_run` spawns a
  PLAIN ``ed4all run --resume <id>``; ``--force`` is not a parameter, not in the
  schema, and no code path can add it.
* Every spawn is a FIXED argv list via ``subprocess.Popen(start_new_session=
  True)`` / ``subprocess.run`` — never ``shell=True``, never string
  interpolation of model input into a command.
* Corpus paths must ``realpath``-resolve under ``<repo>/inputs/`` or the tool
  refuses (a symlink escaping the tree is caught by the realpath check).
* Single-owner + STOP_ALL preflight gate every launch/resume.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lib.paths import PROJECT_ROOT, STATE_PATH, campaign_dir
from lib.assistant.tools import RUN_ID_RE, _clip, _redact

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Path constants (module level, monkeypatchable by tests — S13)
# --------------------------------------------------------------------------- #

# Operator campaign-harness directory, site-configurable via
# ``ED4ALL_CAMPAIGN_DIR`` (default: the neutral repo-relative
# ``plans/campaign``). See ``lib.paths.campaign_dir``.
CAMPAIGN_DIR = campaign_dir()
PENDING_RUNS_DIR = CAMPAIGN_DIR / "pending-runs"
REVIEW_QUEUE_DIR = CAMPAIGN_DIR / "review-queue"
LAUNCHED_RUNS_PATH = CAMPAIGN_DIR / "launched-runs.jsonl"
LAUNCHER_SH = CAMPAIGN_DIR / "launch_book.sh"
INPUTS_ROOT = PROJECT_ROOT / "inputs"

#: Overlay / campaign-slug shape (matches the S5 pending-run name pattern).
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: The fixed review-report kinds (S4 / S6).
REPORT_KINDS: Tuple[str, ...] = (
    "run_failure",
    "run_paused",
    "gate_anomaly",
    "assistant_error",
    "campaign_note",
)

#: Workflow statuses treated as "active" by campaign_run_status.
_ACTIVE_STATUSES = frozenset({"RUNNING", "PAUSED", "FAILED"})

#: Stage-B (LoRA training) — env override + default for the campaign base
#: model. The value is VALIDATED against ``BaseModelRegistry`` in
#: :func:`prepare_training_run`; an unknown name is a LOUD error, never a
#: silent fallback to a different model.
CAMPAIGN_BASE_MODEL_ENV = "ED4ALL_CAMPAIGN_BASE_MODEL"
DEFAULT_CAMPAIGN_BASE_MODEL = "nemotron3-nano-30b"

#: Manifest book statuses eligible for Stage-B training (mirrors the
#: operator training-driver's pick logic).
TRAINABLE_BOOK_STATUSES: Tuple[str, ...] = ("built", "review")

#: Bounded wall-clock ceiling (seconds) on the post-launch poll for the
#: freshly-minted training WF-*.json (same 30s bound as ``launch_book.py``).
_TRAINING_WF_POLL_SECONDS = 30.0

#: Hard cap on a redacted log excerpt persisted in a review report (S6).
_LOG_EXCERPT_CHARS = 2000


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def _iso_now() -> str:
    """UTC ISO8601 with a trailing Z (e.g. ``2026-07-22T14:03:11Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_ts() -> str:
    """UTC ``%Y%m%dT%H%M%SZ`` stamp used for review-report filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write(path: Path, text: str) -> None:
    """Atomic write: tmp sibling + ``os.replace`` (never a torn file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sanitize_name(text: Any) -> Optional[str]:
    """Sanitize a free string to the ``^[a-z0-9][a-z0-9-]{0,63}$`` name shape;
    None when nothing valid survives."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:64].strip("-")
    return slug if slug and NAME_RE.match(slug) else None


def _coerce_num(value: Any) -> float:
    """Numeric sort key; unknown/garbage sorts last."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("inf")


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _elapsed_since(value: Any) -> str:
    dt = _parse_iso(value)
    if dt is None:
        return "?"
    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 0:
        seconds = 0
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def _basename(value: Any) -> str:
    text = str(value or "")
    return text.rsplit("/", 1)[-1] if text else "?"


def _tail(path: Path, lines: int) -> str:
    """Bounded tail: read only trailing bytes, return the last N lines."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(0, size - lines * 400))
            data = handle.read().decode("utf-8", "replace")
    except OSError as exc:
        return f"log unreadable ({type(exc).__name__}: {exc})"
    return "\n".join(data.splitlines()[-lines:])


def _campaign_flags():
    """Monkeypatchable seam returning the ``campaign_flags`` module (imported
    lazily so this module stays importable even if that sibling has not
    landed yet)."""
    from lib.assistant import campaign_flags  # noqa: PLC0415

    return campaign_flags


def _manifest_books() -> List[Dict[str, Any]]:
    manifest_path = CAMPAIGN_DIR / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    books = manifest.get("books") if isinstance(manifest, dict) else None
    return [b for b in books if isinstance(b, dict)] if isinstance(books, list) else []


def _list_overlays() -> List[str]:
    if not PENDING_RUNS_DIR.is_dir():
        return []
    try:
        return sorted(p.stem for p in PENDING_RUNS_DIR.glob("*.json") if p.is_file())
    except OSError:
        return []


def _read_launched_rows() -> List[Dict[str, Any]]:
    if not LAUNCHED_RUNS_PATH.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in LAUNCHED_RUNS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        return rows
    return rows


def _row_kind(row: Dict[str, Any]) -> str:
    """A launched-runs row's kind. Rows predate the ``kind`` field (Stage-A
    builds), so a missing/blank kind reads as ``"build"`` — existing readers
    (``_campaign_run_ids``, the status summaries) stay compatible."""
    kind = str(row.get("kind") or "").strip()
    return kind if kind else "build"


def launched_training_rows() -> List[Dict[str, Any]]:
    """The ``kind == "training"`` launched-runs rows (Stage-B provenance)."""
    return [row for row in _read_launched_rows() if _row_kind(row) == "training"]


def _campaign_log_for(run_id: str) -> Optional[Path]:
    """Resolve a run's campaign log via a launched-runs row (best-effort)."""
    for row in _read_launched_rows():
        if row.get("wf_id") == run_id:
            log_path = row.get("log_path")
            if log_path:
                return Path(str(log_path))
    return None


def _campaign_run_ids() -> set:
    """The wf_ids the CAMPAIGN launched — the ONLY runs a MUTATING campaign
    tool may resume/stop.

    Authority is the launch-manifest ``launched-runs.jsonl`` (one provenance
    row per launch, ``wf_id`` field). ``runtime/state/workflows`` accumulates ~months
    of unrelated dev runs; without this scope a single resume/stop tool call
    could fire a real ``ed4all run --resume`` / ``ed4all stop`` against a run
    the campaign never started. Reads/status stay unrestricted (they only
    display); the MUTATING tools refuse a non-manifest id. A row with a
    null/blank wf_id contributes nothing."""
    ids: set = set()
    for row in _read_launched_rows():
        wf = row.get("wf_id")
        if wf:
            ids.add(str(wf))
    return ids


# --------------------------------------------------------------------------- #
# DecisionCapture seam (monkeypatchable) — mutating tools log one decision each
# --------------------------------------------------------------------------- #


def _get_capture(course_code: str = "campaign"):
    """Build a DecisionCapture for the campaign tool surface. Monkeypatchable
    so tests can inject a recording stub."""
    from lib.decision_capture import DecisionCapture  # noqa: PLC0415

    return DecisionCapture(course_code=course_code, phase="campaign_tools", tool="assistant")


def _log_capture(
    decision_type: str,
    decision: str,
    rationale: str,
    *,
    course_code: str = "campaign",
) -> None:
    """Best-effort decision capture — a capture failure is LOGGED and never
    eats the tool result."""
    try:
        capture = _get_capture(course_code)
        capture.log_decision(
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
            operation="campaign_tools",
        )
    except Exception:  # noqa: BLE001 — capture is best-effort, loud in log
        logger.exception("campaign: decision capture failed (%s)", decision_type)


# --------------------------------------------------------------------------- #
# Public preflight helpers (also imported by launch_book.py + pilot.py)
# --------------------------------------------------------------------------- #


def ed4all_run_pids(proc_root: Union[str, Path] = "/proc") -> List[int]:
    """PIDs of live ``ed4all run`` processes (excluding this one).

    Scans ``<proc_root>/<pid>/cmdline`` (NUL-split tokens) for an ADJACENT
    ``("ed4all", "run")`` token pair. NEVER uses ``pgrep -f`` (self-match
    hazard). Unreadable ``/proc`` entries are skipped.
    """
    root = Path(proc_root)
    me = os.getpid()
    found: List[int] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        tokens = [t.decode("utf-8", "replace") for t in raw.split(b"\x00") if t]
        for i in range(len(tokens) - 1):
            first = tokens[i].rsplit("/", 1)[-1]
            if first == "ed4all" and tokens[i + 1] == "run":
                found.append(pid)
                break
    return found


def trainforge_train_pids(proc_root: Union[str, Path] = "/proc") -> List[int]:
    """PIDs of live ``ed4all run trainforge_train`` processes (excluding this
    one).

    Same discipline as :func:`ed4all_run_pids`: scans
    ``<proc_root>/<pid>/cmdline`` NUL-split tokens for the ADJACENT
    ``("ed4all", "run", "trainforge_train")`` token triple. NEVER uses
    ``pgrep -f`` (self-match hazard). Unreadable ``/proc`` entries skipped.
    """
    root = Path(proc_root)
    me = os.getpid()
    found: List[int] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        tokens = [t.decode("utf-8", "replace") for t in raw.split(b"\x00") if t]
        for i in range(len(tokens) - 2):
            first = tokens[i].rsplit("/", 1)[-1]
            if (
                first == "ed4all"
                and tokens[i + 1] == "run"
                and tokens[i + 2] == "trainforge_train"
            ):
                found.append(pid)
                break
    return found


def preflight_launch(proc_root: Union[str, Path] = "/proc") -> Optional[str]:
    """Read-only launch preflight. ``None`` when clear; else a ``"Refused: "``
    string. Checks the global STOP_ALL sentinel at BOTH sentinel locations and
    single-owner (no live ``ed4all run``)."""
    for candidate in (STATE_PATH / "STOP_ALL", STATE_PATH / "runs" / "STOP_ALL"):
        try:
            if candidate.exists():
                return (
                    "Refused: the global STOP_ALL sentinel is present — it "
                    "pauses AND blocks new/resumed runs until the operator "
                    "clears it (ed4all stop --clear-all). Nothing was launched."
                )
        except OSError:
            return (
                "Refused: the state dir is unreadable — cannot verify STOP_ALL. "
                "Nothing was launched."
            )
    pids = ed4all_run_pids(proc_root)
    if pids:
        return (
            f"Refused: an ed4all run is already active (pid(s) "
            f"{', '.join(str(p) for p in sorted(pids))}) — one build at a time "
            f"(single-owner). Nothing was launched."
        )
    return None


def validate_corpus_path(corpus: str) -> Path:
    """Resolve + confirm a corpus path is under ``<repo>/inputs/`` and exists.

    Uses ``os.path.realpath`` so a symlink escaping the inputs tree is caught.
    Raises ``ValueError`` (the tool layer converts to ``"Refused: ..."``).
    """
    corpus = str(corpus or "").strip()
    if not corpus:
        raise ValueError("corpus path is empty")
    real = os.path.realpath(corpus)
    root = os.path.realpath(str(INPUTS_ROOT))
    if real != root and not real.startswith(root + os.sep):
        raise ValueError(
            f"corpus path {corpus!r} does not resolve under {INPUTS_ROOT} "
            f"(realpath={real}) — corpora must live under <repo>/inputs/"
        )
    path = Path(real)
    if not path.exists():
        raise ValueError(f"corpus path {corpus!r} does not exist (realpath={real})")
    return path


# --------------------------------------------------------------------------- #
# Review-queue helpers (public — pilot imports them)
# --------------------------------------------------------------------------- #


def write_review_report(
    kind: str,
    *,
    summary: str,
    book_slug: Optional[str] = None,
    run_id: Optional[str] = None,
    phase: Optional[str] = None,
    error_class: Optional[str] = None,
    log_excerpt: Optional[str] = None,
    verdict: Optional[str] = None,
) -> Path:
    """Write one review-queue report (S6 shape) + append an INDEX.md line.

    Filename ``<UTC %Y%m%dT%H%M%SZ>-<kind>.json`` (collision → ``-2``, ``-3``…).
    ``log_excerpt`` is clipped to 2000 chars and passed through ``_redact``.
    """
    REVIEW_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    excerpt: Optional[str] = None
    if log_excerpt is not None:
        excerpt = _redact(str(log_excerpt)[:_LOG_EXCERPT_CHARS])

    ts = _iso_now()
    record = {
        "version": 1,
        "ts": ts,
        "kind": kind,
        "book_slug": book_slug,
        "run_id": run_id,
        "phase": phase,
        "error_class": error_class,
        "log_excerpt": excerpt,
        "summary": str(summary),
        "verdict": verdict,
        "status": "open",
    }

    base = f"{_compact_ts()}-{kind}"
    path = REVIEW_QUEUE_DIR / f"{base}.json"
    counter = 2
    while path.exists():
        path = REVIEW_QUEUE_DIR / f"{base}-{counter}.json"
        counter += 1
    _atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False))

    index = REVIEW_QUEUE_DIR / "INDEX.md"
    if not index.exists():
        index.write_text("# Campaign review queue\n\n", encoding="utf-8")
    label = book_slug or run_id or "-"
    line = f"- {ts} [{kind}] {label}: {str(summary)[:120]} ({path.name})\n"
    with open(index, "a", encoding="utf-8") as handle:
        handle.write(line)
    return path


def blocking_reports(book_slug: str) -> List[dict]:
    """Open, blocking review reports for a book.

    A record blocks when ``status == "open"`` AND ``verdict != "CLEAR"`` AND
    (``kind`` in ``("run_failure", "gate_anomaly")`` OR ``verdict == "BLOCK"``).
    An UNPARSEABLE report is treated as BLOCKING (fail-closed) with a synthetic
    record.
    """
    result: List[dict] = []
    if not REVIEW_QUEUE_DIR.is_dir():
        return result
    try:
        files = sorted(REVIEW_QUEUE_DIR.glob("*.json"))
    except OSError:
        return result
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unparseable → fail-closed
            result.append(_synthetic_unparseable(path, book_slug))
            continue
        if not isinstance(record, dict):
            result.append(_synthetic_unparseable(path, book_slug))
            continue
        if record.get("book_slug") != book_slug:
            continue
        if record.get("status") != "open":
            continue
        if record.get("verdict") == "CLEAR":
            continue
        if record.get("kind") in ("run_failure", "gate_anomaly") or record.get("verdict") == "BLOCK":
            result.append(record)
    return result


def _synthetic_unparseable(path: Path, book_slug: str) -> dict:
    return {
        "version": 1,
        "ts": None,
        "kind": "unparseable",
        "book_slug": book_slug,
        "run_id": None,
        "phase": None,
        "error_class": "UnparseableReport",
        "log_excerpt": None,
        "summary": f"unparseable review report {path.name} — treated as BLOCKING",
        "verdict": "BLOCK",
        "status": "open",
        "_path": str(path),
    }


def _open_report_summary() -> Tuple[int, set]:
    """(open report count, set of blocking book slugs)."""
    open_count = 0
    blocking: set = set()
    if not REVIEW_QUEUE_DIR.is_dir():
        return open_count, blocking
    try:
        files = sorted(REVIEW_QUEUE_DIR.glob("*.json"))
    except OSError:
        return open_count, blocking
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            open_count += 1
            continue
        if not isinstance(record, dict) or record.get("status") != "open":
            continue
        open_count += 1
        if record.get("verdict") == "CLEAR":
            continue
        if record.get("kind") in ("run_failure", "gate_anomaly") or record.get("verdict") == "BLOCK":
            slug = record.get("book_slug")
            if slug:
                blocking.add(str(slug))
    return open_count, blocking


# --------------------------------------------------------------------------- #
# Read-only tools
# --------------------------------------------------------------------------- #


def campaign_queue() -> str:
    """Organize view of the campaign: manifest book counts + per-status slugs
    (pending in wave/pages order), prepared overlay names, and the count of
    OPEN review reports (+ blocking book slugs)."""
    books = _manifest_books()
    counts: Dict[str, int] = {}
    by_status: Dict[str, List[Dict[str, Any]]] = {}
    for book in books:
        status = str(book.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        by_status.setdefault(status, []).append(book)

    lines = [f"Campaign: {len(books)} book(s)."]
    lines.append(
        "By status: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    )

    pending = sorted(
        by_status.get("pending", []),
        key=lambda b: (_coerce_num(b.get("wave")), _coerce_num(b.get("pages")), str(b.get("slug", ""))),
    )
    lines.append(
        "Pending (wave,pages order): "
        + (", ".join(str(b.get("slug", "?")) for b in pending) or "none")
    )
    for status in sorted(by_status):
        if status == "pending":
            continue
        slugs = [str(b.get("slug", "?")) for b in by_status[status]]
        lines.append(f"{status}: " + ", ".join(slugs))

    overlays = _list_overlays()
    lines.append("Prepared overlays: " + (", ".join(overlays) or "none"))

    open_count, blocking = _open_report_summary()
    tail = f" (blocking books: {', '.join(sorted(blocking))})" if blocking else ""
    lines.append(f"Open review reports: {open_count}{tail}")
    return _clip("\n".join(lines))


def _run_status_all() -> str:
    wf_dir = STATE_PATH / "workflows"
    rows: List[str] = []
    if wf_dir.is_dir():
        for path in sorted(wf_dir.glob("WF-*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — skip an unreadable doc
                continue
            if not isinstance(doc, dict):
                continue
            status = str(doc.get("status", "")).upper()
            if status not in _ACTIVE_STATUSES:
                continue
            params = doc.get("params") if isinstance(doc.get("params"), dict) else {}
            course = params.get("course_name", "?")
            phase = (
                doc.get("failed_phase")
                or doc.get("paused_phase")
                or doc.get("current_phase")
                or "?"
            )
            elapsed = _elapsed_since(doc.get("started_at") or doc.get("created_at"))
            updated = _elapsed_since(doc.get("updated_at"))
            rows.append(
                f"{path.stem}  status={status}  course={course}  phase={phase}  "
                f"elapsed={elapsed}  updated={updated} ago"
            )
    lines = ["Active runs (RUNNING/PAUSED/FAILED):"]
    lines.extend(rows or ["  none"])

    launched = _read_launched_rows()[-5:]
    if launched:
        lines.append("Recent launches:")
        for row in reversed(launched):
            lines.append(
                f"  {row.get('ts', '?')} name={row.get('name')} "
                f"wf_id={row.get('wf_id')} pid={row.get('pid')} "
                f"corpus={_basename(row.get('corpus'))}"
            )
    return _clip(_redact("\n".join(lines)))


def _run_status_one(run_id: str) -> str:
    path = STATE_PATH / "workflows" / f"{run_id}.json"
    if not path.is_file():
        return f"no workflow state for {run_id} (runtime/state/workflows/{run_id}.json missing)."
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"workflow state for {run_id} unreadable ({type(exc).__name__}: {exc})."
    params = doc.get("params") if isinstance(doc, dict) and isinstance(doc.get("params"), dict) else {}
    lines = [
        f"Run {run_id}  status={doc.get('status', '?') if isinstance(doc, dict) else '?'}  "
        f"course={params.get('course_name', '?')}"
    ]
    if isinstance(doc, dict):
        for key in ("current_phase", "failed_phase", "paused_phase", "failure_reason"):
            if doc.get(key):
                lines.append(f"{key}: {str(doc[key])[:300]}")
        started = doc.get("started_at") or doc.get("created_at")
        if started:
            lines.append(f"elapsed={_elapsed_since(started)}")

    log_path = _campaign_log_for(run_id)
    if log_path is not None and log_path.is_file():
        lines.append(f"[last 40 lines of {log_path.name}]")
        lines.append(_tail(log_path, 40))
    return _clip(_redact("\n".join(lines)))


def campaign_run_status(run_id: str = "") -> str:
    """Read-only run monitor. No ``run_id`` → summarize every active
    runtime/state/workflows record + recent launches. With ``run_id`` (RUN_ID_RE
    validated) → that record + the last 40 lines of its campaign log."""
    run_id = str(run_id or "").strip()
    if run_id:
        if not RUN_ID_RE.match(run_id):
            return (
                f"Refused: {run_id[:80]!r} is not a valid run id "
                f"(expected WF-YYYYMMDD-xxxxxxxx)."
            )
        return _run_status_one(run_id)
    return _run_status_all()


# --------------------------------------------------------------------------- #
# Prepare (validated data overlay only — NEVER a script)
# --------------------------------------------------------------------------- #


def campaign_prepare_run(
    corpus: str,
    env_overlay: Optional[dict] = None,
    note: Optional[str] = None,
) -> str:
    """Prepare a run: validate the corpus path + the env overlay, then write a
    validated JSON DATA overlay under ``pending-runs/<name>.json`` (S5 shape).

    This writes NO script and NO arbitrary file — only the fixed-shape overlay.
    """
    try:
        corpus_path = validate_corpus_path(corpus)
    except ValueError as err:
        return "Refused: " + str(err)

    if env_overlay is None or env_overlay == "":
        overlay_in: Any = {}
    else:
        overlay_in = env_overlay
    if not isinstance(overlay_in, dict):
        return "Refused: env_overlay must be an object mapping flag names to string values."

    flags = _campaign_flags()
    try:
        env = flags.validate_overlay(overlay_in)
    except flags.CampaignFlagError as err:
        return "Refused: " + str(err)

    stem = corpus_path.stem if (corpus_path.is_file() and corpus_path.suffix) else corpus_path.name
    name = _sanitize_name(stem)
    if name is None:
        return (
            f"Refused: could not derive a valid overlay name from corpus "
            f"{corpus_path.name!r} (need ^[a-z0-9][a-z0-9-]{{0,63}}$)."
        )

    record = {
        "version": 1,
        "created": _iso_now(),
        "corpus": str(corpus_path),
        "env": {key: env[key] for key in sorted(env)},
        "note": str(note) if note is not None else None,
        "prepared_by": "assistant",
    }
    path = PENDING_RUNS_DIR / f"{name}.json"
    _atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False))

    keys = ", ".join(sorted(env)) or "none"
    return _clip(
        f"Prepared run overlay {path.name} at {path} (corpus={corpus_path.name}); "
        f"env keys: {keys}. Launch with campaign_launch_run name={name!r}."
    )


# --------------------------------------------------------------------------- #
# Mutating tools (fixed argv, preflight-gated, one DecisionCapture each)
# --------------------------------------------------------------------------- #


def campaign_launch_run(name: str) -> str:
    """Launch a PREPARED run overlay by name. Re-validates the (untrusted)
    overlay file, runs the mandatory preflight, then spawns the FIXED argv
    ``["bash", LAUNCHER_SH, "--overlay", <path>]`` detached."""
    name = str(name or "").strip()
    if not NAME_RE.match(name):
        return (
            f"Refused: {name[:80]!r} is not a valid overlay name "
            f"(need ^[a-z0-9][a-z0-9-]{{0,63}}$). Nothing was launched."
        )
    overlay_path = PENDING_RUNS_DIR / f"{name}.json"
    if not overlay_path.is_file():
        return (
            f"Refused: no prepared overlay {name!r} "
            f"(expected {overlay_path}). Prepare one with campaign_prepare_run."
        )

    # The on-disk overlay is UNTRUSTED — re-validate corpus + env (a tampered
    # file is refused).
    try:
        record = json.loads(overlay_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return (
            f"Refused: overlay {name!r} is unreadable/corrupt "
            f"({type(exc).__name__}). Nothing was launched."
        )
    if not isinstance(record, dict):
        return f"Refused: overlay {name!r} is not a JSON object. Nothing was launched."
    try:
        corpus_path = validate_corpus_path(record.get("corpus"))
    except ValueError as err:
        return "Refused: overlay corpus invalid: " + str(err)
    flags = _campaign_flags()
    try:
        env = flags.validate_overlay(record.get("env") or {})
    except flags.CampaignFlagError as err:
        return "Refused: overlay env invalid: " + str(err)

    refusal = preflight_launch()
    if refusal is not None:
        return refusal

    argv = ["bash", str(LAUNCHER_SH), "--overlay", str(overlay_path)]
    log_dir = CAMPAIGN_DIR / "logs"
    log_path = log_dir / f"launch-{name}.log"
    log_fh = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            argv,
            start_new_session=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    except OSError as exc:
        return f"Failed to spawn launcher: {type(exc).__name__}: {exc}"
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass

    _log_capture(
        "campaign_run_launch",
        f"launch:{name}",
        f"launched prepared overlay {name} (corpus={corpus_path.name}, pid={proc.pid}, "
        f"env keys={sorted(env)})",
    )
    return _clip(
        f"Launched {name} (pid {proc.pid}); log: {log_path}. "
        f"Follow with campaign_run_status."
    )


def campaign_resume_run(run_id: str) -> str:
    """Resume a paused run — PLAIN ``ed4all run --resume <id>``, detached.

    ``--force`` does NOT exist here: it is not a parameter, not in the schema,
    and no code path can add it. Preflight: STOP_ALL + single-owner.
    """
    run_id = str(run_id or "").strip()
    if not RUN_ID_RE.match(run_id):
        return (
            f"Refused: {run_id[:80]!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx). Nothing was resumed."
        )
    if run_id not in _campaign_run_ids():
        return (
            f"Refused: {run_id} is not in the campaign launch-manifest "
            f"(launched-runs.jsonl) — campaign tools resume/stop ONLY runs the "
            f"campaign itself launched, never the unrelated dev runs in "
            f"runtime/state/workflows. If this run really needs resuming, do it directly "
            f"with `ed4all run --resume {run_id}`. Nothing was resumed."
        )
    refusal = preflight_launch()
    if refusal is not None:
        return refusal

    # `ed4all run` requires the WORKFLOW_NAME positional even on the --resume
    # path (cli/commands/run.py declares it a required click argument). The
    # resume path (`_resume_workflow`) resumes purely by workflow_id from the
    # persisted run state and IGNORES this positional, so any supported
    # workflow name satisfies click without affecting what is resumed;
    # `textbook_to_course` is the campaign's build workflow. Omitting it made
    # every resume spawn fail with "Missing argument 'WORKFLOW_NAME'".
    argv = ["ed4all", "run", "textbook_to_course", "--resume", run_id]
    log_dir = CAMPAIGN_DIR / "logs"
    log_path = log_dir / f"launch-resume-{run_id}.log"
    log_fh = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell, no --force
            argv,
            start_new_session=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    except OSError as exc:
        return f"Failed to spawn resume: {type(exc).__name__}: {exc}"
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass

    _log_capture(
        "campaign_run_control",
        f"resume:{run_id}",
        f"resumed paused run {run_id} with a PLAIN --resume (never --force; "
        f"pid={proc.pid}); log {log_path.name}",
    )
    return _clip(
        f"Resume of {run_id} spawned (pid {proc.pid}); log: {log_path}. "
        f"Follow with campaign_run_status."
    )


def campaign_stop_run(run_id: str) -> str:
    """Gracefully stop one run via the FIXED argv ``["ed4all", "stop", <id>]``."""
    run_id = str(run_id or "").strip()
    if not RUN_ID_RE.match(run_id):
        return (
            f"Refused: {run_id[:80]!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx). Nothing was stopped."
        )
    if run_id not in _campaign_run_ids():
        return (
            f"Refused: {run_id} is not in the campaign launch-manifest "
            f"(launched-runs.jsonl) — campaign tools resume/stop ONLY runs the "
            f"campaign itself launched, never the unrelated dev runs in "
            f"runtime/state/workflows. If this run really needs stopping, do it directly "
            f"with `ed4all stop {run_id}`. Nothing was stopped."
        )
    argv = ["ed4all", "stop", run_id]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, validated id, no shell
            argv, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all stop {run_id}` failed to execute: {type(exc).__name__}: {exc}"

    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    _log_capture(
        "campaign_run_control",
        f"stop:{run_id}",
        f"graceful stop sentinel requested for run {run_id} (exit={proc.returncode})",
    )
    if proc.returncode != 0:
        return _clip(_redact(f"`ed4all stop {run_id}` exited {proc.returncode}: {output}"))
    return _clip(_redact(f"Stop sentinel dropped for {run_id}: {output or 'ok'}"))


def campaign_report(
    kind: str,
    summary: str,
    run_id: Optional[str] = None,
    phase: Optional[str] = None,
    error_class: Optional[str] = None,
    log_excerpt: Optional[str] = None,
    book_slug: Optional[str] = None,
) -> str:
    """File a structured review-queue report for Claude (S6 shape)."""
    kind = str(kind or "").strip()
    if kind not in REPORT_KINDS:
        return (
            f"Refused: {kind[:60]!r} is not a valid report kind. "
            f"Valid kinds: {', '.join(REPORT_KINDS)}."
        )
    summary = str(summary or "").strip()
    if not summary:
        return "Refused: campaign_report needs a non-empty summary."
    if run_id is not None and str(run_id).strip():
        candidate = str(run_id).strip()
        if not RUN_ID_RE.match(candidate):
            return (
                f"Refused: {candidate[:80]!r} is not a valid run id "
                f"(expected WF-YYYYMMDD-xxxxxxxx)."
            )
        run_id = candidate
    else:
        run_id = None

    path = write_review_report(
        kind,
        summary=summary,
        book_slug=str(book_slug) if book_slug else None,
        run_id=run_id,
        phase=str(phase) if phase else None,
        error_class=str(error_class) if error_class else None,
        log_excerpt=log_excerpt,
        verdict=None,
    )

    _log_capture(
        "campaign_review_report",
        f"report:{kind}",
        f"filed {kind} review report for {book_slug or run_id or 'campaign'}: {summary[:60]}",
        course_code=str(book_slug) if book_slug else "campaign",
    )

    index_line = ""
    try:
        rows = [
            ln for ln in (REVIEW_QUEUE_DIR / "INDEX.md").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        index_line = rows[-1] if rows else ""
    except OSError:
        pass
    return _clip(f"Wrote review report: {path}\nINDEX: {index_line}")


# --------------------------------------------------------------------------- #
# Stage-B: LoRA training (prepare / launch / status)
#
# Same sandbox posture as Stage-A: fixed argv, no shell, NAME_RE-validated
# slug, manifest-scoped, single-owner preflight via the /proc adjacent-token
# scan (NEVER ``pgrep -f``), every mutation behind dispatch_campaign_tool.
# ``prepare_training_run`` / ``launch_training_run`` are a STABLE import
# contract for the operator campaign pilot driver — keep the signatures and
# return shapes fixed, and keep this module import-light (torch-adjacent
# imports happen lazily INSIDE the functions).
# --------------------------------------------------------------------------- #


def resolve_campaign_base_model() -> str:
    """The campaign Stage-B base-model short name: ``ED4ALL_CAMPAIGN_BASE_MODEL``
    env when non-blank, else :data:`DEFAULT_CAMPAIGN_BASE_MODEL`. NOT validated
    here — :func:`prepare_training_run` validates it against the registry and
    fails LOUDLY on an unknown name (never a fallback to a different model)."""
    raw = os.environ.get(CAMPAIGN_BASE_MODEL_ENV, "")
    raw = str(raw).strip()
    return raw or DEFAULT_CAMPAIGN_BASE_MODEL


def training_env_problems() -> List[str]:
    """train_next-style training-environment readiness checks, factored HERE
    (never imported from ``plans/``): transformers >= 4.57 (nemotron_h
    unloadable below), plus the Mamba fast-path kernels (``mamba_ssm`` +
    ``causal_conv1d``) — the audited 3-10x silent-slowdown trap. Returns a
    list of problem strings (empty == ready). All imports are lazy so this
    module stays import-light."""
    problems: List[str] = []
    try:
        import transformers  # noqa: PLC0415 — lazy, torch-adjacent

        version = str(transformers.__version__)
        major, minor = (int(x) for x in version.split(".")[:2])
        if (major, minor) < (4, 57):
            problems.append(
                f"transformers {version} < 4.57 cannot load nemotron_h — "
                f"run the staged training-env upgrade first"
            )
    except Exception as exc:  # noqa: BLE001 — report, never raise
        problems.append(f"transformers not importable ({exc})")
    for pkg in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(pkg)
        except Exception:  # noqa: BLE001 — report, never raise
            problems.append(
                f"{pkg} missing — Mamba fast-path kernels required; refusing "
                f"the silent 3-10x naive-path slowdown"
            )
    return problems


def instruction_pairs_path(slug: str) -> Optional[Path]:
    """The non-empty synthesized instruction pairs for a LibV2 course, if
    present (mirrors ``train_next.pairs_path``; honors ``ED4ALL_LIBV2_ROOT``
    via ``lib.paths.libv2_path``)."""
    from lib.paths import libv2_path  # noqa: PLC0415 — call-time env resolution

    base = libv2_path() / "courses" / slug
    for rel in (
        "training_specs/instruction_pairs.jsonl",
        "training/instruction_pairs.jsonl",
    ):
        candidate = base / rel
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def training_approval_path(slug: str) -> Path:
    """The human/Claude-reviewer training-approval marker for a book. The
    assistant NEVER writes this file — no campaign tool can; it is created by
    the reviewer under ``review-queue/approvals/``."""
    return REVIEW_QUEUE_DIR / "approvals" / f"{slug}.training-approved"


def _stop_all_problem() -> Optional[str]:
    """The STOP_ALL sentinel check as a problem string (None when clear)."""
    for candidate in (STATE_PATH / "STOP_ALL", STATE_PATH / "runs" / "STOP_ALL"):
        try:
            if candidate.exists():
                return (
                    f"global STOP_ALL sentinel present at {candidate} — clear "
                    f"with `ed4all stop --clear-all` first"
                )
        except OSError:
            return "state dir unreadable — cannot verify STOP_ALL"
    return None


# Seat-teardown seams — thin lazy-import wrappers over
# ``lib.vllm_container_lifecycle`` so tests monkeypatch them directly and the
# heavy module is only imported when a launch actually happens.


def _seat_registry() -> Dict[str, str]:
    """The ``ED4ALL_VLLM_CONTAINERS`` registry as ``{base_url: container}``."""
    from lib.vllm_container_lifecycle import parse_container_registry  # noqa: PLC0415

    return parse_container_registry()


def _docker_stop(container: str) -> bool:
    """Bounded, best-effort ``docker stop <container>`` (NEVER ``rm``) via the
    lifecycle lib's docker runner (sg-docker fallback, 60s timeout, never
    raises). Registry names only — the caller passes only
    ``ED4ALL_VLLM_CONTAINERS`` values, nothing discovered."""
    from lib.vllm_container_lifecycle import _run_docker  # noqa: PLC0415

    return _run_docker(["stop", container])


def _seat_probe(base_url: str) -> bool:
    """True iff ``GET {base_url}/v1/models`` answers 2xx (bounded, never raises)."""
    from lib.vllm_container_lifecycle import _probe_ready  # noqa: PLC0415

    return _probe_ready(base_url)


def teardown_vllm_seats() -> Optional[str]:
    """Stop every registered vLLM seat and VERIFY the card is free.

    Training needs the GPU exclusively: ``docker stop`` (NEVER ``rm``) each
    container in the ``ED4ALL_VLLM_CONTAINERS`` registry — registry names
    only, nothing discovered — best-effort with per-container logging, then
    probe every registered base URL; if ANY still answers ``/v1/models`` the
    teardown FAILED and the caller must NOT launch. Returns ``None`` on
    success, else a loud error string. An empty registry is a clean no-op.
    """
    registry = _seat_registry()
    for base_url in sorted(registry):
        container = registry[base_url]
        stopped = _docker_stop(container)
        logger.info(
            "campaign training: docker stop %r (seat %s) -> %s",
            container, base_url, "ok" if stopped else "FAILED",
        )
    still_up = [url for url in sorted(registry) if _seat_probe(url)]
    if still_up:
        return (
            "seat teardown FAILED — registered seat(s) still answering "
            "/v1/models after docker stop: " + ", ".join(still_up)
            + ". The card is not exclusively free; refusing to launch training."
        )
    return None


def prepare_training_run(slug: str) -> Dict[str, Any]:
    """Validate EVERYTHING for a Stage-B LoRA training run. Mutates NOTHING.

    Returns ``{"ok": bool, "error": Optional[str], "checks": {...}}`` where
    ``checks`` maps check name -> ``{"ok": bool, "detail": str}``. Every check
    is evaluated (all failures surface at once); ``error`` joins the failing
    details. STABLE contract — the operator campaign pilot driver imports
    this directly.

    Checks: slug shape + manifest membership; book status in
    :data:`TRAINABLE_BOOK_STATUSES`; non-empty instruction pairs; base model
    resolves via ``BaseModelRegistry`` (unknown name = loud error, never a
    different model); reviewer approval marker present (the assistant never
    writes it); training env ready (transformers>=4.57 + mamba kernels); no
    STOP_ALL; single-owner (no live build AND no live trainforge_train).
    """
    checks: Dict[str, Dict[str, Any]] = {}

    def _check(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    def _finish() -> Dict[str, Any]:
        failing = [
            f"{name}: {c['detail']}" for name, c in checks.items() if not c["ok"]
        ]
        return {
            "ok": not failing,
            "error": ("; ".join(failing) or None),
            "checks": checks,
        }

    slug = str(slug or "").strip()
    if not NAME_RE.match(slug):
        _check(
            "slug", False,
            f"{slug[:80]!r} is not a valid book slug "
            f"(need ^[a-z0-9][a-z0-9-]{{0,63}}$)",
        )
        return _finish()  # an invalid slug never reaches a path/argv

    book = next(
        (b for b in _manifest_books() if str(b.get("slug")) == slug), None
    )
    if book is None:
        _check("slug", False, f"{slug} is not in the campaign manifest")
        _check("book_status", False, "unknown (book not in manifest)")
    else:
        _check("slug", True, f"{slug} in manifest")
        status = str(book.get("status", ""))
        _check(
            "book_status",
            status in TRAINABLE_BOOK_STATUSES,
            f"status={status!r} (trainable: {', '.join(TRAINABLE_BOOK_STATUSES)})",
        )

    pairs = instruction_pairs_path(slug)
    _check(
        "pairs",
        pairs is not None,
        str(pairs) if pairs is not None
        else f"no non-empty instruction pairs under LibV2/courses/{slug}/ "
             f"(training_specs/ or training/instruction_pairs.jsonl) — "
             f"was the book built with --skip-training?",
    )

    base_model = resolve_campaign_base_model()
    try:
        from Trainforge.training.base_models import BaseModelRegistry  # noqa: PLC0415

        spec = BaseModelRegistry.resolve(base_model)
        _check("base_model", True, f"{base_model} -> {spec.huggingface_repo}")
    except KeyError as exc:
        _check(
            "base_model", False,
            f"unknown base model {base_model!r} "
            f"({CAMPAIGN_BASE_MODEL_ENV} override?): {exc.args[0] if exc.args else exc} "
            f"— refusing (never a fallback to a different model)",
        )
    except Exception as exc:  # noqa: BLE001 — registry import failure is loud
        _check(
            "base_model", False,
            f"base-model registry unavailable ({type(exc).__name__}: {exc})",
        )

    approval = training_approval_path(slug)
    _check(
        "approval",
        approval.is_file(),
        str(approval) if approval.is_file()
        else f"no training-approval marker at {approval} — a human/Claude "
             f"reviewer must write it; the assistant never does",
    )

    env_problems = training_env_problems()
    _check(
        "env_ready",
        not env_problems,
        "; ".join(env_problems) or "transformers>=4.57 + mamba_ssm + causal_conv1d ok",
    )

    stop_all = _stop_all_problem()
    _check("stop_all", stop_all is None, stop_all or "clear")

    train_pids = set(trainforge_train_pids())
    build_pids = [p for p in ed4all_run_pids() if p not in train_pids]
    _check(
        "no_live_build",
        not build_pids,
        (
            f"an ed4all run build is active (pid(s) "
            f"{', '.join(str(p) for p in sorted(build_pids))}) — training "
            f"needs the card exclusively"
        ) if build_pids else "no live ed4all run build",
    )
    _check(
        "no_live_training",
        not train_pids,
        (
            f"a trainforge_train run is already active (pid(s) "
            f"{', '.join(str(p) for p in sorted(train_pids))}) — one training "
            f"run at a time"
        ) if train_pids else "no live trainforge_train",
    )

    return _finish()


def _newest_wf_after(launch_time: float) -> Optional[str]:
    """The newest ``runtime/state/workflows/WF-*.json`` minted at/after ``launch_time``."""
    wf_dir = STATE_PATH / "workflows"
    try:
        names = [
            n for n in os.listdir(wf_dir)
            if n.startswith("WF-") and n.endswith(".json")
        ]
    except OSError:
        return None
    newest = None
    newest_mtime = launch_time
    for name in names:
        try:
            mtime = (wf_dir / name).stat().st_mtime
        except OSError:
            continue
        if mtime >= launch_time and mtime >= newest_mtime:
            newest_mtime = mtime
            newest = name
    return newest[:-5] if newest else None


def _poll_training_wf_id(
    launch_time: float, timeout: float = _TRAINING_WF_POLL_SECONDS
) -> Optional[str]:
    """Bounded poll for the freshly-minted training WF id (monkeypatchable
    seam — tests replace this wholesale so nothing sleeps)."""
    deadline = time.time() + max(0.0, timeout)
    while True:
        wf_id = _newest_wf_after(launch_time)
        if wf_id:
            return wf_id
        if time.time() >= deadline:
            return None
        time.sleep(1.0)


def launch_training_run(slug: str) -> Dict[str, Any]:
    """Launch a Stage-B LoRA training run for one book.

    Returns ``{"ok": bool, "pid": Optional[int], "log_path": Optional[str],
    "wf_id": Optional[str], "error": Optional[str]}``. STABLE contract —
    the operator campaign pilot driver imports this directly.

    Flow: re-run ALL :func:`prepare_training_run` checks (a stale prepare is
    never trusted) → :func:`teardown_vllm_seats` (docker stop every registered
    seat, then VERIFY none still serves — fail loudly, no launch, if one does)
    → spawn the FIXED argv ``["ed4all", "run", "trainforge_train",
    "--course-name", <slug>, "--base-model", <validated-base>]`` detached
    (``start_new_session=True`` — setsid happens IN the spawned child itself,
    so ``proc.pid`` IS the real pipeline pid; no wrapper binary, the same
    solved pattern as :func:`campaign_resume_run`) → bounded WF-id poll →
    append a ``kind: "training"`` provenance row to ``launched-runs.jsonl``.
    """
    result: Dict[str, Any] = {
        "ok": False, "pid": None, "log_path": None, "wf_id": None, "error": None,
    }
    slug = str(slug or "").strip()

    prep = prepare_training_run(slug)
    if not prep["ok"]:
        result["error"] = prep["error"] or "training preflight failed"
        return result

    teardown_error = teardown_vllm_seats()
    if teardown_error is not None:
        result["error"] = teardown_error
        return result

    base_model = resolve_campaign_base_model()
    # ``--course-name`` / ``--base-model`` are the REAL ``ed4all run`` options.
    # ``--course-code`` is the handler-side param alias declared in
    # config/workflows.yaml (``inputs_from: course_code <- course_name``), NOT a
    # CLI spelling: passing it here made every campaign training launch die on
    # a click UsageError before any workflow state was created.
    # ``cli/tests/test_run_base_model.py`` pins this argv against the live
    # click option set so the two can never drift again.
    argv = [
        "ed4all", "run", "trainforge_train",
        "--course-name", slug, "--base-model", base_model,
    ]
    log_dir = CAMPAIGN_DIR / "logs"
    log_path = log_dir / f"train-{slug}-{_compact_ts()}.log"
    launch_time = time.time()
    log_fh = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            argv,
            start_new_session=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    except OSError as exc:
        result["error"] = f"failed to spawn training run: {type(exc).__name__}: {exc}"
        return result
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass

    wf_id = _poll_training_wf_id(launch_time)
    row = {
        "ts": _iso_now(),
        "name": f"train-{slug}",
        "corpus": None,
        "overlay_path": None,
        "env": {},
        "pid": proc.pid,
        "log_path": str(log_path),
        "wf_id": wf_id,
        "book_slug": slug,
        "kind": "training",
    }
    try:
        LAUNCHED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LAUNCHED_RUNS_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        logger.exception("campaign training: failed to append launched-runs row")

    _log_capture(
        "campaign_training_launch",
        f"train:{slug}",
        f"launched trainforge_train for {slug} (base={base_model}, "
        f"pid={proc.pid}, wf_id={wf_id or 'pending'}) after full re-prepare + "
        f"vLLM seat teardown; log {log_path.name}",
        course_code=slug,
    )
    result.update(ok=True, pid=proc.pid, log_path=str(log_path), wf_id=wf_id)
    return result


# ---- string-returning campaign tools over the dict cores ------------------ #


def _format_training_checks(checks: Dict[str, Dict[str, Any]]) -> List[str]:
    return [
        f"  [{'pass' if c.get('ok') else 'FAIL'}] {name}: {c.get('detail', '')}"
        for name, c in checks.items()
    ]


def campaign_prepare_training(slug: str) -> str:
    """Validate a book's Stage-B training readiness (mutates nothing)."""
    slug = str(slug or "").strip()
    result = prepare_training_run(slug)
    lines = _format_training_checks(result["checks"])
    if not result["ok"]:
        return _clip(
            f"Refused: training prepare for {slug!r} failed — "
            f"{result['error']}\n" + "\n".join(lines)
        )
    return _clip(
        f"Training prepare OK for {slug} (all checks passed; nothing mutated).\n"
        + "\n".join(lines)
        + f"\nLaunch with campaign_launch_training slug={slug!r}."
    )


def campaign_launch_training(slug: str) -> str:
    """Launch a Stage-B training run (full re-prepare + seat teardown first)."""
    slug = str(slug or "").strip()
    result = launch_training_run(slug)
    if not result["ok"]:
        error = str(result["error"] or "training launch failed")
        if error.startswith("failed to spawn"):
            return _clip(f"Failed: {error}")
        return _clip(f"Refused: {error}. Nothing was launched.")
    _log = result["log_path"]
    return _clip(
        f"Training launched for {slug} (pid {result['pid']}, "
        f"wf_id={result['wf_id'] or 'pending'}); log: {_log}. "
        f"Follow with campaign_training_status."
    )


def campaign_training_status(slug: Optional[str] = None) -> str:
    """Read-only Stage-B monitor: recent training launches (``kind:
    "training"`` rows), each row's WF record status, a bounded log tail for
    the newest, and the training-env readiness summary."""
    slug = str(slug or "").strip()
    if slug and not NAME_RE.match(slug):
        return (
            f"Refused: {slug[:80]!r} is not a valid book slug "
            f"(need ^[a-z0-9][a-z0-9-]{{0,63}}$)."
        )

    rows = launched_training_rows()
    if slug:
        rows = [r for r in rows if str(r.get("book_slug") or "") == slug]
    rows = rows[-5:]

    lines = [
        "Training runs" + (f" for {slug}" if slug else "")
        + f" (newest last, {len(rows)} shown):"
    ]
    if not rows:
        lines.append("  none launched")
    for row in rows:
        wf_id = row.get("wf_id")
        wf_status = "?"
        if wf_id:
            wf_path = STATE_PATH / "workflows" / f"{wf_id}.json"
            try:
                doc = json.loads(wf_path.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    wf_status = str(doc.get("status", "?"))
            except Exception:  # noqa: BLE001 — display-only
                wf_status = "unreadable"
        lines.append(
            f"  {row.get('ts', '?')} book={row.get('book_slug', '?')} "
            f"pid={row.get('pid', '?')} wf_id={wf_id or '?'} status={wf_status}"
        )
    if rows:
        newest = rows[-1]
        log_path = Path(str(newest.get("log_path") or ""))
        if newest.get("log_path") and log_path.is_file():
            lines.append(f"[last 20 lines of {log_path.name}]")
            lines.append(_tail(log_path, 20))
    env_problems = training_env_problems()
    lines.append(
        "Training env: " + ("READY" if not env_problems else "NOT READY — " + "; ".join(env_problems))
    )
    return _clip(_redact("\n".join(lines)))


# --------------------------------------------------------------------------- #
# Registry + schemas + dispatch (mirrors lib.assistant.tools exactly)
# --------------------------------------------------------------------------- #

CAMPAIGN_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "campaign_queue": campaign_queue,
    "campaign_prepare_run": campaign_prepare_run,
    "campaign_launch_run": campaign_launch_run,
    "campaign_run_status": campaign_run_status,
    "campaign_resume_run": campaign_resume_run,
    "campaign_stop_run": campaign_stop_run,
    "campaign_report": campaign_report,
    "campaign_prepare_training": campaign_prepare_training,
    "campaign_launch_training": campaign_launch_training,
    "campaign_training_status": campaign_training_status,
}

#: Argument whitelist per tool — anything not listed is dropped before the
#: call. NOTE: ``campaign_resume_run`` carries ONLY ``run_id`` — ``force`` is
#: deliberately absent and can never be smuggled in.
_CAMPAIGN_TOOL_ARG_WHITELIST: Dict[str, tuple] = {
    "campaign_queue": (),
    "campaign_prepare_run": ("corpus", "env_overlay", "note"),
    "campaign_launch_run": ("name",),
    "campaign_run_status": ("run_id",),
    "campaign_resume_run": ("run_id",),
    "campaign_stop_run": ("run_id",),
    "campaign_report": (
        "kind",
        "summary",
        "run_id",
        "phase",
        "error_class",
        "log_excerpt",
        "book_slug",
    ),
    "campaign_prepare_training": ("slug",),
    "campaign_launch_training": ("slug",),
    "campaign_training_status": ("slug",),
}

#: Required arguments per tool — missing one is a refusal BEFORE the call.
_CAMPAIGN_TOOL_REQUIRED_ARGS: Dict[str, tuple] = {
    "campaign_prepare_run": ("corpus",),
    "campaign_launch_run": ("name",),
    "campaign_resume_run": ("run_id",),
    "campaign_stop_run": ("run_id",),
    "campaign_report": ("kind", "summary"),
    "campaign_prepare_training": ("slug",),
    "campaign_launch_training": ("slug",),
}


def _schema(
    name: str,
    description: str,
    params: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params or {},
                "required": required or [],
            },
        },
    }


_RUN_ID_PARAM = {
    "type": "string",
    "description": "The workflow run id, e.g. WF-20260420-abc12345.",
}

#: OpenAI function-calling schemas served to the model in campaign mode.
CAMPAIGN_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _schema(
        "campaign_queue",
        "Organize view of the multi-book campaign: manifest book counts + per-status "
        "slugs (pending in wave/pages order), prepared run overlays, and the count of "
        "open review reports (with blocking book slugs).",
    ),
    _schema(
        "campaign_prepare_run",
        "Prepare a run for a corpus by writing a VALIDATED JSON env-overlay data file "
        "(never a script). corpus must resolve under <repo>/inputs/. env_overlay keys must "
        "be in the campaign flag allowlist and values must pass the charset check.",
        {
            "corpus": {
                "type": "string",
                "description": "Absolute or relative corpus path under <repo>/inputs/.",
            },
            "env_overlay": {
                "type": "object",
                "description": "Optional flag->value string map (allowlisted keys only).",
                "additionalProperties": {"type": "string"},
            },
            "note": {"type": "string", "description": "Optional human note recorded in the overlay."},
        },
        ["corpus"],
    ),
    _schema(
        "campaign_launch_run",
        "Launch a PREPARED run overlay by name (detached). Re-validates the overlay file, "
        "runs the single-owner + STOP_ALL preflight, and spawns the campaign launcher.",
        {"name": {"type": "string", "description": "Prepared overlay name (from campaign_prepare_run)."}},
        ["name"],
    ),
    _schema(
        "campaign_run_status",
        "Monitor runs (read-only). Omit run_id to summarize every active run + recent "
        "launches; pass a run_id for that run's record + a bounded log tail.",
        {"run_id": {**_RUN_ID_PARAM, "description": "Optional run id to focus on."}},
    ),
    _schema(
        "campaign_resume_run",
        "Resume a paused run with a PLAIN --resume (never --force). Refuses under STOP_ALL "
        "or when another run is active.",
        {"run_id": _RUN_ID_PARAM},
        ["run_id"],
    ),
    _schema(
        "campaign_stop_run",
        "Gracefully stop one run (ed4all stop <id>) — it checkpoints and pauses at its next "
        "unit boundary.",
        {"run_id": _RUN_ID_PARAM},
        ["run_id"],
    ),
    _schema(
        "campaign_report",
        "File a structured review-queue report for Claude to review. Anything beyond "
        "resume/stop becomes a report — the assistant never edits code or repairs.",
        {
            "kind": {
                "type": "string",
                "description": "One of: run_failure, run_paused, gate_anomaly, assistant_error, campaign_note.",
                "enum": list(REPORT_KINDS),
            },
            "summary": {"type": "string", "description": "The triage summary."},
            "run_id": {**_RUN_ID_PARAM, "description": "Optional related run id."},
            "phase": {"type": "string", "description": "Optional failing phase name."},
            "error_class": {"type": "string", "description": "Optional error class."},
            "log_excerpt": {"type": "string", "description": "Optional bounded log excerpt."},
            "book_slug": {"type": "string", "description": "Optional related campaign book slug."},
        },
        ["kind", "summary"],
    ),
    _schema(
        "campaign_prepare_training",
        "Validate EVERYTHING for a book's Stage-B LoRA training run (mutates "
        "nothing): manifest membership + built/review status, non-empty "
        "instruction pairs, base-model registry resolution, the reviewer's "
        "training-approval marker (the assistant never writes it), training-env "
        "readiness, STOP_ALL, and single-owner (no live build, no live training).",
        {"slug": {"type": "string", "description": "The campaign book slug."}},
        ["slug"],
    ),
    _schema(
        "campaign_launch_training",
        "Launch a Stage-B LoRA training run: re-runs EVERY prepare check, docker-"
        "stops every registered vLLM seat and VERIFIES the card is free (fails "
        "loudly if a seat still serves), then spawns the fixed argv `ed4all run "
        "trainforge_train --course-name <slug> --base-model <validated>` detached.",
        {"slug": {"type": "string", "description": "The campaign book slug."}},
        ["slug"],
    ),
    _schema(
        "campaign_training_status",
        "Read-only Stage-B monitor: recent training launches, their workflow "
        "record status, a bounded log tail for the newest, and the training-env "
        "readiness summary. Optionally filtered to one book slug.",
        {"slug": {"type": "string", "description": "Optional book slug filter."}},
    ),
]


#: The OBSERVE + REPORT campaign tool subset — read/status/report only, no
#: run/queue mutation. This is what the pilot's restricted ``campaign-tick``
#: engine mode exposes so the per-tick LLM review turn cannot bypass the
#: pilot's deterministic policy (bounded auto-resume, queue advance,
#: halt-on-failure) by launching / resuming / stopping runs itself. The
#: interactive ``ed4all assistant --campaign`` session keeps the full set.
CAMPAIGN_READONLY_TOOL_NAMES: Tuple[str, ...] = (
    "campaign_queue",
    "campaign_run_status",
    "campaign_report",
    "campaign_training_status",
)

#: The read-only campaign tool schemas served in ``campaign-tick`` mode — the
#: ``CAMPAIGN_READONLY_TOOL_NAMES`` slice of :data:`CAMPAIGN_TOOL_SCHEMAS`.
CAMPAIGN_READONLY_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    s for s in CAMPAIGN_TOOL_SCHEMAS
    if s["function"]["name"] in CAMPAIGN_READONLY_TOOL_NAMES
]


def dispatch_campaign_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    readonly: bool = False,
) -> str:
    """Dispatch one campaign tool call through the whitelist.

    Unknown names get a refusal (never executed). Arguments are filtered to the
    per-tool whitelist and checked against the required set. SELF-DIAGNOSIS: any
    tool exception is caught, best-effort filed as an ``assistant_error`` review
    report (its own try/except so it can never mask the error return), then a
    loud ``"Tool {name} failed: ..."`` string is returned. Tools never raise to
    the model.

    ``readonly`` (the pilot's ``campaign-tick`` surface) refuses every mutating
    campaign tool — ``campaign_prepare_run`` / ``campaign_launch_run`` /
    ``campaign_resume_run`` / ``campaign_stop_run`` /
    ``campaign_prepare_training`` / ``campaign_launch_training`` — so the tick
    LLM turn can only OBSERVE + REPORT (``campaign_queue`` /
    ``campaign_run_status`` / ``campaign_report`` /
    ``campaign_training_status``); the deterministic pilot policy owns all
    mutations. Default False → the interactive campaign path is byte-identical.
    """
    fn = CAMPAIGN_TOOL_REGISTRY.get(name)
    if fn is None:
        return (
            f"Refused: tool {name!r} is not in the campaign tool whitelist. "
            f"Available tools: {', '.join(sorted(CAMPAIGN_TOOL_REGISTRY))}."
        )
    if readonly and name not in CAMPAIGN_READONLY_TOOL_NAMES:
        return (
            f"Refused: {name} is a mutating campaign tool and is not available "
            f"in the pilot's read-only tick surface — the deterministic pilot "
            f"policy owns launch/resume/stop. File a campaign_report to escalate."
        )
    allowed = _CAMPAIGN_TOOL_ARG_WHITELIST.get(name, ())
    kwargs = {k: v for k, v in (arguments or {}).items() if k in allowed}
    for required in _CAMPAIGN_TOOL_REQUIRED_ARGS.get(name, ()):
        if required not in kwargs:
            return f"Refused: {name} requires a {required} argument."
    try:
        return _clip(_redact(fn(**kwargs)))
    except Exception as exc:  # noqa: BLE001 — never raise to the model
        logger.exception("campaign tool %s raised", name)
        tb = traceback.format_exc()
        try:
            write_review_report(
                "assistant_error",
                summary=f"campaign tool {name} raised",
                error_class=type(exc).__name__,
                log_excerpt=tb[-_LOG_EXCERPT_CHARS:],
            )
        except Exception:  # noqa: BLE001 — must never mask the error return
            logger.exception("campaign: failed to write assistant_error report for %s", name)
        return f"Tool {name} failed: {type(exc).__name__}: {exc}"


__all__ = [
    "CAMPAIGN_DIR",
    "PENDING_RUNS_DIR",
    "REVIEW_QUEUE_DIR",
    "LAUNCHED_RUNS_PATH",
    "LAUNCHER_SH",
    "INPUTS_ROOT",
    "STATE_PATH",
    "NAME_RE",
    "REPORT_KINDS",
    "CAMPAIGN_TOOL_REGISTRY",
    "CAMPAIGN_TOOL_SCHEMAS",
    "CAMPAIGN_READONLY_TOOL_NAMES",
    "CAMPAIGN_READONLY_TOOL_SCHEMAS",
    "dispatch_campaign_tool",
    "campaign_queue",
    "campaign_prepare_run",
    "campaign_launch_run",
    "campaign_run_status",
    "campaign_resume_run",
    "campaign_stop_run",
    "campaign_report",
    "campaign_prepare_training",
    "campaign_launch_training",
    "campaign_training_status",
    "ed4all_run_pids",
    "trainforge_train_pids",
    "preflight_launch",
    "validate_corpus_path",
    "write_review_report",
    "blocking_reports",
    "prepare_training_run",
    "launch_training_run",
    "launched_training_rows",
    "resolve_campaign_base_model",
    "training_env_problems",
    "instruction_pairs_path",
    "training_approval_path",
    "teardown_vllm_seats",
    "DEFAULT_CAMPAIGN_BASE_MODEL",
    "CAMPAIGN_BASE_MODEL_ENV",
    "TRAINABLE_BOOK_STATUSES",
]
