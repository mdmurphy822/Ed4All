"""Run-progress service — the data behind ``GET /api/runs/{run_id}/progress``.

Feeds the Studio stage-tracker rail + live stats band. Everything here is a
READ-ONLY merge of artifacts other subsystems already write; this module never
mutates run state:

- **Phase plan** — the run's workflow phase list from ``config/workflows.yaml``
  (cached by file mtime; NEVER a hardcoded phase list, so ``course_generation``
  / ``rag_training`` / ``trainforge_train`` render as correctly as
  ``textbook_to_course``). Each phase gets a coarse presentation ``group``
  (conversion / planning / generation / validation / packaging / archive)
  derived from name rules, never from phase indices.
- **Run state** — the GUI run record (``state/gui/runs/<run_id>.json``) resolved
  to its orchestrator workflow state (``state/workflows/<workflow_id>.json``):
  ``phase_outputs`` ``_completed``/``_skipped`` markers, ``failed_phase``,
  ``status``. A bare orchestrator workflow id is accepted as a fallback so a
  CLI-launched run can be observed too.
- **Phase wall-clock** — ``state/runs/<params.run_id>/checkpoints/
  <phase>_checkpoint.json`` ``started_at``/``completed_at`` pairs (the same
  files ``BuildCostAggregator`` reads).
- **LLM usage** — the OP2 usage tap's ``state/runs/<params.run_id>/
  llm_usage.jsonl``. Reads are BOUNDED: an incremental accumulator remembers the
  byte offset per file and parses only appended rows (first attach parses at
  most ``_USAGE_FIRST_ATTACH_MAX_BYTES``), keeping the endpoint cheap at a
  2-5s poll. A sliding window over the most recent rows yields tokens/sec.
- **Seat** — a lightweight ``GET {base_url}/v1/models`` probe (the same probe
  ``lib.assistant.client.seat_is_serving`` uses) over the registered
  ``ED4ALL_SEAT_BASE_URLS`` seats, preferring the current phase's ``seats:``
  annotation. Results are TTL-cached; probing never raises and is skipped
  entirely for terminal runs.

Import contract: stdlib + ``gui.shared_state`` at module load; ``yaml`` /
``lib.paths`` are deferred (mirrors ``run_service``).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from gui import shared_state

logger = logging.getLogger("gui.progress_service")

# GUI-record / workflow-file statuses after which the run can no longer advance
# (the frontend stops polling; the rail renders statically with no pulse).
TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "canceled",
    "interrupted",
    "paused",
    "timeout",
    "error",
}

# Sliding-window width (seconds) for the tokens/sec stat.
USAGE_WINDOW_SECONDS = 180.0

# Bounded-read caps for llm_usage.jsonl (see module docstring).
_USAGE_FIRST_ATTACH_MAX_BYTES = 32 * 1024 * 1024
_USAGE_TAIL_ROWS = 500

# Seat probe: short timeout + TTL cache so a 2-5s poll never stacks probes.
_SEAT_PROBE_TIMEOUT_SECONDS = 0.75
_SEAT_PROBE_TTL_SECONDS = 5.0


# --------------------------------------------------------------------- paths


def _state_root() -> Path:
    """Resolve the ``state/`` root exactly like ``gui.shared_state`` does."""
    env_override = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_override:
        return Path(env_override).parent
    from lib.paths import STATE_PATH  # noqa: PLC0415

    return Path(STATE_PATH)


def _runs_dir() -> Path:
    """Resolve the ``state/runs/`` dir (checkpoints + usage tap live here)."""
    from lib.paths import get_state_runs_dir  # noqa: PLC0415

    return Path(get_state_runs_dir())


# --------------------------------------------------- workflow config (cached)


_WF_CONFIG_CACHE: Dict[str, Any] = {"mtime": None, "workflows": {}}
_WF_CONFIG_LOCK = threading.Lock()


def _workflows_config() -> Dict[str, Any]:
    """Parsed ``config/workflows.yaml`` ``workflows`` map, cached by mtime."""
    try:
        from lib.paths import PROJECT_ROOT  # noqa: PLC0415

        path = Path(PROJECT_ROOT) / "config" / "workflows.yaml"
        mtime = path.stat().st_mtime
    except Exception:  # noqa: BLE001 — missing config → empty plan, never raise
        return {}
    with _WF_CONFIG_LOCK:
        if _WF_CONFIG_CACHE["mtime"] == mtime:
            return _WF_CONFIG_CACHE["workflows"]
        try:
            import yaml  # noqa: PLC0415

            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            workflows = cfg.get("workflows") or {}
        except Exception:  # noqa: BLE001 — parse failure → keep last good copy
            logger.warning("workflows.yaml parse failed", exc_info=True)
            return _WF_CONFIG_CACHE["workflows"]
        _WF_CONFIG_CACHE["mtime"] = mtime
        _WF_CONFIG_CACHE["workflows"] = workflows
        return workflows


# Exact-name → presentation group. Names not listed fall through to the keyword
# rules below. Groups are a COARSE visual affordance (rail section labels), so
# an imperfect bucket for a future phase is cosmetic, never functional.
_EXACT_GROUPS: Dict[str, str] = {
    "semantik_conversion": "conversion",
    "dart_conversion": "conversion",  # legacy phase alias read-compat, legacy-token: allow
    "heading_judge": "conversion",
    "staging": "conversion",
    "chunking": "conversion",
    "extraction": "conversion",
    "objective_extraction": "planning",
    "source_mapping": "planning",
    "course_planning": "planning",
    "concept_extraction": "planning",
    "planning": "planning",
    "indexing": "planning",
    "packaging": "packaging",
    "imscc_chunking": "packaging",
    "trainforge_assessment": "archive",
    "training_synthesis": "archive",
    "libv2_archival": "archive",
    "vector_indexing": "archive",
    "finalization": "archive",
    "training": "generation",
}

# Ordered keyword fallbacks (first match wins). "validation" is checked before
# "generation" so inter_tier_validation / post_rewrite_validation land right.
_KEYWORD_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("validation", "validation"),
    ("conversion", "conversion"),
    ("generation", "generation"),
    ("synthesis", "generation"),
    ("assessment", "generation"),
    ("planning", "planning"),
    ("objective", "planning"),
    ("mapping", "planning"),
    ("concept", "planning"),
    ("packaging", "packaging"),
    ("chunking", "packaging"),
    ("archiv", "archive"),
    ("index", "archive"),
    ("final", "archive"),
    ("training", "archive"),
)


def phase_group(name: str) -> str:
    """Coarse presentation group for a phase name (never index-based)."""
    exact = _EXACT_GROUPS.get(name)
    if exact:
        return exact
    lowered = (name or "").lower()
    for needle, group in _KEYWORD_GROUPS:
        if needle in lowered:
            return group
    return "other"


def phase_plan(workflow: str) -> List[Dict[str, Any]]:
    """Ordered phase descriptors for ``workflow`` straight from config.

    Each entry: ``{name, index, group, optional, enabled_when_env}``. An
    unknown workflow returns ``[]`` (the caller degrades to whatever
    ``phase_outputs`` carries — never a fabricated list).
    """
    wf = _workflows_config().get(workflow)
    plan: List[Dict[str, Any]] = []
    if not isinstance(wf, dict):
        return plan
    for idx, ph in enumerate(wf.get("phases") or []):
        if not isinstance(ph, dict) or not ph.get("name"):
            continue
        name = str(ph["name"])
        plan.append(
            {
                "name": name,
                "index": idx,
                "group": phase_group(name),
                "optional": bool(ph.get("optional", False)),
                "enabled_when_env": ph.get("enabled_when_env"),
                "seats": [s for s in (ph.get("seats") or []) if isinstance(s, str)],
            }
        )
    return plan


def _parse_env_condition(condition: Any) -> Optional[Tuple[str, str, str]]:
    """Parse ``VAR=value`` / ``VAR!=value`` → ``(var, op, value)``; None else."""
    if not isinstance(condition, str) or not condition.strip():
        return None
    text = condition.strip()
    if "!=" in text:
        var, _, expected = text.partition("!=")
        return (var.strip(), "!=", expected.strip().lower())
    if "=" in text:
        var, _, expected = text.partition("=")
        return (var.strip(), "=", expected.strip().lower())
    return None


def _env_condition_enabled(
    condition: Any, verdicts: Optional[Dict[Tuple[str, str], bool]] = None
) -> Optional[bool]:
    """Resolve an ``enabled_when_env`` clause for a not-yet-reached phase.

    Resolution (most → least authoritative):
    1. ``verdicts`` — OBSERVED outcomes reconstructed from the run's own
       ``phase_outputs`` markers, keyed ``(var, value) → bool`` meaning
       "``var == value`` held in the RUN's environment". This is what makes a
       CLI-launched two-pass run render correctly even though the GUI process
       env differs from the run's env.
    2. The CURRENT process env (a best-effort prediction).

    Returns None when the clause is unparseable (no prediction). A stamped
    ``_skipped``/``_completed`` marker always wins over this at the call site.
    """
    parsed = _parse_env_condition(condition)
    if parsed is None:
        return None
    var, op, expected = parsed
    if verdicts is not None and (var, expected) in verdicts:
        held = verdicts[(var, expected)]
        return held if op == "=" else not held
    actual = os.environ.get(var, "").strip().lower()
    return (actual == expected) if op == "=" else (actual != expected)


def _observed_env_verdicts(
    plan: List[Dict[str, Any]], phase_outputs: Dict[str, Any]
) -> Dict[Tuple[str, str], bool]:
    """Reconstruct ``(var, value) → held?`` from the run's OWN phase markers.

    A condition-gated phase that the runner stamped ``_skipped`` proves its
    clause evaluated False in the run's env; one it genuinely ran (completed
    without ``_skipped``) proves the clause evaluated True. Both directions
    fold into a ``var == value`` truth so complementary clauses
    (``VAR=true`` vs ``VAR!=true``) resolve from one observation.
    """
    verdicts: Dict[Tuple[str, str], bool] = {}
    for ph in plan:
        parsed = _parse_env_condition(ph.get("enabled_when_env"))
        if parsed is None:
            continue
        var, op, expected = parsed
        out = phase_outputs.get(ph["name"])
        if not isinstance(out, dict) or not out.get("_completed"):
            continue
        if out.get("_skipped"):
            clause_held = False
        else:
            clause_held = True
        # clause_held is about (var op value); normalize to "var == value".
        verdicts[(var, expected)] = clause_held if op == "=" else not clause_held
    return verdicts


# ----------------------------------------------------------- workflow state


def _read_workflow_state(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Read ``state/workflows/<workflow_id>.json`` (bounded, never raises)."""
    if not workflow_id:
        return None
    path = _state_root() / "workflows" / f"{workflow_id}.json"
    try:
        if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — mid-write / corrupt → treat as absent
        return None
    return payload if isinstance(payload, dict) else None


def _parse_dt(value: Any) -> Optional[datetime]:
    """ISO timestamp → naive datetime in the writer's own frame.

    Checkpoint timestamps are written naive (host-local); usage timestamps are
    tz-aware UTC. We normalize aware values to naive-UTC and leave naive values
    alone — elapsed math always subtracts a "now" taken in the SAME frame.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _now_for(dt: datetime, aware_source: bool) -> datetime:
    """A comparable 'now' for a parsed timestamp (see ``_parse_dt``)."""
    del dt
    if aware_source:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    return datetime.now()


def _read_checkpoints(orchestrator_run_id: str) -> Dict[str, Dict[str, Any]]:
    """``{phase_name: {started_at, completed_at, wallclock_s}}`` from checkpoints."""
    result: Dict[str, Dict[str, Any]] = {}
    if not orchestrator_run_id:
        return result
    ckpt_dir = _runs_dir() / orchestrator_run_id / "checkpoints"
    if not ckpt_dir.is_dir():
        return result
    try:
        files = sorted(ckpt_dir.glob("*_checkpoint.json"))
    except OSError:
        return result
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — one bad checkpoint never breaks all
            continue
        if not isinstance(payload, dict):
            continue
        phase = str(payload.get("phase_name") or f.stem.replace("_checkpoint", ""))
        started = _parse_dt(payload.get("started_at"))
        completed = _parse_dt(payload.get("completed_at"))
        wallclock: Optional[float] = None
        if started is not None and completed is not None:
            delta = (completed - started).total_seconds()
            if delta >= 0:
                wallclock = round(delta, 1)
        result[phase] = {
            "status": str(payload.get("status") or "").lower(),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "wallclock_s": wallclock,
        }
    return result


# ------------------------------------------------------------- usage stats


class _UsageAccumulator:
    """Incremental, bounded reader over one append-only ``llm_usage.jsonl``.

    Remembers the byte offset of the last COMPLETE line parsed plus running
    totals, so a 2-5s poll re-reads only appended bytes. A shrunk file
    (rotation / new run reusing the id) resets the accumulator. First attach to
    an already-large file parses at most ``_USAGE_FIRST_ATTACH_MAX_BYTES`` from
    the tail (totals then cover the parsed span — the honest bound, and in
    practice usage files are far below the cap).
    """

    def __init__(self) -> None:
        self.offset = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=_USAGE_TAIL_ROWS)

    def ingest(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.__init__()  # rotated / truncated — start over
        if size == self.offset:
            return
        start = self.offset
        if start == 0 and size > _USAGE_FIRST_ATTACH_MAX_BYTES:
            start = size - _USAGE_FIRST_ATTACH_MAX_BYTES
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                blob = fh.read(size - start)
        except OSError:
            return
        # If we seeked into the middle of a row, drop the partial head line.
        if start > 0 and self.offset == 0:
            nl = blob.find(b"\n")
            if nl < 0:
                return
            start += nl + 1
            blob = blob[nl + 1 :]
        # Only consume up to the last complete line (the tail may be mid-append).
        last_nl = blob.rfind(b"\n")
        if last_nl < 0:
            return
        consumed = blob[: last_nl + 1]
        self.offset = start + last_nl + 1
        for raw in consumed.split(b"\n"):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001 — skip one bad row
                continue
            if not isinstance(row, dict):
                continue
            prompt = _as_int(row.get("prompt_tokens"))
            completion = _as_int(row.get("completion_tokens"))
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.recent.append(
                {
                    "ts": row.get("ts"),
                    "completion_tokens": completion,
                    "duration_ms": _as_float(row.get("duration_ms")),
                    "ttft_ms": _as_float(row.get("ttft_ms")) if "ttft_ms" in row else None,
                }
            )


_USAGE_ACCUMULATORS: Dict[str, _UsageAccumulator] = {}
_USAGE_LOCK = threading.Lock()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ts_epoch(value: Any) -> Optional[float]:
    dt = None
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def usage_window_stats(
    rows: List[Dict[str, Any]],
    *,
    now_ts: Optional[float] = None,
    window_s: float = USAGE_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """Sliding-window throughput over recent usage rows (pure; unit-tested).

    ``rows``: ``[{ts, completion_tokens, duration_ms, ttft_ms?}, ...]`` (oldest
    first). Window membership: the row's completion ``ts`` within the last
    ``window_s`` seconds of ``now_ts``. Tokens/sec is GENERATION throughput —
    completion tokens over the calls' own generation seconds (``duration_ms``),
    which stays meaningful across multi-minute calls; when the window rows
    carry no duration it degrades to tokens over the observed ts span.

    Returns ``{tok_s: float|None, ttft_p50_ms: float|None, window_calls: int}``
    — ``tok_s`` is None when no row falls inside the window.
    """
    now = time.time() if now_ts is None else float(now_ts)
    floor = now - max(1.0, float(window_s))
    window: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        ts = _ts_epoch(row.get("ts"))
        if ts is None or ts < floor or ts > now + 60.0:
            continue
        window.append((ts, row))

    tok_s: Optional[float] = None
    if window:
        completion = sum(_as_int(r.get("completion_tokens")) for _, r in window)
        gen_seconds = sum(max(0.0, _as_float(r.get("duration_ms"))) for _, r in window) / 1000.0
        if gen_seconds > 0:
            tok_s = round(completion / gen_seconds, 1)
        else:
            span = now - min(ts for ts, _ in window)
            tok_s = round(completion / span, 1) if span > 0 else None

    ttft_samples = [
        _as_float(r.get("ttft_ms"))
        for r in rows
        if r.get("ttft_ms") is not None and _as_float(r.get("ttft_ms")) >= 0
    ]
    ttft_p50 = round(statistics.median(ttft_samples), 1) if ttft_samples else None
    return {"tok_s": tok_s, "ttft_p50_ms": ttft_p50, "window_calls": len(window)}


def _usage_stats(orchestrator_run_id: str) -> Dict[str, Any]:
    """Totals + windowed throughput for a run's ``llm_usage.jsonl``."""
    empty = {
        "tok_s": None,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ttft_p50_ms": None,
    }
    if not orchestrator_run_id:
        return empty
    path = _runs_dir() / orchestrator_run_id / "llm_usage.jsonl"
    if not path.is_file():
        return empty
    with _USAGE_LOCK:
        acc = _USAGE_ACCUMULATORS.setdefault(str(path), _UsageAccumulator())
        acc.ingest(path)
        rows = list(acc.recent)
        calls, prompt, completion = acc.calls, acc.prompt_tokens, acc.completion_tokens
    window = usage_window_stats(rows)
    return {
        "tok_s": window["tok_s"],
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_p50_ms": window["ttft_p50_ms"],
    }


# --------------------------------------------------------------- seat probe


_SEAT_PROBE_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}
_SEAT_PROBE_LOCK = threading.Lock()


def _probe_seat_model(base_url: str) -> Optional[str]:
    """``GET {base_url}/v1/models`` → first served model id; None when down.

    The same lightweight liveness probe ``lib.assistant.client.seat_is_serving``
    uses, extended to parse the model id out of the 2xx body. TTL-cached
    (positive AND negative) so a 2-5s poll costs at most one probe per seat per
    ``_SEAT_PROBE_TTL_SECONDS``. Never raises.
    """
    url = f"{str(base_url).rstrip('/')}/v1/models"
    now = time.monotonic()
    with _SEAT_PROBE_LOCK:
        cached = _SEAT_PROBE_CACHE.get(url)
        if cached and cached[0] > now:
            return cached[1]
    model: Optional[str] = None
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_SEAT_PROBE_TIMEOUT_SECONDS) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if 200 <= int(code) < 300:
                body = json.loads(resp.read(65536).decode("utf-8"))
                data = body.get("data") if isinstance(body, dict) else None
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    model = str(data[0].get("id") or "") or None
                else:
                    model = ""  # serving, model id unknown
    except Exception:  # noqa: BLE001 — down / refused / timeout → not serving
        model = None
    with _SEAT_PROBE_LOCK:
        _SEAT_PROBE_CACHE[url] = (now + _SEAT_PROBE_TTL_SECONDS, model)
    return model


def _serving_seat(phase_seats: List[str]) -> Optional[Dict[str, Any]]:
    """First registered seat answering ``/v1/models``; phase seats first.

    Registry = ``ED4ALL_SEAT_BASE_URLS`` via the canonical
    ``lib.vllm_container_lifecycle.parse_seat_registry`` (a new seat is a
    registry entry, never code). Empty registry → None (no probing at all).
    """
    try:
        from lib.vllm_container_lifecycle import parse_seat_registry  # noqa: PLC0415

        registry = parse_seat_registry()
    except Exception:  # noqa: BLE001 — registry parse is best-effort
        return None
    if not registry:
        return None
    ordered: List[Tuple[str, str]] = []
    for name in phase_seats:
        url = registry.get(name)
        if url:
            ordered.append((name, url))
    for name, url in registry.items():
        if (name, url) not in ordered:
            ordered.append((name, url))
    for name, url in ordered:
        model = _probe_seat_model(url)
        if model is not None:
            return {"name": name, "url": url, "model": model or None}
    return None


# ------------------------------------------------------------ the main merge


def run_progress(run_id: str) -> Optional[Dict[str, Any]]:
    """Build the ``/api/runs/{run_id}/progress`` payload; None → 404.

    ``run_id`` is a GUI run id (``state/gui/runs/``); a bare orchestrator
    workflow id (``state/workflows/<id>.json``) is accepted as a fallback so
    CLI-launched runs are observable too.
    """
    record = shared_state.read_run(run_id)
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    status: Optional[str] = None
    if record is not None:
        workflow_id = record.get("workflow_id")
        workflow_name = record.get("workflow")
        status = record.get("status")
        if not workflow_id:
            return None  # phase-only / failed-at-launch runs have no pipeline
    else:
        workflow_id = run_id

    state = _read_workflow_state(str(workflow_id))
    if state is None and record is None:
        return None

    params = (state or {}).get("params") or {}
    if not workflow_name:
        workflow_name = str((state or {}).get("type") or "")
    if not status:
        status = str((state or {}).get("status") or "unknown").lower()
    status = str(status).lower()
    is_terminal = status in TERMINAL_STATUSES

    phase_outputs = (state or {}).get("phase_outputs") or {}
    if not isinstance(phase_outputs, dict):
        phase_outputs = {}
    failed_phase = (state or {}).get("failed_phase") or (record or {}).get("failed_phase")
    failure_reason = (state or {}).get("failure_reason") or (record or {}).get(
        "failure_reason"
    )

    plan = phase_plan(str(workflow_name))
    if not plan:
        # Unknown workflow: degrade to the observed phase_outputs order (real
        # data only — never invent a canonical list).
        plan = [
            {
                "name": name,
                "index": idx,
                "group": phase_group(name),
                "optional": False,
                "enabled_when_env": None,
                "seats": [],
            }
            for idx, name in enumerate(phase_outputs.keys())
        ]

    orchestrator_run_id = str(params.get("run_id") or "")
    checkpoints = _read_checkpoints(orchestrator_run_id)

    # --- stop_after cut: phases strictly after the halt phase never run.
    stop_after = params.get("stop_after")
    stop_after_index: Optional[int] = None
    if isinstance(stop_after, str) and stop_after:
        for ph in plan:
            if ph["name"] == stop_after:
                stop_after_index = ph["index"]
                break

    skip_training = bool(params.get("skip_training"))
    generate_assessments = params.get("generate_assessments")

    # Env-clause truths OBSERVED from the run's own markers (see helper).
    # Suppressed for Courseforge stage-subcommand runs, whose skips are
    # whitelist-driven (courseforge_stage), not env-driven.
    is_stage_run = bool(params.get("courseforge_stage"))
    env_verdicts = (
        {} if is_stage_run else _observed_env_verdicts(plan, phase_outputs)
    )

    run_failed = status in {"failed", "error", "timeout"}
    resolved: Dict[str, Optional[str]] = {}
    checkpoint_current: Optional[str] = None
    for ph in plan:
        name = ph["name"]
        out = phase_outputs.get(name)
        out = out if isinstance(out, dict) else {}
        ckpt = checkpoints.get(name) or {}
        state_token: Optional[str] = None
        if out.get("_skipped"):
            state_token = "skipped"
        elif out.get("_completed"):
            state_token = "done"
        elif ckpt.get("status") == "completed":
            # The checkpoint completes before the workflow state file re-saves
            # (phase-boundary persistence) — trust the checkpoint's verdict.
            state_token = "done"
        elif failed_phase == name and run_failed:
            state_token = "failed"
        elif stop_after_index is not None and ph["index"] > stop_after_index:
            state_token = "skipped"
        elif skip_training and name == "training_synthesis":
            state_token = "skipped"
        elif generate_assessments is False and name in (
            "assessment_synthesis",
            "trainforge_assessment",
        ):
            state_token = "skipped"
        else:
            enabled = _env_condition_enabled(ph.get("enabled_when_env"), env_verdicts)
            if enabled is False:
                state_token = "skipped"
        if (
            state_token is None
            and checkpoint_current is None
            and ckpt.get("status") == "started"
            and not ckpt.get("completed_at")
        ):
            # An in-flight checkpoint is the AUTHORITATIVE current-phase signal
            # (stamped at phase start, before any GUI-visible state re-save).
            checkpoint_current = name
        resolved[name] = state_token

    phases: List[Dict[str, Any]] = []
    current_phase: Optional[str] = None
    for ph in plan:
        name = ph["name"]
        state_token = resolved.get(name)
        if state_token is None:
            if not is_terminal and current_phase is None and (
                checkpoint_current is None or checkpoint_current == name
            ):
                state_token = "current"
                current_phase = name
            else:
                state_token = "pending"
        ckpt = checkpoints.get(name) or {}
        phases.append(
            {
                "name": name,
                "index": ph["index"],
                "state": state_token,
                "group": ph["group"],
                "label": _phase_label(name),
                "wallclock_s": ckpt.get("wallclock_s"),
            }
        )

    # --- elapsed time in the current phase. Anchor preference: the current
    # phase's own in-flight checkpoint ``started_at`` (stamped at phase start)
    # → the latest completed checkpoint (the previous phase's seam) → the
    # workflow's started_at.
    phase_elapsed_s: Optional[float] = None
    if current_phase is not None:
        raw_anchor: Any = None
        cur_ckpt = checkpoints.get(current_phase) or {}
        if cur_ckpt.get("started_at"):
            raw_anchor = cur_ckpt["started_at"]
        else:
            latest: Optional[datetime] = None
            for ckpt in checkpoints.values():
                done = _parse_dt(ckpt.get("completed_at"))
                if done is not None and (latest is None or done > latest):
                    latest = done
                    raw_anchor = ckpt.get("completed_at")
            if latest is None:
                raw_anchor = (state or {}).get("started_at") or (record or {}).get(
                    "started_at"
                )
        anchor = _parse_dt(raw_anchor)
        if anchor is not None:
            aware = isinstance(raw_anchor, str) and (
                "+" in raw_anchor or raw_anchor.endswith("Z")
            )
            delta = (_now_for(anchor, aware) - anchor).total_seconds()
            if delta >= 0:
                phase_elapsed_s = round(delta, 1)

    stats = _usage_stats(orchestrator_run_id)
    stats["phase_elapsed_s"] = phase_elapsed_s
    if is_terminal:
        stats["seat"] = None
    else:
        current_seats = next(
            (ph["seats"] for ph in plan if ph["name"] == current_phase), []
        )
        stats["seat"] = _serving_seat(list(current_seats or []))

    return {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow": workflow_name,
        "status": status,
        "phases": phases,
        "current_phase": current_phase,
        "failed_phase": failed_phase if run_failed else None,
        "failure_reason": failure_reason if run_failed else None,
        "stats": stats,
        "updated_at": shared_state.now_iso(),
    }


def _phase_label(name: str) -> str:
    """Friendly label via the canonical run_service map (fallback: the id)."""
    try:
        from gui.services.run_service import PHASE_LABELS  # noqa: PLC0415

        return PHASE_LABELS.get(name, "") or name
    except Exception:  # noqa: BLE001
        return name


__all__ = [
    "TERMINAL_STATUSES",
    "USAGE_WINDOW_SECONDS",
    "phase_group",
    "phase_plan",
    "run_progress",
    "usage_window_stats",
]
