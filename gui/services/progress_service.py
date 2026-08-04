"""Run-progress service — the data behind ``GET /api/runs/{run_id}/progress``.

Feeds the Studio stage-tracker rail + live stats band. Everything here is a
READ-ONLY merge of artifacts other subsystems already write; this module never
mutates run state:

- **Phase plan** — the run's workflow phase list from ``config/workflows.yaml``
  (cached by file mtime; NEVER a hardcoded phase list, so ``course_generation``
  / ``rag_training`` / ``trainforge_train`` render as correctly as
  ``textbook_to_course``). Each phase gets a coarse presentation ``group``
  (conversion / planning / generation / validation / packaging / archive /
  training / finalization) derived from name rules, never from phase indices.
- **Run state** — the GUI run record (``runtime/state/gui/runs/<run_id>.json``) resolved
  to its orchestrator workflow state (``runtime/state/workflows/<workflow_id>.json``):
  ``phase_outputs`` ``_completed``/``_skipped`` markers, ``failed_phase``,
  ``status``. A bare orchestrator workflow id is accepted as a fallback so a
  CLI-launched run can be observed too.
- **Phase wall-clock** — ``runtime/state/runs/<params.run_id>/checkpoints/
  <phase>_checkpoint.json`` ``started_at``/``completed_at`` pairs (the same
  files ``BuildCostAggregator`` reads).
- **LLM usage** — the OP2 usage tap's ``runtime/state/runs/<params.run_id>/
  llm_usage.jsonl``. Reads are BOUNDED: an incremental accumulator remembers the
  byte offset per file and parses only appended rows (first attach parses at
  most ``_USAGE_FIRST_ATTACH_MAX_BYTES``), keeping the endpoint cheap at a
  2-5s poll. A sliding window over the most recent rows yields the band's
  headline ``tok_s`` (AGGREGATE seat throughput: window tokens over the wall
  span) plus ``streams`` (estimated in-flight request count); the per-stream
  figure lives in ``stats.detail.throughput.per_stream_tok_s``.
  The SAME single-pass accumulator also feeds ``stats.detail`` — the full
  comprehensive matrix (run totals, cumulative avg tok/s, TTFT p50/p95,
  duration mean/median, ``finish_reason == "length"`` truncation +
  ``stream_usage_present: false`` health tripwires, a per-(provider, model)
  breakdown, and per-phase token attribution bucketed against the checkpoint
  wall-clock windows with an explicit ``unattributed`` bucket for rows outside
  every window — never guessed into a phase). Frame note for the attribution
  join: usage ``ts`` is tz-aware UTC, checkpoint times are naive host-local;
  both are joined in EPOCH space (``_ts_epoch`` / ``_wall_epoch``), the same
  per-writer-frame discipline as ``_parse_dt``/``_now_for``.
- **VRAM** — ``runtime/state/runs/<params.run_id>/vram_trajectory.jsonl`` (the
  ``ED4ALL_VRAM_DOCTOR`` output; row shape per
  ``lib/llm/vram_doctor.py::append_trajectory_row``). ``stats.detail.vram``
  surfaces the latest sample (``free_mib``/``total_mib``/phase boundary) + the
  sample count; the whole section is omitted when the file is absent.
- **Seat** — a lightweight ``GET {base_url}/v1/models`` probe (the same probe
  ``lib.assistant.client.seat_is_serving`` uses) over the registered
  ``ED4ALL_SEAT_BASE_URLS`` seats, preferring the current phase's ``seats:``
  annotation. Results are TTL-cached; probing never raises and is skipped
  entirely for terminal runs.
- **Per-phase unit progress** — the pipeline's OWN per-unit resume checkpoint
  sidecars (the ``ED4ALL_GENERATION_CHECKPOINT`` family: one appended JSONL row
  per completed unit, e.g. the outline tier's
  ``01_outline/.blocks_outline_checkpoint.jsonl``). ``stats.phase_units``
  surfaces a bounded newline count of the CURRENT phase's sidecar, resolved
  from the run's real state (``project_path`` / chunkset paths out of
  ``phase_outputs`` — never a hardcoded export pattern). Honesty contract: no
  sidecar on disk → the field is omitted entirely; no totals are ever
  estimated. Counting is incremental (size-keyed cache; only appended bytes
  are re-read) so a 3s poll stays cheap on a tens-of-MB sidecar.

- **Live output tail** — ``output_tail`` (behind ``GET
  /api/runs/{run_id}/output-tail``) returns the last few complete rows of the
  CURRENT phase's per-unit resume sidecar as bounded display records
  (HTML-stripped, hard-truncated) so the build page can show real pipeline
  output scrolling by. Absent sidecar → ``rows: []``, never fabricated.

Import contract: stdlib + ``gui.shared_state`` at module load; ``yaml`` /
``lib.paths`` are deferred (mirrors ``run_service``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from gui import shared_state
from gui.services import liveness

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

# A resume RE-STAMP writes a checkpoint whose ``started_at`` and ``completed_at``
# are IDENTICAL (an exact 0.0s span that measures the restore, not the original
# execution). That one signature is suppressed (see ``run_progress``) rather
# than rendered as "0s". Every genuinely-executed phase keeps its real span,
# however small — a fast deterministic phase (staging / source_mapping complete
# in ~0.1-0.3s) must show an honest "<1s", never a blank. No magnitude floor:
# an arbitrary sub-second threshold could not distinguish a genuinely-fast phase
# from a restore, and the owner contract is "never blank" for fast phases.

# Bounded-read caps for llm_usage.jsonl (see module docstring).
_USAGE_FIRST_ATTACH_MAX_BYTES = 32 * 1024 * 1024
_USAGE_TAIL_ROWS = 500

# Seat probe: short timeout + TTL cache so a 2-5s poll never stacks probes.
_SEAT_PROBE_TIMEOUT_SECONDS = 0.75
_SEAT_PROBE_TTL_SECONDS = 5.0

# Atomic phase snapshots live below the run-scoped telemetry directory.  The
# filename is derived from a phase name already present in the workflow plan;
# arbitrary paths from a snapshot are never followed.
_PHASE_TELEMETRY_SCHEMA_VERSION = 2
_PHASE_TELEMETRY_STATES = frozenset({"running", "paused", "failed", "complete"})
_SAFE_PHASE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


# --------------------------------------------------------------------- paths


def _state_root() -> Path:
    """Resolve the ``runtime/state/`` root exactly like ``gui.shared_state`` does."""
    env_override = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_override:
        return Path(env_override).parent
    from lib.paths import STATE_PATH  # noqa: PLC0415

    return Path(STATE_PATH)


def _runs_dir() -> Path:
    """Resolve the ``runtime/state/runs/`` dir (checkpoints + usage tap live here)."""
    from lib.paths import get_state_runs_dir  # noqa: PLC0415

    return Path(get_state_runs_dir())


def _read_phase_telemetry(
    orchestrator_run_id: str, phase_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Read valid, run-bound atomic phase telemetry snapshots.

    Snapshots are optional presentation evidence.  A missing, partially
    replaced, malformed, wrong-version, wrong-run, or wrong-phase document is
    ignored wholesale so telemetry can never break the progress endpoint or
    leak metrics between runs.
    """
    if not orchestrator_run_id:
        return {}
    root = _runs_dir() / orchestrator_run_id / "telemetry"
    found: Dict[str, Dict[str, Any]] = {}
    for phase_name in phase_names:
        if not _SAFE_PHASE_NAME.fullmatch(phase_name):
            continue
        path = root / f"{phase_name}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != _PHASE_TELEMETRY_SCHEMA_VERSION
            or raw.get("phase") != phase_name
            or str(raw.get("run_id") or "") != orchestrator_run_id
            or raw.get("state") not in _PHASE_TELEMETRY_STATES
            or raw.get("gate_readiness") not in {"pending", "blocked", "ready"}
            or not isinstance(raw.get("backpressure"), bool)
            or not isinstance(raw.get("stop_requested"), bool)
            or not isinstance(raw.get("provider"), str)
            or not isinstance(raw.get("model"), str)
        ):
            continue
        # The writer contract uses non-negative numeric counters/rates.  Reject
        # a whole corrupt snapshot rather than selectively displaying a
        # plausible-looking subset.
        numeric_keys = (
            "elapsed_seconds",
            "total_units",
            "completed_units",
            "terminal_units",
            "accepted_count",
            "rejected_count",
            "transient_count",
            "sft_count",
            "dpo_count",
            "provider_results",
            "cached_replays",
            "transient_attempts",
            "recovered_units",
            "exhausted_units",
            "fatal_units",
            "active_workers",
            "max_concurrent",
            "queued_units",
            "in_flight",
            "throughput_units_per_second",
            "eta_seconds",
        )
        malformed = False
        for key in numeric_keys:
            value = raw.get(key)
            if key not in {"throughput_units_per_second", "eta_seconds"} and value is None:
                malformed = True
                break
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                malformed = True
                break
        reasons = raw.get("rejection_reasons")
        if (
            malformed
            or raw["completed_units"] > raw["total_units"]
            or raw["terminal_units"] > raw["completed_units"]
            or raw["max_concurrent"] < 1
            or raw["active_workers"] > raw["max_concurrent"]
            or raw["in_flight"] > raw["max_concurrent"]
            or raw["active_workers"] + raw["queued_units"]
            != raw["in_flight"]
            or not isinstance(reasons, dict)
            or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in reasons.items()
            )
        ):
            continue
        found[phase_name] = raw
    return found


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
#
# ORDERING CONTRACT: the rail BUCKETS phases by group in FIRST-OCCURRENCE order
# (stage-rail.js::renderRail), so a group's section renders wherever its
# EARLIEST member phase sits in the plan — regardless of where its other
# members are. A group must therefore only span phases that are contiguous in
# execution order, or it drags later phases visually backwards. That is exactly
# why the post-build training tail (training / post_training_validation /
# evaluation) is its OWN group and `finalization` is its own group: build→
# training is one sequenced pipeline whose tail now runs AFTER the archive
# phases, and `finalization` is genuinely last. Folding either back into
# "archive" (whose first member is trainforge_assessment) would render them
# inside that earlier section, i.e. visually BEFORE training.
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
    # Post-build training tail — one sequenced pipeline, so these render as
    # their own section AFTER archive rather than folding into an earlier one.
    "training": "training",
    "post_training_validation": "training",
    "evaluation": "training",
    "finalization": "finalization",
    # Owner design (group consolidation): ONE "generation" section spans the
    # whole authoring slice — the single-pass phase, both two-pass tiers, the
    # inter-tier validators between them, and assessment synthesis — so the
    # rail renders conversion → planning → generation → validation →
    # packaging → archive → training → finalization with each header exactly
    # once instead of interleaving generation/validation headers.
    # post_rewrite_validation is the "validation" section.
    "content_generation": "generation",
    "content_generation_outline": "generation",
    "inter_tier_validation": "generation",
    "content_generation_rewrite": "generation",
    "assessment_synthesis": "generation",
    "post_rewrite_validation": "validation",
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
    # A future evaluation/training-tail phase name lands in the post-build
    # training section, and anything "final" in its own trailing section —
    # mirroring the exact map above so a fallback never renders a late phase
    # inside an earlier group.
    ("eval", "training"),
    ("training", "training"),
    ("final", "finalization"),
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

    ONLY from evidence: ``verdicts`` are OBSERVED outcomes reconstructed from
    the run's own ``phase_outputs`` markers, keyed ``(var, value) → bool``
    meaning "``var == value`` held in the RUN's environment". This is what
    makes a CLI-launched two-pass run render correctly even though the GUI
    process env differs from the run's env.

    Returns None when the clause is unparseable OR no observation exists yet
    (no prediction). The serving process env is deliberately NOT consulted:
    the run's env routinely differs from the GUI's (owner-verified
    inversion: a live two-pass run rendered its tiers "skipped" off the GUI
    env), so with no observational evidence an env-conditional phase must
    render "pending", never "skipped". A stamped ``_skipped``/``_completed``
    marker always wins over this at the call site.
    """
    parsed = _parse_env_condition(condition)
    if parsed is None:
        return None
    var, op, expected = parsed
    if verdicts is not None and (var, expected) in verdicts:
        held = verdicts[(var, expected)]
        return held if op == "=" else not held
    return None


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
    """Read ``runtime/state/workflows/<workflow_id>.json`` (bounded, never raises)."""
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
        # Prefer the CUMULATIVE figure the checkpoint now carries. started_at
        # is rewritten on every resume, so its span measures only the final
        # segment — on a heavily-resumed build that rendered a multi-hour
        # conversion as "1s". Legacy checkpoints predate the field and fall
        # back to the single-segment span.
        cumulative = payload.get("elapsed_seconds")
        if isinstance(cumulative, (int, float)) and cumulative > 0:
            wallclock = round(float(cumulative), 1)
        elif started is not None and completed is not None:
            delta = (completed - started).total_seconds()
            if delta >= 0:
                wallclock = round(delta, 1)
        segments = payload.get("segments")
        result[phase] = {
            "status": str(payload.get("status") or "").lower(),
            "started_at": payload.get("first_started_at") or payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "wallclock_s": wallclock,
            "segments": int(segments) if isinstance(segments, int) and segments > 0 else 1,
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
        # ---- detail-matrix accumulation (same single pass; see stats.detail).
        # Whole-run samples for latency percentiles / means. Only rows that
        # actually CARRY the field contribute (no zero-defaults skewing means).
        self.durations_ms: List[float] = []
        self.ttfts_ms: List[float] = []
        # Health tripwires straight off the tap's row contract.
        self.truncated_calls = 0  # finish_reason == "length"
        self.usage_missing_calls = 0  # stream_usage_present == False
        # (provider, model) → [calls, prompt_tokens, completion_tokens].
        self.by_model: Dict[Tuple[str, str], List[int]] = {}
        # (ts_epoch|None, prompt, completion) per row, for per-phase
        # attribution against the checkpoint windows (re-bucketed per poll —
        # windows move while the run advances, the rows never do).
        # (explicit_phase|None, ts_epoch|None, prompt, completion) per row.
        # Newer taps stamp the spending phase directly; that writer-owned
        # identity is stronger evidence than a checkpoint wall-clock join.
        # Older rows without it retain the temporal fallback.
        self.rows_ts: List[Tuple[Optional[str], Optional[float], int, int]] = []
        # ---- page-metered rows (SemantiK GLM-OCR conversion lane): those
        # rows meter HONEST ZERO tokens and carry their real progress in a
        # positive-int ``pages`` field + ``duration_ms`` instead. Aggregated
        # here so the band can show page progress during conversion.
        self.pages = 0  # total pages across page-bearing rows
        self.pages_rows = 0  # count of page-bearing rows
        self.pages_duration_ms = 0.0  # summed duration_ms of page rows only
        # Wall-span endpoints over the page rows' OWN timestamps. The tap
        # stamps ``ts`` at row COMPLETION (see usage_window_stats), so a
        # row's start is ``ts − duration``; the span is earliest start →
        # latest completion. None until a page row carries a parseable ts.
        self.pages_first_start_ts: Optional[float] = None
        self.pages_last_end_ts: Optional[float] = None

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
            if "duration_ms" in row:
                dur = _as_float(row.get("duration_ms"))
                if dur >= 0:
                    self.durations_ms.append(dur)
            if row.get("ttft_ms") is not None:
                ttft = _as_float(row.get("ttft_ms"))
                if ttft >= 0:
                    self.ttfts_ms.append(ttft)
            if row.get("finish_reason") == "length":
                self.truncated_calls += 1
            if row.get("stream_usage_present") is False:
                self.usage_missing_calls += 1
            key = (str(row.get("provider") or ""), str(row.get("model") or ""))
            bucket = self.by_model.setdefault(key, [0, 0, 0])
            bucket[0] += 1
            bucket[1] += prompt
            bucket[2] += completion
            row_ts = _ts_epoch(row.get("ts"))
            raw_phase = row.get("phase")
            explicit_phase = (
                str(raw_phase).strip()
                if isinstance(raw_phase, str) and raw_phase.strip()
                else None
            )
            self.rows_ts.append((explicit_phase, row_ts, prompt, completion))
            # Page-metered row: a positive int ``pages`` (bool excluded — a
            # JSON ``true`` must never count as one page). Garbage / missing /
            # non-positive values simply don't contribute.
            pages_val = row.get("pages")
            pages_n = 0 if isinstance(pages_val, bool) else _as_int(pages_val)
            if pages_n > 0:
                self.pages += pages_n
                self.pages_rows += 1
                page_dur = _as_float(row.get("duration_ms"))
                if page_dur > 0:
                    self.pages_duration_ms += page_dur
                if row_ts is not None:
                    start = row_ts - max(page_dur, 0.0) / 1000.0
                    if self.pages_first_start_ts is None or start < self.pages_first_start_ts:
                        self.pages_first_start_ts = start
                    if self.pages_last_end_ts is None or row_ts > self.pages_last_end_ts:
                        self.pages_last_end_ts = row_ts
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
    ``window_s`` seconds of ``now_ts``.

    Three throughput figures, each honestly named:

    - ``tok_s`` — AGGREGATE seat throughput: window completion tokens over the
      window's WALL span (first in-window row ``ts`` → now). This is what the
      seat as a whole produces per second, regardless of how many requests it
      is serving concurrently. Degrades to per-stream when the wall span is
      zero (every row just landed); None when no row falls inside the window.
    - ``per_stream_tok_s`` — completion tokens over the calls' own summed
      generation seconds (``duration_ms``): what ONE request experiences.
      Under N-way concurrency this reads ~aggregate/N — the number the band
      used to (misleadingly) headline. None when no in-window row carries a
      duration.
    - ``streams`` — estimated in-flight request count over the window:
      sum(duration_ms)/1000 over the wall span (Little's law; counts
      queued+decoding requests). None when not computable (no rows, no
      durations, or zero wall span).
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
    per_stream: Optional[float] = None
    streams: Optional[float] = None
    if window:
        completion = sum(_as_int(r.get("completion_tokens")) for _, r in window)
        gen_seconds = sum(max(0.0, _as_float(r.get("duration_ms"))) for _, r in window) / 1000.0
        if gen_seconds > 0:
            per_stream = round(completion / gen_seconds, 1)
        span = now - min(ts for ts, _ in window)
        if span > 0:
            tok_s = round(completion / span, 1)
            if gen_seconds > 0:
                streams = round(gen_seconds / span, 1)
        elif gen_seconds > 0:
            tok_s = per_stream  # zero wall span → best honest figure available

    ttft_samples = [
        _as_float(r.get("ttft_ms"))
        for r in rows
        if r.get("ttft_ms") is not None and _as_float(r.get("ttft_ms")) >= 0
    ]
    ttft_p50 = round(statistics.median(ttft_samples), 1) if ttft_samples else None
    return {
        "tok_s": tok_s,
        "per_stream_tok_s": per_stream,
        "streams": streams,
        "ttft_p50_ms": ttft_p50,
        "window_calls": len(window),
    }


def _pages_per_hr(
    pages: int,
    pages_rows: int,
    summed_duration_ms: float,
    first_start_ts: Optional[float],
    last_end_ts: Optional[float],
) -> Optional[int]:
    """Honest pages/hr over the page-metered rows (pure; unit-tested).

    Page batches run CONCURRENTLY, so a rate over the rows' SUMMED durations
    UNDERSTATES real wall-clock throughput by roughly the concurrency factor.
    Preferred denominator is therefore the WALL SPAN of the page rows' own
    timestamps — earliest start (``ts − duration``; the tap stamps ``ts`` at
    completion) → latest completion — falling back to the summed durations
    only when no page row carried a parseable ``ts``. Both denominators come
    from the rows themselves, never this process's wall clock (an attached
    viewer must not dilute a finished lane's rate). None when nothing is
    honestly computable.
    """
    if pages_rows <= 0 or pages <= 0:
        return None
    span_s: Optional[float] = None
    if first_start_ts is not None and last_end_ts is not None:
        wall = last_end_ts - first_start_ts
        if wall > 0:
            span_s = wall
    if span_s is None and summed_duration_ms > 0:
        span_s = summed_duration_ms / 1000.0
    if span_s is None or span_s <= 0:
        return None
    return round(pages / (span_s / 3600.0))


def _usage_stats(orchestrator_run_id: str) -> Dict[str, Any]:
    """Totals + windowed throughput for a run's ``llm_usage.jsonl``."""
    # Backward-compatible shape: the page-metered keys are ALWAYS present
    # (None/0 defaults), so consumers never key-error on an old-style run.
    empty = {
        "tok_s": None,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ttft_p50_ms": None,
        "pages": 0,
        "pages_rows": 0,
        "pages_per_hr": None,
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
        pages, pages_rows = acc.pages, acc.pages_rows
        pages_rate = _pages_per_hr(
            acc.pages,
            acc.pages_rows,
            acc.pages_duration_ms,
            acc.pages_first_start_ts,
            acc.pages_last_end_ts,
        )
    window = usage_window_stats(rows)
    stats = {
        "tok_s": window["tok_s"],
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_p50_ms": window["ttft_p50_ms"],
        "pages": pages,
        "pages_rows": pages_rows,
        "pages_per_hr": pages_rate,
    }
    # Estimated in-flight request count — omitted (not fabricated) when not
    # computable from the window rows.
    if window["streams"] is not None:
        stats["streams"] = window["streams"]
    return stats


# ----------------------------------------------------- detail stats matrix


def _wall_epoch(value: Any) -> Optional[float]:
    """ISO timestamp → epoch seconds in the WRITER's own frame.

    The cross-frame join behind per-phase token attribution: usage-tap ``ts``
    values are tz-aware UTC (their offset is authoritative — ``_ts_epoch``
    handles those), while checkpoint ``started_at``/``completed_at`` are naive
    HOST-LOCAL, so a naive value here is interpreted in the host's local
    timezone (the frame its writer used). Same per-writer-frame discipline as
    ``_parse_dt``/``_now_for``, resolved into comparable epoch seconds.
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
    try:
        return dt.timestamp()  # aware → its own offset; naive → host-local
    except (OverflowError, OSError, ValueError):
        return None


def _percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of ``values`` (q in [0,1]); None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return round(ordered[lo], 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def _attribute_rows_to_phases(
    rows_ts: List[Tuple[Optional[str], Optional[float], int, int]],
    checkpoints: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Bucket usage rows by writer identity, then checkpoint wall-clock.

    A non-blank row ``phase`` is direct evidence emitted by the spending call
    site and wins. Legacy rows without that field land in the first
    (start-ordered) checkpoint window containing their epoch; an in-flight
    window is open-ended. Rows carrying neither form of evidence land in the
    explicit ``unattributed`` bucket — NEVER guessed into a phase.
    """
    windows: List[Tuple[float, Optional[float], str]] = []
    for phase, ckpt in checkpoints.items():
        start = _wall_epoch(ckpt.get("started_at"))
        if start is None:
            continue
        windows.append((start, _wall_epoch(ckpt.get("completed_at")), phase))
    windows.sort(key=lambda w: w[0])

    buckets: Dict[str, List[int]] = {}  # phase → [calls, prompt, completion]
    unattributed = [0, 0, 0]
    explicit_order: List[str] = []
    for explicit_phase, ts, prompt, completion in rows_ts:
        target: Optional[str] = explicit_phase
        if target is not None and target not in explicit_order:
            explicit_order.append(target)
        if target is None and ts is not None:
            for start, end, phase in windows:
                if ts >= start and (end is None or ts <= end):
                    target = phase
                    break
        bucket = buckets.setdefault(target, [0, 0, 0]) if target else unattributed
        bucket[0] += 1
        bucket[1] += prompt
        bucket[2] += completion

    out: List[Dict[str, Any]] = []
    ordered_phases = [phase for _start, _end, phase in windows]
    ordered_phases.extend(
        phase for phase in explicit_order if phase not in ordered_phases
    )
    for phase in ordered_phases:
        counts = buckets.get(phase)
        if counts and counts[0] > 0:
            out.append(
                {
                    "phase": phase,
                    "calls": counts[0],
                    "prompt_tokens": counts[1],
                    "completion_tokens": counts[2],
                }
            )
    if unattributed[0] > 0:
        out.append(
            {
                "phase": "unattributed",
                "calls": unattributed[0],
                "prompt_tokens": unattributed[1],
                "completion_tokens": unattributed[2],
            }
        )
    return out


# vram_trajectory.jsonl is small (a few rows per phase boundary) but reads are
# still bounded: newline count via the shared incremental counter + a tail
# read for the latest complete row, cached by (size, mtime).
_VRAM_TAIL_BYTES = 64 * 1024
_VRAM_CACHE: Dict[str, Tuple[int, float, Optional[Dict[str, Any]]]] = {}
_VRAM_LOCK = threading.Lock()
# The fields surfaced off the latest trajectory row — the writer's real shape
# (lib/llm/vram_doctor.py::append_trajectory_row); resident_models is omitted
# (unbounded list, not a stat).
_VRAM_LATEST_KEYS = ("ts", "phase", "when", "event", "free_mib", "total_mib", "probe_source")


def _vram_stats(orchestrator_run_id: str) -> Optional[Dict[str, Any]]:
    """``stats.detail.vram`` from ``vram_trajectory.jsonl``; None → omitted."""
    if not orchestrator_run_id:
        return None
    path = _runs_dir() / orchestrator_run_id / "vram_trajectory.jsonl"
    try:
        if not path.is_file():
            return None
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    with _VRAM_LOCK:
        cached = _VRAM_CACHE.get(key)
        if cached is not None and cached[0] == st.st_size and cached[1] == st.st_mtime:
            return cached[2]
    samples = _count_sidecar_rows(path, st.st_size)
    if samples is None or samples == 0:
        return None
    latest: Optional[Dict[str, Any]] = None
    try:
        with path.open("rb") as fh:
            start = max(0, st.st_size - _VRAM_TAIL_BYTES)
            fh.seek(start)
            tail = fh.read(_VRAM_TAIL_BYTES)
    except OSError:
        return None
    for raw in reversed(tail.split(b"\n")):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 — partial/corrupt tail line → older row
            continue
        if isinstance(row, dict):
            latest = {k: row.get(k) for k in _VRAM_LATEST_KEYS if k in row}
            break
    payload: Dict[str, Any] = {"samples": samples}
    if latest:
        payload["latest"] = latest
    with _VRAM_LOCK:
        _VRAM_CACHE[key] = (st.st_size, st.st_mtime, payload)
    return payload


def _usage_detail(
    orchestrator_run_id: str, checkpoints: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """``stats.detail`` — the comprehensive matrix, or None (key omitted).

    Every section derives ONLY from real rows/files; an absent source omits
    its section (no usage file → no totals/throughput/latency/health/
    by_model/by_phase; no vram file → no ``vram``). Empty overall → None.
    The usage sections reuse the SAME ``_UsageAccumulator`` single pass the
    band's ``_usage_stats`` runs, so the 3s poll never re-reads the file.
    """
    detail: Dict[str, Any] = {}
    if orchestrator_run_id:
        path = _runs_dir() / orchestrator_run_id / "llm_usage.jsonl"
        if path.is_file():
            with _USAGE_LOCK:
                acc = _USAGE_ACCUMULATORS.setdefault(str(path), _UsageAccumulator())
                acc.ingest(path)
                calls = acc.calls
                prompt = acc.prompt_tokens
                completion = acc.completion_tokens
                durations = list(acc.durations_ms)
                ttfts = list(acc.ttfts_ms)
                truncated = acc.truncated_calls
                usage_missing = acc.usage_missing_calls
                by_model = {k: list(v) for k, v in acc.by_model.items()}
                rows_ts = list(acc.rows_ts)
                recent = list(acc.recent)
            detail["totals"] = {
                "calls": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
            gen_seconds = sum(durations) / 1000.0
            window = usage_window_stats(recent)
            detail["throughput"] = {
                # AGGREGATE window throughput — same semantics (and value) as
                # the band's headline tok_s: tokens over the window WALL span.
                "window_tok_s": window["tok_s"],
                # Per-stream window throughput — what one request experiences:
                # tokens over the calls' own summed generation seconds
                # (~aggregate/N under N-way concurrency).
                "per_stream_tok_s": window["per_stream_tok_s"],
                # Cumulative average: completion tokens over the calls' own
                # generation seconds across the whole run (per-stream-shaped;
                # None until a duration lands).
                "avg_tok_s": round(completion / gen_seconds, 1)
                if gen_seconds > 0
                else None,
            }
            detail["latency"] = {
                "ttft_p50_ms": _percentile(ttfts, 0.50),
                "ttft_p95_ms": _percentile(ttfts, 0.95),
                "duration_mean_ms": round(statistics.fmean(durations), 1)
                if durations
                else None,
                "duration_median_ms": round(statistics.median(durations), 1)
                if durations
                else None,
            }
            detail["health"] = {
                "truncated_calls": truncated,
                "usage_missing_calls": usage_missing,
            }
            detail["by_model"] = [
                {
                    "provider": provider,
                    "model": model,
                    "calls": counts[0],
                    "prompt_tokens": counts[1],
                    "completion_tokens": counts[2],
                }
                for (provider, model), counts in sorted(
                    by_model.items(), key=lambda kv: (-kv[1][0], kv[0])
                )
            ]
            detail["by_phase"] = _attribute_rows_to_phases(rows_ts, checkpoints)
    vram = _vram_stats(orchestrator_run_id)
    if vram is not None:
        detail["vram"] = vram
    return detail or None


# ----------------------------------------------------- per-phase unit progress

# Data-driven map: phase name → where that phase's per-unit resume checkpoint
# sidecar lives + the honest unit label. Only phases whose writers VERIFIABLY
# append one JSONL row per completed unit are listed; sidecar names + dirs are
# pinned to the writing code (do not guess new entries):
#
#   course_planning             MCP/tools/pipeline_tools.py
#                               ``_STAGE2_WINDOWS_CHECKPOINT_NAME`` written to
#                               ``project_path/01_learning_objectives/``.
#   concept_extraction          ``_CONCEPT_EXTRACTION_CHECKPOINT_NAME`` written
#                               to ``<libv2>/courses/<slug>/concept_graph/``.
#   content_generation_outline  ``_OUTLINE_CHECKPOINT_NAME`` written to
#                               ``project_path/01_outline/`` (one row per
#                               authored block, flushed per block).
#   content_generation_rewrite  ``_REWRITE_CHECKPOINT_NAME`` written to
#                               ``project_path/04_rewrite/``.
#   post_rewrite_validation     lib/validators/block_prose_entailment.py
#                               ``_CHECKPOINT_BASENAME`` inside the
#                               ``.prose_entailment_cache/`` dir next to
#                               ``blocks_final.jsonl`` (one row per scored
#                               block).
#   assessment_synthesis        ``_ASSESSMENTS_CHECKPOINT_NAME`` written to
#                               ``project_path/06_assessments/``.
#   training_synthesis          Trainforge/synthesis/synthesize_training.py
#                               ``_append_synthesis_pairs_checkpoint`` (opened
#                               ``"a"`` + flushed per accepted pair) at
#                               ``<corpus>/training_specs/
#                               .synthesis_pairs_checkpoint.jsonl``; the corpus
#                               dir resolves exactly like the registry handler
#                               does — ``Path(assessments_path).parent`` or
#                               ``Path(chunks_path).parent.parent`` off the
#                               trainforge_assessment phase output.
#   heading_judge               MCP/tools/pipeline_tools.py::_run_heading_judge
#                               — a growing DIRECTORY, not one appended file:
#                               one ``{stem}.heading_judgments.json`` (plus
#                               escalations + judged HTML) lands per chapter
#                               under ``runtime/state/runs/<run_id>/heading_judge/``,
#                               so this entry uses ``mode: "dir"`` + ``glob``
#                               (units = matching files, tail = newest files).
#
# Anchors: ``project`` = the Courseforge export dir (``project_path`` from the
# run's params / an earlier phase's outputs); ``libv2_course`` = the LibV2
# course dir derived from the chunking / concept-graph output paths;
# ``trainforge_corpus`` = the Trainforge corpus dir derived from the
# trainforge_assessment output paths; ``run_dir`` = ``runtime/state/runs/
# <params.run_id>/``. All come from REAL run state — never a pattern composed
# from a course name.
#
# Phases investigated and deliberately NOT mapped (no incremental per-unit
# artifact exists — mapping one would fabricate progress):
#   semantik_conversion  — per-PDF outputs ({stem}_accessible.html, GLM-OCR
#                          layout sidecars) are whole-doc writes at the END of
#                          each PDF's cascade, and the output dir lives in
#                          per-task params, not in the workflow state visible
#                          while the phase runs.
#   chunking / imscc_chunking — chunks.jsonl is computed fully in memory and
#                          streamed in ONE tight open("wb") loop at phase end
#                          (MCP/tools/pipeline_tools.py::_run_dart_chunking /  # legacy-token: allow
#                          _run_imscc_chunking); nothing grows during the
#                          phase. <!-- legacy-token: allow -->
#   vector_indexing      — atomic fail-closed index build (no partial index by
#                          design).
#   packaging / source_mapping / objective_extraction / staging /
#   trainforge_assessment / libv2_archival / finalization — single whole-file
#                          (or symlink) emits at phase end.
#   post_training_validation (trainforge_train) — validator-only gate over
#                          the already-written eval_report.json; no artifact.
_PHASE_UNIT_SIDECARS: Dict[str, Dict[str, str]] = {
    "course_planning": {
        "anchor": "project",
        "relpath": "01_learning_objectives/.stage2_windows_checkpoint.jsonl",
        "label": "windows done",
    },
    "concept_extraction": {
        "anchor": "libv2_course",
        "relpath": "concept_graph/.concept_extraction_checkpoint.jsonl",
        "label": "windows done",
    },
    "content_generation_outline": {
        "anchor": "project",
        "relpath": "01_outline/.blocks_outline_checkpoint.jsonl",
        "label": "blocks done",
    },
    "content_generation_rewrite": {
        "anchor": "project",
        "relpath": "04_rewrite/.blocks_final_checkpoint.jsonl",
        "label": "blocks done",
    },
    "post_rewrite_validation": {
        "anchor": "project",
        "relpath": (
            "04_rewrite/.prose_entailment_cache/prose_entailment.checkpoint.jsonl"
        ),
        "label": "blocks scored",
    },
    "assessment_synthesis": {
        "anchor": "project",
        "relpath": "06_assessments/.assessments_checkpoint.jsonl",
        "label": "items done",
    },
    "training_synthesis": {
        "anchor": "trainforge_corpus",
        "relpath": "training_specs/.synthesis_pairs_checkpoint.jsonl",
        "label": "pairs done",
    },
    "heading_judge": {
        # A growing DIRECTORY (one judgments file per chapter), not a single
        # appended JSONL — counted/tailed as matching files, newest last.
        "anchor": "run_dir",
        "relpath": "heading_judge",
        "glob": "*.heading_judgments.json",
        "mode": "dir",
        "label": "chapters judged",
    },
    "training": {
        # The trainforge_train adapter-fit stream is distinct from evaluation:
        # <libv2>/courses/<slug>/models/<model_id>/training_telemetry.v1.jsonl.
        # <model_id> is minted at run start, so the NEWEST matching file is
        # tailed (mode "glob_file"). Evaluation has its own eval_progress
        # stream and must never stand in for live SFT/DPO progress.
        "anchor": "libv2_course_slug",
        "relpath": "models",
        "glob": "*/training_telemetry.v1.jsonl",
        "mode": "glob_file",
        "label": "training events",
    },
}


def _training_telemetry_snapshot(
    current_phase: Optional[str],
    params: Dict[str, Any],
    phase_outputs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the newest trainer-owned atomic snapshot for the training phase.

    The snapshot is display evidence only.  It never implies adapter
    promotion, completion, or success; those decisions remain owned by the
    workflow/checkpoint-selection artifacts.
    """
    if current_phase != "training":
        return None
    spec = _PHASE_UNIT_SIDECARS["training"]
    anchor = _resolve_unit_anchor(spec["anchor"], params, phase_outputs)
    if anchor is None:
        return None
    matches = _unit_dir_files(
        anchor / spec["relpath"], "*/training_telemetry.latest.v1.json"
    )
    if not matches:
        return None
    path = matches[-1]
    try:
        if path.stat().st_size > 1024 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    event = payload.get("event")
    stage = payload.get("stage")
    metrics = payload.get("metrics")
    if not isinstance(event, str) or stage not in {"sft", "dpo"}:
        return None
    if not isinstance(metrics, dict):
        return None
    return payload

# Hard ceiling on how many sidecar bytes a single (first-attach) count may
# read. Sidecars grow to tens of MB on a big corpus; anything past this cap is
# pathological and the field is honestly OMITTED rather than half-counted.
_UNIT_SIDECAR_MAX_BYTES = 256 * 1024 * 1024
_UNIT_COUNT_CHUNK = 1024 * 1024

# path → (bytes_counted, newline_count). Append-only files re-read only the
# appended span; a shrunk/rotated file triggers a full recount.
_UNIT_COUNT_CACHE: Dict[str, Tuple[int, int]] = {}
_UNIT_COUNT_LOCK = threading.Lock()


def _count_sidecar_rows(path: Path, size: int) -> Optional[int]:
    """Newline count of ``path``, incremental over the append-only sidecar.

    A row only counts once its terminating newline is on disk, so a mid-append
    partial tail line is never counted (the bounded-honest count). Returns
    None on any read failure (the caller omits the field).
    """
    key = str(path)
    with _UNIT_COUNT_LOCK:
        cached = _UNIT_COUNT_CACHE.get(key)
    if cached is not None and cached[0] == size:
        return cached[1]
    start, base = 0, 0
    if cached is not None and size > cached[0]:
        start, base = cached
    appended = 0
    read_bytes = 0
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            while True:
                chunk = fh.read(_UNIT_COUNT_CHUNK)
                if not chunk:
                    break
                read_bytes += len(chunk)
                appended += chunk.count(b"\n")
    except OSError:
        return None
    count = base + appended
    with _UNIT_COUNT_LOCK:
        _UNIT_COUNT_CACHE[key] = (start + read_bytes, count)
    return count


def _resolve_unit_anchor(
    kind: str, params: Dict[str, Any], phase_outputs: Dict[str, Any]
) -> Optional[Path]:
    """Resolve a sidecar anchor dir from the run's REAL state (or None)."""
    if kind == "project":
        cand = params.get("project_path")
        if isinstance(cand, str) and cand.strip():
            return Path(cand)
        for out in phase_outputs.values():
            if isinstance(out, dict):
                p = out.get("project_path")
                if isinstance(p, str) and p.strip():
                    return Path(p)
        return None
    if kind == "libv2_course":
        # <course>/semantik_chunks/chunks.jsonl and
        # <course>/concept_graph/concept_graph_semantic.json both sit two
        # levels under the LibV2 course dir.
        for out_key in ("semantik_chunks_path", "concept_graph_path"):
            for out in phase_outputs.values():
                if isinstance(out, dict):
                    p = out.get(out_key)
                    if isinstance(p, str) and p.strip():
                        return Path(p).parent.parent
        return None
    if kind == "trainforge_corpus":
        # The Trainforge corpus dir, resolved EXACTLY like the
        # synthesize_training registry handler (_synthesize_training in
        # MCP/tools/pipeline_tools.py) resolves it from the
        # trainforge_assessment phase output:
        # <corpus>/assessments.json → one parent up;
        # <corpus>/corpus/chunks.jsonl → two parents up.
        # Only the named producer owns these paths. Earlier assessment phases
        # can carry an identically named ``assessments_path`` rooted in a
        # different Courseforge directory; scanning every phase therefore
        # resolves a valid-looking but wrong sidecar path.
        out = phase_outputs.get("trainforge_assessment")
        if not isinstance(out, dict):
            return None
        for out_key, levels in (("assessments_path", 1), ("chunks_path", 2)):
            p = out.get(out_key)
            if isinstance(p, str) and p.strip():
                anchor = Path(p)
                for _ in range(levels):
                    anchor = anchor.parent
                return anchor
        return None
    if kind == "run_dir":
        # runtime/state/runs/<params.run_id>/ — the orchestrator run dir
        # (heading_judge writes its per-chapter outputs here).
        run_id = params.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return _runs_dir() / run_id.strip()
        return None
    if kind == "libv2_course_slug":
        # Resolve the LibV2 course directory with the same canonical slug
        # function and root precedence used by the training runner:
        # params.libv2_root → ED4ALL_LIBV2_ROOT →
        # lib.paths.LIBV2_COURSES.
        code = params.get("course_code") or params.get("course_name")
        if not (isinstance(code, str) and code.strip()):
            return None
        try:
            from lib.ontology.slugs import libv2_course_slug  # noqa: PLC0415

            slug = libv2_course_slug(code.strip())
            cand = params.get("libv2_root")
            if isinstance(cand, str) and cand.strip():
                root = Path(cand.strip())
                courses = root if root.name == "courses" else root / "courses"
            elif os.environ.get("ED4ALL_LIBV2_ROOT", "").strip():
                courses = Path(os.environ["ED4ALL_LIBV2_ROOT"].strip()) / "courses"
            else:
                from lib.paths import LIBV2_COURSES  # noqa: PLC0415

                courses = Path(LIBV2_COURSES)
            return courses / slug
        except Exception:  # noqa: BLE001 — unresolvable → field omitted
            return None
    return None


def _unit_dir_files(path: Path, pattern: str) -> Optional[List[Path]]:
    """Matching unit files in a ``mode: "dir"`` spec's directory, mtime-sorted
    (oldest → newest; name as tie-break). None when the dir is unreadable /
    absent (the caller omits the field)."""
    try:
        if not path.is_dir():
            return None
        files = [f for f in path.glob(pattern) if f.is_file()]
        files.sort(key=lambda f: (f.stat().st_mtime, f.name))
    except OSError:
        return None
    return files


def _phase_unit_progress(
    current_phase: Optional[str],
    params: Dict[str, Any],
    phase_outputs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """``stats.phase_units`` for the current phase, or None (field omitted).

    ``{count, label, source, updated_at}`` — ``count`` is the REAL number of
    complete rows in the phase's resume sidecar (or, for a ``mode: "dir"``
    spec, the number of per-unit files that exist), ``source`` its basename,
    and ``updated_at`` the sidecar/newest-file mtime (UTC ISO). No sidecar /
    unmapped phase / unreadable file → None. Deliberately NO total: a total is
    only honest when a real planned-unit count exists on disk, and none of
    the mapped writers emits one.
    """
    spec = _PHASE_UNIT_SIDECARS.get(current_phase or "")
    if not spec:
        return None
    anchor = _resolve_unit_anchor(spec["anchor"], params, phase_outputs)
    if anchor is None:
        return None
    path = anchor / spec["relpath"]
    if spec.get("mode") == "dir":
        files = _unit_dir_files(path, spec["glob"])
        if not files:
            return None  # absent dir or zero units → omitted, never a fake 0
        try:
            newest_mtime = files[-1].stat().st_mtime
        except OSError:
            return None
        return {
            "count": len(files),
            "label": spec["label"],
            "source": path.name,
            "updated_at": datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    if spec.get("mode") == "glob_file":
        # The unit stream's parent dir is minted at run start (e.g. the
        # trainforge_train models/<model_id>/), so tail the NEWEST matching
        # stream; none yet → omitted (honest — the stage hasn't started).
        matches = _unit_dir_files(path, spec["glob"])
        if not matches:
            return None
        path = matches[-1]
    try:
        if not path.is_file():
            return None
        st = path.stat()
    except OSError:
        return None
    if st.st_size > _UNIT_SIDECAR_MAX_BYTES:
        return None
    count = _count_sidecar_rows(path, st.st_size)
    if count is None:
        return None
    return {
        "count": count,
        "label": spec["label"],
        "source": path.name,
        "updated_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


# ----------------------------------------------------------- live output tail

# The "Live output" panel: the LAST few complete rows of the current phase's
# per-unit resume sidecar (the same ``_PHASE_UNIT_SIDECARS`` map — real
# pipeline output, the units the LLM just finished), each mapped to a bounded
# display record. Honesty contract: absent sidecar / unmapped phase → an empty
# ``rows`` list; nothing estimated, no raw multi-KB blobs (text is
# HTML-stripped and hard-truncated).
_TAIL_MAX_ROWS = 15
_TAIL_MAX_BYTES = 2 * 1024 * 1024  # bounded seek-from-end read
_TAIL_DIR_FILE_MAX_BYTES = 64 * 1024  # per-file cap for mode:"dir" sources
_TAIL_TEXT_MAX_CHARS = 500
# Row keys tried (in order) for the display label — the row's natural unit id
# across the mapped writers (blocks carry block_id; stage-2 windows and
# concept windows carry unit_id; training-pair checkpoint rows carry
# chunk_id; eval-progress rows carry event).
_TAIL_LABEL_KEYS = ("block_id", "unit_id", "item_id", "chunk_id", "id", "event")
# Row keys tried (in order) for the display text — the unit's actual content
# (outline/rewrite block entries carry ``content``; other writers fall back
# to a compact JSON of the row's payload).
_TAIL_TEXT_KEYS = ("content", "html", "text", "statement", "prose", "body")
# Resume-plumbing keys elided from the JSON fallback rendering (fingerprints
# and schema stamps are bookkeeping, not output).
_TAIL_ELIDE_KEYS = frozenset(
    {"fingerprint", "checkpoint_fingerprint", "schema_version", "site_id"}
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Drop HTML tags + collapse whitespace (display-only, never persisted)."""
    return " ".join(_HTML_TAG_RE.sub(" ", text).split())


def _truncate(text: str) -> str:
    if len(text) > _TAIL_TEXT_MAX_CHARS:
        return text[: _TAIL_TEXT_MAX_CHARS - 1].rstrip() + "…"
    return text


def _tail_display_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one sidecar row → ``{seq, label, text}`` (bounded, never raw).

    ``seq`` is the row's OWN sequence/window_index when it carries one, else
    None (a bounded tail read cannot know an absolute file position — never
    fabricated). ``text`` prefers the row's content field (HTML-stripped);
    rows without one (window/concept payloads) render a compact JSON of the
    payload minus resume-plumbing keys. Both are hard-truncated.
    """
    label = ""
    for key in _TAIL_LABEL_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            label = val.strip()
            break
    seq: Optional[int] = None
    for key in ("sequence", "window_index"):
        val = row.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            seq = val
            break
    text = ""
    for key in _TAIL_TEXT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            text = _strip_html(val)
            break
    if not text:
        payload = {k: v for k, v in row.items() if k not in _TAIL_ELIDE_KEYS}
        try:
            text = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
        except (TypeError, ValueError):
            text = ""
    return {"seq": seq, "label": label, "text": _truncate(text)}


def output_tail(run_id: str) -> Optional[Dict[str, Any]]:
    """Build the ``/api/runs/{run_id}/output-tail`` payload; None → 404.

    Resolves the run's CURRENT phase (the same merge the progress payload
    uses), locates that phase's per-unit resume sidecar via
    ``_PHASE_UNIT_SIDECARS`` + the real-state anchor resolution, and returns
    the LAST ``_TAIL_MAX_ROWS`` complete JSONL rows as display records.
    Bounded read (at most ``_TAIL_MAX_BYTES`` from the end); corrupt /
    mid-append partial rows are skipped; absent sidecar / unmapped phase →
    ``rows: []`` (honest, never fabricated).
    """
    progress = run_progress(run_id)
    if progress is None:
        return None
    current = progress.get("current_phase")
    empty: Dict[str, Any] = {
        "run_id": run_id,
        "phase": current,
        "source": None,
        "label": None,
        "row_count": 0,
        "rows": [],
    }
    spec = _PHASE_UNIT_SIDECARS.get(current or "")
    if not spec:
        return empty
    state = _read_workflow_state(str(progress.get("workflow_id") or ""))
    params = (state or {}).get("params") or {}
    phase_outputs = (state or {}).get("phase_outputs") or {}
    if not isinstance(phase_outputs, dict):
        phase_outputs = {}
    anchor = _resolve_unit_anchor(spec["anchor"], params, phase_outputs)
    if anchor is None:
        return empty
    path = anchor / spec["relpath"]
    if spec.get("mode") == "dir":
        # Growing-DIRECTORY source (heading_judge): units are matching files,
        # newest last; each row is the file's (bounded, elided) JSON payload.
        files = _unit_dir_files(path, spec["glob"]) or []
        if not files:
            return empty
        suffix = spec["glob"].lstrip("*")
        rows = []
        for f in files[-_TAIL_MAX_ROWS:]:
            label = (
                f.name[: -len(suffix)]
                if suffix and f.name.endswith(suffix)
                else f.stem
            )
            text = ""
            try:
                if f.stat().st_size <= _TAIL_DIR_FILE_MAX_BYTES:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payload = {
                            k: v
                            for k, v in payload.items()
                            if k not in _TAIL_ELIDE_KEYS
                        }
                    text = json.dumps(
                        payload, ensure_ascii=False, separators=(", ", ": ")
                    )
            except (OSError, ValueError):
                text = ""  # unreadable/corrupt unit file → label-only row
            rows.append({"seq": None, "label": label, "text": _truncate(text)})
        return {
            "run_id": run_id,
            "phase": current,
            "source": path.name,
            "label": spec["label"],
            "row_count": len(files),
            "rows": rows,
        }
    if spec.get("mode") == "glob_file":
        # Newest matching per-run stream (see _phase_unit_progress).
        matches = _unit_dir_files(path, spec["glob"])
        if not matches:
            return empty
        path = matches[-1]
    try:
        if not path.is_file():
            return empty
        size = path.stat().st_size
    except OSError:
        return empty
    start = max(0, size - _TAIL_MAX_BYTES)
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            blob = fh.read(size - start)
    except OSError:
        return empty
    if start > 0:
        # Seeked into the middle of a row — drop the partial head line.
        nl = blob.find(b"\n")
        blob = b"" if nl < 0 else blob[nl + 1 :]
    # Only complete lines count (the tail may be mid-append).
    last_nl = blob.rfind(b"\n")
    if last_nl < 0:
        return empty
    parsed: List[Dict[str, Any]] = []
    for raw in blob[:last_nl].split(b"\n"):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 — one corrupt row never breaks the tail
            continue
        if isinstance(row, dict):
            parsed.append(row)
    rows = [_tail_display_row(r) for r in parsed[-_TAIL_MAX_ROWS:]]
    return {
        "run_id": run_id,
        "phase": current,
        "source": path.name,
        "label": spec["label"],
        "row_count": _count_sidecar_rows(path, size) or len(parsed),
        "rows": rows,
    }


# --------------------------------------------------------------- seat probe


_SEAT_PROBE_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}
_SEAT_PROBE_LOCK = threading.Lock()


def _active_run_seat_registry(
    workflow_id: Optional[str],
    orchestrator_run_id: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, str]:
    """Resolve the seat registry owned by the attributed CLI process.

    A standalone ``ed4all run`` can carry a deliberate per-run seat overlay
    that the already-running GUI process does not inherit. Read only that
    attributed same-user process's ``ED4ALL_SEAT_BASE_URLS`` value from procfs;
    unrelated process environments and all other variables remain unread.
    Missing/unreadable procfs simply yields an empty overlay.
    """
    try:
        from gui.services import liveness  # noqa: PLC0415
        from lib.vllm_container_lifecycle import parse_seat_registry  # noqa: PLC0415

        tokens = liveness.attribution_tokens_from_params(
            params, orchestrator_run_id, wf_id=workflow_id
        )
        if not tokens:
            return {}
        for pid, argv in liveness.scan_pipeline_processes():
            if not any(tok in arg for tok in tokens for arg in argv):
                continue
            try:
                raw = Path(f"/proc/{pid}/environ").read_bytes()
            except OSError:
                continue
            prefix = b"ED4ALL_SEAT_BASE_URLS="
            for entry in raw.split(b"\0"):
                if entry.startswith(prefix):
                    value = entry[len(prefix) :].decode("utf-8", "replace")
                    return parse_seat_registry(value)
    except Exception:  # noqa: BLE001 — optional read-only discovery
        return {}
    return {}


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


def _serving_seat(
    phase_seats: List[str], run_registry: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """First registered seat answering ``/v1/models``; phase seats first.

    Registry = ``ED4ALL_SEAT_BASE_URLS`` via the canonical
    ``lib.vllm_container_lifecycle.parse_seat_registry`` (a new seat is a
    registry entry, never code). Empty registry → None (no probing at all).
    """
    try:
        from lib.vllm_container_lifecycle import parse_seat_registry  # noqa: PLC0415

        registry = parse_seat_registry()
    except Exception:  # noqa: BLE001 — registry parse is best-effort
        registry = {}
    # The attributed runner's overlay is the execution truth for this run.
    registry.update(run_registry or {})
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


def _seat_activity(
    current_phase: Optional[str],
    current_seats: List[str],
    run_registry: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Surface an in-progress SEAT SWAP / cold-start as its OWN state.

    The stage tracker attributes only a phase's own compute time, so a long
    vLLM seat swap AT a phase boundary (the schedule tears down the conversion
    seats and COLD-STARTS the next seat — up to ~10 min: container up,
    ``/v1/models`` silent, then a coherence probe) renders as the phase hanging
    with no progress. This field names that wait for what it is.

    Derived PURELY from the shared cached GLOBAL ``seat_service.seat_overview()``
    probe — NOT the run-scoped ``seat_overview(run_id)``, which would recurse
    back into ``run_progress`` — so there is no second probe path and no
    recursion. The expectation is the current phase's DECLARED ``seats:`` set
    (``current_seats``, already resolved from the plan); a needed seat that is
    not yet ``live`` is surfaced with its own ``since``/``elapsed_ms`` age.

    Two honesty grades, mirroring the seat monitor's satisfying-state contract
    (``{live, loading}``):

    * ``loading`` → PATIENT (``concern=False``): a ~9-min cold start is
      expected, not stuck.
    * anything else a NAMED seat resolves to — ``down`` (container not running),
      an ``unregistered`` name absent from the registry, or ``unknown`` (docker
      unavailable) — → ALARMING (``concern=True``): a dispatch to a seat that is
      not coming up fails the build. This is exactly the seat monitor's
      ``mismatch`` set.

    Returns ``None`` (the key is OMITTED) when the phase declares no seats or
    every needed seat is ``live`` — normal phases are unaffected. Read-only,
    cheap (reuses the ~10 s TTL probe cache), and NEVER raises. Deliberately
    kept OFF ``phase_durations``/``wallclock_s``: the swap seconds must never be
    misattributed as the phase's own compute.
    """
    if not current_phase or not current_seats:
        return None
    try:
        from gui.services import seat_service  # noqa: PLC0415

        # GLOBAL overview only (no run_id) — cached, and never re-enters
        # run_progress via seat_service._run_context.
        overview = seat_service.seat_overview()
    except Exception:  # noqa: BLE001 — seat telemetry is best-effort, never fatal
        return None
    if not isinstance(overview, dict):
        return None
    by_name = {
        str(s.get("name")): s
        for s in (overview.get("seats") or [])
        if isinstance(s, dict) and s.get("name")
    }
    waiting: List[Dict[str, Any]] = []
    concern = False
    for name in current_seats:
        run_url = (run_registry or {}).get(name)
        if run_url and _probe_seat_model(run_url) is not None:
            continue
        seat = by_name.get(name)
        if seat is None:
            # The phase names a seat that is NOT in the registry at all — a loud
            # configuration mismatch, mirroring seat_service's "unregistered".
            waiting.append(
                {
                    "seat": name,
                    "state": "unregistered",
                    "concern": True,
                    "since": None,
                    "elapsed_ms": None,
                    "container": None,
                }
            )
            concern = True
            continue
        state_token = str(seat.get("state") or "unknown")
        if state_token == "live":
            continue  # ready — nothing to surface
        # ``loading`` is the ONLY non-live satisfying state (the ~9-min cold
        # start); everything else is the alarming mismatch set.
        is_concern = state_token != "loading"
        waiting.append(
            {
                "seat": name,
                "state": state_token,
                "concern": is_concern,
                "since": seat.get("since"),
                "elapsed_ms": seat.get("since_ms"),
                "container": seat.get("container"),
            }
        )
        concern = concern or is_concern
    if not waiting:
        return None
    return {"phase": current_phase, "concern": concern, "seats": waiting}


# ------------------------------------------------------------ the main merge


def run_progress(run_id: str) -> Optional[Dict[str, Any]]:
    """Build the ``/api/runs/{run_id}/progress`` payload; None → 404.

    ``run_id`` is a GUI run id (``runtime/state/gui/runs/``); a bare orchestrator
    workflow id (``runtime/state/workflows/<id>.json``) is accepted as a fallback so
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
    phase_telemetry = _read_phase_telemetry(
        orchestrator_run_id, [str(ph["name"]) for ph in plan]
    )

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
        telemetry_state = (phase_telemetry.get(name) or {}).get("state")
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
        elif telemetry_state == "complete":
            state_token = "done"
        elif telemetry_state == "failed":
            state_token = "failed"
        elif telemetry_state == "paused":
            state_token = "paused"
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

    # --- env-branch row hiding (owner design). When the plan carries BOTH
    # sides of an enabled_when_env variable (single-pass content_generation
    # [VAR!=v] vs the two-pass tiers [VAR=v]), the not-taken side is noise:
    #   - a branchy row that resolved "skipped" is HIDDEN (the branch
    #     provably didn't run — its complement did);
    #   - the NEGATIVE ("!=", fallback/single-pass) side is also hidden while
    #     it has NO evidence of running (no marker, no in-flight checkpoint):
    #     the positive side renders "pending" per the no-evidence contract,
    #     so the generation group starts at the outline tier on a two-pass
    #     run. The moment its checkpoint starts (a genuine single-pass run)
    #     it appears as "current" and the complement tiers hide once their
    #     skip is observed.
    # A phase whose complement clause is NOT in the plan is never hidden.
    #
    # "the branch provably didn't run — its complement did" is the LOAD-BEARING
    # half of that premise. A skip alone does NOT prove it: a courseforge-*
    # stage subcommand skips phases by the ``courseforge_stage`` whitelist, and
    # `--stop-after` skips by index, so on those runs BOTH sides of the branch
    # resolve "skipped" and hiding on skip-alone deletes the two-pass tiers
    # (content_generation_outline / inter_tier_validation) from the rail
    # entirely. So a POSITIVE ("=", taken-side) row is hidden only when its
    # complement shows real evidence of having run. The NEGATIVE ("!=",
    # single-pass fallback) side keeps the original skip-hides rule: under
    # two-pass it is genuine noise, and its no-evidence case is handled below.
    plan_clauses = set()
    for ph in plan:
        parsed = _parse_env_condition(ph.get("enabled_when_env"))
        if parsed is not None:
            plan_clauses.add(parsed)

    _RAN_TOKENS = frozenset({"done", "current", "failed"})

    def _complement_ran(var: str, op: str, expected: str) -> bool:
        """True when the OTHER side of this env branch shows it actually ran."""
        want = (var, "!=" if op == "=" else "=", expected)
        for other in plan:
            if _parse_env_condition(other.get("enabled_when_env")) != want:
                continue
            if resolved.get(other["name"]) in _RAN_TOKENS:
                return True
            if checkpoint_current == other["name"]:
                return True
        return False

    hidden_phases: set = set()
    for ph in plan:
        parsed = _parse_env_condition(ph.get("enabled_when_env"))
        if parsed is None:
            continue
        var, op, expected = parsed
        if (var, "!=" if op == "=" else "=", expected) not in plan_clauses:
            continue
        name = ph["name"]
        token = resolved.get(name)
        if token == "skipped":
            if op == "!=" or _complement_ran(var, op, expected):
                hidden_phases.add(name)
        elif token is None and op == "!=" and checkpoint_current != name:
            hidden_phases.add(name)

    phases: List[Dict[str, Any]] = []
    current_phase: Optional[str] = None
    telemetry_current = (
        next(
            (
                ph["name"]
                for ph in plan
                if (phase_telemetry.get(ph["name"]) or {}).get("state") == "running"
            ),
            None,
        )
        if not is_terminal
        else None
    )
    for ph in plan:
        name = ph["name"]
        if name in hidden_phases:
            continue
        state_token = resolved.get(name)
        if state_token is None:
            is_current = (
                name == telemetry_current
                if telemetry_current is not None
                else checkpoint_current is None or checkpoint_current == name
            )
            if not is_terminal and current_phase is None and is_current:
                state_token = "current"
                current_phase = name
            else:
                state_token = "pending"
        ckpt = checkpoints.get(name) or {}
        wallclock = ckpt.get("wallclock_s")
        telemetry = phase_telemetry.get(name) or {}
        started_raw = ckpt.get("started_at")
        completed_raw = ckpt.get("completed_at")
        if (
            wallclock is not None
            and started_raw is not None
            and started_raw == completed_raw
        ):
            # Resume RE-STAMP signature: identical started/completed timestamps
            # (an exact 0.0s span measuring the restore, not the execution) —
            # suppress it rather than render a misleading "0s". A
            # genuinely-executed phase, however fast, keeps its real span so a
            # deterministic phase (staging / source_mapping, ~0.1-0.3s) is never
            # blank; the renderer formats any sub-second span as "<1s".
            wallclock = None
        if wallclock is None and telemetry.get("state") in {
            "paused",
            "failed",
            "complete",
        }:
            # A terminal telemetry snapshot retains the phase's cumulative
            # elapsed time across resumes.  It is stronger than a suppressed
            # zero-duration resume re-stamp, but never replaces a real
            # checkpoint wall-clock.
            elapsed = telemetry.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                wallclock = elapsed
        node = {
            "name": name,
            "index": ph["index"],
            "state": state_token,
            "group": ph["group"],
            "label": _phase_label(name),
            "wallclock_s": wallclock,
        }
        # Preserve a completed/paused phase's last real metrics even after the
        # rail advances.  This is additive and generic: any planned phase may
        # publish the same versioned envelope.
        if telemetry:
            node["telemetry"] = telemetry
        phases.append(node)

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

    # Live elapsed on the IN-PROGRESS phase node: the current phase carries a
    # live ``elapsed_s`` (now − phase start, the same anchor as the band's
    # phase_elapsed_s) recomputed on every poll, so the rail shows an elapsing
    # time under the active phase instead of a blank until it completes.
    # Completed phases keep their final ``wallclock_s``; ``elapsed_s`` is a
    # SEPARATE field so a live value is never confused with a final wall-clock.
    if current_phase is not None and phase_elapsed_s is not None:
        for node in phases:
            if node["name"] == current_phase:
                node["elapsed_s"] = phase_elapsed_s
                break

    stats = _usage_stats(orchestrator_run_id)
    stats["phase_elapsed_s"] = phase_elapsed_s
    # Real in-phase unit progress from the phase's resume sidecar. Honest by
    # construction: the key is OMITTED whenever no sidecar exists on disk.
    phase_units = _phase_unit_progress(current_phase, params, phase_outputs)
    if phase_units is not None:
        stats["phase_units"] = phase_units
    training_telemetry = _training_telemetry_snapshot(
        current_phase, params, phase_outputs
    )
    if training_telemetry is not None:
        stats["training_telemetry"] = training_telemetry
    # The comprehensive detail matrix (additive — the band's top-level keys
    # above stay byte-compatible). Omitted when no source file exists.
    detail = _usage_detail(orchestrator_run_id, checkpoints)
    if detail is not None:
        stats["detail"] = detail
    if phase_telemetry:
        # Keep phase telemetry separate from LLM usage detail.  Consumers can
        # render it even on deterministic/no-usage phases.
        stats["phase_telemetry"] = [
            phase_telemetry[ph["name"]]
            for ph in plan
            if ph["name"] in phase_telemetry
        ]
    if is_terminal:
        stats["seat"] = None
    else:
        current_seats = next(
            (ph["seats"] for ph in plan if ph["name"] == current_phase), []
        )
        run_seat_registry = _active_run_seat_registry(
            workflow_id, orchestrator_run_id, params
        )
        stats["seat"] = _serving_seat(
            list(current_seats or []), run_seat_registry
        )
        # An in-progress seat swap / cold-start on a seat THIS phase declares —
        # surfaced as its OWN state, NEVER folded into the phase's wallclock_s
        # (see _seat_activity). Omitted whenever every needed seat is live.
        seat_activity = _seat_activity(
            current_phase, list(current_seats or []), run_seat_registry
        )
        if seat_activity is not None:
            stats["seat_activity"] = seat_activity

    # Honest process-liveness verdict (see gui/services/liveness.py). A run
    # observed only through its orchestrator workflow file (record is None) is
    # CLI-launched — its liveness is the ``ed4all run`` process; a GUI record is
    # driven in-process. ``effective_status`` never mutates ``status``: it is an
    # ADDITIVE field so a stale RUNNING record renders honestly (stopped /
    # stopping / stalled? / paused) instead of "Building" forever.
    is_cli_run = record is None
    try:
        wf_mtime: Optional[float] = (
            _state_root() / "workflows" / f"{workflow_id}.json"
        ).stat().st_mtime
    except OSError:
        wf_mtime = None

    # Course/book identity for the build-page header. The workflow state's
    # params are the FRESHEST source: an ``--auto-name`` run starts under a
    # provisional slug and the workflow runner REBINDS ``params.course_name``
    # mid-run (workflow_runner._maybe_apply_auto_name persists the rebound
    # name — and ``display_title``, the human H1 title — back into
    # ``runtime/state/workflows/<id>.json``), while the GUI run record keeps its
    # creation-time name forever. Read-only + never-raise: unknown → None
    # (the client omits the element rather than inventing a name).
    course_name: Optional[str] = None
    for cand in (params.get("course_name"), (record or {}).get("course_name")):
        if isinstance(cand, str) and cand.strip():
            course_name = cand.strip()
            break
    display_title = params.get("display_title")
    if not (isinstance(display_title, str) and display_title.strip()):
        display_title = None
    else:
        display_title = display_title.strip()
    eff_status = liveness.effective_status(
        status,
        is_cli=is_cli_run,
        orch_run_id=orchestrator_run_id or None,
        attribution_tokens=liveness.attribution_tokens_from_params(
            params, orchestrator_run_id or None, wf_id=workflow_id or None
        ),
        wf_mtime=wf_mtime,
    )

    return {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow": workflow_name,
        "status": status,
        "effective_status": eff_status,
        "course_name": course_name,
        "display_title": display_title,
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
    "output_tail",
    "phase_group",
    "phase_plan",
    "run_progress",
    "usage_window_stats",
]  # _PHASE_UNIT_SIDECARS / _phase_unit_progress are internal (tested directly)
