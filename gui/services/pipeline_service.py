"""Full-pipeline chain service — the data behind ``GET /api/runs/{run_id}/pipeline``.

Feeds the Studio build-progress page's FULL-pipeline view: Stage-A (the course
BUILD run — ``textbook_to_course`` / ``course_generation``) followed by Stage-B
(the ``trainforge_train`` LoRA-training run). These are two SEPARATE orchestrator
run records correlated ONLY by the course slug (``lib.ontology.slugs``), so this
module discovers the sibling stage by slugifying each run's course and matching.

Everything here is READ-ONLY and best-effort: it never mutates run state and
never fabricates a run/slug/status. Sources:

- **The queried run** — resolved exactly like ``progress_service.run_progress``:
  a GUI run id (``state/gui/runs/`` via ``shared_state.read_run`` →
  ``workflow_id``) OR a bare orchestrator workflow id
  (``state/workflows/<id>.json`` read directly). Its ``params.course_name`` /
  ``params.course_code`` slugifies to the correlation key.
- **Sibling BUILD / TRAINING runs** — the newest run whose course slugifies to
  the same key, discovered by scanning BOTH ``run_service.list_runs()`` (GUI +
  recency-windowed CLI records) AND every ``state/workflows/*.json`` directly
  (CLI/pilot-launched training runs have no GUI record). "Newest" is by
  ``started_at`` then workflow-file mtime.
- **Completed training products** — ``LibV2/courses/<slug>/models/<model_id>/``
  dirs, one per training run; ``present`` for the training stage is a live/recorded
  ``trainforge_train`` run OR ≥1 model dir (a finished train with no run record).
- **Planned phases** — ``progress_service.phase_plan(workflow)`` phase NAMES
  (mtime-cached from ``config/workflows.yaml``; never a hardcoded list). Always
  emitted so the operator sees the full pipeline even before a stage starts.

Bounded + honest: workflow-file reads reuse ``progress_service._read_workflow_state``
(128 MB guard, never raises); the ``state/workflows/`` index scan is cached by
directory mtime; an absent/corrupt/mid-write file is treated as absent; an
unresolvable slug yields a single-stage chain with ``course_slug: null`` (never a
guessed slug).
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui import shared_state
from gui.services import progress_service

logger = logging.getLogger("gui.pipeline_service")

# The two full-course BUILD workflows (Stage A). A run of either slugifies its
# course the same way the trainforge_train (Stage B) run does, so the two
# correlate by slug.
BUILD_WORKFLOWS = ("textbook_to_course", "course_generation")
TRAINING_WORKFLOW = "trainforge_train"

_STAGE_LABELS = {
    "build": "Stage A · Course build",
    "training": "Stage B · LoRA training",
}
_DEFAULT_BUILD_WORKFLOW = "textbook_to_course"


def _norm(name: Any) -> str:
    """Normalize a workflow name to underscore/lowercase form (mirrors run_service)."""
    return str(name or "").replace("-", "_").strip().lower()


def _lower_status(value: Any) -> Optional[str]:
    """Lowercased status string, or None when absent/blank (never fabricated)."""
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _slugify(code: Any) -> Optional[str]:
    """Canonical LibV2 course slug for a course code/name (None on empty/error)."""
    if not (isinstance(code, str) and code.strip()):
        return None
    try:
        from lib.ontology.slugs import libv2_course_slug  # noqa: PLC0415

        slug = libv2_course_slug(code.strip())
    except Exception:  # noqa: BLE001 — slug helper is best-effort here
        return None
    return slug or None


def _course_code(params: Dict[str, Any], record_course: Any) -> Optional[str]:
    """Best human course name/code from a run's params / GUI record (or None).

    ``params.course_name`` (build) → ``params.course_code`` (training) → the GUI
    record's creation-time ``course_name``. Never invents a value.
    """
    for cand in (
        params.get("course_name"),
        params.get("course_code"),
        record_course,
    ):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None


def _plan_names(workflow: str) -> List[str]:
    """Phase NAMES for ``workflow`` from the real config plan ([] if unknown)."""
    return [str(p["name"]) for p in progress_service.phase_plan(workflow)]


def _parse_started(value: Any) -> datetime:
    """Parse a start timestamp for newest-ordering; unparseable → ``datetime.min``."""
    dt = progress_service._parse_dt(value)  # noqa: SLF001 — shared frame discipline
    return dt if dt is not None else datetime.min


def _wf_mtime(workflow_id: str) -> float:
    """mtime of ``state/workflows/<workflow_id>.json`` (0.0 when unreadable)."""
    if not workflow_id:
        return 0.0
    try:
        path = progress_service._state_root() / "workflows" / f"{workflow_id}.json"  # noqa: SLF001
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------- state/workflows/* index

# Cache the (expensive) full scan of ``state/workflows/*.json`` by DIRECTORY
# mtime — the dir mtime bumps whenever a workflow file is added/removed, which
# is what changes the set of discoverable runs. Status is deliberately NOT
# cached here (an in-place rewrite does not bump the dir mtime); the caller
# reads a chosen candidate's status fresh. Mirrors the ``_WF_CONFIG_CACHE``
# pattern in ``progress_service``.
_WF_INDEX_CACHE: Dict[str, Any] = {"mtime": None, "entries": []}
_WF_INDEX_LOCK = threading.Lock()


def _scan_workflow_index() -> List[Dict[str, Any]]:
    """Slug-keyed index of every BUILD/TRAINING workflow file (cached by mtime).

    Each entry: ``{wf_id, type, slug, started_at, mtime}``. Best-effort — a
    corrupt/oversized file (via the bounded ``_read_workflow_state``) is skipped,
    and any directory error yields ``[]``. Only workflow files whose ``type`` is
    a BUILD or TRAINING workflow AND that carry a slugifiable course are indexed.
    """
    try:
        wf_dir = progress_service._state_root() / "workflows"  # noqa: SLF001
        dir_mtime = wf_dir.stat().st_mtime
    except OSError:
        return []
    with _WF_INDEX_LOCK:
        if _WF_INDEX_CACHE["mtime"] == dir_mtime:
            return list(_WF_INDEX_CACHE["entries"])
    try:
        files = list(wf_dir.glob("*.json"))
    except OSError:
        return []
    entries: List[Dict[str, Any]] = []
    for path in files:
        wf_id = path.name[: -len(".json")] if path.name.endswith(".json") else path.stem
        state = progress_service._read_workflow_state(wf_id)  # noqa: SLF001 — bounded/guarded
        if not isinstance(state, dict):
            continue
        wf_type = _norm(state.get("type"))
        if wf_type not in BUILD_WORKFLOWS and wf_type != TRAINING_WORKFLOW:
            continue
        params = state.get("params") if isinstance(state.get("params"), dict) else {}
        slug = _slugify(params.get("course_name") or params.get("course_code"))
        if slug is None:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            {
                "wf_id": wf_id,
                "type": wf_type,
                "slug": slug,
                "started_at": state.get("started_at"),
                "mtime": mtime,
            }
        )
    with _WF_INDEX_LOCK:
        _WF_INDEX_CACHE["mtime"] = dir_mtime
        _WF_INDEX_CACHE["entries"] = list(entries)
    return entries


# --------------------------------------------------------- candidate discovery


class _Candidate:
    """One discovered stage run: a progress-usable id + ordering + status."""

    __slots__ = ("progress_id", "workflow", "status", "started", "mtime")

    def __init__(
        self,
        progress_id: str,
        workflow: str,
        status: Optional[str],
        started: datetime,
        mtime: float,
    ) -> None:
        self.progress_id = progress_id
        self.workflow = workflow
        self.status = status
        self.started = started
        self.mtime = mtime


def _newest(cands: List[_Candidate]) -> Optional[_Candidate]:
    """The newest candidate by (started_at, workflow-file mtime); None if empty."""
    if not cands:
        return None
    return max(cands, key=lambda c: (c.started, c.mtime))


def _collect_candidates(slug: str) -> Tuple[List[_Candidate], List[_Candidate]]:
    """Discover every BUILD + TRAINING run for ``slug`` across both sources.

    Sources: ``run_service.list_runs()`` (GUI + recency-windowed CLI records)
    and a direct scan of ``state/workflows/*.json`` (catches older / never-GUI
    training runs the list omits). De-duped by orchestrator workflow id — a
    workflow already surfaced by ``list_runs`` is not re-added from the scan.
    """
    build: List[_Candidate] = []
    train: List[_Candidate] = []
    seen_wf_ids: set = set()

    # (A) GUI + CLI run records.
    try:
        from gui.services import run_service  # noqa: PLC0415

        records = run_service.list_runs()
    except Exception:  # noqa: BLE001 — listing is best-effort
        logger.debug("run_service.list_runs failed during pipeline discovery", exc_info=True)
        records = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        wf = _norm(rec.get("workflow"))
        if wf not in BUILD_WORKFLOWS and wf != TRAINING_WORKFLOW:
            continue
        params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
        if _slugify(params.get("course_name") or params.get("course_code")
                    or rec.get("course_name")) != slug:
            continue
        wf_id = str(rec.get("workflow_id") or "")
        progress_id = str(rec.get("run_id") or wf_id or "")
        if not progress_id:
            continue
        if wf_id:
            seen_wf_ids.add(wf_id)
        cand = _Candidate(
            progress_id=progress_id,
            workflow=wf,
            status=_lower_status(rec.get("status")),
            started=_parse_started(rec.get("started_at") or rec.get("created_at")),
            mtime=_wf_mtime(wf_id),
        )
        (build if wf in BUILD_WORKFLOWS else train).append(cand)

    # (B) Direct workflow-file scan (indexed + cached).
    for entry in _scan_workflow_index():
        if entry["slug"] != slug:
            continue
        wf_id = str(entry["wf_id"])
        if wf_id in seen_wf_ids:
            continue
        seen_wf_ids.add(wf_id)
        # Fresh status for this scan-only run (not in any GUI record).
        state = progress_service._read_workflow_state(wf_id)  # noqa: SLF001
        status = _lower_status((state or {}).get("status"))
        cand = _Candidate(
            progress_id=wf_id,
            workflow=str(entry["type"]),
            status=status,
            started=_parse_started(entry.get("started_at")),
            mtime=float(entry.get("mtime") or 0.0),
        )
        (build if entry["type"] in BUILD_WORKFLOWS else train).append(cand)

    return build, train


# ------------------------------------------------------------- model products


def _resolve_libv2_courses(params: Dict[str, Any]) -> Optional[Path]:
    """The LibV2 ``courses/`` dir (root precedence mirrors progress_service).

    ``params.libv2_root`` → ``ED4ALL_LIBV2_ROOT`` → ``lib.paths.LIBV2_COURSES``.
    """
    cand = params.get("libv2_root")
    try:
        if isinstance(cand, str) and cand.strip():
            root = Path(cand.strip())
            return root if root.name == "courses" else root / "courses"
        env_root = os.environ.get("ED4ALL_LIBV2_ROOT", "").strip()
        if env_root:
            return Path(env_root) / "courses"
        from lib.paths import LIBV2_COURSES  # noqa: PLC0415

        return Path(LIBV2_COURSES)
    except Exception:  # noqa: BLE001 — unresolvable root → no model discovery
        return None


def _model_ids(slug: str, params: Dict[str, Any]) -> List[str]:
    """Sorted ``<model_id>`` dir names under ``LibV2/courses/<slug>/models/``.

    Each dir is one completed/in-flight training run's products. Best-effort:
    absent dir / read error → ``[]``.
    """
    courses = _resolve_libv2_courses(params)
    if courses is None:
        return []
    models = courses / slug / "models"
    try:
        if not models.is_dir():
            return []
        return sorted(d.name for d in models.iterdir() if d.is_dir())
    except OSError:
        return []


# ------------------------------------------------------------- chain assembly


def _single_stage_chain(
    run_id: str,
    slug: Optional[str],
    course_name: Optional[str],
    workflow: str,
    status: Optional[str],
) -> Dict[str, Any]:
    """A one-stage chain naming just the queried run (no correlatable sibling).

    Used for an unresolvable slug (``course_slug: null``) and for a non-build,
    non-training workflow (e.g. ``rag_training``). The stage is inferred from
    the workflow type.
    """
    is_train = _norm(workflow) == TRAINING_WORKFLOW
    stage = "training" if is_train else "build"
    stage_entry: Dict[str, Any] = {
        "stage": stage,
        "label": _STAGE_LABELS[stage],
        "workflow": workflow or (TRAINING_WORKFLOW if is_train else _DEFAULT_BUILD_WORKFLOW),
        "run_id": run_id,
        "present": True,
        "status": status,
        "planned_phases": _plan_names(workflow),
    }
    if is_train:
        stage_entry["model_ids"] = []
    return {
        "run_id": run_id,
        "course_slug": slug,
        "course_name": course_name,
        "current_stage": stage,
        "stages": [stage_entry],
    }


def pipeline_chain(run_id: str) -> Optional[Dict[str, Any]]:
    """Build the ``GET /api/runs/{run_id}/pipeline`` payload; None → caller 404s.

    Resolves the queried run (GUI run id OR bare workflow id), slugifies its
    course, and assembles the Stage-A (build) + Stage-B (training) chain. See the
    module docstring for the full contract. Never raises: any unexpected error is
    caught and surfaced as ``None`` (a 404), mirroring the read-only, best-effort
    posture of ``progress_service``.
    """
    try:
        return _pipeline_chain_impl(run_id)
    except Exception:  # noqa: BLE001 — read-only best-effort: never raise to caller
        logger.exception("pipeline_chain failed for %s", run_id)
        return None


def _pipeline_chain_impl(run_id: str) -> Optional[Dict[str, Any]]:
    # --- resolve the queried run (mirrors progress_service.run_progress).
    record = shared_state.read_run(run_id)
    workflow_id: Optional[str] = None
    if record is not None:
        workflow_id = record.get("workflow_id")
        if not workflow_id:
            return None  # phase-only / failed-at-launch run has no pipeline
    else:
        workflow_id = run_id

    state = progress_service._read_workflow_state(str(workflow_id))  # noqa: SLF001
    if state is None and record is None:
        return None

    params = (state or {}).get("params") or {}
    if not isinstance(params, dict):
        params = {}
    q_type = _norm((state or {}).get("type") or (record or {}).get("workflow") or "")
    q_status = _lower_status((record or {}).get("status") or (state or {}).get("status"))

    course_name = _course_code(params, (record or {}).get("course_name"))
    slug = _slugify(course_name)

    # --- unresolvable slug → single-stage chain naming just this run.
    if slug is None:
        return _single_stage_chain(run_id, None, course_name, q_type, q_status)

    # --- non-build, non-training workflow → single-stage (no training tail).
    if q_type not in BUILD_WORKFLOWS and q_type != TRAINING_WORKFLOW:
        return _single_stage_chain(run_id, slug, course_name, q_type, q_status)

    # --- full BUILD → TRAINING chain, correlated by slug.
    build_cands, train_cands = _collect_candidates(slug)

    # Build stage: the queried run when it IS the build run, else the newest.
    if q_type in BUILD_WORKFLOWS:
        build_run_id: Optional[str] = run_id
        build_workflow = q_type
        build_status = q_status
        build_present = True
    else:
        nb = _newest(build_cands)
        build_run_id = nb.progress_id if nb else None
        build_workflow = nb.workflow if nb else _DEFAULT_BUILD_WORKFLOW
        build_status = nb.status if nb else None
        build_present = nb is not None

    # Training stage: the queried run when it IS the training run, else newest.
    model_ids = _model_ids(slug, params)
    if q_type == TRAINING_WORKFLOW:
        train_run_id: Optional[str] = run_id
        train_status = q_status
        train_run_present = True
    else:
        nt = _newest(train_cands)
        train_run_id = nt.progress_id if nt else None
        train_status = nt.status if nt else None
        train_run_present = nt is not None
    train_present = train_run_present or bool(model_ids)

    build_stage = {
        "stage": "build",
        "label": _STAGE_LABELS["build"],
        "workflow": build_workflow,
        "run_id": build_run_id,
        "present": build_present,
        "status": build_status,
        "planned_phases": _plan_names(build_workflow),
    }
    training_stage = {
        "stage": "training",
        "label": _STAGE_LABELS["training"],
        "workflow": TRAINING_WORKFLOW,
        "run_id": train_run_id,
        "present": train_present,
        "status": train_status,
        "planned_phases": _plan_names(TRAINING_WORKFLOW),
        "model_ids": model_ids,
    }

    return {
        "run_id": run_id,
        "course_slug": slug,
        "course_name": course_name,
        "current_stage": "training" if train_present else "build",
        "stages": [build_stage, training_stage],
    }


__all__ = ["pipeline_chain", "BUILD_WORKFLOWS", "TRAINING_WORKFLOW"]
