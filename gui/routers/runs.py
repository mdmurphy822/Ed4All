"""RUNS + ACTIVITY router for the Ed4All control-plane GUI.

Mounted at the bare ``/api`` prefix (see ``gui/app.py::_ROUTER_MOUNTS``) so it
owns three distinct path families:

- ``/api/workflows``               — list launchable workflows + phase shapes.
- ``/api/runs`` + ``/api/runs/*``  — launch pipeline/phase, list, status, cancel.
- ``/ws/runs/{run_id}``            — WebSocket: stream log lines + final status.
- ``/api/activity/*``              — the Claude<->GUI events bridge.

All real, no stubs: every handler calls ``gui.services.run_service`` (which
wraps the real orchestrator) or ``gui.shared_state`` directly. Errors surface as
typed ``{error, detail}`` payloads with proper status codes — never a fabricated
success.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gui import shared_state
from gui.services import pipeline_service, progress_service, run_service

logger = logging.getLogger("gui.routers.runs")

router = APIRouter()


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed FLAT ``{error, detail}`` JSON error response.

    Matches the settings/courses/retrieval routers' wire shape
    (``{"error": ..., "detail": ...}``) rather than the nested
    ``{"detail": {"error": ..., "detail": ...}}`` that ``HTTPException`` produces.
    """
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


# ----------------------------------------------------------------- models


class LaunchPipelineRequest(BaseModel):
    """Body for ``POST /api/runs`` — launch a full workflow pipeline."""

    workflow: str = ""
    course_name: str = ""
    corpus: Optional[str] = None  # upload_id or filesystem path
    weeks: Optional[int] = None
    mode: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Phase 4 §5.1(E) Resume: the GUI run_id of a prior interrupted/failed
    # pipeline run to re-drive from its orchestrator checkpoint. When set,
    # workflow/course_name/corpus are ignored (the prior workflow state is
    # authoritative) — so they default empty to keep a resume body minimal.
    resume_run_id: Optional[str] = None
    # I1 review flow: on a ``resume_run_id`` resume, clear the persisted
    # ``--stop-after`` halt marker first so the resumed build continues PAST the
    # objectives-review checkpoint instead of immediately re-pausing. Ignored on
    # a fresh (non-resume) launch.
    clear_stop_after: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)


class LaunchPhaseRequest(BaseModel):
    """Body for ``POST /api/runs/phase`` — execute a single phase."""

    workflow: str
    phase: str
    course_name: Optional[str] = None
    project_id: Optional[str] = None
    mode: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class ActivityPostRequest(BaseModel):
    """Body for ``POST /api/activity/post`` — post a GUI-sourced event."""

    kind: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    # ``source`` is fixed to "gui" server-side; accepted for forward-compat.
    source: str = "gui"


# --------------------------------------------------------------- workflows


@router.get("/workflows")
async def list_workflows() -> Any:
    """Return every launchable workflow with its phase shape (real config)."""
    try:
        return {"workflows": run_service.list_workflows()}
    except Exception as exc:  # noqa: BLE001 — surface the real config error
        logger.exception("list_workflows failed")
        return _error(500, "list_workflows_failed", str(exc))


# -------------------------------------------------------------------- runs


@router.post("/runs")
async def launch_run(req: LaunchPipelineRequest) -> Any:
    """Launch a full workflow pipeline via the real orchestrator."""
    result = await run_service.launch_pipeline(req.model_dump())
    if result.get("status") == "failed":
        # Real failure (bad workflow, creation error) — 422 so the SPA shows it.
        # ``detail`` carries the human error string; the full result dict is
        # serialized in so the SPA still has the structured payload.
        return _error(
            422,
            result.get("error", "launch_failed"),
            str(result.get("error") or result),
        )
    return result


@router.post("/runs/phase")
async def launch_phase_run(req: LaunchPhaseRequest) -> Any:
    """Execute a single workflow phase via the real phase pathway."""
    result = await run_service.launch_phase(req.model_dump())
    if result.get("status") == "failed":
        return _error(
            422,
            result.get("error", "phase_failed"),
            str(result.get("error") or result),
        )
    return result


@router.get("/runs/phase-durations")
async def get_phase_durations(workflow: Optional[str] = Query(default=None)) -> Any:
    """Median per-phase ``duration_ms`` over completed history of ``workflow``.

    Phase 4 §5.1(C) ETA priors. Returns the exact frontend contract::

        {"workflow": <name>,
         "durations": {"<phase>": {"median_ms": <int|null>, "n": <int>}},
         "source": "history" | "prior" | "mixed"}

    ``source="history"`` when >= 2 real runs informed a phase, ``"prior"`` when
    only the static cold-start prior was used, ``"mixed"`` otherwise. A typed 422
    is returned when ``workflow`` is missing or not a known workflow / stage alias.

    NOTE: registered BEFORE ``GET /runs/{run_id}`` so the static ``phase-durations``
    segment is not captured by the ``{run_id}`` path parameter.
    """
    if not workflow:
        return _error(422, "missing_workflow", "query parameter 'workflow' is required")
    known = set(run_service.SUPPORTED_WORKFLOWS) | set(
        run_service.COURSEFORGE_STAGE_SUBCOMMANDS
    )
    if workflow not in known:
        return _error(
            422,
            "unknown_workflow",
            f"unknown workflow {workflow!r}; choose from {sorted(known)}",
        )
    return run_service.phase_duration_medians(workflow)


@router.get("/runs")
async def list_runs(limit: Optional[int] = Query(default=None, ge=0)) -> Dict[str, Any]:
    """Return GUI run records, newest-first.

    Optional ``?limit=`` caps the number of records returned; omitted (default)
    returns all of them, preserving the prior unbounded behavior.
    """
    return {"runs": run_service.list_runs(limit=limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> Any:
    """Return the status record for a single run."""
    record = run_service.run_status(run_id)
    if record is None:
        return _error(404, "unknown_run", run_id)
    return record


@router.get("/runs/{run_id}/timeline")
async def get_run_timeline(run_id: str) -> Any:
    """Return the GUI-side per-phase timing timeline for a run (Phase 0, Tier 1).

    Per-phase durations are derived entirely GUI-side from the ISO-prefixed
    ``[phase] <name> done|skipped|failed`` log lines the backend already emits —
    zero orchestrator change. 404 with the typed ``{error, detail}`` body when the
    run is unknown, mirroring ``get_run``.
    """
    if run_service.run_status(run_id) is None:
        return _error(404, "unknown_run", run_id)
    return run_service.derive_phase_timeline(run_id)


@router.get("/runs/{run_id}/progress")
async def get_run_progress(run_id: str) -> Any:
    """Return the stage-tracker payload for a run (rail + live stats band).

    Read-only merge of the run's ``config/workflows.yaml`` phase plan with its
    orchestrator workflow state, checkpoint wall-clocks, a bounded tail of the
    OP2 ``llm_usage.jsonl`` tap, and a TTL-cached seat probe — cheap enough for
    a 2-5s poll (see ``gui.services.progress_service``). Accepts a GUI run id
    or a bare orchestrator workflow id; typed 404 when neither resolves.
    """
    try:
        payload = progress_service.run_progress(run_id)
    except Exception as exc:  # noqa: BLE001 — surface the real error
        logger.exception("run_progress failed for %s", run_id)
        return _error(500, "run_progress_failed", str(exc))
    if payload is None:
        return _error(404, "unknown_run", run_id)
    return payload


@router.get("/runs/{run_id}/pipeline")
async def get_run_pipeline(run_id: str) -> Any:
    """Return the FULL-pipeline chain for a run: Stage-A build → Stage-B training.

    Read-only correlation of the queried run with its sibling stage by course
    slug: the course BUILD run (``textbook_to_course`` / ``course_generation``)
    and the ``trainforge_train`` LoRA-training run are two separate run records
    tied only by slug (see ``gui.services.pipeline_service``). Each stage carries
    a ``run_id`` usable for ``/api/runs/{id}/progress``, its ``planned_phases``,
    and (training) discovered ``model_ids``. Accepts a GUI run id or a bare
    orchestrator workflow id; typed 404 when neither resolves.
    """
    try:
        payload = pipeline_service.pipeline_chain(run_id)
    except Exception as exc:  # noqa: BLE001 — surface the real error
        logger.exception("pipeline_chain failed for %s", run_id)
        return _error(500, "pipeline_chain_failed", str(exc))
    if payload is None:
        return _error(404, "unknown_run", run_id)
    return payload


@router.get("/runs/{run_id}/output-tail")
async def get_run_output_tail(run_id: str) -> Any:
    """Return the live-output tail for a run's CURRENT phase.

    The last few complete rows of the phase's per-unit resume sidecar, each
    mapped to a bounded ``{seq, label, text}`` display record (HTML-stripped,
    hard-truncated) — the Studio build page's "Live output" panel. Absent
    sidecar / unmapped phase → an honest ``rows: []``. Typed 404 when the run
    is unknown (mirrors ``/progress``).
    """
    try:
        payload = progress_service.output_tail(run_id)
    except Exception as exc:  # noqa: BLE001 — surface the real error
        logger.exception("output_tail failed for %s", run_id)
        return _error(500, "output_tail_failed", str(exc))
    if payload is None:
        return _error(404, "unknown_run", run_id)
    return payload


@router.get("/runs/{run_id}/validation-report")
async def get_validation_report(run_id: str) -> Any:
    """Return the failure / validation report for a run (Marketable-v1 A6).

    Locates the run's ``courseforge_validation_report.json`` (resolved from the
    run record's project export) and returns it plus a digestible failed-gate
    list ``[{phase, gate_id, severity, message, issues_count}]`` derived from the
    persisted ``gate_results``. The report body is ``None`` (with an explanatory
    ``note``) when no aggregator file exists; the failed-gate digest is still
    returned. 404 when the run is unknown.
    """
    try:
        return run_service.validation_report(run_id)
    except KeyError:
        return _error(404, "unknown_run", run_id)
    except Exception as exc:  # noqa: BLE001 — surface the real error
        logger.exception("validation_report failed for %s", run_id)
        return _error(500, "validation_report_failed", str(exc))


@router.get("/runs/{run_id}/review")
async def get_run_review(run_id: str) -> Any:
    """Return the objectives-review context for a paused run (I1 review flow).

    Surfaces the paused-at-phase, the persisted ``--stop-after`` intent, and the
    exact objectives file the editor writes / a plain resume reads (best-effort;
    ``None`` when the export can't be resolved). The Create-wizard review panel
    fetches this to render the "Edit objectives" + "Resume build" affordances.
    404 (typed) when the run is unknown, mirroring ``get_run``.
    """
    info = run_service.paused_review_info(run_id)
    if info is None:
        return _error(404, "unknown_run", run_id)
    return info


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Any:
    """Request cancellation of a run (HONEST two-stage, Phase 4 §5.1(E)).

    Cancellation is cooperative / phase-boundary, so this does NOT claim an
    instant stop. On a fresh request the service writes the NON-terminal
    ``status="cancel_requested"`` and we return **HTTP 202 Accepted** with
    ``{run_id, status:"cancel_requested"}`` — the authoritative flip to the
    terminal ``cancelled`` lands later when the run driver confirms the
    orchestrator exited. An already-terminal or already-requested run returns its
    current status with **200** (idempotent no-op); an unknown run is a typed 404.
    """
    result = run_service.cancel_run(run_id)
    if result.get("error") == "unknown_run":
        return _error(404, "unknown_run", str(result.get("detail", run_id)))
    if result.get("status") == "cancel_requested" and not result.get("note"):
        # Fresh cancel request accepted; the terminal flip is still pending.
        return JSONResponse(status_code=202, content=result)
    return result


# ----------------------------------------------------------------- ws logs


@router.websocket("/ws/runs/{run_id}")
async def ws_run_logs(websocket: WebSocket, run_id: str) -> None:
    """Stream a run's log lines, then a final status frame.

    Frame shapes (JSON):
    - ``{"type": "line", "line": "<log text>"}`` per appended chunk,
    - ``{"type": "status", "status": "<status>", "gates": <gate_results>}`` when
      the run reaches a terminal state, then the socket closes.

    Polls ``tail_log`` (byte-offset cursor) + the run record. No fake data: if
    the run is unknown, an error frame is sent and the socket closes.
    """
    await websocket.accept()
    record = run_service.run_status(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "error": "unknown_run", "run_id": run_id})
        await websocket.close()
        return

    offset = 0
    # ``paused`` (graceful stop / --stop-after) is terminal for THIS run's WS —
    # the run halts at a checkpoint and only continues under a fresh resume run,
    # so the socket must send the status frame + close instead of polling forever.
    terminal = {"completed", "failed", "cancelled", "interrupted", "paused"}
    try:
        while True:
            text, offset = shared_state.tail_log(run_id, offset)
            if text:
                for line in text.splitlines():
                    await websocket.send_json({"type": "line", "line": line})

            record = run_service.run_status(run_id) or record
            status = record.get("status")
            if status in terminal:
                # Drain any trailing log written between the last tail + now.
                text, offset = shared_state.tail_log(run_id, offset)
                if text:
                    for line in text.splitlines():
                        await websocket.send_json({"type": "line", "line": line})
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": status,
                        "gates": record.get("gate_results"),
                        "error": record.get("error"),
                    }
                )
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("ws client disconnected from run %s", run_id)
        return
    except Exception as exc:  # noqa: BLE001 — report, don't crash the socket
        logger.exception("ws stream failed for %s", run_id)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------- activity


@router.get("/activity/events")
async def activity_events(
    since: int = 0, limit: Optional[int] = Query(default=None, ge=0)
) -> Dict[str, Any]:
    """Return events with ``seq >= since`` from the Claude<->GUI bridge.

    Optional ``?limit=`` caps how many events are returned (most recent first by
    ``seq`` is preserved; we keep append order but trim to the last ``limit``).
    ``shared_state.read_events`` only accepts ``since``, so the limit is applied
    by slicing here. Omitted (default) returns every event >= ``since``.
    """
    events = shared_state.read_events(since)
    if limit is not None:
        if limit <= 0:
            events = []
        else:
            # Keep the most recent ``limit`` events (events are in append order).
            events = events[-limit:]
    return {"events": events}


@router.post("/activity/post")
async def activity_post(req: ActivityPostRequest) -> Any:
    """Append a GUI-sourced event to the Claude<->GUI bridge."""
    try:
        record = shared_state.append_event("gui", req.kind, req.payload)
    except ValueError as exc:
        return _error(422, "invalid_event", str(exc))
    return {"event": record}


__all__: List[str] = ["router"]
