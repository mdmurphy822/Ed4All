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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gui import shared_state
from gui.services import run_service

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

    workflow: str
    course_name: str
    corpus: Optional[str] = None  # upload_id or filesystem path
    weeks: Optional[int] = None
    mode: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
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


@router.get("/runs")
async def list_runs(limit: Optional[int] = None) -> Dict[str, Any]:
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


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Any:
    """Request cancellation of a run."""
    result = run_service.cancel_run(run_id)
    if result.get("error") == "unknown_run":
        return _error(404, "unknown_run", str(result.get("detail", run_id)))
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
    terminal = {"completed", "failed", "cancelled", "interrupted"}
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
async def activity_events(since: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
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
