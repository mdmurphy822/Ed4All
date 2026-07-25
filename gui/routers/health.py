"""Health-doctor REST router — ``/api/health/doctor*`` (the in-process doctor).

Mounted at prefix ``/api/health`` by ``gui.app`` (full + studio modes), so the
paths here are relative (``"/doctor"``, ``"/doctor/refresh"``,
``"/doctor/run/{run_id}"``). The router is thin: the diagnostics registry run +
result shaping live in ``gui.services.health_service`` — this module only maps a
request to a service call and a typed error.

Endpoint contract (frontend-facing):

* ``GET  "/doctor"`` → the CACHED global preflight payload
  ``{generated_at, exit_state, exit_code, verdict, groups:[{group, checks:[...]}],
  summary}``. ``exit_state`` is ``healthy | degraded | critical`` (the banner
  verdict); ``checks`` are CheckResult-shaped dicts (name/group/severity/summary/
  detail/remediation/data). Group-agnostic — a newly-registered check group
  appears automatically.
* ``POST "/doctor/refresh"`` → the SAME shape, re-running the checks now
  (bypasses the ~30 s TTL cache).
* ``GET  "/doctor/run/{run_id}"`` → the run-scoped post-mortem payload (the
  global shape PLUS ``{run_id, orchestrator_run_id, effective_status,
  usage:{present, rows}}``). ``run_id`` accepts a GUI run id, a ``WF-*``
  workflow id, or a bare orchestrator run id. An unknown run → 404 typed error.
* ``POST "/doctor/run/{run_id}/refresh"`` → the run-scoped shape, re-running now.

Handlers are plain ``def`` so the (potentially heavy — torch VRAM probe / disk
post-mortem) service call runs in the threadpool rather than stalling the event
loop. Neither the service nor these handlers raise: a compute failure already
degrades to a valid (error-shaped) payload; an unexpected router-level failure
maps to a typed ``{error, detail}`` 500.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gui.services import health_service

router = APIRouter()


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@router.get("/doctor")
def health_doctor() -> Any:
    """Return the cached global preflight doctor payload."""
    try:
        return health_service.get_health()
    except Exception as exc:  # noqa: BLE001 — the service degrades; this is belt-and-braces
        return _error(500, "health_doctor_failed", str(exc))


@router.post("/doctor/refresh")
def health_doctor_refresh() -> Any:
    """Force a re-run of the global preflight doctor (bypasses the TTL cache)."""
    try:
        return health_service.get_health(force_refresh=True)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "health_doctor_refresh_failed", str(exc))


@router.get("/doctor/run/{run_id}")
def health_doctor_run(run_id: str) -> Any:
    """Return the cached run-scoped post-mortem payload (404 for an unknown run)."""
    try:
        payload = health_service.run_health(run_id)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "health_doctor_run_failed", str(exc))
    if payload is None:
        return _error(404, "run_not_found", f"no run resolves for id {run_id[:80]!r}")
    return payload


@router.post("/doctor/run/{run_id}/refresh")
def health_doctor_run_refresh(run_id: str) -> Any:
    """Force a re-run of the run-scoped post-mortem (404 for an unknown run)."""
    try:
        payload = health_service.run_health(run_id, force_refresh=True)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "health_doctor_run_refresh_failed", str(exc))
    if payload is None:
        return _error(404, "run_not_found", f"no run resolves for id {run_id[:80]!r}")
    return payload


__all__ = ["router"]
