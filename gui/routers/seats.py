"""vLLM seat-status REST router — ``/api/seats*`` (the always-on seat monitor).

Mounted at prefix ``/api/seats`` by ``gui.app`` (full + studio modes; NOT
learner), so the paths here are relative (``""``, ``"/run/{run_id}"``). Thin by
contract: every probe, classification, cache and workflow read lives in
``gui.services.seat_service`` — this module maps a request to a service call and
a typed error.

Endpoint contract (frontend-facing):

* ``GET "/api/seats"`` → the GLOBAL seat monitor (renders with no build
  running)::

      {generated_at, registry_configured, docker_available,
       seats: [{name, base_url, live, state, container, model, since, since_ms}]}

  ``state`` ∈ ``live | loading | down | unknown``: ``loading`` means the seat's
  container is running but ``/v1/models`` has not answered yet (a large seat's
  cold start takes ~9 minutes — progress, not an alarm); ``unknown`` means we
  genuinely cannot tell (docker absent / no perms / no container registered).
  ``since`` / ``since_ms`` are the age of the CURRENT state as observed by this
  server process (``null`` on the first observation — nothing to compare
  against). The seat list is exactly the ``ED4ALL_SEAT_BASE_URLS`` registry, in
  registry order — no seat name is hardcoded anywhere in this surface.

* ``GET "/api/seats/run/{run_id}"`` → the same payload PLUS the run's phase
  context::

      {run_id, workflow, status, effective_status, current_phase,
       expected_seats: [...]|null, mismatch: [{seat, state, base_url}]}

  and an ``expected: bool`` flag on each seat. ``expected_seats`` mirrors the
  phase's ``seats:`` annotation in ``config/workflows.yaml``: a name list, ``[]``
  (declared seat-free) or ``null`` (absent — no opinion, so no mismatch is ever
  reported). ``mismatch`` names an expected seat that is neither live nor
  loading — the "this phase needs <seat>, <seat> is DOWN" signal. ``run_id``
  accepts a GUI run id or a ``WF-*`` workflow id; an unknown run → 404 typed
  error.

Both handlers are plain ``def`` so the (bounded, but I/O-bound: HTTP probes +
one ``docker ps`` read) service call runs in the threadpool instead of stalling
the event loop. The service never raises — it degrades to ``unknown`` — so the
``{error, detail}`` 500 arm here is belt-and-braces only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from gui.services import seat_service

router = APIRouter()


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@router.get("")
@router.get("/")
def seats() -> Any:
    """Return the cached global seat overview (every registered seat)."""
    try:
        return seat_service.seat_overview()
    except Exception as exc:  # noqa: BLE001 — the service degrades; belt-and-braces
        return _error(500, "seats_failed", str(exc))


@router.get("/run/{run_id}")
def seats_for_run(run_id: str) -> Any:
    """Return the phase-aware seat overview for ``run_id`` (404 = unknown run)."""
    try:
        payload = seat_service.seat_overview(run_id)
    except Exception as exc:  # noqa: BLE001
        return _error(500, "seats_run_failed", str(exc))
    if payload is None:
        return _error(404, "run_not_found", f"no run resolves for id {run_id[:80]!r}")
    return payload


__all__ = ["router"]
