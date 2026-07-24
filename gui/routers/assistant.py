"""Assistant REST router — ``/api/assistant/*`` (the sandboxed operator chat).

Mounted at prefix ``/api/assistant`` by ``gui.app`` (full + studio modes), so
paths here are relative (``"/chat"``, ``"/status"``). The router is thin: the
engine adapter lives in ``gui.services.assistant_service``; the sandbox itself
(tool whitelist, loopback guard, decision capture, round cap) is owned by
``lib.assistant.AssistantEngine`` — the GUI adds no capability on top.

Endpoint contract (frontend-facing):

* ``POST "/chat"``  → ``{"message", "history"?, "debug"?, "run_id"?}`` →
  ``{"reply", "history", "tool_calls", "rounds", "model", "seat", "debug"?}``.
  ``model`` / ``seat`` name the loopback seat that actually answered the turn
  (``null`` on a stubbed/older turn without the metadata). The endpoint is
  stateless; the returned ``history`` round-trips through the client as the
  next request's ``history``. ``debug: true`` runs the turn on a debug-mode
  engine seeded with the failed-run context for ``run_id`` (omitted =
  auto-resolved last failure); the service builds that context once per
  failed run and the client re-sends the flags on every turn of a debug
  conversation.
* ``GET  "/debug-context"`` (optional ``?run_id=WF-...``) →
  ``{"available", "run_id", "course", "failed_phase", "summary"}`` — the
  404-free "Debug last failure" probe (``available: false`` + ``reason``
  when no failed run resolves).
* ``GET  "/status"`` → ``{"seat_serving": bool, "model": str|null,
  "seat": str|null}`` for the panel's status pill — the dynamically resolved
  live seat/model (a config error adds a ``detail`` string, seat/model null).
* ``GET  "/seat/status"`` → ``{"live_seat": str|null, "seats": {name: bool},
  "starting": bool, "last_seat_error": str|null}`` — the "Seat model" button's
  own probe. Deliberately a SEPARATE endpoint from ``/status`` (whose
  three-key pill shape is asserted verbatim by clients/tests and stays
  untouched): this one reports EVERY candidate seat in the priority walk plus
  the in-flight-start latch. A config error adds a ``detail`` string.
* ``POST "/seat"`` (no body) → start the assistant's OWN seat
  (``ED4ALL_ASSISTANT_SEAT``) through the lifecycle coherent-start path, on a
  background thread. ``200 {"ok": true, "state": "starting", "seat", "error":
  null}`` — returned immediately, and identically for a second click while a
  start is in flight (SINGLE-FLIGHT: never a second launch). Refusals carry NO
  side effect: ``409 assistant_seat_already_live`` when a seat already answers
  (the detail names it), ``500 assistant_seat_no_launch_spec`` when the seat has
  no ``ED4ALL_SEAT_LAUNCH_SPECS`` entry (detail is that message verbatim), and
  ``500 assistant_config_error`` on a non-loopback seat config. Progress is
  observed by polling ``GET "/seat/status"``; a failed start surfaces there as
  ``last_seat_error``.

Errors surface as a typed ``{error, detail}`` body: 503 when the assistant
seat is down (``detail`` carries the start instructions), 500 on a
non-loopback misconfiguration, 422 on an empty message or a malformed
``run_id``, 409 when a debug chat is requested but no failed run resolves —
never a fabricated success. Endpoints are plain ``def`` so the blocking LLM
call runs in the threadpool rather than stalling the event loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gui.services import assistant_service
from gui.services.assistant_service import (
    AssistantProviderNotLocal,
    AssistantSeatUnavailable,
    DebugContextUnavailable,
)

router = APIRouter()


# ----------------------------------------------------------------- request models


class AssistantChatRequest(BaseModel):
    """Body for ``POST "/chat"`` — one user message + the round-tripped history.

    ``debug`` + ``run_id`` select the debug-mode engine (failed-run context);
    the client re-sends both on every turn of a debug conversation.
    """

    message: str
    history: Optional[List[Dict[str, Any]]] = None
    debug: bool = False
    run_id: Optional[str] = None


# ------------------------------------------------------------------- error helper


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


# ----------------------------------------------------------------------- endpoints


def _validated_run_id(raw: Optional[str]) -> Any:
    """Return a stripped run id / None, or an error response on garbage."""
    run_id = (raw or "").strip() or None
    if run_id is not None and not assistant_service.is_valid_run_id(run_id):
        return _error(
            422,
            "invalid_run_id",
            f"{run_id[:80]!r} is not a valid run id (expected WF-YYYYMMDD-xxxxxxxx)",
        )
    return run_id


@router.post("/chat")
def assistant_chat(req: AssistantChatRequest) -> Any:
    """Run one assistant turn (stateless; history round-trips via the client)."""
    message = (req.message or "").strip()
    if not message:
        return _error(422, "empty_message", "message must be a non-empty string")
    run_id: Optional[str] = None
    if req.debug:
        run_id = _validated_run_id(req.run_id)
        if isinstance(run_id, JSONResponse):
            return run_id
    try:
        return assistant_service.chat(
            message, history=req.history, debug=bool(req.debug), run_id=run_id
        )
    except AssistantSeatUnavailable as exc:
        # The exception text IS the operator-facing start instructions.
        return _error(503, "assistant_seat_unavailable", str(exc))
    except AssistantProviderNotLocal as exc:
        return _error(
            500,
            "assistant_config_error",
            f"Assistant misconfigured (non-loopback seat URL): {exc}",
        )
    except DebugContextUnavailable as exc:
        # Debug chat requested but no failed run resolves — the probe's
        # available=false case surfacing on the chat path.
        return _error(409, "debug_context_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 — unexpected engine failure
        return _error(500, "assistant_chat_failed", str(exc))


@router.get("/debug-context")
def assistant_debug_context(run_id: Optional[str] = None) -> Any:
    """Probe the failed-run debug context (404-free; powers the panel button)."""
    validated = _validated_run_id(run_id)
    if isinstance(validated, JSONResponse):
        return validated
    try:
        return assistant_service.debug_context_probe(validated)
    except Exception as exc:  # noqa: BLE001 — probe must degrade, not 500 blindly
        return _error(500, "assistant_debug_context_failed", str(exc))


@router.get("/status")
def assistant_status() -> Any:
    """Return the seat-liveness + model probe for the panel's status pill."""
    try:
        return assistant_service.status()
    except Exception as exc:  # noqa: BLE001 — service handles config errors itself
        return _error(500, "assistant_status_failed", str(exc))


@router.get("/seat/status")
def assistant_seat_status() -> Any:
    """Return the per-seat liveness + in-flight-start state for the button."""
    try:
        return assistant_service.seat_status()
    except Exception as exc:  # noqa: BLE001 — service handles config errors itself
        return _error(500, "assistant_seat_status_failed", str(exc))


#: ``seat_nano`` state → (HTTP status, typed error code). ``starting`` is the
#: only success state; every other state is a REFUSAL with no side effect.
_SEAT_ERROR_STATUS = {
    "already_live": (409, "assistant_seat_already_live"),
    "no_launch_spec": (500, "assistant_seat_no_launch_spec"),
    "config_error": (500, "assistant_config_error"),
    "start_failed": (500, "assistant_seat_start_failed"),
}


@router.post("/seat")
def assistant_seat_start() -> Any:
    """Seat the assistant's own model (explicit human action; single-flight)."""
    try:
        result = assistant_service.seat_nano()
    except Exception as exc:  # noqa: BLE001 — never a fabricated success
        return _error(500, "assistant_seat_start_failed", str(exc))
    if result.get("ok"):
        return result
    state = str(result.get("state") or "start_failed")
    status_code, error_code = _SEAT_ERROR_STATUS.get(
        state, (500, "assistant_seat_start_failed")
    )
    # The service's message is surfaced VERBATIM (the no-launch-spec text is
    # the operator's remediation instruction).
    return _error(status_code, error_code, str(result.get("error") or state))


__all__ = ["router"]
