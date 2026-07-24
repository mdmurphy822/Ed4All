"""Assistant chat service — thin GUI adapter over ``lib.assistant.AssistantEngine``.

The engine owns the whole sandbox (system prompt, typed tool whitelist,
loopback guard, decision capture, the 6-round tool cap); this service adds
NOTHING on top. It only:

* lazily builds + caches ONE engine per server process (construction is
  loopback-enforced — a non-loopback ``ED4ALL_ASSISTANT_BASE_URL`` raises
  ``AssistantProviderNotLocal`` straight through to the router);
* resolves the GUI-level ``assistant.autostart`` setting and passes it to
  ``AssistantEngine.ensure_seat(allow_autostart=...)`` — the engine's own
  ``ED4ALL_ASSISTANT_AUTOSTART`` policy still governs whether a start actually
  occurs (the GUI never widens the engine's policy, it can only decline to
  attempt);
* runs one stateless turn (history round-trips through the client);
* appends one ``assistant``-sourced event per exchange to the shared
  ``state/gui/events.jsonl`` activity bridge (best-effort — telemetry never
  eats a reply), so exchanges surface in the Activity tab beside the
  Claude/GUI events;
* exposes the panel's explicit "Seat model" action (:func:`seat_status` /
  :func:`seat_nano`) — a HUMAN-triggered start of the assistant's OWN seat,
  routed through the same lifecycle coherent-start path the engine's autostart
  uses (:func:`lib.assistant.client.autostart_seat` →
  ``lib.vllm_container_lifecycle.start_seat_coherent``). The GUI never names a
  script: the seat name comes from ``ED4ALL_ASSISTANT_SEAT`` and the script
  from the ``ED4ALL_SEAT_LAUNCH_SPECS`` registry entry for that seat.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from lib.assistant import (
    AssistantEngine,
    AssistantProviderNotLocal,  # noqa: F401 — re-exported for the router
    AssistantSeatUnavailable,  # noqa: F401 — re-exported for the router
)
from lib.assistant.debug_context import (
    DebugContextUnavailable,  # noqa: F401 — re-exported for the router
    build_debug_context,
)
from lib.assistant.tools import RUN_ID_RE

try:  # S1 dynamic seat resolution lands with the sibling client task.
    from lib.assistant.client import resolve_active_seat
except ImportError:  # pragma: no cover — resolver not yet present in the tree

    def resolve_active_seat(*_args, **_kwargs):  # type: ignore[misc]
        """Fallback until ``lib.assistant.client.resolve_active_seat`` (S1) lands.

        Raises the same config error the loopback guard uses so ``status()``
        surfaces an honest ``detail`` instead of fabricating a serving seat.
        """
        raise AssistantProviderNotLocal(
            "dynamic seat resolution unavailable "
            "(lib.assistant.client.resolve_active_seat not present)"
        )

from lib.assistant.client import (  # noqa: E402 — sits with the seat imports
    autostart_seat,
    reset_seat_cache,
    resolve_assistant_base_url,
    resolve_assistant_seat,
    resolve_assistant_seat_priority,
    seat_is_serving,
)
from lib.vllm_container_lifecycle import (  # noqa: E402
    ENV_SEAT_LAUNCH_SPECS,
    parse_seat_registry,
    resolve_seat_launch_spec,
)

from gui import settings_store, shared_state

logger = logging.getLogger(__name__)

#: Max characters of the user question echoed into the activity-event summary.
_SUMMARY_QUESTION_CHARS = 160

# One engine per server process (built lazily on first use). Requests are
# stateless — the conversation history round-trips through the client — so a
# single shared engine is safe; it holds only the client config + capture.
_ENGINE: Optional[AssistantEngine] = None

# Debug-mode state, keyed by the REQUESTED run id ("" = auto-resolve "last
# failure"), with an alias under the RESOLVED run id. ``build_debug_context``
# shells out to ``ed4all doctor`` (bounded but slow), so the context is built
# ONCE per failed run per server process: the panel's mount probe pays the
# cost and every debug chat turn of that conversation reuses the same context
# + engine. ``reset_engine()`` clears both (tests / config changes).
_AUTO_KEY = ""
_DEBUG_CONTEXTS: Dict[str, Dict[str, Any]] = {}
_DEBUG_ENGINES: Dict[str, AssistantEngine] = {}


def _build_engine() -> AssistantEngine:
    """Engine factory (module-level so tests can monkeypatch it)."""
    return AssistantEngine()


def get_engine() -> AssistantEngine:
    """Return the process-cached engine, building it on first call.

    Raises ``AssistantProviderNotLocal`` when the resolved assistant base URL
    is non-loopback (the engine's construction-time guard — surfaced to the
    router as a config error, never swallowed).
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


def reset_engine() -> None:
    """Drop the cached engine + debug caches (tests / config changes)."""
    global _ENGINE
    _ENGINE = None
    _DEBUG_CONTEXTS.clear()
    _DEBUG_ENGINES.clear()


def is_valid_run_id(run_id: str) -> bool:
    """True when ``run_id`` matches the WF-YYYYMMDD-xxxxxxxx shape.

    The same validation ``lib.assistant`` applies before any id reaches a
    path or argv — the router uses it to 422 on garbage BEFORE the debug
    machinery is touched.
    """
    return bool(RUN_ID_RE.match(str(run_id)))


def _get_debug_context(run_id: Optional[str]) -> Dict[str, Any]:
    """Return the (cached) debug context for ``run_id`` (None = auto-resolve).

    Raises ``DebugContextUnavailable`` when no failed run resolves — never
    cached, so a failure appearing later becomes probeable without a restart.
    """
    key = run_id or _AUTO_KEY
    ctx = _DEBUG_CONTEXTS.get(key)
    if ctx is None:
        ctx = build_debug_context(run_id)
        _DEBUG_CONTEXTS[key] = ctx
        resolved = str(ctx.get("run_id") or "")
        if resolved and resolved != key:
            _DEBUG_CONTEXTS[resolved] = ctx
    return ctx


def _build_debug_engine(debug_context: str) -> AssistantEngine:
    """Debug-engine factory (module-level so tests can monkeypatch/observe)."""
    return AssistantEngine(mode="debug", debug_context=debug_context)


def _get_debug_engine(ctx: Dict[str, Any]) -> AssistantEngine:
    """One cached debug-mode engine per resolved failed run."""
    key = str(ctx.get("run_id") or _AUTO_KEY)
    engine = _DEBUG_ENGINES.get(key)
    if engine is None:
        engine = _build_debug_engine(str(ctx.get("text") or ""))
        _DEBUG_ENGINES[key] = engine
    return engine


def debug_context_probe(run_id: Optional[str] = None) -> Dict[str, Any]:
    """Probe the debug context — the 404-free ``/debug-context`` payload.

    Returns ``{available: True, run_id, course, failed_phase, summary}`` when
    a failed run resolves, else ``{available: False, ..., reason}`` (never an
    HTTP error for the "nothing to debug" case).
    """
    try:
        ctx = _get_debug_context(run_id)
    except DebugContextUnavailable as exc:
        return {
            "available": False,
            "run_id": run_id,
            "course": None,
            "failed_phase": None,
            "summary": None,
            "reason": str(exc),
        }
    return {
        "available": True,
        "run_id": ctx.get("run_id"),
        "course": ctx.get("slug"),
        "failed_phase": ctx.get("failed_phase"),
        "summary": ctx.get("summary"),
    }


def autostart_enabled() -> bool:
    """Resolve the GUI ``assistant.autostart`` setting (default ``False``).

    Read from the canonical settings doc so both the GUI Settings surface and
    the MCP bridge (``gui_set_setting("assistant.autostart", ...)``) govern it.
    """
    try:
        doc = settings_store.load_settings()
    except Exception:  # noqa: BLE001 — a broken store must not break chat
        logger.exception("assistant: settings load failed; autostart off")
        return False
    block = doc.get("assistant") if isinstance(doc, dict) else None
    if not isinstance(block, dict):
        return False
    return bool(block.get("autostart"))


def chat(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    *,
    debug: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one assistant turn; return the wire-shaped response dict.

    ``debug=True`` routes the turn through a debug-mode engine seeded with the
    failed-run context for ``run_id`` (None = auto-resolved last failure). The
    context/engine are cached per resolved run, so turn-0 builds them once and
    later turns of the conversation (the client re-sends ``debug`` +
    ``run_id`` with the round-tripped history) reuse them. Same sandbox: the
    debug engine keeps the identical tool whitelist.

    Raises ``AssistantSeatUnavailable`` (seat down; message carries the start
    instructions), ``AssistantProviderNotLocal`` (misconfiguration), and — in
    debug mode — ``DebugContextUnavailable`` (no failed run) for the router
    to map to 503 / 500 / 409.
    """
    ctx: Optional[Dict[str, Any]] = None
    if debug:
        ctx = _get_debug_context(run_id)
        engine = _get_debug_engine(ctx)
    else:
        engine = get_engine()
    # Seat policy stays engine-owned: when the GUI setting is on we ATTEMPT the
    # engine's lazy-start path once before failing; when off we only probe.
    engine.ensure_seat(allow_autostart=autostart_enabled())
    turn = engine.run_turn(str(message), history=history)
    _post_activity_event(str(message), turn, debug=debug)
    response: Dict[str, Any] = {
        "reply": turn.reply,
        "history": turn.messages,
        "tool_calls": turn.tool_calls,
        "rounds": turn.rounds,
        # Which loopback seat/model actually answered (S11). ``getattr`` with a
        # ``None`` default keeps a stubbed turn without the fields working.
        "model": getattr(turn, "model", None),
        "seat": getattr(turn, "seat_name", None),
    }
    if ctx is not None:
        response["debug"] = {
            "run_id": ctx.get("run_id"),
            "failed_phase": ctx.get("failed_phase"),
            "summary": ctx.get("summary"),
        }
    return response


def status() -> Dict[str, Any]:
    """Return ``{seat_serving, model, seat}`` for the panel's status pill.

    Seat + model are resolved DYNAMICALLY via
    ``lib.assistant.client.resolve_active_seat`` (S1), so the pill reflects
    WHICH loopback seat is currently live (the priority seat when up, else the
    fallback) and the served model id it reports — never a hardcoded model.
    ``model`` is ``None`` when the resolved seat is not live.

    A misconfigured (non-loopback) assistant — surfaced either at engine
    construction OR from a bad seat-registry entry the resolver walks — reports
    ``seat_serving=False`` / ``model=None`` / ``seat=None`` with an explanatory
    ``detail``. The status probe never raises for a config error (the chat
    endpoint surfaces it as a 500).
    """
    try:
        # Engine build first: preserves the construction-time loopback guard.
        get_engine()
        seat = resolve_active_seat()
    except AssistantProviderNotLocal as exc:
        return {"seat_serving": False, "model": None, "seat": None, "detail": str(exc)}
    return {
        "seat_serving": bool(seat.live),
        "model": seat.model if seat.live else None,
        "seat": seat.seat_name,
    }


# --------------------------------------------------------------------------- #
# Seat model (explicit human action)
# --------------------------------------------------------------------------- #
#
# The panel's "Seat model" button seats the assistant's OWN seat
# (``ED4ALL_ASSISTANT_SEAT``, default the small always-available nano seat) and
# is offered ONLY when nothing in the priority walk answers ``/v1/models``.
# Because it is an EXPLICIT human action it does NOT consult
# ``ED4ALL_ASSISTANT_AUTOSTART`` (that flag governs the implicit lazy-start on a
# chat turn) — but it takes exactly the same road: the lifecycle lib's
# coherent-start path for the RESOLVED seat, whose script comes from the
# ``ED4ALL_SEAT_LAUNCH_SPECS`` registry entry. No script path is ever named here.
#
# A cold start takes MINUTES, so the start runs on ONE background thread; the
# call returns ``state="starting"`` immediately and progress is observed through
# :func:`seat_status` (``starting`` while the thread runs, ``last_seat_error``
# once it has failed). SINGLE-FLIGHT: the module-level lock + state mean a
# second click observes the in-flight start instead of launching a second one.

_SEAT_LOCK = threading.Lock()

#: In-flight seat-start state. ``starting`` is the single-flight latch;
#: ``error`` is the LAST completed start's failure text (``None`` after a
#: success); ``thread`` is retained so a caller (and the tests) can observe the
#: worker. Only ever mutated under ``_SEAT_LOCK``.
_SEAT_START: Dict[str, Any] = {
    "starting": False,
    "seat": None,
    "error": None,
    "thread": None,
}


def reset_seat_start_state() -> None:
    """Clear the seat-start latch + last error (tests / config changes).

    Does NOT kill a running worker thread — it is a daemon that only writes the
    latch back under the lock; clearing state mid-flight is a test affordance,
    not a production path.
    """
    with _SEAT_LOCK:
        _SEAT_START.update({"starting": False, "seat": None, "error": None, "thread": None})


def _candidate_seat_names() -> List[str]:
    """Ordered, de-duplicated seat names the status probe reports on.

    The dynamic priority walk (``ED4ALL_ASSISTANT_SEAT_PRIORITY``, default the
    large shared pipeline seat then the small one) plus the assistant's OWN seat
    (``ED4ALL_ASSISTANT_SEAT``) when the priority list omits it. Registry-driven
    and model-agnostic — no seat name is hardcoded here.
    """
    names = list(resolve_assistant_seat_priority())
    own = resolve_assistant_seat()
    if own not in names:
        names.append(own)
    return names


def _seat_base_url(name: str, registry: Dict[str, str]) -> Optional[str]:
    """Resolve one seat's base URL, or ``None`` when it is not probeable.

    Registry entry (``ED4ALL_SEAT_BASE_URLS``) first; the assistant's own seat
    additionally falls back to ``ED4ALL_ASSISTANT_BASE_URL``. An unregistered
    seat is reported as *absent from the map*, never as a fabricated ``False``.
    """
    url = registry.get(name)
    if url:
        return url.rstrip("/")
    if name == resolve_assistant_seat():
        try:
            return resolve_assistant_base_url()
        except AssistantProviderNotLocal:
            return None
    return None


def seat_status() -> Dict[str, Any]:
    """Return ``{live_seat, seats, starting, last_seat_error}`` for the button.

    ``live_seat`` is the seat the DYNAMIC priority walk
    (``lib.assistant.client.resolve_active_seat``) says is answering right now,
    or ``None`` when nothing in the walk is live — which is exactly the
    condition under which the "Seat model" button may be offered. ``seats`` maps
    each probeable candidate seat name to its liveness (an unregistered seat is
    omitted rather than reported ``False``). ``starting`` is the single-flight
    latch; ``last_seat_error`` carries the previous start's failure text.

    A misconfiguration (non-loopback seat URL) surfaces as an extra ``detail``
    key with ``live_seat=None`` and an empty ``seats`` map — this probe never
    raises, mirroring :func:`status`.
    """
    with _SEAT_LOCK:
        payload: Dict[str, Any] = {
            "live_seat": None,
            "seats": {},
            "starting": bool(_SEAT_START["starting"]),
            "last_seat_error": _SEAT_START["error"],
        }
    try:
        # Engine build first (construction-time loopback guard), then the
        # authoritative priority walk — which also enforces the per-seat
        # loopback guard, so the map below never re-implements it.
        get_engine()
        active = resolve_active_seat(force=True)
    except AssistantProviderNotLocal as exc:
        payload["detail"] = str(exc)
        return payload

    live_name = active.seat_name if getattr(active, "live", False) else None
    payload["live_seat"] = live_name
    try:
        registry = parse_seat_registry()
    except Exception:  # noqa: BLE001 — fail-soft registry, never break the probe
        logger.exception("assistant: seat registry parse failed")
        registry = {}
    seats: Dict[str, bool] = {}
    for name in _candidate_seat_names():
        if live_name is not None and name == live_name:
            seats[name] = True
            continue
        base_url = _seat_base_url(name, registry)
        if not base_url:
            continue
        try:
            seats[name] = bool(seat_is_serving(base_url))
        except Exception:  # noqa: BLE001 — a probe failure means "not serving"
            logger.exception("assistant: seat probe failed for %r", name)
            seats[name] = False
    payload["seats"] = seats
    return payload


def _no_launch_spec_error(seat: str) -> str:
    """The verbatim operator-facing error when the seat has no launch spec."""
    return (
        f"Seat {seat!r} has no launch spec registered in {ENV_SEAT_LAUNCH_SPECS}, "
        f"so the assistant cannot seat it. Add a "
        f"'{seat}=<launch script path>' entry to {ENV_SEAT_LAUNCH_SPECS} (the "
        f"registry is the only source of the launch command — the GUI never "
        f"names a script), then try again."
    )


def _seat_start_worker(seat: str) -> None:
    """Background worker: coherent-start ``seat``, then clear the latch.

    NEVER raises (it is a thread body): any failure is recorded as
    ``last_seat_error`` for the next :func:`seat_status` poll.
    """
    error: Optional[str] = None
    try:
        result = autostart_seat()
        if not getattr(result, "ok", False):
            error = (
                f"Seat {seat!r} failed to come up coherent "
                f"(reason: {getattr(result, 'reason', 'unknown')}). "
                f"Check the seat's container logs."
            )
    except Exception as exc:  # noqa: BLE001 — thread body: record, never raise
        logger.exception("assistant: seat start failed for %r", seat)
        error = f"Seat {seat!r} start failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            # A freshly-started seat must be visible to the next resolution.
            reset_seat_cache()
        except Exception:  # noqa: BLE001 — cache reset is best-effort
            logger.exception("assistant: seat cache reset failed")
        with _SEAT_LOCK:
            _SEAT_START["starting"] = False
            _SEAT_START["error"] = error


def seat_nano() -> Dict[str, Any]:
    """Start the assistant's own seat — the panel's explicit "Seat model" action.

    Returns ``{ok, state, seat, error}`` where ``state`` is one of:

    * ``"starting"`` — a background coherent-start is now in flight (or already
      was: single-flight means a second call NEVER launches a second start);
    * ``"already_live"`` — REFUSED with no side effect because a seat is already
      answering (``seat`` names it);
    * ``"no_launch_spec"`` — REFUSED with no side effect because the resolved
      seat has no ``ED4ALL_SEAT_LAUNCH_SPECS`` entry (``error`` is that message
      verbatim);
    * ``"config_error"`` — REFUSED because the seat config is non-loopback.

    Deliberately independent of ``ED4ALL_ASSISTANT_AUTOSTART`` (this is an
    explicit human action, not the implicit lazy-start), but it takes the
    identical lifecycle road: :func:`lib.assistant.client.autostart_seat` →
    ``start_seat_coherent`` for the seat resolved from ``ED4ALL_ASSISTANT_SEAT``.
    """
    with _SEAT_LOCK:
        if _SEAT_START["starting"]:
            return {
                "ok": True,
                "state": "starting",
                "seat": _SEAT_START["seat"],
                "error": None,
            }

    # REFUSE loudly (and with zero side effects) when a seat already answers.
    probe = seat_status()
    if probe.get("detail"):
        return {
            "ok": False,
            "state": "config_error",
            "seat": None,
            "error": str(probe["detail"]),
        }
    live = probe.get("live_seat")
    if live:
        return {
            "ok": False,
            "state": "already_live",
            "seat": live,
            "error": (
                f"Refused: seat {live!r} is already serving — nothing to seat. "
                f"The 'Seat model' action only applies when no assistant seat "
                f"answers /v1/models."
            ),
        }

    seat = resolve_assistant_seat()
    try:
        spec = resolve_seat_launch_spec(seat)
    except Exception:  # noqa: BLE001 — fail-soft registry parse
        logger.exception("assistant: launch-spec lookup failed for %r", seat)
        spec = None
    if not spec:
        return {
            "ok": False,
            "state": "no_launch_spec",
            "seat": seat,
            "error": _no_launch_spec_error(seat),
        }

    with _SEAT_LOCK:
        if _SEAT_START["starting"]:  # raced with another click — single-flight
            return {
                "ok": True,
                "state": "starting",
                "seat": _SEAT_START["seat"],
                "error": None,
            }
        thread = threading.Thread(
            target=_seat_start_worker,
            args=(seat,),
            name="assistant-seat-start",
            daemon=True,
        )
        _SEAT_START.update({"starting": True, "seat": seat, "error": None, "thread": thread})
    try:
        thread.start()
    except Exception as exc:  # noqa: BLE001 — thread could not be spawned
        logger.exception("assistant: seat-start thread spawn failed")
        with _SEAT_LOCK:
            _SEAT_START["starting"] = False
            _SEAT_START["error"] = f"Seat start could not be scheduled: {exc}"
        return {
            "ok": False,
            "state": "start_failed",
            "seat": seat,
            "error": f"Seat start could not be scheduled: {exc}",
        }
    return {"ok": True, "state": "starting", "seat": seat, "error": None}


def _post_activity_event(message: str, turn: Any, *, debug: bool = False) -> None:
    """Append one ``assistant``-sourced exchange event (best-effort)."""
    try:
        tools = sorted({t.get("tool", "") for t in (turn.tool_calls or []) if t.get("tool")})
        question = message.strip()
        if len(question) > _SUMMARY_QUESTION_CHARS:
            question = question[: _SUMMARY_QUESTION_CHARS - 1] + "…"
        summary = f"Assistant answered{' (debug)' if debug else ''}: {question!r}"
        summary += f" (tools: {', '.join(tools)})" if tools else " (no tools)"
        shared_state.append_event(
            "assistant",
            "assistant_exchange",
            {
                "summary": summary,
                "question": question,
                "tools": tools,
                "rounds": int(getattr(turn, "rounds", 0) or 0),
                "mode": "debug" if debug else "operator",
            },
        )
    except Exception:  # noqa: BLE001 — telemetry must never eat the reply
        logger.exception("assistant: activity-event append failed")


__all__ = [
    "autostart_enabled",
    "chat",
    "debug_context_probe",
    "get_engine",
    "is_valid_run_id",
    "reset_engine",
    "reset_seat_start_state",
    "seat_nano",
    "seat_status",
    "status",
]
