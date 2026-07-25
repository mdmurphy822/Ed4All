"""vLLM seat-status service — the always-on "which seat is up" monitor.

WHY THIS EXISTS (real incident, 2026-07-22): two pipeline runs ran
concurrently; run B's seat schedule stopped a seat while run A was mid-dispatch
to it, so every call to that seat was refused → poison pill → a 2.5-hour build
FAILED. Nothing in the GUI showed that the seat the current phase DEPENDED ON
had gone away. The single most valuable thing this service reports is therefore
**expected-vs-actual**: "this phase needs <seat> — <seat> is DOWN".

Two surfaces, one shared probe cache:

* :func:`seat_overview` with no ``run_id`` — the GLOBAL monitor (every seat in
  the registry, its state, and how long it has been in that state). Backs
  ``GET /api/seats`` and the Dashboard's always-visible seat section, which
  renders even when no build is running.
* :func:`seat_overview` with a ``run_id`` — the same seat list PLUS the run's
  ``current_phase``, that phase's DECLARED ``expected_seats`` and the
  ``mismatch`` list. Backs ``GET /api/seats/run/{run_id}`` and the build
  page's seat strip.

Design contracts:

* **Registry-driven / model-agnostic.** The seat list is exactly
  ``lib.vllm_container_lifecycle.parse_seat_registry()``
  (``ED4ALL_SEAT_BASE_URLS``) and the container map is
  ``parse_container_registry()`` (``ED4ALL_VLLM_CONTAINERS``). NO seat name is
  ever hardcoded — a new seat is a registry entry, never code here.
* **loading ≠ down.** A large seat's cold start takes ~9 MINUTES: the container
  is up but ``/v1/models`` does not answer yet. That is ``loading`` (progress,
  not an alarm). ``down`` is asserted ONLY when the seat's registered container
  is verifiably NOT running. When we cannot tell (docker absent / no perms / no
  container registered for that base URL) the state is ``unknown`` — never a
  guessed ``down``.
* **Expected seats are DECLARED, not inferred.** The current phase's ``seats:``
  annotation in ``config/workflows.yaml`` has three distinct cases and all three
  are honoured: named seats (an expectation), an explicit ``[]`` (declared
  seat-free — no expectation), and ABSENT (no opinion — no expectation, so no
  mismatch can ever be reported). A mismatch is raised only for a NAMED seat
  that is neither ``live`` nor ``loading``.
* **Bounded + cached + thread-safe + never raises.** Every probe has a short
  timeout, the whole payload is TTL-cached (~10 s) behind a module lock so N
  pollers cost ONE probe round, and any failure degrades to ``unknown`` /an
  empty seat list rather than propagating.
* **"since" is honest.** ``since`` / ``since_ms`` report when THIS PROCESS first
  observed the seat in its current state — not a claim about the seat's real
  history. The first observation of a seat therefore reports ``None``: there is
  no prior observation to compare against.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

#: Freshness window (seconds) for the cached overview. The UI polls ~15 s from
#: every open page; this keeps the real probe cost flat regardless of how many
#: viewers (dashboard + build page + a second tab) are watching.
CACHE_TTL_SECONDS = 10.0

#: Per-seat ``/v1/models`` HTTP timeout (seconds). Short by design: a seat that
#: cannot answer in this window is not usefully "serving" for a dispatch.
PROBE_TIMEOUT_SECONDS = 1.5

#: Bounded timeout (seconds) for the single ``docker ps`` read per refresh.
DOCKER_PS_TIMEOUT_SECONDS = 8.0

#: The four states a seat can be reported in (see the module docstring).
STATE_LIVE = "live"
STATE_LOADING = "loading"
STATE_DOWN = "down"
STATE_UNKNOWN = "unknown"

#: States that satisfy a phase's declared seat expectation. ``loading`` counts:
#: a ~9-minute cold start is progress, not an alarm.
_SATISFYING_STATES = frozenset({STATE_LIVE, STATE_LOADING})


# --------------------------------------------------------------------- cache
_lock = threading.RLock()
_cache: Optional[Dict[str, Any]] = None  # {"monotonic", "payload"} (global)
_run_cache: Dict[str, Dict[str, Any]] = {}  # run_id -> {"monotonic", "payload"}

#: Per-seat state-transition markers: ``{seat: {"state", "monotonic", "iso"}}``.
#: Populated on the FIRST observation of each seat and reset whenever its state
#: changes; the age reported to the UI is measured from here. Process-scoped by
#: construction (see the docstring's honesty note).
_since: Dict[str, Dict[str, Any]] = {}


def reset_cache() -> None:
    """Drop the cached payloads AND the transition markers (tests / config)."""
    global _cache
    with _lock:
        _cache = None
        _run_cache.clear()
        _since.clear()
        _WF_CACHE["mtime"] = None
        _WF_CACHE["workflows"] = {}


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------- probes
def _probe_seat(base_url: str) -> Optional[str]:
    """``GET {base_url}/v1/models`` → served model id; ``None`` when not serving.

    Returns ``""`` when the seat answers 2xx but no model id is parseable (it IS
    serving — we just cannot name the model). Bounded by
    :data:`PROBE_TIMEOUT_SECONDS`; never raises.

    The seat registry's convention is a base URL WITHOUT the ``/v1`` suffix
    (``spark-x=http://host:8001``), exactly as
    ``lib.vllm_container_lifecycle._probe_ready`` expects. A registry entry that
    nonetheless carries ``/v1`` (the OpenAI-client convention that
    ``ED4ALL_ASSISTANT_BASE_URL`` uses) is tolerated here rather than reported
    as a false ``down``: the suffix is detected, never doubled.
    """
    base = str(base_url).rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SECONDS) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if not (200 <= int(code) < 300):
                return None
            body = json.loads(resp.read(65536).decode("utf-8"))
            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("id") or "") or ""
            return ""
    except Exception:  # noqa: BLE001 — down / refused / timeout / bad URL
        return None


def _docker_running_names() -> Optional[Set[str]]:
    """Names of RUNNING containers (``docker ps``), or ``None`` when unknowable.

    ``None`` means docker is unavailable (CLI absent / no perms / timeout /
    error) — the caller must report ``unknown``, never ``down``. On a
    permission-shaped failure the query is retried once through ``sg docker -c``
    (the Spark docker-group wrapping), mirroring
    ``lib.diagnostics.seat_schedule._docker_container_names`` and
    ``lib.vllm_container_lifecycle._run_docker``. Bounded; NEVER raises; this is
    a READ ONLY — nothing here starts or stops a container.
    """
    cmd = ["docker", "ps", "--format", "{{.Names}}"]

    def _run(argv: List[str]) -> Optional[Tuple[int, str, str]]:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=DOCKER_PS_TIMEOUT_SECONDS
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        except Exception as exc:  # noqa: BLE001 — docker absent / timeout / etc.
            logger.debug("seat_service: %r failed: %s", " ".join(argv), exc)
            return None

    res = _run(cmd)
    if res is not None and res[0] == 0:
        return {ln.strip() for ln in res[1].splitlines() if ln.strip()}
    if res is not None:
        stderr = (res[2] or "").lower()
        if "permission denied" in stderr or "dial unix" in stderr:
            wrapped = _run(["sg", "docker", "-c", " ".join(cmd)])
            if wrapped is not None and wrapped[0] == 0:
                return {ln.strip() for ln in wrapped[1].splitlines() if ln.strip()}
    return None


# ------------------------------------------------------- workflow seat plan
_WF_CACHE: Dict[str, Any] = {"mtime": None, "workflows": {}}


def _workflows_config() -> Dict[str, Any]:
    """Parsed ``config/workflows.yaml`` ``workflows`` map, cached by mtime.

    Deliberately re-read here (rather than borrowing the progress service's
    private cache) because this service needs the RAW phase dict: only the raw
    mapping distinguishes an ABSENT ``seats:`` key from an explicit ``seats: []``.
    Never raises — an unreadable config yields ``{}`` (→ "no opinion").
    """
    try:
        from lib.paths import PROJECT_ROOT  # noqa: PLC0415

        path = Path(PROJECT_ROOT) / "config" / "workflows.yaml"
        mtime = path.stat().st_mtime
    except Exception:  # noqa: BLE001 — missing config → no expectations
        return {}
    with _lock:
        if _WF_CACHE["mtime"] == mtime:
            return _WF_CACHE["workflows"]
    try:
        import yaml  # noqa: PLC0415

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        workflows = cfg.get("workflows") or {}
    except Exception:  # noqa: BLE001 — parse failure → keep the last good copy
        logger.warning("seat_service: workflows.yaml parse failed", exc_info=True)
        with _lock:
            return _WF_CACHE["workflows"]
    if not isinstance(workflows, dict):
        workflows = {}
    with _lock:
        _WF_CACHE["mtime"] = mtime
        _WF_CACHE["workflows"] = workflows
        return workflows


def phase_expected_seats(workflow: str, phase: str) -> Optional[List[str]]:
    """The phase's DECLARED ``seats:`` annotation — the three-case contract.

    * ``["<seat-name>"]`` → the phase names seats: an expectation to enforce.
    * ``[]``              → explicitly declared seat-free: no expectation.
    * ``None``            → the key is ABSENT (no opinion), the workflow/phase
      is unknown, or the config is unreadable: NO expectation, so no mismatch
      can ever be reported for it.
    """
    if not workflow or not phase:
        return None
    wf = _workflows_config().get(str(workflow))
    if not isinstance(wf, dict):
        return None
    for ph in wf.get("phases") or []:
        if not isinstance(ph, dict) or str(ph.get("name") or "") != str(phase):
            continue
        if "seats" not in ph:
            return None  # absent → no opinion
        raw = ph.get("seats")
        if not isinstance(raw, list):
            return None  # `seats:` null / malformed → no opinion (never a guess)
        return [str(s).strip() for s in raw if isinstance(s, str) and str(s).strip()]
    return None


# ----------------------------------------------------------- state assembly
def _classify(
    live: bool, container: Optional[str], running: Optional[Set[str]]
) -> str:
    """Map (liveness, container, docker view) onto one of the four states."""
    if live:
        return STATE_LIVE
    if running is None:
        return STATE_UNKNOWN  # docker unavailable — cannot distinguish
    if not container:
        return STATE_UNKNOWN  # no container registered for this base URL
    if container in running:
        return STATE_LOADING  # container up, endpoint not answering YET
    return STATE_DOWN


def _mark_since(name: str, state: str, now: float) -> Tuple[Optional[str], Optional[int]]:
    """Record ``state`` for ``name`` and return ``(since_iso, since_ms)``.

    First observation → ``(None, None)`` (nothing to compare against). A CHANGED
    state resets the marker (age restarts at 0). An unchanged state preserves
    the original marker, so a flapping seat is visible as a resetting age.
    """
    prev = _since.get(name)
    if prev is None:
        _since[name] = {"state": state, "monotonic": now, "iso": _now_iso()}
        return None, None
    if prev.get("state") != state:
        marker = {"state": state, "monotonic": now, "iso": _now_iso()}
        _since[name] = marker
        return str(marker["iso"]), 0
    elapsed = max(0, int((now - float(prev.get("monotonic") or now)) * 1000))
    return str(prev.get("iso") or "") or None, elapsed


def _compute_seats() -> Dict[str, Any]:
    """Probe every registered seat once; shape the global payload. Never raises."""
    try:
        from lib.vllm_container_lifecycle import (  # noqa: PLC0415
            parse_container_registry,
            parse_seat_registry,
        )

        registry = parse_seat_registry() or {}
    except Exception:  # noqa: BLE001 — a broken registry is not an outage
        logger.warning("seat_service: seat registry parse failed", exc_info=True)
        return {
            "generated_at": _now_iso(),
            "seats": [],
            "registry_configured": False,
            "docker_available": None,
            "detail": "seat registry could not be read",
        }
    try:
        containers = parse_container_registry() or {}
    except Exception:  # noqa: BLE001 — container map is an enrichment, not a gate
        logger.warning("seat_service: container registry parse failed", exc_info=True)
        containers = {}

    running: Optional[Set[str]] = None
    if registry:
        # ONE docker read per refresh, shared by every seat below.
        running = _docker_running_names()

    # Probe OUTSIDE the lock: N bounded HTTP round-trips must never block a
    # concurrent poller (the lock only guards the marker map + caches).
    probed: List[Tuple[str, str, Optional[str], str]] = []
    for name, base_url in registry.items():
        url = str(base_url).rstrip("/")
        container = containers.get(url)
        try:
            model = _probe_seat(url)
        except Exception:  # noqa: BLE001 — a probe failure is "not serving"
            logger.debug("seat_service: probe raised for %r", name, exc_info=True)
            model = None
        probed.append((str(name), url, model, _classify(model is not None, container, running)))

    now = time.monotonic()
    seats: List[Dict[str, Any]] = []
    with _lock:
        for name, url, model, state in probed:
            since_iso, since_ms = _mark_since(name, state, now)
            seats.append(
                {
                    "name": name,
                    "base_url": url,
                    "live": state == STATE_LIVE,
                    "state": state,
                    "container": containers.get(url),
                    "model": model or None,
                    "since": since_iso,
                    "since_ms": since_ms,
                }
            )
        # Keep the marker map bounded to the CURRENT registry (a seat removed
        # from the registry must not keep a stale age forever).
        for stale in [k for k in _since if k not in registry]:
            _since.pop(stale, None)

    return {
        "generated_at": _now_iso(),
        "seats": seats,
        "registry_configured": bool(registry),
        "docker_available": None if running is None else True,
    }


def _global_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    """The TTL-cached global seat payload (one shared cache for every caller)."""
    global _cache
    with _lock:
        now = time.monotonic()
        cached = _cache
        if not force_refresh and cached and (now - cached["monotonic"]) < CACHE_TTL_SECONDS:
            return cached["payload"]
    try:
        payload = _compute_seats()
    except Exception as exc:  # noqa: BLE001 — belt-and-braces: never raise
        logger.exception("seat_service: seat overview compute failed")
        payload = {
            "generated_at": _now_iso(),
            "seats": [],
            "registry_configured": False,
            "docker_available": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    with _lock:
        _cache = {"monotonic": time.monotonic(), "payload": payload}
    return payload


# ------------------------------------------------------------ run-scoped view
def _run_context(run_id: str) -> Optional[Dict[str, Any]]:
    """Resolve ``run_id`` → ``{current_phase, expected_seats, ...}`` (None = unknown).

    Reuses the progress service's run resolution (GUI run id / ``WF-*``
    workflow id) so a CLI-launched run is observable here too. A terminal run
    has no current phase and therefore no expectation.
    """
    try:
        from gui.services import progress_service  # noqa: PLC0415

        payload = progress_service.run_progress(str(run_id))
    except Exception:  # noqa: BLE001 — unresolvable → the router 404s honestly
        logger.debug("seat_service: run resolution failed for %r", run_id, exc_info=True)
        return None
    if payload is None:
        return None
    workflow = str(payload.get("workflow") or "")
    current_phase = payload.get("current_phase")
    current_phase = str(current_phase) if isinstance(current_phase, str) else None
    expected = phase_expected_seats(workflow, current_phase or "")
    return {
        "run_id": str(run_id),
        "workflow": workflow or None,
        "status": payload.get("status"),
        "effective_status": payload.get("effective_status"),
        "current_phase": current_phase,
        # None → the phase declares nothing (absent annotation / unknown phase).
        "expected_seats": expected,
    }


def _run_payload(run_id: str, *, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """TTL-cached run context (never caches a miss, so a new run appears fast)."""
    key = str(run_id or "").strip()
    if not key:
        return None
    with _lock:
        now = time.monotonic()
        cached = _run_cache.get(key)
        if not force_refresh and cached and (now - cached["monotonic"]) < CACHE_TTL_SECONDS:
            return cached["payload"]
    ctx = _run_context(key)
    if ctx is None:
        return None
    with _lock:
        _run_cache[key] = {"monotonic": time.monotonic(), "payload": ctx}
    return ctx


def seat_overview(
    run_id: Optional[str] = None, *, force_refresh: bool = False
) -> Optional[Dict[str, Any]]:
    """Return the seat overview; ``None`` ONLY when a given ``run_id`` is unknown.

    Global shape (``run_id=None``)::

        {generated_at, registry_configured, docker_available,
         seats: [{name, base_url, live, state, container, model, since, since_ms}]}

    Run-scoped shape adds::

        {run_id, workflow, status, effective_status, current_phase,
         expected_seats: [...]|null, mismatch: [{seat, state, base_url}]}

    ``expected_seats`` is ``null`` when the current phase declares no opinion
    (absent ``seats:``) and ``[]`` when it is declared seat-free; ``mismatch``
    is non-empty ONLY for a NAMED seat that is neither live nor loading. Each
    seat additionally carries ``expected: bool`` on the run-scoped shape.

    Never raises. ``state`` degrades to ``unknown`` rather than fabricating a
    verdict, and a broken registry yields an empty seat list with a ``detail``.
    """
    base = _global_payload(force_refresh=force_refresh)
    payload: Dict[str, Any] = dict(base)
    payload["seats"] = [dict(s) for s in base.get("seats", [])]
    if run_id is None:
        return payload

    ctx = _run_payload(str(run_id), force_refresh=force_refresh)
    if ctx is None:
        return None
    payload.update(ctx)

    expected = ctx.get("expected_seats")
    by_name = {s["name"]: s for s in payload["seats"]}
    if isinstance(expected, list):
        for seat in payload["seats"]:
            seat["expected"] = seat["name"] in expected
        mismatch: List[Dict[str, Any]] = []
        for name in expected:
            seat = by_name.get(name)
            if seat is None:
                # The phase names a seat that is NOT in the registry at all —
                # a real (and loud) configuration mismatch, not a silent pass.
                mismatch.append({"seat": name, "state": "unregistered", "base_url": None})
                continue
            if seat["state"] not in _SATISFYING_STATES:
                mismatch.append(
                    {"seat": name, "state": seat["state"], "base_url": seat["base_url"]}
                )
        payload["mismatch"] = mismatch
    else:
        for seat in payload["seats"]:
            seat["expected"] = False
        payload["mismatch"] = []
    return payload


__all__ = [
    "CACHE_TTL_SECONDS",
    "DOCKER_PS_TIMEOUT_SECONDS",
    "PROBE_TIMEOUT_SECONDS",
    "STATE_DOWN",
    "STATE_LIVE",
    "STATE_LOADING",
    "STATE_UNKNOWN",
    "phase_expected_seats",
    "reset_cache",
    "seat_overview",
]
