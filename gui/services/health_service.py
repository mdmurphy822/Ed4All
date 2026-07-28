"""In-process ``ed4all doctor`` health service for the control-plane GUI.

Runs the pluggable diagnostics registry (:mod:`lib.diagnostics`) IN-PROCESS —
no ``ed4all doctor`` subprocess — and shapes the results into a JSON-able
payload the Dashboard's "System health" card renders. Two surfaces:

* :func:`get_health` — the GLOBAL preflight doctor (the default check groups
  plus any env-gated opt-in group whose prerequisite env is present). Backs
  ``GET /api/health/doctor``.
* :func:`run_health` — a RUN-SCOPED post-mortem (the ``postmortem`` group over
  one run's persisted ``state/runs/<run_id>/`` checkpoints + VRAM trajectory,
  the in-process equivalent of ``ed4all doctor --run-id <id>``), folded with the
  run's honest ``effective_status`` and an ``llm_usage.jsonl`` presence/row
  probe. Backs ``GET /api/health/doctor/run/{run_id}``.

Design contracts (mirroring the diagnostics foundation + the assistant
service):

* **Group-agnostic.** The registry is bootstrapped exactly as the CLI does
  (reusing ``cli.commands.doctor._bootstrap_checks`` when importable); the set
  of groups to RUN is derived from the *live registry* minus a small, stable
  policy exclusion (``provider`` — a run-preflight/network concern; ``postmortem``
  — the run-scoped mode) and gated by env presence for opt-in groups (``seat``).
  A NEW group the concurrent doctor work registers auto-appears with no edit
  here.
* **Never raises.** ``run_checks`` already turns a raising check into a single
  WARN; every public entry point additionally wraps its compute so a failure
  degrades to a valid (error-shaped) payload rather than propagating.
* **Thread-offload + short-TTL cache.** The routers call these from plain ``def``
  handlers (FastAPI runs those in the threadpool, so a heavy torch VRAM probe
  never blocks the event loop). Results are cached ~30 s (global + per run id)
  with an explicit force-refresh, so a Dashboard mount does not re-probe on
  every poll.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from lib.diagnostics import (
    CheckContext,
    registered_checks,
    resolve_exit_code,
    resolve_verdict,
    results_to_json,
    run_checks,
)

logger = logging.getLogger(__name__)

#: Cache freshness window (seconds) for both the global + per-run payloads.
CACHE_TTL_SECONDS = 30.0

#: Groups NEVER auto-run by the global dashboard doctor. ``provider`` is a
#: run-preflight seat/network concern (the CLI excludes it from its bare
#: default too); ``postmortem`` needs a run id (the run-scoped endpoint only).
#: This is a small, stable POLICY set — NOT the group list — so any other
#: registered group (present or future) auto-runs.
_EXCLUDED_GROUPS = frozenset({"provider", "postmortem"})

#: The run-scoped post-mortem group (``ed4all doctor --run-id`` equivalent).
_POSTMORTEM_GROUP = "postmortem"


def _seat_env_present() -> bool:
    """True when any vLLM seat-registry / seat-schedule env is configured.

    Gates the opt-in ``seat`` group so it only runs when the operator has
    actually declared seats (matches "include the seat group when a seat
    registry env is set"). Presence-only — never dereferences a real seat.
    """
    return any(
        os.environ.get(key, "").strip()
        for key in (
            "ED4ALL_SEAT_BASE_URLS",
            "ED4ALL_VLLM_CONTAINERS",
            "ED4ALL_SEAT_SCHEDULE",
            "ED4ALL_VLLM_CONTAINER_LIFECYCLE",
        )
    )


#: Opt-in groups gated on an env prerequisite: ``{group_name: predicate}``. A
#: registered group named here runs ONLY when its predicate is truthy; a group
#: absent from both this map and ``_EXCLUDED_GROUPS`` always runs (the
#: group-agnostic default).
_ENV_GATED_GROUPS: Dict[str, Callable[[], bool]] = {"seat": _seat_env_present}


# --------------------------------------------------------------------- cache
_lock = threading.Lock()
_global_cache: Optional[Dict[str, Any]] = None  # {"monotonic", "payload"}
_run_cache: Dict[str, Dict[str, Any]] = {}  # run_id -> {"monotonic", "payload"}


def reset_cache() -> None:
    """Drop the cached global + per-run payloads (tests / config changes)."""
    global _global_cache
    with _lock:
        _global_cache = None
        _run_cache.clear()


# --------------------------------------------------------------- bootstrap
def _bootstrap() -> None:
    """Register the built-in check groups exactly as the CLI does (idempotent).

    Reuses ``cli.commands.doctor._bootstrap_checks`` (read-only import) so a
    group the doctor work adds to that bootstrap — e.g. a landing ``seat``
    group — is picked up here for free. If that import is transiently
    unavailable (the module is mid-edit) we fall back to registering the
    known groups directly. Never raises.
    """
    try:
        from cli.commands.doctor import _bootstrap_checks  # noqa: PLC0415

        _bootstrap_checks()
        return
    except Exception:  # noqa: BLE001 — degrade to direct registration
        logger.warning(
            "health: doctor bootstrap import unavailable; registering groups directly",
            exc_info=True,
        )
    _fallback_bootstrap()


def _fallback_bootstrap() -> None:
    """Register the known groups directly (belt-and-suspenders for _bootstrap)."""
    from lib.diagnostics import clear_registry  # noqa: PLC0415

    clear_registry()
    # Each register_* is imported + called defensively so a single broken
    # module never blocks the rest of the doctor.
    registrations = (
        ("lib.diagnostics.vram", "register_gpu_checks"),
        ("lib.diagnostics.gpu_profile", "register_gpu_profile_checks"),
        ("lib.diagnostics.serving_window", "register_serving_window_checks"),
        ("lib.diagnostics.environment", "register_environment_checks"),
        ("lib.diagnostics.provider", "register_provider_checks"),
        ("lib.diagnostics.postmortem", "register_postmortem_checks"),
    )
    import importlib  # noqa: PLC0415

    for module_path, fn_name in registrations:
        try:
            module = importlib.import_module(module_path)
            getattr(module, fn_name)()
        except Exception:  # noqa: BLE001 — one bad module must not sink the rest
            logger.warning("health: fallback register %s failed", fn_name, exc_info=True)


def _select_groups() -> List[str]:
    """Group-agnostically pick which registered groups the global doctor runs.

    Every registered group runs EXCEPT the stable policy exclusions
    (``_EXCLUDED_GROUPS``); an env-gated group (``_ENV_GATED_GROUPS``) runs only
    when its predicate is truthy. Order + de-dup follow registration order.
    """
    selected: List[str] = []
    seen: set = set()
    for group, _fn in registered_checks():
        if group in seen:
            continue
        seen.add(group)
        if group in _EXCLUDED_GROUPS:
            continue
        gate = _ENV_GATED_GROUPS.get(group)
        if gate is not None and not _safe_gate(gate):
            continue
        selected.append(group)
    return selected


def _local_synthesis_active() -> bool:
    """Whether the GUI's current provider configuration uses local synthesis.

    The endpoint registry contains historical/available backends; presence in
    that catalog does not make a backend active. Resolve activation from the
    live process env plus the persisted GUI settings operators actually edit.
    This prevents an unused legacy Ollama default from degrading health while
    retaining the OpenAI-compatible probe whenever a local endpoint is
    explicitly configured.
    """
    local_keys = ("LOCAL_SYNTHESIS_BASE_URL", "LOCAL_SYNTHESIS_MODEL")
    provider_keys = (
        "LLM_PROVIDER",
        "COURSEFORGE_OUTLINE_PROVIDER",
        "COURSEFORGE_REWRITE_PROVIDER",
        "TRAINFORGE_SYNTHESIS_PROVIDER",
        "TRAINFORGE_ASSESSMENT_PROVIDER",
    )
    if any(os.environ.get(k, "").strip() for k in local_keys):
        return True
    if any(os.environ.get(k, "").strip().lower() == "local" for k in provider_keys):
        return True
    try:
        from gui import settings_store  # noqa: PLC0415

        doc = settings_store.load_settings()
    except Exception:  # noqa: BLE001 — absent/corrupt settings means inactive
        return False
    env = doc.get("env") if isinstance(doc, dict) else {}
    if isinstance(env, dict):
        if any(str(env.get(k) or "").strip() for k in local_keys):
            return True
        if any(
            str(env.get(k) or "").strip().lower() == "local"
            for k in provider_keys
        ):
            return True
    routing = doc.get("model_routing") if isinstance(doc, dict) else {}
    if isinstance(routing, dict):
        for row in routing.values():
            if isinstance(row, dict) and str(row.get("provider") or "").strip().lower() == "local":
                return True
    return False


def _safe_gate(gate: Callable[[], bool]) -> bool:
    """Evaluate an env-gate predicate, treating any failure as 'absent'."""
    try:
        return bool(gate())
    except Exception:  # noqa: BLE001 — a broken gate never enables its group
        return False


# ------------------------------------------------------------------ shaping
_SEVERITIES = ("ok", "info", "warn", "fail")
_EXIT_STATE = {0: "healthy", 1: "degraded", 2: "critical"}


def _now_iso() -> str:
    """UTC ISO-8601 timestamp for ``generated_at`` (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _shape_payload(results: List[Any], *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Shape a flat ``CheckResult`` list into the wire payload.

    ``{generated_at, exit_state, exit_code, verdict, groups:[{group, checks:[...]}],
    summary:{total, ok, info, warn, fail, verdict}}`` — ``checks`` are the plain
    ``results_to_json`` CheckResult dicts (name/group/severity/summary/detail/
    remediation/data). ``extra`` merges run-scoped fields (run_id / usage / ...).
    """
    exit_code = resolve_exit_code(results)
    verdict = resolve_verdict(results)
    dicts = results_to_json(results)

    group_order: List[str] = []
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    counts = {sev: 0 for sev in _SEVERITIES}
    for d in dicts:
        group = d.get("group") or "?"
        if group not in by_group:
            by_group[group] = []
            group_order.append(group)
        by_group[group].append(d)
        sev = d.get("severity")
        if sev in counts:
            counts[sev] += 1

    payload: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "exit_state": _EXIT_STATE.get(exit_code, "healthy"),
        "exit_code": exit_code,
        "verdict": verdict,
        "groups": [{"group": g, "checks": by_group[g]} for g in group_order],
        "summary": {
            "total": len(dicts),
            "ok": counts["ok"],
            "info": counts["info"],
            "warn": counts["warn"],
            "fail": counts["fail"],
            "verdict": verdict,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _error_payload(exc: Exception, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A valid (degraded) payload for a compute that failed unexpectedly.

    The diagnostics foundation never raises, so this is belt-and-suspenders: a
    single synthetic WARN check under a ``health`` group carries the error so
    the UI shows an honest degraded banner rather than a blank card.
    """
    check = {
        "name": "health_service_error",
        "group": "health",
        "severity": "warn",
        "summary": f"health check compute failed: {exc}",
        "detail": f"{type(exc).__name__}: {exc}",
        "remediation": "this is a health-service bug — the doctor should never raise",
        "data": {"error": str(exc), "error_type": type(exc).__name__},
    }
    payload: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "exit_state": "degraded",
        "exit_code": 1,
        "verdict": f"DEGRADED: {check['name']}",
        "groups": [{"group": "health", "checks": [check]}],
        "summary": {"total": 1, "ok": 0, "info": 0, "warn": 1, "fail": 0, "verdict": check["summary"]},
    }
    if extra:
        payload.update(extra)
    return payload


# ------------------------------------------------------------- global doctor
def _compute_global() -> Dict[str, Any]:
    """Bootstrap + run the selected global groups; never raises."""
    try:
        _bootstrap()
        groups = _select_groups()
        results = run_checks(
            CheckContext(
                base_url=None,
                local_synthesis_active=_local_synthesis_active(),
            ),
            groups=groups,
        )
        return _shape_payload(results)
    except Exception as exc:  # noqa: BLE001 — degrade to an error payload
        logger.exception("health: global doctor compute failed")
        return _error_payload(exc)


def get_health(*, force_refresh: bool = False) -> Dict[str, Any]:
    """Return the cached global doctor payload (re-running past its TTL).

    ``force_refresh`` re-runs the checks immediately (the ``/refresh`` route).
    Serialized under the module lock so a stampede of Dashboard polls does not
    launch parallel torch probes; the router calls this from a threadpool
    handler so the event loop is never blocked.
    """
    global _global_cache
    with _lock:
        now = time.monotonic()
        cached = _global_cache
        if not force_refresh and cached and (now - cached["monotonic"]) < CACHE_TTL_SECONDS:
            return cached["payload"]
        payload = _compute_global()
        _global_cache = {"monotonic": time.monotonic(), "payload": payload}
        return payload


# ---------------------------------------------------- run-scoped post-mortem
def _run_dir_exists(run_id: str) -> bool:
    """True when ``state/runs/<run_id>/`` exists (a bare orchestrator run id)."""
    try:
        from lib.paths import get_state_runs_dir  # noqa: PLC0415

        return (get_state_runs_dir() / run_id).is_dir()
    except Exception:  # noqa: BLE001 — unresolvable path → treat as absent
        return False


def _read_workflow_state(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Read ``state/workflows/<workflow_id>.json`` (bounded, never raises).

    Mirrors ``progress_service._read_workflow_state`` — a CLI-launched run is
    only observable through its ``WF-*.json`` workflow-state file.
    """
    if not workflow_id:
        return None
    try:
        from gui import shared_state  # noqa: PLC0415

        path = shared_state._state_root() / "workflows" / f"{workflow_id}.json"
        if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
            return None
        import json  # noqa: PLC0415

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — mid-write / corrupt → absent
        return None
    return payload if isinstance(payload, dict) else None


def _as_dict(params: Any) -> Dict[str, Any]:
    """Coerce a (possibly JSON-string) params blob to a dict (else ``{}``)."""
    if isinstance(params, str):
        try:
            import json  # noqa: PLC0415

            params = json.loads(params)
        except ValueError:
            return {}
    return params if isinstance(params, dict) else {}


def _orch_run_id_from_params(params: Any) -> Optional[str]:
    """Best-effort orchestrator run id (``params.run_id``) — the state/runs key."""
    from gui.services.run_service import _record_orch_run_id  # noqa: PLC0415

    return _record_orch_run_id(params)


def _resolve_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an input id to its orchestrator run id + effective status.

    Accepts a GUI run id, an orchestrator ``WF-*`` workflow id, OR a bare
    orchestrator run id (a ``state/runs/<id>/`` dir). Resolution precedent is
    the run + progress services: a GUI record's ``params.run_id`` / a
    ``WF-*.json`` workflow-state's ``params.run_id`` is the ``state/runs/<...>``
    key the post-mortem reads. Returns ``None`` when NOTHING resolves (→ the
    router answers a clean 404); a resolvable record with no post-mortem dir
    still returns (the post-mortem then reports ``run_not_found`` honestly).
    """
    from gui import shared_state  # noqa: PLC0415
    from gui.services import liveness  # noqa: PLC0415

    rec: Optional[Dict[str, Any]] = None
    try:
        rec = shared_state.read_run(run_id)
    except Exception:  # noqa: BLE001 — a broken registry read → treat as absent
        rec = None

    orch: Optional[str] = None
    status: Any = None
    is_cli = False
    tokens: List[str] = []
    resolved = False

    if rec is not None:
        resolved = True
        params = rec.get("params")
        orch = _orch_run_id_from_params(params)
        status = rec.get("status")
        is_cli = False  # GUI runs are driven in-process (no ed4all run process)
    else:
        state = _read_workflow_state(run_id)
        if state is not None:
            resolved = True
            params = state.get("params")
            orch = _orch_run_id_from_params(params)
            status = state.get("status")
            is_cli = True
            # The state file is keyed by workflow id; its ``id`` field is the
            # canonical ``WF-<id>`` (fall back to the resolving id). Threading it
            # lets a ``--resume WF-<id>`` process self-attribute (see liveness.py).
            wf_id = str(state.get("id") or run_id or "").strip() or None
            tokens = liveness.attribution_tokens_from_params(
                _as_dict(params), orch, wf_id=wf_id
            )

    # Bare orchestrator run id (or a record whose params omit run_id but whose
    # run dir is named by the input id).
    if not orch and _run_dir_exists(run_id):
        orch = run_id
        resolved = True

    if not resolved:
        return None

    effective: Optional[str] = None
    if status is not None and str(status).strip():
        try:
            effective = liveness.effective_status(
                status,
                is_cli=is_cli,
                orch_run_id=orch,
                attribution_tokens=tokens,
            )
        except Exception:  # noqa: BLE001 — derivation is best-effort
            effective = str(status).strip().lower()

    return {"orch_run_id": orch or "", "effective_status": effective}


def _usage_probe(orch_run_id: str) -> Dict[str, Any]:
    """Read-only presence + row count of ``state/runs/<id>/llm_usage.jsonl``.

    ``{present, rows}`` — the OP2 usage tap the run/progress services read.
    Streams line-by-line (never loads the file); any failure → not present.
    """
    if not orch_run_id:
        return {"present": False, "rows": 0}
    try:
        from lib.paths import get_state_runs_dir  # noqa: PLC0415

        path = get_state_runs_dir() / orch_run_id / "llm_usage.jsonl"
        if not path.is_file():
            return {"present": False, "rows": 0}
        rows = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows += 1
        return {"present": True, "rows": rows}
    except Exception:  # noqa: BLE001 — probe is best-effort
        return {"present": False, "rows": 0}


def _compute_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Resolve + run the post-mortem group for ``run_id``; None → unknown run."""
    resolved = _resolve_run(run_id)
    if resolved is None:
        return None
    orch_run_id = resolved["orch_run_id"]
    extra = {
        "run_id": run_id,
        "orchestrator_run_id": orch_run_id,
        "effective_status": resolved["effective_status"],
        "usage": _usage_probe(orch_run_id),
    }
    try:
        _bootstrap()
        results = run_checks(CheckContext(run_id=orch_run_id), groups=[_POSTMORTEM_GROUP])
        return _shape_payload(results, extra=extra)
    except Exception as exc:  # noqa: BLE001 — degrade to an error payload
        logger.exception("health: run post-mortem compute failed for %s", run_id)
        return _error_payload(exc, extra=extra)


def run_health(run_id: str, *, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Return the cached run-scoped post-mortem payload, or ``None`` (unknown run).

    Cache is keyed per run id with the same TTL + force-refresh contract as the
    global payload. A miss (unknown run) is NEVER cached, so a run that appears
    later becomes observable without a restart.
    """
    key = str(run_id or "").strip()
    if not key:
        return None
    with _lock:
        now = time.monotonic()
        cached = _run_cache.get(key)
        if not force_refresh and cached and (now - cached["monotonic"]) < CACHE_TTL_SECONDS:
            return cached["payload"]
        payload = _compute_run(key)
        if payload is None:
            return None  # unknown run → router 404 (never cache the miss)
        _run_cache[key] = {"monotonic": time.monotonic(), "payload": payload}
        return payload


__all__ = [
    "CACHE_TTL_SECONDS",
    "get_health",
    "run_health",
    "reset_cache",
]
