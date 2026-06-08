"""Real wrappers around the Ed4All orchestrator for the control-plane GUI.

This module is the *service* layer behind the RUNS vertical of the GUI. It
wraps the SAME backend functions the ``ed4all run`` CLI drives — there are no
stubs here. Every launch creates a real workflow state under
``state/workflows/`` and runs it through ``PipelineOrchestrator``.

Import contract: this module imports cleanly WITHOUT FastAPI. All heavy MCP /
orchestrator imports are deferred inside the launch functions so
``python -c "import gui.services.run_service"`` stays light. The only top-level
dependency is ``gui.shared_state`` + ``gui.settings_store`` (which themselves
have no web deps) and stdlib.

Run-record schema persisted to ``state/gui/runs/<run_id>.json`` (via
``shared_state.register_run`` / ``update_run``)::

    {
      "run_id":       "GUI-20260531-ab12cd",   # GUI run id (log + ws key)
      "kind":         "pipeline" | "phase",
      "workflow":     "textbook_to_course",     # workflow name (or stage alias)
      "workflow_id":  "WF-20260531-abcd1234",   # orchestrator workflow id
      "course_name":  "PHYS_101",
      "phase":        "courseforge-outline",    # phase runs only
      "mode":         "api" | "local",
      "provider":     "anthropic",
      "model":        "claude-sonnet-4-6" | null,
      "status":       "queued" | "running" | "completed" | "failed" | "cancelled",
      "params":       { ... },                  # the params handed to the backend
      "gate_results": [ ... ] | null,           # populated for phase runs
      "tasks":        [ ... ] | null,           # populated for phase runs
      "error":        "<real error>" | null,
      "created_at":   "<iso>",
      "updated_at":   "<iso>",
      "started_at":   "<iso>" | null,
      "finished_at":  "<iso>" | null
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui import settings_store, shared_state

logger = logging.getLogger("gui.run_service")

# Workflow names the GUI can launch as a full pipeline. Mirrors the CLI's
# SUPPORTED_WORKFLOWS, normalized to underscore form.
SUPPORTED_WORKFLOWS = (
    "textbook_to_course",
    "course_generation",
    "intake_remediation",
    "batch_dart",
    "rag_training",
    "trainforge_train",
)

# Courseforge stage subcommands — thin aliases over ``textbook_to_course``
# that re-execute only one Phase 3 tier. Mirrors
# ``cli/commands/run.py::COURSEFORGE_STAGE_SUBCOMMANDS``.
COURSEFORGE_STAGE_SUBCOMMANDS = (
    "courseforge",
    "courseforge_outline",
    "courseforge_validate",
    "courseforge_rewrite",
)

DEFAULT_DART_OUTPUT_DIR = "DART/output"

# Keep handles on the background asyncio tasks so they aren't garbage-collected
# mid-flight (asyncio holds only a weak reference to scheduled tasks) AND so
# ``cancel_run`` can look up the running task by run_id and actually cancel it.
# Keyed by GUI run_id -> the driving asyncio.Task.
_BACKGROUND_TASKS: Dict[str, asyncio.Task] = {}


def _normalize_workflow(name: str) -> str:
    """Normalize a workflow name to underscore/lowercase form."""
    return (name or "").replace("-", "_").strip().lower()


# --------------------------------------------------------------------- list


def list_workflows() -> List[Dict[str, Any]]:
    """Return every launchable workflow with its phase shape.

    Reads from ``OrchestratorConfig.load()`` (the real YAML config) so the GUI
    never hardcodes phase lists. Each phase carries ``name``, ``agents``,
    ``depends_on`` and a ``validation_gates_count``. Courseforge stage
    subcommands are appended as aliases of ``textbook_to_course``.
    """
    from MCP.core.config import OrchestratorConfig  # noqa: PLC0415

    config = OrchestratorConfig.load()
    workflows: List[Dict[str, Any]] = []
    for name, wf in config.workflows.items():
        phases = []
        for phase in wf.phases:
            gates = phase.validation_gates or []
            phases.append(
                {
                    "name": phase.name,
                    "agents": list(phase.agents or []),
                    "depends_on": list(phase.depends_on or []),
                    "validation_gates_count": len(gates),
                    "optional": bool(getattr(phase, "optional", False)),
                }
            )
        workflows.append(
            {
                "name": name,
                "description": getattr(wf, "description", "") or "",
                "phases": phases,
                "stage_subcommand": False,
            }
        )

    # Append the Courseforge stage subcommands as aliases. They re-execute a
    # single tier of the textbook_to_course pipeline; surface them so the GUI
    # can offer them as launch targets.
    for alias in COURSEFORGE_STAGE_SUBCOMMANDS:
        workflows.append(
            {
                "name": alias,
                "description": (
                    f"Courseforge stage subcommand — re-run the '{alias}' tier "
                    "against an existing textbook_to_course project export."
                ),
                "phases": [],
                "stage_subcommand": True,
            }
        )
    return workflows


# ------------------------------------------------------------------ helpers


def _resolve_corpus(corpus: Any) -> str:
    """Resolve a ``corpus`` request value into a filesystem path string.

    Accepts:
    - a path string (returned as-is),
    - an ``upload_id`` referencing ``uploads_dir()/<upload_id>/`` — resolved to
      the directory (if multiple files) or the single file inside it.

    Raises ``ValueError`` when an upload_id names no saved files.
    """
    if not corpus:
        return ""
    text = str(corpus).strip()
    if not text:
        return ""

    # If it looks like an existing path, use it directly.
    candidate = Path(text)
    if candidate.exists():
        return str(candidate)

    # Otherwise treat it as an upload_id under uploads_dir().
    upload_root = shared_state.uploads_dir() / text
    if upload_root.is_dir():
        files = sorted(
            p for p in upload_root.iterdir() if p.is_file() and not p.name.startswith(".")
        )
        if not files:
            raise ValueError(f"upload {text!r} contains no files")
        if len(files) == 1:
            return str(files[0])
        # Multiple files -> pass the directory; downstream globs *.pdf.
        return str(upload_root)

    # Last resort: not a path, not an upload — surface the raw value so the
    # backend's own validation produces the authoritative error.
    return text


def _apply_request_env(req: Dict[str, Any]) -> Dict[str, str]:
    """Apply the saved settings env, then overlay per-request overrides.

    First applies the persisted settings doc (provider/model/key env vars) so
    the launched run inherits the user's saved configuration. Then overlays any
    per-request ``mode`` / ``provider`` / ``model`` into ``os.environ`` so a
    one-off launch can override the saved defaults without mutating settings.
    """
    applied = settings_store.apply_env(settings_store.load_settings())

    overrides: Dict[str, str] = {}
    mode = req.get("mode")
    if mode:
        overrides["LLM_MODE"] = str(mode)
    provider = req.get("provider")
    if provider:
        overrides["LLM_PROVIDER"] = str(provider)
    model = req.get("model")
    if model:
        overrides["LLM_MODEL"] = str(model)

    for key, value in overrides.items():
        os.environ[key] = value
        applied[key] = value
    return applied


def _resolve_mode(req: Dict[str, Any]) -> str:
    """Resolve execution mode: request > env LLM_MODE > 'local'."""
    return str(req.get("mode") or os.environ.get("LLM_MODE", "local"))


def _resolve_provider(req: Dict[str, Any]) -> str:
    """Resolve provider: request > env LLM_PROVIDER > 'anthropic'."""
    return str(req.get("provider") or os.environ.get("LLM_PROVIDER", "anthropic"))


# ------------------------------------------------------------------ launch


async def launch_pipeline(req: Dict[str, Any]) -> Dict[str, Any]:
    """Launch a full workflow pipeline.

    Steps (no stubs):
    1. Apply saved settings env + per-request overrides into ``os.environ``.
    2. Create a REAL workflow via ``create_textbook_pipeline`` (textbook_to_course
       + Courseforge stage subcommands) or ``create_workflow_impl`` (others).
    3. Register a GUI run record in ``state/gui/runs/``.
    4. Kick off execution in a background asyncio task that drives
       ``PipelineOrchestrator.run`` and streams status/log to disk.

    Never fabricates success: if workflow creation raises or returns an error,
    the run record is stamped ``status="failed"`` with the real error and that
    error is returned.

    ``req`` shape (frontend contract)::

        {workflow, course_name, corpus, weeks?, mode?, provider?, model?, options{}}
    """
    workflow = _normalize_workflow(req.get("workflow", ""))
    course_name = req.get("course_name")
    options = req.get("options") or {}
    if not isinstance(options, dict):
        options = {}

    run_id = shared_state.new_run_id("GUI")
    mode = _resolve_mode(req)
    provider = _resolve_provider(req)
    model = req.get("model")

    # Validate workflow name up-front so a typo doesn't create orphan state.
    is_stage = workflow in COURSEFORGE_STAGE_SUBCOMMANDS
    if workflow not in SUPPORTED_WORKFLOWS and not is_stage:
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error=(
                f"unknown workflow {workflow!r}; choose from "
                f"{sorted(SUPPORTED_WORKFLOWS) + list(COURSEFORGE_STAGE_SUBCOMMANDS)}"
            ),
        )
    if not course_name or len(str(course_name)) < 2:
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error="course_name is required (>= 2 chars)",
        )

    # Apply env BEFORE creating the workflow so canonical-course-code +
    # capture wiring see the user's settings.
    try:
        _apply_request_env(req)
    except Exception as exc:  # noqa: BLE001 — surface real settings error
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error=f"failed to apply settings env: {exc}",
        )

    # Build params + create the real workflow.
    try:
        created, params = await _create_workflow(
            workflow=workflow,
            course_name=str(course_name),
            corpus=req.get("corpus"),
            weeks=req.get("weeks"),
            options=options,
            is_stage=is_stage,
        )
    except Exception as exc:  # noqa: BLE001 — never fabricate success
        logger.exception("workflow creation crashed")
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error=f"{exc}",
            traceback_text=traceback.format_exc(),
        )

    if "error" in created:
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error=str(created.get("error")),
            detail=created,
        )

    workflow_id = created.get("workflow_id")
    if not workflow_id:
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error="workflow creation returned no workflow_id",
            detail=created,
        )

    # Register the run record (status=queued) and start streaming.
    record = {
        "run_id": run_id,
        "kind": "pipeline",
        "workflow": workflow,
        "workflow_id": workflow_id,
        "course_name": course_name,
        "phase": None,
        "mode": mode,
        "provider": provider,
        "model": model,
        "status": "queued",
        "params": params,
        "gate_results": None,
        "tasks": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    shared_state.register_run(record)
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] queued workflow {workflow_id} "
        f"({workflow}) mode={mode} provider={provider}\n",
    )
    shared_state.append_event(
        "gui",
        "run_launched",
        {"run_id": run_id, "workflow_id": workflow_id, "workflow": workflow},
    )

    # Kick off the orchestrator in the background.
    task = asyncio.ensure_future(
        _drive_pipeline(run_id, workflow_id, mode=mode, provider=provider, model=model)
    )
    _BACKGROUND_TASKS[run_id] = task
    # Clean up the handle when the task finishes (only if it's still ours — a
    # later run could never collide because run_ids are unique, but guard anyway).
    task.add_done_callback(lambda _t, _rid=run_id: _BACKGROUND_TASKS.pop(_rid, None))

    return {"run_id": run_id, "workflow_id": workflow_id, "status": "queued"}


async def _create_workflow(
    *,
    workflow: str,
    course_name: str,
    corpus: Any,
    weeks: Any,
    options: Dict[str, Any],
    is_stage: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create the real backend workflow; return ``(created_json, params)``.

    Textbook_to_course + Courseforge stage subcommands route through
    ``create_textbook_pipeline``; everything else through ``create_workflow_impl``.
    """
    from MCP.tools.orchestrator_tools import create_workflow_impl  # noqa: PLC0415
    from MCP.tools.pipeline_tools import create_textbook_pipeline  # noqa: PLC0415

    weeks_explicit = weeks is not None
    duration_weeks = int(weeks) if weeks is not None else 12

    courseforge_stage = workflow if is_stage else None

    if is_stage or workflow == "textbook_to_course":
        corpus_path = _resolve_corpus(corpus) if not is_stage else ""
        generate_assessments = not bool(options.get("no_assessments", False))
        raw = await create_textbook_pipeline(
            pdf_paths=corpus_path,
            course_name=course_name,
            objectives_path=options.get("objectives_path"),
            duration_weeks=duration_weeks,
            duration_weeks_explicit=weeks_explicit,
            generate_assessments=generate_assessments,
            assessment_count=int(options.get("assessment_count", 50)),
            bloom_levels=options.get(
                "bloom_levels", "remember,understand,apply,analyze"
            ),
            priority=str(options.get("priority", "normal")),
            skip_dart=bool(options.get("skip_dart", False)),
            dart_output_dir=options.get("dart_output_dir"),
            reuse_objectives_path=options.get("reuse_objectives_path"),
            courseforge_stage=courseforge_stage,
            force_rerun=bool(options.get("force_rerun", False)),
            skip_training=bool(options.get("skip_training", False)),
        )
        created = json.loads(raw)
        params = created.get("params") or {
            "course_name": course_name,
            "corpus": corpus_path,
            "duration_weeks": duration_weeks,
            "courseforge_stage": courseforge_stage,
        }
        return created, params

    # Generic workflow path.
    params: Dict[str, Any] = {
        "course_name": course_name,
        "corpus": _resolve_corpus(corpus),
        "priority": str(options.get("priority", "normal")),
    }
    if weeks is not None:
        params["duration_weeks"] = duration_weeks
    params.update({k: v for k, v in options.items() if k not in params})

    raw = await create_workflow_impl(
        workflow_type=workflow,
        params=json.dumps(params),
        priority=params.get("priority", "normal"),
    )
    created = json.loads(raw)
    return created, params


async def _drive_pipeline(
    run_id: str,
    workflow_id: str,
    *,
    mode: str,
    provider: str,
    model: Optional[str],
) -> None:
    """Background driver: run the orchestrator + persist status/log transitions.

    Deferred imports keep ``run_service`` light at import time. Any exception is
    captured into the run record as ``status="failed"`` with the real error —
    never swallowed into a fake success.
    """
    try:
        shared_state.update_run(run_id, {"status": "running", "started_at": shared_state.now_iso()})
    except FileNotFoundError:
        # Run was deleted out from under us; nothing to drive.
        return
    shared_state.append_log(run_id, f"[{shared_state.now_iso()}] running {workflow_id}\n")

    try:
        from MCP.orchestrator import PipelineOrchestrator  # noqa: PLC0415
        from MCP.orchestrator.llm_backend import BackendSpec  # noqa: PLC0415

        spec = BackendSpec(mode=mode, provider=provider, model=model)
        orchestrator = PipelineOrchestrator(mode=mode, backend_spec=spec)
        result = await orchestrator.run(workflow_id)
    except asyncio.CancelledError:
        shared_state.append_log(run_id, f"[{shared_state.now_iso()}] cancelled\n")
        _safe_update(
            run_id,
            {"status": "cancelled", "finished_at": shared_state.now_iso()},
        )
        raise
    except Exception as exc:  # noqa: BLE001 — record the real error
        logger.exception("pipeline run crashed for %s", workflow_id)
        tb = traceback.format_exc()
        shared_state.append_log(run_id, f"[{shared_state.now_iso()}] ERROR: {exc}\n{tb}\n")
        if _finalize_status(
            run_id,
            {
                "status": "failed",
                "error": str(exc),
                "finished_at": shared_state.now_iso(),
            },
        ):
            shared_state.append_event("gui", "run_failed", {"run_id": run_id, "error": str(exc)})
        return

    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    status = "completed" if payload.get("status") == "ok" else "failed"
    gates_passed = payload.get("gates_passed")
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] finished status={payload.get('status')} "
        f"gates_passed={gates_passed}\n",
    )
    if payload.get("phase_results"):
        for name, info in payload["phase_results"].items():
            if isinstance(info, dict):
                shared_state.append_log(
                    run_id,
                    f"  phase {name}: completed={info.get('completed', 0)}/"
                    f"{info.get('task_count', 0)} gates="
                    f"{'pass' if info.get('gates_passed') else 'fail'}\n",
                )
    if _finalize_status(
        run_id,
        {
            "status": status,
            "error": payload.get("error"),
            "gate_results": payload.get("phase_results"),
            "gates_passed": gates_passed,
            "finished_at": shared_state.now_iso(),
        },
    ):
        shared_state.append_event(
            "gui",
            "run_finished",
            {"run_id": run_id, "workflow_id": workflow_id, "status": status},
        )


def _safe_update(run_id: str, patch: Dict[str, Any]) -> None:
    """Best-effort ``update_run`` that tolerates a deleted run record."""
    try:
        shared_state.update_run(run_id, patch)
    except FileNotFoundError:
        logger.warning("run %s vanished before update %s", run_id, patch)


def _finalize_status(run_id: str, patch: Dict[str, Any]) -> bool:
    """Write a terminal-status ``patch`` UNLESS the run was already cancelled.

    Cancellation (``cancel_run``) flips the record to ``status="cancelled"`` and
    calls ``task.cancel()`` on the driver. But the orchestrator's
    cancellation granularity is phase-boundary / cooperative — the underlying
    ``orchestrator.run`` may finish its current phase (or the whole workflow)
    before the ``CancelledError`` is delivered, then fall through to the normal
    ``completed``/``failed`` terminal write. Re-read the record here and refuse
    to clobber a ``cancelled`` status with ``completed``/``failed`` so a cancel
    requested mid-run is never silently overwritten.

    Returns ``True`` when the patch was written, ``False`` when it was skipped
    (already cancelled) — so the caller can suppress the matching event emission.
    """
    current = shared_state.read_run(run_id)
    if current is not None and current.get("status") == "cancelled":
        logger.info("run %s already cancelled; not overwriting with %s", run_id, patch.get("status"))
        return False
    _safe_update(run_id, patch)
    return True


def _fail_phase(run_id: str, msg: str) -> None:
    """Stamp a phase run as failed with ``msg`` (terse helper)."""
    _safe_update(
        run_id,
        {"status": "failed", "error": msg, "finished_at": shared_state.now_iso()},
    )


def _record_launch_failure(
    run_id: str,
    *,
    workflow: str,
    course_name: Any,
    kind: str,
    mode: str,
    provider: str,
    model: Optional[str],
    error: str,
    detail: Optional[Dict[str, Any]] = None,
    traceback_text: Optional[str] = None,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a failed run record (never fabricate success) and return it.

    Persists the real error to ``state/gui/runs/<run_id>.json`` + the log so the
    failure is visible in the GUI / via the registry. Returns the frontend
    response payload carrying ``status="failed"`` and the real error.
    """
    record = {
        "run_id": run_id,
        "kind": kind,
        "workflow": workflow,
        "workflow_id": None,
        "course_name": course_name,
        "phase": phase,
        "mode": mode,
        "provider": provider,
        "model": model,
        "status": "failed",
        "params": None,
        "gate_results": None,
        "tasks": None,
        "error": error,
        "detail": detail,
        "started_at": None,
        "finished_at": shared_state.now_iso(),
    }
    try:
        shared_state.register_run(record)
        shared_state.append_log(run_id, f"[{shared_state.now_iso()}] FAILED: {error}\n")
        if traceback_text:
            shared_state.append_log(run_id, traceback_text + "\n")
        shared_state.append_event("gui", "run_failed", {"run_id": run_id, "error": error})
    except Exception:  # noqa: BLE001 — registration is best-effort on failure
        logger.exception("failed to persist launch-failure record for %s", run_id)
    return {"run_id": run_id, "workflow_id": None, "status": "failed", "error": error}


# ------------------------------------------------------------------ phase


async def launch_phase(req: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single workflow phase via the documented phase pathway.

    Pathway (per spec §5):
    ``OrchestratorConfig`` -> pick the named phase ->
    ``WorkflowRunner._route_params`` -> ``_create_phase_tasks`` ->
    ``TaskExecutor.execute_phase``. Upstream ``phase_outputs`` are pre-populated
    from any ``project_id``-named export via ``_synthesize_outline_output`` when
    available, so the target phase has its inputs.

    ``req`` shape (frontend contract)::

        {workflow, phase, course_name, project_id?, options{}}

    Returns ``{run_id, status, tasks?, gate_results?}``. On error, records the
    real error into the run registry and returns ``status="failed"``.
    """
    workflow = _normalize_workflow(req.get("workflow", ""))
    phase_name = req.get("phase")
    course_name = req.get("course_name")
    project_id = req.get("project_id")
    options = req.get("options") or {}
    if not isinstance(options, dict):
        options = {}

    run_id = shared_state.new_run_id("GUI")
    mode = _resolve_mode(req)
    provider = _resolve_provider(req)
    model = req.get("model")

    if not phase_name:
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="phase",
            mode=mode,
            provider=provider,
            model=model,
            error="phase is required for a single-phase run",
            phase=phase_name,
        )

    try:
        _apply_request_env(req)
    except Exception as exc:  # noqa: BLE001
        return _record_launch_failure(
            run_id,
            workflow=workflow,
            course_name=course_name,
            kind="phase",
            mode=mode,
            provider=provider,
            model=model,
            error=f"failed to apply settings env: {exc}",
            phase=phase_name,
        )

    # Register the run as queued before doing the heavy lifting.
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "phase",
            "workflow": workflow,
            "workflow_id": None,
            "course_name": course_name,
            "phase": phase_name,
            "mode": mode,
            "provider": provider,
            "model": model,
            "status": "running",
            "params": {"project_id": project_id, "options": options},
            "gate_results": None,
            "tasks": None,
            "error": None,
            "started_at": shared_state.now_iso(),
            "finished_at": None,
        }
    )
    shared_state.append_log(
        run_id, f"[{shared_state.now_iso()}] phase run '{phase_name}' ({workflow})\n"
    )

    try:
        outcome = await _run_single_phase(
            run_id=run_id,
            workflow=workflow,
            phase_name=str(phase_name),
            course_name=course_name,
            project_id=project_id,
            options=options,
        )
    except Exception as exc:  # noqa: BLE001 — never fabricate success
        logger.exception("single-phase run crashed")
        tb = traceback.format_exc()
        shared_state.append_log(run_id, f"[{shared_state.now_iso()}] ERROR: {exc}\n{tb}\n")
        _safe_update(
            run_id,
            {"status": "failed", "error": str(exc), "finished_at": shared_state.now_iso()},
        )
        return {"run_id": run_id, "status": "failed", "error": str(exc)}

    return outcome


async def _run_single_phase(
    *,
    run_id: str,
    workflow: str,
    phase_name: str,
    course_name: Any,
    project_id: Optional[str],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Drive the real single-phase execution pathway and persist the outcome."""
    from MCP.core.config import OrchestratorConfig  # noqa: PLC0415
    from MCP.core.executor import TaskExecutor  # noqa: PLC0415
    from MCP.core.workflow_runner import WorkflowRunner  # noqa: PLC0415
    from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: PLC0415

    config = OrchestratorConfig.load()

    # Courseforge stage aliases run against the textbook_to_course state machine.
    wf_name = workflow
    if workflow in COURSEFORGE_STAGE_SUBCOMMANDS:
        wf_name = "textbook_to_course"
    wf = config.get_workflow(wf_name)
    if wf is None:
        msg = f"unknown workflow {wf_name!r}"
        _fail_phase(run_id, msg)
        return {"run_id": run_id, "status": "failed", "error": msg}

    phase = next((p for p in wf.phases if p.name == phase_name), None)
    if phase is None:
        available = [p.name for p in wf.phases]
        msg = f"phase {phase_name!r} not found in {wf_name!r}; available: {available}"
        _fail_phase(run_id, msg)
        return {"run_id": run_id, "status": "failed", "error": msg}

    # Build the workflow params + pre-populate upstream phase_outputs.
    workflow_params: Dict[str, Any] = {
        "course_name": course_name,
        "duration_weeks": int(options.get("duration_weeks", 12)),
    }
    if project_id:
        workflow_params["project_id"] = project_id
    workflow_params.update({k: v for k, v in options.items() if k not in workflow_params})

    phase_outputs = _prepopulate_phase_outputs(project_id, course_name)

    # Wire an executor with the full pipeline tool registry so phase tasks can
    # resolve their tool names (same registry the orchestrator uses).
    executor = TaskExecutor(tool_registry=_build_tool_registry(), run_id=run_id)
    runner = WorkflowRunner(executor, config)

    routed = runner._route_params(phase_name, workflow_params, phase_outputs)
    tasks = runner._create_phase_tasks(
        workflow_id=run_id,
        phase=phase,
        routed_params=routed,
        workflow_params=workflow_params,
    )
    shared_state.append_log(
        run_id, f"[{shared_state.now_iso()}] created {len(tasks)} task(s) for phase {phase_name}\n"
    )

    gate_configs = phase.validation_gates or None
    results, gates_passed, gate_results = await executor.execute_phase(
        workflow_id=run_id,
        phase_name=phase_name,
        phase_index=0,
        tasks=tasks,
        gate_configs=gate_configs,
        max_concurrent=getattr(phase, "max_concurrent", 5),
        phase_outputs=phase_outputs,
        workflow_params=workflow_params,
    )

    task_summaries = _summarize_task_results(results)
    status = "completed" if gates_passed else "failed"
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] phase {phase_name} done: gates_passed={gates_passed}\n",
    )
    _safe_update(
        run_id,
        {
            "status": status,
            "tasks": task_summaries,
            "gate_results": gate_results,
            "gates_passed": bool(gates_passed),
            "finished_at": shared_state.now_iso(),
        },
    )
    shared_state.append_event(
        "gui",
        "phase_finished",
        {"run_id": run_id, "phase": phase_name, "status": status},
    )
    return {
        "run_id": run_id,
        "status": status,
        "tasks": task_summaries,
        "gate_results": gate_results,
    }


def _prepopulate_phase_outputs(
    project_id: Optional[str], course_name: Any
) -> Dict[str, Dict[str, Any]]:
    """Synthesize upstream ``phase_outputs`` from an existing project export.

    Resolves the Courseforge project export directory (from ``project_id`` or,
    failing that, the newest ``PROJ-<course_name>-*`` export) and calls
    ``WorkflowRunner._synthesize_outline_output(outline_dir)`` so a single-phase
    re-run (e.g. ``courseforge-outline``) finds the inputs its upstream phases
    would have produced. Best-effort: returns ``{}`` when no export is found
    (the phase then runs against whatever inputs its routing can resolve, and
    any genuine missing-input is surfaced by the executor — not faked).
    """
    export_dir = _resolve_project_export(project_id, course_name)
    if export_dir is None:
        return {}
    try:
        from MCP.core.config import OrchestratorConfig  # noqa: PLC0415
        from MCP.core.executor import TaskExecutor  # noqa: PLC0415
        from MCP.core.workflow_runner import WorkflowRunner  # noqa: PLC0415
        from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: PLC0415

        runner = WorkflowRunner(
            TaskExecutor(tool_registry=_build_tool_registry()),
            OrchestratorConfig.load(),
        )
        outputs = runner._synthesize_outline_output(export_dir)
        if isinstance(outputs, dict):
            return outputs
    except Exception as exc:  # noqa: BLE001 — synthesis is best-effort
        logger.debug("phase_outputs pre-population skipped: %s", exc)
    return {}


def _resolve_project_export(
    project_id: Optional[str], course_name: Any
) -> Optional[Path]:
    """Return the Courseforge export dir for a project_id / course_name.

    ``project_id`` may be a bare ``PROJ-...`` dir name or an absolute path.
    Otherwise the newest ``PROJ-<course_name>-*`` export under
    ``Courseforge/exports/`` is chosen. Returns ``None`` when nothing matches.
    """
    from lib.paths import COURSEFORGE_PATH  # noqa: PLC0415

    exports = Path(COURSEFORGE_PATH) / "exports"
    if project_id:
        candidate = Path(project_id)
        if candidate.is_absolute() and candidate.is_dir():
            return candidate
        named = exports / project_id
        if named.is_dir():
            return named
    if course_name and exports.is_dir():
        prefix = f"PROJ-{course_name}-"
        matches = sorted(
            (p for p in exports.iterdir() if p.is_dir() and p.name.startswith(prefix)),
            key=lambda p: p.name,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


def _summarize_task_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map ``ExecutionResult`` objects to JSON-safe summary dicts."""
    out: List[Dict[str, Any]] = []
    for task_id, res in (results or {}).items():
        out.append(
            {
                "task_id": task_id,
                "status": getattr(res, "status", None) or _attr(res, "state"),
                "success": bool(getattr(res, "success", False)),
                "error": getattr(res, "error", None),
            }
        )
    return out


def _attr(obj: Any, name: str) -> Any:
    """Safe attribute read returning ``None`` when absent."""
    return getattr(obj, name, None)


# ------------------------------------------------------------------ status


def run_status(run_id: str) -> Optional[Dict[str, Any]]:
    """Return the run record for ``run_id`` (or ``None`` if unknown)."""
    return shared_state.read_run(run_id)


def list_runs(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return GUI run records, newest-first.

    ``limit`` (optional) caps the number of records returned; ``None`` (default)
    returns all of them, preserving the prior unbounded behavior for callers that
    don't pass it. A non-positive limit returns an empty list.
    """
    runs = shared_state.list_runs()
    if limit is not None:
        if limit <= 0:
            return []
        return runs[:limit]
    return runs


# Non-terminal states a fresh process should reconcile on boot. ``queued`` and
# ``running`` (pipeline) plus phase runs that register straight as ``running``.
_NON_TERMINAL_STATES = ("queued", "running")


def reconcile_orphans() -> List[str]:
    """Flip any non-terminal run left over from a previous process to terminal.

    Background driver tasks live only in-process, so a uvicorn restart leaves any
    ``queued``/``running`` record stuck forever (no task is driving it, and its WS
    clients would poll indefinitely waiting for a terminal frame). On boot, scan
    the registry and stamp every non-terminal run as ``status="interrupted"`` with
    a short note + ``finished_at`` so the registry and WS clients see a terminal
    state. Best-effort: a record that vanishes / fails to update is skipped.

    Returns the list of run_ids that were reconciled (for logging/tests).
    """
    reconciled: List[str] = []
    for record in shared_state.list_runs():
        if record.get("status") not in _NON_TERMINAL_STATES:
            continue
        run_id = record.get("run_id")
        if not run_id:
            continue
        try:
            shared_state.update_run(
                run_id,
                {
                    "status": "interrupted",
                    "error": record.get("error")
                    or "process restarted; run was no longer being driven",
                    "finished_at": shared_state.now_iso(),
                },
            )
            shared_state.append_log(
                run_id,
                f"[{shared_state.now_iso()}] interrupted: orchestrator process "
                "restarted; run was no longer being driven\n",
            )
            reconciled.append(run_id)
        except FileNotFoundError:
            logger.warning("run %s vanished during orphan reconciliation", run_id)
        except Exception:  # noqa: BLE001 — reconciliation is best-effort
            logger.exception("failed to reconcile orphan run %s", run_id)
    if reconciled:
        logger.info("reconciled %d orphaned run(s): %s", len(reconciled), reconciled)
    return reconciled


def tail_log(run_id: str, offset: int = 0) -> Tuple[str, int]:
    """Return ``(new_text, new_offset)`` from the run's log for ws streaming."""
    return shared_state.tail_log(run_id, offset)


def cancel_run(run_id: str) -> Dict[str, Any]:
    """Request cancellation of a run.

    Flips the registry record to ``status="cancelled"`` AND cancels the
    in-process background driver task (when this process owns it). The
    ``_finalize_status`` guard in ``_drive_pipeline`` then refuses to clobber the
    ``cancelled`` status with a late ``completed``/``failed`` terminal write.

    Cancellation granularity is phase-boundary / cooperative: cancelling the
    asyncio task injects ``CancelledError`` at the next ``await`` boundary, but
    the orchestrator may already be mid-phase and may not honor mid-phase
    interruption — the currently-executing phase can run to completion before the
    cancellation is observed. The registry flip is the authoritative signal; the
    task-cancel is best-effort acceleration.

    Returns the updated record, or a typed error when the run is unknown or
    already terminal.
    """
    record = shared_state.read_run(run_id)
    if record is None:
        return {"error": "unknown_run", "detail": f"no run with id {run_id!r}"}
    if record.get("status") in ("completed", "failed", "cancelled", "interrupted"):
        return {
            "run_id": run_id,
            "status": record.get("status"),
            "note": "run already terminal; nothing to cancel",
        }
    updated = shared_state.update_run(
        run_id, {"status": "cancelled", "finished_at": shared_state.now_iso()}
    )
    # Actually cancel the driving asyncio task if this process owns it. After a
    # uvicorn restart the handle is gone (in-process only); the status flip above
    # still terminates the run from the registry's / WS clients' point of view.
    task = _BACKGROUND_TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    shared_state.append_log(run_id, f"[{shared_state.now_iso()}] cancellation requested\n")
    shared_state.append_event("gui", "run_cancelled", {"run_id": run_id})
    return {"run_id": run_id, "status": updated.get("status")}


__all__ = [
    "list_workflows",
    "launch_pipeline",
    "launch_phase",
    "run_status",
    "list_runs",
    "reconcile_orphans",
    "tail_log",
    "cancel_run",
    "SUPPORTED_WORKFLOWS",
    "COURSEFORGE_STAGE_SUBCOMMANDS",
]
