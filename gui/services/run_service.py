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
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui import settings_store, shared_state
from gui.services import liveness

logger = logging.getLogger("gui.run_service")

# Workflow names the GUI can launch as a full pipeline. Mirrors the CLI's
# SUPPORTED_WORKFLOWS, normalized to underscore form.
SUPPORTED_WORKFLOWS = (
    "textbook_to_course",
    "course_generation",
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

DEFAULT_DART_OUTPUT_DIR = "SemantiK/output"

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
                    "label": PHASE_LABELS.get(phase.name, ""),
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


# Marketable-v1 A3 — the blessed turnkey authoring route. The GUI launches
# runs headless (no Claude Code session servicing the mailbox), so every
# LLM-needing agent MUST resolve its generation through the in-process
# provider lattice. Each entry below is an agent's provider env var; when
# set, ``TaskExecutor._invoke_tool`` short-circuits the mailbox subagent
# dispatch and runs the in-process tool, which routes through the
# OpenAI-compatible provider registry. The env-var literals are the canonical
# ``MCP.core.executor.AGENT_AUTHORING_PROVIDER_ENV_MAP`` values, reproduced as
# a plain tuple so this module stays import-light (no MCP import at module
# load). A drift guard test asserts they match the executor map.
_AUTHORING_PROVIDER_ENVS: Tuple[str, ...] = (
    "COURSEFORGE_PROVIDER",        # content-generator
    "COURSEPLANNER_PROVIDER",      # course-outliner
    "TRAINFORGE_ASSESSMENT_PROVIDER",  # assessment-generator
    "TRAINFORGE_SYNTHESIS_PROVIDER",   # training-synthesizer
)


def _apply_authoring_route_env(req: Dict[str, Any]) -> Dict[str, str]:
    """Set the blessed authoring-route provider env for an enqueued run.

    A GUI / headless run has no Claude session draining the mailbox, so the
    workflow_runner guardrail (``_enforce_authoring_provider_route``) would
    fail any LLM-needing phase whose ``<AGENT>_PROVIDER`` env is unset. This
    helper fills every such env that isn't already set (via settings
    ``model_routing`` or a prior overlay) with the resolved authoring
    provider so the run routes through the in-process lattice by default.

    Resolution per env var (only when currently unset/empty):
      request ``provider`` > env ``LLM_PROVIDER`` (global routing provider)
      > ``"local"`` (license-clean default; an air-gapped Ollama/vLLM lattice
      provider that needs no key).

    Returns the env vars this helper set (for logging / tests). Idempotent:
    an env already populated (e.g. ``COURSEPLANNER_PROVIDER`` set via
    ``model_routing.courseplanner.provider``) is left untouched, so per-task
    routing the user configured in settings still wins.
    """
    resolved = (
        str(req.get("provider") or "").strip()
        or os.environ.get("LLM_PROVIDER", "").strip()
        or "local"
    )
    applied: Dict[str, str] = {}
    for env_var in _AUTHORING_PROVIDER_ENVS:
        if os.environ.get(env_var, "").strip():
            continue
        os.environ[env_var] = resolved
        applied[env_var] = resolved
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

        {workflow, course_name, corpus, weeks?, mode?, provider?, model?,
         resume_run_id?, options{}}

    Phase 4 §5.1(E) RESUME: when ``resume_run_id`` (the GUI run_id of a prior
    ``interrupted``/``failed`` pipeline run with a workflow-state checkpoint) is
    present, this delegates to :func:`resume_run` — re-driving the EXISTING
    orchestrator ``workflow_id`` (the documented CLI ``--resume WF-...`` pathway,
    which the orchestrator honors by skipping already-``_completed`` phases) under
    a fresh GUI run record. No re-upload — the staged corpus + checkpoints persist.
    All other launch params are ignored on a resume (the prior workflow's state is
    authoritative).
    """
    resume_run_id = req.get("resume_run_id")
    if resume_run_id:
        return await resume_run(str(resume_run_id), req)

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
    # capture wiring see the user's settings. Then set the blessed
    # authoring-route provider env (Marketable-v1 A3) so every LLM-needing
    # phase resolves generation through the in-process provider lattice —
    # a GUI/headless run has no Claude session servicing the mailbox, so
    # without this the workflow_runner guardrail would fail those phases.
    try:
        _apply_request_env(req)
        _apply_authoring_route_env(req)
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
            # Persist the operator's --stop-after intent so a bare resume
            # honors the halt point (mirrors the CLI create path). Without
            # this, a GUI-launched run dropped stop_after and a resume ran
            # past the operator's stop marker.
            stop_after=options.get("stop_after"),
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

    # Live phase-progress poller (C3 Create-wizard checklist). The orchestrator
    # only logs phase summaries at the very end, so without this the WS stream
    # carries no per-phase signal mid-run. The poller watches the workflow state
    # file's ``phase_outputs`` completed-markers and appends a friendly
    # ``[phase] <name> <state>`` line as each phase completes; those lines stream
    # over the existing ``/ws/runs/{run_id}`` socket, and the wizard parses them
    # into the checklist. Best-effort + cancelled with the run; never affects the
    # run outcome.
    progress_task = asyncio.ensure_future(_poll_phase_progress(run_id, workflow_id))
    try:
        from MCP.orchestrator import PipelineOrchestrator  # noqa: PLC0415
        from MCP.orchestrator.llm_backend import BackendSpec  # noqa: PLC0415

        spec = BackendSpec(mode=mode, provider=provider, model=model)
        orchestrator = PipelineOrchestrator(mode=mode, backend_spec=spec)
        result = await orchestrator.run(workflow_id)
    except asyncio.CancelledError:
        progress_task.cancel()
        shared_state.append_log(run_id, f"[{shared_state.now_iso()}] cancelled\n")
        _safe_update(
            run_id,
            {"status": "cancelled", "finished_at": shared_state.now_iso()},
        )
        raise
    except Exception as exc:  # noqa: BLE001 — record the real error
        progress_task.cancel()
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
    finally:
        # Stop the poller (best-effort) once the orchestrator returns / raises.
        if not progress_task.done():
            progress_task.cancel()

    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    # Graceful-stop / ``--stop-after`` (I1 review flow): the orchestrator returns
    # ``status="paused"`` (exit-code-3 semantics) when a run deliberately halts at
    # a unit/phase boundary — NOT a failure. Map it to a first-class GUI
    # ``paused`` run status so the Create-wizard progress view can render the
    # objectives-review + resume panel instead of the A6 failure panel. Anything
    # other than ok/paused is a real failure.
    raw_status = payload.get("status")
    if raw_status == "ok":
        status = "completed"
    elif raw_status == "paused":
        status = "paused"
    else:
        status = "failed"
    gates_passed = payload.get("gates_passed")
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] finished status={payload.get('status')} "
        f"gates_passed={gates_passed}\n",
    )
    if payload.get("phase_results"):
        # Resolve the workflow name from the run record so validator-only
        # (``agents: []``) phases are suppressed correctly. Falls back to
        # ``textbook_to_course`` (the only multi-phase pipeline with
        # ``agents: []`` phases) when the record is unreadable.
        _rec = shared_state.read_run(run_id) or {}
        _wf_name = str(_rec.get("workflow") or "textbook_to_course")
        validator_only = _validator_only_phases(_wf_name)
        # Phase 5 §5.1(D) — per-phase declared gate counts (real, from config)
        # size the full-pipeline ``__summary__`` denominator. The rollup carries
        # NO per-gate id/severity detail (see ``_emit_gate_lines`` limitation
        # note), so the full-pipeline path emits ONLY the truthful summary line.
        _gate_counts = _phase_gate_counts(_wf_name)
        for name, info in payload["phase_results"].items():
            if isinstance(info, dict):
                shared_state.append_log(
                    run_id,
                    f"  phase {name}: completed={info.get('completed', 0)}/"
                    f"{info.get('task_count', 0)} gates="
                    f"{'pass' if info.get('gates_passed') else 'fail'}\n",
                )
                # Phase-3 (§5.1(B)) — emit the frontend's
                # ``[<iso>] [progress] <phase> <done>/<total>`` ring signal from
                # the REAL per-phase task count the runner reports. Suppressed
                # for validator-only (``agents: []``) phases whose synthesized
                # virtual task makes ``task_count`` meaningless (their ring
                # stays indeterminate) and for any phase with no real tasks.
                # This is the single truthful END-OF-PHASE total; no mid-phase
                # incremental signal exists GUI-side (see _emit_phase_progress
                # _line's HONEST GAP note).
                if name not in validator_only:
                    _emit_phase_progress_line(
                        run_id,
                        name,
                        info.get("completed", 0),
                        info.get("task_count", 0),
                    )
                # Phase 5 §5.1(D) — emit the per-phase gate ``__summary__`` line
                # from the REAL ``gates_passed`` boolean + the declared gate
                # count. No per-id lines here (the rollup strips them). Emitted
                # for validator-only phases too — a gate-only phase still has a
                # meaningful "all checks passing" summary even with no tasks.
                _emit_pipeline_gate_summary(
                    run_id, name, info, _gate_counts.get(name, 0)
                )

    # Marketable-v1 A6 operator-failure-UX: when the workflow failed, emit a
    # STRUCTURED per-phase failure line + persist ``failed_phase`` /
    # ``failure_reason`` onto the run record. The Create-wizard progress view
    # parses this line to mark the right phase failed (no longer guessing from
    # "last running"), and the failure panel reads the persisted fields. The
    # runner surfaces both fields; fall back to inferring the failed phase from
    # ``phase_results`` (the phase reporting gates_passed=False or failed>0) so
    # a workflow that fails without setting them still gets a structured line.
    failed_phase = payload.get("failed_phase")
    failure_reason = payload.get("failure_reason")
    if status == "failed":
        if not failed_phase:
            failed_phase, failure_reason = _infer_failed_phase(
                payload.get("phase_results"), failure_reason or payload.get("error")
            )
        if failed_phase:
            label = PHASE_LABELS.get(failed_phase, failed_phase)
            reason = failure_reason or payload.get("error") or "unknown failure"
            shared_state.append_log(
                run_id,
                f"[{shared_state.now_iso()}] [phase] {failed_phase} failed "
                f"— {label}: {reason}\n",
            )

    # I1 review flow: when the run PAUSED (graceful stop / ``--stop-after``),
    # surface which phase it halted at so the progress view can render the
    # objectives-review panel. The runner persists ``stopped_after`` (a clean
    # ``--stop-after`` halt) or ``paused_phase`` (a graceful mid-phase stop) on
    # the workflow-state file; read it best-effort (a missing/corrupt file
    # leaves it ``None`` and the view falls back to the persisted stop_after
    # param). Persist it as ``paused_phase`` on the run record + log a friendly
    # checkpoint line for the build log.
    paused_phase: Optional[str] = None
    if status == "paused":
        paused_phase = _read_paused_after(workflow_id)
        label = PHASE_LABELS.get(paused_phase, paused_phase) if paused_phase else None
        shared_state.append_log(
            run_id,
            f"[{shared_state.now_iso()}] paused for review after phase "
            f"{paused_phase or '?'}"
            + (f" — {label}" if label else "")
            + " (resume to continue the build)\n",
        )

    # Phase 0 (Tier-1) — persist the GUI-side per-phase duration vector onto the
    # final run record. Best-effort: derived purely from log lines the backend
    # already emits (zero orchestrator change). A derivation failure here must
    # NEVER change ``final_status`` or break finalize, so it is wrapped and
    # defaults to ``None`` (additive field; absent ⇒ unchanged record shape).
    phase_durations: Optional[List[Dict[str, Any]]] = None
    try:
        phase_durations = derive_phase_timeline(run_id).get("timeline")
    except Exception:  # noqa: BLE001 — timing derivation is best-effort only
        logger.warning("phase-duration derivation failed for %s", run_id, exc_info=True)

    finalize_patch: Dict[str, Any] = {
        "status": status,
        "error": payload.get("error"),
        "gate_results": payload.get("phase_results"),
        "gates_passed": gates_passed,
        "failed_phase": failed_phase,
        "failure_reason": failure_reason,
        "paused_phase": paused_phase,
        "finished_at": shared_state.now_iso(),
    }
    if phase_durations is not None:
        finalize_patch["phase_durations"] = phase_durations

    if _finalize_status(run_id, finalize_patch):
        # A paused run emits a distinct ``run_paused`` activity event (the
        # review-flow signal); a completed/failed run keeps ``run_finished``.
        shared_state.append_event(
            "gui",
            "run_paused" if status == "paused" else "run_finished",
            {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": status,
                **({"paused_phase": paused_phase} if status == "paused" else {}),
            },
        )


# Friendly, end-user-facing phase labels for the textbook_to_course pipeline.
# Keyed by the internal phase name (config/workflows.yaml). The Create wizard's
# progress checklist renders these; the canonical phase order is still read from
# the run record / config, this only maps id -> human label. Unknown phases fall
# back to a title-cased id client-side, so a new phase never breaks the UI.
PHASE_LABELS: Dict[str, str] = {
    # Task #19 Stage 3d renamed the conversion phase; keep the legacy id so
    # old workflow records still render a friendly label.
    "semantik_conversion": "Convert textbook to accessible HTML",
    "dart_conversion": "Convert textbook to accessible HTML",
    "staging": "Stage source files",
    "chunking": "Chunk source content",
    "objective_extraction": "Read textbook structure",
    "source_mapping": "Map sources to modules",
    "course_planning": "Plan learning objectives",
    "concept_extraction": "Extract key concepts",
    "content_generation": "Generate course content",
    "content_generation_outline": "Outline course content",
    "inter_tier_validation": "Validate content",
    "content_generation_rewrite": "Refine course content",
    "post_rewrite_validation": "Re-validate content",
    "packaging": "Package course",
    "imscc_chunking": "Chunk packaged course",
    "trainforge_assessment": "Generate assessments",
    "training_synthesis": "Synthesize training data",
    "libv2_archival": "Archive course",
    "vector_indexing": "Build search index",
    "finalization": "Finalize course",
}


def _validator_only_phases(workflow: str) -> set:
    """Return the set of phase names declaring ``agents: []`` for ``workflow``.

    These phases synthesize a single virtual ``phase-handler`` task (so their
    ``task_count`` is a meaningless ``1``); per the Phase-3 contract they must
    emit NO ``[progress]`` line — their ring stays indeterminate. The canonical
    source of truth is ``config/workflows.yaml`` (``agents: []`` declarations).

    Read directly from the YAML (no MCP import) so the full-pipeline path can
    discriminate without loading ``OrchestratorConfig``. Courseforge stage
    aliases run against the ``textbook_to_course`` machine, so they resolve
    through the same map. Best-effort: any read/parse failure returns an empty
    set (we then emit for every counted phase rather than crash — but never
    fabricate a count).
    """
    name = "textbook_to_course" if workflow in COURSEFORGE_STAGE_SUBCOMMANDS else workflow
    cached = _VALIDATOR_ONLY_PHASES_CACHE.get(name)
    if cached is not None:
        return cached
    result: set = set()
    try:
        import yaml  # noqa: PLC0415
        from lib.paths import PROJECT_ROOT  # noqa: PLC0415

        cfg = yaml.safe_load((Path(PROJECT_ROOT) / "config" / "workflows.yaml").read_text())
        wf = (cfg or {}).get("workflows", {}).get(name)
        if isinstance(wf, dict):
            for ph in wf.get("phases", []) or []:
                if isinstance(ph, dict) and ph.get("agents") == []:
                    result.add(ph.get("name"))
    except Exception:  # noqa: BLE001 — discrimination is best-effort
        logger.debug("validator-only phase lookup failed for %s", name, exc_info=True)
    _VALIDATOR_ONLY_PHASES_CACHE[name] = result
    return result


_VALIDATOR_ONLY_PHASES_CACHE: Dict[str, set] = {}


def _emit_phase_progress_line(
    run_id: str, phase_name: str, completed: int, task_count: int
) -> bool:
    """Append a single TRUTHFUL ``[<iso>] [progress] <phase> <done>/<total>`` line.

    The frontend Build Console (§5.1(B)) parses ``[progress] <phase> X/Y`` to
    fill a per-phase progress ring. We emit ONLY from a genuine task count:
    ``completed`` / ``task_count`` derived from the executor's real per-phase
    results. Returns ``True`` when a line was emitted, ``False`` when suppressed.

    Suppression (never fabricate a count):
      * ``task_count <= 0`` — no real tasks ran (pure gate-chain / count absent).
        Caller is responsible for filtering validator-only ``agents: []`` phases
        (whose synthesized virtual task makes ``task_count == 1`` meaningless).
      * non-int / negative inputs — refuse rather than invent.

    HONEST GAP: the workflow state file is re-saved only at PHASE BOUNDARIES
    (``WorkflowRunner._save_workflow_state`` is never called mid-phase), and the
    executor's live ``progress["completed"]`` (``MCP/core/executor.py``) is held
    in-memory and never persisted GUI-visibly. So there is NO incremental
    mid-phase task count available GUI-side — this emits the single truthful
    END-OF-PHASE total. Incremental mid-phase fill requires the DEFERRED Tier-2
    orchestrator stamp (roadmap §5.2, the ``MCP/core/executor.py`` edit this
    Phase-3 backend is explicitly forbidden from making).
    """
    try:
        done = int(completed)
        total = int(task_count)
    except (TypeError, ValueError):
        return False
    if total <= 0 or done < 0:
        return False
    if done > total:
        done = total
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] [progress] {phase_name} {done}/{total}\n",
    )
    return True


# Phase 5 §5.1(D) — live per-gate log lines feeding the frontend Build Console's
# inline gate strip ("37 of 37 checks passing," warnings amber). Grammar
# (matches the roadmap contract + the frontend parser):
#
#   [<iso>] [gate] <phase> <gate_id> <pass|fail> <severity>
#   [<iso>] [gate] <phase> __summary__ <passed>/<total>
#
# HONEST CEILING — per-id-pass-detail limitation:
#   * SINGLE-PHASE runs (``_run_single_phase``) DO carry real per-gate detail:
#     ``execute_phase`` returns a ``gate_results`` list of per-gate dicts
#     (``gate_id`` + ``passed`` + ``issues[]``), and the phase's gate CONFIGS
#     (``phase.validation_gates``) carry the REAL declared ``severity`` per
#     gate_id. So there we emit a real per-id line for EVERY resolved gate
#     (pass and fail) plus an exact ``__summary__``.
#   * FULL-PIPELINE runs (``_drive_pipeline``) do NOT: the orchestrator's
#     ``phase_results`` rollup carries ONLY ``{task_count, completed, failed,
#     gates_passed}`` per phase — the per-gate ``_gate_results`` chain is
#     STRIPPED from the ``run_workflow`` return payload (underscore-prefixed
#     keys removed in ``WorkflowRunner.run_workflow``). So GUI-side we have NO
#     per-id detail there, not even for failures. The only honest per-phase
#     gate signal is the ``gates_passed`` boolean + the phase's TOTAL gate
#     count (from config). Therefore the full-pipeline path emits ONLY the
#     ``__summary__ <total>/<total>`` line, and ONLY when ``gates_passed`` is
#     True (every gate genuinely passed). When ``gates_passed`` is False the
#     individual passed count is UNKNOWN GUI-side, so we emit nothing rather
#     than fabricate an ``N/M``. Per-id full-pipeline lines require the
#     DEFERRED Tier-2 orchestrator stamp (roadmap §5.2) — the
#     ``MCP/core/executor.py`` / ``workflow_runner.py`` edit this GUI-side
#     backend is explicitly forbidden from making.

# Recognized gate severities, in descending order of seriousness. Used to pick
# the most-serious issue severity as a fallback when a gate config severity is
# unavailable (single-phase passing gates with no config map entry).
_GATE_SEVERITY_ORDER: Tuple[str, ...] = ("critical", "warning", "info")


def _gate_severity_for(
    gate: Any, gate_id: str, config_severity: Dict[str, str]
) -> str:
    """Resolve the REAL severity for one resolved gate (never fabricate).

    Resolution (most → least authoritative):
      1. The gate CONFIG's declared ``severity`` for this ``gate_id``
         (``config_severity`` map built from ``phase.validation_gates``) — the
         canonical ``config/workflows.yaml`` severity.
      2. The most-serious severity among the gate result's own ``issues`` — a
         real signal emitted by the validator when (1) is absent.
      3. ``"warning"`` — the conservative default the GUI already uses
         (mirrors ``failed_gate_digest``) when neither real signal exists.
    """
    declared = config_severity.get(gate_id)
    if declared:
        return str(declared)

    if hasattr(gate, "issues"):
        issues = getattr(gate, "issues", None) or []
    elif isinstance(gate, dict):
        issues = gate.get("issues") or []
    else:
        issues = []
    seen = set()
    for issue in issues:
        sev = (
            issue.get("severity")
            if isinstance(issue, dict)
            else getattr(issue, "severity", None)
        )
        if sev:
            seen.add(str(sev))
    for candidate in _GATE_SEVERITY_ORDER:
        if candidate in seen:
            return candidate
    return "warning"


def _emit_gate_lines(
    run_id: str,
    phase_name: str,
    gate_results: Any,
    *,
    config_severity: Optional[Dict[str, str]] = None,
) -> int:
    """Emit ``[gate]`` log lines for one phase from REAL per-gate detail.

    For each gate in ``gate_results`` (a list of per-gate dicts / ``GateResult``
    objects from ``execute_phase``) carrying a real ``gate_id``, append::

        [<iso>] [gate] <phase> <gate_id> <pass|fail> <severity>

    then a single truthful aggregate::

        [<iso>] [gate] <phase> __summary__ <passed>/<total>

    Severity comes from the phase's gate CONFIG (``config_severity``,
    ``gate_id -> declared severity``) with a real-issue-severity fallback — see
    :func:`_gate_severity_for`. Returns the number of per-gate lines emitted.

    NEVER fabricates: a gate with no real ``gate_id`` is skipped (it carries no
    per-id signal — e.g. the coarse rollup row). When NO gate has a real
    ``gate_id``, emits nothing at all (no ``__summary__`` over zero real gates).
    """
    if not isinstance(gate_results, list) or not gate_results:
        return 0
    config_severity = config_severity or {}

    emitted = 0
    passed_count = 0
    lines: List[str] = []
    for gate in gate_results:
        if hasattr(gate, "gate_id"):
            gate_id = getattr(gate, "gate_id", "") or ""
            passed = bool(getattr(gate, "passed", True))
        elif isinstance(gate, dict):
            gate_id = gate.get("gate_id", "") or ""
            passed = bool(gate.get("passed", True))
        else:
            continue
        if not gate_id:
            # No real per-id signal — skip rather than emit a blank gate_id.
            continue
        severity = _gate_severity_for(gate, gate_id, config_severity)
        verdict = "pass" if passed else "fail"
        lines.append(
            f"[{shared_state.now_iso()}] [gate] {phase_name} "
            f"{gate_id} {verdict} {severity}\n"
        )
        emitted += 1
        if passed:
            passed_count += 1

    if emitted == 0:
        return 0

    for line in lines:
        shared_state.append_log(run_id, line)
    # The truthful aggregate over the gates that DID carry per-id detail.
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] [gate] {phase_name} "
        f"__summary__ {passed_count}/{emitted}\n",
    )
    return emitted


def _emit_pipeline_gate_summary(
    run_id: str, phase_name: str, info: Dict[str, Any], total_gates: int
) -> bool:
    """Emit the full-pipeline ``__summary__`` line from the rollup (real data only).

    The orchestrator's ``phase_results`` rollup carries no per-gate detail — see
    the per-id-pass-detail limitation note above ``_emit_gate_lines``. The only
    honest signal here is ``info["gates_passed"]`` + ``total_gates`` (the phase's
    declared gate count from config). We therefore emit::

        [<iso>] [gate] <phase> __summary__ <total>/<total>

    ONLY when every gate genuinely passed (``gates_passed is True``) and the
    phase declares ≥1 gate. When ``gates_passed`` is False the individual passed
    count is UNKNOWN GUI-side, so we emit nothing rather than fabricate an
    ``N/M``. Returns ``True`` when a line was emitted.
    """
    if total_gates <= 0:
        return False
    if info.get("gates_passed") is not True:
        # Passed count unknown on a fail — refuse to invent it.
        return False
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] [gate] {phase_name} "
        f"__summary__ {total_gates}/{total_gates}\n",
    )
    return True


def _phase_gate_counts(workflow: str) -> Dict[str, int]:
    """Return ``{phase_name: declared_gate_count}`` for ``workflow`` from config.

    Read from ``config/workflows.yaml`` (no MCP import) so the full-pipeline
    path can size the ``__summary__`` denominator from the REAL declared gate
    count. Courseforge stage aliases run against ``textbook_to_course``. Cached.
    Best-effort: any read/parse failure returns an empty map (the summary line
    is then suppressed for every phase — never fabricate a count).
    """
    name = "textbook_to_course" if workflow in COURSEFORGE_STAGE_SUBCOMMANDS else workflow
    cached = _PHASE_GATE_COUNT_CACHE.get(name)
    if cached is not None:
        return cached
    counts: Dict[str, int] = {}
    try:
        import yaml  # noqa: PLC0415
        from lib.paths import PROJECT_ROOT  # noqa: PLC0415

        cfg = yaml.safe_load((Path(PROJECT_ROOT) / "config" / "workflows.yaml").read_text())
        wf = (cfg or {}).get("workflows", {}).get(name)
        if isinstance(wf, dict):
            for ph in wf.get("phases", []) or []:
                if isinstance(ph, dict) and ph.get("name"):
                    gates = ph.get("validation_gates") or []
                    counts[ph.get("name")] = len(gates) if isinstance(gates, list) else 0
    except Exception:  # noqa: BLE001 — count lookup is best-effort
        logger.debug("phase gate-count lookup failed for %s", name, exc_info=True)
    _PHASE_GATE_COUNT_CACHE[name] = counts
    return counts


_PHASE_GATE_COUNT_CACHE: Dict[str, Dict[str, int]] = {}


async def _poll_phase_progress(run_id: str, workflow_id: str) -> None:
    """Append friendly per-phase progress lines as the workflow advances.

    Watches ``state/workflows/<workflow_id>.json``'s ``phase_outputs`` — the
    WorkflowRunner stamps ``_completed`` (and ``_skipped``) on each phase's
    output dict as it finishes a phase and re-saves the state file. We poll that
    file and, when a phase newly reaches a completed/skipped marker, append a
    ``[phase] <name> done|skipped`` line to the run log so the WS stream carries
    a per-phase signal mid-run (the orchestrator itself only logs a summary at
    the very end). Pure observation: never mutates the workflow state, never
    affects the run outcome; cancelled by ``_drive_pipeline`` when the run ends.
    """
    from lib.paths import STATE_PATH  # noqa: PLC0415

    state_file = Path(STATE_PATH) / "workflows" / f"{workflow_id}.json"
    seen: Dict[str, str] = {}
    try:
        while True:
            await asyncio.sleep(1.0)
            try:
                raw = state_file.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                continue
            try:
                state = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                continue  # mid-write; try again next tick
            phase_outputs = state.get("phase_outputs") if isinstance(state, dict) else None
            if not isinstance(phase_outputs, dict):
                continue
            for name, out in phase_outputs.items():
                if not isinstance(out, dict) or not out.get("_completed"):
                    continue
                marker = "skipped" if out.get("_skipped") else "done"
                if seen.get(name) == marker:
                    continue
                seen[name] = marker
                label = PHASE_LABELS.get(name, name)
                shared_state.append_log(
                    run_id,
                    f"[{shared_state.now_iso()}] [phase] {name} {marker} — {label}\n",
                )
    except asyncio.CancelledError:
        return
    except Exception:  # noqa: BLE001 — progress polling is best-effort
        logger.debug("phase-progress poll stopped for %s", run_id, exc_info=True)


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

    HONEST two-stage cancel (Phase 4 §5.1(E)): only the TERMINAL ``cancelled``
    blocks a write. The intermediate ``cancel_requested`` status (written by
    ``cancel_run`` the instant a cancel is asked for) is deliberately NON-terminal
    here, so the driver can still write the authoritative terminal status — the
    cooperative ``cancelled`` from the ``CancelledError`` handler, or the natural
    ``completed``/``failed`` if the orchestrator raced to completion before the
    cancel landed.

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
        _apply_authoring_route_env(req)
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


def _normalize_csv_param(
    workflow_params: Dict[str, Any], src_key: str, dst_key: str
) -> None:
    """Pop a CSV/list ``src_key`` option into a de-duplicated ``dst_key`` list.

    Shared normalizer for the failure panel's scope options. Accepts a CSV
    string or a list/tuple; writes the de-duplicated (order-preserving) list
    under ``dst_key``. Mutates ``workflow_params`` in place. No/empty source
    → nothing added (byte-identical). An explicit ``dst_key`` already present
    wins (never overwritten).
    """
    raw = workflow_params.pop(src_key, None)
    if dst_key in workflow_params:
        return
    tokens: List[str]
    if isinstance(raw, str):
        tokens = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        tokens = [str(t) for t in raw]
    else:
        return
    seen: set = set()
    out: List[str] = []
    for tok in (t.strip() for t in tokens):
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    if out:
        workflow_params[dst_key] = out


def _page_covers(page: str, token: str) -> bool:
    """True when ``token`` names ``page`` exactly or as its module prefix.

    MIRRORS ``MCP.tools.pipeline_tools._page_membership_match`` — the SOURCE OF
    TRUTH for the rewrite-phase page eviction. Reproduced here (not imported) so
    ``run_service`` keeps its import-light contract (no MCP import at module
    load — see the module docstring). A parity test
    (``test_page_covers_matches_pipeline_tools``) imports BOTH and asserts
    identical verdicts over a table of cases (``week_01`` vs ``week_01_overview``
    vs ``week_010``, exact match, no match) so the two never drift.

    A ``--pages`` token covers a page when it equals the page id (single-page
    scope) OR is an underscore-bounded PREFIX of it (module scope — e.g.
    ``week_01`` covers ``week_01_overview`` but never ``week_010``).
    """
    if not page or not token:
        return False
    return page == token or page.startswith(token + "_")


def _drop_page_covered_block_ids(workflow_params: Dict[str, Any]) -> None:
    """Subsumption dedup: drop block-instance ids already covered by a page token.

    When a selected ``pages`` token (``target_page_ids``) covers the page of a
    selected block-instance id (``target_block_instance_ids``), the page
    eviction already re-authors that block, so the redundant instance id is
    dropped. A block-instance id is shaped ``{page_id}#{block_type}_{slug}_{idx}``
    (:func:`MCP.tools.pipeline_tools._page_id_for`); its page is the substring
    BEFORE the ``#`` separator. Page coverage uses the SAME rule as the rewrite
    eviction (:func:`_page_covers` ⇔ ``_page_membership_match``).

    Defensive: a block-instance id with no ``#`` separator has no resolvable
    page, so it is NEVER dropped (passes through). If every id is subsumed the
    param key is removed entirely (no empty-list param emitted). Runs after both
    lists are normalized; a no-op when either list is absent.
    """
    block_ids = workflow_params.get("target_block_instance_ids")
    pages = workflow_params.get("target_page_ids")
    if not block_ids or not pages:
        return
    kept: List[str] = []
    for bid in block_ids:
        page = bid.split("#", 1)[0] if "#" in bid else None
        if page is not None and any(_page_covers(page, tok) for tok in pages):
            continue  # page eviction already covers this instance — subsumed
        kept.append(bid)
    if kept:
        workflow_params["target_block_instance_ids"] = kept
    else:
        workflow_params.pop("target_block_instance_ids", None)


def _normalize_blocks_param(workflow_params: Dict[str, Any]) -> None:
    """Normalize failure-panel scope options into rewrite-eviction params.

    Q1 (--blocks) + I4 stage 2 (--block-ids / --pages). The studio failure
    panel posts scope options as phase options; the rewrite-phase routing
    consumes list params:

    - ``blocks`` (block TYPES) → ``target_block_ids`` (stage-1 type eviction);
    - ``block_ids`` (exact block-instance IDs) →
      ``target_block_instance_ids`` (I4 stage-2 instance eviction);
    - ``pages`` (page/module ids) → ``target_page_ids`` (I4 stage-2 page
      eviction).

    Each source accepts BOTH a CSV string (API callers) and a JSON array of
    strings (what the per-page/per-block picker posts) — see
    :func:`_normalize_csv_param`. Each is trimmed, empties dropped, de-duplicated
    (order-preserving) and additive; an unknown id/page fails LOUD in the
    rewrite handler (validated against the real outline block set), never a
    silent no-op. No/empty → nothing added (byte-identical). An explicit
    destination param already present wins.

    Finally a subsumption pass (:func:`_drop_page_covered_block_ids`) drops any
    block-instance id whose page a selected ``pages`` token already covers, so
    the picker can post overlapping selections without double-scoping.
    """
    _normalize_csv_param(workflow_params, "blocks", "target_block_ids")
    _normalize_csv_param(
        workflow_params, "block_ids", "target_block_instance_ids"
    )
    _normalize_csv_param(workflow_params, "pages", "target_page_ids")
    _drop_page_covered_block_ids(workflow_params)


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

    # Q1 (--blocks) — normalize the failure panel's ``options.blocks`` into
    # the ``target_block_ids`` param the rewrite-phase routing consumes
    # (additive type-eviction over the reuse cache). Absent/empty → no key
    # added → byte-identical behavior.
    _normalize_blocks_param(workflow_params)

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
    # Phase 5 §5.1(D) — emit per-gate ``[gate]`` lines + a truthful summary from
    # the REAL ``gate_results`` the executor returned (this single-phase path
    # DOES have per-id detail, unlike the full pipeline). Severity is the gate
    # CONFIG's declared severity (``gate_configs``) per gate_id, with a
    # real-issue-severity fallback. A phase with no gate data emits no line.
    _config_severity: Dict[str, str] = {}
    for _gc in gate_configs or []:
        if isinstance(_gc, dict) and _gc.get("gate_id"):
            _config_severity[str(_gc["gate_id"])] = str(_gc.get("severity") or "")
        elif getattr(_gc, "gate_id", None):
            _config_severity[str(_gc.gate_id)] = str(getattr(_gc, "severity", "") or "")
    _emit_gate_lines(
        run_id, phase_name, gate_results, config_severity=_config_severity
    )
    # Phase-3 (§5.1(B)) — emit the REAL end-of-phase
    # ``[<iso>] [progress] <phase> <done>/<total>`` ring signal for the
    # single-phase pathway. ``len(tasks)`` is the genuine task count and the
    # COMPLETE results are the genuine done count. Suppressed for validator-only
    # (``agents: []``) phases whose synthesized virtual task is not a meaningful
    # count (ring stays indeterminate). No mid-phase incremental signal exists
    # GUI-side (see _emit_phase_progress_line's HONEST GAP note).
    if getattr(phase, "agents", None) != []:
        completed_tasks = sum(
            1 for r in results.values() if getattr(r, "status", None) == "COMPLETE"
        )
        _emit_phase_progress_line(run_id, phase_name, completed_tasks, len(tasks))
    # A6: on a single-phase gate failure, emit a structured failure line +
    # persist the failed phase/reason so the GUI failure panel is consistent
    # with the full-pipeline path.
    update_patch: Dict[str, Any] = {
        "status": status,
        "tasks": task_summaries,
        "gate_results": gate_results,
        "gates_passed": bool(gates_passed),
        "finished_at": shared_state.now_iso(),
    }
    if status == "failed":
        digest = failed_gate_digest(gate_results)
        reason = digest[0]["message"] if digest else "failed validation gates"
        label = PHASE_LABELS.get(phase_name, phase_name)
        shared_state.append_log(
            run_id,
            f"[{shared_state.now_iso()}] [phase] {phase_name} failed "
            f"— {label}: {reason}\n",
        )
        update_patch["failed_phase"] = phase_name
        update_patch["failure_reason"] = reason
    _safe_update(run_id, update_patch)
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


def _infer_failed_phase(
    phase_results: Any, reason_fallback: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort: pick the failing phase from a ``phase_results`` rollup.

    Used only when the runner did not surface ``failed_phase`` directly (older
    payloads / a crash before the loop break recorded it). Walks the
    ``phase_results`` dict in insertion order and returns the FIRST phase that
    reports ``gates_passed is False`` or ``failed > 0`` — the workflow is
    sequential, so the first such phase is where it stopped. Returns
    ``(None, reason_fallback)`` when nothing structured is found.
    """
    if isinstance(phase_results, dict):
        for name, info in phase_results.items():
            if not isinstance(info, dict):
                continue
            if info.get("gates_passed") is False:
                return name, reason_fallback or "failed validation gates"
            if info.get("failed"):
                return name, reason_fallback or (
                    f"{info.get('failed')} task(s) failed"
                )
    return None, reason_fallback


def failed_gate_digest(gate_results: Any) -> List[Dict[str, Any]]:
    """Reduce a ``gate_results`` chain to a digestible failed-gate list.

    ``gate_results`` may be:
    - the per-phase ``phase_results`` rollup (``{phase: {gates_passed,...}}``),
      which carries no per-gate detail — skipped here, or
    - a list of per-gate result dicts / objects (``gate_id``, ``severity``,
      ``passed``, ``issues``), as the single-phase pathway persists in
      ``run_record["gate_results"]``.

    Returns one row per FAILED gate: ``{phase, gate_id, severity, message,
    issues_count}``. ``phase`` is ``None`` for a flat list (the caller knows the
    phase). Non-failing gates are dropped.
    """
    rows: List[Dict[str, Any]] = []

    def _one(gr: Any, phase: Optional[str]) -> Optional[Dict[str, Any]]:
        if hasattr(gr, "gate_id"):
            passed = getattr(gr, "passed", True)
            gate_id = getattr(gr, "gate_id", "") or ""
            severity = getattr(gr, "severity", "warning") or "warning"
            issues = list(getattr(gr, "issues", None) or [])
        elif isinstance(gr, dict):
            passed = gr.get("passed", True)
            gate_id = gr.get("gate_id", "") or ""
            severity = gr.get("severity", "warning") or "warning"
            issues = list(gr.get("issues") or [])
        else:
            return None
        if passed:
            return None
        return {
            "phase": phase,
            "gate_id": gate_id,
            "severity": severity,
            "message": str(issues[0]) if issues else "validation gate failed",
            "issues_count": len(issues),
        }

    if isinstance(gate_results, list):
        for gr in gate_results:
            row = _one(gr, None)
            if row:
                rows.append(row)
    elif isinstance(gate_results, dict):
        # A phase_results rollup: each value may itself carry a per-gate list
        # under a conventional key, but the rollup we persist only has the
        # boolean; surface the failing phases as coarse rows.
        for phase, info in gate_results.items():
            if not isinstance(info, dict):
                continue
            sub = info.get("gate_results") or info.get("_gate_results")
            if isinstance(sub, list):
                for gr in sub:
                    row = _one(gr, phase)
                    if row:
                        rows.append(row)
            elif info.get("gates_passed") is False:
                rows.append(
                    {
                        "phase": phase,
                        "gate_id": "",
                        "severity": "critical",
                        "message": "phase failed validation gates",
                        "issues_count": 0,
                    }
                )
    return rows


def locate_validation_report(record: Dict[str, Any]) -> Optional[Path]:
    """Resolve the ``courseforge_validation_report.json`` for a run record.

    Resolution: the run's ``params.project_id`` (or newest
    ``PROJ-<course_name>-*`` export) -> ``<export_dir>/courseforge_validation_report.json``.
    Courses discovered dynamically; no hardcoded paths. Returns ``None`` when no
    export dir is found or the report file is absent.
    """
    params = record.get("params") if isinstance(record, dict) else None
    project_id = None
    if isinstance(params, dict):
        project_id = params.get("project_id")
    course_name = record.get("course_name") if isinstance(record, dict) else None
    export_dir = _resolve_project_export(project_id, course_name)
    if export_dir is None:
        return None
    report = export_dir / "courseforge_validation_report.json"
    return report if report.is_file() else None


def validation_report(run_id: str) -> Dict[str, Any]:
    """Return the validation-report payload for a run (A6 endpoint backing).

    Shape on success::

        {
          "run_id": ...,
          "report": {<courseforge_validation_report.json>} | null,
          "report_path": "<abs path>" | null,
          "failed_gates": [ {phase, gate_id, severity, message, issues_count} ],
          "failed_phase": ... | null,
          "failure_reason": ... | null,
        }

    ``report`` is ``None`` (with an explanatory ``note``) when no
    ``courseforge_validation_report.json`` exists for the run. ``failed_gates``
    is always populated from the run record's persisted ``gate_results`` so the
    operator sees the failing-gate digest even when no aggregator file exists.
    Raises ``KeyError`` when the run is unknown (router maps to 404).
    """
    record = shared_state.read_run(run_id)
    if record is None:
        raise KeyError(run_id)

    out: Dict[str, Any] = {
        "run_id": run_id,
        "report": None,
        "report_path": None,
        "failed_gates": failed_gate_digest(record.get("gate_results")),
        "failed_phase": record.get("failed_phase"),
        "failure_reason": record.get("failure_reason"),
    }
    report_path = locate_validation_report(record)
    if report_path is None:
        out["note"] = (
            "no courseforge_validation_report.json found for this run "
            "(only present after a two-pass Courseforge slice with a project "
            "export); the failed-gate digest is still available"
        )
        return out
    try:
        out["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        out["report_path"] = str(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        out["note"] = f"validation report present but unreadable: {exc}"
    return out


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


# Matches the per-phase log lines appended by ``_poll_phase_progress`` and the
# failure path in ``_drive_pipeline``:
#   ``[<iso>] [phase] <name> done — <label>``
#   ``[<iso>] [phase] <name> skipped — <label>``
#   ``[<iso>] [phase] <name> failed — <label>: <reason>``
# Capture group 1 = the bracketed ISO prefix, 2 = the phase name, 3 = the state.
_PHASE_LINE_RE = re.compile(
    r"^\[(?P<iso>[^\]]+)\]\s+\[phase\]\s+(?P<name>\S+)\s+(?P<state>done|skipped|failed)\b"
)


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (the ``now_iso()`` form) → aware ``datetime``.

    Returns ``None`` for anything unparseable so the timeline derivation never
    raises on a malformed / truncated log line. Mirrors ``now_iso()``'s
    ``datetime.now(timezone.utc).isoformat()`` output (offset-aware).
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return parsed


def derive_phase_timeline(run_id: str) -> Dict[str, Any]:
    """Derive per-phase timing for a run from its log file (Tier-1, GUI-side).

    The roadmap's "invisible keystone signal": every per-phase completion the
    backend ALREADY logs carries an ISO timestamp prefix
    (``[<iso>] [phase] <name> done|skipped|failed — ...``). We parse those lines
    and reconstruct a sequential per-phase duration vector with ZERO orchestrator
    change. Sequential duration of a phase = ``t(this completion) − t(previous
    completion)``; the first phase's start is the run record's ``started_at``.

    Robustness contract (never raises):
    - Unknown run / absent log / empty log → empty ``timeline`` (``total_ms`` 0).
    - An unparseable or missing timestamp on a line → that phase's
      ``duration_ms`` is ``null`` (and it can't anchor the next phase's start).
    - ``started_at`` missing/garbage → the first phase's ``duration_ms`` is
      ``null`` (no anchor), but later phases still time off each other.

    Returns::

        {
          "run_id": <run_id>,
          "timeline": [
            {"phase": <name>, "state": "done|skipped|failed",
             "completed_at": <iso|None>, "duration_ms": <int|None>},
            ...
          ],
          "total_ms": <int>,   # sum of the non-null per-phase durations
        }
    """
    record = shared_state.read_run(run_id)
    timeline: List[Dict[str, Any]] = []

    # The first phase's "start" anchor is the run's started_at (if parseable).
    prev_dt = _parse_iso(record.get("started_at")) if isinstance(record, dict) else None

    log_text = ""
    try:
        path = shared_state.log_path(run_id)
        if path.exists():
            log_text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        log_text = ""

    for line in log_text.splitlines():
        match = _PHASE_LINE_RE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        state = match.group("state")
        iso = match.group("iso")
        this_dt = _parse_iso(iso)

        duration_ms: Optional[int] = None
        if this_dt is not None and prev_dt is not None:
            delta_ms = int((this_dt - prev_dt).total_seconds() * 1000)
            # Guard against clock skew / out-of-order lines: never emit a
            # negative duration (degrade to null instead of a misleading value).
            duration_ms = delta_ms if delta_ms >= 0 else None

        timeline.append(
            {
                "phase": name,
                "state": state,
                "completed_at": iso if this_dt is not None else None,
                "duration_ms": duration_ms,
            }
        )

        # Anchor the next phase off this completion only when it had a valid
        # timestamp; an unparseable line cannot anchor the following phase.
        if this_dt is not None:
            prev_dt = this_dt

    total_ms = sum(
        entry["duration_ms"] for entry in timeline if isinstance(entry["duration_ms"], int)
    )
    return {"run_id": run_id, "timeline": timeline, "total_ms": total_ms}


# Phase 4 (§5.1(C)) — static per-phase ETA priors (milliseconds) for the
# COLD-START case (< 2 historical runs of a workflow). Keyed by the internal
# phase name (the same keys as ``PHASE_LABELS``). These are deliberately ROUGH
# order-of-magnitude estimates — the roadmap's ETA discipline (§5.1(C), §11) is
# emphatic that with < 2 real runs the UI labels the number a "rough estimate"
# and renders a RANGE, never a zeroing countdown. The honest fallback exists so
# the first build still shows an emotional "about Nm left," not a blank.
# Real history (>= 2 runs) always supersedes these per phase.
_PHASE_DURATION_PRIOR: Dict[str, int] = {
    # Task #19 Stage 3d renamed dart_conversion -> semantik_conversion; keep
    # the legacy key so old workflow records still get a prior estimate.
    "semantik_conversion": 600_000,       # OCR/synthesis over a full PDF — minutes
    "dart_conversion": 600_000,           # legacy alias (old runs)
    "staging": 5_000,
    "chunking": 30_000,
    "objective_extraction": 60_000,
    "source_mapping": 45_000,
    "course_planning": 300_000,           # 7B objective synthesis — minutes
    "concept_extraction": 120_000,
    "content_generation": 900_000,        # the dominant phase (1800s on record)
    "content_generation_outline": 600_000,
    "inter_tier_validation": 60_000,
    "content_generation_rewrite": 600_000,
    "post_rewrite_validation": 60_000,
    "packaging": 30_000,
    "imscc_chunking": 30_000,
    "trainforge_assessment": 300_000,
    "training_synthesis": 600_000,
    "libv2_archival": 30_000,
    "vector_indexing": 120_000,
    "finalization": 15_000,
}


def _median_ms(values: List[int]) -> Optional[int]:
    """Integer median of a non-empty list of durations (``None`` if empty)."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return int(ordered[mid])
    # Even count: average the two central samples, rounded to an int ms.
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def phase_duration_medians(workflow: str) -> Dict[str, Any]:
    """Median per-phase ``duration_ms`` over COMPLETED runs of ``workflow``.

    Phase 4 §5.1(C) ETA history. Scans the run registry (:func:`list_runs`) for
    COMPLETED runs of ``workflow`` carrying a persisted ``phase_durations`` vector
    (the list of ``{phase, state, completed_at, duration_ms}`` finalize writes,
    Phase 0) and returns the median real ``duration_ms`` per phase across that
    history. A phase with FEWER than 2 historical samples honestly falls back to
    the static :data:`_PHASE_DURATION_PRIOR` and is marked accordingly (``n`` is
    always the count of REAL samples that informed the phase, even when the prior
    was used because ``n < 2``).

    Courseforge stage aliases resolve through the ``textbook_to_course`` machine,
    so their history is read under that workflow name (mirroring
    :func:`_validator_only_phases`).

    Return shape (the frontend depends on it EXACTLY)::

        {
          "workflow": <name>,
          "durations": {
            "<phase>": {"median_ms": <int|null>, "n": <int>},
            ...
          },
          "source": "history" | "prior" | "mixed",
        }

    ``source`` semantics:
      * ``"history"`` — every phase in ``durations`` was informed by >= 2 real runs.
      * ``"prior"``   — no phase had >= 2 real runs; the static prior carried all.
      * ``"mixed"``   — some phases had history, others fell back to the prior.

    A phase that has neither >= 2 samples nor a static prior gets
    ``{"median_ms": null, "n": <real sample count>}`` (honest null, no fabrication).
    """
    name = "textbook_to_course" if workflow in COURSEFORGE_STAGE_SUBCOMMANDS else workflow

    # Collect each phase's real per-run durations from COMPLETED history.
    samples: Dict[str, List[int]] = {}
    try:
        runs = list_runs()
    except Exception:  # noqa: BLE001 — registry read is best-effort
        logger.warning("phase_duration_medians: list_runs failed", exc_info=True)
        runs = []
    for rec in runs:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "completed":
            continue
        rec_wf = rec.get("workflow")
        rec_name = "textbook_to_course" if rec_wf in COURSEFORGE_STAGE_SUBCOMMANDS else rec_wf
        if rec_name != name:
            continue
        vector = rec.get("phase_durations")
        if not isinstance(vector, list):
            continue
        for entry in vector:
            if not isinstance(entry, dict):
                continue
            phase = entry.get("phase")
            dur = entry.get("duration_ms")
            if not isinstance(phase, str) or not isinstance(dur, int) or dur < 0:
                continue
            samples.setdefault(phase, []).append(dur)

    # Union of phases we can speak to: anything with history OR a static prior.
    phases = set(samples) | set(_PHASE_DURATION_PRIOR)

    durations: Dict[str, Dict[str, Any]] = {}
    any_history = False
    any_prior = False
    for phase in phases:
        real = samples.get(phase, [])
        n = len(real)
        if n >= 2:
            durations[phase] = {"median_ms": _median_ms(real), "n": n}
            any_history = True
        else:
            # < 2 real samples → honest prior fallback (n is the real count, 0/1).
            prior = _PHASE_DURATION_PRIOR.get(phase)
            durations[phase] = {"median_ms": prior, "n": n}
            if prior is not None:
                any_prior = True

    if any_history and any_prior:
        source = "mixed"
    elif any_history:
        source = "history"
    else:
        source = "prior"

    return {"workflow": name, "durations": durations, "source": source}


def list_runs(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return GUI + CLI-launched run records, newest-first.

    GUI-registry records (``state/gui/runs/``) are tagged ``source: "gui"``.
    CLI-launched orchestrator workflows (``state/workflows/WF-*.json`` records
    with no GUI registry entry) are additionally surfaced as run-record-shaped
    dicts tagged ``source: "cli"`` (see :func:`_list_cli_workflow_runs`), so a
    live ``ed4all run`` campaign is discoverable from the Runs tab instead of
    requiring a hand-built ``#/create/<WF-...>`` URL. De-duped: a workflow id
    already referenced by a GUI record's ``workflow_id`` is never listed twice.

    ``limit`` (optional) caps the number of records returned; ``None`` (default)
    returns all of them, preserving the prior unbounded behavior for callers that
    don't pass it. A non-positive limit returns an empty list.
    """
    # Scan for live ``ed4all run`` processes ONCE per listing — every CLI
    # record's process-liveness verdict (building / stopped / stalled?) reads
    # the same snapshot (no per-record /proc walk).
    processes = liveness.scan_pipeline_processes()

    runs = shared_state.list_runs()
    for record in runs:
        record.setdefault("source", "gui")
        # GUI runs are driven in-process (no external ``ed4all run`` process),
        # so only PAUSED / stop-sentinel signals refine their status here.
        record["effective_status"] = liveness.effective_status(
            record.get("status"),
            is_cli=False,
            orch_run_id=_record_orch_run_id(record.get("params")),
            processes=processes,
        )
    gui_workflow_ids = {
        str(r.get("workflow_id")) for r in runs if r.get("workflow_id")
    }
    runs.extend(
        _list_cli_workflow_runs(
            exclude_workflow_ids=gui_workflow_ids, processes=processes
        )
    )
    # Newest-first merge across both sources. Sort on the PARSED timestamp
    # (GUI records carry tz-aware UTC ISO strings, orchestrator state files
    # naive-local ones — a plain string sort would skew across the tz offset);
    # unparsable/missing timestamps sink to the bottom.
    runs.sort(
        key=lambda r: _parse_state_timestamp(r.get("created_at")) or datetime.min,
        reverse=True,
    )
    if limit is not None:
        if limit <= 0:
            return []
        return runs[:limit]
    return runs


# ---- CLI-launched workflow surfacing (Runs tab) -------------------------
#
# DISPLAY HEURISTIC (not a lifecycle truth-source): ``state/workflows/``
# accumulates stale RUNNING records from old crashed runs whose processes are
# long dead — the orchestrator has no reaper stamping them terminal. Dumping
# them all into the Runs tab would bury the live build under dozens of
# zombie "running" rows. So a CLI record is surfaced only when it is
# plausibly still relevant:
#   * non-terminal status AND updated within the last 48 h, OR
#   * terminal status AND updated within the last 7 days.
# This filters the LISTING only — it never mutates workflow state, and every
# record (stale or not) stays reachable via ``#/create/<WF-...>`` /
# ``GET /api/runs/<id>/progress`` directly.
_CLI_SCAN_CAP = 100  # newest state files (by mtime) considered per listing
_CLI_ACTIVE_WINDOW_HOURS = 48
_CLI_TERMINAL_WINDOW_DAYS = 7
# Lowercased terminal workflow statuses (WorkflowRunner writes COMPLETE /
# FAILED / PAUSED / RUNNING / PENDING; TIMEOUT survives from older runs).
_CLI_TERMINAL_STATUSES = {"complete", "completed", "failed", "cancelled", "timeout"}


def _parse_state_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp into a NAIVE LOCAL datetime (``None`` on garbage).

    GUI records carry tz-aware UTC ISO strings; orchestrator workflow state
    files carry naive local-time ones. Normalize both onto naive local time so
    recency windows and sort keys compare consistently.
    """
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _record_orch_run_id(params: Any) -> Optional[str]:
    """Best-effort orchestrator run id (``params.run_id``) for a run record.

    The stop-sentinel + checkpoint dirs live under ``state/runs/<run_id>/`` keyed
    by this orchestrator run id (NOT the ``WF-...`` workflow id). Tolerates a
    JSON-string params blob.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            return None
    if isinstance(params, dict):
        rid = params.get("run_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return None


def _cli_workflow_run_record(
    state: Dict[str, Any],
    mtime: float,
    *,
    processes: Optional[List[Tuple[int, List[str]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Map one orchestrator workflow-state dict onto the Runs-tab record shape.

    Returns ``None`` when the record fails the display heuristic above (stale)
    or carries no usable id. Honest data only: fields the workflow state does
    not carry (mode/provider/model/params/gate_results) are ``None`` — never
    invented. ``params`` is deliberately ``None`` (not the workflow's params
    dict) so the frontend's "Run again" affordance — which re-POSTs GUI-shaped
    params — never fires with the orchestrator's incompatible param shape.
    """
    workflow_id = str(state.get("id") or "").strip()
    if not workflow_id:
        return None

    status = str(state.get("status") or "").strip().lower()
    if status == "complete":
        status = "completed"  # frontend terminal-set vocabulary

    updated = _parse_state_timestamp(state.get("updated_at"))
    if updated is None:
        updated = datetime.fromtimestamp(mtime)
    age = datetime.now() - updated
    if status in _CLI_TERMINAL_STATUSES:
        if age > timedelta(days=_CLI_TERMINAL_WINDOW_DAYS):
            return None
    elif age > timedelta(hours=_CLI_ACTIVE_WINDOW_HOURS):
        return None

    params = state.get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if not isinstance(params, dict):
        params = {}

    # Honest process-liveness verdict for a CLI-launched run (see liveness.py):
    # a RUNNING record whose ``ed4all run`` process is gone → "incomplete", one
    # with a stop sentinel → "stopping", etc. Uses the shared per-listing scan.
    orch_run_id = _record_orch_run_id(params)
    eff_status = liveness.effective_status(
        status,
        is_cli=True,
        orch_run_id=orch_run_id,
        attribution_tokens=liveness.attribution_tokens_from_params(
            params, orch_run_id, wf_id=workflow_id
        ),
        processes=processes,
        wf_mtime=mtime,
    )

    return {
        "run_id": workflow_id,
        "kind": "pipeline",
        "workflow": state.get("type"),
        "workflow_id": workflow_id,
        "course_name": params.get("course_name"),
        "phase": None,
        "mode": None,
        "provider": None,
        "model": None,
        "status": status or "unknown",
        "effective_status": eff_status,
        "params": None,
        "gate_results": None,
        "tasks": None,
        "error": state.get("failure_reason") or None,
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "started_at": state.get("started_at"),
        "finished_at": (
            state.get("completed_at") if status in _CLI_TERMINAL_STATUSES else None
        ),
        "source": "cli",
    }


def _list_cli_workflow_runs(
    *,
    exclude_workflow_ids: set,
    processes: Optional[List[Tuple[int, List[str]]]] = None,
) -> List[Dict[str, Any]]:
    """Return CLI-launched workflow runs as Runs-tab records (best-effort).

    Reads ``<state>/workflows/WF-*.json`` bounded to the newest
    ``_CLI_SCAN_CAP`` files by mtime. Skips: ids in ``exclude_workflow_ids``
    (already represented by a GUI registry record — the de-dupe contract),
    mid-write/corrupt/non-dict JSON, and records failing the staleness display
    heuristic (see the comment block above). Any directory-level failure
    returns ``[]`` — the GUI listing must never break on orchestrator state.
    """
    workflows_dir = _workflow_state_file("WF-probe").parent
    try:
        candidates = [
            p
            for p in workflows_dir.iterdir()
            if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
        ]
    except (FileNotFoundError, OSError):
        return []

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for path in candidates[:_CLI_SCAN_CAP]:
        if path.stem in exclude_workflow_ids:
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue  # mid-write / corrupt — tolerate and skip
        if not isinstance(state, dict):
            continue
        if str(state.get("id") or path.stem) in exclude_workflow_ids:
            continue
        record = _cli_workflow_run_record(state, _mtime(path), processes=processes)
        if record is not None:
            out.append(record)
    return out


# Non-terminal states a fresh process should reconcile on boot. ``queued`` and
# ``running`` (pipeline) plus phase runs that register straight as ``running``.
_NON_TERMINAL_STATES = ("queued", "running")

# Marketable-v1 D3 durable-runs cap: one automatic resume per run. A run that is
# re-orphaned AFTER an auto-resume (i.e. the resumed process also died, or the
# resume itself crashed) is marked failed instead of resumed again — this bounds
# crash loops where a poisoned run would otherwise be resumed on every boot. The
# attempt count is persisted on the run record under ``resume_attempts``.
_MAX_AUTO_RESUME_ATTEMPTS = 1


def _workflow_state_file(workflow_id: str) -> Path:
    """Resolve ``<state>/workflows/<workflow_id>.json`` for the active state root.

    Mirrors ``shared_state._state_root``: when ``ED4ALL_STATE_RUNS_DIR`` is set
    (tests / a redirected deployment), the ``workflows/`` dir is a sibling of the
    named ``runs/`` dir under the same state root; otherwise it falls back to
    ``lib.paths.STATE_PATH`` (which the WorkflowRunner writes to). Keeping this
    in lock-step with where the runner persists state is what makes the
    checkpoint detectable on boot.
    """
    env_runs = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_runs:
        return Path(env_runs).parent / "workflows" / f"{workflow_id}.json"
    from lib.paths import STATE_PATH  # noqa: PLC0415

    return Path(STATE_PATH) / "workflows" / f"{workflow_id}.json"


def _load_workflow_state(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Return the parsed workflow-state dict for ``workflow_id`` (or ``None``).

    Best-effort: a missing / unreadable / corrupt / non-dict state file returns
    ``None`` so callers never raise on a mid-write or absent checkpoint.
    """
    try:
        state = json.loads(_workflow_state_file(workflow_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _read_paused_after(workflow_id: str) -> Optional[str]:
    """Return the phase a paused run halted at (``None`` when not derivable).

    The runner persists ``stopped_after`` for a clean ``--stop-after`` halt and
    ``paused_phase`` for a graceful mid-phase stop. Prefer the ``--stop-after``
    marker (the review-flow halt point); fall back to the graceful-stop marker.
    """
    state = _load_workflow_state(workflow_id)
    if state is None:
        return None
    marker = state.get("stopped_after") or state.get("paused_phase")
    return str(marker) if marker else None


def _clear_stop_after(workflow_id: str) -> bool:
    """Strip the persisted ``--stop-after`` marker from a workflow-state file.

    The runner reads ``params.stop_after`` from the persisted workflow state, so
    a PLAIN resume of a ``--stop-after``-paused run would immediately re-pause at
    the same phase. The I1 review-flow "Resume build" action continues the build
    PAST the review checkpoint, so we remove ``params.stop_after`` (and the
    ``stopped_after`` halt marker) before the re-drive. Mirrors the CLI's
    ``_apply_resume_stop_after_override`` write path.

    Returns ``True`` when the file was changed (a stop marker existed and was
    removed), ``False`` otherwise (nothing to clear / unreadable / unwritable).
    Best-effort: a missing / unreadable / unwritable state file returns
    ``False`` and the persisted value continues to govern.
    """
    path = _workflow_state_file(workflow_id)
    state = _load_workflow_state(workflow_id)
    if state is None:
        return False
    changed = False
    params = state.get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if isinstance(params, dict) and params.get("stop_after"):
        params.pop("stop_after", None)
        state["params"] = params
        changed = True
    if state.get("stopped_after"):
        state.pop("stopped_after", None)
        changed = True
    if not changed:
        return False
    try:
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except OSError:
        return False
    return True


def paused_review_info(run_id: str) -> Optional[Dict[str, Any]]:
    """Return the objectives-review context for a paused run (I1 review flow).

    Backs ``GET /api/runs/{run_id}/review``. Shape::

        {
          "run_id": ...,
          "status": "paused" | ...,
          "paused": <bool>,                 # status == "paused"
          "paused_phase": "course_planning" | null,
          "stop_after": "course_planning" | null,   # the persisted halt intent
          "course_name": ...,
          "course_id": ... | null,          # project_id for the editor (best-effort)
          "objectives_path": "<abs path>" | null,   # the file the editor writes +
                                            # a plain resume reads (best-effort)
          "objectives_available": <bool>,   # the file exists on disk
        }

    ``course_id`` / ``objectives_path`` are resolved best-effort from the
    Courseforge export via ``course_service`` — they degrade to ``None`` (never
    raise) when the export can't be resolved. Returns ``None`` for an unknown
    run (the router maps that to a 404).
    """
    record = shared_state.read_run(run_id)
    if record is None:
        return None
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    stop_after = (params or {}).get("stop_after") or record.get("stop_after")
    course_name = record.get("course_name")
    out: Dict[str, Any] = {
        "run_id": run_id,
        "status": record.get("status"),
        "paused": record.get("status") == "paused",
        "paused_phase": record.get("paused_phase"),
        "stop_after": stop_after,
        "course_name": course_name,
        "course_id": None,
        "objectives_path": None,
        "objectives_available": False,
    }
    # Best-effort: resolve the exact objectives file the editor writes (and a
    # plain resume reads in place) so the review panel can surface it.
    if course_name:
        try:
            from gui.services import course_service  # noqa: PLC0415

            resolved = course_service._resolve(str(course_name))
            export = resolved.get("export") if isinstance(resolved, dict) else None
            if export:
                out["course_id"] = (
                    export.get("project_id")
                    or export.get("course_name")
                    or course_name
                )
                obj_path = export.get("objectives_path")
                if obj_path is not None:
                    out["objectives_path"] = str(obj_path)
                    out["objectives_available"] = Path(str(obj_path)).exists()
        except Exception:  # noqa: BLE001 — objectives resolve is best-effort
            logger.debug(
                "paused_review_info: objectives resolve failed for %s",
                run_id,
                exc_info=True,
            )
    if out["course_id"] is None:
        out["course_id"] = course_name
    return out


def _resumable_workflow_id(record: Dict[str, Any]) -> Optional[str]:
    """Return the orchestrator ``workflow_id`` IFF the run has resumable state.

    A run is resumable when its workflow-state JSON exists and carries at least
    one phase persisted as ``_completed`` (a checkpoint the resume path can skip
    past). Without a completed phase there is nothing to resume — re-driving
    would just restart from phase 0, so we treat it as non-resumable and let the
    caller mark it failed/interrupted.

    Returns ``None`` for phase runs (``kind != "pipeline"`` / no ``workflow_id``)
    and for any state file that is missing, corrupt, or has no completed phase.
    """
    if record.get("kind") != "pipeline":
        return None
    workflow_id = record.get("workflow_id")
    if not workflow_id:
        return None
    state_file = _workflow_state_file(workflow_id)
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    phase_outputs = state.get("phase_outputs") if isinstance(state, dict) else None
    if not isinstance(phase_outputs, dict):
        return None
    has_checkpoint = any(
        isinstance(out, dict) and out.get("_completed") for out in phase_outputs.values()
    )
    return workflow_id if has_checkpoint else None


def _mark_orphan_failed(run_id: str, reason: str, *, terminal_status: str) -> bool:
    """Stamp an orphaned run terminal (failed/interrupted) with ``reason``.

    Returns ``True`` when the patch was written. Tolerates a vanished record.
    """
    try:
        shared_state.update_run(
            run_id,
            {
                "status": terminal_status,
                "error": reason,
                "finished_at": shared_state.now_iso(),
            },
        )
        shared_state.append_log(
            run_id, f"[{shared_state.now_iso()}] {terminal_status}: {reason}\n"
        )
        return True
    except FileNotFoundError:
        logger.warning("run %s vanished during orphan reconciliation", run_id)
        return False


def reconcile_orphans() -> List[str]:
    """Reconcile non-terminal runs left over from a previous process on boot.

    Background driver tasks live only in-process, so a uvicorn restart leaves any
    ``queued``/``running`` record stuck forever (no task drives it, WS clients
    poll indefinitely). On boot we scan the registry and, per orphan:

    * **Resumable** (pipeline run with a workflow-state checkpoint AND under the
      auto-resume cap): re-enter the drive loop via :func:`_resume_orphan` — the
      SAME pathway a fresh launch uses (authoring-route env setup +
      ``_drive_pipeline`` against the existing ``workflow_id``, which the
      orchestrator resumes by skipping already-``_completed`` phases). The run
      goes back to ``status="queued"`` and ``resume_attempts`` is incremented.
    * **Not resumable** (no checkpoint, corrupt state, phase run): stamped
      ``status="interrupted"`` with the reason.
    * **Resume cap exhausted** (already auto-resumed ``_MAX_AUTO_RESUME_ATTEMPTS``
      times and re-orphaned): stamped ``status="failed"`` with the reason, to
      avoid a crash loop.

    Resume launches require a running event loop (the FastAPI ``startup`` hook is
    async). When no loop is running (a synchronous caller / test of the
    mark-only path), a resumable orphan is conservatively marked ``interrupted``
    rather than silently resumed.

    Returns the list of run_ids that were reconciled — resumed OR marked terminal
    (for logging/tests).
    """
    reconciled: List[str] = []
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    for record in shared_state.list_runs():
        if record.get("status") not in _NON_TERMINAL_STATES:
            continue
        run_id = record.get("run_id")
        if not run_id:
            continue

        base_reason = record.get("error") or (
            "process restarted; run was no longer being driven"
        )
        try:
            workflow_id = _resumable_workflow_id(record)
            attempts = int(record.get("resume_attempts") or 0)

            if workflow_id and attempts >= _MAX_AUTO_RESUME_ATTEMPTS:
                # Auto-resume already spent; refuse to resume again (crash-loop
                # guard) and mark failed with the reason.
                if _mark_orphan_failed(
                    run_id,
                    f"{base_reason}; auto-resume cap reached "
                    f"({attempts}/{_MAX_AUTO_RESUME_ATTEMPTS}) — not resuming again",
                    terminal_status="failed",
                ):
                    reconciled.append(run_id)
                continue

            if workflow_id and loop is not None:
                _resume_orphan(run_id, workflow_id, record, attempts)
                reconciled.append(run_id)
                continue

            # Non-resumable (no checkpoint / corrupt state / phase run), or a
            # resumable run but no event loop to drive it: mark interrupted.
            note = (
                base_reason
                if workflow_id is None
                else f"{base_reason}; resumable but no event loop to drive resume"
            )
            if _mark_orphan_failed(run_id, note, terminal_status="interrupted"):
                reconciled.append(run_id)
        except Exception:  # noqa: BLE001 — reconciliation is best-effort
            logger.exception("failed to reconcile orphan run %s", run_id)

    if reconciled:
        logger.info("reconciled %d orphaned run(s): %s", len(reconciled), reconciled)
    return reconciled


def _resume_orphan(
    run_id: str,
    workflow_id: str,
    record: Dict[str, Any],
    prior_attempts: int,
) -> None:
    """Re-drive an orphaned run from its checkpoint on the same pathway as launch.

    Re-applies the blessed authoring-route provider env (Marketable-v1 A3 — a
    headless/GUI resume has no Claude session servicing the mailbox, so every
    LLM-needing phase must resolve through the in-process provider lattice, just
    like a fresh launch) using the run record's persisted ``provider``, bumps the
    persisted ``resume_attempts`` counter, flips the record back to ``queued``,
    and schedules ``_drive_pipeline`` against the EXISTING ``workflow_id``. The
    orchestrator resumes by skipping already-``_completed`` phases.

    The attempt bump is persisted BEFORE the drive task starts so that if the
    resumed process also dies (re-orphaning the run), the next boot sees the
    incremented count and marks the run failed instead of resuming forever.
    """
    mode = record.get("mode") or _resolve_mode({})
    provider = record.get("provider") or _resolve_provider({})
    model = record.get("model")

    # Re-apply the authoring-route env so the resumed run routes generation
    # through the in-process lattice (A3 guardrail). Best-effort — a failure
    # here is logged; the drive task's own guardrail surfaces a hard error if
    # the route is still unsatisfied.
    try:
        _apply_authoring_route_env({"provider": provider})
    except Exception:  # noqa: BLE001
        logger.exception("resume: failed to apply authoring-route env for %s", run_id)

    shared_state.update_run(
        run_id,
        {
            "status": "queued",
            "error": None,
            "finished_at": None,
            "resume_attempts": prior_attempts + 1,
        },
    )
    shared_state.append_log(
        run_id,
        f"[{shared_state.now_iso()}] auto-resuming from checkpoint "
        f"(attempt {prior_attempts + 1}/{_MAX_AUTO_RESUME_ATTEMPTS}) "
        f"workflow={workflow_id} mode={mode} provider={provider}\n",
    )
    shared_state.append_event(
        "gui",
        "run_resumed",
        {"run_id": run_id, "workflow_id": workflow_id, "attempt": prior_attempts + 1},
    )

    task = asyncio.ensure_future(
        _drive_pipeline(run_id, workflow_id, mode=mode, provider=provider, model=model)
    )
    _BACKGROUND_TASKS[run_id] = task
    task.add_done_callback(lambda _t, _rid=run_id: _BACKGROUND_TASKS.pop(_rid, None))


def tail_log(run_id: str, offset: int = 0) -> Tuple[str, int]:
    """Return ``(new_text, new_offset)`` from the run's log for ws streaming."""
    return shared_state.tail_log(run_id, offset)


def cancel_run(run_id: str) -> Dict[str, Any]:
    """Request cancellation of a run (HONEST two-stage, Phase 4 §5.1(E)).

    Cancellation is cooperative / phase-boundary: the orchestrator may already be
    mid-phase and may not honor mid-phase interruption, so the currently-executing
    phase can run to completion before the cancel is observed. We therefore do NOT
    claim an instant stop. Instead this writes the NON-terminal intermediate
    status ``cancel_requested`` (+ ``cancel_requested_at``) immediately and signals
    the in-process driver task. The authoritative flip to the terminal
    ``status="cancelled"`` happens later, when the driver's ``CancelledError``
    handler (or ``_finalize_status``) confirms the orchestrator actually exited.

    Two-stage flow:
      1. ``cancel_run``  → ``status="cancel_requested"`` (HTTP 202 at the endpoint),
         best-effort ``task.cancel()`` to accelerate the cooperative exit.
      2. ``_drive_pipeline`` ``CancelledError``/finalize → ``status="cancelled"``.

    ``cancel_requested`` is treated as NON-terminal by the ``_finalize_status``
    clobber guard (only ``cancelled`` blocks a terminal write) so a phase that
    races to completion can STILL write the terminal ``cancelled`` (or, if the
    cancel never lands cooperatively, the natural ``completed``/``failed``).

    Returns ``{run_id, status:"cancel_requested"}`` on a fresh request, a typed
    ``{"error": "unknown_run", ...}`` when the run is unknown, the current record
    (no-op note) when the run is already terminal, and an idempotent no-op note
    when a cancel was already requested.
    """
    record = shared_state.read_run(run_id)
    if record is None:
        return {"error": "unknown_run", "detail": f"no run with id {run_id!r}"}
    status = record.get("status")
    if status in ("completed", "failed", "cancelled", "interrupted"):
        return {
            "run_id": run_id,
            "status": status,
            "note": "run already terminal; nothing to cancel",
        }
    if status == "cancel_requested":
        # Idempotent: a second cancel while the first is still settling is a no-op.
        return {
            "run_id": run_id,
            "status": "cancel_requested",
            "note": "cancellation already requested; awaiting orchestrator exit",
        }
    updated = shared_state.update_run(
        run_id,
        {"status": "cancel_requested", "cancel_requested_at": shared_state.now_iso()},
    )
    # Accelerate the cooperative exit: cancel the driving asyncio task if this
    # process owns it. After a uvicorn restart the handle is gone (in-process
    # only); the ``cancel_requested`` flip above is still the signal the orphan
    # reconciler / WS clients see, and the terminal ``cancelled`` lands when the
    # driver finally exits.
    task = _BACKGROUND_TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    shared_state.append_log(run_id, f"[{shared_state.now_iso()}] cancellation requested\n")
    shared_state.append_event("gui", "run_cancel_requested", {"run_id": run_id})
    return {"run_id": run_id, "status": updated.get("status")}


async def resume_run(resume_run_id: str, req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resume a prior pipeline run from its orchestrator checkpoint (Phase 4 §5.1(E)).

    Operator-facing counterpart to the boot-time auto-resume (:func:`_resume_orphan`)
    and the CLI ``ed4all run ... --resume WF-...`` pathway. Given the GUI run_id of
    a prior pipeline run that holds a resumable workflow-state checkpoint, this
    starts a FRESH GUI run record that re-drives the SAME orchestrator
    ``workflow_id`` — the orchestrator resumes by skipping already-``_completed``
    phases (no re-upload, the staged corpus + checkpoints persist on disk).

    Never fabricates success: an unknown prior run, a phase (non-pipeline) run, or
    a prior run with no checkpoint returns a typed ``{status:"failed", error}`` (so
    the router maps it to a 422), exactly like a bad launch.

    ``req`` (optional) lets the caller override ``mode``/``provider``/``model`` for
    the resumed drive; absent, the prior run's persisted values are reused.
    ``req["clear_stop_after"]`` (I1 review flow): when truthy, the persisted
    ``--stop-after`` halt marker is stripped from the workflow-state file before
    the re-drive so the resumed build continues PAST the review checkpoint rather
    than immediately re-pausing at the same phase.
    """
    req = req or {}
    prior = shared_state.read_run(resume_run_id)
    new_run_id = shared_state.new_run_id("GUI")

    if prior is None:
        return _record_launch_failure(
            new_run_id,
            workflow=None,
            course_name=None,
            kind="pipeline",
            mode=_resolve_mode(req),
            provider=_resolve_provider(req),
            model=req.get("model"),
            error=f"cannot resume: unknown run {resume_run_id!r}",
        )

    workflow = prior.get("workflow")
    course_name = prior.get("course_name")
    mode = req.get("mode") or prior.get("mode") or _resolve_mode(req)
    provider = req.get("provider") or prior.get("provider") or _resolve_provider(req)
    model = req.get("model") if req.get("model") is not None else prior.get("model")

    workflow_id = _resumable_workflow_id(prior)
    if not workflow_id:
        return _record_launch_failure(
            new_run_id,
            workflow=workflow,
            course_name=course_name,
            kind="pipeline",
            mode=mode,
            provider=provider,
            model=model,
            error=(
                f"cannot resume {resume_run_id!r}: no resumable workflow checkpoint "
                "(phase run, missing/corrupt state, or no completed phase)"
            ),
        )

    # Re-apply the blessed authoring-route env so the resumed run routes
    # generation through the in-process provider lattice (A3 guardrail) — a
    # headless/GUI resume has no Claude session servicing the mailbox. Best-effort.
    try:
        _apply_authoring_route_env({"provider": provider})
    except Exception:  # noqa: BLE001 — the drive guardrail surfaces a hard error
        logger.exception("resume: failed to apply authoring-route env for %s", new_run_id)

    # I1 review-flow resume: continue PAST the objectives-review checkpoint. A
    # run paused via ``--stop-after`` persists that marker in its workflow
    # params, so a PLAIN re-drive would immediately re-pause at the same phase.
    # When the caller asks to clear it, strip the marker from the state file
    # first (the runner reads it from there) AND from the resumed record's
    # params echo so the new run doesn't misleadingly advertise a halt point.
    resumed_params = prior.get("params")
    cleared_stop_after = False
    if req.get("clear_stop_after"):
        try:
            cleared_stop_after = _clear_stop_after(workflow_id)
        except Exception:  # noqa: BLE001 — best-effort; the re-drive still runs
            logger.exception("resume: failed to clear stop_after for %s", workflow_id)
        if isinstance(resumed_params, dict) and resumed_params.get("stop_after"):
            resumed_params = {
                k: v for k, v in resumed_params.items() if k != "stop_after"
            }

    record = {
        "run_id": new_run_id,
        "kind": "pipeline",
        "workflow": workflow,
        "workflow_id": workflow_id,
        "course_name": course_name,
        "phase": None,
        "mode": mode,
        "provider": provider,
        "model": model,
        "status": "queued",
        "params": resumed_params,
        "resumed_from": resume_run_id,
        "gate_results": None,
        "tasks": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    shared_state.register_run(record)
    shared_state.append_log(
        new_run_id,
        f"[{shared_state.now_iso()}] resuming workflow {workflow_id} from "
        f"{resume_run_id} mode={mode} provider={provider}\n",
    )
    if cleared_stop_after:
        shared_state.append_log(
            new_run_id,
            f"[{shared_state.now_iso()}] cleared --stop-after marker; continuing "
            "the build past the objectives-review checkpoint\n",
        )
    shared_state.append_event(
        "gui",
        "run_resumed",
        {"run_id": new_run_id, "workflow_id": workflow_id, "resumed_from": resume_run_id},
    )

    task = asyncio.ensure_future(
        _drive_pipeline(new_run_id, workflow_id, mode=mode, provider=provider, model=model)
    )
    _BACKGROUND_TASKS[new_run_id] = task
    task.add_done_callback(lambda _t, _rid=new_run_id: _BACKGROUND_TASKS.pop(_rid, None))

    return {**record, "resume_run_id": resume_run_id}


__all__ = [
    "list_workflows",
    "launch_pipeline",
    "resume_run",
    "paused_review_info",
    "launch_phase",
    "run_status",
    "list_runs",
    "reconcile_orphans",
    "tail_log",
    "cancel_run",
    "phase_duration_medians",
    "validation_report",
    "failed_gate_digest",
    "locate_validation_report",
    "SUPPORTED_WORKFLOWS",
    "COURSEFORGE_STAGE_SUBCOMMANDS",
    "PHASE_LABELS",
]
