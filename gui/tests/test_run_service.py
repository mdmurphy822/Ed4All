"""Tests for ``gui.services.run_service`` — workflows, launch, phase.

The heavy orchestrator calls (``create_textbook_pipeline``,
``create_workflow_impl``, ``PipelineOrchestrator``) are monkeypatched to RECORD
their invocation and assert the REAL functions are called with correctly-routed
params + applied env — NOT to short-circuit into a fabricated "ok".

No fastapi needed (run_service is web-free). State isolated via ``state_dir``.
"""

from __future__ import annotations

import asyncio
import json
import os

from gui import shared_state
from gui.services import run_service


def test_list_workflows_shape(state_dir):
    workflows = run_service.list_workflows()
    by_name = {w["name"]: w for w in workflows}
    # Real OrchestratorConfig workflows present.
    assert "textbook_to_course" in by_name
    tbc = by_name["textbook_to_course"]
    assert tbc["phases"], "textbook_to_course must carry phases from the real config"
    phase = tbc["phases"][0]
    assert {"name", "agents", "depends_on", "validation_gates_count"} <= set(phase.keys())
    assert isinstance(phase["validation_gates_count"], int)
    # Stage subcommands appended as aliases.
    for alias in run_service.COURSEFORGE_STAGE_SUBCOMMANDS:
        assert alias in by_name
        assert by_name[alias]["stage_subcommand"] is True


def test_launch_pipeline_invokes_real_create_textbook(
    state_dir, monkeypatch
):
    """launch_pipeline must call the REAL create_textbook_pipeline + apply env."""
    import MCP.tools.pipeline_tools as pt

    recorded = {}

    async def fake_create(**kwargs):
        recorded["kwargs"] = kwargs
        # Mirror the real return contract (a JSON string with workflow_id).
        return json.dumps({"workflow_id": "WF-TEST-0001", "status": "created"})

    monkeypatch.setattr(pt, "create_textbook_pipeline", fake_create)

    # Stop the background driver from actually running the orchestrator.
    driven = {}

    async def fake_drive(run_id, workflow_id, **kw):
        driven["run_id"] = run_id
        driven["workflow_id"] = workflow_id

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    # A pre-existing corpus path so _resolve_corpus returns it verbatim.
    corpus = state_dir / "fixture.pdf"
    corpus.write_bytes(b"%PDF-1.4 test")

    req = {
        "workflow": "textbook_to_course",
        "course_name": "PHYS_101",
        "corpus": str(corpus),
        "weeks": 14,
        "mode": "api",
        "provider": "anthropic",
        "options": {"assessment_count": 7},
    }
    result = asyncio.run(run_service.launch_pipeline(req))

    assert result["status"] == "queued"
    assert result["workflow_id"] == "WF-TEST-0001"
    # Real function was called with routed params.
    kw = recorded["kwargs"]
    assert kw["course_name"] == "PHYS_101"
    assert kw["pdf_paths"] == str(corpus)
    assert kw["duration_weeks"] == 14
    assert kw["duration_weeks_explicit"] is True
    assert kw["assessment_count"] == 7
    # Per-request env override applied to os.environ.
    assert os.environ.get("LLM_MODE") == "api"
    assert os.environ.get("LLM_PROVIDER") == "anthropic"
    # Run record persisted as queued.
    record = shared_state.read_run(result["run_id"])
    assert record["status"] == "queued"
    assert record["workflow_id"] == "WF-TEST-0001"


def test_launch_pipeline_generic_workflow_uses_create_workflow_impl(
    state_dir, monkeypatch
):
    import MCP.tools.orchestrator_tools as ot

    recorded = {}

    async def fake_impl(workflow_type, params, priority):
        recorded["workflow_type"] = workflow_type
        recorded["params"] = json.loads(params)
        recorded["priority"] = priority
        return json.dumps({"workflow_id": "WF-RAG-0001", "status": "created"})

    monkeypatch.setattr(ot, "create_workflow_impl", fake_impl)

    async def fake_drive(*a, **kw):
        return None

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    req = {
        "workflow": "rag_training",
        "course_name": "CHEM_101",
        "corpus": "/tmp/course.imscc",
        "options": {"priority": "high"},
    }
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["workflow_id"] == "WF-RAG-0001"
    assert recorded["workflow_type"] == "rag_training"
    assert recorded["params"]["course_name"] == "CHEM_101"
    assert recorded["priority"] == "high"


def test_launch_pipeline_unknown_workflow_fails_closed(state_dir):
    req = {"workflow": "not_a_workflow", "course_name": "X1", "corpus": "/tmp/x.pdf"}
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "failed"
    assert "unknown workflow" in result["error"]
    # Failure persisted (no fabricated success).
    record = shared_state.read_run(result["run_id"])
    assert record["status"] == "failed"
    assert record["workflow_id"] is None


def test_launch_pipeline_creation_error_persists_failed(state_dir, monkeypatch):
    """A backend that returns an {error} must persist status=failed, not ok."""
    import MCP.tools.pipeline_tools as pt

    async def fake_create(**kwargs):
        return json.dumps({"error": "SemantiK staging produced no HTML"})

    monkeypatch.setattr(pt, "create_textbook_pipeline", fake_create)

    req = {
        "workflow": "textbook_to_course",
        "course_name": "PHYS_101",
        "corpus": "/nonexistent/path.pdf",
    }
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "failed"
    assert "SemantiK staging produced no HTML" in result["error"]
    record = shared_state.read_run(result["run_id"])
    assert record["status"] == "failed"


def test_launch_pipeline_crash_persists_failed(state_dir, monkeypatch):
    import MCP.tools.pipeline_tools as pt

    async def boom(**kwargs):
        raise RuntimeError("orchestrator exploded")

    monkeypatch.setattr(pt, "create_textbook_pipeline", boom)

    req = {"workflow": "textbook_to_course", "course_name": "PHYS_101", "corpus": "x.pdf"}
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "failed"
    assert "orchestrator exploded" in result["error"]


def test_drive_pipeline_invokes_real_orchestrator(state_dir, monkeypatch):
    """_drive_pipeline must construct + run the REAL PipelineOrchestrator."""
    import MCP.orchestrator as orch_pkg

    constructed = {}

    class FakeResult:
        def to_dict(self):
            return {"status": "ok", "gates_passed": True, "phase_results": {}}

    class FakeOrchestrator:
        def __init__(self, mode, backend_spec):
            constructed["mode"] = mode
            constructed["spec"] = backend_spec

        async def run(self, workflow_id):
            constructed["workflow_id"] = workflow_id
            return FakeResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", FakeOrchestrator)

    # Register a run so _drive_pipeline can update it.
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "queued"})

    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-DRIVE-1", mode="api", provider="anthropic", model="m1"
        )
    )

    assert constructed["mode"] == "api"
    assert constructed["workflow_id"] == "WF-DRIVE-1"
    # BackendSpec carried the routed provider/model.
    assert constructed["spec"].provider == "anthropic"
    assert constructed["spec"].model == "m1"
    record = shared_state.read_run(run_id)
    assert record["status"] == "completed"
    assert record["gates_passed"] is True


def test_drive_pipeline_orchestrator_crash_persists_failed(state_dir, monkeypatch):
    import MCP.orchestrator as orch_pkg

    class CrashingOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            raise RuntimeError("run blew up")

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", CrashingOrchestrator)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "queued"})
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-CRASH", mode="api", provider="anthropic", model=None
        )
    )
    record = shared_state.read_run(run_id)
    assert record["status"] == "failed"
    assert "run blew up" in record["error"]


def test_drive_pipeline_persists_failed_phase_and_structured_line(
    state_dir, monkeypatch
):
    """A6: a failed run records failed_phase/failure_reason + a structured line."""
    import MCP.orchestrator as orch_pkg

    class FailResult:
        def to_dict(self):
            return {
                "status": "failed",
                "gates_passed": False,
                "failed_phase": "inter_tier_validation",
                "failure_reason": "failed validation gate(s): curie_anchoring (0.8<0.95)",
                "phase_results": {
                    "content_generation_outline": {"gates_passed": True, "failed": 0},
                    "inter_tier_validation": {"gates_passed": False, "failed": 0},
                },
                "error": "Phase inter_tier_validation failed validation gates",
            }

    class FailOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return FailResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", FailOrchestrator)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "queued"})
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-FAIL", mode="api", provider="anthropic", model=None
        )
    )
    record = shared_state.read_run(run_id)
    assert record["status"] == "failed"
    assert record["failed_phase"] == "inter_tier_validation"
    assert "curie_anchoring" in record["failure_reason"]
    # The structured per-phase failure line is in the log (the wizard parses it).
    log_text, _ = shared_state.tail_log(run_id, 0)
    assert "[phase] inter_tier_validation failed" in log_text


def test_drive_pipeline_infers_failed_phase_when_runner_omits_it(
    state_dir, monkeypatch
):
    """When the payload lacks failed_phase, it's inferred from phase_results."""
    import MCP.orchestrator as orch_pkg

    class FailResult:
        def to_dict(self):
            return {
                "status": "failed",
                "gates_passed": False,
                "phase_results": {
                    "staging": {"gates_passed": True, "failed": 0},
                    "course_planning": {"gates_passed": False, "failed": 0},
                },
                "error": "boom",
            }

    class FailOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return FailResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", FailOrchestrator)
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "queued"})
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-INFER", mode="api", provider="anthropic", model=None
        )
    )
    record = shared_state.read_run(run_id)
    assert record["failed_phase"] == "course_planning"


def test_launch_phase_invokes_real_execute_phase(state_dir, monkeypatch):
    """launch_phase must route through WorkflowRunner + TaskExecutor.execute_phase."""
    import MCP.core.executor as executor_mod
    import MCP.core.workflow_runner as wr_mod

    calls = {}

    # Patch the WorkflowRunner internals the service relies on, plus the
    # executor's execute_phase, to record they were invoked with routed args.
    def fake_route(self, phase_name, workflow_params, phase_outputs):
        calls["routed_phase"] = phase_name
        calls["workflow_params"] = dict(workflow_params)
        return {"routed": True}

    def fake_create_tasks(self, *, workflow_id, phase, routed_params, workflow_params):
        calls["create_tasks_phase"] = phase.name
        calls["routed_params"] = routed_params
        return [{"task_id": "t1"}]

    async def fake_execute_phase(self, **kwargs):
        calls["execute_phase_kwargs"] = kwargs
        # (results, gates_passed, gate_results)
        return ({"t1": _FakeRes()}, True, [{"gate": "content_structure", "passed": True}])

    monkeypatch.setattr(wr_mod.WorkflowRunner, "_route_params", fake_route)
    monkeypatch.setattr(wr_mod.WorkflowRunner, "_create_phase_tasks", fake_create_tasks)
    monkeypatch.setattr(executor_mod.TaskExecutor, "execute_phase", fake_execute_phase)

    # Pick a phase that really exists in textbook_to_course.
    from MCP.core.config import OrchestratorConfig

    wf = OrchestratorConfig.load().get_workflow("textbook_to_course")
    phase_name = wf.phases[0].name

    req = {
        "workflow": "textbook_to_course",
        "phase": phase_name,
        "course_name": "PHYS_101",
        "options": {"duration_weeks": 9},
    }
    result = asyncio.run(run_service.launch_phase(req))

    assert result["status"] == "completed"
    assert calls["routed_phase"] == phase_name
    assert calls["create_tasks_phase"] == phase_name
    assert calls["execute_phase_kwargs"]["phase_name"] == phase_name
    assert result["gate_results"] == [{"gate": "content_structure", "passed": True}]
    record = shared_state.read_run(result["run_id"])
    assert record["status"] == "completed"
    assert record["gates_passed"] is True


def test_launch_phase_missing_phase_field_fails(state_dir):
    req = {"workflow": "textbook_to_course", "course_name": "PHYS_101"}
    result = asyncio.run(run_service.launch_phase(req))
    assert result["status"] == "failed"
    assert "phase is required" in result["error"]


def test_launch_phase_unknown_phase_fails(state_dir, monkeypatch):
    # Avoid pre-population scanning the real Courseforge exports.
    monkeypatch.setattr(run_service, "_prepopulate_phase_outputs", lambda *a, **k: {})
    req = {
        "workflow": "textbook_to_course",
        "phase": "not_a_real_phase",
        "course_name": "PHYS_101",
    }
    result = asyncio.run(run_service.launch_phase(req))
    assert result["status"] == "failed"
    assert "not found" in result["error"]


class _FakeRes:
    status = "completed"
    success = True
    error = None


# --------------------------------------------------------------- cancellation


def test_finalize_status_refuses_to_clobber_cancelled(state_dir):
    """A cancelled run must NOT be overwritten back to completed/failed.

    The driver's terminal write goes through ``_finalize_status``; if a cancel
    landed first (status="cancelled"), the late completed/failed write must be
    refused so a mid-run cancel is never silently lost.
    """
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "cancelled"})

    # Driver tries to write a terminal "completed" after the cancel landed.
    wrote = run_service._finalize_status(
        run_id, {"status": "completed", "finished_at": shared_state.now_iso()}
    )
    assert wrote is False
    record = shared_state.read_run(run_id)
    assert record["status"] == "cancelled"

    # A "failed" finalize is likewise refused.
    assert run_service._finalize_status(run_id, {"status": "failed"}) is False
    assert shared_state.read_run(run_id)["status"] == "cancelled"


def test_finalize_status_writes_when_not_cancelled(state_dir):
    """The guard is a no-op for a normal (non-cancelled) terminal write."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    wrote = run_service._finalize_status(run_id, {"status": "completed"})
    assert wrote is True
    assert shared_state.read_run(run_id)["status"] == "completed"


def test_drive_pipeline_does_not_overwrite_cancelled(state_dir, monkeypatch):
    """End-to-end: a run cancelled before the orchestrator finishes stays cancelled.

    Simulates the race where ``orchestrator.run`` completes normally (returns
    status=ok) AFTER a cancel already flipped the record — the driver's
    finalize path must keep status="cancelled", not "completed".
    """
    import MCP.orchestrator as orch_pkg

    class FakeResult:
        def to_dict(self):
            return {"status": "ok", "gates_passed": True, "phase_results": {}}

    class LateFinishOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            # Cancel landed mid-run: flip the record before we return ok.
            shared_state.update_run(
                run_id, {"status": "cancelled", "finished_at": shared_state.now_iso()}
            )
            return FakeResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", LateFinishOrchestrator)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-CANCEL-RACE", mode="api", provider="anthropic", model=None
        )
    )
    assert shared_state.read_run(run_id)["status"] == "cancelled"


# --------------------------------------------- honest two-stage cancel (P4 §5.1E)


def test_cancel_run_writes_cancel_requested_not_cancelled(state_dir):
    """A fresh cancel writes the NON-terminal ``cancel_requested`` + timestamp.

    Phase 4 §5.1(E): cancellation is cooperative / phase-boundary, so cancel_run
    must NOT flip straight to the terminal ``cancelled`` — it stages
    ``cancel_requested`` and the driver lands the terminal flip later.
    """
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})

    result = run_service.cancel_run(run_id)
    assert result == {"run_id": run_id, "status": "cancel_requested"}
    record = shared_state.read_run(run_id)
    assert record["status"] == "cancel_requested"
    assert record.get("cancel_requested_at")  # iso stamp present


def test_cancel_run_unknown_is_typed_error(state_dir):
    """An unknown run yields the typed ``unknown_run`` error (router -> 404)."""
    result = run_service.cancel_run("GUI-nope-000000")
    assert result["error"] == "unknown_run"


def test_cancel_run_already_terminal_is_noop(state_dir):
    """A terminal run is a no-op: current status echoed with an explanatory note."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "completed"})
    result = run_service.cancel_run(run_id)
    assert result["status"] == "completed"
    assert "terminal" in result["note"]
    # The record is untouched.
    assert shared_state.read_run(run_id)["status"] == "completed"


def test_cancel_run_idempotent_while_requested(state_dir):
    """A second cancel while one is settling is an idempotent no-op."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    run_service.cancel_run(run_id)
    again = run_service.cancel_run(run_id)
    assert again["status"] == "cancel_requested"
    assert "already requested" in again["note"]


def test_cancel_requested_does_not_block_terminal_cancelled(state_dir):
    """The two-stage flip: ``cancel_requested`` -> ``cancelled`` is allowed.

    The clobber guard only blocks a write over the TERMINAL ``cancelled``; the
    intermediate ``cancel_requested`` must let the driver write the authoritative
    terminal ``cancelled`` (the CancelledError handler) — AND also let a natural
    completed/failed land if the orchestrator raced to completion uncancelled.
    """
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    run_service.cancel_run(run_id)
    assert shared_state.read_run(run_id)["status"] == "cancel_requested"

    # Driver confirms the orchestrator exited -> terminal cancelled write lands.
    wrote = run_service._finalize_status(
        run_id, {"status": "cancelled", "finished_at": shared_state.now_iso()}
    )
    assert wrote is True
    assert shared_state.read_run(run_id)["status"] == "cancelled"

    # And once terminal-cancelled, a late completed/failed is refused.
    assert run_service._finalize_status(run_id, {"status": "completed"}) is False
    assert shared_state.read_run(run_id)["status"] == "cancelled"


def test_cancel_requested_allows_natural_completion(state_dir):
    """If the cancel never lands cooperatively, a natural completed still writes."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    run_service.cancel_run(run_id)
    # Orchestrator raced to completion before the cancel was observed.
    wrote = run_service._finalize_status(run_id, {"status": "completed"})
    assert wrote is True
    assert shared_state.read_run(run_id)["status"] == "completed"


# ------------------------------------------------------------- orphan reconcile


def test_reconcile_orphans_flips_non_terminal_to_interrupted(state_dir):
    """A queued/running record left over from a dead process -> interrupted."""
    # Explicit, distinct ids: ``new_run_id`` can collide when called within the
    # same millisecond (its suffix is millisecond-hex), which would collapse
    # these into one registry file.
    queued = "GUI-20260101-000001"
    running = "GUI-20260101-000002"
    done = "GUI-20260101-000003"
    shared_state.register_run({"run_id": queued, "status": "queued"})
    shared_state.register_run({"run_id": running, "status": "running"})
    shared_state.register_run({"run_id": done, "status": "completed"})

    reconciled = run_service.reconcile_orphans()

    assert set(reconciled) == {queued, running}
    # No kind/workflow_id -> not resumable -> interrupted.
    assert shared_state.read_run(queued)["status"] == "interrupted"
    assert shared_state.read_run(running)["status"] == "interrupted"
    assert shared_state.read_run(queued)["finished_at"]
    # A terminal run is left untouched.
    assert shared_state.read_run(done)["status"] == "completed"


# ------------------------------------------------- durable-runs resume (D3)


def _write_workflow_state(state_dir, workflow_id, *, completed_phases):
    """Write a synthetic workflow-state JSON with the given completed phases."""
    workflows = state_dir / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    phase_outputs = {
        name: {"_completed": True, "some_output": "x"} for name in completed_phases
    }
    (workflows / f"{workflow_id}.json").write_text(
        json.dumps({"workflow_id": workflow_id, "phase_outputs": phase_outputs})
    )


def test_reconcile_resumes_orphan_with_checkpoint(state_dir, monkeypatch):
    """A pipeline orphan with a checkpoint is auto-resumed, not failed."""
    workflow_id = "WF-RESUME-1"
    _write_workflow_state(state_dir, workflow_id, completed_phases=["semantik_conversion"])

    driven = {}

    async def fake_drive(run_id, wf_id, *, mode, provider, model):
        driven["run_id"] = run_id
        driven["workflow_id"] = wf_id
        driven["provider"] = provider

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    run_id = "GUI-20260101-100001"
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow_id": workflow_id,
            "status": "running",
            "provider": "local",
            "mode": "local",
        }
    )

    async def _go():
        return run_service.reconcile_orphans()

    reconciled = asyncio.run(_go())

    assert run_id in reconciled
    # Drive task was scheduled against the SAME workflow_id (resume).
    assert driven["workflow_id"] == workflow_id
    record = shared_state.read_run(run_id)
    # Flipped back to queued with the attempt counter bumped.
    assert record["status"] == "queued"
    assert record["resume_attempts"] == 1


# ------------------------------------------- operator-facing resume (P4 §5.1E)


def test_resume_run_redrives_existing_workflow(state_dir, monkeypatch):
    """``resume_run`` re-drives the SAME workflow_id under a fresh GUI run id."""
    workflow_id = "WF-OPRESUME-1"
    _write_workflow_state(state_dir, workflow_id, completed_phases=["staging"])

    driven = {}

    async def fake_drive(run_id, wf_id, *, mode, provider, model):
        driven["run_id"] = run_id
        driven["workflow_id"] = wf_id
        driven["provider"] = provider

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    prior_id = "GUI-20260101-200001"
    shared_state.register_run(
        {
            "run_id": prior_id,
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "workflow_id": workflow_id,
            "course_name": "PHYS_101",
            "status": "interrupted",
            "provider": "local",
            "mode": "local",
            "params": {"course_name": "PHYS_101"},
        }
    )

    async def _go():
        return await run_service.resume_run(prior_id)

    result = asyncio.run(_go())

    new_id = result["run_id"]
    assert new_id != prior_id
    assert result["status"] == "queued"
    assert result["resumed_from"] == prior_id
    # The drive task targets the EXISTING workflow_id (the --resume pathway).
    assert driven["workflow_id"] == workflow_id
    assert driven["run_id"] == new_id
    # Fresh record persisted, carrying the resume linkage + inherited params.
    rec = shared_state.read_run(new_id)
    assert rec["workflow_id"] == workflow_id
    assert rec["resumed_from"] == prior_id
    assert rec["params"] == {"course_name": "PHYS_101"}


def test_resume_run_unknown_prior_fails_closed(state_dir):
    """Resuming an unknown prior run returns a typed failed record (router 422)."""
    result = asyncio.run(run_service.resume_run("GUI-not-here"))
    assert result["status"] == "failed"
    assert "unknown run" in result["error"]


def test_resume_run_no_checkpoint_fails_closed(state_dir):
    """A prior run with no resumable checkpoint fails closed (no fabrication)."""
    workflow_id = "WF-OPRESUME-NOCK"
    _write_workflow_state(state_dir, workflow_id, completed_phases=[])
    prior_id = "GUI-20260101-200002"
    shared_state.register_run(
        {
            "run_id": prior_id,
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "workflow_id": workflow_id,
            "status": "interrupted",
        }
    )
    result = asyncio.run(run_service.resume_run(prior_id))
    assert result["status"] == "failed"
    assert "no resumable workflow checkpoint" in result["error"]


def test_launch_pipeline_resume_run_id_delegates(state_dir, monkeypatch):
    """``launch_pipeline`` with ``resume_run_id`` delegates to ``resume_run``."""
    captured = {}

    async def fake_resume(rid, req):
        captured["rid"] = rid
        captured["req"] = req
        return {"run_id": "GUI-resumed", "status": "queued"}

    monkeypatch.setattr(run_service, "resume_run", fake_resume)
    req = {"resume_run_id": "GUI-prior-xyz"}
    result = asyncio.run(run_service.launch_pipeline(req))
    assert captured["rid"] == "GUI-prior-xyz"
    assert result == {"run_id": "GUI-resumed", "status": "queued"}


def test_reconcile_orphan_without_checkpoint_marks_interrupted(state_dir, monkeypatch):
    """A pipeline orphan whose state has no completed phase is not resumable."""
    workflow_id = "WF-NOCHECK-1"
    _write_workflow_state(state_dir, workflow_id, completed_phases=[])

    called = {"drive": False}

    async def fake_drive(*a, **k):  # pragma: no cover - must not run
        called["drive"] = True

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    run_id = "GUI-20260101-100002"
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow_id": workflow_id,
            "status": "running",
        }
    )

    reconciled = asyncio.run(_reconcile_in_loop())

    assert run_id in reconciled
    assert called["drive"] is False
    record = shared_state.read_run(run_id)
    assert record["status"] == "interrupted"
    assert "no longer being driven" in record["error"]


def test_reconcile_missing_workflow_state_marks_interrupted(state_dir, monkeypatch):
    """No workflow-state file at all -> non-resumable -> interrupted."""

    async def fake_drive(*a, **k):  # pragma: no cover
        raise AssertionError("must not drive")

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)
    run_id = "GUI-20260101-100003"
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow_id": "WF-DOES-NOT-EXIST",
            "status": "running",
        }
    )

    reconciled = asyncio.run(_reconcile_in_loop())

    assert run_id in reconciled
    assert shared_state.read_run(run_id)["status"] == "interrupted"


def test_reconcile_resume_cap_marks_failed(state_dir, monkeypatch):
    """A run already auto-resumed once is marked failed (crash-loop guard)."""
    workflow_id = "WF-CAP-1"
    _write_workflow_state(state_dir, workflow_id, completed_phases=["semantik_conversion"])

    called = {"drive": False}

    async def fake_drive(*a, **k):  # pragma: no cover
        called["drive"] = True

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    run_id = "GUI-20260101-100004"
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow_id": workflow_id,
            "status": "running",
            "resume_attempts": run_service._MAX_AUTO_RESUME_ATTEMPTS,
        }
    )

    reconciled = asyncio.run(_reconcile_in_loop())

    assert run_id in reconciled
    assert called["drive"] is False
    record = shared_state.read_run(run_id)
    assert record["status"] == "failed"
    assert "auto-resume cap reached" in record["error"]


def test_reconcile_resumable_without_loop_marks_interrupted(state_dir, monkeypatch):
    """Called synchronously (no event loop), a resumable orphan is interrupted."""
    workflow_id = "WF-NOLOOP-1"
    _write_workflow_state(state_dir, workflow_id, completed_phases=["semantik_conversion"])

    run_id = "GUI-20260101-100005"
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow_id": workflow_id,
            "status": "running",
        }
    )

    # No asyncio.run wrapper -> no running loop -> mark-only path.
    reconciled = run_service.reconcile_orphans()

    assert run_id in reconciled
    record = shared_state.read_run(run_id)
    assert record["status"] == "interrupted"
    assert "no event loop" in record["error"]


async def _reconcile_in_loop():
    """Run reconcile_orphans inside a running event loop (for resume scheduling)."""
    result = run_service.reconcile_orphans()
    # Let any scheduled drive tasks start (they're monkeypatched no-ops).
    await asyncio.sleep(0)
    return result


# ---------------------------------------------------------------------------
# Marketable-v1 A3 — blessed authoring-route env defaulting
# ---------------------------------------------------------------------------


def test_authoring_route_envs_match_executor_map():
    """The reproduced provider-env tuple must match the canonical executor map.

    ``run_service`` reproduces the env-var literals (to stay import-light); a
    drift between the tuple and ``AGENT_AUTHORING_PROVIDER_ENV_MAP`` would let
    a newly-blessed LLM agent slip through the GUI default-env wiring.
    """
    from MCP.core.executor import AGENT_AUTHORING_PROVIDER_ENV_MAP

    assert set(run_service._AUTHORING_PROVIDER_ENVS) == set(
        AGENT_AUTHORING_PROVIDER_ENV_MAP.values()
    )


def test_apply_authoring_route_env_defaults_to_local(monkeypatch):
    """With no provider configured, every LLM-agent env defaults to 'local'."""
    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    applied = run_service._apply_authoring_route_env({})

    assert set(applied) == set(run_service._AUTHORING_PROVIDER_ENVS)
    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        assert os.environ.get(env_var) == "local"


def test_apply_authoring_route_env_uses_request_provider(monkeypatch):
    """A per-request provider overrides the 'local' default for unset envs."""
    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        monkeypatch.delenv(env_var, raising=False)

    run_service._apply_authoring_route_env({"provider": "together"})

    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        assert os.environ.get(env_var) == "together"


def test_apply_authoring_route_env_preserves_existing(monkeypatch):
    """A provider env already set (via settings model_routing) is left intact."""
    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        monkeypatch.delenv(env_var, raising=False)
    # Operator pinned the course-planner to anthropic via settings.
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "anthropic")

    run_service._apply_authoring_route_env({"provider": "local"})

    # Per-task pin preserved; the rest defaulted to the request provider.
    assert os.environ.get("COURSEPLANNER_PROVIDER") == "anthropic"
    assert os.environ.get("COURSEFORGE_PROVIDER") == "local"


def test_launch_pipeline_sets_blessed_route_env(state_dir, monkeypatch):
    """A GUI pipeline launch wires the blessed authoring-route provider env.

    Without a Claude session servicing the mailbox, the workflow_runner
    guardrail would fail every LLM-needing phase whose provider env is
    unset. ``launch_pipeline`` must set them so the run routes via the
    in-process lattice.
    """
    import MCP.tools.pipeline_tools as pt

    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        monkeypatch.delenv(env_var, raising=False)

    async def fake_create(**kwargs):
        return json.dumps({"workflow_id": "WF-BLESSED-1", "status": "created"})

    async def fake_drive(*a, **kw):
        return None

    monkeypatch.setattr(pt, "create_textbook_pipeline", fake_create)
    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    corpus = state_dir / "fixture.pdf"
    corpus.write_bytes(b"%PDF-1.4 test")

    req = {
        "workflow": "textbook_to_course",
        "course_name": "PHYS_101",
        "corpus": str(corpus),
        "provider": "local",
    }
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "queued"
    for env_var in run_service._AUTHORING_PROVIDER_ENVS:
        assert os.environ.get(env_var) == "local"


# --------------------------------------------------------------------------- #
# Marketable-v1 A6 — operator failure UX (structured failure surface)
# --------------------------------------------------------------------------- #


def test_failed_gate_digest_from_flat_list():
    """A flat per-gate result list reduces to failed-only rows w/ the A6 shape."""
    gate_results = [
        {"gate_id": "content_structure", "severity": "critical", "passed": True, "issues": []},
        {
            "gate_id": "objective_assessment_similarity",
            "severity": "warning",
            "passed": False,
            "issues": ["3 objectives below the similarity floor", "see report"],
        },
        {
            "gate_id": "curie_anchoring",
            "severity": "critical",
            "passed": False,
            "issues": ["pair anchoring rate 0.81 < 0.95"],
        },
    ]
    rows = run_service.failed_gate_digest(gate_results)
    assert {r["gate_id"] for r in rows} == {
        "objective_assessment_similarity",
        "curie_anchoring",
    }
    by_id = {r["gate_id"]: r for r in rows}
    assert by_id["curie_anchoring"]["severity"] == "critical"
    assert by_id["curie_anchoring"]["issues_count"] == 1
    assert "pair anchoring" in by_id["curie_anchoring"]["message"]
    assert by_id["objective_assessment_similarity"]["issues_count"] == 2
    assert set(rows[0].keys()) == {"phase", "gate_id", "severity", "message", "issues_count"}


def test_failed_gate_digest_from_phase_rollup():
    """A phase_results rollup surfaces failing phases as coarse rows."""
    phase_results = {
        "staging": {"task_count": 1, "completed": 1, "failed": 0, "gates_passed": True},
        "content_generation": {
            "task_count": 12,
            "completed": 12,
            "failed": 0,
            "gates_passed": False,
        },
    }
    rows = run_service.failed_gate_digest(phase_results)
    assert len(rows) == 1
    assert rows[0]["phase"] == "content_generation"
    assert rows[0]["severity"] == "critical"


def test_infer_failed_phase_picks_first_failing():
    phase_results = {
        "staging": {"gates_passed": True, "failed": 0},
        "objective_extraction": {"gates_passed": True, "failed": 0},
        "course_planning": {"gates_passed": False, "failed": 0},
        "content_generation": {"gates_passed": False, "failed": 2},
    }
    name, reason = run_service._infer_failed_phase(phase_results, None)
    assert name == "course_planning"
    assert "validation" in reason


def _register_failed_run(course_name="PHYS_101", project_id=None, gate_results=None):
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "course_name": course_name,
            "status": "failed",
            "params": {"project_id": project_id} if project_id else {},
            "gate_results": gate_results,
            "failed_phase": "content_generation",
            "failure_reason": "failed validation gate(s): curie_anchoring",
            "error": "phase content_generation failed",
        }
    )
    return run_id


def test_validation_report_absent_returns_note(state_dir, monkeypatch):
    """No courseforge_validation_report.json -> report None + explanatory note."""
    import lib.paths as paths

    monkeypatch.setattr(paths, "COURSEFORGE_PATH", state_dir / "Courseforge")
    run_id = _register_failed_run(
        gate_results=[
            {"gate_id": "curie_anchoring", "severity": "critical", "passed": False,
             "issues": ["anchoring 0.8 < 0.95"]},
        ],
    )
    out = run_service.validation_report(run_id)
    assert out["run_id"] == run_id
    assert out["report"] is None
    assert "note" in out
    # The failed-gate digest is still available even without the aggregator file.
    assert out["failed_gates"][0]["gate_id"] == "curie_anchoring"
    assert out["failed_phase"] == "content_generation"


def test_validation_report_locates_synthetic_report(state_dir, monkeypatch):
    """A courseforge_validation_report.json under the project export is returned."""
    import lib.paths as paths

    cf_root = state_dir / "Courseforge"
    project_id = "PROJ-PHYS_101-20260610-deadbeef"
    export_dir = cf_root / "exports" / project_id
    export_dir.mkdir(parents=True, exist_ok=True)
    report_doc = {
        "schema_version": "1.1",
        "course_code": "PHYS_101",
        "status": "fail",
        "summary": {"total_gates": 4, "passed_count": 2, "failed_count": 2},
        "per_block_results": [
            {"block_id": "b1", "block_type": "assessment_item", "status": "failed"},
            {"block_id": "b2", "block_type": "assessment_item", "status": "passed"},
            {"block_id": "b3", "block_type": "objective", "status": "passed",
             "escalation_marker": "rewrite_2"},
        ],
    }
    (export_dir / "courseforge_validation_report.json").write_text(
        json.dumps(report_doc), encoding="utf-8"
    )
    monkeypatch.setattr(paths, "COURSEFORGE_PATH", cf_root)

    run_id = _register_failed_run(project_id=project_id)
    out = run_service.validation_report(run_id)
    assert out["report"] is not None
    assert out["report"]["course_code"] == "PHYS_101"
    assert out["report_path"].endswith("courseforge_validation_report.json")


def test_validation_report_unknown_run_raises_keyerror(state_dir):
    import pytest

    with pytest.raises(KeyError):
        run_service.validation_report("GUI-nope-000000")


# ---------------------------------------------------------------------------
# Phase 3 — per-phase task-progress ``[progress] <phase> <done>/<total>`` lines
# ---------------------------------------------------------------------------

import re as _re

_PROGRESS_RE = _re.compile(
    r"^\[(?P<iso>[^\]]+)\]\s+\[progress\]\s+(?P<name>\S+)\s+(?P<done>\d+)/(?P<total>\d+)$"
)


def _progress_lines(run_id):
    """Return parsed ``[progress]`` log lines for ``run_id`` (in order)."""
    text, _ = shared_state.tail_log(run_id, 0)
    out = []
    for line in text.splitlines():
        m = _PROGRESS_RE.match(line.strip())
        if m:
            out.append(
                {
                    "iso": m.group("iso"),
                    "name": m.group("name"),
                    "done": int(m.group("done")),
                    "total": int(m.group("total")),
                }
            )
    return out


def test_emit_phase_progress_line_grammar_and_iso_prefix(state_dir):
    """A real count emits ``[<iso>] [progress] <phase> <done>/<total>`` exactly."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    emitted = run_service._emit_phase_progress_line(run_id, "content_generation", 7, 50)
    assert emitted is True
    lines = _progress_lines(run_id)
    assert len(lines) == 1
    assert lines[0]["name"] == "content_generation"
    assert lines[0]["done"] == 7
    assert lines[0]["total"] == 50
    # The bracketed prefix is a real parseable ISO-8601 timestamp.
    from datetime import datetime

    datetime.fromisoformat(lines[0]["iso"])


def test_emit_phase_progress_line_suppresses_zero_and_garbage_counts(state_dir):
    """No fabricated line when the count is absent / non-numeric / negative."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    assert run_service._emit_phase_progress_line(run_id, "p", 0, 0) is False
    assert run_service._emit_phase_progress_line(run_id, "p", 1, 0) is False
    assert run_service._emit_phase_progress_line(run_id, "p", 0, -3) is False
    assert run_service._emit_phase_progress_line(run_id, "p", None, "x") is False
    assert _progress_lines(run_id) == []


def test_emit_phase_progress_line_clamps_done_to_total(state_dir):
    """A done count exceeding total is clamped (never emits an impossible ratio)."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    assert run_service._emit_phase_progress_line(run_id, "p", 99, 12) is True
    lines = _progress_lines(run_id)
    assert lines == [{"iso": lines[0]["iso"], "name": "p", "done": 12, "total": 12}]


def test_validator_only_phases_enumerates_agents_empty(state_dir):
    """The validator-only set is the ``agents: []`` phases of the workflow."""
    vo = run_service._validator_only_phases("textbook_to_course")
    assert "inter_tier_validation" in vo
    assert "post_rewrite_validation" in vo
    # Content phases with real agents are NOT validator-only.
    assert "content_generation" not in vo
    assert "course_planning" not in vo
    # Courseforge stage aliases resolve through the textbook_to_course machine.
    assert run_service._validator_only_phases("courseforge_validate") == vo


def test_drive_pipeline_emits_progress_from_real_phase_results(state_dir, monkeypatch):
    """Full-pipeline path emits ``[progress]`` from REAL per-phase task counts,
    and emits NOTHING for validator-only ``agents: []`` phases."""
    import MCP.orchestrator as orch_pkg

    class OkResult:
        def to_dict(self):
            return {
                "status": "ok",
                "gates_passed": True,
                "phase_results": {
                    # Real agent phase with a genuine task count.
                    "content_generation": {
                        "task_count": 12,
                        "completed": 12,
                        "gates_passed": True,
                    },
                    # Validator-only phase: synthesized virtual task -> task_count
                    # 1 is meaningless; MUST emit no [progress] line.
                    "inter_tier_validation": {
                        "task_count": 1,
                        "completed": 1,
                        "gates_passed": True,
                    },
                    # A real phase that partially completed (honest done<total).
                    "course_planning": {
                        "task_count": 4,
                        "completed": 3,
                        "gates_passed": True,
                    },
                    # A pure gate-chain phase with no real tasks -> no count.
                    "finalization": {
                        "task_count": 0,
                        "completed": 0,
                        "gates_passed": True,
                    },
                },
            }

    class OkOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return OkResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", OkOrchestrator)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "status": "queued", "workflow": "textbook_to_course"}
    )
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-PROG", mode="api", provider="anthropic", model=None
        )
    )

    lines = {ln["name"]: ln for ln in _progress_lines(run_id)}
    # Real agent phases emit their genuine count.
    assert lines["content_generation"]["done"] == 12
    assert lines["content_generation"]["total"] == 12
    assert lines["course_planning"]["done"] == 3
    assert lines["course_planning"]["total"] == 4
    # Validator-only phase: NO [progress] line (ring stays indeterminate).
    assert "inter_tier_validation" not in lines
    # No-task phase: nothing fabricated.
    assert "finalization" not in lines
    # Every emitted line carries a parseable ISO prefix.
    from datetime import datetime

    for ln in lines.values():
        datetime.fromisoformat(ln["iso"])


def test_run_single_phase_emits_progress_and_suppresses_validator_phase(
    state_dir, monkeypatch
):
    """Single-phase path emits a real ``[progress]`` count for an agent phase
    and emits NOTHING for an ``agents: []`` validator-only phase."""

    class FakeResult:
        def __init__(self, status):
            self.status = status

    async def _run_phase(phase_name, agents):
        run_id = shared_state.new_run_id("GUI")
        shared_state.register_run({"run_id": run_id, "status": "running"})

        class FakePhase:
            def __init__(self):
                self.name = phase_name
                self.agents = agents
                self.validation_gates = None
                self.max_concurrent = 5

        class FakeWorkflow:
            phases = [FakePhase()]

        class FakeConfig:
            def get_workflow(self, name):
                return FakeWorkflow()

        # 3 tasks; 2 COMPLETE -> honest 2/3 for an agent phase.
        fake_results = {
            "t1": FakeResult("COMPLETE"),
            "t2": FakeResult("COMPLETE"),
            "t3": FakeResult("FAILED"),
        }

        class FakeExecutor:
            def __init__(self, *a, **k):
                pass

            async def execute_phase(self, **kwargs):
                return fake_results, True, []

        class FakeRunner:
            def __init__(self, *a, **k):
                pass

            def _route_params(self, *a, **k):
                return {}

            def _create_phase_tasks(self, *a, **k):
                return [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]

        import MCP.core.config as cfg_mod
        import MCP.core.executor as exec_mod
        import MCP.core.workflow_runner as wr_mod
        import MCP.tools.pipeline_tools as pt_mod

        monkeypatch.setattr(cfg_mod.OrchestratorConfig, "load", classmethod(lambda cls: FakeConfig()))
        monkeypatch.setattr(exec_mod, "TaskExecutor", FakeExecutor)
        monkeypatch.setattr(wr_mod, "WorkflowRunner", FakeRunner)
        monkeypatch.setattr(pt_mod, "_build_tool_registry", lambda: {})
        monkeypatch.setattr(run_service, "_prepopulate_phase_outputs", lambda *a, **k: {})

        await run_service._run_single_phase(
            run_id=run_id,
            workflow="textbook_to_course",
            phase_name=phase_name,
            course_name="PHYS_101",
            project_id=None,
            options={},
        )
        return run_id

    # Agent phase: emits a real 2/3 progress line.
    agent_run = asyncio.run(_run_phase("content_generation", ["content-generator"]))
    agent_lines = _progress_lines(agent_run)
    assert len(agent_lines) == 1
    assert agent_lines[0]["name"] == "content_generation"
    assert agent_lines[0]["done"] == 2
    assert agent_lines[0]["total"] == 3

    # Validator-only phase (agents: []): emits NO progress line.
    vo_run = asyncio.run(_run_phase("inter_tier_validation", []))
    assert _progress_lines(vo_run) == []


# ----------------------------------------------------------- Phase 5 gate lines


import re as _re_gates  # noqa: E402

# Matches ``[<iso>] [gate] <phase> <gate_id> <pass|fail> <severity>`` (the
# frontend Build Console contract) AND the ``__summary__ <passed>/<total>``
# variant. The summary line is captured via the ``token == "__summary__"`` arm.
_GATE_RE = _re_gates.compile(
    r"^\[(?P<iso>[^\]]+)\]\s+\[gate\]\s+(?P<phase>\S+)\s+(?P<token>\S+)\s+(?P<rest>.+)$"
)


def _gate_lines(run_id):
    """Return parsed ``[gate]`` log lines for ``run_id`` (in order).

    Each entry is either a per-gate line
    ``{kind:"gate", iso, phase, gate_id, verdict, severity}`` or a summary line
    ``{kind:"summary", iso, phase, passed, total}``.
    """
    text, _ = shared_state.tail_log(run_id, 0)
    out = []
    for line in text.splitlines():
        m = _GATE_RE.match(line.strip())
        if not m:
            continue
        token = m.group("token")
        rest = m.group("rest").strip()
        if token == "__summary__":
            passed, _, total = rest.partition("/")
            out.append(
                {
                    "kind": "summary",
                    "iso": m.group("iso"),
                    "phase": m.group("phase"),
                    "passed": int(passed),
                    "total": int(total),
                }
            )
        else:
            verdict, _, severity = rest.partition(" ")
            out.append(
                {
                    "kind": "gate",
                    "iso": m.group("iso"),
                    "phase": m.group("phase"),
                    "gate_id": token,
                    "verdict": verdict,
                    "severity": severity.strip(),
                }
            )
    return out


def test_emit_gate_lines_real_detail_and_summary(state_dir):
    """Per-gate ``[gate]`` lines carry real ``<gate_id> <pass|fail> <severity>``
    (severity from the gate CONFIG), plus a real ``__summary__ passed/total``."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    # Real-shaped per-gate result dicts (GateResult.to_dict() shape): note there
    # is NO top-level severity key — severity is supplied by the config map.
    gate_results = [
        {"gate_id": "content_structure", "passed": True, "issues": []},
        {
            "gate_id": "curie_anchoring",
            "passed": False,
            "issues": [{"severity": "critical", "message": "0.8 < 0.95"}],
        },
    ]
    config_severity = {
        "content_structure": "critical",
        "curie_anchoring": "critical",
    }
    emitted = run_service._emit_gate_lines(
        run_id,
        "inter_tier_validation",
        gate_results,
        config_severity=config_severity,
    )
    assert emitted == 2

    lines = _gate_lines(run_id)
    gates = [ln for ln in lines if ln["kind"] == "gate"]
    summaries = [ln for ln in lines if ln["kind"] == "summary"]
    assert len(gates) == 2
    assert len(summaries) == 1

    by_id = {g["gate_id"]: g for g in gates}
    # The PASS gate carries the real config severity + ``pass`` verdict.
    assert by_id["content_structure"]["verdict"] == "pass"
    assert by_id["content_structure"]["severity"] == "critical"
    # The FAIL gate carries the real config severity + ``fail`` verdict.
    assert by_id["curie_anchoring"]["verdict"] == "fail"
    assert by_id["curie_anchoring"]["severity"] == "critical"
    assert by_id["curie_anchoring"]["phase"] == "inter_tier_validation"
    # The summary reports the REAL passed/total (1 of 2).
    assert summaries[0]["passed"] == 1
    assert summaries[0]["total"] == 2
    assert summaries[0]["phase"] == "inter_tier_validation"
    # Every line carries a parseable ISO prefix matching now_iso().
    from datetime import datetime

    for ln in lines:
        datetime.fromisoformat(ln["iso"])


def test_emit_gate_lines_severity_falls_back_to_issue_when_no_config(state_dir):
    """With no config severity, the gate's most-serious ISSUE severity is used
    (real signal), defaulting to ``warning`` only when truly absent."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    gate_results = [
        # No config entry; issue carries a real critical severity.
        {
            "gate_id": "numeric_literal_grounding",
            "passed": False,
            "issues": [{"severity": "warning"}, {"severity": "critical"}],
        },
        # No config entry, no issues -> conservative warning default.
        {"gate_id": "source_coverage", "passed": True, "issues": []},
    ]
    run_service._emit_gate_lines(
        run_id, "post_rewrite_validation", gate_results, config_severity={}
    )
    by_id = {g["gate_id"]: g for g in _gate_lines(run_id) if g["kind"] == "gate"}
    # Most-serious issue severity wins (critical beats warning).
    assert by_id["numeric_literal_grounding"]["severity"] == "critical"
    # No real signal at all -> conservative warning default (not fabricated).
    assert by_id["source_coverage"]["severity"] == "warning"


def test_emit_gate_lines_no_data_emits_nothing(state_dir):
    """A phase with no gate data emits NO ``[gate]`` line (no fabrication)."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    # Empty list, None, and a non-list all emit nothing.
    assert run_service._emit_gate_lines(run_id, "staging", []) == 0
    assert run_service._emit_gate_lines(run_id, "staging", None) == 0
    assert run_service._emit_gate_lines(run_id, "staging", {"not": "a list"}) == 0
    # A list whose entries carry NO real gate_id emits nothing (no blank-id line,
    # no summary over zero real gates).
    assert (
        run_service._emit_gate_lines(
            run_id, "staging", [{"gate_id": "", "passed": True}]
        )
        == 0
    )
    assert _gate_lines(run_id) == []


def test_emit_pipeline_gate_summary_only_on_all_pass(state_dir):
    """Full-pipeline summary emits ``<total>/<total>`` ONLY when gates_passed is
    True (passed count unknown on a fail -> emit nothing rather than fabricate)."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    # All gates passed + a real declared count -> truthful 3/3 summary.
    assert (
        run_service._emit_pipeline_gate_summary(
            run_id, "staging", {"gates_passed": True}, 3
        )
        is True
    )
    # A FAILED phase: passed count unknown GUI-side -> NO summary line.
    assert (
        run_service._emit_pipeline_gate_summary(
            run_id, "course_planning", {"gates_passed": False}, 5
        )
        is False
    )
    # Zero declared gates -> no summary.
    assert (
        run_service._emit_pipeline_gate_summary(
            run_id, "vector_indexing", {"gates_passed": True}, 0
        )
        is False
    )
    summaries = [ln for ln in _gate_lines(run_id) if ln["kind"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["phase"] == "staging"
    assert summaries[0]["passed"] == 3
    assert summaries[0]["total"] == 3


def test_phase_gate_counts_reads_real_config(state_dir):
    """Declared per-phase gate counts come from the real workflows.yaml."""
    counts = run_service._phase_gate_counts("textbook_to_course")
    # semantik_conversion declares at least its semantik_markers gate.
    assert counts.get("semantik_conversion", 0) >= 1
    # Stage aliases resolve through the textbook_to_course machine.
    assert run_service._phase_gate_counts("courseforge_validate") == counts


def test_drive_pipeline_emits_gate_summary_from_real_phase_results(
    state_dir, monkeypatch
):
    """Full-pipeline path emits a per-phase ``__summary__`` from the REAL
    gates_passed boolean + declared gate count; emits nothing for a failed phase
    and no per-id lines (the rollup carries none)."""
    import MCP.orchestrator as orch_pkg

    # Pin a deterministic gate-count map so the assertion is independent of the
    # live config gate counts.
    monkeypatch.setattr(
        run_service,
        "_phase_gate_counts",
        lambda wf: {
            "content_generation": 4,
            "inter_tier_validation": 7,
            "course_planning": 5,
            "finalization": 0,
        },
    )

    class OkResult:
        def to_dict(self):
            return {
                "status": "ok",
                "gates_passed": True,
                "phase_results": {
                    "content_generation": {
                        "task_count": 12,
                        "completed": 12,
                        "gates_passed": True,
                    },
                    # Validator-only phase still gets a gate summary (gate-only
                    # phase has a meaningful "all checks passing" signal).
                    "inter_tier_validation": {
                        "task_count": 1,
                        "completed": 1,
                        "gates_passed": True,
                    },
                    # A phase that failed gates -> NO summary (passed unknown).
                    "course_planning": {
                        "task_count": 4,
                        "completed": 4,
                        "gates_passed": False,
                    },
                    # Zero declared gates -> no summary.
                    "finalization": {
                        "task_count": 0,
                        "completed": 0,
                        "gates_passed": True,
                    },
                },
            }

    class OkOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return OkResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", OkOrchestrator)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "status": "queued", "workflow": "textbook_to_course"}
    )
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-GATE", mode="api", provider="anthropic", model=None
        )
    )

    lines = _gate_lines(run_id)
    # No per-id gate lines on the full-pipeline path (the rollup strips them).
    assert [ln for ln in lines if ln["kind"] == "gate"] == []
    summaries = {ln["phase"]: ln for ln in lines if ln["kind"] == "summary"}
    # All-pass phases with a declared count emit a truthful total/total summary.
    assert summaries["content_generation"]["passed"] == 4
    assert summaries["content_generation"]["total"] == 4
    assert summaries["inter_tier_validation"]["passed"] == 7
    assert summaries["inter_tier_validation"]["total"] == 7
    # Failed phase: passed count unknown -> NO summary line.
    assert "course_planning" not in summaries
    # Zero declared gates -> no summary.
    assert "finalization" not in summaries


def test_run_single_phase_emits_gate_lines_from_real_results(
    state_dir, monkeypatch
):
    """Single-phase path emits per-gate ``[gate]`` lines + a real ``__summary__``
    from the executor's REAL gate_results, with config-declared severity."""

    class FakeResult:
        def __init__(self, status):
            self.status = status

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})

    # Real-shaped gate config (dicts with gate_id + severity) + gate results.
    gate_configs = [
        {"gate_id": "block_structure", "severity": "critical"},
        {"gate_id": "claim_support", "severity": "warning"},
    ]
    gate_results = [
        {"gate_id": "block_structure", "passed": True, "issues": []},
        {
            "gate_id": "claim_support",
            "passed": False,
            "issues": [{"severity": "warning", "message": "weak support"}],
        },
    ]

    class FakePhase:
        name = "post_rewrite_validation"
        agents = []  # validator-only: progress suppressed, gates still emitted
        validation_gates = gate_configs
        max_concurrent = 5

    class FakeWorkflow:
        phases = [FakePhase()]

    class FakeConfig:
        def get_workflow(self, name):
            return FakeWorkflow()

    class FakeExecutor:
        def __init__(self, *a, **k):
            pass

        async def execute_phase(self, **kwargs):
            # gates_passed False because claim_support failed (warning, but the
            # executor's return is what we mirror); per-gate detail is real.
            return ({"t1": FakeResult("COMPLETE")}, False, gate_results)

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def _route_params(self, *a, **k):
            return {}

        def _create_phase_tasks(self, *a, **k):
            return [{"id": "t1"}]

    import MCP.core.config as cfg_mod
    import MCP.core.executor as exec_mod
    import MCP.core.workflow_runner as wr_mod
    import MCP.tools.pipeline_tools as pt_mod

    monkeypatch.setattr(
        cfg_mod.OrchestratorConfig, "load", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(exec_mod, "TaskExecutor", FakeExecutor)
    monkeypatch.setattr(wr_mod, "WorkflowRunner", FakeRunner)
    monkeypatch.setattr(pt_mod, "_build_tool_registry", lambda: {})
    monkeypatch.setattr(run_service, "_prepopulate_phase_outputs", lambda *a, **k: {})

    asyncio.run(
        run_service._run_single_phase(
            run_id=run_id,
            workflow="textbook_to_course",
            phase_name="post_rewrite_validation",
            course_name="PHYS_101",
            project_id=None,
            options={},
        )
    )

    lines = _gate_lines(run_id)
    gates = {g["gate_id"]: g for g in lines if g["kind"] == "gate"}
    summaries = [ln for ln in lines if ln["kind"] == "summary"]
    assert gates["block_structure"]["verdict"] == "pass"
    assert gates["block_structure"]["severity"] == "critical"  # from config
    assert gates["claim_support"]["verdict"] == "fail"
    assert gates["claim_support"]["severity"] == "warning"  # from config
    # Real summary: 1 of 2 passed.
    assert len(summaries) == 1
    assert summaries[0]["passed"] == 1
    assert summaries[0]["total"] == 2


def test_run_single_phase_no_gates_emits_no_gate_line(state_dir, monkeypatch):
    """A single-phase run whose executor returns NO gate_results emits no
    ``[gate]`` line (honest: no gate data -> no line)."""

    class FakeResult:
        def __init__(self, status):
            self.status = status

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})

    class FakePhase:
        name = "staging"
        agents = ["textbook-stager"]
        validation_gates = None
        max_concurrent = 5

    class FakeWorkflow:
        phases = [FakePhase()]

    class FakeConfig:
        def get_workflow(self, name):
            return FakeWorkflow()

    class FakeExecutor:
        def __init__(self, *a, **k):
            pass

        async def execute_phase(self, **kwargs):
            return ({"t1": FakeResult("COMPLETE")}, True, [])

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def _route_params(self, *a, **k):
            return {}

        def _create_phase_tasks(self, *a, **k):
            return [{"id": "t1"}]

    import MCP.core.config as cfg_mod
    import MCP.core.executor as exec_mod
    import MCP.core.workflow_runner as wr_mod
    import MCP.tools.pipeline_tools as pt_mod

    monkeypatch.setattr(
        cfg_mod.OrchestratorConfig, "load", classmethod(lambda cls: FakeConfig())
    )
    monkeypatch.setattr(exec_mod, "TaskExecutor", FakeExecutor)
    monkeypatch.setattr(wr_mod, "WorkflowRunner", FakeRunner)
    monkeypatch.setattr(pt_mod, "_build_tool_registry", lambda: {})
    monkeypatch.setattr(run_service, "_prepopulate_phase_outputs", lambda *a, **k: {})

    asyncio.run(
        run_service._run_single_phase(
            run_id=run_id,
            workflow="textbook_to_course",
            phase_name="staging",
            course_name="PHYS_101",
            project_id=None,
            options={},
        )
    )
    assert _gate_lines(run_id) == []


# ---------------------------------------------------------------------------
# I4 stage-2 — failure-panel scope normalization (--block-ids / --pages picker)
# ---------------------------------------------------------------------------


def test_normalize_blocks_param_accepts_json_arrays():
    """The picker posts JSON arrays; they normalize like CSV strings do."""
    params = {
        "block_ids": ["week_01_overview#objective_intro_01", "week_02_content_01#hook_a_02"],
        "pages": ["week_03"],
        "blocks": ["objective", "hook"],
    }
    run_service._normalize_blocks_param(params)
    assert params["target_block_instance_ids"] == [
        "week_01_overview#objective_intro_01",
        "week_02_content_01#hook_a_02",
    ]
    assert params["target_page_ids"] == ["week_03"]
    # blocks (type-level) is untouched + composable alongside the instance/page scopes.
    assert params["target_block_ids"] == ["objective", "hook"]
    # Source keys are consumed.
    assert "block_ids" not in params and "pages" not in params and "blocks" not in params


def test_normalize_blocks_param_accepts_csv_strings():
    """API callers can still post CSV strings for every scope."""
    params = {
        "block_ids": "week_01_overview#objective_intro_01,week_02_content_01#hook_a_02",
        "pages": "week_03,week_04",
    }
    run_service._normalize_blocks_param(params)
    assert params["target_block_instance_ids"] == [
        "week_01_overview#objective_intro_01",
        "week_02_content_01#hook_a_02",
    ]
    assert params["target_page_ids"] == ["week_03", "week_04"]


def test_normalize_blocks_param_dedups_preserving_order():
    """Repeated tokens (array or CSV) collapse to first-seen order."""
    params = {
        "block_ids": ["a#x_1", "b#y_2", "a#x_1"],
        "pages": ["week_02", "week_02"],
    }
    run_service._normalize_blocks_param(params)
    assert params["target_block_instance_ids"] == ["a#x_1", "b#y_2"]
    assert params["target_page_ids"] == ["week_02"]


def test_normalize_blocks_param_subsumption_drops_page_covered_ids():
    """A block-instance id whose page a selected pages token covers is dropped."""
    params = {
        "block_ids": [
            "week_01_overview#objective_a_01",   # covered by module prefix week_01
            "week_01#hook_b_02",                 # exact page match week_01
            "week_02_content_01#objective_c_03",  # NOT covered -> kept
        ],
        "pages": ["week_01"],
    }
    run_service._normalize_blocks_param(params)
    assert params["target_block_instance_ids"] == ["week_02_content_01#objective_c_03"]
    assert params["target_page_ids"] == ["week_01"]


def test_normalize_blocks_param_subsumption_prefix_boundary():
    """Module-prefix coverage is underscore-bounded: week_01 never covers week_010."""
    params = {
        "block_ids": ["week_010_content_01#objective_a_01"],
        "pages": ["week_01"],
    }
    run_service._normalize_blocks_param(params)
    # week_010 is NOT covered by week_01 -> the id survives.
    assert params["target_block_instance_ids"] == ["week_010_content_01#objective_a_01"]


def test_normalize_blocks_param_subsumption_empties_the_key():
    """When every id is page-covered no empty-list param is emitted."""
    params = {
        "block_ids": ["week_01_overview#objective_a_01"],
        "pages": ["week_01"],
    }
    run_service._normalize_blocks_param(params)
    assert "target_block_instance_ids" not in params
    assert params["target_page_ids"] == ["week_01"]


def test_normalize_blocks_param_block_id_without_separator_survives():
    """Defensive: a block id with no '#' has no page, so it is never subsumed."""
    params = {
        "block_ids": ["loose_id_no_hash"],
        "pages": ["week_01"],
    }
    run_service._normalize_blocks_param(params)
    assert params["target_block_instance_ids"] == ["loose_id_no_hash"]


def test_normalize_blocks_param_empty_and_whitespace_dropped():
    """Empty / whitespace-only selections emit NO param keys (byte-identical)."""
    params = {"block_ids": ["", "  ", "\t"], "pages": "", "blocks": []}
    run_service._normalize_blocks_param(params)
    assert "target_block_instance_ids" not in params
    assert "target_page_ids" not in params
    assert "target_block_ids" not in params


def test_page_covers_matches_pipeline_tools():
    """Parity: _page_covers agrees with the canonical _page_membership_match.

    ``run_service`` mirrors the eviction matching rule rather than importing it
    (import-light contract). This asserts the two verdicts are identical over a
    table of cases, so the mirror never drifts from the source of truth.
    """
    from MCP.tools.pipeline_tools import _page_membership_match

    cases = [
        ("week_01_overview", "week_01"),   # module prefix -> covered
        ("week_01", "week_01"),            # exact -> covered
        ("week_010", "week_01"),           # prefix boundary -> NOT covered
        ("week_01_content_02", "week_02"),  # different module -> NOT covered
        ("week_01_overview", ""),          # empty token -> NOT covered
        ("", "week_01"),                   # empty page -> NOT covered
    ]
    for page, token in cases:
        assert run_service._page_covers(page, token) == _page_membership_match(
            page, token
        ), f"parity mismatch for page={page!r} token={token!r}"
