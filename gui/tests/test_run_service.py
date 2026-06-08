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
        return json.dumps({"error": "DART staging produced no HTML"})

    monkeypatch.setattr(pt, "create_textbook_pipeline", fake_create)

    req = {
        "workflow": "textbook_to_course",
        "course_name": "PHYS_101",
        "corpus": "/nonexistent/path.pdf",
    }
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "failed"
    assert "DART staging produced no HTML" in result["error"]
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
    assert shared_state.read_run(queued)["status"] == "interrupted"
    assert shared_state.read_run(running)["status"] == "interrupted"
    assert shared_state.read_run(queued)["finished_at"]
    # A terminal run is left untouched.
    assert shared_state.read_run(done)["status"] == "completed"
