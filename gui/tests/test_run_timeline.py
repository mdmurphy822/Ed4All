"""Tests for the Phase-0 (Tier-1) GUI-side per-phase timing timeline.

``run_service.derive_phase_timeline`` reconstructs a per-phase duration vector
purely from the ISO-prefixed ``[phase] <name> done|skipped|failed`` log lines the
backend ALREADY emits — zero orchestrator change. These tests pin:

- correct sequential ``duration_ms`` per phase + ``total_ms`` from KNOWN times,
- graceful degrade to ``duration_ms: null`` on a garbage/missing timestamp
  (never raises),
- the ``GET /api/runs/{id}/timeline`` endpoint (timeline for a known run, typed
  404 for an unknown one),
- persisting ``phase_durations`` does NOT alter the run's ``status``.

State isolated via ``state_dir``; the router tests need fastapi (opt-in extra).
"""

from __future__ import annotations

from gui import shared_state
from gui.services import run_service


def _register(run_id: str, **fields) -> None:
    """Register a minimal run record carrying the given fields."""
    record = {"run_id": run_id, "kind": "pipeline", "status": "running"}
    record.update(fields)
    shared_state.register_run(record)


def test_derive_timeline_sequential_durations(state_dir):
    """Known, increasing ISO completion times → correct sequential durations."""
    run_id = "GUI-timeline-0001"
    started = "2026-06-17T12:00:00+00:00"
    _register(run_id, started_at=started)

    # done lines at +5s, +20s (=+15s from prev), +30s (=+10s from prev).
    shared_state.append_log(
        run_id,
        "[2026-06-17T12:00:05+00:00] [phase] staging done — Stage source files\n"
        "[2026-06-17T12:00:20+00:00] [phase] objective_extraction done — Read textbook structure\n"
        "[2026-06-17T12:00:30+00:00] [phase] course_planning skipped — Plan learning objectives\n",
    )

    result = run_service.derive_phase_timeline(run_id)
    assert result["run_id"] == run_id
    timeline = result["timeline"]
    assert [e["phase"] for e in timeline] == [
        "staging",
        "objective_extraction",
        "course_planning",
    ]
    assert [e["state"] for e in timeline] == ["done", "done", "skipped"]
    # First phase anchors off started_at (12:00:00 → 12:00:05 = 5000ms).
    assert timeline[0]["duration_ms"] == 5000
    assert timeline[1]["duration_ms"] == 15000
    assert timeline[2]["duration_ms"] == 10000
    assert timeline[0]["completed_at"] == "2026-06-17T12:00:05+00:00"
    # total_ms = sum of the non-null durations.
    assert result["total_ms"] == 5000 + 15000 + 10000


def test_derive_timeline_failed_line_parsed(state_dir):
    """The failure-path ``[phase] <name> failed — <label>: <reason>`` line parses."""
    run_id = "GUI-timeline-fail"
    _register(run_id, started_at="2026-06-17T09:00:00+00:00")
    shared_state.append_log(
        run_id,
        "[2026-06-17T09:00:10+00:00] [phase] staging done — Stage source files\n"
        "[2026-06-17T09:00:40+00:00] [phase] course_planning failed "
        "— Plan learning objectives: 1800s timeout\n",
    )
    timeline = run_service.derive_phase_timeline(run_id)["timeline"]
    assert [e["state"] for e in timeline] == ["done", "failed"]
    assert timeline[1]["phase"] == "course_planning"
    assert timeline[1]["duration_ms"] == 30000


def test_derive_timeline_garbage_timestamp_degrades_to_null(state_dir):
    """A garbage/missing timestamp → duration_ms null for that phase, no raise."""
    run_id = "GUI-timeline-garbage"
    _register(run_id, started_at="2026-06-17T12:00:00+00:00")
    shared_state.append_log(
        run_id,
        "[2026-06-17T12:00:05+00:00] [phase] staging done — Stage source files\n"
        "[NOT-A-TIMESTAMP] [phase] objective_extraction done — Read textbook structure\n"
        "[2026-06-17T12:00:30+00:00] [phase] course_planning done — Plan learning objectives\n",
    )
    # Must not raise.
    result = run_service.derive_phase_timeline(run_id)
    timeline = result["timeline"]
    assert len(timeline) == 3
    assert timeline[0]["duration_ms"] == 5000
    # Unparseable timestamp → null duration + null completed_at for that phase.
    assert timeline[1]["duration_ms"] is None
    assert timeline[1]["completed_at"] is None
    # The unparseable line cannot ANCHOR the next phase, so the third phase times
    # off the last VALID completion (12:00:05 → 12:00:30 = 25000ms) rather than
    # against the garbage line. The bad line is skipped, not poisoning.
    assert timeline[2]["duration_ms"] == 25000
    # total_ms ignores the null entry.
    assert result["total_ms"] == 5000 + 25000


def test_derive_timeline_missing_started_at_first_phase_null(state_dir):
    """No started_at anchor → first phase null, later phases still time off each other."""
    run_id = "GUI-timeline-nostart"
    _register(run_id)  # no started_at
    shared_state.append_log(
        run_id,
        "[2026-06-17T12:00:05+00:00] [phase] staging done — Stage source files\n"
        "[2026-06-17T12:00:20+00:00] [phase] objective_extraction done — Read textbook structure\n",
    )
    timeline = run_service.derive_phase_timeline(run_id)["timeline"]
    assert timeline[0]["duration_ms"] is None  # no anchor
    assert timeline[1]["duration_ms"] == 15000


def test_derive_timeline_empty_log_and_unknown_run(state_dir):
    """Absent/empty log → empty timeline; unknown run → empty timeline (no raise)."""
    # Registered run, no log written yet.
    run_id = "GUI-timeline-empty"
    _register(run_id, started_at="2026-06-17T12:00:00+00:00")
    result = run_service.derive_phase_timeline(run_id)
    assert result == {"run_id": run_id, "timeline": [], "total_ms": 0}

    # Wholly unknown run id.
    unknown = run_service.derive_phase_timeline("GUI-does-not-exist")
    assert unknown == {"run_id": "GUI-does-not-exist", "timeline": [], "total_ms": 0}


def test_finalize_persists_phase_durations_without_changing_status(state_dir, monkeypatch):
    """Persisting phase_durations at finalize must not alter status/final_status.

    Drives the real ``_drive_pipeline`` finalize path with a patched orchestrator
    that returns a successful result. The status must be the byte-identical
    ``completed`` it was before this Phase-0 change, and ``phase_durations`` must
    be persisted additively as a list derived from the run's per-phase log lines.
    """
    import asyncio

    run_id = "GUI-finalize-timing"
    _register(run_id, workflow_id="WF-X")

    # A successful orchestrator result (mirrors the WorkflowRunner contract).
    class _Result:
        def to_dict(self):
            return {
                "status": "ok",
                "gates_passed": True,
                "phase_results": {
                    "staging": {"completed": 1, "task_count": 1, "gates_passed": True}
                },
            }

    async def fake_run(self_orch, workflow_id):  # noqa: ANN001
        # Emit per-phase completion lines the way the live poller would, so the
        # Tier-1 derivation has real log lines to parse at finalize time.
        shared_state.append_log(
            run_id,
            f"[{shared_state.now_iso()}] [phase] staging done — Stage source files\n",
        )
        shared_state.append_log(
            run_id,
            f"[{shared_state.now_iso()}] [phase] course_planning done — Plan learning objectives\n",
        )
        return _Result()

    class _Orch:
        def __init__(self, *a, **k):
            pass

        run = fake_run

    class _Spec:
        def __init__(self, *a, **k):
            pass

    # Patch the deferred imports the driver resolves inside the try-block.
    import MCP.orchestrator as orch_pkg
    import MCP.orchestrator.llm_backend as backend_mod

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", _Orch, raising=False)
    monkeypatch.setattr(backend_mod, "BackendSpec", _Spec, raising=False)

    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-X", mode="local", provider="local", model=None
        )
    )

    record = shared_state.read_run(run_id)
    # Status is the byte-identical successful outcome (Phase 0 changed nothing).
    assert record["status"] == "completed"
    # phase_durations persisted additively as a list derived from the log.
    durations = record.get("phase_durations")
    assert isinstance(durations, list) and durations
    assert {e["phase"] for e in durations} == {"staging", "course_planning"}


# ----------------------------------------------------------------- router

import pytest  # noqa: E402

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


def test_timeline_endpoint_known_run(client, state_dir):
    run_id = "GUI-ep-known"
    _register(run_id, started_at="2026-06-17T12:00:00+00:00")
    shared_state.append_log(
        run_id,
        "[2026-06-17T12:00:05+00:00] [phase] staging done — Stage source files\n"
        "[2026-06-17T12:00:20+00:00] [phase] course_planning done — Plan learning objectives\n",
    )
    resp = client.get(f"/api/runs/{run_id}/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert [e["phase"] for e in body["timeline"]] == ["staging", "course_planning"]
    assert body["timeline"][0]["duration_ms"] == 5000
    assert body["total_ms"] == 5000 + 15000


def test_timeline_endpoint_unknown_run_typed_404(client, state_dir):
    resp = client.get("/api/runs/GUI-unknown-run/timeline")
    assert resp.status_code == 404
    body = resp.json()
    assert body == {"error": "unknown_run", "detail": "GUI-unknown-run"}
