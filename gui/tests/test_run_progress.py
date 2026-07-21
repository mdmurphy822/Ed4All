"""Tests for the stage-tracker progress endpoint + service.

``gui.services.progress_service`` merges the run's ``config/workflows.yaml``
phase plan with its orchestrator workflow state, checkpoint wall-clocks, and a
bounded tail of the OP2 ``llm_usage.jsonl`` tap. These tests pin:

- the sliding-window tok/s math (pure ``usage_window_stats``),
- the happy-path payload (states, groups, wall-clocks, usage totals),
- ``stats`` nulls for a run with no usage rows yet,
- a typed 404 for an unknown run id,
- skipped-phase mapping (``_skipped`` markers + params-driven prediction),
- a workflow-agnostic phase list (``course_generation`` renders ITS OWN config
  phases, never the 21-phase textbook list).

State isolated via ``state_dir``; the endpoint tests need fastapi (opt-in
``gui`` extra) and are skipped without it. No network: the seat registry env is
cleared so no probe is attempted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gui import shared_state
from gui.services import progress_service


@pytest.fixture(autouse=True)
def _no_seat_probes(monkeypatch: pytest.MonkeyPatch):
    """No registered seats → the service never opens a socket in tests."""
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    # The two-pass env must not leak into skip prediction from the dev shell.
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
    # Fresh usage accumulators per test (path-keyed module cache).
    progress_service._USAGE_ACCUMULATORS.clear()


def _seed_run(
    state_dir: Path,
    *,
    run_id: str = "GUI-prog-0001",
    workflow_id: str = "WF-20260101-prog0001",
    workflow: str = "textbook_to_course",
    gui_status: str = "running",
    wf_status: str = "RUNNING",
    orch_run_id: str = "TTC_prog_20260101_000000",  # synthetic run id, slug-guard: allow
    params: Optional[Dict[str, Any]] = None,
    phase_outputs: Optional[Dict[str, Any]] = None,
    failed_phase: Optional[str] = None,
    failure_reason: Optional[str] = None,
    usage_rows: Optional[List[Dict[str, Any]]] = None,
    checkpoints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Register a GUI run + write its workflow state / run-dir artifacts."""
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": workflow,
            "workflow_id": workflow_id,
            "course_name": "PHYS_101",
            "status": gui_status,
            "started_at": "2026-01-01T00:00:00",
        }
    )
    wf_doc: Dict[str, Any] = {
        "id": workflow_id,
        "type": workflow,
        "status": wf_status,
        "started_at": "2026-01-01T00:00:00",
        "params": {"run_id": orch_run_id, **(params or {})},
        "phase_outputs": phase_outputs or {},
    }
    if failed_phase:
        wf_doc["failed_phase"] = failed_phase
        wf_doc["failure_reason"] = failure_reason or "gate failure"
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(json.dumps(wf_doc), encoding="utf-8")

    run_dir = state_dir / "runs" / orch_run_id
    if usage_rows is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "llm_usage.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in usage_rows), encoding="utf-8"
        )
    for phase, ckpt in (checkpoints or {}).items():
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"{phase}_checkpoint.json").write_text(
            json.dumps({"phase_name": phase, **ckpt}), encoding="utf-8"
        )
    return run_id


def _phase(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(p for p in payload["phases"] if p["name"] == name)


# ------------------------------------------------- usage window (pure math)


def test_usage_window_stats_tok_s_math():
    """tok/s = window completion tokens over the calls' generation seconds."""
    now = 1_000_000.0

    def _iso(epoch: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    rows = [
        # Outside the 180s window — must be excluded from tok/s.
        {"ts": _iso(now - 3600), "completion_tokens": 99999, "duration_ms": 1000.0},
        # In-window: 500 tokens over 10s + 250 tokens over 5s → 750/15 = 50.0.
        {"ts": _iso(now - 100), "completion_tokens": 500, "duration_ms": 10000.0,
         "ttft_ms": 100.0},
        {"ts": _iso(now - 10), "completion_tokens": 250, "duration_ms": 5000.0,
         "ttft_ms": 300.0},
    ]
    out = progress_service.usage_window_stats(rows, now_ts=now, window_s=180.0)
    assert out["tok_s"] == 50.0
    assert out["window_calls"] == 2
    # p50 over ALL rows carrying ttft samples (100, 300) → 200.
    assert out["ttft_p50_ms"] == 200.0


def test_usage_window_stats_empty_window_is_null():
    """No rows inside the window → tok_s None (never fabricated)."""
    rows = [{"ts": "2020-01-01T00:00:00+00:00", "completion_tokens": 10, "duration_ms": 1000.0}]
    out = progress_service.usage_window_stats(rows, now_ts=2_000_000_000.0)
    assert out["tok_s"] is None
    assert out["ttft_p50_ms"] is None
    assert progress_service.usage_window_stats([], now_ts=1.0)["tok_s"] is None


# ------------------------------------------------------- service-level merge


def test_run_progress_happy_path(state_dir):
    """Done/current/pending states, groups, wall-clocks, and usage totals."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    recent_iso = now.isoformat()
    run_id = _seed_run(
        state_dir,
        params={"skip_training": True},
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
        },
        checkpoints={
            "staging": {
                "status": "completed",
                "started_at": "2026-01-01T00:01:00",
                "completed_at": "2026-01-01T00:01:30",
            },
        },
        usage_rows=[
            {"ts": recent_iso, "provider": "local", "model": "m", "prompt_tokens": 100,
             "completion_tokens": 400, "duration_ms": 8000.0, "ttft_ms": 500.0},
            {"ts": recent_iso, "provider": "local", "model": "m", "prompt_tokens": 50,
             "completion_tokens": 100, "duration_ms": 2000.0},
        ],
    )

    payload = progress_service.run_progress(run_id)
    assert payload is not None
    assert payload["workflow"] == "textbook_to_course"
    assert payload["status"] == "running"

    # Phase list is the CONFIG plan (21 phases for textbook_to_course), ordered.
    names = [p["name"] for p in payload["phases"]]
    assert names[0] == "semantik_conversion"
    assert "packaging" in names and "finalization" in names
    assert [p["index"] for p in payload["phases"]] == sorted(
        p["index"] for p in payload["phases"]
    )

    assert _phase(payload, "semantik_conversion")["state"] == "done"
    assert _phase(payload, "staging")["state"] == "done"
    # wall-clock from the checkpoint pair: 30s.
    assert _phase(payload, "staging")["wallclock_s"] == 30.0
    # The first unresolved phase is current.
    assert payload["current_phase"] == "chunking"
    assert _phase(payload, "chunking")["state"] == "current"
    assert _phase(payload, "packaging")["state"] == "pending"
    # --skip-training prediction → training_synthesis renders skipped.
    assert _phase(payload, "training_synthesis")["state"] == "skipped"

    # Groups derive from names, not indices.
    assert _phase(payload, "semantik_conversion")["group"] == "conversion"
    assert _phase(payload, "course_planning")["group"] == "planning"
    assert _phase(payload, "inter_tier_validation")["group"] == "validation"
    assert _phase(payload, "packaging")["group"] == "packaging"
    assert _phase(payload, "libv2_archival")["group"] == "archive"

    # Usage totals + in-window throughput: 500 tokens over 10 gen-seconds.
    stats = payload["stats"]
    assert stats["calls"] == 2
    assert stats["prompt_tokens"] == 150
    assert stats["completion_tokens"] == 500
    assert stats["tok_s"] == 50.0
    assert stats["ttft_p50_ms"] == 500.0
    assert stats["seat"] is None  # no registered seats in tests


def test_run_progress_no_usage_rows_yet(state_dir):
    """A run with no llm_usage.jsonl → stats nulls / zeros, no crash."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-nousage",
        workflow_id="WF-20260101-prog0002",
        orch_run_id="TTC_prog_nousage",
        usage_rows=None,
    )
    payload = progress_service.run_progress(run_id)
    assert payload is not None
    stats = payload["stats"]
    assert stats["tok_s"] is None
    assert stats["ttft_p50_ms"] is None
    assert stats["calls"] == 0
    assert stats["prompt_tokens"] == 0
    assert stats["completion_tokens"] == 0


def test_run_progress_unknown_run_is_none(state_dir):
    assert progress_service.run_progress("GUI-does-not-exist") is None


def test_run_progress_skipped_marker_and_env_verdict(state_dir):
    """``_skipped`` markers map to skipped; an OBSERVED env skip re-enables the
    complementary two-pass tiers even when the serving process env differs."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-skips",
        workflow_id="WF-20260101-prog0003",
        orch_run_id="TTC_prog_skips",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
            "course_planning": {"_completed": True},
            "concept_extraction": {"_completed": True},
            # The runner skipped the single-pass phase → the run had
            # COURSEFORGE_TWO_PASS=true, whatever THIS process env says.
            "content_generation": {"_completed": True, "_skipped": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    assert _phase(payload, "content_generation")["state"] == "skipped"
    # The observed verdict enables the two-pass tiers: outline is current,
    # the downstream tiers pending — NOT predicted-skipped off the test env.
    assert payload["current_phase"] == "content_generation_outline"
    assert _phase(payload, "inter_tier_validation")["state"] == "pending"
    assert _phase(payload, "content_generation_rewrite")["state"] == "pending"


def test_run_progress_failed_run_static(state_dir):
    """A failed run: failed phase marked, no current phase (static render)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-failed",
        workflow_id="WF-20260101-prog0004",
        orch_run_id="TTC_prog_failed",
        gui_status="failed",
        wf_status="FAILED",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
        },
        failed_phase="objective_extraction",
        failure_reason="failed validation gate(s): chunk_health",
    )
    payload = progress_service.run_progress(run_id)
    assert payload["status"] == "failed"
    assert payload["current_phase"] is None
    assert _phase(payload, "objective_extraction")["state"] == "failed"
    assert payload["failed_phase"] == "objective_extraction"
    assert "chunk_health" in payload["failure_reason"]
    # Nothing after the failure is current; the rest stays pending/skipped.
    assert _phase(payload, "packaging")["state"] in ("pending", "skipped")
    assert payload["stats"]["seat"] is None  # terminal → no probe


def test_run_progress_workflow_agnostic_phase_list(state_dir):
    """course_generation renders ITS OWN config phases — never the 21-phase
    textbook list (the workflow-agnostic contract)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-cg",
        workflow_id="WF-20260101-prog0005",
        workflow="course_generation",
        orch_run_id="TTC_prog_cg",
        phase_outputs={"planning": {"_completed": True}},
    )
    payload = progress_service.run_progress(run_id)
    names = [p["name"] for p in payload["phases"]]
    assert names[0] == "planning"
    assert "semantik_conversion" not in names
    assert "packaging" in names and "validation" in names
    assert len(names) < 15  # course_generation's own (10-phase) plan
    assert _phase(payload, "planning")["state"] == "done"
    assert _phase(payload, "planning")["group"] == "planning"
    assert _phase(payload, "validation")["group"] == "validation"


def test_checkpoint_start_marks_current_and_completion_marks_done(state_dir):
    """An in-flight ``status: started`` checkpoint names the current phase
    (ahead of the phase-boundary state re-save); a ``completed`` checkpoint
    marks done even before the workflow state file catches up."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-ckpt",
        workflow_id="WF-20260101-prog0006",
        orch_run_id="TTC_prog_ckpt",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
        },
        checkpoints={
            # State file hasn't re-saved yet, but the checkpoint completed.
            "heading_judge": {
                "status": "completed",
                "started_at": "2026-01-01T00:02:00",
                "completed_at": "2026-01-01T00:02:10",
            },
            # …and staging is in flight right now.
            "staging": {
                "status": "started",
                "started_at": "2026-01-01T00:02:11",
                "completed_at": None,
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    assert _phase(payload, "heading_judge")["state"] == "done"
    assert _phase(payload, "heading_judge")["wallclock_s"] == 10.0
    assert payload["current_phase"] == "staging"
    assert payload["stats"]["phase_elapsed_s"] is not None


# ----------------------------------------------------------------- endpoint


@pytest.fixture
def client(state_dir, libv2_root):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_endpoint_happy_path(client, state_dir):
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-http",
        workflow_id="WF-20260101-prog0007",
        orch_run_id="TTC_prog_http",
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    resp = client.get(f"/api/runs/{run_id}/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["workflow"] == "textbook_to_course"
    assert {"name", "index", "state", "group", "wallclock_s"} <= set(
        body["phases"][0].keys()
    )
    assert body["current_phase"] == "heading_judge"
    assert "updated_at" in body and "stats" in body


def test_endpoint_unknown_run_404(client):
    resp = client.get("/api/runs/GUI-no-such-run/progress")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_run", "detail": "GUI-no-such-run"}
