"""Run-scoped phase telemetry consumption for the generic stage rail."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gui import shared_state
from gui.services import progress_service


def _seed(
    state_dir: Path,
    *,
    gui_status: str = "running",
    wf_status: str = "RUNNING",
) -> tuple[str, str]:
    gui_run_id = "GUI-telemetry-test"
    workflow_id = "WF-telemetry-test"
    orch_run_id = "RUN_telemetry_test"
    shared_state.register_run(
        {
            "run_id": gui_run_id,
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "workflow_id": workflow_id,
            "course_name": "synthetic-course",
            "status": gui_status,
            "started_at": "2026-01-01T00:00:00",
        }
    )
    wf_path = state_dir / "workflows" / f"{workflow_id}.json"
    wf_path.parent.mkdir(parents=True, exist_ok=True)
    wf_path.write_text(
        json.dumps(
            {
                "id": workflow_id,
                "type": "textbook_to_course",
                "status": wf_status,
                "started_at": "2026-01-01T00:00:00",
                "params": {"run_id": orch_run_id},
                "phase_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    return gui_run_id, orch_run_id


def _snapshot(orch_run_id: str, state: str = "running") -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "phase": "training_synthesis",
        "state": state,
        "run_id": orch_run_id,
        "started_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:01:00Z",
        "elapsed_seconds": 60.0,
        "total_units": 100,
        "completed_units": 25,
        "terminal_units": 25,
        "accepted_count": 20,
        "rejected_count": 7,
        "transient_count": 2,
        "sft_count": 12,
        "dpo_count": 8,
        "provider_results": 30,
        "cached_replays": 8,
        "transient_attempts": 4,
        "recovered_units": 1,
        "exhausted_units": 0,
        "fatal_units": 0,
        "active_workers": 16,
        "max_concurrent": 32,
        "queued_units": 16,
        "in_flight": 32,
        "backpressure": False,
        "throughput_units_per_second": 0.5,
        "eta_seconds": 150.0,
        "checkpoint_timestamp": "2026-01-01T00:00:59Z",
        "stop_requested": False,
        "gate_readiness": "pending",
        "provider": "local",
        "model": "teacher-model",
        "rejection_reasons": {"claim_support": 7},
    }


def _write_snapshot(state_dir: Path, orch_run_id: str, doc: Any) -> None:
    path = (
        state_dir
        / "runs"
        / orch_run_id
        / "telemetry"
        / "training_synthesis.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        doc if isinstance(doc, str) else json.dumps(doc),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _no_seat_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)


def _phase(payload: Dict[str, Any]) -> Dict[str, Any]:
    return next(
        phase
        for phase in payload["phases"]
        if phase["name"] == "training_synthesis"
    )


def test_running_snapshot_drives_current_node_and_metrics(state_dir: Path) -> None:
    run_id, orch_run_id = _seed(state_dir)
    snapshot = _snapshot(orch_run_id)
    _write_snapshot(state_dir, orch_run_id, snapshot)

    payload = progress_service.run_progress(run_id)

    assert payload is not None
    assert payload["current_phase"] == "training_synthesis"
    assert _phase(payload)["state"] == "current"
    assert _phase(payload)["telemetry"] == snapshot
    assert payload["stats"]["phase_telemetry"] == [snapshot]


def test_paused_snapshot_is_paused_and_preserves_last_metrics(
    state_dir: Path,
) -> None:
    run_id, orch_run_id = _seed(
        state_dir, gui_status="paused", wf_status="PAUSED"
    )
    snapshot = _snapshot(orch_run_id, state="paused")
    _write_snapshot(state_dir, orch_run_id, snapshot)

    payload = progress_service.run_progress(run_id)

    assert payload is not None
    node = _phase(payload)
    assert payload["current_phase"] is None
    assert node["state"] == "paused"
    assert node["wallclock_s"] == 60.0
    assert node["telemetry"]["accepted_count"] == 20
    assert payload["stats"]["phase_telemetry"][0]["max_concurrent"] == 32


def test_complete_snapshot_marks_done_and_keeps_metrics(state_dir: Path) -> None:
    run_id, orch_run_id = _seed(state_dir)
    snapshot = _snapshot(orch_run_id, state="complete")
    snapshot["completed_units"] = 100
    snapshot["terminal_units"] = 100
    snapshot["accepted_count"] = 84
    snapshot["gate_readiness"] = "ready"
    _write_snapshot(state_dir, orch_run_id, snapshot)

    payload = progress_service.run_progress(run_id)

    assert payload is not None
    assert _phase(payload)["state"] == "done"
    assert _phase(payload)["telemetry"]["accepted_count"] == 84
    assert payload["stats"]["phase_telemetry"][0]["gate_readiness"] == "ready"


@pytest.mark.parametrize(
    "bad_doc",
    [
        "{not-json",
        {"schema_version": 99},
        {
            **_snapshot("some-other-run"),
            "run_id": "some-other-run",
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "phase": "../training_synthesis",
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "completed_units": -1,
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "completed_units": 101,
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "terminal_units": 26,
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "active_workers": 33,
            "queued_units": 0,
            "in_flight": 33,
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "active_workers": 10,
            "queued_units": 10,
            "in_flight": 19,
        },
        {
            **_snapshot("RUN_telemetry_test"),
            "max_concurrent": 0,
            "active_workers": 0,
            "queued_units": 0,
            "in_flight": 0,
        },
    ],
)
def test_missing_or_malformed_snapshot_is_ignored_fail_safe(
    state_dir: Path, bad_doc: Any
) -> None:
    run_id, orch_run_id = _seed(state_dir)
    _write_snapshot(state_dir, orch_run_id, bad_doc)

    payload = progress_service.run_progress(run_id)

    assert payload is not None
    assert "telemetry" not in _phase(payload)
    assert "phase_telemetry" not in payload["stats"]


def test_frontend_supports_paused_and_generic_phase_telemetry() -> None:
    source = (
        Path(__file__).parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")

    assert "paused: 'Ⅱ'" in source
    assert "phaseTelemetry.forEach" in source
    assert "tm.phase" in source
    assert "training_synthesis" not in source
