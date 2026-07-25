"""Endpoint + service tests for the in-process ``ed4all doctor`` health surface.

Hermetic by construction: the diagnostics registry run is monkeypatched with
canned ``CheckResult``s and ``_bootstrap`` is stubbed to a no-op, so NO torch /
GPU / docker / seat / network probe is ever touched. These tests exercise the
GUI adapter contract only — the group-agnostic payload shape, the severity
rollup, the TTL cache + forced refresh, auth parity with the sibling open
endpoints, the run-scoped post-mortem resolution (GUI id / WF id / unknown),
per-run cache keying, and the never-raise degrade. Skipped without the ``gui``
extra.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui import auth  # noqa: E402
from gui import shared_state  # noqa: E402
from gui.app import create_app  # noqa: E402
from gui.services import health_service  # noqa: E402
from lib.diagnostics import CheckResult, Severity  # noqa: E402


# ------------------------------------------------------------------ fixtures
def _cr(name, group, severity, *, summary=None, remediation="", detail="", data=None):
    return CheckResult(
        name=name,
        group=group,
        severity=severity,
        summary=summary or f"{name} says something",
        remediation=remediation,
        detail=detail,
        data=data or {},
    )


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Stub the registry bootstrap + reset the cache around every test."""
    monkeypatch.setattr(health_service, "_bootstrap", lambda: None)
    health_service.reset_cache()
    yield
    health_service.reset_cache()


@pytest.fixture
def client(state_dir):
    """A TestClient over a fresh full-mode app with isolated state."""
    return TestClient(create_app())


def _install_checks(monkeypatch, results, *, counter=None):
    """Monkeypatch ``run_checks`` to return canned results (optionally counting)."""

    def _fake(ctx, groups=None):
        if counter is not None:
            counter.append((groups, getattr(ctx, "run_id", None)))
        return list(results)

    monkeypatch.setattr(health_service, "run_checks", _fake)


# ------------------------------------------------- GET /api/health/doctor
def test_doctor_payload_group_agnostic_shape(client, monkeypatch):
    # Two groups, one a NOVEL name the code never enumerates → still rendered.
    results = [
        _cr("gpu_fit", "gpu", Severity.OK),
        _cr("win", "window", Severity.WARN, remediation="widen the window", detail="ctx=4096"),
        _cr("future_check", "future_group", Severity.INFO),
    ]
    _install_checks(monkeypatch, results)
    resp = client.get("/api/health/doctor")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) >= {"generated_at", "exit_state", "groups", "summary"}
    # Groups reflect exactly what the backend reported, in order — group-agnostic.
    assert [g["group"] for g in body["groups"]] == ["gpu", "window", "future_group"]
    # Checks are CheckResult-shaped dicts.
    win = body["groups"][1]["checks"][0]
    assert win["name"] == "win" and win["severity"] == "warn"
    assert win["remediation"] == "widen the window" and win["detail"] == "ctx=4096"
    # Summary counts.
    assert body["summary"] == {
        "total": 3, "ok": 1, "info": 1, "warn": 1, "fail": 0,
        "verdict": body["verdict"],
    }


@pytest.mark.parametrize(
    "sev, state",
    [
        (Severity.OK, "healthy"),
        (Severity.INFO, "healthy"),  # INFO never escalates the verdict
        (Severity.WARN, "degraded"),
        (Severity.FAIL, "critical"),
    ],
)
def test_severity_rollup(client, monkeypatch, sev, state):
    _install_checks(monkeypatch, [_cr("c", "gpu", Severity.OK), _cr("d", "gpu", sev)])
    body = client.get("/api/health/doctor").json()
    assert body["exit_state"] == state


def test_ttl_cache_and_forced_refresh(client, monkeypatch):
    calls: list = []
    _install_checks(monkeypatch, [_cr("c", "gpu", Severity.OK)], counter=calls)
    client.get("/api/health/doctor")
    client.get("/api/health/doctor")
    assert len(calls) == 1  # second GET served from the TTL cache
    resp = client.post("/api/health/doctor/refresh")
    assert resp.status_code == 200
    assert len(calls) == 2  # force-refresh bypasses the cache


def test_check_compute_failure_still_yields_payload(client, monkeypatch):
    def _boom(ctx, groups=None):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(health_service, "run_checks", _boom)
    resp = client.get("/api/health/doctor")
    assert resp.status_code == 200  # never raises — degrades to a valid payload
    body = resp.json()
    assert body["exit_state"] == "degraded"
    names = [c["name"] for g in body["groups"] for c in g["checks"]]
    assert "health_service_error" in names


# --------------------------------------------------- group selection (unit)
def test_group_selection_excludes_provider_postmortem_and_gates_seat(monkeypatch):
    pairs = [
        ("gpu", None), ("window", None), ("provider", None),
        ("postmortem", None), ("seat", None), ("future_x", None),
    ]
    monkeypatch.setattr(health_service, "registered_checks", lambda: list(pairs))
    # No seat env → seat gated out; provider/postmortem always excluded; a
    # novel group auto-runs.
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    monkeypatch.delenv("ED4ALL_VLLM_CONTAINERS", raising=False)
    monkeypatch.delenv("ED4ALL_SEAT_SCHEDULE", raising=False)
    monkeypatch.delenv("ED4ALL_VLLM_CONTAINER_LIFECYCLE", raising=False)
    assert health_service._select_groups() == ["gpu", "window", "future_x"]
    # Seat registry env present → the seat group is included.
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-super=http://localhost:8001/v1")
    assert health_service._select_groups() == ["gpu", "window", "seat", "future_x"]


# ------------------------------------------------------------- auth parity
def test_health_doctor_is_not_operator_classified():
    # Parity with /api/health, /api/runs — open (the Dashboard is the product).
    assert auth.is_operator_path("/api/health/doctor") is False
    assert auth.is_operator_path("/api/health/doctor/run/WF-1") is False


def test_health_doctor_open_even_with_token(state_dir, monkeypatch):
    _install_checks(monkeypatch, [_cr("c", "gpu", Severity.OK)])
    monkeypatch.setenv("ED4ALL_GUI_TOKEN", "sekret")
    gated = TestClient(create_app())
    # An operator route 401s without the token; the health doctor stays open.
    assert gated.get("/api/courses").status_code == 401
    assert gated.get("/api/health/doctor").status_code == 200


# --------------------------------------------------------------- serve modes
def test_studio_mode_mounts_health(state_dir, monkeypatch):
    _install_checks(monkeypatch, [_cr("c", "gpu", Severity.OK)])
    studio = TestClient(create_app(mode="studio"))
    assert studio.get("/api/health/doctor").status_code == 200


def test_learner_mode_does_not_mount_health(state_dir):
    learner = TestClient(create_app(mode="learner"))
    assert learner.get("/api/health/doctor").status_code == 404


# ------------------------------------------ run-scoped post-mortem endpoint
_PM_OK = [
    CheckResult(
        name="postmortem_phase_course_planning",
        group="postmortem",
        severity=Severity.OK,
        summary="phase course_planning completed",
    ),
]


def _seed_gui_run(state_dir, gui_run_id, orch_run_id, *, status="failed"):
    """Register a GUI run record whose params.run_id names the orchestrator dir."""
    shared_state.register_run(
        {
            "run_id": gui_run_id,
            "workflow_id": "WF-20260722-abcd1234",
            "workflow": "textbook_to_course",
            "status": status,
            "params": {"run_id": orch_run_id, "course_name": "demo"},
        }
    )
    (state_dir / "runs" / orch_run_id).mkdir(parents=True, exist_ok=True)


def _seed_workflow_file(state_dir, workflow_id, orch_run_id, *, status="failed"):
    """Write a CLI-style ``state/workflows/<id>.json`` naming the orchestrator dir."""
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "id": workflow_id,
                "type": "textbook_to_course",
                "status": status,
                "params": {"run_id": orch_run_id, "course_name": "demo"},
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "runs" / orch_run_id).mkdir(parents=True, exist_ok=True)


def test_run_doctor_resolves_gui_record(client, state_dir, monkeypatch):
    _install_checks(monkeypatch, _PM_OK)
    _seed_gui_run(state_dir, "GUI-20260722-aa11bb", "run-orch-1", status="failed")
    resp = client.get("/api/health/doctor/run/GUI-20260722-aa11bb")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == "GUI-20260722-aa11bb"
    assert body["orchestrator_run_id"] == "run-orch-1"
    assert body["effective_status"] == "failed"  # terminal passes through
    assert body["usage"] == {"present": False, "rows": 0}
    # Post-mortem checks are folded into the same group-shaped payload.
    assert body["groups"][0]["group"] == "postmortem"


def test_run_doctor_resolves_workflow_id(client, state_dir, monkeypatch):
    counter: list = []
    _install_checks(monkeypatch, _PM_OK, counter=counter)
    _seed_workflow_file(state_dir, "WF-20260722-ffff9999", "run-orch-2")
    resp = client.get("/api/health/doctor/run/WF-20260722-ffff9999")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["orchestrator_run_id"] == "run-orch-2"
    # The post-mortem group ran against the RESOLVED orchestrator run id.
    assert counter[-1] == (["postmortem"], "run-orch-2")


def test_run_doctor_unknown_run_is_404(client, monkeypatch):
    _install_checks(monkeypatch, _PM_OK)
    resp = client.get("/api/health/doctor/run/WF-nope-nothing")
    assert resp.status_code == 404
    assert resp.json()["error"] == "run_not_found"


def test_run_doctor_usage_probe_counts_rows(client, state_dir, monkeypatch):
    _install_checks(monkeypatch, _PM_OK)
    _seed_gui_run(state_dir, "GUI-20260722-usage1", "run-orch-3", status="completed")
    usage = state_dir / "runs" / "run-orch-3" / "llm_usage.jsonl"
    usage.write_text('{"a":1}\n{"a":2}\n\n{"a":3}\n', encoding="utf-8")
    body = client.get("/api/health/doctor/run/GUI-20260722-usage1").json()
    assert body["usage"] == {"present": True, "rows": 3}  # blank line ignored


def test_run_doctor_cache_keyed_per_run(client, state_dir, monkeypatch):
    counter: list = []
    _install_checks(monkeypatch, _PM_OK, counter=counter)
    _seed_gui_run(state_dir, "GUI-20260722-k1", "run-orch-k1")
    _seed_gui_run(state_dir, "GUI-20260722-k2", "run-orch-k2")
    client.get("/api/health/doctor/run/GUI-20260722-k1")
    client.get("/api/health/doctor/run/GUI-20260722-k1")  # cached
    client.get("/api/health/doctor/run/GUI-20260722-k2")  # distinct key → computes
    assert len(counter) == 2
    # Force refresh re-runs just that key.
    assert client.post("/api/health/doctor/run/GUI-20260722-k1/refresh").status_code == 200
    assert len(counter) == 3


def test_run_doctor_refresh_unknown_is_404(client, monkeypatch):
    _install_checks(monkeypatch, _PM_OK)
    resp = client.post("/api/health/doctor/run/GUI-nope/refresh")
    assert resp.status_code == 404
    assert resp.json()["error"] == "run_not_found"
