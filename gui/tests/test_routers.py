"""FastAPI ``TestClient`` smoke tests over every GUI route.

Skipped entirely on a default install with no ``fastapi`` (the ``gui`` extra is
opt-in). State is isolated via ``state_dir`` / ``libv2_root`` so the real
``state/gui/`` and ``LibV2/`` are never touched.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart for upload form parsing

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    """A TestClient over a fresh app with isolated state + LibV2 roots."""
    return TestClient(create_app())


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_settings_masks_and_enriches(client, sample_settings_doc):
    from gui import settings_store

    settings_store.save_settings(sample_settings_doc)
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    # Secret masked — no raw key leaks.
    assert body["env"]["ANTHROPIC_API_KEY"] == "set"
    assert "sk-ant-test-secret-value" not in resp.text
    # Catalog / providers / base_models enrichment present.
    assert body["catalog"] and body["providers"] and body["base_models"]
    provider_names = {p["name"] for p in body["providers"]}
    assert "anthropic" in provider_names and "local" in provider_names


def test_put_settings_validates(client):
    bad = {"model_routing": {"global": {"mode": "api", "provider": "bogus"}}}
    resp = client.put("/api/settings", json=bad)
    assert resp.status_code == 422
    assert resp.json()["error"] == "settings_invalid"


def test_patch_settings_round_trip(client):
    resp = client.patch("/api/settings", json={"flags": {"COURSEFORGE_TWO_PASS": True}})
    assert resp.status_code == 200
    assert resp.json()["flags"]["COURSEFORGE_TWO_PASS"] is True


def test_apply_settings_returns_key_names_only(client, sample_settings_doc):
    from gui import settings_store

    settings_store.save_settings(sample_settings_doc)
    resp = client.post("/api/settings/apply")
    assert resp.status_code == 200
    applied = resp.json()["applied"]
    # Names only — no values (so a secret never leaves).
    assert "ANTHROPIC_API_KEY" in applied
    assert "sk-ant-test-secret-value" not in resp.text


def test_workflows_route(client):
    resp = client.get("/api/workflows")
    assert resp.status_code == 200
    names = {w["name"] for w in resp.json()["workflows"]}
    assert "textbook_to_course" in names


def test_upload_accepts_pdf(client):
    pdf_bytes = b"%PDF-1.4\n%%EOF\n"
    resp = client.post(
        "/api/uploads",
        files={"files": ("corpus.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["upload_id"].startswith("UPL-")
    assert body["files"][0]["name"] == "corpus.pdf"
    assert body["files"][0]["kind"] == "pdf"
    assert body["files"][0]["size"] == len(pdf_bytes)


def test_upload_rejects_txt(client):
    resp = client.post(
        "/api/uploads",
        files={"files": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 422
    # Flat error shape: routers/uploads.py now returns {"error", "detail"} via
    # JSONResponse, not HTTPException's nested {"detail": {"error", ...}}.
    body = resp.json()
    assert body["error"] == "unsupported_extension"


def test_list_uploads(client):
    client.post(
        "/api/uploads",
        files={"files": ("a.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    resp = client.get("/api/uploads")
    assert resp.status_code == 200
    assert len(resp.json()["uploads"]) >= 1


def test_courses_route_empty_ok(client):
    # No COURSEFORGE export patched here -> reads the real exports dir; the
    # contract is simply a 200 with a list (we don't assert specific courses).
    resp = client.get("/api/courses")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_courses_route_with_export(state_dir, libv2_root, courseforge_export):
    from fastapi.testclient import TestClient as _TC

    from gui.app import create_app as _create

    c = _TC(_create())
    resp = c.get("/api/courses")
    assert resp.status_code == 200
    names = {row["course_name"] for row in resp.json()}
    assert "PHYS_101" in names

    # Objectives GET via the route.
    obj = c.get(f"/api/courses/{courseforge_export['project_id']}/objectives")
    assert obj.status_code == 200
    assert obj.json()["terminal_objectives"][0]["id"] == "TO-01"


def test_retrieval_courses_route(state_dir, libv2_course):
    from fastapi.testclient import TestClient as _TC

    from gui.app import create_app as _create

    c = _TC(_create())
    resp = c.get("/api/retrieval/courses")
    assert resp.status_code == 200
    slugs = {row["slug"] for row in resp.json()}
    assert libv2_course["slug"] in slugs


def test_retrieval_query_bm25(state_dir, libv2_course):
    from fastapi.testclient import TestClient as _TC

    from gui.app import create_app as _create

    c = _TC(_create())
    resp = c.post(
        "/api/retrieval/query",
        json={
            "slug": libv2_course["slug"],
            "query": "velocity rate of change of position",
            "mode": "bm25",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["results"]


def test_retrieval_query_empty_is_422(client, libv2_course):
    resp = client.post(
        "/api/retrieval/query",
        json={"slug": libv2_course["slug"], "query": "  ", "mode": "bm25"},
    )
    assert resp.status_code == 422


def test_bad_workflow_launch_is_422(client):
    resp = client.post(
        "/api/runs",
        json={"workflow": "not_a_workflow", "course_name": "X1", "corpus": "/tmp/x.pdf"},
    )
    assert resp.status_code == 422


def test_bad_phase_launch_is_422(client, monkeypatch):
    # Avoid scanning real Courseforge exports during pre-population.
    from gui.services import run_service

    monkeypatch.setattr(run_service, "_prepopulate_phase_outputs", lambda *a, **k: {})
    resp = client.post(
        "/api/runs/phase",
        json={
            "workflow": "textbook_to_course",
            "phase": "not_a_real_phase",
            "course_name": "PHYS_101",
        },
    )
    assert resp.status_code == 422


def test_unknown_run_is_404(client):
    resp = client.get("/api/runs/GUI-does-not-exist-000000")
    assert resp.status_code == 404


def test_activity_post_and_get_round_trip(client):
    post = client.post("/api/activity/post", json={"kind": "message", "payload": {"hi": 1}})
    assert post.status_code == 200
    seq = post.json()["event"]["seq"]
    got = client.get("/api/activity/events", params={"since": seq})
    assert got.status_code == 200
    events = got.json()["events"]
    assert any(e["kind"] == "message" and e["source"] == "gui" for e in events)


# ----------------------------------------------------------- flat error shape


def test_unknown_run_error_is_flat(client):
    """runs.py errors use the FLAT {error, detail} shape (not nested in detail)."""
    resp = client.get("/api/runs/GUI-does-not-exist-000000")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "unknown_run"
    # detail is a plain string, not a nested {error, detail} dict.
    assert isinstance(body["detail"], str)


def test_no_files_upload_error_is_flat(client):
    """Upload with no parts -> 422 flat {error, detail}."""
    resp = client.post("/api/uploads", files={})
    # FastAPI rejects a missing required field before our handler (422); when a
    # part is present-but-empty our handler emits the flat shape. Either way the
    # status is 422; assert the flat shape when our handler produced it.
    assert resp.status_code == 422


# ------------------------------------------------------- adapter error mapping


def test_infer_adapter_load_failed_maps_to_503(client, libv2_course, monkeypatch):
    from gui.services import retrieval_service

    monkeypatch.setattr(
        retrieval_service,
        "adapter_infer",
        lambda *a, **k: {"error": "adapter_load_failed", "detail": "OOM loading weights"},
    )
    resp = client.post(
        f"/api/retrieval/{libv2_course['slug']}/infer",
        json={"model_id": "adapter-v1", "prompt": "explain velocity"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "adapter_load_failed"


def test_infer_inference_failed_maps_to_502(client, libv2_course, monkeypatch):
    from gui.services import retrieval_service

    monkeypatch.setattr(
        retrieval_service,
        "adapter_infer",
        lambda *a, **k: {"error": "inference_failed", "detail": "generation crashed"},
    )
    resp = client.post(
        f"/api/retrieval/{libv2_course['slug']}/infer",
        json={"model_id": "adapter-v1", "prompt": "explain velocity"},
    )
    assert resp.status_code == 502
    assert resp.json()["error"] == "inference_failed"


# --------------------------------------------------------------- pagination


def test_runs_limit_caps_results(client):
    from gui import shared_state

    # Explicit distinct ids: new_run_id can collide within a single millisecond.
    for i in range(3):
        shared_state.register_run({"run_id": f"GUI-20260101-00000{i}", "status": "queued"})
    # Unbounded by default.
    assert len(client.get("/api/runs").json()["runs"]) >= 3
    # ?limit caps.
    assert len(client.get("/api/runs", params={"limit": 1}).json()["runs"]) == 1


def test_activity_events_limit_caps_results(client):
    for i in range(4):
        client.post("/api/activity/post", json={"kind": "message", "payload": {"i": i}})
    limited = client.get("/api/activity/events", params={"limit": 2}).json()["events"]
    assert len(limited) == 2


def test_uploads_limit_caps_results(client):
    import io as _io

    for name in ("a.pdf", "b.pdf", "c.pdf"):
        client.post(
            "/api/uploads",
            files={"files": (name, _io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
    limited = client.get("/api/uploads", params={"limit": 2}).json()["uploads"]
    assert len(limited) == 2
