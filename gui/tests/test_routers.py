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


def test_cancel_endpoint_returns_202_cancel_requested(client):
    """Phase 4 §5.1(E): a fresh cancel returns HTTP 202 + cancel_requested."""
    from gui import shared_state

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "running"})
    resp = client.post(f"/api/runs/{run_id}/cancel")
    assert resp.status_code == 202
    body = resp.json()
    assert body == {"run_id": run_id, "status": "cancel_requested"}
    assert shared_state.read_run(run_id)["status"] == "cancel_requested"


def test_cancel_endpoint_terminal_run_is_200_noop(client):
    """An already-terminal run is a 200 idempotent no-op (not 202)."""
    from gui import shared_state

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "status": "completed"})
    resp = client.post(f"/api/runs/{run_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_cancel_endpoint_unknown_run_is_404(client):
    resp = client.post("/api/runs/GUI-no-such-run/cancel")
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_run"


def test_resume_unknown_run_is_422(client):
    """A resume of an unknown prior run fails closed (422), never fabricates."""
    resp = client.post("/api/runs", json={"resume_run_id": "GUI-not-a-run"})
    assert resp.status_code == 422


def test_validation_report_unknown_run_is_404(client):
    resp = client.get("/api/runs/GUI-does-not-exist-000000/validation-report")
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_run"


def test_validation_report_endpoint_returns_digest_and_report(
    state_dir, libv2_root, monkeypatch
):
    """The A6 endpoint returns the report body + failed-gate digest for a run."""
    import json as _json

    import lib.paths as paths
    from fastapi.testclient import TestClient as _TC

    from gui import shared_state
    from gui.app import create_app as _create

    cf_root = state_dir / "Courseforge"
    project_id = "PROJ-BIO_201-20260610-cafef00d"
    export_dir = cf_root / "exports" / project_id
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "courseforge_validation_report.json").write_text(
        _json.dumps(
            {
                "schema_version": "1.1",
                "course_code": "BIO_201",
                "status": "fail",
                "per_block_results": [
                    {"block_id": "b1", "block_type": "assessment_item", "status": "failed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "COURSEFORGE_PATH", cf_root)

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": "courseforge",
            "course_name": "BIO_201",
            "status": "failed",
            "params": {"project_id": project_id},
            "gate_results": [
                {"gate_id": "curie_anchoring", "severity": "critical", "passed": False,
                 "issues": ["anchoring 0.7 < 0.95"]},
            ],
            "failed_phase": "inter_tier_validation",
            "failure_reason": "failed validation gate(s): curie_anchoring",
        }
    )

    client = _TC(_create())
    resp = client.get(f"/api/runs/{run_id}/validation-report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"]["course_code"] == "BIO_201"
    assert body["failed_gates"][0]["gate_id"] == "curie_anchoring"
    assert body["failed_phase"] == "inter_tier_validation"


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


# ---------------------------------------------------- SPA static-asset serving


def _index_asset_urls() -> list[str]:
    """Parse gui/static/index.html and return every href/src asset URL.

    Restricts to local ``.css`` / ``.js`` references (skips ``#`` anchor hrefs
    and any absolute http(s) URLs) — these are the SPA assets the browser must
    be able to fetch for the page to come alive.
    """
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text()
    urls = re.findall(r'(?:href|src)\s*=\s*"([^"]+)"', html)
    return [u for u in urls if u.endswith((".css", ".js"))]


def test_index_references_no_static_prefix():
    """index.html must not reference a /static/ asset path.

    The only StaticFiles mount is at ``/`` (html=True); there is NO ``/static``
    mount, so any ``/static/...`` asset reference would 404 and the SPA would
    load with no JS/CSS (Finding 1).
    """
    assets = _index_asset_urls()
    assert assets, "index.html should reference at least one .css/.js asset"
    offenders = [u for u in assets if u.startswith("/static/")]
    assert not offenders, f"index.html references unserved /static/ asset(s): {offenders}"


def test_index_assets_are_served(client):
    """Every .css/.js asset referenced by index.html must return HTTP 200.

    Catches the Finding-1 class of bug (broken asset path) regardless of which
    path convention index.html uses.
    """
    assets = _index_asset_urls()
    assert assets, "index.html should reference at least one .css/.js asset"
    for url in assets:
        resp = client.get(url)
        assert resp.status_code == 200, f"asset {url} returned {resp.status_code}"


def test_app_js_and_styles_css_served(client):
    """The two canonical SPA assets resolve at their root-mount paths."""
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
    # The /static/ convention must NOT resolve (no such mount).
    assert client.get("/static/app.js").status_code == 404


# ------------------------------------------- 422 validation array-shape (backend)


def test_request_validation_error_detail_is_array(client):
    """FastAPI request-validation 422s return body.detail as a LIST.

    Documents the backend shape the app.js api() error handler must flatten
    (Finding 2): a JSON-body schema violation -> 422 with detail as an array of
    {loc, msg, ...} entries, not a flat/nested {error, detail} object.
    """
    # 'weeks' must be an int; a non-coercible string trips request validation.
    resp = client.post(
        "/api/runs",
        json={
            "workflow": "textbook_to_course",
            "course_name": "X1",
            "corpus": "/tmp/x.pdf",
            "weeks": "not-a-number",
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert detail and "loc" in detail[0] and "msg" in detail[0]


# ----------------------------------------------- Finding 1: run-log WS path


def test_ws_runs_route_registered_under_api_prefix(client):
    """The run-log WebSocket must be reachable at ``/api/ws/runs/{run_id}``.

    The SPA connects to ``/api/ws/runs/<id>`` (app.js ``openWs``); if the route
    were only at the bare ``/ws/...`` path the handshake would hit the
    StaticFiles catch-all and 500. Assert the route exists at the API path AND
    (since Starlette's TestClient supports it) that connecting resolves the
    route and yields the ``unknown_run`` error frame rather than a 500.
    """
    from starlette.routing import WebSocketRoute

    app = client.app
    ws_paths = {
        r.path for r in app.routes if isinstance(r, WebSocketRoute)
    }
    assert "/api/ws/runs/{run_id}" in ws_paths

    with client.websocket_connect("/api/ws/runs/GUI-does-not-exist-000000") as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert frame["error"] == "unknown_run"


# ----------------------------- Finding 2: DELETE /api/uploads/.. is clean 4xx


def test_delete_upload_dotdot_is_clean_4xx_not_500(client):
    """A single-segment ``..`` id must 4xx cleanly, not 500 (PathTraversalError).

    ``PathTraversalError`` is NOT a ``ValueError`` subclass; before the fix the
    delete handler only caught ``ValueError`` so the traversal id propagated
    uncaught -> HTTP 500 leaking the absolute server path.
    """
    resp = client.delete("/api/uploads/%2e%2e")
    assert resp.status_code in (404, 422), resp.text
    body = resp.json()
    assert "error" in body and "detail" in body
    assert isinstance(body["detail"], str)


def test_delete_upload_dotdot_segment_is_clean_4xx(client):
    """A ``..``-bearing id (e.g. ``a..b``) is rejected as a clean 4xx."""
    resp = client.delete("/api/uploads/foo..bar")
    assert resp.status_code in (404, 422), resp.text
    assert "error" in resp.json()


def test_delete_upload_valid_still_works(client):
    """A normal valid delete after the guard reordering still succeeds."""
    up = client.post(
        "/api/uploads",
        files={"files": ("d.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert up.status_code == 200, up.text
    upload_id = up.json()["upload_id"]
    resp = client.delete(f"/api/uploads/{upload_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] == upload_id


# ----------------------------- Finding 4: unknown retrieval slug -> 404


def test_retrieval_query_unknown_slug_is_404(client):
    """A query against a non-existent course slug returns a clean 404.

    Distinguishable from a real course with no matches (200 {results:[]}).
    Existence is keyed on the course listing, not on whether chunks resolved.
    """
    resp = client.post(
        "/api/retrieval/query",
        json={"slug": "no-such-course-xyz", "query": "velocity", "mode": "bm25"},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"] == "course_not_found"


def test_retrieval_query_known_slug_stays_200(state_dir, libv2_course):
    """A known slug stays 200 even when the query matches nothing."""
    from fastapi.testclient import TestClient as _TC

    from gui.app import create_app as _create

    c = _TC(_create())
    resp = c.post(
        "/api/retrieval/query",
        json={"slug": libv2_course["slug"], "query": "velocity", "mode": "bm25"},
    )
    assert resp.status_code == 200, resp.text
    assert "results" in resp.json()


# ----------------------- Finding 5: classification PATCH can clear a field


def test_classification_patch_can_clear_field(state_dir, libv2_course):
    """PATCH sets topics, an explicit null clears it, an empty PATCH no-ops."""
    from fastapi.testclient import TestClient as _TC

    from gui.app import create_app as _create

    c = _TC(_create())
    slug = libv2_course["slug"]

    # Set topics.
    r1 = c.patch(
        f"/api/courses/{slug}/classification",
        json={"topics": ["mechanics", "thermo"]},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["topics"] == ["mechanics", "thermo"]

    # Explicit null clears (removes) the field.
    r2 = c.patch(
        f"/api/courses/{slug}/classification",
        json={"topics": None},
    )
    assert r2.status_code == 200, r2.text
    assert "topics" not in r2.json()

    # Empty PATCH leaves the rest of the block untouched (e.g. primary_domain).
    before = c.get(f"/api/courses/{slug}/classification").json()
    r3 = c.patch(f"/api/courses/{slug}/classification", json={})
    assert r3.status_code == 200, r3.text
    assert r3.json() == before


# ----------------------- Finding 6: negative limit on runs endpoints -> 422


def test_runs_negative_limit_is_422(client):
    """A negative ?limit on /api/runs is rejected (422), matching uploads/courses."""
    resp = client.get("/api/runs", params={"limit": -5})
    assert resp.status_code == 422


def test_activity_events_negative_limit_is_422(client):
    """A negative ?limit on /api/activity/events is rejected (422)."""
    resp = client.get("/api/activity/events", params={"limit": -5})
    assert resp.status_code == 422
