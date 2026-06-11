"""Router contract tests for the async ask-jobs endpoints + the ask-ready gate.

The ask-job *store* is stubbed at the ``ask_jobs.submit`` / ``ask_jobs.status``
seam (no worker thread, no model): the router is exercised as a pure typed-error
+ rendering adapter, mirroring how ``test_learn_router`` stubs
``answer_service.ask``. Covers submit (422/404 gates + happy), poll (pending /
done-renders-fragment / error-maps-typed / unknown-404), and the ask-ready gate.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402
from gui.services import ask_jobs, retrieval_service  # noqa: E402
from gui.routers import learn as learn_router  # noqa: E402


def _grounded(status, *, answer_text=None, citations=None, slug="phys-101"):
    return {
        "status": status,
        "query": "what is velocity?",
        "course_slug": slug,
        "engine": "lexical",
        "answer_text": answer_text,
        "citations": citations or [],
        "refusal": None,
        "confidence": {},
        "groundedness": None,
        "warnings": [],
        "model_id": "qwen2.5:7b" if answer_text else None,
        "prompt_version": "v1" if answer_text else None,
        "generated_at": "2026-06-10T00:00:00Z",
        "latency_ms": 12.0,
    }


_CITATION = {
    "chunk_id": "c1",
    "item_path": "ch01/velocity.html",
    "section_heading": "Velocity",
    "module_id": "M1",
    "page_label": "Velocity",
    "anchor_status": "resolved_exact",
    "source_path": "/x.html",
    "text_quote": "Velocity is a vector.",
    "link_target": {
        "kind": "course_page",
        "item_path": "ch01/velocity.html",
        "fragment": {"kind": "heading", "value": "velocity"},
        "char_span": [0, 5],
    },
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "list_courses",
        lambda: [{"slug": "phys-101", "chunk_count": 3}],
    )
    return TestClient(create_app())


# ------------------------------------------------------------------ submit gates


def test_submit_empty_query_is_422(client):
    resp = client.post("/api/learn/ask-jobs", json={"slug": "phys-101", "query": " "})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_query"


def test_submit_unknown_course_is_404(client):
    resp = client.post("/api/learn/ask-jobs", json={"slug": "nope", "query": "x"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_submit_returns_ask_id_and_pending(client, monkeypatch):
    monkeypatch.setattr(
        ask_jobs,
        "submit",
        lambda slug, query, engine: {
            "ask_id": "ASK-test-1",
            "status": "pending",
            "queue_position": 0,
        },
    )
    resp = client.post("/api/learn/ask-jobs", json={"slug": "phys-101", "query": "q"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ask_id": "ASK-test-1", "status": "pending", "queue_position": 0}


def test_submit_passes_engine(client, monkeypatch):
    seen = {}

    def stub(slug, query, engine):
        seen["engine"] = engine
        return {"ask_id": "ASK-e", "status": "pending", "queue_position": 0}

    monkeypatch.setattr(ask_jobs, "submit", stub)
    client.post(
        "/api/learn/ask-jobs",
        json={"slug": "phys-101", "query": "q", "engine": "semantic"},
    )
    assert seen["engine"] == "semantic"


# -------------------------------------------------------------------- poll states


def test_poll_pending_returns_queue_position(client, monkeypatch):
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {"ask_id": ask_id, "status": "pending", "queue_position": 2},
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200
    assert resp.json() == {"ask_id": "ASK-x", "status": "pending", "queue_position": 2}


def test_poll_done_renders_fragment(client, monkeypatch):
    answer = _grounded("answered", answer_text="Velocity is a vector.", citations=[_CITATION])
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {"ask_id": ask_id, "status": "done", "answer": answer},
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["answer"]["status"] == "answered"
    assert 'data-status="answered"' in body["html"]
    assert "Source: Velocity" in body["html"]
    # The citation link points at the in-context source URL the drawer intercepts.
    assert "/api/learn/source/" in body["html"]


def test_poll_done_refusal_renders_no_sources(client, monkeypatch):
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {"ask_id": ask_id, "status": "done", "answer": _grounded("refused_not_in_course")},
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200
    body = resp.json()
    assert 'data-status="refused_not_in_course"' in body["html"]
    assert "Source:" not in body["html"]


@pytest.mark.parametrize(
    "err_name,expected_code,copy_marker",
    [
        ("AnswerBackendUnavailable", "answer_backend_unavailable", "The answer engine"),
        ("SemanticIndexMissing", "semantic_index_unavailable", "facilitator"),
        ("RuntimeError", "engine_unavailable", "facilitator"),
        ("ValueError", "ask_failed", "facilitator"),
    ],
)
def test_poll_error_maps_typed_error(client, monkeypatch, err_name, expected_code, copy_marker):
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {
            "ask_id": ask_id,
            "status": "error",
            "error": err_name,
            "detail": f"{err_name} raised in worker",
        },
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200  # the job *state* is data, not an HTTP error
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] == expected_code
    assert copy_marker in body["html"]
    # Operator-only raw class name is never leaked into the rendered copy.
    assert err_name not in body["html"]
    assert err_name in body["detail"]


def test_poll_unknown_id_is_404(client, monkeypatch):
    monkeypatch.setattr(ask_jobs, "status", lambda ask_id: None)
    resp = client.get("/api/learn/ask-jobs/ASK-missing")
    assert resp.status_code == 404
    assert resp.json()["error"] == "ask_job_not_found"


# -------------------------------------------------------------------- ask-ready


def test_ask_ready_unknown_course_is_404(client):
    resp = client.get("/api/learn/ask-ready/nope")
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


def test_ask_ready_reports_index_presence(client, monkeypatch):
    from gui.services import answer_service

    monkeypatch.setattr(answer_service, "_has_vector_index", lambda root, slug: True)
    resp = client.get("/api/learn/ask-ready/phys-101")
    assert resp.status_code == 200
    assert resp.json() == {"slug": "phys-101", "exists": True, "has_vector_index": True}


def test_ask_ready_no_index(client, monkeypatch):
    from gui.services import answer_service

    monkeypatch.setattr(answer_service, "_has_vector_index", lambda root, slug: False)
    resp = client.get("/api/learn/ask-ready/phys-101")
    assert resp.status_code == 200
    assert resp.json()["has_vector_index"] is False


def test_ask_ready_stat_failure_degrades_to_no_index(client, monkeypatch):
    from gui.services import answer_service

    def boom(root, slug):
        raise OSError("fs gone")

    monkeypatch.setattr(answer_service, "_has_vector_index", boom)
    resp = client.get("/api/learn/ask-ready/phys-101")
    assert resp.status_code == 200
    assert resp.json()["has_vector_index"] is False


# ------------------------------------------------------- mounted on studio + full


def test_ask_jobs_mounted_on_studio_app(monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "list_courses",
        lambda: [{"slug": "phys-101", "chunk_count": 3}],
    )
    monkeypatch.setattr(
        ask_jobs,
        "submit",
        lambda slug, query, engine: {"ask_id": "ASK-s", "status": "pending", "queue_position": 0},
    )
    app = create_app(mode="studio")
    sc = TestClient(app)
    resp = sc.post("/api/learn/ask-jobs", json={"slug": "phys-101", "query": "q"})
    assert resp.status_code == 200
    assert resp.json()["ask_id"] == "ASK-s"
