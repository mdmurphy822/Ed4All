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
        lambda slug, query, engine, library_wide=None: {
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

    def stub(slug, query, engine, library_wide=None):
        seen["engine"] = engine
        seen["library_wide"] = library_wide
        return {"ask_id": "ASK-e", "status": "pending", "queue_position": 0}

    monkeypatch.setattr(ask_jobs, "submit", stub)
    client.post(
        "/api/learn/ask-jobs",
        json={"slug": "phys-101", "query": "q", "engine": "semantic"},
    )
    assert seen["engine"] == "semantic"
    assert seen["library_wide"] is None


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
        lambda slug, query, engine, library_wide=None: {
            "ask_id": "ASK-s",
            "status": "pending",
            "queue_position": 0,
        },
    )
    app = create_app(mode="studio")
    sc = TestClient(app)
    resp = sc.post("/api/learn/ask-jobs", json={"slug": "phys-101", "query": "q"})
    assert resp.status_code == 200
    assert resp.json()["ask_id"] == "ASK-s"


# ------------------------------------------------ L4 poll passages / L3 library-wide


def test_poll_running_surfaces_passages(client, monkeypatch):
    """A running job carrying disclosed passages surfaces them on the poll (L4)."""
    passages = [{"chunk_id": "c1", "snippet": "hello", "score": 1.0, "course_slug": ""}]
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {
            "ask_id": ask_id,
            "status": "running",
            "queue_position": 0,
            "passages": passages,
            "passages_refused": False,
        },
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["passages"] == passages
    assert body["passages_refused"] is False


def test_poll_running_without_passages_omits_field(client, monkeypatch):
    """Byte-identical default: no passages on the record → no passages key."""
    monkeypatch.setattr(
        ask_jobs,
        "status",
        lambda ask_id: {"ask_id": ask_id, "status": "running", "queue_position": 0},
    )
    resp = client.get("/api/learn/ask-jobs/ASK-x")
    assert resp.status_code == 200
    body = resp.json()
    assert "passages" not in body


def test_submit_threads_explicit_library_wide(client, monkeypatch):
    seen = {}

    def stub(slug, query, engine, library_wide=None):
        seen["library_wide"] = library_wide
        return {"ask_id": "ASK-lw", "status": "pending", "queue_position": 0}

    monkeypatch.setattr(ask_jobs, "submit", stub)
    resp = client.post(
        "/api/learn/ask-jobs",
        json={"slug": "phys-101", "query": "q", "library_wide": True},
    )
    assert resp.status_code == 200
    assert seen["library_wide"] is True


# ------------------------------------------------------------- ask-capabilities (L3)


def test_ask_capabilities_reports_single_course_not_eligible(client, monkeypatch):
    from gui.services import answer_service

    # A libv2 root whose courses dir has zero/one indexed course → not eligible.
    monkeypatch.setattr(answer_service, "_libv2_root", lambda: __import__("pathlib").Path("/nonexistent"))
    resp = client.get("/api/learn/ask-capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["library_wide_eligible"] is False
    assert body["indexed_course_count"] == 0
    assert "library_wide_default" in body


def test_ask_capabilities_eligible_when_two_indexed(client, monkeypatch, tmp_path):
    from gui.services import answer_service

    courses = tmp_path / "courses"
    for slug in ("a", "b"):
        (courses / slug / "vector_index").mkdir(parents=True)
        (courses / slug / "vector_index" / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(answer_service, "_libv2_root", lambda: tmp_path)
    resp = client.get("/api/learn/ask-capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["indexed_course_count"] == 2
    assert body["library_wide_eligible"] is True


# --------------------------------------------------------------- answer feedback (I6)


def test_feedback_up_fans_out_to_event_log(client, monkeypatch):
    seen = {}

    def _cap(source, kind, payload):
        seen["source"] = source
        seen["kind"] = kind
        seen["payload"] = payload
        return {"seq": 0}

    monkeypatch.setattr(learn_router.shared_state, "append_event", _cap)
    resp = client.post(
        "/api/learn/feedback",
        json={"slug": "phys-101", "ask_id": "ASK-1", "verdict": "up", "comment": "nice"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "verdict": "up"}
    assert seen["source"] == "gui"
    assert seen["kind"] == "answer_feedback"
    assert seen["payload"]["verdict"] == "up"
    assert seen["payload"]["comment"] == "nice"
    assert seen["payload"]["ask_id"] == "ASK-1"


def test_feedback_rejects_unknown_verdict(client):
    resp = client.post(
        "/api/learn/feedback", json={"slug": "phys-101", "verdict": "maybe"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_verdict"


def test_feedback_comment_is_size_capped(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        learn_router.shared_state,
        "append_event",
        lambda s, k, p: seen.setdefault("payload", p) or {"seq": 0},
    )
    long_comment = "x" * (learn_router._FEEDBACK_COMMENT_MAX + 500)
    resp = client.post(
        "/api/learn/feedback",
        json={"slug": "phys-101", "verdict": "down", "comment": long_comment},
    )
    assert resp.status_code == 200
    assert len(seen["payload"]["comment"]) == learn_router._FEEDBACK_COMMENT_MAX


def test_feedback_writes_course_feedback_jsonl(client, monkeypatch, tmp_path):
    import json as _json
    from gui.services import answer_service

    course_dir = tmp_path / "courses" / "phys-101"
    course_dir.mkdir(parents=True)
    monkeypatch.setattr(answer_service, "_libv2_root", lambda: tmp_path)
    # Silence the event-log arm; only exercise the per-course file.
    monkeypatch.setattr(learn_router.shared_state, "append_event", lambda s, k, p: {"seq": 0})
    resp = client.post(
        "/api/learn/feedback", json={"slug": "phys-101", "verdict": "up"}
    )
    assert resp.status_code == 200
    fpath = course_dir / "feedback.jsonl"
    assert fpath.is_file()
    rec = _json.loads(fpath.read_text(encoding="utf-8").strip())
    assert rec["verdict"] == "up"
    assert rec["slug"] == "phys-101"


def test_feedback_tolerates_absent_course_dir(client, monkeypatch, tmp_path):
    from gui.services import answer_service

    monkeypatch.setattr(answer_service, "_libv2_root", lambda: tmp_path)  # no courses/
    monkeypatch.setattr(learn_router.shared_state, "append_event", lambda s, k, p: {"seq": 0})
    resp = client.post(
        "/api/learn/feedback", json={"slug": "phys-101", "verdict": "down"}
    )
    # No course dir → no file, but the request still succeeds.
    assert resp.status_code == 200
    assert not (tmp_path / "courses" / "phys-101" / "feedback.jsonl").exists()


def test_feedback_rate_limit_returns_429(client, monkeypatch, tmp_path):
    from gui.services import answer_service

    monkeypatch.setattr(learn_router.shared_state, "append_event", lambda s, k, p: {"seq": 0})
    monkeypatch.setattr(answer_service, "_libv2_root", lambda: tmp_path)  # no courses/
    # Reset the shared limiter window, then exhaust it deterministically.
    learn_router._feedback_hits.clear()
    try:
        for _ in range(learn_router._FEEDBACK_RATE_MAX):
            ok = client.post(
                "/api/learn/feedback", json={"slug": "phys-101", "verdict": "up"}
            )
            assert ok.status_code == 200
        blocked = client.post(
            "/api/learn/feedback", json={"slug": "phys-101", "verdict": "up"}
        )
        assert blocked.status_code == 429
        assert blocked.json()["error"] == "rate_limited"
    finally:
        learn_router._feedback_hits.clear()  # leave the limiter clean for others
