"""HTTP QUERY method contract tests for the retrieval-shaped GUI endpoints.

Covers the ``N1`` integration: the safe + idempotent, body-carrying endpoints
(learner ``/api/learn/ask`` + operator ``/api/retrieval/query``) accept the IETF
QUERY method (RFC 10008) as the canonical method AND keep POST as a deprecated
alias that carries a ``Deprecation`` response header. Custom-method requests are
issued through the httpx-backed ``TestClient.request(<method>, ...)`` seam.

All service seams are stubbed — no model call, no LibV2 read, no network — so the
suite runs fully offline (``HF_HUB_OFFLINE=1``), mirroring ``test_learn_router``.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402
from gui.routers import http_query  # noqa: E402
from gui.services import answer_service, retrieval_service  # noqa: E402


_COURSES = [{"slug": "phys-101", "chunk_count": 3}]


def _grounded(status="answered"):
    """A minimal ``GroundedAnswer.to_dict()``-shaped dict for the ask stub."""
    return {
        "status": status,
        "query": "what is velocity?",
        "course_slug": "phys-101",
        "engine": "lexical",
        "answer_text": "Velocity is a vector." if status == "answered" else None,
        "citations": [],
        "refusal": None,
        "confidence": {},
        "groundedness": None,
        "warnings": [],
        "model_id": None,
        "prompt_version": None,
        "generated_at": "2026-06-09T00:00:00Z",
        "latency_ms": 12.0,
    }


@pytest.fixture
def client(monkeypatch):
    """Full app with the retrieval + answer service seams stubbed."""
    monkeypatch.setattr(retrieval_service, "list_courses", lambda: list(_COURSES))
    monkeypatch.setattr(
        retrieval_service,
        "retrieve",
        lambda slug, query, top_k, filters: [
            {"chunk_id": "c1", "text": "Velocity is a vector.", "score": 0.9}
        ],
    )
    monkeypatch.setattr(
        answer_service, "ask", lambda slug, query, engine, **kw: _grounded("answered")
    )
    monkeypatch.setattr(
        answer_service, "source_materials_enabled", lambda slug: False
    )
    return TestClient(create_app())


# ------------------------------------------------------------- module constant


def test_query_methods_constant_is_query_then_post():
    # QUERY canonical first, POST the deprecated alias — the ordered contract the
    # front-end fallback mirrors.
    assert http_query.QUERY_METHODS == ["QUERY", "POST"]


# --------------------------------------------------------- /api/retrieval/query


def _query_body():
    return {"slug": "phys-101", "query": "what is velocity?", "mode": "bm25"}


def test_retrieval_query_via_QUERY_matches_POST(client):
    q = client.request("QUERY", "/api/retrieval/query", json=_query_body())
    p = client.post("/api/retrieval/query", json=_query_body())
    assert q.status_code == 200
    assert p.status_code == 200
    # Identical payload regardless of method.
    assert q.json() == p.json()
    assert q.json()["results"][0]["chunk_id"] == "c1"


def test_retrieval_query_POST_carries_deprecation_header(client):
    p = client.post("/api/retrieval/query", json=_query_body())
    assert p.status_code == 200
    assert p.headers.get("deprecation") == "true"
    assert 'rel="successor-version"' in (p.headers.get("link") or "")


def test_retrieval_query_QUERY_has_no_deprecation_header(client):
    q = client.request("QUERY", "/api/retrieval/query", json=_query_body())
    assert q.status_code == 200
    # The canonical method is NOT deprecated.
    assert "deprecation" not in {k.lower() for k in q.headers.keys()}


def test_retrieval_query_unknown_method_is_405(client):
    r = client.request("FROB", "/api/retrieval/query", json=_query_body())
    assert r.status_code == 405


def test_retrieval_query_QUERY_empty_query_still_422(client):
    # The shared core runs identically under QUERY: input gates are unchanged.
    r = client.request(
        "QUERY",
        "/api/retrieval/query",
        json={"slug": "phys-101", "query": "  ", "mode": "bm25"},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_query"


def test_retrieval_query_POST_error_path_also_carries_deprecation(client):
    # A typed-error (JSONResponse) POST result is tagged in place too.
    r = client.post(
        "/api/retrieval/query",
        json={"slug": "phys-101", "query": "  ", "mode": "bm25"},
    )
    assert r.status_code == 422
    assert r.headers.get("deprecation") == "true"


# ---------------------------------------------------------------- /api/learn/ask


def _ask_body():
    return {"slug": "phys-101", "query": "what is velocity?"}


def test_learn_ask_via_QUERY_matches_POST(client):
    q = client.request("QUERY", "/api/learn/ask", json=_ask_body())
    p = client.post("/api/learn/ask", json=_ask_body())
    assert q.status_code == 200
    assert p.status_code == 200
    assert q.json() == p.json()
    assert q.json()["answer"]["status"] == "answered"
    assert "html" in q.json()


def test_learn_ask_POST_carries_deprecation_header(client):
    p = client.post("/api/learn/ask", json=_ask_body())
    assert p.status_code == 200
    assert p.headers.get("deprecation") == "true"


def test_learn_ask_QUERY_has_no_deprecation_header(client):
    q = client.request("QUERY", "/api/learn/ask", json=_ask_body())
    assert q.status_code == 200
    assert "deprecation" not in {k.lower() for k in q.headers.keys()}


def test_learn_ask_unknown_method_is_405(client):
    r = client.request("FROB", "/api/learn/ask", json=_ask_body())
    assert r.status_code == 405


def test_learn_ask_QUERY_empty_query_still_422_with_fragment(client):
    r = client.request(
        "QUERY", "/api/learn/ask", json={"slug": "phys-101", "query": "  "}
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_query"
    assert "html" in body  # learner-safe fragment rides along under QUERY too


def test_learn_ask_GET_is_405(client):
    # GET is not in the method set (a body-less read of this endpoint is invalid).
    r = client.get("/api/learn/ask")
    assert r.status_code == 405
