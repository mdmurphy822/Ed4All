"""Router contract tests for ``/api/learn/*`` — happy / refusal / blocked / typed-error.

The answer pipeline is stubbed at the service seam (``answer_service.ask``): no
model call, no LibV2 read. Typed-error mapping is exercised by raising dummy
exception classes whose ``__name__`` matches the real typed errors (the router
matches by class name, so no heavy imports are needed). The learner-only app is
asserted to expose ONLY the learner surface (operator routes 404).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402
from gui.services import answer_service, retrieval_service  # noqa: E402
from gui.routers import learn as learn_router  # noqa: E402


# --------------------------------------------------------------------- fixtures


def _grounded(status, *, answer_text=None, citations=None, slug="phys-101"):
    """A minimal ``GroundedAnswer.to_dict()``-shaped dict for one status."""
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
        "generated_at": "2026-06-09T00:00:00Z",
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
    """A TestClient over the full app with ``_course_exists`` stubbed true."""
    monkeypatch.setattr(
        retrieval_service,
        "list_courses",
        lambda: [{"slug": "phys-101", "chunk_count": 3}],
    )
    return TestClient(create_app())


def _stub_ask(monkeypatch, fn):
    monkeypatch.setattr(answer_service, "ask", fn)
    # The router imports ``answer_service`` as a module reference, so patching
    # the attribute on the module is sufficient.


# ------------------------------------------------------------------------ /courses


def test_courses_delegates_list_courses(client, monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "list_courses",
        lambda: [{"slug": "phys-101", "chunk_count": 3}],
    )
    resp = client.get("/api/learn/courses")
    assert resp.status_code == 200
    assert resp.json() == [{"slug": "phys-101", "chunk_count": 3}]


def test_courses_failure_is_500(client, monkeypatch):
    def boom():
        raise OSError("disk gone")

    monkeypatch.setattr(retrieval_service, "list_courses", boom)
    resp = client.get("/api/learn/courses")
    assert resp.status_code == 500
    assert resp.json()["error"] == "list_courses_failed"


# ---------------------------------------------------------------------------- ask


def test_ask_answered_200_json_and_html(client, monkeypatch):
    _stub_ask(
        monkeypatch,
        lambda slug, query, engine: _grounded(
            "answered", answer_text="Velocity is a vector.", citations=[_CITATION]
        ),
    )
    resp = client.post(
        "/api/learn/ask", json={"slug": "phys-101", "query": "what is velocity?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"answer", "html"}
    assert body["answer"]["status"] == "answered"
    assert 'data-status="answered"' in body["html"]
    assert "Source: Velocity" in body["html"]


@pytest.mark.parametrize(
    "status",
    [
        "refused_low_confidence",
        "refused_not_in_course",
        "blocked_invalid_citation",
        "blocked_citation_gate",
    ],
)
def test_ask_refusal_and_block_are_200(client, monkeypatch, status):
    _stub_ask(monkeypatch, lambda slug, query, engine: _grounded(status))
    resp = client.post(
        "/api/learn/ask", json={"slug": "phys-101", "query": "anything"}
    )
    # refusals/blocks are data, not errors → 200
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]["status"] == status
    assert f'data-status="{status}"' in body["html"]
    assert "Source:" not in body["html"]


def test_ask_empty_query_is_422(client):
    resp = client.post("/api/learn/ask", json={"slug": "phys-101", "query": "   "})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "invalid_query"
    assert "html" in body  # error fragment rides along


def test_ask_unknown_course_is_404(client, monkeypatch):
    _stub_ask(monkeypatch, lambda slug, query, engine: _grounded("answered"))
    resp = client.post(
        "/api/learn/ask", json={"slug": "no-such-course", "query": "x"}
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "course_not_found"


# -------------------------------------------------------------- typed-error map


def _named_exc(name, base=Exception):
    """Build an exception instance whose ``type(...).__name__ == name``."""
    cls = type(name, (base,), {})
    return cls(f"{name} raised")


@pytest.mark.parametrize(
    "exc_name,expected_status,expected_code",
    [
        ("AnswerBackendUnavailable", 503, "answer_backend_unavailable"),
        ("AnswerProviderNotLocal", 503, "answer_provider_not_local"),
        ("AnswerComposeError", 502, "answer_compose_failed"),
        ("InvalidCitationError", 502, "answer_compose_failed"),
        ("SemanticIndexMissing", 503, "semantic_index_unavailable"),
        ("SemanticIndexStale", 503, "semantic_index_unavailable"),
        ("EmbeddingBackendUnavailable", 503, "embedding_backend_unavailable"),
        ("RuntimeError", 503, "engine_unavailable"),
        ("ValueError", 500, "ask_failed"),  # unmapped → generic 500
    ],
)
def test_ask_typed_error_mapping(
    client, monkeypatch, exc_name, expected_status, expected_code
):
    def raiser(slug, query, engine):
        raise _named_exc(exc_name)

    _stub_ask(monkeypatch, raiser)
    resp = client.post(
        "/api/learn/ask", json={"slug": "phys-101", "query": "x"}
    )
    assert resp.status_code == expected_status
    body = resp.json()
    assert body["error"] == expected_code
    # learner-safe rendered fragment rides along; detail is operator-only text
    assert "html" in body and body["html"].startswith("<section")
    assert exc_name in body["detail"]
    # the rendered copy never exposes the raw exception text
    assert exc_name not in body["html"]


def test_backend_down_renders_facilitator_copy(client, monkeypatch):
    def raiser(slug, query, engine):
        raise _named_exc("AnswerBackendUnavailable")

    _stub_ask(monkeypatch, raiser)
    resp = client.post("/api/learn/ask", json={"slug": "phys-101", "query": "x"})
    assert resp.status_code == 503
    assert "The answer engine" in resp.json()["html"]
    assert "facilitator" in resp.json()["html"]


# --------------------------------------------------------- engine pass-through


def test_ask_passes_engine_to_service(client, monkeypatch):
    seen = {}

    def capture(slug, query, engine):
        seen["engine"] = engine
        return _grounded("answered", answer_text="x", citations=[_CITATION])

    _stub_ask(monkeypatch, capture)
    client.post(
        "/api/learn/ask",
        json={"slug": "phys-101", "query": "x", "engine": "semantic"},
    )
    assert seen["engine"] == "semantic"


# --------------------------------------------------------------- /source delegate


def test_source_requires_item_path(client):
    resp = client.get("/api/learn/source/phys-101")
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_item_path"


def test_source_delegates_to_source_page(client, monkeypatch):
    # Drive the router against the frozen ``source_page`` contract by patching
    # the real module's ``render_source_page`` seam (the module is present in the
    # live tree; we never call its heavy resolution path here).
    from gui.services import source_page

    def render_source_page(slug, item_path, libv2_root, fragment=None):
        if item_path == "boom.html":
            raise source_page.SourcePageError(404, "source_page_missing", "no such page")
        return source_page.SourcePageResult(
            html="<!DOCTYPE html><html lang='en'><body><h1>Page</h1></body></html>",
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    monkeypatch.setattr(source_page, "render_source_page", render_source_page)

    resp = client.get("/api/learn/source/phys-101?item_path=ch01/p.html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "<h1>Page</h1>" in resp.text

    miss = client.get("/api/learn/source/phys-101?item_path=boom.html")
    assert miss.status_code == 404
    assert miss.json()["error"] == "source_page_missing"


# -------------------------------------------------------------- learner-only app


def test_learner_only_app_exposes_only_learner_surface(monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "list_courses",
        lambda: [{"slug": "phys-101", "chunk_count": 3}],
    )
    app = create_app(learner_only=True)
    lc = TestClient(app)

    # learner API is mounted
    assert lc.get("/api/learn/courses").status_code == 200
    # liveness probe stays
    assert lc.get("/api/health").status_code == 200
    # operator routes are NOT mounted → 404
    assert lc.get("/api/settings").status_code == 404
    assert lc.get("/api/retrieval/courses").status_code == 404
    assert lc.get("/api/courses").status_code == 404


def test_full_app_mounts_learn_router(client):
    # Smoke: the learn router IS mounted on the full operator app too.
    monkeyless = client.get("/api/learn/courses")
    assert monkeyless.status_code == 200


def test_ask_path_wires_capture_and_disables_groundedness(client, monkeypatch):
    """Exercise the REAL ``answer_service.ask`` against a stubbed pipeline.

    Confirms the service threads a DecisionCapture handle (the existing
    grounded-answer emit sites consume it) AND always calls with
    ``with_groundedness=False`` (D4: the learner path never loads the NLI model).
    """
    import gui.services.answer_service as svc
    import lib.retrieval.library_wide as lw_mod

    seen = {}

    # library-wide flag off (default) => answer_library_question delegates
    # verbatim to answer_course_question on the single course.
    monkeypatch.delenv(lw_mod.ENV_LIBRARY_WIDE, raising=False)

    # Avoid the real LibV2 path resolution + index probe.
    monkeypatch.setattr(svc, "_resolve_engine", lambda engine, root, slug: "lexical")
    monkeypatch.setattr(svc, "_build_capture", lambda slug: ("CAPTURE", seen.setdefault("capture_slug", slug))[0])

    # Patch the grounded-answer boundary at the seam library_wide delegates to
    # (its module-bound ``answer_course_question``). answer_service now routes
    # through ``answer_library_question``; with the flag off it forwards every
    # kwarg (capture, with_groundedness) to this delegate unchanged.
    def fake_answer_course_question(libv2_root, slug, query, **kwargs):
        seen["capture"] = kwargs.get("capture")
        seen["with_groundedness"] = kwargs.get("with_groundedness")

        class _R:
            def to_dict(self_inner):
                return _grounded("answered", answer_text="x", citations=[_CITATION])

        return _R()

    monkeypatch.setattr(lw_mod, "answer_course_question", fake_answer_course_question)

    resp = client.post("/api/learn/ask", json={"slug": "phys-101", "query": "x"})
    assert resp.status_code == 200
    assert seen["capture"] == "CAPTURE"  # capture handle threaded into the pipeline
    assert seen["with_groundedness"] is False  # D4
    assert seen["capture_slug"] == "phys-101"
