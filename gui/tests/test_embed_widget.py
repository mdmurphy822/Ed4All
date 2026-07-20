"""Embed-widget contract: iframe frame-ancestors CSP + query-param course pin.

Covers TRACK-EMBED's OpenOLAT-demo surface on OUR side:

* ``resolve_frame_ancestors`` / ``FrameAncestorsMiddleware`` — the opt-in
  ``ED4ALL_GUI_FRAME_ANCESTORS`` allowlist. Default (unset) = NO header, so the
  ask surface stays framed-by-anyone exactly as today (byte-identical); set = a
  ``Content-Security-Policy: frame-ancestors <allowlist>`` header on every HTTP
  response that doesn't already carry a CSP (the archived-source viewer's own
  restrictive CSP is never clobbered).
* The served learner page ships the embed-widget wiring (``?course=`` pin +
  ``?embed=1`` compact mode) — asserted against the served static assets, the
  way the rest of the suite validates client JS it can't drive headless.
* The ask surface stays OPEN even when the operator token gate is armed (the
  embed widget needs no token — auth scoping is structural, per the design).

Skipped on a default install (no fastapi), mirroring the rest of ``gui/tests``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gui import auth  # noqa: E402
from gui.app import (  # noqa: E402
    FRAME_ANCESTORS_ENV,
    FrameAncestorsMiddleware,
    create_app,
    resolve_frame_ancestors,
)

_CSP = "content-security-policy"


# --------------------------------------------------------------------------- #
# resolve_frame_ancestors — parse-with-fallback
# --------------------------------------------------------------------------- #


def test_frame_ancestors_unset_returns_none(monkeypatch):
    monkeypatch.delenv(FRAME_ANCESTORS_ENV, raising=False)
    assert resolve_frame_ancestors() is None


@pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
def test_frame_ancestors_blank_returns_none(monkeypatch, value):
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, value)
    assert resolve_frame_ancestors() is None


def test_frame_ancestors_single_origin(monkeypatch):
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, "http://lms.example:8080")
    assert resolve_frame_ancestors() == "frame-ancestors http://lms.example:8080"


def test_frame_ancestors_multiple_origins(monkeypatch):
    monkeypatch.setenv(
        FRAME_ANCESTORS_ENV, "'self'   http://lms.example:8080  https://lms.example"
    )
    # Interior whitespace collapses to single spaces (a valid source list).
    assert resolve_frame_ancestors() == (
        "frame-ancestors 'self' http://lms.example:8080 https://lms.example"
    )


def test_frame_ancestors_scrubs_crlf_injection(monkeypatch):
    # A hostile value must not smuggle a second header line via CR/LF.
    monkeypatch.setenv(
        FRAME_ANCESTORS_ENV, "http://ok:8080\r\nSet-Cookie: evil=1"
    )
    resolved = resolve_frame_ancestors()
    assert "\r" not in resolved and "\n" not in resolved
    assert resolved == "frame-ancestors http://ok:8080 Set-Cookie: evil=1"


# --------------------------------------------------------------------------- #
# FrameAncestorsMiddleware — inject when absent, preserve when present
# --------------------------------------------------------------------------- #


def _mini_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(FrameAncestorsMiddleware)

    @app.get("/plain")
    def _plain():  # noqa: ANN202
        return PlainTextResponse("y")

    @app.get("/own-csp")
    def _own():  # noqa: ANN202
        return PlainTextResponse(
            "x", headers={"Content-Security-Policy": "default-src 'none'"}
        )

    return app


def test_middleware_injects_header_when_env_set(monkeypatch):
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, "http://lms.example:8080")
    client = TestClient(_mini_app())
    resp = client.get("/plain")
    assert resp.headers[_CSP] == "frame-ancestors http://lms.example:8080"


def test_middleware_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv(FRAME_ANCESTORS_ENV, raising=False)
    client = TestClient(_mini_app())
    resp = client.get("/plain")
    assert _CSP not in resp.headers  # byte-identical: no header added


def test_middleware_preserves_existing_csp(monkeypatch):
    # The archived-source viewer sends its OWN restrictive CSP — never clobber it.
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, "http://lms.example:8080")
    client = TestClient(_mini_app())
    resp = client.get("/own-csp")
    assert resp.headers[_CSP] == "default-src 'none'"


# --------------------------------------------------------------------------- #
# Serve-mode integration: header present/absent per env, across all modes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["full", "studio", "learner"])
def test_no_csp_header_when_unset_all_modes(state_dir, libv2_root, monkeypatch, mode):
    monkeypatch.delenv(FRAME_ANCESTORS_ENV, raising=False)
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode=mode))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert _CSP not in resp.headers  # default posture: framed-by-anyone


@pytest.mark.parametrize("mode", ["full", "studio", "learner"])
def test_csp_header_present_when_set_all_modes(state_dir, libv2_root, monkeypatch, mode):
    origin = "http://lms.example:8080"
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, origin)
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode=mode))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.headers[_CSP] == f"frame-ancestors {origin}"


def test_learn_page_static_carries_frame_ancestors(state_dir, libv2_root, monkeypatch):
    # The learner page itself (the framed document) carries the header when set.
    origin = "http://lms.example:8080"
    monkeypatch.setenv(FRAME_ANCESTORS_ENV, origin)
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    resp = client.get("/learn/")
    assert resp.status_code == 200
    assert resp.headers[_CSP] == f"frame-ancestors {origin}"


# --------------------------------------------------------------------------- #
# Widget wiring shipped in the served static assets (course pin + embed mode)
# --------------------------------------------------------------------------- #


def test_learn_index_ships_course_field_hook(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    resp = client.get("/learn/index.html")
    assert resp.status_code == 200, resp.text
    # The course field carries the id the embed CSS hides in compact mode.
    assert 'id="course-field"' in resp.text


def test_learn_js_ships_course_pin_and_embed_wiring(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    resp = client.get("/learn/learn.js")
    assert resp.status_code == 200, resp.text
    body = resp.text
    # Query-param course pin + compact embed mode are wired.
    assert "applyCoursePin" in body
    assert 'getParam("course")' in body
    assert 'getParam("embed")' in body
    assert 'classList.add("embed")' in body


def test_learn_css_ships_embed_hide_rules(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    resp = client.get("/learn/learn.css")
    assert resp.status_code == 200, resp.text
    assert "body.embed #course-field" in resp.text
    assert "body.embed #quizzes" in resp.text


# --------------------------------------------------------------------------- #
# Embed-scope auth: the ask surface is OPEN even with the operator gate armed
# --------------------------------------------------------------------------- #


def test_ask_surface_not_operator_classified():
    # Structural: /api/learn/* is never operator-classified, so the embed widget
    # reaches the ask/courses endpoints without any token.
    assert auth.is_operator_path("/api/learn/ask", "QUERY") is False
    assert auth.is_operator_path("/api/learn/ask", "POST") is False
    assert auth.is_operator_path("/api/learn/courses", "GET") is False


def test_ask_surface_open_when_operator_token_armed(state_dir, libv2_root, monkeypatch):
    # With a token configured (operator surface gated), the learner ask surface
    # stays reachable — the embed widget needs no token.
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    monkeypatch.setenv("ED4ALL_GUI_TOKEN", "s3cret-token")
    client = TestClient(create_app())  # full mode installs the token gate
    # Operator surface requires the token...
    assert client.get("/api/courses").status_code == 401
    # ...but the learner ask surface stays open (200 or a backend 503, never 401).
    assert client.get("/api/learn/courses").status_code in (200, 503)
