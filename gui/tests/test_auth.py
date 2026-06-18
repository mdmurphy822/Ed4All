"""Operator token gate — middleware + startup-warning tests (Marketable-v1 D2).

Two tiers:

* Pure-unit tests over ``gui.auth`` (path classification, constant-time compare,
  token resolution) — run with no fastapi installed.
* fastapi-gated tests — the startup-warning helper (it imports ``gui.server`` →
  ``gui.app`` → fastapi) and the ``TestClient`` integration over the gated
  full-mode app — skipped on a default install with no fastapi (mirrors the rest
  of ``gui/tests``).

State is isolated via ``state_dir`` / ``libv2_root`` so the real ``state/gui/``
and ``LibV2/`` are never touched.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gui import auth

# fastapi is the opt-in ``gui`` extra. The pure-unit tier (path classification,
# token resolution, constant-time compare) runs on a default install; the
# startup-warning tests import ``gui.server`` (which pulls ``gui.app`` →
# fastapi), and the gated-app integration tier needs the TestClient — both are
# gated on fastapi. A module-level importorskip would skip the pure-unit tier
# too, so we probe once here and decorate only the fastapi-dependent tests.
_HAS_FASTAPI = True
try:  # pragma: no cover - import probe
    from fastapi.testclient import TestClient  # noqa: E402

    from gui.app import create_app  # noqa: E402
except Exception:  # noqa: BLE001
    _HAS_FASTAPI = False

pytestmark_integration = pytest.mark.skipif(
    not _HAS_FASTAPI, reason="fastapi (gui extra) not installed"
)


# --------------------------------------------------------------------------- #
# Pure-unit: path classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        # GUI-redesign Phase 2: the operator SPA root moved to /advanced/ (the
        # bare ``/`` is now the OPEN Studio shell).
        "/advanced",
        "/advanced/",
        "/advanced/index.html",
        "/advanced/app.js",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/courses",
        "/api/courses/PROJ-x/objectives",
        "/api/retrieval",
        "/api/retrieval/slug/adapters",
        "/api/activity/events",
        "/api/activity/post",
        "/api/ws/runs/RUN-123",
        # The developer console (C4), now /advanced/dev/ — whole subtree gated.
        "/advanced/dev",
        "/advanced/dev/",
        "/advanced/dev/index.html",
        "/advanced/dev/dev.js",
        "/advanced/dev/dev.css",
    ],
)
def test_operator_paths_classified(path):
    assert auth.is_operator_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # The bare ``/`` is now the OPEN Studio shell — NOT operator-classified.
        "/",
        "/index.html",
        "/api/health",
        "/api/settings",
        "/api/settings/studio",
        "/api/uploads",
        "/api/runs",
        "/api/runs/RUN-1",
        "/api/learn/courses",
        "/api/library",
        "/api/workflows",
        "/shared/api.js",
        "/studio/",
        "/learn/index.html",
        # The operator stylesheet stays open under /advanced/ (styles.css posture).
        "/advanced/styles.css",
        # The old root-mounted operator assets no longer exist there — and even
        # if requested they are NOT gated (the bare ``/`` tree is open Studio).
        "/app.js",
        "/styles.css",
        # A path that merely shares the /advanced prefix without the dev boundary
        # is NOT the dev console (no false-positive on /advancement*).
        "/advancement",
        "/advanced-thing/index.html",
    ],
)
def test_non_operator_paths_open(path):
    assert auth.is_operator_path(path) is False


# --------------------------------------------------------------------------- #
# Method-gated destructive routes (D5): DELETE /api/library is operator-gated;
# GET /api/library stays open.
# --------------------------------------------------------------------------- #


def test_delete_library_is_operator_classified():
    assert auth.is_operator_path("/api/library/demo-101", "DELETE") is True
    assert auth.is_operator_path("/api/library", "DELETE") is True
    # Lower-case method still matches.
    assert auth.is_operator_path("/api/library/demo-101", "delete") is True


def test_get_library_stays_open():
    assert auth.is_operator_path("/api/library", "GET") is False
    assert auth.is_operator_path("/api/library/demo-101", "GET") is False
    # No method (path-only) is the open/GET-equivalent classification.
    assert auth.is_operator_path("/api/library/demo-101") is False


# --------------------------------------------------------------------------- #
# Pure-unit: token resolution + constant-time compare
# --------------------------------------------------------------------------- #


def test_resolve_token_env_precedence(monkeypatch):
    monkeypatch.setenv(auth.TOKEN_ENV, "  env-token  ")
    # Env wins over a settings-store token + is stripped.
    assert auth.resolve_token("settings-token") == "env-token"


def test_resolve_token_settings_fallback(monkeypatch):
    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    assert auth.resolve_token("settings-token") == "settings-token"


def test_resolve_token_unset(monkeypatch):
    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    assert auth.resolve_token(None) is None
    assert auth.resolve_token("   ") is None


def test_tokens_match_uses_constant_time(monkeypatch):
    """The compare must route through hmac.compare_digest (timing-safe)."""
    import hmac

    calls = {"n": 0}
    real = hmac.compare_digest

    def _spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", _spy)
    assert auth.tokens_match("secret", "secret") is True
    assert auth.tokens_match("wrong", "secret") is False
    # A missing presented token short-circuits to False WITHOUT a compare.
    assert auth.tokens_match(None, "secret") is False
    assert auth.tokens_match("", "secret") is False
    # The two real comparisons went through compare_digest.
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Pure-unit: startup warning helper
# --------------------------------------------------------------------------- #


@pytestmark_integration
def test_is_loopback_host():
    from gui.server import _is_loopback_host

    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("0.0.0.0") is False
    assert _is_loopback_host("192.168.1.10") is False
    assert _is_loopback_host("") is False
    assert _is_loopback_host("not-an-ip") is False


@pytestmark_integration
def test_startup_warns_when_exposed_without_token(monkeypatch, caplog):
    from gui.server import _warn_if_exposed_without_token

    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    with caplog.at_level("WARNING", logger="gui.server"):
        _warn_if_exposed_without_token("0.0.0.0", "full")
    assert any("UNAUTHENTICATED" in r.message for r in caplog.records)


@pytestmark_integration
def test_startup_no_warning_when_token_set(monkeypatch, caplog):
    from gui.server import _warn_if_exposed_without_token

    monkeypatch.setenv(auth.TOKEN_ENV, "tok")
    with caplog.at_level("WARNING", logger="gui.server"):
        _warn_if_exposed_without_token("0.0.0.0", "full")
    assert not caplog.records


@pytestmark_integration
def test_startup_no_warning_on_loopback(monkeypatch, caplog):
    from gui.server import _warn_if_exposed_without_token

    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    with caplog.at_level("WARNING", logger="gui.server"):
        _warn_if_exposed_without_token("127.0.0.1", "full")
    assert not caplog.records


@pytestmark_integration
def test_startup_no_warning_in_studio_mode(monkeypatch, caplog):
    from gui.server import _warn_if_exposed_without_token

    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    with caplog.at_level("WARNING", logger="gui.server"):
        # Studio/learner modes don't mount the operator surface → no warning.
        _warn_if_exposed_without_token("0.0.0.0", "studio")
        _warn_if_exposed_without_token("0.0.0.0", "learner")
    assert not caplog.records


# --------------------------------------------------------------------------- #
# Integration: gated full-mode app
# --------------------------------------------------------------------------- #


@pytest.fixture
def open_client(state_dir, libv2_root, monkeypatch):
    """Full-mode app with NO token configured → open (current default)."""
    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    return TestClient(create_app())


@pytest.fixture
def gated_client(state_dir, libv2_root, monkeypatch):
    """Full-mode app WITH a token configured → operator paths gated."""
    monkeypatch.setenv(auth.TOKEN_ENV, "s3cret-token")
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    return TestClient(create_app())


@pytestmark_integration
def test_open_mode_operator_paths_unauthenticated(open_client):
    # No token configured → operator paths answer normally (no 401).
    assert open_client.get("/api/courses").status_code in (200, 404, 500)
    # The operator SPA now lives under /advanced/ (open when no token set).
    assert open_client.get("/advanced/", follow_redirects=False).status_code == 200
    assert open_client.get("/openapi.json").status_code == 200
    assert open_client.get("/api/activity/events").status_code == 200
    # The bare ``/`` is the OPEN Studio shell in every token posture.
    assert open_client.get("/", follow_redirects=False).status_code == 200


@pytestmark_integration
def test_gated_operator_path_401_without_token(gated_client):
    resp = gated_client.get("/api/courses")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")
    assert resp.json()["error"] == "unauthorized"


@pytestmark_integration
def test_gated_operator_path_401_wrong_token(gated_client):
    resp = gated_client.get(
        "/api/retrieval/courses", headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401


@pytestmark_integration
def test_gated_operator_path_200_with_token(gated_client):
    resp = gated_client.get(
        "/api/courses", headers={"Authorization": "Bearer s3cret-token"}
    )
    # Authorized → the route runs (whatever real status it returns, NOT 401).
    assert resp.status_code != 401


@pytestmark_integration
def test_gated_advanced_spa_root_and_docs_require_token(gated_client):
    # GUI-redesign Phase 2: the operator SPA root + docs are gated under
    # /advanced/ (the bare ``/`` is now the OPEN Studio shell — see below).
    assert gated_client.get("/advanced", follow_redirects=False).status_code == 401
    assert gated_client.get("/advanced/", follow_redirects=False).status_code == 401
    assert gated_client.get("/advanced/index.html").status_code == 401
    assert gated_client.get("/advanced/app.js").status_code == 401
    assert gated_client.get("/openapi.json").status_code == 401
    # ...but the operator SPA root opens with the token.
    assert (
        gated_client.get(
            "/advanced/", headers={"Authorization": "Bearer s3cret-token"}
        ).status_code
        == 200
    )


@pytestmark_integration
def test_gated_studio_root_stays_open(gated_client):
    # CRITICAL: with a token SET, the bare ``/`` Studio shell stays OPEN (200) —
    # it is the open product surface, NOT operator-classified anymore.
    assert gated_client.get("/", follow_redirects=False).status_code == 200
    # The other open product surfaces stay reachable so the Author surface works.
    assert gated_client.get("/studio/", follow_redirects=False).status_code == 200
    assert gated_client.get("/learn/", follow_redirects=False).status_code == 200
    # The operator stylesheet under /advanced/ stays open (styles.css posture).
    assert gated_client.get("/advanced/styles.css").status_code == 200


@pytestmark_integration
def test_gated_dev_console_requires_token(gated_client):
    # The developer console (C4), now /advanced/dev/, is operator-classified.
    assert gated_client.get("/advanced/dev/", follow_redirects=False).status_code == 401
    assert gated_client.get("/advanced/dev/dev.js").status_code == 401
    assert gated_client.get("/advanced/dev/dev.css").status_code == 401
    # ...and opens with the token.
    ok = gated_client.get(
        "/advanced/dev/", headers={"Authorization": "Bearer s3cret-token"}
    )
    assert ok.status_code == 200


@pytestmark_integration
def test_open_mode_dev_console_unauthenticated(open_client):
    # No token configured → the dev console serves normally (no 401).
    assert open_client.get("/advanced/dev/", follow_redirects=False).status_code == 200
    assert open_client.get("/advanced/dev/dev.js").status_code == 200


@pytestmark_integration
def test_gated_health_and_studio_paths_exempt(gated_client):
    # Health stays open for the Docker healthcheck.
    assert gated_client.get("/api/health").status_code == 200
    # Studio-shared API routers stay open even in full mode (Studio uses them).
    assert gated_client.get("/api/settings").status_code == 200
    assert gated_client.get("/api/workflows").status_code == 200
    assert gated_client.get("/api/runs").status_code == 200
    assert gated_client.get("/api/uploads").status_code == 200
    assert gated_client.get("/api/library").status_code == 200
    assert gated_client.get("/api/learn/courses").status_code in (200, 404, 500)
    # Static shared/studio assets stay open.
    assert gated_client.get("/shared/api.js").status_code == 200
    assert gated_client.get("/shared/tokens.css").status_code == 200


@pytestmark_integration
def test_open_mode_advanced_paths_unauthenticated(open_client):
    # Token UNSET → EVERYTHING is open pass-through, including /advanced/.
    assert open_client.get("/advanced/", follow_redirects=False).status_code == 200
    assert open_client.get("/advanced/app.js").status_code == 200
    assert open_client.get("/advanced/styles.css").status_code == 200
    assert open_client.get("/api/courses").status_code in (200, 404, 500)
    assert open_client.get("/api/activity/events").status_code == 200


@pytestmark_integration
def test_gated_delete_library_requires_token(gated_client):
    # DELETE on the otherwise-open Library is operator-gated (D5).
    resp = gated_client.request("DELETE", "/api/library/demo-101?confirm=demo-101")
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


@pytestmark_integration
def test_gated_delete_library_passes_with_token(gated_client):
    # With the token the middleware lets the DELETE reach the route (404 here:
    # no such synthetic course — but NOT a 401).
    resp = gated_client.request(
        "DELETE",
        "/api/library/demo-101?confirm=demo-101",
        headers={"Authorization": "Bearer s3cret-token"},
    )
    assert resp.status_code != 401


@pytestmark_integration
def test_open_mode_delete_library_unauthenticated(open_client, libv2_root):
    # No token → DELETE answers normally (404 for a missing course, not 401).
    (libv2_root / "courses").mkdir(parents=True, exist_ok=True)
    resp = open_client.request(
        "DELETE", "/api/library/nope-999?confirm=nope-999"
    )
    assert resp.status_code != 401


@pytestmark_integration
def test_gated_ws_rejects_without_token(gated_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with gated_client.websocket_connect("/api/ws/runs/RUN-x"):
            pass
    assert exc.value.code == 4401


@pytestmark_integration
def test_gated_ws_accepts_with_query_token(gated_client):
    # With the right ?token= the WS handshake is accepted (the run is unknown,
    # so the handler sends an error frame then closes — but NOT a 4401 reject).
    with gated_client.websocket_connect(
        "/api/ws/runs/RUN-unknown?token=s3cret-token"
    ) as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert frame["error"] == "unknown_run"


@pytestmark_integration
def test_studio_mode_has_no_gate(state_dir, libv2_root, monkeypatch):
    # Even with a token set, studio mode doesn't mount the operator surface and
    # installs no gate → its routes stay open.
    monkeypatch.setenv(auth.TOKEN_ENV, "s3cret-token")
    client = TestClient(create_app(mode="studio"))
    assert client.get("/api/library").status_code == 200
    assert client.get("/api/settings").status_code == 200


# --------------------------------------------------------------------------- #
# Frontend: app.js stays syntactically valid (node --check)
# --------------------------------------------------------------------------- #


def test_app_js_node_check():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    proc = subprocess.run(
        [node, "--check", str(app_js)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
