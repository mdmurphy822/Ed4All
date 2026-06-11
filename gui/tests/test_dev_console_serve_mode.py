"""Serve-mode wiring for the developer console (Marketable-v1 C4).

Contract: the ``/dev/`` pane shell + its ``dev.js`` / ``dev.css`` assets are
mounted in FULL mode only. Studio + learner serve modes do NOT mount the
operator surface, so ``/dev/`` is absent (404) there. The subtree is
operator-classified in ``gui.auth`` so the D2 token gates it in full mode (gate
behavior is covered in ``test_auth``); here we only assert the mount presence /
absence per serve mode.

Skipped on a default install (no fastapi), mirroring the rest of ``gui/tests``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


def test_dev_console_served_in_full_mode(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_TOKEN", raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    client = TestClient(create_app())  # full mode, open (no token)
    # The pane shell + its assets are served.
    assert client.get("/dev/").status_code == 200
    assert client.get("/dev/dev.js").status_code == 200
    assert client.get("/dev/dev.css").status_code == 200
    # The shell loads the shared ES toolkit via the catch-all / mount.
    assert client.get("/shared/api.js").status_code == 200
    # The legacy SPA + its app.js still survive (the six tabs live inside /dev/
    # AND the legacy root stays functional during the transition).
    assert client.get("/", follow_redirects=False).status_code == 200
    assert client.get("/app.js").status_code == 200


def test_dev_console_absent_in_studio_mode(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    # No operator surface in studio mode → the dev console is not mounted.
    assert client.get("/dev/", follow_redirects=False).status_code == 404
    assert client.get("/dev/dev.js").status_code == 404


def test_dev_console_absent_in_learner_mode(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="learner"))
    assert client.get("/dev/", follow_redirects=False).status_code == 404
    assert client.get("/dev/dev.js").status_code == 404


def test_dev_console_shell_static_shape():
    """The shell ships as a module-script page with the expected pane rail."""
    from pathlib import Path  # noqa: PLC0415

    html = (
        Path(__file__).resolve().parents[1] / "static" / "dev" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'type="module"' in html, "dev shell must load dev.js as an ES module"
    assert 'src="/dev/dev.js"' in html
    # Nine panes (six ported tabs + Run Inspector + Env + Events + API).
    for pane in ("runs", "upload", "settings", "routing", "courses", "retrieval", "env", "events", "api"):
        assert f'data-pane="{pane}"' in html, f"missing nav entry for pane {pane}"


def test_dev_js_node_check():
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    dev_js = Path(__file__).resolve().parents[1] / "static" / "dev" / "dev.js"
    proc = subprocess.run([node, "--check", str(dev_js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_shared_api_js_node_check():
    """The shared api() gained an opt-in error hook — stays syntactically valid."""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    api_js = Path(__file__).resolve().parents[1] / "static" / "shared" / "api.js"
    proc = subprocess.run([node, "--check", str(api_js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_dev_js_wires_panes_and_drawer():
    """The dev console wires the run inspector, error-drawer hook, and panes."""
    from pathlib import Path  # noqa: PLC0415

    js = (
        Path(__file__).resolve().parents[1] / "static" / "dev" / "dev.js"
    ).read_text(encoding="utf-8")
    # Run Inspector reuses the WS + the A6 validation-report endpoint.
    assert "/api/ws/runs/" in js, "run inspector must reuse the run-log WebSocket"
    assert "/validation-report" in js, "run inspector must fetch the A6 validation report"
    assert "failed_phase" in js and "failure_reason" in js
    # Console-wide ApiError drawer registers the shared opt-in hook.
    assert "onApiError" in js, "error drawer must register the shared api() error hook"
    # Six legacy tabs are ported (embedded via the legacy render functions).
    for fn in ("renderUpload", "renderSettings", "renderRouting", "renderCourses", "renderRetrieval"):
        assert fn in js, f"legacy pane {fn} must be wired"
    # Env-catalog editor reuses the settings API; API explorer keeps history.
    assert "/api/settings" in js
    assert "ed4all_dev_api_history" in js
