"""Serve-mode wiring for the end-user Studio surface (``ED4ALL_GUI_MODE=studio``).

The contract: ``create_app(mode="studio")`` (and the ``ED4ALL_GUI_MODE=studio``
env, the channel uvicorn's no-arg import-string factory uses) mounts the Studio
Library + IMSCC viewer API (``/api/library`` + ``/api/courses/*``) and the
learner answer API (``/api/learn/*`` — the C2 ask drawer) PLUS the studio /
shared / learn page subtrees, but NOT the operator routers. Full mode keeps the
operator surface; learner mode is unaffected by the new knob.

Also pins the mode-precedence contract (``_resolve_mode``): explicit arg >
``learner_only`` kwarg > ``ED4ALL_GUI_MODE`` > legacy ``ED4ALL_GUI_LEARNER`` >
default ``full``, with ``ED4ALL_GUI_MODE`` winning over the legacy learner env.

Skipped on a default install (no fastapi), mirroring the rest of ``gui/tests``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import _resolve_mode, create_app  # noqa: E402


# --------------------------------------------------------------------------- #
# _resolve_mode precedence
# --------------------------------------------------------------------------- #


def test_explicit_mode_wins(monkeypatch):
    monkeypatch.setenv("ED4ALL_GUI_MODE", "learner")
    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")
    assert _resolve_mode("studio", False) == "studio"


def test_learner_only_kwarg_maps_to_learner(monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    assert _resolve_mode(None, True) == "learner"


def test_env_mode_studio(monkeypatch):
    monkeypatch.setenv("ED4ALL_GUI_MODE", "studio")
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    assert _resolve_mode(None, False) == "studio"


def test_env_mode_wins_over_legacy_learner(monkeypatch):
    monkeypatch.setenv("ED4ALL_GUI_MODE", "studio")
    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")
    assert _resolve_mode(None, False) == "studio"


def test_legacy_learner_env_still_honored(monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")
    assert _resolve_mode(None, False) == "learner"


def test_default_is_full(monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    assert _resolve_mode(None, False) == "full"


def test_unknown_env_mode_falls_through(monkeypatch):
    monkeypatch.setenv("ED4ALL_GUI_MODE", "bogus")
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    assert _resolve_mode(None, False) == "full"


# --------------------------------------------------------------------------- #
# Studio mode surface (mounted vs not-mounted)
# --------------------------------------------------------------------------- #


def test_studio_mode_exposes_library_and_learn_not_operator(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="studio"))
    # Liveness + Studio API present.
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/library").status_code == 200
    # Learner answer API present (the C2 ask drawer rides it).
    assert client.get("/api/learn/courses").status_code in (200, 503)
    # Studio shell + shared toolkit served.
    assert client.get("/", follow_redirects=False).status_code in (307, 308)
    assert client.get("/studio/").status_code == 200
    assert client.get("/shared/api.js").status_code == 200
    assert client.get("/studio/viewer-shim.js").status_code == 200
    # Studio shell now ships the Create wizard + Settings page.
    assert client.get("/studio/create.js").status_code == 200
    assert client.get("/studio/settings.js").status_code == 200
    # C3: the Create wizard + Studio settings page need upload + run + settings,
    # so studio mode now mounts those surfaces too.
    assert client.get("/api/uploads").status_code == 200
    assert client.get("/api/workflows").status_code == 200
    assert client.get("/api/settings/studio").status_code == 200
    assert client.get("/api/settings").status_code == 200


def test_studio_mode_via_env(state_dir, libv2_root, monkeypatch):
    monkeypatch.setenv("ED4ALL_GUI_MODE", "studio")
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app())  # no kwargs — env-driven (factory path)
    assert client.get("/api/library").status_code == 200
    # C3 wizard surface present under the env-driven factory path too.
    assert client.get("/api/settings/studio").status_code == 200
    assert client.get("/api/uploads").status_code == 200


def test_full_mode_unaffected(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app())
    # Operator surface present...
    assert client.get("/api/settings").status_code == 200
    # ...and the Studio API rides the full app too.
    assert client.get("/api/library").status_code == 200


def test_learner_mode_has_no_library_api(state_dir, libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="learner"))
    # Learner answer API present...
    assert client.get("/api/learn/courses").status_code in (200, 503)
    # ...but the Studio Library API is NOT mounted in learner mode.
    assert client.get("/api/library").status_code == 404
    assert client.get("/api/settings").status_code == 404


def test_learner_mode_serves_shared_kit_assets(state_dir, libv2_root, monkeypatch):
    """The learner page imports the shared design-token + component-kit assets
    via ``/shared/...`` (Phase 6 thinking ring). Those subtree assets must
    resolve in pure learner mode (the page links components.css + tokens.css and
    the script imports ``/shared/components/ring.js``)."""
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(mode="learner"))
    # The learner index is served and loads learn.js as an ES module.
    idx = client.get("/")
    assert idx.status_code == 200, idx.text
    assert '<script type="module" src="/learn/learn.js">' in idx.text
    # Shared assets the page references resolve (200, not the 404 they 'd 404 on
    # without the /shared mount in learner mode).
    assert client.get("/shared/tokens.css").status_code == 200
    assert client.get("/shared/components/components.css").status_code == 200
    assert client.get("/shared/components/ring.js").status_code == 200
    # The presentation-only kit carries NO run-control endpoint into learner
    # mode (the shared mount is StaticFiles only).
    assert client.get("/api/library").status_code == 404
    assert client.get("/api/runs").status_code == 404


# --------------------------------------------------------------------------- #
# CLI / server flag threading (mirrors test_learner_serve_mode patterns)
# --------------------------------------------------------------------------- #


def _patch_server(monkeypatch):
    import sys
    import types

    import gui.server as server

    captured = {"create_app_kwargs": None, "run_kwargs": None}

    def _fake_create_app(*args, **kwargs):
        captured["create_app_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(server, "create_app", _fake_create_app)
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **k: captured.__setitem__("run_kwargs", k)  # noqa: E731
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    return captured


def test_server_mode_studio_threads_kwarg(monkeypatch):
    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    rc = main(["--mode", "studio"])
    assert rc == 0
    assert captured["create_app_kwargs"].get("mode") == "studio"


def test_server_reload_exports_mode_env(monkeypatch):
    import os

    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    rc = main(["--mode", "studio", "--reload"])
    assert rc == 0
    assert os.environ.get("ED4ALL_GUI_MODE") == "studio"
    assert captured["run_kwargs"].get("factory") is True


def _invoke_gui_command(monkeypatch, args):
    import sys
    import types

    from click.testing import CliRunner

    import cli.commands.gui_cmd as gui_cmd

    captured = {"run_kwargs": None}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **k: captured.__setitem__("run_kwargs", k)  # noqa: E731
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    fake_launch = types.ModuleType("gui.launch")
    fake_launch._port_is_free = lambda host, port: True  # noqa: E731
    fake_launch.pick_port = lambda host, port: port  # noqa: E731
    monkeypatch.setitem(sys.modules, "gui.launch", fake_launch)
    return CliRunner().invoke(gui_cmd.gui_command, args), captured


def test_cli_mode_studio_exports_env(monkeypatch):
    import os

    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    result, _ = _invoke_gui_command(monkeypatch, ["--mode", "studio"])
    assert result.exit_code == 0, result.output
    assert os.environ.get("ED4ALL_GUI_MODE") == "studio"
    assert "studio" in result.output.lower()
