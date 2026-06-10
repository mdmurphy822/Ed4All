"""Serve-mode wiring for the learner answer surface.

Covers the ``--learner`` flag on both entry points (``python -m gui.server`` and
``ed4all gui``) plus the ``ED4ALL_GUI_LEARNER`` env fallback. The contract: the
flag (or a truthy env) selects the learner-only app surface — for the direct
uvicorn entrypoint that means ``create_app(learner_only=True)``; for the
reload/CLI factory path (uvicorn re-imports the no-arg factory in each worker)
that means ``ED4ALL_GUI_LEARNER=1`` is exported so the factory honors it.

``create_app(learner_only=...)`` is the frozen kwarg owned by the app factory;
these tests stub ``create_app`` and ``uvicorn.run`` so they never start a server
or require a real LibV2/model — they only assert the flag is threaded through.

Skipped on a default install (no ``fastapi``/``uvicorn``), mirroring the rest of
``gui/tests`` (``test_routers.py`` uses the same ``importorskip`` guard).
"""

from __future__ import annotations

import pytest

# gui.server imports create_app from gui.app, which imports fastapi at module
# top; ed4all gui shells out to uvicorn. Skip cleanly without the gui extra.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


# --------------------------------------------------------------------------- #
# gui/server.py  (python -m gui.server)
# --------------------------------------------------------------------------- #


def _patch_server(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub gui.server's create_app + uvicorn.run; return a captures dict."""
    import sys
    import types

    import gui.server as server

    captured: dict = {"create_app_kwargs": None, "run_args": None, "run_kwargs": None}

    def _fake_create_app(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["create_app_kwargs"] = kwargs
        return object()  # a sentinel "app" instance; uvicorn.run is stubbed

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["run_args"] = args
        captured["run_kwargs"] = kwargs

    monkeypatch.setattr(server, "create_app", _fake_create_app)
    # main() does `import uvicorn` lazily; inject a stub module so the import
    # resolves to our fake run() without a real uvicorn server.
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = _fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    return captured


def test_learner_flag_passes_learner_only_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)

    rc = main(["--learner"])

    assert rc == 0
    assert captured["create_app_kwargs"] == {"learner_only": True}


def test_no_learner_flag_defaults_to_full_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)

    rc = main([])

    assert rc == 0
    assert captured["create_app_kwargs"] == {"learner_only": False}


def test_env_learner_default_selects_learner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")

    rc = main([])

    assert rc == 0
    assert captured["create_app_kwargs"] == {"learner_only": True}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nope", False),
    ],
)
def test_env_learner_default_truthiness(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    from gui.server import _env_learner_default

    monkeypatch.setenv("ED4ALL_GUI_LEARNER", value)
    assert _env_learner_default() is expected


def test_env_learner_default_unset_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from gui.server import _env_learner_default

    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    assert _env_learner_default() is False


def test_reload_mode_exports_learner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload mode can't pass kwargs to the import-string factory, so the flag
    must ride ED4ALL_GUI_LEARNER for the reloaded workers."""
    from gui.server import main

    captured = _patch_server(monkeypatch)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)

    rc = main(["--learner", "--reload"])

    assert rc == 0
    # Reload path uses the import-string factory, not an app instance.
    assert captured["create_app_kwargs"] is None
    import os

    assert os.environ.get("ED4ALL_GUI_LEARNER") == "1"
    assert captured["run_kwargs"].get("factory") is True


# --------------------------------------------------------------------------- #
# cli/commands/gui_cmd.py  (ed4all gui)
# --------------------------------------------------------------------------- #


def _invoke_gui_command(monkeypatch: pytest.MonkeyPatch, args: list[str]):
    """Run ``ed4all gui`` via CliRunner with uvicorn + port scan stubbed."""
    import sys
    import types

    from click.testing import CliRunner

    import cli.commands.gui_cmd as gui_cmd

    captured: dict = {"run_kwargs": None}

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **k: captured.__setitem__("run_kwargs", k)  # noqa: E731
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    # gui.launch port helpers are imported lazily inside the command.
    fake_launch = types.ModuleType("gui.launch")
    fake_launch._port_is_free = lambda host, port: True  # noqa: E731
    fake_launch.pick_port = lambda host, port: port  # noqa: E731
    monkeypatch.setitem(sys.modules, "gui.launch", fake_launch)

    result = CliRunner().invoke(gui_cmd.gui_command, args)
    return result, captured


def test_cli_learner_flag_exports_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)

    result, captured = _invoke_gui_command(monkeypatch, ["--learner"])

    import os

    assert result.exit_code == 0, result.output
    assert os.environ.get("ED4ALL_GUI_LEARNER") == "1"
    assert "learner-only" in result.output
    assert captured["run_kwargs"].get("factory") is True


def test_cli_no_learner_flag_does_not_export_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)

    result, _ = _invoke_gui_command(monkeypatch, [])

    import os

    assert result.exit_code == 0, result.output
    assert os.environ.get("ED4ALL_GUI_LEARNER") is None
    assert "operator + learner" in result.output


def test_cli_env_fallback_selects_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")

    result, _ = _invoke_gui_command(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert "learner-only" in result.output


def test_create_app_honors_learner_env_for_factory_paths(monkeypatch):
    """uvicorn import-string factories call create_app() with no kwargs, so
    ED4ALL_GUI_LEARNER must flip learner-only mode inside create_app itself —
    otherwise `ed4all gui --learner` silently serves the operator surface."""
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from gui.app import create_app

    monkeypatch.setenv("ED4ALL_GUI_LEARNER", "1")
    client = TestClient(create_app())
    # Learner surface present...
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/learn/courses").status_code in (200, 503)
    # ...operator surface NOT mounted.
    assert client.get("/api/settings").status_code == 404


def test_create_app_env_unset_serves_operator_surface(monkeypatch):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from gui.app import create_app

    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app())
    assert client.get("/api/settings").status_code == 200
