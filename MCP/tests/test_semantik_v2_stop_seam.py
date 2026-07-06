"""Wave E / plan P6 — Ed4All-side SemantiK cascade graceful-stop boundary.

Pins the ``MCP/tools/pipeline_tools.py`` half of the P6 contract (the SemantiK
seam owns ``dart_semantic.stop_seam`` + its ``CascadeStopRequested``; this suite
owns the Ed4All boundary that hands the sentinel PATH in and translates the
SemantiK-local stop exception back to ``lib.generation.stop_control``):

  * ``_semantik_stop_sentinel_env_value`` builds the ``SEMANTIK_STOP_SENTINEL``
    declared-env value — the global ``STOP_ALL`` always, plus the run-scoped
    ``<run_id>/control/STOP_REQUESTED`` when ``ED4ALL_RUN_ID`` resolves — from
    the SAME ``lib.paths.get_state_runs_dir`` the CLI writes through.
  * the in-process cascade arm sets ``SEMANTIK_STOP_SENTINEL`` before running.
  * a cascade that raises the SemantiK-local ``CascadeStopRequested`` is
    translated to ``GracefulStopRequested`` (paused, never a success=False
    fail-closed dict) with the sentinel path threaded through.
  * the bridge arm: an error return WHILE a stop sentinel is authoritative is
    translated to ``GracefulStopRequested`` (not surfaced as a failed dict).

CPU-only, stdlib-only, synthetic modules injected into ``sys.modules`` (no
SemantiK heavy deps loaded, no models, no live state dirs). Sentinels isolated
via ``ED4ALL_STATE_RUNS_DIR`` (read at call time by ``get_state_runs_dir``).

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
    MCP/tests/test_semantik_v2_stop_seam.py -q
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from lib.generation import stop_control
from lib.generation.stop_control import GracefulStopRequested


@pytest.fixture(autouse=True)
def _clean_stop_env(monkeypatch):
    """Keep the graceful-stop env vars isolated per test.

    ``_run_semantik_v2_conversion`` sets ``SEMANTIK_STOP_SENTINEL`` on
    ``os.environ`` directly (the in-process cascade reads it per seam call), so
    delenv both so a leaked value never crosses tests. ``delenv`` also restores
    the (absent) original on teardown.
    """
    monkeypatch.delenv("SEMANTIK_STOP_SENTINEL", raising=False)
    monkeypatch.delenv("ED4ALL_REUSE_CONVERSION", raising=False)
    yield


# --------------------------------------------------------------------------
# synthetic SemantiK package: a cascade whose run_pipeline_v2 raises the
# SemantiK-local CascadeStopRequested, + a stop_seam carrying that class.
# --------------------------------------------------------------------------
class _FakeCascadeStop(RuntimeError):
    """Stand-in for ``dart_semantic.stop_seam.CascadeStopRequested``."""

    def __init__(self, site_id, sentinel_path=None):
        self.site_id = site_id
        self.sentinel_path = sentinel_path
        super().__init__(f"cascade stop requested at {site_id!r}")


def _install_stopping_cascade(monkeypatch, sentinel_path):
    """Inject SemantiK cascade + stop_seam so the in-process arm takes and the
    cascade raises the SemantiK-local stop at Stage-6 boundary."""
    def _raise_stop(pdf_path, *args, **kwargs):
        raise _FakeCascadeStop(
            "cascade:post-stage5e-pre-stage6", Path(sentinel_path)
        )

    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    cascade = types.ModuleType("SemantiK.dart_semantic.cascade")
    cascade.run_pipeline_v2 = _raise_stop
    seam = types.ModuleType("SemantiK.dart_semantic.stop_seam")
    seam.CascadeStopRequested = _FakeCascadeStop
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.cascade", cascade)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.stop_seam", seam)


def _install_importfail_cascade(monkeypatch):
    """Inject SemantiK packages with NO cascade.run_pipeline_v2 so the seam's
    in-process import raises ImportError and falls to the bridge arm."""
    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    # No ``SemantiK.dart_semantic.cascade`` submodule → ImportError on
    # ``from SemantiK.dart_semantic.cascade import run_pipeline_v2``.


# --------------------------------------------------------------------------
# _semantik_stop_sentinel_env_value
# --------------------------------------------------------------------------
def test_env_value_global_plus_run_scoped(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import (
        _semantik_stop_sentinel_env_value,
        _STOP_GLOBAL_SENTINEL_NAME,
        _STOP_RUN_SENTINEL_NAME,
    )

    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-P6-0001")

    value = _semantik_stop_sentinel_env_value()
    parts = value.split(os.pathsep)
    assert str(runs / _STOP_GLOBAL_SENTINEL_NAME) in parts
    assert (
        str(runs / "RUN-P6-0001" / "control" / _STOP_RUN_SENTINEL_NAME)
        in parts
    )
    # Both sentinels are watched (global + run-scoped).
    assert len(parts) == 2


def test_env_value_global_only_without_run_id(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import (
        _semantik_stop_sentinel_env_value,
        _STOP_GLOBAL_SENTINEL_NAME,
    )

    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)

    value = _semantik_stop_sentinel_env_value()
    assert value == str(runs / _STOP_GLOBAL_SENTINEL_NAME)


# --------------------------------------------------------------------------
# in-process cascade arm — env plumbing + stop translation
# --------------------------------------------------------------------------
def test_inprocess_stop_translates_to_graceful_stop(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import (
        _run_semantik_v2_conversion,
        _SEMANTIK_STOP_SENTINEL_ENV,
    )

    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-P6-STOP")
    sentinel = runs / "RUN-P6-STOP" / "control" / "STOP_REQUESTED"
    _install_stopping_cascade(monkeypatch, sentinel)

    out = tmp_path / "ch01_accessible.html"
    with pytest.raises(GracefulStopRequested) as exc_info:
        _run_semantik_v2_conversion(str(tmp_path / "ch01.pdf"), str(out))

    # site_id carries the seam, sentinel_path threaded through, units=0.
    assert "semantik_cascade" in exc_info.value.site_id
    assert exc_info.value.units_completed == 0
    assert exc_info.value.sentinel_path == sentinel
    # The declared-env plumbing was published before the cascade ran.
    assert _SEMANTIK_STOP_SENTINEL_ENV in os.environ
    assert str(sentinel) in os.environ[_SEMANTIK_STOP_SENTINEL_ENV]
    # NO HTML written — the stop raised before the write.
    assert not out.exists()


# --------------------------------------------------------------------------
# bridge arm — error-while-stop-pending translates to graceful stop
# --------------------------------------------------------------------------
def test_bridge_error_while_stop_pending_translates(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("SEMANTIK_PYTHON", "/nonexistent/python")  # bridge arm
    _install_importfail_cascade(monkeypatch)

    # Arm the global stop sentinel — the authoritative signal.
    assert stop_control.request_stop(scope="all") is not None
    assert stop_control.stop_requested() is True

    # The bridge child would poll + self-terminate; today it surfaces the stop
    # as a generic error dict. Simulate that.
    monkeypatch.setattr(
        pt,
        "_run_semantik_bridge_subprocess",
        lambda *_a, **_k: {"error": "SemantiK cascade (bridge): stopped"},
    )

    out = tmp_path / "ch01_accessible.html"
    with pytest.raises(GracefulStopRequested) as exc_info:
        pt._run_semantik_v2_conversion(str(tmp_path / "ch01.pdf"), str(out))
    assert "semantik_bridge" in exc_info.value.site_id


def test_bridge_error_without_stop_stays_failed_dict(monkeypatch, tmp_path):
    """A genuine bridge error with NO stop pending is still a failed dict (not
    a spurious pause)."""
    import MCP.tools.pipeline_tools as pt

    runs = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("SEMANTIK_PYTHON", "/nonexistent/python")
    _install_importfail_cascade(monkeypatch)

    assert stop_control.stop_requested() is False
    monkeypatch.setattr(
        pt,
        "_run_semantik_bridge_subprocess",
        lambda *_a, **_k: {"error": "real cascade failure"},
    )

    out = tmp_path / "ch01_accessible.html"
    result = pt._run_semantik_v2_conversion(str(tmp_path / "ch01.pdf"), str(out))
    assert result["success"] is False
    assert "real cascade failure" in result["error"]
