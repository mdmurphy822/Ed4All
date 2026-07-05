"""Unit tests for the graceful-stop sentinel control.

``lib/generation/stop_control.py`` is the stdlib-only "checkpoint on command"
sentinel layer: it writes / probes / clears the run-scoped and global stop
sentinels under the resolved ``state/runs`` parent and raises the typed
``GracefulStopRequested`` at unit boundaries. These tests pin path resolution
(explicit arg beats env; ``ED4ALL_STATE_RUNS_DIR`` honored), request/clear
roundtrip, run-vs-global scoping isolation, the typed raise payload, the
best-effort OSError degrade, and the ``StopPoller`` interval cache. Hermetic
(tmp dirs, no LLM, no network, no course slugs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import (  # noqa: E402
    STOP_MARKER,
    GracefulStopRequested,
    StopPoller,
    check_stop,
    clear_stop,
    request_stop,
    stop_requested,
)


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """Isolate the state/runs parent into a tmp dir for the whole test."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(d))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    return d


def _run_sentinel(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id / "control" / "STOP_REQUESTED"


def _global_sentinel(runs_dir: Path) -> Path:
    return runs_dir / "STOP_ALL"


# --------------------------------------------------------------------------- #
# STOP_MARKER identity contract (AMENDMENT #1)
# --------------------------------------------------------------------------- #
def test_stop_marker_is_distinct_from_none():
    assert STOP_MARKER is not None
    assert STOP_MARKER is stop_control.STOP_MARKER  # singleton
    assert repr(STOP_MARKER) == "<STOP_MARKER>"


# --------------------------------------------------------------------------- #
# Path resolution: ED4ALL_STATE_RUNS_DIR honored; explicit arg beats env run_id
# --------------------------------------------------------------------------- #
def test_request_writes_under_state_runs_dir(runs_dir):
    path = request_stop("RUN-A", scope="run")
    assert path == _run_sentinel(runs_dir, "RUN-A")
    assert path.exists()


def test_explicit_run_id_beats_env(runs_dir, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "ENV-RUN")
    path = request_stop("ARG-RUN", scope="run")
    assert path == _run_sentinel(runs_dir, "ARG-RUN")
    # The env run's sentinel was NOT written.
    assert not _run_sentinel(runs_dir, "ENV-RUN").exists()


def test_env_run_id_used_when_no_arg(runs_dir, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", "ENV-RUN")
    path = request_stop(scope="run")
    assert path == _run_sentinel(runs_dir, "ENV-RUN")
    assert stop_requested() is True  # resolves the same env run_id


def test_run_scoped_request_without_any_run_id_degrades(runs_dir):
    # No arg, no ED4ALL_RUN_ID → no sentinel, falsy return, no crash.
    assert request_stop(scope="run") is None
    assert list(runs_dir.rglob("STOP_REQUESTED")) == []


# --------------------------------------------------------------------------- #
# request / clear roundtrip
# --------------------------------------------------------------------------- #
def test_request_clear_roundtrip(runs_dir):
    assert stop_requested("RUN-A") is False
    request_stop("RUN-A", scope="run", reason="operator", source="cli")
    assert stop_requested("RUN-A") is True
    clear_stop("RUN-A")
    assert stop_requested("RUN-A") is False
    assert not _run_sentinel(runs_dir, "RUN-A").exists()


def test_clear_stop_idempotent_on_missing(runs_dir):
    clear_stop("RUN-A")  # nothing to remove → no raise
    clear_stop("RUN-A", include_global=True)


def test_request_body_is_diagnostic_json(runs_dir):
    import json

    path = request_stop("RUN-A", scope="run", reason="why", source="test")
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["reason"] == "why"
    assert body["source"] == "test"
    assert body["scope"] == "run"
    assert body["run_id"] == "RUN-A"
    assert isinstance(body["pid"], int)


# --------------------------------------------------------------------------- #
# Global vs run scoping: run stop is isolated; global trips all
# --------------------------------------------------------------------------- #
def test_run_stop_does_not_trip_other_runs(runs_dir):
    request_stop("RUN-A", scope="run")
    assert stop_requested("RUN-A") is True
    assert stop_requested("RUN-B") is False  # sibling run unaffected


def test_global_stop_trips_every_run(runs_dir):
    path = request_stop(scope="all")
    assert path == _global_sentinel(runs_dir)
    assert stop_requested("RUN-A") is True
    assert stop_requested("RUN-B") is True
    assert stop_requested(None) is True  # no run_id at all still trips


def test_clear_run_does_not_touch_global(runs_dir):
    request_stop(scope="all")
    request_stop("RUN-A", scope="run")
    clear_stop("RUN-A")  # include_global defaults False
    assert stop_requested("RUN-A") is True  # global still tripping it
    assert _global_sentinel(runs_dir).exists()
    clear_stop("RUN-A", include_global=True)
    assert not _global_sentinel(runs_dir).exists()
    assert stop_requested("RUN-A") is False


def test_unknown_scope_raises(runs_dir):
    with pytest.raises(ValueError):
        request_stop("RUN-A", scope="bogus")


# --------------------------------------------------------------------------- #
# check_stop: typed raise carrying payload attributes
# --------------------------------------------------------------------------- #
def test_check_stop_noop_when_not_requested(runs_dir):
    check_stop("site.x", 3, run_id="RUN-A")  # no raise


def test_check_stop_raises_with_run_payload(runs_dir):
    request_stop("RUN-A", scope="run")
    with pytest.raises(GracefulStopRequested) as ei:
        check_stop("stage2_windows", 7, run_id="RUN-A")
    exc = ei.value
    assert exc.site_id == "stage2_windows"
    assert exc.units_completed == 7
    assert exc.sentinel_path == _run_sentinel(runs_dir, "RUN-A")
    assert isinstance(exc, RuntimeError)  # generic except still catches it


def test_check_stop_reports_global_sentinel(runs_dir):
    request_stop(scope="all")
    with pytest.raises(GracefulStopRequested) as ei:
        check_stop("concept_windows", 2, run_id="RUN-A")
    assert ei.value.sentinel_path == _global_sentinel(runs_dir)


# --------------------------------------------------------------------------- #
# OSError best-effort: request / probe / clear never propagate
# --------------------------------------------------------------------------- #
def test_request_stop_oserror_returns_falsy(runs_dir, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _boom)
    assert request_stop("RUN-A", scope="run") is None  # degraded, no raise


def test_request_stop_write_oserror_returns_falsy(runs_dir, monkeypatch):
    def _boom(*a, **k):
        raise OSError("read-only fs")

    monkeypatch.setattr(Path, "write_text", _boom)
    assert request_stop("RUN-A", scope="run") is None


def test_stop_requested_oserror_degrades_false(runs_dir, monkeypatch):
    request_stop("RUN-A", scope="run")

    def _boom(self):
        raise OSError("stat failed")

    monkeypatch.setattr(Path, "exists", _boom)
    assert stop_requested("RUN-A") is False  # stat failure != stop


def test_clear_stop_oserror_no_raise(runs_dir, monkeypatch):
    request_stop("RUN-A", scope="run")

    def _boom(self):
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "unlink", _boom)
    clear_stop("RUN-A", include_global=True)  # logged, never raised


# --------------------------------------------------------------------------- #
# StopPoller: monotonic-time interval caching
# --------------------------------------------------------------------------- #
def test_stop_poller_caches_within_interval(runs_dir, monkeypatch):
    calls = {"n": 0}
    real = stop_control.stop_requested

    def _counting(run_id=None):
        calls["n"] += 1
        return real(run_id)

    monkeypatch.setattr(stop_control, "stop_requested", _counting)

    clock = {"t": 100.0}
    monkeypatch.setattr(stop_control.time, "monotonic", lambda: clock["t"])

    poller = StopPoller(min_interval_s=2.0)
    assert poller.should_stop("RUN-A") is False  # first probe
    assert calls["n"] == 1
    # Within the interval → cached, no new probe.
    clock["t"] = 101.0
    assert poller.should_stop("RUN-A") is False
    assert calls["n"] == 1

    # Arm the sentinel; still cached until the interval elapses.
    request_stop("RUN-A", scope="run")
    clock["t"] = 101.9
    assert poller.should_stop("RUN-A") is False
    assert calls["n"] == 1

    # Interval elapsed → re-probe picks up the stop.
    clock["t"] = 102.0
    assert poller.should_stop("RUN-A") is True
    assert calls["n"] == 2


def test_stop_poller_check_raises_after_interval(runs_dir, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(stop_control.time, "monotonic", lambda: clock["t"])
    request_stop("RUN-A", scope="run")

    poller = StopPoller(min_interval_s=1.0)
    with pytest.raises(GracefulStopRequested) as ei:
        poller.check("site.tight_loop", 42, run_id="RUN-A")
    assert ei.value.units_completed == 42
