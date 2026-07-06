"""Unit tests for the SemantiK-local graceful-stop seam (cross-venv twin).

Covers the sentinel-PATH plumbing the Ed4All side hands in via
``SEMANTIK_STOP_SENTINEL``: env parsing, presence probe (fail-soft), the typed
``CascadeStopRequested`` raise, the no-wiring no-op, and the ``StopPoller``
monotonic throttle. Stdlib-only, CPU-only, no models.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dart_semantic.stop_seam import (
    STOP_SENTINEL_ENV,
    CascadeStopRequested,
    StopPoller,
    check_cascade_stop,
    resolve_stop_sentinel_paths,
    stop_sentinel_present,
)


# --------------------------------------------------------------------------- #
# resolve_stop_sentinel_paths — env parsing (read per call).
# --------------------------------------------------------------------------- #
def test_resolve_unset_is_empty(monkeypatch):
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    assert resolve_stop_sentinel_paths() == []


def test_resolve_blank_is_empty(monkeypatch):
    monkeypatch.setenv(STOP_SENTINEL_ENV, "   ")
    assert resolve_stop_sentinel_paths() == []


def test_resolve_single_path(monkeypatch, tmp_path):
    p = tmp_path / "STOP_REQUESTED"
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    assert resolve_stop_sentinel_paths() == [p]


def test_resolve_multiple_pathsep_joined(monkeypatch, tmp_path):
    run = tmp_path / "run" / "control" / "STOP_REQUESTED"
    glob = tmp_path / "STOP_ALL"
    monkeypatch.setenv(STOP_SENTINEL_ENV, os.pathsep.join([str(run), str(glob)]))
    assert resolve_stop_sentinel_paths() == [run, glob]


def test_resolve_drops_blank_segments(monkeypatch, tmp_path):
    p = tmp_path / "STOP_ALL"
    monkeypatch.setenv(STOP_SENTINEL_ENV, os.pathsep.join(["", str(p), "  "]))
    assert resolve_stop_sentinel_paths() == [p]


def test_resolve_read_per_call(monkeypatch, tmp_path):
    # A sentinel var set AFTER the module imported must still be seen.
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    assert resolve_stop_sentinel_paths() == []
    p = tmp_path / "STOP_REQUESTED"
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    assert resolve_stop_sentinel_paths() == [p]


# --------------------------------------------------------------------------- #
# stop_sentinel_present — presence probe (fail-soft).
# --------------------------------------------------------------------------- #
def test_present_false_when_no_paths(monkeypatch):
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    assert stop_sentinel_present() is False


def test_present_false_when_file_absent(monkeypatch, tmp_path):
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(tmp_path / "STOP_REQUESTED"))
    assert stop_sentinel_present() is False


def test_present_true_when_file_exists(monkeypatch, tmp_path):
    p = tmp_path / "STOP_REQUESTED"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    assert stop_sentinel_present() is True


def test_present_true_when_any_of_several_exists(monkeypatch, tmp_path):
    run = tmp_path / "STOP_REQUESTED"
    glob = tmp_path / "STOP_ALL"
    glob.write_text("{}", encoding="utf-8")  # only the global one exists
    monkeypatch.setenv(STOP_SENTINEL_ENV, os.pathsep.join([str(run), str(glob)]))
    assert stop_sentinel_present() is True


def test_present_failsoft_on_oserror(monkeypatch):
    # A probe that raises OSError degrades to "absent", never propagates.
    class _Boom:
        def exists(self):
            raise OSError("stat blew up")

    assert stop_sentinel_present([_Boom()]) is False


# --------------------------------------------------------------------------- #
# check_cascade_stop — typed raise / no-op.
# --------------------------------------------------------------------------- #
def test_check_noop_when_unset(monkeypatch):
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    check_cascade_stop("site:x")  # must not raise


def test_check_noop_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(tmp_path / "STOP_REQUESTED"))
    check_cascade_stop("site:x")  # must not raise


def test_check_raises_typed_with_path(monkeypatch, tmp_path):
    p = tmp_path / "STOP_REQUESTED"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    with pytest.raises(CascadeStopRequested) as ei:
        check_cascade_stop("site:seam-a")
    assert ei.value.site_id == "site:seam-a"
    assert ei.value.sentinel_path == p
    assert "seam-a" in str(ei.value)


def test_check_is_runtimeerror_subclass(monkeypatch, tmp_path):
    p = tmp_path / "STOP_ALL"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    # A generic except Exception still catches it (bridge boundary catches the
    # specific type FIRST — verified on the Ed4All side).
    with pytest.raises(RuntimeError):
        check_cascade_stop("site:x")


# --------------------------------------------------------------------------- #
# StopPoller — throttle + no-op.
# --------------------------------------------------------------------------- #
def test_poller_noop_when_unset(monkeypatch):
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    poller = StopPoller()
    assert poller.should_stop() is False
    poller.check("site:x")  # no-op


def test_poller_sees_stop(monkeypatch, tmp_path):
    p = tmp_path / "STOP_REQUESTED"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    poller = StopPoller(min_interval_s=0.0)
    assert poller.should_stop() is True
    with pytest.raises(CascadeStopRequested):
        poller.check("site:seam-c")


def test_poller_throttles_probe(monkeypatch, tmp_path):
    # With a large interval the poller re-stats at most once; a sentinel written
    # AFTER the first (false) probe is NOT observed until the interval elapses.
    p = tmp_path / "STOP_REQUESTED"
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    poller = StopPoller(min_interval_s=1000.0)
    assert poller.should_stop() is False  # primes the cache (absent)
    p.write_text("{}", encoding="utf-8")  # sentinel appears
    assert poller.should_stop() is False  # throttled — cache still sticky-false


def test_poller_caches_resolved_paths(monkeypatch, tmp_path):
    # Path resolution is cached on first use; clearing the env afterwards does
    # not disarm a poller already watching a path (stable for a cascade run).
    p = tmp_path / "STOP_REQUESTED"
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    poller = StopPoller(min_interval_s=0.0)
    assert poller.should_stop() is False
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    p.write_text("{}", encoding="utf-8")
    assert poller.should_stop() is True
