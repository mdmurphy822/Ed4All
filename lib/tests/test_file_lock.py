"""Regression tests for the shared cross-process file lock (W0.2 helper)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.file_lock import file_lock


@pytest.mark.unit
def test_file_lock_runs_body_and_creates_sentinel(tmp_path):
    lock_path = tmp_path / "sub" / ".lock"
    ran = []
    with file_lock(lock_path):
        ran.append(True)
    assert ran == [True]
    assert lock_path.exists()  # sentinel created (parent auto-made)


@pytest.mark.unit
def test_file_lock_reentrant_across_sequential_calls(tmp_path):
    lock_path = tmp_path / ".lock"
    with file_lock(lock_path):
        pass
    # Re-acquire after release must not deadlock or error.
    with file_lock(lock_path):
        pass
    assert lock_path.exists()


@pytest.mark.unit
def test_file_lock_unwritable_sentinel_degrades_gracefully(tmp_path, monkeypatch):
    # Force the open() to fail; the body must still run (unlocked fallback).
    import builtins

    real_open = builtins.open

    def boom(path, *a, **k):
        if str(path).endswith(".lock"):
            raise OSError("simulated unwritable")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", boom)
    ran = []
    with file_lock(tmp_path / ".lock"):
        ran.append(True)
    assert ran == [True]


@pytest.mark.unit
def test_file_lock_timeout_acquires_when_uncontended(tmp_path):
    lock_path = tmp_path / ".lock"
    ran = []
    with file_lock(lock_path, timeout=10.0):
        ran.append(True)
    assert ran == [True]
    # Timeout-mode acquirers stamp holder identity into the sentinel so a
    # later contender that times out can name them.
    stamp = lock_path.read_text()
    assert "pid=" in stamp


@pytest.mark.unit
def test_file_lock_timeout_contended_proceeds_with_loud_warning(tmp_path, caplog):
    """Contended + timed out → body still runs (unlocked) after a LOUD
    warning naming the other holder from its sentinel stamp."""
    import fcntl
    import logging

    lock_path = tmp_path / ".lock"
    # Simulate the other process: hold LOCK_EX on a separate fd and leave a
    # holder stamp behind (what a timeout-mode acquirer writes).
    holder_fh = open(lock_path, "w")
    holder_fh.write("pid=99999 cmd=ed4all acquired=2026-07-21T20:55:00\n")
    holder_fh.flush()
    fcntl.flock(holder_fh.fileno(), fcntl.LOCK_EX)
    try:
        ran = []
        with caplog.at_level(logging.WARNING, logger="lib.file_lock"):
            with file_lock(lock_path, timeout=0.3):
                ran.append(True)
        assert ran == [True]  # proceeded despite contention
        warning = " ".join(r.getMessage() for r in caplog.records)
        assert "TIMED OUT" in warning
        assert "pid=99999" in warning  # names the other holder
    finally:
        fcntl.flock(holder_fh.fileno(), fcntl.LOCK_UN)
        holder_fh.close()


@pytest.mark.unit
def test_file_lock_timeout_serializes_two_threads(tmp_path):
    """A second acquirer waits (within its timeout) for the first to release."""
    import threading
    import time

    lock_path = tmp_path / ".lock"
    order = []

    def first():
        with file_lock(lock_path, timeout=5.0):
            order.append("first-in")
            time.sleep(0.4)
            order.append("first-out")

    def second():
        time.sleep(0.1)  # let first grab the lock
        with file_lock(lock_path, timeout=5.0):
            order.append("second-in")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert order == ["first-in", "first-out", "second-in"]
