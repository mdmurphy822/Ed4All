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
