"""
Worker W3 regression tests for fcntl.flock around the legacy
decision-capture write path (lib/decision_capture.py::_write_to_streams).

Coverage:
- test_concurrent_writes_do_not_interleave: 8 threads x 100 records each;
  every line in the JSONL output must parse cleanly.
- test_flock_acquired_on_write: monkey-patch fcntl.flock to record calls;
  assert LOCK_EX is acquired on every write.
- test_oserror_falls_through_with_warning: monkey-patch fcntl.flock to
  raise OSError; assert the write still completes and a warning is logged.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib import decision_capture as dc_module
    from lib.decision_capture import DecisionCapture
except ImportError:
    pytest.skip("decision_capture not available", allow_module_level=True)

# fcntl is POSIX-only; Windows runs degrade to unlocked behavior and the
# flock-acquisition + OSError-fallback tests do not apply.
fcntl_required = pytest.mark.skipif(
    dc_module._fcntl is None,
    reason="fcntl unavailable (non-POSIX); flock path inactive",
)


@pytest.fixture
def capture(tmp_path):
    """A streaming DecisionCapture rooted in tmp_path.

    Patches LibV2Storage + LEGACY_TRAINING_DIR so both stream paths land
    inside the temp tree. Returns the live capture; teardown closes it.
    """
    legacy_dir = tmp_path / "legacy-training"
    legacy_dir.mkdir(parents=True)

    libv2_root = tmp_path / "libv2" / "training"
    libv2_root.mkdir(parents=True)

    storage = Mock()
    storage.get_training_capture_path.return_value = libv2_root

    with patch("lib.decision_capture.LibV2Storage") as mock_cls, patch(
        "lib.decision_capture.LEGACY_TRAINING_DIR", legacy_dir
    ):
        mock_cls.return_value = storage
        cap = DecisionCapture(
            course_code="W3_TEST",
            phase="content-generator",
            tool="courseforge",
            streaming=True,
        )
        try:
            yield cap
        finally:
            cap.close()


# =============================================================================
# 1. Concurrent writes do not interleave
# =============================================================================


@fcntl_required
def test_concurrent_writes_do_not_interleave(capture):
    """8 threads x 100 records must produce 800 cleanly-parsing JSON lines.

    Without flock, large records straddling PIPE_BUF (4 KiB) interleave
    mid-line and downstream JSON parsing fails. With flock, every line
    parses.
    """
    threads = 8
    per_thread = 100
    payload_filler = "X" * 6000  # >4 KiB so writes straddle PIPE_BUF

    def worker(tid: int) -> None:
        for i in range(per_thread):
            record = {
                "thread": tid,
                "seq": i,
                "filler": payload_filler,
            }
            capture._write_to_streams(record)

    workers = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    stream_path = capture._stream_path
    assert stream_path is not None and stream_path.exists()

    raw = stream_path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    # Drop trailing empty produced by terminating newline.
    if lines and lines[-1] == "":
        lines = lines[:-1]

    expected = threads * per_thread
    assert len(lines) == expected, (
        f"line count {len(lines)} != expected {expected}; "
        "missing or extra newlines indicate interleaving"
    )

    parsed = 0
    for line in lines:
        obj = json.loads(line)  # raises if any line is malformed
        assert obj["filler"] == payload_filler
        parsed += 1
    assert parsed == expected


# =============================================================================
# 2. flock LOCK_EX acquired on every write
# =============================================================================


@fcntl_required
def test_flock_acquired_on_write(capture):
    """Every legacy-path write must call fcntl.flock with LOCK_EX."""
    real_flock = dc_module._fcntl.flock
    LOCK_EX = dc_module._fcntl.LOCK_EX
    LOCK_UN = dc_module._fcntl.LOCK_UN

    calls: list[tuple[int, int]] = []

    def recording_flock(fd: int, op: int) -> None:
        calls.append((fd, op))
        real_flock(fd, op)

    with patch.object(dc_module._fcntl, "flock", side_effect=recording_flock):
        capture._write_to_streams({"event": "first"})
        capture._write_to_streams({"event": "second"})

    # Per write: 1 LOCK_EX + 1 LOCK_UN per active stream handle. Confirm at
    # least one LOCK_EX fired per write; the LOCK_UN pair-up is implicit in
    # the try/finally — failure to release would already block subsequent
    # acquires in the threaded test above.
    lock_ex_count = sum(1 for _, op in calls if op == LOCK_EX)
    lock_un_count = sum(1 for _, op in calls if op == LOCK_UN)
    assert lock_ex_count >= 2, (
        f"expected >=2 LOCK_EX (one per write call); got {lock_ex_count}"
    )
    assert lock_ex_count == lock_un_count, (
        f"LOCK_EX ({lock_ex_count}) != LOCK_UN ({lock_un_count}); "
        "unbalanced acquire/release"
    )


# =============================================================================
# 3. OSError on flock falls through unlocked + warns
# =============================================================================


@fcntl_required
def test_oserror_falls_through_with_warning(capture, caplog):
    """If fcntl.flock raises OSError (WSL2 DrvFS / NFS), the write still
    completes and a warning is emitted."""
    import logging as _logging

    def raising_flock(fd: int, op: int) -> None:
        raise OSError("flock not supported on this filesystem")

    with patch.object(dc_module._fcntl, "flock", side_effect=raising_flock):
        with caplog.at_level(_logging.WARNING, logger="lib.decision_capture"):
            capture._write_to_streams({"event": "drvfs_fallback"})

    # Write still landed on disk despite the OSError.
    stream_path = capture._stream_path
    assert stream_path is not None and stream_path.exists()
    raw = stream_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) >= 1
    obj = json.loads(raw[-1])
    assert obj["event"] == "drvfs_fallback"

    # Warning surfaced naming the unlocked-fallback condition.
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno >= _logging.WARNING
        and "flock unavailable" in rec.getMessage()
    ]
    assert warnings, (
        "expected at least one 'flock unavailable' warning in the "
        f"OSError fallback path; got records: {[r.getMessage() for r in caplog.records]}"
    )
