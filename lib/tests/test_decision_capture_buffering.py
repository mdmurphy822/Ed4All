"""
Builder-2 regression tests for decision-capture write buffering
(lib/decision_capture.py, ED4ALL_CAPTURE_BUFFER).

PROFILE the change addresses: ``_write_to_streams`` fsyncs three open JSONL
handles per ``log_decision``; a ~424-decision metadata-only gate pays ~424*3
disk syncs (~5.8s). ``ED4ALL_CAPTURE_BUFFER`` coalesces those into one batched
write+fsync per ``ED4ALL_CAPTURE_BUFFER_ROWS`` rows.

Coverage:
- flag OFF (default) → one write+fsync emit per decision (byte-identical path).
- flag ON → rows buffered; a single batched emit fires per N rows.
- flush-on-close drains a partial (< N) buffer to disk.
- thread-safety smoke: concurrent writers, every row lands + parses.
- rows byte-identical on disk between the buffered and unbuffered paths.
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


def _make_capture(tmp_path: Path) -> DecisionCapture:
    """Construct a streaming DecisionCapture whose three JSONL mirrors all
    land inside ``tmp_path`` (LibV2Storage + legacy dir patched).

    Env (``ED4ALL_CAPTURE_BUFFER`` / ``_ROWS``) is read in ``__init__``, so
    set it via ``monkeypatch.setenv`` BEFORE calling this.
    """
    legacy_dir = tmp_path / "legacy-training"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    libv2_root = tmp_path / "libv2" / "training"
    libv2_root.mkdir(parents=True, exist_ok=True)

    storage = Mock()
    storage.get_training_capture_path.return_value = libv2_root

    with patch("lib.decision_capture.LibV2Storage") as mock_cls, patch(
        "lib.decision_capture.LEGACY_TRAINING_DIR", legacy_dir
    ):
        mock_cls.return_value = storage
        return DecisionCapture(
            course_code="BUF_TEST",
            phase="content-generator",
            tool="courseforge",
            streaming=True,
        )


def _read_rows(cap: DecisionCapture) -> list:
    raw = cap._stream_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line]


# =============================================================================
# 1. Flag OFF → per-call write preserved (one emit per decision)
# =============================================================================


def test_flag_off_emits_per_call(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_CAPTURE_BUFFER", raising=False)
    cap = _make_capture(tmp_path)
    assert cap._buffer_enabled is False
    try:
        real_emit = cap._emit_to_stream_handles
        with patch.object(
            cap, "_emit_to_stream_handles", side_effect=real_emit
        ) as spy:
            for i in range(5):
                cap._write_to_streams({"seq": i})
            # One batched-write helper call per decision (fsync-per-call).
            assert spy.call_count == 5
        # Nothing buffered on the off-path.
        assert cap._buffer == []
        rows = _read_rows(cap)
        assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    finally:
        cap.close()


# =============================================================================
# 2. Flag ON → single batched emit per N rows
# =============================================================================


def test_flag_on_batches_every_n_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "1")
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", "3")
    cap = _make_capture(tmp_path)
    assert cap._buffer_enabled is True
    assert cap._buffer_rows == 3
    try:
        real_emit = cap._emit_to_stream_handles
        with patch.object(
            cap, "_emit_to_stream_handles", side_effect=real_emit
        ) as spy:
            # Two rows: buffered, no emit yet.
            cap._write_to_streams({"seq": 0})
            cap._write_to_streams({"seq": 1})
            assert spy.call_count == 0
            assert len(cap._buffer) == 2
            # Third row hits the boundary → exactly one batched emit.
            cap._write_to_streams({"seq": 2})
            assert spy.call_count == 1
            assert cap._buffer == []
            # The single emit carried all three newline-terminated rows.
            (blob,), _ = spy.call_args
            assert blob.count("\n") == 3

            # Two more rows buffer again without a new emit.
            cap._write_to_streams({"seq": 3})
            cap._write_to_streams({"seq": 4})
            assert spy.call_count == 1
            assert len(cap._buffer) == 2

        rows = _read_rows(cap)
        # Only the first flushed batch is on disk pre-flush.
        assert [r["seq"] for r in rows] == [0, 1, 2]
    finally:
        cap.close()


# =============================================================================
# 3. flush() and close() drain a partial (< N) buffer
# =============================================================================


def test_flush_drains_partial_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "yes")
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", "50")
    cap = _make_capture(tmp_path)
    try:
        for i in range(4):
            cap._write_to_streams({"seq": i})
        # Below the 50-row boundary → still buffered, nothing on disk.
        assert len(cap._buffer) == 4
        assert cap._stream_path.read_text(encoding="utf-8") == ""

        cap.flush()
        assert cap._buffer == []
        assert [r["seq"] for r in _read_rows(cap)] == [0, 1, 2, 3]

        # flush() is idempotent (empty buffer → no-op, no duplicate rows).
        cap.flush()
        assert [r["seq"] for r in _read_rows(cap)] == [0, 1, 2, 3]
    finally:
        cap.close()


def test_close_drains_partial_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "on")
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", "100")
    cap = _make_capture(tmp_path)
    stream_path = cap._stream_path
    for i in range(7):
        cap._write_to_streams({"seq": i})
    assert len(cap._buffer) == 7
    assert stream_path.read_text(encoding="utf-8") == ""

    cap.close()  # must drain BEFORE handles close
    rows = [json.loads(line) for line in stream_path.read_text().splitlines() if line]
    assert [r["seq"] for r in rows] == list(range(7))


# =============================================================================
# 4. Thread-safety smoke — concurrent writers, every row lands + parses
# =============================================================================


def test_buffered_concurrent_writes_all_land(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "1")
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", "16")
    cap = _make_capture(tmp_path)
    threads = 8
    per_thread = 50
    filler = "Y" * 200

    def worker(tid: int) -> None:
        for i in range(per_thread):
            cap._write_to_streams({"thread": tid, "seq": i, "filler": filler})

    workers = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    cap.flush()  # drain the final partial buffer
    rows = _read_rows(cap)  # json.loads raises on any interleaved/malformed line
    assert len(rows) == threads * per_thread
    assert all(r["filler"] == filler for r in rows)
    cap.close()


# =============================================================================
# 5. Rows byte-identical between buffered and unbuffered paths
# =============================================================================


def test_buffered_and_unbuffered_bytes_identical(tmp_path, monkeypatch):
    records = [{"seq": i, "decision": f"d{i}", "rationale": "r" * (i + 1)} for i in range(25)]

    monkeypatch.delenv("ED4ALL_CAPTURE_BUFFER", raising=False)
    cap_off = _make_capture(tmp_path / "off")
    for rec in records:
        cap_off._write_to_streams(dict(rec))
    cap_off.close()
    bytes_off = cap_off._stream_path.read_bytes()

    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "1")
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", "7")
    cap_on = _make_capture(tmp_path / "on")
    for rec in records:
        cap_on._write_to_streams(dict(rec))
    cap_on.close()  # drains the final partial buffer
    bytes_on = cap_on._stream_path.read_bytes()

    assert bytes_on == bytes_off, "buffered on-disk bytes must match the per-call path"


# =============================================================================
# 6. Env parsing — garbage/falsey → OFF, rows fallback
# =============================================================================


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage", ""])
def test_flag_falsey_disables_buffering(tmp_path, monkeypatch, val):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", val)
    cap = _make_capture(tmp_path / val.replace("/", "_") or "empty")
    try:
        assert cap._buffer_enabled is False
    finally:
        cap.close()


@pytest.mark.parametrize("val,expected", [("", 50), ("garbage", 50), ("0", 50), ("-4", 50), ("25", 25)])
def test_buffer_rows_fallback(tmp_path, monkeypatch, val, expected):
    monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER", "1")
    if val:
        monkeypatch.setenv("ED4ALL_CAPTURE_BUFFER_ROWS", val)
    else:
        monkeypatch.delenv("ED4ALL_CAPTURE_BUFFER_ROWS", raising=False)
    cap = _make_capture(tmp_path)
    try:
        assert cap._buffer_rows == expected
    finally:
        cap.close()
