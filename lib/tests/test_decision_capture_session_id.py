"""
Worker W6 regression tests for DecisionCapture.session_id uniqueness.

Pre-W6, ``self.session_id`` was set via
``datetime.now().strftime("%Y%m%d_%H%M%S")`` — 1-second granularity, so two
captures initialized within the same wall-second from parallel workers
(DART per-PDF max_concurrent=4, assessment_generation max_concurrent=5,
etc.) shared a session_id, JSONL filename, and run_id fallback. W6 appends
``_{pid}_{6-hex}`` so concurrent inits get distinct ids while the leading
15-char ``%Y%m%d_%H%M%S`` timestamp prefix is preserved (so glob patterns
``decisions_*.jsonl`` still match).

Coverage:

- test_session_ids_unique_under_concurrent_init: spawn 100 captures in
  parallel via threads in the same wall-second; assert all 100 session_ids
  are unique.
- test_session_id_format_remains_parseable: assert the format follows the
  documented pattern ``%Y%m%d_%H%M%S_{pid}_{6-hex}``.
- test_session_id_timestamp_prefix_preserved: assert the leading 15 chars
  are still the legacy ``%Y%m%d_%H%M%S`` prefix so existing glob consumers
  keep matching.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.decision_capture import DecisionCapture
except ImportError:
    pytest.skip("decision_capture not available", allow_module_level=True)


# Documented format: YYYYMMDD_HHMMSS_<pid>_<6-hex>
# - chars 0..15: %Y%m%d_%H%M%S (legacy prefix preserved for glob compat)
# - then "_<pid>_<6-hex>" suffix (W6 disambiguator)
SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_\d+_[0-9a-f]{6}$")
TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}_\d{6}$")


def _build_capture(course_code: str = "TEST_001") -> DecisionCapture:
    """Helper: build a DecisionCapture with streaming disabled.

    Streaming=False sidesteps file-handle setup so the test stays fast and
    doesn't pollute training-captures/. session_id is set in __init__ before
    any I/O so the disambiguator fires regardless of streaming mode.
    """
    return DecisionCapture(
        course_code=course_code,
        phase="content-generator",
        tool="courseforge",
        streaming=False,
    )


def test_session_ids_unique_under_concurrent_init(tmp_path, monkeypatch):
    """100 captures spawned in parallel from the same process must all
    receive distinct session_ids."""
    # Redirect captures away from the real training-captures/ tree so the
    # test is hermetic. LibV2Storage + LEGACY_TRAINING_DIR each create
    # directories on init; pointing them at tmp_path avoids polluting the
    # repo and keeps the test fast.
    monkeypatch.setenv("LIBV2_ROOT", str(tmp_path / "libv2"))
    monkeypatch.setattr(
        "lib.decision_capture.LEGACY_TRAINING_DIR",
        tmp_path / "training-captures",
    )

    n_workers = 100
    session_ids: list[str] = []
    lock = threading.Lock()

    def _spawn() -> None:
        cap = _build_capture()
        with lock:
            session_ids.append(cap.session_id)

    threads = [threading.Thread(target=_spawn) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(session_ids) == n_workers, (
        f"expected {n_workers} session_ids collected, got {len(session_ids)}"
    )
    assert len(set(session_ids)) == n_workers, (
        f"expected {n_workers} distinct session_ids, got "
        f"{len(set(session_ids))} distinct out of {len(session_ids)} "
        f"(collisions: "
        f"{[sid for sid in session_ids if session_ids.count(sid) > 1][:5]})"
    )


def test_session_id_format_remains_parseable(tmp_path, monkeypatch):
    """session_id must match ``YYYYMMDD_HHMMSS_<pid>_<6-hex>`` and embed the
    current PID."""
    monkeypatch.setenv("LIBV2_ROOT", str(tmp_path / "libv2"))
    monkeypatch.setattr(
        "lib.decision_capture.LEGACY_TRAINING_DIR",
        tmp_path / "training-captures",
    )

    cap = _build_capture()

    assert SESSION_ID_RE.match(cap.session_id), (
        f"session_id {cap.session_id!r} does not match expected pattern "
        f"YYYYMMDD_HHMMSS_<pid>_<6-hex>"
    )

    parts = cap.session_id.split("_")
    # parts: ['YYYYMMDD', 'HHMMSS', '<pid>', '<6-hex>']
    assert len(parts) == 4, f"expected 4 underscore-separated parts, got {parts!r}"
    assert int(parts[2]) == os.getpid(), (
        f"PID component {parts[2]!r} does not match os.getpid()={os.getpid()}"
    )
    assert len(parts[3]) == 6 and all(c in "0123456789abcdef" for c in parts[3]), (
        f"hex suffix {parts[3]!r} is not 6 lowercase hex chars"
    )


def test_session_id_timestamp_prefix_preserved(tmp_path, monkeypatch):
    """The leading 15 chars must remain the legacy ``%Y%m%d_%H%M%S`` prefix
    so glob patterns ``decisions_*.jsonl`` and any consumer that slices
    ``session_id[:15]`` still works."""
    monkeypatch.setenv("LIBV2_ROOT", str(tmp_path / "libv2"))
    monkeypatch.setattr(
        "lib.decision_capture.LEGACY_TRAINING_DIR",
        tmp_path / "training-captures",
    )

    cap = _build_capture()
    prefix = cap.session_id[:15]
    assert TIMESTAMP_PREFIX_RE.match(prefix), (
        f"leading 15 chars {prefix!r} are not %Y%m%d_%H%M%S"
    )
