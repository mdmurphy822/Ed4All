"""Wall-clock accounting and pause status across resume segments.

Two defects surfaced on a build that took 17 run segments:

1. ``start_phase`` unconditionally rewrote ``started_at``, so each phase
   reported only its LAST segment. A multi-hour scan conversion rendered in
   the GUI as "1s" because the final resume merely skipped it.
2. A phase that stopped gracefully still had ``complete_phase`` called on it,
   writing ``status="completed"``. ``training_synthesis`` showed as done after
   synthesizing ~2% of its chunks.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from MCP.hardening.checkpoint import CheckpointManager, PhaseCheckpoint


def _mgr(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path)


def _start(mgr: CheckpointManager, phase: str = "training_synthesis"):
    return mgr.start_phase("run-1", "WF-1", phase, 0, ["t-1"])


def test_first_segment_records_its_own_elapsed(tmp_path: Path):
    mgr = _mgr(tmp_path)
    _start(mgr)
    ck = mgr.complete_phase("training_synthesis")

    assert ck.segments == 1
    assert ck.elapsed_seconds >= 0.0
    assert ck.first_started_at == ck.started_at


def test_resume_accumulates_instead_of_overwriting(tmp_path: Path):
    """The core regression: a resumed phase must not lose the prior segment."""
    mgr = _mgr(tmp_path)

    # Segment 1 — hand-write a checkpoint with a known 2h span.
    _start(mgr)
    ck = mgr.load_checkpoint("training_synthesis")
    start = datetime(2026, 7, 24, 10, 0, 0)
    ck.started_at = start.isoformat()
    ck.completed_at = (start + timedelta(hours=2)).isoformat()
    ck.status = "completed"
    ck.elapsed_seconds = 7200.0
    ck.first_started_at = ck.started_at
    mgr._write_checkpoint(ck)

    # Segment 2 — resume.
    resumed = _start(mgr)
    assert resumed.segments == 2
    assert resumed.elapsed_seconds == 7200.0, "prior segment's time was dropped"
    assert resumed.first_started_at == start.isoformat()
    assert resumed.started_at != start.isoformat(), "new segment needs its own start"

    final = mgr.complete_phase("training_synthesis")
    assert final.elapsed_seconds >= 7200.0
    assert final.segments == 2


def test_legacy_checkpoint_without_elapsed_is_folded_in(tmp_path: Path):
    """A checkpoint written before the field existed still has timestamps."""
    mgr = _mgr(tmp_path)
    start = datetime(2026, 7, 24, 10, 0, 0)
    legacy = PhaseCheckpoint(
        run_id="run-1",
        workflow_id="WF-1",
        phase_name="course_planning",
        phase_index=6,
        status="completed",
        started_at=start.isoformat(),
        completed_at=(start + timedelta(minutes=40)).isoformat(),
    )
    # Simulate the on-disk legacy shape: no elapsed/segments/first_started_at.
    payload = legacy.to_dict()
    for key in ("elapsed_seconds", "segments", "first_started_at"):
        payload.pop(key, None)
    path = mgr._checkpoint_path("course_planning")
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = _start(mgr, "course_planning")
    assert resumed.elapsed_seconds == 2400.0
    assert resumed.segments == 2


def test_graceful_stop_is_paused_not_completed(tmp_path: Path):
    """A stopped phase must be distinguishable from a finished one."""
    mgr = _mgr(tmp_path)
    _start(mgr)

    ck = mgr.pause_phase("training_synthesis", reason="Graceful stop requested")

    assert ck.status == "paused"
    assert ck.is_paused
    assert not ck.is_complete
    assert ck.can_resume, "a paused phase with pending tasks must re-enter"


def test_pause_also_closes_its_segment(tmp_path: Path):
    """A pause is a real segment boundary — its time counts."""
    mgr = _mgr(tmp_path)
    _start(mgr)
    ck = mgr.load_checkpoint("training_synthesis")
    start = datetime(2026, 7, 25, 12, 0, 0)
    ck.started_at = start.isoformat()
    mgr._write_checkpoint(ck)

    paused = mgr.pause_phase("training_synthesis", reason="stop")
    assert paused.elapsed_seconds > 0.0
    assert paused.completed_at is not None


def test_failed_phase_also_accumulates(tmp_path: Path):
    mgr = _mgr(tmp_path)
    _start(mgr)
    ck = mgr.fail_phase("training_synthesis", "Validation gates failed")

    assert ck.status == "failed"
    assert ck.elapsed_seconds >= 0.0
    assert ck.first_started_at is not None
