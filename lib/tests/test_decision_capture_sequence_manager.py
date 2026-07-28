"""Regression coverage for run-scoped DecisionCapture event sequencing.

These tests deliberately construct captures outside ``RunManager``.  Each
test uses a UUID-backed run ID because the sequence manager retains counters
for the life of the Python process.
"""

from __future__ import annotations

import threading
import uuid

from lib.decision_capture import DecisionCapture


def _run_id(label: str) -> str:
    return f"test-{label}-{uuid.uuid4().hex}"


def _capture() -> DecisionCapture:
    return DecisionCapture(
        course_code="TEST_001",
        phase="training-synthesis",
        tool="trainforge",
        streaming=False,
    )


def _log(capture: DecisionCapture, ordinal: int) -> None:
    capture.log_decision(
        decision_type="instruction_pair_synthesis",
        decision=f"accepted synthetic decision {ordinal}",
        rationale=(
            f"Sequence regression decision {ordinal} uses a dynamic ordinal "
            "and explicit run identity."
        ),
    )


def test_standalone_capture_sequence_is_positive_and_strictly_increasing(
    monkeypatch,
) -> None:
    run_id = _run_id("increasing")
    monkeypatch.setenv("ED4ALL_RUN_ID", run_id)
    monkeypatch.delenv("RUN_ID", raising=False)

    capture = _capture()
    for ordinal in range(4):
        _log(capture, ordinal)

    assert capture.run_id == run_id
    assert [record["seq"] for record in capture.decisions] == [1, 2, 3, 4]


def test_ed4all_run_id_takes_precedence_over_legacy_run_id(monkeypatch) -> None:
    canonical_run_id = _run_id("canonical")
    monkeypatch.setenv("ED4ALL_RUN_ID", canonical_run_id)
    monkeypatch.setenv("RUN_ID", _run_id("legacy"))

    capture = _capture()
    _log(capture, 0)

    assert capture.run_id == canonical_run_id
    assert capture.decisions[0]["run_id"] == canonical_run_id
    assert capture.decisions[0]["seq"] == 1


def test_shared_capture_allocates_thread_safe_unique_sequences_and_event_ids(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ED4ALL_RUN_ID", _run_id("threaded"))
    capture = _capture()
    worker_count = 64

    threads = [
        threading.Thread(target=_log, args=(capture, ordinal))
        for ordinal in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sequences = [record["seq"] for record in capture.decisions]
    event_ids = [record["event_id"] for record in capture.decisions]
    assert len(sequences) == worker_count
    assert all(sequence > 0 for sequence in sequences)
    assert len(set(sequences)) == worker_count
    assert set(sequences) == set(range(1, worker_count + 1))
    assert len(set(event_ids)) == worker_count
    assert all(event_id.startswith("EVT_") for event_id in event_ids)


def test_distinct_run_ids_each_begin_at_one(monkeypatch) -> None:
    first_run_id = _run_id("first")
    second_run_id = _run_id("second")

    monkeypatch.setenv("ED4ALL_RUN_ID", first_run_id)
    first_capture = _capture()
    _log(first_capture, 0)

    monkeypatch.setenv("ED4ALL_RUN_ID", second_run_id)
    second_capture = _capture()
    _log(second_capture, 0)

    assert first_capture.decisions[0]["seq"] == 1
    assert second_capture.decisions[0]["seq"] == 1
    assert first_capture.decisions[0]["run_id"] == first_run_id
    assert second_capture.decisions[0]["run_id"] == second_run_id
