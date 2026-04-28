#!/usr/bin/env python3
"""
Wave 112 Task 1 regression net.

Locks in the contract that ``Trainforge.synthesize_training._last_event_id``
fails loud when called against a ``DecisionCapture`` that has not logged any
decisions. The pre-Wave-112 implementation returned ``""`` here, which then
rode into the emitted ``instruction_pair`` / ``preference_pair`` JSONL as
``decision_capture_id: ""`` -- a schema-violating value that broke
strict-mode pair validation downstream.

The production synthesis loop logs a stage-start decision before it ever
asks for ``_last_event_id``, so a real run never hits the empty branch.
Failing loud at the helper level prevents future refactors from silently
re-introducing the empty fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402
from Trainforge.synthesize_training import _last_event_id  # noqa: E402


def test_last_event_id_raises_on_empty_capture():
    """An empty ``DecisionCapture`` must surface a RuntimeError, not ``""``."""
    capture = DecisionCapture(
        course_code="TEST_001",
        phase="synthesize-training",
        tool="trainforge",
        streaming=False,
    )
    assert capture.decisions == []
    with pytest.raises(RuntimeError, match="no decisions logged"):
        _last_event_id(capture)


def test_last_event_id_returns_event_id_after_log():
    """After at least one ``log_decision``, the helper returns the event_id."""
    capture = DecisionCapture(
        course_code="TEST_001",
        phase="synthesize-training",
        tool="trainforge",
        streaming=False,
    )
    capture.log_decision(
        decision_type="instruction_pair_synthesis",
        decision="Stage start sentinel for the integrity test.",
        rationale=(
            "Need at least one logged decision so _last_event_id has a tail "
            "entry to pull event_id off of; this mirrors the production "
            "synthesis loop's stage-start log."
        ),
    )
    event_id = _last_event_id(capture)
    assert isinstance(event_id, str)
    assert event_id, "event_id must be a non-empty string"
