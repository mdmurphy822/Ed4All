"""
Regression tests for Wave4-I5 boilerplate-rationale detector.

Finding 6 of plans/dispatch-7-execution-inspection-2026-05.md:
62 of 63 source_selection decisions in a real dispatch-7 run shared a
byte-identical 165-char rationale. The detector fires at close() time by
scanning self.decisions grouped by decision_type; it warns when > 60% of
entries in a group (minimum 5 entries) share the same first-80-char prefix.
"""

import logging

import pytest


def _make_capture():
    """Return a DecisionCapture in non-streaming mode to avoid filesystem I/O."""
    from unittest.mock import MagicMock, patch

    # Patch the heavy __init__ dependencies so we don't need a full env
    with (
        patch("lib.decision_capture.LibV2Storage") as mock_storage,
        patch("lib.decision_capture.get_current_run", return_value=None),
        patch("lib.decision_capture.HARDENING_AVAILABLE", False),
    ):
        mock_storage.return_value.get_training_capture_path.return_value = MagicMock(
            __truediv__=lambda s, o: MagicMock()
        )
        from lib.decision_capture import DecisionCapture

        cap = DecisionCapture.__new__(DecisionCapture)
        cap.course_code = "TEST_001"
        cap.phase = "content-generator"
        cap.tool = "courseforge"
        cap.session_id = "20260513_120000_12345_aabbcc"
        cap.streaming_mode = False
        cap.task_id = None
        cap._run_context = None
        cap.run_id = "test_run"
        cap.course_id = "TEST_001"
        cap.module_id = None
        cap.artifact_id = None
        cap.decisions = []
        cap.prompts_responses = []
        cap.web_searches = []
        cap.files_created = []
        cap.sources_used = {}
        cap._stream_file = None
        cap._legacy_stream_file = None
        cap._run_stream_file = None
        cap._stream_path = None
        cap._legacy_stream_path = None
        cap._run_decisions_path = None
        return cap


def _make_decision(decision_type: str, rationale: str) -> dict:
    return {
        "decision_type": decision_type,
        "decision": "some decision",
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Test 1: 60 decisions, all sharing the same 80-char prefix → warning fires
# ---------------------------------------------------------------------------

def test_boilerplate_warning_fires_when_all_same(caplog):
    """60 source_selection decisions with identical rationale triggers warning."""
    cap = _make_capture()
    boilerplate = (
        "Selected source based on relevance to course objectives "
        "and alignment with the module theme for all weeks."
    )
    cap.decisions = [
        _make_decision("source_selection", boilerplate) for _ in range(60)
    ]

    with caplog.at_level(logging.WARNING, logger="lib.decision_capture"):
        cap._scan_boilerplate_rationales()

    assert any(
        "DUPLICATE_BOILERPLATE_RATIONALE" in msg and "source_selection" in msg
        for msg in caplog.messages
    ), f"Expected boilerplate warning, got: {caplog.messages}"


# ---------------------------------------------------------------------------
# Test 2: 60 decisions, only 50% share a prefix (below 60% threshold) → no warn
# ---------------------------------------------------------------------------

def test_no_warning_when_below_threshold(caplog):
    """60 decisions where 50% share the same prefix does not trigger warning."""
    cap = _make_capture()
    boilerplate = "A" * 100  # same prefix for 30 entries
    diverse_base = "B" * 80   # different prefix for other 30

    decisions = [_make_decision("source_selection", boilerplate) for _ in range(30)]
    decisions += [
        _make_decision("source_selection", diverse_base + str(i)) for i in range(30)
    ]
    cap.decisions = decisions

    with caplog.at_level(logging.WARNING, logger="lib.decision_capture"):
        cap._scan_boilerplate_rationales()

    assert not any(
        "DUPLICATE_BOILERPLATE_RATIONALE" in msg for msg in caplog.messages
    ), f"Unexpected boilerplate warning: {caplog.messages}"


# ---------------------------------------------------------------------------
# Test 3: Only 4 decisions (below 5-count floor) → no warning
# ---------------------------------------------------------------------------

def test_no_warning_below_min_count(caplog):
    """4 boilerplate decisions (below 5-count floor) does not trigger warning."""
    cap = _make_capture()
    boilerplate = "Identical rationale string repeated four times exactly here."
    cap.decisions = [
        _make_decision("source_selection", boilerplate) for _ in range(4)
    ]

    with caplog.at_level(logging.WARNING, logger="lib.decision_capture"):
        cap._scan_boilerplate_rationales()

    assert not any(
        "DUPLICATE_BOILERPLATE_RATIONALE" in msg for msg in caplog.messages
    ), f"Unexpected boilerplate warning: {caplog.messages}"


# ---------------------------------------------------------------------------
# Test 4: 60 decisions across 3 types, each type is diverse → no warning
# ---------------------------------------------------------------------------

def test_no_warning_when_diverse_within_each_type(caplog):
    """60 decisions across 3 diverse decision_types emits no boilerplate warning."""
    cap = _make_capture()
    decisions = []
    for dt in ("source_selection", "content_structure", "alt_text_generation"):
        for i in range(20):
            # Each rationale is unique within its type
            rationale = f"Decision {i} for {dt}: block_id=BLK-{i:04d} confidence={0.5 + i * 0.02:.3f}"
            decisions.append(_make_decision(dt, rationale))
    cap.decisions = decisions

    with caplog.at_level(logging.WARNING, logger="lib.decision_capture"):
        cap._scan_boilerplate_rationales()

    assert not any(
        "DUPLICATE_BOILERPLATE_RATIONALE" in msg for msg in caplog.messages
    ), f"Unexpected boilerplate warning: {caplog.messages}"
