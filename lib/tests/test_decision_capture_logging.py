"""Wave 29 Defect 4 — decision-capture stderr quieting.

Pre-Wave-29 every non-strict validation issue hit ``logger.warning``,
flooding stderr with hundreds of ``Decision validation issues: [...]``
lines on a normal run and burying real errors. Wave 29 demotes the
non-strict path to ``logger.debug`` and adds a one-line INFO summary
at capture close.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.decision_capture import DecisionCapture


@pytest.fixture
def mock_storage(tmp_path):
    """Mock LibV2Storage + LEGACY_TRAINING_DIR to redirect capture dirs."""
    with patch("lib.decision_capture.LibV2Storage") as mock_cls:
        storage = Mock()
        capture_path = tmp_path / "libv2" / "training"
        capture_path.mkdir(parents=True)
        storage.get_training_capture_path.return_value = capture_path
        mock_cls.return_value = storage

        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir(parents=True)
        with patch("lib.decision_capture.LEGACY_TRAINING_DIR", legacy_dir):
            yield tmp_path


def _emit_invalid_decision(cap: DecisionCapture):
    """Emit a decision whose ``decision_type`` is unknown — triggers
    the validation-issue path."""
    cap.log_decision(
        decision_type="definitely_not_a_real_decision_type_xyz",
        decision="x",
        rationale="short",  # too short on top of the unknown type
    )


def test_validation_issue_does_not_hit_warning_in_non_strict(
    mock_storage, caplog, monkeypatch
):
    """Wave 29: validation issues in non-strict mode emit at DEBUG,
    not WARNING. Previous behaviour flooded stderr with one WARNING per
    record; real errors got buried."""
    monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)

    cap = DecisionCapture(
        course_code="TEST_001",
        phase="test-phase",
        tool="trainforge",
        streaming=False,
    )

    with caplog.at_level(logging.WARNING, logger="lib.decision_capture"):
        _emit_invalid_decision(cap)

    # No WARNING-level record should contain "Decision validation issues".
    warning_messages = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    offenders = [m for m in warning_messages if "Decision validation issues" in m]
    assert not offenders, (
        f"Validation issues should log at DEBUG in non-strict mode; "
        f"unexpected WARNINGs: {offenders}"
    )


def test_validation_issue_fires_at_debug_level(mock_storage, caplog, monkeypatch):
    """The debug-level emission still fires so ``-v`` users see per-record
    detail — we don't silently drop the signal."""
    monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    # Force reload of constants flag since it's cached on import.
    import importlib
    import lib.constants
    importlib.reload(lib.constants)
    import lib.decision_capture as dc_module
    importlib.reload(dc_module)

    cap = dc_module.DecisionCapture(
        course_code="TEST_001",
        phase="test-phase",
        tool="trainforge",
        streaming=False,
    )

    with caplog.at_level(logging.DEBUG, logger="lib.decision_capture"):
        _emit_invalid_decision(cap)

    debug_messages = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG
    ]
    assert any(
        "Decision validation issues" in m for m in debug_messages
    ), f"Expected DEBUG Decision validation issue msg, got: {debug_messages}"


def test_strict_mode_still_raises(mock_storage, monkeypatch):
    """Wave 29 must not quiet the strict path — fail-closed callers
    still raise."""
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    import importlib
    import lib.constants
    importlib.reload(lib.constants)
    import lib.decision_capture as dc_module
    importlib.reload(dc_module)

    cap = dc_module.DecisionCapture(
        course_code="TEST_001",
        phase="test-phase",
        tool="trainforge",
        streaming=False,
    )

    with pytest.raises(ValueError, match="strict mode"):
        cap.log_decision(
            decision_type="definitely_not_a_real_decision_type_xyz",
            decision="x",
            rationale="short",
        )


def test_summary_info_line_emitted_on_save(mock_storage, caplog, monkeypatch):
    """Wave 29: a single INFO-level summary line fires at ``.save()``
    reporting the total decision + validation-issue count."""
    monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)

    cap = DecisionCapture(
        course_code="TEST_001",
        phase="test-phase",
        tool="trainforge",
        streaming=False,
    )
    # Emit a few valid decisions.
    for _ in range(3):
        cap.log_decision(
            decision_type="structure_detection",
            decision="standard layout",
            rationale="This rationale is definitely long enough to satisfy the 20-char quality gate.",
        )

    with caplog.at_level(logging.INFO, logger="lib.decision_capture"):
        cap.save("test_save.json")

    info_messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    ]
    summary_lines = [
        m for m in info_messages if "Captured" in m and "decisions" in m
    ]
    assert summary_lines, (
        f"Expected a 'Captured N decisions' INFO summary line, got: "
        f"{info_messages}"
    )


def test_validation_issue_count_tracked_internally(mock_storage, monkeypatch):
    """The internal counter increments as invalid decisions come through,
    and surfaces in the saved summary block."""
    monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    import importlib
    import lib.constants
    importlib.reload(lib.constants)
    import lib.decision_capture as dc_module
    importlib.reload(dc_module)

    cap = dc_module.DecisionCapture(
        course_code="TEST_001",
        phase="test-phase",
        tool="trainforge",
        streaming=False,
    )
    for _ in range(4):
        cap.log_decision(
            decision_type="never_a_real_decision_type_abc",
            decision="x",
            rationale="short",
        )

    assert getattr(cap, "_validation_issue_count", 0) >= 1, (
        "Expected the internal validation-issue counter to increment"
    )

    output = cap.save("count_test.json")
    import json as _json
    data = _json.loads(Path(output).read_text())
    # Summary carries the aggregate count.
    assert "total_validation_issues" in data["summary"]
    assert data["summary"]["total_validation_issues"] >= 1
