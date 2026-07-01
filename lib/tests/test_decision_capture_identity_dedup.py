"""DecisionCapture never creates a LibV2/courses skeleton (root fix).

Historically ``DecisionCapture`` constructed ``LibV2Storage(auto_create=True)``,
which created an empty ``courses/<slug>/`` skeleton just to log a decision (the
recurring split-brain "empty twin"). The root fix builds the storage handle
with ``auto_create=False``, so no course skeleton is ever created — regardless
of the (now no-op-from-decision-capture) ``ED4ALL_COURSE_IDENTITY_DEDUP`` flag.
These tests pin that behavior; the W0.5 dedup layer they used to exercise is
superseded by the root fix.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.decision_capture import DecisionCapture


@pytest.fixture
def libv2_root(tmp_path, monkeypatch):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(root))
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures"))
    return root


def _courses_children(root: Path):
    """Return the list of child entries under ``courses/`` (should stay empty)."""
    return list((root / "courses").iterdir())


@pytest.mark.integration
def test_flag_off_creates_no_course_skeleton(libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_COURSE_IDENTITY_DEDUP", raising=False)
    cap = DecisionCapture("demo-course", "phase-x", tool="trainforge")
    try:
        # Storage course_id is the verbatim course_code (unchanged).
        assert cap._storage.course_id == "demo-course"
        # ROOT FIX: no courses/<slug>/ skeleton was created just to log.
        assert _courses_children(libv2_root) == []
        assert not cap._storage.course_path.exists()
        # The capture output_dir still resolves under catalog/.../training/.
        assert cap.output_dir.exists()
        assert "training" in cap.output_dir.parts
    finally:
        cap.close()


@pytest.mark.integration
def test_flag_on_creates_no_course_skeleton(libv2_root, monkeypatch):
    # The dedup flag is now a no-op from decision capture: still no skeleton.
    monkeypatch.setenv("ED4ALL_COURSE_IDENTITY_DEDUP", "true")
    cap = DecisionCapture("Ed4All", "phase-x", tool="trainforge")
    try:
        assert _courses_children(libv2_root) == []
        assert not cap._storage.course_path.exists()
    finally:
        cap.close()


@pytest.mark.integration
def test_logging_writes_capture_but_no_skeleton(libv2_root, monkeypatch):
    monkeypatch.delenv("ED4ALL_COURSE_IDENTITY_DEDUP", raising=False)
    cap = DecisionCapture("Ed4All", "phase-x", tool="trainforge")
    try:
        cap.log_decision(
            decision_type="content_selection",
            decision="noop",
            rationale="exercise the capture write path and assert no skeleton.",
        )
        # A decisions_*.jsonl landed under the training path...
        written = list(cap.output_dir.glob("decisions_*.jsonl"))
        assert written, "expected a decisions_*.jsonl under the training path"
        assert "training" in cap.output_dir.parts
        # ...and still no courses/ skeleton was created.
        assert _courses_children(libv2_root) == []
    finally:
        cap.close()
