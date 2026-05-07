"""Wave W-D1 T1.5 — ``QuestionFactory`` capture-wiring regression test.

Pins one ``question_creation`` event per ``create_*`` call (six call
sites at ``Trainforge/generators/question_factory.py:158-413``) plus
the ``bloom_alignment_rejection`` event emitted by
``validate_bloom_alignment`` at
``question_factory.py:455-460`` when a question type doesn't match its
target Bloom's level.

Capture is **optional** — ``capture=None`` is a silent no-op (the
factory's ``if self.capture:`` guards short-circuit cleanly).

Four assertions per file (per plan §2.5):

1. ``create_multiple_choice`` fires exactly one ``question_creation``
   event with rationale carrying bloom level + objective id signals.
2. ``create_true_false`` likewise fires one ``question_creation``
   event (regression net for the second call site).
3. ``capture=None`` is a silent no-op — no events, no exceptions, the
   ``Question`` is returned with the expected ``question_type``.
4. ``validate_bloom_alignment`` emits ``bloom_alignment_rejection``
   when an unsupported (type, bloom_level) pair is requested.
   ``create_essay(bloom_level="remember")`` triggers this path —
   ``BLOOM_QUESTION_MAP["remember"]`` does not include ``essay``
   per ``question_factory.py:91-98``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.question_factory import QuestionFactory  # noqa: E402


# T1.5 stub — see plan §2 "Shared test-stub helpers". The
# question_factory path does NOT call ``_last_event_id``, so event_id
# stamping is unnecessary; we still expose ``decisions`` for symmetry.
class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def test_create_mcq_fires_question_creation():
    capture = _RecordingCapture()
    factory = QuestionFactory(capture=capture, enforce_bloom_alignment=False)
    factory.create_multiple_choice(
        stem="What is X?",
        choices=[
            {"text": "A", "is_correct": True},
            {"text": "B", "is_correct": False},
        ],
        bloom_level="remember",
        objective_id="TO-01",
    )
    creation_events = [
        e for e in capture.events
        if e["decision_type"] == "question_creation"
    ]
    assert len(creation_events) == 1
    ev = creation_events[0]
    rationale = ev["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals per question_factory.py:160-162.
    assert "remember" in rationale
    assert "TO-01" in rationale


def test_create_true_false_fires_question_creation():
    capture = _RecordingCapture()
    factory = QuestionFactory(capture=capture, enforce_bloom_alignment=False)
    factory.create_true_false(
        stem="X is true.",
        correct_answer=True,
        bloom_level="remember",
        objective_id="TO-01",
    )
    creation_events = [
        e for e in capture.events
        if e["decision_type"] == "question_creation"
    ]
    assert len(creation_events) == 1
    ev = creation_events[0]
    rationale = ev["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals per question_factory.py:268.
    assert "remember" in rationale


def test_no_capture_silent_no_op():
    """``capture=None`` is the back-compat default; the factory must
    still return the expected ``Question`` shape without exceptions."""
    factory = QuestionFactory(capture=None, enforce_bloom_alignment=False)
    q = factory.create_multiple_choice(
        stem="What is X?",
        choices=[
            {"text": "A", "is_correct": True},
            {"text": "B", "is_correct": False},
        ],
        bloom_level="remember",
        objective_id="TO-01",
    )
    assert q.question_type == "multiple_choice"


def test_bloom_alignment_rejection_event_fires_on_mismatch():
    """``validate_bloom_alignment`` emits ``bloom_alignment_rejection``
    when an unexpected (type, bloom_level) pair is requested with
    ``enforce_bloom_alignment=True``. Per question_factory.py:455-460.

    ``create_essay(bloom_level="remember")`` is the trigger: the
    ``BLOOM_QUESTION_MAP["remember"]`` entry is
    ``[multiple_choice, true_false, fill_in_blank, matching]`` — no
    ``essay`` — so the rejection path fires + raises
    ``BloomAlignmentError``. The event is logged BEFORE the raise so
    audit replay records the rejection."""
    capture = _RecordingCapture()
    factory = QuestionFactory(capture=capture, enforce_bloom_alignment=True)
    try:
        factory.create_essay(
            stem="Discuss X.",
            bloom_level="remember",
            objective_id="TO-01",
        )
    except Exception:  # noqa: BLE001 — rejection-raise is expected
        pass
    rejections = [
        e for e in capture.events
        if e["decision_type"] == "bloom_alignment_rejection"
    ]
    assert len(rejections) >= 1
    rationale = rejections[0]["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals — the rationale interpolates the rejected
    # question_type + bloom_level combo per question_factory.py:450-460.
    assert "essay" in rationale
    assert "remember" in rationale
