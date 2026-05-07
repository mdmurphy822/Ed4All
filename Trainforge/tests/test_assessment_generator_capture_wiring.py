"""Wave W-D1 T1.5 — ``AssessmentGenerator`` capture-wiring regression test.

Pins the per-`generate()` decision-capture event chain emitted by
``Trainforge/generators/assessment_generator.py``. Closes the audit gap
flagged by ``plans/wave-D1-p0-fixes-2026-05-07.md`` § 2.5 (#1 of 6) and
enforces the CLAUDE.md "LLM call-site instrumentation" rule for the
assessment-generation surface.

Three assertions per file (mirrors
``Trainforge/tests/test_anthropic_synthesis_provider.py:336-398`` and
``Trainforge/tests/test_qualitative_judge_capture_wiring.py``):

1. With a wired capture, at least one event fires; the first event's
   ``decision_type`` is ``"assessment_planning"`` (the per-call planning
   emit at ``assessment_generator.py:292-309``).
2. Rationale interpolates dynamic signals — objective count + Bloom's
   level distribution — and is at least 20 chars.
3. ``capture=None`` is the back-compat default; no exceptions, no
   events. Pre-Wave-1.5 corpora rely on this surface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentGenerator,
)


# T1.5 stub — see plan §2 "Shared test-stub helpers". The
# assessment_generator path does NOT call ``_last_event_id``, so the
# event_id stamping is unnecessary; we still expose ``decisions`` for
# symmetry with the other five tests in the wave.
class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def _stub_chunks() -> List[Dict[str, Any]]:
    """Minimal stub chunk so ``_generate_question`` has source content
    to template against. The text contains a learning anchor so the
    leak checker (when enabled) doesn't flag every stem."""
    return [
        {
            "id": "c1",
            "text": (
                "Topic X is the foundational concept introduced in chapter 1. "
                "Learners must define and apply it across subsequent material."
            ),
            "learning_outcome_refs": ["TO-01"],
            "key_terms": [
                {"term": "Topic X", "definition": "foundational concept"}
            ],
        }
    ]


def test_capture_fires_on_generate_call():
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)
    gen.generate(
        course_code="TEST_101",
        objective_ids=["TO-01"],
        bloom_levels=["remember"],
        question_count=1,
        source_chunks=_stub_chunks(),
    )
    # First event MUST be assessment_planning per
    # ``assessment_generator.py:292-309``.
    assert len(capture.events) >= 1
    first = capture.events[0]
    assert first["decision_type"] == "assessment_planning"
    rationale = first["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals interpolated by ``assessment_generator.py:284-291``.
    assert "1 objectives" in rationale
    assert "Bloom" in rationale or "bloom" in rationale.lower()


def test_capture_rationale_carries_dynamic_signals():
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)
    gen.generate(
        course_code="TEST_101",
        objective_ids=["TO-01", "TO-02", "CO-01"],
        bloom_levels=["remember", "apply"],
        question_count=3,
        source_chunks=_stub_chunks(),
    )
    planning = next(
        e for e in capture.events
        if e["decision_type"] == "assessment_planning"
    )
    # 3-objective N must appear in rationale per the production format
    # string at assessment_generator.py:284-291.
    assert "3 objectives" in planning["rationale"]
    # Bloom's joined-list of two levels.
    assert "remember" in planning["rationale"]
    assert "apply" in planning["rationale"]


def test_no_capture_silent_no_op():
    """``capture=None`` is the back-compat default; ``generate()`` must
    return an ``AssessmentData`` without raising. Pre-Wave-1.5 corpora
    rely on this surface."""
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    result = gen.generate(
        course_code="TEST_101",
        objective_ids=["TO-01"],
        bloom_levels=["remember"],
        question_count=1,
        source_chunks=_stub_chunks(),
    )
    assert result is not None
