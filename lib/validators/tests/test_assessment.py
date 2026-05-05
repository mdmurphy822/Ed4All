"""H3 Wave W5 — AssessmentQualityValidator decision-capture wiring.

Pins per-question ``assessment_quality_check`` emission with dynamic
signals (question_id, placeholder_hits, bloom_level, is_mcq, issue_codes)
interpolated into the rationale. Mirrors the canonical exemplar at
``lib/validators/rewrite_source_grounding.py::_emit_decision``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.assessment import AssessmentQualityValidator  # noqa: E402


class _StubCapture:
    """Captures ``log_decision`` invocations for assertion."""

    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def _make_question(
    *,
    question_id: str,
    stem: str = "What is the meaning of federation in identity systems?",
    question_type: str = "multiple_choice",
    bloom_level: str = "understand",
    correct_text: str = "Federation describes a trust relationship across security domains.",
    distractors=None,
):
    if distractors is None:
        distractors = [
            "Bridging firewalls between two private networks securely",
            "Using a single sign-on token shared between two providers",
            "Synchronizing user data across replicated database servers",
        ]
    choices = [{"text": correct_text, "is_correct": True}]
    for d in distractors:
        choices.append({"text": d, "is_correct": False})
    return {
        "question_id": question_id,
        "stem": stem,
        "question_type": question_type,
        "bloom_level": bloom_level,
        "choices": choices,
        "feedback": (
            "Federation establishes shared trust across security domains "
            "via cryptographic assertions exchanged between providers."
        ),
    }


def test_emits_one_decision_per_question() -> None:
    """Per-question cardinality contract: 3 questions → 3 emits."""
    capture = _StubCapture()
    questions = [
        _make_question(question_id="Q1"),
        _make_question(question_id="Q2"),
        _make_question(question_id="Q3"),
    ]
    AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 3
    types = {c["decision_type"] for c in capture.calls}
    assert types == {"assessment_quality_check"}


def test_rationale_carries_dynamic_signals() -> None:
    """Rationale interpolates question_id, placeholder_hits, bloom_level,
    is_mcq, and issue_codes (regression-pin against static rationales)."""
    capture = _StubCapture()
    AssessmentQualityValidator().validate({
        "assessment_data": {"questions": [_make_question(question_id="Q1")]},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "Q1" in rationale
    assert "placeholder_hits=" in rationale
    assert "bloom_level=" in rationale
    assert "is_mcq=" in rationale
    assert "issue_codes=" in rationale


def test_no_capture_no_emit_no_crash() -> None:
    """Absent decision_capture → identical GateResult, no exception."""
    validator = AssessmentQualityValidator()
    base = validator.validate({
        "assessment_data": {"questions": [_make_question(question_id="Q1")]},
    })
    captured = validator.validate({
        "assessment_data": {"questions": [_make_question(question_id="Q1")]},
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
    assert [i.code for i in base.issues] == [i.code for i in captured.issues]


def test_placeholder_hit_surfaces_in_capture() -> None:
    """A placeholder-stem question emits a failure-coded decision."""
    capture = _StubCapture()
    q = _make_question(
        question_id="QP",
        stem="The key concept from LO-01 is _______ in this passage.",
    )
    AssessmentQualityValidator().validate({
        "assessment_data": {"questions": [q]},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision"].startswith("failed:")
    assert "PLACEHOLDER_QUESTION" in call["rationale"]
