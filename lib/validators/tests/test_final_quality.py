"""H3 Wave W5 — FinalQualityValidator decision-capture wiring.

Pins corpus-wide ``final_quality_check`` cardinality (one emit per
``validate()`` call) with dynamic signals (n_assessments,
total_questions, duplicate_count, score, min_score) interpolated.

Note: ``FinalQualityValidator`` lives in ``lib/validators/assessment.py``
(canonical location per ``config/workflows.yaml``); there is no
``lib/validators/final_quality.py`` file. This test file is named for
the validator class, not the module path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.assessment import FinalQualityValidator  # noqa: E402


class _StubCapture:
    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def _assessment(question_count: int, *, prefix: str = "Q") -> dict:
    return {
        "questions": [
            {
                "question_id": f"{prefix}{i}",
                "stem": (
                    f"What does identity federation establish in cryptographic "
                    f"trust scenario {i}?"
                ),
                "question_type": "multiple_choice",
                "bloom_level": "understand",
                "choices": [
                    {"text": "A trust relationship across security domains.", "is_correct": True},
                    {"text": "A direct VPN tunnel between two providers.", "is_correct": False},
                    {"text": "A shared encryption key on both endpoints.", "is_correct": False},
                ],
                "feedback": (
                    "Federation describes a trust relationship across multiple "
                    "independent security domains via cryptographic assertions."
                ),
            }
            for i in range(question_count)
        ],
    }


def test_emits_one_corpus_wide_decision() -> None:
    """Corpus-wide cardinality contract: one emit per validate() call."""
    capture = _StubCapture()
    FinalQualityValidator().validate({
        "assessments": [_assessment(5)],
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision_type"] == "final_quality_check"


def test_rationale_carries_dynamic_signals() -> None:
    capture = _StubCapture()
    FinalQualityValidator().validate({
        "assessments": [_assessment(5)],
        "decision_capture": capture,
    })
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "n_assessments=1" in rationale
    assert "total_questions=5" in rationale
    assert "duplicate_stem_count=" in rationale
    assert "score=" in rationale
    assert "min_score=" in rationale


def test_empty_assessments_emits_no_assessments_failure() -> None:
    """Empty input still emits one decision, with NO_ASSESSMENTS code."""
    capture = _StubCapture()
    FinalQualityValidator().validate({
        "assessments": [],
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:NO_ASSESSMENTS"


def test_no_capture_no_emit_no_crash() -> None:
    validator = FinalQualityValidator()
    base = validator.validate({"assessments": [_assessment(5)]})
    capture = _StubCapture()
    captured = validator.validate({
        "assessments": [_assessment(5)],
        "decision_capture": capture,
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
