"""H3 Wave W5 — BloomAlignmentValidator decision-capture wiring.

Pins per-question ``bloom_alignment_check`` emission with dynamic
signals (question_id, declared_level, detected_level, match,
permissive_mode, aligned).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.bloom import BloomAlignmentValidator  # noqa: E402


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


def test_emits_one_decision_per_question() -> None:
    capture = _StubCapture()
    questions = [
        {
            "question_id": "Q1",
            "stem": "Explain how federation establishes shared trust.",
            "bloom_level": "understand",
        },
        {
            "question_id": "Q2",
            "stem": "Analyze the security trade-offs of single sign-on.",
            "bloom_level": "analyze",
        },
        {
            "question_id": "Q3",
            "stem": "Apply the federation pattern to a multi-tenant SaaS scenario.",
            "bloom_level": "apply",
        },
    ]
    BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 3
    types = {c["decision_type"] for c in capture.calls}
    assert types == {"bloom_alignment_check"}


def test_rationale_carries_dynamic_signals() -> None:
    capture = _StubCapture()
    BloomAlignmentValidator().validate({
        "assessment_data": {"questions": [{
            "question_id": "Q1",
            "stem": "Explain how federation establishes shared trust.",
            "bloom_level": "understand",
        }]},
        "decision_capture": capture,
    })
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "Q1" in rationale
    assert "declared_level=" in rationale
    assert "detected_level=" in rationale
    assert "verb_match=" in rationale
    assert "permissive_mode=" in rationale
    assert "aligned=" in rationale


def test_mismatch_emits_unaligned_decision() -> None:
    """A declared level that doesn't match the detected verb emits
    an ``unaligned`` decision with detected_level interpolated."""
    capture = _StubCapture()
    BloomAlignmentValidator().validate({
        "assessment_data": {"questions": [{
            "question_id": "Q1",
            # 'analyze' verb but declared 'remember'
            "stem": "Analyze the trade-offs of federation.",
            "bloom_level": "remember",
        }]},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "unaligned"
    assert "detected_level=analyze" in capture.calls[0]["rationale"]


def test_no_capture_no_emit_no_crash() -> None:
    questions = [{
        "question_id": "Q1",
        "stem": "Explain how federation establishes shared trust.",
        "bloom_level": "understand",
    }]
    base = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
    })
    captured = BloomAlignmentValidator().validate({
        "assessment_data": {"questions": questions},
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
