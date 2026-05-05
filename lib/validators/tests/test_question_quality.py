"""H3 Wave W5 — QuestionQualityValidator decision-capture wiring.

Pins per-question ``question_quality_check`` emission with dynamic
signals (question_id, stem_grounding_jaccard, distractor_pairwise_max,
score, distractor_count).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.question_quality import QuestionQualityValidator  # noqa: E402


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


def _make_question(question_id: str, *, bloom_level: str = "understand"):
    return {
        "question_id": question_id,
        "stem": (
            "Explain how identity federation establishes trust across "
            "independent security domains."
        ),
        "question_type": "multiple_choice",
        "bloom_level": bloom_level,
        "choices": [
            {
                "text": (
                    "Federation establishes trust between independent security "
                    "domains via cryptographic assertions exchanged between "
                    "providers."
                ),
                "is_correct": True,
            },
            {
                "text": (
                    "A virtual private network tunnel between two enterprise "
                    "office locations sharing routing tables."
                ),
                "is_correct": False,
            },
            {
                "text": (
                    "A cluster of replicated database servers synchronizing "
                    "user records across data centers."
                ),
                "is_correct": False,
            },
        ],
        "feedback": (
            "Federation describes a trust relationship across security "
            "domains using cryptographic assertions issued by identity "
            "providers."
        ),
    }


_SOURCE_CHUNKS = [
    {"text": (
        "Federation establishes trust between independent security domains "
        "via cryptographic assertions. Identity providers issue assertions "
        "about authenticated users to relying parties."
    )},
]


def test_emits_one_decision_per_question() -> None:
    capture = _StubCapture()
    questions = [
        _make_question("Q1"),
        _make_question("Q2"),
        _make_question("Q3"),
    ]
    QuestionQualityValidator().validate({
        "assessment_data": {"questions": questions},
        "source_chunks": _SOURCE_CHUNKS,
        "decision_capture": capture,
    })
    assert len(capture.calls) == 3
    types = {c["decision_type"] for c in capture.calls}
    assert types == {"question_quality_check"}


def test_rationale_carries_dynamic_signals() -> None:
    capture = _StubCapture()
    QuestionQualityValidator().validate({
        "assessment_data": {"questions": [_make_question("Q1")]},
        "source_chunks": _SOURCE_CHUNKS,
        "decision_capture": capture,
    })
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "Q1" in rationale
    assert "stem_grounding_jaccard=" in rationale
    assert "distractor_pairwise_jaccard_max=" in rationale
    assert "composite_score=" in rationale
    assert "distractor_count=" in rationale


def test_no_capture_no_emit_no_crash() -> None:
    base = QuestionQualityValidator().validate({
        "assessment_data": {"questions": [_make_question("Q1")]},
        "source_chunks": _SOURCE_CHUNKS,
    })
    captured = QuestionQualityValidator().validate({
        "assessment_data": {"questions": [_make_question("Q1")]},
        "source_chunks": _SOURCE_CHUNKS,
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
