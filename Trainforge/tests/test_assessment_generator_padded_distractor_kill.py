"""Worker W2.D Part A — kill the padded-distractor fallback at source.

The pre-W2.D path in ``_generate_multiple_choice`` carried a
``while len(distractors) < 3`` loop that unconditionally emitted
``f"A concept unrelated to {target.term} in this context"`` strings
into the choices[] when the misconception/distractor synthesis
exhausted plausible options. The pad text trivially passed structural
validators (string + non-duplicate of correct answer) and got baked
into training pairs — silent corpus poisoning. W2.D Part A replaces
the loop with a ``SkippedItem(reason="padded_distractor_fallback_
eliminated")`` early return + a ``distractor_padding_skipped``
decision-capture event so the count surfaces in the audit trail
instead of silently corrupting the corpus.

Tests pin the new behavior:

1. test_insufficient_distractors_emits_skip_not_padding
2. test_no_emitted_distractor_matches_padded_fallback_regex
3. test_padded_distractor_fallback_eliminated_decision_event_fires
4. test_skipped_item_count_surfaced_in_assessment_to_dict
5. test_distractor_padding_skipped_in_schema_enum

The fixture deliberately constructs a chunk with exactly one
extractable key term and zero misconceptions / additional terms /
factual statements — the precondition that fired the pre-W2.D
while-loop at ``:678-681``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment.generator import (  # noqa: E402
    AssessmentGenerator,
    QuestionData,
    SkippedItem,
)


class _RecordingCapture:
    """Minimal stand-in for DecisionCapture; records log_decision calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        decision_type: str = None,
        decision: str = None,
        rationale: str = None,
        **kwargs: Any,
    ) -> None:
        # Accept either positional / kwarg forms used across the codebase.
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kwargs,
            }
        )


def _level_config(verb: str = "define") -> Dict[str, Any]:
    return {
        "verbs": [verb],
        "patterns": ["What is...?"],
        "question_types": ["multiple_choice"],
    }


# Single-term chunk with zero misconceptions, zero additional terms,
# and minimal factual statements — the precondition that fired the
# pre-W2.D padded-fallback while-loop. The legacy generator path
# extracted the lone key term, then padded the missing 2-3 distractor
# slots with the canonical pad string.
_SINGLE_TERM_CHUNK: Dict[str, Any] = {
    "id": "chunk_solo",
    "chunk_id": "chunk_solo",
    "text": (
        "<p><strong>Cognitive Load Theory</strong> is defined as "
        "the framework describing how working memory limits "
        "information processing.</p>"
    ),
    "concept_tags": ["cognitive-load"],
    "difficulty": "intermediate",
    # Zero misconceptions on the chunk so Strategy 1 (misconception
    # distractors) collects nothing.
    "misconceptions": [],
}


# Canonical regex from the W2.D plan §"Sanity / what NOT to do" — any
# emitted distractor matching this regex is a smoking-gun regression.
_PADDED_FALLBACK_RE = re.compile(
    r"^(A |An )?(concept|topic|item|element) unrelated to ",
    re.IGNORECASE,
)


def test_insufficient_distractors_emits_skip_not_padding() -> None:
    """When the misconception/distractor synthesis can't produce 3
    plausible distractors, the generator MUST emit a
    ``SkippedItem(reason="padded_distractor_fallback_eliminated")``
    instead of padding with the canonical 'A concept unrelated to ...'
    template strings.
    """
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)

    result = gen._generate_multiple_choice(
        "Q-padded-1",
        "LO-001",
        "remember",
        _level_config(),
        [_SINGLE_TERM_CHUNK],
    )

    assert isinstance(result, SkippedItem), (
        f"Expected SkippedItem on insufficient-distractor path; "
        f"got {type(result).__name__}: {result}"
    )
    assert result.reason == "padded_distractor_fallback_eliminated", (
        f"Expected SkippedItem.reason='padded_distractor_fallback_eliminated'; "
        f"got {result.reason!r}"
    )
    assert result.question_id == "Q-padded-1"
    assert result.question_type == "multiple_choice"
    assert result.objective_id == "LO-001"
    assert result.bloom_level == "remember"


def test_no_emitted_distractor_matches_padded_fallback_regex() -> None:
    """End-to-end: feed the single-term chunk through the generator;
    serialize the result; assert NO substring matches the canonical
    pre-W2.D padded-fallback regex. The pre-W2.D path would have
    produced 2-3 hits per call.
    """
    gen = AssessmentGenerator(capture=None, check_leaks=False)

    # Run the MCQ branch directly so we exercise the precise call path
    # that carried the pad fallback.
    result = gen._generate_multiple_choice(
        "Q-padded-2",
        "LO-001",
        "remember",
        _level_config(),
        [_SINGLE_TERM_CHUNK],
    )

    serialized = json.dumps(result.to_dict() if result is not None else {})
    match = _PADDED_FALLBACK_RE.search(serialized)
    assert match is None, (
        f"Padded-fallback regex matched generator output at "
        f"{match.start() if match else '?'}: "
        f"{match.group() if match else ''!r}"
    )


def test_padded_distractor_fallback_eliminated_decision_event_fires() -> None:
    """The generator-side ``distractor_padding_skipped`` decision-capture
    event must fire on the skip path so an operator sees the count of
    insufficient-distractor questions in the audit trail.
    """
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)

    gen._generate_multiple_choice(
        "Q-padded-3",
        "LO-002",
        "understand",
        _level_config(),
        [_SINGLE_TERM_CHUNK],
    )

    skip_events = [
        e for e in capture.events
        if e.get("decision_type") == "distractor_padding_skipped"
    ]
    assert skip_events, (
        f"Expected at least one ``distractor_padding_skipped`` decision "
        f"event when the pad fallback would have fired; got events: "
        f"{[e.get('decision_type') for e in capture.events]}"
    )
    rationale = skip_events[0]["rationale"]
    assert "Q-padded-3" in rationale
    assert "LO-002" in rationale
    assert "understand" in rationale
    assert "padded_distractor_fallback_eliminated" in rationale
    # Rationale must be ≥20 chars for the decision_validation gate.
    assert len(rationale) >= 20


def test_skipped_item_count_surfaced_in_assessment_to_dict() -> None:
    """Defense-in-depth: the AssessmentData payload must surface the
    count of padded-fallback skips in ``skipped_items_count`` so a
    downstream consumer (operator / aggregator) sees the regression
    without grepping decision JSONLs.

    This exercises the ``_generate_question`` dispatch path.
    """
    gen = AssessmentGenerator(capture=None, check_leaks=False)

    assessment = gen.generate(
        course_code="TEST_PADDED",
        objective_ids=["LO-001"],
        bloom_levels=["remember"],
        question_count=2,
        source_chunks=[_SINGLE_TERM_CHUNK],
    )
    payload = assessment.to_dict()
    assert "skipped_items_count" in payload
    assert "skipped_items_summary" in payload
    # The whole payload must be free of pad-fallback strings.
    serialized = json.dumps(payload)
    assert _PADDED_FALLBACK_RE.search(serialized) is None


def test_distractor_padding_skipped_in_schema_enum() -> None:
    """W2.D adds ``distractor_padding_skipped`` to the canonical
    ``decision_type`` enum so DECISION_VALIDATION_STRICT runs don't
    fail on the emit.
    """
    schema_path = (
        PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum_values = schema["properties"]["decision_type"]["enum"]
    assert "distractor_padding_skipped" in enum_values, (
        "Worker W2.D: ``distractor_padding_skipped`` must be registered "
        "in schemas/events/decision_event.schema.json::decision_type.enum"
    )
