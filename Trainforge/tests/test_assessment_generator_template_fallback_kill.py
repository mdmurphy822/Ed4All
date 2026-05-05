"""Worker W1 — kill the assessment generator's deterministic template fallback.

The pre-W1 path emitted placeholder strings ("Correct answer based on
content", "Plausible distractor A", "Statement about ... content.",
etc.) into the assessment data when source_chunks was empty. The
strings then leaked into the Trainforge corpus and biased the trained
adapter. Worker W1 converts the fallback to a structured SkippedItem +
decision-capture event, never a placeholder.

These regressions pin the new behavior:

1. test_no_source_chunks_emits_skip_not_placeholder
2. test_with_source_chunks_no_skip
3. test_assessment_placeholder_patterns_no_longer_match_outputs
4. test_skip_count_surfaced_in_assessment_output
5. test_decision_event_assessment_template_skip_in_schema_enum
6. test_all_five_methods_skip_when_empty
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentGenerator,
    QuestionData,
    SkippedItem,
)
from lib.validators.assessment import ASSESSMENT_PLACEHOLDER_PATTERNS  # noqa: E402


# ---------- Fakes ----------


class _RecordingCapture:
    """Minimal stand-in for DecisionCapture; records log_decision calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kwargs,
            }
        )


# Sample chunks rich enough to exercise the content-grounded paths.
_SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "chunk_001",
        "chunk_id": "chunk_001",
        "text": (
            "<p><strong>Cognitive Load Theory</strong> is defined as "
            "the framework describing how working memory capacity "
            "limits the amount of information a learner can process "
            "simultaneously. Cognitive Load Theory was developed by "
            "John Sweller in 1988.</p>"
            "<p><strong>Intrinsic load</strong> refers to the inherent "
            "difficulty of the material being learned.</p>"
        ),
        "concept_tags": ["cognitive-load", "working-memory"],
        "difficulty": "intermediate",
    },
    {
        "id": "chunk_002",
        "chunk_id": "chunk_002",
        "text": (
            "<p>Extraneous load is caused by poorly designed instruction. "
            "Unlike intrinsic load, extraneous load can always be reduced "
            "through better instructional design.</p>"
        ),
        "concept_tags": ["cognitive-load", "instructional-design"],
    },
]


def _level_config(verb: str = "define", qtype: str = "multiple_choice") -> Dict[str, Any]:
    return {
        "verbs": [verb],
        "patterns": ["What is...?"],
        "question_types": [qtype],
    }


# ---------- Tests ----------


def test_no_source_chunks_emits_skip_not_placeholder() -> None:
    """When source_chunks is empty, the generator MUST emit a SkippedItem
    instead of a QuestionData carrying placeholder strings, AND the
    capture MUST log an `assessment_template_skip` decision event.
    """
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)

    result = gen._generate_question(
        objective_id="LO-001",
        bloom_level="remember",
        source_chunks=[],
    )

    assert isinstance(result, SkippedItem), (
        f"Expected SkippedItem on empty chunks; got {type(result).__name__}: {result}"
    )
    assert result.reason == "no_source_chunks"
    assert result.objective_id == "LO-001"
    assert result.bloom_level == "remember"

    skip_events = [
        e for e in capture.events if e["decision_type"] == "assessment_template_skip"
    ]
    assert skip_events, (
        "Expected at least one `assessment_template_skip` decision event "
        "to be logged when the fallback would have fired."
    )
    # Rationale must interpolate dynamic signals (objective_id, bloom).
    assert "LO-001" in skip_events[0]["rationale"]
    assert "remember" in skip_events[0]["rationale"]


def test_with_source_chunks_no_skip() -> None:
    """Happy path: with a real source corpus, the generator returns
    QuestionData (not SkippedItem) and no placeholder strings appear.
    """
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)

    result = gen._generate_multiple_choice(
        "Q-test-1",
        "LO-001",
        "remember",
        _level_config(),
        _SAMPLE_CHUNKS,
    )
    assert isinstance(result, QuestionData), (
        f"Expected content-grounded QuestionData; got {type(result).__name__}"
    )
    # Placeholder strings must not appear in the emitted question.
    serialized = json.dumps(result.to_dict())
    for pat in ASSESSMENT_PLACEHOLDER_PATTERNS:
        assert not pat.search(serialized), (
            f"Placeholder pattern {pat.pattern!r} matched grounded output: {serialized}"
        )


def test_assessment_placeholder_patterns_no_longer_match_outputs() -> None:
    """End-to-end: feed empty source_chunks through `generate()`; collect
    every emitted question's full JSON serialization; assert NONE of the
    `ASSESSMENT_PLACEHOLDER_PATTERNS` regex matches anything in the output.
    The pre-W1 path would have produced 5+ placeholder hits per run.
    """
    capture = _RecordingCapture()
    gen = AssessmentGenerator(capture=capture, check_leaks=False)

    assessment = gen.generate(
        course_code="TEST_101",
        objective_ids=["LO-001", "LO-002"],
        bloom_levels=["remember", "understand"],
        question_count=4,
        source_chunks=None,
    )

    # All slots should have skipped — no real chunks → no real questions.
    assert len(assessment.questions) == 0, (
        f"Expected zero emitted questions on empty chunks; got "
        f"{len(assessment.questions)}: {[q.to_dict() for q in assessment.questions]}"
    )
    assert len(assessment.skipped_items) > 0

    full = json.dumps(assessment.to_dict())
    for pat in ASSESSMENT_PLACEHOLDER_PATTERNS:
        match = pat.search(full)
        assert match is None, (
            f"Placeholder pattern {pat.pattern!r} matched generator output "
            f"at {match.start() if match else '?'}: {match.group() if match else ''}"
        )


def test_skip_count_surfaced_in_assessment_output() -> None:
    """The AssessmentData.to_dict() shape MUST surface skipped_items_count
    and skipped_items_summary so the workflow phase output (and the MCP
    tool envelope built from it) carries visibility on the dropped slots.
    """
    gen = AssessmentGenerator(capture=None, check_leaks=False)

    assessment = gen.generate(
        course_code="TEST_201",
        objective_ids=["LO-001"],
        bloom_levels=["apply"],
        question_count=3,
        source_chunks=None,
    )
    payload = assessment.to_dict()

    assert "skipped_items_count" in payload
    assert payload["skipped_items_count"] == len(assessment.skipped_items)
    assert payload["skipped_items_count"] > 0
    assert "skipped_items_summary" in payload
    assert isinstance(payload["skipped_items_summary"], list)
    # Summary capped at 3 entries.
    assert len(payload["skipped_items_summary"]) <= 3
    if payload["skipped_items_summary"]:
        first = payload["skipped_items_summary"][0]
        assert first["reason"] == "no_source_chunks"
        assert first["objective_id"] == "LO-001"
        assert first["bloom_level"] == "apply"


def test_decision_event_assessment_template_skip_in_schema_enum() -> None:
    """Worker W1 added `assessment_template_skip` to the canonical
    decision_event enum so DECISION_VALIDATION_STRICT runs don't fail.
    """
    schema_path = (
        PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum_values = schema["properties"]["decision_type"]["enum"]
    assert "assessment_template_skip" in enum_values, (
        "Worker W1: `assessment_template_skip` must be registered in "
        "schemas/events/decision_event.schema.json::decision_type.enum"
    )


def test_all_five_methods_skip_when_empty() -> None:
    """Each of the five `_generate_*` methods must return SkippedItem
    when source_chunks is empty (no placeholder paths remain).
    """
    gen = AssessmentGenerator(capture=None, check_leaks=False)

    cases = [
        ("_generate_multiple_choice", "multiple_choice"),
        ("_generate_true_false", "true_false"),
        ("_generate_fill_in_blank", "fill_in_blank"),
        ("_generate_essay", "essay"),
        ("_generate_short_answer", "short_answer"),
    ]
    for method_name, qtype in cases:
        method = getattr(gen, method_name)
        result = method(
            f"Q-{qtype}",
            "LO-001",
            "understand",
            _level_config(qtype=qtype),
            None,
        )
        assert isinstance(result, SkippedItem), (
            f"{method_name} returned {type(result).__name__}, expected SkippedItem"
        )
        assert result.question_type == qtype
        assert result.reason == "no_source_chunks"


def test_skipped_items_count_field_visible_via_to_dict() -> None:
    """Defense-in-depth: assert that even a partially-skipped run (some
    questions emit, some skip) surfaces the count correctly. We can't
    easily produce a half-skip on the heuristic content extractor, so
    we exercise the all-skip and all-emit branches separately as a
    sanity proxy for the dataclass plumbing.
    """
    gen = AssessmentGenerator(capture=None, check_leaks=False)

    # All-emit branch: real chunks → no skips.
    all_emit = gen.generate(
        course_code="TEST",
        objective_ids=["LO-001"],
        bloom_levels=["remember"],
        question_count=2,
        source_chunks=_SAMPLE_CHUNKS,
    )
    payload_emit = all_emit.to_dict()
    assert payload_emit["skipped_items_count"] == 0
    assert payload_emit["skipped_items_summary"] == []
    assert payload_emit["question_count"] == len(all_emit.questions)
