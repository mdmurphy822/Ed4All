"""Worker W2 — AssessmentRetrievalGroundingValidator unit tests.

Pins the validator's per-block dispatch + four GateIssue codes on
representative outline-tier (dict content) and rewrite-tier (HTML
string content) Block fixtures + non-assessment_item no-op path +
decision-capture wiring assertion.

Failure modes:

- ``ANSWER_NOT_GROUNDED`` — Jaccard overlap below threshold.
- ``ANSWER_TEXT_MISSING`` — assessment_item with no recoverable
  correct-answer text on either content shape.
- ``NO_SOURCE_ATTRIBUTION`` (warning) — block declares no source IDs;
  validator falls back to all-chunk union.
- ``ESSAY_SKIPPED`` (info) — essay-mode block; skipped from gate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Block lives at Courseforge/scripts/blocks.py — mirror the import
# bridge the validator uses internally.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.assessment_retrieval_grounding import (  # noqa: E402
    AssessmentRetrievalGroundingValidator,
    DEFAULT_MIN_OVERLAP_JACCARD,
)


# --------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal DecisionCapture stand-in recording log_decision kwargs."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _outline_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    answer_key: Optional[str] = (
        "An RDF triple is a subject-predicate-object statement that "
        "expresses a fact in the resource description framework graph."
    ),
    correct_answer: Optional[str] = None,
    distractors: Optional[List[Dict[str, Any]]] = None,
    correct_answer_index: int = 0,
    drop_answer_key: bool = False,
    question_type: Optional[str] = None,
    content_type: str = "assessment_item",
    source_ids: tuple = ("dart:rdf_intro#blk_0",),
    source_references: tuple = (),
) -> Block:
    if distractors is None:
        distractors = [
            {"text": answer_key or ""},
            {"text": "An XML namespace declaration."},
            {"text": "A Turtle prefix declaration."},
            {"text": "A SHACL node shape."},
        ]
    content: Dict[str, Any] = {
        "curies": ["rdf:Statement"],
        "key_claims": [
            "An RDF triple is a subject-predicate-object statement.",
        ],
        "content_type": content_type,
        "stem": "What is the primary purpose of an RDF triple?",
        "distractors": distractors,
        "correct_answer_index": correct_answer_index,
    }
    if not drop_answer_key and answer_key is not None:
        content["answer_key"] = answer_key
    if correct_answer is not None:
        content["correct_answer"] = correct_answer
    if question_type is not None:
        content["question_type"] = question_type

    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=("TO-01",),
        source_ids=source_ids,
        source_references=source_references,
    )


def _rewrite_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    correct_answer_text: str = (
        "An RDF triple is a subject-predicate-object statement expressing "
        "a fact in the resource description framework graph."
    ),
    source_ids: tuple = ("dart:rdf_intro#blk_0",),
) -> Block:
    html = (
        "<section data-cf-block-id=\"" + block_id + "\">\n"
        "  <p>What is the primary purpose of an RDF triple?</p>\n"
        "  <ol data-cf-correct-answer-index=\"0\">\n"
        "    <li data-cf-distractor-index=\"0\" data-cf-correct=\"true\">"
        + correct_answer_text + "</li>\n"
        "    <li data-cf-distractor-index=\"1\">An XML namespace declaration.</li>\n"
        "    <li data-cf-distractor-index=\"2\">A Turtle prefix declaration.</li>\n"
        "    <li data-cf-distractor-index=\"3\">A SHACL node shape.</li>\n"
        "  </ol>\n"
        "</section>"
    )
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=html,
        objective_ids=("TO-01",),
        source_ids=source_ids,
    )


def _non_assessment_block() -> Block:
    return Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "curies": ["ed4all:Concept"],
            "key_claims": ["A concept block introduces a new term."],
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


def _grounded_chunk_text() -> str:
    return (
        "An RDF triple is the foundational unit of the resource description "
        "framework graph: it expresses a fact as a subject, predicate, and "
        "object statement. Each component identifies a node or property in "
        "the directed labelled graph that the framework constructs."
    )


def _ungrounded_chunk_text() -> str:
    return (
        "Computer networks transport data across physical and wireless "
        "links between hosts. Routers forward packets through autonomous "
        "systems using border gateway protocol exchanges."
    )


def _codes(result) -> List[str]:
    return [issue.code for issue in result.issues]


# --------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------- #


def test_outline_grounded_answer_passes():
    """Outline assessment_item whose answer text shares >=30% Jaccard
    with the referenced source chunk → passed=True, action=None."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None
    assert result.score == 1.0
    assert "ANSWER_NOT_GROUNDED" not in _codes(result)


def test_rewrite_grounded_answer_passes():
    """Rewrite-tier assessment_item with <li data-cf-correct=\"true\">
    body that overlaps the source chunk → passed=True."""
    blocks = [_rewrite_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None


# --------------------------------------------------------------------- #
# Fail path — ANSWER_NOT_GROUNDED
# --------------------------------------------------------------------- #


def test_outline_ungrounded_answer_fails_critical():
    """Outline assessment_item whose answer talks about RDF but the
    source chunk discusses computer networks → ANSWER_NOT_GROUNDED."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _ungrounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ANSWER_NOT_GROUNDED" in _codes(result)
    # Critical-severity per spec.
    not_grounded = [
        i for i in result.issues if i.code == "ANSWER_NOT_GROUNDED"
    ]
    assert all(i.severity == "critical" for i in not_grounded)


def test_rewrite_ungrounded_answer_fails_critical():
    """Rewrite-tier assessment_item whose <li data-cf-correct> body
    is unrelated to source chunks → ANSWER_NOT_GROUNDED."""
    blocks = [_rewrite_assessment_block(
        correct_answer_text=(
            "Quantum chromodynamics describes the strong interaction "
            "between quarks via gluon exchange in particle accelerator "
            "experiments and lattice simulations."
        )
    )]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert "ANSWER_NOT_GROUNDED" in _codes(result)


def test_threshold_override_via_inputs():
    """``inputs['min_overlap_jaccard']`` overrides the constructor floor."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    # Reasonable grounding passes the default 0.30 floor; pushing the
    # threshold to 0.99 should fail it.
    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "min_overlap_jaccard": 0.99,
    })

    assert result.passed is False
    assert "ANSWER_NOT_GROUNDED" in _codes(result)


# --------------------------------------------------------------------- #
# ANSWER_TEXT_MISSING
# --------------------------------------------------------------------- #


def test_outline_no_answer_text_fires_critical():
    """Outline assessment_item with no answer_key / correct_answer /
    distractors[cai].text → ANSWER_TEXT_MISSING critical."""
    blocks = [_outline_assessment_block(
        drop_answer_key=True,
        # Make distractor[0].text empty too so the index fallback fails.
        distractors=[
            {"text": ""},
            {"text": "An XML namespace declaration."},
        ],
        correct_answer_index=0,
    )]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert "ANSWER_TEXT_MISSING" in _codes(result)


def test_rewrite_no_correct_li_fires_critical():
    """Rewrite-tier assessment_item with no <li data-cf-correct> AND
    no data-cf-correct-answer-index → ANSWER_TEXT_MISSING critical."""
    html = (
        "<section data-cf-block-id=\"page_01#assessment_item_q_0\">\n"
        "  <p>What is the primary purpose of an RDF triple?</p>\n"
        "  <ol>\n"
        "    <li>A subject-predicate-object statement.</li>\n"
        "    <li>An XML namespace declaration.</li>\n"
        "  </ol>\n"
        "</section>"
    )
    block = Block(
        block_id="page_01#assessment_item_q_0",
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=html,
        objective_ids=("TO-01",),
        source_ids=("dart:rdf_intro#blk_0",),
    )
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": [block],
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert "ANSWER_TEXT_MISSING" in _codes(result)


# --------------------------------------------------------------------- #
# NO_SOURCE_ATTRIBUTION (warning)
# --------------------------------------------------------------------- #


def test_no_source_ids_emits_warning_and_falls_back():
    """Block declaring no source_ids / source_references → warning,
    validator falls back to union of all chunks for the overlap check.

    Two chunks are supplied so the fallback path exercises the
    all-chunk-union behaviour rather than degenerating to a
    single-chunk lookup. The grounded chunk provides enough overlap
    against the answer that the union path passes the 0.30 floor
    even when the unrelated chunk dilutes the denominator slightly;
    the unrelated-chunk dilution is bounded because the union token
    set is small and the grounded token overlap is high.
    """
    blocks = [_outline_assessment_block(source_ids=(), source_references=())]
    # Two grounded chunks (plus an unrelated one) so the union still
    # carries enough overlap signal that the answer passes the floor —
    # the test's load-bearing assertion is that the warning fires AND
    # fallback occurred, not the dilution math.
    chunks_lookup = {
        "dart:rdf_intro#blk_0": _grounded_chunk_text(),
        "dart:rdf_intro#blk_1": _grounded_chunk_text(),
        "dart:misc#blk_0": _ungrounded_chunk_text(),
    }
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "NO_SOURCE_ATTRIBUTION" in codes, codes
    warnings = [i for i in result.issues if i.code == "NO_SOURCE_ATTRIBUTION"]
    assert all(i.severity == "warning" for i in warnings)
    # The validator records that fallback was used in ml_features
    # regardless of pass/fail outcome — assert the per-block payload
    # carries the fallback flag (via a fresh validate() with capture).
    capture = _RecordingCapture()
    validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })
    per_block = capture.calls[0]["ml_features"]["per_block"]
    assert per_block, "expected at least one per-block feature row"
    assert per_block[0].get("fallback_used") is True


def test_no_source_ids_with_unrelated_chunks_still_fails_grounding():
    """When the all-chunk union ALSO doesn't ground the answer, the
    block correctly fails with ANSWER_NOT_GROUNDED on top of the
    NO_SOURCE_ATTRIBUTION warning — the fallback isn't a free pass."""
    blocks = [_outline_assessment_block(source_ids=(), source_references=())]
    chunks_lookup = {
        "dart:misc#blk_0": _ungrounded_chunk_text(),
        "dart:misc#blk_1": _ungrounded_chunk_text(),
    }
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "NO_SOURCE_ATTRIBUTION" in codes
    assert "ANSWER_NOT_GROUNDED" in codes
    assert result.passed is False


# --------------------------------------------------------------------- #
# ESSAY_SKIPPED (info)
# --------------------------------------------------------------------- #


def test_essay_question_skipped_info():
    """Essay-mode assessment_item → ESSAY_SKIPPED info; not failed."""
    blocks = [_outline_assessment_block(
        question_type="essay",
        answer_key=None,
        drop_answer_key=True,
        distractors=[],
    )]
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": {},
    })

    codes = _codes(result)
    assert "ESSAY_SKIPPED" in codes
    info = [i for i in result.issues if i.code == "ESSAY_SKIPPED"]
    assert all(i.severity == "info" for i in info)
    # Essay skip is not a failure.
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# No-op path — non-assessment_item blocks are skipped silently
# --------------------------------------------------------------------- #


def test_non_assessment_blocks_silently_skipped():
    """Non-assessment_item blocks → no audit, no issues, passed."""
    blocks = [_non_assessment_block(), _non_assessment_block()]
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": {"dart:rdf_intro#blk_0": _grounded_chunk_text()},
    })

    assert result.passed is True
    assert result.score == 1.0
    assert _codes(result) == []
    assert result.action is None


def test_empty_blocks_input_passes():
    """Empty blocks list → trivial pass, score 1.0."""
    validator = AssessmentRetrievalGroundingValidator()
    result = validator.validate({"blocks": [], "chunks_lookup": {}})
    assert result.passed is True
    assert result.score == 1.0


def test_missing_blocks_input_fails():
    """Absent ``blocks`` key → MISSING_BLOCKS_INPUT critical."""
    validator = AssessmentRetrievalGroundingValidator()
    result = validator.validate({"chunks_lookup": {}})
    assert result.passed is False
    assert "MISSING_BLOCKS_INPUT" in _codes(result)


def test_invalid_blocks_input_type_fails():
    """Non-list ``blocks`` value → INVALID_BLOCKS_INPUT critical."""
    validator = AssessmentRetrievalGroundingValidator()
    result = validator.validate({"blocks": "not-a-list", "chunks_lookup": {}})
    assert result.passed is False
    assert "INVALID_BLOCKS_INPUT" in _codes(result)


# --------------------------------------------------------------------- #
# Decision-capture wiring
# --------------------------------------------------------------------- #


def test_capture_emit_per_validate_call():
    """``decision_capture`` is invoked exactly once per validate() with
    decision_type='assessment_retrieval_grounding_check' and ml_features
    carrying per-block overlap scores + threshold + verdict."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _grounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    assert result.passed is True
    assert len(capture.calls) == 1, (
        f"expected one capture call per validate(), got {len(capture.calls)}"
    )
    call = capture.calls[0]
    assert call["decision_type"] == "assessment_retrieval_grounding_check"
    assert call["decision"] == "passed"
    assert "rationale" in call and len(call["rationale"]) >= 20
    features = call["ml_features"]
    assert features["min_overlap_jaccard"] == DEFAULT_MIN_OVERLAP_JACCARD
    assert features["audited_count"] == 1
    assert features["grounded_count"] == 1
    assert features["failed_count"] == 0
    per_block = features["per_block"]
    assert len(per_block) == 1
    assert per_block[0]["block_id"] == "page_01#assessment_item_q_0"
    assert per_block[0]["verdict"] == "grounded"
    assert per_block[0]["overlap"] is not None
    assert per_block[0]["overlap"] >= DEFAULT_MIN_OVERLAP_JACCARD


def test_capture_records_failure_verdict_for_ungrounded_block():
    """Capture verdict reflects the ungrounded outcome with overlap
    score in ml_features."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"dart:rdf_intro#blk_0": _ungrounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    assert result.passed is False
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision"] == "failed"
    features = call["ml_features"]
    assert features["audited_count"] == 1
    assert features["failed_count"] == 1
    assert features["grounded_count"] == 0
    per_block = features["per_block"]
    assert per_block[0]["verdict"] == "not_grounded"
    assert per_block[0]["overlap"] < DEFAULT_MIN_OVERLAP_JACCARD


# --------------------------------------------------------------------- #
# Threshold sanity
# --------------------------------------------------------------------- #


def test_default_min_overlap_is_thirty_percent():
    """Plan W2 § config default — sanity-pin the floor."""
    assert DEFAULT_MIN_OVERLAP_JACCARD == pytest.approx(0.30)
