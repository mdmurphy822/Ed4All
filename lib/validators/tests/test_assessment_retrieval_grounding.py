"""Worker W2 — AssessmentRetrievalGroundingValidator unit tests.

Pins the validator's per-block dispatch + GateIssue codes on
representative outline-tier (dict content) and rewrite-tier (HTML
string content) Block fixtures + non-assessment_item no-op path +
decision-capture wiring assertion.

Wave 3 W3.A extends the suite with five new sub-check codes
(``STEM_NOT_SELF_CONTAINED``, ``ANSWER_LEAKED_IN_STEM``,
``FEEDBACK_NOT_GROUNDED``, ``ANSWER_SPAN_MISSING``, plus the
calibration-gated promotion of ``NO_SOURCE_ATTRIBUTION``) and the
parametrized calibration-flip threshold test.

Failure modes:

- ``ANSWER_NOT_GROUNDED`` (critical) — Jaccard overlap below threshold.
- ``ANSWER_TEXT_MISSING`` (critical) — assessment_item with no
  recoverable correct-answer text on either content shape.
- ``NO_SOURCE_ATTRIBUTION`` (warning, calibration-gated → critical) —
  block declares no source IDs; W3.A drops the legacy
  all-chunk-union fallback.
- ``STEM_NOT_SELF_CONTAINED`` (warning, W3.A) — dangling reference or
  unbound pronoun in the stem.
- ``ANSWER_LEAKED_IN_STEM`` (warning, W3.A) — stem-answer Jaccard
  above the ceiling.
- ``FEEDBACK_NOT_GROUNDED`` (warning, W3.A) — feedback prose
  overlap with chunks below the floor.
- ``ANSWER_SPAN_MISSING`` (warning, W3.A) — no explicit answer-span
  pointer.
- ``ESSAY_SKIPPED`` (info) — essay-mode block; skipped from gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Wave W-D10 T10.2: validator wraps the [embedding] sentence-transformers
# extra; tests use Jaccard fallback today but the file is slow-marked so
# the nightly extras-installed run picks them up via -m "slow".
pytestmark = pytest.mark.slow

# Block lives at Courseforge/scripts/blocks.py — mirror the import
# bridge the validator uses internally.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.assessment_retrieval_grounding import (  # noqa: E402
    AssessmentRetrievalGroundingValidator,
    CALIBRATION_FLIP_THRESHOLD,
    CALIBRATION_SIGNAL_NAME,
    DEFAULT_MAX_STEM_ANSWER_JACCARD,
    DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD,
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
    source_ids: tuple = ("semantik:rdf_intro#blk_0",),
    source_references: tuple = (),
    stem: str = "What is the primary purpose of an RDF triple?",
    feedback: Optional[str] = None,
    answer_span: Optional[Dict[str, Any]] = None,
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
        "stem": stem,
        "distractors": distractors,
        "correct_answer_index": correct_answer_index,
    }
    if not drop_answer_key and answer_key is not None:
        content["answer_key"] = answer_key
    if correct_answer is not None:
        content["correct_answer"] = correct_answer
    if question_type is not None:
        content["question_type"] = question_type
    if feedback is not None:
        content["feedback"] = feedback
    if answer_span is not None:
        content["answer_span"] = answer_span

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
    source_ids: tuple = ("semantik:rdf_intro#blk_0",),
    stem_html: str = "<p>What is the primary purpose of an RDF triple?</p>",
    feedback_html: str = "",
    answer_span_attr: str = "",
) -> Block:
    span_extra = (
        f' data-cf-answer-span="{answer_span_attr}"'
        if answer_span_attr
        else ""
    )
    html = (
        "<section data-cf-block-id=\"" + block_id + "\"" + span_extra + ">\n"
        + stem_html + "\n"
        "  <ol data-cf-correct-answer-index=\"0\">\n"
        "    <li data-cf-distractor-index=\"0\" data-cf-correct=\"true\">"
        + correct_answer_text + "</li>\n"
        "    <li data-cf-distractor-index=\"1\">An XML namespace declaration.</li>\n"
        "    <li data-cf-distractor-index=\"2\">A Turtle prefix declaration.</li>\n"
        "    <li data-cf-distractor-index=\"3\">A SHACL node shape.</li>\n"
        "  </ol>\n"
        + feedback_html + "\n"
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
    with the referenced source chunk → passed=True (with warning-tier
    sub-check noise on ANSWER_SPAN_MISSING since the fixture omits it)."""
    blocks = [_outline_assessment_block(
        answer_span={
            "source_chunk_id": "semantik:rdf_intro#blk_0",
            "char_start": 0,
            "char_end": 50,
        },
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
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
    blocks = [_rewrite_assessment_block(
        answer_span_attr="semantik:rdf_intro#blk_0:0-50",
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
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
    chunks_lookup = {"semantik:rdf_intro#blk_0": _ungrounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ANSWER_NOT_GROUNDED" in _codes(result)
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
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
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
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

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
        distractors=[
            {"text": ""},
            {"text": "An XML namespace declaration."},
        ],
        correct_answer_index=0,
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
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
        source_ids=("semantik:rdf_intro#blk_0",),
    )
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator()

    result = validator.validate({
        "blocks": [block],
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    assert "ANSWER_TEXT_MISSING" in _codes(result)


# --------------------------------------------------------------------- #
# NO_SOURCE_ATTRIBUTION (W3.A: no fallback; calibration-gated severity)
# --------------------------------------------------------------------- #


def test_no_source_ids_emits_warning_no_fallback(monkeypatch, tmp_path):
    """Block declaring no source_ids / source_references → warning
    severity (default flip-deferred) AND skips the per-block overlap
    check; the per-block feature row records ``no_source_attribution``
    rather than ``grounded`` / ``not_grounded``.
    """
    blocks = [_outline_assessment_block(source_ids=(), source_references=())]
    chunks_lookup = {
        "semantik:rdf_intro#blk_0": _grounded_chunk_text(),
        "semantik:rdf_intro#blk_1": _grounded_chunk_text(),
    }
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "NO_SOURCE_ATTRIBUTION" in codes, codes
    warnings = [i for i in result.issues if i.code == "NO_SOURCE_ATTRIBUTION"]
    assert all(i.severity == "warning" for i in warnings)
    # The legacy fallback path is gone; the validator must emit no
    # ANSWER_NOT_GROUNDED for this block (the overlap check was skipped).
    assert "ANSWER_NOT_GROUNDED" not in codes
    # Decision-capture per-block payload records the no-source verdict.
    capture = _RecordingCapture()
    validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })
    grounding_calls = [
        c for c in capture.calls
        if c["decision_type"] == "assessment_retrieval_grounding_check"
    ]
    assert grounding_calls, "expected at least one grounding-check capture"
    per_block = grounding_calls[0]["ml_features"]["per_block"]
    assert per_block, "expected at least one per-block feature row"
    assert per_block[0]["verdict"] == "no_source_attribution"
    assert per_block[0].get("fallback_used") is True


def test_no_source_promoted_to_critical_when_flip_applied(tmp_path):
    """When the calibration signal clears the floor the
    NO_SOURCE_ATTRIBUTION severity flips warning -> critical AND the
    block counts as a failure for the gate result."""
    _write_calibration_report(tmp_path, alignment_rate=0.92)
    blocks = [_outline_assessment_block(source_ids=(), source_references=())]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    assert validator.severity_flip_applied is True

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert result.passed is False
    no_source = [
        i for i in result.issues if i.code == "NO_SOURCE_ATTRIBUTION"
    ]
    assert no_source, _codes(result)
    assert all(i.severity == "critical" for i in no_source)


# --------------------------------------------------------------------- #
# W3.A — STEM_NOT_SELF_CONTAINED
# --------------------------------------------------------------------- #


def test_stem_not_self_contained_dangling_reference(tmp_path):
    """Stem with a dangling reference like 'see above' → warning."""
    blocks = [_outline_assessment_block(
        stem="See above and answer the following: what is RDF?",
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "STEM_NOT_SELF_CONTAINED" in codes, codes
    issues = [i for i in result.issues if i.code == "STEM_NOT_SELF_CONTAINED"]
    assert all(i.severity == "warning" for i in issues)


def test_stem_not_self_contained_unbound_pronoun(tmp_path):
    """Stem opening with an unbound 'This' pronoun → warning."""
    blocks = [_outline_assessment_block(
        stem="This is the primary purpose of an RDF triple, isn't it?",
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "STEM_NOT_SELF_CONTAINED" in _codes(result)


def test_stem_self_contained_does_not_fire(tmp_path):
    """A clean self-contained stem MUST NOT fire STEM_NOT_SELF_CONTAINED."""
    blocks = [_outline_assessment_block(
        stem="What is the primary purpose of an RDF triple?",
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "STEM_NOT_SELF_CONTAINED" not in _codes(result)


# --------------------------------------------------------------------- #
# W3.A — ANSWER_LEAKED_IN_STEM
# --------------------------------------------------------------------- #


def test_answer_leaked_in_stem_fires_warning(tmp_path):
    """Stem and answer share > 0.50 Jaccard token overlap → warning."""
    blocks = [_outline_assessment_block(
        stem=(
            "An RDF triple is a subject predicate object statement that "
            "expresses a fact: what is the primary purpose of this triple?"
        ),
        answer_key=(
            "An RDF triple is a subject predicate object statement "
            "expressing a fact."
        ),
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "ANSWER_LEAKED_IN_STEM" in codes, codes
    issues = [i for i in result.issues if i.code == "ANSWER_LEAKED_IN_STEM"]
    assert all(i.severity == "warning" for i in issues)


def test_clean_stem_does_not_fire_answer_leak(tmp_path):
    """Disjoint stem and answer tokens MUST NOT fire ANSWER_LEAKED_IN_STEM."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "ANSWER_LEAKED_IN_STEM" not in _codes(result)


# --------------------------------------------------------------------- #
# W3.A — FEEDBACK_NOT_GROUNDED
# --------------------------------------------------------------------- #


def test_feedback_not_grounded_fires_warning(tmp_path):
    """Feedback prose unrelated to the cited chunk → warning."""
    blocks = [_outline_assessment_block(
        feedback=(
            "Quantum chromodynamics describes strong nuclear interactions "
            "via gluon exchanges between quark colour charges in collider "
            "experiments and lattice gauge simulations."
        ),
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "FEEDBACK_NOT_GROUNDED" in codes, codes
    issues = [i for i in result.issues if i.code == "FEEDBACK_NOT_GROUNDED"]
    assert all(i.severity == "warning" for i in issues)


def test_grounded_feedback_does_not_fire(tmp_path):
    """Feedback prose anchored in the cited chunk MUST NOT fire."""
    blocks = [_outline_assessment_block(
        feedback=(
            "An RDF triple represents a subject predicate object statement "
            "in the resource description framework graph."
        ),
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "FEEDBACK_NOT_GROUNDED" not in _codes(result)


def test_no_feedback_surface_does_not_fire(tmp_path):
    """A block carrying no feedback at all does not fire FEEDBACK_NOT_GROUNDED."""
    blocks = [_outline_assessment_block()]  # no feedback kwarg
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "FEEDBACK_NOT_GROUNDED" not in _codes(result)


# --------------------------------------------------------------------- #
# W3.A — ANSWER_SPAN_MISSING
# --------------------------------------------------------------------- #


def test_answer_span_missing_outline_fires_warning(tmp_path):
    """Outline block without content['answer_span'] → warning."""
    blocks = [_outline_assessment_block()]  # no answer_span kwarg
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    codes = _codes(result)
    assert "ANSWER_SPAN_MISSING" in codes
    issues = [i for i in result.issues if i.code == "ANSWER_SPAN_MISSING"]
    assert all(i.severity == "warning" for i in issues)


def test_answer_span_present_outline_does_not_fire(tmp_path):
    """A well-formed outline answer_span suppresses ANSWER_SPAN_MISSING."""
    blocks = [_outline_assessment_block(
        answer_span={
            "source_chunk_id": "semantik:rdf_intro#blk_0",
            "char_start": 10,
            "char_end": 60,
        },
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "ANSWER_SPAN_MISSING" not in _codes(result)


def test_answer_span_present_rewrite_does_not_fire(tmp_path):
    """A rewrite block carrying data-cf-answer-span suppresses the warning."""
    blocks = [_rewrite_assessment_block(
        answer_span_attr="semantik:rdf_intro#blk_0:0-50",
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    assert "ANSWER_SPAN_MISSING" not in _codes(result)


def test_answer_span_stays_warning_after_flip(tmp_path):
    """Even when the calibration flip applies, ANSWER_SPAN_MISSING stays
    at warning severity (it's hygiene, not correctness)."""
    _write_calibration_report(tmp_path, alignment_rate=0.95)
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    assert validator.severity_flip_applied is True

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
    })

    span_issues = [
        i for i in result.issues if i.code == "ANSWER_SPAN_MISSING"
    ]
    assert span_issues, _codes(result)
    assert all(i.severity == "warning" for i in span_issues)


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
        "chunks_lookup": {"semantik:rdf_intro#blk_0": _grounded_chunk_text()},
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


def test_capture_emit_per_validate_call(tmp_path):
    """``decision_capture`` is invoked with both the per-validate
    grounding-check event AND the calibration-flip event (once per
    capture handle)."""
    blocks = [_outline_assessment_block(
        answer_span={
            "source_chunk_id": "semantik:rdf_intro#blk_0",
            "char_start": 0,
            "char_end": 50,
        },
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    assert result.passed is True
    grounding_calls = [
        c for c in capture.calls
        if c["decision_type"] == "assessment_retrieval_grounding_check"
    ]
    flip_calls = [
        c for c in capture.calls
        if c["decision_type"]
        in {"severity_flip_applied", "severity_flip_deferred"}
    ]
    assert len(grounding_calls) == 1
    assert len(flip_calls) == 1
    call = grounding_calls[0]
    assert call["decision"] == "passed"
    assert "rationale" in call and len(call["rationale"]) >= 20
    features = call["ml_features"]
    assert features["min_overlap_jaccard"] == DEFAULT_MIN_OVERLAP_JACCARD
    assert features["audited_count"] == 1
    assert features["grounded_count"] == 1
    assert features["failed_count"] == 0
    assert "sub_check_counts" in features
    assert features["sub_check_counts"]["stem_not_self_contained"] == 0
    assert features["sub_check_counts"]["answer_leaked_in_stem"] == 0
    assert features["sub_check_counts"]["feedback_not_grounded"] == 0
    assert features["sub_check_counts"]["answer_span_missing"] == 0
    assert features["sub_check_counts"]["no_source_attribution"] == 0
    assert features["severity_flip_applied"] is False
    per_block = features["per_block"]
    assert len(per_block) == 1
    assert per_block[0]["block_id"] == "page_01#assessment_item_q_0"
    assert per_block[0]["verdict"] == "grounded"
    assert per_block[0]["overlap"] is not None
    assert per_block[0]["overlap"] >= DEFAULT_MIN_OVERLAP_JACCARD


def test_capture_records_failure_verdict_for_ungrounded_block(tmp_path):
    """Capture verdict reflects the ungrounded outcome."""
    blocks = [_outline_assessment_block()]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _ungrounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    result = validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    assert result.passed is False
    grounding_calls = [
        c for c in capture.calls
        if c["decision_type"] == "assessment_retrieval_grounding_check"
    ]
    assert len(grounding_calls) == 1
    call = grounding_calls[0]
    assert call["decision"] == "failed"
    features = call["ml_features"]
    assert features["audited_count"] == 1
    assert features["failed_count"] == 1
    assert features["grounded_count"] == 0
    per_block = features["per_block"]
    assert per_block[0]["verdict"] == "not_grounded"
    assert per_block[0]["overlap"] < DEFAULT_MIN_OVERLAP_JACCARD


def test_capture_records_per_sub_check_signals(tmp_path):
    """Each sub-check increments its dedicated counter in
    ml_features.sub_check_counts so an audit can grep the rationale."""
    blocks = [_outline_assessment_block(
        stem="See above and tell me what RDF is.",  # STEM_NOT_SELF_CONTAINED
        answer_key=(
            "An RDF triple is a subject predicate object statement "
            "expressing a fact in the resource description framework."
        ),
        feedback=(
            "Quantum chromodynamics describes strong nuclear interactions "
            "via gluon exchange between quarks in collider experiments."
        ),  # FEEDBACK_NOT_GROUNDED
        # No answer_span → ANSWER_SPAN_MISSING
    )]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    grounding_calls = [
        c for c in capture.calls
        if c["decision_type"] == "assessment_retrieval_grounding_check"
    ]
    assert grounding_calls
    counts = grounding_calls[0]["ml_features"]["sub_check_counts"]
    assert counts["stem_not_self_contained"] == 1
    assert counts["feedback_not_grounded"] == 1
    assert counts["answer_span_missing"] == 1
    rationale = grounding_calls[0]["rationale"]
    assert "sub_checks" in rationale


# --------------------------------------------------------------------- #
# Threshold sanity
# --------------------------------------------------------------------- #


def test_default_min_overlap_is_thirty_percent():
    """Plan W2 § config default — sanity-pin the floor."""
    assert DEFAULT_MIN_OVERLAP_JACCARD == pytest.approx(0.30)


def test_default_max_stem_answer_is_fifty_percent():
    """Plan W3.A § ANSWER_LEAKED_IN_STEM sanity-pin."""
    assert DEFAULT_MAX_STEM_ANSWER_JACCARD == pytest.approx(0.50)


def test_default_min_feedback_overlap_is_thirty_percent():
    """Plan W3.A § FEEDBACK_NOT_GROUNDED sanity-pin."""
    assert DEFAULT_MIN_FEEDBACK_OVERLAP_JACCARD == pytest.approx(0.30)


def test_calibration_flip_threshold_is_eighty_five():
    """Plan W3.A § calibration-gated severity flip sanity-pin."""
    assert CALIBRATION_FLIP_THRESHOLD == pytest.approx(0.85)
    assert CALIBRATION_SIGNAL_NAME == "answerability_alignment_rate"


# --------------------------------------------------------------------- #
# Calibration-gated severity flip
# --------------------------------------------------------------------- #


def _write_calibration_report(
    libv2_root: Path,
    *,
    alignment_rate: float,
    course_slug: str = "calibration_textbook",
) -> Path:
    """Write a minimal calibration report fixture under <libv2_root>."""
    report_dir = (
        libv2_root / "courses" / course_slug / "quality"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "trainforge_assessment_quality_report.json"
    report_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "course_code": course_slug,
            "summary": {
                "answerability_alignment_rate": alignment_rate,
            },
        }),
        encoding="utf-8",
    )
    return report_path


@pytest.mark.parametrize(
    "alignment_rate,expected_flip",
    [
        (0.85, True),   # at threshold → flip applies
        (0.84, False),  # just below → flip holds
        (0.95, True),   # well above → flip applies
        (0.10, False),  # well below → flip holds
    ],
)
def test_calibration_flip_threshold_parametrized(
    tmp_path, alignment_rate, expected_flip
):
    """The flip fires iff observed signal >= 0.85 floor."""
    _write_calibration_report(tmp_path, alignment_rate=alignment_rate)
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )
    assert validator.severity_flip_applied is expected_flip


def test_calibration_report_missing_defers_flip(tmp_path):
    """Clean-checkout posture: no calibration report → flip deferred,
    severity stays at warning, ``severity_flip_deferred`` event fires
    with ``calibration_report_missing`` reason."""
    # tmp_path has no LibV2/courses/<slug>/quality/... payload.
    blocks = [_outline_assessment_block(source_ids=(), source_references=())]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _grounded_chunk_text()}
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    assert validator.severity_flip_applied is False

    validator.validate({
        "blocks": blocks,
        "chunks_lookup": chunks_lookup,
        "decision_capture": capture,
    })

    flip_events = [
        c for c in capture.calls
        if c["decision_type"]
        in {"severity_flip_applied", "severity_flip_deferred"}
    ]
    assert len(flip_events) == 1
    event = flip_events[0]
    assert event["decision_type"] == "severity_flip_deferred"
    assert event["decision"] == "warning"
    assert (
        event["ml_features"]["reason"] == "calibration_report_missing"
    )
    assert event["ml_features"]["observed_rate"] is None


def test_calibration_signal_missing_defers_flip(tmp_path):
    """Calibration report present but signal field absent → flip
    deferred with reason=calibration_signal_missing."""
    report_dir = tmp_path / "courses" / "calibration_textbook" / "quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "trainforge_assessment_quality_report.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "summary": {"placeholder_rate": 0.0},  # no answerability_*
        }),
        encoding="utf-8",
    )
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    assert validator.severity_flip_applied is False

    validator.validate({
        "blocks": [_outline_assessment_block()],
        "chunks_lookup": {"semantik:rdf_intro#blk_0": _grounded_chunk_text()},
        "decision_capture": capture,
    })

    flip_events = [
        c for c in capture.calls
        if c["decision_type"]
        in {"severity_flip_applied", "severity_flip_deferred"}
    ]
    assert len(flip_events) == 1
    assert flip_events[0]["decision_type"] == "severity_flip_deferred"
    assert (
        flip_events[0]["ml_features"]["reason"]
        == "calibration_signal_missing"
    )


def test_calibration_signal_below_floor_defers_flip(tmp_path):
    """When the signal is below 0.85 the flip event carries
    reason=alignment_rate_below_floor and the observed value."""
    _write_calibration_report(tmp_path, alignment_rate=0.40)
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    validator.validate({
        "blocks": [_outline_assessment_block()],
        "chunks_lookup": {"semantik:rdf_intro#blk_0": _grounded_chunk_text()},
        "decision_capture": capture,
    })

    flip_events = [
        c for c in capture.calls
        if c["decision_type"] == "severity_flip_deferred"
    ]
    assert flip_events
    payload = flip_events[0]["ml_features"]
    assert payload["reason"] == "alignment_rate_below_floor"
    assert payload["observed_rate"] == pytest.approx(0.40)


def test_calibration_signal_above_floor_emits_flip_applied(tmp_path):
    """At alignment_rate >= 0.85 the validator emits a
    ``severity_flip_applied`` event with decision='critical' and the
    observed value."""
    _write_calibration_report(tmp_path, alignment_rate=0.92)
    capture = _RecordingCapture()
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )

    assert validator.severity_flip_applied is True

    validator.validate({
        "blocks": [_outline_assessment_block()],
        "chunks_lookup": {"semantik:rdf_intro#blk_0": _grounded_chunk_text()},
        "decision_capture": capture,
    })

    flip_events = [
        c for c in capture.calls
        if c["decision_type"] == "severity_flip_applied"
    ]
    assert len(flip_events) == 1
    payload = flip_events[0]
    assert payload["decision"] == "critical"
    assert payload["ml_features"]["observed_rate"] == pytest.approx(0.92)
    assert payload["ml_features"]["threshold"] == pytest.approx(0.85)


def test_uses_default_calibration_value_when_threshold_match(tmp_path):
    """Sanity: at exactly 0.85 the flip fires (threshold is inclusive)."""
    _write_calibration_report(tmp_path, alignment_rate=0.85)
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )
    assert validator.severity_flip_applied is True


def test_just_below_threshold_holds_flip(tmp_path):
    """At 0.84 the flip is held."""
    _write_calibration_report(tmp_path, alignment_rate=0.84)
    validator = AssessmentRetrievalGroundingValidator(
        calibration_libv2_root=tmp_path,
    )
    assert validator.severity_flip_applied is False
