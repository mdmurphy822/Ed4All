"""Worker W7 — BlockAssessmentItemPayloadValidator unit tests.

Pins the validator's two-mode dispatch + four GateIssue codes on
representative outline-tier (dict content) and rewrite-tier (HTML
string content) Block fixtures. Mirrors the test pattern used by
``Courseforge/router/tests/test_inter_tier_gates.py`` for the four
sibling Block validators.

Failure modes:

- ``ASSESSMENT_ITEM_MISSING_DISTRACTORS`` — no ``distractors`` field
  (outline) / fewer than 2 ``<li data-cf-distractor-index>`` siblings
  (rewrite); fewer than 2 entries on either path.
- ``ASSESSMENT_ITEM_INVALID_MISCONCEPTION_REF`` — a distractor's
  ``misconception_ref`` doesn't match
  ``^[A-Z]{2,}-\\d{2,}#m\\d+$`` (outline-only).
- ``ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE`` —
  ``correct_answer_index`` < 0 or >= len(distractors) on the dict
  path; non-contiguous ``data-cf-distractor-index`` values on the
  rewrite path.
- ``ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING`` — a distractor entry has
  no ``text`` field, an empty string, or (on the rewrite path) a
  ``<li data-cf-distractor-index>`` element with no body text after
  HTML strip.

Failure ``action`` is always ``"regenerate"`` (rewrite-tier re-roll
fixes any payload-shape miss without external manifest dependencies).
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

from lib.validators.assessment_item_payload import (  # noqa: E402
    BlockAssessmentItemPayloadValidator,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _outline_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    page_id: str = "page_01",
    sequence: int = 0,
    stem: str = "What is the primary purpose of an RDF triple?",
    answer_key: str = "To express a subject-predicate-object statement.",
    distractors: Optional[List[Dict[str, Any]]] = None,
    correct_answer_index: int = 0,
    drop_distractors: bool = False,
    drop_correct_index: bool = False,
) -> Block:
    """Outline-tier (dict-content) assessment_item Block fixture.

    Required dict keys: stem, answer_key, distractors[],
    correct_answer_index. The validator filters on
    block_type == "assessment_item" so other block_type values are
    silently skipped.
    """
    if distractors is None:
        distractors = [
            {"text": "To express a subject-predicate-object statement."},
            {"text": "To define an XML namespace.", "misconception_ref": "TO-01#m1"},
            {"text": "To anchor a Turtle prefix declaration."},
            {"text": "To declare a SHACL node shape."},
        ]
    content: Dict[str, Any] = {
        "curies": ["rdf:Statement"],
        "key_claims": [
            "An RDF triple is a subject-predicate-object statement.",
        ],
        "content_type": "assessment_item",
        "stem": stem,
        "answer_key": answer_key,
    }
    if not drop_distractors:
        content["distractors"] = distractors
    if not drop_correct_index:
        content["correct_answer_index"] = correct_answer_index

    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content=content,
        objective_ids=("TO-01",),
    )


def _rewrite_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    page_id: str = "page_01",
    sequence: int = 0,
    html: Optional[str] = None,
) -> Block:
    """Rewrite-tier (HTML-string) assessment_item Block fixture."""
    if html is None:
        html = (
            "<section data-cf-block-id=\"page_01#assessment_item_q_0\">\n"
            "  <p>What is the primary purpose of an RDF triple?</p>\n"
            "  <ol>\n"
            "    <li data-cf-distractor-index=\"0\">A subject-predicate-object statement.</li>\n"
            "    <li data-cf-distractor-index=\"1\">An XML namespace declaration.</li>\n"
            "    <li data-cf-distractor-index=\"2\">A Turtle prefix declaration.</li>\n"
            "    <li data-cf-distractor-index=\"3\">A SHACL node shape.</li>\n"
            "  </ol>\n"
            "</section>"
        )
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content=html,
        objective_ids=("TO-01",),
    )


def _non_assessment_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
) -> Block:
    """Non-assessment_item Block fixture for the no-op test path."""
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content={
            "curies": ["ed4all:Concept"],
            "key_claims": ["A concept block introduces a new term."],
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


def _codes(result) -> List[str]:
    return [issue.code for issue in result.issues]


# --------------------------------------------------------------------- #
# Outline-mode tests
# --------------------------------------------------------------------- #


def test_outline_happy_path_passes():
    """Outline-mode block with 4 valid distractors + valid index +
    valid misconception_ref strings → passed=True, action=None."""
    blocks = [_outline_assessment_block()]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None
    assert result.score == 1.0


def test_outline_missing_distractors_fires_regenerate():
    """No distractors field → ASSESSMENT_ITEM_MISSING_DISTRACTORS."""
    blocks = [_outline_assessment_block(drop_distractors=True)]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_MISSING_DISTRACTORS" in _codes(result)


def test_outline_single_distractor_fires_regenerate():
    """distractors length < 2 → ASSESSMENT_ITEM_MISSING_DISTRACTORS."""
    blocks = [
        _outline_assessment_block(
            distractors=[{"text": "Only one option."}],
            correct_answer_index=0,
        ),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_MISSING_DISTRACTORS" in _codes(result)


def test_outline_invalid_misconception_ref_fires_regenerate():
    """Distractor with misconception_ref='bad-pattern' →
    ASSESSMENT_ITEM_INVALID_MISCONCEPTION_REF."""
    blocks = [
        _outline_assessment_block(
            distractors=[
                {"text": "Correct."},
                {"text": "Wrong A.", "misconception_ref": "bad-pattern"},
                {"text": "Wrong B."},
            ],
            correct_answer_index=0,
        ),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_INVALID_MISCONCEPTION_REF" in _codes(result)


def test_outline_correct_index_out_of_range_fires_regenerate():
    """correct_answer_index=5 with 3 distractors →
    ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE."""
    blocks = [
        _outline_assessment_block(
            distractors=[
                {"text": "A"},
                {"text": "B"},
                {"text": "C"},
            ],
            correct_answer_index=5,
        ),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE" in _codes(result)


def test_outline_distractor_missing_text_fires_regenerate():
    """A distractor with empty text → ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING."""
    blocks = [
        _outline_assessment_block(
            distractors=[
                {"text": "Correct."},
                {"text": ""},  # empty text
                {"text": "Wrong B."},
            ],
            correct_answer_index=0,
        ),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING" in _codes(result)


def test_outline_correct_index_missing_fires_regenerate():
    """No correct_answer_index field at all →
    ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE (treated as non-int)."""
    blocks = [_outline_assessment_block(drop_correct_index=True)]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE" in _codes(result)


# --------------------------------------------------------------------- #
# Rewrite-mode tests
# --------------------------------------------------------------------- #


def test_rewrite_happy_path_passes():
    """Rewrite-mode HTML with 4 contiguous <li data-cf-distractor-index>
    siblings → passed=True."""
    blocks = [_rewrite_assessment_block()]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None


def test_rewrite_single_li_fires_missing_distractors():
    """Rewrite-mode HTML with only one <li data-cf-distractor-index>
    sibling → ASSESSMENT_ITEM_MISSING_DISTRACTORS."""
    html = (
        "<section>\n"
        "  <p>Stem.</p>\n"
        "  <ol>\n"
        "    <li data-cf-distractor-index=\"0\">Only one.</li>\n"
        "  </ol>\n"
        "</section>"
    )
    blocks = [_rewrite_assessment_block(html=html)]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_MISSING_DISTRACTORS" in _codes(result)


def test_rewrite_non_contiguous_indices_fires_out_of_range():
    """Rewrite-mode HTML with 3 <li> siblings indexed 0/1/5 (non-
    contiguous) → ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE."""
    html = (
        "<ol>\n"
        "  <li data-cf-distractor-index=\"0\">A.</li>\n"
        "  <li data-cf-distractor-index=\"1\">B.</li>\n"
        "  <li data-cf-distractor-index=\"5\">C.</li>\n"
        "</ol>"
    )
    blocks = [_rewrite_assessment_block(html=html)]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE" in _codes(result)


def test_rewrite_empty_li_body_fires_text_missing():
    """Rewrite-mode <li> with no body text → ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING."""
    html = (
        "<ol>\n"
        "  <li data-cf-distractor-index=\"0\">Real text.</li>\n"
        "  <li data-cf-distractor-index=\"1\">  </li>\n"
        "</ol>"
    )
    blocks = [_rewrite_assessment_block(html=html)]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING" in _codes(result)


# --------------------------------------------------------------------- #
# No-op + input-shape tests
# --------------------------------------------------------------------- #


def test_non_assessment_item_blocks_pass_no_op():
    """Validator filters on block_type == 'assessment_item'; non-
    assessment blocks are silently skipped (passed=True, no issues)."""
    blocks = [
        _non_assessment_block(block_id="page_01#concept_a_0", block_type="concept"),
        _non_assessment_block(
            block_id="page_01#example_a_0",
            block_type="example",
        ),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_mixed_batch_audits_only_assessment_items():
    """A batch with one passing assessment_item + one bad concept Block
    (which the validator ignores) → passed=True; the concept block
    isn't audited."""
    blocks = [
        _outline_assessment_block(),
        _non_assessment_block(),
    ]
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True


def test_missing_blocks_input_fires_structured_skip():
    """No 'blocks' key in inputs → MISSING_BLOCKS_INPUT issue,
    action='regenerate'."""
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "MISSING_BLOCKS_INPUT" in _codes(result)


def test_invalid_blocks_input_fires_structured_skip():
    """blocks input not a list → INVALID_BLOCKS_INPUT issue."""
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": "not-a-list"})

    assert result.passed is False
    assert result.action == "regenerate"
    assert "INVALID_BLOCKS_INPUT" in _codes(result)


# --------------------------------------------------------------------- #
# Empty-batch test (matches inter_tier_gates score semantics)
# --------------------------------------------------------------------- #


def test_empty_blocks_list_passes():
    """No blocks to audit → passed=True, score=1.0 (mirrors the four
    sibling Block*Validator's empty-batch contract)."""
    validator = BlockAssessmentItemPayloadValidator()

    result = validator.validate({"blocks": []})

    assert result.passed is True
    assert result.action is None
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# H3 Wave W5 — decision-capture wiring
# --------------------------------------------------------------------- #


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


def test_capture_emits_one_decision_per_assessment_item_block() -> None:
    """Per-block cardinality: 2 assessment_item blocks → 2 emits;
    non-assessment_item blocks are silently skipped (no emit)."""
    capture = _StubCapture()
    BlockAssessmentItemPayloadValidator().validate({
        "blocks": [
            _outline_assessment_block(block_id="page_01#assessment_item_q_0"),
            _rewrite_assessment_block(block_id="page_01#assessment_item_q_1"),
            _non_assessment_block(),
        ],
        "decision_capture": capture,
    })
    assert len(capture.calls) == 2
    types = {c["decision_type"] for c in capture.calls}
    assert types == {"block_assessment_item_payload_check"}


def test_capture_rationale_carries_dynamic_signals() -> None:
    capture = _StubCapture()
    BlockAssessmentItemPayloadValidator().validate({
        "blocks": [_outline_assessment_block(block_id="page_01#assessment_item_q_0")],
        "decision_capture": capture,
    })
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "page_01#assessment_item_q_0" in rationale
    assert "distractors_count=" in rationale
    assert "correct_answer_index_valid=" in rationale
    assert "misconception_refs_resolved=" in rationale
    assert "mode=outline" in rationale


def test_capture_rewrite_mode_signals() -> None:
    capture = _StubCapture()
    BlockAssessmentItemPayloadValidator().validate({
        "blocks": [_rewrite_assessment_block(block_id="page_01#assessment_item_q_0")],
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert "mode=rewrite" in capture.calls[0]["rationale"]


def test_capture_failure_block_emits_failure_decision() -> None:
    """A block missing distractors emits a failed:* decision."""
    capture = _StubCapture()
    BlockAssessmentItemPayloadValidator().validate({
        "blocks": [_outline_assessment_block(drop_distractors=True)],
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert "failed:" in capture.calls[0]["decision"]
    assert "ASSESSMENT_ITEM_MISSING_DISTRACTORS" in capture.calls[0]["decision"]


def test_capture_no_capture_no_emit_no_crash() -> None:
    """Absent decision_capture → identical GateResult, no exception."""
    validator = BlockAssessmentItemPayloadValidator()
    blocks = [_outline_assessment_block(block_id="page_01#assessment_item_q_0")]
    base = validator.validate({"blocks": blocks})
    captured = validator.validate({
        "blocks": blocks,
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
    assert [i.code for i in base.issues] == [i.code for i in captured.issues]
