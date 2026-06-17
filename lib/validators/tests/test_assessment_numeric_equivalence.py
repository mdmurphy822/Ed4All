"""AssessmentNumericEquivalenceValidator unit tests.

Pins the validator's single GateIssue code
(``DISTRACTOR_EQUALS_ANSWER_VALUE``, warning, ``action="regenerate"``)
against representative outline-tier (dict content) and rewrite-tier
(``<li data-cf-distractor-index="N">`` HTML) assessment_item fixtures.

The blind spot under test (investigation a6d6291b): token-Jaccard
distinctness tokenises ``"4/6"`` to ``{"4", "6"}`` and ``"2/3"`` to
``{"2", "3"}`` → Jaccard 0.0 → "distinct". But ``4/6`` EQUALS ``2/3``
in VALUE, so the distractor is a SECOND CORRECT ANSWER. This validator
parses each option to a ``fractions.Fraction`` and fails any distractor
equal in value to the key.

Scope-limit: non-numeric text options skip (pass) — never false-
positive on a non-math MCQ. Empty input → pass. Decision capture fires
exactly one ``assessment_numeric_equivalence_check`` per AUDITED block.
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

from lib.validators.assessment_numeric_equivalence import (  # noqa: E402
    AssessmentNumericEquivalenceValidator,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _outline_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    answer_key: str,
    distractor_texts: List[str],
    stem: str = "Simplify 20/30 to lowest terms.",
) -> Block:
    """Outline-tier (dict-content) assessment_item Block fixture.

    The correct answer is ``answer_key``; the wrong options are the
    per-distractor ``text`` fields.
    """
    content: Dict[str, Any] = {
        "curies": ["math:Fraction"],
        "key_claims": ["20/30 reduces to 2/3 by dividing out the GCF 10."],
        "content_type": "assessment_item",
        "stem": stem,
        "answer_key": answer_key,
        "distractors": [{"text": t} for t in distractor_texts],
        "correct_answer_index": 0,
    }
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=("TO-01",),
    )


def _rewrite_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    correct_text: str,
    distractor_texts: List[str],
) -> Block:
    """Rewrite-tier (str-content) assessment_item Block fixture.

    Builds the canonical ``<li data-cf-distractor-index="N">`` markup;
    the correct option carries ``data-cf-correct="true"``. The correct
    answer is rendered as index 0, distractors follow.
    """
    lis: List[str] = [
        f'<li data-cf-distractor-index="0" data-cf-correct="true">'
        f"{correct_text}</li>"
    ]
    for i, dtext in enumerate(distractor_texts, start=1):
        lis.append(
            f'<li data-cf-distractor-index="{i}">{dtext}</li>'
        )
    html = (
        "<section data-cf-source-ids=\"dart:math#q1\">"
        "<p>Simplify 20/30 to lowest terms.</p>"
        "<ol>" + "".join(lis) + "</ol>"
        "</section>"
    )
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=html,
        objective_ids=("TO-01",),
    )


def _non_assessment_block(
    *, block_id: str = "page_01#concept_intro_0", block_type: str = "concept"
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content={
            "curies": ["math:Concept"],
            "key_claims": ["A concept block introduces a new term."],
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


def _codes(result) -> List[str]:
    return [issue.code for issue in result.issues]


class _RecordingCapture:
    """Minimal stand-in for DecisionCapture that records log_decision."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **extra: Any,
    ) -> None:
        event: Dict[str, Any] = {
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
        }
        event.update(extra)
        self.events.append(event)


# --------------------------------------------------------------------- #
# Core blind-spot: 2/3 with 4/6 distractor → FAIL
# --------------------------------------------------------------------- #


def test_fraction_equivalent_distractor_fails_outline():
    """correct=2/3 with distractor 4/6 → DISTRACTOR_EQUALS_ANSWER_VALUE,
    action=regenerate (the a6d6291b blind spot)."""
    block = _outline_block(answer_key="2/3", distractor_texts=["4/6", "5/7"])
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is False, f"expected fail, codes={_codes(result)}"
    assert result.action == "regenerate"
    assert "DISTRACTOR_EQUALS_ANSWER_VALUE" in _codes(result)
    # Exactly the 4/6 distractor fires (5/7 is genuinely distinct).
    assert _codes(result).count("DISTRACTOR_EQUALS_ANSWER_VALUE") == 1
    issue = next(
        i for i in result.issues
        if i.code == "DISTRACTOR_EQUALS_ANSWER_VALUE"
    )
    assert "4/6" in issue.message
    assert issue.severity == "warning"


def test_fraction_equivalent_distractor_fails_rewrite():
    """Same blind spot through the rewrite-tier <li data-cf-correct>
    HTML shape."""
    block = _rewrite_block(
        correct_text="2/3", distractor_texts=["4/6", "5/7"]
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is False, f"expected fail, codes={_codes(result)}"
    assert result.action == "regenerate"
    assert _codes(result).count("DISTRACTOR_EQUALS_ANSWER_VALUE") == 1


def test_equivalent_distractor_with_rationale_suffix_fails():
    """Canonical 'VALUE — rationale' option text: the leading value is
    parsed (the trailing rationale's numbers must not make it look like a
    multi-value expression). 4/6 still detected ≡ 2/3."""
    block = _rewrite_block(
        correct_text="2/3 — correct: the GCF of 20 and 30 is 10, divide by 10",
        distractor_texts=[
            "4/6 — divided by an incorrect common factor instead of the GCF",
            "5/7 — subtracted the GCF instead of dividing by it",
        ],
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )
    assert result.passed is False, f"expected fail, codes={_codes(result)}"
    assert result.action == "regenerate"
    assert _codes(result).count("DISTRACTOR_EQUALS_ANSWER_VALUE") == 1


# --------------------------------------------------------------------- #
# Genuinely-distinct distractors → PASS
# --------------------------------------------------------------------- #


def test_distinct_fraction_distractors_pass_outline():
    """correct=2/3 with distractors {2/5, 3/4} → passed=True."""
    block = _outline_block(
        answer_key="2/3", distractor_texts=["2/5", "3/4"]
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is True, f"unexpected: {_codes(result)}"
    assert result.action is None
    assert result.score == 1.0
    assert _codes(result) == []


def test_distinct_fraction_distractors_pass_rewrite():
    block = _rewrite_block(
        correct_text="2/3", distractor_texts=["2/5", "3/4"]
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is True, f"unexpected: {_codes(result)}"
    assert result.action is None


# --------------------------------------------------------------------- #
# Integer / decimal equivalence
# --------------------------------------------------------------------- #


def test_decimal_fraction_equivalence_fails():
    """0.5 (correct) vs 1/2 (distractor) → FAIL (cross-form equality)."""
    block = _outline_block(
        answer_key="0.5",
        distractor_texts=["1/2", "0.3"],
        stem="What is 1 divided by 2 as a decimal?",
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is False, f"expected fail, codes={_codes(result)}"
    assert _codes(result).count("DISTRACTOR_EQUALS_ANSWER_VALUE") == 1


def test_integer_fraction_equivalence_fails():
    """2 (correct) vs 4/2 (distractor) → FAIL."""
    block = _outline_block(
        answer_key="2",
        distractor_texts=["4/2", "3"],
        stem="Evaluate 6 divided by 3.",
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is False, f"expected fail, codes={_codes(result)}"
    assert _codes(result).count("DISTRACTOR_EQUALS_ANSWER_VALUE") == 1


# --------------------------------------------------------------------- #
# Graceful scope-limit: non-numeric text options → PASS
# --------------------------------------------------------------------- #


def test_non_numeric_text_options_graceful_pass():
    """Text MCQ (no parseable numeric value) → passed=True, never a
    false positive."""
    block = _outline_block(
        answer_key="subject, predicate, object",
        distractor_texts=[
            "A class hierarchy declaration.",
            "An XML namespace prefix mapping.",
        ],
        stem="Which is a valid RDF triple?",
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is True, f"unexpected: {_codes(result)}"
    assert result.action is None
    assert _codes(result) == []


def test_non_assessment_blocks_silently_skipped():
    blocks = [
        _non_assessment_block(block_id="page_01#concept_a"),
        _non_assessment_block(block_id="page_01#concept_b"),
    ]
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": blocks}
    )

    assert result.passed is True
    assert result.action is None
    assert _codes(result) == []


# --------------------------------------------------------------------- #
# Empty / missing input
# --------------------------------------------------------------------- #


def test_empty_blocks_passes():
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": []}
    )
    assert result.passed is True
    assert result.action is None
    assert _codes(result) == []


def test_missing_blocks_input_fails_closed():
    result = AssessmentNumericEquivalenceValidator().validate({})
    assert result.passed is False
    assert "MISSING_BLOCKS_INPUT" in _codes(result)


def test_rewrite_no_correct_marker_scope_skips():
    """Rewrite-tier HTML with no data-cf-correct marker can't resolve a
    correct value → scope-skip (pass); the W7 payload gate owns the
    missing-marker axis."""
    html = (
        "<section><ol>"
        '<li data-cf-distractor-index="0">2/3</li>'
        '<li data-cf-distractor-index="1">4/6</li>'
        "</ol></section>"
    )
    block = Block(
        block_id="page_01#assessment_item_q_0",
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=html,
        objective_ids=("TO-01",),
    )
    result = AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block]}
    )

    assert result.passed is True, f"unexpected: {_codes(result)}"
    assert _codes(result) == []


# --------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------- #


def test_capture_fires_once_per_audited_block_on_fail():
    """The numeric block (4/6 == 2/3) audits → exactly one
    assessment_numeric_equivalence_check event with decision='failed'
    and equivalent_found=True."""
    capture = _RecordingCapture()
    block = _outline_block(answer_key="2/3", distractor_texts=["4/6", "5/7"])
    AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [block], "decision_capture": capture}
    )

    events = [
        e for e in capture.events
        if e["decision_type"] == "assessment_numeric_equivalence_check"
    ]
    assert len(events) == 1
    assert events[0]["decision"] == "failed"
    assert events[0]["metrics"]["equivalent_found"] is True
    assert events[0]["metrics"]["n_options"] == 3
    assert events[0]["metrics"]["n_numeric_parsed"] == 3
    assert len(events[0]["rationale"]) >= 20


def test_capture_fires_on_pass_and_skips_non_numeric():
    """A clean numeric block emits one passed event; a non-numeric text
    block is scope-skipped (NOT audited → no event)."""
    capture = _RecordingCapture()
    numeric = _outline_block(
        answer_key="2/3", distractor_texts=["2/5", "3/4"]
    )
    text = _outline_block(
        block_id="page_01#assessment_item_q_1",
        answer_key="A valid RDF triple.",
        distractor_texts=["A class declaration.", "A namespace prefix."],
    )
    AssessmentNumericEquivalenceValidator().validate(
        {"blocks": [numeric, text], "decision_capture": capture}
    )

    events = [
        e for e in capture.events
        if e["decision_type"] == "assessment_numeric_equivalence_check"
    ]
    # Only the numeric block is audited; the text block scope-skips.
    assert len(events) == 1
    assert events[0]["decision"] == "passed"
    assert events[0]["metrics"]["equivalent_found"] is False
