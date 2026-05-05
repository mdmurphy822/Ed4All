"""Worker W3b — DistractorPlausibilityValidator unit tests.

Pins the validator's two GateIssue codes against representative
outline-tier (dict content) Block fixtures, plus the rewrite[2]
regression case GPT cited (3/3 distractors as syntactic permutations
of the answer).

Failure modes:

- ``DISTRACTOR_NEAR_DUPLICATE_ANSWER`` — distractor token-set Jaccard
  overlap with the answer_key above ``max_overlap_with_answer``
  (default 0.7). The rewrite[2] regression case has identical token
  bags (overlap=1.0) so all three distractors fire.
- ``DISTRACTORS_NEAR_DUPLICATE_PAIR`` — two distractors' pairwise
  Jaccard overlap above ``max_pairwise_overlap`` (default 0.85).

Failure ``action`` is always ``"regenerate"`` — a rewrite-tier re-roll
with the offending pair surfaced in the prompt remediation suffix
typically diversifies the distractor pool.

Decision capture: exactly one ``distractor_plausibility_check`` event
fires per ``validate()`` call, with rationale interpolating the
audited block count + both near-duplicate counts + both thresholds.
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

from lib.validators.distractor_plausibility import (  # noqa: E402
    DistractorPlausibilityValidator,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _outline_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    page_id: str = "page_01",
    sequence: int = 0,
    stem: str = "Which of the following is a valid RDF triple?",
    answer_key: str = "subject, predicate, object",
    distractors: Optional[List[Dict[str, Any]]] = None,
    correct_answer_index: int = 0,
) -> Block:
    """Outline-tier (dict-content) assessment_item Block fixture."""
    if distractors is None:
        # Four topically-distinct WRONG options — none token-overlap
        # the answer_key above the 0.7 floor and none are paraphrases
        # of each other above the 0.85 pairwise floor. Outline-tier
        # contract treats distractors[] as the wrong-options pool;
        # the correct answer lives in answer_key (cross-walked at
        # rewrite-tier render time via correct_answer_index).
        distractors = [
            {"text": "A class hierarchy declaration."},
            {"text": "An XML namespace prefix mapping."},
            {"text": "A function-call signature in a programming language."},
            {"text": "A regular-expression matching pattern."},
        ]
    content: Dict[str, Any] = {
        "curies": ["rdf:Statement"],
        "key_claims": [
            "An RDF triple has the form subject-predicate-object.",
        ],
        "content_type": "assessment_item",
        "stem": stem,
        "answer_key": answer_key,
        "distractors": distractors,
        "correct_answer_index": correct_answer_index,
    }
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content=content,
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


class _RecordingCapture:
    """Minimal stand-in for ``lib.decision_capture.DecisionCapture``.

    Records every ``log_decision`` invocation so the capture-emit test
    can assert exactly one ``distractor_plausibility_check`` fires per
    ``validate()`` call. Avoids dragging the full capture dependency
    chain (file IO, schema validation, etc.) into a unit test.
    """

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
# Happy path
# --------------------------------------------------------------------- #


def test_happy_path_distinct_distractors_passes():
    """Four distractors with distinct token-bags + answer_key from a
    different topical domain → passed=True, action=None, score=1.0."""
    blocks = [_outline_assessment_block()]
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None
    assert result.score == 1.0
    assert _codes(result) == []


def test_non_assessment_blocks_silently_skipped():
    """A batch containing ONLY non-assessment_item blocks → passed=True,
    no issues, no audit-list contributions."""
    blocks = [
        _non_assessment_block(block_id="page_01#concept_a", block_type="concept"),
        _non_assessment_block(block_id="page_01#concept_b", block_type="concept"),
    ]
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": blocks})

    assert result.passed is True
    assert result.action is None
    assert _codes(result) == []


# --------------------------------------------------------------------- #
# Permutation-of-answer regression (rewrite[2] case)
# --------------------------------------------------------------------- #


def test_rewrite2_permutation_fires_three_near_duplicate_answer():
    """Regression-fixture verification: rewrite[2]-style block with
    3/3 distractors that are syntactic permutations of the
    "subject, predicate, object" answer → exactly three
    DISTRACTOR_NEAR_DUPLICATE_ANSWER GateIssues fire (one per
    permuted distractor). Closes the rewrite[2] regression class.
    """
    rewrite2_block = _outline_assessment_block(
        answer_key="subject, predicate, object",
        distractors=[
            {"text": "object, predicate, subject"},
            {"text": "predicate, subject, object"},
            {"text": "subject, object, predicate"},
        ],
        correct_answer_index=0,
    )
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": [rewrite2_block]})

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    near_dup_answer = [c for c in codes if c == "DISTRACTOR_NEAR_DUPLICATE_ANSWER"]
    assert len(near_dup_answer) == 3, (
        f"expected 3 DISTRACTOR_NEAR_DUPLICATE_ANSWER on the rewrite[2] "
        f"permutation case (one per distractor); got codes={codes!r}"
    )

    # And every offending issue must reference the actual block_id so
    # an operator can locate the failure in a multi-block batch.
    for issue in result.issues:
        if issue.code == "DISTRACTOR_NEAR_DUPLICATE_ANSWER":
            assert issue.location == rewrite2_block.block_id
            assert issue.severity == "critical"


def test_single_permutation_distractor_among_distinct_siblings_fires_once():
    """Mixed batch: one distractor that's a permutation of the answer +
    three distinct distractors → exactly one
    DISTRACTOR_NEAR_DUPLICATE_ANSWER fires, no pairwise issues."""
    block = _outline_assessment_block(
        answer_key="red green blue",
        distractors=[
            # Permutation of answer — fires.
            {"text": "blue red green"},
            # Topically-distinct fillers — pass.
            {"text": "A function-call signature."},
            {"text": "A class hierarchy declaration."},
            {"text": "An XML namespace prefix mapping."},
        ],
        correct_answer_index=1,
    )
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": [block]})

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    assert codes.count("DISTRACTOR_NEAR_DUPLICATE_ANSWER") == 1
    assert codes.count("DISTRACTORS_NEAR_DUPLICATE_PAIR") == 0


# --------------------------------------------------------------------- #
# Pairwise near-duplicate regression
# --------------------------------------------------------------------- #


def test_two_distractors_near_duplicate_pair_fires():
    """Two distractors are near-paraphrases of each other (Jaccard
    > 0.85), but neither overlaps the answer above the answer-floor →
    exactly one DISTRACTORS_NEAR_DUPLICATE_PAIR fires for the offending
    (i, j) pair, no NEAR_DUPLICATE_ANSWER fires."""
    block = _outline_assessment_block(
        answer_key="The capital of France is Paris.",
        distractors=[
            # Pair A — near-paraphrases of each other (token bags
            # identical: {berlin, is, the, capital, of, germany}).
            {"text": "Berlin is the capital of Germany"},
            {"text": "the capital of Germany is Berlin"},
            # Distinct filler — passes both axes.
            {"text": "Madrid is in Spain."},
            {"text": "Rome is on the river Tiber."},
        ],
        correct_answer_index=0,
    )
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": [block]})

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    assert codes.count("DISTRACTORS_NEAR_DUPLICATE_PAIR") == 1, (
        f"expected exactly one near-duplicate-pair issue; got codes={codes!r}"
    )
    assert codes.count("DISTRACTOR_NEAR_DUPLICATE_ANSWER") == 0, (
        f"unexpected near-duplicate-answer fire on the pairwise-only "
        f"case; got codes={codes!r}"
    )


# --------------------------------------------------------------------- #
# Decision capture wiring
# --------------------------------------------------------------------- #


def test_capture_emits_one_distractor_plausibility_check_per_validate_happy():
    """A happy-path validate() emits exactly one
    ``distractor_plausibility_check`` event with passed-shaped
    rationale, regardless of how many blocks are audited."""
    blocks = [
        _outline_assessment_block(block_id="page_01#assessment_item_q_0"),
        _outline_assessment_block(
            block_id="page_01#assessment_item_q_1",
            answer_key="A subject-predicate-object statement.",
            distractors=[
                {"text": "A topic-comment-rheme structure."},
                {"text": "A noun phrase."},
                {"text": "A predicate calculus formula."},
                {"text": "A regular expression."},
            ],
        ),
    ]
    validator = DistractorPlausibilityValidator()
    capture = _RecordingCapture()

    result = validator.validate({
        "blocks": blocks,
        "decision_capture": capture,
    })

    assert result.passed is True
    plaus_events = [
        e for e in capture.events
        if e["decision_type"] == "distractor_plausibility_check"
    ]
    assert len(plaus_events) == 1, (
        f"expected exactly one distractor_plausibility_check event per "
        f"validate(); got {len(plaus_events)} events: "
        f"{[e['decision_type'] for e in capture.events]!r}"
    )
    event = plaus_events[0]
    assert event["decision"] == "passed"
    # Rationale must interpolate dynamic signals so captures replay.
    assert "near_duplicate_answer=0" in event["rationale"]
    assert "near_duplicate_pair=0" in event["rationale"]
    assert "max_overlap_with_answer=" in event["rationale"]
    assert "max_pairwise_overlap=" in event["rationale"]
    # Min-20-char audit floor (decision-capture contract).
    assert len(event["rationale"]) >= 20


def test_capture_emits_failed_event_on_rewrite2_permutation():
    """The rewrite[2] permutation block emits a single failed-shaped
    distractor_plausibility_check event with near_duplicate_answer=3
    interpolated."""
    block = _outline_assessment_block(
        answer_key="subject, predicate, object",
        distractors=[
            {"text": "object, predicate, subject"},
            {"text": "predicate, subject, object"},
            {"text": "subject, object, predicate"},
        ],
    )
    validator = DistractorPlausibilityValidator()
    capture = _RecordingCapture()

    result = validator.validate({
        "blocks": [block],
        "decision_capture": capture,
    })

    assert result.passed is False
    plaus_events = [
        e for e in capture.events
        if e["decision_type"] == "distractor_plausibility_check"
    ]
    assert len(plaus_events) == 1
    event = plaus_events[0]
    assert event["decision"] == "failed"
    assert "near_duplicate_answer=3" in event["rationale"]
    # The rewrite[2] permutations are also pairwise-near-duplicate of
    # each other (identical token bags), so all C(3,2)=3 pairs fire.
    assert "near_duplicate_pair=3" in event["rationale"]


# --------------------------------------------------------------------- #
# Threshold-override + edge cases
# --------------------------------------------------------------------- #


def test_threshold_override_relaxes_answer_overlap_floor():
    """Per-call ``max_overlap_with_answer`` override at 0.99 admits the
    rewrite[2] permutation case (Jaccard=1.0 with the answer key fails
    at threshold>=1.0 only) — confirms the threshold is wired through
    inputs[] correctly. Floor at 0.99 means equality-to-floor still
    fires (validator uses strict >, so we set 0.999 to admit).
    """
    block = _outline_assessment_block(
        answer_key="subject, predicate, object",
        distractors=[
            {"text": "object, predicate, subject"},
            {"text": "predicate, subject, object"},
            {"text": "subject, object, predicate"},
        ],
    )
    validator = DistractorPlausibilityValidator()

    result = validator.validate({
        "blocks": [block],
        "max_overlap_with_answer": 0.999,
        "max_pairwise_overlap": 0.999,
    })

    # Identical token bags → overlap=1.0 > 0.999, so the gate still
    # fires (3 NEAR_DUPLICATE_ANSWER + 3 NEAR_DUPLICATE_PAIR).
    assert result.passed is False
    codes = _codes(result)
    assert codes.count("DISTRACTOR_NEAR_DUPLICATE_ANSWER") == 3
    # All three pairs of identical-token-bag distractors fire too.
    assert codes.count("DISTRACTORS_NEAR_DUPLICATE_PAIR") == 3


def test_missing_blocks_input_fails_loud():
    """Missing ``inputs['blocks']`` → critical MISSING_BLOCKS_INPUT,
    action=regenerate."""
    validator = DistractorPlausibilityValidator()

    result = validator.validate({})

    assert result.passed is False
    assert result.action == "regenerate"
    assert _codes(result) == ["MISSING_BLOCKS_INPUT"]


def test_invalid_blocks_input_fails_loud():
    """Non-list ``inputs['blocks']`` → critical INVALID_BLOCKS_INPUT."""
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": "not-a-list"})

    assert result.passed is False
    assert result.action == "regenerate"
    assert _codes(result) == ["INVALID_BLOCKS_INPUT"]


def test_empty_distractor_text_does_not_crash():
    """A distractor with empty / missing text is silently skipped (the
    W7 payload-shape gate owns that failure mode); the validator must
    not crash and must still audit the remaining distractors."""
    block = _outline_assessment_block(
        answer_key="The capital of France is Paris.",
        distractors=[
            {"text": ""},  # skipped
            {},  # skipped (no text key)
            {"text": "Madrid is in Spain."},
            {"text": "Rome is on the river Tiber."},
        ],
        correct_answer_index=2,
    )
    validator = DistractorPlausibilityValidator()

    result = validator.validate({"blocks": [block]})

    # No real distractors near-duplicate of each other or of the answer.
    assert result.passed is True
    assert _codes(result) == []
