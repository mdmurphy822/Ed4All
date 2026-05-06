"""Worker W2.D — PaddedDistractorValidator unit tests.

Pins the validator's 5 canonical padding patterns against representative
``assessment_item`` Block fixtures plus the boundary cases (pattern in
stem / pattern in correct answer must NOT fire), the action-transition
contract (outline tier → ``regenerate``; rewrite tier → ``block``), and
the per-call decision-emit regression.

Failure modes:

- ``PADDED_DISTRACTOR_PATTERN_1`` — ``"A concept unrelated to ..."``
  pre-W2.D canonical pad and lexical variants.
- ``PADDED_DISTRACTOR_PATTERN_2`` — ``"Not related/associated/connected
  to ..."`` paraphrased pads.
- ``PADDED_DISTRACTOR_PATTERN_3`` — ``"Distractor"`` / ``"Wrong answer
  2"`` / ``"Incorrect"`` literal placeholder strings.
- ``PADDED_DISTRACTOR_PATTERN_4`` — ``"None of the above"`` lazy
  non-distractor pad.
- ``PADDED_DISTRACTOR_PATTERN_5`` — ``"This is a wrong/incorrect/
  distractor answer"`` meta-description pads.
- ``MISSING_BLOCKS_INPUT`` / ``INVALID_BLOCKS_INPUT`` — symmetric
  sibling-validator input-shape guards.

Decision-emit regression: every ``validate()`` call (including the
no-fire happy path) emits exactly one
``decision_type="padded_distractor_check"`` event with rationale
interpolating the dynamic per-call signal. Mirrors W2.C's per-block
emit pattern.
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

from lib.validators.padded_distractor import (  # noqa: E402
    PaddedDistractorValidator,
    _PADDING_PATTERNS,
)


# --------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal DecisionCapture stand-in recording log_decision kwargs."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    stem: str = "What is an RDF triple?",
    correct_text: str = (
        "An RDF triple is a subject-predicate-object statement expressing "
        "a fact in a directed labelled graph."
    ),
    distractors: Optional[List[str]] = None,
    block_type: str = "assessment_item",
) -> Block:
    """Trainforge ``choices[]`` shape assessment_item block.

    ``distractors`` is a list of plain wrong-option text; the fixture
    wraps each in ``{"text": ..., "is_correct": False}`` and prepends
    the correct entry.
    """
    if distractors is None:
        distractors = [
            "An XML namespace prefix mapping.",
            "A function-call signature in code.",
            "A regular-expression matching pattern.",
        ]
    choices: List[Dict[str, Any]] = [
        {"text": f"<p>{correct_text}</p>", "is_correct": True},
    ]
    for d in distractors:
        choices.append({"text": f"<p>{d}</p>", "is_correct": False})

    content: Dict[str, Any] = {
        "stem": f"<p>{stem}</p>",
        "choices": choices,
        "content_type": "assessment_item",
    }
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=("TO-01",),
        source_ids=("dart:rdf_intro#blk_0",),
    )


def _legacy_distractors_block(
    *,
    block_id: str = "page_01#assessment_item_q_legacy",
    answer_key: str = (
        "An RDF triple is a subject-predicate-object statement."
    ),
    distractors: Optional[List[str]] = None,
) -> Block:
    """Legacy ``distractors[] + answer_key`` shape (W2.D must catch this too)."""
    if distractors is None:
        distractors = [
            "An XML namespace prefix mapping.",
            "A function-call signature in code.",
            "A regular-expression matching pattern.",
        ]
    content: Dict[str, Any] = {
        "stem": "What is an RDF triple?",
        "distractors": [{"text": d} for d in distractors],
        "answer_key": answer_key,
        "correct_answer_index": 0,
        "content_type": "assessment_item",
    }
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=("TO-01",),
        source_ids=("dart:rdf_intro#blk_0",),
    )


def _padded_check_events(capture: _RecordingCapture) -> List[Dict[str, Any]]:
    return [
        e for e in capture.events
        if e.get("decision_type") == "padded_distractor_check"
    ]


def _codes(result) -> List[str]:
    return [issue.code for issue in result.issues]


# --------------------------------------------------------------------- #
# Per-pattern coverage — each of the 5 canonical pads triggers the
# correct PADDED_DISTRACTOR_PATTERN_N code.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "padded_text,expected_code",
    [
        # Pattern 1 — canonical pre-W2.D pad (the actual generator string).
        (
            "A concept unrelated to RDF in this context",
            "PADDED_DISTRACTOR_PATTERN_1",
        ),
        # Pattern 1 — lexical variant ("topic" instead of "concept").
        # The canonical regex matches only the determiner ``A`` (the
        # generator's pre-W2.D pad string was always 'A concept ...');
        # ``An`` variants are out-of-band per the W2.D brief's
        # verbatim ``[Aa]`` character class.
        (
            "A topic unrelated to SHACL in this context",
            "PADDED_DISTRACTOR_PATTERN_1",
        ),
        # Pattern 1 — "idea" + lowercase determiner.
        (
            "a idea unrelated to graphs in this context",
            "PADDED_DISTRACTOR_PATTERN_1",
        ),
        # Pattern 2 — "Not related to".
        (
            "Not related to the subject under discussion",
            "PADDED_DISTRACTOR_PATTERN_2",
        ),
        # Pattern 2 — "not associated with" wording.
        (
            "Not associated to the topic in question",
            "PADDED_DISTRACTOR_PATTERN_2",
        ),
        # Pattern 2 — "Not connected to" wording.
        (
            "not connected to the cited concept directly",
            "PADDED_DISTRACTOR_PATTERN_2",
        ),
        # Pattern 3 — bare "Distractor".
        ("Distractor", "PADDED_DISTRACTOR_PATTERN_3"),
        # Pattern 3 — "Wrong answer 2".
        ("Wrong answer 2", "PADDED_DISTRACTOR_PATTERN_3"),
        # Pattern 3 — "Incorrect".
        ("Incorrect", "PADDED_DISTRACTOR_PATTERN_3"),
        # Pattern 4 — "None of the above".
        ("None of the above", "PADDED_DISTRACTOR_PATTERN_4"),
        # Pattern 4 — lowercase variant.
        ("none of the above", "PADDED_DISTRACTOR_PATTERN_4"),
        # Pattern 5 — meta-description "This is a wrong answer".
        (
            "This is a wrong answer to the question",
            "PADDED_DISTRACTOR_PATTERN_5",
        ),
        # Pattern 5 — "This is incorrect answer".
        (
            "This is incorrect answer for the prompt",
            "PADDED_DISTRACTOR_PATTERN_5",
        ),
        # Pattern 5 — "This is distractor answer".
        (
            "this is distractor answer placeholder text",
            "PADDED_DISTRACTOR_PATTERN_5",
        ),
    ],
)
def test_each_padding_pattern_triggers_correct_code(padded_text, expected_code):
    """Every canonical padding pattern must fire the matching code."""
    block = _assessment_block(distractors=[
        "A plausible misconception-targeted distractor about RDF graphs.",
        padded_text,
        "Another plausible misconception about graph predicates.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is False, (
        f"Expected fail on padded distractor {padded_text!r}; "
        f"got passed=True"
    )
    assert expected_code in _codes(result), (
        f"Expected GateIssue code {expected_code!r} for distractor "
        f"{padded_text!r}; got codes={_codes(result)}"
    )


# --------------------------------------------------------------------- #
# Happy path — clean assessment passes
# --------------------------------------------------------------------- #


def test_clean_assessment_with_plausible_distractors_passes():
    """An assessment_item with three plausible misconception-targeted
    distractors must NOT fire any PADDED_DISTRACTOR_PATTERN_N code.
    """
    block = _assessment_block(distractors=[
        "An XML namespace prefix mapping declared in a schema header.",
        "A function-call signature in code with named parameters.",
        "A regular-expression pattern matching directed graph traversals.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is True, (
        f"Clean distractors triggered the gate; codes={_codes(result)}"
    )
    assert result.action is None
    assert _codes(result) == []


def test_legacy_distractors_shape_also_audited():
    """The legacy ``distractors[] + answer_key`` shape must be audited
    too — the W2.D gate fires on the legacy emit path the same way it
    fires on the canonical Trainforge ``choices[]`` shape.
    """
    block = _legacy_distractors_block(distractors=[
        "An XML namespace prefix mapping.",
        "A concept unrelated to RDF in this context",  # padded pattern 1
        "A regular-expression pattern.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is False
    assert "PADDED_DISTRACTOR_PATTERN_1" in _codes(result)


# --------------------------------------------------------------------- #
# Boundary cases — pattern in stem / correct answer must NOT fire
# --------------------------------------------------------------------- #


def test_pattern_in_stem_does_not_fire():
    """A padded-pattern string in the stem (not a distractor) must NOT
    fire the gate. The validator's domain is wrong-options exclusively.
    """
    block = _assessment_block(
        # Stem itself contains canonical pad text; distractors are clean.
        stem="A concept unrelated to RDF in this context — what is RDF?",
        distractors=[
            "An XML namespace prefix mapping.",
            "A function-call signature in code.",
            "A regular-expression matching pattern.",
        ],
    )
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is True, (
        f"Pattern in stem triggered the gate; codes={_codes(result)}"
    )


def test_pattern_in_correct_answer_does_not_fire():
    """A padded-pattern string in the correct answer (is_correct=True)
    must NOT fire the gate. Distractors only.
    """
    block = _assessment_block(
        # Correct answer carries pad-like text; the validator must skip
        # it because the validator's domain is wrong-options only. The
        # canonical answer-grounding gate covers the correct-answer side.
        correct_text="A concept unrelated to RDF in this context",
        distractors=[
            "An XML namespace prefix mapping.",
            "A function-call signature in code.",
            "A regular-expression matching pattern.",
        ],
    )
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is True


def test_non_assessment_blocks_silently_skipped():
    """Concept / example / chrome blocks must be silently skipped even
    when their content shape happens to carry a 'choices' key.
    """
    fake_concept = Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "stem": "A concept unrelated to RDF in this context",
            "choices": [
                {"text": "A concept unrelated to RDF in this context",
                 "is_correct": False},
            ],
            "content_type": "concept",
        },
        objective_ids=("TO-01",),
    )
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [fake_concept]})

    assert result.passed is True
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# Action-transition contract — outline tier ⇒ regenerate; rewrite /
# packaging / trainforge_assessment tier ⇒ block.
# --------------------------------------------------------------------- #


def test_outline_tier_action_is_regenerate():
    """At the inter_tier_validation seam the action must be
    ``regenerate`` so the rewrite tier re-rolls the block.
    """
    block = _assessment_block(distractors=[
        "A concept unrelated to RDF in this context",
        "An XML namespace prefix mapping.",
        "A function-call signature in code.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({
        "blocks": [block],
        "phase": "inter_tier_validation",
    })

    assert result.passed is False
    assert result.action == "regenerate"


def test_rewrite_tier_action_is_block():
    """At the post_rewrite_validation seam (canonical published HTML)
    a padded distractor is a structural unfixable; the action MUST be
    ``block`` instead of ``regenerate``.
    """
    block = _assessment_block(distractors=[
        "A concept unrelated to RDF in this context",
        "An XML namespace prefix mapping.",
        "A function-call signature in code.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({
        "blocks": [block],
        "phase": "post_rewrite_validation",
    })

    assert result.passed is False
    assert result.action == "block"


def test_trainforge_assessment_tier_action_is_block():
    """trainforge_assessment phase mirrors rewrite-tier posture
    (post-publication seam → block, not regenerate).
    """
    block = _assessment_block(distractors=[
        "None of the above",
        "An XML namespace prefix mapping.",
        "A function-call signature in code.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({
        "blocks": [block],
        "phase": "trainforge_assessment",
    })

    assert result.passed is False
    assert result.action == "block"


def test_default_phase_outline_tier_action():
    """When no ``phase`` input is supplied, default to outline-tier
    posture (``regenerate``). Mirrors the validator's default kwarg.
    """
    block = _assessment_block(distractors=[
        "Distractor",
        "An XML namespace prefix mapping.",
        "A function-call signature in code.",
    ])
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": [block]})

    assert result.passed is False
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Decision-emit regression — every validate() call emits one
# padded_distractor_check event with rationale interpolating dynamic
# signals.
# --------------------------------------------------------------------- #


def test_decision_emit_on_pass_path():
    """Even on the happy path (no padding), one
    ``padded_distractor_check`` decision must fire so the audit log
    has a positive signal that the gate ran cleanly.
    """
    capture = _RecordingCapture()
    block = _assessment_block()
    validator = PaddedDistractorValidator()
    result = validator.validate({
        "blocks": [block],
        "decision_capture": capture,
    })

    assert result.passed is True
    events = _padded_check_events(capture)
    assert len(events) == 1, (
        f"Expected exactly one padded_distractor_check event on the "
        f"pass path; got {len(events)}: {events}"
    )
    rationale = events[0].get("rationale", "")
    assert "matched_count=0" in rationale
    assert "audited" not in rationale or "1" in rationale
    # Rationale must be ≥20 chars for the schema-side decision_validation
    # gate.
    assert len(rationale) >= 20


def test_decision_emit_on_fail_path_carries_matched_codes():
    """On a fail, the decision rationale must interpolate the matched
    pattern codes so a post-hoc audit can replay which pad regex fired.
    """
    capture = _RecordingCapture()
    block = _assessment_block(distractors=[
        "A concept unrelated to RDF in this context",  # pattern 1
        "None of the above",  # pattern 4
        "A regular-expression matching pattern.",
    ])
    validator = PaddedDistractorValidator()
    validator.validate({
        "blocks": [block],
        "decision_capture": capture,
    })

    events = _padded_check_events(capture)
    assert len(events) == 1
    rationale = events[0]["rationale"]
    assert "PADDED_DISTRACTOR_PATTERN_1" in rationale
    assert "PADDED_DISTRACTOR_PATTERN_4" in rationale
    assert "matched_count=2" in rationale


def test_decision_emit_on_input_shape_error_path():
    """Even on the input-shape error path (MISSING_BLOCKS_INPUT), the
    GateResult must still surface the failure properly.
    """
    validator = PaddedDistractorValidator()
    result = validator.validate({})  # blocks input missing
    assert result.passed is False
    codes = _codes(result)
    assert "MISSING_BLOCKS_INPUT" in codes


# --------------------------------------------------------------------- #
# Schema enum — decision_type registered in canonical enum
# --------------------------------------------------------------------- #


def test_decision_type_in_canonical_enum():
    """The ``padded_distractor_check`` decision_type must be in the
    canonical ``decision_event.schema.json::decision_type.enum`` so
    DECISION_VALIDATION_STRICT runs don't fail-close on the emit.
    """
    import json
    schema_path = (
        _PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum_values = schema["properties"]["decision_type"]["enum"]
    assert "padded_distractor_check" in enum_values, (
        "Worker W2.D: `padded_distractor_check` must be registered in "
        "schemas/events/decision_event.schema.json::decision_type.enum"
    )


# --------------------------------------------------------------------- #
# Input-shape guards mirror the sibling-validator contract.
# --------------------------------------------------------------------- #


def test_invalid_blocks_input_fails_closed():
    """``inputs['blocks']`` must be a list; any other type fails closed
    with ``INVALID_BLOCKS_INPUT``.
    """
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": {"not": "a list"}})
    assert result.passed is False
    assert "INVALID_BLOCKS_INPUT" in _codes(result)


def test_empty_blocks_list_passes_vacuously():
    """An empty blocks list passes vacuously — no audited blocks → no
    fire path. Score 1.0.
    """
    validator = PaddedDistractorValidator()
    result = validator.validate({"blocks": []})
    assert result.passed is True
    assert result.score == 1.0


def test_padding_patterns_constant_size():
    """The W2.D plan calls out exactly 5 canonical patterns; if a
    refactor ever drops or adds one without updating this test, the
    contract drifted.
    """
    assert len(_PADDING_PATTERNS) == 5
    codes = [code for code, _, _ in _PADDING_PATTERNS]
    assert codes == [
        "PADDED_DISTRACTOR_PATTERN_1",
        "PADDED_DISTRACTOR_PATTERN_2",
        "PADDED_DISTRACTOR_PATTERN_3",
        "PADDED_DISTRACTOR_PATTERN_4",
        "PADDED_DISTRACTOR_PATTERN_5",
    ]
