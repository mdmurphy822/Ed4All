"""Worker W6 — tests for ``BloomStructuralEnforcementValidator``.

Coverage matrix:

* Six happy-path tests (one per Bloom level) — declared level + stem +
  answer + question_type wired so every per-level structural rule
  passes; ``GateResult`` is a clean pass with empty ``issues`` and
  ``action=None``.
* Six mismatch tests (one per Bloom level, including the canonical
  W6 regression: analyze-declared block with a "remember"-shaped
  ``"What is TCP?"`` stem). Each assert the correct
  ``BLOOM_*`` GateIssue code fires + ``action="regenerate"``.
* Capture-emit test — passes a stub capture and asserts a single
  ``decision_type="bloom_structural_enforcement_check"`` event fires
  per validate() call.
* Plus auxiliary regression coverage for skip behaviour (non-audited
  block types, undeclared bloom_level, missing input shape).

Verbs used in stems are pulled from the canonical bloom_verbs schema
via :func:`lib.ontology.bloom.get_verbs` so a future schema update
won't silently regress these tests; the test suite imports the same
helper the validator does.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Block lives at Courseforge/scripts/blocks.py — same import bridge
# the bloom_classifier_disagreement test uses.
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.ontology.bloom import get_verbs  # noqa: E402
from lib.validators.bloom_structural_enforcement import (  # noqa: E402
    _ANSWER_TOKEN_FLOORS,
    _CLAUSE_FLOORS,
    BloomStructuralEnforcementValidator,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _make_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    page_id: str = "page_01",
    sequence: int = 0,
    stem: str = "Identify the protocol layer.",
    bloom_level: Optional[str] = "remember",
    correct_answer: str = "TCP is a transport layer protocol.",
    question_type: str = "short_answer",
    options: Optional[List[Dict[str, Any]]] = None,
) -> Block:
    """Build an assessment_item Block with a stem + answer surface.

    The validator pulls the stem from ``content["stem"]`` first, the
    answer from ``content["correct_answer"]`` (or ``options[]`` when
    multiple-choice), and the question_type from
    ``content["question_type"]``. Test surfaces all three fields so a
    test can isolate which structural rule trips.
    """
    content: Dict[str, Any] = {
        "stem": stem,
        "correct_answer": correct_answer,
        "question_type": question_type,
    }
    if options is not None:
        content["options"] = options
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content=content,
        bloom_level=bloom_level,
    )


def _make_concept_block() -> Block:
    """Non-audited Block (block_type='concept' is filtered out)."""
    return Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={"key_claims": ["Networking layers stack."]},
    )


class _StubCapture:
    """Records every ``log_decision`` call; mirrors DecisionCapture API."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(dict(kwargs))


# --------------------------------------------------------------------- #
# Happy-path: one per Bloom level
# --------------------------------------------------------------------- #


def test_remember_happy_path_passes() -> None:
    """remember: 1-clause, remember-set verb, answer floor=0."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem="List the protocol layers in the TCP/IP stack.",
        bloom_level="remember",
        correct_answer="Application, transport, internet, link.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_understand_happy_path_passes() -> None:
    """understand: 1-clause, understand-set verb (explain), no floor."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem="Explain how TCP guarantees ordered delivery.",
        bloom_level="understand",
        correct_answer="TCP uses sequence numbers and acknowledgements.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_apply_happy_path_passes() -> None:
    """apply: >=2 clauses, apply-set verb, answer >=4 tokens."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem=(
            "Solve the routing puzzle, and apply the shortest-path "
            "algorithm to find the next hop."
        ),
        bloom_level="apply",
        correct_answer="The next hop is router R3 via interface eth0.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_analyze_happy_path_passes() -> None:
    """analyze: >=2 clauses, analyze-set verb (differentiate), comparison
    marker (contrast), answer >=6 tokens."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem=(
            "Differentiate TCP from UDP, and contrast their reliability "
            "guarantees."
        ),
        bloom_level="analyze",
        correct_answer=(
            "TCP provides ordered, reliable delivery via sequence "
            "numbers and acknowledgements; UDP is connectionless and "
            "best-effort."
        ),
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed, f"unexpected issues: {[i.code for i in result.issues]}"
    assert result.action is None
    assert result.issues == []


def test_evaluate_happy_path_passes() -> None:
    """evaluate: >=2 clauses, evaluate-set verb (assess), judgment
    marker (justify), answer >=8 tokens."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem=(
            "Assess the security trade-offs of TLS 1.3, and justify "
            "your reasoning with reference to forward secrecy."
        ),
        bloom_level="evaluate",
        correct_answer=(
            "TLS 1.3 enforces forward secrecy by default through "
            "ephemeral key exchange, materially reducing long-term "
            "key compromise risk relative to TLS 1.2."
        ),
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed, f"unexpected issues: {[i.code for i in result.issues]}"
    assert result.action is None
    assert result.issues == []


def test_create_happy_path_passes() -> None:
    """create: >=2 clauses, create-set verb (design), design marker
    (construct), open-ended question_type (essay)."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem=(
            "Design a load-balanced web architecture, and construct a "
            "deployment diagram showing failover paths."
        ),
        bloom_level="create",
        # create-level intentionally has no answer-token floor; the
        # open-ended question_type check supersedes it.
        correct_answer="(open-ended response expected)",
        question_type="essay",
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed, f"unexpected issues: {[i.code for i in result.issues]}"
    assert result.action is None
    assert result.issues == []


# --------------------------------------------------------------------- #
# Mismatch: one per Bloom level
# --------------------------------------------------------------------- #


def test_remember_mismatch_emits_verb_mismatch() -> None:
    """remember-declared block with no remember-set verb in the stem
    fires BLOOM_VERB_MISMATCH."""
    validator = BloomStructuralEnforcementValidator()
    # "Why" / "Justify" — no remember-set verb (define/list/recall/etc.).
    block = _make_assessment_block(
        stem="Justify why the sky appears blue at noon.",
        bloom_level="remember",
        correct_answer="Rayleigh scattering favours short wavelengths.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "BLOOM_VERB_MISMATCH" in codes


def test_understand_mismatch_emits_verb_mismatch() -> None:
    """understand-declared block with no understand-set verb fires
    BLOOM_VERB_MISMATCH."""
    validator = BloomStructuralEnforcementValidator()
    # "Construct" — create-set, not understand-set.
    block = _make_assessment_block(
        stem="Construct a sentence using the new vocabulary term.",
        bloom_level="understand",
        correct_answer="Any well-formed sentence using the term.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "BLOOM_VERB_MISMATCH" in codes


def test_apply_mismatch_emits_insufficient_clauses() -> None:
    """apply-declared block with a single-clause stem fires
    BLOOM_STRUCTURE_INSUFFICIENT_CLAUSES."""
    validator = BloomStructuralEnforcementValidator()
    # Single-clause "Apply X" stem — one clause, apply-set verb, but
    # the clause floor of 2 trips first.
    block = _make_assessment_block(
        stem="Apply the formula.",
        bloom_level="apply",
        correct_answer="The result is 42.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "BLOOM_STRUCTURE_INSUFFICIENT_CLAUSES" in codes


def test_analyze_remember_shaped_stem_fires_insufficient_clauses() -> None:
    """W6 canonical regression: analyze-declared block with the
    "remember"-shaped ``What is TCP?`` stem trips the clause floor
    AND verb-set check AND missing-comparison-marker check."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem="What is TCP?",
        bloom_level="analyze",
        correct_answer="TCP is the Transmission Control Protocol.",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    # All three structural rules trip on this canonical bad-stem case.
    assert "BLOOM_STRUCTURE_INSUFFICIENT_CLAUSES" in codes
    assert "BLOOM_VERB_MISMATCH" in codes
    assert "BLOOM_MISSING_COMPARISON_MARKER" in codes


def test_evaluate_mismatch_emits_missing_judgment_marker() -> None:
    """evaluate-declared block with no judgment marker fires
    BLOOM_MISSING_JUDGMENT_MARKER."""
    validator = BloomStructuralEnforcementValidator()
    # "Argue" is in the evaluate verb set, two-clause stem, but no
    # judgment marker (justify/assess/critique/defend) — so the
    # judgment-marker rule trips.
    block = _make_assessment_block(
        stem="Argue the policy stance, and present supporting evidence.",
        bloom_level="evaluate",
        correct_answer=(
            "The policy reduces friction, accelerates throughput, "
            "and lowers operational cost."
        ),
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "BLOOM_MISSING_JUDGMENT_MARKER" in codes


def test_create_mismatch_fires_must_be_open_ended() -> None:
    """create-declared block with a fixed-answer question_type fires
    CREATE_LEVEL_MUST_BE_OPEN_ENDED."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem="Design a system, and construct the deployment plan.",
        bloom_level="create",
        correct_answer="Option B",
        question_type="multiple_choice",
        options=[
            {"text": "Option A", "correct": False},
            {"text": "Option B", "correct": True},
        ],
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "regenerate"
    codes = {i.code for i in result.issues}
    assert "CREATE_LEVEL_MUST_BE_OPEN_ENDED" in codes


# --------------------------------------------------------------------- #
# Per-level marker / answer-floor coverage (defense-in-depth)
# --------------------------------------------------------------------- #


def test_analyze_missing_comparison_marker_fires() -> None:
    """analyze-declared block with two clauses and an analyze-set verb
    but NO comparison marker fires BLOOM_MISSING_COMPARISON_MARKER."""
    validator = BloomStructuralEnforcementValidator()
    # ``investigate`` is in the analyze verb set; two clauses; but no
    # compare/contrast/differentiate marker. Answer is long enough to
    # clear the 6-token floor.
    block = _make_assessment_block(
        stem=(
            "Investigate the failure mode, and report the underlying "
            "cause."
        ),
        bloom_level="analyze",
        correct_answer=(
            "The cache invalidation race condition produced stale "
            "reads under load."
        ),
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "BLOOM_MISSING_COMPARISON_MARKER" in codes


def test_create_missing_design_marker_fires() -> None:
    """create-declared block with a create-set verb but no design
    marker fires BLOOM_MISSING_DESIGN_MARKER."""
    validator = BloomStructuralEnforcementValidator()
    # ``invent`` is in the create verb set; two-clause stem; but no
    # design / construct / develop / formulate marker.
    block = _make_assessment_block(
        stem="Invent something new, and write it up.",
        bloom_level="create",
        correct_answer="(open-ended response expected)",
        question_type="essay",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "BLOOM_MISSING_DESIGN_MARKER" in codes


def test_apply_short_answer_fires_answer_too_short() -> None:
    """apply-declared block with a 2-clause stem + apply-set verb but
    a 1-token answer fires BLOOM_ANSWER_TOO_SHORT."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem=(
            "Solve for x, and use the quadratic formula to verify the "
            "result."
        ),
        bloom_level="apply",
        # 1 token < 4-token floor.
        correct_answer="42",
        question_type="short_answer",
    )
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "BLOOM_ANSWER_TOO_SHORT" in codes


# --------------------------------------------------------------------- #
# Capture emit
# --------------------------------------------------------------------- #


def test_capture_emits_one_decision_per_validate_call() -> None:
    """A wired-in capture receives one
    ``bloom_structural_enforcement_check`` event per validate() call,
    with rationale interpolating the audited / passed / failed counts.
    """
    capture = _StubCapture()
    validator = BloomStructuralEnforcementValidator(capture=capture)

    blocks = [
        _make_assessment_block(
            block_id="page_01#assessment_item_q_0",
            stem="List the protocol layers in the TCP/IP stack.",
            bloom_level="remember",
            correct_answer="Application, transport, internet, link.",
            question_type="short_answer",
        ),
        _make_assessment_block(
            block_id="page_01#assessment_item_q_1",
            stem="What is TCP?",
            bloom_level="analyze",
            correct_answer="TCP is the Transmission Control Protocol.",
            question_type="short_answer",
        ),
    ]
    result = validator.validate({"blocks": blocks})

    # The second block fails three structural rules so action='regenerate'.
    assert result.action == "regenerate"

    # Exactly one capture event per validate() call.
    assert len(capture.calls) == 1
    event = capture.calls[0]
    assert event["decision_type"] == "bloom_structural_enforcement_check"
    assert "audited=2" in event["decision"]
    assert "passed=1" in event["decision"]
    assert "failed=1" in event["decision"]
    # Rationale must hit the 20-char floor + carry the per-level
    # distribution so a downstream replay can reconstruct what was
    # audited.
    assert len(event["rationale"]) >= 20
    assert "remember=1" in event["rationale"]
    assert "analyze=1" in event["rationale"]


def test_capture_no_op_when_capture_is_none() -> None:
    """Validator without a capture wired in must not raise — capture
    is optional. Mirrors the courseforge_outline_shacl pattern."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block()
    # Just exercising the no-capture path doesn't raise.
    result = validator.validate({"blocks": [block]})
    assert isinstance(result.passed, bool)


# --------------------------------------------------------------------- #
# Skip behaviour
# --------------------------------------------------------------------- #


def test_skips_non_audited_block_types() -> None:
    """concept blocks (and any block_type outside _AUDITED_BLOCK_TYPES)
    are not structurally enforced — only assessment_item is in scope."""
    validator = BloomStructuralEnforcementValidator()
    # Even a "remember-shaped" stem on a concept block must NOT trip
    # — the gate only enforces assessment_item.
    blocks = [_make_concept_block()]
    result = validator.validate({"blocks": blocks})
    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_skips_assessment_blocks_without_declared_bloom_level() -> None:
    """Assessment blocks with empty/None bloom_level are silently
    skipped — the gate can't enforce a level that wasn't claimed."""
    validator = BloomStructuralEnforcementValidator()
    block = _make_assessment_block(
        stem="What is TCP?",  # would trip every analyze rule
        bloom_level=None,
    )
    result = validator.validate({"blocks": [block]})
    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_missing_blocks_input_is_critical_block() -> None:
    """inputs['blocks'] absent -> critical, passed=False, action='block'."""
    validator = BloomStructuralEnforcementValidator()
    result = validator.validate({})
    assert not result.passed
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"
    assert result.issues[0].severity == "critical"


def test_invalid_blocks_input_type_is_critical_block() -> None:
    """inputs['blocks'] not-a-list -> critical, passed=False, action='block'."""
    validator = BloomStructuralEnforcementValidator()
    result = validator.validate({"blocks": "not-a-list"})
    assert not result.passed
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "INVALID_BLOCKS_INPUT"


# --------------------------------------------------------------------- #
# Module-level constants regression
# --------------------------------------------------------------------- #


def test_clause_floors_match_plan() -> None:
    """Clause floors per the W6 plan: remember/understand=1, others=2."""
    assert _CLAUSE_FLOORS["remember"] == 1
    assert _CLAUSE_FLOORS["understand"] == 1
    assert _CLAUSE_FLOORS["apply"] == 2
    assert _CLAUSE_FLOORS["analyze"] == 2
    assert _CLAUSE_FLOORS["evaluate"] == 2
    assert _CLAUSE_FLOORS["create"] == 2


def test_answer_token_floors_match_plan() -> None:
    """Answer-token floors per the W6 plan: apply=4, analyze=6,
    evaluate=8; remember/understand have no floor; create uses the
    open-ended question_type check instead."""
    assert _ANSWER_TOKEN_FLOORS["apply"] == 4
    assert _ANSWER_TOKEN_FLOORS["analyze"] == 6
    assert _ANSWER_TOKEN_FLOORS["evaluate"] == 8


def test_validator_uses_canonical_verb_sets() -> None:
    """The validator's verb sets must be pulled from the same source
    of truth as ``lib.ontology.bloom.get_verbs()`` — no duplicate
    registry per the W6 acceptance criteria."""
    validator = BloomStructuralEnforcementValidator()
    canonical = get_verbs()
    # Validator caches the verb sets at construction time.
    assert validator._verb_sets == canonical


# --------------------------------------------------------------------- #
# Multi-block aggregation
# --------------------------------------------------------------------- #


def test_passing_blocks_alongside_failing_block_aggregate_correctly() -> None:
    """Score = pass_rate over audited blocks. Mixed batch: 2 pass /
    1 fail -> score=0.6667 (rounded to 4dp), action='regenerate'."""
    validator = BloomStructuralEnforcementValidator()
    blocks = [
        # pass: remember + remember-set verb.
        _make_assessment_block(
            block_id="b0",
            stem="Define the term TCP.",
            bloom_level="remember",
            correct_answer="Transmission Control Protocol.",
            question_type="short_answer",
        ),
        # pass: understand + understand-set verb.
        _make_assessment_block(
            block_id="b1",
            stem="Explain how TCP guarantees ordering.",
            bloom_level="understand",
            correct_answer="Sequence numbers + ack.",
            question_type="short_answer",
        ),
        # fail: analyze + remember-shaped stem.
        _make_assessment_block(
            block_id="b2",
            stem="What is TCP?",
            bloom_level="analyze",
            correct_answer="TCP is a transport-layer protocol.",
            question_type="short_answer",
        ),
    ]
    result = validator.validate({"blocks": blocks})

    assert not result.passed
    assert result.action == "regenerate"
    assert result.score == pytest.approx(2 / 3, abs=1e-4)
    # Issues attribute the failing block.
    assert all(i.location == "b2" for i in result.issues)
