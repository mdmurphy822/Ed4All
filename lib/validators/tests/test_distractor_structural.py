"""Worker W2.C — DistractorStructuralValidator unit tests.

Pins the validator's six GateIssue codes against representative
``assessment_item`` Block fixtures plus the Trainforge ``choices[]``
shape and the legacy ``distractors[]`` + ``answer_key`` fallback
shape. Decision-emit regression covers the per-block audit-log
contract.

Failure modes:

- ``DISTRACTOR_STRUCTURAL_MULTI_CORRECT`` (critical, regenerate) —
  ``choices[]`` declares >1 correct entry.
- ``DISTRACTOR_STRUCTURAL_NO_CORRECT`` (critical, regenerate) —
  ``choices[]`` declares zero correct entries.
- ``DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED`` (critical,
  regenerate) — distractor token-set Jaccard overlap with any cited
  source chunk above ``max_source_overlap`` floor (default 0.70).
- ``DISTRACTOR_STRUCTURAL_NO_CHOICES`` (warning) — block declares
  neither ``choices[]`` nor a legacy fallback; block skipped.
- ``DISTRACTOR_STRUCTURAL_NO_SOURCES`` (warning) — ``source_chunks``
  input absent; entailment sub-check skipped (exactly-one-correct
  still runs).
- ``MISSING_BLOCKS_INPUT`` / ``INVALID_BLOCKS_INPUT`` — symmetric
  sibling-validator input-shape guards.
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

from lib.validators.distractor_structural import (  # noqa: E402
    DEFAULT_MAX_SOURCE_OVERLAP,
    DistractorStructuralValidator,
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


_GROUNDED_CHUNK_TEXT = (
    "An RDF triple is the foundational unit of the resource description "
    "framework graph: it expresses a fact as a subject, predicate, and "
    "object statement. Each component identifies a node or property in "
    "the directed labelled graph that the framework constructs."
)

_UNRELATED_CHUNK_TEXT = (
    "Quantum chromodynamics describes the strong interaction between "
    "quarks via gluon exchange in particle accelerator experiments and "
    "lattice gauge simulations across the standard model regime."
)


def _choices_block(
    *,
    block_id: str = "page_01#assessment_item_q_0",
    choices: Optional[List[Dict[str, Any]]] = None,
    source_ids: tuple = ("semantik:rdf_intro#blk_0",),
    block_type: str = "assessment_item",
) -> Block:
    """Trainforge-shape ``choices[]`` assessment_item Block fixture."""
    if choices is None:
        choices = [
            {
                "text": (
                    "An RDF triple is a subject-predicate-object statement "
                    "expressing a fact in the graph."
                ),
                "is_correct": True,
            },
            {"text": "An XML namespace prefix mapping.", "is_correct": False},
            {"text": "A function-call signature in code.", "is_correct": False},
            {"text": "A regular-expression matching pattern.", "is_correct": False},
        ]
    content: Dict[str, Any] = {
        "stem": "What is an RDF triple?",
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
        source_ids=source_ids,
    )


def _legacy_split_block(
    *,
    block_id: str = "page_01#assessment_item_q_legacy",
    answer_key: str = (
        "An RDF triple is a subject-predicate-object statement expressing "
        "a fact in the graph."
    ),
    distractors: Optional[List[Dict[str, Any]]] = None,
    source_ids: tuple = ("semantik:rdf_intro#blk_0",),
) -> Block:
    """Legacy ``distractors[] + answer_key`` shape fixture."""
    if distractors is None:
        distractors = [
            {"text": "An XML namespace prefix mapping."},
            {"text": "A function-call signature in code."},
            {"text": "A regular-expression matching pattern."},
        ]
    content: Dict[str, Any] = {
        "stem": "What is an RDF triple?",
        "distractors": distractors,
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
        source_ids=source_ids,
    )


def _no_choices_block(
    *,
    block_id: str = "page_01#assessment_item_q_no_choices",
) -> Block:
    """Assessment block with no resolvable choices."""
    content: Dict[str, Any] = {
        "stem": "What is an RDF triple?",
        "content_type": "assessment_item",
    }
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=("TO-01",),
    )


def _non_assessment_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content={
            "stem": "An RDF triple is a subject-predicate-object statement.",
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


def _codes(result) -> List[str]:
    return [issue.code for issue in result.issues]


def _struct_events(capture: _RecordingCapture) -> List[Dict[str, Any]]:
    return [
        e for e in capture.events
        if e.get("decision_type") == "distractor_structural_check"
    ]


# --------------------------------------------------------------------- #
# Happy path — choices shape + legacy split shape
# --------------------------------------------------------------------- #


def test_happy_path_choices_shape_single_correct_no_entailment_passes():
    """Trainforge ``choices[]`` block with exactly one correct entry +
    distractors that don't overlap source chunks above the floor →
    passed=True, action=None, no critical issues."""
    blocks = [_choices_block()]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT}
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": blocks,
        "source_chunks": chunks_lookup,
    })

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None
    assert result.score == 1.0
    assert _codes(result) == []


def test_happy_path_legacy_split_shape_passes():
    """Legacy ``distractors[]`` + ``answer_key`` shape: validator
    reconstructs the implicit choices list and audits exactly one
    correct entry. Distractors topically distinct from the source chunk
    → passed=True."""
    blocks = [_legacy_split_block()]
    # Source chunk talks about RDF triples; the legacy distractors talk
    # about XML / function calls / regex — no overlap above the 0.70
    # floor.
    chunks_lookup = {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT}
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": blocks,
        "source_chunks": chunks_lookup,
    })

    assert result.passed is True, f"unexpected issues: {_codes(result)}"
    assert result.action is None


def test_non_assessment_blocks_silently_skipped():
    """Concept / example / chrome / recap blocks → vacuously passed."""
    blocks = [
        _non_assessment_block(block_id="page_01#concept_a", block_type="concept"),
        _non_assessment_block(block_id="page_01#example_b", block_type="example"),
    ]
    chunks_lookup = {"semantik:foo#blk_0": _GROUNDED_CHUNK_TEXT}
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": blocks,
        "source_chunks": chunks_lookup,
    })

    assert result.passed is True
    assert _codes(result) == []
    assert result.action is None


# --------------------------------------------------------------------- #
# Sub-check 1 — exactly-one-correct
# --------------------------------------------------------------------- #


def test_two_correct_answers_fails_multi_correct():
    """choices[] declaring 2 entries with is_correct=true → critical
    DISTRACTOR_STRUCTURAL_MULTI_CORRECT, action=regenerate."""
    block = _choices_block(choices=[
        {"text": "An RDF triple is a subject-predicate-object statement.", "is_correct": True},
        {"text": "An RDF triple expresses a fact in the graph.", "is_correct": True},
        {"text": "An XML namespace prefix mapping.", "is_correct": False},
        {"text": "A function-call signature.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
    })

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    assert "DISTRACTOR_STRUCTURAL_MULTI_CORRECT" in codes
    multi = [
        i for i in result.issues
        if i.code == "DISTRACTOR_STRUCTURAL_MULTI_CORRECT"
    ]
    assert all(i.severity == "critical" for i in multi)


def test_zero_correct_answers_fails_no_correct():
    """choices[] with all is_correct=false → critical
    DISTRACTOR_STRUCTURAL_NO_CORRECT, action=regenerate."""
    block = _choices_block(choices=[
        {"text": "Option A.", "is_correct": False},
        {"text": "Option B.", "is_correct": False},
        {"text": "Option C.", "is_correct": False},
        {"text": "Option D.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
    })

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    assert "DISTRACTOR_STRUCTURAL_NO_CORRECT" in codes


# --------------------------------------------------------------------- #
# Sub-check 2 — distractor-entailed-by-source
# --------------------------------------------------------------------- #


def test_distractor_entailed_by_source_fails_critical():
    """Distractor whose token-set Jaccard overlap against the cited
    source chunk is above the 0.70 floor → critical
    DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED, action=regenerate. The
    distractor is essentially the source claim verbatim, so it's a
    true alternative, not a misconception."""
    # Tight source chunk so a verbatim distractor crosses the 0.70
    # Jaccard floor. (Long source chunks dilute Jaccard fast — the
    # gate's calibration assumes per-block-attributed chunks of
    # comparable length to the distractor surface.)
    source_text = (
        "An RDF triple is a subject predicate object statement that "
        "expresses a fact."
    )
    entailing_distractor = (
        "An RDF triple is a subject predicate object statement."
    )
    block = _choices_block(choices=[
        {"text": "A graph claim about three nodes.", "is_correct": True},
        {"text": entailing_distractor, "is_correct": False},
        {"text": "An XML namespace prefix mapping.", "is_correct": False},
        {"text": "A function-call signature.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": source_text},
    })

    assert result.passed is False
    assert result.action == "regenerate"
    codes = _codes(result)
    assert "DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED" in codes
    entailed = [
        i for i in result.issues
        if i.code == "DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED"
    ]
    assert all(i.severity == "critical" for i in entailed)


def test_threshold_override_admits_high_overlap_distractor():
    """Per-call ``max_source_overlap=0.99`` admits a distractor that
    would fire at the default 0.70 floor — confirms inputs[] override
    is wired through."""
    source_text = (
        "An RDF triple is a subject predicate object statement that "
        "expresses a fact."
    )
    entailing_distractor = (
        "An RDF triple is a subject predicate object statement."
    )
    block = _choices_block(choices=[
        {"text": "A graph claim about three nodes.", "is_correct": True},
        {"text": entailing_distractor, "is_correct": False},
        {"text": "An XML namespace prefix mapping.", "is_correct": False},
        {"text": "A function-call signature.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": source_text},
        "max_source_overlap": 0.99,
    })

    # The relaxed floor doesn't fire ENTAILED on the distractor; with
    # exactly one correct and source-chunks present, the block passes
    # both sub-checks.
    assert "DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED" not in _codes(result)
    assert result.passed is True


# --------------------------------------------------------------------- #
# Skip / warning paths
# --------------------------------------------------------------------- #


def test_block_missing_choices_emits_warning_and_skips():
    """Block with neither ``choices[]`` nor a legacy fallback → warning
    DISTRACTOR_STRUCTURAL_NO_CHOICES; block skipped, validator passes
    with no critical issues."""
    block = _no_choices_block()
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
    })

    assert result.passed is True  # warning-only is non-blocking
    codes = _codes(result)
    assert "DISTRACTOR_STRUCTURAL_NO_CHOICES" in codes
    no_choices = [
        i for i in result.issues
        if i.code == "DISTRACTOR_STRUCTURAL_NO_CHOICES"
    ]
    assert all(i.severity == "warning" for i in no_choices)


def test_no_source_chunks_emits_warning_runs_exactly_one_correct():
    """``inputs['source_chunks']`` absent → entailment sub-check skips
    (warning DISTRACTOR_STRUCTURAL_NO_SOURCES) but the
    exactly-one-correct sub-check still runs and catches a 2-correct
    block as MULTI_CORRECT."""
    block = _choices_block(choices=[
        {"text": "Answer A.", "is_correct": True},
        {"text": "Answer B.", "is_correct": True},  # second correct
        {"text": "Wrong C.", "is_correct": False},
        {"text": "Wrong D.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()

    result = validator.validate({
        "blocks": [block],
        # no source_chunks — entailment sub-check skips
    })

    codes = _codes(result)
    # Warning surfaced once for the no-sources path.
    assert "DISTRACTOR_STRUCTURAL_NO_SOURCES" in codes
    no_sources = [
        i for i in result.issues
        if i.code == "DISTRACTOR_STRUCTURAL_NO_SOURCES"
    ]
    assert len(no_sources) == 1
    assert no_sources[0].severity == "warning"
    # exactly-one-correct sub-check still runs.
    assert "DISTRACTOR_STRUCTURAL_MULTI_CORRECT" in codes
    assert result.passed is False
    assert result.action == "regenerate"


def test_no_source_chunks_happy_path_passes_with_warning_only():
    """Source-chunks absent + a well-formed single-correct block →
    passes (warning DISTRACTOR_STRUCTURAL_NO_SOURCES is non-blocking)."""
    block = _choices_block()
    validator = DistractorStructuralValidator()

    result = validator.validate({"blocks": [block]})

    codes = _codes(result)
    assert "DISTRACTOR_STRUCTURAL_NO_SOURCES" in codes
    # No critical issues -> passes despite warning.
    assert result.passed is True


# --------------------------------------------------------------------- #
# Decision-emit regression
# --------------------------------------------------------------------- #


def test_capture_emits_one_distractor_structural_check_per_block():
    """A two-block batch emits exactly two
    ``distractor_structural_check`` events — one per audited block —
    with rationale interpolating block_id, choice_count, correct_count,
    threshold, and outcome. Mirrors the per-block decision-emit pattern
    of DistractorMisconceptionAlignmentValidator."""
    blocks = [
        _choices_block(block_id="page_01#assessment_item_q_0"),
        _choices_block(block_id="page_01#assessment_item_q_1"),
    ]
    chunks_lookup = {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT}
    validator = DistractorStructuralValidator()
    capture = _RecordingCapture()

    result = validator.validate({
        "blocks": blocks,
        "source_chunks": chunks_lookup,
        "decision_capture": capture,
    })

    assert result.passed is True
    events = _struct_events(capture)
    assert len(events) == 2, (
        f"expected exactly 2 distractor_structural_check events (one per "
        f"audited block); got {len(events)}: {events!r}"
    )
    for event in events:
        assert event["decision"] == "passed"
        # Rationale must interpolate dynamic signals so captures replay.
        rationale = event["rationale"]
        assert "choice_count=4" in rationale
        assert "correct_count=1" in rationale
        assert "threshold=" in rationale
        assert "max_source_overlap=" in rationale
        # Min-20-char audit floor (decision-capture contract).
        assert len(rationale) >= 20


def test_capture_emits_failed_event_on_multi_correct():
    """A multi-correct block emits a single failed-shaped event with
    decision='multi_correct' and correct_count=2 interpolated."""
    block = _choices_block(choices=[
        {"text": "Answer A.", "is_correct": True},
        {"text": "Answer B.", "is_correct": True},
        {"text": "Wrong C.", "is_correct": False},
        {"text": "Wrong D.", "is_correct": False},
    ])
    validator = DistractorStructuralValidator()
    capture = _RecordingCapture()

    result = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
        "decision_capture": capture,
    })

    assert result.passed is False
    events = _struct_events(capture)
    assert len(events) == 1
    event = events[0]
    assert event["decision"] == "multi_correct"
    assert "correct_count=2" in event["rationale"]


def test_capture_emits_no_choices_event():
    """A block missing choices emits a ``no_choices``-decision event."""
    block = _no_choices_block()
    validator = DistractorStructuralValidator()
    capture = _RecordingCapture()

    validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
        "decision_capture": capture,
    })

    events = _struct_events(capture)
    assert len(events) == 1
    assert events[0]["decision"] == "no_choices"


# --------------------------------------------------------------------- #
# Edge cases — input-shape guards
# --------------------------------------------------------------------- #


def test_missing_blocks_input_fails_loud():
    validator = DistractorStructuralValidator()
    result = validator.validate({})
    assert result.passed is False
    assert result.action == "regenerate"
    assert _codes(result) == ["MISSING_BLOCKS_INPUT"]


def test_invalid_blocks_input_fails_loud():
    validator = DistractorStructuralValidator()
    result = validator.validate({"blocks": "not-a-list"})
    assert result.passed is False
    assert result.action == "regenerate"
    assert _codes(result) == ["INVALID_BLOCKS_INPUT"]


def test_default_threshold_constant_exposed():
    """The DEFAULT_MAX_SOURCE_OVERLAP constant is part of the module's
    public surface; downstream callers (and the workflows.yaml gate
    config) reference it for threshold tuning."""
    assert isinstance(DEFAULT_MAX_SOURCE_OVERLAP, float)
    assert 0.0 < DEFAULT_MAX_SOURCE_OVERLAP <= 1.0


def test_chunks_lookup_alias_resolves_same_as_source_chunks():
    """``inputs['chunks_lookup']`` is honoured as an alias for
    ``inputs['source_chunks']`` so calls from the assessment-grounding
    builder share inputs with this validator without re-keying."""
    block = _choices_block()
    validator = DistractorStructuralValidator()

    result_via_alias = validator.validate({
        "blocks": [block],
        "chunks_lookup": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
    })
    result_via_canonical = validator.validate({
        "blocks": [block],
        "source_chunks": {"semantik:rdf_intro#blk_0": _GROUNDED_CHUNK_TEXT},
    })

    assert result_via_alias.passed == result_via_canonical.passed
    assert _codes(result_via_alias) == _codes(result_via_canonical)
