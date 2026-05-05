"""H3 Wave W1 regression: per-block decision-capture wiring on the
four ``Block*Validator``s in ``Courseforge.router.inter_tier_gates``.

Plan: ``plans/h3-validator-capture-wiring-2026-05.md`` § "W1.
Block-input validators (the four ``Block*Validator``s)".

Pattern A (from ``lib/validators/rewrite_source_grounding.py:268-311``)
borrowed verbatim into ``inter_tier_gates._emit_block_decision``. The
S0.5 runtime-injection commit (``8914fce``) populates both
``inputs["decision_capture"]`` and ``inputs["capture"]`` at gate-runner
time; this suite exercises the canonical key (``decision_capture``)
and asserts the four W1 validators each emit exactly one Pattern A
event per validate() call per audited Block.

Cardinality contract per W1 acceptance:
    1 emit/block × 3 blocks × 4 validators = 12 captures per parametrized run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Block lives at Courseforge/scripts/blocks.py — mirror the import
# bridge used by the rest of the router test suite.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
)


# --------------------------------------------------------------------------- #
# MockCapture — minimal DecisionCapture seam
# --------------------------------------------------------------------------- #


class _MockCapture:
    """Minimal DecisionCapture stand-in. Records every log_decision
    call as a ``(decision_type, decision, rationale)`` tuple."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **_kw: Any,
    ) -> None:
        self.calls.append((decision_type, decision, rationale))


# --------------------------------------------------------------------------- #
# Block fixture builders — mirror test_inter_tier_gates._outline_block
# --------------------------------------------------------------------------- #


def _outline_block(
    *,
    block_id: str,
    curies: Tuple[str, ...] = ("ed4all:Foo",),
    key_claims: Optional[List[str]] = None,
    content_type: str = "definition",
    objective_ids: Tuple[str, ...] = ("TO-01",),
    source_ids: Tuple[str, ...] = (),
) -> Block:
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": (
                list(key_claims)
                if key_claims is not None
                else ["The ed4all:Foo predicate marks anchoring."]
            ),
            "content_type": content_type,
        },
        objective_ids=tuple(objective_ids),
        source_ids=tuple(source_ids),
    )


def _three_block_fixture_for_curie() -> Tuple[List[Block], Dict[str, Any]]:
    """Three-block input for the curie-anchoring validator."""
    blocks = [
        _outline_block(
            block_id="page_01#concept_a_0",
            curies=("ed4all:Foo",),
            key_claims=["The ed4all:Foo predicate marks anchoring."],
        ),
        _outline_block(
            block_id="page_01#concept_b_1",
            curies=(),  # empty → MISSING_CURIES branch
            key_claims=["No curies declared at all."],
        ),
        _outline_block(
            block_id="page_01#concept_c_2",
            curies=("ed4all:Bar",),
            key_claims=["This text mentions nothing matching."],
        ),
    ]
    return blocks, {"blocks": blocks}


def _three_block_fixture_for_content_type() -> Tuple[List[Block], Dict[str, Any]]:
    blocks = [
        _outline_block(block_id="page_01#a_0", content_type="definition"),
        _outline_block(block_id="page_01#b_1", content_type=""),  # MISSING
        _outline_block(block_id="page_01#c_2", content_type="not_a_real_type"),  # INVALID
    ]
    return blocks, {"blocks": blocks}


def _three_block_fixture_for_objectives() -> Tuple[List[Block], Dict[str, Any]]:
    blocks = [
        _outline_block(
            block_id="page_01#a_0",
            objective_ids=("TO-01",),
        ),
        _outline_block(
            block_id="page_01#b_1",
            objective_ids=(),  # MISSING_OBJECTIVE_REF
        ),
        _outline_block(
            block_id="page_01#c_2",
            objective_ids=("TO-99",),  # UNKNOWN_OBJECTIVE under {TO-01}
        ),
    ]
    inputs = {
        "blocks": blocks,
        "valid_objective_ids": {"TO-01"},
    }
    return blocks, inputs


def _three_block_fixture_for_source_refs() -> Tuple[List[Block], Dict[str, Any]]:
    blocks = [
        _outline_block(
            block_id="page_01#a_0",
            source_ids=("dart:my-textbook#blk_42",),
        ),
        _outline_block(
            block_id="page_01#b_1",
            source_ids=(),  # No source_ids — passes structural check
        ),
        _outline_block(
            block_id="page_01#c_2",
            source_ids=("not-a-canonical-id",),  # INVALID_SOURCE_ID_SHAPE
        ),
    ]
    inputs = {
        "blocks": blocks,
        "valid_source_ids": {"dart:my-textbook#blk_42"},
    }
    return blocks, inputs


# --------------------------------------------------------------------------- #
# Parametrization — one row per validator class
# --------------------------------------------------------------------------- #


_W1_VALIDATOR_PARAMS = [
    pytest.param(
        BlockCurieAnchoringValidator,
        _three_block_fixture_for_curie,
        "block_curie_anchoring_check",
        id="curie_anchoring",
    ),
    pytest.param(
        BlockContentTypeValidator,
        _three_block_fixture_for_content_type,
        "block_content_type_check",
        id="content_type",
    ),
    pytest.param(
        BlockPageObjectivesValidator,
        _three_block_fixture_for_objectives,
        "block_page_objectives_check",
        id="page_objectives",
    ),
    pytest.param(
        BlockSourceRefValidator,
        _three_block_fixture_for_source_refs,
        "block_source_ref_check",
        id="source_refs",
    ),
]


@pytest.mark.parametrize(
    "validator_cls,fixture_factory,expected_decision_type", _W1_VALIDATOR_PARAMS
)
def test_block_validators_emit_decision_capture(
    validator_cls,
    fixture_factory,
    expected_decision_type,
):
    """Per-block emit cardinality: every block in inputs['blocks']
    triggers exactly one decision-capture event with the validator's
    expected decision_type. Three blocks → three captures."""
    blocks, inputs = fixture_factory()
    capture = _MockCapture()
    inputs_with_capture = dict(inputs)
    inputs_with_capture["decision_capture"] = capture

    validator_cls().validate(inputs_with_capture)

    # Exactly one event per audited block.
    assert len(capture.calls) == len(blocks), (
        f"{validator_cls.__name__} emitted {len(capture.calls)} "
        f"decision-capture events; expected {len(blocks)} (one per "
        f"audited Block)."
    )
    # Every event carries the validator's expected decision_type.
    for decision_type, decision, rationale in capture.calls:
        assert decision_type == expected_decision_type, (
            f"{validator_cls.__name__} emitted decision_type="
            f"{decision_type!r}; expected {expected_decision_type!r}."
        )
        # Pattern A rationale floor — pins against static / boilerplate
        # rationales (root CLAUDE.md "Decision rationale" contract:
        # ≥20 chars; H3 plan §3 W1 regression test asserts ≥60).
        assert len(rationale) >= 60, (
            f"{validator_cls.__name__} rationale too short ({len(rationale)} "
            f"chars); rationale must interpolate dynamic per-block signals."
        )
        # Decision is either 'passed' or starts with 'failed:<code>'.
        assert decision == "passed" or decision.startswith("failed:"), (
            f"{validator_cls.__name__} emitted decision={decision!r}; "
            f"expected 'passed' or 'failed:<code>'."
        )

    # Per-block block_id appears in each rationale (sanity-check
    # that the emit really walks the per-block axis, not just emits
    # one canned event N times).
    block_ids_seen = {
        block.block_id
        for block in blocks
        for (_, _, rat) in capture.calls
        if block.block_id in rat
    }
    assert block_ids_seen == {b.block_id for b in blocks}


@pytest.mark.parametrize(
    "validator_cls,fixture_factory,expected_decision_type", _W1_VALIDATOR_PARAMS
)
def test_block_validators_no_capture_no_emit_no_crash(
    validator_cls,
    fixture_factory,
    expected_decision_type,
):
    """When inputs['decision_capture'] is absent, validate() must not
    crash and must produce a GateResult identical to the captured
    counterpart (capture wiring is purely additive)."""
    _, inputs = fixture_factory()

    # Run with capture and without; assert GateResult fields agree.
    capture = _MockCapture()
    inputs_with = dict(inputs)
    inputs_with["decision_capture"] = capture
    result_with = validator_cls().validate(inputs_with)
    result_without = validator_cls().validate(dict(inputs))

    assert result_with.passed == result_without.passed
    assert result_with.action == result_without.action
    assert result_with.score == result_without.score
    assert len(result_with.issues) == len(result_without.issues)


@pytest.mark.parametrize(
    "validator_cls,fixture_factory,expected_decision_type", _W1_VALIDATOR_PARAMS
)
def test_block_validators_capture_alias_key_honoured(
    validator_cls,
    fixture_factory,
    expected_decision_type,
):
    """S0.5 runtime injection populates both ``decision_capture`` and
    ``capture`` keys; the canonical key takes precedence, and the
    alias works when the canonical key is absent."""
    _, inputs = fixture_factory()

    # Alias-only input: pass capture under inputs['capture'].
    capture = _MockCapture()
    inputs_alias = dict(inputs)
    inputs_alias["capture"] = capture
    validator_cls().validate(inputs_alias)
    assert len(capture.calls) > 0, (
        f"{validator_cls.__name__} did not honour inputs['capture'] alias."
    )


@pytest.mark.parametrize(
    "validator_cls,fixture_factory,expected_decision_type", _W1_VALIDATOR_PARAMS
)
def test_block_validators_capture_failure_does_not_propagate(
    validator_cls,
    fixture_factory,
    expected_decision_type,
):
    """A capture instance whose log_decision() raises must not abort
    the validate() walk — the gate is the source of truth, not the
    audit trail."""

    class _RaisingCapture:
        def log_decision(self, **_kw: Any) -> None:
            raise RuntimeError("simulated capture-side failure")

    _, inputs = fixture_factory()
    inputs_with_failing = dict(inputs)
    inputs_with_failing["decision_capture"] = _RaisingCapture()

    # Must not raise.
    result = validator_cls().validate(inputs_with_failing)
    assert result is not None
    assert hasattr(result, "passed")
