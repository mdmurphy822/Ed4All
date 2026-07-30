"""Bloom-ladder initiative (WI-17) — per-Bloom entailment floors on the
W1.7.C ``BlockObjectiveDeliveryValidator`` machinery.

Covers the per-(block_type, bloom_rung) entailment-floor extension over
``_PER_BLOCK_TYPE_ENTAILMENT_FLOOR``: resolved ONLY when the audited
block carries a non-empty ``Block.ladder_rung`` (WI-06's
planner-stamped rung provenance — only ever set under
``ED4ALL_BLOOM_LADDER``); a legacy block (``ladder_rung is None``, the
byte-identical default) always hits the pre-existing per-block-type
floor.

The NLI loader is stubbed via direct injection (``nli=...``) exactly as
in ``test_block_objective_delivery.py`` — no GPU / model download
involved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.slow

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Block lives at Courseforge/scripts/blocks.py — import bridge mirror.
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.block_objective_delivery import (  # noqa: E402
    BlockObjectiveDeliveryValidator,
    _CODE_STATEMENT_UNDERSUPPORTED,
    _PER_BLOCK_TYPE_BLOOM_RUNG_ENTAILMENT_FLOOR,
    _PER_BLOCK_TYPE_ENTAILMENT_FLOOR,
)


# --------------------------------------------------------------------- #
# Stubs / fixtures — mirrors test_block_objective_delivery.py
# --------------------------------------------------------------------- #


class _StubNli:
    """Drop-in test stub for ``NliClassifier``. See sibling test module
    for the full docstring; behavior is identical here."""

    def __init__(
        self,
        response_map: Optional[Dict[Tuple[str, str], NliScore]] = None,
        default_score: Optional[NliScore] = None,
    ) -> None:
        self.response_map: Dict[Tuple[str, str], NliScore] = (
            response_map or {}
        )
        self.default_score = default_score or NliScore(
            entailment=0.85, neutral=0.10, contradiction=0.05,
        )
        self.calls: List[Tuple[str, str]] = []

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        self.calls.append((premise, hypothesis))
        return self.response_map.get(
            (premise, hypothesis), self.default_score,
        )


def _make_block(
    *,
    block_id: str = "page_01#assessment_item_x_0",
    block_type: str = "assessment_item",
    page_id: str = "page_01",
    sequence: int = 0,
    statement: str = "We construct the federation trust anchor here.",
    objective_ids: Tuple[str, ...] = ("CO-08",),
    observed_bloom_level: Optional[str] = "create",
    ladder_rung: Optional[str] = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content={"statement": statement},
        objective_ids=objective_ids,
        observed_bloom_level=observed_bloom_level,
        ladder_rung=ladder_rung,
    )


def _objectives_for(
    *,
    bloom_level: str = "create",
    bloom_verb: str = "construct",
    statement: str = "Construct subclass and subproperty hierarchies in Turtle.",
    obj_id: str = "CO-08",
) -> Dict[str, Dict[str, Any]]:
    return {
        obj_id: {
            "id": obj_id,
            "bloom_level": bloom_level,
            "bloom_verb": bloom_verb,
            "statement": statement,
        }
    }


def _run(
    validator: BlockObjectiveDeliveryValidator,
    block: Block,
    objectives: Dict[str, Dict[str, Any]],
    *,
    extra_inputs: Optional[Dict[str, Any]] = None,
    decision_capture: Any = None,
):
    inputs: Dict[str, Any] = {
        "blocks": [block],
        "objectives": objectives,
        "objective_statements": {
            k: v["statement"] for k, v in objectives.items()
        },
    }
    if decision_capture is not None:
        inputs["decision_capture"] = decision_capture
    if extra_inputs:
        inputs.update(extra_inputs)
    return validator.validate(inputs)


# --------------------------------------------------------------------- #
# Sanity — the new table's create-rung floors are actually looser than
# the base per-block-type table (the property the rest of this module
# exercises through the validator).
# --------------------------------------------------------------------- #


def test_bloom_rung_table_create_floors_looser_than_base_table() -> None:
    assert (
        _PER_BLOCK_TYPE_BLOOM_RUNG_ENTAILMENT_FLOOR[("assessment_item", "create")]
        < _PER_BLOCK_TYPE_ENTAILMENT_FLOOR["assessment_item"]
    )
    assert (
        _PER_BLOCK_TYPE_BLOOM_RUNG_ENTAILMENT_FLOOR[("activity", "create")]
        < _PER_BLOCK_TYPE_ENTAILMENT_FLOOR["activity"]
    )


# --------------------------------------------------------------------- #
# Test 1: legacy block (ladder_rung=None) hits the per-type floor
# unchanged — the byte-identical-when-flags-unset contract.
# --------------------------------------------------------------------- #


def test_legacy_block_no_ladder_rung_uses_per_type_floor_unchanged() -> None:
    """entailment=0.50 is BELOW the assessment_item per-type floor
    (0.55) — legacy (no ladder_rung) block fires UNDERSUPPORTED."""
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.50, neutral=0.0, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung=None, statement=statement)
    objectives = _objectives_for(statement=hypothesis)

    result = _run(validator, block, objectives)

    assert any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    ), "expected the base 0.55 assessment_item floor to fire at 0.50"


# --------------------------------------------------------------------- #
# Test 2: ladder-authored block (ladder_rung="create") resolves the
# LOOSER per-(block_type, bloom_rung) floor (0.45) instead — same NLI
# score now passes.
# --------------------------------------------------------------------- #


def test_ladder_authored_block_uses_looser_bloom_rung_floor() -> None:
    """Same entailment=0.50 score — ladder_rung='create' resolves the
    0.45 floor, so the SAME miss the legacy path fires now passes."""
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.50, neutral=0.0, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung="create", statement=statement)
    objectives = _objectives_for(statement=hypothesis)

    result = _run(validator, block, objectives)

    assert not any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    ), "expected the looser 0.45 create-rung floor to admit 0.50"


# --------------------------------------------------------------------- #
# Test 3: (block_type, bloom_rung) combo absent from the ladder table
# falls back to the per-block-type floor — NOT the global default.
# --------------------------------------------------------------------- #


def test_unmatched_bloom_rung_combo_falls_back_to_per_type_floor() -> None:
    """``concept`` is only ever tabled at 'understand'; ladder_rung=
    'create' has no (concept, create) entry, so the floor falls back to
    the concept per-type floor (0.30) — NOT the 0.40 global default.
    entailment=0.35 sits between the two, so the assertion discriminates
    a correct fallback from an accidental fall-through to the default.
    """
    statement = "Concept prose about federated trust."
    hypothesis = "Explain the concept of federated trust."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.35, neutral=0.05, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(
        block_type="concept",
        ladder_rung="create",  # no (concept, create) table entry
        statement=statement,
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="explain", statement=hypothesis,
    )

    result = _run(validator, block, objectives)

    assert _PER_BLOCK_TYPE_ENTAILMENT_FLOOR["concept"] == 0.30
    assert not any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    ), "0.35 should clear the 0.30 per-type fallback floor"


# --------------------------------------------------------------------- #
# Test 4: a garbage / non-canonical ladder_rung value degrades gracefully
# to the per-block-type floor rather than raising.
# --------------------------------------------------------------------- #


def test_garbage_ladder_rung_falls_back_to_per_type_floor_without_raising() -> None:
    statement = "Concept prose about federated trust."
    hypothesis = "Explain the concept of federated trust."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.35, neutral=0.05, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(
        block_type="concept",
        ladder_rung="not_a_bloom_level",
        statement=statement,
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="explain", statement=hypothesis,
    )

    result = _run(validator, block, objectives)  # must not raise

    assert not any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    ), "garbage ladder_rung should fall back to the 0.30 per-type floor"


# --------------------------------------------------------------------- #
# Test 5: gate.config override merge hook —
# ``per_block_type_bloom_rung_entailment_floor`` (nested mapping) tightens
# a specific (block_type, bloom_rung) pair.
# --------------------------------------------------------------------- #


def test_bloom_rung_threshold_override_merge_hook() -> None:
    """A gate.config override for (assessment_item, create) tightens the
    default 0.45 floor to 0.95 — the same 0.50 entailment that PASSED
    under the default table now fires UNDERSUPPORTED."""
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.50, neutral=0.0, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung="create", statement=statement)
    objectives = _objectives_for(statement=hypothesis)

    result = _run(
        validator, block, objectives,
        extra_inputs={
            "per_block_type_bloom_rung_entailment_floor": {
                "assessment_item": {"create": 0.95}
            }
        },
    )

    assert any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    ), "tight 0.95 override should fire UNDERSUPPORTED at 0.50"


# --------------------------------------------------------------------- #
# Test 6: severity stays warning day-1 — the new floors do not promote
# to critical (calibration-gated flip is a FUTURE step per
# lib.governance.calibration_gate.resolve_severity_flip).
# --------------------------------------------------------------------- #


def test_bloom_rung_floor_miss_stays_warning_severity() -> None:
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.10, neutral=0.0, contradiction=0.80,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung="create", statement=statement)
    objectives = _objectives_for(statement=hypothesis)

    result = _run(validator, block, objectives)

    miss_issues = [
        i for i in result.issues if i.code == _CODE_STATEMENT_UNDERSUPPORTED
    ]
    assert miss_issues, "expected the 0.45 create-rung floor to fire at 0.10"
    assert all(i.severity == "warning" for i in miss_issues)
    assert result.passed is True


# --------------------------------------------------------------------- #
# Test 7: decision capture fires with the ladder_rung interpolated for a
# ladder-authored block, and stays absent for a legacy block — dynamic,
# per-call rationale (repo decision-capture contract).
# --------------------------------------------------------------------- #


class _CaptureStub:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **_: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


def test_decision_capture_interpolates_ladder_rung_when_set() -> None:
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.50, neutral=0.0, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung="create", statement=statement)
    objectives = _objectives_for(statement=hypothesis)
    capture = _CaptureStub()

    _run(validator, block, objectives, decision_capture=capture)

    delivery_events = [
        e for e in capture.events
        if e["decision_type"] == "block_objective_delivery_check"
    ]
    assert len(delivery_events) == 1
    assert "ladder_rung='create'" in delivery_events[0]["rationale"]
    # entailment_threshold in the rationale reflects the LOOSER
    # bloom-rung floor (0.45), not the base per-type floor (0.55).
    assert "entailment_threshold=0.4500" in delivery_events[0]["rationale"]


def test_decision_capture_omits_ladder_rung_for_legacy_block() -> None:
    statement = "We construct the federation trust anchor here."
    hypothesis = "Construct subclass and subproperty hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.50, neutral=0.0, contradiction=0.60,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)
    block = _make_block(ladder_rung=None, statement=statement)
    objectives = _objectives_for(statement=hypothesis)
    capture = _CaptureStub()

    _run(validator, block, objectives, decision_capture=capture)

    delivery_events = [
        e for e in capture.events
        if e["decision_type"] == "block_objective_delivery_check"
    ]
    assert len(delivery_events) == 1
    assert "ladder_rung=" not in delivery_events[0]["rationale"]
    # entailment_threshold in the rationale reflects the base per-type
    # floor (0.55) — unchanged from the pre-WI-17 validator.
    assert "entailment_threshold=0.5500" in delivery_events[0]["rationale"]
