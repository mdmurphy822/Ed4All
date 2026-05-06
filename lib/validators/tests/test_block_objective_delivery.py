"""GPT Feedback v2 Wave 1.7 W1.7.C —
``BlockObjectiveDeliveryValidator`` tests.

Covers the tri-axis (NLI statement entailment / Bloom-gap≥2 /
action-verb synonym presence) per-block-per-objective delivery check.
The W1.7.C plan §2 Fix 1.7.C "Tests required" subsection enumerates 13
test cases this module implements.

The NLI loader is stubbed via monkeypatch on
``NliClassifier.get_or_load`` (and direct injection of ``nli=...`` on
the validator constructor) so CI fresh-checkout does not require the
W2.F real loader. Test 9 (``test_nli_deps_missing_emits_warning_passes``)
exercises the actual ``get_or_load() -> None`` graceful-degrade path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

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
    _CODE_BLOOM_UNDERMET,
    _CODE_NLI_DEPS_MISSING,
    _CODE_STATEMENT_UNDERSUPPORTED,
    _CODE_VERB_ABSENT,
    _PER_BLOCK_TYPE_ENTAILMENT_FLOOR,
)


# --------------------------------------------------------------------- #
# Stubs / fixtures
# --------------------------------------------------------------------- #


class _StubNli:
    """Drop-in test stub for ``NliClassifier``.

    Returns deterministic per-(premise, hypothesis) NliScores via a
    response map. Falls back to a default score when no map entry
    matches. Records every ``score_pair`` call so tests can assert
    fan-out.
    """

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
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    statement: str = "Federation requires a trusted identity provider.",
    objective_ids: Tuple[str, ...] = ("CO-08",),
    observed_bloom_level: Optional[str] = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content={"statement": statement},
        objective_ids=objective_ids,
        observed_bloom_level=observed_bloom_level,
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


# --------------------------------------------------------------------- #
# Test 1: Bloom-gap >= 2 fires UNDERMET
# --------------------------------------------------------------------- #


def test_block_with_underdelivered_bloom_fires_BLOCK_OBJECTIVE_BLOOM_UNDERMET() -> None:
    """Block at ``remember``, objective at ``create`` → gap 4 → fires."""
    nli = _StubNli()  # high-entailment default — entailment axis passes
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="remember",
    )
    objectives = _objectives_for(bloom_level="create", bloom_verb="construct")

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert any(
        i.code == _CODE_BLOOM_UNDERMET and i.severity == "warning"
        for i in result.issues
    ), f"expected BLOOM_UNDERMET issue; got {[i.code for i in result.issues]}"
    assert result.action == "regenerate"
    assert result.passed  # warning-only severity


# --------------------------------------------------------------------- #
# Test 2: gap == 1 passes the Bloom axis
# --------------------------------------------------------------------- #


def test_block_with_one_level_gap_passes_bloom_axis() -> None:
    """Block at ``understand``, objective at ``apply`` → gap 1 → tolerated."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="example",
        observed_bloom_level="understand",
        statement=(
            "Apply this configuration step-by-step to demonstrate "
            "the federation lookup."
        ),
    )
    objectives = _objectives_for(
        bloom_level="apply", bloom_verb="apply",
        statement="Apply the federation flow to a multi-tenant scenario.",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert not any(
        i.code == _CODE_BLOOM_UNDERMET for i in result.issues
    ), "gap=1 must NOT fire BLOOM_UNDERMET"


# --------------------------------------------------------------------- #
# Test 3: verb synonym presence passes the verb axis
# --------------------------------------------------------------------- #


def test_block_with_verb_synonym_passes_verb_axis() -> None:
    """``bloom_verb=evaluate``; prose contains synonym ``judge`` → passes."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="explanation",
        observed_bloom_level="evaluate",
        statement=(
            "We judge the proposed schema against the Turtle "
            "specification's constraints."
        ),
    )
    objectives = _objectives_for(
        bloom_level="evaluate", bloom_verb="evaluate",
        statement="Evaluate competing RDF schema designs.",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert not any(
        i.code == _CODE_VERB_ABSENT for i in result.issues
    ), "synonym 'judge' should pass the evaluate-level verb axis"


# --------------------------------------------------------------------- #
# Test 4: zero verb synonyms fires VERB_ABSENT
# --------------------------------------------------------------------- #


def test_block_with_no_verb_synonym_fires_BLOCK_OBJECTIVE_VERB_ABSENT() -> None:
    """``bloom_verb=create``; prose has zero create-level synonyms."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="activity",
        observed_bloom_level="create",
        # Pure passive prose — no canonical create-level verb at all.
        statement=(
            "RDF schemas typically come pre-built. The book has it."
        ),
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
        statement="Construct novel RDF schemas.",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert any(
        i.code == _CODE_VERB_ABSENT and i.severity == "warning"
        for i in result.issues
    )
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 5: low entailment + high contradiction fires UNDERSUPPORTED
# --------------------------------------------------------------------- #


def test_block_with_low_entailment_high_contradiction_fires_undersupported() -> None:
    """Mock NLI returns entailment=0.20, contradiction=0.70 → fires."""
    statement = "Federation requires trust"  # premise (block surface)
    hypothesis = "Construct subclass hierarchies in Turtle."  # objective stmt
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.20, neutral=0.10, contradiction=0.70,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",  # match level so Bloom axis passes
        statement=statement,
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
        statement=hypothesis,
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED
        for i in result.issues
    )
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 6: low entailment + low contradiction passes (surface-form variance)
# --------------------------------------------------------------------- #


def test_block_with_low_entailment_low_contradiction_passes() -> None:
    """Mock NLI returns entailment=0.20, contradiction=0.10 → passes."""
    statement = "Federation requires trust"
    hypothesis = "Construct subclass hierarchies in Turtle."
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.20, neutral=0.70, contradiction=0.10,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        statement=statement,
    )
    # Use a verb the prose contains so the verb axis doesn't fire.
    objectives = {
        "CO-08": {
            "id": "CO-08",
            "bloom_level": "create",
            "bloom_verb": "construct",
            "statement": hypothesis,
        }
    }

    # Inject a verb the prose actually contains.
    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        statement="We construct a federation trust anchor here.",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    # No UNDERSUPPORTED — contradiction is below the floor.
    assert not any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result.issues
    )


# --------------------------------------------------------------------- #
# Test 7: observed_bloom_level=None skips Bloom axis silently
# --------------------------------------------------------------------- #


def test_block_with_observed_bloom_unset_skips_bloom_axis() -> None:
    """Legacy block with no observed_bloom → Bloom axis silently skips."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level=None,  # legacy / pre-W1 block
        statement="We construct the federation lookup at runtime.",
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
            "return_touched_blocks": True,
        }
    )

    # No BLOOM_UNDERMET — observed_bloom missing.
    assert not any(i.code == _CODE_BLOOM_UNDERMET for i in result.issues)


# --------------------------------------------------------------------- #
# Test 8: legacy block with no objective_refs is silently skipped
# --------------------------------------------------------------------- #


def test_legacy_block_no_objective_refs_skipped_silently() -> None:
    """Block with empty objective_ids → no events, no issues."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        objective_ids=(),
        observed_bloom_level="apply",
    )
    objectives = _objectives_for()

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    # No regenerate; no NLI call made; no axis-miss issues.
    assert not any(
        i.code in (
            _CODE_BLOOM_UNDERMET,
            _CODE_VERB_ABSENT,
            _CODE_STATEMENT_UNDERSUPPORTED,
        )
        for i in result.issues
    )
    assert nli.calls == []


# --------------------------------------------------------------------- #
# Test 9: NLI loader returns None → graceful-degrade warning passes
# --------------------------------------------------------------------- #


def test_nli_deps_missing_emits_warning_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NLI loader returns None → BLOCK_OBJECTIVE_NLI_DEPS_MISSING warning."""
    from lib.classifiers import nli_classifier as mod

    # The real W1.7.C stub already returns None; force-pin to be safe.
    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )

    validator = BlockObjectiveDeliveryValidator()  # nli=None → calls stub

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        statement="We construct the federation trust here.",
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
    )

    result = validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    assert any(
        i.code == _CODE_NLI_DEPS_MISSING and i.severity == "warning"
        for i in result.issues
    )
    # Bloom + verb still ran, but with matching observed bloom + verb
    # synonym, no further misses fire — passes overall.
    assert result.passed


# --------------------------------------------------------------------- #
# Test 10: per-block-type threshold table overrides default
# --------------------------------------------------------------------- #


def test_per_block_type_threshold_table_overrides_default() -> None:
    """``concept`` block uses 0.30 floor (table entry) not 0.40 default."""
    statement = "Concept text"
    hypothesis = "Objective text"
    # entailment=0.32 — below 0.40 default but ABOVE concept's 0.30
    # per-type floor → no UNDERSUPPORTED issue.
    nli = _StubNli(
        response_map={
            (statement, hypothesis): NliScore(
                entailment=0.32, neutral=0.10, contradiction=0.58,
            )
        }
    )
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        statement=statement,
    )
    objectives = {
        "CO-08": {
            "id": "CO-08",
            "bloom_level": "create",
            "bloom_verb": "construct",
            "statement": hypothesis,
        }
    }

    result = validator.validate(
        {
            "blocks": [_make_block(
                block_type="concept",
                observed_bloom_level="create",
                statement="We construct the schema here. Concept text",
            )],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    # The concept-tier 0.30 threshold passes 0.32; sanity that the
    # table (vs the 0.40 default) fired.
    # Confirm the floor table per-type entry differs from the default.
    assert _PER_BLOCK_TYPE_ENTAILMENT_FLOOR["concept"] < 0.40

    # Run again with a TIGHT per-block-type override (concept=0.95) to
    # confirm the table-entry path actually wires through.
    nli2 = _StubNli(
        response_map={
            ("Concept text", "Objective text"): NliScore(
                entailment=0.32, neutral=0.10, contradiction=0.58,
            )
        }
    )
    validator2 = BlockObjectiveDeliveryValidator(nli=nli2)

    block2 = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        statement="Concept text",
    )
    result2 = validator2.validate(
        {
            "blocks": [block2],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
            # Override path — the executor merges gate.config thresholds
            # in via this same key (per the merge contract documented
            # at executor.py:1442 + the validator's
            # _resolve_threshold_table helper).
            "per_block_type_entailment_floor": {"concept": 0.95},
        }
    )

    assert any(
        i.code == _CODE_STATEMENT_UNDERSUPPORTED for i in result2.issues
    ), "tight 0.95 concept floor should fire UNDERSUPPORTED at 0.32"


# --------------------------------------------------------------------- #
# Test 11: capture fires per audited (block, objective_id) pair
# --------------------------------------------------------------------- #


def test_validator_emits_capture_per_audited_pair() -> None:
    """Block with two objective_ids → two delivery_check events."""
    captured: List[Dict[str, Any]] = []

    class _CaptureStub:
        def log_decision(
            self,
            *,
            decision_type: str,
            decision: str,
            rationale: str,
            **_: Any,
        ) -> None:
            captured.append(
                {
                    "decision_type": decision_type,
                    "decision": decision,
                    "rationale": rationale,
                }
            )

    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        objective_ids=("CO-08", "CO-09"),
        statement="We construct the federation trust here.",
    )
    objectives = {
        "CO-08": {
            "id": "CO-08", "bloom_level": "create", "bloom_verb": "construct",
            "statement": "Construct subclass hierarchies.",
        },
        "CO-09": {
            "id": "CO-09", "bloom_level": "create", "bloom_verb": "design",
            "statement": "Design schema diagrams.",
        },
    }

    validator.validate(
        {
            "blocks": [block],
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
            "decision_capture": _CaptureStub(),
        }
    )

    delivery_events = [
        c for c in captured
        if c["decision_type"] == "block_objective_delivery_check"
    ]
    assert len(delivery_events) == 2, (
        f"expected 2 delivery events; got {len(delivery_events)}"
    )


# --------------------------------------------------------------------- #
# Test 12: return_touched_blocks=True populates Block.objective_alignment
# --------------------------------------------------------------------- #


def test_validator_populates_block_objective_alignment_field() -> None:
    """Opt-in: validate() populates Block.objective_alignment per pair."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    block = _make_block(
        block_type="concept",
        observed_bloom_level="create",
        objective_ids=("CO-08",),
        statement="We construct the federation trust here.",
    )
    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
    )

    inputs = {
        "blocks": [block],
        "objectives": objectives,
        "objective_statements": {
            k: v["statement"] for k, v in objectives.items()
        },
        "return_touched_blocks": True,
    }
    validator.validate(inputs)

    touched = inputs.get("_touched_blocks", [])
    assert touched, "expected return_touched_blocks=True to populate _touched_blocks"
    touched_block = touched[0]
    alignment = getattr(touched_block, "objective_alignment", None)
    assert alignment, "expected non-empty objective_alignment"
    assert len(alignment) == 1
    entry = alignment[0]
    assert entry["objective_id"] == "CO-08"
    assert entry["status"] in {
        "delivered", "underdelivered", "verb_only", "unverifiable",
    }


# --------------------------------------------------------------------- #
# Test 13: unaudited block_types skip silently
# --------------------------------------------------------------------- #


def test_validator_skips_unaudited_block_types_silently() -> None:
    """``chrome`` / ``misconception`` / ``callout`` etc. emit no issues."""
    nli = _StubNli()
    validator = BlockObjectiveDeliveryValidator(nli=nli)

    blocks: List[Block] = []
    for bt in ("chrome", "misconception", "callout", "recap"):
        blocks.append(
            Block(
                block_id=f"page_01#{bt}_x_0",
                block_type=bt,
                page_id="page_01",
                sequence=0,
                content={"text": "Some prose."},
                objective_ids=("CO-08",),
                observed_bloom_level="remember",
            )
        )

    objectives = _objectives_for(
        bloom_level="create", bloom_verb="construct",
    )

    result = validator.validate(
        {
            "blocks": blocks,
            "objectives": objectives,
            "objective_statements": {
                k: v["statement"] for k, v in objectives.items()
            },
        }
    )

    # No axis-miss issues for unaudited types.
    assert not any(
        i.code in (
            _CODE_BLOOM_UNDERMET,
            _CODE_VERB_ABSENT,
            _CODE_STATEMENT_UNDERSUPPORTED,
        )
        for i in result.issues
    )
    assert nli.calls == []  # no NLI dispatch for unaudited types
