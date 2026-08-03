"""Phase 2 Subtask 5 — Block + Touch dataclass regression suite.

Covers:
    - Block.block_type validates against BLOCK_TYPES enum
    - Block is frozen (FrozenInstanceError on assignment)
    - with_touch returns a new instance and grows the chain
    - Touch chain composes across all three tiers
    - compute_content_hash is stable across touch chain (audit-only)
    - compute_content_hash changes when content changes
    - compute_content_hash excludes sequence (and validation_attempts /
      escalation_marker — Phase 3 feedback fields)
    - stable_id format
    - Touch validates decision_capture_id non-empty (Wave 112)
    - Touch validates tier enum
    - Touch validates provider enum
    - Block validates validation_attempts >= 0 (Phase 3 amendment)
    - Block validates escalation_marker against _ESCALATION_MARKERS
      (Phase 3 amendment)
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import BLOCK_TYPES, Block, Touch  # noqa: E402


def _make_touch(tier: str = "outline", capture_id: str = "decisions:0") -> Touch:
    return Touch(
        model="claude-sonnet-4",
        provider="local",
        tier=tier,
        timestamp="2026-05-02T00:00:00Z",
        decision_capture_id=capture_id,
        purpose="draft",
    )


def _make_block(**overrides) -> Block:
    base = {
        "block_id": "page_01#objective_to_01_0",
        "block_type": "objective",
        "page_id": "page_01",
        "sequence": 0,
        "content": "Define X concretely",
        "objective_ids": ("TO-01",),
        "bloom_level": "remember",
    }
    base.update(overrides)
    return Block(**base)


def test_block_type_validates_against_enum():
    with pytest.raises(ValueError, match="block_type"):
        Block(
            block_id="x",
            block_type="not_a_real_type",
            page_id="p",
            sequence=0,
            content="c",
        )


def test_block_is_frozen():
    b = _make_block()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.content = "mutated"  # type: ignore[misc]


def test_with_touch_appends_and_returns_new_instance():
    b = _make_block()
    t = _make_touch()
    b2 = b.with_touch(t)
    assert id(b) != id(b2)
    assert len(b.touched_by) == 0
    assert len(b2.touched_by) == 1
    assert b2.touched_by[0] is t


def test_with_touch_chain_grows_three_tiers():
    b = _make_block()
    chained = b
    for tier in ("outline", "validation", "rewrite"):
        chained = chained.with_touch(_make_touch(tier=tier, capture_id=f"d:{tier}"))
    assert len(chained.touched_by) == 3
    assert [t.tier for t in chained.touched_by] == ["outline", "validation", "rewrite"]


def test_compute_content_hash_is_stable_across_touch_chain():
    b = _make_block()
    base_hash = b.compute_content_hash()
    chained = b
    for tier in ("outline", "validation", "rewrite"):
        chained = chained.with_touch(_make_touch(tier=tier, capture_id=f"d:{tier}"))
    assert chained.compute_content_hash() == base_hash


def test_compute_content_hash_changes_when_content_changes():
    a = _make_block(content="Define X concretely")
    b = _make_block(content="Define Y concretely")
    assert a.compute_content_hash() != b.compute_content_hash()


def test_compute_content_hash_excludes_sequence():
    a = _make_block(sequence=0)
    b = _make_block(sequence=99)
    assert a.compute_content_hash() == b.compute_content_hash()


def test_compute_content_hash_excludes_validation_attempts_and_escalation():
    """Phase 3 feedback fields don't shift the canonical hash."""
    a = _make_block(validation_attempts=0, escalation_marker=None)
    b = _make_block(
        validation_attempts=3, escalation_marker="outline_budget_exhausted"
    )
    assert a.compute_content_hash() == b.compute_content_hash()


def test_stable_id_format():
    sid = Block.stable_id("week_01_overview", "objective", "TO-01", 0)
    assert sid == "week_01_overview#objective_TO-01_0"


def test_touch_validates_decision_capture_id_non_empty():
    with pytest.raises(ValueError, match="decision_capture_id"):
        Touch(
            model="m",
            provider="local",
            tier="outline",
            timestamp="t",
            decision_capture_id="",
            purpose="p",
        )


def test_touch_validates_tier_enum():
    with pytest.raises(ValueError, match="tier"):
        Touch(
            model="m",
            provider="local",
            tier="bogus_tier",
            timestamp="t",
            decision_capture_id="d:0",
            purpose="p",
        )


def test_touch_validates_provider_enum():
    with pytest.raises(ValueError, match="provider"):
        Touch(
            model="m",
            provider="not_a_provider",
            tier="outline",
            timestamp="t",
            decision_capture_id="d:0",
            purpose="p",
        )


def test_validation_attempts_non_negative_validates():
    """Phase 3 amendment — negative validation_attempts is rejected."""
    with pytest.raises(ValueError, match="validation_attempts"):
        _make_block(validation_attempts=-1)


def test_escalation_marker_enum_validates():
    """Phase 3 amendment — escalation_marker constrained to enum."""
    # None is fine
    b_none = _make_block(escalation_marker=None)
    assert b_none.escalation_marker is None
    # Canonical marker is fine
    b_ok = _make_block(escalation_marker="outline_budget_exhausted")
    assert b_ok.escalation_marker == "outline_budget_exhausted"
    # Off-enum marker raises
    with pytest.raises(ValueError, match="escalation_marker"):
        _make_block(escalation_marker="totally_made_up_marker")


def test_block_page_id_required_non_empty():
    with pytest.raises(ValueError, match="page_id"):
        Block(
            block_id="x",
            block_type="objective",
            page_id="",
            sequence=0,
            content="c",
        )


def test_block_sequence_non_negative_validates():
    with pytest.raises(ValueError, match="sequence"):
        Block(
            block_id="x",
            block_type="objective",
            page_id="p",
            sequence=-1,
            content="c",
        )


def test_block_types_count_is_twenty_eight():
    """Canonical 30-type set (16 Phase-2 + 5 Wave-2 + 3 I6 palette-v2 + 4 IB5 +
    1 B15 `resources` + 1 FR-INT-02 `guided_practice`)."""
    assert len(BLOCK_TYPES) == 30
    for required in (
        "objective",
        "concept",
        "example",
        "assessment_item",
        "explanation",
        "prereq_set",
        "activity",
        "misconception",
        "callout",
        "flip_card_grid",
        "self_check_question",
        "summary_takeaway",
        "reflection_prompt",
        "discussion_prompt",
        "chrome",
        "recap",
        # IB5 framework-aligned pedagogical block types.
        "hook",
        "multimedia",
        "worked_example",
        "diagram",
    ):
        assert required in BLOCK_TYPES


def test_touch_to_jsonld_camelcase_keys():
    t = _make_touch()
    out = t.to_jsonld()
    assert out["decisionCaptureId"] == "decisions:0"
    assert out["model"] == "claude-sonnet-4"
    assert out["provider"] == "local"
    assert "decision_capture_id" not in out


# ---------------------------------------------------------------------------
# IB1 — six-slot anatomy contract + five-stage micro-lifecycle
# ---------------------------------------------------------------------------

from blocks import (  # noqa: E402
    LIFECYCLE_STAGES,
    _BODY_SLOT,
    _STAGE_SLOTS,
    derive_anatomy_slots,
    lifecycle_stage_coverage,
    slots_to_lifecycle,
)


def test_anatomy_slots_default_none():
    """IB1.1 — a Block built with no slot kwargs has all five slots None."""
    b = _make_block()
    assert b.heading is None
    assert b.purpose_tag is None
    assert b.interaction is None
    assert b.feedback is None
    assert b.transition is None


def test_anatomy_slots_settable_and_frozen():
    """IB1.1 — slots are settable, read back, and immutable (frozen)."""
    b = _make_block(
        heading="Simplify",
        purpose_tag="formative-assessment",
        interaction="self-check",
        feedback="Check your work",
        transition="Up next",
    )
    assert b.heading == "Simplify"
    assert b.purpose_tag == "formative-assessment"
    assert b.interaction == "self-check"
    assert b.feedback == "Check your work"
    assert b.transition == "Up next"
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.heading = "mutated"  # type: ignore[misc]


def test_compute_content_hash_excludes_anatomy_slots():
    """IB1.3 — the five anatomy slots are excluded from the content hash."""
    a = _make_block()
    b = dataclasses.replace(
        a,
        heading="H",
        purpose_tag="P",
        interaction="I",
        feedback="F",
        transition="T",
    )
    assert a.compute_content_hash() == b.compute_content_hash()


def test_option_feedback_default_none_and_hash_excluded():
    """FR-INT-05 — option_feedback defaults None and is excluded from the hash."""
    a = _make_block(block_type="self_check_question", page_id="p", content="c")
    assert a.option_feedback is None
    b = dataclasses.replace(a, option_feedback={"wrong": "because you forgot X"})
    assert b.option_feedback == {"wrong": "because you forgot X"}
    assert a.compute_content_hash() == b.compute_content_hash()


def test_callout_kind_default_none_and_hash_excluded():
    """FR-A11Y-03 — callout_kind defaults None and is excluded from the hash."""
    a = _make_block(block_type="callout", content="c")
    assert a.callout_kind is None
    b = dataclasses.replace(a, callout_kind="warning")
    assert b.callout_kind == "warning"
    assert a.compute_content_hash() == b.compute_content_hash()


def test_derive_anatomy_slots_heading_from_html():
    """IB1.5 — heading is parsed from content HTML; feedback/transition None."""
    b = _make_block(
        block_type="concept",
        content="<h2>Simplify</h2><p>Combine like terms.</p>",
    )
    out = derive_anatomy_slots(b)
    assert out.heading == "Simplify"
    assert out.interaction is None  # concept is non-interactive
    assert out.feedback is None
    assert out.transition is None


def test_derive_anatomy_slots_interaction_marker():
    """IB1.5 — interaction-bearing block types get a presence marker."""
    b = _make_block(block_type="self_check_question", content="What is 2+2?")
    out = derive_anatomy_slots(b)
    assert out.interaction == "self-check"


def test_derive_anatomy_slots_purpose_tag_from_purpose():
    """IB1.5 — purpose_tag consolidates purpose / teaching_role."""
    b = _make_block(block_type="concept", purpose="formative-assessment")
    out = derive_anatomy_slots(b)
    assert out.purpose_tag == "formative-assessment"


def test_derive_anatomy_slots_returns_new_instance_never_mutates():
    """IB1.5 — pure: new frozen instance returned, input untouched."""
    b = _make_block(
        block_type="self_check_question",
        content="<h2>Quiz</h2><p>Q?</p>",
        purpose="formative-assessment",
    )
    out = derive_anatomy_slots(b)
    assert out is not b
    assert b.heading is None
    assert b.interaction is None
    assert b.purpose_tag is None
    assert out.heading == "Quiz"
    assert out.interaction == "self-check"
    assert out.purpose_tag == "formative-assessment"


def test_lifecycle_stages_exactly_five_in_order():
    """IB1.6 — LIFECYCLE_STAGES is exactly the five stages in order."""
    assert LIFECYCLE_STAGES == (
        "activate",
        "present",
        "apply",
        "check",
        "consolidate",
    )


def test_lifecycle_stage_slots_mapping_matches_framework():
    """IB1.6 — _STAGE_SLOTS keys == stages; values ⊆ the six anatomy slots."""
    assert tuple(_STAGE_SLOTS.keys()) == LIFECYCLE_STAGES
    valid = {
        "heading",
        "purpose_tag",
        "content",
        "interaction",
        "feedback",
        "transition",
    }
    for slots in _STAGE_SLOTS.values():
        assert set(slots) <= valid
    assert _BODY_SLOT == "content"


def test_lifecycle_stage_coverage_for_self_check():
    """IB1.6 — coverage reflects which mapped slots are present."""
    b = derive_anatomy_slots(
        _make_block(
            block_type="self_check_question",
            content="<h2>Quiz</h2><p>Q?</p>",
        )
    )
    coverage = lifecycle_stage_coverage(b)
    assert coverage["activate"] is True  # heading derived
    assert coverage["present"] is True  # content is truthy
    assert coverage["apply"] is True  # interaction marker derived
    assert coverage["check"] is False  # feedback never derived
    assert coverage["consolidate"] is False  # transition never derived


def test_slots_to_lifecycle_lists_present_slot_names():
    """IB1.6 — slots_to_lifecycle reports the present slot names per stage."""
    b = _make_block(heading="H", interaction="self-check")
    mapping = slots_to_lifecycle(b)
    assert mapping["activate"] == ["heading"]
    assert mapping["present"] == ["content"]  # _make_block content is truthy
    assert mapping["apply"] == ["interaction"]
    assert mapping["check"] == []
    assert mapping["consolidate"] == []
