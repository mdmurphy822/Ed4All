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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parents[1]
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


def test_block_types_count_is_sixteen():
    """Phase 2 plan locks the canonical 16-type set."""
    assert len(BLOCK_TYPES) == 16
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
    ):
        assert required in BLOCK_TYPES


def test_touch_to_jsonld_camelcase_keys():
    t = _make_touch()
    out = t.to_jsonld()
    assert out["decisionCaptureId"] == "decisions:0"
    assert out["model"] == "claude-sonnet-4"
    assert out["provider"] == "local"
    assert "decision_capture_id" not in out
