"""IB6.2 — AnatomySlotPresenceValidator."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators.anatomy_slot_presence import AnatomySlotPresenceValidator


def test_interactive_missing_feedback_caps_dims():
    # self_check_question == B07 (interactive); has interaction, no feedback.
    block = Block(
        block_id="b1",
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content="<p>What is x?</p>",
        heading="Check",
        interaction="Answer the question.",
        feedback=None,
        transition="Next we cover y.",
    )
    res = AnatomySlotPresenceValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True  # warning-day-1
    codes = {i.code for i in res.issues}
    assert "ANATOMY_FEEDBACK_MISSING_ON_INTERACTIVE" in codes
    caps = res.metadata["anatomy_slot_presence"]["caps_dims_by_block"]
    assert caps["b1"] == ["feedback", "coherence"]


def test_concept_with_heading_body_transition_passes():
    block = Block(
        block_id="b2",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        heading="A concept",
        purpose_tag="present",
        transition="Now consider the next idea.",
    )
    res = AnatomySlotPresenceValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    # concept is not interactive → no interaction/feedback issue, transition present.
    assert not res.issues or all(i.severity != "critical" for i in res.issues)
    assert not res.metadata["anatomy_slot_presence"]["caps_dims_by_block"]


def test_slots_absent_skip_pass():
    # No anatomy slot populated anywhere → graceful skip.
    block = Block(
        block_id="b3",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
    )
    res = AnatomySlotPresenceValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"ANATOMY_SLOTS_ABSENT"}


def test_transition_missing_flagged():
    block = Block(
        block_id="b4",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        heading="A concept",
    )
    res = AnatomySlotPresenceValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "ANATOMY_TRANSITION_MISSING" for i in res.issues)


def test_disabled_is_noop_pass():
    block = Block(
        block_id="b5",
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content="<p>Q?</p>",
        heading="h",
    )
    res = AnatomySlotPresenceValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"ANATOMY_PRESENCE_DISABLED"}
