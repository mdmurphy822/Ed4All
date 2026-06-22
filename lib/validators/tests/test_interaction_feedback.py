"""IB6.3 — InteractionFeedbackValidator."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators.interaction_feedback import InteractionFeedbackValidator


def _self_check(feedback, block_id="b1") -> Block:
    return Block(
        block_id=block_id,
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content="<p>Q?</p>",
        interaction="Answer.",
        feedback=feedback,
    )


def test_thin_feedback_flagged():
    block = _self_check("Good try.")  # 2 tokens < 8
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    assert any(i.code == "INTERACTION_FEEDBACK_THIN" for i in res.issues)
    assert (
        res.metadata["interaction_feedback"]["per_block"]["b1"]["feedback_status"]
        == "thin"
    )


def test_elaborated_feedback_passes():
    block = _self_check(
        "Not quite — you added before multiplying. Order of operations "
        "requires you to evaluate multiplication first, so the answer is 14."
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
    assert not any(
        i.code
        in (
            "INTERACTION_NO_FEEDBACK",
            "INTERACTION_BARE_RIGHT_WRONG",
            "INTERACTION_FEEDBACK_THIN",
        )
        for i in res.issues
    )
    assert (
        res.metadata["interaction_feedback"]["per_block"]["b1"]["feedback_status"]
        == "elaborated"
    )


def test_empty_feedback_caps_dims():
    block = _self_check(None)
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)
    caps = res.metadata["interaction_feedback"]["caps_dims_by_block"]
    assert caps["b1"] == ["feedback", "coherence"]


def test_bare_right_wrong_flagged():
    block = _self_check("Correct!")
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "INTERACTION_BARE_RIGHT_WRONG" for i in res.issues)


def test_non_interactive_skipped():
    block = Block(
        block_id="c1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.metadata["interaction_feedback"]["interactive_blocks_audited"] == 0


def test_disabled_is_noop_pass():
    block = _self_check(None)
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"INTERACTION_FEEDBACK_DISABLED"}
