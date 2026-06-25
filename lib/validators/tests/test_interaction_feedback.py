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


# --- IB6.3 FP fix: feedback lives INLINE in content, not the feedback slot ----


def test_inline_details_reveal_counts_as_feedback():
    # Regression for the IB6.3 false-fire: Block.feedback is reserved for an
    # authoring wave and NEVER populated by derive_anatomy_slots, so 0/N
    # interactive blocks carry it — the renderers emit corrective feedback
    # INLINE as a <details>/Show-answer reveal. A self-check whose feedback
    # slot is empty BUT whose content carries an explained inline reveal must
    # NOT fire NO_FEEDBACK.
    block = Block(
        block_id="b1",
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content=(
            "<p>What is 2 + 3 x 4?</p>"
            "<details><summary>Show answer</summary>"
            "<p>The answer is 14 — order of operations requires you to "
            "multiply 3 x 4 first, then add 2.</p></details>"
        ),
        interaction="Answer.",
        feedback=None,  # never populated in practice
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)
    assert (
        res.metadata["interaction_feedback"]["per_block"]["b1"]["feedback_status"]
        == "elaborated"
    )


def test_inline_dict_explanation_counts_as_feedback():
    block = Block(
        block_id="kc1",
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content={
            "question": "Q?",
            "options": [{"text": "right", "correct": True}],
            "explanation": "Because the distributive property applies here, "
            "you expand the product before combining like terms.",
        },
        interaction="Answer.",
        feedback=None,
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)


def test_problem_with_no_reveal_still_fires_no_feedback():
    # A problem (B08) with NO feedback slot AND no inline reveal in its body
    # genuinely lacks feedback — it still fires (the fix only changes WHERE the
    # validator looks, not whether a real miss fires).
    block = Block(
        block_id="p1",
        block_type="problem",
        page_id="week_01_practice",
        sequence=1,
        content="<p>Solve for x: 2x + 3 = 11.</p>",
        interaction="Answer.",
        feedback=None,
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)


def test_checklist_excluded_from_feedback_requirement():
    # A checklist (B08 listing) is non-answer-bearing — nothing to reveal — so
    # it is excluded from the feedback-presence requirement and never audited.
    block = Block(
        block_id="cl1",
        block_type="checklist",
        page_id="week_01_summary",
        sequence=1,
        content="<ul><li>Item A</li><li>Item B</li></ul>",
        interaction="Check.",
        feedback=None,
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)
    assert res.metadata["interaction_feedback"]["interactive_blocks_audited"] == 0


def test_discussion_prompt_excluded_from_feedback_requirement():
    block = Block(
        block_id="dp1",
        block_type="discussion_prompt",
        page_id="week_01_discussion",
        sequence=1,
        content="<p>Discuss how order of operations applies in real life.</p>",
        interaction="Discuss.",
        feedback=None,
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(i.code == "INTERACTION_NO_FEEDBACK" for i in res.issues)
    assert res.metadata["interaction_feedback"]["interactive_blocks_audited"] == 0


# --- FR-INT-05: per-distractor misconception-targeted feedback ----------------


def _knowledge_check(*, options, option_feedback=None, feedback="Elaborated why "
                     "the order of operations matters here.", block_id="kc1") -> Block:
    return Block(
        block_id=block_id,
        block_type="self_check_question",  # B07
        page_id="week_01_self_check",
        sequence=1,
        content={"question": "Q?", "options": options},
        interaction="Answer.",
        feedback=feedback,
        option_feedback=option_feedback,
    )


_OPTIONS = [
    {"text": "right", "correct": True},
    {"text": "wrong-a", "correct": False},
    {"text": "wrong-b", "correct": False},
]


def test_distractors_without_option_feedback_flagged():
    block = _knowledge_check(options=_OPTIONS, option_feedback=None)
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert any(
        i.code == "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK" for i in res.issues
    )


def test_distractors_with_option_feedback_pass():
    block = _knowledge_check(
        options=_OPTIONS,
        option_feedback={"wrong-a": "you added first", "wrong-b": "you skipped X"},
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(
        i.code == "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK" for i in res.issues
    )


def test_threaded_signal_true_clears_check():
    block = _knowledge_check(options=_OPTIONS, option_feedback=None)
    res = InteractionFeedbackValidator().validate(
        {
            "blocks": [block],
            "rubric_enabled": True,
            "distractor_signals_by_block": {
                "kc1": {"misconception_targeted": True}
            },
        }
    )
    assert not any(
        i.code == "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK" for i in res.issues
    )


def test_threaded_signal_false_fires_even_with_option_feedback():
    block = _knowledge_check(
        options=_OPTIONS, option_feedback={"wrong-a": "x", "wrong-b": "y"}
    )
    res = InteractionFeedbackValidator().validate(
        {
            "blocks": [block],
            "rubric_enabled": True,
            "distractor_signals_by_block": {
                "kc1": {"misconception_targeted": False}
            },
        }
    )
    assert any(
        i.code == "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK" for i in res.issues
    )


def test_no_distractors_never_fires():
    # Free-response-style: only a correct option, no distractors.
    block = _knowledge_check(
        options=[{"text": "right", "correct": True}], option_feedback=None
    )
    res = InteractionFeedbackValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert not any(
        i.code == "DISTRACTOR_NO_MISCONCEPTION_FEEDBACK" for i in res.issues
    )
