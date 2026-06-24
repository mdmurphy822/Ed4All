"""C3-4 — B07 self-check CONFIDENCE capture (metacognition / calibration).

CPU-pinned, no-GPU, no-model unit tests for:
* the C3-4 ``confidence_prompt`` Block field (Optional default-None, hash-excluded),
* the InteractionFeedbackValidator SELF_CHECK_NO_CONFIDENCE_CAPTURE arm,
* the ``_render_confidence_capture`` self-check render control.

Reuses the EXISTING ``ED4ALL_REFLECTION_CALIBRATION`` flag (confidence capture is
the same calibration theme). PRIMARY gate: byte-stability with the flag off — the
validator arm no-ops and the render control returns "". The validator arm also
rides ``ED4ALL_BLOCK_QUALITY_RUBRIC`` (its parent gate flag); we enable it
explicitly via ``rubric_enabled`` so the arm is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Courseforge.scripts.blocks import Block  # noqa: E402
from lib.generation.reflection_calibration import (  # noqa: E402
    ENV_REFLECTION_CALIBRATION,
)
from lib.validators.interaction_feedback import (  # noqa: E402
    InteractionFeedbackValidator,
)


def _sc(**kw) -> Block:
    """A minimal B07 self_check_question Block with elaborated feedback so the
    only arm under test is the confidence-capture arm (not INTERACTION_NO_
    FEEDBACK / thin-feedback)."""
    return Block(
        block_id="w1_sc#self_check_question_q_0",
        block_type="self_check_question",
        page_id="w1_sc",
        sequence=0,
        content="Which option is correct and why does it follow?",
        feedback="The correct option follows because the rule applies here in full.",
        **kw,
    )


# --------------------------------------------------------------------------- #
# Block field: default-None + hash-excluded
# --------------------------------------------------------------------------- #
def test_confidence_field_default_none():
    assert _sc().confidence_prompt is None


def test_confidence_field_hash_excluded():
    base = _sc()
    enriched = _sc(confidence_prompt="How sure are you of this answer?")
    assert base.compute_content_hash() == enriched.compute_content_hash()


# --------------------------------------------------------------------------- #
# InteractionFeedbackValidator SELF_CHECK_NO_CONFIDENCE_CAPTURE arm
# --------------------------------------------------------------------------- #
def test_confidence_arm_noop_when_calibration_off():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_sc()], "rubric_enabled": True,
         "reflection_calibration_enabled": False}
    )
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" not in [i.code for i in r.issues]


def test_confidence_arm_fires_on_no_capture():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_sc()], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert r.passed is True  # warning-day-1
    codes = [i.code for i in r.issues]
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" in codes
    # The firing issue is a warning, not a hard fail.
    fired = next(i for i in r.issues
                 if i.code == "SELF_CHECK_NO_CONFIDENCE_CAPTURE")
    assert fired.severity == "warning"


def test_confidence_arm_clean_with_confidence_prompt():
    blk = _sc(confidence_prompt="How sure are you of this answer?")
    r = InteractionFeedbackValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" not in [i.code for i in r.issues]


def test_confidence_arm_only_scopes_b07():
    # A B14 graded-assessment (assessment_item) is interactive but NOT B07, so
    # the confidence arm must not fire on it.
    blk = Block(
        block_id="w1_q#assessment_item_q_0",
        block_type="assessment_item",
        page_id="w1_q",
        sequence=0,
        content="Choose the best answer.",
        feedback="That follows because the definition requires it across the set.",
    )
    r = InteractionFeedbackValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" not in [i.code for i in r.issues]


def test_confidence_arm_byte_stable_when_rubric_off():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_sc()], "rubric_enabled": False,
         "reflection_calibration_enabled": True}
    )
    assert r.passed is True
    assert r.issues[0].code == "INTERACTION_FEEDBACK_DISABLED"


def test_confidence_arm_reads_env_when_seam_unset(monkeypatch):
    # With no explicit reflection_calibration_enabled seam, the validator reads
    # ED4ALL_REFLECTION_CALIBRATION; off → no-op.
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_sc()], "rubric_enabled": True}
    )
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" not in [i.code for i in r.issues]

    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    r2 = InteractionFeedbackValidator().validate(
        {"blocks": [_sc()], "rubric_enabled": True}
    )
    assert "SELF_CHECK_NO_CONFIDENCE_CAPTURE" in [i.code for i in r2.issues]


# --------------------------------------------------------------------------- #
# Render control
# --------------------------------------------------------------------------- #
def test_confidence_render_byte_stable_off(monkeypatch):
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    import generate_course as gc

    blk = _sc(confidence_prompt="How sure are you?")
    assert gc._render_confidence_capture(blk) == ""


def test_confidence_render_emits_control_on(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    blk = _sc(confidence_prompt="HowSurePrompt")
    out = gc._render_confidence_capture(blk)
    assert "confidence-capture" in out
    assert "HowSurePrompt" in out
    assert 'type="radio"' in out
    # Four-point scale (no neutral midpoint).
    assert out.count('type="radio"') == 4


def test_confidence_render_empty_when_field_unset(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    assert gc._render_confidence_capture(_sc()) == ""


def test_self_check_render_byte_stable_off(monkeypatch):
    # The full _render_self_check emit must be byte-identical with the flag off
    # whether or not a question supplies a confidence_prompt.
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    import generate_course as gc

    questions = [{
        "question": "Q1?",
        "options": [
            {"text": "A", "correct": True, "feedback": "yes"},
            {"text": "B", "correct": False, "feedback": "no"},
        ],
        "bloom_level": "understand",
        "confidence_prompt": "How sure are you?",
    }]
    out = gc._render_self_check(questions, page_id="w1_sc")
    assert "confidence-capture" not in out


def test_self_check_render_includes_control_on(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    questions = [{
        "question": "Q1?",
        "options": [
            {"text": "A", "correct": True, "feedback": "yes"},
            {"text": "B", "correct": False, "feedback": "no"},
        ],
        "bloom_level": "understand",
        "confidence_prompt": "HowSurePromptInline",
    }]
    out = gc._render_self_check(questions, page_id="w1_sc")
    assert "confidence-capture" in out
    assert "HowSurePromptInline" in out


def test_self_check_render_no_control_without_prompt_even_on(monkeypatch):
    # Flag on but the question supplies no confidence_prompt → no control
    # (byte-stable for legacy questions).
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    questions = [{
        "question": "Q1?",
        "options": [
            {"text": "A", "correct": True, "feedback": "yes"},
            {"text": "B", "correct": False, "feedback": "no"},
        ],
        "bloom_level": "understand",
    }]
    out = gc._render_self_check(questions, page_id="w1_sc")
    assert "confidence-capture" not in out
