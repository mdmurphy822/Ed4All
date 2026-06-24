"""FR-INT-03 — B11 reflection predict-then-reveal calibration.

CPU-pinned, no-GPU, no-model unit tests for:
* the ED4ALL_REFLECTION_CALIBRATION flag resolver,
* the three B11 calibration Block fields (hash-excluded),
* the InteractionFeedbackValidator REFLECTION_NO_CAPTURE arm,
* the AnatomySlotPresenceValidator benchmark-in-feedback-slot check,
* the <details> predict-then-reveal render scaffold.

PRIMARY gate: byte-stability with ED4ALL_REFLECTION_CALIBRATION unset — both
validator arms no-op and the render scaffold returns "". The two validators also
ride ED4ALL_BLOCK_QUALITY_RUBRIC (their parent gate flag); we enable it
explicitly via rubric_enabled so the reflection arm is exercised.
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
    resolve_reflection_calibration,
)
from lib.validators.anatomy_slot_presence import (  # noqa: E402
    AnatomySlotPresenceValidator,
)
from lib.validators.interaction_feedback import (  # noqa: E402
    InteractionFeedbackValidator,
)


def _refl(**kw) -> Block:
    return Block(
        block_id="w1_sc#reflection_prompt_x_0",
        block_type="reflection_prompt",
        page_id="w1_sc",
        sequence=0,
        content="What did you learn?",
        **kw,
    )


# --------------------------------------------------------------------------- #
# Flag resolver
# --------------------------------------------------------------------------- #
def test_flag_resolver_default_off(monkeypatch):
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    assert resolve_reflection_calibration() is False


def test_flag_resolver_truthy(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "on")
    assert resolve_reflection_calibration() is True


def test_flag_resolver_garbage_off(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "banana")
    assert resolve_reflection_calibration() is False


# --------------------------------------------------------------------------- #
# Block fields hash-excluded
# --------------------------------------------------------------------------- #
def test_calibration_fields_hash_excluded():
    base = _refl()
    enriched = _refl(
        prediction_prompt="pp", reveal_content="rc", calibration_feedback="cf"
    )
    assert base.compute_content_hash() == enriched.compute_content_hash()


# --------------------------------------------------------------------------- #
# InteractionFeedbackValidator REFLECTION_NO_CAPTURE
# --------------------------------------------------------------------------- #
def test_reflection_arm_noop_when_calibration_off():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_refl()], "rubric_enabled": True,
         "reflection_calibration_enabled": False}
    )
    assert "REFLECTION_NO_CAPTURE" not in [i.code for i in r.issues]


def test_reflection_arm_fires_on_bare_reflection():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_refl()], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert r.passed is True  # warning-day-1
    codes = [i.code for i in r.issues]
    assert "REFLECTION_NO_CAPTURE" in codes


def test_reflection_arm_clean_with_predict_then_reveal():
    blk = _refl(prediction_prompt="Predict X", reveal_content="Answer Y")
    r = InteractionFeedbackValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert "REFLECTION_NO_CAPTURE" not in [i.code for i in r.issues]


def test_reflection_arm_clean_with_calibration_feedback():
    blk = _refl(calibration_feedback="Compare your judgment to the benchmark.")
    r = InteractionFeedbackValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert "REFLECTION_NO_CAPTURE" not in [i.code for i in r.issues]


def test_interaction_feedback_byte_stable_when_rubric_off():
    r = InteractionFeedbackValidator().validate(
        {"blocks": [_refl()], "rubric_enabled": False,
         "reflection_calibration_enabled": True}
    )
    assert r.passed is True
    assert r.issues[0].code == "INTERACTION_FEEDBACK_DISABLED"


# --------------------------------------------------------------------------- #
# AnatomySlotPresenceValidator benchmark-in-feedback
# --------------------------------------------------------------------------- #
def test_anatomy_benchmark_noop_when_calibration_off():
    blk = _refl(heading="Reflect", transition="next")
    r = AnatomySlotPresenceValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": False}
    )
    assert "ANATOMY_REFLECTION_NO_BENCHMARK" not in [i.code for i in r.issues]


def test_anatomy_benchmark_fires_without_benchmark():
    blk = _refl(heading="Reflect", transition="next")
    r = AnatomySlotPresenceValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert r.passed is True  # warning-day-1
    assert "ANATOMY_REFLECTION_NO_BENCHMARK" in [i.code for i in r.issues]


def test_anatomy_benchmark_clean_with_reveal():
    blk = _refl(heading="Reflect", transition="next", reveal_content="benchmark")
    r = AnatomySlotPresenceValidator().validate(
        {"blocks": [blk], "rubric_enabled": True,
         "reflection_calibration_enabled": True}
    )
    assert "ANATOMY_REFLECTION_NO_BENCHMARK" not in [i.code for i in r.issues]


# --------------------------------------------------------------------------- #
# Render scaffold
# --------------------------------------------------------------------------- #
def test_calibration_render_byte_stable_off(monkeypatch):
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    import generate_course as gc

    blk = _refl(prediction_prompt="pp", reveal_content="rc",
                calibration_feedback="cf")
    assert gc._render_reflection_calibration(blk) == ""


def test_calibration_render_emits_details_on(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    blk = _refl(prediction_prompt="PredictMove", reveal_content="RevealMove",
                calibration_feedback="CalibrateMove")
    out = gc._render_reflection_calibration(blk)
    assert "<details" in out
    assert "PredictMove" in out and "RevealMove" in out and "CalibrateMove" in out


def test_calibration_render_empty_when_no_fields_set(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    assert gc._render_reflection_calibration(_refl()) == ""
