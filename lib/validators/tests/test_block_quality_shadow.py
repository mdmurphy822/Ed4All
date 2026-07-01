"""W8.8 — ED4ALL_BLOCK_QUALITY_SHADOW shadow-collect mode.

The IB6 block-quality gates are default-OFF (keystone
``ED4ALL_BLOCK_QUALITY_RUBRIC``), so they never COMPUTE and never generate
the fire-rate telemetry the calibration harness needs (chicken-and-egg).
Shadow-collect runs the MEASUREMENT path (metadata + warning issues +
captures) WITHOUT gating (warning-day-1 → no verdict change) and WITHOUT the
emit-side render, so calibration data accrues while product bytes stay stable.

These tests pin: (a) the shared resolvers, (b) that shadow makes the
validators compute, (c) that default (both flags off) is a byte-stable no-op.
"""
from __future__ import annotations

import pytest

from Courseforge.scripts.blocks import Block
from lib.validators import _block_rubric_helpers as h
from lib.validators.content import BlockCognitiveLoadValidator
from lib.validators.block_quality_rubric import BlockQualityRubricValidator
from lib.validators.anatomy_slot_presence import AnatomySlotPresenceValidator
from lib.validators.interaction_feedback import InteractionFeedbackValidator
from lib.validators.qa_checklist import QaChecklistValidator


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_RUBRIC", raising=False)
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_SHADOW", raising=False)


def _concept(content: str, block_id: str = "b1") -> Block:
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content=content,
    )


# --------------------------------------------------------------------------------------
# Resolvers
# --------------------------------------------------------------------------------------
def test_shadow_resolver_reflects_env(monkeypatch):
    assert h.block_quality_shadow_enabled() is False
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "1")
    assert h.block_quality_shadow_enabled() is True
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "garbage")
    assert h.block_quality_shadow_enabled() is False  # parse-with-fallback


def test_scoring_active_is_rubric_or_shadow(monkeypatch):
    assert h.block_quality_scoring_active() is False
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "true")
    assert h.block_quality_scoring_active() is True
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_SHADOW", raising=False)
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "on")
    assert h.block_quality_scoring_active() is True


# --------------------------------------------------------------------------------------
# Default (both flags off) — byte-stable disabled no-op preserved.
# --------------------------------------------------------------------------------------
def test_default_off_is_disabled_no_op():
    block = _concept("<p>" + ("word " * 240) + "</p>")  # would overflow if active
    res = BlockCognitiveLoadValidator().validate({"blocks": [block]})
    assert res.passed is True
    assert res.metadata["block_cognitive_load"]["enabled"] is False
    assert any(i.code == "COGNITIVE_LOAD_DISABLED" for i in res.issues)

    r2 = BlockQualityRubricValidator().validate({"blocks": [block]})
    assert r2.metadata["block_quality_rubric"]["enabled"] is False
    assert any(i.code == "RUBRIC_DISABLED" for i in r2.issues)


# --------------------------------------------------------------------------------------
# Shadow on (rubric off) — validators COMPUTE, verdict unchanged (warning-day-1).
# --------------------------------------------------------------------------------------
def test_shadow_makes_cognitive_load_compute(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "1")
    block = _concept("<p>" + ("word " * 240) + "</p>")  # ~1200 chars → overflow
    res = BlockCognitiveLoadValidator().validate({"blocks": [block]})
    # Verdict unchanged (warning-day-1) but the signal is now COMPUTED + recorded.
    assert res.passed is True
    meta = res.metadata["block_cognitive_load"]
    assert meta["enabled"] is True
    assert meta["blocks_overflow"] == 1
    assert meta["per_block"]["b1"]["overflow"] is True
    assert any(i.code == "BLOCK_BODY_OVERFLOW" for i in res.issues)


def test_shadow_makes_rubric_compute_and_marks_shadow(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "1")
    block = _concept("<p>A short developed idea about slope.</p>")
    res = BlockQualityRubricValidator().validate({"blocks": [block]})
    assert res.passed is True  # warning-day-1: never blocks
    meta = res.metadata["block_quality_rubric"]
    assert meta["enabled"] is True
    assert meta["shadow"] is True  # tagged as shadow telemetry, not a real rubric run
    assert meta["blocks_scored"] >= 1


def test_rubric_run_is_not_marked_shadow(monkeypatch):
    # Keystone rubric flag on → real run, shadow marker False.
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    block = _concept("<p>A short developed idea about slope.</p>")
    res = BlockQualityRubricValidator().validate({"blocks": [block]})
    meta = res.metadata["block_quality_rubric"]
    assert meta["enabled"] is True
    assert meta["shadow"] is False


def test_shadow_activates_anatomy_interaction_qa(monkeypatch):
    # The other IB6 gates also leave their disabled early-return under shadow.
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_SHADOW", "1")
    block = _concept("<p>concept body</p>")
    a = AnatomySlotPresenceValidator().validate({"blocks": [block]})
    assert not any(i.code == "ANATOMY_PRESENCE_DISABLED" for i in a.issues)
    i = InteractionFeedbackValidator().validate({"blocks": [block]})
    assert not any(x.code == "INTERACTION_FEEDBACK_DISABLED" for x in i.issues)
    q = QaChecklistValidator().validate({"blocks": [block]})
    assert not any(x.code == "QA_CHECKLIST_DISABLED" for x in q.issues)
