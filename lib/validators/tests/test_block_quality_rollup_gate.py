"""IB6.6 — BlockQualityRollupValidator (the rollup GATE, FR-07/13, §6.5).

Asserts the gate (a) no-ops byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is
unset, (b) passes (warning) when the rollup's course_pass is True, and
(c) fails the course rollup (still warning-day-1) + enumerates the failing
block when a block trips a hard gate (Accessibility==0).
"""
from __future__ import annotations

import os

import pytest

from Courseforge.scripts.blocks import Block
from lib.validators.block_quality_rollup import BlockQualityRollupValidator

_FLAG = "ED4ALL_BLOCK_QUALITY_RUBRIC"


@pytest.fixture
def rubric_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    yield
    monkeypatch.delenv(_FLAG, raising=False)


def _block(block_id, block_type="concept", **fields):
    """Minimal real Block the rubric scorer + helpers read.

    The rubric scorer calls ``dataclasses.replace`` on each block, so the
    rollup gate must be fed real Block dataclasses (not dicts).
    """
    base = dict(
        block_id=block_id,
        block_type=block_type,
        page_id="week_01_content",
        sequence=1,
        content="<p>" + ("idea " * 20).strip() + ".</p>",
        bloom_level="understand",
        bloom_verb="explain",
        objective_ids=("CO-01",),
        n_representations=2,
    )
    base.update(fields)
    return Block(**base)


def test_disabled_flag_is_byte_stable_noop(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    res = BlockQualityRollupValidator().validate(
        {"gate_id": "block_quality_rollup", "blocks": [_block("b1")]}
    )
    assert res.passed is True
    assert res.metadata["block_quality_rollup"]["enabled"] is False
    codes = {i.code for i in res.issues}
    assert codes == {"RUBRIC_DISABLED"}


def test_passes_when_course_pass_true(rubric_on):
    # Two clean content blocks: thin-signal defaults score the core dims at >=2
    # (no a11y==0, no alignment orphan) → course_pass True.
    blocks = [_block("b1"), _block("b2")]
    res = BlockQualityRollupValidator().validate(
        {"gate_id": "block_quality_rollup", "blocks": blocks}
    )
    meta = res.metadata["block_quality_rollup"]
    assert meta["enabled"] is True
    assert meta["course_pass"] is True
    assert res.passed is True
    # No course-rollup-fail issue when course_pass holds.
    assert not any(i.code == "COURSE_QUALITY_ROLLUP_FAIL" for i in res.issues)


def test_accessibility_hard_gate_fail_warns_not_blocks(rubric_on):
    # Thread an upstream signal driving Accessibility==0 for b2 so the hard
    # gate fires (course_pass False). Warning-day-1: passed stays True.
    blocks = [_block("b1"), _block("b2")]
    gate_results_by_block = {
        "b2": {"wcag_clean": False},  # Form-C per-block signal → Accessibility 0
    }
    res = BlockQualityRollupValidator().validate(
        {
            "gate_id": "block_quality_rollup",
            "blocks": blocks,
            "gate_results_by_block": gate_results_by_block,
        }
    )
    meta = res.metadata["block_quality_rollup"]
    assert meta["course_pass"] is False
    assert meta["accessibility_gate_failures"] == 1
    assert meta["quality_decision"] == "failed"
    # Warning-day-1: the gate still passes (does not alter final_status).
    assert res.passed is True
    # ...but it enumerates the failures.
    codes = {i.code for i in res.issues}
    assert "COURSE_QUALITY_ROLLUP_FAIL" in codes
    assert "BLOCK_ACCESSIBILITY_GATE_FAIL" in codes
    # the offending block is localized
    a11y_issue = next(
        i for i in res.issues if i.code == "BLOCK_ACCESSIBILITY_GATE_FAIL"
    )
    assert a11y_issue.location == "b2"


def test_decision_capture_fires(rubric_on):
    class _Capture:
        def __init__(self):
            self.events = []

        def log_decision(self, **kw):
            self.events.append(kw)

    cap = _Capture()
    BlockQualityRollupValidator().validate(
        {
            "gate_id": "block_quality_rollup",
            "blocks": [_block("b1")],
            "decision_capture": cap,
        }
    )
    ev = next(
        e for e in cap.events if e["decision_type"] == "validation_result"
    )
    assert "block_quality_rollup" in ev["decision"]
    assert len(ev["rationale"]) >= 20


def test_gate_registered_in_router():
    from MCP.hardening.gate_input_routing import default_router

    r = default_router()
    assert (
        "lib.validators.block_quality_rollup.BlockQualityRollupValidator"
        in r.builders
    )
