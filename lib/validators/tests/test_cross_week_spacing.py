"""C3-6 — CrossWeekSpacingValidator (course-level §6.5 cross-week spacing).

Asserts the gate (a) no-ops byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is
unset, (b) FIRES CONCEPT_MASSED_SINGLE_WEEK on a substantive concept taught +
assessed entirely within one week with no spaced revisit, (c) PASSES on the same
concept distributed across two weeks, (d) graceful-skips when week info is
unresolvable, (e) graceful-skips a single-week course, (f) is warning-day-1
(passed=True) throughout, and (g) is registered in the router.
"""
from __future__ import annotations

import pytest

from Courseforge.scripts.blocks import Block
from lib.validators.cross_week_spacing import CrossWeekSpacingValidator

_FLAG = "ED4ALL_BLOCK_QUALITY_RUBRIC"


@pytest.fixture
def rubric_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    yield
    monkeypatch.delenv(_FLAG, raising=False)


def _block(block_id, block_type="concept", page_id="week_01_content", **fields):
    base = dict(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=1,
        content="<p>" + ("idea " * 20).strip() + ".</p>",
        bloom_level="understand",
        bloom_verb="explain",
        objective_ids=("CO-01",),
    )
    base.update(fields)
    return Block(**base)


def _codes(res):
    return {i.code for i in res.issues}


# --------------------------------------------------------------------------- #
# (a) Byte-stable no-op when the flag is unset.
# --------------------------------------------------------------------------- #


def test_disabled_flag_is_byte_stable_noop(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": [_block("b1")]}
    )
    assert res.passed is True
    assert res.metadata["cross_week_spacing"]["enabled"] is False
    assert _codes(res) == {"RUBRIC_DISABLED"}


# --------------------------------------------------------------------------- #
# (b) Massed concept (one week, no revisit) -> fires.
# --------------------------------------------------------------------------- #


def test_massed_single_week_concept_fires(rubric_on):
    # CO-01 is substantive (>=3 blocks incl. a check) but entirely in week 1;
    # CO-02 lives in week 2 so the course spans >=2 weeks (cross-week axis
    # exists) — CO-02 itself is thin (1 block) so it is not audited.
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "explanation", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "self_check_question", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b4", "concept", page_id="week_02_content", objective_ids=("CO-02",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert "CONCEPT_MASSED_SINGLE_WEEK" in _codes(res)
    meta = res.metadata["cross_week_spacing"]
    assert meta["n_weeks"] == 2
    assert meta["n_substantive_concepts"] == 1
    assert meta["n_massed_concepts"] == 1
    assert "objective::CO-01" in meta["massed_concepts"]
    # Warning-day-1: the gate never blocks.
    assert res.passed is True


# --------------------------------------------------------------------------- #
# (c) Same concept distributed across two weeks -> passes (spaced).
# --------------------------------------------------------------------------- #


def test_spaced_concept_passes(rubric_on):
    # CO-01: taught in week 1, retrieved/applied again in week 2 -> spaced.
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "explanation", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "self_check_question", page_id="week_02_content",
               objective_ids=("CO-01",)),
        _block("b4", "assessment_item", page_id="week_02_assessment",
               objective_ids=("CO-01",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert "CONCEPT_MASSED_SINGLE_WEEK" not in _codes(res)
    meta = res.metadata["cross_week_spacing"]
    assert meta["n_substantive_concepts"] == 1
    assert meta["n_spaced_concepts"] == 1
    assert meta["n_massed_concepts"] == 0
    assert res.passed is True


# --------------------------------------------------------------------------- #
# Thin concept (below MIN_BLOCKS_FOR_SPACING) is not audited.
# --------------------------------------------------------------------------- #


def test_thin_concept_not_flagged(rubric_on):
    # CO-01: only 2 blocks (below the spacing threshold) all in week 1; not
    # eligible for the massed flag. CO-02 lives in week 2 (multi-week course).
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "self_check_question", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "concept", page_id="week_02_content", objective_ids=("CO-02",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert "CONCEPT_MASSED_SINGLE_WEEK" not in _codes(res)
    assert res.metadata["cross_week_spacing"]["n_substantive_concepts"] == 0


# --------------------------------------------------------------------------- #
# No-practice concept (>=3 blocks but no retrieval/check) is not audited.
# --------------------------------------------------------------------------- #


def test_no_practice_concept_not_flagged(rubric_on):
    # CO-01: 3 exposition blocks, NO retrieval/check -> there is no practice to
    # space, so the concept is not eligible for the massed flag.
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "explanation", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "example", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b4", "concept", page_id="week_02_content", objective_ids=("CO-02",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert "CONCEPT_MASSED_SINGLE_WEEK" not in _codes(res)
    assert res.metadata["cross_week_spacing"]["n_substantive_concepts"] == 0


# --------------------------------------------------------------------------- #
# (d) Unresolvable week info -> graceful structured skip.
# --------------------------------------------------------------------------- #


def test_unresolvable_week_info_structured_skip(rubric_on):
    blocks = [
        _block("b1", "concept", page_id="intro_overview", objective_ids=("CO-01",)),
        _block("b2", "self_check_question", page_id="syllabus",
               objective_ids=("CO-01",)),
        _block("b3", "explanation", page_id="appendix", objective_ids=("CO-01",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert res.passed is True
    assert "WEEK_INFO_UNRESOLVABLE" in _codes(res)
    assert res.metadata["cross_week_spacing"]["skipped"] == "WEEK_INFO_UNRESOLVABLE"
    assert "CONCEPT_MASSED_SINGLE_WEEK" not in _codes(res)


# --------------------------------------------------------------------------- #
# (e) Single-week course -> graceful structured skip (no cross-week axis).
# --------------------------------------------------------------------------- #


def test_single_week_course_structured_skip(rubric_on):
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "explanation", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "self_check_question", page_id="week_01_content",
               objective_ids=("CO-01",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert res.passed is True
    assert "SINGLE_WEEK_COURSE" in _codes(res)
    assert res.metadata["cross_week_spacing"]["skipped"] == "SINGLE_WEEK_COURSE"
    assert "CONCEPT_MASSED_SINGLE_WEEK" not in _codes(res)


# --------------------------------------------------------------------------- #
# Concept-tag fallback when no objective id resolves.
# --------------------------------------------------------------------------- #


def test_concept_tag_axis_when_no_objective(rubric_on):
    # No objective ids; the concept-tag axis (key_terms) carries the spacing
    # signal.
    blocks = [
        _block("b1", "concept", page_id="week_01_content",
               objective_ids=(), key_terms=("slope",)),
        _block("b2", "explanation", page_id="week_01_content",
               objective_ids=(), key_terms=("slope",)),
        _block("b3", "self_check_question", page_id="week_01_content",
               objective_ids=(), key_terms=("slope",)),
        _block("b4", "concept", page_id="week_02_content",
               objective_ids=(), key_terms=("intercept",)),
    ]
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": blocks}
    )
    assert "CONCEPT_MASSED_SINGLE_WEEK" in _codes(res)
    assert "concept::slope" in res.metadata["cross_week_spacing"]["massed_concepts"]


# --------------------------------------------------------------------------- #
# Decision capture fires with a dynamic, >=20-char rationale.
# --------------------------------------------------------------------------- #


def test_decision_capture_fires(rubric_on):
    class _Capture:
        def __init__(self):
            self.events = []

        def log_decision(self, **kw):
            self.events.append(kw)

    cap = _Capture()
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "self_check_question", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b3", "explanation", page_id="week_01_content",
               objective_ids=("CO-01",)),
        _block("b4", "concept", page_id="week_02_content", objective_ids=("CO-02",)),
    ]
    CrossWeekSpacingValidator().validate(
        {
            "gate_id": "cross_week_spacing",
            "blocks": blocks,
            "decision_capture": cap,
        }
    )
    ev = next(e for e in cap.events if e["decision_type"] == "validation_result")
    assert "cross_week_spacing" in ev["decision"]
    assert len(ev["rationale"]) >= 20
    assert "n_weeks" in ev["ml_features"]


# --------------------------------------------------------------------------- #
# (g) Gate is registered in the router.
# --------------------------------------------------------------------------- #


def test_gate_registered_in_router():
    from MCP.hardening.gate_input_routing import default_router

    r = default_router()
    assert (
        "lib.validators.cross_week_spacing.CrossWeekSpacingValidator" in r.builders
    )


# --------------------------------------------------------------------------- #
# No blocks -> structured skip (warning), never crashes.
# --------------------------------------------------------------------------- #


def test_no_blocks_structured_skip(rubric_on):
    res = CrossWeekSpacingValidator().validate(
        {"gate_id": "cross_week_spacing", "blocks": []}
    )
    assert res.passed is True
    assert "BLOCKS_UNAVAILABLE" in _codes(res)
