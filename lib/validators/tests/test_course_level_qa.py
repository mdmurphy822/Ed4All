"""FR-COURSE-01 — CourseLevelQaValidator (course-level §6.5 emergent QA).

Asserts the gate (a) no-ops byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is
unset, (b) FIRES COURSE_INTERACTION_MIX_NARROW + COURSE_NO_INTEGRATION_CLOSE on
a read-only (student-content-only, no closing summative) course, (c) PASSES the
interaction-mix + integration-close checks on a balanced course that spans all
three interaction modes and closes with a summative block, and (d) is
warning-day-1 (passed=True) throughout.
"""
from __future__ import annotations

import json

import pytest

from Courseforge.scripts.blocks import Block
from lib.validators.course_level_qa import CourseLevelQaValidator

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
        n_representations=2,
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
    res = CourseLevelQaValidator().validate(
        {"gate_id": "course_level_qa", "blocks": [_block("b1")]}
    )
    assert res.passed is True
    assert res.metadata["course_level_qa"]["enabled"] is False
    assert _codes(res) == {"RUBRIC_DISABLED"}


# --------------------------------------------------------------------------- #
# (b) Narrow-interaction / no-integration-close course -> fires.
# --------------------------------------------------------------------------- #


def test_narrow_interaction_mix_and_no_close_fire(rubric_on):
    # A read-only course: only student-content blocks (concept / explanation /
    # example), spread across two modules, and the last module does NOT close
    # with a summative/integration block.
    blocks = [
        _block("b1", "concept", page_id="week_01_content"),
        _block("b2", "explanation", page_id="week_01_content"),
        _block("b3", "example", page_id="week_02_content"),
        _block("b4", "concept", page_id="week_02_content"),
    ]
    res = CourseLevelQaValidator().validate(
        {"gate_id": "course_level_qa", "blocks": blocks}
    )
    codes = _codes(res)
    assert "COURSE_INTERACTION_MIX_NARROW" in codes
    assert "COURSE_NO_INTEGRATION_CLOSE" in codes
    meta = res.metadata["course_level_qa"]
    assert meta["n_interaction_modes"] == 1
    assert meta["interaction_modes_present"] == ["student_content"]
    assert meta["has_integration_close"] is False
    # Warning-day-1: the gate never blocks.
    assert res.passed is True


# --------------------------------------------------------------------------- #
# (c) Balanced course -> interaction-mix + integration-close PASS.
# --------------------------------------------------------------------------- #


def test_balanced_course_passes_mix_and_close(rubric_on):
    # Spans all three interaction modes (content + discussion B10 + graded B14)
    # and the last module closes with a summative (B13 summary) + a graded final.
    blocks = [
        _block("b1", "concept", page_id="week_01_content"),
        _block("b2", "self_check_question", page_id="week_01_content"),
        _block("b3", "discussion_prompt", page_id="week_01_discussion"),
        _block("b4", "explanation", page_id="week_02_content"),
        _block("b5", "assessment_item", page_id="week_02_assessment"),
        _block("b6", "summary_takeaway", page_id="week_02_summary"),
    ]
    res = CourseLevelQaValidator().validate(
        {"gate_id": "course_level_qa", "blocks": blocks}
    )
    codes = _codes(res)
    assert "COURSE_INTERACTION_MIX_NARROW" not in codes
    assert "COURSE_NO_INTEGRATION_CLOSE" not in codes
    meta = res.metadata["course_level_qa"]
    assert meta["n_interaction_modes"] == 3
    assert set(meta["interaction_modes_present"]) == {
        "student_content",
        "student_student",
        "student_instructor",
    }
    assert meta["has_integration_close"] is True
    assert res.passed is True


# --------------------------------------------------------------------------- #
# Manifest supplies the student-instructor mode when no inline B14 exists.
# --------------------------------------------------------------------------- #


def test_manifest_graded_supplies_instructor_mode(rubric_on, tmp_path):
    # Content + discussion only inline (2 modes); the 06_assessments manifest
    # adds the graded student-instructor surface -> 3 modes -> mix passes.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"assessments": [{"id": "quiz-1", "type": "quiz"}]}),
        encoding="utf-8",
    )
    blocks = [
        _block("b1", "concept", page_id="week_01_content"),
        _block("b2", "discussion_prompt", page_id="week_01_discussion"),
        _block("b3", "summary_takeaway", page_id="week_01_summary"),
    ]
    res = CourseLevelQaValidator().validate(
        {
            "gate_id": "course_level_qa",
            "blocks": blocks,
            "assessments_path": str(tmp_path),
        }
    )
    meta = res.metadata["course_level_qa"]
    assert meta["manifest_graded"] is True
    assert "student_instructor" in meta["interaction_modes_present"]
    assert "COURSE_INTERACTION_MIX_NARROW" not in _codes(res)


# --------------------------------------------------------------------------- #
# Dimension A: orphan terminal objective flagged from real objectives doc.
# --------------------------------------------------------------------------- #


def test_orphan_terminal_objective_flagged(rubric_on, tmp_path):
    obj = tmp_path / "synthesized_objectives.json"
    obj.write_text(
        json.dumps(
            {
                "terminal_objectives": [{"id": "TO-01"}, {"id": "TO-02"}],
                "chapter_objectives": [
                    {"id": "CO-01", "terminal_id": "TO-01"},
                    {"id": "CO-02", "terminal_id": "TO-02"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Blocks only target CO-01 (-> TO-01); TO-02 is an orphan.
    blocks = [
        _block("b1", "concept", page_id="week_01_content", objective_ids=("CO-01",)),
        _block("b2", "discussion_prompt", page_id="week_01_discussion",
               objective_ids=("CO-01",)),
        _block("b3", "assessment_item", page_id="week_01_assessment",
               objective_ids=("CO-01",)),
    ]
    res = CourseLevelQaValidator().validate(
        {
            "gate_id": "course_level_qa",
            "blocks": blocks,
            "synthesized_objectives_path": str(obj),
        }
    )
    assert "COURSE_TO_COVERAGE_GAP" in _codes(res)
    meta = res.metadata["course_level_qa"]
    assert meta["n_terminal_objectives"] == 2
    assert meta["n_orphan_terminal_objectives"] == 1


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
    CourseLevelQaValidator().validate(
        {
            "gate_id": "course_level_qa",
            "blocks": [_block("b1")],
            "decision_capture": cap,
        }
    )
    ev = next(e for e in cap.events if e["decision_type"] == "validation_result")
    assert "course_level_qa" in ev["decision"]
    assert len(ev["rationale"]) >= 20
    assert "n_interaction_modes" in ev["ml_features"]


# --------------------------------------------------------------------------- #
# Gate is registered in the router.
# --------------------------------------------------------------------------- #


def test_gate_registered_in_router():
    from MCP.hardening.gate_input_routing import default_router

    r = default_router()
    assert (
        "lib.validators.course_level_qa.CourseLevelQaValidator" in r.builders
    )


# --------------------------------------------------------------------------- #
# No blocks -> structured skip (warning), never crashes.
# --------------------------------------------------------------------------- #


def test_no_blocks_structured_skip(rubric_on):
    res = CourseLevelQaValidator().validate(
        {"gate_id": "course_level_qa", "blocks": []}
    )
    assert res.passed is True
    assert "BLOCKS_UNAVAILABLE" in _codes(res)
