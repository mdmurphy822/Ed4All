"""State-repair regression coverage for rewinding a paused workflow."""

from __future__ import annotations

from pathlib import Path

from scripts.ops.repair_partial_resume_state import apply_plan, build_plan


def test_rewind_clears_later_pause_and_exposes_incomplete_phase():
    state = {
        "status": "PAUSED",
        "created_at": "2026-07-24T00:00:00Z",
        "started_at": "2026-07-25T01:00:00Z",
        "completed_at": "2026-07-25T02:00:00Z",
        "updated_at": "2026-07-25T02:00:00Z",
        "paused_phase": "training_synthesis",
        "failed_phase": None,
        "tasks": [
            {"id": "upstream-complete", "phase": "course_planning",
             "status": "COMPLETE"},
            {"id": "assessment-pending", "phase": "assessment_synthesis",
             "status": "PENDING"},
            {"id": "assessment-running", "phase": "assessment_synthesis",
             "status": "IN_PROGRESS"},
            {"id": "package-complete", "phase": "packaging",
             "status": "COMPLETE"},
            {"id": "training-pending", "phase": "training_synthesis",
             "status": "PENDING"},
        ],
        "phase_outputs": {
            "course_planning": {
                "_completed": True,
                "_gates_passed": True,
            },
            "assessment_synthesis": {
                "_completed": True,
                "_gates_passed": True,
                "assessments_dir": "/tmp/export/06_assessments",
            },
            "post_rewrite_validation": {
                "_completed": True,
                "_gates_passed": True,
            },
            "packaging": {
                "_completed": True,
                "_gates_passed": True,
            },
            "training_synthesis": {
                "_completed": False,
            },
        },
    }
    plan = build_plan(
        state,
        incomplete_phase="assessment_synthesis",
        reset_phases=[
            "post_rewrite_validation",
            "packaging",
            "training_synthesis",
        ],
    )
    apply_plan(state, plan)

    assert state["status"] == "CREATED"
    assert "paused_phase" not in state
    assert "started_at" not in state
    assert "completed_at" not in state
    assert state["created_at"] == "2026-07-24T00:00:00Z"
    assert state["updated_at"] != "2026-07-25T02:00:00Z"
    assert state["phase_outputs"]["assessment_synthesis"]["_completed"] is False
    assert set(state["phase_outputs"]) == {
        "course_planning",
        "assessment_synthesis",
    }
    assert state["tasks"] == [
        {
            "id": "upstream-complete",
            "phase": "course_planning",
            "status": "COMPLETE",
        }
    ]
    assert plan["checkpoint_reset_phases"] == [
        "assessment_synthesis",
        "post_rewrite_validation",
        "packaging",
        "training_synthesis",
    ]
    assert plan["local_checkpoint_paths"] == [
        str(
            Path("/tmp/export")
            / "06_assessments"
            / ".assessments_checkpoint.jsonl"
        )
    ]

    workflow_order = [
        "course_planning",
        "assessment_synthesis",
        "post_rewrite_validation",
        "packaging",
        "training_synthesis",
    ]
    next_phase = next(
        phase for phase in workflow_order
        if not state["phase_outputs"].get(phase, {}).get("_completed")
    )
    assert next_phase == "assessment_synthesis"
