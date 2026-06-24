"""C3-2 — tests for DiscussionAssignmentGroundingValidator.

Confirms the per-item-type grounding contract for the B10 discussion +
assignment surface:

  * an ungrounded discussion item (objective_id resolves to no chunk) →
    DISCUSSION_UNGROUNDED (warning, passed stays True);
  * an ungrounded assignment item → ASSIGNMENT_UNGROUNDED;
  * a grounded discussion/assignment (objective_id present in a chunk's
    learning_outcome_refs) → PASS, no issues;
  * explicit resolvable source_chunk_ids → grounded;
  * missing chunks_path → graceful skip (GROUNDING_INPUTS_UNAVAILABLE);
  * no items → graceful skip (NO_DISCUSSION_ASSIGNMENT_ITEMS);
  * manifest reconstruction (items recovered from a 06_assessments manifest);
  * decision-capture fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.validators.discussion_assignment_grounding import (
    DiscussionAssignmentGroundingValidator,
)


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **kw: Any
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kw,
            }
        )


def _write_chunks(
    tmp_path: Path, chunks: List[Dict[str, Any]]
) -> str:
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    return str(p)


def _validate(
    *,
    discussion_items: Optional[List[Dict[str, Any]]] = None,
    assignment_items: Optional[List[Dict[str, Any]]] = None,
    chunks_path: Optional[str] = None,
    assessments_path: Optional[str] = None,
    synthesized_objectives_path: Optional[str] = None,
    capture: Optional[_SpyCapture] = None,
):
    v = DiscussionAssignmentGroundingValidator()
    inputs: Dict[str, Any] = {}
    if discussion_items is not None:
        inputs["discussion_items"] = discussion_items
    if assignment_items is not None:
        inputs["assignment_items"] = assignment_items
    if chunks_path is not None:
        inputs["chunks_path"] = chunks_path
    if assessments_path is not None:
        inputs["assessments_path"] = assessments_path
    if synthesized_objectives_path is not None:
        inputs["synthesized_objectives_path"] = synthesized_objectives_path
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Ungrounded → FLAG (warning, passed stays True)
# ---------------------------------------------------------------------------


def test_ungrounded_discussion_flagged(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [{"id": "c1", "learning_outcome_refs": ["TO-01"], "text": "x"}],
    )
    capture = _SpyCapture()
    res = _validate(
        discussion_items=[
            {"discussion_id": "DISC-TO-99", "objective_id": "TO-99", "text": "?"}
        ],
        chunks_path=chunks,
        capture=capture,
    )
    codes = {i.code for i in res.issues}
    assert "DISCUSSION_UNGROUNDED" in codes
    # Warning-severity → passed stays True (warning-day-1 posture).
    assert res.passed is True
    assert all(
        i.severity == "warning"
        for i in res.issues
        if i.code == "DISCUSSION_UNGROUNDED"
    )
    assert res.action == "regenerate"
    # Decision-capture fired with the ungrounded verdict.
    assert capture.events
    ev = capture.events[-1]
    assert ev["decision_type"] == "assessment_quality"
    assert ev["ml_features"]["n_discussion_ungrounded"] == 1


def test_ungrounded_assignment_flagged(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [{"id": "c1", "learning_outcome_refs": ["CO-01"], "text": "x"}],
    )
    res = _validate(
        assignment_items=[
            {"assignment_id": "ASGN-CO-99", "objective_id": "CO-99", "text": "do x"}
        ],
        chunks_path=chunks,
    )
    codes = {i.code for i in res.issues}
    assert "ASSIGNMENT_UNGROUNDED" in codes
    assert res.passed is True


def test_item_with_no_objective_and_no_chunks_flagged(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [{"id": "c1", "learning_outcome_refs": ["TO-01"], "text": "x"}],
    )
    res = _validate(
        discussion_items=[{"discussion_id": "DISC-orphan", "text": "?"}],
        chunks_path=chunks,
    )
    assert any(i.code == "DISCUSSION_UNGROUNDED" for i in res.issues)


# ---------------------------------------------------------------------------
# Grounded → PASS, no issues
# ---------------------------------------------------------------------------


def test_grounded_discussion_passes(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [
            {"id": "c1", "learning_outcome_refs": ["TO-01"], "text": "a"},
            {"id": "c2", "learning_outcome_refs": ["CO-01"], "text": "b"},
        ],
    )
    capture = _SpyCapture()
    res = _validate(
        discussion_items=[
            {"discussion_id": "DISC-TO-01", "objective_id": "TO-01", "text": "?"}
        ],
        assignment_items=[
            {"assignment_id": "ASGN-CO-01", "objective_id": "CO-01", "text": "do"}
        ],
        chunks_path=chunks,
        capture=capture,
    )
    assert res.passed is True
    assert not any(
        i.code in ("DISCUSSION_UNGROUNDED", "ASSIGNMENT_UNGROUNDED")
        for i in res.issues
    )
    assert res.score == 1.0
    assert capture.events[-1]["ml_features"]["passed"] is True


def test_case_insensitive_objective_resolution(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [{"id": "c1", "learning_outcome_refs": ["to-01"], "text": "a"}],
    )
    res = _validate(
        discussion_items=[{"objective_id": "TO-01", "text": "?"}],
        chunks_path=chunks,
    )
    assert res.passed is True
    assert not res.issues


def test_explicit_source_chunk_ids_ground_the_item(tmp_path):
    # objective_id resolves to NO chunk, but explicit source_chunk_ids do.
    chunks = _write_chunks(
        tmp_path,
        [{"id": "chunk-7", "learning_outcome_refs": [], "text": "a"}],
    )
    res = _validate(
        assignment_items=[
            {
                "assignment_id": "ASGN-x",
                "objective_id": "CO-99",
                "source_chunk_ids": ["chunk-7"],
                "text": "do",
            }
        ],
        chunks_path=chunks,
    )
    assert res.passed is True
    assert not any(i.code == "ASSIGNMENT_UNGROUNDED" for i in res.issues)


def test_explicit_source_chunk_ids_pointing_nowhere_flagged(tmp_path):
    chunks = _write_chunks(
        tmp_path,
        [{"id": "chunk-7", "learning_outcome_refs": [], "text": "a"}],
    )
    res = _validate(
        assignment_items=[
            {"objective_id": "CO-99", "source_chunk_ids": ["ghost-chunk"]}
        ],
        chunks_path=chunks,
    )
    assert any(i.code == "ASSIGNMENT_UNGROUNDED" for i in res.issues)


def test_synthesized_objectives_union_grounds_item(tmp_path):
    # Chunk has no refs; synthesized objectives' id set grounds the objective.
    chunks = _write_chunks(
        tmp_path,
        [{"id": "c1", "learning_outcome_refs": [], "text": "a"}],
    )
    obj = tmp_path / "synthesized_objectives.json"
    obj.write_text(
        json.dumps(
            {
                "terminal_objectives": [{"id": "TO-01", "statement": "t"}],
                "chapter_objectives": [],
            }
        ),
        encoding="utf-8",
    )
    res = _validate(
        discussion_items=[{"objective_id": "TO-01", "text": "?"}],
        chunks_path=chunks,
        synthesized_objectives_path=str(obj),
    )
    assert res.passed is True
    assert not any(i.code == "DISCUSSION_UNGROUNDED" for i in res.issues)


# ---------------------------------------------------------------------------
# Graceful skips
# ---------------------------------------------------------------------------


def test_missing_chunks_path_graceful_skip(tmp_path):
    capture = _SpyCapture()
    res = _validate(
        discussion_items=[{"objective_id": "TO-01", "text": "?"}],
        chunks_path=str(tmp_path / "does_not_exist.jsonl"),
        capture=capture,
    )
    assert res.passed is True
    assert any(i.code == "GROUNDING_INPUTS_UNAVAILABLE" for i in res.issues)
    assert capture.events[-1]["ml_features"]["applied"] is False


def test_no_chunks_input_graceful_skip():
    res = _validate(discussion_items=[{"objective_id": "TO-01"}])
    assert res.passed is True
    assert any(i.code == "GROUNDING_INPUTS_UNAVAILABLE" for i in res.issues)


def test_no_items_graceful_skip(tmp_path):
    chunks = _write_chunks(
        tmp_path, [{"id": "c1", "learning_outcome_refs": ["TO-01"]}]
    )
    capture = _SpyCapture()
    res = _validate(
        discussion_items=[],
        assignment_items=[],
        chunks_path=chunks,
        capture=capture,
    )
    assert res.passed is True
    assert any(i.code == "NO_DISCUSSION_ASSIGNMENT_ITEMS" for i in res.issues)
    assert capture.events[-1]["ml_features"]["applied"] is False


def test_no_item_inputs_at_all_reconstructs_empty(tmp_path):
    # No item lists, no assessments_path → reconstruct yields nothing → skip.
    chunks = _write_chunks(
        tmp_path, [{"id": "c1", "learning_outcome_refs": ["TO-01"]}]
    )
    res = _validate(chunks_path=chunks)
    assert res.passed is True
    assert any(i.code == "NO_DISCUSSION_ASSIGNMENT_ITEMS" for i in res.issues)


# ---------------------------------------------------------------------------
# Manifest reconstruction (the on-disk persisted path)
# ---------------------------------------------------------------------------


def test_reconstruct_items_from_manifest(tmp_path):
    out = tmp_path / "06_assessments"
    out.mkdir()
    # A discussion XML whose body carries the objective id (the persisted
    # manifest entry drops objective_id; it's recovered from the title/XML).
    (out / "week_01_discussion.xml").write_text(
        '<topic ident="DISC-TO-01"><title>Week 1 Discussion</title>'
        "<text>Discuss TO-01 ...</text></topic>",
        encoding="utf-8",
    )
    (out / "week_01_assignment.xml").write_text(
        '<resource ident="ASGN-CO-99"><title>Week 1 Assignment</title>'
        "<text>Do CO-99 task ...</text></resource>",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "v1",
        "assessments": [
            {
                "file": "week_01_discussion.xml",
                "type": "discussion",
                "title": "Week 1 Discussion",
                "identifier": "RES_week_01_discussion",
            },
            {
                "file": "week_01_assignment.xml",
                "type": "assignment",
                "title": "Week 1 Assignment",
                "identifier": "RES_week_01_assignment",
            },
            {
                "file": "week_01_quiz.xml",
                "type": "qti",
                "title": "Week 1 Quiz",
                "identifier": "RES_week_01_quiz",
            },
        ],
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    # TO-01 is grounded; CO-99 is not.
    chunks = _write_chunks(
        tmp_path, [{"id": "c1", "learning_outcome_refs": ["TO-01"], "text": "a"}]
    )
    res = _validate(assessments_path=str(mpath), chunks_path=chunks)
    codes = {i.code for i in res.issues}
    # The grounded discussion (TO-01) does not flag; the assignment (CO-99) does.
    assert "DISCUSSION_UNGROUNDED" not in codes
    assert "ASSIGNMENT_UNGROUNDED" in codes
    # The quiz entry is ignored entirely (per-type validator scope).
    assert res.passed is True


# ---------------------------------------------------------------------------
# Gate-input-routing builder + registry wiring
# ---------------------------------------------------------------------------


def test_builder_registered_and_resolves_paths(tmp_path):
    from MCP.hardening.gate_input_routing import default_router

    out = tmp_path / "06_assessments"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps({"schema_version": "v1", "assessments": []}), encoding="utf-8"
    )
    (out / "chunks.jsonl").write_text(
        json.dumps({"id": "c1", "learning_outcome_refs": ["TO-01"]}) + "\n",
        encoding="utf-8",
    )
    router = default_router()
    path = (
        "lib.validators.discussion_assignment_grounding."
        "DiscussionAssignmentGroundingValidator"
    )
    assert path in router.builders
    # phase_outputs is keyed by phase name; the builder's _locate walks the
    # nested per-phase dicts (mirrors the real assessment_synthesis output).
    phase_outputs = {
        "assessment_synthesis": {"assessments_path": str(out / "manifest.json")}
    }
    inputs, missing = router.build(path, phase_outputs, {})
    assert missing == []
    assert inputs["assessments_path"] == str(out / "manifest.json")
    assert inputs.get("chunks_path", "").endswith("chunks.jsonl")


def test_builder_skips_without_assessments():
    from MCP.hardening.gate_input_routing import default_router

    router = default_router()
    path = (
        "lib.validators.discussion_assignment_grounding."
        "DiscussionAssignmentGroundingValidator"
    )
    inputs, missing = router.build(path, {}, {})
    assert "assessments_path" in missing
