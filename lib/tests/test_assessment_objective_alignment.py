"""Tests for AssessmentObjectiveAlignmentValidator (Wave 24 scope 7).

The validator fails loudly when any assessment question's objective_id
is not present in any chunk's learning_outcome_refs. This guards
against the pre-Wave-24 failure mode (disjoint LO naming schemes) from
resurfacing silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.validators.assessment_objective_alignment import (
    AssessmentObjectiveAlignmentValidator,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_chunks(path: Path, chunks: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def test_all_aligned_passes(tmp_path):
    """Every question's objective_id appears in chunk refs → pass."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
            {"question_id": "Q2", "objective_id": "CO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
        {"id": "c2", "learning_outcome_refs": ["co-01", "to-02"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert result.passed
    assert result.score == 1.0


def test_phantom_refs_fail_loudly(tmp_path):
    """Questions referencing non-existent objective_ids → critical fail."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "PHYS_101_OBJ_1"},
            {"question_id": "Q2", "objective_id": "PHYS_101_OBJ_2"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01", "co-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    # PHANTOM_OBJECTIVE_REFS code should surface.
    codes = {i.code for i in result.issues}
    assert "PHANTOM_OBJECTIVE_REFS" in codes


def test_empty_questions_list_is_skipped(tmp_path):
    """No questions in payload → warning-only, not critical."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {"questions": []})
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    # Passes with a warning.
    assert result.passed


def test_missing_inputs_error(tmp_path):
    """Missing required inputs → critical fail with descriptive code."""
    result = AssessmentObjectiveAlignmentValidator().validate({})
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "MISSING_ASSESSMENTS_PATH" in codes


def test_case_insensitive_match(tmp_path):
    """TO-01 in question matches to-01 in chunks (Trainforge lowercases)."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert result.passed


def test_empty_chunkset_fails_closed(tmp_path):
    """C4 audit fix: zero learning_outcome_refs across chunks file
    indicates an upstream chunking failure. Fail closed with the
    named code rather than vacuously passing.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    # File exists, parses cleanly, but no chunk carries refs.
    _write_chunks(chunks, [
        {"id": "c1"},
        {"id": "c2", "learning_outcome_refs": []},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in codes
    crit = [i for i in result.issues if i.code == "ASSESSMENT_ALIGNMENT_NO_CHUNKS"]
    assert crit and crit[0].severity == "critical"
    # Operator hint must reference the upstream phase + the chunks path.
    msg = crit[0].message
    assert str(chunks) in msg
    assert "imscc_chunking" in msg or "trainforge_assessment" in msg


def test_completely_empty_chunks_file_fails_closed(tmp_path):
    """An empty chunks.jsonl file (zero lines) also fails closed."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    chunks.write_text("", encoding="utf-8")
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in codes


def test_question_missing_objective_id_fails(tmp_path):
    """Question with no objective_id → critical."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1"},  # no objective_id
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "QUESTION_MISSING_OBJECTIVE" in codes
