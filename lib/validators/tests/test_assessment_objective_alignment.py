"""H3 Wave W5 — AssessmentObjectiveAlignmentValidator capture wiring.

Pins per-question ``assessment_objective_alignment_check`` emission
with dynamic signals (question_id, declared_objective_ids,
resolved_in_chunk, n_chunks_searched).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.assessment_objective_alignment import (  # noqa: E402
    AssessmentObjectiveAlignmentValidator,
)


class _StubCapture:
    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_chunks(path: Path, chunks: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def test_emits_one_decision_per_question(tmp_path) -> None:
    capture = _StubCapture()
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
            {"question_id": "Q2", "objective_id": "CO-01"},
            {"question_id": "Q3", "objective_id": "PHANTOM-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01", "co-01"]},
    ])
    AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 3
    types = {c["decision_type"] for c in capture.calls}
    assert types == {"assessment_objective_alignment_check"}


def test_rationale_carries_dynamic_signals(tmp_path) -> None:
    capture = _StubCapture()
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [{"question_id": "Q1", "objective_id": "TO-01"}],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
        {"id": "c2", "learning_outcome_refs": ["co-01"]},
    ])
    AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    rationale = capture.calls[0]["rationale"]
    assert len(rationale) >= 60
    assert "Q1" in rationale
    assert "declared_objective_ids=" in rationale
    assert "resolved_in_chunk=" in rationale
    assert "n_chunks_searched=" in rationale


def test_phantom_ref_emits_failure_decision(tmp_path) -> None:
    capture = _StubCapture()
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [{"question_id": "Q1", "objective_id": "PHANTOM"}],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert "failed:" in capture.calls[0]["decision"]
    assert "PHANTOM_OBJECTIVE_REFS" in capture.calls[0]["decision"]


def test_no_capture_no_emit_no_crash(tmp_path) -> None:
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [{"question_id": "Q1", "objective_id": "TO-01"}],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    base = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    captured = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
