"""IB3.3 — evidence-form gate on AssessmentObjectiveAlignmentValidator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.assessment_objective_alignment import (  # noqa: E402
    AssessmentObjectiveAlignmentValidator,
)


def _setup(tmp_path, questions, *, obj_bloom="apply"):
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    objectives = tmp_path / "synthesized_objectives.json"
    assessments.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    with open(chunks, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "learning_outcome_refs": ["co-01"]}) + "\n")
    objectives.write_text(json.dumps({
        "learning_outcomes": [
            {"id": "CO-01", "bloom_level": obj_bloom,
             "statement": "Apply the method"},
        ]
    }), encoding="utf-8")
    return {
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(objectives),
    }


def _codes(res):
    return {i.code for i in res.issues}


def test_apply_objective_recall_mcq_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = _setup(tmp_path, [
        {"question_id": "Q1", "objective_id": "CO-01",
         "question_type": "multiple_choice"},
    ])
    res = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert "ASSESSMENT_EVIDENCE_FORM_TOO_LOW" in _codes(res)
    # Warning never flips the critical id-resolution pass.
    assert res.passed is True


def test_apply_objective_numeric_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = _setup(tmp_path, [
        {"question_id": "Q1", "objective_id": "CO-01",
         "question_type": "numeric"},
    ])
    res = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert "ASSESSMENT_EVIDENCE_FORM_TOO_LOW" not in _codes(res)


def test_apply_objective_scenario_evidence_form_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = _setup(tmp_path, [
        {"question_id": "Q1", "objective_id": "CO-01",
         "question_type": "multiple_choice", "evidence_form": "scenario"},
    ])
    res = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert "ASSESSMENT_EVIDENCE_FORM_TOO_LOW" not in _codes(res)


def test_understand_objective_recall_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = _setup(tmp_path, [
        {"question_id": "Q1", "objective_id": "CO-01",
         "question_type": "multiple_choice"},
    ], obj_bloom="understand")
    res = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert "ASSESSMENT_EVIDENCE_FORM_TOO_LOW" not in _codes(res)


def test_flag_off_no_evidence_code(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    inputs = _setup(tmp_path, [
        {"question_id": "Q1", "objective_id": "CO-01",
         "question_type": "multiple_choice"},
    ])
    res = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert "ASSESSMENT_EVIDENCE_FORM_TOO_LOW" not in _codes(res)
    # The critical id-resolution path is unchanged: Q1 resolves to co-01.
    assert res.passed is True
    assert res.score == 1.0
