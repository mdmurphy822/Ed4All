"""Regression net for ``Trainforge/generators/assessment_sft_generator.py``.

Covers the SFT-program Phase-1 contract:

* required DecisionCapture (None raises) + one event per format batch;
* the six rationale-augmented formats emit with valid schema-bounded text;
* full per-pair provenance (§B) on every pair;
* D1 — the generator NEVER sources a ``practice_bank`` / ``assessment_item``
  harvested chunk;
* verbatim-safe ``chunk_id`` downgrade keeps the leakage gate clean;
* hint pairs never leak the answer; error-diagnosis pairs carry DPO metadata;
* template-instance caps + holdout skip + determinism.

Offline / deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment_sft_generator import (  # noqa: E402
    generate_assessment_sft_pairs,
    _FORMATS,
    ASSESSMENT_SFT_SOURCE_MARKER,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)


def _chunks() -> Dict[str, Any]:
    return {
        "c_real": {
            "text": "Real instructional prose about solving linear equations "
                    "by isolating the variable on one side.",
            "chunk_type": "explanation",
        },
        # D1 — a harvested end-of-section item chunk the generator must skip.
        "c_practice": {
            "text": "46. 550 47. 22,335 48. 39,075 (end-of-section answer key).",
            "chunk_type": "assessment_item",
            "practice_bank": True,
        },
    }


def _assessments_doc() -> Dict[str, Any]:
    return {
        "assessment_id": "a1",
        "title": "Quiz",
        "course_code": "TEST_101",
        "questions": [
            {  # Q1 numeric solve
                "question_id": "q-001",
                "question_type": "fill_in_blank",
                "item_subtype": "fib_numeric",
                "stem": "Solve 2x + 3 = 11 for x.",
                "bloom_level": "apply",
                "objective_id": "CO-01",
                "correct_answer": "4",
                "choices": [],
                "feedback": "<ol><li>Subtract 3 from both sides to get 2x = 8."
                            "</li><li>Divide both sides by 2 to find x.</li></ol>",
                "source_chunks": ["c_real"],
            },
            {  # Q2 error-analysis MC
                "question_id": "q-002",
                "question_type": "multiple_choice",
                "item_subtype": "error_analysis",
                "stem": "Simplify -(3 - 5).",
                "bloom_level": "analyze",
                "objective_id": "CO-02",
                "correct_answer": "2",
                "choices": [
                    {"text": "2", "is_correct": True},
                    {"text": "-2", "is_correct": False,
                     "misconception_note": "failed to distribute the negative sign"},
                    {"text": "8", "is_correct": False,
                     "misconception_note": "added instead of subtracting"},
                ],
                "feedback": "<li>Distribute the negative: -(3-5) = -3+5 = 2.</li>",
                "source_chunks": ["c_real"],
            },
            {  # Q3 written short-answer with rubric
                "question_id": "q-003",
                "question_type": "essay",
                "item_subtype": "short_answer",
                "stem": "Explain why dividing by zero is undefined.",
                "bloom_level": "understand",
                "objective_id": "CO-03",
                "correct_answer": "No number times zero yields a nonzero result.",
                "choices": [],
                "feedback": "<li>Because multiplication by zero always gives zero."
                            "</li>",
                "rubric": {
                    "criteria": [
                        {"criterion": "States the definition of division",
                         "cites": ["c_real"],
                         "levels": [{"score": 2, "descriptor": "complete"}]},
                    ],
                    "deductions": [],
                },
                "source_chunks": ["c_real"],
            },
            {  # Q4 grounded ONLY in a practice-bank chunk (D1 test)
                "question_id": "q-004",
                "question_type": "fill_in_blank",
                "item_subtype": "fib_numeric",
                "stem": "Compute 3 + 4.",
                "bloom_level": "apply",
                "objective_id": "CO-04",
                "correct_answer": "7",
                "choices": [],
                "feedback": "<li>Add the two numbers.</li>",
                "source_chunks": ["c_practice"],
            },
        ],
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_requires_capture() -> None:
    with pytest.raises(ValueError, match="requires a DecisionCapture"):
        list(generate_assessment_sft_pairs(_assessments_doc(), {}, None))


def test_emits_pairs_and_one_event_per_format_batch() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    assert pairs, "expected assessment-SFT pairs"
    # One assessment_generation event per format batch (6 formats).
    batch_events = [
        d for d in cap.decisions
        if d.get("decision_type") == "assessment_generation"
    ]
    assert len(batch_events) == len(_FORMATS)
    for d in batch_events:
        assert len(d.get("rationale", "")) >= 20
    # A spread of formats actually fired.
    formats = {p["pair_format"] for p in pairs}
    assert {"solve_steps", "error_diagnosis", "explain_why",
            "grade_rubric", "hint_no_reveal", "verify_answer"} <= formats


def test_every_pair_has_provenance_and_marker() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    required = {
        "prompt", "completion", "chunk_id", "lo_refs", "bloom_level",
        "content_type", "seed", "decision_capture_id", "template_id",
        "provider", "generation_method", "generating_seat", "seat_license",
        "verifier_results", "source_chunk_ids", "holdout_safe",
        "decontam_checked", "source",
    }
    for p in pairs:
        assert required <= set(p), f"missing keys: {required - set(p)}"
        assert p["provider"] == "local"                    # closed enum reuse
        assert p["source"] == ASSESSMENT_SFT_SOURCE_MARKER  # carve-out marker
        assert p["generation_method"] == "deterministic_template"
        assert p["decision_capture_id"]                    # non-empty
        assert p["lo_refs"] and p["lo_refs"][0]
        assert p["bloom_level"] in {
            "remember", "understand", "apply", "analyze", "evaluate", "create"
        }
        assert p["holdout_safe"] is True
        assert p["decontam_checked"] is False


def test_schema_bounds_respected() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    for p in pairs:
        assert 40 <= len(p["prompt"]) <= 400
        assert 50 <= len(p["completion"]) <= 600


def test_d1_never_sources_practice_bank_chunk() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    for p in pairs:
        assert "c_practice" not in p["source_chunk_ids"]
        assert p["chunk_id"] != "c_practice"
    # Q4 was grounded ONLY in the practice-bank chunk -> its pairs must carry
    # an empty grounding set + a synthetic anchor (never the practice chunk).
    q4 = [p for p in pairs if p["lo_refs"] == ["CO-04"]]
    assert q4, "expected Q4 pairs"
    for p in q4:
        assert p["source_chunk_ids"] == []
        assert p["chunk_id"].startswith("assessment_item:")


def test_error_diagnosis_carries_dpo_metadata() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    ed = [p for p in pairs if p["pair_format"] == "error_diagnosis"]
    assert ed
    for p in ed:
        dpo = p.get("dpo_metadata")
        assert isinstance(dpo, dict)
        assert dpo.get("chosen") and dpo.get("rejected")
        assert dpo["kind"] == "error_analysis"


def test_hint_never_reveals_answer() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    hints = [p for p in pairs if p["pair_format"] == "hint_no_reveal"]
    assert hints
    # Map objective -> answer for a targeted leak check.
    answers = {"CO-01": "4", "CO-02": "2", "CO-03": None, "CO-04": "7"}
    for p in hints:
        ans = answers.get(p["lo_refs"][0])
        if ans:
            assert ans.lower() not in p["completion"].lower()
        assert p["verifier_results"].get("answer_leak_check") == "passed"


def test_verbatim_span_downgrades_chunk_anchor() -> None:
    # A worked-solution feedback that copies a >=50-char span of the cited
    # chunk must NOT be anchored to that chunk (keeps the verbatim gate clean).
    doc = {
        "questions": [{
            "question_id": "q-vb",
            "question_type": "fill_in_blank",
            "stem": "Solve the equation shown.",
            "bloom_level": "apply",
            "objective_id": "CO-09",
            "correct_answer": "4",
            "feedback": "<li>Real instructional prose about solving linear "
                        "equations by isolating the variable on one side.</li>",
            "source_chunks": ["c_real"],
        }],
    }
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(doc, _chunks(), cap))
    solve = [p for p in pairs if p["pair_format"] == "solve_steps"]
    assert solve
    for p in solve:
        assert p["chunk_id"] != "c_real"
        assert p["chunk_id_downgraded_verbatim"] is True
        # Grounding provenance is still preserved.
        assert p["source_chunk_ids"] == ["c_real"]


def test_template_caps_enforced() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(
        _assessments_doc(), _chunks(), cap,
        per_template_caps={"explain_why": 1},
    ))
    explain = [p for p in pairs if p["pair_format"] == "explain_why"]
    assert len(explain) == 1


def test_holdout_items_skipped() -> None:
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(
        _assessments_doc(), _chunks(), cap,
        holdout_item_ids=["q-001"],
    ))
    # No pair should derive from the held-out numeric item (its objective).
    assert all(p["lo_refs"] != ["CO-01"] for p in pairs)


def test_deterministic() -> None:
    cap1, cap2 = _RecordingCapture(), _RecordingCapture()
    a = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap1))
    b = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap2))
    assert [(p["template_id"], p["prompt"], p["completion"], p["seed"]) for p in a] \
        == [(p["template_id"], p["prompt"], p["completion"], p["seed"]) for p in b]


def test_schema_valid_when_jsonschema_available() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    import json
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    cap = _RecordingCapture()
    pairs = list(generate_assessment_sft_pairs(_assessments_doc(), _chunks(), cap))
    for p in pairs:
        jsonschema.validate(p, schema)
