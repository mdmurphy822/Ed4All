"""Wave 26 — assessment dimension in ``quality_report.json`` tests.

Pre-Wave-26 bug: ``quality_report.json`` carried chunk/corpus dimensions
only — a user reviewing it couldn't see WHICH question was broken, just
an aggregate score. Wave 26 grafts an ``assessments`` dimension onto the
report that surfaces per-question issues.

See :func:`Trainforge.generators.assessment_quality_report.build_assessment_dimension`.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment_quality_report import (  # noqa: E402
    build_assessment_dimension,
)


def _mcq(qid: str, stem: str, correct: str, distractors=None,
         bloom="understand", obj="LO-01"):
    """Helper to build a minimal MCQ question dict."""
    if distractors is None:
        distractors = [f"wrong-{qid}-{i}" for i in range(3)]
    choices = [{"text": correct, "is_correct": True}]
    for d in distractors:
        choices.append({"text": d, "is_correct": False})
    return {
        "question_id": qid,
        "question_type": "multiple_choice",
        "stem": stem,
        "bloom_level": bloom,
        "objective_id": obj,
        "choices": choices,
    }


def test_clean_assessment_all_ratios_high_empty_issues():
    """A clean assessment: distinct stems, distinct answers, no
    templated distractors, verbful Bloom verbs. All ratios >= 0.8 and
    per_question_issues is empty."""
    questions = [
        _mcq("q-001", "<p>Explain photosynthesis</p>", "light-to-glucose 1"),
        _mcq("q-002", "<p>Describe the Calvin cycle</p>", "carbon-fixation 2"),
        _mcq("q-003", "<p>Identify photosystem II role</p>", "water-splitting 3", bloom="remember"),
        _mcq("q-004", "<p>Compare aerobic to anaerobic respiration</p>",
             "oxygen-presence-matters 4"),
        _mcq("q-005", "<p>Explain ATP synthesis</p>", "chemiosmotic-gradient 5",
             obj="LO-02"),
    ]
    assessment = {
        "assessment_id": "ASM-TEST",
        "questions": questions,
        "objectives_targeted": ["LO-01", "LO-02"],
    }

    dim = build_assessment_dimension(assessment)
    assert dim is not None
    assert dim["total_questions"] == 5
    assert dim["distinct_stems"] == 5
    assert dim["distinct_correct_answers"] == 5
    assert dim["distinct_stem_ratio"] >= 0.8
    assert dim["distinct_correct_answer_ratio"] >= 0.8
    # Per-question issues is empty (no broken questions).
    assert dim["per_question_issues"] == [], dim["per_question_issues"]
    # Bloom distribution observed
    assert "understand" in dim["bloom_distribution_observed"]
    # Objective coverage
    assert dim["objective_coverage_ratio"] == 1.0


def test_broken_assessment_populates_per_question_issues():
    """An assessment where every question has the SAME stem: ratios
    drop, per_question_issues carries LOW_STEM_DIVERSITY at the
    cross-question position."""
    same_stem = "<p>Which of the following best describes Structural?</p>"
    questions = [
        _mcq(f"q-{i:03d}", same_stem, f"answer-{i}") for i in range(10)
    ]
    assessment = {
        "assessment_id": "ASM-BROKEN",
        "questions": questions,
    }

    dim = build_assessment_dimension(assessment)
    assert dim is not None
    assert dim["total_questions"] == 10
    # 1 distinct stem out of 10 → ratio 0.1
    assert dim["distinct_stems"] == 1
    assert dim["distinct_stem_ratio"] < 0.2

    # per_question_issues must carry the LOW_STEM_DIVERSITY marker.
    flattened_codes = []
    for entry in dim["per_question_issues"]:
        flattened_codes.extend(entry["issues"])
    assert "LOW_STEM_DIVERSITY" in flattened_codes, dim["per_question_issues"]


def test_toc_fragment_answer_surfaces_per_question_issue():
    """A specific question with a TOC fragment as the correct answer
    must appear in per_question_issues with its question_id + the
    TOC_FRAGMENT_ANSWER code."""
    questions = [
        _mcq("q-001", "<p>Explain mitosis phases</p>", "prophase metaphase 1"),
        _mcq(
            "q-bad-002",
            "<p>Explain structural changes in the economy</p>",
            "1.1 Structural changes in the economy 14 1.7 From the "
            "periphery 22",
        ),
        _mcq("q-003", "<p>Explain DNA replication</p>", "semi-conservative 3"),
        _mcq("q-004", "<p>Describe RNA transcription</p>", "mRNA synthesis 4"),
    ]
    assessment = {
        "assessment_id": "ASM-TOC",
        "questions": questions,
    }
    dim = build_assessment_dimension(assessment)
    assert dim is not None

    # Find q-bad-002 in per_question_issues
    by_qid = {e["question_id"]: e["issues"] for e in dim["per_question_issues"]}
    assert "q-bad-002" in by_qid, by_qid
    assert "TOC_FRAGMENT_ANSWER" in by_qid["q-bad-002"]


def test_no_assessments_returns_none():
    """Missing or empty assessments returns None — caller omits the
    dimension cleanly."""
    assert build_assessment_dimension(None) is None
    assert build_assessment_dimension({}) is None
    assert build_assessment_dimension({"questions": []}) is None


def test_distractor_entropy_range():
    """avg_distractor_entropy is a float in [0.0, 1.0]."""
    questions = [
        _mcq("q-001", "<p>Explain A</p>", "a 1",
             distractors=["wrong-1", "wrong-2", "wrong-3"]),
        _mcq("q-002", "<p>Describe B</p>", "b 2",
             distractors=["wrong-4", "wrong-5", "wrong-6"]),
    ]
    dim = build_assessment_dimension({"questions": questions})
    assert dim is not None
    assert 0.0 <= dim["avg_distractor_entropy"] <= 1.0
