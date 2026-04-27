"""Wave 103 - tests for the ED4ALL-Bench headline-delta computation.

The renderer must emit:
* Hallucination reduction percentage from base -> adapter+rag.
* Source-grounded lift multiplier.
* Accuracy lift multiplier.
* A rendered marketing sentence carrying ED4ALL-Bench v1.0 + the
  course slug + the holdout-hash + the scoring-commit pin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_report(*, base_h, final_h, base_s, final_s, base_acc, final_acc):
    return {
        "headline_table": [
            {"setup": "base",        "accuracy": base_acc,
             "hallucination_rate": base_h, "source_match": base_s,
             "faithfulness": 1 - base_h, "qualitative_score": None},
            {"setup": "base+rag",    "accuracy": 0.55,
             "hallucination_rate": 0.35, "source_match": 0.4,
             "faithfulness": 0.65, "qualitative_score": None},
            {"setup": "adapter",     "accuracy": 0.65,
             "hallucination_rate": 0.30, "source_match": 0.2,
             "faithfulness": 0.70, "qualitative_score": None},
            {"setup": "adapter+rag", "accuracy": final_acc,
             "hallucination_rate": final_h, "source_match": final_s,
             "faithfulness": 1 - final_h, "qualitative_score": None},
        ],
        "retrieval_method_table": [],
    }


def test_compute_headline_delta_basic():
    from Trainforge.eval.headline_delta import compute_headline_delta

    report = _build_report(
        base_h=0.5, final_h=0.1,
        base_s=0.1, final_s=0.6,
        base_acc=0.4, final_acc=0.85,
    )
    out = compute_headline_delta(
        report,
        course_slug="rdf-shacl-551-2",
        holdout_hash="a" * 64,
        scoring_commit="b" * 40,
    )
    # 80% reduction (0.5 -> 0.1)
    assert out["hallucination_reduction_pct"] == pytest.approx(0.8, abs=1e-3)
    # 6x source lift (0.1 -> 0.6)
    assert out["source_grounded_lift_x"] == pytest.approx(6.0, abs=1e-3)
    # 2.125x accuracy lift (0.4 -> 0.85)
    assert out["accuracy_lift_x"] == pytest.approx(2.125, abs=1e-3)


def test_headline_sentence_contains_required_tokens():
    from Trainforge.eval.headline_delta import compute_headline_delta

    report = _build_report(
        base_h=0.5, final_h=0.1,
        base_s=0.1, final_s=0.6,
        base_acc=0.4, final_acc=0.85,
    )
    out = compute_headline_delta(
        report,
        course_slug="rdf-shacl-551-2",
        holdout_hash="a" * 64,
        scoring_commit="b" * 40,
    )
    sentence = out["headline_sentence"]
    assert "ED4ALL-Bench v1.0" in sentence
    assert "ed4all-bench/rdf-shacl-551-2" in sentence
    assert "a" * 64 in sentence
    assert "b" * 40 in sentence
    # Hallucination reduction shows as 80%
    assert "80%" in sentence
    # Source-grounded lift shows as 6x (formatted as "6.0×")
    assert "6.0" in sentence


def test_headline_delta_handles_missing_rows():
    from Trainforge.eval.headline_delta import compute_headline_delta

    out = compute_headline_delta(
        {"headline_table": []},
        course_slug="x",
        holdout_hash="h",
        scoring_commit="c",
    )
    assert out["hallucination_reduction_pct"] is None
    assert out["source_grounded_lift_x"] is None
    assert out["accuracy_lift_x"] is None
    assert "incomplete" in out["headline_sentence"]


def test_headline_delta_handles_zero_base_source():
    from Trainforge.eval.headline_delta import compute_headline_delta

    report = _build_report(
        base_h=0.5, final_h=0.1,
        base_s=0.0, final_s=0.6,  # zero base source -> N/A lift
        base_acc=0.4, final_acc=0.85,
    )
    out = compute_headline_delta(
        report,
        course_slug="x",
        holdout_hash="h",
        scoring_commit="c",
    )
    assert out["source_grounded_lift_x"] is None
    # Sentence falls back to "n/a" without crashing
    assert "n/a" in out["headline_sentence"]
