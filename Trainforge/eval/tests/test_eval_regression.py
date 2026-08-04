"""Wave 92 — RegressionEvaluator tests.

The pointer file format lands in Wave 93. Wave 92's regression module
is forward-compatible: it raises a clean ``FileNotFoundError`` with
a useful message when the prior pointer doesn't exist, and it
correctly compares scores when both are present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.regression import RegressionEvaluator  # noqa: E402


def test_missing_pointer_raises_clean_filenotfound(tmp_path):
    course = tmp_path / "tst-101"
    course.mkdir()
    (course / "models").mkdir()
    rev = RegressionEvaluator(
        course_path=course,
        current_eval_report={"faithfulness": 0.5, "coverage": 0.5},
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        rev.evaluate()
    msg = str(excinfo.value)
    assert "pointer" in msg.lower() or "prior" in msg.lower()


def test_compares_against_prior_when_pointer_exists(tmp_path):
    course = tmp_path / "tst-101"
    (course / "models").mkdir(parents=True)
    prior_id = "qwen-tst-aaaaaaaa"
    prior_dir = course / "models" / prior_id
    prior_dir.mkdir()
    (prior_dir / "model_card.json").write_text(json.dumps({
        "model_id": prior_id,
        "course_slug": "tst-101",
        "eval_scores": {
            "faithfulness": 0.6,
            "coverage": 0.7,
            "baseline_delta": 0.1,
        },
        "provenance": {
            "vocabulary_ttl_hash": "a" * 64,
            "pedagogy_graph_hash": "b" * 64,
        },
    }), encoding="utf-8")
    (course / "models" / "_pointers.json").write_text(json.dumps({
        "current": prior_id,
    }), encoding="utf-8")

    rev = RegressionEvaluator(
        course_path=course,
        current_eval_report={
            "_model_id": "qwen-tst-bbbbbbbb",
            "faithfulness": 0.75,
            "coverage": 0.8,
            "baseline_delta": 0.15,
            "_provenance": {
                "vocabulary_ttl_hash": "a" * 64,
                "pedagogy_graph_hash": "b" * 64,
            },
        },
    )
    out = rev.evaluate()
    assert out["comparable"] is True
    assert out["drift_warnings"] == []
    assert out["deltas"]["faithfulness"]["delta"] == pytest.approx(0.15)
    assert out["deltas"]["coverage"]["delta"] == pytest.approx(0.1, rel=1e-3)
    assert out["prior_model_id"] == prior_id


def test_drift_warning_on_provenance_mismatch(tmp_path):
    course = tmp_path / "tst-101"
    (course / "models").mkdir(parents=True)
    prior_id = "qwen-tst-aaaaaaaa"
    prior_dir = course / "models" / prior_id
    prior_dir.mkdir()
    (prior_dir / "model_card.json").write_text(json.dumps({
        "model_id": prior_id,
        "eval_scores": {"faithfulness": 0.6},
        "provenance": {
            "vocabulary_ttl_hash": "a" * 64,
            "pedagogy_graph_hash": "b" * 64,
        },
    }), encoding="utf-8")
    (course / "models" / "_pointers.json").write_text(
        json.dumps({"current": prior_id}), encoding="utf-8",
    )

    out = RegressionEvaluator(
        course_path=course,
        current_eval_report={
            "faithfulness": 0.65,
            "_provenance": {
                "vocabulary_ttl_hash": "c" * 64,  # changed
                "pedagogy_graph_hash": "b" * 64,
            },
        },
    ).evaluate()
    assert out["comparable"] is False
    assert any("vocabulary_ttl_hash" in w for w in out["drift_warnings"])
