"""Wave 92 — CalibrationEvaluator tests.

Synthesise model outputs that state a fixed confidence at a known
accuracy level; assert ECE matches the expected gap. The test
fixture builds a 10-edge holdout split and a model_callable that
answers Yes/No with a stated confidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.calibration import CalibrationEvaluator  # noqa: E402
from Trainforge.eval.holdout_builder import HoldoutBuilder  # noqa: E402


def _build_course(tmp_path: Path, n_edges: int = 20) -> Path:
    course = tmp_path / "tst-101"
    (course / "graph").mkdir(parents=True)
    edges = [
        {
            "source": f"a_{i}", "target": f"b_{i}",
            "relation_type": "prerequisite_of",
        }
        for i in range(n_edges)
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": [], "edges": edges}), encoding="utf-8",
    )
    HoldoutBuilder(course, holdout_pct=0.5, seed=42).build()
    return course


def test_perfect_calibration_yields_low_ece(tmp_path):
    """Model that says 'yes, 100% confident' on a true statement and is
    always right: ECE should be 0 on that bin."""
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def model(prompt: str) -> str:
        return "yes\nconfidence: 100%"

    out = CalibrationEvaluator(holdout, model, bins=5).evaluate()
    assert out["ece"] == 0.0
    assert out["scored"] >= 1


def test_overconfident_wrong_model_has_high_ece(tmp_path):
    """Model says 100% confident but answers wrongly: ECE should be 1.0."""
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def overconfident(prompt: str) -> str:
        return "no\nconfidence: 100%"

    out = CalibrationEvaluator(holdout, overconfident, bins=5).evaluate()
    # All "no" answers, all incorrect, all 100% conf → ECE = 1.0
    assert out["ece"] == 1.0


def test_unparsable_response_is_dropped(tmp_path):
    """Responses without a confidence number are not scored toward ECE."""
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def vague(prompt: str) -> str:
        return "yes maybe"

    out = CalibrationEvaluator(holdout, vague).evaluate()
    assert out["scored"] == 0
    assert out["ece"] == 0.0


def test_bin_summary_buckets_responses(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def fifty_fifty(prompt: str) -> str:
        return "yes\nconfidence: 50%"

    out = CalibrationEvaluator(holdout, fifty_fifty, bins=10).evaluate()
    # All responses fall in the 0.5-0.6 bucket
    bucket_counts = sum(b["count"] for b in out["bins"])
    assert bucket_counts == out["scored"]
