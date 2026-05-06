"""Tests for ``SingleCorrectRateEvaluator``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.single_correct_rate import SingleCorrectRateEvaluator


def test_single_correct_marker_passes() -> None:
    """One ``data-cf-correct=\"true\"`` marker → single-correct."""
    html = """
    <ul>
      <li data-cf-correct="true">RDFS describes classes</li>
      <li data-cf-correct="false">SHACL is a programming language</li>
      <li data-cf-correct="false">OWL is unrelated to RDF</li>
    </ul>
    """
    result = SingleCorrectRateEvaluator().evaluate([html])
    assert result["single_correct_rate"] == 1.0
    assert result["total_questions"] == 1
    assert result["multi_correct_count"] == 0
    assert result["no_correct_count"] == 0
    assert result["per_question"][0]["correct_count"] == 1
    assert result["per_question"][0]["outcome"] == "single"


def test_multi_correct_count_increments() -> None:
    """Two ``data-cf-correct=\"true\"`` markers → multi-correct."""
    html = """
    <ul>
      <li data-cf-correct="true">A</li>
      <li data-cf-correct="true">B</li>
      <li data-cf-correct="false">C</li>
    </ul>
    """
    result = SingleCorrectRateEvaluator().evaluate([html])
    assert result["single_correct_rate"] == 0.0
    assert result["multi_correct_count"] == 1
    assert result["no_correct_count"] == 0
    assert result["per_question"][0]["correct_count"] == 2
    assert result["per_question"][0]["outcome"] == "multi"


def test_no_correct_count_increments() -> None:
    """No ``data-cf-correct=\"true\"`` markers → no-correct."""
    html = """
    <ul>
      <li data-cf-correct="false">A</li>
      <li data-cf-correct="false">B</li>
    </ul>
    """
    result = SingleCorrectRateEvaluator().evaluate([html])
    assert result["single_correct_rate"] == 0.0
    assert result["multi_correct_count"] == 0
    assert result["no_correct_count"] == 1
    assert result["per_question"][0]["outcome"] == "none"


def test_mixed_three_questions_one_each() -> None:
    """Mixed batch: 1 single, 1 multi, 1 none → rate = 1/3."""
    blocks = [
        # Single
        """<ul><li data-cf-correct="true">A</li><li data-cf-correct="false">B</li></ul>""",
        # Multi
        """<ul><li data-cf-correct="true">A</li><li data-cf-correct="true">B</li></ul>""",
        # None
        """<ul><li data-cf-correct="false">A</li></ul>""",
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)
    assert result["total_questions"] == 3
    # 1 / 3 ≈ 0.3333
    assert abs(result["single_correct_rate"] - (1 / 3)) < 1e-3
    assert result["multi_correct_count"] == 1
    assert result["no_correct_count"] == 1
