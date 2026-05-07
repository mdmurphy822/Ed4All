"""Tests for ``SingleCorrectRateEvaluator``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.single_correct_rate import SingleCorrectRateEvaluator


# Wave 7 W7.B: input shape is now List[Tuple[str, str]] of
# (html, question_type). Bare-string entries still work via the
# defensive isinstance(entry, tuple) branch in the evaluator (degraded —
# no per-type segmentation possible) so the legacy single-shape tests
# below pass `[(html, "")]` to exercise the new contract.
def test_single_correct_marker_passes() -> None:
    """One ``data-cf-correct=\"true\"`` marker → single-correct."""
    html = """
    <ul>
      <li data-cf-correct="true">RDFS describes classes</li>
      <li data-cf-correct="false">SHACL is a programming language</li>
      <li data-cf-correct="false">OWL is unrelated to RDF</li>
    </ul>
    """
    result = SingleCorrectRateEvaluator().evaluate([(html, "multiple_choice")])
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
    result = SingleCorrectRateEvaluator().evaluate([(html, "multiple_choice")])
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
    result = SingleCorrectRateEvaluator().evaluate([(html, "multiple_choice")])
    assert result["single_correct_rate"] == 0.0
    assert result["multi_correct_count"] == 0
    assert result["no_correct_count"] == 1
    assert result["per_question"][0]["outcome"] == "none"


def test_mixed_three_questions_one_each() -> None:
    """Mixed batch: 1 single, 1 multi, 1 none → rate = 1/3."""
    blocks = [
        # Single
        ("""<ul><li data-cf-correct="true">A</li><li data-cf-correct="false">B</li></ul>""", "multiple_choice"),
        # Multi
        ("""<ul><li data-cf-correct="true">A</li><li data-cf-correct="true">B</li></ul>""", "multiple_choice"),
        # None
        ("""<ul><li data-cf-correct="false">A</li></ul>""", "multiple_choice"),
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)
    assert result["total_questions"] == 3
    # 1 / 3 ≈ 0.3333
    assert abs(result["single_correct_rate"] - (1 / 3)) < 1e-3
    assert result["multi_correct_count"] == 1
    assert result["no_correct_count"] == 1


# ---------------------------------------------------------------------------
# Wave 7 W7.B — per-question-type segmentation tests.
# ---------------------------------------------------------------------------


def _mc_single() -> str:
    return """<ul><li data-cf-correct="true">A</li><li data-cf-correct="false">B</li></ul>"""


def _mc_multi() -> str:
    return """<ul><li data-cf-correct="true">A</li><li data-cf-correct="true">B</li></ul>"""


def _tf_single() -> str:
    return """<ul><li data-cf-correct="true">True</li><li data-cf-correct="false">False</li></ul>"""


def test_single_correct_rate_unpacks_html_qtype_tuple() -> None:
    """Per-question records carry the qt and bucketing routes correctly."""
    blocks = [
        (_mc_single(), "multiple_choice"),
        (_mc_multi(), "multiple_choice"),
        (_tf_single(), "true_false"),
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)

    # Per-question records carry the qt label.
    assert result["per_question"][0]["question_type"] == "multiple_choice"
    assert result["per_question"][1]["question_type"] == "multiple_choice"
    assert result["per_question"][2]["question_type"] == "true_false"

    pqt = result["per_question_type"]
    assert set(pqt.keys()) == {"multiple_choice", "true_false"}
    assert pqt["multiple_choice"]["total_questions"] == 2
    assert pqt["multiple_choice"]["correct_count"] == 1
    assert pqt["multiple_choice"]["single_correct_rate"] == 0.5
    assert pqt["true_false"]["total_questions"] == 1
    assert pqt["true_false"]["correct_count"] == 1
    assert pqt["true_false"]["single_correct_rate"] == 1.0


def test_single_correct_rate_emits_per_question_type_block() -> None:
    """4-tuple fixture (3 MC + 1 TF) emits expected buckets + rates."""
    blocks = [
        (_mc_single(), "multiple_choice"),
        (_mc_single(), "multiple_choice"),
        (_mc_multi(), "multiple_choice"),
        (_tf_single(), "true_false"),
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)
    pqt = result["per_question_type"]

    assert pqt["multiple_choice"]["total_questions"] == 3
    assert pqt["multiple_choice"]["correct_count"] == 2
    assert abs(pqt["multiple_choice"]["single_correct_rate"] - (2 / 3)) < 1e-3
    assert pqt["multiple_choice"]["relevant"] is True

    assert pqt["true_false"]["total_questions"] == 1
    assert pqt["true_false"]["correct_count"] == 1
    assert pqt["true_false"]["single_correct_rate"] == 1.0
    assert pqt["true_false"]["relevant"] is True


def test_single_correct_rate_marks_irrelevant_buckets_as_not_relevant() -> None:
    """Essay bucket emits relevant=False (single_correct_rate is MC + TF only)."""
    blocks = [
        (_mc_single(), "multiple_choice"),
        # Essay rendered as a single-correct shell — structurally not
        # meaningful for the metric.
        (_mc_single(), "essay"),
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)
    pqt = result["per_question_type"]
    assert pqt["multiple_choice"]["relevant"] is True
    assert pqt["essay"]["relevant"] is False


def test_single_correct_rate_drops_empty_qtype_bucket_from_per_type_emit() -> None:
    """Untyped questions still count in corpus rate but not per-type emit."""
    blocks = [
        (_mc_single(), "multiple_choice"),
        # Untyped question — empty-string bucket never lands in per_question_type.
        (_mc_single(), ""),
    ]
    result = SingleCorrectRateEvaluator().evaluate(blocks)
    assert result["total_questions"] == 2
    assert result["single_correct_rate"] == 1.0
    pqt = result["per_question_type"]
    assert "" not in pqt
    assert set(pqt.keys()) == {"multiple_choice"}


def test_load_rendered_html_for_metrics_returns_tuples(tmp_path: Path) -> None:
    """Harness loader returns ``List[Tuple[str, str]]`` of (html, qt)."""
    from Trainforge.eval.slm_eval_harness import SLMEvalHarness

    harness = SLMEvalHarness.__new__(SLMEvalHarness)
    harness.course_path = tmp_path  # type: ignore[attr-defined]

    prompts = [
        {
            "question_id": "q-mc",
            "question_type": "multiple_choice",
            "rendered_html": _mc_single(),
        },
        {
            "question_id": "q-tf",
            "question_type": "true_false",
            "rendered_html": _tf_single(),
        },
        {
            "question_id": "q-no-html",
            "question_type": "essay",
            # Missing rendered_html → skipped entirely.
        },
    ]
    blocks = harness._load_rendered_html_for_metrics(prompts)

    assert len(blocks) == 2
    assert all(isinstance(b, tuple) and len(b) == 2 for b in blocks)
    assert blocks[0][1] == "multiple_choice"
    assert blocks[1][1] == "true_false"
    assert "data-cf-correct" in blocks[0][0]
