"""Wave 108 / Phase B: NegativeGroundingEvaluator tests.

The evaluator scores a model's no-rate on probes that ask about facts
that do not exist in the graph. A 'yes always' adapter scores 0.0
here even when faithfulness is 1.0 — that's the gameable-eval gap
this evaluator closes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.negative_grounding import NegativeGroundingEvaluator


def _write_split(tmp_path: Path) -> Path:
    split_path = tmp_path / "holdout_split.json"
    split_path.write_text(json.dumps({
        "withheld_edges": [],
        "probes": [],
        "negative_probes": [
            {"source": "concept_a", "target": "concept_z",
             "relation_type": "prerequisite_of", "ground_truth": "no"},
            {"source": "concept_b", "target": "concept_y",
             "relation_type": "prerequisite_of", "ground_truth": "no"},
            {"source": "concept_c", "target": "concept_x",
             "relation_type": "prerequisite_of", "ground_truth": "no"},
            {"source": "concept_d", "target": "concept_w",
             "relation_type": "prerequisite_of", "ground_truth": "no"},
        ],
    }), encoding="utf-8")
    return split_path


def test_no_always_model_scores_perfect(tmp_path: Path) -> None:
    split = _write_split(tmp_path)
    no_always = lambda _prompt: "No, that statement is false."
    result = NegativeGroundingEvaluator(split, no_always).evaluate()
    assert result["negative_grounding_accuracy"] == 1.0
    assert result["false_yes_rate"] == 0.0
    assert result["scored_total"] == 4


def test_yes_always_model_scores_zero(tmp_path: Path) -> None:
    """The exact regression class Phase B catches."""
    split = _write_split(tmp_path)
    yes_always = lambda _prompt: "Yes."
    result = NegativeGroundingEvaluator(split, yes_always).evaluate()
    assert result["negative_grounding_accuracy"] == 0.0
    assert result["false_yes_rate"] == 1.0


def test_empty_negative_probes_emits_none(tmp_path: Path) -> None:
    """Old corpora without negative_probes don't crash; the harness
    can fold None into the report and the gating validator treats
    None as an unscored signal."""
    split_path = tmp_path / "holdout.json"
    split_path.write_text(json.dumps({
        "withheld_edges": [],
        "probes": [],
    }), encoding="utf-8")
    result = NegativeGroundingEvaluator(split_path, lambda _p: "yes").evaluate()
    assert result["negative_grounding_accuracy"] is None
    assert result["scored_total"] == 0
