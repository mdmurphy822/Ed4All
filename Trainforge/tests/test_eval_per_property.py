"""Wave 109 / Phase C: PerPropertyEvaluator tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.property_eval import PerPropertyEvaluator


def _write_split(tmp_path: Path) -> Path:
    p = tmp_path / "split.json"
    p.write_text(json.dumps({
        "withheld_edges": [
            {"source": "chunk_001", "target": "shape_a",
             "relation_type": "teaches",
             "probe_text": "Does sh:datatype constrain literal types?"},
            {"source": "chunk_002", "target": "shape_b",
             "relation_type": "teaches",
             "probe_text": "Does sh:class constrain class types?"},
            {"source": "chunk_003", "target": "shape_c",
             "relation_type": "teaches",
             "probe_text": "Does owl:sameAs assert identity?"},
        ],
    }), encoding="utf-8")
    return p


def test_perfect_model_scores_all_properties_one(tmp_path: Path) -> None:
    split = _write_split(tmp_path)
    yes_model = lambda _prompt: "Yes."
    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="rdf-shacl-551-2",
        model_callable=yes_model,
    ).evaluate()
    per_prop = result["per_property_accuracy"]
    assert per_prop["sh_datatype"] == 1.0
    assert per_prop["sh_class"] == 1.0
    assert per_prop["owl_sameas"] == 1.0


def test_no_model_scores_zero_on_property_probes(tmp_path: Path) -> None:
    split = _write_split(tmp_path)
    no_model = lambda _prompt: "No, that statement is false."
    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="rdf-shacl-551-2",
        model_callable=no_model,
    ).evaluate()
    per_prop = result["per_property_accuracy"]
    assert per_prop["sh_datatype"] == 0.0


def test_property_with_zero_probes_returns_none(tmp_path: Path) -> None:
    """When the holdout split has no probes referencing a declared
    surface form, that property's accuracy is None (unscored), not 0."""
    split = _write_split(tmp_path)
    yes_model = lambda _prompt: "Yes."
    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="rdf-shacl-551-2",
        model_callable=yes_model,
    ).evaluate()
    assert result["per_property_accuracy"]["rdfs_subclassof"] is None
    assert result["per_property_scored"]["rdfs_subclassof"] == 0


def test_unknown_course_returns_empty_report(tmp_path: Path) -> None:
    split = _write_split(tmp_path)
    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="course-with-no-manifest",
        model_callable=lambda _p: "yes",
    ).evaluate()
    assert result["per_property_accuracy"] == {}
