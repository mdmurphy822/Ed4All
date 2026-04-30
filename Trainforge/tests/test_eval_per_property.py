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


def _build_property_probes_split(tmp_path: Path) -> Path:
    """Helper: write a holdout_split.json carrying the new
    `property_probes` array covering ALL SIX rdf-shacl declared
    properties, with 2 probes per property."""
    split = tmp_path / "split.json"
    property_ids = (
        "sh_datatype", "sh_class", "sh_nodeshape",
        "sh_propertyshape", "rdfs_subclassof", "owl_sameas",
    )
    surface_forms = {
        "sh_datatype": "sh:datatype",
        "sh_class": "sh:class",
        "sh_nodeshape": "sh:NodeShape",
        "sh_propertyshape": "sh:PropertyShape",
        "rdfs_subclassof": "rdfs:subClassOf",
        "owl_sameas": "owl:sameAs",
    }
    property_probes = []
    for i, pid in enumerate(property_ids):
        sf = surface_forms[pid]
        for j in range(2):
            property_probes.append({
                "probe_id": f"property-{pid}-{j:04d}",
                "property_id": pid,
                "prompt": f"Does the chunk use `{sf}`?",
                "probe_text": f"Does the chunk use `{sf}`?",
                "ground_truth_chunk_id": f"chunk_{i*2+j:03d}",
                "surface_form": sf,
                "expected_response": "affirm",
            })
    split.write_text(json.dumps({
        "withheld_edges": [],  # empty — would have produced all-null
        "property_probes": property_probes,
    }), encoding="utf-8")
    return split


def test_property_probes_array_scores_all_six_properties(tmp_path: Path) -> None:
    """Audit 2026-04-30 fix: when holdout_split.json carries a
    `property_probes` array (the Wave-122-followup emission path),
    PerPropertyEvaluator uses those probes directly instead of
    surface-form-filtering withheld_edges. With ``withheld_edges``
    empty, the legacy path would have produced all-null
    per_property_accuracy (the cc07cc76 silent-skip bug); the new
    path scores every declared property the manifest covers.

    Asserts coverage of ALL SIX rdf-shacl manifest properties — not
    just one — so a regression that drops any property's emit path
    fails this test rather than silently passing."""
    split = _build_property_probes_split(tmp_path)
    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="rdf-shacl-551-2",
        model_callable=lambda _p: "Yes.",
    ).evaluate()
    expected = {
        "sh_datatype", "sh_class", "sh_nodeshape",
        "sh_propertyshape", "rdfs_subclassof", "owl_sameas",
    }
    for prop_id in expected:
        assert result["per_property_accuracy"][prop_id] == 1.0, (
            f"{prop_id} not scored 1.0; got "
            f"{result['per_property_accuracy'].get(prop_id)!r}"
        )
        assert result["per_property_scored"][prop_id] == 2


def test_property_probes_path_isolates_each_property(tmp_path: Path) -> None:
    """A model that says 'no' to one property's probes but 'yes' to
    the rest must score that property at 0.0 and the others at 1.0.
    Catches a regression where property_probes are pooled instead of
    grouped by property_id."""
    split = _build_property_probes_split(tmp_path)

    def _selective(prompt: str) -> str:
        # Say "no" only to sh:datatype probes; "yes" to everything else.
        if "sh:datatype" in prompt:
            return "No, that is incorrect."
        return "Yes."

    result = PerPropertyEvaluator(
        holdout_split=split,
        course_slug="rdf-shacl-551-2",
        model_callable=_selective,
    ).evaluate()
    assert result["per_property_accuracy"]["sh_datatype"] == 0.0
    for prop_id in (
        "sh_class", "sh_nodeshape", "sh_propertyshape",
        "rdfs_subclassof", "owl_sameas",
    ):
        assert result["per_property_accuracy"][prop_id] == 1.0
