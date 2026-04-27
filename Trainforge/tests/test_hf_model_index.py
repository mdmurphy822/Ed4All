"""Wave 101 - HF model-index converter + README writer tests.

Synthesizes a Wave 92 ``eval_report`` shape with all 5 layers
populated and asserts:

* ``eval_report_to_model_index`` produces >=3 entries each carrying
  task / dataset / metrics keys.
* ``write_hf_readme`` emits a parseable YAML frontmatter + non-empty
  body.
* Round-trip: the metric values in the rendered frontmatter equal
  the input eval_report values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------- #
# Synthetic eval_report                                                   #
# ---------------------------------------------------------------------- #


def _build_eval_report() -> dict:
    """Mirror the Wave 92 ``EvalReport.to_dict()`` shape with all
    optional fields populated. Used as the round-trip fixture."""
    return {
        "faithfulness": 0.8412,
        "coverage": 0.7531,
        "baseline_delta": 0.1245,
        "calibration_ece": 0.0532,
        "profile": "rdf_shacl",
        "per_tier": {
            "syntactic_pass_rate": 0.92,
            "calibration": {"ece": 0.0532, "scored": 100, "total": 120},
            "key_term_precision": {
                "avg_similarity": 0.65,
                "required_element_precision": 0.78,
                "scoring_method": "jaccard",
                "total": 50,
            },
        },
        "per_invariant": {
            "prerequisite_order": {"pass_rate": 0.83, "scored": 30, "passed": 25},
            "bloom_level": {"pass_rate": 0.71, "scored": 25, "passed": 18},
            "misconception_rejection": {"pass_rate": 0.89, "scored": 18, "passed": 16},
            "disambiguation": {"pass_rate": 0.66, "scored": 30, "passed": 20},
        },
    }


def _build_model_card() -> dict:
    return {
        "model_id": "rdf-shacl-551-2-qwen2-5-1-5b-01e31284",
        "course_slug": "rdf-shacl-551-2",
        "base_model": {
            "name": "qwen2.5-1.5b",
            "revision": "main",
            "huggingface_repo": "Qwen/Qwen2.5-1.5B",
        },
        "adapter_format": "safetensors",
        "training_config": {
            "seed": 42,
            "learning_rate": 0.0002,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_seq_length": 2048,
            "batch_size": 4,
        },
        "provenance": {
            "chunks_hash": "a" * 64,
            "pedagogy_graph_hash": "b" * 64,
            "instruction_pairs_hash": "c" * 64,
            "preference_pairs_hash": "d" * 64,
            "concept_graph_hash": "e" * 64,
            "vocabulary_ttl_hash": "f" * 64,
            "holdout_graph_hash": "0" * 64,
        },
        "created_at": "2026-04-26T18:00:00Z",
        "license": "apache-2.0",
    }


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_eval_report_to_model_index_emits_at_least_three_entries():
    from Trainforge.eval.hf_model_index import eval_report_to_model_index

    report = _build_eval_report()
    results = eval_report_to_model_index(
        eval_report=report,
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="rdf-shacl-551-2-qwen2-5-1-5b-01e31284",
    )
    assert len(results) >= 3, (
        f"Expected >=3 model-index entries; got {len(results)}: {results}"
    )
    for entry in results:
        assert "task" in entry
        assert "dataset" in entry
        assert "metrics" in entry
        assert isinstance(entry["metrics"], list) and entry["metrics"]
        # Every metric needs type + value
        for metric in entry["metrics"]:
            assert "type" in metric
            assert "value" in metric


def test_eval_report_to_model_index_dataset_namespace():
    from Trainforge.eval.hf_model_index import eval_report_to_model_index

    report = _build_eval_report()
    results = eval_report_to_model_index(
        eval_report=report,
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="m-01",
    )
    for entry in results:
        ds = entry["dataset"]
        assert ds["type"] == "ed4all/rdf-shacl-551-2"
        assert ds["split"] == "holdout"


def test_eval_report_to_model_index_metric_types():
    """Each canonical-layer score must map to its expected metric.type."""
    from Trainforge.eval.hf_model_index import eval_report_to_model_index

    report = _build_eval_report()
    results = eval_report_to_model_index(
        eval_report=report,
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="m-01",
    )
    type_index: dict = {}
    for entry in results:
        for metric in entry["metrics"]:
            type_index.setdefault(metric["type"], []).append(metric["name"])

    # Faithfulness -> f1
    assert "f1" in type_index
    # Coverage / invariants / Tier-1 syntactic -> accuracy
    assert "accuracy" in type_index
    # Calibration -> custom expected_calibration_error
    assert "expected_calibration_error" in type_index
    # Baseline delta -> custom accuracy_delta
    assert "accuracy_delta" in type_index


def test_write_hf_readme_renders_yaml_frontmatter(tmp_path):
    from Trainforge.eval.hf_model_index import write_hf_readme

    readme_path = write_hf_readme(
        run_dir=tmp_path,
        eval_report=_build_eval_report(),
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="rdf-shacl-551-2-qwen2-5-1-5b-01e31284",
        model_card=_build_model_card(),
        base_model_repo="Qwen/Qwen2.5-1.5B",
    )
    assert readme_path.exists()
    text = readme_path.read_text(encoding="utf-8")
    # YAML frontmatter delimiters
    assert text.startswith("---\n")
    assert "\n---\n" in text[4:]
    # Body sections (post-frontmatter)
    assert "## Training Data" in text
    assert "## Evaluation" in text
    assert "## Limitations" in text
    assert "## Provenance" in text
    # Body is non-empty after frontmatter
    body = text.split("\n---\n", 1)[1]
    assert len(body.strip()) > 100


def test_write_hf_readme_round_trip_metric_values(tmp_path):
    """Parse the rendered README's frontmatter and assert every metric
    value matches the input eval_report (within rounding)."""
    from Trainforge.eval.hf_model_index import write_hf_readme

    report = _build_eval_report()
    readme_path = write_hf_readme(
        run_dir=tmp_path,
        eval_report=report,
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="rdf-shacl-551-2-qwen2-5-1-5b-01e31284",
        model_card=_build_model_card(),
    )
    text = readme_path.read_text(encoding="utf-8")
    # Slice out the frontmatter between the two --- delimiters.
    parts = text.split("---\n", 2)
    assert len(parts) >= 3, "README is not in valid frontmatter form"
    front = yaml.safe_load(parts[1])

    # Required top-level keys
    assert front["library_name"] == "peft"
    assert front["base_model"] == "Qwen/Qwen2.5-1.5B"
    assert front["license"] == "apache-2.0"
    assert "model-index" in front
    assert isinstance(front["model-index"], list)
    assert front["model-index"][0]["name"] == (
        "rdf-shacl-551-2-qwen2-5-1-5b-01e31284"
    )

    # Walk the metrics and round-trip the headline values.
    metric_value_by_name: dict = {}
    for entry in front["model-index"][0]["results"]:
        for metric in entry["metrics"]:
            metric_value_by_name[metric["name"]] = float(metric["value"])

    # Faithfulness lands under "Faithfulness (KG-anchored)"
    assert metric_value_by_name["Faithfulness (KG-anchored)"] == pytest.approx(
        report["faithfulness"], abs=1e-3,
    )
    # Coverage
    assert "Coverage (Tier-1 x Tier-2 pass rate)" in metric_value_by_name
    assert metric_value_by_name["Coverage (Tier-1 x Tier-2 pass rate)"] == pytest.approx(
        report["coverage"], abs=1e-3,
    )
    # Per-invariant pass rates
    assert metric_value_by_name["prerequisite_order_pass_rate"] == pytest.approx(
        report["per_invariant"]["prerequisite_order"]["pass_rate"], abs=1e-3,
    )
    assert metric_value_by_name["misconception_rejection_pass_rate"] == pytest.approx(
        report["per_invariant"]["misconception_rejection"]["pass_rate"], abs=1e-3,
    )
    assert metric_value_by_name["bloom_level_pass_rate"] == pytest.approx(
        report["per_invariant"]["bloom_level"]["pass_rate"], abs=1e-3,
    )
    # Calibration ECE
    assert metric_value_by_name["Calibration ECE"] == pytest.approx(
        report["calibration_ece"], abs=1e-3,
    )
    # Baseline delta
    assert metric_value_by_name["Baseline delta (trained - base)"] == pytest.approx(
        report["baseline_delta"], abs=1e-3,
    )


def test_write_hf_readme_includes_provenance_hashes(tmp_path):
    from Trainforge.eval.hf_model_index import write_hf_readme

    card = _build_model_card()
    readme_path = write_hf_readme(
        run_dir=tmp_path,
        eval_report=_build_eval_report(),
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="m-01",
        model_card=card,
    )
    text = readme_path.read_text(encoding="utf-8")
    # All 7 hashes must appear by name in the body provenance section.
    for key in (
        "chunks_hash",
        "pedagogy_graph_hash",
        "instruction_pairs_hash",
        "preference_pairs_hash",
        "concept_graph_hash",
        "vocabulary_ttl_hash",
        "holdout_graph_hash",
    ):
        assert key in text, f"Provenance hash {key} missing from README"


def test_write_hf_readme_tags_for_rdf_shacl_slug(tmp_path):
    """RDF/SHACL slugs auto-tag with rdf + shacl for HF discovery."""
    from Trainforge.eval.hf_model_index import write_hf_readme

    readme_path = write_hf_readme(
        run_dir=tmp_path,
        eval_report=_build_eval_report(),
        course_slug="rdf-shacl-551-2",
        base_model="qwen2.5-1.5b",
        model_id="m-01",
        model_card=_build_model_card(),
    )
    text = readme_path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    front = yaml.safe_load(parts[1])
    assert "rdf" in front["tags"]
    assert "shacl" in front["tags"]
    assert "education" in front["tags"]


def test_write_hf_readme_partial_eval_report(tmp_path):
    """Partial eval_report (only faithfulness) still yields a valid README."""
    from Trainforge.eval.hf_model_index import write_hf_readme

    minimal = {"faithfulness": 0.5, "coverage": 0.4, "profile": "generic"}
    readme_path = write_hf_readme(
        run_dir=tmp_path,
        eval_report=minimal,
        course_slug="generic-101",
        base_model="qwen2.5-1.5b",
        model_id="m-mini",
        model_card=_build_model_card(),
    )
    text = readme_path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    front = yaml.safe_load(parts[1])
    # Two metric entries (faithfulness + coverage)
    metrics = front["model-index"][0]["results"]
    assert 1 <= len(metrics) <= 4
