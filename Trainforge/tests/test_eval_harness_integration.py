"""Wave 92 — End-to-end harness integration test.

Builds a synthetic LibV2 course tree (manifest + corpus + graph),
wires a mock model_callable, runs ``SLMEvalHarness.run_all()``, and
asserts the emitted ``eval_report.json`` shape conforms to the
``model_card.json::eval_scores`` schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.slm_eval_harness import SLMEvalHarness  # noqa: E402


SCHEMA_PATH = PROJECT_ROOT / "schemas" / "models" / "model_card.schema.json"


def _build_course(tmp_path: Path, *, classification: dict) -> Path:
    course = tmp_path / "tst-101"
    (course / "graph").mkdir(parents=True)
    (course / "corpus").mkdir(parents=True)

    # Manifest
    (course / "manifest.json").write_text(json.dumps({
        "classification": classification,
    }), encoding="utf-8")

    # Pedagogy graph
    nodes = [
        {"id": "bloom:remember", "class": "BloomLevel", "level": "remember"},
        {"id": "bloom:apply", "class": "BloomLevel", "level": "apply"},
        {
            "id": "mc_001", "class": "Misconception",
            "label": "wrong idea",
            "statement": "This is a misconception that should be rejected.",
        },
        {"id": "concept_x", "class": "Concept", "label": "Concept X"},
        {"id": "concept_y", "class": "Concept", "label": "Concept Y"},
    ]
    edges = [
        {"source": "concept_x", "target": "concept_y", "relation_type": "prerequisite_of"},
        {"source": "concept_y", "target": "concept_x", "relation_type": "prerequisite_of"},
        {"source": "chunk_01", "target": "bloom:remember", "relation_type": "at_bloom_level"},
        {"source": "chunk_02", "target": "bloom:apply", "relation_type": "at_bloom_level"},
        {"source": "mc_001", "target": "concept_x", "relation_type": "interferes_with"},
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8",
    )

    # Chunks
    chunks = [
        {
            "id": "c_001",
            "key_terms": [
                {"term": "concept x", "definition": "A foundational idea in this corpus."},
            ],
            "misconceptions": [
                {
                    "misconception": "This is a misconception that should be rejected.",
                    "correction": "The actual correct understanding is different and rigorous.",
                },
            ],
        },
    ]
    with (course / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    return course


def _mock_model(prompt: str) -> str:
    """Mock that always rejects + cites distinguishing language."""
    return (
        "yes, that holds. confidence: 80%. "
        "First, define the concept; rather than relying on rows, "
        "we have first-class facts. Actually false - misconception."
    )


def test_harness_emits_eval_scores_compatible_report(tmp_path):
    course = _build_course(tmp_path, classification={
        "subdomains": ["semantic web"],
        "topics": ["rdf and shacl"],
    })
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=5,
    )
    report_path = harness.run_all()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Required keys per the harness contract
    assert "faithfulness" in report
    assert "coverage" in report
    assert 0.0 <= report["faithfulness"] <= 1.0
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["profile"] == "rdf_shacl"

    # Validate the report's eval-score subset against the model_card
    # schema's eval_scores subschema.
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    eval_subschema = schema["properties"]["eval_scores"]
    # Build a stripped version with only canonical scores. Wave 102
    # added required scoring_commit + tolerance_band to eval_scores;
    # the harness emits the metric values, but the surrounding
    # reproducibility envelope is stamped by the runner (Wave 102
    # reproduce_eval.sh path), so we synthesize the required fields
    # here to validate the metric subset shape only.
    canonical = {
        k: report[k]
        for k in (
            "faithfulness", "coverage", "baseline_delta",
            "calibration_ece", "source_match",
        )
        if k in report
    }
    canonical["scoring_commit"] = "0" * 40
    canonical["tolerance_band"] = {"faithfulness": 0.05}
    jsonschema.validate(canonical, eval_subschema)


def test_harness_writes_eval_progress_artifact(tmp_path):
    course = _build_course(tmp_path, classification={
        "subdomains": ["semantic web"],
        "topics": ["rdf and shacl"],
    })
    output_path = course / "eval" / "custom_eval_report.json"
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
    )

    report_path = harness.run_all(output_path=output_path)

    progress_path = report_path.parent / "eval_progress.jsonl"
    assert progress_path.exists()
    records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["event"] == "run_start"
    assert any(r["event"] == "stage_start" for r in records)
    assert any(r["event"] == "model_call" for r in records)
    assert records[-1]["event"] == "run_end"
    assert records[-1]["total_calls"] > 0


def test_harness_picks_generic_profile_for_non_semantic_corpus(tmp_path):
    course = _build_course(tmp_path, classification={
        "subdomains": ["mathematics"],
        "topics": ["linear algebra"],
    })
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    report_path = harness.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["profile"] == "generic"


def test_harness_explicit_profile_override(tmp_path):
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        profile="rdf_shacl",
        max_holdout_questions=3,
    )
    report_path = harness.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["profile"] == "rdf_shacl"


def test_harness_with_baseline_callable_emits_delta(tmp_path):
    course = _build_course(tmp_path, classification={"subdomains": ["semantic web"]})

    def base(prompt: str) -> str:
        return "no"

    def trained(prompt: str) -> str:
        return "yes"

    harness = SLMEvalHarness(
        course_path=course,
        model_callable=trained,
        base_callable=base,
        max_holdout_questions=3,
    )
    report_path = harness.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "baseline_delta" in report
    # trained "yes" beats base "no" on every probe → delta = 1.0
    assert report["baseline_delta"] == 1.0


def test_harness_creates_holdout_split_lazily(tmp_path):
    """When ``eval/holdout_split.json`` doesn't exist yet, the harness
    builds it on the fly."""
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    assert not (course / "eval" / "holdout_split.json").exists()
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    harness.run_all()
    assert (course / "eval" / "holdout_split.json").exists()


def test_unknown_course_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SLMEvalHarness(
            course_path=tmp_path / "does-not-exist",
            model_callable=_mock_model,
        )
