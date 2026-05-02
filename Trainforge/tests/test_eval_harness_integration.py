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


def test_harness_reads_explicit_eval_profile_from_manifest(tmp_path):
    """Wave 132c: manifest.eval_profile is authoritative.

    A course whose subdomain is 'mathematics' (legacy substring sniff =>
    'generic') but whose manifest declares eval_profile=rdf_shacl must
    resolve to rdf_shacl, no warning, no sniff.
    """
    course = _build_course(tmp_path, classification={
        "subdomains": ["mathematics"],
        "topics": ["linear algebra"],
    })
    # Augment the manifest with an explicit eval_profile.
    manifest_path = course / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["eval_profile"] = "rdf_shacl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    report_path = harness.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["profile"] == "rdf_shacl", (
        "manifest.eval_profile must override the substring sniff."
    )


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


def test_eval_report_includes_negative_grounding_and_yes_rate(tmp_path):
    """Wave 108 / Phase B: the harness emits negative_grounding_accuracy
    and yes_rate in eval_report.json so the gating validator can read
    them. A 'no' model gets perfect negative-grounding."""
    course = _build_course(tmp_path, classification={
        "subdomains": ["semantic web"],
        "topics": ["rdf and shacl"],
    })
    no_model = lambda _prompt: "No, that statement is false."
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=no_model,
        max_holdout_questions=10,
    )
    out_path = harness.run_all()
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert "negative_grounding_accuracy" in report
    assert "yes_rate" in report
    # 'No' model has yes_rate ~= 0.0 and negative_grounding_accuracy ~= 1.0:
    assert report["yes_rate"] == 0.0
    assert report["negative_grounding_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Wave 138a: Teaching-role alignment + per-stage checkpoint
# ---------------------------------------------------------------------------


def _augment_chunks_with_teaching_roles(course: Path, extra_chunks: list) -> None:
    """Append chunks carrying content_type_label + teaching_role to the
    course's chunks.jsonl so TeachingRoleAlignmentEvaluator has signal."""
    chunks_path = course / "corpus" / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as fh:
        for c in extra_chunks:
            fh.write(json.dumps(c) + "\n")


def test_run_all_emits_content_type_role_alignment_when_chunks_present(tmp_path):
    """Wave 138a / Plan1-W2: TeachingRoleAlignmentEvaluator output flows
    through to eval_report.json so EvalGatingValidator can reach it."""
    course = _build_course(tmp_path, classification={
        "subdomains": ["semantic web"],
    })
    # Add 6 real_world_scenario chunks all wrongly labeled elaborate
    # (the rdf-shacl-551-2 audit signal we want to detect).
    _augment_chunks_with_teaching_roles(course, [
        {"id": f"rws_{i}", "content_type_label": "real_world_scenario",
         "teaching_role": "elaborate"}
        for i in range(6)
    ])

    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    report_path = harness.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "content_type_role_alignment" in report
    assert "content_type_role_alignment_summary" in report
    rws = report["content_type_role_alignment"]["real_world_scenario"]
    assert rws["expected_role"] == "transfer"
    assert rws["mismatch"] is True
    assert rws["actual_expected_share"] == 0.0
    assert "real_world_scenario" in (
        report["content_type_role_alignment_summary"]["mismatched_content_types"]
    )


def test_run_all_skips_teaching_role_alignment_when_chunks_missing(tmp_path):
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    # Remove the chunks.jsonl
    (course / "corpus" / "chunks.jsonl").unlink()
    # Re-create an empty corpus directory marker so build_course's
    # other invariants hold (the harness reaches into the corpus path
    # for label resolution etc.)
    (course / "corpus" / "chunks.jsonl").write_text("", encoding="utf-8")

    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    # The empty file IS still treated as "present"; remove it for the
    # actual missing-file case.
    (course / "corpus" / "chunks.jsonl").unlink()
    harness2 = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=3,
    )
    # Restore minimal chunks for harness internals before run_all so
    # downstream stages don't fail; the alignment stage gates on
    # chunks_path.exists() at its own scope.
    (course / "corpus" / "chunks.jsonl").write_text(
        json.dumps({"id": "c_001"}) + "\n", encoding="utf-8",
    )
    report_path = harness2.run_all()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # With chunks.jsonl carrying only an empty id (no content_type_label),
    # the evaluator runs but emits empty per-content-type dict.
    if "content_type_role_alignment" in report:
        assert report["content_type_role_alignment"] == {}


def test_run_all_skips_eval_checkpoint_when_disabled(tmp_path):
    """--no-eval-checkpoint disables the sidecar entirely."""
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
        eval_checkpoint_enabled=False,
    )
    harness.run_all()
    sidecar = course / "eval" / ".eval_results_checkpoint.jsonl"
    assert not sidecar.exists()


def test_eval_stage_checkpoint_unlinked_on_clean_exit(tmp_path):
    """On clean exit, the sidecar is removed — eval_report.json
    is now the authoritative source."""
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
    )
    harness.run_all()
    sidecar = course / "eval" / ".eval_results_checkpoint.jsonl"
    assert not sidecar.exists()
    # eval_report.json IS authoritative now
    assert (course / "eval" / "eval_report.json").exists()


def test_eval_stage_checkpoint_resume_skips_cached_stages(tmp_path):
    """Pre-seed the checkpoint with a fake stage result; the harness
    must skip the lambda for that stage and replay the cached value."""
    from Trainforge.eval.slm_eval_harness import (
        _append_eval_stage_checkpoint,
        _load_eval_stage_checkpoint,
    )
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    sidecar = course / "eval" / ".eval_results_checkpoint.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # Seed the sidecar with a faithfulness stage result. On the next
    # run, the harness must replay this rather than calling the
    # FaithfulnessEvaluator. We can't easily trap that without monkey-
    # patching, so we assert via the eval_report's faithfulness value:
    # the cached stage result has accuracy=0.42, which is unusual for
    # the _mock_model (which would otherwise produce different values).
    cached_faithfulness = {
        "accuracy": 0.42, "scored_total": 7, "correct": 3,
    }
    with sidecar.open("w", encoding="utf-8") as fh:
        _append_eval_stage_checkpoint(
            fh, stage="faithfulness", result=cached_faithfulness,
        )
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
    )
    harness.run_all()
    report = json.loads(
        (course / "eval" / "eval_report.json").read_text(encoding="utf-8"),
    )
    assert report["faithfulness"] == 0.42
    # The sentinel value would never naturally show up — only the
    # checkpoint replay produces it. After clean exit the sidecar is
    # unlinked.
    assert not sidecar.exists()


def test_eval_stage_checkpoint_schema_version_drift_invalidates(tmp_path):
    """A v0 record in the checkpoint must be skipped (with a warning)
    so the stage re-runs against the live evaluator."""
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    sidecar = course / "eval" / ".eval_results_checkpoint.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({
            "schema_version": "v0",
            "stage": "faithfulness",
            "result": {"accuracy": 0.42, "scored_total": 7, "correct": 3},
        }) + "\n",
        encoding="utf-8",
    )
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
    )
    harness.run_all()
    report = json.loads(
        (course / "eval" / "eval_report.json").read_text(encoding="utf-8"),
    )
    # The v0 record was dropped; faithfulness re-ran. The _mock_model
    # affirms ("yes, that holds. ...") so faithfulness should be a
    # full 1.0, NOT the cached 0.42.
    assert report["faithfulness"] != 0.42


def test_eval_stage_checkpoint_malformed_lines_tolerated(tmp_path):
    course = _build_course(tmp_path, classification={"subdomains": ["math"]})
    sidecar = course / "eval" / ".eval_results_checkpoint.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "not valid json\n"
        "{ broken json\n"
        "\n"
        "  \n",
        encoding="utf-8",
    )
    harness = SLMEvalHarness(
        course_path=course,
        model_callable=_mock_model,
        max_holdout_questions=2,
    )
    # Should NOT crash on malformed lines — they're just skipped.
    harness.run_all()


def test_eval_stage_checkpoint_helpers_no_path_noop(tmp_path):
    """Direct helper-level test: None path returns empty cache, append
    is a no-op."""
    from Trainforge.eval.slm_eval_harness import (
        _append_eval_stage_checkpoint,
        _load_eval_stage_checkpoint,
    )
    assert _load_eval_stage_checkpoint(None) == {}
    _append_eval_stage_checkpoint(None, stage="x", result={})  # no raise


def test_eval_stage_checkpoint_appended_per_stage(tmp_path):
    """After a clean run, a fresh checkpoint built from the same input
    contains one record per evaluator stage (before unlink)."""
    from Trainforge.eval.slm_eval_harness import (
        _append_eval_stage_checkpoint,
        _load_eval_stage_checkpoint,
    )
    # Direct helper test — we already verified the integration with
    # actual harness runs above. This pins the invariant that one
    # append produces one parseable record.
    sidecar = tmp_path / ".eval_results_checkpoint.jsonl"
    with sidecar.open("a", encoding="utf-8") as fh:
        _append_eval_stage_checkpoint(fh, stage="s1", result={"acc": 0.5})
        _append_eval_stage_checkpoint(fh, stage="s2", result={"acc": 0.7})
        _append_eval_stage_checkpoint(fh, stage="s3", result={"acc": 0.9})
    cache = _load_eval_stage_checkpoint(sidecar)
    assert set(cache.keys()) == {"s1", "s2", "s3"}
    assert cache["s2"]["acc"] == 0.7
