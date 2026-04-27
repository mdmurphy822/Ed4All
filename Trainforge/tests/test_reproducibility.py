"""Wave 102 - Reproducibility envelope tests.

Asserts:

* ``write_reproduce_script`` writes an executable bash script that
  pins the commit / model_id / course_slug / profile.
* ``verify_eval`` returns OK when stored = actual.
* ``verify_eval`` flags drift when a metric exceeds its tolerance
  band.
* ``verify_eval`` flags ablation-table drift on row mismatches.
* The end-to-end loop (write_reproduce_script + verify_eval) runs
  cleanly on a synthetic run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_HASH64 = "a" * 64


def _build_card(**eval_overrides):
    eval_scores = {
        "faithfulness": 0.80,
        "coverage": 0.70,
        "source_match": 0.50,
        "scoring_commit": "f" * 40,
        "tolerance_band": {
            "accuracy": 0.0,
            "faithfulness": 0.05,
            "hallucination_rate": 0.05,
            "source_match": 0.0,
            "coverage": 0.0,
        },
    }
    eval_scores.update(eval_overrides)
    return {
        "model_id": "rdf-shacl-551-2-qwen2-5-1-5b-deadbeef",
        "course_slug": "rdf-shacl-551-2",
        "base_model": {
            "name": "qwen2.5-1.5b",
            "revision": "main",
            "huggingface_repo": "Qwen/Qwen2.5-1.5B",
        },
        "adapter_format": "safetensors",
        "training_config": {
            "seed": 42, "learning_rate": 2e-4, "epochs": 3,
            "lora_rank": 16, "lora_alpha": 32, "max_seq_length": 2048,
            "batch_size": 4,
        },
        "provenance": {
            "chunks_hash": _HASH64,
            "pedagogy_graph_hash": _HASH64,
            "instruction_pairs_hash": _HASH64,
            "preference_pairs_hash": _HASH64,
            "concept_graph_hash": _HASH64,
            "vocabulary_ttl_hash": _HASH64,
            "holdout_graph_hash": _HASH64,
        },
        "created_at": "2026-04-26T18:00:00Z",
        "eval_scores": eval_scores,
    }


def _build_eval_report(**overrides):
    base = {
        "faithfulness": 0.80,
        "coverage": 0.70,
        "source_match": 0.50,
        "metrics": {
            "hallucination_rate": 0.20,
            "source_match": 0.50,
        },
        "profile": "rdf_shacl",
    }
    base.update(overrides)
    return base


def test_write_reproduce_script_pins_commit_and_model(tmp_path):
    from Trainforge.eval.reproducibility import write_reproduce_script

    card = _build_card()
    script = write_reproduce_script(
        run_dir=tmp_path,
        model_card=card,
    )
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    # All four pinned values land in the body.
    assert ("f" * 40) in text
    assert "rdf-shacl-551-2-qwen2-5-1-5b-deadbeef" in text
    assert "rdf-shacl-551-2" in text
    assert "verify_eval" in text


def test_write_reproduce_script_falls_back_when_no_commit(tmp_path):
    from Trainforge.eval.reproducibility import write_reproduce_script

    card = _build_card()
    # Drop scoring_commit; the writer should still produce a script.
    card["eval_scores"].pop("scoring_commit")
    script = write_reproduce_script(
        run_dir=tmp_path,
        model_card=card,
        commit_sha="abcdef",
    )
    text = script.read_text(encoding="utf-8")
    assert "abcdef" in text


def test_verify_eval_passes_when_stored_matches_actual(tmp_path):
    from Trainforge.eval.verify_eval import verify

    card_path = tmp_path / "model_card.json"
    eval_path = tmp_path / "eval_report.json"
    card_path.write_text(json.dumps(_build_card()), encoding="utf-8")
    eval_path.write_text(json.dumps(_build_eval_report()), encoding="utf-8")

    passed, drift = verify(
        model_card_path=card_path,
        eval_report_path=eval_path,
    )
    assert passed is True
    assert drift == []


def test_verify_eval_flags_drift_outside_tolerance(tmp_path):
    from Trainforge.eval.verify_eval import verify

    card_path = tmp_path / "model_card.json"
    eval_path = tmp_path / "eval_report.json"
    card = _build_card()
    card["eval_scores"]["tolerance_band"]["faithfulness"] = 0.01
    card_path.write_text(json.dumps(card), encoding="utf-8")
    # Actual faithfulness is 0.5; stored is 0.8; tolerance is 0.01.
    eval_path.write_text(
        json.dumps(_build_eval_report(faithfulness=0.5)),
        encoding="utf-8",
    )
    passed, drift = verify(
        model_card_path=card_path,
        eval_report_path=eval_path,
    )
    assert passed is False
    assert any("faithfulness" in line for line in drift)


def test_verify_eval_flags_ablation_table_row_count_drift(tmp_path):
    from Trainforge.eval.verify_eval import verify

    card = _build_card()
    card["eval_scores"]["headline_table"] = [
        {"setup": "base", "accuracy": 0.4, "faithfulness": 0.5,
         "hallucination_rate": 0.5, "source_match": 0.1},
        {"setup": "adapter", "accuracy": 0.7, "faithfulness": 0.8,
         "hallucination_rate": 0.2, "source_match": 0.5},
    ]
    card_path = tmp_path / "model_card.json"
    eval_path = tmp_path / "eval_report.json"
    abl_path = tmp_path / "ablation_report.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    eval_path.write_text(json.dumps(_build_eval_report()), encoding="utf-8")
    # Actual ablation has only 1 row -> count drift
    abl_path.write_text(json.dumps({
        "headline_table": [
            {"setup": "base", "accuracy": 0.4, "faithfulness": 0.5,
             "hallucination_rate": 0.5, "source_match": 0.1},
        ],
        "retrieval_method_table": [],
    }), encoding="utf-8")
    passed, drift = verify(
        model_card_path=card_path,
        eval_report_path=eval_path,
        ablation_report_path=abl_path,
    )
    assert passed is False
    assert any("row count" in line for line in drift)


def test_end_to_end_reproduce_then_verify(tmp_path):
    """Synthetic run dir round-trip: write the script + run the verifier."""
    from Trainforge.eval.reproducibility import write_reproduce_script
    from Trainforge.eval.verify_eval import verify

    card = _build_card()
    card_path = tmp_path / "model_card.json"
    eval_path = tmp_path / "eval_report.json"
    abl_path = tmp_path / "ablation_report.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    eval_path.write_text(json.dumps(_build_eval_report()), encoding="utf-8")
    abl_path.write_text(json.dumps({
        "headline_table": [],
        "retrieval_method_table": [],
    }), encoding="utf-8")

    script = write_reproduce_script(run_dir=tmp_path, model_card=card)
    assert script.exists()
    passed, drift = verify(
        model_card_path=card_path,
        eval_report_path=eval_path,
        ablation_report_path=abl_path,
    )
    assert passed, f"unexpected drift: {drift}"


def test_verify_eval_main_cli_exit_code(tmp_path):
    """The CLI entry point exits 0 on OK and 1 on drift."""
    from Trainforge.eval.verify_eval import main

    card = _build_card()
    card_path = tmp_path / "model_card.json"
    eval_path = tmp_path / "eval_report.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    eval_path.write_text(json.dumps(_build_eval_report()), encoding="utf-8")

    rc = main([
        "--model-card", str(card_path),
        "--eval-report", str(eval_path),
    ])
    assert rc == 0

    # Now corrupt the eval report so faithfulness drifts past tol.
    eval_path.write_text(json.dumps(_build_eval_report(faithfulness=0.0)),
                         encoding="utf-8")
    rc_drift = main([
        "--model-card", str(card_path),
        "--eval-report", str(eval_path),
    ])
    assert rc_drift == 1
