"""Wave 137d-2: tests for the form_data coverage helper +
EvalGatingValidator checkpoint emission.

Nine tests pin the contract:

1. ``test_compute_coverage_metrics_all_complete`` — synthetic
   all-complete form_data => coverage 1.0.
2. ``test_compute_coverage_metrics_all_degraded`` => coverage 0.0.
3. ``test_compute_coverage_metrics_partial`` — half-complete => 0.5.
4. ``test_compute_coverage_metrics_no_family_map_returns_empty_map`` —
   absent family map => empty family_coverage_map dict.
5. ``test_eval_gating_emits_checkpoint_row_on_pass`` — passing gate
   appends a row with promotion_decision="passed".
6. ``test_eval_gating_emits_checkpoint_row_on_block`` — blocked gate
   appends a row with promotion_decision="blocked" + block_reasons.
7. ``test_eval_gating_checkpoint_failure_does_not_break_gate`` —
   monkey-patched _emit_coverage_checkpoint raising still produces a
   well-formed GateResult.
8. ``test_checkpoint_jsonl_is_append_only`` — two consecutive
   validate() calls produce two rows in the JSONL.
9. ``test_checkpoint_row_schema`` — required keys present, types
   match.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from lib.validators.eval_gating import EvalGatingValidator
from lib.validators.form_data_coverage import compute_coverage_metrics


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_property(curie: str) -> SimpleNamespace:
    return SimpleNamespace(curie=curie)


def _make_manifest(curies: List[str], family: str = "synthetic") -> SimpleNamespace:
    return SimpleNamespace(
        family=family,
        properties=[_make_property(c) for c in curies],
    )


def _make_form_entry(status: str = "complete") -> SimpleNamespace:
    return SimpleNamespace(anchored_status=status)


def _write_eval_report(model_dir: Path, **fields: Any) -> Path:
    """Mirror the helper from test_eval_gating_validator.py."""
    report: Dict[str, Any] = {
        "faithfulness": 0.80,
        "coverage": 0.90,
        "profile": "rdf_shacl",
        "per_tier": {},
        "per_invariant": {},
        "baseline_delta": 0.10,
        "source_match": 0.65,
        "negative_grounding_accuracy": 0.70,
        "yes_rate": 0.55,
        "metrics": {"hallucination_rate": 0.20},
    }
    report.update(fields)
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / "eval_report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# 1. all-complete => coverage 1.0
# ----------------------------------------------------------------------


def test_compute_coverage_metrics_all_complete():
    curies = ["test:A", "test:B", "test:C"]
    manifest = _make_manifest(curies)
    form_data = {c: _make_form_entry("complete") for c in curies}
    metrics = compute_coverage_metrics(
        "synthetic", form_data=form_data, manifest=manifest,
    )
    assert metrics["manifest_coverage_pct"] == 1.0
    assert metrics["complete_count"] == 3
    assert metrics["degraded_count"] == 0


# ----------------------------------------------------------------------
# 2. all-degraded => coverage 0.0
# ----------------------------------------------------------------------


def test_compute_coverage_metrics_all_degraded():
    curies = ["test:A", "test:B", "test:C"]
    manifest = _make_manifest(curies)
    form_data = {c: _make_form_entry("degraded_placeholder") for c in curies}
    metrics = compute_coverage_metrics(
        "synthetic", form_data=form_data, manifest=manifest,
    )
    assert metrics["manifest_coverage_pct"] == 0.0
    assert metrics["complete_count"] == 0
    assert metrics["degraded_count"] == 3


# ----------------------------------------------------------------------
# 3. half-complete => coverage 0.5
# ----------------------------------------------------------------------


def test_compute_coverage_metrics_partial():
    curies = ["test:A", "test:B", "test:C", "test:D"]
    manifest = _make_manifest(curies)
    form_data = {
        "test:A": _make_form_entry("complete"),
        "test:B": _make_form_entry("complete"),
        "test:C": _make_form_entry("degraded_placeholder"),
        "test:D": _make_form_entry("degraded_placeholder"),
    }
    metrics = compute_coverage_metrics(
        "synthetic", form_data=form_data, manifest=manifest,
    )
    assert metrics["manifest_coverage_pct"] == 0.5
    assert metrics["complete_count"] == 2
    assert metrics["degraded_count"] == 2


# ----------------------------------------------------------------------
# 4. No family map => empty family_coverage_map dict.
# ----------------------------------------------------------------------


def test_compute_coverage_metrics_no_family_map_returns_empty_map():
    """Synthetic family slug => no family_map.synthetic.yaml on disk =>
    family_coverage_map must be an empty dict, not ``None``."""
    curies = ["test:A", "test:B"]
    manifest = _make_manifest(curies)
    form_data = {c: _make_form_entry("complete") for c in curies}
    metrics = compute_coverage_metrics(
        "synthetic", form_data=form_data, manifest=manifest,
    )
    assert metrics["family_coverage_map"] == {}


# ----------------------------------------------------------------------
# 5. Eval gating emits checkpoint row on pass.
# ----------------------------------------------------------------------


def test_eval_gating_emits_checkpoint_row_on_pass(tmp_path: Path) -> None:
    course_root = tmp_path / "rdf-shacl-551-2"
    model_dir = course_root / "models" / "test-v1"
    model_dir.mkdir(parents=True)
    _write_eval_report(model_dir)

    # Stub the manifest + coverage helper so the test isn't coupled
    # to the real rdf_shacl manifest contents.
    fake_manifest = _make_manifest(["test:A"], family="synthetic")
    fake_metrics = {
        "manifest_coverage_pct": 0.50,
        "complete_count": 5,
        "degraded_count": 5,
        "family_coverage_map": {"cardinality": {"complete": 1, "total": 2,
                                                "status": "partial",
                                                "curies": ["test:A", "test:B"]}},
    }

    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=fake_manifest,
    ), patch(
        "lib.validators.form_data_coverage.compute_coverage_metrics",
        return_value=fake_metrics,
    ):
        result = EvalGatingValidator().validate({"model_dir": str(model_dir)})

    assert result.passed is True

    checkpoint_path = course_root / "eval" / "form_data_coverage_checkpoint.jsonl"
    assert checkpoint_path.exists()
    rows = [json.loads(l) for l in checkpoint_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["promotion_decision"] == "passed"
    assert row["promotion_block_reasons"] == []
    assert row["model_id"] == "test-v1"
    assert row["course_slug"] == "rdf-shacl-551-2"
    assert row["family"] == "synthetic"
    assert row["manifest_coverage_pct"] == 0.50


# ----------------------------------------------------------------------
# 6. Eval gating emits checkpoint row on block.
# ----------------------------------------------------------------------


def test_eval_gating_emits_checkpoint_row_on_block(tmp_path: Path) -> None:
    course_root = tmp_path / "rdf-shacl-551-2"
    model_dir = course_root / "models" / "test-v2"
    model_dir.mkdir(parents=True)
    # Faithfulness below threshold => critical block.
    _write_eval_report(model_dir, faithfulness=0.10)

    fake_manifest = _make_manifest(["test:A"], family="synthetic")
    fake_metrics = {
        "manifest_coverage_pct": 0.10,
        "complete_count": 1,
        "degraded_count": 9,
        "family_coverage_map": {},
    }

    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=fake_manifest,
    ), patch(
        "lib.validators.form_data_coverage.compute_coverage_metrics",
        return_value=fake_metrics,
    ):
        result = EvalGatingValidator().validate({"model_dir": str(model_dir)})

    assert result.passed is False

    checkpoint_path = course_root / "eval" / "form_data_coverage_checkpoint.jsonl"
    rows = [json.loads(l) for l in checkpoint_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["promotion_decision"] == "blocked"
    assert "EVAL_FAITHFULNESS_BELOW_THRESHOLD" in row["promotion_block_reasons"]


# ----------------------------------------------------------------------
# 7. Checkpoint failure does not break the gate.
# ----------------------------------------------------------------------


def test_eval_gating_checkpoint_failure_does_not_break_gate(
    tmp_path: Path,
) -> None:
    """A raise inside _emit_coverage_checkpoint must NOT propagate —
    the gate result is canonical regardless of checkpoint write
    failures."""
    course_root = tmp_path / "rdf-shacl-551-2"
    model_dir = course_root / "models" / "test-v3"
    model_dir.mkdir(parents=True)
    _write_eval_report(model_dir)

    def _raise(self, **kwargs):
        raise RuntimeError("synthetic checkpoint write failure")

    with patch.object(
        EvalGatingValidator, "_emit_coverage_checkpoint", _raise,
    ):
        result = EvalGatingValidator().validate({"model_dir": str(model_dir)})

    # Gate result is the canonical pass — the checkpoint failure is
    # logged but never breaks the result.
    assert result.passed is True


# ----------------------------------------------------------------------
# 8. JSONL is append-only.
# ----------------------------------------------------------------------


def test_checkpoint_jsonl_is_append_only(tmp_path: Path) -> None:
    course_root = tmp_path / "rdf-shacl-551-2"
    model_dir = course_root / "models" / "test-v4"
    model_dir.mkdir(parents=True)
    _write_eval_report(model_dir)

    fake_manifest = _make_manifest(["test:A"], family="synthetic")
    fake_metrics = {
        "manifest_coverage_pct": 0.30,
        "complete_count": 3,
        "degraded_count": 7,
        "family_coverage_map": {},
    }

    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=fake_manifest,
    ), patch(
        "lib.validators.form_data_coverage.compute_coverage_metrics",
        return_value=fake_metrics,
    ):
        # Two validate() calls.
        EvalGatingValidator().validate({"model_dir": str(model_dir)})
        EvalGatingValidator().validate({"model_dir": str(model_dir)})

    checkpoint_path = course_root / "eval" / "form_data_coverage_checkpoint.jsonl"
    rows = [json.loads(l) for l in checkpoint_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    # Both rows should carry the same model_id / course_slug.
    assert rows[0]["model_id"] == rows[1]["model_id"] == "test-v4"
    assert rows[0]["course_slug"] == rows[1]["course_slug"] == "rdf-shacl-551-2"


# ----------------------------------------------------------------------
# 9. Row schema: required keys present, types match.
# ----------------------------------------------------------------------


def test_checkpoint_row_schema(tmp_path: Path) -> None:
    course_root = tmp_path / "rdf-shacl-551-2"
    model_dir = course_root / "models" / "test-v5"
    model_dir.mkdir(parents=True)
    _write_eval_report(model_dir)

    fake_manifest = _make_manifest(["test:A", "test:B"], family="synthetic")
    fake_metrics = {
        "manifest_coverage_pct": 0.75,
        "complete_count": 3,
        "degraded_count": 1,
        "family_coverage_map": {
            "cardinality": {
                "complete": 2,
                "total": 3,
                "status": "partial",
                "curies": ["test:A", "test:B", "test:C"],
            }
        },
    }

    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=fake_manifest,
    ), patch(
        "lib.validators.form_data_coverage.compute_coverage_metrics",
        return_value=fake_metrics,
    ):
        EvalGatingValidator().validate({"model_dir": str(model_dir)})

    checkpoint_path = course_root / "eval" / "form_data_coverage_checkpoint.jsonl"
    rows = [json.loads(l) for l in checkpoint_path.read_text().splitlines() if l.strip()]
    row = rows[0]
    required_keys = {
        "timestamp",
        "model_id",
        "course_slug",
        "family",
        "manifest_coverage_pct",
        "complete_count",
        "degraded_count",
        "family_coverage_map",
        "promotion_decision",
        "promotion_block_reasons",
    }
    assert required_keys.issubset(row.keys()), (
        f"missing keys: {required_keys - set(row.keys())}"
    )
    # Type sanity.
    assert isinstance(row["timestamp"], str)
    assert isinstance(row["model_id"], str)
    assert isinstance(row["course_slug"], str)
    assert isinstance(row["family"], str)
    assert isinstance(row["manifest_coverage_pct"], float)
    assert isinstance(row["complete_count"], int)
    assert isinstance(row["degraded_count"], int)
    assert isinstance(row["family_coverage_map"], dict)
    assert row["promotion_decision"] in ("passed", "blocked")
    assert isinstance(row["promotion_block_reasons"], list)
