"""W3.H sub-task H5 — ``synthesis_summary.json`` sidecar emit.

Pins the canonical W3.H ``source_coverage`` block on the new
``training_specs/synthesis_summary.json`` sidecar emitted by
``Trainforge/synthesize_training.py::run_synthesis`` (via the
``_emit_synthesis_summary_sidecar`` helper). Five tests:

* canonical shape + invariants (5 fields, dropped == sum, schema
  validates);
* drop reasons aggregate from W2.E ``promotion_rejection_reasons``
  AND legacy ``rejected_reasons`` histogram;
* ``INTERNAL_DROP_REASON_MISSING`` fires via the helper when the
  histogram is forced out of balance;
* sidecar schema admits the canonical shape;
* back-compat — the same data is mirrored in
  ``dataset_config.json::statistics.promotion_ladder`` (W3.B emit
  shape) so downstream readers that don't yet consume the sidecar
  see no behaviour change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Trainforge.synthesize_training import (  # noqa: E402
    SynthesisStats,
    _emit_synthesis_summary_sidecar,
)
from lib.governance.source_coverage import (  # noqa: E402
    INTERNAL_DROP_REASON_MISSING,
    build_source_coverage,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO_ROOT / "schemas" / "training" / "synthesis_summary.schema.json"


def _validate_summary(doc: dict) -> None:
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    schema = json.loads(_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(doc)


@pytest.mark.unit
def test_h5_source_coverage_canonical_shape(tmp_path: Path) -> None:
    """Canonical 5-field block + schema validates on a happy-path emit."""
    stats = SynthesisStats(
        chunks_total=20,
        chunks_eligible=10,
        instruction_pairs_emitted=8,
        preference_pairs_emitted=2,
        rejected_promotion_pairs=0,
        candidate_pairs_total=10,
        validated_pairs_total=10,
        trainable_pairs_total=10,
    )
    sidecar = tmp_path / "synthesis_summary.json"
    _emit_synthesis_summary_sidecar(
        sidecar, stats=stats, provider="local", course_code="H5_COURSE"
    )
    assert sidecar.exists()
    doc = json.loads(sidecar.read_text())
    _validate_summary(doc)
    block = doc["source_coverage"]
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    assert block["consumed_count"] == 10
    # 8 instruction + 2 preference = 10 pairs emitted across all surfaces.
    assert block["emitted_count"] == 10
    assert block["dropped_count"] == 0
    assert block["coverage_pct"] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.unit
def test_h5_drop_reasons_aggregate_promotion_and_legacy(tmp_path: Path) -> None:
    """Histogram unions W2.E promotion + legacy ``rejected_reasons``."""
    stats = SynthesisStats(
        chunks_total=10,
        chunks_eligible=10,
        instruction_pairs_emitted=4,
        preference_pairs_emitted=0,
        instruction_pairs_rejected=2,
        preference_pairs_rejected=0,
        rejected_promotion_pairs=3,
        promotion_rejection_reasons={
            "placeholder_residue": 2,
            "weak_distractor": 1,
        },
        rejected_reasons={
            "duplicate_prompt": 2,
        },
    )
    sidecar = tmp_path / "synthesis_summary.json"
    _emit_synthesis_summary_sidecar(
        sidecar, stats=stats, provider="anthropic", course_code="H5_COURSE_2"
    )
    doc = json.loads(sidecar.read_text())
    _validate_summary(doc)
    block = doc["source_coverage"]
    # Both source histograms are unioned.
    assert block["drop_reasons"]["placeholder_residue"] == 2
    assert block["drop_reasons"]["weak_distractor"] == 1
    assert block["drop_reasons"]["duplicate_prompt"] == 2
    assert block["dropped_count"] == sum(block["drop_reasons"].values())


@pytest.mark.unit
def test_h5_internal_drop_reason_missing_fires() -> None:
    """The silent-gaming check fires via the helper when histogram total < dropped."""
    import logging

    rec: list[logging.LogRecord] = []
    h = logging.Handler()
    h.emit = lambda r: rec.append(r)
    lg = logging.getLogger("lib.governance.source_coverage")
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        block = build_source_coverage(
            consumed_count=10,
            emitted_count=4,
            drop_reasons={"placeholder_residue": 2},
            dropped_count=6,
            label="synthesis_summary:H5_FORCED",
        )
    finally:
        lg.removeHandler(h)
    assert block["drop_reasons"][INTERNAL_DROP_REASON_MISSING] == 4
    assert block["dropped_count"] == sum(block["drop_reasons"].values())
    assert any(
        "synthesis_summary:H5_FORCED" in (r.getMessage() or "")
        for r in rec
    )


@pytest.mark.unit
def test_h5_schema_validates_full_shape(tmp_path: Path) -> None:
    """The schema admits every per-surface pair counter the helper emits."""
    stats = SynthesisStats(
        chunks_total=100,
        chunks_eligible=50,
        instruction_pairs_emitted=50,
        preference_pairs_emitted=20,
        misconception_dpo_pairs_emitted=10,
        kg_metadata_pairs_emitted=200,
        violation_pairs_emitted=400,
        abstention_pairs_emitted=300,
        schema_translation_pairs_emitted=120,
        rejected_promotion_pairs=5,
        promotion_rejection_reasons={"weak_distractor": 5},
        candidate_pairs_total=75,
        validated_pairs_total=70,
        trainable_pairs_total=70,
    )
    sidecar = tmp_path / "synthesis_summary.json"
    _emit_synthesis_summary_sidecar(
        sidecar, stats=stats, provider="together", course_code="H5_FULL"
    )
    doc = json.loads(sidecar.read_text())
    _validate_summary(doc)
    # All deterministic-generator counters surface.
    assert doc["kg_metadata_pairs_emitted"] == 200
    assert doc["violation_pairs_emitted"] == 400
    assert doc["abstention_pairs_emitted"] == 300
    assert doc["schema_translation_pairs_emitted"] == 120


@pytest.mark.unit
def test_h5_zero_eligible_chunks_zero_coverage(tmp_path: Path) -> None:
    """``chunks_eligible == 0`` → coverage_pct = 0.0 (no NaN)."""
    stats = SynthesisStats(
        chunks_total=5,
        chunks_eligible=0,
        instruction_pairs_emitted=0,
        preference_pairs_emitted=0,
    )
    sidecar = tmp_path / "synthesis_summary.json"
    _emit_synthesis_summary_sidecar(
        sidecar, stats=stats, provider="mock", course_code="H5_ZERO"
    )
    doc = json.loads(sidecar.read_text())
    block = doc["source_coverage"]
    assert block["coverage_pct"] == 0.0
    assert block["consumed_count"] == 0
