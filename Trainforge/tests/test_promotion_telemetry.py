"""Worker W3.B — promotion-ladder telemetry round-trip tests.

Round-trips the promotion-ladder counters from
:class:`SynthesisStats` through to:

* ``dataset_config.json::statistics.promotion_ladder``
* the W2.B aggregator's
  ``phase_results.synthesis.promotion_ladder`` block (read-back via
  :class:`TrainforgeAssessmentQualityReport._read_promotion_ladder`)

Coverage matrix (per the W3.B plan):

* Round-trip: SynthesisStats -> dataset_config -> aggregator. Build a
  fixture run with N candidate pairs, M validated, the rest rejected
  by reason. Assert each sink carries the same numbers.
* Empty-promotion-ladder edge case (zero pairs). Should emit zeros,
  not omit the block.
* Reason-key alphabetisation invariant (so a future histogram diff
  is byte-stable).
* SynthesisStats.as_dict() projection asserts (5 new fields surface).

These tests exercise the wire-shape directly without spinning up
the full ``run_synthesis`` LLM pipeline. The per-pair filter site
is unit-tested in
:mod:`lib.validators.tests.test_training_pair_promotion`; this file
asserts the counters the filter site populates flow through the
telemetry surfaces unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from Trainforge.synthesis.synthesize_training import (
    SynthesisStats,
    _update_dataset_config,
)
from lib.aggregators.trainforge_assessment_quality_report import (
    TrainforgeAssessmentQualityReport,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

# Canonical 7 rejection reasons mirrored from
# lib/validators/training_pair_promotion.py. Pinned here so the test
# alphabetisation fixture remains stable when the validator gains an
# 8th reason.
_CANONICAL_REJECTION_REASONS = (
    "placeholder_residue",
    "unsupported_answer",
    "weak_distractor",
    "unanswerable_stem",
    "source_free_generation",
    "low_bloom_alignment",
    "generic_rationale",
)


def _stats_with_ladder(
    *,
    candidate: int,
    validated: int,
    trainable: int,
    rejection_reasons: Dict[str, int],
) -> SynthesisStats:
    """Build a SynthesisStats with promotion-ladder counters preset.

    The ladder invariants must hold on input:
        candidate == validated + sum(rejection_reasons.values())
        rejected_promotion_pairs == sum(rejection_reasons.values())
    The fixture asserts both so a typo never silently propagates.
    """
    rejected = sum(rejection_reasons.values())
    assert candidate == validated + rejected, (
        f"fixture invariant broken: candidate={candidate}, "
        f"validated={validated}, rejected={rejected}"
    )
    stats = SynthesisStats(chunks_total=candidate)
    stats.candidate_pairs_total = candidate
    stats.validated_pairs_total = validated
    stats.trainable_pairs_total = trainable
    stats.rejected_promotion_pairs = rejected
    stats.promotion_rejection_reasons = dict(rejection_reasons)
    return stats


def _seed_dataset_config(
    trainforge_dir: Path,
    stats: SynthesisStats,
) -> Path:
    """Run the synthesize_training emit path against a real path."""
    training_specs = trainforge_dir / "training_specs"
    training_specs.mkdir(parents=True, exist_ok=True)
    config_path = training_specs / "dataset_config.json"
    _update_dataset_config(config_path, stats)
    return config_path


# -------------------------------------------------------------------------
# 1. SynthesisStats.as_dict() projects the 5 new ladder fields
# -------------------------------------------------------------------------


class TestSynthesisStatsProjection:
    """The 5 new fields land in as_dict() with expected wire shape."""

    def test_default_stats_emits_zero_ladder(self):
        stats = SynthesisStats()
        payload = stats.as_dict()
        for field in (
            "candidate_pairs_total",
            "validated_pairs_total",
            "trainable_pairs_total",
            "rejected_promotion_pairs",
            "promotion_rejection_reasons",
        ):
            assert field in payload, f"missing as_dict field: {field}"
        assert payload["candidate_pairs_total"] == 0
        assert payload["validated_pairs_total"] == 0
        assert payload["trainable_pairs_total"] == 0
        assert payload["rejected_promotion_pairs"] == 0
        assert payload["promotion_rejection_reasons"] == {}

    def test_populated_ladder_round_trips_through_as_dict(self):
        stats = _stats_with_ladder(
            candidate=10,
            validated=7,
            trainable=7,
            rejection_reasons={
                "placeholder_residue": 2,
                "weak_distractor": 1,
            },
        )
        payload = stats.as_dict()
        assert payload["candidate_pairs_total"] == 10
        assert payload["validated_pairs_total"] == 7
        assert payload["trainable_pairs_total"] == 7
        assert payload["rejected_promotion_pairs"] == 3
        assert payload["promotion_rejection_reasons"] == {
            "placeholder_residue": 2,
            "weak_distractor": 1,
        }

    def test_promotion_rejection_reasons_alphabetised_in_as_dict(self):
        # Insert in reverse-alpha order; assert as_dict re-orders by key.
        reasons_in = {
            "weak_distractor": 1,
            "unanswerable_stem": 2,
            "placeholder_residue": 3,
            "generic_rationale": 4,
        }
        stats = _stats_with_ladder(
            candidate=10,
            validated=0,
            trainable=0,
            rejection_reasons=reasons_in,
        )
        payload = stats.as_dict()
        keys = list(payload["promotion_rejection_reasons"].keys())
        assert keys == sorted(reasons_in), (
            f"expected alphabetised keys; got {keys}"
        )


# -------------------------------------------------------------------------
# 2. dataset_config.json carries statistics.promotion_ladder
# -------------------------------------------------------------------------


class TestDatasetConfigEmit:
    """``_update_dataset_config`` writes the promotion_ladder block."""

    def test_promotion_ladder_block_written(self, tmp_path):
        stats = _stats_with_ladder(
            candidate=12,
            validated=9,
            trainable=9,
            rejection_reasons={
                "placeholder_residue": 1,
                "weak_distractor": 2,
            },
        )
        config_path = _seed_dataset_config(tmp_path, stats)
        assert config_path.exists()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        ladder = payload["statistics"]["promotion_ladder"]
        assert ladder["candidate_pairs_total"] == 12
        assert ladder["validated_pairs_total"] == 9
        assert ladder["trainable_pairs_total"] == 9
        assert ladder["rejected_promotion_pairs"] == 3
        assert ladder["promotion_rejection_reasons"] == {
            "placeholder_residue": 1,
            "weak_distractor": 2,
        }

    def test_zero_pair_run_emits_empty_ladder_block(self, tmp_path):
        # Edge case from the W3.B plan: an empty run must EMIT the
        # block, not OMIT it. Operators reading dataset_config can
        # then distinguish "ran with zero pairs" from "legacy run".
        stats = SynthesisStats()
        config_path = _seed_dataset_config(tmp_path, stats)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        ladder = payload["statistics"]["promotion_ladder"]
        assert ladder == {
            "candidate_pairs_total": 0,
            "validated_pairs_total": 0,
            "trainable_pairs_total": 0,
            "rejected_promotion_pairs": 0,
            "promotion_rejection_reasons": {},
        }

    def test_promotion_ladder_keys_alphabetised_on_emit(self, tmp_path):
        reasons_in = {
            "weak_distractor": 1,
            "placeholder_residue": 2,
            "generic_rationale": 3,
        }
        stats = _stats_with_ladder(
            candidate=6,
            validated=0,
            trainable=0,
            rejection_reasons=reasons_in,
        )
        config_path = _seed_dataset_config(tmp_path, stats)
        # Read via raw text so insertion order is observable
        # (json.loads in CPython preserves insertion order).
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        keys = list(
            payload["statistics"]["promotion_ladder"][
                "promotion_rejection_reasons"
            ].keys()
        )
        assert keys == sorted(reasons_in), (
            f"reason keys not alphabetised in dataset_config: {keys}"
        )

    def test_synthesis_last_run_carries_ladder_too(self, tmp_path):
        # Belt-and-suspenders: the same numbers also land in the
        # ``synthesis.last_run`` block (full SynthesisStats dump) so an
        # operator inspecting either surface gets the same numbers.
        stats = _stats_with_ladder(
            candidate=5,
            validated=4,
            trainable=4,
            rejection_reasons={"unsupported_answer": 1},
        )
        config_path = _seed_dataset_config(tmp_path, stats)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        last_run = payload["synthesis"]["last_run"]
        assert last_run["candidate_pairs_total"] == 5
        assert last_run["validated_pairs_total"] == 4
        assert last_run["trainable_pairs_total"] == 4
        assert last_run["rejected_promotion_pairs"] == 1
        assert last_run["promotion_rejection_reasons"] == {
            "unsupported_answer": 1,
        }


# -------------------------------------------------------------------------
# 3. Round-trip: SynthesisStats -> dataset_config -> aggregator
# -------------------------------------------------------------------------


class TestAggregatorRoundTrip:
    """The W2.B aggregator copies the ladder through unchanged."""

    def test_full_round_trip_carries_same_numbers(self, tmp_path):
        stats = _stats_with_ladder(
            candidate=20,
            validated=14,
            trainable=14,
            rejection_reasons={
                "placeholder_residue": 2,
                "unsupported_answer": 1,
                "weak_distractor": 1,
                "unanswerable_stem": 1,
                "source_free_generation": 1,
            },
        )
        _seed_dataset_config(tmp_path, stats)

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C",
            run_id="WF-RT",
            trainforge_dir=tmp_path,
        )
        report = agg.build()
        ladder = report["phase_results"]["synthesis"]["promotion_ladder"]
        assert ladder["candidate_pairs_total"] == 20
        assert ladder["validated_pairs_total"] == 14
        assert ladder["trainable_pairs_total"] == 14
        assert ladder["rejected_promotion_pairs"] == 6
        assert ladder["promotion_rejection_reasons"] == {
            "placeholder_residue": 2,
            "source_free_generation": 1,
            "unanswerable_stem": 1,
            "unsupported_answer": 1,
            "weak_distractor": 1,
        }

    def test_aggregator_emits_empty_ladder_when_dataset_config_missing(
        self, tmp_path,
    ):
        # No dataset_config seeded; aggregator returns the empty-shape
        # placeholder so downstream consumers see a uniform wire shape.
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C",
            run_id="WF-EMPTY",
            trainforge_dir=tmp_path,
        )
        report = agg.build()
        ladder = report["phase_results"]["synthesis"]["promotion_ladder"]
        assert ladder == {
            "candidate_pairs_total": 0,
            "validated_pairs_total": 0,
            "trainable_pairs_total": 0,
            "rejected_promotion_pairs": 0,
            "promotion_rejection_reasons": {},
        }

    def test_aggregator_alphabetises_reason_keys(self, tmp_path):
        # Even if dataset_config emit order regressed, the aggregator's
        # read-back path re-sorts the histogram keys.
        ts = tmp_path / "training_specs"
        ts.mkdir()
        (ts / "dataset_config.json").write_text(
            json.dumps({
                "format": "instruction-following",
                "statistics": {
                    "promotion_ladder": {
                        "candidate_pairs_total": 6,
                        "validated_pairs_total": 0,
                        "trainable_pairs_total": 0,
                        "rejected_promotion_pairs": 6,
                        # Insertion order intentionally non-alpha.
                        "promotion_rejection_reasons": {
                            "weak_distractor": 1,
                            "placeholder_residue": 2,
                            "generic_rationale": 3,
                        },
                    },
                },
            }),
            encoding="utf-8",
        )

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C",
            run_id="WF-A",
            trainforge_dir=tmp_path,
        )
        report = agg.build()
        keys = list(
            report["phase_results"]["synthesis"]["promotion_ladder"][
                "promotion_rejection_reasons"
            ].keys()
        )
        assert keys == [
            "generic_rationale",
            "placeholder_residue",
            "weak_distractor",
        ], f"reason keys not alphabetised on read-back: {keys}"


# -------------------------------------------------------------------------
# 4. Canonical-reason coverage sanity test
# -------------------------------------------------------------------------


class TestCanonicalReasonCoverage:
    """All 7 canonical rejection reasons round-trip through the surfaces."""

    def test_all_seven_reasons_round_trip(self, tmp_path):
        rejection_reasons = {
            reason: idx + 1
            for idx, reason in enumerate(_CANONICAL_REJECTION_REASONS)
        }
        total_rejected = sum(rejection_reasons.values())
        stats = _stats_with_ladder(
            candidate=total_rejected + 5,
            validated=5,
            trainable=5,
            rejection_reasons=rejection_reasons,
        )
        _seed_dataset_config(tmp_path, stats)

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C",
            run_id="WF-CRC",
            trainforge_dir=tmp_path,
        )
        report = agg.build()
        ladder = report["phase_results"]["synthesis"]["promotion_ladder"]
        for reason in _CANONICAL_REJECTION_REASONS:
            assert reason in ladder["promotion_rejection_reasons"], (
                f"missing canonical reason: {reason}"
            )
            assert (
                ladder["promotion_rejection_reasons"][reason]
                == rejection_reasons[reason]
            )
        # Histogram sum invariant.
        assert (
            sum(ladder["promotion_rejection_reasons"].values())
            == ladder["rejected_promotion_pairs"]
        )
