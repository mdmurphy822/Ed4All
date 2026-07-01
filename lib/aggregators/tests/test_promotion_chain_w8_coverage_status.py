"""W8.5 / W8.6 regression tests for the promotion-chain aggregator.

W8.5 — a NON-training run (``--skip-training`` / ``--stop-after
imscc_chunking``) legitimately skips arrows 6-9. The derived course_status
must NOT be forced to ``failed`` when the completed arrows (1-5) certified,
and the derived status must be stamped onto the LibV2 course manifest.

W8.6 — the chain reads source-coverage (per-arrow + coverage_map) and
surfaces a warning-day-1 ``COVERAGE_DROP`` signal below a floor; the stricter
gating is opt-in behind ``ED4ALL_COVERAGE_DROP_STRICT`` and default-preserving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.aggregators.promotion_chain_report import (
    COVERAGE_DROP,
    MISSING_STAGE_REPORT,
    PromotionChainAggregator,
    _resolve_coverage_floor,
    _resolve_coverage_strict,
)
from lib.governance.course_status import derive_course_status

# Reuse the canonical fixture builder from the sibling suite.
from lib.aggregators.tests.test_promotion_chain_report import (  # noqa: E402
    _build_full_layout,
    _coverage_block,
    _write_json,
)


def _strip_training_arrows(layout: Dict[str, Path]) -> None:
    """Delete the assessment/training arrow artifacts (arrows 6-9).

    Leaves arrows 1-5 (staging, dart chunks, rewrite, packaging, imscc
    chunks) intact so the run looks exactly like a non-training build that
    stopped after imscc_chunking.
    """
    course = layout["course_dir"]
    for rel in (
        "training_specs/assessments.json",
        "training_specs/synthesis_summary.json",
        "training_specs/dataset_config.json",
    ):
        p = course / rel
        if p.exists():
            p.unlink()
    models = course / "models"
    if models.exists():
        import shutil

        shutil.rmtree(models)


# ---------------------------------------------------------------------------
# W8.5 — non-training run course_status
# ---------------------------------------------------------------------------


class TestW85NonTrainingCourseStatus:
    def test_non_training_run_certifies_not_failed(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        _strip_training_arrows(layout)
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-NONTRAIN",
            # No training phase keys in phase_outputs -> non-training scope.
            phase_outputs={},
        )
        report = agg.build()

        # Arrows 6-9 are missing-stage-report sentinels...
        for aid in (6, 7, 8, 9):
            arrow = next(a for a in report["arrows"] if a["arrow_id"] == aid)
            assert MISSING_STAGE_REPORT in arrow["validator_set"]
        # ...yet the course is NOT failed: arrows 1-5 certified the
        # accessible + instructional tiers.
        assert report["course_status"] == "certified_instructional"

    def test_training_run_missing_report_still_failed(self, tmp_path):
        # A run that DECLARED training (phase key present) but is missing the
        # arrow-7 report is a genuine regression -> failed (strict).
        layout = _build_full_layout(tmp_path)
        (layout["course_dir"] / "training_specs" / "synthesis_summary.json").unlink()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TRAIN-BROKEN",
            phase_outputs={"training_synthesis": {"_completed": True}},
        )
        report = agg.build()
        arrow_7 = next(a for a in report["arrows"] if a["arrow_id"] == 7)
        assert MISSING_STAGE_REPORT in arrow_7["validator_set"]
        assert report["course_status"] == "failed"

    def test_non_training_run_missing_core_arrow_still_failed(self, tmp_path):
        # Even on a non-training run, a missing CORE arrow (1-5) is a real
        # failure and must still short-circuit to failed.
        layout = _build_full_layout(tmp_path)
        _strip_training_arrows(layout)
        (layout["course_dir"] / "imscc_chunks" / "manifest.json").unlink()  # arrow 5
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-NONTRAIN-BROKEN",
            phase_outputs={},
        )
        report = agg.build()
        assert report["course_status"] == "failed"

    def test_derive_course_status_training_expected_default_is_strict(self):
        # A missing training arrow with default (None) training_expected still
        # forces failed — byte-identical legacy contract.
        arrows = [
            {
                "arrow_id": i,
                "name": f"arrow{i}",
                "validator_set": (
                    ["wcag_compliance"] if i == 1 else []
                ),
                "passed": True,
                "warnings_count": 0,
                "source_coverage": None,
                "promotion_decision": "pass",
            }
            for i in range(1, 6)
        ] + [
            {
                "arrow_id": i,
                "name": f"arrow{i}",
                "validator_set": [MISSING_STAGE_REPORT],
                "passed": None,
                "warnings_count": 0,
                "source_coverage": None,
                "promotion_decision": "fail",
            }
            for i in range(6, 10)
        ]
        # Default: strict -> failed.
        assert derive_course_status(arrows) == "failed"
        # training_expected=False -> certified (arrows 1-5 pass, only accessibility gate).
        assert derive_course_status(arrows, training_expected=False) == (
            "certified_accessible"
        )


# ---------------------------------------------------------------------------
# W8.5 — manifest stamping
# ---------------------------------------------------------------------------


class TestW85ManifestStamp:
    def test_course_status_stamped_onto_manifest(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-STAMP",
        )
        report = agg.build()
        manifest = json.loads(
            (layout["course_dir"] / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["course_status"] == report["course_status"]
        assert manifest["quality_metadata"]["final_status"] == (
            report["course_status"]
        )
        # Pre-existing field untouched.
        assert manifest["imscc_sha256"] == "c" * 64

    def test_stamp_is_best_effort_when_manifest_absent(self, tmp_path):
        ghost = tmp_path / "no_course"
        agg = PromotionChainAggregator(
            course_path=ghost,
            course_code="GHOST",
            run_id="WF-GHOST",
        )
        # Must not raise even though there is no manifest to stamp.
        agg.build()


# ---------------------------------------------------------------------------
# W8.6 — COVERAGE_DROP signal
# ---------------------------------------------------------------------------


class TestW86CoverageDrop:
    def test_full_coverage_is_byte_identical_no_signal(self, tmp_path):
        # All fixture coverage is 100% -> no COVERAGE_DROP anywhere.
        layout = _build_full_layout(tmp_path)
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-FULLCOV",
        )
        report = agg.build()
        for arrow in report["arrows"]:
            assert COVERAGE_DROP not in arrow["validator_set"]

    def test_sub_floor_arrow_gets_warning_but_still_passes(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        # Rewrite the arrow-5 imscc manifest with a 1% retention (99% drop).
        _write_json(
            layout["course_dir"] / "imscc_chunks" / "manifest.json",
            {
                "chunks_sha256": "d" * 64,
                "chunker_version": "v4",
                "chunkset_kind": "imscc",
                "source_imscc_sha256": "c" * 64,
                "chunks_count": 1,
                "source_coverage": _coverage_block(100, 1),
            },
        )
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-DROP",
        )
        report = agg.build()
        arrow_5 = next(a for a in report["arrows"] if a["arrow_id"] == 5)
        assert COVERAGE_DROP in arrow_5["validator_set"]
        assert arrow_5["warnings_count"] >= 1
        # Warning-day-1: promotion_decision NOT flipped by default.
        assert arrow_5["promotion_decision"] == "pass"
        # ...and the course still certifies (no verdict change).
        assert report["course_status"] == "certified_trainable"

    def test_strict_flag_flips_drop_to_failed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ED4ALL_COVERAGE_DROP_STRICT", "1")
        layout = _build_full_layout(tmp_path)
        _write_json(
            layout["course_dir"] / "imscc_chunks" / "manifest.json",
            {
                "chunks_sha256": "d" * 64,
                "chunker_version": "v4",
                "chunkset_kind": "imscc",
                "source_imscc_sha256": "c" * 64,
                "chunks_count": 1,
                "source_coverage": _coverage_block(100, 1),
            },
        )
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-DROP-STRICT",
        )
        report = agg.build()
        arrow_5 = next(a for a in report["arrows"] if a["arrow_id"] == 5)
        assert COVERAGE_DROP in arrow_5["validator_set"]
        assert arrow_5["promotion_decision"] == "fail"
        assert report["course_status"] == "failed"

    def test_coverage_map_objective_coverage_drop(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        # 1 of 100 objectives has chunks -> objective coverage 1% -> drop.
        _write_json(
            layout["course_dir"] / "coverage_map.json",
            {
                "schema_version": "1.0",
                "course_code": "TEST",
                "run_id": "WF-TEST",
                "generated_at": "2026-05-06T00:00:00Z",
                "objectives": [],
                "summary": {
                    "total_objectives": 100,
                    "objectives_with_chunks": 1,
                    "objectives_with_questions": 0,
                    "objectives_with_training_pairs": 0,
                    "orphan_objectives": [],
                    "orphan_chunks": [],
                    "orphan_questions": [],
                },
            },
        )
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-OBJDROP",
        )
        report = agg.build()
        arrow_5 = next(a for a in report["arrows"] if a["arrow_id"] == 5)
        assert COVERAGE_DROP in arrow_5["validator_set"]

    def test_decision_capture_surfaces_coverage_rollup(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        _write_json(
            layout["course_dir"] / "imscc_chunks" / "manifest.json",
            {
                "chunks_sha256": "d" * 64,
                "chunker_version": "v4",
                "chunkset_kind": "imscc",
                "source_imscc_sha256": "c" * 64,
                "chunks_count": 1,
                "source_coverage": _coverage_block(100, 1),
            },
        )

        class _Spy:
            def __init__(self):
                self.events = []

            def log_decision(self, **kw):
                self.events.append(kw)

        spy = _Spy()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-COVCAP",
            decision_capture=spy,
        )
        agg.build()
        assert len(spy.events) == 1
        ev = spy.events[0]
        assert "coverage_drop_count=" in ev["rationale"]
        assert ev["metrics"]["coverage_drop_count"] >= 1


class TestCoverageResolvers:
    def test_floor_parse_with_fallback(self, monkeypatch):
        monkeypatch.delenv("ED4ALL_COVERAGE_FLOOR", raising=False)
        assert _resolve_coverage_floor() == 0.80
        monkeypatch.setenv("ED4ALL_COVERAGE_FLOOR", "0.5")
        assert _resolve_coverage_floor() == 0.5
        monkeypatch.setenv("ED4ALL_COVERAGE_FLOOR", "garbage")
        assert _resolve_coverage_floor() == 0.80
        monkeypatch.setenv("ED4ALL_COVERAGE_FLOOR", "1.5")
        assert _resolve_coverage_floor() == 0.80

    def test_strict_parse_with_fallback(self, monkeypatch):
        monkeypatch.delenv("ED4ALL_COVERAGE_DROP_STRICT", raising=False)
        assert _resolve_coverage_strict() is False
        monkeypatch.setenv("ED4ALL_COVERAGE_DROP_STRICT", "on")
        assert _resolve_coverage_strict() is True
        monkeypatch.setenv("ED4ALL_COVERAGE_DROP_STRICT", "nope")
        assert _resolve_coverage_strict() is False
