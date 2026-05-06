"""Worker W3.G — :class:`PromotionChainAggregator` tests.

Coverage matrix (per the W3.G plan):

1. Full happy path — all 9 arrows have reports, all pass; course_status
   = certified_trainable.
2. Missing arrow 4 — packaging_report.json absent; arrow 4
   promotion_decision="fail", validator_set carries
   ``missing_stage_report``; course_status=failed.
3. Course-status decision-logic table — 8-combination parametric matrix
   over (accessibility_pass, instructional_pass, trainable_pass) per plan
   §"Worker W3.G" Test 3.
4. Anti-silent-degradation — confirm a missing report does NOT silently
   default to pass.
5. Chain-hash determinism — same input rows produce the same chain_hash;
   reordered rows produce the same hash (we sort before hashing) but a
   single field flip changes the digest.
6. Schema validates emitted report — emit + Draft202012Validator
   ``iter_errors`` against ``schemas/governance/promotion_chain.schema.json``,
   assert empty.
7. Decision capture — one ``promotion_chain_aggregated`` event per build
   with the 5 required signals interpolated in the rationale.
8. Best-effort posture — missing course_path returns gracefully without
   raising.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lib.aggregators.promotion_chain_report import (
    ARROW_NAMES,
    MISSING_STAGE_REPORT,
    PromotionChainAggregator,
    SCHEMA_VERSION,
)
from lib.governance.course_status import (
    ACCESSIBILITY_GATE_IDS,
    INSTRUCTIONAL_GATE_IDS,
    TRAINABLE_GATE_IDS,
    derive_course_status,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "governance" / "promotion_chain.schema.json"
)
COURSE_STATUS_SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "governance" / "course_status.schema.json"
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _coverage_block(consumed: int, emitted: int) -> Dict[str, Any]:
    return {
        "consumed_count": consumed,
        "emitted_count": emitted,
        "dropped_count": max(0, consumed - emitted),
        "drop_reasons": {},
        "coverage_pct": (emitted / consumed) if consumed > 0 else 0.0,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _build_full_layout(tmp_path: Path) -> Dict[str, Path]:
    """Build a complete fixture layout with all 9 arrows present + passing."""
    course_dir = tmp_path / "libv2_course"
    project_dir = tmp_path / "courseforge_project"
    staging_dir = tmp_path / "staging"
    course_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Arrow 1 — staging manifest
    _write_json(
        staging_dir / "staging_manifest.json",
        {
            "run_id": "WF-TEST",
            "course_name": "TEST",
            "staged_files": ["a.html", "b.html"],
            "files": [{"path": "a.html"}, {"path": "b.html"}],
            "errors": None,
        },
    )

    # Arrow 2 — dart_chunks/manifest.json
    _write_json(
        course_dir / "dart_chunks" / "manifest.json",
        {
            "chunks_sha256": "a" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "dart",
            "source_dart_html_sha256": "b" * 64,
            "chunks_count": 100,
            "source_coverage": _coverage_block(100, 100),
        },
    )

    # Arrow 3 — inter-tier + post-rewrite reports + cf validation report
    _write_json(
        project_dir / "02_validation_report" / "report.json",
        {
            "schema_version": "1.0",
            "total_blocks": 50,
            "passed": 50,
            "failed": 0,
            "escalated": 0,
            "per_block": [],
            "source_coverage": _coverage_block(50, 50),
        },
    )
    _write_json(
        project_dir / "04_rewrite" / "02_validation_report" / "report.json",
        {
            "schema_version": "1.0",
            "total_blocks": 50,
            "passed": 50,
            "failed": 0,
            "escalated": 0,
            "per_block": [],
            "source_coverage": _coverage_block(50, 50),
        },
    )
    _write_json(
        project_dir / "courseforge_validation_report.json",
        {
            "schema_version": "1.1",
            "course_code": "TEST",
            "run_id": "WF-TEST",
            "generated_at": "2026-05-06T00:00:00Z",
            "status": "pass",
            "summary": {
                "total_gates": 0,
                "passed_count": 0,
                "failed_count": 0,
                "warning_count": 0,
                "skipped_count": 0,
            },
            "blocking_failures": [],
            "warnings": [],
            "per_phase": [],
            "final_promotion_decision": {
                "value": "certified_trainable",
                "rationale": "all gates passed",
                "contributing_gate_ids": [],
            },
        },
    )

    # Arrow 4 — packaging_report.json
    _write_json(
        project_dir / "05_final_package" / "packaging_report.json",
        {
            "source_coverage": _coverage_block(50, 50),
        },
    )

    # LibV2 manifest carrying imscc hash
    _write_json(
        course_dir / "manifest.json",
        {
            "imscc_sha256": "c" * 64,
        },
    )

    # Arrow 5 — imscc_chunks/manifest.json
    _write_json(
        course_dir / "imscc_chunks" / "manifest.json",
        {
            "chunks_sha256": "d" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "imscc",
            "source_imscc_sha256": "c" * 64,
            "chunks_count": 100,
            "source_coverage": _coverage_block(100, 100),
        },
    )

    # Arrow 6 — assessments.json
    _write_json(
        course_dir / "training_specs" / "assessments.json",
        {
            "assessment_id": "asmt-001",
            "title": "TEST",
            "course_code": "TEST",
            "questions": [],
            "source_coverage": _coverage_block(20, 20),
        },
    )

    # Arrow 7 — synthesis_summary.json + dataset_config.json
    _write_json(
        course_dir / "training_specs" / "synthesis_summary.json",
        {
            "schema_version": "1.0",
            "course_code": "TEST",
            "provider": "local",
            "instruction_pairs_emitted": 100,
            "preference_pairs_emitted": 50,
            "source_coverage": _coverage_block(150, 150),
        },
    )
    _write_json(
        course_dir / "training_specs" / "dataset_config.json",
        {
            "statistics": {"promotion_ladder": {"draft": 200, "trainable": 150}},
        },
    )

    # Arrows 8 + 9 — model_card.json + eval_report.json
    model_dir = course_dir / "models" / "test-model-001"
    _write_json(
        model_dir / "model_card.json",
        {
            "model_id": "test-model-001",
            "provenance": {
                "instruction_pairs_hash": "e" * 64,
                "preference_pairs_hash": "f" * 64,
                "chunks_hash": "1" * 64,
                "pedagogy_graph_hash": "2" * 64,
                "concept_graph_hash": "3" * 64,
                "vocabulary_ttl_hash": "4" * 64,
                "holdout_graph_hash": "5" * 64,
            },
            "weights_sha256": "6" * 64,
        },
    )
    _write_json(
        model_dir / "eval" / "eval_report.json",
        {
            "eval_config_hash": "7" * 64,
            "status": "pass",
            "metrics": {"faithfulness": 0.85, "source_match": 0.65},
        },
    )

    return {
        "course_dir": course_dir,
        "project_dir": project_dir,
        "staging_dir": staging_dir,
    }


# --------------------------------------------------------------------------
# Schema validator helper (skips cleanly when jsonschema is missing)
# --------------------------------------------------------------------------


def _load_schema_validator():
    try:
        from jsonschema import Draft202012Validator
        from jsonschema import RefResolver
    except ImportError:  # pragma: no cover - depends on env
        pytest.skip("jsonschema not installed")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    course_status_schema = json.loads(
        COURSE_STATUS_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    # Pre-resolve the cross-schema $ref locally so jsonschema doesn't try
    # to dereference the $id over HTTPS.
    resolver = RefResolver.from_schema(schema)
    resolver.store[course_status_schema["$id"]] = course_status_schema
    return Draft202012Validator(schema, resolver=resolver)


# --------------------------------------------------------------------------
# Decision-capture spy
# --------------------------------------------------------------------------


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------
# Test 1 — full happy path
# --------------------------------------------------------------------------


class TestFullHappyPath:
    def test_all_arrows_present_certified_trainable(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        report = agg.build()

        assert report["schema_version"] == SCHEMA_VERSION
        assert report["course_code"] == "TEST"
        assert report["run_id"] == "WF-TEST"
        assert len(report["arrows"]) == 9
        for i, arrow in enumerate(report["arrows"], start=1):
            assert arrow["arrow_id"] == i, f"arrow_id mismatch: {arrow}"
            assert arrow["name"] == ARROW_NAMES[i]
            assert arrow["promotion_decision"] in (
                "pass", "warn", "fail", "escalate"
            )

        # Every arrow MUST be passing in the happy path.
        for arrow in report["arrows"]:
            assert arrow["promotion_decision"] in ("pass", "warn"), (
                f"arrow {arrow['arrow_id']} failed in happy path: {arrow}"
            )

        assert report["course_status"] == "certified_trainable"


# --------------------------------------------------------------------------
# Test 2 — missing arrow 4 (anti-silent-degradation)
# --------------------------------------------------------------------------


class TestMissingArrowFour:
    def test_packaging_report_missing_fails_chain(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        # DELETE arrow 4
        (
            layout["project_dir"] / "05_final_package" / "packaging_report.json"
        ).unlink()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        report = agg.build()

        arrow_4 = next(a for a in report["arrows"] if a["arrow_id"] == 4)
        assert arrow_4["promotion_decision"] == "fail"
        assert MISSING_STAGE_REPORT in arrow_4["validator_set"]
        # Fall-through to course_status="failed" because missing_stage_report
        # is in the critical cohort.
        assert report["course_status"] == "failed"


# --------------------------------------------------------------------------
# Test 3 — course-status decision-logic table (8 combinations)
# --------------------------------------------------------------------------


def _arrow_with_gate(arrow_id: int, gate_id: str, *, promotion: str = "pass") -> Dict[str, Any]:
    return {
        "arrow_id": arrow_id,
        "name": ARROW_NAMES[arrow_id],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [gate_id],
        "passed": promotion in ("pass", "warn"),
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": promotion,
    }


def _archive_passing_arrows() -> List[Dict[str, Any]]:
    """Build arrows 1-5 in a clean passing state, no critical-cohort gates."""
    return [
        {
            "arrow_id": i,
            "name": ARROW_NAMES[i],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [],
            "passed": True,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass",
        }
        for i in range(1, 6)
    ]


def _make_chain(
    *,
    accessibility_pass: bool,
    instructional_pass: bool,
    trainable_pass: bool,
) -> List[Dict[str, Any]]:
    """Compose a 9-arrow chain that exercises a row of the decision table.

    accessibility_pass=False is encoded by attaching a failing wcag gate
    (so the failure is critical via the cohort table). instructional_pass
    + trainable_pass follow the same pattern using their respective
    cohort gates.
    """
    arrows = _archive_passing_arrows()
    # Embed accessibility gate on arrow 5 so when it fails the helper
    # short-circuits to "failed" via the critical-cohort branch.
    arrows[4] = _arrow_with_gate(
        5, "wcag_compliance",
        promotion="pass" if accessibility_pass else "fail",
    )
    # Arrow 6 — instructional cohort. Distribute the four gates across
    # arrows 6/7 so the cohort_passes() helper can find them.
    if instructional_pass:
        arrows.append({
            "arrow_id": 6,
            "name": ARROW_NAMES[6],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [
                "content_grounding", "oscqr_score",
                "page_objectives", "source_refs",
            ],
            "passed": True,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass",
        })
    else:
        # Mark cohort as failing — at least one gate fires non-passing.
        arrows.append({
            "arrow_id": 6,
            "name": ARROW_NAMES[6],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [
                "content_grounding", "oscqr_score",
                "page_objectives", "source_refs",
            ],
            "passed": False,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "fail",
        })
    if trainable_pass:
        arrows.append({
            "arrow_id": 7,
            "name": ARROW_NAMES[7],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [
                "min_edge_count", "synthesis_diversity",
                "eval_gating", "family_completeness",
            ],
            "passed": True,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass",
        })
    else:
        arrows.append({
            "arrow_id": 7,
            "name": ARROW_NAMES[7],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [
                "min_edge_count", "synthesis_diversity",
                "eval_gating", "family_completeness",
            ],
            "passed": False,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "fail",
        })
    # Arrows 8 + 9 — bookkeeping rows so the chain has all 9.
    arrows.append({
        "arrow_id": 8,
        "name": ARROW_NAMES[8],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [],
        "passed": True,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": "pass",
    })
    arrows.append({
        "arrow_id": 9,
        "name": ARROW_NAMES[9],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [],
        "passed": True,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": "pass",
    })
    return arrows


@pytest.mark.parametrize(
    "accessibility_pass,instructional_pass,trainable_pass",
    [
        (False, False, False),
        (False, False, True),
        (False, True,  False),
        (False, True,  True),
        (True,  False, False),
        (True,  False, True),
        (True,  True,  False),
        (True,  True,  True),
    ],
)
def test_course_status_decision_yields_enum(
    accessibility_pass, instructional_pass, trainable_pass,
):
    arrows = _make_chain(
        accessibility_pass=accessibility_pass,
        instructional_pass=instructional_pass,
        trainable_pass=trainable_pass,
    )
    decision = derive_course_status(arrows)
    valid_enum = {
        "failed", "non_certified_archive",
        "certified_accessible", "certified_instructional",
        "certified_trainable",
    }
    assert decision in valid_enum, (
        f"({accessibility_pass},{instructional_pass},{trainable_pass}) "
        f"-> {decision!r} not in {valid_enum}"
    )
    assert decision is not None

    # Spot-check the load-bearing branches:
    if accessibility_pass and instructional_pass and trainable_pass:
        assert decision == "certified_trainable"
    if accessibility_pass and instructional_pass and not trainable_pass:
        # trainable cohort fails -> instructional ceiling
        assert decision == "certified_instructional"
    if accessibility_pass and not instructional_pass and not trainable_pass:
        # instructional cohort hard-fails but is NOT in the critical
        # cohort; accessibility passing certifies the accessible tier
        # without short-circuiting to "failed".
        assert decision == "certified_accessible"
    if not accessibility_pass:
        # accessibility hard-fail is in critical cohort -> "failed"
        assert decision == "failed"


# --------------------------------------------------------------------------
# Test 4 — anti-silent-degradation: missing report does not silently pass
# --------------------------------------------------------------------------


class TestAntiSilentDegradation:
    def test_missing_arrow_2_does_not_default_to_pass(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        # Drop dart_chunks/manifest.json — arrow 2.
        (layout["course_dir"] / "dart_chunks" / "manifest.json").unlink()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        report = agg.build()
        arrow_2 = next(a for a in report["arrows"] if a["arrow_id"] == 2)
        assert arrow_2["promotion_decision"] == "fail"
        assert MISSING_STAGE_REPORT in arrow_2["validator_set"]
        # Confirm it is NOT silently a pass row.
        assert arrow_2["passed"] is None
        assert report["course_status"] == "failed"

    def test_missing_arrow_5_imscc_chunks_fails_chain(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        (layout["course_dir"] / "imscc_chunks" / "manifest.json").unlink()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        report = agg.build()
        arrow_5 = next(a for a in report["arrows"] if a["arrow_id"] == 5)
        assert arrow_5["promotion_decision"] == "fail"
        assert MISSING_STAGE_REPORT in arrow_5["validator_set"]
        assert report["course_status"] == "failed"


# --------------------------------------------------------------------------
# Test 5 — chain-hash determinism
# --------------------------------------------------------------------------


class TestChainHashDeterminism:
    def test_same_input_same_chain_hash(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        agg1 = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        agg2 = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        h1 = agg1.build()["chain_hash"]
        h2 = agg2.build()["chain_hash"]
        assert h1 == h2
        assert len(h1) == 64
        # Hex-only.
        int(h1, 16)

    def test_field_flip_changes_chain_hash(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        agg1 = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        h1 = agg1.build()["chain_hash"]
        # Drop arrow 6 -> different row -> different hash.
        (
            layout["course_dir"] / "training_specs" / "assessments.json"
        ).unlink()
        agg2 = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        h2 = agg2.build()["chain_hash"]
        assert h1 != h2

    def test_chain_hash_canonical_over_arrow_order(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        report = agg.build()
        # Recompute the hash with arrows reversed; canonical sort must
        # produce the same digest.
        arrows_reversed = list(reversed(report["arrows"]))
        h2 = PromotionChainAggregator._compute_chain_hash(arrows_reversed)
        assert h2 == report["chain_hash"]


# --------------------------------------------------------------------------
# Test 6 — schema validation
# --------------------------------------------------------------------------


class TestSchemaValidation:
    def test_emit_validates_against_schema(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        out_path = layout["course_dir"] / "courseforge_promotion_chain_report.json"
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        agg.write(out_path)
        emitted = json.loads(out_path.read_text(encoding="utf-8"))

        validator = _load_schema_validator()
        errors = list(validator.iter_errors(emitted))
        assert errors == [], (
            f"emitted report failed schema validation: "
            f"{[e.message for e in errors]}"
        )

    def test_missing_arrow_emit_still_validates(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        # Delete arrow 4 to exercise the missing_stage_report row shape.
        (
            layout["project_dir"] / "05_final_package" / "packaging_report.json"
        ).unlink()
        out_path = layout["course_dir"] / "courseforge_promotion_chain_report.json"
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
        )
        agg.write(out_path)
        emitted = json.loads(out_path.read_text(encoding="utf-8"))
        validator = _load_schema_validator()
        errors = list(validator.iter_errors(emitted))
        assert errors == [], (
            f"missing-arrow emit failed schema validation: "
            f"{[e.message for e in errors]}"
        )


# --------------------------------------------------------------------------
# Test 7 — decision-capture emit
# --------------------------------------------------------------------------


class TestDecisionCapture:
    def test_one_event_per_build_with_required_signals(self, tmp_path):
        layout = _build_full_layout(tmp_path)
        spy = _SpyCapture()
        agg = PromotionChainAggregator(
            course_path=layout["course_dir"],
            project_path=layout["project_dir"],
            staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
            course_code="TEST",
            run_id="WF-TEST",
            decision_capture=spy,
        )
        agg.build()

        assert len(spy.events) == 1
        event = spy.events[0]
        assert event["decision_type"] == "promotion_chain_aggregated"
        rationale = event["rationale"]
        # Confirm all 5 required signals are interpolated.
        for token in (
            "chain_hash=", "course_status=",
            "total_arrows=", "passed_arrows=",
            "failed_arrows=", "missing_arrows=",
        ):
            assert token in rationale, (
                f"missing '{token}' in rationale: {rationale!r}"
            )
        # Rationale floor (>= 20 chars per Ed4All decision-capture contract).
        assert len(rationale) >= 20
        # Metrics should carry the same five signals.
        metrics = event.get("metrics") or {}
        for key in (
            "total_arrows", "passed_arrows", "failed_arrows", "missing_arrows",
        ):
            assert key in metrics, f"metrics missing key: {key}"


# --------------------------------------------------------------------------
# Test 8 — best-effort posture (graceful degradation)
# --------------------------------------------------------------------------


class TestBestEffortPosture:
    def test_no_inputs_returns_chain_of_missing_rows(self, tmp_path):
        agg = PromotionChainAggregator(
            course_path=None,
            project_path=None,
            staging_manifest_path=None,
            course_code="EMPTY",
            run_id="WF-EMPTY",
        )
        report = agg.build()
        assert len(report["arrows"]) == 9
        for arrow in report["arrows"]:
            assert MISSING_STAGE_REPORT in arrow["validator_set"], (
                f"empty fixture arrow {arrow['arrow_id']} did not surface "
                f"missing_stage_report"
            )
            assert arrow["promotion_decision"] == "fail"
        assert report["course_status"] == "failed"

    def test_nonexistent_course_path_does_not_crash(self, tmp_path):
        ghost = tmp_path / "no_such_course"
        agg = PromotionChainAggregator(
            course_path=ghost,
            course_code="GHOST",
            run_id="WF-GHOST",
        )
        # Build and write must not raise.
        report = agg.build()
        out = tmp_path / "ghost_promotion_chain.json"
        agg.write(out)
        # Round-trip through disk works.
        rehydrated = json.loads(out.read_text(encoding="utf-8"))
        assert rehydrated["course_status"] == report["course_status"]
        assert rehydrated["chain_hash"] == report["chain_hash"]
