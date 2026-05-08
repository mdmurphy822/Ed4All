"""Worker W2.B — :class:`TrainforgeAssessmentQualityReport` aggregator tests.

Coverage matrix (per the W2.B plan):

* synthetic phase_outputs fixture covering each contributing gate
* schema invariants (each top-level key present, ``summary`` non-empty)
* promotion-decision matrix
  (``failed`` / ``non_certified_archive`` / ``trainable``)
* missing-eval-report graceful degradation
  (sets ``eval_summary: null``)
* missing ``_gate_results`` graceful degradation
  (per-gate-derived fields ``null``, top-level status ``warn``)
* decision-emit regression test
* schema-version invariance test
* deterministic write test (sort_keys=True)

The aggregator is exercised directly (no WorkflowRunner) so the
fixture surface stays minimal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lib.aggregators.trainforge_assessment_quality_report import (
    SCHEMA_VERSION,
    TrainforgeAssessmentQualityReport,
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _gate_row(
    gate_id: str,
    *,
    passed: bool = True,
    severity: str = "critical",
    score: Optional[float] = None,
    code: Optional[str] = None,
    message: Optional[str] = None,
    action: Optional[str] = None,
    validator_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ``GateResult.to_dict()``-shaped row for ``_gate_results``."""
    issues: List[Dict[str, Any]] = []
    if not passed:
        issues.append({
            "severity": severity,
            "code": code or "GATE_FAIL",
            "message": message or "synthetic failure",
            "location": None,
            "suggestion": None,
        })
    return {
        "gate_id": gate_id,
        "validator_name": validator_name or gate_id,
        "validator_version": "1.0",
        "passed": passed,
        "severity": severity,
        "score": score,
        "issues": issues,
        "execution_time_ms": 0,
        "action": action,
    }


def _phase(*gate_rows: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap gate rows into a phase_outputs payload."""
    return {
        "_completed": True,
        "_gates_passed": all(g.get("passed", True) for g in gate_rows),
        "_gate_results": list(gate_rows),
    }


def _write_assessment_quality_report(
    trainforge_dir: Path,
    *,
    total_questions: int = 5,
    placeholder_issues: int = 0,
    bloom_misaligned_issues: int = 0,
    distinct_correct_answer_ratio: float = 1.0,
) -> Path:
    """Write a synthetic ``quality_report.json`` carrying ``assessments``."""
    per_question_issues: List[Dict[str, Any]] = []
    for i in range(placeholder_issues):
        per_question_issues.append({
            "question_id": f"q-p{i}",
            "issues": ["TEMPLATED_DISTRACTOR_PLACEHOLDER"],
        })
    for i in range(bloom_misaligned_issues):
        per_question_issues.append({
            "question_id": f"q-b{i}",
            "issues": ["BLOOM_LEVEL_MISALIGNED"],
        })
    payload = {
        "assessments": {
            "total_questions": total_questions,
            "distinct_stems": total_questions,
            "distinct_correct_answers": total_questions,
            "distinct_stem_ratio": 1.0,
            "distinct_correct_answer_ratio": distinct_correct_answer_ratio,
            "avg_distractor_entropy": 0.85,
            "bloom_distribution_observed": {
                "remember": 2, "understand": 3,
            },
            "objective_coverage_ratio": 0.9,
            "per_question_issues": per_question_issues,
        }
    }
    qr_dir = trainforge_dir / "quality"
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr_path = qr_dir / "quality_report.json"
    qr_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return qr_path


def _write_eval_report(
    libv2_course: Path,
    *,
    model_id: str = "rdf-shacl-551-2-qwen2.5-1.5b-abc1234",
    faithfulness: float = 0.65,
    yes_rate: float = 0.55,
    negative_grounding_accuracy: float = 0.62,
    smoke_mode: bool = False,
) -> Path:
    """Write a synthetic ``eval_report.json`` under the model dir."""
    eval_dir = libv2_course / "models" / model_id / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "eval_report.json"
    payload = {
        "faithfulness": faithfulness,
        "yes_rate": yes_rate,
        "negative_grounding_accuracy": negative_grounding_accuracy,
        "source_match": 0.42,
        "baseline_delta": 0.05,
        "per_property_accuracy": {"sh:datatype": 0.55},
        "alignment_rate": 0.74,
    }
    if smoke_mode:
        payload["smoke_mode"] = True
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return eval_path


# --------------------------------------------------------------------------
# Schema invariants + minimal-fixture tests
# --------------------------------------------------------------------------


class TestSchemaInvariants:
    """Top-level keys, schema_version, summary always present."""

    def test_top_level_keys_present_on_minimal_fixture(self):
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="TEST",
            run_id="WF-MIN",
        )
        report = agg.build()

        assert report["schema_version"] == SCHEMA_VERSION
        # Every canonical top-level key surfaces (even when empty).
        for key in (
            "schema_version",
            "generated_at",
            "course_code",
            "run_id",
            "status",
            "summary",
            "synthesis_quality",
            "kg_quality",
            "eval_summary",
            "per_question_issues",
            "blocking_failures",
            "warnings",
            "promotion_decision",
        ):
            assert key in report, f"missing top-level key: {key}"

        # summary is non-empty (six canonical fields).
        for field in (
            "total_questions",
            "answerable_rate",
            "single_correct_rate",
            "bloom_alignment_rate",
            "placeholder_rate",
            "source_support_rate",
        ):
            assert field in report["summary"], f"missing summary field: {field}"

        # promotion_decision shape.
        promo = report["promotion_decision"]
        assert promo["value"] in {"failed", "non_certified_archive", "trainable"}
        assert isinstance(promo["rationale"], str)
        assert len(promo["rationale"]) >= 20
        assert isinstance(promo["contributing_gate_ids"], list)

    def test_schema_version_is_1_0(self):
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={}, course_code="X", run_id="X",
        )
        assert SCHEMA_VERSION == "1.0"
        assert agg.build()["schema_version"] == "1.0"


# --------------------------------------------------------------------------
# Synthetic phase_outputs covering each contributing gate
# --------------------------------------------------------------------------


class TestPhaseInputCoverage:
    """One fixture per source surface — every gate-id contributes."""

    def test_synthesis_quality_gates_project_to_bucket(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True, score=0.78),
                _gate_row("synthesis_leakage", passed=True, score=0.02),
                _gate_row(
                    "curie_anchoring", passed=True, score=0.99,
                ),
                _gate_row(
                    "property_coverage", passed=True, score=1.0,
                ),
                _gate_row(
                    "min_edge_count", passed=True, score=1.0,
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C",
            run_id="WF-SYNTH",
        )
        report = agg.build()
        sq = report["synthesis_quality"]
        for gate_id in (
            "synthesis_diversity",
            "synthesis_leakage",
            "curie_anchoring",
            "property_coverage",
            "min_edge_count",
        ):
            assert sq[gate_id] is not None, f"missing: {gate_id}"
            assert sq[gate_id]["gate_id"] == gate_id
            assert sq[gate_id]["passed"] is True

    def test_trainforge_assessment_summary_fields_populated(self, tmp_path):
        # quality_report::assessments dimension provides total_questions,
        # placeholder_rate, single_correct_rate, bloom_alignment_rate.
        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        _write_assessment_quality_report(
            tdir,
            total_questions=10,
            placeholder_issues=0,
            bloom_misaligned_issues=0,
            distinct_correct_answer_ratio=0.95,
        )
        phase_outputs = {
            "trainforge_assessment": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("assessment_quality", passed=True, score=0.95),
                ],
                "trainforge_dir": str(tdir),
            },
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C",
            run_id="WF-A",
            trainforge_dir=tdir,
        )
        report = agg.build()
        summary = report["summary"]
        assert summary["total_questions"] == 10
        assert summary["placeholder_rate"] == 0.0
        # Bloom alignment rate is 1.0 (no misaligned issues).
        assert summary["bloom_alignment_rate"] == 1.0
        # single_correct_rate populated from distinct_correct_answer_ratio
        # in the dimension.
        assert summary["single_correct_rate"] == 0.95

    def test_libv2_archival_kg_quality_bucket(self):
        phase_outputs = {
            "libv2_archival": _phase(
                _gate_row("kg_quality_report", passed=True, score=0.97),
                _gate_row("libv2_manifest", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-KG",
        )
        report = agg.build()
        kg = report["kg_quality"]
        assert kg is not None
        gate_ids = {row["gate_id"] for row in kg["gates"]}
        assert "kg_quality_report" in gate_ids
        assert "libv2_manifest" in gate_ids
        assert kg["summary"]["passed"] == 2
        assert kg["summary"]["any_blocking_fail"] is False

    def test_retrieval_grounding_score_drives_answerable_rate(self):
        phase_outputs = {
            "post_rewrite_validation": _phase(
                _gate_row(
                    "rewrite_assessment_retrieval_grounding",
                    passed=True, severity="warning", score=0.82,
                ),
            ),
            "inter_tier_validation": _phase(
                _gate_row(
                    "outline_assessment_retrieval_grounding",
                    passed=True, severity="warning", score=0.65,
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-RG",
        )
        report = agg.build()
        # Rewrite (post-rewrite) score wins over outline.
        assert report["summary"]["answerable_rate"] == 0.82
        assert report["summary"]["source_support_rate"] == 0.82


# --------------------------------------------------------------------------
# Promotion-decision matrix
# --------------------------------------------------------------------------


class TestPromotionDecision:
    """Three-way heuristic — failed / non_certified_archive / trainable."""

    def test_trainable_clean_run(self):
        # All gates pass, no warnings, no blocking signals.
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
                _gate_row("synthesis_leakage", passed=True),
                _gate_row("curie_anchoring", passed=True),
                _gate_row("property_coverage", passed=True),
                _gate_row("min_edge_count", passed=True),
            ),
            "trainforge_assessment": _phase(
                _gate_row("assessment_quality", passed=True),
            ),
            "libv2_archival": _phase(
                _gate_row("kg_quality_report", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-T",
        )
        report = agg.build()
        assert report["promotion_decision"]["value"] == "trainable"
        assert report["status"] == "pass"
        assert report["blocking_failures"] == []

    def test_non_certified_archive_warnings_only(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
                _gate_row(
                    "synthesis_quota",
                    passed=False,
                    severity="warning",
                    code="SYNTHESIS_QUOTA_EXCEEDED",
                    message="estimated 1800 dispatches",
                ),
            ),
            "trainforge_assessment": _phase(
                _gate_row("assessment_quality", passed=True),
            ),
            "libv2_archival": _phase(
                _gate_row("kg_quality_report", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-NCA",
        )
        report = agg.build()
        assert report["promotion_decision"]["value"] == "non_certified_archive"
        assert report["status"] == "warn"
        assert report["blocking_failures"] == []
        assert len(report["warnings"]) == 1

    def test_failed_synthesis_diversity_blocking(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row(
                    "synthesis_diversity",
                    passed=False,
                    severity="critical",
                    code="SYNTHESIS_TEMPLATE_COLLAPSE",
                    message="top-3 templates 78% of pairs",
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-F1",
        )
        report = agg.build()
        assert report["promotion_decision"]["value"] == "failed"
        assert report["status"] == "fail"
        assert any(
            bf["gate_id"] == "synthesis_diversity"
            for bf in report["blocking_failures"]
        )
        # Contributing gate IDs surface the failed gate.
        assert (
            "synthesis_diversity"
            in report["promotion_decision"]["contributing_gate_ids"]
        )

    def test_failed_min_edge_count_blocking(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row(
                    "min_edge_count",
                    passed=False,
                    severity="critical",
                    code="MIN_EDGE_COUNT_BELOW_FLOOR",
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-F2",
        )
        report = agg.build()
        assert report["promotion_decision"]["value"] == "failed"

    def test_failed_synthesis_leakage_blocking(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row(
                    "synthesis_leakage",
                    passed=False,
                    severity="critical",
                    code="LEAKAGE_VERBATIM_SPAN",
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-F3",
        )
        report = agg.build()
        assert report["promotion_decision"]["value"] == "failed"

    def test_failed_placeholder_rate_above_zero(self, tmp_path):
        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        _write_assessment_quality_report(
            tdir,
            total_questions=10,
            placeholder_issues=2,  # 0.2 placeholder rate -> failed
        )
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
            "trainforge_assessment": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("assessment_quality", passed=True, score=0.9),
                ],
                "trainforge_dir": str(tdir),
            },
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-F4",
            trainforge_dir=tdir,
        )
        report = agg.build()
        assert report["summary"]["placeholder_rate"] == 0.2
        assert report["promotion_decision"]["value"] == "failed"
        assert (
            "placeholder_rate=0.2000"
            in report["promotion_decision"]["rationale"]
        )

    def test_failed_answerable_rate_below_floor(self):
        phase_outputs = {
            "post_rewrite_validation": _phase(
                _gate_row(
                    "rewrite_assessment_retrieval_grounding",
                    passed=True, severity="warning", score=0.40,
                ),
            ),
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-F5",
        )
        report = agg.build()
        assert report["summary"]["answerable_rate"] == 0.40
        assert report["promotion_decision"]["value"] == "failed"

    def test_failed_kg_quality_blocking(self):
        phase_outputs = {
            "libv2_archival": _phase(
                _gate_row(
                    "kg_quality_report",
                    passed=False,
                    severity="critical",
                    code="KG_COMPLETENESS_BELOW_THRESHOLD",
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-F6",
        )
        report = agg.build()
        assert report["kg_quality"] is not None
        assert report["kg_quality"]["summary"]["any_blocking_fail"] is True
        assert report["promotion_decision"]["value"] == "failed"


# --------------------------------------------------------------------------
# Graceful-degradation tests
# --------------------------------------------------------------------------


class TestGracefulDegradation:
    """Missing inputs degrade to ``null`` / ``warn`` rather than raising."""

    def test_missing_eval_report_sets_eval_summary_null(self, tmp_path):
        # libv2_archival ran but no model adapter — eval_summary is null.
        course_dir = tmp_path / "rdf-shacl-551-2"
        course_dir.mkdir()
        # No models/ dir created.

        phase_outputs = {
            "libv2_archival": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("kg_quality_report", passed=True),
                ],
                "course_dir": str(course_dir),
            },
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-NEV",
            libv2_course_path=course_dir,
        )
        report = agg.build()
        assert report["eval_summary"] is None

    def test_eval_report_loaded_when_present(self, tmp_path):
        course_dir = tmp_path / "rdf-shacl-551-2"
        course_dir.mkdir()
        _write_eval_report(course_dir, faithfulness=0.65, yes_rate=0.50)
        phase_outputs = {
            "libv2_archival": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("kg_quality_report", passed=True),
                ],
                "course_dir": str(course_dir),
            },
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-EV",
            libv2_course_path=course_dir,
        )
        report = agg.build()
        assert report["eval_summary"] is not None
        assert report["eval_summary"]["faithfulness"] == 0.65
        assert report["eval_summary"]["yes_rate"] == 0.50

    def test_smoke_mode_eval_does_not_block_promotion(self, tmp_path):
        # smoke_mode eval_report MUST NOT trigger ``failed`` even when
        # its faithfulness is below the gating floor.
        course_dir = tmp_path / "rdf-shacl-551-2"
        course_dir.mkdir()
        _write_eval_report(
            course_dir,
            faithfulness=0.10,  # would normally fail
            yes_rate=0.99,      # would normally fail
            smoke_mode=True,
        )
        phase_outputs = {
            "libv2_archival": _phase(
                _gate_row("kg_quality_report", passed=True),
            ),
        }
        # Override course_dir on the phase_outputs lookup path.
        phase_outputs["libv2_archival"]["course_dir"] = str(course_dir)
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-SMOKE",
            libv2_course_path=course_dir,
        )
        report = agg.build()
        assert report["eval_summary"]["smoke_mode"] is True
        # Smoke-mode signals don't trigger ``failed``.
        assert report["promotion_decision"]["value"] != "failed"

    def test_missing_gate_results_top_level_warn(self):
        # phase_outputs is empty; aggregator should set status to ``warn``
        # and per-gate-derived fields to None.
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={}, course_code="C", run_id="WF-EMPTY",
        )
        report = agg.build()
        assert report["status"] == "warn"
        # Every synthesis_quality gate is None.
        for gate_id in (
            "synthesis_diversity", "synthesis_leakage", "curie_anchoring",
            "property_coverage", "min_edge_count",
        ):
            assert report["synthesis_quality"][gate_id] is None
        # KG quality and eval_summary degrade to None.
        assert report["kg_quality"] is None
        assert report["eval_summary"] is None
        # Summary fields collapse to None.
        for field in (
            "total_questions", "answerable_rate", "single_correct_rate",
            "bloom_alignment_rate", "placeholder_rate", "source_support_rate",
        ):
            assert report["summary"][field] is None

    def test_missing_quality_report_does_not_raise(self, tmp_path):
        # trainforge_dir resolved but quality_report.json is absent.
        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        # No quality_report.json.
        phase_outputs = {
            "trainforge_assessment": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("assessment_quality", passed=True, score=0.9),
                ],
                "trainforge_dir": str(tdir),
            },
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs, course_code="C", run_id="WF-NoQR",
            trainforge_dir=tdir,
        )
        # Must not raise.
        report = agg.build()
        # total_questions degrades to None.
        assert report["summary"]["total_questions"] is None
        # single_correct_rate falls through to assessment_quality gate score.
        assert report["summary"]["single_correct_rate"] == 0.9


# --------------------------------------------------------------------------
# Decision-emit regression test (Wave 2 Test 1 of the gate)
# --------------------------------------------------------------------------


class TestDecisionEmit:
    """Capture-wiring contract — aggregator emits one decision per build."""

    def test_emit_aggregator_decision_fires(self):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
        }
        captured: List[Dict[str, Any]] = []

        class _StubCapture:
            def log_decision(self, **kwargs):
                captured.append(kwargs)

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-EMIT",
            decision_capture=_StubCapture(),
        )
        agg.build()

        assert len(captured) == 1
        emit = captured[0]
        assert emit["decision_type"] == "trainforge_quality_aggregated"
        # Rationale is dynamic + ≥20 chars.
        assert isinstance(emit["rationale"], str)
        assert len(emit["rationale"]) >= 20
        # Rationale interpolates load-bearing dynamic signals.
        assert "verdict=" in emit["rationale"]
        assert "total_questions=" in emit["rationale"]
        assert "answerable_rate=" in emit["rationale"]
        assert "placeholder_rate=" in emit["rationale"]
        # Decision value is one of the canonical 3 enum values.
        assert emit["decision"] in {
            "failed", "non_certified_archive", "trainable",
        }
        # metrics carries the structured signals.
        metrics = emit.get("metrics") or {}
        assert "promotion_decision" in metrics
        assert "blocking_count" in metrics
        assert "warning_count" in metrics


class TestDecisionTypeEnum:
    """Decision-type schema enum check (Wave 2 Test 2 surface)."""

    def test_trainforge_quality_aggregated_in_schema_enum(self):
        schema_path = (
            Path(__file__).resolve().parents[3]
            / "schemas" / "events" / "decision_event.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        decision_type = (
            schema.get("properties", {}).get("decision_type", {})
        )
        enum = decision_type.get("enum") or []
        assert "trainforge_quality_aggregated" in enum, (
            "trainforge_quality_aggregated MUST appear in the canonical "
            "decision_type enum at "
            "schemas/events/decision_event.schema.json so "
            "DECISION_VALIDATION_STRICT=true does not fail-close on the "
            "aggregator emit."
        )


# --------------------------------------------------------------------------
# Deterministic write test
# --------------------------------------------------------------------------


class TestDeterministicWrite:
    """``write()`` produces sort_keys=True JSON at the requested path."""

    def test_write_emits_deterministic_sorted_json(self, tmp_path):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-WRITE",
        )
        out_path = tmp_path / "out" / "trainforge_assessment_quality_report.json"
        result_path = agg.write(out_path)
        assert result_path == out_path
        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == "1.0"
        assert loaded["run_id"] == "WF-WRITE"
        # Re-write produces the same bytes (modulo timestamp), so
        # canonicalising the file confirms deterministic key ordering:
        # the top-level keys match the alphabetical sort.
        top_keys = list(loaded.keys())
        assert top_keys == sorted(top_keys)


# --------------------------------------------------------------------------
# Workflow-runner wiring contract
# --------------------------------------------------------------------------


class TestWorkflowRunnerWiring:
    """The workflow_runner._maybe_write_trainforge_assessment_quality_report
    helper is reachable and returns a path on a healthy fixture."""

    def test_workflow_runner_helper_exists(self):
        from MCP.core.workflow_runner import WorkflowRunner

        assert hasattr(
            WorkflowRunner,
            "_maybe_write_trainforge_assessment_quality_report",
        ), (
            "WorkflowRunner must expose the W2.B aggregator wiring "
            "method symmetric to "
            "_maybe_write_courseforge_validation_report."
        )

    def test_workflow_runner_returns_path_on_archival_phase_output(
        self, tmp_path
    ):
        # Build a stub WorkflowRunner-like object exposing only the
        # static helper; we don't construct a full runner here.
        from MCP.core.workflow_runner import WorkflowRunner

        course_dir = tmp_path / "rdf-shacl-551-2"
        course_dir.mkdir()

        phase_outputs = {
            "libv2_archival": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("kg_quality_report", passed=True),
                ],
                "course_dir": str(course_dir),
            },
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
        }

        # Bind the unbound method against a fake-self that doesn't need
        # initialisation (the helper only reads phase_outputs +
        # workflow_params and writes to a path).
        helper = WorkflowRunner._maybe_write_trainforge_assessment_quality_report
        result = helper(
            WorkflowRunner.__new__(WorkflowRunner),
            workflow_id="WF-WIRE",
            workflow_params={"course_name": "C"},
            phase_outputs=phase_outputs,
        )
        assert result is not None
        result_path = Path(result)
        assert result_path.exists()
        assert (
            result_path.name == "trainforge_assessment_quality_report.json"
        )
        # Lives under the course quality dir.
        assert result_path.parent.name == "quality"

    def test_workflow_runner_falls_back_to_trainforge_dir(self, tmp_path):
        from MCP.core.workflow_runner import WorkflowRunner

        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        phase_outputs = {
            "training_synthesis": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row("synthesis_diversity", passed=True),
                ],
                "corpus_dir": str(tdir),
            },
        }
        helper = WorkflowRunner._maybe_write_trainforge_assessment_quality_report
        result = helper(
            WorkflowRunner.__new__(WorkflowRunner),
            workflow_id="WF-FB",
            workflow_params={"course_name": "C"},
            phase_outputs=phase_outputs,
        )
        assert result is not None
        result_path = Path(result)
        assert result_path.exists()
        # Falls back to the trainforge_dir root (no quality/ subdir).
        assert result_path.parent == tdir

    def test_workflow_runner_returns_none_when_unresolvable(self):
        from MCP.core.workflow_runner import WorkflowRunner

        helper = WorkflowRunner._maybe_write_trainforge_assessment_quality_report
        result = helper(
            WorkflowRunner.__new__(WorkflowRunner),
            workflow_id="WF-NONE",
            workflow_params={"course_name": "C"},
            phase_outputs={},  # no archival, no training_synthesis, nothing
        )
        assert result is None


# --------------------------------------------------------------------------
# Worker W3.B — promotion-ladder pass-through tests
# --------------------------------------------------------------------------


def _seed_promotion_ladder_dataset_config(
    trainforge_dir: Path,
    *,
    candidate: int = 12,
    validated: int = 9,
    trainable: int = 9,
    rejected: int = 3,
    rejection_reasons: Optional[Dict[str, int]] = None,
) -> Path:
    """Write a dataset_config.json carrying a statistics.promotion_ladder block."""
    if rejection_reasons is None:
        rejection_reasons = {
            "placeholder_residue": 1,
            "weak_distractor": 2,
        }
    ts = trainforge_dir / "training_specs"
    ts.mkdir(parents=True, exist_ok=True)
    config_path = ts / "dataset_config.json"
    config_path.write_text(
        json.dumps({
            "format": "instruction-following",
            "statistics": {
                "instruction_pairs": validated,
                "preference_pairs": 0,
                "promotion_ladder": {
                    "candidate_pairs_total": candidate,
                    "validated_pairs_total": validated,
                    "trainable_pairs_total": trainable,
                    "rejected_promotion_pairs": rejected,
                    "promotion_rejection_reasons": dict(rejection_reasons),
                },
            },
        }),
        encoding="utf-8",
    )
    return config_path


class TestPromotionLadderPassthrough:
    """Worker W3.B — phase_results.synthesis.promotion_ladder copy-through."""

    def test_promotion_ladder_copied_from_dataset_config(self, tmp_path):
        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        _seed_promotion_ladder_dataset_config(
            tdir,
            candidate=20,
            validated=14,
            trainable=14,
            rejected=6,
            rejection_reasons={
                "placeholder_residue": 2,
                "unsupported_answer": 1,
                "weak_distractor": 1,
                "unanswerable_stem": 1,
                "source_free_generation": 1,
            },
        )

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C", run_id="WF-PL",
            trainforge_dir=tdir,
        )
        report = agg.build()

        # phase_results bucket present + carries the synthesis sub-dict.
        assert "phase_results" in report
        assert "synthesis" in report["phase_results"]
        ladder = report["phase_results"]["synthesis"]["promotion_ladder"]
        assert ladder["candidate_pairs_total"] == 20
        assert ladder["validated_pairs_total"] == 14
        assert ladder["trainable_pairs_total"] == 14
        assert ladder["rejected_promotion_pairs"] == 6
        # Histogram alphabetised on read-back regardless of source order.
        assert list(ladder["promotion_rejection_reasons"].keys()) == [
            "placeholder_residue",
            "source_free_generation",
            "unanswerable_stem",
            "unsupported_answer",
            "weak_distractor",
        ]

    def test_missing_dataset_config_emits_empty_ladder_placeholder(
        self, tmp_path,
    ):
        # Legacy run: trainforge_dir resolved but no dataset_config.
        tdir = tmp_path / "trainforge"
        tdir.mkdir()

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C", run_id="WF-LEGACY",
            trainforge_dir=tdir,
        )
        report = agg.build()

        # Aggregator MUST NOT crash; the empty-shape placeholder ships.
        ladder = report["phase_results"]["synthesis"]["promotion_ladder"]
        assert ladder == {
            "candidate_pairs_total": 0,
            "validated_pairs_total": 0,
            "trainable_pairs_total": 0,
            "rejected_promotion_pairs": 0,
            "promotion_rejection_reasons": {},
        }

    def test_dataset_config_without_promotion_ladder_block(self, tmp_path):
        # Pre-W3.B dataset_config (no promotion_ladder under statistics):
        # the aggregator emits the empty-shape placeholder, not None.
        tdir = tmp_path / "trainforge"
        ts = tdir / "training_specs"
        ts.mkdir(parents=True)
        (ts / "dataset_config.json").write_text(
            json.dumps({
                "format": "instruction-following",
                "statistics": {
                    "instruction_pairs": 5,
                    "preference_pairs": 2,
                },
            }),
            encoding="utf-8",
        )

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C", run_id="WF-PRE-W3B",
            trainforge_dir=tdir,
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


# --------------------------------------------------------------------------
# Worker W-D11 T11.5 — evidence_quote_* metadata flow into synthesis_quality
# --------------------------------------------------------------------------


def _gate_row_with_metadata(
    gate_id: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    passed: bool = True,
    severity: str = "warning",
) -> Dict[str, Any]:
    """GateResult.to_dict() row carrying a free-form metadata block."""
    row = _gate_row(
        gate_id, passed=passed, severity=severity, score=1.0,
    )
    if metadata is not None:
        row["metadata"] = metadata
    return row


class TestEvidenceQuoteSurfacing:
    """Wave W-D11 T11.5 — pair-side claim_support metadata flows into the
    ``synthesis_quality.evidence_quote`` block. Mirrors the promotion-chain
    aggregator wiring on the chunks-and-pairs surface.
    """

    def test_pair_claim_support_metadata_flows_to_evidence_quote_block(
        self,
    ):
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row_with_metadata(
                    "pair_claim_support",
                    metadata={
                        "evidence_quote_coverage_rate": 0.87,
                        "evidence_quote_substring_fail_rate": 0.06,
                        "evidence_quote_char_span_mismatch_rate": 0.02,
                    },
                ),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-EQ-PAIR",
        )
        report = agg.build()
        evidence = report["synthesis_quality"]["evidence_quote"]
        assert evidence == {
            "evidence_quote_coverage_rate": 0.87,
            "evidence_quote_substring_fail_rate": 0.06,
            "evidence_quote_char_span_mismatch_rate": 0.02,
        }

    def test_legacy_pair_claim_support_without_metadata_defaults_to_zero(
        self,
    ):
        # GateResult that ran before T11.2 metadata stamping. The
        # aggregator MUST default each rate to 0.0 (graceful-degrade
        # contract) without raising.
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row_with_metadata("pair_claim_support", metadata=None),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-EQ-LEG",
        )
        report = agg.build()
        evidence = report["synthesis_quality"]["evidence_quote"]
        assert evidence == {
            "evidence_quote_coverage_rate": 0.0,
            "evidence_quote_substring_fail_rate": 0.0,
            "evidence_quote_char_span_mismatch_rate": 0.0,
        }

    def test_pair_claim_support_absent_yields_zero_evidence_block(self):
        # No pair_claim_support gate fired. The aggregator still emits
        # the evidence_quote block with zeros so the wire shape is
        # invariant.
        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True),
            ),
        }
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="C", run_id="WF-EQ-NONE",
        )
        report = agg.build()
        evidence = report["synthesis_quality"]["evidence_quote"]
        assert evidence == {
            "evidence_quote_coverage_rate": 0.0,
            "evidence_quote_substring_fail_rate": 0.0,
            "evidence_quote_char_span_mismatch_rate": 0.0,
        }


# --------------------------------------------------------------------------
# Wave W-D11.A — schema conformance
# --------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = (
    _REPO_ROOT / "schemas" / "aggregators"
    / "trainforge_assessment_quality_report.schema.json"
)
_PROMOTION_CHAIN_SCHEMA_PATH = (
    _REPO_ROOT / "schemas" / "governance" / "promotion_chain.schema.json"
)


def _load_schema_validator():
    """Build a Draft202012Validator with the cross-schema $ref pre-resolved.

    Mirrors ``test_promotion_chain_report.py::_load_schema_validator`` so the
    cross-schema reference to ``promotion_chain.schema.json#/$defs/EvidenceQuoteMetrics``
    resolves locally without an HTTPS dereference.
    """
    try:
        from jsonschema import Draft202012Validator, RefResolver
    except ImportError:  # pragma: no cover — depends on env
        pytest.skip("jsonschema not installed")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    promotion_chain_schema = json.loads(
        _PROMOTION_CHAIN_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    resolver = RefResolver.from_schema(schema)
    resolver.store[promotion_chain_schema["$id"]] = promotion_chain_schema
    return Draft202012Validator(schema, resolver=resolver)


class TestSchemaConformance:
    """Wave W-D11.A — emitted reports validate against the pinned schema."""

    def test_schema_self_validates(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:  # pragma: no cover
            pytest.skip("jsonschema not installed")
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        # MUST NOT raise.
        Draft202012Validator.check_schema(schema)

    def test_minimal_fixture_validates(self):
        validator = _load_schema_validator()
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="MIN",
            run_id="WF-MIN-SCHEMA",
        )
        report = agg.build()
        errors = sorted(
            validator.iter_errors(report), key=lambda e: e.path
        )
        assert errors == [], (
            "minimal fixture failed schema validation: "
            + "\n".join(
                f"{list(e.absolute_path)}: {e.message}" for e in errors
            )
        )

    def test_full_fixture_validates(self, tmp_path):
        """Rich fixture: every contributing surface populated."""
        validator = _load_schema_validator()

        # Trainforge dir with quality_report.json + dataset_config.json.
        tdir = tmp_path / "trainforge"
        tdir.mkdir()
        _write_assessment_quality_report(
            tdir,
            total_questions=10,
            placeholder_issues=0,
            bloom_misaligned_issues=1,
            distinct_correct_answer_ratio=0.95,
        )
        _seed_promotion_ladder_dataset_config(
            tdir,
            candidate=20,
            validated=14,
            trainable=14,
            rejected=6,
            rejection_reasons={
                "placeholder_residue": 2,
                "unsupported_answer": 1,
                "weak_distractor": 1,
                "unanswerable_stem": 1,
                "source_free_generation": 1,
            },
        )

        # LibV2 course dir with eval_report.json.
        course_dir = tmp_path / "rdf-shacl-551-2"
        course_dir.mkdir()
        _write_eval_report(
            course_dir, faithfulness=0.85, yes_rate=0.55,
            negative_grounding_accuracy=0.82,
        )

        phase_outputs = {
            "training_synthesis": _phase(
                _gate_row("synthesis_diversity", passed=True, score=0.78),
                _gate_row("synthesis_leakage", passed=True, score=0.02),
                _gate_row("curie_anchoring", passed=True, score=0.99),
                _gate_row("property_coverage", passed=True, score=1.0),
                _gate_row("min_edge_count", passed=True, score=1.0),
                _gate_row_with_metadata(
                    "pair_claim_support",
                    metadata={
                        "evidence_quote_coverage_rate": 0.87,
                        "evidence_quote_substring_fail_rate": 0.06,
                        "evidence_quote_char_span_mismatch_rate": 0.02,
                    },
                ),
            ),
            "trainforge_assessment": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row(
                        "assessment_quality", passed=True, score=0.95,
                    ),
                    _gate_row(
                        "assessment_objective_alignment", passed=True,
                    ),
                ],
                "trainforge_dir": str(tdir),
            },
            "libv2_archival": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _gate_row(
                        "kg_quality_report", passed=True, score=0.97,
                    ),
                    _gate_row("libv2_manifest", passed=True),
                ],
                "course_dir": str(course_dir),
            },
            "post_rewrite_validation": _phase(
                _gate_row(
                    "rewrite_assessment_retrieval_grounding",
                    passed=True, severity="warning", score=0.82,
                ),
            ),
        }

        agg = TrainforgeAssessmentQualityReport(
            phase_outputs=phase_outputs,
            course_code="rdf-shacl-551-2",
            run_id="WF-FULL-SCHEMA",
            trainforge_dir=tdir,
            libv2_course_path=course_dir,
        )
        report = agg.build()
        errors = sorted(
            validator.iter_errors(report), key=lambda e: e.path
        )
        assert errors == [], (
            "full fixture failed schema validation: "
            + "\n".join(
                f"{list(e.absolute_path)}: {e.message}" for e in errors
            )
        )

    def test_evidence_quote_cross_ref_resolves(self):
        """Cross-schema $ref to promotion_chain::EvidenceQuoteMetrics resolves.

        Constructs a payload whose synthesis_quality.evidence_quote block
        violates the cross-referenced shape (rate > 1.0) and asserts the
        validator surfaces the violation. If the $ref were broken the
        validator would silently pass the bad payload.
        """
        validator = _load_schema_validator()
        agg = TrainforgeAssessmentQualityReport(
            phase_outputs={},
            course_code="C",
            run_id="WF-XREF",
        )
        report = agg.build()
        # Mutate the evidence_quote block to violate maximum: 1.0 on the
        # cross-referenced EvidenceQuoteMetrics.evidence_quote_coverage_rate.
        report["synthesis_quality"]["evidence_quote"][
            "evidence_quote_coverage_rate"
        ] = 5.0
        errors = list(validator.iter_errors(report))
        assert any(
            "evidence_quote_coverage_rate" in str(list(e.absolute_path))
            or "5.0" in e.message
            for e in errors
        ), (
            "cross-schema $ref to EvidenceQuoteMetrics did not resolve — "
            "validator failed to flag rate=5.0 (>1.0) on the "
            "synthesis_quality.evidence_quote block"
        )
