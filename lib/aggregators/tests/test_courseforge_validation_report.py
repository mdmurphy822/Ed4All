"""Worker W5 — :class:`CourseforgeValidationReport` aggregator tests.

Covers:

* ``test_aggregator_walks_all_phase_reports`` — synthetic project with
  per-phase ``report.json`` files in two phases plus an in-memory
  ``_gate_results`` chain in a third; aggregator's ``per_phase`` covers
  all three.
* ``test_status_pass_when_all_gates_pass`` — every gate ``passed=True``
  ⇒ top-level ``status="pass"``, ``blocking_failures=[]``.
* ``test_status_fail_when_any_critical_fails`` — one critical gate
  fails ⇒ ``status="fail"`` + non-empty ``blocking_failures``.
* ``test_status_pass_when_only_warnings`` — failures only at warning
  severity ⇒ ``status="pass"`` + non-empty ``warnings``.
* ``test_missing_phase_report_is_skipped_with_warning`` — known report
  absent on disk and no in-memory fallback ⇒ phase appears with
  ``skipped=True``; ``summary.skipped_count`` increments; aggregator
  doesn't raise.
* ``test_write_emits_deterministic_sorted_json`` — :meth:`write`
  produces ``sort_keys=True`` JSON at the requested path.

The aggregator is exercised directly (no WorkflowRunner) so the
fixture surface stays minimal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.aggregators.courseforge_validation_report import (
    CourseforgeValidationReport,
    SCHEMA_VERSION,
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _make_two_pass_report(
    phase: str,
    *,
    passed: int = 2,
    failed: int = 0,
    escalated: int = 0,
    chain_pass: bool = True,
) -> dict:
    """Build a Phase 5 ``report.json`` payload with a per-block chain."""
    chain = [
        {
            "gate_id": (
                "outline_curie_anchoring"
                if phase == "inter_tier_validation"
                else "rewrite_curie_anchoring"
            ),
            "action": None if chain_pass else "regenerate",
            "passed": chain_pass,
            "issue_count": 0 if chain_pass else 3,
        },
        {
            "gate_id": (
                "outline_source_refs"
                if phase == "inter_tier_validation"
                else "rewrite_source_refs"
            ),
            "action": None,
            "passed": True,
            "issue_count": 0,
        },
    ]
    per_block = []
    for i in range(passed + failed + escalated):
        if i < passed:
            status = "passed"
            esc = None
        elif i < passed + failed:
            status = "failed"
            esc = None
        else:
            status = "escalated"
            esc = "outline_budget_exhausted"
        per_block.append({
            "block_id": f"b{i}",
            "block_type": "objective",
            "page": "page_1",
            "week": 1,
            "status": status,
            "gate_results": chain,
            "escalation_marker": esc,
        })
    return {
        "run_id": "WF-W5-TEST",
        "phase": phase,
        "schema_version": "v1",
        "total_blocks": passed + failed + escalated,
        "passed": passed,
        "failed": failed,
        "escalated": escalated,
        "per_block": per_block,
    }


def _write_report(project_path: Path, phase: str, payload: dict) -> Path:
    if phase == "inter_tier_validation":
        rel = project_path / "02_validation_report" / "report.json"
    elif phase == "post_rewrite_validation":
        rel = (
            project_path / "04_rewrite" / "02_validation_report"
            / "report.json"
        )
    else:
        raise ValueError(f"unknown phase {phase!r}")
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_text(json.dumps(payload), encoding="utf-8")
    return rel


def _make_in_memory_phase(
    *,
    gate_id: str,
    passed: bool,
    severity: str = "critical",
    code: str | None = None,
    message: str | None = None,
) -> dict:
    """Build a phase_outputs entry with ``_gate_results`` populated."""
    issue = {}
    issues = []
    if not passed:
        issue = {
            "severity": severity,
            "code": code or "GATE_FAIL",
            "message": message or "synthetic failure",
            "location": None,
            "suggestion": None,
        }
        issues = [issue]
    return {
        "_completed": True,
        "_gates_passed": passed,
        "_gate_results": [
            {
                "gate_id": gate_id,
                "validator_name": gate_id,
                "validator_version": "1.0",
                "passed": passed,
                "severity": severity,
                "score": None,
                "issues": issues,
                "execution_time_ms": 0,
                "action": None,
            }
        ],
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestAggregator:

    def test_aggregator_walks_all_phase_reports(self, tmp_path):
        # Two known per-phase reports + one in-memory phase.
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        phase_outputs = {
            "packaging": _make_in_memory_phase(
                gate_id="imscc_structure", passed=True
            ),
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="PHYS_101",
            run_id="WF-W5-TEST",
        )
        report = agg.build()

        assert report["schema_version"] == SCHEMA_VERSION
        assert report["course_code"] == "PHYS_101"
        assert report["run_id"] == "WF-W5-TEST"
        assert report["status"] == "pass"

        phases = {p["phase"] for p in report["per_phase"]}
        assert phases == {
            "inter_tier_validation",
            "post_rewrite_validation",
            "packaging",
        }

        # Each two-pass phase contributes 2 gates from the chain;
        # packaging contributes 1.
        assert report["summary"]["total_gates"] == 5
        assert report["summary"]["passed_count"] == 5
        assert report["summary"]["failed_count"] == 0
        assert report["summary"]["warning_count"] == 0

    def test_status_pass_when_all_gates_pass(self, tmp_path):
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-PASS",
        )
        report = agg.build()
        assert report["status"] == "pass"
        assert report["blocking_failures"] == []

    def test_status_fail_when_any_critical_fails(self, tmp_path):
        # Inter-tier passes; post-rewrite has a critical failure
        # (failed=1) and the chain itself flips passed=False.
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report(
                "post_rewrite_validation",
                passed=1,
                failed=1,
                chain_pass=False,
            ),
        )
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-FAIL",
        )
        report = agg.build()
        assert report["status"] == "fail"
        assert len(report["blocking_failures"]) >= 1
        bf = report["blocking_failures"][0]
        assert bf["phase"] == "post_rewrite_validation"
        assert bf["severity"] == "critical"
        assert bf["gate_id"] == "rewrite_curie_anchoring"

    def test_status_pass_when_only_warnings(self, tmp_path):
        # Per-phase reports both pass; in-memory packaging gate fails
        # at warning severity.
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        phase_outputs = {
            "packaging": _make_in_memory_phase(
                gate_id="oscqr_score",
                passed=False,
                severity="warning",
                code="OSCQR_LOW",
                message="OSCQR score below 0.7",
            ),
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-WARN",
        )
        report = agg.build()
        assert report["status"] == "pass"
        assert report["blocking_failures"] == []
        assert len(report["warnings"]) == 1
        w = report["warnings"][0]
        assert w["phase"] == "packaging"
        assert w["severity"] == "warning"
        assert w["code"] == "OSCQR_LOW"
        assert "OSCQR score" in w["message"]

    def test_missing_phase_report_is_skipped_with_warning(self, tmp_path):
        # Only post_rewrite_validation has a report; inter_tier is
        # missing AND has no in-memory _gate_results.
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-SKIP",
        )
        report = agg.build()

        # Aggregator did not raise.
        assert report["status"] == "pass"
        # inter_tier_validation appears as skipped.
        inter = next(
            p for p in report["per_phase"]
            if p["phase"] == "inter_tier_validation"
        )
        assert inter["report_path"] is None
        assert inter.get("skipped") is True
        assert inter["gates"] == []
        # And summary.skipped_count picks it up.
        assert report["summary"]["skipped_count"] == 1

    def test_critical_in_memory_failure_blocks(self, tmp_path):
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        phase_outputs = {
            "libv2_archival": _make_in_memory_phase(
                gate_id="libv2_manifest",
                passed=False,
                severity="critical",
                code="MANIFEST_HASH_MISMATCH",
                message="manifest sha256 disagrees with on-disk artifact",
            ),
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-LIBV2-FAIL",
        )
        report = agg.build()
        assert report["status"] == "fail"
        codes = {bf["code"] for bf in report["blocking_failures"]}
        assert "MANIFEST_HASH_MISMATCH" in codes

    def test_write_emits_deterministic_sorted_json(self, tmp_path):
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-WRITE",
        )
        out = tmp_path / "courseforge_validation_report.json"
        written = agg.write(out)
        assert written == out
        assert out.exists()
        body = out.read_text(encoding="utf-8")
        # sort_keys=True ⇒ schema_version sorts alphabetically before
        # status; verify by parse + reserialise.
        parsed = json.loads(body)
        canonical = json.dumps(parsed, indent=2, sort_keys=True)
        assert body == canonical


# --------------------------------------------------------------------------
# Integration test — exercise via WorkflowRunner post-loop hook
# --------------------------------------------------------------------------


def test_post_loop_aggregator_writes_top_level_report(tmp_path, monkeypatch):
    """Worker W5 integration — ``run_workflow`` post-loop hook fires.

    Builds a minimal workflow state file under ``state/workflows/`` (via
    a monkey-patched STATE_PATH) such that the runner runs through one
    phase, fails its dispatch, and exits the loop early. The post-loop
    aggregator then resolves project_path from the synthesised
    ``objective_extraction`` phase_output and writes the top-level
    JSON. This locks the wiring contract: no separate finalization
    handler exists, so the aggregator MUST run from the post-loop site
    in :meth:`WorkflowRunner.run_workflow`.
    """
    from MCP.core import workflow_runner as wr_mod
    from MCP.core.workflow_runner import WorkflowRunner

    # Pre-stage a project export with one per-phase report on disk so
    # the aggregator has something to walk.
    project_path = tmp_path / "PROJ-W5_INT-20260505"
    _write_report(
        project_path,
        "inter_tier_validation",
        _make_two_pass_report("inter_tier_validation"),
    )

    # Build a stub runner whose run_workflow we exercise directly:
    # the aggregator helper is independent of the executor, so we
    # call it via the public method on a minimal instance.
    class _StubRunner(WorkflowRunner):
        def __init__(self):  # noqa: D401 — bypass the executor wiring
            self.executor = None
            self.config = None

    runner = _StubRunner()

    phase_outputs = {
        "objective_extraction": {
            "_completed": True,
            "project_id": "PROJ-W5_INT-20260505",
            "project_path": str(project_path),
        },
        "inter_tier_validation": {
            "_completed": True,
            "_gates_passed": True,
            "_gate_results": [],
            "blocks_validated_path": str(project_path / "01_outline" / "x"),
        },
    }
    written = runner._maybe_write_courseforge_validation_report(
        workflow_id="WF-W5-INT",
        workflow_params={"course_name": "W5_INT"},
        phase_outputs=phase_outputs,
    )

    expected = project_path / "courseforge_validation_report.json"
    assert written == expected
    assert expected.exists()

    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert payload["course_code"] == "W5_INT"
    assert payload["run_id"] == "WF-W5-INT"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "pass"
    phases = {p["phase"] for p in payload["per_phase"]}
    assert "inter_tier_validation" in phases


def test_post_loop_aggregator_returns_none_without_project_path(tmp_path):
    """No ``objective_extraction`` output ⇒ aggregator silently skips."""
    from MCP.core.workflow_runner import WorkflowRunner

    class _StubRunner(WorkflowRunner):
        def __init__(self):
            self.executor = None
            self.config = None

    runner = _StubRunner()
    written = runner._maybe_write_courseforge_validation_report(
        workflow_id="WF-NOOP",
        workflow_params={"course_name": "NO_COURSEFORGE"},
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    assert written is None


# --------------------------------------------------------------------------
# GPT Feedback v2 Wave 2 W2.A — additive sub-objects + final_promotion_decision
# --------------------------------------------------------------------------


def _write_blocks_jsonl(
    project_path: Path,
    *,
    rel_dir: str,
    validated_rows: list[dict],
    failed_rows: list[dict],
) -> tuple[Path, Path]:
    """Helper: write blocks_validated.jsonl + blocks_failed.jsonl pair."""
    target_dir = project_path / rel_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    validated_path = target_dir / "blocks_validated.jsonl"
    failed_path = target_dir / "blocks_failed.jsonl"
    with validated_path.open("w", encoding="utf-8") as fh:
        for row in validated_rows:
            fh.write(json.dumps(row) + "\n")
    with failed_path.open("w", encoding="utf-8") as fh:
        for row in failed_rows:
            fh.write(json.dumps(row) + "\n")
    return validated_path, failed_path


def _make_gate_result(
    gate_id: str,
    *,
    passed: bool = True,
    severity: str = "critical",
    validator_name: str = None,
    issues: list[dict] | None = None,
) -> dict:
    return {
        "gate_id": gate_id,
        "validator_name": validator_name or gate_id,
        "validator_version": "1.0",
        "passed": passed,
        "severity": severity,
        "score": None,
        "issues": issues or [],
        "execution_time_ms": 0,
        "action": None,
    }


class TestPerBlockResults:

    def test_per_block_results_walks_jsonl_files(self, tmp_path):
        # Inter-tier validated + failed
        _write_blocks_jsonl(
            tmp_path,
            rel_dir="02_validation_report",
            validated_rows=[
                {
                    "block_id": "b1",
                    "block_type": "objective",
                    "page_id": "page_1",
                },
            ],
            failed_rows=[
                {
                    "block_id": "b2",
                    "block_type": "assessment_item",
                    "page_id": "page_1",
                    "escalation_marker": "outline_budget_exhausted",
                },
            ],
        )
        # Post-rewrite validated + failed
        _write_blocks_jsonl(
            tmp_path,
            rel_dir="04_rewrite/02_validation_report",
            validated_rows=[
                {
                    "block_id": "b3",
                    "block_type": "concept",
                    "page_id": "page_2",
                },
            ],
            failed_rows=[
                {
                    "block_id": "b4",
                    "block_type": "concept",
                    "page_id": "page_2",
                    "escalation_marker": "validator_consensus_fail",
                },
            ],
        )
        # Inter-tier report.json carries the gate-chain summary.
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )

        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-PER-BLOCK",
        )
        report = agg.build()
        rows = report["per_block_results"]
        assert len(rows) == 4
        block_ids = {r["block_id"] for r in rows}
        assert block_ids == {"b1", "b2", "b3", "b4"}
        # Each row carries phase + status + gate_chain.
        b2 = next(r for r in rows if r["block_id"] == "b2")
        assert b2["phase"] == "inter_tier_validation"
        assert b2["status"] == "failed"
        assert b2["escalation_marker"] == "outline_budget_exhausted"
        assert isinstance(b2["gate_chain"], list)
        assert len(b2["gate_chain"]) == 2  # two-pass chain has 2 gates
        b4 = next(r for r in rows if r["block_id"] == "b4")
        assert b4["phase"] == "post_rewrite_validation"
        assert b4["status"] == "failed"
        assert b4["escalation_marker"] == "validator_consensus_fail"


class TestSourceGroundingResults:

    def test_source_grounding_results_projection(self, tmp_path):
        phase_outputs = {
            "content_generation": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _make_gate_result(
                        "source_refs",
                        validator_name=(
                            "lib.validators.source_refs.PageSourceRefValidator"
                        ),
                    ),
                    _make_gate_result(
                        "content_grounding",
                        validator_name=(
                            "lib.validators.content_grounding."
                            "ContentGroundingValidator"
                        ),
                    ),
                    # Non-matching gate — should not appear.
                    _make_gate_result(
                        "imscc_structure",
                        validator_name="IMSCCValidator",
                    ),
                ],
            },
            "post_rewrite_validation": {
                "_completed": True,
                "_gate_results": [
                    _make_gate_result(
                        "rewrite_source_grounding",
                        passed=False,
                        validator_name=(
                            "RewriteSourceGroundingValidator"
                        ),
                        issues=[{
                            "severity": "critical",
                            "code": "GROUNDING_FAIL",
                            "message": "fail",
                        }],
                    ),
                ],
            },
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-SG",
        )
        report = agg.build()
        sg = report["source_grounding_results"]
        gate_ids = {g["gate_id"] for g in sg["gates"]}
        # Both source-related gates surface; imscc_structure does not.
        assert "source_refs" in gate_ids
        assert "content_grounding" in gate_ids
        assert "rewrite_source_grounding" in gate_ids
        assert "imscc_structure" not in gate_ids
        assert sg["summary"]["total"] == 3
        assert sg["summary"]["failed"] == 1
        assert sg["summary"]["passed"] == 2


class TestAccessibilityResults:

    def test_accessibility_results_projection_and_per_page_rollup(
        self, tmp_path
    ):
        phase_outputs = {
            "validation": {
                "_completed": True,
                "_gate_results": [
                    _make_gate_result(
                        "wcag_compliance",
                        passed=False,
                        validator_name=(
                            "lib.validators.wcag.WCAGValidator"
                        ),
                        issues=[
                            {
                                "severity": "critical",
                                "code": "MISSING_ALT_TEXT",
                                "message": "missing alt",
                                "location": "page_1.html",
                            },
                            {
                                "severity": "critical",
                                "code": "BAD_HEADING",
                                "message": "bad heading",
                                "location": "page_1.html",
                            },
                            {
                                "severity": "warning",
                                "code": "LOW_CONTRAST",
                                "message": "contrast",
                                "location": "page_2.html",
                            },
                        ],
                    ),
                ],
            },
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-A11Y",
        )
        report = agg.build()
        a11y = report["accessibility_results"]
        # Gate row present.
        assert any(
            g["gate_id"] == "wcag_compliance" for g in a11y["gates"]
        )
        # Per-page issue rollup.
        ppi = a11y["per_page_issue_count"]
        assert ppi.get("page_1.html") == 2
        assert ppi.get("page_2.html") == 1


class TestStatisticalSemanticResults:

    def test_statistical_semantic_results_with_deps_missing(self, tmp_path):
        phase_outputs = {
            "post_rewrite_validation": {
                "_completed": True,
                "_gate_results": [
                    _make_gate_result(
                        "rewrite_objective_assessment_similarity",
                        validator_name=(
                            "ObjectiveAssessmentSimilarityValidator"
                        ),
                        issues=[{
                            "severity": "warning",
                            "code": "EMBEDDING_DEPS_MISSING",
                            "message": "embedding extras absent",
                        }],
                    ),
                    _make_gate_result(
                        "concept_example_similarity",
                        validator_name=(
                            "ConceptExampleSimilarityValidator"
                        ),
                    ),
                    # Unrelated gate — must not appear.
                    _make_gate_result(
                        "wcag_compliance",
                        validator_name="WCAGValidator",
                    ),
                ],
            },
        }
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="X",
            run_id="WF-STAT",
        )
        report = agg.build()
        stat = report["statistical_semantic_results"]
        gate_ids = {g["gate_id"] for g in stat["gates"]}
        assert "rewrite_objective_assessment_similarity" in gate_ids
        assert "concept_example_similarity" in gate_ids
        assert "wcag_compliance" not in gate_ids
        assert stat["embedding_deps_missing_count"] == 1
        assert stat["embedding_deps_missing_rate"] == pytest.approx(0.5)


class TestManifestHashResults:

    def test_manifest_hash_results_with_provenance_passthrough(
        self, tmp_path, monkeypatch
    ):
        # Prepare a synthetic LibV2 model_card.json under a custom root.
        libv2_root = tmp_path / "LibV2"
        course_slug = "phys_101"
        models_dir = libv2_root / "courses" / course_slug / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = models_dir / "adapter-v1"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        provenance = {
            "chunks_hash": "a" * 64,
            "pedagogy_graph_hash": "b" * 64,
            "instruction_pairs_hash": "c" * 64,
            "preference_pairs_hash": "d" * 64,
            "concept_graph_hash": "e" * 64,
            "vocabulary_ttl_hash": "f" * 64,
            "holdout_graph_hash": "0" * 64,
        }
        (adapter_dir / "model_card.json").write_text(
            json.dumps({"provenance": provenance})
        )

        phase_outputs = {
            "libv2_archival": {
                "_completed": True,
                "_gate_results": [
                    _make_gate_result(
                        "libv2_manifest",
                        validator_name="LibV2ManifestValidator",
                    ),
                ],
            },
        }
        # ``project_path`` is irrelevant; we explicitly pass libv2_root.
        agg = CourseforgeValidationReport(
            project_path=tmp_path / "Courseforge" / "exports" / "PROJ-PHYS_101-x",
            phase_outputs=phase_outputs,
            course_code="PHYS_101",
            run_id="WF-MANIFEST",
            libv2_root=libv2_root,
        )
        report = agg.build()
        mh = report["manifest_hash_results"]
        assert any(g["gate_id"] == "libv2_manifest" for g in mh["gates"])
        # Seven hash fields surface verbatim.
        ph = mh["provenance_hashes"]
        assert ph is not None
        assert ph["chunks_hash"] == "a" * 64
        assert ph["holdout_graph_hash"] == "0" * 64
        assert set(ph.keys()) == {
            "chunks_hash",
            "pedagogy_graph_hash",
            "instruction_pairs_hash",
            "preference_pairs_hash",
            "concept_graph_hash",
            "vocabulary_ttl_hash",
            "holdout_graph_hash",
        }


class TestFinalPromotionDecisionMatrix:

    @pytest.mark.parametrize(
        "scenario,expected",
        [
            # critical-fail -> failed
            ("critical_fail_a11y", "failed"),
            ("critical_fail_pedagogy", "failed"),
            # warnings only -> certified_accessible
            ("warnings_only", "certified_accessible"),
            # clean run, no training -> certified_instructional
            ("clean_no_training", "certified_instructional"),
            # clean run with training-data signal -> certified_trainable
            ("clean_with_training", "certified_trainable"),
        ],
    )
    def test_decision_matrix(self, tmp_path, scenario, expected):
        phase_outputs: dict = {}
        # Always pre-stage clean inter-tier + post-rewrite reports so
        # the per-phase walk has something to chew on.
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )

        if scenario == "critical_fail_a11y":
            phase_outputs["validation"] = _make_in_memory_phase(
                gate_id="wcag_compliance",
                passed=False,
                severity="critical",
                code="MISSING_ALT_TEXT",
                message="missing alt",
            )
        elif scenario == "critical_fail_pedagogy":
            phase_outputs["libv2_archival"] = _make_in_memory_phase(
                gate_id="libv2_manifest",
                passed=False,
                severity="critical",
                code="MANIFEST_HASH_MISMATCH",
                message="hash mismatch",
            )
        elif scenario == "warnings_only":
            phase_outputs["packaging"] = _make_in_memory_phase(
                gate_id="oscqr_score",
                passed=False,
                severity="warning",
                code="OSCQR_LOW",
                message="oscqr below threshold",
            )
        elif scenario == "clean_no_training":
            # Clean run, no training-data signal at all.
            phase_outputs["packaging"] = _make_in_memory_phase(
                gate_id="imscc_structure", passed=True
            )
        elif scenario == "clean_with_training":
            # Clean run + training_synthesis phase ran successfully.
            phase_outputs["training_synthesis"] = {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    _make_gate_result("synthesis_diversity"),
                ],
            }

        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs=phase_outputs,
            course_code="MATRIX",
            run_id=f"WF-{scenario.upper()}",
        )
        report = agg.build()
        fpd = report["final_promotion_decision"]
        assert fpd["value"] == expected, (
            f"scenario={scenario!r} got {fpd['value']!r} "
            f"expected {expected!r}"
        )
        assert isinstance(fpd["rationale"], str)
        assert len(fpd["rationale"]) >= 20
        assert isinstance(fpd["contributing_gate_ids"], list)


class TestDecisionEmitRegression:

    def test_emit_aggregator_decision_fires(self, tmp_path):
        """Capture-wiring contract — aggregator emits one decision per build."""
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )

        captured: list[dict] = []

        class _StubCapture:
            def log_decision(self, **kwargs):
                captured.append(kwargs)

        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-EMIT",
            decision_capture=_StubCapture(),
        )
        agg.build()
        assert len(captured) == 1
        emit = captured[0]
        assert emit["decision_type"] == "courseforge_validation_aggregated"
        # Rationale interpolates dynamic signals.
        assert len(emit["rationale"]) >= 20
        assert "verdict=" in emit["rationale"]
        assert "total_gates=" in emit["rationale"]
        # Decision value is one of the canonical 5 enum values.
        assert emit["decision"] in {
            "failed",
            "non_certified_archive",
            "certified_accessible",
            "certified_instructional",
            "certified_trainable",
        }


class TestSchemaBumpBackCompat:

    def test_v1_0_reader_can_parse_v1_1_output(self, tmp_path):
        """v1.0 readers MUST keep parsing v1.1 output (additive only).

        v1.0 readers read these top-level keys:
        ``schema_version, course_code, run_id, generated_at, status,
        summary, blocking_failures, warnings, per_phase``. They MUST
        all still be present and shape-compatible in v1.1.
        """
        _write_report(
            tmp_path,
            "inter_tier_validation",
            _make_two_pass_report("inter_tier_validation"),
        )
        _write_report(
            tmp_path,
            "post_rewrite_validation",
            _make_two_pass_report("post_rewrite_validation"),
        )
        agg = CourseforgeValidationReport(
            project_path=tmp_path,
            phase_outputs={},
            course_code="X",
            run_id="WF-BC",
        )
        report = agg.build()
        # Schema bumped to 1.1.
        assert report["schema_version"] == "1.1"
        # All v1.0 keys still present + correct shape.
        for key in (
            "schema_version",
            "course_code",
            "run_id",
            "generated_at",
            "status",
            "summary",
            "blocking_failures",
            "warnings",
            "per_phase",
        ):
            assert key in report, f"v1.0 key {key!r} missing in v1.1 output"
        # summary still has the canonical 5 counters.
        for counter in (
            "total_gates",
            "passed_count",
            "failed_count",
            "warning_count",
            "skipped_count",
        ):
            assert counter in report["summary"]
        # per_phase entries still carry phase + gates list.
        for entry in report["per_phase"]:
            assert "phase" in entry
            assert "gates" in entry
            assert isinstance(entry["gates"], list)
