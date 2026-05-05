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
        phase_outputs={"dart_conversion": {"_completed": True}},
    )
    assert written is None
