"""Roadmap T3 — :class:`AccessibilityConformanceAggregator` tests.

Coverage:

1. Clean run — no WCAG issues → evaluable criteria support, non-evaluable
   criteria are explicit not_evaluated rows (contrast / media / cognitive).
2. Critical issue inverts to does_not_support with evidence counts + pages.
3. Warning-only issue inverts to partially_supports.
4. On-disk courseforge_validation_report.json accessibility_results fallback.
5. Union of in-memory gate results + on-disk report.
6. Schema validation of the emitted report.
7. Every WCAG 2.2 A+AA criterion appears exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.aggregators.accessibility_conformance import (
    DOES_NOT_SUPPORT,
    NOT_EVALUATED,
    PARTIALLY_SUPPORTS,
    SCHEMA_VERSION,
    SUPPORTS,
    AccessibilityConformanceAggregator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "aggregators" / "accessibility_conformance.schema.json"
)


def _wcag_gate_phase(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a phase_outputs entry carrying one WCAG gate result."""
    return {
        "_gate_results": [
            {
                "gate_id": "wcag_compliance",
                "validator_name": "WCAGValidator",
                "passed": not any(
                    i.get("severity") == "critical" for i in issues
                ),
                "issues": issues,
            }
        ]
    }


def _rows_by_criterion(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["criterion"]: r for r in report["criteria"]}


def test_clean_run_supports_and_not_evaluated():
    agg = AccessibilityConformanceAggregator(
        phase_outputs={}, course_code="X", run_id="R1"
    )
    report = agg.build()
    assert report["schema_version"] == SCHEMA_VERSION
    rows = _rows_by_criterion(report)

    # Evaluable structural criterion with no issues → supports.
    assert rows["1.1.1"]["status"] == SUPPORTS
    # Contrast COMPUTATION cannot be statically evaluated → not_evaluated.
    assert rows["1.4.3"]["status"] == NOT_EVALUATED
    assert rows["1.4.3"]["reason"] == "contrast_computation"
    # Time-based media → not_evaluated.
    assert rows["1.2.2"]["status"] == NOT_EVALUATED
    assert rows["1.2.2"]["reason"] == "time_based_media"

    summary = report["summary"]
    assert summary["does_not_support"] == 0
    assert summary["partially_supports"] == 0
    assert summary["conformant"] is True
    assert summary["not_evaluated"] > 0
    assert summary["supports"] > 0


def test_critical_issue_inverts_to_does_not_support():
    phase_outputs = {
        "packaging": _wcag_gate_phase(
            [
                {
                    "code": "WCAG_1_1_1",
                    "severity": "critical",
                    "location": "Module_1.html",
                },
                {
                    "code": "WCAG_1_1_1",
                    "severity": "critical",
                    "location": "Module_2.html",
                },
            ]
        )
    }
    report = AccessibilityConformanceAggregator(
        phase_outputs=phase_outputs, run_id="R2"
    ).build()
    row = _rows_by_criterion(report)["1.1.1"]
    assert row["status"] == DOES_NOT_SUPPORT
    assert row["evidence_counts"]["critical"] == 2
    assert row["evidence_counts"]["pages"] == 2
    assert row["pages"] == ["Module_1.html", "Module_2.html"]
    assert report["summary"]["conformant"] is False


def test_warning_only_inverts_to_partially_supports():
    phase_outputs = {
        "packaging": _wcag_gate_phase(
            [{"code": "WCAG_2_4_6", "severity": "warning", "location": "p1"}]
        )
    }
    report = AccessibilityConformanceAggregator(
        phase_outputs=phase_outputs
    ).build()
    row = _rows_by_criterion(report)["2.4.6"]
    assert row["status"] == PARTIALLY_SUPPORTS
    assert row["evidence_counts"]["warning"] == 1
    assert report["summary"]["conformant"] is False


def test_on_disk_report_fallback(tmp_path: Path):
    project = tmp_path / "PROJ"
    project.mkdir()
    (project / "courseforge_validation_report.json").write_text(
        json.dumps(
            {
                "accessibility_results": {
                    "results": [
                        {
                            "gate_id": "wcag_compliance",
                            "top_issues": [
                                {
                                    "code": "WCAG_1_3_1",
                                    "severity": "critical",
                                    "location": "page-3",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    report = AccessibilityConformanceAggregator(
        phase_outputs={}, project_path=project
    ).build()
    assert _rows_by_criterion(report)["1.3.1"]["status"] == DOES_NOT_SUPPORT


def test_union_memory_and_disk(tmp_path: Path):
    project = tmp_path / "PROJ"
    project.mkdir()
    (project / "courseforge_validation_report.json").write_text(
        json.dumps(
            {
                "accessibility_results": {
                    "results": [
                        {
                            "top_issues": [
                                {
                                    "code": "WCAG_1_1_1",
                                    "severity": "warning",
                                    "location": "disk-page",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    phase_outputs = {
        "packaging": _wcag_gate_phase(
            [{"code": "WCAG_1_1_1", "severity": "critical", "location": "mem-page"}]
        )
    }
    report = AccessibilityConformanceAggregator(
        phase_outputs=phase_outputs, project_path=project
    ).build()
    row = _rows_by_criterion(report)["1.1.1"]
    # Critical (memory) + warning (disk) → does_not_support, both pages.
    assert row["status"] == DOES_NOT_SUPPORT
    assert row["evidence_counts"]["critical"] == 1
    assert row["evidence_counts"]["warning"] == 1
    assert set(row["pages"]) == {"mem-page", "disk-page"}


def test_every_criterion_appears_once():
    report = AccessibilityConformanceAggregator().build()
    criteria = [r["criterion"] for r in report["criteria"]]
    assert len(criteria) == len(set(criteria))
    assert report["summary"]["total_criteria"] == len(criteria)
    # sanity: canonical anchors present
    assert "1.1.1" in criteria
    assert "3.3.8" in criteria


def test_emitted_report_validates_against_schema(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    phase_outputs = {
        "packaging": _wcag_gate_phase(
            [
                {"code": "WCAG_1_1_1", "severity": "critical", "location": "p1"},
                {"code": "WCAG_2_4_6", "severity": "warning", "location": "p2"},
            ]
        )
    }
    out = tmp_path / "quality" / "accessibility_conformance.json"
    AccessibilityConformanceAggregator(
        phase_outputs=phase_outputs, course_code="C", run_id="R"
    ).write(out)
    report = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
