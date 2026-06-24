"""Per-block ``issue_count`` attribution in ``_write_validation_report``.

Regression for the schema-v1 bug where every block in
``02_validation_report/report.json`` received the SAME phase-level
``gate_results`` list — each entry's ``issue_count`` was the COURSE-WIDE
total — so a gate that fired on ONE block (or on an objective/page, not a
block at all) appeared to fire on EVERY block, inflating the calibration
harness's per-gate fire-rates toward 100%.

Schema v2 fix:

* Each ``GateResult.issues[]`` carries a ``location`` field. A BLOCK-level
  gate sets ``location`` to a ``block_id``; OBJECTIVE / PAGE / summary
  issues set it to a non-block id or ``None``.
* Per block, ``gate_results[].issue_count`` counts ONLY the issues whose
  ``location`` equals THAT block's ``block_id``.
* A new top-level ``phase_level_gate_results`` preserves structural
  (non-block-attributable) issues: per gate the TOTAL issue_count + the
  ``unattributed_issue_count`` (issues whose location matched no block).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    return WorkflowRunner(executor=object(), config=object())


def _scaffold_two_block_export(tmp_path: Path) -> tuple[Path, str]:
    """Two validated blocks (A, B) under a rewrite-tier project export.

    Returns (project_dir, blocks_validated_path).
    """
    project = tmp_path / "exports" / "PROJ-ATTRIB_101-20260624"
    rewrite = project / "04_rewrite"
    rewrite.mkdir(parents=True)

    blocks = [
        {
            "block_id": "blockA",
            "block_type": "hook",
            "page_id": "week_01_overview",
            "week": 1,
            "content": "<section><p>Block A.</p></section>",
        },
        {
            "block_id": "blockB",
            "block_type": "concept",
            "page_id": "week_01_overview",
            "week": 1,
            "content": "<section><p>Block B.</p></section>",
        },
    ]
    blocks_validated = rewrite / "blocks_validated.jsonl"
    blocks_validated.write_text(
        "\n".join(json.dumps(b) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return project, str(blocks_validated)


def test_issue_count_attributed_per_block_not_smeared(runner_stub, tmp_path):
    project, blocks_validated_path = _scaffold_two_block_export(tmp_path)

    # A BLOCK-level gate whose issues name block A ONLY (twice), plus an
    # OBJECTIVE-level gate whose issue locates a CO id (no block), plus a
    # summary issue with location=None.
    gate_results_list = [
        {
            "gate_id": "udl_coverage",
            "action": "regenerate",
            "passed": False,  # phase-level: the gate failed for the phase
            "issues": [
                {"severity": "warning", "code": "UDL_X",
                 "message": "m1", "location": "blockA"},
                {"severity": "warning", "code": "UDL_Y",
                 "message": "m2", "location": "blockA"},
            ],
        },
        {
            "gate_id": "triangle_completeness",
            "action": "block",
            "passed": False,
            "issues": [
                {"severity": "critical", "code": "TRIANGLE_X",
                 "message": "co", "location": "CO-01"},
                # summary issue, no location
                {"severity": "warning", "code": "TRIANGLE_SUMMARY",
                 "message": "rollup", "location": None},
            ],
        },
    ]

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-ATTRIB-TEST",
        phase_name="post_rewrite_validation",
        phase_output={"blocks_validated_path": blocks_validated_path},
        gate_results_list=gate_results_list,
    )
    assert report_path is not None
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "v2"

    # --- Per-block attribution: block A shows udl_coverage=2, block B=0. ---
    by_id = {b["block_id"]: b for b in report["per_block"]}
    assert set(by_id) == {"blockA", "blockB"}

    def _count(block_id: str, gate_id: str) -> int:
        for gr in by_id[block_id]["gate_results"]:
            if gr["gate_id"] == gate_id:
                return gr["issue_count"]
        raise AssertionError(f"{gate_id} missing on {block_id}")

    # The bug this guards against: udl_coverage was 2 on BOTH blocks.
    assert _count("blockA", "udl_coverage") == 2
    assert _count("blockB", "udl_coverage") == 0

    # The objective-level gate names NO block, so it is 0 on every block.
    assert _count("blockA", "triangle_completeness") == 0
    assert _count("blockB", "triangle_completeness") == 0

    # The per-gate ``passed`` flag stays PHASE-level on every block.
    for block_id in ("blockA", "blockB"):
        for gr in by_id[block_id]["gate_results"]:
            assert gr["passed"] is False

    # --- Phase-level section: structural issues are visible here. ---
    phase = {g["gate_id"]: g for g in report["phase_level_gate_results"]}
    assert set(phase) == {"udl_coverage", "triangle_completeness"}

    # udl_coverage: 2 total issues, both attributed to block A -> 0
    # unattributed.
    assert phase["udl_coverage"]["issue_count"] == 2
    assert phase["udl_coverage"]["unattributed_issue_count"] == 0

    # triangle_completeness: 2 total (CO-01 + None-location), neither maps
    # to a block_id -> both unattributed.
    assert phase["triangle_completeness"]["issue_count"] == 2
    assert phase["triangle_completeness"]["unattributed_issue_count"] == 2


def test_empty_gate_results_yields_empty_phase_section(runner_stub, tmp_path):
    project, blocks_validated_path = _scaffold_two_block_export(tmp_path)
    report_path = runner_stub._write_validation_report(
        workflow_id="WF-EMPTY",
        phase_name="post_rewrite_validation",
        phase_output={"blocks_validated_path": blocks_validated_path},
        gate_results_list=[],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["phase_level_gate_results"] == []
    for b in report["per_block"]:
        assert b["gate_results"] == []
