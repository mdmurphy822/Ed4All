"""``courseforge-validate`` stage subcommand end-to-end report write.

Regression for the bug where ``ed4all run courseforge-validate
--course-name <X>`` finished ``status=failed`` with the
``post_rewrite_validation`` phase reporting "zero successful tasks and
no outputs (dispatched=1, complete=0, keys=[])" and NO
``02_validation_report/report.json`` ever written.

Root cause: ``post_rewrite_validation`` resolves its ``blocks_final_path``
``inputs_from`` against ``content_generation_rewrite``'s phase output,
but the validate stage SKIPS ``content_generation_rewrite`` and the
pre-loop synthesizer (``_synthesize_outline_output``) stopped at
``inter_tier_validation`` — so ``blocks_final_path`` was empty, the
handler returned ``{"error": "blocks_final_path is required"}``, the
anti-zombie guard failed the phase, and the in-loop report writer (which
only runs when the phase does NOT break) was never reached.

These tests pin the two halves of the fix:

1. ``_synthesize_outline_output`` reconstructs
   ``content_generation_rewrite.blocks_final_path`` from
   ``04_rewrite/blocks_final.jsonl`` for the validate stage.
2. The post_rewrite handler, fed that real on-disk path, runs the
   validators and the report writer emits
   ``04_rewrite/02_validation_report/report.json`` with per-gate /
   per-block pass-fail counts.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _run_post_rewrite_validation,
)


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    return WorkflowRunner(executor=object(), config=object())


def _scaffold_validate_export(tmp_path: Path) -> Path:
    """Build a minimal project export with a real 04_rewrite/blocks_final.jsonl.

    Mirrors the on-disk layout a completed two-pass run leaves behind:
    project_config.json + 01_learning_objectives/synthesized_objectives.json
    + 04_rewrite/blocks_final.jsonl.
    """
    project = tmp_path / "exports" / "PROJ-FXCALIB_101-20260620"
    (project / "01_learning_objectives").mkdir(parents=True)
    (project / "04_rewrite").mkdir(parents=True)

    project.joinpath("project_config.json").write_text(
        json.dumps({
            "course_name": "FXCALIB_101",
            "project_id": project.name,
            "duration_weeks": 8,
        }),
        encoding="utf-8",
    )
    project.joinpath(
        "01_learning_objectives", "synthesized_objectives.json"
    ).write_text(
        json.dumps({
            "course_name": "FXCALIB_101",
            "duration_weeks": 8,
            "terminal_objectives": [
                {"id": "TO-01", "statement": "T1", "bloom_level": "understand"},
            ],
            "chapter_objectives": [
                {
                    "chapter": "Week 1",
                    "objectives": [
                        {"id": "CO-01", "statement": "C1",
                         "parent_terminal": "TO-01"},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )

    # A real rewrite-tier blocks_final.jsonl (snake_case emit shape).
    blocks = [
        {
            "block_id": "w1-b1",
            "block_type": "objective",
            "page_id": "week_01_overview",
            "sequence": 0,
            "week": 1,
            "content": "<section><p>By the end you can do CO-01.</p></section>",
            "objective_ids": ["CO-01"],
            "source_ids": ["semantik:ch1#p1"],
        },
        {
            "block_id": "w1-b2",
            "block_type": "concept",
            "page_id": "week_01_overview",
            "sequence": 1,
            "week": 1,
            "content": "<section><p>The concept is explained here.</p></section>",
            "objective_ids": ["CO-01"],
            "source_ids": ["semantik:ch1#p2"],
        },
    ]
    project.joinpath("04_rewrite", "blocks_final.jsonl").write_text(
        "\n".join(json.dumps(b) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return project


def test_validate_stage_resolves_path_and_writes_report(
    runner_stub, tmp_path, monkeypatch
):
    project = _scaffold_validate_export(tmp_path)

    # The handler resolves objectives via courseforge_exports_dir()/<id>.
    exports_root = project.parent
    monkeypatch.setattr(
        "MCP.tools.pipeline_tools.courseforge_exports_dir",
        lambda: exports_root,
    )

    # 1. The synthesizer reconstructs the rewrite-tier output for the
    #    validate stage (the dispatch-side half of the fix).
    validate_active = (
        WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_validate"
        )
    )
    synth = runner_stub._synthesize_outline_output(
        project, stage_active_phases=validate_active
    )
    assert "content_generation_rewrite" in synth
    blocks_final_path = synth["content_generation_rewrite"]["blocks_final_path"]
    assert Path(blocks_final_path).exists()

    # 2. The post_rewrite handler, fed the resolved real path, runs the
    #    validators and returns a non-empty envelope (NOT the
    #    "blocks_final_path is required" / no-output failure).
    envelope = asyncio.run(
        _run_post_rewrite_validation(
            blocks_final_path=blocks_final_path,
            project_id=project.name,
        )
    )
    payload = json.loads(envelope)
    assert payload["success"] is True, payload
    assert payload["block_count"] == 2
    assert "gate_results" in payload and payload["gate_results"]
    validated_path = Path(payload["blocks_validated_path"])
    assert validated_path.exists()
    # The validated JSONL lives under 04_rewrite/ (out_dir = blocks parent).
    assert validated_path.parent == project / "04_rewrite"

    # 3. The report writer emits 04_rewrite/02_validation_report/report.json
    #    with per-gate / per-block counts (the operator-facing deliverable
    #    that was never written when the phase failed early).
    report_path = runner_stub._write_validation_report(
        workflow_id="WF-CALIB-TEST",
        phase_name="post_rewrite_validation",
        phase_output={
            "blocks_validated_path": payload["blocks_validated_path"],
            "blocks_failed_path": payload["blocks_failed_path"],
        },
        gate_results_list=payload["gate_results"],
    )
    assert report_path is not None
    assert report_path == (
        project / "04_rewrite" / "02_validation_report" / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["phase"] == "post_rewrite_validation"
    assert report["total_blocks"] == 2
    # Per-block records present with a status verdict each.
    assert len(report["per_block"]) == 2
    for rec in report["per_block"]:
        assert rec["status"] in ("passed", "failed", "escalated")
    # passed + failed + escalated accounts for every block.
    assert (
        report["passed"] + report["failed"] + report["escalated"]
        == report["total_blocks"]
    )


def test_validate_stage_missing_rewrite_fails_loud_not_silent(
    runner_stub, tmp_path, monkeypatch
):
    """Anti-fabrication: no 04_rewrite/blocks_final.jsonl => the phase is
    NOT synthesized (no placeholder path), and the handler invoked with
    an empty path fails LOUDLY with a clear message rather than silently
    emitting no outputs."""
    project = _scaffold_validate_export(tmp_path)
    # Remove the rewrite output to simulate an export missing the tier.
    (project / "04_rewrite" / "blocks_final.jsonl").unlink()

    validate_active = (
        WorkflowRunner._resolve_courseforge_stage_active_phases(
            "courseforge_validate"
        )
    )
    synth = runner_stub._synthesize_outline_output(
        project, stage_active_phases=validate_active
    )
    assert "content_generation_rewrite" not in synth

    # The handler with an empty path returns a clear, explicit error.
    envelope = asyncio.run(
        _run_post_rewrite_validation(
            blocks_final_path="",
            project_id=project.name,
        )
    )
    payload = json.loads(envelope)
    assert payload["success"] is False
    assert "blocks_final_path is required" in payload["error"]
