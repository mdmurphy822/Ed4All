"""W3.H sub-task H2 — two-pass validation report ``source_coverage`` emit.

Pins the canonical W3.H ``source_coverage`` block on the two-pass
``02_validation_report/report.json`` writer in
``MCP/core/workflow_runner.py::_write_validation_report``. Three tests:

* canonical shape (5 fields present, coverage_pct calc, dropped_count
  == sum invariant);
* ``INTERNAL_DROP_REASON_MISSING`` fires when drop reasons don't
  balance (synthesized via more failed blocks than enumerable
  escalation_marker buckets);
* back-compat — a legacy report (no ``source_coverage`` field) still
  has every other key the old writer produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402
from lib.governance.source_coverage import (  # noqa: E402
    INTERNAL_DROP_REASON_MISSING,
)


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    return WorkflowRunner(executor=object(), config=object())


def _write_blocks_jsonl(path: Path, blocks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(b) for b in blocks),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_h2_source_coverage_canonical_shape(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """Validated + failed JSONL drives a well-formed source_coverage block."""
    project_root = tmp_path / "project"
    outline_dir = project_root / "01_outline"
    validated_path = outline_dir / "blocks_validated.jsonl"
    failed_path = outline_dir / "blocks_failed.jsonl"

    _write_blocks_jsonl(validated_path, [
        {"block_id": "b1", "block_type": "objective"},
        {"block_id": "b2", "block_type": "concept"},
        {"block_id": "b3", "block_type": "concept"},
    ])
    _write_blocks_jsonl(failed_path, [
        {"block_id": "b4", "block_type": "objective",
         "escalation_marker": "outline_budget_exhausted"},
        {"block_id": "b5", "block_type": "concept",
         "escalation_marker": "validator_consensus_fail"},
    ])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-TEST-H2-001",
        phase_name="inter_tier_validation",
        phase_output={
            "blocks_validated_path": str(validated_path),
            "blocks_failed_path": str(failed_path),
        },
        gate_results_list=[],
    )
    assert report_path is not None and report_path.exists()
    report = json.loads(report_path.read_text())
    assert "source_coverage" in report
    block = report["source_coverage"]
    assert set(block.keys()) == {
        "consumed_count",
        "emitted_count",
        "dropped_count",
        "drop_reasons",
        "coverage_pct",
    }
    # 3 passed + 2 escalated = 5 attempted; 3 emitted; 2 dropped.
    assert block["consumed_count"] == 5
    assert block["emitted_count"] == 3
    assert block["dropped_count"] == 2
    assert block["coverage_pct"] == pytest.approx(0.6, rel=1e-3)
    # Both canonical W3.H drop-reason buckets surface.
    assert block["drop_reasons"].get("regen_budget") == 1
    assert block["drop_reasons"].get("validator_consensus_fail") == 1
    assert block["dropped_count"] == sum(block["drop_reasons"].values())


@pytest.mark.unit
def test_h2_internal_drop_reason_missing_fires(
    runner_stub: WorkflowRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the silent-gaming check by forcing an unattributed drop.

    Patches the build helper's behaviour by populating the
    failed-blocks JSONL with entries whose escalation_marker is None
    AND status is reclassified as 'failed' — those produce
    ``validation_failed`` bucket entries (canonical fallback). To
    actually trigger ``INTERNAL_DROP_REASON_MISSING`` we'd need the
    histogram total to be less than dropped_count; the structural
    contract enumerates EVERY non-passing block so the only way to
    trip the silent-gaming check at this layer is via the helper
    directly. We do that here so the assertion fires at the public
    surface.
    """
    import logging
    from lib.governance.source_coverage import build_source_coverage

    caplog_logger = logging.getLogger("lib.governance.source_coverage")
    handler_records: list[logging.LogRecord] = []
    h = logging.Handler()
    h.emit = lambda r: handler_records.append(r)
    caplog_logger.addHandler(h)
    caplog_logger.setLevel(logging.WARNING)
    try:
        block = build_source_coverage(
            consumed_count=10,
            emitted_count=4,
            drop_reasons={"validator_consensus_fail": 2},
            dropped_count=6,
            label="two_pass_inter_tier_validation_test",
        )
    finally:
        caplog_logger.removeHandler(h)

    # The unattributed delta (6 - 2 = 4) lands in the canonical bucket.
    assert block["drop_reasons"].get(INTERNAL_DROP_REASON_MISSING) == 4
    assert block["dropped_count"] == sum(block["drop_reasons"].values())
    assert any(
        "two_pass_inter_tier_validation_test" in (r.getMessage() or "")
        for r in handler_records
    )


@pytest.mark.unit
def test_h2_zero_blocks_zero_coverage(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """Empty validated + failed JSONLs yield 0 coverage_pct (no NaN)."""
    project_root = tmp_path / "project"
    outline_dir = project_root / "01_outline"
    validated_path = outline_dir / "blocks_validated.jsonl"
    failed_path = outline_dir / "blocks_failed.jsonl"
    _write_blocks_jsonl(validated_path, [])
    _write_blocks_jsonl(failed_path, [])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-EMPTY",
        phase_name="inter_tier_validation",
        phase_output={
            "blocks_validated_path": str(validated_path),
            "blocks_failed_path": str(failed_path),
        },
        gate_results_list=[],
    )
    report = json.loads(report_path.read_text())
    assert report["source_coverage"]["coverage_pct"] == 0.0


@pytest.mark.unit
def test_h2_post_rewrite_writes_to_rewrite_dir(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """``post_rewrite_validation`` writes under ``04_rewrite/`` per Phase 5 §6."""
    project_root = tmp_path / "project"
    rewrite_dir = project_root / "04_rewrite"
    validated_path = rewrite_dir / "blocks_validated.jsonl"
    _write_blocks_jsonl(validated_path, [
        {"block_id": "rb1", "block_type": "objective"},
    ])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-RW-001",
        phase_name="post_rewrite_validation",
        phase_output={
            "blocks_validated_path": str(validated_path),
        },
        gate_results_list=[],
    )
    assert report_path is not None
    # Per the writer contract: post-rewrite reports live under
    # 04_rewrite/02_validation_report/.
    assert "04_rewrite" in str(report_path)
    assert "02_validation_report" in str(report_path)
    report = json.loads(report_path.read_text())
    assert "source_coverage" in report
    assert report["source_coverage"]["consumed_count"] == 1
    assert report["source_coverage"]["emitted_count"] == 1
    assert report["source_coverage"]["coverage_pct"] == pytest.approx(1.0, rel=1e-3)
