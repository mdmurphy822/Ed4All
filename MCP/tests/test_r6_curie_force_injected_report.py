"""R6 — force-injected CURIE blocks visible in ``02_validation_report``.

Pins the R6 silent-degradation fix on the two-pass validation report
writer ``MCP/core/workflow_runner.py::_write_validation_report``.

Problem closed: when the rewrite LLM drops a block's CURIEs after the
remediation budget, ``RewriteProvider`` force-injects them as an
appended ``<span hidden data-cf-curie="..." data-cf-curie-forced="true">``
and the block legitimately PASSES the ``rewrite_curie_anchoring`` gate.
Before R6 such a block was recorded as a plain ``status="passed"`` —
indistinguishable from a clean rewrite. R6 adds a durable, report-
reachable signal (the ``data-cf-curie-forced`` boolean attribute lives
inside ``block.content``, the one Block field that survives every JSONL
round trip) and surfaces it as ``per_block[*].curie_force_injected`` plus
a top-level ``curie_force_injected`` aggregate count — WITHOUT flipping
the block's pass verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402
from Courseforge.generators.rewrite._rewrite_provider import (  # noqa: E402
    _force_inject_curies,
    html_has_forced_curie_marker,
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


# ---------------------------------------------------------------------------
# Marker detection (RewriteProvider side)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_force_inject_stamps_forced_marker() -> None:
    """``_force_inject_curies`` appends the ``data-cf-curie-forced`` marker."""
    out = _force_inject_curies("<p>plain prose</p>", ["ex:concept_a"])
    assert 'data-cf-curie-forced="true"' in out
    assert "ex:concept_a" in out
    assert html_has_forced_curie_marker(out) is True


@pytest.mark.unit
def test_clean_html_has_no_forced_marker() -> None:
    """A clean rewrite (no force-injection) carries no marker."""
    clean = "<section><p>The block discusses ex:concept_a in prose.</p></section>"
    assert html_has_forced_curie_marker(clean) is False
    # A bare data-cf-curie attribute (no forced marker) is NOT a match.
    assert html_has_forced_curie_marker(
        '<span data-cf-curie="ex:c">ex:c</span>'
    ) is False
    assert html_has_forced_curie_marker(None) is False
    assert html_has_forced_curie_marker("") is False


# ---------------------------------------------------------------------------
# Report writer surfacing (workflow_runner side)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r6_force_injected_block_flagged_in_report(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """A force-injected passing block carries ``curie_force_injected: true``.

    Its ``status`` stays ``passed`` (the gate legitimately passed it);
    a clean passing sibling carries no ``curie_force_injected`` key.
    """
    project_root = tmp_path / "project"
    rewrite_dir = project_root / "04_rewrite"
    validated_path = rewrite_dir / "blocks_validated.jsonl"

    forced_html = _force_inject_curies(
        "<section><p>prose body</p></section>", ["ex:dropped_curie"]
    )
    _write_blocks_jsonl(validated_path, [
        # Clean rewrite — no forced span.
        {"block_id": "clean1", "block_type": "concept", "page_id": "p1",
         "content": "<section><p>A clean concept body.</p></section>"},
        # Force-injected block — passes the gate, but the span is the tell.
        {"block_id": "forced1", "block_type": "objective", "page_id": "p1",
         "content": forced_html},
    ])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-R6-001",
        phase_name="post_rewrite_validation",
        phase_output={"blocks_validated_path": str(validated_path)},
        gate_results_list=[],
    )
    assert report_path is not None and report_path.exists()
    report = json.loads(report_path.read_text())

    # Aggregate header count.
    assert report["curie_force_injected"] == 1
    # Both blocks still count as passed — verdict is NOT flipped.
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["escalated"] == 0

    by_id = {b["block_id"]: b for b in report["per_block"]}
    forced = by_id["forced1"]
    clean = by_id["clean1"]

    # Force-injected block: flagged, but status stays passed.
    assert forced["status"] == "passed"
    assert forced["curie_force_injected"] is True

    # Clean block: passed and the flag key is absent (byte-stable emit).
    assert clean["status"] == "passed"
    assert "curie_force_injected" not in clean


@pytest.mark.unit
def test_r6_no_force_injection_zero_count(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """A report with only clean rewrites has ``curie_force_injected == 0``."""
    project_root = tmp_path / "project"
    validated_path = project_root / "01_outline" / "blocks_validated.jsonl"
    _write_blocks_jsonl(validated_path, [
        {"block_id": "b1", "block_type": "objective", "page_id": "p1",
         "content": "<section><p>clean</p></section>"},
        {"block_id": "b2", "block_type": "concept", "page_id": "p1",
         "content": "<section><p>also clean</p></section>"},
    ])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-R6-002",
        phase_name="inter_tier_validation",
        phase_output={"blocks_validated_path": str(validated_path)},
        gate_results_list=[],
    )
    report = json.loads(report_path.read_text())
    assert report["curie_force_injected"] == 0
    assert all(
        "curie_force_injected" not in b for b in report["per_block"]
    )


@pytest.mark.unit
def test_r6_force_injected_failed_block_still_flagged(
    runner_stub: WorkflowRunner, tmp_path: Path
) -> None:
    """A force-injected block that also lands in ``blocks_failed.jsonl``
    keeps its failed/escalated status AND the force-injected flag.

    Defensive: force-injection and a separate gate failure are
    orthogonal — the flag is content-derived, not status-derived.
    """
    project_root = tmp_path / "project"
    outline_dir = project_root / "01_outline"
    validated_path = outline_dir / "blocks_validated.jsonl"
    failed_path = outline_dir / "blocks_failed.jsonl"

    forced_html = _force_inject_curies("<p>body</p>", ["ex:c"])
    _write_blocks_jsonl(validated_path, [])
    _write_blocks_jsonl(failed_path, [
        {"block_id": "f1", "block_type": "concept", "page_id": "p1",
         "content": forced_html},
    ])

    report_path = runner_stub._write_validation_report(
        workflow_id="WF-R6-003",
        phase_name="inter_tier_validation",
        phase_output={
            "blocks_validated_path": str(validated_path),
            "blocks_failed_path": str(failed_path),
        },
        gate_results_list=[],
    )
    report = json.loads(report_path.read_text())
    assert report["curie_force_injected"] == 1
    rec = report["per_block"][0]
    assert rec["status"] == "failed"
    assert rec["curie_force_injected"] is True
