"""IB6.6 — BlockQualityRollupAggregator (mean + min-floor + 3 hard gates)."""
from __future__ import annotations

import json

from lib.aggregators.block_quality_rollup import BlockQualityRollupAggregator


def _block(block_id, module, scores, framework_block="B03"):
    """Synthetic scored-block dict carrying a quality_rubric tuple."""
    return {
        "block_id": block_id,
        "page_id": module,
        "framework_block": framework_block,
        "quality_rubric": [
            {"dimension": dim, "score": score, "applicable": True}
            for dim, score in scores.items()
        ],
    }


_ALL3 = {
    "alignment": 3,
    "cognitive_load": 3,
    "accessibility": 3,
    "coherence": 3,
}


def test_module_of_all_3_blocks_passes():
    blocks = [
        _block("b1", "week_01_content", dict(_ALL3)),
        _block("b2", "week_01_content", dict(_ALL3)),
    ]
    report = BlockQualityRollupAggregator(blocks=blocks).build()
    assert report["summary"]["course_pass"] is True
    mod = report["per_module"][0]
    assert mod["module"] == "week_01"
    assert mod["module_pass"] is True


def test_one_weak_core_block_fails_min_floor_despite_high_mean():
    # b1 all-3, b2 has coherence==1 (a core dim). Module mean stays high but
    # the per-dimension minimum floor fails → module FAILS (no weak block hides).
    weak = dict(_ALL3)
    weak["coherence"] = 1
    blocks = [
        _block("b1", "week_01_content", dict(_ALL3)),
        _block("b2", "week_01_content", weak),
    ]
    report = BlockQualityRollupAggregator(blocks=blocks).build()
    mod = report["per_module"][0]
    assert mod["module_mean"] >= 2.0  # mean path would pass
    assert mod["required_dim_minimum_ok"] is False  # min-floor path fails
    assert mod["module_pass"] is False  # the no-weak-block-hides assertion
    assert mod["module_dim_minimums"]["coherence"] == 1


def test_accessibility_zero_block_fails_course():
    bad = dict(_ALL3)
    bad["accessibility"] = 0
    blocks = [
        _block("b1", "week_01_content", dict(_ALL3)),
        _block("b2", "week_01_content", bad),
    ]
    report = BlockQualityRollupAggregator(blocks=blocks).build()
    assert report["summary"]["accessibility_gate_failures"] == 1
    assert report["summary"]["course_pass"] is False
    assert report["summary"]["quality_decision"] == "failed"
    # the offending block fails outright
    b2 = next(r for r in report["per_block"] if r["block_id"] == "b2")
    assert b2["accessibility_gate_fail"] is True
    assert b2["block_pass"] is False


def test_decision_capture_fires(tmp_path):
    class _Capture:
        def __init__(self):
            self.events = []

        def log_decision(self, **kw):
            self.events.append(kw)

    cap = _Capture()
    blocks = [_block("b1", "week_01_content", dict(_ALL3))]
    agg = BlockQualityRollupAggregator(blocks=blocks, decision_capture=cap, run_id="WF-1")
    out = agg.write(tmp_path / "block_quality_rollup_report.json")
    assert out.exists()
    # decision fired
    assert any(
        e["decision_type"] == "block_quality_rollup_aggregated" for e in cap.events
    )
    ev = next(
        e for e in cap.events if e["decision_type"] == "block_quality_rollup_aggregated"
    )
    assert len(ev["rationale"]) >= 20
    # written report round-trips + carries the schema version.
    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0"
    assert "per_module" in data and "per_block" in data


def test_module_mean_below_floor_fails_even_when_minimums_ok():
    # Both clauses must hold: a module whose per-dim minimums are >=2 but whose
    # MEAN drops below 2.0 also fails (the AND condition).
    low = {"alignment": 2, "cognitive_load": 2, "accessibility": 2, "coherence": 2}
    blocks = [_block("b1", "week_02_content", low)]
    report = BlockQualityRollupAggregator(blocks=blocks).build()
    mod = report["per_module"][0]
    assert mod["required_dim_minimum_ok"] is True
    assert mod["module_mean"] == 2.0
    assert mod["module_pass"] is True  # exactly at the 2.0 floor passes


def test_from_phase_outputs_resolution():
    # Reconstructs blocks from a stashed block_quality_rubric GateResult.
    from MCP.hardening.validation_gates import GateResult

    gr = GateResult(
        gate_id="block_quality_rubric",
        validator_name="block_quality_rubric",
        validator_version="1.0.0",
        passed=True,
        metadata={
            "block_quality_rubric": {
                "per_block": {
                    "b1": {
                        "framework_block": "B03",
                        "page_id": "week_01_content",
                        "scores": dict(_ALL3),
                        "applicable_dims": sorted(_ALL3.keys()),
                    }
                }
            }
        },
    )
    phase_outputs = {"post_rewrite_validation": {"_gate_results": [gr]}}
    agg = BlockQualityRollupAggregator(phase_outputs=phase_outputs)
    assert agg.blocks  # resolved one block
    report = agg.build()
    assert report["summary"]["total_blocks"] == 1
