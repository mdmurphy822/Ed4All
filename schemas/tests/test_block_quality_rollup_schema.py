"""IB6.6 — block_quality_rollup_report.json validates against its schema."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from lib.aggregators.block_quality_rollup import BlockQualityRollupAggregator

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "aggregators"
    / "block_quality_rollup.schema.json"
)


def _schema():
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _block(block_id, module, scores):
    return {
        "block_id": block_id,
        "page_id": module,
        "framework_block": "B03",
        "quality_rubric": [
            {"dimension": dim, "score": score, "applicable": True}
            for dim, score in scores.items()
        ],
    }


def test_schema_is_valid_draft_2020():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_rollup_report_validates():
    blocks = [
        _block(
            "b1",
            "week_01_content",
            {
                "alignment": 3,
                "cognitive_load": 3,
                "accessibility": 3,
                "coherence": 2,
            },
        ),
        _block(
            "b2",
            "week_01_content",
            {
                "alignment": 0,
                "cognitive_load": 1,
                "accessibility": 0,
                "coherence": 1,
            },
        ),
    ]
    report = BlockQualityRollupAggregator(
        blocks=blocks, course_code="TEST_101", run_id="WF-1"
    ).build()
    jsonschema.validate(report, _schema())


def test_empty_rollup_validates():
    report = BlockQualityRollupAggregator(blocks=[]).build()
    jsonschema.validate(report, _schema())
