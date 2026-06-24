"""FR-PLAN-01 / FR-INT-01 — interaction_type + fade_state survive rehydration.

The outline-JSONL → rewrite round-trip (``_block_to_snake_case_entry`` →
``Block(**entry)``) must preserve both the planner-selected ``interaction_type``
and the ``fade_state``, mirroring how ``anchored_rubric`` is threaded. When both
are None (the flag-off / legacy path) the serializer must OMIT them so the
JSONL stays byte-stable.
"""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from MCP.tools.pipeline_tools import _block_to_snake_case_entry


def test_serializer_emits_both_when_set():
    b = Block(
        block_id="p#assessment_item_x_0", block_type="assessment_item",
        page_id="p", sequence=0, content="c",
        interaction_type="numeric", fade_state="completion",
    )
    entry = _block_to_snake_case_entry(b)
    assert entry["interaction_type"] == "numeric"
    assert entry["fade_state"] == "completion"


def test_serializer_omits_both_when_none():
    b = Block(
        block_id="p#concept_x_0", block_type="concept", page_id="p",
        sequence=0, content="c",
    )
    entry = _block_to_snake_case_entry(b)
    assert "interaction_type" not in entry
    assert "fade_state" not in entry


def test_round_trip_preserves_both():
    b = Block(
        block_id="p#problem_x_0", block_type="problem", page_id="p",
        sequence=0, content="c",
        interaction_type="drag_drop", fade_state="completion",
    )
    entry = _block_to_snake_case_entry(b)
    restored = Block(**entry)
    assert restored.interaction_type == "drag_drop"
    assert restored.fade_state == "completion"
