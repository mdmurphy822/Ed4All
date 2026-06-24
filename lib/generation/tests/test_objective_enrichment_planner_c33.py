"""C3-3 — block_planner ``_enrich_objective_blocks`` pass.

The pass stamps ``self_rating_prompt`` + ``objective_assessment_thread`` on
objective blocks, gated by the EXISTING ``ED4ALL_BLOCK_ANATOMY`` flag. Strict
identity no-op (byte-stable) when off; deterministic when on; anti-fabrication
(assessment refs come from real selected blocks).
"""
from __future__ import annotations

import copy

import pytest

from lib.generation.block_planner import _enrich_objective_blocks


@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_ANATOMY", raising=False)
    yield


def _selected():
    return [
        {"block_type": "objective", "page_type": "overview",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "concept", "page_type": "content",
         "target_bloom": "understand", "target_co_ids": ["CO-01"]},
        {"block_type": "assessment_item", "page_type": "self_check",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_bloom": "remember", "target_co_ids": ["CO-01"]},
    ]


def test_noop_when_flag_off():
    sel = _selected()
    before = copy.deepcopy(sel)
    out = _enrich_objective_blocks(selected=sel)
    assert sel == before  # input not mutated
    assert all(
        b.get("self_rating_prompt") is None
        and b.get("objective_assessment_thread") is None
        for b in out
    )


def test_stamps_objective_when_flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    out = _enrich_objective_blocks(selected=_selected())
    obj = next(b for b in out if b["block_type"] == "objective")
    assert isinstance(obj["self_rating_prompt"], str) and obj["self_rating_prompt"]
    thread = obj["objective_assessment_thread"]
    assert isinstance(thread, dict)
    # Anti-fabrication: refs are real assessing block types sharing the CO.
    assert "assessment_item" in thread["assessment_refs"]
    assert "self_check_question" in thread["assessment_refs"]
    # Non-assessing blocks are NOT refs.
    assert "concept" not in thread["assessment_refs"]


def test_non_objective_blocks_untouched(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    out = _enrich_objective_blocks(selected=_selected())
    for b in out:
        if b["block_type"] != "objective":
            assert b.get("self_rating_prompt") is None
            assert b.get("objective_assessment_thread") is None


def test_deterministic(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    a = _enrich_objective_blocks(selected=_selected())
    b = _enrich_objective_blocks(selected=_selected())
    obj_a = next(x for x in a if x["block_type"] == "objective")
    obj_b = next(x for x in b if x["block_type"] == "objective")
    assert obj_a["objective_assessment_thread"] == obj_b["objective_assessment_thread"]
    assert obj_a["self_rating_prompt"] == obj_b["self_rating_prompt"]


def test_existing_field_preserved(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_ANATOMY", "1")
    sel = _selected()
    sel[0]["self_rating_prompt"] = "custom prompt"
    out = _enrich_objective_blocks(selected=sel)
    obj = next(b for b in out if b["block_type"] == "objective")
    assert obj["self_rating_prompt"] == "custom prompt"
