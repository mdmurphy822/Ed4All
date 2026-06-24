"""Unit tests for FR-PLAN-01 (interaction-type resolver) + FR-INT-01 (fading).

Covers:

- the deterministic ``interaction_type`` resolver (off → identity no-op; on →
  picks a Bloom-affine type from the catalog ``default_activity_types``; non-
  interaction-bearing types stay None; deterministic);
- the B08 fading-sequence pass (off → identity; on → injects a B08 completion
  block after a worked_example, stamps fade_state, no dup);
- ``_to_page_plan`` arity (byte-stable 3-tuple when off; 4/5-tuple when set);
- the ``_apply_ib7_passes`` integration is a strict identity no-op when all
  flags are off.
"""
from __future__ import annotations

import copy
import os

import pytest

from lib.generation.block_planner import (
    _apply_fading_sequence,
    _apply_ib7_passes,
    _resolve_block_types,
    _resolve_interaction_types,
    _resolve_one_interaction_type,
    _to_page_plan,
    load_block_catalog,
)


@pytest.fixture()
def catalog_by_type():
    return {e["block_type"]: e for e in load_block_catalog() if e.get("block_type")}


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for env in (
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        "ED4ALL_PLANNER_FADING",
        "ED4ALL_PLANNER_BLOOM_CLIMB",
        "ED4ALL_PLANNER_LIFECYCLE",
        "ED4ALL_PLANNER_SPACING",
        "ED4ALL_PLANNER_BLOOM_CEILING",
    ):
        monkeypatch.delenv(env, raising=False)
    yield


# --------------------------------------------------------------------------- #
# FR-PLAN-01 — interaction-type resolver.
# --------------------------------------------------------------------------- #
def test_interaction_resolver_noop_when_flag_off(catalog_by_type):
    sel = [{"block_type": "assessment_item", "page_type": "self_check",
            "target_bloom": "apply", "target_co_ids": ["CO-01"]}]
    before = copy.deepcopy(sel)
    out = _resolve_interaction_types(selected=sel, catalog_by_type=catalog_by_type)
    assert sel == before
    assert all(b.get("interaction_type") is None for b in out)


def test_interaction_resolver_assigns_when_flag_on(monkeypatch, catalog_by_type):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    sel = [
        {"block_type": "assessment_item", "page_type": "self_check",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_bloom": "remember", "target_co_ids": []},
        {"block_type": "concept", "page_type": "content",
         "target_bloom": "understand", "target_co_ids": []},
    ]
    out = _resolve_interaction_types(selected=sel, catalog_by_type=catalog_by_type)
    by_type = {b["block_type"]: b.get("interaction_type") for b in out}
    # interaction-bearing types get a type drawn from their catalog list.
    assert by_type["assessment_item"] in catalog_by_type["assessment_item"]["default_activity_types"]
    assert by_type["self_check_question"] in catalog_by_type["self_check_question"]["default_activity_types"]
    # a non-interaction-bearing type stays None.
    assert by_type["concept"] is None


def test_interaction_resolver_is_deterministic(monkeypatch, catalog_by_type):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    blk = {"block_type": "assessment_item", "page_type": "self_check",
           "target_bloom": "apply", "target_co_ids": ["CO-01"]}
    a = _resolve_interaction_types(selected=[dict(blk)], catalog_by_type=catalog_by_type)
    b = _resolve_interaction_types(selected=[dict(blk)], catalog_by_type=catalog_by_type)
    assert a[0]["interaction_type"] == b[0]["interaction_type"]


def test_interaction_resolver_bloom_affinity(catalog_by_type):
    # higher Bloom on the assessment_item prefers a constructed-response type;
    # lower Bloom prefers a recognition type. Both must be from the catalog list.
    hi = _resolve_one_interaction_type(
        block_type="assessment_item", target_bloom="evaluate",
        catalog_by_type=catalog_by_type)
    lo = _resolve_one_interaction_type(
        block_type="assessment_item", target_bloom="understand",
        catalog_by_type=catalog_by_type)
    allowed = set(catalog_by_type["assessment_item"]["default_activity_types"])
    assert hi in allowed and lo in allowed
    # an understand-level item lands on a recognition type (mc / matching / ...)
    assert lo in {"multiple_choice", "matching", "fill_in_blank", "true_false"}


def test_interaction_resolver_none_for_non_interactive(catalog_by_type):
    assert _resolve_one_interaction_type(
        block_type="concept", target_bloom="understand",
        catalog_by_type=catalog_by_type) is None


# --------------------------------------------------------------------------- #
# FR-INT-01 — fading sequence.
# --------------------------------------------------------------------------- #
def _worked_then_concept():
    return [
        {"block_type": "worked_example", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "concept", "page_type": "content",
         "target_bloom": "understand", "target_co_ids": []},
    ]


def test_fading_noop_when_flag_off():
    sel = _worked_then_concept()
    before = copy.deepcopy(sel)
    out = _apply_fading_sequence(
        selected=sel, chapter_objectives=[{"id": "CO-01"}],
        block_types=_resolve_block_types())
    assert [b["block_type"] for b in out] == [b["block_type"] for b in before]
    assert all("fade_state" not in b for b in out)


def test_fading_injects_b08_when_flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_PLANNER_FADING", "1")
    out = _apply_fading_sequence(
        selected=_worked_then_concept(), chapter_objectives=[{"id": "CO-01"}],
        block_types=_resolve_block_types())
    types = [b["block_type"] for b in out]
    # a B08 completion block was injected after the worked_example.
    assert types[0] == "worked_example"
    assert types[1] in ("problem", "activity", "checklist")
    injected = out[1]
    assert injected["fade_state"] == "completion"
    # anti-fabrication: inherits the worked_example's CO ids.
    assert injected["target_co_ids"] == ["CO-01"]
    # the worked_example is stamped "worked".
    assert out[0]["fade_state"] == "worked"


def test_fading_no_dup_when_b08_already_present(monkeypatch):
    monkeypatch.setenv("ED4ALL_PLANNER_FADING", "1")
    sel = [
        {"block_type": "worked_example", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "problem", "page_type": "application",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
    ]
    out = _apply_fading_sequence(
        selected=sel, chapter_objectives=[{"id": "CO-01"}],
        block_types=_resolve_block_types())
    # no extra block injected; existing B08 gets a completion fade_state.
    assert [b["block_type"] for b in out] == ["worked_example", "problem"]
    assert out[1]["fade_state"] == "completion"


# --------------------------------------------------------------------------- #
# _to_page_plan arity / byte-stability.
# --------------------------------------------------------------------------- #
def test_page_plan_byte_stable_3_tuple_when_no_extra():
    plan = _to_page_plan([
        {"block_type": "concept", "page_type": "content",
         "target_bloom": "understand", "target_co_ids": ["CO-01"]},
    ])
    assert plan["content"] == [("concept", "understand", ["CO-01"])]


def test_page_plan_4_tuple_with_interaction():
    plan = _to_page_plan([
        {"block_type": "assessment_item", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": [], "interaction_type": "numeric"},
    ])
    assert plan["content"] == [("assessment_item", "apply", [], "numeric")]


def test_page_plan_5_tuple_with_fade():
    plan = _to_page_plan([
        {"block_type": "problem", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": [],
         "interaction_type": "numeric", "fade_state": "completion"},
    ])
    assert plan["content"] == [("problem", "apply", [], "numeric", "completion")]


# --------------------------------------------------------------------------- #
# _apply_ib7_passes integration — identity no-op when all flags off.
# --------------------------------------------------------------------------- #
def test_ib7_passes_identity_when_all_flags_off(catalog_by_type):
    sel = [
        {"block_type": "worked_example", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "assessment_item", "page_type": "self_check",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
    ]
    before = copy.deepcopy(sel)
    signals: dict = {}
    out = _apply_ib7_passes(
        selected=sel, chapter_objectives=[{"id": "CO-01"}],
        catalog_by_type=catalog_by_type,
        block_types=_resolve_block_types(), signals=signals)
    assert [b["block_type"] for b in out] == [b["block_type"] for b in before]
    assert all(b.get("interaction_type") is None for b in out)
    assert all("fade_state" not in b for b in out)
    assert signals["fading_blocks_injected"] == 0
    assert signals["interaction_types_assigned"] == 0


def test_ib7_passes_run_both_when_flags_on(monkeypatch, catalog_by_type):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    monkeypatch.setenv("ED4ALL_PLANNER_FADING", "1")
    sel = [
        {"block_type": "worked_example", "page_type": "content",
         "target_bloom": "apply", "target_co_ids": ["CO-01"]},
        {"block_type": "concept", "page_type": "content",
         "target_bloom": "understand", "target_co_ids": []},
    ]
    signals: dict = {}
    out = _apply_ib7_passes(
        selected=sel, chapter_objectives=[{"id": "CO-01"}],
        catalog_by_type=catalog_by_type,
        block_types=_resolve_block_types(), signals=signals)
    # a B08 completion block was injected, and the injected interaction-bearing
    # block (and any others) got an interaction_type.
    assert signals["fading_blocks_injected"] >= 1
    injected = [b for b in out if b["block_type"] in ("problem", "activity", "checklist")]
    assert injected and injected[0]["fade_state"] == "completion"
    assert injected[0].get("interaction_type") is not None
