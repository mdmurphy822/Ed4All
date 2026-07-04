"""Unit tests for the W6 (Round-2) planner-pedagogy passes.

Covers the four opt-in passes added to ``lib.generation.block_planner``:

- ``_apply_interleave`` (W6.2, ``ED4ALL_PLANNER_INTERLEAVE``): default-off
  identity; on → partition-invariant reorder mixing practice blocks ACROSS COs.
- ``_apply_far_transfer_floor`` (W6.3, ``ED4ALL_PLANNER_FAR_TRANSFER``):
  default-off identity; on → >= 1 novel-context application block per TO,
  grounded to a real CO id.
- ``_apply_dual_coding_floor`` (W6.5, ``ED4ALL_PLANNER_DUAL_CODING``):
  default-off identity; on → a co-located visual paired with each verbal-only
  CO, grounded to a real CO id.
- ``build_cross_week_retrieval_blocks`` (W6.1,
  ``ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL``): default-off empty; on → cumulative
  retrieval blocks over prior-week COs (only real prior ids).

Plus the cross-cutting contracts: flags-off byte-identical (the whole
``plan_week_blocks`` output), partition invariance for the reorder, no invented
CO ids.
"""
from __future__ import annotations

import copy

import pytest

from lib.generation.block_planner import (
    _apply_dual_coding_floor,
    _apply_far_transfer_floor,
    _apply_integration_floor,
    _apply_interleave,
    _apply_w6_within_week_passes,
    _resolve_block_types,
    build_cross_week_retrieval_blocks,
    plan_week_blocks,
)


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for env in (
        "ED4ALL_PLANNER_INTERLEAVE",
        "ED4ALL_PLANNER_FAR_TRANSFER",
        "ED4ALL_PLANNER_INTEGRATION",
        "ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP",
        "ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK",
        "ED4ALL_PLANNER_DUAL_CODING",
        "ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL",
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        "ED4ALL_NEW_BLOCK_TYPES",
        "ED4ALL_PLANNER_BLOOM_CLIMB",
        "ED4ALL_PLANNER_LIFECYCLE",
        "ED4ALL_PLANNER_SPACING",
        "ED4ALL_PLANNER_BLOOM_CEILING",
        "ED4ALL_PLANNER_FADING",
        "ED4ALL_TRIANGLE_FLOOR",
        "ED4ALL_RETRIEVAL_INTERLEAVE",
        "ED4ALL_WORKED_EXAMPLE_FLOOR",
        "ED4ALL_BLOOM_SPREAD_FLOOR",
        "ED4ALL_BLOCK_ANATOMY",
    ):
        monkeypatch.delenv(env, raising=False)
    yield


@pytest.fixture()
def block_types():
    return _resolve_block_types()


def _cos():
    return [
        {"id": "CO-01", "statement": "Add integers", "bloom_level": "apply"},
        {"id": "CO-02", "statement": "Compare integers", "bloom_level": "understand"},
    ]


# --------------------------------------------------------------------------- #
# W6.2 — interleave.
# --------------------------------------------------------------------------- #
def _blocked_practice_selection():
    """AABB blocked practice: two CO-01 problems then two CO-02 problems."""
    return [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-02"], "target_bloom": "apply"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-02"], "target_bloom": "apply"},
    ]


def test_interleave_noop_when_flag_off():
    sel = _blocked_practice_selection()
    before = copy.deepcopy(sel)
    out = _apply_interleave(selected=sel)
    assert out == before


def test_interleave_mixes_across_cos_when_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_PLANNER_INTERLEAVE", "1")
    sel = _blocked_practice_selection()
    out = _apply_interleave(selected=sel)
    # Non-practice block (the concept) stays at index 0.
    assert out[0]["block_type"] == "concept"
    # The four practice slots (1..4) now alternate CO-01 / CO-02 (ABAB).
    practice_cos = [out[i]["target_co_ids"][0] for i in (1, 2, 3, 4)]
    assert practice_cos == ["CO-01", "CO-02", "CO-01", "CO-02"]


def test_interleave_is_partition_invariant(monkeypatch):
    monkeypatch.setenv("ED4ALL_PLANNER_INTERLEAVE", "1")
    sel = _blocked_practice_selection()
    out = _apply_interleave(selected=sel)
    # Same multiset of blocks (no add/drop/re-type) — sort a stable projection.
    def _key(b):
        return (b["block_type"], tuple(b["target_co_ids"]), b["target_bloom"])
    assert sorted(map(_key, out)) == sorted(map(_key, sel))
    assert len(out) == len(sel)


def test_interleave_noop_single_co(monkeypatch):
    monkeypatch.setenv("ED4ALL_PLANNER_INTERLEAVE", "1")
    sel = [
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_interleave(selected=sel)
    assert out == before


# --------------------------------------------------------------------------- #
# W6.3 — far-transfer floor.
# --------------------------------------------------------------------------- #
def _verbal_selection():
    return [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "explanation", "page_type": "content",
         "target_co_ids": ["CO-02"], "target_bloom": "understand"},
    ]


def test_far_transfer_noop_when_flag_off(block_types):
    sel = _verbal_selection()
    before = copy.deepcopy(sel)
    out = _apply_far_transfer_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    assert out == before


def test_far_transfer_injects_novel_context_block(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_FAR_TRANSFER", "1")
    sel = _verbal_selection()
    out = _apply_far_transfer_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    ft = [b for b in out if b.get("far_transfer")]
    assert len(ft) == 1
    assert ft[0]["block_type"] in ("scenario", "problem")
    # Grounded to a REAL referenced CO id (anti-fabrication).
    assert ft[0]["target_co_ids"] == ["CO-01"]


def test_far_transfer_idempotent_when_already_present(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_FAR_TRANSFER", "1")
    sel = _verbal_selection() + [
        {"block_type": "scenario", "page_type": "application",
         "target_co_ids": ["CO-01"], "target_bloom": "apply",
         "far_transfer": True},
    ]
    before = copy.deepcopy(sel)
    out = _apply_far_transfer_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    assert out == before


def test_far_transfer_no_co_is_noop(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_FAR_TRANSFER", "1")
    sel = [{"block_type": "concept", "page_type": "content",
            "target_co_ids": [], "target_bloom": "understand"}]
    out = _apply_far_transfer_floor(
        selected=sel, chapter_objectives=[], block_types=block_types)
    assert out == sel  # never invents a CO id


# --------------------------------------------------------------------------- #
# Integration floor — weak-membership ("oddball") CO → cross-CO integrated
# practice (ED4ALL_PLANNER_INTEGRATION).
# --------------------------------------------------------------------------- #
from lib.generation.block_planner import (  # noqa: E402
    _resolve_integration_max_overlap,
    _resolve_integration_max_per_week,
)

# TO whose theme an oddball CO only weakly matches. 8 content tokens: apply,
# properties, of, real, numbers, and, algebraic, expressions.
_INTEG_TO = {
    "id": "TO-02",
    "statement": "Apply properties of real numbers and algebraic expressions",
}


def _integ_cos():
    """One strong-membership anchor + one weak-membership oddball CO."""
    return [
        # High token-overlap with the TO (apply / properties / of / real /
        # numbers / and) → the anchor.
        {"id": "CO-11",
         "statement": "Apply the commutative and associative properties of "
                      "real numbers",
         "bloom_level": "apply"},
        # Moderate overlap (algebraic / expressions / of), still above the floor.
        {"id": "CO-12",
         "statement": "Simplify algebraic expressions using order of operations",
         "bloom_level": "apply"},
        # Near-zero overlap (only "and") → the oddball.
        {"id": "CO-14",
         "statement": "Explain the relationship between Fahrenheit and Celsius "
                      "temperature scales",
         "bloom_level": "understand"},
    ]


def _integ_taught_selection():
    """Each CO taught by a real block (non-empty target_co_ids)."""
    return [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-11"], "target_bloom": "apply"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-12"], "target_bloom": "apply"},
        {"block_type": "explanation", "page_type": "content",
         "target_co_ids": ["CO-14"], "target_bloom": "understand"},
    ]


def test_integration_noop_when_flag_off(block_types):
    sel = _integ_taught_selection()
    before = copy.deepcopy(sel)
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    assert out == before


def test_integration_injects_for_weak_membership_co(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    sel = _integ_taught_selection()
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    injected = [b for b in out if b.get("integration")]
    assert len(injected) == 1
    blk = injected[0]
    assert blk["block_type"] in ("problem", "scenario")
    assert blk["page_type"] == "application"
    assert blk["target_bloom"] == "apply"
    # REAL ids only: [oddball, anchor]. Anchor = highest-overlap CO (CO-11, not
    # CO-12); oddball = the weak-membership CO-14.
    assert blk["target_co_ids"] == ["CO-14", "CO-11"]
    # content_focus names BOTH CO statements + directs the rewrite tier.
    focus = blk["content_focus"]
    assert "CO-14" in focus and "CO-11" in focus
    assert "Fahrenheit" in focus          # oddball statement fragment
    assert "commutative" in focus         # anchor statement fragment
    assert "cited source chunks" in focus


def test_integration_anchor_is_highest_overlap_co(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    sel = _integ_taught_selection()
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    injected = [b for b in out if b.get("integration")][0]
    # CO-11 overlaps the TO more than CO-12, so it wins the anchor slot.
    assert injected["target_co_ids"][1] == "CO-11"


def test_integration_skips_already_co_targeted_oddball(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    sel = _integ_taught_selection() + [
        # A block already co-targets the oddball together with the anchor.
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-14", "CO-11"], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    assert out == before  # already integrated → no injection


def test_integration_noop_when_no_weak_co(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    # Only the two strong-membership COs are taught (no oddball present).
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-11"], "target_bloom": "apply"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-12"], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos()[:2], block_types=block_types)
    assert out == before


def test_integration_noop_single_co_week(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    sel = [
        {"block_type": "explanation", "page_type": "content",
         "target_co_ids": ["CO-14"], "target_bloom": "understand"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    assert out == before  # < 2 taught COs → no anchor distinct from oddball


def test_integration_static_fallback_empty_co_is_noop(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    # Static fallback plan: real TO + COs, but blocks carry empty target_co_ids.
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": [], "target_bloom": "understand"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": [], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    assert out == before  # no taught CO → natural no-op


# Three distinct weak-membership overlaps (all < 0.10, each shares only "and"
# with the TO) so weakest-first ordering is observable, plus a strong anchor.
def _integ_multi_weak_cos():
    return [
        {"id": "CO-A",
         "statement": "Apply the commutative and associative properties of "
                      "real numbers",
         "bloom_level": "apply"},                          # strong anchor
        {"id": "CO-W3",                                    # overlap ~0.083
         "statement": "Describe historical timelines and events",
         "bloom_level": "understand"},
        {"id": "CO-W1",                                    # overlap ~0.056 (weakest)
         "statement": "Explain the relationship between Fahrenheit and Celsius "
                      "temperature scales here now",
         "bloom_level": "understand"},
        {"id": "CO-W2",                                    # overlap ~0.067
         "statement": "Convert distance units across measurement systems and "
                      "contexts",
         "bloom_level": "apply"},
    ]


def _integ_multi_weak_selection():
    return [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": [cid], "target_bloom": "apply"}
        for cid in ("CO-A", "CO-W3", "CO-W1", "CO-W2")
    ]


def test_integration_max_per_week_cap_weakest_first(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "2")
    out = _apply_integration_floor(
        selected=_integ_multi_weak_selection(),
        terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_multi_weak_cos(), block_types=block_types)
    injected = [b for b in out if b.get("integration")]
    assert len(injected) == 2  # capped at 2 despite 3 weak COs
    # Weakest first: CO-W1 (0.056) then CO-W2 (0.067); CO-W3 (0.083) dropped.
    oddballs = [b["target_co_ids"][0] for b in injected]
    assert oddballs == ["CO-W1", "CO-W2"]
    # Anchor is always the strong CO-A.
    assert all(b["target_co_ids"][1] == "CO-A" for b in injected)


def test_integration_max_per_week_cap_one(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "1")
    out = _apply_integration_floor(
        selected=_integ_multi_weak_selection(),
        terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_multi_weak_cos(), block_types=block_types)
    injected = [b for b in out if b.get("integration")]
    assert len(injected) == 1
    assert injected[0]["target_co_ids"][0] == "CO-W1"  # the single weakest


def test_integration_max_overlap_satellite_widens_oddball_set(
    monkeypatch, block_types
):
    # Raise the floor so CO-12 (moderate ~0.25 overlap) also counts as weak.
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP", "0.30")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "5")
    out = _apply_integration_floor(
        selected=_integ_taught_selection(), terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    oddballs = {b["target_co_ids"][0] for b in out if b.get("integration")}
    assert oddballs == {"CO-14", "CO-12"}  # both now below the 0.30 floor


def test_integration_satellite_env_parse_with_fallback(monkeypatch):
    # Garbage → defaults.
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP", "not-a-float")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "bogus")
    assert _resolve_integration_max_overlap() == 0.10
    assert _resolve_integration_max_per_week() == 2
    # Out-of-range → defaults.
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP", "1.5")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "0")
    assert _resolve_integration_max_overlap() == 0.10
    assert _resolve_integration_max_per_week() == 2
    # Valid → parsed.
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_OVERLAP", "0.25")
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION_MAX_PER_WEEK", "3")
    assert _resolve_integration_max_overlap() == 0.25
    assert _resolve_integration_max_per_week() == 3
    # Explicit arg wins over env.
    assert _resolve_integration_max_overlap(0.5) == 0.5
    assert _resolve_integration_max_per_week(7) == 7


def test_integration_is_deterministic(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    kwargs = dict(
        terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_multi_weak_cos(), block_types=block_types)
    out1 = _apply_integration_floor(
        selected=_integ_multi_weak_selection(), **kwargs)
    out2 = _apply_integration_floor(
        selected=_integ_multi_weak_selection(), **kwargs)
    assert out1 == out2


# --------------------------------------------------------------------------- #
# W6.5 — dual-coding floor.
# --------------------------------------------------------------------------- #
def test_dual_coding_noop_when_flag_off(block_types):
    sel = _verbal_selection()
    before = copy.deepcopy(sel)
    out = _apply_dual_coding_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    assert out == before


def test_dual_coding_pairs_visual_with_verbal_only_co(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_DUAL_CODING", "1")
    sel = _verbal_selection()
    out = _apply_dual_coding_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    visuals = [b for b in out if b["block_type"] in ("diagram", "multimedia", "table")]
    # One visual per verbal-only CO (CO-01, CO-02).
    covered = {b["target_co_ids"][0] for b in visuals}
    assert covered == {"CO-01", "CO-02"}
    # Co-located: the visual for CO-01 sits immediately after the CO-01 concept.
    types = [b["block_type"] for b in out]
    idx_concept = next(i for i, b in enumerate(out)
                       if b["block_type"] == "concept")
    assert out[idx_concept + 1]["block_type"] in ("diagram", "multimedia", "table")


def test_dual_coding_skips_co_with_visual(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_DUAL_CODING", "1")
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "diagram", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    before = copy.deepcopy(sel)
    out = _apply_dual_coding_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types)
    assert out == before  # CO-01 already has a visual → no pairing


# --------------------------------------------------------------------------- #
# W6.1 — cross-week cumulative retrieval helper.
# --------------------------------------------------------------------------- #
def test_cross_week_empty_when_flag_off():
    out = build_cross_week_retrieval_blocks(
        prior_week_co_ids=["CO-01", "CO-02"], week_index=3)
    assert out == []


def test_cross_week_targets_only_real_prior_cos(monkeypatch, block_types):
    monkeypatch.setenv("ED4ALL_PLANNER_CROSS_WEEK_RETRIEVAL", "1")
    out = build_cross_week_retrieval_blocks(
        prior_week_co_ids=["CO-01", "CO-02", "CO-01"],
        week_index=4, block_types=block_types)
    assert len(out) == 2  # deduped
    assert {b["target_co_ids"][0] for b in out} == {"CO-01", "CO-02"}
    assert all(b.get("cumulative_retrieval") for b in out)
    assert all(b["page_type"] == "self_check" for b in out)


def test_cross_week_enabled_override(block_types):
    out = build_cross_week_retrieval_blocks(
        prior_week_co_ids=["CO-01"], week_index=2,
        block_types=block_types, enabled=True)
    assert len(out) == 1
    assert out[0]["target_co_ids"] == ["CO-01"]


def test_cross_week_empty_prior_is_empty(block_types):
    out = build_cross_week_retrieval_blocks(
        prior_week_co_ids=[], week_index=5,
        block_types=block_types, enabled=True)
    assert out == []


def test_cross_week_max_targets_caps(block_types):
    out = build_cross_week_retrieval_blocks(
        prior_week_co_ids=[f"CO-{i:02d}" for i in range(1, 20)],
        week_index=9, block_types=block_types, enabled=True, max_targets=3)
    assert len(out) == 3


# --------------------------------------------------------------------------- #
# Cross-cutting: flags-off byte-identical plan_week_blocks.
# --------------------------------------------------------------------------- #
def test_plan_week_blocks_byte_identical_when_all_w6_flags_off():
    to = {"id": "TO-01", "statement": "Work with integers"}
    cos = _cos()
    plan_off = plan_week_blocks(
        terminal_objective=to, chapter_objectives=cos, provider=None)
    # A second call with the flags still off must match (deterministic + no W6).
    plan_off2 = plan_week_blocks(
        terminal_objective=to, chapter_objectives=cos, provider=None)
    assert plan_off.selected == plan_off2.selected
    assert plan_off.page_plan == plan_off2.page_plan
    # No W6 stamp fields leak onto an all-off run.
    for b in plan_off.selected:
        assert "far_transfer" not in b
        assert "integration" not in b
        assert "cumulative_retrieval" not in b


def test_w6_within_week_passes_identity_when_off(block_types):
    sel = _blocked_practice_selection()
    before = copy.deepcopy(sel)
    signals = {}
    out = _apply_w6_within_week_passes(
        selected=sel, chapter_objectives=_cos(),
        block_types=block_types, signals=signals)
    assert out == before
    assert signals == {
        "far_transfer_injected": 0,
        "integration_injected": 0,
        "dual_coding_injected": 0,
        "interleave_moves": 0,
    }


def test_integration_non_practice_co_targeting_does_not_suppress(
    monkeypatch, block_types
):
    """Regression: the first live run injected 0 blocks across 11 weeks because
    roll-up blocks (objective / summary) co-target several COs as a matter of
    course and the skip guard counted them. Only PRACTICE-type co-targeting
    (_INTERLEAVE_PRACTICE_TYPES) marks a CO as already integrated."""
    monkeypatch.setenv("ED4ALL_PLANNER_INTEGRATION", "1")
    sel = _integ_taught_selection() + [
        # An OBJECTIVE roll-up block listing the oddball with other COs —
        # NOT integrated practice; must not suppress the injection.
        {"block_type": "objective", "page_type": "overview",
         "target_co_ids": ["CO-14", "CO-11", "CO-19"],
         "target_bloom": "understand"},
    ]
    out = _apply_integration_floor(
        selected=sel, terminal_objective=_INTEG_TO,
        chapter_objectives=_integ_cos(), block_types=block_types)
    injected = [b for b in out if b.get("integration")]
    assert injected, "objective-block co-targeting must not suppress injection"
    assert any("CO-14" in (b.get("target_co_ids") or []) for b in injected)
