"""ITEM2 (region-bbox) — Region.page_bboxes carry + stamp tests.

Covers the compute helper's union / sort / skip semantics, the bare-constructor
default, the graph-exit stamp (both reading-order-fix states), the
kind/payload-rewrite carry-through contract, and the byte-identity of
``cascade._build_region_provenance`` (the field is inert provenance — no
emit/cache surface reads it).

Cases 7-8 (Stage-5e merge/split geometry recompute) live in
``qwen_specialists/tests/test_block_resegment.py`` — see the ITEM2 Phase-2 spec.
"""
from __future__ import annotations

import dataclasses

import pytest

from semantik_structure.cascade import _build_region_provenance
from semantik_structure.structure_graph import (
    Region,
    build_structure_graph,
    compute_region_page_bboxes,
    stamp_region_page_bboxes,
)
from semantik_structure.tests.test_structure_reading_order import _build_interleaved_graph
from semantik_structure.types import FeatureBlock, RawBlock


def _fb(page: int, bbox) -> FeatureBlock:
    """Minimal FeatureBlock carrying a RawBlock with the given page + bbox."""
    raw = RawBlock(
        text="x",
        page=page,
        bbox=bbox,
        page_width=200.0,
        page_height=300.0,
        font_size=11.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
        is_image=False,
    )


# ---------------------------------------------------------------------------
# Case 1 — single-page union
# ---------------------------------------------------------------------------


def test_compute_single_page_union():
    fbs = [
        _fb(1, (10.0, 20.0, 30.0, 40.0)),
        _fb(1, (5.0, 25.0, 35.0, 38.0)),
        _fb(1, (12.0, 15.0, 28.0, 50.0)),
    ]
    result = compute_region_page_bboxes((0, 1, 2), fbs)
    # Exact min/min/max/max union over the three bboxes on page 1.
    assert result == ((1, (5.0, 15.0, 35.0, 50.0)),)


# ---------------------------------------------------------------------------
# Case 2 — multi-page, entries sorted ascending by page, per-page independent
# ---------------------------------------------------------------------------


def test_compute_multi_page_sorted_entries():
    fbs = [
        _fb(2, (0.0, 0.0, 10.0, 10.0)),
        _fb(1, (1.0, 1.0, 5.0, 5.0)),
        _fb(2, (3.0, 4.0, 20.0, 6.0)),
        _fb(1, (2.0, 0.0, 4.0, 9.0)),
    ]
    result = compute_region_page_bboxes((0, 1, 2, 3), fbs)
    # Sorted (1, ...), (2, ...); each page's union computed independently.
    assert result == (
        (1, (1.0, 0.0, 5.0, 9.0)),
        (2, (0.0, 0.0, 20.0, 10.0)),
    )


# ---------------------------------------------------------------------------
# Case 3 — skips unusable geometry; all-unusable region -> () not None
# ---------------------------------------------------------------------------


def test_compute_skips_unusable_geometry():
    good = _fb(1, (1.0, 2.0, 3.0, 4.0))
    raw_none = _fb(1, (0.0, 0.0, 1.0, 1.0))
    raw_none.raw = None  # type: ignore[assignment]
    bbox_none = _fb(1, (0.0, 0.0, 1.0, 1.0))
    bbox_none.raw.bbox = None  # type: ignore[assignment]
    short_bbox = _fb(1, (0.0, 0.0, 1.0, 1.0))
    short_bbox.raw.bbox = (1.0, 2.0, 3.0)  # type: ignore[assignment]  # 3-element

    fbs = [good, raw_none, bbox_none, short_bbox]

    # Out-of-range index 99, raw=None (1), bbox=None (2), short bbox (3) all
    # skipped; only the good FB (0) survives.
    result = compute_region_page_bboxes((0, 1, 2, 3, 99), fbs)
    assert result == ((1, (1.0, 2.0, 3.0, 4.0)),)

    # A region whose members are ALL unusable -> () (computed, no geometry),
    # never None.
    all_unusable = compute_region_page_bboxes((1, 2, 3, 99), fbs)
    assert all_unusable == ()


# ---------------------------------------------------------------------------
# Case 4 — bare constructor defaults to None
# ---------------------------------------------------------------------------


def test_bare_constructor_defaults_none():
    r = Region(kind="paragraph", feature_block_indices=(0,))
    assert r.page_bboxes is None


# ---------------------------------------------------------------------------
# Case 5 — build_structure_graph stamps every region (both flag states)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_value", ["0", "1"])
def test_build_structure_graph_stamps_every_region(monkeypatch, flag_value):
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", flag_value)
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)

    assert regions, "no regions formed"
    for r in regions:
        assert r.page_bboxes is not None, f"region {r.kind} was not stamped"
        assert r.page_bboxes == compute_region_page_bboxes(
            r.feature_block_indices, fbs
        )


# ---------------------------------------------------------------------------
# Case 6 — kind/payload rewrite via dataclasses.replace carries geometry
# ---------------------------------------------------------------------------


def test_replace_kind_carries_geometry():
    fbs = [_fb(1, (1.0, 2.0, 3.0, 4.0))]
    [stamped] = stamp_region_page_bboxes(
        [Region(kind="paragraph", feature_block_indices=(0,))], fbs
    )
    assert stamped.page_bboxes == ((1, (1.0, 2.0, 3.0, 4.0)),)

    # A kind-only rewrite (reviewer / deterministic_structure precedent) keeps
    # page_bboxes — the carry-through contract that ITEM2 relies on.
    retyped = dataclasses.replace(stamped, kind="list")
    assert retyped.page_bboxes == stamped.page_bboxes


# ---------------------------------------------------------------------------
# Case 9 — _build_region_provenance byte-identical with/without page_bboxes
# ---------------------------------------------------------------------------


def test_region_provenance_byte_identical():
    state, fbs, cands, decs = _build_interleaved_graph()
    stamped = build_structure_graph(state, fbs, cands, decs)
    # Every stamped region carries a non-None page_bboxes.
    assert all(r.page_bboxes is not None for r in stamped)

    forced_none = [dataclasses.replace(r, page_bboxes=None) for r in stamped]

    region_order = list(range(len(stamped)))
    prov_stamped = _build_region_provenance(region_order, stamped, fbs, {})
    prov_none = _build_region_provenance(region_order, forced_none, fbs, {})

    # No provenance key leaks the new field -> the two lists compare equal.
    assert prov_stamped == prov_none
