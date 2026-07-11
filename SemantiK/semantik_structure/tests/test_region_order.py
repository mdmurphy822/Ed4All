"""Stage-5 geometric region-order derivation (ITEM5 prong 2, region_order.py).

Covers the SEMANTIK_REGION_ORDER three-mode owner: resolver semantics + legacy
alias, ``fb`` byte-identity to the min-FB sort, ``off`` byte-identity to the
segregated pass-order list, the ``geom`` per-page column derivation (page ->
column -> y0 -> x0 -> min_fb), float exclusion, full-width straddle pinning,
bbox-less re-insertion, the normalized-inversion divergence tripwire + page
ceiling fallback, the audit shape (shadow vs applied), permutation conservation,
and the build_structure_graph order_audit out-param plumbing.

All CPU-only, synthetic no-GPU CouncilState/FB harness (reused from
test_structure_reading_order.py) + hand-built Region/FeatureBlock objects. No
model, no PDF, no network.
"""
from __future__ import annotations

import pytest

from semantik_structure import region_order
from semantik_structure.region_order import (
    _inversion_distance,
    apply_region_order,
    resolve_region_order_mode,
)
from semantik_structure.structure_graph import (
    Region,
    build_structure_graph,
    compute_region_page_bboxes,
    resolve_reading_order_fix,
)
from semantik_structure.types import FeatureBlock, RawBlock

# Reuse the synthetic interleaved council harness.
from semantik_structure.tests.test_structure_reading_order import (
    _build_interleaved_graph,
)


_MODE = "SEMANTIK_REGION_ORDER"
_LEGACY = "SEMANTIK_READING_ORDER_FIX"


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    monkeypatch.delenv(_MODE, raising=False)
    monkeypatch.delenv(_LEGACY, raising=False)
    yield


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(page: int, bbox, pw: float = 1224.0) -> FeatureBlock:
    raw = RawBlock(
        text="x",
        page=page,
        bbox=bbox,
        page_width=pw,
        page_height=2000.0,
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


def _region(kind: str, fb_idx, fbs, *, stamp: bool = True) -> Region:
    idxs = tuple(fb_idx) if isinstance(fb_idx, (list, tuple)) else (fb_idx,)
    pb = compute_region_page_bboxes(idxs, fbs) if stamp else None
    return Region(kind=kind, feature_block_indices=idxs, page_bboxes=pb)


# ---------------------------------------------------------------------------
# 1-2. Resolver semantics + legacy alias.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val,expected", [
    ("geom", "geom"), ("fb", "fb"), ("off", "off"),
    ("GEOM", "geom"), (" fb ", "fb"),
    ("garbage", "fb"), ("", "fb"), ("2x", "fb"),
])
def test_resolver_modes_and_garbage(monkeypatch, val, expected):
    monkeypatch.setenv(_MODE, val)
    assert resolve_region_order_mode() == expected


def test_resolver_unset_default_fb(monkeypatch):
    assert resolve_region_order_mode() == "fb"


def test_resolver_legacy_alias(monkeypatch):
    # New flag unset + legacy explicitly falsey -> off, shim False.
    monkeypatch.setenv(_LEGACY, "0")
    assert resolve_region_order_mode() == "off"
    assert resolve_reading_order_fix() is False
    # Legacy truthy / unset -> default mode, shim True.
    monkeypatch.setenv(_LEGACY, "1")
    assert resolve_region_order_mode() == "fb"
    assert resolve_reading_order_fix() is True
    monkeypatch.delenv(_LEGACY, raising=False)
    assert resolve_reading_order_fix() is True
    # An explicit new-flag mode WINS over the legacy alias.
    monkeypatch.setenv(_LEGACY, "0")
    monkeypatch.setenv(_MODE, "geom")
    assert resolve_region_order_mode() == "geom"


# ---------------------------------------------------------------------------
# 3. fb mode byte-identical to the min-FB stable sort.
# ---------------------------------------------------------------------------


def test_fb_mode_byte_identical_to_min_fb_sort():
    fbs = [
        _fb(1, (100.0, 50.0, 300.0, 90.0)),
        _fb(1, (100.0, 200.0, 300.0, 240.0)),
        _fb(1, (100.0, 10.0, 300.0, 40.0)),
    ]
    regions = [
        _region("paragraph", [1], fbs),
        _region("paragraph", [2], fbs),
        _region("paragraph", [0], fbs),
        Region(kind="metadata_drop", feature_block_indices=()),  # empty-FB sentinel
    ]
    out = apply_region_order(list(regions), fbs, mode="fb")
    expected = sorted(
        regions,
        key=lambda r: min(r.feature_block_indices)
        if r.feature_block_indices else len(fbs),
    )
    assert [id(r) for r in out] == [id(r) for r in expected]


# ---------------------------------------------------------------------------
# 4. off mode byte-identical to the segregated pass-order list.
# ---------------------------------------------------------------------------


def test_off_mode_byte_identical_to_segregated(monkeypatch):
    def _seq(regs):
        return [(r.kind, min(r.feature_block_indices)) for r in regs]

    monkeypatch.setenv(_MODE, "off")
    s, f, c, d = _build_interleaved_graph()
    off_seq = _seq(build_structure_graph(s, f, c, d))

    monkeypatch.delenv(_MODE, raising=False)
    monkeypatch.setenv(_LEGACY, "0")
    s2, f2, c2, d2 = _build_interleaved_graph()
    legacy_seq = _seq(build_structure_graph(s2, f2, c2, d2))

    assert off_seq == legacy_seq
    # And it IS the segregated pass order (headings, tables, paragraphs).
    assert off_seq == [
        ("heading", 0), ("heading", 3),
        ("table", 2), ("table", 5),
        ("paragraph", 1), ("paragraph", 4),
    ]


# ---------------------------------------------------------------------------
# 5-6. geom groups columns; page-major.
# ---------------------------------------------------------------------------


def _two_col_raster():
    # FB stream frozen in RASTER order across a 2-col page (col0/col1 alternate).
    fbs = [
        _fb(1, (100.0, 50.0, 300.0, 90.0)),    # 0 col0 top
        _fb(1, (700.0, 60.0, 900.0, 100.0)),   # 1 col1 top
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 2 col0 mid
        _fb(1, (700.0, 210.0, 900.0, 250.0)),  # 3 col1 mid
    ]
    regions = [_region("paragraph", [i], fbs) for i in range(4)]
    return fbs, regions


def test_geom_mode_groups_columns():
    fbs, regions = _two_col_raster()
    fb_out = apply_region_order(list(regions), fbs, mode="fb")
    geom_out = apply_region_order(list(regions), fbs, mode="geom")
    fb_idx = [min(r.feature_block_indices) for r in fb_out]
    geom_idx = [min(r.feature_block_indices) for r in geom_out]
    # fb: raster/interleaved order 0,1,2,3; geom: column 0 (0,2) then column 1 (1,3).
    assert fb_idx == [0, 1, 2, 3]
    assert geom_idx == [0, 2, 1, 3]


def test_geom_multipage_page_major():
    fbs = [
        _fb(1, (100.0, 100.0, 300.0, 140.0)),  # 0 page1
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 1 page1
        _fb(2, (100.0, 10.0, 300.0, 50.0)),    # 2 page2 (tiny y0)
    ]
    regions = [_region("paragraph", [i], fbs) for i in range(3)]
    out = apply_region_order(list(regions), fbs, mode="geom")
    # Page precedes column/row: the page-2 region sorts AFTER both page-1 regions
    # despite its smaller y0.
    assert [min(r.feature_block_indices) for r in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 7. bbox-less region keeps its relative FB position.
# ---------------------------------------------------------------------------


def test_bboxless_region_keeps_relative_fb_position():
    fbs = [
        _fb(1, (100.0, 50.0, 300.0, 90.0)),    # 0 geometry-bearing
        _fb(1, None),                          # 1 bbox-less (raw.bbox None)
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 2 geometry-bearing
    ]
    regions = [_region("paragraph", [i], fbs) for i in range(3)]
    # region 1 has no usable geometry -> bbox-less accessor returns None.
    assert region_order._region_geom(regions[1], fbs) is None
    out = apply_region_order(list(regions), fbs, mode="geom")
    # Re-inserted immediately after its FB-order predecessor (region 0).
    assert [id(r) for r in out] == [id(regions[0]), id(regions[1]), id(regions[2])]


# ---------------------------------------------------------------------------
# 8. float exclusion — a figure does not spawn a column.
# ---------------------------------------------------------------------------


def test_float_exclusion_figure_does_not_spawn_column():
    fbs = [
        _fb(1, (100.0, 50.0, 300.0, 90.0)),    # 0 body col0
        _fb(1, (100.0, 150.0, 300.0, 190.0)),  # 1 body col0
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 2 body col0
        _fb(1, (700.0, 250.0, 1100.0, 400.0)), # 3 figure, right margin, bottom
    ]
    regions = [_region("paragraph", [i], fbs) for i in range(3)]
    regions.append(_region("figure", [3], fbs))
    audit: dict = {}
    out = apply_region_order(list(regions), fbs, mode="geom", audit=audit)
    # The right-margin figure is EXCLUDED from seeding -> single column.
    assert audit["pages"][0]["n_columns"] == 1
    # Single column -> order is y0-sorted == fb order here.
    assert [min(r.feature_block_indices) for r in out] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 9. full-width heading pinned to column 0.
# ---------------------------------------------------------------------------


def test_full_width_heading_pinned_col0():
    fbs = [
        _fb(1, (100.0, 30.0, 900.0, 70.0)),    # 0 full-width heading (straddles)
        _fb(1, (100.0, 100.0, 300.0, 140.0)),  # 1 col0
        _fb(1, (700.0, 100.0, 900.0, 140.0)),  # 2 col1
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 3 col0
        _fb(1, (700.0, 200.0, 900.0, 240.0)),  # 4 col1
    ]
    regions = [_region("heading", [0], fbs)] + [
        _region("paragraph", [i], fbs) for i in range(1, 5)
    ]
    out = apply_region_order(list(regions), fbs, mode="geom")
    order = [min(r.feature_block_indices) for r in out]
    # The straddling heading (FB 0) sorts FIRST (column 0), before both columns.
    assert order[0] == 0
    # Then column 0 (1, 3) precedes column 1 (2, 4).
    assert order == [0, 1, 3, 2, 4]


# ---------------------------------------------------------------------------
# 10. divergence metric.
# ---------------------------------------------------------------------------


def test_divergence_metric():
    assert _inversion_distance([0, 1, 2, 3]) == 0.0    # identity
    assert _inversion_distance([3, 2, 1, 0]) == 1.0    # full reversal
    assert _inversion_distance([0]) == 0.0             # n<2
    assert _inversion_distance([]) == 0.0


# ---------------------------------------------------------------------------
# 11. page ceiling -> fb fallback.
# ---------------------------------------------------------------------------


def test_page_ceiling_falls_back_to_fb():
    # Single-column page whose y0 DECREASES as FB index increases -> geometric
    # (y0) order fully reverses fb order (divergence 1.0 > 0.45 ceiling).
    fbs = [
        _fb(1, (100.0, 400.0, 300.0, 440.0)),  # 0 (bottom)
        _fb(1, (100.0, 300.0, 300.0, 340.0)),  # 1
        _fb(1, (100.0, 200.0, 300.0, 240.0)),  # 2
        _fb(1, (100.0, 100.0, 300.0, 140.0)),  # 3 (top)
    ]
    regions = [_region("paragraph", [i], fbs) for i in range(4)]
    audit: dict = {}
    out = apply_region_order(list(regions), fbs, mode="geom", audit=audit)
    assert audit["pages"][0]["fallback"] is True
    assert audit["pages"][0]["divergence"] == 1.0
    # Fallback -> the page keeps fb order.
    assert [min(r.feature_block_indices) for r in out] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 12. audit stamped in shadow (fb) and geom.
# ---------------------------------------------------------------------------


def test_audit_stamped_in_shadow_and_geom():
    fbs, regions = _two_col_raster()
    shadow: dict = {}
    apply_region_order(list(regions), fbs, mode="fb", audit=shadow)
    assert shadow["mode"] == "fb"
    assert shadow["applied"] is False
    assert "doc_divergence" in shadow and shadow["pages"]
    # fb mode still MEASURES divergence (raster vs column order diverges).
    assert shadow["doc_divergence"] > 0.0

    applied: dict = {}
    apply_region_order(list(regions), fbs, mode="geom", audit=applied)
    assert applied["mode"] == "geom"
    assert applied["applied"] is True


# ---------------------------------------------------------------------------
# 13. permutation conserved (the defensive assert fires on a drop).
# ---------------------------------------------------------------------------


def test_permutation_conserved(monkeypatch):
    fbs, regions = _two_col_raster()
    real_sorted = sorted

    def _dropping_sorted(seq, **kw):
        out = real_sorted(list(seq), **kw)
        # Drop one element from any sort OVER Region objects -> conservation break.
        if out and all(isinstance(x, Region) for x in out):
            return out[1:]
        return out

    monkeypatch.setattr(region_order, "sorted", _dropping_sorted, raising=False)
    with pytest.raises(AssertionError):
        apply_region_order(list(regions), fbs, mode="fb")


# ---------------------------------------------------------------------------
# 14. build_structure_graph threads order_audit.
# ---------------------------------------------------------------------------


def test_build_structure_graph_threads_order_audit(monkeypatch):
    monkeypatch.delenv(_MODE, raising=False)  # default fb (shadow)
    s, f, c, d = _build_interleaved_graph()
    audit: dict = {}
    regions = build_structure_graph(s, f, c, d, order_audit=audit)
    assert regions  # sanity
    assert audit.get("mode") == "fb"
    assert audit.get("applied") is False
    assert "doc_divergence" in audit
    assert isinstance(audit.get("pages"), list)
