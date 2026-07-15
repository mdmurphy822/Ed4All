"""Wiring tests for the Super-120B fused-page re-segmentation stage (Step 2).

Covers the ``snap_elements_to_units`` helper, the ``resegment_arranger_paragraph``
entry point, and the ``page_arranger.build_regions_for_page`` call-site wiring.

NO network: every test that reaches ``resegment_page`` monkeypatches the sole
HTTP boundary (``super_resegment._super_completion``); the byte-identical-OFF
test never touches the module at all.

The module lives at ``semantik_structure/super_resegment.py`` and the arranger at
``semantik_structure/page_arranger.py``; this file is outside the package tree,
so it bootstraps ``SemantiK/`` onto ``sys.path`` the same way the in-package
``conftest.py`` / the sibling ``test_super_resegment.py`` does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# .../SemantiK/tests/test_super_resegment_wiring.py → parents[1] == SemantiK/
_SEMANTIK_ROOT = Path(__file__).resolve().parents[1]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from semantik_structure import page_arranger as pa  # noqa: E402
from semantik_structure import super_resegment as sr  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402


# ---------------------------------------------------------------------------
# Builders (NO course/corpus text).
# ---------------------------------------------------------------------------
def _fb(text, *, page=1, y0=0.0):
    raw = RawBlock(
        text=text, page=page, bbox=(0.0, y0, 100.0, y0 + 10.0),
        page_width=612.0, page_height=792.0, source="tesseract",
    )
    return FeatureBlock(
        raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
        is_centered=False, caps=None, indent_bucket=0, relative_font_ratio=1.0,
        provenance="tesseract", is_image=False,
    )


def _units(texts):
    """Mint member-unit dicts {id, text, fb_index} in order (fb_index == pos)."""
    return [
        {"id": f"p1_b{i:02d}", "text": t, "fb_index": i}
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# snap_elements_to_units — exact, single, fail→None, multi-unit element, list.
# ---------------------------------------------------------------------------
def test_snap_exact_multi_element():
    units = _units(["Introduction", "This chapter covers integers.", "TRY IT Solve for x."])
    elements = [
        {"type": "heading", "text": "Introduction"},
        {"type": "para", "text": "This chapter covers integers."},
        {"type": "callout", "text": "TRY IT Solve for x."},
    ]
    snapped = sr.snap_elements_to_units(elements, units)
    assert snapped is not None
    assert [d["type"] for d in snapped] == ["heading", "para", "callout"]
    assert [d["fb_idxs"] for d in snapped] == [[0], [1], [2]]
    assert snapped[1]["text"] == "This chapter covers integers."


def test_snap_single_whole_block_element():
    units = _units(["Alpha", "beta", "gamma"])
    elements = [{"type": "para", "text": "Alpha beta gamma"}]
    snapped = sr.snap_elements_to_units(elements, units)
    assert snapped is not None
    assert len(snapped) == 1
    assert snapped[0]["fb_idxs"] == [0, 1, 2]


def test_snap_element_spans_multiple_units():
    units = _units(["The cat", "sat on", "the mat.", "The end."])
    elements = [
        {"type": "para", "text": "The cat sat on the mat."},
        {"type": "para", "text": "The end."},
    ]
    snapped = sr.snap_elements_to_units(elements, units)
    assert snapped is not None
    assert [d["fb_idxs"] for d in snapped] == [[0, 1, 2], [3]]
    # every parent unit claimed exactly once, in order
    claimed = [i for d in snapped for i in d["fb_idxs"]]
    assert claimed == [0, 1, 2, 3]


def test_snap_list_items_element():
    units = _units(["First item", "Second item", "Third item"])
    elements = [{"type": "list", "items": ["First item", "Second item", "Third item"]}]
    snapped = sr.snap_elements_to_units(elements, units)
    assert snapped is not None
    assert len(snapped) == 1
    assert snapped[0]["type"] == "list"
    assert snapped[0]["fb_idxs"] == [0, 1, 2]
    assert snapped[0]["items"] == ["First item", "Second item", "Third item"]


def test_snap_misaligned_boundary_returns_none():
    # An element boundary that cuts INSIDE a unit (overshoots) cannot snap.
    units = _units(["Introduction This", "chapter covers integers."])
    elements = [
        {"type": "heading", "text": "Introduction"},   # only part of unit 0
        {"type": "para", "text": "This chapter covers integers."},
    ]
    assert sr.snap_elements_to_units(elements, units) is None


def test_snap_leftover_units_returns_none():
    units = _units(["Alpha", "beta", "gamma"])
    elements = [{"type": "para", "text": "Alpha beta"}]  # never claims gamma
    assert sr.snap_elements_to_units(elements, units) is None


def test_snap_empty_element_returns_none():
    units = _units(["Alpha", "beta"])
    assert sr.snap_elements_to_units([{"type": "para", "text": ""}], units) is None


# ---------------------------------------------------------------------------
# resegment_arranger_paragraph — combine resegment_page + snap.
# ---------------------------------------------------------------------------
_FUSED = "Introduction This chapter covers integers. TRY IT Solve for x."
_UNITS3 = _units(["Introduction", "This chapter covers integers.", "TRY IT Solve for x."])
_TAGGED_OK = (
    "<<heading>>Introduction\n"
    "<<para>>This chapter covers integers.\n"
    "<<callout>>TRY IT Solve for x.\n"
)


def test_entry_on_split_snaps(monkeypatch):
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": _TAGGED_OK, "finish": "stop"},
    )
    descs = sr.resegment_arranger_paragraph(
        _UNITS3, _FUSED, mode="on", seat=sr.resolve_super_seat(),
    )
    assert descs is not None
    assert [d["type"] for d in descs] == ["heading", "para", "callout"]
    assert [d["fb_idxs"] for d in descs] == [[0], [1], [2]]


def test_entry_table_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("Super must not be called on a table")

    monkeypatch.setattr(sr, "_super_completion", _boom)
    tabular = (
        r"\begin{tabular}{cc} A & B \\ \(a\) & \(b\) \\ \end{tabular}"
    )
    units = _units([tabular])
    assert sr.resegment_arranger_paragraph(
        units, tabular, mode="on", seat=sr.resolve_super_seat()
    ) is None


def test_entry_snap_failure_returns_none(monkeypatch):
    # Conserved split, but the element boundary does not align to any unit
    # boundary → snap None → entry None.
    tagged = "<<para>>Introduction This\n<<para>>chapter covers integers. TRY IT Solve for x.\n"
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": tagged, "finish": "stop"},
    )
    assert sr.resegment_arranger_paragraph(
        _UNITS3, _FUSED, mode="on", seat=sr.resolve_super_seat()
    ) is None


def test_entry_error_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(sr, "_super_completion", _raise)
    assert sr.resegment_arranger_paragraph(
        _UNITS3, _FUSED, mode="on", seat=sr.resolve_super_seat()
    ) is None


def test_entry_shadow_returns_none_but_audits(monkeypatch):
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": _TAGGED_OK, "finish": "stop"},
    )
    audit_rows: list = []
    descs = sr.resegment_arranger_paragraph(
        _UNITS3, _FUSED, mode="shadow", seat=sr.resolve_super_seat(),
        audit=audit_rows.append,
    )
    assert descs is None  # shadow applies nothing
    assert len(audit_rows) == 1
    row = audit_rows[0]
    assert row["shadow"] is True
    assert row["action"] == "split"
    assert row["snapped"] is True  # the snap was computed for calibration
    assert row["n_regions"] == 3


# ---------------------------------------------------------------------------
# build_regions_for_page wiring.
# ---------------------------------------------------------------------------
def _result_one_paragraph(unit_ids):
    return {
        "status": "ok",
        "arrangement": {"blocks": [{"ids": list(unit_ids), "type": "paragraph"}], "confidence": 0.9},
        "attempts": 1,
    }


def _baseline_regions(fbs, units, result):
    """Capture the pre-wiring Region list by forcing the flag OFF."""
    import os

    prev = os.environ.pop("SEMANTIK_SUPER_RESEGMENT", None)
    try:
        return pa.build_regions_for_page(1, units, result, fbs)
    finally:
        if prev is not None:
            os.environ["SEMANTIK_SUPER_RESEGMENT"] = prev


def test_flag_off_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SUPER_RESEGMENT", raising=False)
    # Super must NEVER be reached on the off path.
    monkeypatch.setattr(
        sr, "resegment_arranger_paragraph",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run when off")),
    )
    fbs = [_fb("Introduction"), _fb("This chapter covers integers."), _fb("TRY IT Solve for x.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = _result_one_paragraph([u["id"] for u in units])

    regions = pa.build_regions_for_page(1, units, result, fbs)
    # The whole fused paragraph is ONE region claiming all three FBs.
    assert [r.kind for r in regions] == ["paragraph"]
    assert regions[0].feature_block_indices == (0, 1, 2)
    assert regions[0].payload["text"] == "Introduction This chapter covers integers. TRY IT Solve for x."


def test_flag_on_split_replaces_one_region_with_three(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SUPER_RESEGMENT", "on")
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": _TAGGED_OK, "finish": "stop"},
    )
    fbs = [_fb("Introduction"), _fb("This chapter covers integers."), _fb("TRY IT Solve for x.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = _result_one_paragraph([u["id"] for u in units])

    regions = pa.build_regions_for_page(1, units, result, fbs)
    assert [r.kind for r in regions] == ["heading", "paragraph", "paragraph"]
    # fb_idxs partitioned EXACTLY across the sub-regions, none dropped/overlapping
    claimed = sorted(i for r in regions for i in r.feature_block_indices)
    assert claimed == [0, 1, 2]
    assert regions[0].feature_block_indices == (0,)
    assert regions[1].feature_block_indices == (1,)
    assert regions[2].feature_block_indices == (2,)
    # the callout sub-region carries the pedagogy css_class mapping
    assert regions[2].payload.get("css_class") == "pedagogy-try-it"
    assert all(r.payload.get("super_resegment") for r in regions)


def test_flag_on_snap_failure_keeps_single_region(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SUPER_RESEGMENT", "on")
    # A conserved split whose boundaries DON'T align to the unit boundaries.
    tagged = "<<para>>Introduction This\n<<para>>chapter covers integers. TRY IT Solve for x.\n"
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": tagged, "finish": "stop"},
    )
    fbs = [_fb("Introduction"), _fb("This chapter covers integers."), _fb("TRY IT Solve for x.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = _result_one_paragraph([u["id"] for u in units])

    regions = pa.build_regions_for_page(1, units, result, fbs)
    baseline = _baseline_regions(fbs, units, result)
    # Coverage preserved: falls back to the single fused region (baseline shape).
    assert [r.kind for r in regions] == [r.kind for r in baseline]
    assert regions[0].feature_block_indices == baseline[0].feature_block_indices == (0, 1, 2)


def test_flag_shadow_emits_single_region(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SUPER_RESEGMENT", "shadow")
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda seat, system, user, *, timeout: {"content": _TAGGED_OK, "finish": "stop"},
    )
    fbs = [_fb("Introduction"), _fb("This chapter covers integers."), _fb("TRY IT Solve for x.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = _result_one_paragraph([u["id"] for u in units])

    regions = pa.build_regions_for_page(1, units, result, fbs)
    baseline = _baseline_regions(fbs, units, result)
    # No mutation in shadow: the single fused region is emitted, unchanged.
    assert [r.kind for r in regions] == [r.kind for r in baseline]
    assert regions[0].feature_block_indices == (0, 1, 2)


def test_flag_on_single_member_not_split(monkeypatch):
    # A paragraph block with only ONE member unit is INELIGIBLE (>=2 required):
    # Super is never called and the region is byte-identical.
    monkeypatch.setenv("SEMANTIK_SUPER_RESEGMENT", "on")
    monkeypatch.setattr(
        sr, "_super_completion",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("single-unit block is ineligible")),
    )
    fbs = [_fb("A lone paragraph.")]
    units = pa.mint_units_by_page(fbs)[1]
    result = _result_one_paragraph([units[0]["id"]])

    regions = pa.build_regions_for_page(1, units, result, fbs)
    assert [r.kind for r in regions] == ["paragraph"]
    assert regions[0].feature_block_indices == (0,)
