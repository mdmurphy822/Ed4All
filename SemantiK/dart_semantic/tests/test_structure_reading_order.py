"""Stage-5 reading-order diagnostic — pins the region-order bug both ways.

``build_structure_graph`` appends regions in PASS order (Pass-2 all
headings, Pass-3 tables/math, Pass-5 paragraphs, ...) and never re-sorts to
reading order, so the flat Region list is SEGREGATED by kind (every heading
clustered at the front, every body paragraph at the back) rather than
monotone in document order. On OpenStax ch1-3 that surfaces as EXAMPLE
labels (council=heading) at byte-4% and their bodies (paragraph) at byte-40%.

The fix is a final STABLE sort by ``min(feature_block_indices)`` at the exit
of ``build_structure_graph``, gated behind ``SEMANTIK_READING_ORDER_FIX``
(default off, byte-identical when off).

This test drives the REAL ``build_structure_graph`` path on a synthetic,
GPU-free CouncilState (mirroring the harness in
``test_figure_table_assembly_e2e.py`` — synthetic ``CouncilState`` +
``TypedSignal`` + ``TableCandidate`` + ``arbitrate``, no model). It builds an
interleaved heading/paragraph/table FB stream so the flag-OFF pass order
produces the heading→table→paragraph segregation, and asserts the flag-ON
sort restores reading order with zero inversions while preserving FB coverage
under both states.
"""

from __future__ import annotations

import pytest

from dart_semantic.council.cross_reranker import arbitrate
from dart_semantic.council.types import (
    BertOutput,
    CouncilState,
    TableCandidate,
    TypedSignal,
)
from dart_semantic.structure_graph import (
    build_structure_graph,
    resolve_reading_order_fix,
)
from dart_semantic.types import FeatureBlock, RawBlock

_FLAG = "SEMANTIK_READING_ORDER_FIX"


# ---------------------------------------------------------------------------
# Builders — a synthetic, no-GPU interleaved FB stream + CouncilState.
# ---------------------------------------------------------------------------


def _fb(text: str, *, ratio: float = 1.0, size: str = "md", bold: bool = False) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        page_width=200.0,
        page_height=300.0,
        font_size=11.0,
    )
    # ``is_bold`` is consulted by the heading passes; RawBlock may or may not
    # carry the attribute, so set it defensively.
    try:
        raw.is_bold = bold  # type: ignore[attr-defined]
    except Exception:
        pass
    return FeatureBlock(
        raw=raw,
        size_bucket=size,
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=ratio,
        is_image=False,
    )


def _heading_signals(idx: int) -> list[TypedSignal]:
    """Structure head: confident heading (is_heading >= 0.8 + role=heading)."""
    return [
        TypedSignal("is_heading", idx, ["heading", "body"], [0.97, 0.03]),
        TypedSignal("structural_role", idx, ["heading", "paragraph"], [0.95, 0.05]),
    ]


def _paragraph_signals(idx: int) -> list[TypedSignal]:
    """Structure head: confident body paragraph (falls to the prose pass)."""
    return [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, ["paragraph", "heading"], [0.95, 0.05]),
    ]


def _build_interleaved_graph():
    """Run the REAL arbitrate -> build_structure_graph path on a synthetic,
    interleaved heading/paragraph/table FB stream (no GPU, no model).

    Returns ``(state, feature_blocks, candidates, decisions)`` so the caller
    can re-run ``build_structure_graph`` under either flag state.
    """
    # Three interleaved "units": (heading, body-paragraph, table) x2, so the
    # council labels alternate kinds and the flag-OFF pass order segregates
    # them (all headings first, then tables, then paragraphs).
    feature_blocks = [
        _fb("Chapter 1 Introduction", ratio=1.5, bold=True),                       # 0 heading
        _fb("This is body text that explains the concept in some detail here."),   # 1 paragraph
        _fb("Number Quantity"),                                                    # 2 table cells
        _fb("Section 1.1 Details", ratio=1.3, bold=True),                          # 3 heading
        _fb("More body prose follows here with a full explanatory sentence."),     # 4 paragraph
        _fb("Item Count"),                                                         # 5 table cells
    ]

    struct_signals: list[TypedSignal] = []
    struct_signals += _heading_signals(0)
    struct_signals += _paragraph_signals(1)
    struct_signals += _paragraph_signals(2)  # claimed by the table passthrough
    struct_signals += _heading_signals(3)
    struct_signals += _paragraph_signals(4)
    struct_signals += _paragraph_signals(5)  # claimed by the table passthrough

    state = CouncilState(outputs={"structure": BertOutput("structure", struct_signals)})

    table_a = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 2, "bordered": False},
        member_block_indices=[2],
        cell_grid=[["Number", "Quantity"], ["1", "10"], ["2", "20"], ["3", "30"]],
        header_row_indices=[0],
        bordered=False,
    )
    table_b = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 2, "bordered": False},
        member_block_indices=[5],
        cell_grid=[["Item", "Count"], ["a", "1"], ["b", "2"], ["c", "3"]],
        header_row_indices=[0],
        bordered=False,
    )

    candidates = [table_a, table_b]
    decisions = arbitrate(state, candidates)
    assert [d.route for d in decisions] == ["table", "table"], (
        "deterministic table routing precondition failed"
    )
    return state, feature_blocks, candidates, decisions


def _inversions(regions) -> int:
    """Count adjacent pairs out of reading order — ``i`` where the next
    region's first FB index is SMALLER than this region's."""
    return sum(
        1
        for i in range(len(regions) - 1)
        if min(regions[i + 1].feature_block_indices) < min(regions[i].feature_block_indices)
    )


def _first_index_of_kind(regions, kind: str) -> int:
    for i, r in enumerate(regions):
        if r.kind == kind:
            return i
    raise AssertionError(f"no {kind!r} region formed (saw {[r.kind for r in regions]})")


# ---------------------------------------------------------------------------
# Phase 0 — the diagnostic, both flag states.
# ---------------------------------------------------------------------------


def test_segregated_when_flag_off(monkeypatch):
    """Flag unset -> the legacy pass-order segregation: >0 inversions AND a
    heading region precedes its body-paragraph region by list index."""
    monkeypatch.delenv(_FLAG, raising=False)
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)

    assert _inversions(regions) > 0, (
        f"expected segregation inversions, got reading order: "
        f"{[r.kind for r in regions]}"
    )
    # The first heading clusters BEFORE the first body paragraph (segregation).
    assert _first_index_of_kind(regions, "heading") < _first_index_of_kind(regions, "paragraph")


def test_reading_order_when_flag_on(monkeypatch):
    """Flag on -> 0 inversions AND the region list equals itself sorted by
    ``min(feature_block_indices)`` (canonical reading order)."""
    monkeypatch.setenv(_FLAG, "1")
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)

    assert _inversions(regions) == 0, (
        f"reading-order sort left inversions: {[r.kind for r in regions]}"
    )
    expected = sorted(regions, key=lambda r: min(r.feature_block_indices))
    assert [id(r) for r in regions] == [id(r) for r in expected], (
        "region list is not in canonical min-FB reading order"
    )


@pytest.mark.parametrize("flag_value", [None, "1"])
def test_coverage_invariant_holds_both_states(monkeypatch, flag_value):
    """Under BOTH flag states every input FB index is owned by exactly one
    output region (the sort permutes, never drops/adds)."""
    if flag_value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, flag_value)
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)

    owned: list[int] = [j for r in regions for j in r.feature_block_indices]
    assert sorted(owned) == list(range(len(fbs))), (
        f"FB coverage broken (flag={flag_value!r}): owned={sorted(owned)}"
    )
    assert len(owned) == len(set(owned)), "an FB is owned by more than one region"


def test_flag_off_region_order_byte_identical_to_baseline(monkeypatch):
    """The flag-off region list is the EXACT segregated baseline (the sort is
    gated). Pinned as the (kind, min-FB) sequence so a future accidental
    always-on sort is caught."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert resolve_reading_order_fix() is False
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)

    baseline = [(r.kind, min(r.feature_block_indices)) for r in regions]
    assert baseline == [
        ("heading", 0),
        ("heading", 3),
        ("table", 2),
        ("table", 5),
        ("paragraph", 1),
        ("paragraph", 4),
    ], f"flag-off baseline drifted: {baseline}"


# ---------------------------------------------------------------------------
# Phase 1 — flag-resolver semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "0", "false", "no", "off", "garbage", "2", "  "])
def test_reading_order_flag_off_by_default(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, raw)
    assert resolve_reading_order_fix() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "YES", "on", " on ", "Yes"])
def test_reading_order_flag_truthy(monkeypatch, raw):
    monkeypatch.setenv(_FLAG, raw)
    assert resolve_reading_order_fix() is True
