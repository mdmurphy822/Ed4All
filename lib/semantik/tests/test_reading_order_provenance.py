"""Phase 4 — provenance / reviewer block_id blast-radius coverage for the
Stage-5 reading-order fix (``SEMANTIK_READING_ORDER_FIX``).

The reading-order fix is a final STABLE sort of ``build_structure_graph``'s
region list by ``min(feature_block_indices)``. It reorders the region LIST;
it must NOT touch the WIRE contract. The downstream sourceId / provenance
values are all FeatureBlock-derived (``cascade.py`` ~``:548``
``first_raw = min(fb_indices)``; the adapter mints ``data-semantik-block-id``
from that anchor, ``adapter.py::_mint_sid``), so they are invariant under the
reorder.

This module proves, with NO models / GPU:

  Part A (structure-graph reorder, the real sort)
    For the SAME synthetic interleaved FB stream driven through the REAL
    ``build_structure_graph`` both flag states, the MULTISET of
    ``first_raw_block_index`` values that ``_build_region_provenance`` emits
    is IDENTICAL flag-off vs flag-on (only the LIST ORDER differs), and the
    flag-on provenance list is sorted ASCENDING by ``first_raw_block_index``.
    The ``region_index`` integers (and, downstream, the reviewer ``block_id``
    = region-list index, ``reviewer.py:536``) DO move — that is expected and
    explicitly asserted, to document the blast radius.

  Part B (adapter sourceId mint, the real wire value)
    For a hand-built ``region_provenance`` and a REORDERED (segregated) copy
    of it — same per-region dict values, different list order — the REAL
    adapter (``build_chapters_ir`` -> ``normalize_cascade_to_ed4all``) emits
    an IDENTICAL multiset of ``data-semantik-block-id`` HTML values and an
    identical multiset of ``first_raw_block_index``. No id VALUE is dropped,
    added, or rewritten by the reorder.

Run:
  PYTHONPATH=SemantiK:. ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_reading_order_provenance.py -q
"""

from __future__ import annotations

import collections
import re

import pytest

from semantik_structure.cascade import _build_region_provenance
from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.types import (
    BertOutput,
    CouncilState,
    TableCandidate,
    TypedSignal,
)
from semantik_structure.structure_graph import build_structure_graph
from semantik_structure.types import FeatureBlock, RawBlock

_FLAG = "SEMANTIK_READING_ORDER_FIX"


# ---------------------------------------------------------------------------
# Part A — synthetic interleaved FB stream + the REAL build_structure_graph.
# Mirrors the harness in semantik_structure/tests/test_structure_reading_order.py.
# ---------------------------------------------------------------------------


def _fb(text: str, *, ratio: float = 1.0, size: str = "md") -> FeatureBlock:
    raw = RawBlock(
        text=text, page=1, bbox=(0.0, 0.0, 100.0, 20.0),
        page_width=200.0, page_height=300.0, font_size=11.0,
    )
    return FeatureBlock(
        raw=raw, size_bucket=size, gap_above=None, is_top_of_page=False,
        is_centered=False, caps=None, indent_bucket=0,
        relative_font_ratio=ratio, is_image=False,
    )


def _heading_signals(idx: int) -> list[TypedSignal]:
    return [
        TypedSignal("is_heading", idx, ["heading", "body"], [0.97, 0.03]),
        TypedSignal("structural_role", idx, ["heading", "paragraph"], [0.95, 0.05]),
    ]


def _paragraph_signals(idx: int) -> list[TypedSignal]:
    return [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, ["paragraph", "heading"], [0.95, 0.05]),
    ]


def _build_interleaved_graph():
    """Run the REAL arbitrate -> build_structure_graph path on a synthetic,
    interleaved heading/paragraph/table FB stream (no GPU, no model)."""
    feature_blocks = [
        _fb("Chapter 1 Introduction", ratio=1.5),                              # 0 heading
        _fb("This is body text that explains the concept in some detail here."),  # 1 paragraph
        _fb("Number Quantity"),                                                # 2 table cells
        _fb("Section 1.1 Details", ratio=1.3),                                 # 3 heading
        _fb("More body prose follows here with a full explanatory sentence."),  # 4 paragraph
        _fb("Item Count"),                                                     # 5 table cells
    ]

    struct_signals: list[TypedSignal] = []
    struct_signals += _heading_signals(0)
    struct_signals += _paragraph_signals(1)
    struct_signals += _paragraph_signals(2)
    struct_signals += _heading_signals(3)
    struct_signals += _paragraph_signals(4)
    struct_signals += _paragraph_signals(5)

    state = CouncilState(outputs={"structure": BertOutput("structure", struct_signals)})

    table_a = TableCandidate(
        kind="table", bbox=(0.0, 0.0, 100.0, 100.0), pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 2, "bordered": False},
        member_block_indices=[2],
        cell_grid=[["Number", "Quantity"], ["1", "10"], ["2", "20"], ["3", "30"]],
        header_row_indices=[0], bordered=False,
    )
    table_b = TableCandidate(
        kind="table", bbox=(0.0, 0.0, 100.0, 100.0), pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 2, "bordered": False},
        member_block_indices=[5],
        cell_grid=[["Item", "Count"], ["a", "1"], ["b", "2"], ["c", "3"]],
        header_row_indices=[0], bordered=False,
    )

    candidates = [table_a, table_b]
    decisions = arbitrate(state, candidates)
    assert [d.route for d in decisions] == ["table", "table"]
    return state, feature_blocks, candidates, decisions


def _provenance_for_flag(monkeypatch, flag_value) -> list[dict]:
    """build_structure_graph both flag states -> _build_region_provenance.

    ``region_order`` is ``range(len(regions))`` (every region emitted in the
    region-list order — reading order when on, segregated when off), exactly
    the shape ``AssembledDoc.region_provenance`` carries for a 1:1 emit.
    """
    if flag_value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, flag_value)
    state, fbs, cands, decs = _build_interleaved_graph()
    regions = build_structure_graph(state, fbs, cands, decs)
    region_order = list(range(len(regions)))
    return _build_region_provenance(region_order, regions, fbs, {}, None)


def test_first_raw_block_index_multiset_identical(monkeypatch):
    """The WIRE anchor (first_raw_block_index) multiset is byte-stable under
    the reorder — only the LIST ORDER differs flag-off vs flag-on."""
    off = _provenance_for_flag(monkeypatch, "0")
    on = _provenance_for_flag(monkeypatch, "1")

    off_first_raw = collections.Counter(p["first_raw_block_index"] for p in off)
    on_first_raw = collections.Counter(p["first_raw_block_index"] for p in on)
    assert off_first_raw == on_first_raw, (off_first_raw, on_first_raw)
    # Same number of regions — none dropped/added by the sort.
    assert len(off) == len(on)


def test_provenance_sorted_ascending_when_flag_on(monkeypatch):
    """Flag on -> region_provenance is monotonic ascending in
    first_raw_block_index (document reading order)."""
    on = _provenance_for_flag(monkeypatch, "1")
    first_raws = [p["first_raw_block_index"] for p in on]
    assert first_raws == sorted(first_raws), first_raws

    # And flag-off is NOT sorted (the segregated pass order) — proves the
    # fix is what produces the monotonic list, not an accident of the input.
    off = _provenance_for_flag(monkeypatch, "0")
    off_first_raws = [p["first_raw_block_index"] for p in off]
    assert off_first_raws != sorted(off_first_raws), off_first_raws


def test_region_index_integers_move_documented(monkeypatch):
    """Blast-radius doc: the region_index integer (== reviewer block_id,
    reviewer.py:536) DOES move under the reorder while the FB-derived anchor
    stays put. The (first_raw -> region_index) mapping differs between
    states."""
    off = _provenance_for_flag(monkeypatch, "0")
    on = _provenance_for_flag(monkeypatch, "1")

    off_map = {p["first_raw_block_index"]: p["region_index"] for p in off}
    on_map = {p["first_raw_block_index"]: p["region_index"] for p in on}
    # Same keys (anchors), but at least one anchor lands at a DIFFERENT
    # region_index — the integer block_id moved (expected, not a regression).
    assert set(off_map) == set(on_map)
    assert off_map != on_map, (off_map, on_map)


def test_no_raw_text_dropped_under_reorder(monkeypatch):
    """No region text is dropped/added by the reorder — the multiset of
    per-region raw_text is identical flag-off vs flag-on."""
    off = _provenance_for_flag(monkeypatch, "0")
    on = _provenance_for_flag(monkeypatch, "1")
    off_text = collections.Counter(p["raw_text"] for p in off)
    on_text = collections.Counter(p["raw_text"] for p in on)
    assert off_text == on_text, (off_text, on_text)


# ---------------------------------------------------------------------------
# Part B — REAL adapter sourceId mint: data-semantik-block-id multiset stable.
# ---------------------------------------------------------------------------


def _reading_order_provenance() -> list[dict]:
    """A document-order ``region_provenance`` (the flag-ON shape): two
    content-bearing level-1 chapter headings, a section heading, prose, a
    list, a figure. Mirrors lib/semantik/tests/test_cascade_ir.py."""
    return [
        {
            "region_index": 0, "region_kind": "heading", "role": "heading",
            "confidence": 0.95, "wcag_status": "passed",
            "first_raw_block_index": 0, "pages": [1],
            "heading_text": "Chapter 1: Foundations", "level": 1,
            "figure_alt": None, "raw_text": "Chapter 1: Foundations",
        },
        {
            "region_index": 1, "region_kind": "heading", "role": "heading",
            "confidence": 0.83, "wcag_status": "passed",
            "first_raw_block_index": 1, "pages": [1],
            "heading_text": "Introduction to Algebra", "level": 2,
            "figure_alt": None, "raw_text": "Introduction to Algebra",
        },
        {
            "region_index": 2, "region_kind": "paragraph", "role": "body",
            "confidence": 0.61, "wcag_status": "passed",
            "first_raw_block_index": 2, "pages": [2, 3],
            "heading_text": None, "level": None, "figure_alt": None,
            "raw_text": "The order of operations is PEMDAS.",
        },
        {
            "region_index": 3, "region_kind": "list", "role": "list",
            "confidence": 0.7, "wcag_status": "passed",
            "first_raw_block_index": 3, "pages": [2],
            "heading_text": None, "level": None, "figure_alt": None,
            "raw_text": "• Add integers. • Subtract integers. • Multiply integers.",
        },
        {
            "region_index": 4, "region_kind": "figure", "role": "figure",
            "confidence": 1.0, "wcag_status": "passed",
            "first_raw_block_index": 4, "pages": [3],
            "heading_text": "Number Line", "level": None,
            "figure_alt": "A number line from -5 to 5.",
            "raw_text": "Figure 1.1 A number line from -5 to 5.",
        },
        {
            "region_index": 5, "region_kind": "heading", "role": "heading",
            "confidence": 0.9, "wcag_status": "passed",
            "first_raw_block_index": 10, "pages": [5],
            "heading_text": "Chapter 2: Linear Equations", "level": 1,
            "figure_alt": None, "raw_text": "Chapter 2: Linear Equations",
        },
        {
            "region_index": 6, "region_kind": "paragraph", "role": "body",
            "confidence": 0.79, "wcag_status": "passed",
            "first_raw_block_index": 11, "pages": [5],
            "heading_text": None, "level": None, "figure_alt": None,
            "raw_text": "Solve for x in 2x + 3 = 7.",
        },
    ]


def _segregate_pass_order(prov: list[dict]) -> list[dict]:
    """Re-order a reading-order provenance into the SEGREGATED pass order the
    flag-OFF ``build_structure_graph`` produces (all headings, then lists,
    then paragraphs, then figures), re-stamping ``region_index`` to the new
    list position — exactly what ``_build_region_provenance`` does on the
    segregated region list. Per-region dict VALUES (incl. first_raw /
    heading_text) are otherwise untouched."""
    order = {"heading": 0, "table": 1, "list": 2, "paragraph": 3, "figure": 4}
    seg = sorted(prov, key=lambda p: order.get(p["region_kind"], 9))
    out = []
    for i, p in enumerate(seg):
        q = dict(p)
        q["region_index"] = i  # the integer moves; the FB anchor does not
        out.append(q)
    return out


class _Result:
    """Duck-typed PipelineV2Result for build_chapters_ir."""

    def __init__(self, provenance: list[dict]):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = "ship_with_confidence"
        self.theta_score = 0.91
        self.flags: list[str] = []
        self.lane_used = "fast-lane"
        self.lang = "en"
        self.region_provenance = provenance
        self.heading_tree = [
            (1, "Chapter 1: Foundations"),
            (2, "Introduction to Algebra"),
            (1, "Chapter 2: Linear Equations"),
        ]


def _adapter_html(provenance: list[dict]) -> dict:
    from lib.semantik.adapter import normalize_cascade_to_ed4all
    from lib.semantik.cascade_ir import build_chapters_ir

    chapters = build_chapters_ir(_Result(provenance))

    class _Res:
        chapters = None
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        theta_score = 0.91
        flags: list[str] = []
        lane_used = "fast-lane"
        lang = "en"

    res = _Res()
    res.chapters = chapters
    return normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")


def _block_ids(html: str) -> collections.Counter:
    return collections.Counter(re.findall(r'data-semantik-block-id="([^"]+)"', html))


def test_data_semantik_block_id_multiset_stable_under_reorder():
    """The adapter mints an IDENTICAL multiset of data-semantik-block-id values
    from the reading-order provenance and its segregated reorder — no id
    VALUE dropped, added, or rewritten by the reorder."""
    reading = _reading_order_provenance()
    segregated = _segregate_pass_order(reading)

    reading_out = _adapter_html(reading)
    seg_out = _adapter_html(segregated)

    reading_ids = _block_ids(reading_out["html"])
    seg_ids = _block_ids(seg_out["html"])
    assert reading_ids, "no data-semantik-block-id emitted"
    assert reading_ids == seg_ids, (reading_ids, seg_ids)


def test_first_raw_multiset_stable_through_adapter():
    """The first_raw_block_index multiset on the produced IR blocks is
    identical for the reading-order and segregated provenance."""
    from lib.semantik.cascade_ir import build_chapters_ir

    reading = _reading_order_provenance()
    segregated = _segregate_pass_order(reading)

    r_blocks = [b for c in build_chapters_ir(_Result(reading)) for b in c.blocks]
    s_blocks = [b for c in build_chapters_ir(_Result(segregated)) for b in c.blocks]

    r_anchors = collections.Counter(b.raw_block_index for b in r_blocks)
    s_anchors = collections.Counter(b.raw_block_index for b in s_blocks)
    assert r_anchors == s_anchors, (r_anchors, s_anchors)


@pytest.mark.parametrize("flag", ["1", None])
def test_adapter_sids_resolve_against_sidecar(monkeypatch, flag):
    """Whichever order the regions arrive in, every HTML data-semantik-block-id
    still resolves against the synthesized sidecar (the source_refs gate's
    valid-id universe) — the reorder never strands a sourceId."""
    if flag is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, flag)
    prov = _reading_order_provenance() if flag else _segregate_pass_order(
        _reading_order_provenance()
    )
    out = _adapter_html(prov)
    sidecar_ids = {s["section_id"] for s in out["synthesized_sidecar"]["sections"]}
    html_ids = set(re.findall(r'data-semantik-block-id="([^"]+)"', out["html"]))
    assert html_ids
    assert html_ids <= sidecar_ids, (html_ids - sidecar_ids)
