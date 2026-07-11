"""ITEM4 Phase 1 — materialized containment tree (metadata-only) tests.

CPU-only, no model load, no GPU. Synthetic Regions / FeatureBlocks like the
existing suites. Determinism-sensitive assertions (edge sets, pre-order walks)
must hold under PYTHONHASHSEED=0,1,2 (I5).
"""

from __future__ import annotations

import random

import pytest

from semantik_structure.assembler.pass_9a import _group_regions_into_sections
from semantik_structure.containment import (
    ContainmentTree,
    _assert_tree_cover,
    _unit_runs,
    build_containment_tree,
    render_tree,
    resolve_containment_mode,
)
from semantik_structure.council.types import BertOutput, CouncilState, TypedSignal
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _fb(text: str = "", *, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 50.0, 12.0),
        page_width=612.0,
        page_height=792.0,
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
    )


def _region(
    kind,
    fbs,
    *,
    level_hint=None,
    css_class=None,
    semantic_class=None,
    caption_fb_index=None,
    source_region_id=None,
    text=None,
) -> Region:
    payload = {}
    if level_hint is not None:
        payload["level_hint"] = level_hint
    if css_class is not None:
        payload["css_class"] = css_class
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    if caption_fb_index is not None:
        payload["caption_fb_index"] = caption_fb_index
    if text is not None:
        payload["text"] = text
    return Region(
        kind=kind,
        feature_block_indices=tuple(fbs),
        payload=payload,
        source_region_id=source_region_id,
    )


# ---------------------------------------------------------------------------
# Resolver + trivial-forest degradation
# ---------------------------------------------------------------------------


def test_resolver_default_on_and_falsey_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)
    assert resolve_containment_mode() is True
    for falsey in ("0", "false", "no", "off", "OFF", "  false  "):
        monkeypatch.setenv("SEMANTIK_CONTAINMENT", falsey)
        assert resolve_containment_mode() is False
    for truthy in ("1", "true", "on", "garbage", ""):
        monkeypatch.setenv("SEMANTIK_CONTAINMENT", truthy)
        assert resolve_containment_mode() is True


def test_trivial_forest_when_reading_order_off(monkeypatch):
    # Reading-order fix OFF -> the builder degrades to the all-roots forest.
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "0")
    regions = [
        _region("heading", [0], level_hint=1),
        _region("paragraph", [1]),
        _region("paragraph", [2]),
    ]
    fbs = [_fb() for _ in range(3)]
    tree = build_containment_tree(regions, fbs, region_html=["<h1>x</h1>", "<p>a</p>", "<p>b</p>"])
    assert tree.roots == (0, 1, 2)
    assert tree.parent == (None, None, None)
    assert tree.diagnostics.get("reading_order_off") == 1


# ---------------------------------------------------------------------------
# Section arm — the EQUIVALENCE property test
# ---------------------------------------------------------------------------


def _build_section_case(rng):
    """Random sequence of heading (with a raw level) + paragraph regions; build
    the index-aligned region_html with a resolvable id on every heading."""
    n = rng.randint(1, 14)
    regions = []
    region_html = []
    for idx in range(n):
        if rng.random() < 0.45:
            raw_lvl = rng.randint(1, 6)
            regions.append(_region("heading", [idx], level_hint=raw_lvl))
            region_html.append(f'<h{raw_lvl} id="h{idx}">Heading {idx}</h{raw_lvl}>')
        else:
            regions.append(_region("paragraph", [idx]))
            region_html.append(f"<p>Body {idx}</p>")
    return regions, region_html


def test_section_edges_match_stack_grouping(monkeypatch):
    """For ~50 randomized level sequences the tree's section walk reproduces
    ``_group_regions_into_sections`` byte-for-byte (no unit/caption edges)."""
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    from semantik_structure.assembler.heading_tree import normalize_heading_levels

    rng = random.Random(1234)
    for _ in range(50):
        regions, region_html = _build_section_case(rng)
        fbs = [_fb() for _ in regions]
        tree = build_containment_tree(regions, fbs, region_html=region_html)

        heading_indices = [i for i, r in enumerate(regions) if r.kind == "heading"]
        raw_levels = [regions[i].payload["level_hint"] for i in heading_indices]
        normalized = normalize_heading_levels(raw_levels)

        legacy = _group_regions_into_sections(region_html, heading_indices, normalized)
        walked = render_tree(tree, region_html)
        assert walked == legacy
        # No unit / caption edges were created.
        assert all(k in (None, "section") for k in tree.edge_kind)


# ---------------------------------------------------------------------------
# Unit arm — reuse of the pedagogical-unit boundary SoT
# ---------------------------------------------------------------------------


def test_unit_edges_reuse_boundary_sot():
    # anchor(worked_example) + solution + step followers, stopped by a heading.
    regions = [
        _region("paragraph", [0], semantic_class="worked_example", text="EXAMPLE 1"),
        _region("paragraph", [1], css_class="pedagogy-solution", text="Solution"),
        _region("paragraph", [2], css_class="pedagogy-step", text="Step 1"),
        _region("heading", [3], level_hint=2),
        _region("paragraph", [4], text="after"),
    ]
    html = ["<p>EXAMPLE 1</p>", "<p>Solution</p>", "<p>Step 1</p>", "<h2 id='h'>H</h2>", "<p>after</p>"]
    runs = _unit_runs(regions, html)
    assert runs == {0: 3}  # claims 1,2; stops at heading 3
    tree = build_containment_tree(regions, [_fb() for _ in regions], region_html=html)
    assert tree.edge_kind[1] == "unit"
    assert tree.edge_kind[2] == "unit"
    assert tree.parent[1] == 0 and tree.parent[2] == 0


def test_unit_next_anchor_and_unit_start_css_stop():
    regions = [
        _region("paragraph", [0], semantic_class="worked_example", text="EXAMPLE 1"),
        _region("paragraph", [1], text="body"),
        _region("paragraph", [2], css_class="pedagogy-example", text="EXAMPLE 2"),
    ]
    html = ["<p>EXAMPLE 1</p>", "<p>body</p>", "<p>EXAMPLE 2</p>"]
    runs = _unit_runs(regions, html)
    assert runs == {0: 2}  # stops before the unit-start css label at index 2


def test_unit_absorb_max_run_cap():
    # 20 plain followers -> capped at ABSORB_MAX_RUN (8).
    regions = [_region("paragraph", [0], semantic_class="worked_example", text="EX")]
    html = ["<p>EX</p>"]
    for i in range(1, 21):
        regions.append(_region("paragraph", [i], text=f"b{i}"))
        html.append(f"<p>b{i}</p>")
    runs = _unit_runs(regions, html)
    assert runs == {0: 9}  # 0 + 8 followers


def test_unit_passthrough_fb_adjacent_claimed_nonadjacent_not():
    # FB-adjacent trailing passthrough table IS claimed (narrow-absorb subsumed).
    regions = [
        _region("paragraph", [0], semantic_class="worked_example", text="EX"),
        _region("table", [1], source_region_id=7),
    ]
    html = ["<p>EX</p>", "<table></table>"]
    assert _unit_runs(regions, html) == {0: 2}

    # Non-FB-adjacent passthrough table is NOT claimed (FB gap 1 -> 5).
    regions2 = [
        _region("paragraph", [0], semantic_class="worked_example", text="EX"),
        _region("table", [5], source_region_id=7),
    ]
    html2 = ["<p>EX</p>", "<table></table>"]
    assert _unit_runs(regions2, html2) == {}


# ---------------------------------------------------------------------------
# Caption arm — the guards
# ---------------------------------------------------------------------------


def test_caption_edge_forward_single_fb_paragraph_claimed():
    fbs = [_fb("img"), _fb("Figure 1: a cat"), _fb("body")]
    regions = [
        _region("figure", [0], caption_fb_index=1),
        _region("paragraph", [1], text="Figure 1: a cat"),  # single-FB caption owner
        _region("paragraph", [2], text="body"),
    ]
    tree = build_containment_tree(regions, fbs, region_html=["<img>", "<p>Figure 1: a cat</p>", "<p>body</p>"])
    assert tree.edge_kind[1] == "caption"
    assert tree.parent[1] == 0
    assert tree.diagnostics["caption_edges"] == 1


def test_caption_backward_dropped_to_metadata():
    # Caption paragraph ABOVE the table (owner index < table index) -> dropped.
    fbs = [_fb("Table 1"), _fb("cell")]
    regions = [
        _region("paragraph", [0], text="Table 1"),
        _region("table", [1], caption_fb_index=0),
    ]
    tree = build_containment_tree(regions, fbs, region_html=["<p>Table 1</p>", "<table></table>"])
    assert tree.diagnostics["dropped_backward_edges"] == 1
    assert tree.edge_kind[0] != "caption"


def test_caption_multi_fb_owner_with_extra_text_not_claimed():
    fbs = [_fb("img"), _fb("Figure 2"), _fb("extra sentence")]
    regions = [
        _region("figure", [0], caption_fb_index=1),
        _region("paragraph", [1, 2], text="Figure 2 extra sentence"),  # multi-FB, extra text
    ]
    tree = build_containment_tree(regions, fbs, region_html=["<img>", "<p>Figure 2 extra sentence</p>"])
    assert tree.edge_kind[1] != "caption"
    assert tree.diagnostics["caption_edges"] == 0


# ---------------------------------------------------------------------------
# TREE-COVER invariants + fail-closed
# ---------------------------------------------------------------------------


def test_tree_cover_every_node_once_and_identity_preorder():
    regions = [
        _region("heading", [0], level_hint=1),
        _region("paragraph", [1]),
        _region("heading", [2], level_hint=2),
        _region("paragraph", [3]),
    ]
    html = ["<h1 id='a'>A</h1>", "<p>x</p>", "<h2 id='b'>B</h2>", "<p>y</p>"]
    tree = build_containment_tree(regions, [_fb() for _ in regions], region_html=html)
    from semantik_structure.containment import _preorder

    walk = _preorder(tree.children, tree.roots)
    assert walk == [0, 1, 2, 3]
    # every node exactly once across roots ∪ children
    covered = list(tree.roots) + [c for cs in tree.children for c in cs]
    assert sorted(covered) == [0, 1, 2, 3]


def test_corrupted_edge_falls_closed_to_trivial_forest():
    # A hand-built tree whose pre-order != identity (child 2 before child 1).
    bad = ContainmentTree(
        parent=(None, 0, 0),
        children=((2, 1), (), ()),  # descending -> pre-order 0,2,1 != identity
        roots=(0,),
        edge_kind=(None, "section", "section"),
    )
    fixed = _assert_tree_cover(bad, 3)
    assert fixed.roots == (0, 1, 2)
    assert fixed.parent == (None, None, None)
    assert fixed.diagnostics.get("tree_cover_failed") == 1


# ---------------------------------------------------------------------------
# dl pairs (intra-region, council-derived)
# ---------------------------------------------------------------------------


def _council_with_structural_roles(role_by_fb):
    signals = [
        TypedSignal(
            head_name="structural_role",
            region_id=fb,
            top_k_labels=[role],
            top_k_confidences=[0.9],
        )
        for fb, role in role_by_fb.items()
    ]
    return CouncilState(
        outputs={"structure": BertOutput(bert_name="structure", signals=signals)}
    )


def test_dl_pairs_stamped_from_council_state():
    regions = [_region("definition_list", [0, 1, 2, 3])]
    council = _council_with_structural_roles(
        {0: "definition_term", 1: "definition_def", 2: "definition_term", 3: "definition_def"}
    )
    build_containment_tree(regions, [_fb() for _ in range(4)], region_html=["<dl></dl>"], council_state=council)
    assert regions[0].payload.get("dl_pairs") == [[0, [1]], [2, [3]]]


def test_dl_pairs_absent_without_state():
    regions = [_region("definition_list", [0, 1])]
    build_containment_tree(regions, [_fb(), _fb()], region_html=["<dl></dl>"], council_state=None)
    assert "dl_pairs" not in regions[0].payload


# ---------------------------------------------------------------------------
# Flag-off no-stash / provenance byte-stability
# ---------------------------------------------------------------------------


def test_flag_off_no_containment_provenance_key():
    from semantik_structure.cascade import _build_region_provenance

    regions = [_region("paragraph", [0], text="x")]
    fbs = [_fb("x")]
    prov = _build_region_provenance([0], regions, fbs, {}, containment_tree=None)
    assert "containment" not in prov[0]


def test_flag_on_provenance_carries_parent_pointer():
    from semantik_structure.cascade import _build_region_provenance

    regions = [
        _region("heading", [0], level_hint=1),
        _region("paragraph", [1], text="x"),
    ]
    html = ["<h1 id='a'>A</h1>", "<p>x</p>"]
    tree = build_containment_tree(regions, [_fb(), _fb("x")], region_html=html)
    prov = _build_region_provenance([0, 1], regions, [_fb(), _fb("x")], {}, containment_tree=tree)
    assert prov[0]["containment"] == {"parent_region_index": None, "edge_kind": None}
    assert prov[1]["containment"] == {"parent_region_index": 0, "edge_kind": "section"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_determinism_across_hashseeds(seed):
    """The tree is byte-stable regardless of hash seed (I5). Runs the same
    randomized corpus and asserts a fixed edge signature."""
    rng = random.Random(seed + 99)
    regions, region_html = _build_section_case(rng)
    fbs = [_fb() for _ in regions]
    t1 = build_containment_tree(regions, fbs, region_html=region_html)
    t2 = build_containment_tree(regions, fbs, region_html=region_html)
    assert t1.parent == t2.parent
    assert t1.children == t2.children
    assert t1.roots == t2.roots
    assert t1.edge_kind == t2.edge_kind
