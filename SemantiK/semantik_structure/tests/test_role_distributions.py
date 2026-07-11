"""ITEM6 Phase 1 — persist council top-k role distributions on Region
provenance + region_provenance.

CPU-only: every ``CouncilState`` is synthetic ``TypedSignal`` dataclasses
(mirroring ``test_content_type_guard.py``); no GPU, no model download. Pins the
additive, idempotent ``stamp_role_distributions`` helper + its end-to-end
plumbing through ``build_structure_graph`` and the ``_build_region_provenance``
wire lift.
"""

from __future__ import annotations

from dataclasses import replace

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.types import BertOutput, CouncilState, TypedSignal
from semantik_structure.structure_graph import (
    Region,
    build_structure_graph,
    stamp_role_distributions,
)
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures — synthetic FeatureBlocks + CouncilState.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
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


def _state_with_role(idx: int, labels, confs, *, ped=None) -> CouncilState:
    """A CouncilState whose ``structure`` BERT emits a ``structural_role``
    signal (and optionally a ``pedagogical_role`` signal) for FB ``idx``."""
    sigs = [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, list(labels), list(confs)),
    ]
    if ped is not None:
        ped_labels, ped_confs = ped
        sigs.append(
            TypedSignal("pedagogical_role", idx, list(ped_labels), list(ped_confs))
        )
    return CouncilState(outputs={"structure": BertOutput("structure", sigs)})


def _region(fbs=(0,), *, kind="paragraph", source_region_id=None, provenance=None):
    return Region(
        kind=kind,
        feature_block_indices=tuple(fbs),
        payload={"text": "x"},
        provenance=dict(provenance or {}),
        source_region_id=source_region_id,
    )


# ---------------------------------------------------------------------------
# stamp_role_distributions — unit behaviour.
# ---------------------------------------------------------------------------


def test_stamp_present_k3_rounded():
    # 5-class distribution; only top-3 persisted, probs at 4dp.
    labels = ["code_block", "paragraph", "list_item", "blockquote", "heading"]
    confs = [0.51234, 0.30111, 0.09214, 0.05, 0.04441]
    state = _state_with_role(0, labels, confs)
    out = stamp_role_distributions([_region((0,))], state)
    top_k = out[0].provenance["role_top_k"]
    assert len(top_k) == 3
    assert top_k == [
        ["code_block", 0.5123],
        ["paragraph", 0.3011],
        ["list_item", 0.0921],
    ]


def test_stamp_pedagogical_top_k_when_head_present():
    state = _state_with_role(
        0,
        ["paragraph", "code_block"],
        [0.8, 0.2],
        ped=(["worked_example", "none"], [0.7, 0.3]),
    )
    out = stamp_role_distributions([_region((0,))], state)
    assert out[0].provenance["role_top_k"] == [["paragraph", 0.8], ["code_block", 0.2]]
    assert out[0].provenance["pedagogical_top_k"] == [
        ["worked_example", 0.7],
        ["none", 0.3],
    ]


def test_stamp_absent_without_signal():
    # Mock/legacy state (no structure signals) -> region object UNCHANGED
    # (is-identity preserved for unstamped regions).
    state = CouncilState(outputs={})
    reg = _region((0,))
    out = stamp_role_distributions([reg], state)
    assert out[0] is reg
    assert "role_top_k" not in out[0].provenance


def test_stamp_skips_passthrough_and_empty_fb():
    state = _state_with_role(0, ["paragraph", "code_block"], [0.8, 0.2])
    passthrough = _region((0,), kind="table", source_region_id=7)
    empty_fb = _region(())
    out = stamp_role_distributions([passthrough, empty_fb], state)
    assert out[0] is passthrough
    assert out[1] is empty_fb
    assert "role_top_k" not in out[0].provenance
    assert "role_top_k" not in out[1].provenance


def test_stamp_pure_additive_and_idempotent():
    state = _state_with_role(0, ["paragraph", "code_block"], [0.8, 0.2])
    reg = _region((0,), kind="list", provenance={"pass": "list"})
    reg = replace(reg, payload={"text": "x", "items": [1, 2]}, aria_hints=("nav",))
    once = stamp_role_distributions([reg], state)[0]
    # Everything except provenance byte-identical.
    assert once.kind == reg.kind
    assert once.feature_block_indices == reg.feature_block_indices
    assert once.payload == reg.payload
    assert once.aria_hints == reg.aria_hints
    assert once.source_region_id == reg.source_region_id
    # Pre-existing provenance keys preserved; role_top_k added.
    assert once.provenance["pass"] == "list"
    assert once.provenance["role_top_k"] == [["paragraph", 0.8], ["code_block", 0.2]]
    # Idempotent: double-stamp yields identical provenance.
    twice = stamp_role_distributions([once], state)[0]
    assert twice.provenance == once.provenance


def test_stamp_uses_min_fb_representative():
    # Signal only on FB 3 (the min of a merged tuple); rep = min(indices).
    state = _state_with_role(3, ["math", "paragraph"], [0.6, 0.4])
    merged = _region((3, 5, 9))
    out = stamp_role_distributions([merged], state)
    assert out[0].provenance["role_top_k"] == [["math", 0.6], ["paragraph", 0.4]]


def test_restamp_after_merge_uses_label_min_fb():
    # A _merged_region-shaped Region (fresh empty provenance, merged FB tuple).
    state = _state_with_role(2, ["paragraph", "list_item"], [0.55, 0.45])
    merged = _region((2, 3, 4), provenance={})
    out = stamp_role_distributions([merged], state)
    assert out[0].provenance["role_top_k"] == [["paragraph", 0.55], ["list_item", 0.45]]


# ---------------------------------------------------------------------------
# End-to-end through build_structure_graph + the wire lift.
# ---------------------------------------------------------------------------


def _role_signals(idx: int, role: str) -> list[TypedSignal]:
    return [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, [role, "paragraph", "list_item"], [0.7, 0.2, 0.1]),
    ]


def _build_graph_inputs():
    fbs = [_fb("The quick brown fox jumps over the lazy dog."),
           _fb("A second paragraph of ordinary running prose here.")]
    sigs: list[TypedSignal] = []
    sigs += _role_signals(0, "paragraph")
    sigs += _role_signals(1, "paragraph")
    state = CouncilState(outputs={"structure": BertOutput("structure", sigs)})
    decisions = arbitrate(state, [])
    return state, fbs, [], decisions


def test_build_structure_graph_regions_carry_role_top_k():
    state, fbs, cands, decs = _build_graph_inputs()
    regions = build_structure_graph(state, fbs, cands, decs)
    # Every non-passthrough region carries the additive key.
    stamped = [r for r in regions if r.source_region_id is None and r.feature_block_indices]
    assert stamped, "no stampable regions produced"
    for r in stamped:
        assert "role_top_k" in r.provenance
        assert len(r.provenance["role_top_k"]) == 3


def test_build_structure_graph_kinds_order_unchanged_by_stamp():
    """No-behavior-change pin: kinds + FB order identical to a manual
    pre-stamp reconstruction (the stamp only adds provenance)."""
    state, fbs, cands, decs = _build_graph_inputs()
    regions = build_structure_graph(state, fbs, cands, decs)
    # Kind + FB tuples are the load-bearing structure; the two prose FBs fuse
    # into one continuation-merged paragraph region. The stamp is post-coverage
    # + provenance-only, so this shape is invariant to it — pin it, and confirm
    # every FB is still covered exactly once.
    shape = [(r.kind, r.feature_block_indices) for r in regions]
    assert shape == [("paragraph", (0, 1))]
    owned = sorted(j for r in regions for j in r.feature_block_indices)
    assert owned == [0, 1]


def test_region_provenance_lifts_role_top_k_additively():
    from semantik_structure.cascade import _build_region_provenance

    state, fbs, cands, decs = _build_graph_inputs()
    regions = build_structure_graph(state, fbs, cands, decs)
    prov = _build_region_provenance(
        list(range(len(regions))), regions, fbs, {}
    )
    assert any("role_top_k" in e for e in prov)

    # Unstamped regions (mock state) -> key ABSENT (byte-stable baseline).
    bare = [replace(r, provenance={}) for r in regions]
    prov_bare = _build_region_provenance(
        list(range(len(bare))), bare, fbs, {}
    )
    assert all("role_top_k" not in e for e in prov_bare)
