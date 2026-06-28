"""Stage-5e block JOIN/SPLIT pass — unit tests (mocked runtimes, NO GPU).

Covers the R-PART invariant + the deterministic merge/split triggers + the
fail-closed reverts + the flag-off byte-stability + the LLM-layer
endpoint-down behaviour. No network, no model — the only runtime used is a
trivial in-test mock.
"""

from __future__ import annotations

import dataclasses

import pytest

from dart_semantic.qwen_specialists.block_resegment import (
    _UNIT_REGROUP_TOKEN_BUDGET,
    PartitionConservationError,
    ResegmentOp,
    _detect_unit_merges,
    apply_resegment,
    assert_partition_conservation,
    resegment_blocks,
    resolve_block_resegment_llm_mode,
    resolve_block_resegment_mode,
    resolve_unit_regroup_mode,
)
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
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
    fb_indices,
    *,
    text=None,
    source_region_id=None,
    pages=None,
    css_class=None,
    semantic_class=None,
):
    payload = {}
    if text is not None:
        payload["text"] = text
    if pages is not None:
        payload["pages"] = pages
    if css_class is not None:
        payload["css_class"] = css_class
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_indices),
        payload=payload,
        source_region_id=source_region_id,
    )


# --- a CouncilState-shaped mock for the MergeOrSplit head ----------------


class _MergeSignal:
    def __init__(self, region_id, label, conf):
        self.head_name = "same_logical_block"
        self.region_id = region_id
        self.top_k_labels = [label]
        self.top_k_confidences = [conf]


class _BertOut:
    def __init__(self, signals):
        self.signals = signals


class _State:
    """Minimal CouncilState stand-in carrying MergeOrSplit signals.

    ``signals`` maps the LEFT FB index of an adjacent pair -> (label, conf).
    """

    def __init__(self, signals=None):
        sigs = [
            _MergeSignal(left, label, conf)
            for left, (label, conf) in (signals or {}).items()
        ]
        self.outputs = {"merge_or_split": _BertOut(sigs)}


class _EmptyState:
    outputs = {}


class _DownRuntime:
    """A runtime whose generate_batch always raises (endpoint down)."""

    def generate_batch(self, prompts, **kwargs):
        raise RuntimeError("endpoint down")


class _LlmRuntime:
    """A runtime returning a crafted JSON ops payload."""

    def __init__(self, payload):
        self._payload = payload

    def generate_batch(self, prompts, *, max_tokens=1024, fail_soft=False, **kwargs):
        return [self._payload]


# ---------------------------------------------------------------------------
# Mode resolvers.
# ---------------------------------------------------------------------------


def test_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    assert resolve_block_resegment_mode() is False


def test_mode_on_when_truthy(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    assert resolve_block_resegment_mode() is True


def test_llm_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT_LLM", raising=False)
    assert resolve_block_resegment_llm_mode() is False


def test_mode_garbage_is_off(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "maybe")
    assert resolve_block_resegment_mode() is False


# ---------------------------------------------------------------------------
# MERGE — adjacent same-kind paragraph regions across a page break.
# ---------------------------------------------------------------------------


def test_merge_preserves_token_multiset_and_lowest_id():
    """Two adjacent same-kind paragraphs continued across a page merge into
    one whose FBs are the doc-order concat, with the lowest id preserved."""
    fbs = [_fb("the cat sat on the", page=1), _fb("warm mat by the fire", page=2)]
    regions = [
        _region("paragraph", [0], text="the cat sat on the", pages=[1]),
        _region("paragraph", [1], text="warm mat by the fire", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())

    assert len(out) == 1
    merged = out[0]
    assert merged.feature_block_indices == (0, 1)
    # Lowest source id preserved (min FB index).
    assert min(merged.feature_block_indices) == 0
    # Merged text is the doc-order joined source.
    assert merged.payload["text"] == "the cat sat on the warm mat by the fire"
    assert merged.payload["pages"] == [1, 2]
    # Breadcrumb stamped.
    assert merged.payload["resegment"]["op"] == "merge"
    assert merged.payload["resegment"]["conservation_verified"] is True
    assert [o.op for o in ops] == ["merge"]


def test_cross_page_paragraph_merge_with_no_merge_signal():
    """The deterministic cross-page cue (prev ends mid-sentence, next starts
    lowercase) fires with NO MergeOrSplit signal present."""
    fbs = [_fb("a sentence that runs", page=1), _fb("onto the next page", page=2)]
    regions = [
        _region("paragraph", [0], text="a sentence that runs", pages=[1]),
        _region("paragraph", [1], text="onto the next page", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1)


def test_no_merge_when_prev_ends_with_terminal_punct():
    """A prev region ending in a period is NOT a continuation -> no merge."""
    fbs = [_fb("a complete sentence.", page=1), _fb("another one entirely", page=2)]
    regions = [
        _region("paragraph", [0], text="a complete sentence.", pages=[1]),
        _region("paragraph", [1], text="another one entirely", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 2
    assert ops == []


def test_shattered_list_across_page_merges():
    """Adjacent list regions on different pages merge (shattered list)."""
    fbs = [_fb("first item", page=1), _fb("second item", page=2)]
    regions = [
        _region("list", [0], text="first item", pages=[1]),
        _region("list", [1], text="second item", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 1
    assert out[0].kind == "list"
    assert out[0].feature_block_indices == (0, 1)


def test_no_merge_across_kinds():
    """A heading + paragraph never merge (v1 same-kind-only scope cut)."""
    fbs = [_fb("Heading", page=1), _fb("body text", page=2)]
    regions = [
        _region("heading", [0], text="Heading", pages=[1]),
        _region("paragraph", [1], text="body text", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 2
    assert ops == []


def test_passthrough_region_never_merged():
    """A table/math passthrough (source_region_id set) is skipped."""
    fbs = [_fb("cell a", page=1), _fb("cell b", page=2)]
    regions = [
        _region("paragraph", [0], text="cell a", pages=[1], source_region_id=5),
        _region("paragraph", [1], text="cell b", pages=[2], source_region_id=6),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 2
    assert ops == []


def test_confident_not_same_vetoes_merge():
    """A confident 'not_same' boundary signal blocks a cross-page merge."""
    fbs = [_fb("runs onward", page=1), _fb("but distinct here", page=2)]
    regions = [
        _region("paragraph", [0], text="runs onward", pages=[1]),
        _region("paragraph", [1], text="but distinct here", pages=[2]),
    ]
    # FB pair (0,1): left index 0 -> not_same.
    state = _State({0: ("not_same", 0.9)})
    out, ops = resegment_blocks(regions, fbs, state)
    assert len(out) == 2
    assert ops == []


# ---------------------------------------------------------------------------
# SPLIT — at an FB boundary.
# ---------------------------------------------------------------------------


def test_split_at_pedagogical_label_boundary():
    """A region whose 2nd FB starts a pedagogical label splits before it; K
    children get disjoint contiguous FB slices and child 0 keeps the parent id."""
    fbs = [_fb("intro prose here"), _fb("EXAMPLE 2 demonstrates this")]
    regions = [_region("paragraph", [0, 1], text="intro prose here EXAMPLE 2 demonstrates this")]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())

    assert len(out) == 2
    assert out[0].feature_block_indices == (0,)
    assert out[1].feature_block_indices == (1,)
    # Disjoint + contiguous.
    assert set(out[0].feature_block_indices).isdisjoint(out[1].feature_block_indices)
    # Child 0 keeps the parent's id (lowest FB index 0).
    assert min(out[0].feature_block_indices) == 0
    # Child texts are the verbatim slices.
    assert out[0].payload["text"] == "intro prose here"
    assert out[1].payload["text"] == "EXAMPLE 2 demonstrates this"
    assert out[1].payload["resegment"]["op"] == "split"
    assert [o.op for o in ops] == ["split"]


def test_split_at_confident_not_same_boundary():
    """An interior boundary where MergeOrSplit confidently says not_same splits."""
    fbs = [_fb("one unit of text"), _fb("a different unit here")]
    regions = [_region("paragraph", [0, 1], text="one unit of text a different unit here")]
    # FB pair (0,1): left index 0 -> not_same.
    state = _State({0: ("not_same", 0.85)})
    out, ops = resegment_blocks(regions, fbs, state)
    assert len(out) == 2
    assert out[0].feature_block_indices == (0,)
    assert out[1].feature_block_indices == (1,)


def test_split_into_three_children():
    """Two interior pedagogical boundaries yield three contiguous children."""
    fbs = [_fb("intro"), _fb("EXAMPLE 1 first"), _fb("Solution to it")]
    regions = [_region("paragraph", [0, 1, 2], text="intro EXAMPLE 1 first Solution to it")]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert [r.feature_block_indices for r in out] == [(0,), (1,), (2,)]


def test_no_split_on_single_fb_region():
    """A 1-FB region can never split."""
    fbs = [_fb("EXAMPLE 1 alone")]
    regions = [_region("paragraph", [0], text="EXAMPLE 1 alone")]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 1
    assert ops == []


# ---------------------------------------------------------------------------
# R-PART conservation + fail-closed revert.
# ---------------------------------------------------------------------------


def test_partition_conservation_catches_dropped_fb():
    """An op that DROPS an FB raises PartitionConservationError."""
    in_regions = [_region("paragraph", [0, 1])]
    out_regions = [_region("paragraph", [0])]  # FB 1 dropped
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(in_regions, out_regions)


def test_partition_conservation_catches_duplicated_fb():
    """An op that DUPLICATES an FB raises PartitionConservationError."""
    in_regions = [_region("paragraph", [0, 1])]
    out_regions = [_region("paragraph", [0]), _region("paragraph", [0, 1])]
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(in_regions, out_regions)


def test_partition_conservation_catches_reorder():
    """An op that RE-ORDERS FBs raises PartitionConservationError."""
    in_regions = [_region("paragraph", [0, 1])]
    out_regions = [_region("paragraph", [1, 0])]
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(in_regions, out_regions)


def test_partition_conservation_passes_on_pure_repartition():
    """A merge then a split that re-partition the same FBs passes."""
    in_regions = [_region("paragraph", [0]), _region("paragraph", [1])]
    out_merged = [_region("paragraph", [0, 1])]
    assert_partition_conservation(in_regions, out_merged)  # no raise


def test_dropping_op_fails_closed_to_input(monkeypatch):
    """A synthetic op that drops an FB -> resegment_blocks reverts to input.

    We monkeypatch apply_resegment to return a content-losing list; the driver
    must catch PartitionConservationError and return the INPUT unchanged."""
    import dart_semantic.qwen_specialists.block_resegment as mod

    fbs = [_fb("EXAMPLE 1 intro"), _fb("EXAMPLE 2 more")]
    regions = [_region("paragraph", [0, 1], text="EXAMPLE 1 intro EXAMPLE 2 more")]

    def _bad_apply(rs, fb, ops):
        # Drop FB 1 entirely (content loss).
        return [dataclasses.replace(regions[0], feature_block_indices=(0,))]

    monkeypatch.setattr(mod, "apply_resegment", _bad_apply)
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    # Fail-closed: input returned unchanged, empty op list.
    assert out == regions
    assert ops == []


def test_token_conservation_orphan_reverts(monkeypatch):
    """An op set that orphans a real source FB (drops its tokens from coverage)
    reverts. ``assert_token_conservation`` raises on the orphaned FB; the
    driver fails closed to the input list."""
    import dart_semantic.qwen_specialists.block_resegment as mod

    fbs = [_fb("EXAMPLE 1 intro"), _fb("EXAMPLE 2 more")]
    regions = [_region("paragraph", [0, 1], text="EXAMPLE 1 intro EXAMPLE 2 more")]

    def _bad_apply(rs, fb, ops):
        # Return a region claiming FB 0 twice: multiset is {0:2}, source is
        # {0:1, 1:1}. FB 1's tokens are orphaned (never owned) AND FB 0 is
        # duplicated -> token conservation flags the orphan / R-PART flags the
        # multiset; either way the driver reverts.
        return [dataclasses.replace(regions[0], feature_block_indices=(0, 0))]

    monkeypatch.setattr(mod, "apply_resegment", _bad_apply)
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert out == regions
    assert ops == []


# ---------------------------------------------------------------------------
# Flag-OFF byte-identical pass-through (no triggers).
# ---------------------------------------------------------------------------


def test_no_triggers_is_pass_through():
    """With no merge/split triggers, the region list passes through unchanged."""
    fbs = [_fb("a complete sentence.", page=1), _fb("Another complete one.", page=1)]
    regions = [
        _region("paragraph", [0], text="a complete sentence.", pages=[1]),
        _region("paragraph", [1], text="Another complete one.", pages=[1]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert out == regions
    assert ops == []


def test_empty_regions_pass_through():
    out, ops = resegment_blocks([], [], _EmptyState())
    assert out == []
    assert ops == []


# ---------------------------------------------------------------------------
# LLM layer.
# ---------------------------------------------------------------------------


def test_llm_endpoint_down_keeps_deterministic_result(monkeypatch):
    """An endpoint-down LLM runtime keeps the deterministic merge result."""
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT_LLM", "on")
    fbs = [_fb("runs onto", page=1), _fb("the next page", page=2)]
    regions = [
        _region("paragraph", [0], text="runs onto", pages=[1]),
        _region("paragraph", [1], text="the next page", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=_DownRuntime())
    # Deterministic cross-page merge still happened; LLM proposed nothing.
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1)


def test_llm_proposes_extra_merge(monkeypatch):
    """A valid LLM merge op is applied (re-validated by the gates)."""
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT_LLM", "on")
    # No deterministic trigger (same page, terminal punct), but the LLM
    # proposes a merge of regions 0 and 1.
    fbs = [_fb("part one.", page=1), _fb("part two.", page=1)]
    regions = [
        _region("paragraph", [0], text="part one.", pages=[1]),
        _region("paragraph", [1], text="part two.", pages=[1]),
    ]
    payload = '{"ops":[{"op":"merge","region_indices":[0,1]}]}'
    out, ops = resegment_blocks(
        regions, fbs, _EmptyState(), runtime=_LlmRuntime(payload)
    )
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1)
    assert any(o.origin == "llm" for o in ops)


def test_llm_malformed_payload_keeps_deterministic(monkeypatch):
    """A malformed LLM payload is dropped; deterministic result stands."""
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT_LLM", "on")
    fbs = [_fb("a complete sentence.", page=1), _fb("Another complete one.", page=1)]
    regions = [
        _region("paragraph", [0], text="a complete sentence.", pages=[1]),
        _region("paragraph", [1], text="Another complete one.", pages=[1]),
    ]
    out, ops = resegment_blocks(
        regions, fbs, _EmptyState(), runtime=_LlmRuntime("not json at all")
    )
    # No deterministic trigger + unparseable LLM -> pass-through.
    assert out == regions
    assert ops == []


# ---------------------------------------------------------------------------
# apply_resegment directly — overlapping op guard.
# ---------------------------------------------------------------------------


def test_apply_skips_overlapping_ops():
    """A split op on a region a merge op already consumed is skipped."""
    fbs = [_fb("aa"), _fb("bb"), _fb("cc")]
    regions = [
        _region("paragraph", [0]),
        _region("paragraph", [1]),
        _region("paragraph", [2]),
    ]
    # Merge regions 0,1; a (conflicting) split on region 0 must be ignored.
    ops = [
        ResegmentOp(op="merge", region_indices=(0, 1)),
        ResegmentOp(op="split", region_indices=(0,), split_at=(1,)),
    ]
    out = apply_resegment(regions, fbs, ops)
    # Region 0+1 merged, region 2 untouched -> 2 output regions.
    assert len(out) == 2
    assert out[0].feature_block_indices == (0, 1)


# ---------------------------------------------------------------------------
# Phase 0 — SEMANTIK_UNIT_REGROUP resolver (mirrors the resegment resolver tests).
# ---------------------------------------------------------------------------


def test_unit_regroup_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    assert resolve_unit_regroup_mode() is False
    for v in ("", "  "):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is False


def test_unit_regroup_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "yes", "on", "  On  ", "YES"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is True


def test_unit_regroup_mode_garbage_is_off(monkeypatch):
    for v in ("banana", "0", "off", "false", "no", "maybe"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is False


# ---------------------------------------------------------------------------
# Phase 2 — pure region-index-keyed unit-merge detector (UNCALLED, no behavior).
# ---------------------------------------------------------------------------


def test_detect_unit_merge_label_plus_body():
    """label(pedagogy-example) + problem + Solution + step -> ONE regroup op
    spanning (0,1,2,3), anchored at index 0."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor"), _fb("Step 1")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
        _region("paragraph", [3], text="Step 1", css_class="pedagogy-step"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "merge"
    assert op.subtype == "regroup"
    assert op.region_indices == (0, 1, 2, 3)
    assert op.region_indices[0] == 0


def test_detect_stops_at_next_unit_start():
    """A SECOND pedagogy-example after the body is a boundary — the run stops
    before it."""
    fbs = [_fb("EXAMPLE 1"), _fb("body one"), _fb("EXAMPLE 2"), _fb("body two")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="body one"),
        _region("paragraph", [2], text="EXAMPLE 2", css_class="pedagogy-example"),
        _region("paragraph", [3], text="body two"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    # First unit = (0,1); the second example starts its own run but has only
    # its own body adjacency to absorb -> (2,3).
    assert ops[0].region_indices == (0, 1)
    assert all(idx not in ops[0].region_indices for idx in (2, 3))


def test_detect_stops_at_heading():
    """A heading region is a hard boundary."""
    fbs = [_fb("EXAMPLE 1"), _fb("body"), _fb("Next Section"), _fb("after")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="body"),
        _region("heading", [2], text="Next Section"),
        _region("paragraph", [3], text="after"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    assert ops[0].region_indices == (0, 1)


def test_detect_caps_at_absorb_max_run():
    """A 12-region body absorbs at most ABSORB_MAX_RUN (8) followers."""
    from dart_semantic.pedagogical_units import ABSORB_MAX_RUN

    n_body = 12
    fbs = [_fb("EXAMPLE 1")] + [_fb(f"p{i}") for i in range(1, 1 + n_body)]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example")
    ] + [_region("paragraph", [i], text=f"p{i}") for i in range(1, 1 + n_body)]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    # The run is the anchor + up to (ABSORB_MAX_RUN - 1) followers (j - i <
    # ABSORB_MAX_RUN), so the run length is ABSORB_MAX_RUN.
    assert len(ops[0].region_indices) == ABSORB_MAX_RUN
    assert ops[0].region_indices[0] == 0


def test_detect_caps_at_token_budget():
    """A run whose summed FB text would exceed the token budget caps early."""
    # Each body region ~ (budget // 2) + 1 words so the THIRD body would overflow.
    big = " ".join(["w"] * ((_UNIT_REGROUP_TOKEN_BUDGET // 2) + 1))
    fbs = [_fb("EXAMPLE 1"), _fb(big), _fb(big), _fb(big)]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text=big),
        _region("paragraph", [2], text=big),
        _region("paragraph", [3], text=big),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    # anchor(small) + first big fits (~budget/2); the second big overflows ->
    # run stops at (0, 1).
    assert ops[0].region_indices == (0, 1)


def test_detect_run_of_one_no_op():
    """A lone anchor with an immediate boundary emits no op."""
    fbs = [_fb("EXAMPLE 1"), _fb("Next")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("heading", [1], text="Next"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert ops == []


def test_detect_requires_fb_adjacency():
    """A non-contiguous (kind-segregated) list yields no spurious run."""
    fbs = [_fb(f"t{i}") for i in range(6)]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        # The body FB index is NOT prev_last+1 (the un-reordered, segregated
        # case) -> adjacency fails -> no absorption.
        _region("paragraph", [5], text="far away body"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert ops == []


def test_detect_anchor_is_index_zero():
    """Every emitted op's first index is the unit anchor (semantic_class anchor
    here), even when it is not region 0 of the document."""
    fbs = [_fb("Intro"), _fb("EXAMPLE"), _fb("body a"), _fb("body b")]
    regions = [
        _region("heading", [0], text="Intro"),
        _region("paragraph", [1], text="EXAMPLE", semantic_class="worked_example"),
        _region("paragraph", [2], text="body a"),
        _region("paragraph", [3], text="body b"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    first_idx = ops[0].region_indices[0]
    assert first_idx == 1
    anchor = regions[first_idx]
    assert anchor.payload.get("semantic_class") == "worked_example"
