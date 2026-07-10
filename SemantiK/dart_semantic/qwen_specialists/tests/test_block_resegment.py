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
    _detect_splits,
    _detect_unit_merges,
    _is_passthrough,
    apply_proposed_regroups,
    apply_resegment,
    assert_partition_conservation,
    build_resegment_audit_rows,
    is_passthrough_region,
    resegment_blocks,
    resolve_block_resegment_llm_mode,
    resolve_block_resegment_mode,
    resolve_unit_regroup_mode,
    resolve_unit_regroup_table_mode,
)
from dart_semantic.structure_graph import (
    Region,
    compute_region_page_bboxes,
    stamp_region_page_bboxes,
)
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


@pytest.fixture(autouse=True)
def block_resegment_on_by_default(monkeypatch):
    """Default the SAME-KIND Stage-5e pass ON for this module.

    Phase 6 moved the per-flag gating INSIDE ``resegment_blocks`` —
    ``_detect_merges`` / ``_detect_splits`` now run ONLY when
    ``SEMANTIK_BLOCK_RESEGMENT`` is on. The same-kind merge/split unit tests
    were written for the pre-gating world where those detectors ran
    unconditionally, so default the flag ON here. Tests that exercise the gate
    itself (orthogonality / flag-off byte-stability) explicitly
    ``monkeypatch.delenv`` it in-body to override this (same monkeypatch
    instance → in-body wins)."""
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")


@pytest.fixture
def unit_regroup_on(monkeypatch):
    """Enable SEMANTIK_UNIT_REGROUP for a detector-direct test.

    The Phase-3 self-gate makes ``_detect_unit_merges`` return ``[]`` when the
    flag is off, so any test that calls the detector directly and expects ops
    requests this fixture."""
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")


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


def test_unit_regroup_mode_on_by_default(monkeypatch):
    # ITEM1: default-ON falsey-set (mirrors resolve_reading_order_fix). Unset /
    # blank -> on.
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    assert resolve_unit_regroup_mode() is True
    for v in ("", "  "):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is True


def test_unit_regroup_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "yes", "on", "  On  ", "YES"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is True


def test_unit_regroup_mode_garbage_is_on_only_falsey_reverts(monkeypatch):
    # Garbage is ON (default-ON parse-with-fallback); ONLY an explicit falsey
    # token reverts to the byte-identical legacy partition.
    for v in ("banana", "maybe"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is True
    for v in ("0", "off", "false", "no"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is False


# ---------------------------------------------------------------------------
# table-v2 Phase 0 — SEMANTIK_UNIT_REGROUP_TABLE sub-flag resolver.
# ---------------------------------------------------------------------------


def test_unit_regroup_table_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP_TABLE", raising=False)
    assert resolve_unit_regroup_table_mode() is False
    for v in ("", "  "):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", v)
        assert resolve_unit_regroup_table_mode() is False


def test_unit_regroup_table_mode_truthy_tokens(monkeypatch):
    for v in ("1", "true", "yes", "on", "  On  ", "YES"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", v)
        assert resolve_unit_regroup_table_mode() is True


def test_unit_regroup_table_mode_garbage_is_off(monkeypatch):
    for v in ("banana", "0", "off", "false", "no", "maybe"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", v)
        assert resolve_unit_regroup_table_mode() is False


def test_is_passthrough_region_aliases_single_source():
    """``block_resegment._is_passthrough`` IS the neutral
    ``is_passthrough_region`` (single-sourced; cycle-free)."""
    assert _is_passthrough is is_passthrough_region
    none_region = Region(kind="paragraph", feature_block_indices=(0,), payload={})
    table_region = Region(
        kind="table", feature_block_indices=(1,), payload={}, source_region_id=3
    )
    assert is_passthrough_region(none_region) is False
    assert is_passthrough_region(table_region) is True


# ---------------------------------------------------------------------------
# Phase 2 — pure region-index-keyed unit-merge detector (UNCALLED, no behavior).
# ---------------------------------------------------------------------------


def test_detect_unit_merge_label_plus_body(unit_regroup_on):
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


def test_detect_stops_at_next_unit_start(unit_regroup_on):
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


def test_detect_stops_at_uncss_stamped_example(unit_regroup_on):
    """OVER-MERGE FIX: a following EXAMPLE region with NO css_class (the clean
    pass only stamps a re-tagged heading, so a council-typed paragraph label
    slips through unstamped) is a boundary via the text-pattern arm, so the
    anchor TRY-IT unit's run STOPS before it. Pre-fix the unstamped EXAMPLE
    was absorbed (two examples fused into one box)."""
    fbs = [_fb("TRY IT 1.1"), _fb("body one"), _fb("EXAMPLE 1.114"), _fb("body two")]
    regions = [
        _region("paragraph", [0], text="TRY IT 1.1", css_class="pedagogy-try-it"),
        _region("paragraph", [1], text="body one"),
        # The over-merge bug: a real EXAMPLE label region whose css_class was
        # NEVER stamped (no css_class set). The text-pattern arm catches it.
        _region("paragraph", [2], text="EXAMPLE 1.114 Simplify"),
        _region("paragraph", [3], text="body two"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    # The TRY-IT run is (0,1) and STOPS before the unstamped EXAMPLE at 2.
    assert ops[0].region_indices == (0, 1)
    assert all(idx not in ops[0].region_indices for idx in (2, 3))


def test_detect_stops_at_heading(unit_regroup_on):
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


def test_detect_caps_at_absorb_max_run(unit_regroup_on):
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


def test_detect_caps_at_token_budget(unit_regroup_on):
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


def test_detect_run_of_one_no_op(unit_regroup_on):
    """A lone anchor with an immediate boundary emits no op."""
    fbs = [_fb("EXAMPLE 1"), _fb("Next")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("heading", [1], text="Next"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert ops == []


def test_detect_requires_fb_adjacency(unit_regroup_on):
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


def test_detect_anchor_is_index_zero(unit_regroup_on):
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


# ---------------------------------------------------------------------------
# Phase 3 — cross-kind unit-MERGE op (subtype='regroup') + semantic_class carry.
# ---------------------------------------------------------------------------


def _unit_regions():
    """label(pedagogy-example) + problem + Solution + step -> a 4-region unit."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor"), _fb("Step 1")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
        _region("paragraph", [3], text="Step 1", css_class="pedagogy-step"),
    ]
    return regions, fbs


def test_unit_merge_fuses_run_into_one_region(unit_regroup_on):
    """Applying the regroup op fuses the 4-region unit into ONE region whose
    FBs are the doc-order concat and whose pages are unioned."""
    regions, fbs = _unit_regions()
    # Give each region a page so the union is observable.
    regions = [dataclasses.replace(r, payload={**r.payload, "pages": [i + 1]})
               for i, r in enumerate(regions)]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    out = apply_resegment(regions, fbs, ops)
    assert len(out) == len(regions) - 3  # 4 -> 1
    merged = out[0]
    assert merged.feature_block_indices == (0, 1, 2, 3)  # doc-order concat
    assert merged.payload["pages"] == [1, 2, 3, 4]  # unioned


def test_merged_region_carries_label_semantic_class(unit_regroup_on):
    """A label region already carrying semantic_class='worked_example' (reviewer
    ran) -> the merged region carries it (anchor payload inherited for free)."""
    fbs = [_fb("EXAMPLE 1"), _fb("body a"), _fb("body b")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1",
                css_class="pedagogy-example", semantic_class="worked_example"),
        _region("paragraph", [1], text="body a"),
        _region("paragraph", [2], text="body b"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    out = apply_resegment(regions, fbs, ops)
    assert out[0].payload.get("semantic_class") == "worked_example"


def test_deterministic_semantic_class_from_css_class(unit_regroup_on):
    """Reviewer OFF: a label with ONLY css_class='pedagogy-example' gains
    semantic_class='worked_example' via component_for_pedagogy_class."""
    fbs = [_fb("EXAMPLE 1"), _fb("body a"), _fb("body b")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="body a"),
        _region("paragraph", [2], text="body b"),
    ]
    assert regions[0].payload.get("semantic_class") is None  # reviewer off
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    out = apply_resegment(regions, fbs, ops)
    assert out[0].payload.get("semantic_class") == "worked_example"


def test_unknown_css_class_no_fabricated_component():
    """A regroup merge whose anchor css_class has no catalog mapping (and no
    prior semantic_class) gets NO semantic_class stamped — anti-fabrication,
    never invents a component. Exercised directly on _merged_region (the carry
    site) with an unmapped css that component_for_pedagogy_class returns None
    for."""
    from dart_semantic.qwen_specialists.block_resegment import _merged_region

    fbs = [_fb("EXAMPLE 1"), _fb("body a")]
    anchor = _region("paragraph", [0], text="EXAMPLE 1",
                     css_class="pedagogy-bogus-unmapped")
    body = _region("paragraph", [1], text="body a")
    merged = _merged_region([anchor, body], fbs, subtype="regroup")
    assert merged.payload.get("semantic_class") is None  # never invented
    # The breadcrumb still records the regroup (the carry just had nothing to
    # derive).
    assert merged.payload["resegment"]["op"] == "regroup"


def test_merged_region_breadcrumb_op_regroup(unit_regroup_on):
    """A regroup-subtype merge stamps breadcrumb op='regroup'; a same-kind
    merge keeps op='merge'."""
    regions, fbs = _unit_regions()
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    out = apply_resegment(regions, fbs, ops)
    assert out[0].payload["resegment"]["op"] == "regroup"

    # Same-kind merge keeps 'merge'.
    fbs2 = [_fb("the cat sat on the", page=1), _fb("warm mat by the fire", page=2)]
    regions2 = [
        _region("paragraph", [0], text="the cat sat on the", pages=[1]),
        _region("paragraph", [1], text="warm mat by the fire", pages=[2]),
    ]
    out2, _ = resegment_blocks(regions2, fbs2, _EmptyState())
    assert out2[0].payload["resegment"]["op"] == "merge"


def test_subtype_default_merge_byte_stable():
    """A same-kind _detect_merges op has subtype=='merge' (the default) and its
    audit-shape fields are byte-identical to the pre-field op."""
    op = ResegmentOp(op="merge", region_indices=(0, 1), source_ids=(0, 1))
    assert op.subtype == "merge"
    # The serialized audit shape the cascade builds reads op/source_ids/origin —
    # all byte-identical to a no-subtype op.
    assert op.op == "merge"
    assert op.source_ids == (0, 1)
    assert op.origin == "deterministic"


def test_anchor_first_invariant_asserts():
    """A hand-built body-first regroup run raises at construction; a same-kind
    merge with a non-anchor run[0] does NOT raise."""
    from dart_semantic.qwen_specialists.block_resegment import (
        _make_regroup_op,
        _merged_region,
    )
    # A body-first run (run[0] is a plain paragraph, not a unit anchor) raises.
    fbs = [_fb("plain body"), _fb("EXAMPLE 1")]
    body_first = [
        _region("paragraph", [0], text="plain body"),
        _region("paragraph", [1], text="EXAMPLE 1", css_class="pedagogy-example"),
    ]
    with pytest.raises(AssertionError):
        _make_regroup_op([0, 1], body_first)

    # A SAME-KIND merge through the shared _merged_region with a non-anchor
    # run[0] does NOT raise (the assert is scoped to op construction).
    same_kind = [
        _region("paragraph", [0], text="plain a"),
        _region("paragraph", [1], text="plain b"),
    ]
    merged = _merged_region(same_kind, fbs, subtype="merge")
    assert merged.feature_block_indices == (0, 1)  # no raise


def test_rpart_and_token_conservation_hold_cross_kind(unit_regroup_on):
    """assert_partition_conservation + assert_token_conservation pass on the
    cross-kind merge output."""
    from dart_semantic.qwen_specialists.reviewer import assert_token_conservation

    regions, fbs = _unit_regions()
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    out = apply_resegment(regions, fbs, ops)
    assert_partition_conservation(regions, out)  # R-PART
    assert_token_conservation(regions, out, fbs)  # C3 token conservation


# ---------------------------------------------------------------------------
# Phase 4 — passthrough (table/math) is a unit BOUNDARY (v1, no orphaning).
# ---------------------------------------------------------------------------


def test_regroup_stops_at_passthrough_table(unit_regroup_on):
    """label + problem + Solution + table(source_region_id set) -> the merged
    region is label+problem+Solution ONLY; the table stays a SEPARATE region
    with its source_region_id intact."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor"), _fb("grid")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
        _region("table", [3], text="grid", source_region_id="tbl-7"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    assert ops[0].region_indices == (0, 1, 2)  # table excluded
    out = apply_resegment(regions, fbs, ops)
    assert len(out) == 2  # merged unit + the standalone table
    merged, table = out[0], out[1]
    assert merged.feature_block_indices == (0, 1, 2)
    assert table.source_region_id == "tbl-7"  # Stage-4 link intact
    assert table.feature_block_indices == (3,)


def test_passthrough_link_never_orphaned(unit_regroup_on):
    """No merged region carries a fused-away source_region_id — the merged
    unit's source_region_id == the label's (unchanged, None here)."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("grid")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("table", [2], text="grid", source_region_id="tbl-9"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    out = apply_resegment(regions, fbs, ops)
    merged = out[0]
    assert merged.source_region_id is None  # label's, unchanged — not fused away
    # The table survives standalone with its link.
    assert any(r.source_region_id == "tbl-9" for r in out)


def test_math_passthrough_also_boundary(unit_regroup_on):
    """A math passthrough (source_region_id set) inside a body is a boundary,
    parity with the table case."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("x^2")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("math", [2], text="x^2", source_region_id="math-3"),
    ]
    ops = _detect_unit_merges(regions, fbs, _EmptyState())
    assert len(ops) == 1
    assert ops[0].region_indices == (0, 1)  # math excluded


# ---------------------------------------------------------------------------
# Phase 6 — driver wiring: per-flag gating, reading-order guard,
# regroup-FIRST precedence, fail-closed.
# ---------------------------------------------------------------------------


def test_regroup_fires_through_driver(monkeypatch):
    """SEMANTIK_UNIT_REGROUP on (+ reading-order on, the default) + a contiguous
    label->body unit -> resegment_blocks returns the merged list + the regroup
    op (the detector is wired into the driver)."""
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)  # default on
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1, 2)
    assert [o.op for o in ops] == ["merge"]
    assert ops[0].subtype == "regroup"
    assert ops[0].region_indices == (0, 1, 2)
    # The merged unit carries the label's gold semantic_class.
    assert out[0].payload["semantic_class"] == "worked_example"


def test_regroup_only_emits_no_same_kind_ops(monkeypatch):
    """SEMANTIK_UNIT_REGROUP on + SEMANTIK_BLOCK_RESEGMENT OFF -> ZERO same-kind
    merge/split ops (orthogonality): the same input that WOULD trigger a
    cross-page same-kind merge under block-resegment yields no such op here."""
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    # A pure same-kind cross-page continuation (a _detect_merges candidate) with
    # NO unit anchor -> under regroup-only it must NOT merge.
    fbs = [_fb("a clause that runs", page=1), _fb("over the page edge", page=2)]
    regions = [
        _region("paragraph", [0], text="a clause that runs", pages=[1]),
        _region("paragraph", [1], text="over the page edge", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    # No anchor -> no regroup; block-resegment off -> no same-kind merge.
    assert out == regions
    assert ops == []
    # Directly: the same-kind detectors are never consulted -> no op of any kind.
    assert all(o.subtype != "merge" for o in ops)


def test_regroup_noop_when_reading_order_off(monkeypatch):
    """resolve_reading_order_fix monkeypatched False + regroup on -> the driver
    guard suppresses the regroup entirely; input byte-identical."""
    import dart_semantic.qwen_specialists.block_resegment as mod

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setattr(mod, "resolve_reading_order_fix", lambda: False)
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert out == regions
    assert ops == []


def test_bad_regroup_op_fails_closed_to_input(monkeypatch):
    """A synthetic dropping/dup regroup op -> assert_partition_conservation
    raises -> the driver reverts to the input region list (reuses the _bad_apply
    monkeypatch pattern)."""
    import dart_semantic.qwen_specialists.block_resegment as mod

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
    ]

    def _bad_apply(rs, fb, ops):
        # Drop FB 2 entirely (content loss) -> partition multiset shrinks.
        return [dataclasses.replace(regions[0], feature_block_indices=(0, 1))]

    monkeypatch.setattr(mod, "apply_resegment", _bad_apply)
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert out == regions
    assert ops == []


def test_regroup_wins_overlapping_same_kind_merge(monkeypatch):
    """BOTH flags on. A unit whose body has an adjacent same-kind cross-page
    continuation (a _detect_merges candidate overlapping the unit run) -> the
    regroup APPLIES (regroup-first), the merged unit carries the label's
    semantic_class, and the overlapping same-kind merge is the op DROPPED."""
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    # Anchor label (FB0), body paragraph that ends mid-sentence (FB1, p1),
    # continued body paragraph starting lowercase (FB2, p2): the (1,2) pair is a
    # _detect_merges cross-page continuation candidate; the regroup run is
    # (0,1,2) and overlaps it.
    fbs = [
        _fb("EXAMPLE 1", page=1),
        _fb("a clause that runs", page=1),
        _fb("over the page edge", page=2),
    ]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example", pages=[1]),
        _region("paragraph", [1], text="a clause that runs", pages=[1]),
        _region("paragraph", [2], text="over the page edge", pages=[2]),
    ]
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    # Regroup WON the overlap: one merged unit spanning all three FBs, and its
    # breadcrumb is 'regroup' (had the same-kind merge won instead, region 0
    # would stand alone and a merged (1,2) would carry a 'merge' breadcrumb).
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1, 2)
    assert out[0].payload["resegment"]["op"] == "regroup"
    # The regroup op is FIRST in the candidate set (regroup-first precedence);
    # the overlapping same-kind merge is present but was DROPPED at apply time.
    assert ops[0].subtype == "regroup"
    assert ops[0].region_indices == (0, 1, 2)
    same_kind = [o for o in ops if o.subtype != "regroup"]
    assert len(same_kind) == 1 and same_kind[0].region_indices == (1, 2)
    # The merged unit carries the LABEL's semantic_class (regroup, not same-kind).
    assert out[0].payload["semantic_class"] == "worked_example"


def test_regroup_wins_overlapping_split(monkeypatch):
    """BOTH flags on. A unit body region carrying an interior pedagogical label
    (a _detect_splits candidate inside the unit run) -> the regroup wins, the
    split is dropped, and the body is kept whole inside the unit."""
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.delenv("SEMANTIK_READING_ORDER_FIX", raising=False)
    # Anchor label (FB0); body region with TWO FBs whose 2nd FB starts a
    # pedagogical "Solution" label (a _detect_splits interior-cut candidate).
    fbs = [_fb("EXAMPLE 1"), _fb("intro prose"), _fb("Solution factor it")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1, 2], text="intro prose Solution factor it"),
    ]
    # Sanity: the split detector WOULD fire on region 1 in isolation.
    assert any(
        o.op == "split" for o in _detect_splits(regions, fbs, _EmptyState())
    )
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    # Regroup WON: one merged unit spanning (0,1,2); the body region was NOT
    # split out (the overlapping split op was dropped at apply time).
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1, 2)
    assert out[0].payload["resegment"]["op"] == "regroup"
    # The regroup is first in the candidate set; the split is present but dropped.
    assert ops[0].subtype == "regroup"
    assert any(o.op == "split" for o in ops)  # present in candidates...
    assert all(r.payload.get("resegment", {}).get("op") != "split" for r in out)  # ...not applied
    assert out[0].payload["semantic_class"] == "worked_example"


# ---------------------------------------------------------------------------
# Phase 8 — provenance / block_id blast-radius (sourceId folds into label
# min-FB). The merge REDUCES region count (N->1): the FB partition multiset
# stays conserved (R-PART), but the sourceId SET shrinks — every body region's
# id folds into the LABEL's min(feature_block_indices). NOT a stable multiset
# like the reading-order reorder fix.
# ---------------------------------------------------------------------------


def _worked_example_unit():
    """label(pedagogy-example, FB0) + problem(FB1) + Solution(FB2) + step(FB3)
    -> a single regroup unit anchored at the label."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor"), _fb("Step 1")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="solve x"),
        _region("paragraph", [2], text="Solution factor", css_class="pedagogy-solution"),
        _region("paragraph", [3], text="Step 1", css_class="pedagogy-step"),
    ]
    return regions, fbs


def test_merged_unit_sourceid_is_label_min_fb(unit_regroup_on, monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)  # regroup-only
    regions, fbs = _worked_example_unit()
    out, ops = resegment_blocks(regions, fbs, _EmptyState())
    assert len(out) == 1
    merged = out[0]
    # The unit's stable id = the LABEL's min-FB. Downstream first_raw resolves
    # the same (cascade's first_raw = min(feature_block_indices)).
    assert min(merged.feature_block_indices) == 0
    assert min(merged.feature_block_indices) == min(regions[0].feature_block_indices) == 0


def test_body_region_ids_absent_after_merge(unit_regroup_on, monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    regions, fbs = _worked_example_unit()
    body_ids = {min(regions[k].feature_block_indices) for k in (1, 2, 3)}  # {1,2,3}
    out, _ = resegment_blocks(regions, fbs, _EmptyState())
    surviving_ids = {min(r.feature_block_indices) for r in out}
    # The honest fold: the body regions' former ids do NOT survive as their own
    # provenance entries; only the label's id remains.
    assert surviving_ids == {0}
    assert body_ids.isdisjoint(surviving_ids)


def test_provenance_one_entry_per_unit(unit_regroup_on, monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    regions, fbs = _worked_example_unit()
    out, _ = resegment_blocks(regions, fbs, _EmptyState())
    regroup_regions = [
        r for r in out if (r.payload or {}).get("resegment", {}).get("op") == "regroup"
    ]
    assert len(regroup_regions) == 1
    breadcrumb = regroup_regions[0].payload["resegment"]
    assert breadcrumb["op"] == "regroup"
    assert breadcrumb["source_ids"] == [0, 1, 2, 3]


def test_fb_multiset_conserved_provenance(unit_regroup_on, monkeypatch):
    import collections

    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    regions, fbs = _worked_example_unit()
    out, _ = resegment_blocks(regions, fbs, _EmptyState())
    in_fb = collections.Counter(
        i for r in regions for i in (r.feature_block_indices or ())
    )
    out_fb = collections.Counter(
        i for r in out for i in (r.feature_block_indices or ())
    )
    # R-PART at the provenance layer: no FB lost / duplicated under the N->1
    # region-count reduction.
    assert in_fb == out_fb


# ---------------------------------------------------------------------------
# Phase 9 — decision-capture audit-row enrichment (subtype='regroup' rides the
# already-present block_resegment enum; NO schema change).
# ---------------------------------------------------------------------------


def test_regroup_op_in_audit_rows(unit_regroup_on, monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    regions, fbs = _worked_example_unit()
    _, ops = resegment_blocks(regions, fbs, _EmptyState())
    rows = build_resegment_audit_rows(ops)
    regroup_rows = [r for r in rows if r["op"] == "regroup"]
    assert len(regroup_rows) == 1
    row = regroup_rows[0]
    assert row["semantic_class"] == "worked_example"   # pedagogy-example FLOOR
    assert row["source_ids"] == [0, 1, 2, 3]
    assert row["regions_folded"] == 3                   # 4 regions, 1 anchor
    assert row["conservation_verified"] is True
    # No decision_type leaks into the audit row (the bridge stamps it).
    assert "decision_type" not in row


def test_audit_row_byte_stable_flag_off():
    # A same-kind op (subtype default 'merge') -> the original 4-key row,
    # byte-identical to the pre-Phase-9 SEMANTIK_BLOCK_RESEGMENT-only baseline.
    same_kind = ResegmentOp(
        op="merge",
        region_indices=(0, 1),
        source_ids=(0, 1),
        origin="deterministic",
    )
    split = ResegmentOp(
        op="split",
        region_indices=(2,),
        split_at=(5,),
        source_ids=(2,),
        origin="llm",
    )
    rows = build_resegment_audit_rows([same_kind, split])
    assert rows == [
        {"op": "merge", "source_ids": [0, 1], "origin": "deterministic",
         "conservation_verified": True},
        {"op": "split", "source_ids": [2], "origin": "llm",
         "conservation_verified": True},
    ]
    # No regroup-only keys on a same-kind row.
    for row in rows:
        assert "semantic_class" not in row
        assert "regions_folded" not in row


def test_no_decision_schema_change():
    import json
    from pathlib import Path

    schema_path = (
        Path(__file__).resolve().parents[4]
        / "schemas" / "events" / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    enum = schema["properties"]["decision_type"]["enum"]
    # The audit rides the ALREADY-present block_resegment enum member — no new
    # decision_type was added for the regroup subtype.
    assert "block_resegment" in enum


# ---------------------------------------------------------------------------
# Phase 10 — verifier-drivable ResegmentOp seam (second-pass section_no_body
# fix). An external proposer passes explicit region_indices into the SAME
# apply_resegment + R-PART/token gates; a hallucinated/over-broad run is
# dropped per-op.
# ---------------------------------------------------------------------------


def test_externally_proposed_merge_revalidated():
    fbs = [_fb("EXAMPLE 1"), _fb("body one"), _fb("body two")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="body one"),
        _region("paragraph", [2], text="body two"),
    ]
    # A GOOD contiguous run -> applied + gated.
    out, ops = apply_proposed_regroups(regions, fbs, [[0, 1]])
    assert len(ops) == 1 and ops[0].subtype == "regroup"
    assert out[0].feature_block_indices == (0, 1)
    assert len(out) == 2  # merged(0,1) + region 2

    # A BAD non-contiguous run (skips region 1) -> apply_resegment would drop
    # region 1's FB -> R-PART raises -> the op is dropped, input preserved.
    out_bad, ops_bad = apply_proposed_regroups(regions, fbs, [[0, 2]])
    assert ops_bad == []
    assert out_bad is regions  # byte-identical input


def test_section_no_body_unit_mergeable():
    # section_no_body shape: a heading region with zero body, then the unit's
    # real body in the next (adjacent, post-reading-order) region. The verifier
    # proposes fusing them; the regroup op fixes it within the gates.
    fbs = [_fb("Pythagorean Theorem"), _fb("a^2 + b^2 = c^2 explanation")]
    regions = [
        _region("heading", [0], text="Pythagorean Theorem"),
        _region("paragraph", [1], text="a^2 + b^2 = c^2 explanation"),
    ]
    out, ops = apply_proposed_regroups(regions, fbs, [[0, 1]])
    assert len(ops) == 1 and ops[0].subtype == "regroup"
    assert len(out) == 1
    assert out[0].feature_block_indices == (0, 1)
    assert out[0].payload["resegment"]["op"] == "regroup"


def test_seam_inert_without_driver():
    fbs = [_fb("EXAMPLE 1"), _fb("body")]
    regions = [
        _region("paragraph", [0], text="EXAMPLE 1", css_class="pedagogy-example"),
        _region("paragraph", [1], text="body"),
    ]
    # No proposer (empty driver) -> byte-identical input, no ops.
    out, ops = apply_proposed_regroups(regions, fbs, [])
    assert ops == []
    assert out is regions


# ---------------------------------------------------------------------------
# ITEM2 (region-bbox) Phase-2 — Stage-5e merge/split recompute Region.page_bboxes.
# ---------------------------------------------------------------------------


def _fb_bbox(page: int, bbox) -> FeatureBlock:
    """A FeatureBlock with an explicit page + bbox (distinct geometry per FB)."""
    raw = RawBlock(
        text="x",
        page=page,
        bbox=bbox,
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


def test_merge_recomputes_geometry():
    """A MERGE fuses two stamped same-kind regions on pages 1+2; the merged
    region's page_bboxes is the union over BOTH pages, not the first region's
    stale single-page value."""
    a = (1.0, 2.0, 3.0, 4.0)
    b = (5.0, 6.0, 7.0, 8.0)
    fbs = [_fb_bbox(1, a), _fb_bbox(2, b)]
    r0, r1 = stamp_region_page_bboxes(
        [
            Region(kind="paragraph", feature_block_indices=(0,)),
            Region(kind="paragraph", feature_block_indices=(1,)),
        ],
        fbs,
    )
    assert r0.page_bboxes == ((1, a),)
    assert r1.page_bboxes == ((2, b),)

    op = ResegmentOp(op="merge", region_indices=(0, 1), source_ids=(0, 1))
    out = apply_resegment([r0, r1], fbs, [op])
    assert len(out) == 1
    merged = out[0]

    # Recomputed union over both pages — NOT the stale first-region value.
    assert merged.page_bboxes == ((1, a), (2, b))
    assert merged.page_bboxes != r0.page_bboxes
    assert merged.page_bboxes == compute_region_page_bboxes((0, 1), fbs)


def test_split_recomputes_geometry():
    """A SPLIT cuts one stamped cross-page region into two children; each child's
    page_bboxes is its own slice geometry, and the parent's union value appears
    on neither child."""
    a = (1.0, 2.0, 3.0, 4.0)
    b = (5.0, 6.0, 7.0, 8.0)
    fbs = [_fb_bbox(1, a), _fb_bbox(2, b)]
    [parent] = stamp_region_page_bboxes(
        [Region(kind="paragraph", feature_block_indices=(0, 1))], fbs
    )
    parent_geometry = parent.page_bboxes
    assert parent_geometry == ((1, a), (2, b))

    # Cut before FB 1 -> child0 = {FB0}, child1 = {FB1}.
    op = ResegmentOp(
        op="split", region_indices=(0,), split_at=(1,), source_ids=(0,)
    )
    out = apply_resegment([parent], fbs, [op])
    assert len(out) == 2
    child0, child1 = out

    assert child0.page_bboxes == ((1, a),)
    assert child1.page_bboxes == ((2, b),)
    # Each child equals its own slice's recomputed geometry.
    assert child0.page_bboxes == compute_region_page_bboxes(
        child0.feature_block_indices, fbs
    )
    assert child1.page_bboxes == compute_region_page_bboxes(
        child1.feature_block_indices, fbs
    )
    # The parent's union value is on NEITHER child (staleness recomputed away).
    assert child0.page_bboxes != parent_geometry
    assert child1.page_bboxes != parent_geometry


# ---------------------------------------------------------------------------
# ITEM6 — the Stage-5e op-proposal digest carries the additive role_top_k
# evidence read off region.provenance (stamped before Stage-5e runs).
# ---------------------------------------------------------------------------


def test_resegment_digest_carries_role_top_k_when_stamped():
    from dart_semantic.qwen_specialists.block_resegment_prompt import _region_obj
    from dart_semantic.structure_graph import Region as _R

    region = _R(
        kind="paragraph",
        feature_block_indices=(0,),
        payload={"text": "hi"},
        provenance={"role_top_k": [["paragraph", 0.7], ["code_block", 0.3]]},
    )
    obj = _region_obj(region, 0, [_fb("hi")])
    assert obj["role_top_k"] == [["paragraph", 0.7], ["code_block", 0.3]]


def test_resegment_digest_role_top_k_absent_when_unstamped():
    from dart_semantic.qwen_specialists.block_resegment_prompt import _region_obj
    from dart_semantic.structure_graph import Region as _R

    region = _R(kind="paragraph", feature_block_indices=(0,), payload={"text": "hi"})
    obj = _region_obj(region, 0, [_fb("hi")])
    assert "role_top_k" not in obj
