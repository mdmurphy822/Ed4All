"""Stage-5e MOVE op (ITEM3) — unit tests (CPU-only, pure, NO GPU / model).

Phase 1 covers the audited MoveOp vocabulary + the move-aware R-PART invariant
+ the apply machinery (``_apply_moves`` / ``_replay_moves`` /
``assert_partition_conservation(move_ops=...)`` / the audit-row move branch /
``resolve_move_op_mode``). No detector / driver / cascade is wired in Phase 1,
so every case here builds ``Region`` fixtures directly and calls the primitives.

Phase 2 covers the deterministic detector + driver + shadow default.
"""

from __future__ import annotations

import dataclasses

import pytest

from dart_semantic.qwen_specialists.block_resegment import (
    _MOVE_MAX_FB_GAP,
    _MOVE_MAX_REGION_DISTANCE,
    PartitionConservationError,
    ResegmentOp,
    _apply_moves,
    _detect_unit_merges,
    _detect_unit_moves,
    _ordered_fb_sequence,
    _replay_moves,
    apply_proposed_regroups,
    apply_proposed_unit_fix,
    assert_partition_conservation,
    build_resegment_audit_rows,
    resegment_blocks,
    resolve_move_op_mode,
)
from dart_semantic.structure_graph import Region, compute_region_page_bboxes
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Builders — a Region is fully described by its FB-index tuple for R-PART.
# ---------------------------------------------------------------------------


def _reg(fb_indices, *, kind="paragraph", text=None, source_region_id=None):
    payload = {}
    if text is not None:
        payload["text"] = text
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_indices),
        payload=payload,
        source_region_id=source_region_id,
    )


def _move(src, dest, *, reason="unit_label_distant_body", evidence=()):
    return ResegmentOp(
        op="move",
        region_indices=(src,),
        move_after=dest,
        reason=reason,
        evidence=tuple(evidence),
        origin="deterministic",
    )


def _reorder(regions, order):
    """Build an out-region list in the given INPUT-index order."""
    return [regions[k] for k in order]


# ---------------------------------------------------------------------------
# _replay_moves + the move-aware R-PART clause.
# ---------------------------------------------------------------------------


def test_replay_moves_explains_permutation():
    regions = [_reg((0,)), _reg((1,)), _reg((2,)), _reg((3,))]
    op = _move(2, 0)  # relocate region 2 to immediately after region 0
    assert _replay_moves(regions, [op]) == [0, 2, 1, 3]
    # An out-list carrying exactly that permutation passes with the op.
    out = _reorder(regions, [0, 2, 1, 3])
    assert_partition_conservation(regions, out, move_ops=[op])


def test_replay_moves_front_anchor():
    regions = [_reg((0,)), _reg((1,)), _reg((2,))]
    op = _move(2, -1)  # -1 == document front
    assert _replay_moves(regions, [op]) == [2, 0, 1]
    out = _reorder(regions, [2, 0, 1])
    assert_partition_conservation(regions, out, move_ops=[op])


def test_replay_moves_empty_is_identity():
    regions = [_reg((0,)), _reg((1, 2)), _reg((3,))]
    assert _replay_moves(regions, []) == [0, 1, 2, 3]
    # The default (no move_ops) path is byte-identical to the legacy assertion.
    assert_partition_conservation(regions, list(regions))


def test_unexplained_permutation_still_raises():
    regions = [_reg((0,)), _reg((1,)), _reg((2,)), _reg((3,))]
    out = _reorder(regions, [0, 2, 1, 3])
    # Legacy behaviour pinned: a permutation with NO move op raises.
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(regions, out)
    # A NON-matching move op (expects [3,0,1,2]) also raises against this out.
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(regions, out, move_ops=[_move(3, -1)])


def test_multiset_clause_unconditional():
    regions = [_reg((0,)), _reg((1,)), _reg((2,))]
    op = _move(2, 0)
    # Drop an FB -> clause 1 raises EVEN with a move op supplied.
    dropped = [_reg((0,)), _reg((1,))]
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(regions, dropped, move_ops=[op])
    # Duplicate an FB -> clause 1 raises too.
    dup = [_reg((0,)), _reg((2,)), _reg((1,)), _reg((2,))]
    with pytest.raises(PartitionConservationError):
        assert_partition_conservation(regions, dup, move_ops=[op])


# ---------------------------------------------------------------------------
# _apply_moves — permutation-only, breadcrumb, identity, replay determinism.
# ---------------------------------------------------------------------------


def test_apply_moves_permutation_only():
    regions = [
        _reg((0,), text="LABEL"),
        _reg((1,), text="between"),
        _reg((2,), text="body"),
        _reg((3,), text="tail"),
    ]
    op = _move(0, 1)  # move the label to right after region 1 (before its body)
    out = _apply_moves(regions, [], [op])
    # Same length; FB multiset conserved.
    assert len(out) == len(regions)
    assert sorted(_ordered_fb_sequence(out)) == sorted(_ordered_fb_sequence(regions))
    # Unmoved regions keep object identity; the moved region is a clone.
    assert out[0] is regions[1]
    assert out[1] is not regions[0]
    assert out[1].feature_block_indices == regions[0].feature_block_indices
    # Locate the moved (breadcrumbed) region: only region 0 carries the crumb.
    moved = [r for r in out if "resegment_move" in (r.payload or {})]
    assert len(moved) == 1
    crumb = moved[0].payload["resegment_move"]
    assert crumb["op"] == "move"
    assert crumb["conservation_verified"] is True
    # No payload text change except the added breadcrumb key.
    assert moved[0].payload["text"] == "LABEL"
    assert set(moved[0].payload) == {"text", "resegment_move"}


def test_apply_moves_replay_determinism():
    regions = [_reg((0,)), _reg((1,)), _reg((2,)), _reg((3,)), _reg((4,))]
    ops = [_move(3, 0)]
    applied = _apply_moves(regions, [], ops)
    assert _ordered_fb_sequence(applied) == _replay_moves(regions, ops)
    # And the produced list gates clean against the move-aware invariant.
    assert_partition_conservation(regions, applied, move_ops=ops)


def test_apply_moves_multi_fb_region():
    regions = [_reg((0,)), _reg((1, 2, 3)), _reg((4,))]
    ops = [_move(1, -1)]  # a multi-FB body region moved to the front
    applied = _apply_moves(regions, [], ops)
    assert _ordered_fb_sequence(applied) == [1, 2, 3, 0, 4]
    assert_partition_conservation(regions, applied, move_ops=ops)


# ---------------------------------------------------------------------------
# Overlapping moves — fail closed (replay raises; apply drops the second).
# ---------------------------------------------------------------------------


def test_overlapping_moves_fail_closed():
    regions = [_reg((0,)), _reg((1,)), _reg((2,)), _reg((3,))]
    op1 = _move(2, 0)
    op2 = _move(3, 2)  # dest anchor 2 was op1's SOURCE -> overlap
    # replay (strict) raises.
    with pytest.raises(PartitionConservationError):
        _replay_moves(regions, [op1, op2])
    # apply (non-strict) drops the second; the result equals op1 applied alone.
    applied = _apply_moves(regions, [], [op1, op2])
    assert _ordered_fb_sequence(applied) == _replay_moves(regions, [op1])


def test_out_of_range_move_dropped_on_apply():
    regions = [_reg((0,)), _reg((1,))]
    bad = _move(5, 0)  # src out of range
    applied = _apply_moves(regions, [], [bad])
    assert _ordered_fb_sequence(applied) == [0, 1]  # no-op
    with pytest.raises(PartitionConservationError):
        _replay_moves(regions, [bad])


# ---------------------------------------------------------------------------
# Audit rows.
# ---------------------------------------------------------------------------


def test_audit_row_move_branch():
    shadow_op = _move(
        3,
        0,
        evidence=[
            ("dest_source_id", 0),
            ("fb_gap", 7),
            ("mode", "shadow"),
            ("region_distance", 3),
        ],
    )
    (row,) = build_resegment_audit_rows([shadow_op])
    assert row["op"] == "move"
    assert row["reason"] == "unit_label_distant_body"
    assert row["dest_source_id"] == 0
    assert row["region_distance"] == 3
    assert row["fb_gap"] == 7
    assert row["mode"] == "shadow"
    assert row["applied"] is False
    assert row["conservation_verified"] is True

    live_op = _move(3, 0, evidence=[("dest_source_id", 0), ("region_distance", 2)])
    (live_row,) = build_resegment_audit_rows([live_op])
    assert live_row["mode"] == "live"
    assert live_row["applied"] is True


def test_audit_rows_without_moves_byte_identical():
    merge = ResegmentOp(op="merge", region_indices=(0, 1), source_ids=(0, 1))
    split = ResegmentOp(op="split", region_indices=(2,), split_at=(5,), source_ids=(2,))
    regroup = ResegmentOp(
        op="merge",
        subtype="regroup",
        region_indices=(3, 4),
        source_ids=(3, 4),
        semantic_class="worked_example",
    )
    rows = build_resegment_audit_rows([merge, split, regroup])
    # No move keys leak onto non-move rows.
    for row in rows:
        assert "mode" not in row
        assert "applied" not in row
        assert "dest_source_id" not in row
    assert rows[0]["op"] == "merge"
    assert rows[1]["op"] == "split"
    assert rows[2]["op"] == "regroup"
    assert rows[2]["semantic_class"] == "worked_example"
    assert rows[2]["regions_folded"] == 1


# ---------------------------------------------------------------------------
# resolve_move_op_mode — three-valued parse table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "shadow"),
        ("", "shadow"),
        ("   ", "shadow"),
        ("0", "off"),
        ("false", "off"),
        ("no", "off"),
        ("off", "off"),
        ("Off", "off"),
        ("OFF", "off"),
        ("live", "live"),
        ("LIVE", "live"),
        ("  live  ", "live"),
        ("1", "shadow"),
        ("true", "shadow"),
        ("yes", "shadow"),
        ("on", "shadow"),
        ("garbage", "shadow"),
    ],
)
def test_resolve_move_op_mode_parse(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SEMANTIK_MOVE_OP", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_MOVE_OP", value)
    assert resolve_move_op_mode() == expected


# ===========================================================================
# Phase 2 — deterministic detector + driver + shadow default.
# ===========================================================================


def _fb(text, page=1, bbox=(0.0, 0.0, 10.0, 10.0)):
    raw = RawBlock(
        text=text, page=page, bbox=bbox, page_width=100.0, page_height=100.0
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


def _mk(
    kind,
    fb_idx,
    feature_blocks,
    *,
    text=None,
    css_class=None,
    semantic_class=None,
    source_region_id=None,
    with_bbox=True,
):
    payload = {}
    if text is not None:
        payload["text"] = text
    if css_class is not None:
        payload["css_class"] = css_class
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    pb = (
        compute_region_page_bboxes(tuple(fb_idx), feature_blocks)
        if with_bbox
        else None
    )
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_idx),
        payload=payload,
        source_region_id=source_region_id,
        page_bboxes=pb,
    )


class _State:
    """Minimal CouncilState stand-in (the move arm never reads it)."""

    outputs: dict = {}


def _misordered_fixture(with_bbox=True):
    """label(0) -> passthrough(1) -> body(2) -> heading(3): the body is the
    UNIQUE same-page candidate; move_after == 1 (j-1, j==2)."""
    fbs = [
        _fb("EXAMPLE 2 here", page=1, bbox=(0.0, 0.0, 10.0, 10.0)),
        _fb("some table", page=1, bbox=(0.0, 12.0, 10.0, 20.0)),
        _fb("Consider the derivative of the function.", page=1, bbox=(0.0, 11.0, 10.0, 18.0)),
        _fb("Next Section", page=1, bbox=(0.0, 30.0, 10.0, 34.0)),
    ]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 2 here", css_class="pedagogy-example", with_bbox=with_bbox),
        _mk("table", [1], fbs, text="some table", source_region_id=99, with_bbox=with_bbox),
        _mk("paragraph", [2], fbs, text="Consider the derivative of the function.", with_bbox=with_bbox),
        _mk("heading", [3], fbs, text="Next Section", with_bbox=with_bbox),
    ]
    return regions, fbs


@pytest.fixture(autouse=True)
def _move_shadow_default(monkeypatch):
    """Default the module env clean; each test sets SEMANTIK_MOVE_OP itself.
    UNIT_REGROUP off so the driver tests isolate the MOVE arm."""
    monkeypatch.delenv("SEMANTIK_MOVE_OP", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)


def test_detect_unit_moves_fires_on_misordered_label(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    regions, fbs = _misordered_fixture()
    ops = _detect_unit_moves(regions, fbs)
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "move"
    assert op.region_indices == (0,)
    assert op.move_after == 1  # j - 1, j == 2
    assert op.reason == "unit_label_distant_body"
    assert op.source_ids == (0,)
    ev = dict(op.evidence)
    assert ev["bbox"] == "ok"
    assert ev["dest_source_id"] == 1
    assert ev["region_distance"] == 2
    assert ev["fb_gap"] == 2


def test_detect_ambiguous_two_candidates_fail_closed(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    fbs = [
        _fb("EXAMPLE 2"),
        _fb("passthrough"),
        _fb("first body candidate here"),
        _fb("second body candidate here"),
    ]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 2", css_class="pedagogy-example"),
        _mk("table", [1], fbs, text="passthrough", source_region_id=7),
        _mk("paragraph", [2], fbs, text="first body candidate here"),
        _mk("paragraph", [3], fbs, text="second body candidate here"),
    ]
    assert _detect_unit_moves(regions, fbs) == []


def test_heading_barrier_blocks_move(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    fbs = [_fb("EXAMPLE 2"), _fb("passthrough"), _fb("A Heading"), _fb("the body prose here")]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 2", css_class="pedagogy-example"),
        _mk("table", [1], fbs, text="passthrough", source_region_id=7),
        _mk("heading", [2], fbs, text="A Heading"),
        _mk("paragraph", [3], fbs, text="the body prose here"),
    ]
    assert _detect_unit_moves(regions, fbs) == []


def test_distance_and_fb_gap_caps(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    # Body FB index far beyond the FB-gap cap -> not a candidate.
    big = _MOVE_MAX_FB_GAP + 5
    fbs = [_fb("EXAMPLE 2")] + [_fb("filler")] * (big - 1) + [_fb("the body prose here")]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 2", css_class="pedagogy-example"),
        _mk("table", [1], fbs, text="filler", source_region_id=7),
        _mk("paragraph", [big], fbs, text="the body prose here"),
    ]
    # Region distance is only 2, but the FB gap (big) exceeds the cap.
    assert _detect_unit_moves(regions, fbs) == []


def test_passthrough_never_moves(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    # (a) a passthrough LABEL is never the src.
    fbs = [_fb("EXAMPLE 2"), _fb("body prose here")]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 2", css_class="pedagogy-example", source_region_id=5),
        _mk("paragraph", [1], fbs, text="body prose here"),
    ]
    assert _detect_unit_moves(regions, fbs) == []
    # (b) when the only follower is passthrough there is no body candidate.
    fbs2 = [_fb("EXAMPLE 2"), _fb("tbl a"), _fb("tbl b")]
    regions2 = [
        _mk("paragraph", [0], fbs2, text="EXAMPLE 2", css_class="pedagogy-example"),
        _mk("table", [1], fbs2, text="tbl a", source_region_id=7),
        _mk("table", [2], fbs2, text="tbl b", source_region_id=8),
    ]
    assert _detect_unit_moves(regions2, fbs2) == []


def test_no_bbox_shadow_only_evidence(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    regions, fbs = _misordered_fixture(with_bbox=False)
    ops = _detect_unit_moves(regions, fbs)
    assert len(ops) == 1
    assert dict(ops[0].evidence)["bbox"] == "bbox_missing"


def test_detector_off_returns_empty(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "0")
    regions, fbs = _misordered_fixture()
    assert _detect_unit_moves(regions, fbs) == []


# --- driver ---------------------------------------------------------------


def test_driver_shadow_no_region_change(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    regions, fbs = _misordered_fixture()
    out, ops = resegment_blocks(regions, fbs, _State())
    # Shadow: the returned region list is the UNCHANGED pre-move input (identity).
    assert out is regions
    move_ops = [o for o in ops if o.op == "move"]
    assert len(move_ops) == 1
    assert dict(move_ops[0].evidence)["mode"] == "shadow"
    (row,) = [r for r in build_resegment_audit_rows(ops) if r["op"] == "move"]
    assert row["applied"] is False
    assert row["mode"] == "shadow"


def test_driver_live_applies_and_gates(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    regions, fbs = _misordered_fixture()
    out, ops = resegment_blocks(regions, fbs, _State())
    # Live: the label (region 0) is relocated to immediately after region 1.
    assert _ordered_fb_sequence(out) == _replay_moves(regions, [_apply_ref(regions, fbs)])
    move_ops = [o for o in ops if o.op == "move"]
    assert len(move_ops) == 1
    (row,) = [r for r in build_resegment_audit_rows(ops) if r["op"] == "move"]
    assert row["applied"] is True
    assert row["mode"] == "live"
    # The whole-set gate held (out is a clean permutation of the input FBs).
    assert sorted(_ordered_fb_sequence(out)) == sorted(_ordered_fb_sequence(regions))


def _apply_ref(regions, fbs):
    """The single move op the detector emits on the fixture (for replay-cmp)."""
    return _detect_unit_moves(regions, fbs)[0]


def test_driver_live_drops_poisoned_move(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    regions, fbs = _misordered_fixture()
    good = _detect_unit_moves(regions, fbs)[0]
    poisoned = ResegmentOp(
        op="move",
        region_indices=(0,),
        move_after=999,  # out of range -> individually dropped
        source_ids=(0,),
        reason="unit_label_distant_body",
        evidence=(("bbox", "ok"),),
    )
    import dart_semantic.qwen_specialists.block_resegment as br

    monkeypatch.setattr(br, "_detect_unit_moves", lambda r, f: [good, poisoned])
    out, ops = resegment_blocks(regions, fbs, _State())
    # Only the good op survived + applied; the poisoned one was dropped alone.
    move_ops = [o for o in ops if o.op == "move"]
    assert len(move_ops) == 1
    assert _ordered_fb_sequence(out) == _replay_moves(regions, [good])


def test_driver_flag_off_byte_identical(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "0")
    regions, fbs = _misordered_fixture()
    out, ops = resegment_blocks(regions, fbs, _State())
    assert out is regions
    assert [o for o in ops if o.op == "move"] == []


def test_driver_live_no_bbox_stays_shadow(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    regions, fbs = _misordered_fixture(with_bbox=False)
    out, ops = resegment_blocks(regions, fbs, _State())
    # Live mode but NO bbox corroboration -> the op stays shadow-only (not applied).
    assert out is regions
    move_ops = [o for o in ops if o.op == "move"]
    assert len(move_ops) == 1
    assert dict(move_ops[0].evidence)["mode"] == "shadow"


# ===========================================================================
# Phase 3 — verifier-channel graduation (MOVE+merge) + breadcrumb adjacency.
# ===========================================================================


def _contiguous_unit_fixture():
    """label(0, pedagogy-example) + body(1), FB-adjacent -> a contiguous run."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve the problem here")]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 1", css_class="pedagogy-example"),
        _mk("paragraph", [1], fbs, text="solve the problem here"),
    ]
    return regions, fbs


def _noncontiguous_unit_fixture():
    """label(0) then two intervening regions then body(3): NON-contiguous run
    (0, 3). All same page, no heading between -> boundable."""
    fbs = [
        _fb("EXAMPLE 1"),
        _fb("unrelated aside one"),
        _fb("unrelated aside two"),
        _fb("solve the derivative here"),
    ]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 1", css_class="pedagogy-example"),
        _mk("paragraph", [1], fbs, text="unrelated aside one"),
        _mk("paragraph", [2], fbs, text="unrelated aside two"),
        _mk("paragraph", [3], fbs, text="solve the derivative here"),
    ]
    return regions, fbs


def test_apply_proposed_unit_fix_contiguous_delegates(monkeypatch):
    regions, fbs = _contiguous_unit_fixture()
    # Non-live: byte-parity with apply_proposed_regroups on a contiguous run.
    for mode in ("0", "shadow"):
        monkeypatch.setenv("SEMANTIK_MOVE_OP", mode)
        out_fix, ops_fix = apply_proposed_unit_fix(regions, fbs, [(0, 1)])
        out_reg, ops_reg = apply_proposed_regroups(regions, fbs, [(0, 1)])
        assert _ordered_fb_sequence(out_fix) == _ordered_fb_sequence(out_reg)
        assert [o.op for o in ops_fix] == [o.op for o in ops_reg]
    # Live contiguous also delegates (single regroup, no move).
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    out_live, ops_live = apply_proposed_unit_fix(regions, fbs, [(0, 1)])
    assert [o.op for o in ops_live if o.op == "move"] == []
    assert len(out_live) == 1  # fused


def test_non_contiguous_run_decomposes_live(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    regions, fbs = _noncontiguous_unit_fixture()
    out, ops = apply_proposed_unit_fix(regions, fbs, [(0, 3)])
    move_ops = [o for o in ops if o.op == "move"]
    merge_ops = [o for o in ops if o.subtype == "regroup"]
    assert len(move_ops) == 1
    assert len(merge_ops) == 1
    # The label(fb0) + body(fb3) are fused into ONE region; asides survive.
    merged = [r for r in out if set(r.feature_block_indices) == {0, 3}]
    assert len(merged) == 1
    # FB multiset conserved end-to-end.
    assert sorted(_ordered_fb_sequence(out)) == [0, 1, 2, 3]


def test_non_contiguous_run_dropped_when_not_live(monkeypatch):
    regions, fbs = _noncontiguous_unit_fixture()
    for mode in ("0", "shadow"):
        monkeypatch.setenv("SEMANTIK_MOVE_OP", mode)
        out, ops = apply_proposed_unit_fix(regions, fbs, [(0, 3)])
        # Delegates to apply_proposed_regroups -> non-contiguous run dropped.
        assert ops == []
        assert _ordered_fb_sequence(out) == [0, 1, 2, 3]
        assert len(out) == len(regions)


def test_unboundable_member_drops_whole_run(monkeypatch):
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    fbs = [
        _fb("EXAMPLE 1"),
        _fb("A Heading"),
        _fb("solve the derivative here"),
    ]
    regions = [
        _mk("paragraph", [0], fbs, text="EXAMPLE 1", css_class="pedagogy-example"),
        _mk("heading", [1], fbs, text="A Heading"),
        _mk("paragraph", [2], fbs, text="solve the derivative here"),
    ]
    out, ops = apply_proposed_unit_fix(regions, fbs, [(0, 2)])
    # Heading strictly between -> unboundable -> whole run dropped.
    assert ops == []
    assert _ordered_fb_sequence(out) == [0, 1, 2]


def test_regroup_breadcrumb_adjacency(monkeypatch):
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    # label(fb0) then body(fb5) list-adjacent but FB-GAPPED; the body carries a
    # landed-move breadcrumb whose dest chains to the label -> regroup fuses them.
    fbs = [_fb("EXAMPLE 1")] + [_fb(f"x{i}") for i in range(1, 6)]
    label = _mk("paragraph", [0], fbs, text="EXAMPLE 1", css_class="pedagogy-example")
    body = Region(
        kind="paragraph",
        feature_block_indices=(5,),
        payload={
            "text": "solve it",
            "resegment_move": {"op": "move", "dest_source_id": 0},
        },
    )
    ops = _detect_unit_merges([label, body], fbs, _State())
    assert len(ops) == 1
    assert ops[0].subtype == "regroup"
    assert ops[0].region_indices == (0, 1)


def test_regroup_breadcrumb_absent_still_breaks(monkeypatch):
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    # Same FB gap but NO breadcrumb -> the non-adjacency is a hard boundary
    # (flag-off / no-move byte-identical to the pre-ITEM3 behaviour).
    fbs = [_fb("EXAMPLE 1")] + [_fb(f"x{i}") for i in range(1, 6)]
    label = _mk("paragraph", [0], fbs, text="EXAMPLE 1", css_class="pedagogy-example")
    body = _mk("paragraph", [5], fbs, text="solve it")
    assert _detect_unit_merges([label, body], fbs, _State()) == []
