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
    PartitionConservationError,
    ResegmentOp,
    _apply_moves,
    _ordered_fb_sequence,
    _replay_moves,
    assert_partition_conservation,
    build_resegment_audit_rows,
    resolve_move_op_mode,
)
from dart_semantic.structure_graph import Region


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
