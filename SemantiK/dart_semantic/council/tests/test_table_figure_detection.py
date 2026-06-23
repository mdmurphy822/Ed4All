"""Unit tests — Part T (table structural-confidence confirm) + Part F
(deterministic figure route) in the cross-BERT reranker.

All CPU, no GPU, no model load: synthetic CouncilState + candidates only.
"""

from __future__ import annotations

import pytest

from dart_semantic.council.cross_reranker import (
    _structurally_confident_table,
    _table_confirmed,
    arbitrate,
    confirm_table_candidates,
    resolve_table_structural_confirm,
)
from dart_semantic.council.types import (
    BertOutput,
    CouncilState,
    ImageCandidate,
    TableCandidate,
    TypedSignal,
)


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _state_with_low_table_head(member_idx: list[int]) -> CouncilState:
    """A CouncilState whose ``table_region`` head fires BELOW 0.5 on every
    member span — i.e. the regression that drops every real table."""
    sigs = [
        TypedSignal(
            head_name="table_region",
            region_id=i,
            top_k_labels=["not_table_region", "table_region"],
            top_k_confidences=[0.9, 0.1],
        )
        for i in member_idx
    ]
    return CouncilState(outputs={"structure": BertOutput("structure", sigs)})


def _empty_state() -> CouncilState:
    return CouncilState(outputs={})


def _grid_table(member_idx: list[int]) -> TableCandidate:
    return TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        evidence_features={"n_rows": 4, "n_cols": 3, "bordered": False},
        member_block_indices=member_idx,
        cell_grid=[["a", "b", "c"]] * 4,
        bordered=False,
    )


def _thin_table(member_idx: list[int]) -> TableCandidate:
    """A 1xN borderless 'table' — a page-layout false positive."""
    return TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 20.0),
        pages=[1],
        evidence_features={"n_rows": 1, "n_cols": 5, "bordered": False},
        member_block_indices=member_idx,
        cell_grid=[["a", "b", "c", "d", "e"]],
        bordered=False,
    )


# ---------------------------------------------------------------------------
# Part T — structural confidence guard.
# ---------------------------------------------------------------------------


def test_structural_guard_accepts_real_grid():
    assert _structurally_confident_table(_grid_table([1])) is True


def test_structural_guard_rejects_thin_layout_table():
    assert _structurally_confident_table(_thin_table([1])) is False


def test_structural_guard_accepts_bordered_keyvalue():
    cand = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 40.0),
        pages=[1],
        evidence_features={"n_rows": 2, "n_cols": 1, "bordered": True},
        member_block_indices=[1],
        cell_grid=[["k"], ["v"]],
        bordered=True,
    )
    assert _structurally_confident_table(cand) is True


def test_table_confirmed_load_bearing_over_low_bert():
    """A structurally-strong table is confirmed even when the BERT head
    fires < 0.5 on every member span (the 0-tables regression fix)."""
    cand = _grid_table([0, 1])
    confirmed, signal_present = _table_confirmed(
        cand, _state_with_low_table_head([0, 1]), structural_confirm=True
    )
    assert confirmed is True
    assert signal_present is True


def test_table_confirmed_load_bearing_with_no_bert_signal():
    cand = _grid_table([0, 1])
    confirmed, signal_present = _table_confirmed(
        cand, _empty_state(), structural_confirm=True
    )
    assert confirmed is True
    assert signal_present is True


def test_table_confirmed_legacy_off_drops_on_low_bert():
    """structural_confirm=False reverts to the legacy BERT-gated-only path:
    the low head drops the table (byte-stable to the old behaviour)."""
    cand = _grid_table([0, 1])
    confirmed, signal_present = _table_confirmed(
        cand, _state_with_low_table_head([0, 1]), structural_confirm=False
    )
    assert confirmed is False
    assert signal_present is True


def test_thin_table_not_confirmed_by_structural_arm():
    cand = _thin_table([0])
    confirmed, _ = _table_confirmed(
        cand, _state_with_low_table_head([0]), structural_confirm=True
    )
    assert confirmed is False


def test_arbitrate_routes_real_grid_to_table_default():
    """End-to-end through arbitrate: a real grid lands on the table track
    despite the low BERT head (default structural-confirm ON)."""
    cand = _grid_table([0, 1])
    decisions = arbitrate(_state_with_low_table_head([0, 1]), [cand])
    assert len(decisions) == 1
    assert decisions[0].route == "table"
    assert decisions[0].final_role == "table"


def test_confirm_table_candidates_uses_structural_arm():
    cand = _grid_table([0, 1])
    kept = confirm_table_candidates(
        _state_with_low_table_head([0, 1]), [cand]
    )
    assert kept == [cand]


def test_resolve_default_on(monkeypatch):
    monkeypatch.delenv("SEMANTIK_TABLE_STRUCTURAL_CONFIRM", raising=False)
    assert resolve_table_structural_confirm() is True
    monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURAL_CONFIRM", "off")
    assert resolve_table_structural_confirm() is False
    monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURAL_CONFIRM", "garbage")
    assert resolve_table_structural_confirm() is True


# ---------------------------------------------------------------------------
# Part F — deterministic figure route.
# ---------------------------------------------------------------------------


def _image_cand(member_idx: list[int]) -> ImageCandidate:
    return ImageCandidate(
        kind="figure",
        bbox=(50.0, 50.0, 250.0, 250.0),
        pages=[1],
        evidence_features={},
        member_block_indices=member_idx,
        px_size=(400, 400),
        image_index=0,
    )


def test_image_candidate_routes_to_figure_deterministically():
    """An ImageCandidate routes to figure regardless of the is_image_block
    head — the geometric detector is load-bearing."""
    cand = _image_cand([3])
    decisions = arbitrate(_empty_state(), [cand])
    assert decisions[0].route == "figure"
    assert decisions[0].final_role == "figure"


def test_image_candidate_route_unaffected_by_low_image_head():
    cand = _image_cand([3])
    state = CouncilState(
        outputs={
            "structure": BertOutput(
                "structure",
                [
                    TypedSignal(
                        head_name="is_image_block",
                        region_id=3,
                        top_k_labels=["not_image_block", "image_block"],
                        top_k_confidences=[0.95, 0.05],
                    )
                ],
            )
        }
    )
    decisions = arbitrate(state, [cand])
    assert decisions[0].route == "figure"


def test_image_candidate_high_head_adds_confirmation_flag():
    cand = _image_cand([3])
    state = CouncilState(
        outputs={
            "structure": BertOutput(
                "structure",
                [
                    TypedSignal(
                        head_name="is_image_block",
                        region_id=3,
                        top_k_labels=["image_block", "not_image_block"],
                        top_k_confidences=[0.9, 0.1],
                    )
                ],
            )
        }
    )
    decisions = arbitrate(state, [cand])
    assert decisions[0].route == "figure"
    assert "image_block_confirmed" in decisions[0].flags
