"""Torch-FREE unit tests — Structure→Semantic cascade AXIS-1 role fold.

Covers ``council.orchestrator._derive_cascades`` / ``_fold_role_to_legacy``:
the deployed structure adapter emits a 9-class ``structural_role``
distribution (AXIS-1 added ``definition_term`` / ``definition_def`` /
``caption``), but the Semantic head was trained on the LEGACY 6-role cascade
and consumes an 8-dim vector positionally. ``_derive_cascades`` folds the 3
AXIS-1 roles' probability mass into ``paragraph`` so the vector stays 8-dim.

These are pure-Python (no torch): the fold + dim invariant are the whole
surface. Importing ``orchestrator`` pulls in ``structure`` for the role-name
tuples only (torch is imported lazily inside the runtime functions).
"""

from __future__ import annotations

import math

import pytest

from dart_semantic.council import orchestrator as orch
from dart_semantic.council.semantic import CASCADE_DIM
from dart_semantic.council.structure import ROLE_NAMES
from dart_semantic.council.types import BertOutput, TypedSignal


def _role_signal(region_id: int, role_probs: dict[str, float]) -> TypedSignal:
    """A full-distribution ``structural_role`` signal for one span."""
    labels = list(role_probs.keys())
    return TypedSignal(
        head_name="structural_role",
        region_id=region_id,
        top_k_labels=labels,
        top_k_confidences=[role_probs[name] for name in labels],
    )


def _binary_signal(head_name: str, region_id: int, p_positive: float) -> TypedSignal:
    """A 2-class gate signal (is_heading / table_region), index-1 = positive."""
    # _signal_to_full slots by label name, so carry both labels explicitly.
    if head_name == "is_heading":
        neg, pos = "not_heading", "heading"
    else:
        neg, pos = "not_table_region", "table_region"
    return TypedSignal(
        head_name=head_name,
        region_id=region_id,
        top_k_labels=[pos, neg],
        top_k_confidences=[p_positive, 1.0 - p_positive],
    )


def _structure_out(signals: list[TypedSignal]) -> BertOutput:
    return BertOutput(bert_name="structure", signals=signals)


# ---------------------------------------------------------------------------
# Sanity: the legacy 6-slot order the fold depends on is the deployed order.
# ---------------------------------------------------------------------------


def test_legacy_role_names_match_deployed_first_six():
    assert orch._LEGACY_ROLE_NAMES == (
        "paragraph",
        "heading",
        "list_item",
        "form_label",
        "blockquote",
        "code_block",
    )
    assert orch._LEGACY_ROLE_NAMES == ROLE_NAMES[:6]
    assert orch._LEGACY_CASCADE_DIM == CASCADE_DIM == 8


# ---------------------------------------------------------------------------
# (a) 9-role signal → 8-dim cascade, AXIS-1 mass folded into paragraph.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    len(ROLE_NAMES) != 9,
    reason="AXIS-1 fold test requires the 9-role deployed adapter taxonomy",
)
def test_nine_role_folds_axis1_mass_into_paragraph():
    # paragraph .10 + def_term .05 + def_def .07 + caption .03 -> .25 paragraph.
    role_probs = {
        "paragraph": 0.10,
        "heading": 0.20,
        "list_item": 0.15,
        "form_label": 0.05,
        "blockquote": 0.05,
        "code_block": 0.10,
        "definition_term": 0.05,
        "definition_def": 0.07,
        "caption": 0.03,
    }
    # sums to 0.80; remainder is irrelevant to the fold arithmetic.
    out = _structure_out(
        [
            _role_signal(0, role_probs),
            _binary_signal("is_heading", 0, 0.9),
            _binary_signal("table_region", 0, 0.4),
        ]
    )
    cascades = orch._derive_cascades([object()], out)
    assert len(cascades) == 1
    vec = cascades[0]
    assert len(vec) == 8

    # 6 legacy role slots in _LEGACY_ROLE_NAMES order.
    para, heading, list_item, form_label, blockquote, code_block = vec[:6]
    assert math.isclose(para, 0.10 + 0.05 + 0.07 + 0.03, rel_tol=1e-9)
    # Other legacy slots preserved (no fold target touches them).
    assert math.isclose(heading, 0.20, rel_tol=1e-9)
    assert math.isclose(list_item, 0.15, rel_tol=1e-9)
    assert math.isclose(form_label, 0.05, rel_tol=1e-9)
    assert math.isclose(blockquote, 0.05, rel_tol=1e-9)
    assert math.isclose(code_block, 0.10, rel_tol=1e-9)
    # Gates land at indices 6/7 (positive-class probability).
    assert math.isclose(vec[6], 0.9, rel_tol=1e-9)
    assert math.isclose(vec[7], 0.4, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (b) 6-role (pre-AXIS-1) taxonomy → straight passthrough, no fold.
# ---------------------------------------------------------------------------


def test_six_role_taxonomy_passthrough(monkeypatch):
    six = (
        "paragraph",
        "heading",
        "list_item",
        "form_label",
        "blockquote",
        "code_block",
    )
    # Simulate an older / pre-AXIS-1 checkout where ROLE_NAMES has 6 entries.
    monkeypatch.setattr(orch, "ROLE_NAMES", six)
    role_probs = {
        "paragraph": 0.30,
        "heading": 0.10,
        "list_item": 0.20,
        "form_label": 0.05,
        "blockquote": 0.15,
        "code_block": 0.20,
    }
    out = _structure_out(
        [
            _role_signal(0, role_probs),
            _binary_signal("is_heading", 0, 0.25),
            _binary_signal("table_region", 0, 0.75),
        ]
    )
    cascades = orch._derive_cascades([object()], out)
    vec = cascades[0]
    assert len(vec) == 8
    # Each legacy slot equals its own mass — nothing folded in.
    assert math.isclose(vec[0], 0.30, rel_tol=1e-9)  # paragraph unchanged
    assert math.isclose(vec[1], 0.10, rel_tol=1e-9)
    assert math.isclose(vec[2], 0.20, rel_tol=1e-9)
    assert math.isclose(vec[3], 0.05, rel_tol=1e-9)
    assert math.isclose(vec[4], 0.15, rel_tol=1e-9)
    assert math.isclose(vec[5], 0.20, rel_tol=1e-9)
    assert math.isclose(vec[6], 0.25, rel_tol=1e-9)
    assert math.isclose(vec[7], 0.75, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (c) missing signal for a span → zero-filled 8-dim cascade.
# ---------------------------------------------------------------------------


def test_missing_signal_zero_filled():
    # structure_out=None -> every span zero-filled.
    cascades = orch._derive_cascades([object(), object()], None)
    assert cascades == [[0.0] * 8, [0.0] * 8]

    # structure ran but emitted no signal for span 1 (only span 0).
    out = _structure_out(
        [
            _role_signal(0, {"paragraph": 1.0}),
            _binary_signal("is_heading", 0, 0.5),
            _binary_signal("table_region", 0, 0.5),
        ]
    )
    cascades = orch._derive_cascades([object(), object()], out)
    assert len(cascades) == 2
    assert len(cascades[0]) == 8
    # span 1 had no role/gate signals -> all zeros (trips Semantic's validator).
    assert cascades[1] == [0.0] * 8


# ---------------------------------------------------------------------------
# (d) descriptive errors fire when the projection is bypassed / mis-mapped.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    len(ROLE_NAMES) != 9,
    reason="requires the 9-role taxonomy so a bypassed fold widens the vector",
)
def test_bypassed_fold_raises_descriptive_dim_error(monkeypatch):
    # Bypass the fold: return the raw (9-dim) distribution unchanged so the
    # concatenated cascade is 11-dim -> the dim check must fire descriptively
    # (this is the exact bare-assert regression we replaced).
    monkeypatch.setattr(orch, "_fold_role_to_legacy", lambda role_full: list(role_full))
    out = _structure_out(
        [
            _role_signal(0, {name: 1.0 / len(ROLE_NAMES) for name in ROLE_NAMES}),
            _binary_signal("is_heading", 0, 0.5),
            _binary_signal("table_region", 0, 0.5),
        ]
    )
    with pytest.raises(ValueError) as exc:
        orch._derive_cascades([object()], out)
    msg = str(exc.value)
    assert "11 != 8" in msg
    assert "ROLE_NAMES has 9 roles" in msg


def test_bad_fold_map_raises_descriptive_error(monkeypatch):
    if len(ROLE_NAMES) != 9:
        pytest.skip("bad-fold-map test requires AXIS-1 roles present")
    # A fold map pointing an AXIS-1 role at a non-existent legacy slot must
    # fail loud rather than silently drop mass.
    monkeypatch.setattr(
        orch,
        "_AXIS1_ROLE_FOLD",
        {"definition_term": "no_such_slot", "definition_def": "paragraph", "caption": "paragraph"},
    )
    out = _structure_out(
        [
            _role_signal(0, {name: 1.0 / len(ROLE_NAMES) for name in ROLE_NAMES}),
            _binary_signal("is_heading", 0, 0.5),
            _binary_signal("table_region", 0, 0.5),
        ]
    )
    with pytest.raises(ValueError) as exc:
        orch._derive_cascades([object()], out)
    assert "no legacy-slot mapping" in str(exc.value)
