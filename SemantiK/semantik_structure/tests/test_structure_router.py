"""Genre/confidence-gated STRUCTURE ROUTER (structure_router.py).

Covers the SEMANTIK_STRUCTURE_ROUTER opt-in:
  * resolver parse-with-fallback (default OFF; truthy / falsey / garbage);
  * the detector FIRING off-domain (few real headings + high apparatus ratio +
    high VLM divergence) and NOT firing on a healthy on-domain document;
  * the natural NO-OP without any VLM heading hint (dual-gate);
  * the switch PROMOTING VLM headings + DEMOTING furniture headings ONLY when
    the detector verdicts VLM-authoritative — and being byte-identical (same
    region objects) when the detector keeps the council authoritative.

All CPU-only, synthetic hand-built Region / FeatureBlock objects — no council,
no model, no PDF, no network. (`.venv` lacks pytest; run under system
`python3 -m pytest`.)
"""
from __future__ import annotations

import pytest

from semantik_structure.structure_router import (
    RouterDecision,
    apply_structure_router,
    resolve_structure_router_mode,
    route_decision,
)
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock

_ENV = "SEMANTIK_STRUCTURE_ROUTER"


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    yield


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(page: int, *, text: str = "x", vlm_heading_level: int | None = None) -> FeatureBlock:
    """A synthetic FeatureBlock, optionally carrying a whole-block VLM heading hint."""
    hint = None
    if vlm_heading_level is not None:
        hint = {
            "kind": "heading",
            "level": vlm_heading_level,
            "marker": None,
            "coverage": "whole_block",
        }
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 100.0, 12.0),
        page_width=612.0,
        page_height=792.0,
        font_size=11.0,
        vlm_hint=hint,
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
        vlm_hint=hint,
    )


def _heading(fb_idx: int, text: str, *, conf: float = 0.9) -> Region:
    return Region(
        kind="heading",
        feature_block_indices=(fb_idx,),
        payload={"text": text, "level_hint": 2, "confidence": conf},
        provenance={"pass": "heading", "is_heading_conf": conf},
    )


def _paragraph(fb_idx: int, text: str) -> Region:
    return Region(
        kind="paragraph",
        feature_block_indices=(fb_idx,),
        payload={"text": text},
    )


# ---------------------------------------------------------------------------
# Corpus builders — a healthy on-domain doc and a collapsed off-domain doc.
# ---------------------------------------------------------------------------


def _healthy_on_domain():
    """Council found clean section headings; VLM agrees → keep BERT."""
    regions = [
        _heading(0, "1.1 Whole Numbers", conf=0.95),
        _paragraph(1, "The set of whole numbers begins at zero and continues."),
        _heading(2, "1.2 Rounding", conf=0.93),
        _paragraph(3, "Rounding approximates a value to a given place."),
        _heading(4, "1.3 Estimation", conf=0.9),
        _paragraph(5, "Estimation gives a quick approximate answer."),
    ]
    # VLM corroborates the SAME headings the council already typed.
    fbs = [
        _fb(1, vlm_heading_level=2),
        _fb(1),
        _fb(2, vlm_heading_level=2),
        _fb(2),
        _fb(3, vlm_heading_level=2),
        _fb(3),
    ]
    return regions, fbs


def _collapsed_off_domain():
    """Council collapse: 2 real headings buried under furniture; VLM diverges.

    Council typed a pile of pedagogical-label / apparatus FURNITURE as headings
    and MISSED the real sections, which land as prose. The VLM marks the real
    sections (and only those) as whole-block headings on several pages.
    """
    regions = [
        # Furniture the council mis-typed as headings (the off-domain FP pile).
        _heading(0, "Example 1", conf=0.4),
        _heading(1, "Try It", conf=0.35),
        _heading(2, "Solution", conf=0.5),
        _heading(3, "Step 2", conf=0.45),
        _heading(4, "Key Terms", conf=0.4),
        _heading(5, "Chapter Review", conf=0.42),
        # Two real section titles the council DEMOTED to prose (recall collapse).
        _paragraph(6, "Anatomy of the Circulatory System"),
        _paragraph(7, "Regulation of Blood Pressure"),
        # Ordinary body prose (no VLM hint).
        _paragraph(8, "Blood carries oxygen from the lungs to the tissues."),
    ]
    fbs = [
        _fb(1),
        _fb(1),
        _fb(2),
        _fb(2),
        _fb(3),
        _fb(4),
        # The two real sections carry whole-block VLM heading hints across pages.
        _fb(2, vlm_heading_level=2),
        _fb(5, vlm_heading_level=2),
        _fb(2),
    ]
    # A third VLM heading so n_vlm >= _MIN_VLM_HEADINGS (3): stamp one on the
    # last prose block too? Keep it clean — add one more VLM-only heading region.
    regions.append(_paragraph(9, "The Heart as a Pump"))
    fbs.append(_fb(3, vlm_heading_level=2))
    return regions, fbs


# ---------------------------------------------------------------------------
# Resolver.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("garbage", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        ("On", True),
    ],
)
def test_resolver_parse_with_fallback(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)
    assert resolve_structure_router_mode() is expected


# ---------------------------------------------------------------------------
# Detector.
# ---------------------------------------------------------------------------


def test_detector_on_domain_keeps_bert_authoritative():
    regions, fbs = _healthy_on_domain()
    d = route_decision(regions, fbs)
    assert isinstance(d, RouterDecision)
    assert d.bert_authoritative is True
    # Healthy: low apparatus ratio, VLM agrees (no divergence).
    assert d.apparatus_ratio < 0.5
    assert d.divergence < 0.5


def test_detector_off_domain_flips_to_vlm_authoritative():
    regions, fbs = _collapsed_off_domain()
    d = route_decision(regions, fbs)
    assert d.vlm_present is True
    assert d.bert_authoritative is False
    # The three off-domain fingerprints all surface.
    assert d.apparatus_ratio >= 0.5
    assert d.divergence >= 0.5
    assert d.n_vlm_headings >= 3
    assert "vlm_divergence" in d.reasons


def test_detector_no_vlm_is_bert_authoritative():
    """No VLM heading hint anywhere → natural no-op (keep BERT) even when the
    council heading set is furniture-polluted."""
    regions, fbs = _collapsed_off_domain()
    # Strip every VLM hint from the FeatureBlocks.
    stripped = [_fb(fb.raw.page, text=fb.raw.text) for fb in fbs]
    d = route_decision(regions, stripped)
    assert d.vlm_present is False
    assert d.bert_authoritative is True


# ---------------------------------------------------------------------------
# The switch.
# ---------------------------------------------------------------------------


def test_switch_noop_when_bert_authoritative():
    regions, fbs = _healthy_on_domain()
    out, diag = apply_structure_router(regions, fbs)
    # Byte-identical: the same region objects are returned unchanged.
    assert out is regions
    assert diag["promoted"] == 0
    assert diag["demoted"] == 0
    assert diag["decision"]["bert_authoritative"] is True


def test_switch_noop_without_vlm():
    regions, fbs = _collapsed_off_domain()
    stripped = [_fb(fb.raw.page, text=fb.raw.text) for fb in fbs]
    out, diag = apply_structure_router(regions, stripped)
    assert out is regions
    assert diag["promoted"] == 0
    assert diag["demoted"] == 0


def test_switch_promotes_vlm_headings_and_demotes_furniture_off_domain():
    regions, fbs = _collapsed_off_domain()
    out, diag = apply_structure_router(regions, fbs)

    assert diag["decision"]["bert_authoritative"] is False
    assert not diag["reverted_for_invariant"]
    # The FB partition is immutable — same count, same total FB multiset.
    assert len(out) == len(regions)
    assert sum(len(r.feature_block_indices) for r in out) == sum(
        len(r.feature_block_indices) for r in regions
    )

    # The two prose regions the VLM marked as whole-block headings are promoted.
    promoted_texts = {
        r.payload.get("text")
        for r in out
        if r.kind == "heading"
        and (r.payload.get("structure_router") or {}).get("promoted") == "vlm_heading"
    }
    assert "Anatomy of the Circulatory System" in promoted_texts
    assert "Regulation of Blood Pressure" in promoted_texts
    assert diag["promoted"] >= 2

    # Furniture council headings the VLM did not corroborate are demoted to <p>.
    demoted_texts = {
        r.payload.get("text")
        for r in out
        if r.kind == "paragraph"
        and (r.payload.get("structure_router") or {}).get("demoted")
        == "apparatus_furniture"
    }
    assert "Example 1" in demoted_texts
    assert "Try It" in demoted_texts
    assert diag["demoted"] >= 2


def test_switch_preserves_all_text_tokens_off_domain():
    """Re-type is partition-immutable: every FB's owning region stays non-drop,
    so token conservation holds and the pass is never reverted."""
    regions, fbs = _collapsed_off_domain()
    out, diag = apply_structure_router(regions, fbs)
    assert diag["reverted_for_invariant"] is False
    # No region was dropped to metadata_drop.
    assert all(r.kind != "metadata_drop" for r in out)
