"""SEMANTIK_SCAN_LANE_DEBERT — OCR/scan-lane BERT de-poisoning gate (task #43).

No real course / corpus text: a tiny synthesized FeatureBlock + CouncilState set
mimicking OCR (tesseract-provenance) input. Asserts (a) flag off → byte-identical
current behavior (no mask, no provenance key); (b) flag on + scan lane → the
council structural heads are non-binding (masked out) and every provenance region
is ``typing_authority`` ``deterministic``/``vlm`` — ZERO ``bert``; (c) a
born-digital lane is unaffected by the flag.
"""

from __future__ import annotations

import dataclasses

import pytest

from semantik_structure.cascade import _build_region_provenance
from semantik_structure.council.types import BertOutput, CouncilState, TypedSignal
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock
from semantik_structure import scan_lane
from semantik_structure.scan_lane import (
    NEUTRALIZED_HEADS,
    is_ocr_scan_lane,
    mask_council_state,
    resolve_scan_lane_debert_mode,
    scan_lane_debert_active,
    typing_authority_for_payload,
)


# ---------------------------------------------------------------------------
# Synthetic builders (NO course/corpus text).
# ---------------------------------------------------------------------------


def _fb(text: str, *, source: str, page: int = 1, is_image: bool = False) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 50.0, 12.0),
        page_width=612.0,
        page_height=792.0,
        source=source,
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
        provenance=source,
        is_image=is_image,
    )


def _sig(head: str, idx: int, label: str, conf: float) -> TypedSignal:
    return TypedSignal(
        head_name=head, region_id=idx, top_k_labels=[label], top_k_confidences=[conf]
    )


def _council_state() -> CouncilState:
    """A state carrying a signal for every neutralized head + one text-repair
    head (hyphen_repair) that MUST survive the mask."""
    structure_sigs = [
        _sig("structural_role", 0, "heading", 0.9),
        _sig("pedagogical_role", 0, "example", 0.8),
        _sig("is_heading", 0, "heading", 0.9),
        _sig("is_heading_calibrated", 0, "heading", 0.9),
        _sig("table_region", 0, "table_region", 0.9),
        _sig("is_image_block", 0, "image_block", 0.9),
        _sig("list_nesting", 0, "depth0", 0.9),
    ]
    return CouncilState(
        outputs={
            "structure": BertOutput(bert_name="structure", signals=structure_sigs),
            "semantic": BertOutput(
                bert_name="semantic", signals=[_sig("doc_role", 0, "footer", 0.9)]
            ),
            "merge_or_split": BertOutput(
                bert_name="merge_or_split",
                signals=[
                    _sig("same_logical_block", 0, "same", 0.9),
                    _sig("hyphen_repair", 0, "join", 0.9),  # text head — survives
                ],
            ),
        }
    )


def _regions() -> list[Region]:
    return [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "alpha"}),
        Region(
            kind="heading",
            feature_block_indices=(1,),
            payload={"text": "beta", "level_hint": 2, "vlm_corroborated": True},
        ),
    ]


# ---------------------------------------------------------------------------
# Resolver + lane detection.
# ---------------------------------------------------------------------------


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv(scan_lane.SEMANTIK_SCAN_LANE_DEBERT_ENV, raising=False)
    assert resolve_scan_lane_debert_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_resolver_truthy(monkeypatch, val):
    monkeypatch.setenv(scan_lane.SEMANTIK_SCAN_LANE_DEBERT_ENV, val)
    assert resolve_scan_lane_debert_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "off", "", "garbage"])
def test_resolver_falsey_and_garbage(monkeypatch, val):
    monkeypatch.setenv(scan_lane.SEMANTIK_SCAN_LANE_DEBERT_ENV, val)
    assert resolve_scan_lane_debert_mode() is False


def test_is_ocr_scan_lane_tesseract():
    fbs = [_fb("a", source="tesseract"), _fb("b", source="tesseract")]
    assert is_ocr_scan_lane(fbs) is True


def test_is_ocr_scan_lane_born_digital():
    fbs = [_fb("a", source="pdfplumber+pypdfium2"), _fb("b", source="pypdfium2")]
    assert is_ocr_scan_lane(fbs) is False


def test_is_ocr_scan_lane_excludes_synthetic_images():
    # A born-digital doc with an image FB is still NOT the scan lane.
    fbs = [_fb("a", source="pypdfium2"), _fb("", source="pypdfium2:image", is_image=True)]
    assert is_ocr_scan_lane(fbs) is False


def test_is_ocr_scan_lane_empty():
    assert is_ocr_scan_lane([]) is False


# ---------------------------------------------------------------------------
# mask_council_state.
# ---------------------------------------------------------------------------


def test_mask_drops_every_neutralized_head_keeps_text_head():
    state = _council_state()
    masked = mask_council_state(state)

    def _heads(s, bert):
        out = s.outputs.get(bert)
        return {sig.head_name for sig in (out.signals if out else [])}

    # Every neutralized (bert, head) pair is gone.
    for bert, head in NEUTRALIZED_HEADS:
        assert head not in _heads(masked, bert), (bert, head)
    # The non-enumerated text-repair head survives.
    assert "hyphen_repair" in _heads(masked, "merge_or_split")
    # Original state is untouched (pure transform).
    assert "structural_role" in _heads(state, "structure")


def test_mask_noop_on_non_state():
    sentinel = object()
    assert mask_council_state(sentinel) is sentinel


# ---------------------------------------------------------------------------
# typing_authority stamping in _build_region_provenance.
# ---------------------------------------------------------------------------


def test_provenance_no_key_when_inactive_byte_identical():
    """(a) flag off / born-digital → NO typing_authority key (byte-identical)."""
    regions = _regions()
    fbs = [_fb("alpha", source="pypdfium2"), _fb("beta", source="pypdfium2")]
    prov = _build_region_provenance(
        [0, 1], regions, fbs, {}, typing_authority_active=False
    )
    assert all("typing_authority" not in e for e in prov)


def test_provenance_zero_bert_when_active():
    """(b) flag on + scan lane → every region deterministic/vlm, ZERO bert."""
    regions = _regions()
    fbs = [_fb("alpha", source="tesseract"), _fb("beta", source="tesseract")]
    prov = _build_region_provenance(
        [0, 1], regions, fbs, {}, typing_authority_active=True
    )
    authorities = [e["typing_authority"] for e in prov]
    assert all(a in {"deterministic", "vlm"} for a in authorities)
    assert authorities.count("bert") == 0
    # The vlm_corroborated region reads as vlm; the plain one as deterministic.
    assert set(authorities) == {"deterministic", "vlm"}


def test_typing_authority_for_payload():
    assert typing_authority_for_payload(None) == "deterministic"
    assert typing_authority_for_payload({}) == "deterministic"
    assert typing_authority_for_payload({"vlm_corroborated": True}) == "vlm"


# ---------------------------------------------------------------------------
# End-to-end gating (resolver + lane) — (b) vs (c).
# ---------------------------------------------------------------------------


def test_active_only_on_scan_lane_with_flag(monkeypatch):
    scan_fbs = [_fb("a", source="tesseract")]
    digital_fbs = [_fb("a", source="pypdfium2")]

    # (c) born-digital lane is unaffected even with the flag on.
    monkeypatch.setenv(scan_lane.SEMANTIK_SCAN_LANE_DEBERT_ENV, "1")
    assert scan_lane_debert_active(digital_fbs) is False
    # (b) scan lane + flag → active.
    assert scan_lane_debert_active(scan_fbs) is True
    # flag off → never active, even on the scan lane.
    monkeypatch.setenv(scan_lane.SEMANTIK_SCAN_LANE_DEBERT_ENV, "0")
    assert scan_lane_debert_active(scan_fbs) is False
