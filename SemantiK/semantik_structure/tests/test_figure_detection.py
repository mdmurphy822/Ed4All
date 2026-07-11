"""Unit tests — Part F figure detection stages B/C/D (no GPU, no PDF IO).

Covers:
  * region_detection image candidates + spacer filter (Y-flip is exercised
    via the extract-side test below using a synthetic ``shared``).
  * synthetic image FeatureBlock interleave in reading order.
  * figure-region formation + Stage-5 coverage invariant.
  * flag-OFF byte-stability for the figure path.
"""

from __future__ import annotations

from semantik_structure.features import (
    _interleave_image_feature_blocks,
    featurize_with_regions,
)
from semantik_structure.region_detection import (
    ImageCandidate,
    detect_image_region_candidates,
)
from semantik_structure.structure_graph import build_structure_graph
from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.types import CouncilState
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _text_fb(text: str, *, page: int, y0: float, x0: float = 0.0) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(x0, y0, x0 + 50.0, y0 + 12.0),
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


def _shared_with_images(images):
    """A minimal ``shared`` dict carrying one page with an ``images`` list."""
    return {
        "pages": [
            {
                "page_num": 1,
                "width": 612.0,
                "height": 792.0,
                "merged": {"text_blocks": []},
                "pdfplumber": {"tables": []},
                "images": images,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Stage B — image candidate detection + spacer filter.
# ---------------------------------------------------------------------------


def test_detect_image_candidates_basic():
    shared = _shared_with_images(
        [{"bbox": [50.0, 100.0, 250.0, 300.0], "px_size": [400, 400], "index": 0}]
    )
    cands = detect_image_region_candidates(shared)
    assert len(cands) == 1
    c = cands[0]
    assert isinstance(c, ImageCandidate)
    assert c.kind == "figure"
    assert c.pages == [1]
    assert c.px_size == (400, 400)
    assert c.bbox == (50.0, 100.0, 250.0, 300.0)


def test_spacer_image_filtered():
    # A 2px-tall hairline rule and a 1xN spacer are both rejected.
    shared = _shared_with_images(
        [
            {"bbox": [0.0, 0.0, 500.0, 1.0], "px_size": [500, 1], "index": 0},
            {"bbox": [0.0, 5.0, 4.0, 9.0], "px_size": [4, 4], "index": 1},
        ]
    )
    assert detect_image_region_candidates(shared) == []


def test_no_images_key_returns_empty():
    shared = {"pages": [{"page_num": 1, "merged": {"text_blocks": []}}]}
    assert detect_image_region_candidates(shared) == []


# ---------------------------------------------------------------------------
# Stage C — synthetic FB interleave in reading order.
# ---------------------------------------------------------------------------


def test_synthetic_fb_interleaved_in_reading_order():
    fbs = [
        _text_fb("top", page=1, y0=10.0),
        _text_fb("bottom", page=1, y0=500.0),
    ]
    # Image candidate sits between the two text blocks (y0=200).
    cand = ImageCandidate(
        kind="figure",
        bbox=(50.0, 200.0, 250.0, 400.0),
        pages=[1],
        px_size=(400, 400),
        image_index=0,
    )
    merged = _interleave_image_feature_blocks(fbs, [cand])
    assert len(merged) == 3
    # Order: top text (y=10), image (y=200), bottom text (y=500).
    assert merged[0].raw.text == "top"
    assert merged[1].is_image is True
    assert merged[1].raw.source == "pypdfium2:image"
    assert merged[2].raw.text == "bottom"
    # The candidate's member index points at the synthetic FB (index 1).
    assert cand.member_block_indices == [1]


def test_synthetic_fb_degenerate_bbox_skipped():
    fbs = [_text_fb("only", page=1, y0=10.0)]
    cand = ImageCandidate(
        kind="figure",
        bbox=(50.0, 50.0, 50.0, 50.0),  # zero area
        pages=[1],
        px_size=(10, 10),
        image_index=0,
    )
    merged = _interleave_image_feature_blocks(fbs, [cand])
    assert len(merged) == 1  # no synthetic FB created
    assert cand.member_block_indices == []


# ---------------------------------------------------------------------------
# Stage D — figure-region formation + coverage invariant.
# ---------------------------------------------------------------------------


def test_figure_region_formed_and_coverage_invariant_holds():
    # Two text FBs + one synthetic image FB interleaved.
    fbs = [
        _text_fb("intro", page=1, y0=10.0),
        _text_fb("Figure 1: a widget", page=1, y0=500.0),
    ]
    cand = ImageCandidate(
        kind="figure",
        bbox=(50.0, 200.0, 250.0, 400.0),
        pages=[1],
        px_size=(400, 400),
        image_index=0,
    )
    merged = _interleave_image_feature_blocks(fbs, [cand])
    state = CouncilState(outputs={})
    regions = [cand]  # the only Stage-4 region
    decisions = arbitrate(state, regions)
    assert decisions[0].route == "figure"

    out = build_structure_graph(state, merged, regions, decisions)
    figures = [r for r in out if r.kind == "figure"]
    assert len(figures) == 1
    fig = figures[0]
    assert fig.provenance.get("pass") == "figure_image_candidate"
    # The figure claims the synthetic image FB (index 1 after interleave).
    assert fig.feature_block_indices == (1,)
    # Caption FB ("Figure 1: ...") is recorded (index 2 in merged stream).
    assert fig.payload.get("caption_fb_index") == 2

    # Coverage invariant: every FB index appears in exactly one region.
    seen = [i for r in out for i in r.feature_block_indices]
    assert sorted(seen) == list(range(len(merged)))


def test_figure_region_no_caption_when_none_below():
    fbs = [_text_fb("intro", page=1, y0=10.0)]
    cand = ImageCandidate(
        kind="figure",
        bbox=(50.0, 200.0, 250.0, 400.0),
        pages=[1],
        px_size=(400, 400),
        image_index=0,
    )
    merged = _interleave_image_feature_blocks(fbs, [cand])
    state = CouncilState(outputs={})
    decisions = arbitrate(state, [cand])
    out = build_structure_graph(state, merged, [cand], decisions)
    fig = [r for r in out if r.kind == "figure"][0]
    assert fig.payload.get("caption_fb_index") is None
    seen = [i for r in out for i in r.feature_block_indices]
    assert sorted(seen) == list(range(len(merged)))


# ---------------------------------------------------------------------------
# Flag-OFF byte-stability — no images key, no figure regions.
# ---------------------------------------------------------------------------


def test_featurize_with_regions_flag_off_byte_stable(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    # shared with NO images key (flag off → extract emits no images list).
    shared = {
        "pages": [
            {
                "page_num": 1,
                "width": 612.0,
                "height": 792.0,
                "merged": {
                    "text_blocks": [
                        {
                            "bbox": [0.0, 10.0, 50.0, 22.0],
                            "text": "hello",
                            "font_size": 10.0,
                            "font_name": "Times",
                        }
                    ]
                },
                "pdfplumber": {"tables": []},
                "pikepdf": {"widgets": []},
            }
        ]
    }
    fs = featurize_with_regions(shared)
    assert fs.image_candidates == []
    # No synthetic image FBs in the stream.
    assert all(not fb.is_image for fb in fs.feature_blocks)
