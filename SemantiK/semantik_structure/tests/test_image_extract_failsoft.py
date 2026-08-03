"""Verify Stage-5c render degradation through the Stage-6b caption boundary.

Image-pixel-space OCR boxes can fall outside the PDF-point render. Text-bearing
regions with unusable geometry must join the prose track; synthetic image
regions must retain their figure type with ``figure_render_skipped`` so the
captioner can skip them explicitly. An unmarked payload-less figure remains a
loud contract failure.

The pypdfium2 render + SmolVLM2 boundary are mocked; no real PDF, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantik_structure.figures import captioner as figure_captioner
from semantik_structure.figures import render as render_module
from semantik_structure.figures.captioner import (
    FigureCaptionError,
    caption_figure_regions,
)
from semantik_structure.figures.render import (
    FigureRenderError,
    render_figure_regions_to_bytes,
)
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


# --------------------------------------------------------------------------
# fakes — a small PDF-point page image; a tesseract-space bbox overshoots it
# --------------------------------------------------------------------------


class _FakePIL:
    def __init__(self, size):
        self.size = size

    def crop(self, box):
        return self

    def save(self, buf, **kwargs):
        buf.write(b"PNGBYTES")


class _FakeBitmap:
    def to_pil(self):
        # ~612x792pt page rendered at 144 DPI (scale 2.0) → 1224x1584 px.
        return _FakePIL((1224, 1584))


class _FakePage:
    def render(self, *a, **k):
        return _FakeBitmap()


class _FakeDoc:
    def __init__(self, *a, **k):
        pass

    def __getitem__(self, i):
        return _FakePage()

    def close(self):
        pass


def _fb(bbox, page=1, text="Figure 9.2 fused caption prose", is_image=False):
    raw = RawBlock(
        text=text,
        page=page,
        bbox=tuple(float(v) for v in bbox),
        page_width=1836.0,
        page_height=2376.0,
        source="tesseract",
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
        is_image=is_image,
    )


def _fig_region(fb_index):
    return Region(kind="figure", feature_block_indices=(fb_index,), payload={})


# A Tesseract-pixel bbox valid in its own space (y 1971..1994 on
# a 2376px page) but ×2 = 3942 overshoots the 1584px PDF-point render →
# inverted clamp → empty crop.
_TESS_SPACE_BBOX = (409.0, 1971.0, 515.0, 1994.0)


@pytest.fixture(autouse=True)
def _mock_pdfium(monkeypatch):
    monkeypatch.setattr(render_module.pdfium, "PdfDocument", _FakeDoc)


# --------------------------------------------------------------------------
# renderer: fail-soft + REAL degrade
# --------------------------------------------------------------------------


def test_tesseract_space_bbox_degrades_to_paragraph_not_raises():
    # Geometry degradation applies even when rasterizer failures remain loud.
    fbs = [_fb(_TESS_SPACE_BBOX)]
    regions = [_fig_region(0)]
    out = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    assert len(out) == 1
    # REAL degrade: the text-bearing region is re-typed onto the prose track
    # so no downstream figure consumer (Stage 6b) sees a payload-less figure.
    assert out[0].kind == "paragraph"
    assert "image_png_bytes" not in (out[0].payload or {})
    assert "crop empty" in out[0].payload["figure_render_degraded"]


def test_degenerate_bbox_degrades_not_raises():
    fbs = [_fb((100.0, 200.0, 100.0, 50.0))]  # x0==x1, y0>y1
    regions = [_fig_region(0)]
    out = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    assert out[0].kind == "paragraph"
    assert "image_png_bytes" not in (out[0].payload or {})


def test_synthetic_image_fb_stays_figure_with_skip_marker():
    # A synthetic image FB (is_image=True, empty text) has no prose
    # form — the region stays a figure but carries the skip marker the
    # captioner honours (assembler ships the type-level alt).
    fbs = [_fb(_TESS_SPACE_BBOX, text="", is_image=True)]
    regions = [_fig_region(0)]
    out = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    assert out[0].kind == "figure"
    assert "image_png_bytes" not in (out[0].payload or {})
    assert "figure_render_skipped" in out[0].payload


def test_valid_bbox_still_renders():
    # A sane PDF-point bbox renders normally (guards against over-eager skip).
    fbs = [_fb((100.0, 100.0, 300.0, 400.0))]
    regions = [_fig_region(0)]
    out = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    assert out[0].kind == "figure"
    assert out[0].payload["image_png_bytes"] == b"PNGBYTES"


def test_one_bad_region_does_not_lose_the_good_ones():
    fbs = [
        _fb((100.0, 100.0, 300.0, 400.0)),  # good
        _fb(_TESS_SPACE_BBOX),              # tesseract-space → degrade
        _fb((120.0, 120.0, 320.0, 420.0)),  # good
    ]
    regions = [_fig_region(0), _fig_region(1), _fig_region(2)]
    out = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    assert len(out) == 3
    assert out[0].payload["image_png_bytes"] == b"PNGBYTES"
    assert out[1].kind == "paragraph"
    assert out[2].payload["image_png_bytes"] == b"PNGBYTES"


def test_genuine_rasterizer_failure_still_honors_fail_soft():
    # A non-geometry failure (the renderer raising) stays LOUD when
    # fail_soft=False; fail_soft=True degrades it AND stamps the skip marker
    # so Stage 6b won't fail closed on it later.
    class _BoomDoc(_FakeDoc):
        def __getitem__(self, i):
            raise RuntimeError("pypdfium2 decode exploded")

    import semantik_structure.figures.render as ie

    orig = ie.pdfium.PdfDocument
    ie.pdfium.PdfDocument = _BoomDoc
    try:
        fbs = [_fb((100.0, 100.0, 300.0, 400.0))]
        regions = [_fig_region(0)]
        with pytest.raises(FigureRenderError):
            render_figure_regions_to_bytes(
                regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
            )
        out = render_figure_regions_to_bytes(
            regions, fbs, Path("/nonexistent.pdf"), fail_soft=True
        )
        assert out[0].kind == "figure"
        assert "image_png_bytes" not in (out[0].payload or {})
        assert "figure_render_skipped" in out[0].payload
    finally:
        ie.pdfium.PdfDocument = orig


# --------------------------------------------------------------------------
# caption stage: a deliberately-skipped figure is never chapter-fatal;
# a genuinely-missing payload still raises (no-silent-fallback preserved)
# --------------------------------------------------------------------------


@pytest.fixture
def _mock_vlm(monkeypatch):
    monkeypatch.setattr(
        figure_captioner,
        "_run_smolvlm_caption",
        lambda image_bytes, prompt, *, max_new_tokens: "a mocked caption",
    )
    monkeypatch.setattr(
        figure_captioner, "_build_caption_capture", lambda: None
    )


def test_caption_stage_skips_marked_payloadless_figure(_mock_vlm):
    marked = Region(
        kind="figure",
        feature_block_indices=(5,),
        payload={"figure_render_skipped": "crop empty after page-clamp"},
    )
    out = caption_figure_regions([marked], run_extended=False)
    assert len(out) == 1
    # Skipped, not raised; no alt_text minted — the assembler's type-level
    # alt guard owns the honest fallback.
    assert "alt_text" not in (out[0].payload or {})


def test_caption_stage_still_raises_on_unmarked_payloadless_figure(_mock_vlm):
    unmarked = Region(kind="figure", feature_block_indices=(5,), payload={})
    with pytest.raises(FigureCaptionError):
        caption_figure_regions([unmarked], run_extended=False)


def test_caption_stage_captions_good_figures_around_a_skipped_one(_mock_vlm):
    good = Region(
        kind="figure",
        feature_block_indices=(1,),
        payload={"image_png_bytes": b"PNGBYTES"},
    )
    marked = Region(
        kind="figure",
        feature_block_indices=(2,),
        payload={"figure_render_skipped": "degenerate bbox"},
    )
    out = caption_figure_regions([good, marked], run_extended=False)
    assert out[0].payload["alt_text"] == "a mocked caption"
    assert "alt_text" not in (out[1].payload or {})


# --------------------------------------------------------------------------
# Stage 5c skip → Stage 6b caption integration contract.
# --------------------------------------------------------------------------


def test_render_skip_flows_through_caption_stage_without_raising(_mock_vlm):
    fbs = [
        _fb((100.0, 100.0, 300.0, 400.0)),               # renders fine
        _fb(_TESS_SPACE_BBOX),                           # prose-track degrade
        _fb(_TESS_SPACE_BBOX, text="", is_image=True),   # skip, stays figure
    ]
    regions = [_fig_region(0), _fig_region(1), _fig_region(2)]

    rendered = render_figure_regions_to_bytes(
        regions, fbs, Path("/nonexistent.pdf"), fail_soft=False
    )
    # The full Stage-6b pass over the post-5c list must not raise.
    out = caption_figure_regions(rendered, run_extended=False)

    assert out[0].kind == "figure"
    assert out[0].payload["alt_text"] == "a mocked caption"
    assert out[1].kind == "paragraph"          # real degrade: prose track
    assert "alt_text" not in (out[1].payload or {})
    assert out[2].kind == "figure"             # marker-skipped figure
    assert "alt_text" not in (out[2].payload or {})
