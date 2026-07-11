"""Regression — the Stage-5c figure renderer FAIL-SOFTs a bad/empty crop AND
the degrade is REAL end-to-end (Stage 6b never fails closed on it).

Live-fire defect pair (full-cascade ch09 scan, VLM fusion on):

1. A figure Region minted over VLM-fused tesseract blocks carries a bbox in
   tesseract IMAGE-PIXEL space (page ~1836x2376 at OCR scale 3.0). The
   renderer assumes PDF-POINT space, so ``bbox * scale`` overshoots the
   smaller PDF-point-rendered page and the page-clamp INVERTS the crop box
   (``y0 > y1``) → the original code raised a chapter-fatal
   ``FigureRenderError``.
2. SECOND-ORDER: a skip that merely omitted ``image_png_bytes`` left the
   region ``kind="figure"``, and Stage 6b's captioner fails closed on a
   payload-less figure (``FigureCaptionError``) — also chapter-fatal.

The degrade must therefore be REAL: a text-bearing source FB → the region is
RE-TYPED to ``kind="paragraph"`` (prose track end-to-end); a synthetic image
FB (no prose form) → stays a figure but stamped ``figure_render_skipped`` so
the captioner skips it with a warning. A payload-less figure WITHOUT a skip
marker still raises (genuine "Stage 5c didn't run" stays loud).

The pypdfium2 render + SmolVLM2 boundary are mocked; no real PDF, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantik_structure import figure_captioner, image_extract
from semantik_structure.figure_captioner import (
    FigureCaptionError,
    caption_figure_regions,
)
from semantik_structure.image_extract import (
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


# The live-fire tesseract-pixel bbox: valid IN ITS OWN SPACE (y 1971..1994 on
# a 2376px page) but ×2 = 3942 overshoots the 1584px PDF-point render →
# inverted clamp → empty crop.
_TESS_SPACE_BBOX = (409.0, 1971.0, 515.0, 1994.0)


@pytest.fixture(autouse=True)
def _mock_pdfium(monkeypatch):
    monkeypatch.setattr(image_extract.pdfium, "PdfDocument", _FakeDoc)


# --------------------------------------------------------------------------
# renderer: fail-soft + REAL degrade
# --------------------------------------------------------------------------


def test_tesseract_space_bbox_degrades_to_paragraph_not_raises():
    # fail_soft=False is the legacy demote path that used to abort the chapter.
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
    # A Part-F synthetic image FB (is_image=True, empty text) has no prose
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

    import semantik_structure.image_extract as ie

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
# end-to-end-ish: Stage 5c skip → Stage 6b caption, the exact live-fire chain
# (the prior test stopped at the renderer boundary — this is why it leaked)
# --------------------------------------------------------------------------


def test_render_skip_flows_through_caption_stage_without_raising(_mock_vlm):
    fbs = [
        _fb((100.0, 100.0, 300.0, 400.0)),               # renders fine
        _fb(_TESS_SPACE_BBOX),                           # live-fire degrade
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
