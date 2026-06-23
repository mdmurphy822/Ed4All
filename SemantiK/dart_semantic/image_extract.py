"""Stage 5c — render figure Region bboxes to PNG bytes (no model, CPU only).

Plans/09 §1. Image pixels never reach Stage 6 because ``extract.py`` is
text-only. This module bridges the gap. After Stage 5
(:func:`build_structure_graph`) emits figure Regions from the Stage-4
``image_block_demoted`` flags, render each region's source FeatureBlock bbox
from the PDF page via ``pypdfium2`` and attach the PNG bytes to the Region
payload under ``"image_png_bytes"``.

Stage 5c sits between Stage 5b (GLM-OCR table enrichment) and Stage 6
(``run_qwen_specialists`` / ``enrich``). Unlike Stage 5b it is NOT opt-in:
without pixels, the alt-text path can't function. The function is a no-op
when no figure Regions are present (no PDF open, no overhead).

PDF backend: ``pypdfium2`` (Apache-2 / BSD-3, ``feedback_license_policy``) —
a permissively-licensed, commercial-safe rasterizer.

No silent fallback (``feedback_no_silent_fallbacks``): a bbox that fails to
render raises :class:`FigureRenderError` so the cascade surfaces the failure
rather than producing an unalt'd figure downstream.

Bbox unit assumption: PDF-point coordinates with top-left origin (pdfplumber
convention), as carried in ``FeatureBlock.raw.bbox``. Tesseract-OCR-sourced
pixels would need pre-conversion before this stage (not yet wired).
"""
from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium


class FigureRenderError(RuntimeError):
    """A figure Region's bbox could not be rendered to image bytes."""


def render_figure_regions_to_bytes(
    regions: list[Any],
    feature_blocks: list[Any],
    pdf_path: Path,
    *,
    render_dpi: int = 144,
) -> list[Any]:
    """Stage 5c — attach ``image_png_bytes`` to every figure Region.

    For each Region with ``kind == "figure"``, look up its first source
    FeatureBlock via ``feature_block_indices[0]``, render the FB's PDF page
    once via pypdfium2 (cached per page for multi-figure pages), crop the
    bbox via PIL, encode PNG, and return a NEW Region (the class is frozen)
    with ``payload["image_png_bytes"]`` + ``payload["image_render_dpi"]``
    set. Non-figure Regions pass through unchanged.

    Parameters
    ----------
    regions
        Stage-5 Region list (typically from ``build_structure_graph``).
    feature_blocks
        Shared FeatureBlock list (carries ``raw.page`` 1-indexed and
        ``raw.bbox`` in PDF-point units, top-left origin).
    pdf_path
        Source PDF for the cascade run.
    render_dpi
        Render resolution. 144 trades off chart legibility vs payload size
        (~1.5× a 96 DPI screen capture).

    Returns
    -------
    list of Region — same length and order as the input; figure entries are
    replaced copies, non-figures returned as-is.

    Raises
    ------
    FigureRenderError
        Any per-region rendering failure (bad page index, invalid bbox,
        pypdfium2/PIL error). Per the no-silent-fallback discipline, we
        surface rather than skip.
    """
    fig_idxs = {i for i, r in enumerate(regions)
                if getattr(r, "kind", None) == "figure"}
    if not fig_idxs:
        return regions

    scale = render_dpi / 72.0  # PDF native: 72 dpi
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        # Render each touched page at most once.
        page_cache: dict[int, Any] = {}
        out: list[Any] = []
        for i, region in enumerate(regions):
            if i not in fig_idxs:
                out.append(region)
                continue
            try:
                fb_idx = region.feature_block_indices[0]
                fb = feature_blocks[fb_idx]
                page_idx_0based = int(fb.raw.page) - 1
                bbox = tuple(fb.raw.bbox)
                if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError(f"degenerate bbox {bbox!r}")

                if page_idx_0based not in page_cache:
                    page = doc[page_idx_0based]
                    page_cache[page_idx_0based] = page.render(
                        scale=scale,
                    ).to_pil()
                page_img = page_cache[page_idx_0based]

                # bbox is top-left origin (pdfplumber); PIL is top-left too.
                x0, y0_top, x1, y1_bottom = bbox
                pil_box = (
                    int(x0 * scale), int(y0_top * scale),
                    int(x1 * scale), int(y1_bottom * scale),
                )
                w, h = page_img.size
                pil_box = (
                    max(0, pil_box[0]), max(0, pil_box[1]),
                    min(w, pil_box[2]), min(h, pil_box[3]),
                )
                if pil_box[2] <= pil_box[0] or pil_box[3] <= pil_box[1]:
                    raise ValueError(f"crop empty after page-clamp: {pil_box!r}")

                cropped = page_img.crop(pil_box)
                buf = BytesIO()
                cropped.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
                if not png_bytes:
                    raise ValueError("empty PNG bytes")
            except Exception as exc:  # noqa: BLE001 — surface, don't swallow
                raise FigureRenderError(
                    f"failed to render figure region {i} "
                    f"(fb={region.feature_block_indices!r}) from "
                    f"{pdf_path}: {type(exc).__name__}: {exc}"
                ) from exc

            new_payload = {
                **(region.payload or {}),
                "image_png_bytes": png_bytes,
                "image_render_dpi": render_dpi,
            }
            out.append(replace(region, payload=new_payload))
        return out
    finally:
        doc.close()
