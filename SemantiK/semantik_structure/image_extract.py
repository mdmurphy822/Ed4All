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

import logging
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


class FigureRenderError(RuntimeError):
    """A figure Region's bbox could not be rendered to image bytes."""


class _EmptyCropError(ValueError):
    """A figure Region's bbox produced a degenerate / out-of-page crop.

    A distinct class so the render loop can treat a BAD-GEOMETRY crop (an
    inverted or out-of-range bbox — e.g. a tesseract IMAGE-PIXEL-space bbox
    fed to the PDF-point render, or a VLM-fusion-interpolated insert) as a
    per-region SKIP that is NEVER chapter-fatal, while a genuine rasterizer
    failure still honours ``fail_soft``. Mirrors the per-page extraction
    fail-soft idiom (a bad input degrades that unit, it never aborts the doc).
    """


def _degrade_unrenderable_figure(
    region: Any, feature_blocks: list[Any], *, reason: str
) -> Any:
    """Degrade a figure Region whose bbox cannot produce a crop — FOR REAL.

    Downstream consumers must never see the result as a render-expecting
    figure (Stage 6b's captioner fails closed on a payload-less figure):

    * Source FB carries prose text (the live case: a VLM-fused OCR text block
      the ``is_image_block`` head demoted) → RE-TYPE to ``kind="paragraph"``
      so it rides the prose track end-to-end; ``figure_render_degraded``
      records the reason for audit. FB partition untouched (re-tag only, the
      deterministic_structure idiom).
    * No prose form (a synthetic image FB — ``is_image=True``, empty text) →
      keep ``kind="figure"`` but stamp ``figure_render_skipped``; the
      captioner skips it with a warning and the assembler ships the honest
      type-level alt.
    """
    fb = None
    try:
        fb = feature_blocks[region.feature_block_indices[0]]
    except Exception:  # noqa: BLE001 — defensive: bad index was the failure
        pass
    text = ""
    if fb is not None and not getattr(fb, "is_image", False):
        text = (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
    if text:
        return replace(
            region,
            kind="paragraph",
            payload={
                **(region.payload or {}),
                "figure_render_degraded": reason,
            },
        )
    return replace(
        region,
        payload={
            **(region.payload or {}),
            "figure_render_skipped": reason,
        },
    )


def render_figure_regions_to_bytes(
    regions: list[Any],
    feature_blocks: list[Any],
    pdf_path: Path,
    *,
    render_dpi: int = 144,
    fail_soft: bool = False,
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
    fail_soft
        Part F — when True, a per-region render failure (bad bbox, decode
        error) is SKIPPED (the region passes through without
        ``image_png_bytes``) instead of aborting the whole document, so
        one bad bbox on a 39-image page does not lose the other 38. The
        legacy default (False) keeps the loud no-silent-fallback path for
        the legacy ``image_block_demoted`` figures.

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
                    raise _EmptyCropError(f"degenerate bbox {bbox!r}")

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
                    raise _EmptyCropError(
                        f"crop empty after page-clamp: {pil_box!r}"
                    )

                cropped = page_img.crop(pil_box)
                buf = BytesIO()
                cropped.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
                if not png_bytes:
                    raise _EmptyCropError("empty PNG bytes")
            except _EmptyCropError as exc:
                # BAD-GEOMETRY crop → per-region degrade, ALWAYS (independent
                # of ``fail_soft``). A degenerate / out-of-page bbox (a
                # tesseract-pixel-space or VLM-fusion-interpolated bbox that
                # the PDF-point render can't crop) must never be chapter-fatal.
                # The degrade must be REAL: a region left ``kind="figure"``
                # without ``image_png_bytes`` is chapter-fatal ONE STAGE LATER
                # (Stage 6b's captioner fails closed on the missing payload —
                # the live ch09 second-order defect). A region whose source FB
                # carries prose text is RE-TYPED to the prose track
                # (``kind="paragraph"`` via ``dataclasses.replace`` — the
                # deterministic_structure re-tag idiom; FB partition
                # untouched); a region with no prose form (synthetic image FB
                # / empty text) stays a figure but is stamped
                # ``figure_render_skipped`` so Stage 6b skips it and the
                # assembler ships the honest type-level alt.
                logger.warning(
                    "figure region %d (fb=%r) render skipped, degrading: %s",
                    i, region.feature_block_indices, exc,
                )
                out.append(_degrade_unrenderable_figure(
                    region, feature_blocks, reason=str(exc),
                ))
                continue
            except Exception as exc:  # noqa: BLE001 — surface, don't swallow
                if fail_soft:
                    # Part F — degrade this ONE region; keep the rest. Stamp
                    # the skip marker so Stage 6b's captioner (when active)
                    # skips it instead of failing closed on the missing
                    # ``image_png_bytes`` (the same second-order trap the
                    # _EmptyCropError arm closes).
                    out.append(replace(region, payload={
                        **(region.payload or {}),
                        "figure_render_skipped":
                            f"{type(exc).__name__}: {exc}",
                    }))
                    continue
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
