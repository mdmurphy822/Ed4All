"""Stage 1: Extraction — thin adapter over extract_shared.py.

The heavy lifting (running all extractors, building the shared JSON,
reconciling blocks) lives in extract_shared.py. This module's job is to
keep the simple `extract_blocks(pdf_path) -> list[RawBlock]` interface
downstream stages have always used, so the pipeline doesn't need to
change when the extractor stack evolves.

For callers that want the full multi-extractor view (e.g. training-data
generators that benefit from cross-extractor disagreement signals), use
`extract_shared(pdf_path)` directly.
"""
from __future__ import annotations

from pathlib import Path

from .extract_shared import EncryptedPDFError, extract_shared  # re-exported
from .types import RawBlock


def extract_blocks(pdf_path: Path) -> list[RawBlock]:
    """Return a flat list of RawBlocks across all pages.

    Internally:
      - Runs every licence-compatible extractor (pikepdf + pypdfium2 +
        pdfplumber + Tesseract-when-needed) via extract_shared.
      - Flattens the merged text_blocks list per page into RawBlock.
      - Preserves which extractor(s) contributed each block in RawBlock.source
        so downstream features can flag the extractor profile.
    """
    shared = extract_shared(pdf_path)
    return blocks_from_shared(shared)


def blocks_from_shared(shared: dict) -> list[RawBlock]:
    """Flatten the shared JSON's merged per-page text_blocks into RawBlocks."""
    out: list[RawBlock] = []
    for page in shared.get("pages", []):
        page_num = page.get("page_num", 1)
        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))
        merged = page.get("merged", {})
        blocks = merged.get("text_blocks", [])
        # When Tesseract is the source, bboxes are in image-pixel space; the
        # feature normalizer in stage 2 computes relative sizes so the unit
        # mismatch is absorbed. We still prefer reporting the Tesseract
        # image dims in that case so page-fraction flags (top, centered) are
        # computed correctly.
        use_tess_dims = (page.get("tesseract_width") is not None
                         and all("tesseract" in b.get("provenance", [])
                                 for b in blocks))
        if use_tess_dims:
            page_w = float(page["tesseract_width"])
            page_h = float(page["tesseract_height"])

        for b in blocks:
            bbox = tuple(float(x) for x in b["bbox"])
            prov = tuple(b.get("provenance", []))
            # source is a compact tag for downstream consumers (and training
            # signal). The first provenance wins as the short tag; the full
            # provenance is concatenated with '+'.
            source_tag = "+".join(prov) if prov else "unknown"
            out.append(RawBlock(
                text=b["text"],
                page=page_num,
                bbox=bbox,
                page_width=page_w,
                page_height=page_h,
                font_size=b.get("font_size"),
                font_name=b.get("font_name"),
                is_bold=b.get("is_bold"),
                is_italic=b.get("is_italic"),
                confidence=float(b.get("confidence", 1.0)),
                source=source_tag,
                # P2 VLM hint carriage. The merged block dict gains a
                # ``vlm_hint`` key ONLY when extract_shared's fusion attached
                # one under SEMANTIK_VLM_STRUCT_HINTS; a block without the key
                # -> None (no KeyError, byte-identical when the flag is off).
                vlm_hint=b.get("vlm_hint"),
                # P1 fusion provenance. The merged block dict gains a
                # ``fusion`` key ONLY when extract_shared's P1 fusion rewrote
                # the tesseract blocks (SEMANTIK_VLM_FUSION); absent -> None
                # (byte-identical when the flag is off).
                fusion=b.get("fusion"),
            ))
    return out


__all__ = ["extract_blocks", "blocks_from_shared", "extract_shared",
           "EncryptedPDFError"]
