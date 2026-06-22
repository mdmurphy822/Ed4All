"""Run GLM-OCR on detected math / table regions of a shared-extract doc.

Sits between `extract_shared` and the rest of the pipeline. Does NOT
modify the shared output's existing keys — only appends a new key
`glm_ocr` per page:

  page["glm_ocr"] = [
      {
        "kind": "math" | "table",
        "bbox": [x0, y0, x1, y1],      # pdf-space, top-left origin
        "text": "<OCR result>",
        "prompt": "Formula Recognition:" | "Table Recognition:",
        "source_text": "<what pypdfium2/pdfplumber produced for this region>",
        "cached": bool,                # whether this came from disk cache
      },
      ...
  ]

Idempotent: every call is keyed through `glm_ocr_cache`, so re-running
`enrich_shared` on the same PDF is free after the first pass.

Tesseract fallback: pages that had to use Tesseract (missing text
layer) are OCR'd whole with GLM-OCR as a secondary signal. The result
goes in `page["glm_ocr_full_page"]`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .glm_ocr import (
    PROMPT_FORMULA,
    PROMPT_TABLE,
    PROMPT_TEXT,
    ocr_image,
)
from .glm_ocr_cache import CacheKey, get as cache_get, put as cache_put, sha256_file
from .region_detection import (
    Region,
    detect_low_conf_tesseract_regions,
    detect_math_regions,
    detect_table_regions,
)


# Per-kind generation budget. Math equations are short (~100-300 tokens),
# small tables fit in ~500 tokens, low-confidence Tesseract fragments are
# usually a single line. Capping per-kind avoids the default 2048 that
# made the pre-pass 4× slower than necessary on math-heavy docs.
MAX_NEW_TOKENS_BY_KIND = {
    "math": 512,
    "table": 1024,
    "tess_low_conf": 512,
}
DEFAULT_MAX_NEW_TOKENS = 2048


def _render_page(pdf_path: Path, page_num: int, scale: float):
    from .extract_shared import _pypdfium2_render_page_to_image
    return _pypdfium2_render_page_to_image(pdf_path, page_num, scale=scale)


def _crop(img, page_w: float, page_h: float,
          bbox: tuple[float, float, float, float], scale: float,
          pad: float = 4.0):
    x0, y0, x1, y1 = bbox
    # Some upstream bboxes come out inverted (y0 > y1) — normalize so
    # PIL's crop always sees upper < lower.
    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(page_w, x1 + pad)
    y1 = min(page_h, y1 + pad)
    # After clamping to the page, an out-of-page bbox can collapse
    # (e.g. the whole bbox was below page_h). Skip empty/inverted results.
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((int(x0 * scale), int(y0 * scale),
                     int(x1 * scale), int(y1 * scale)))


def _prompt_for(kind: str) -> str:
    if kind == "math":
        return PROMPT_FORMULA
    if kind == "table":
        return PROMPT_TABLE
    # tess_low_conf and fallback
    return PROMPT_TEXT


def enrich_shared(shared: dict, pdf_path: Path, glm: dict | None,
                  *,
                  render_scale: float = 2.0,
                  tesseract_conf_threshold: float = 0.70) -> dict:
    """In-place: add `glm_ocr` per page. Returns `shared` for chaining.

    Regions OCR'd:
      - math (detect_math_regions)
      - table (detect_table_regions, every pdfplumber-detected bbox)
      - tess_low_conf (Tesseract blocks with confidence below the
        threshold; only fires on pages that actually used Tesseract)

    `glm` None → cache-only mode (worker processes, no GPU).
    `glm` loaded → full mode (main process, OCRs misses).
    """
    pages = shared.get("pages") or []
    if not pages:
        return shared

    pdf_sha = sha256_file(pdf_path)
    for pg in pages:
        page_num = int(pg["page_num"])
        regions: list[Region] = (
            detect_table_regions(pg)
            + detect_math_regions(pg)
            + detect_low_conf_tesseract_regions(
                pg, conf_threshold=tesseract_conf_threshold)
        )

        page_img = None
        page_w = float(pg.get("width", 612))
        page_h = float(pg.get("height", 792))

        out_rows: list[dict[str, Any]] = []
        for r in regions:
            prompt = _prompt_for(r.kind)
            key = CacheKey(pdf_sha=pdf_sha, page_num=page_num,
                           bbox=r.bbox, prompt=prompt)
            hit = cache_get(key)
            if hit is not None:
                out_rows.append({
                    "kind": r.kind,
                    "bbox": list(r.bbox),
                    "text": hit.get("text", ""),
                    "prompt": prompt,
                    "source_text": r.src_text,
                    "error": hit.get("error"),
                    "cached": True,
                })
                continue

            if glm is None:
                # cache-only mode: skip misses
                continue

            if page_img is None:
                page_img = _render_page(pdf_path, page_num, render_scale)

            crop = _crop(page_img, page_w, page_h, r.bbox, render_scale)
            mnt = MAX_NEW_TOKENS_BY_KIND.get(r.kind, DEFAULT_MAX_NEW_TOKENS)
            if crop is None:
                payload = {"text": "", "error": "empty_bbox"}
            else:
                try:
                    text = ocr_image(glm, crop, prompt=prompt,
                                     max_new_tokens=mnt)
                    payload = {"text": text}
                except Exception as exc:
                    payload = {"text": "",
                               "error": f"{type(exc).__name__}: {exc}"}
            cache_put(key, payload)
            out_rows.append({
                "kind": r.kind,
                "bbox": list(r.bbox),
                "text": payload.get("text", ""),
                "prompt": prompt,
                "source_text": r.src_text,
                "error": payload.get("error"),
                "cached": False,
            })

        pg["glm_ocr"] = out_rows

    return shared


def enrich_confirmed_tables(
    shared: dict,
    pdf_path: Path,
    glm: dict | None,
    *,
    confirmed_bboxes_by_page: dict[int, list[tuple[float, float, float, float]]],
    render_scale: float = 2.0,
) -> dict:
    """In-place: OCR ONLY the table bboxes the caller has already confirmed.

    Strict-by-default counterpart to ``enrich_shared``: skip math
    regions, skip low-confidence-Tesseract regions, skip pdfplumber's
    raw table proposals. The caller — typically having run the
    Structure council and called
    :func:`dart_semantic.council.cross_reranker.confirm_table_candidates`
    — passes in the per-page list of confirmed bboxes, and we OCR
    exactly those. This is the cost-savings entry point for "GLM-OCR
    is expensive; only run it where the trained head agrees there's
    a table."

    Output shape on each page matches the existing ``glm_ocr`` key
    so downstream consumers don't need a new branch:

        page["glm_ocr"] = [
            {"kind": "table", "bbox": [...], "text": "...", ...},
            ...
        ]

    Pages with no confirmed bboxes get ``page["glm_ocr"] = []`` so the
    key is always present (matches ``enrich_shared``'s contract).

    ``glm`` ``None`` → cache-only mode (worker processes, no GPU);
    misses are silently skipped.
    """
    pages = shared.get("pages") or []
    if not pages:
        return shared

    pdf_sha = sha256_file(pdf_path)
    for pg in pages:
        page_num = int(pg["page_num"])
        bboxes = confirmed_bboxes_by_page.get(page_num, [])

        page_img = None
        page_w = float(pg.get("width", 612))
        page_h = float(pg.get("height", 792))
        out_rows: list[dict[str, Any]] = []
        prompt = _prompt_for("table")

        for bbox in bboxes:
            key = CacheKey(pdf_sha=pdf_sha, page_num=page_num,
                           bbox=tuple(bbox), prompt=prompt)
            hit = cache_get(key)
            if hit is not None:
                out_rows.append({
                    "kind": "table",
                    "bbox": list(bbox),
                    "text": hit.get("text", ""),
                    "prompt": prompt,
                    "source_text": "",  # caller didn't provide; downstream may.
                    "error": hit.get("error"),
                    "cached": True,
                    "gated_by": "council:table_region",
                })
                continue

            if glm is None:
                continue  # cache-only worker

            if page_img is None:
                page_img = _render_page(pdf_path, page_num, render_scale)
            crop = _crop(page_img, page_w, page_h, tuple(bbox), render_scale)
            mnt = MAX_NEW_TOKENS_BY_KIND.get("table", DEFAULT_MAX_NEW_TOKENS)
            if crop is None:
                payload = {"text": "", "error": "empty_bbox"}
            else:
                try:
                    text = ocr_image(glm, crop, prompt=prompt,
                                     max_new_tokens=mnt)
                    payload = {"text": text}
                except Exception as exc:
                    payload = {"text": "",
                               "error": f"{type(exc).__name__}: {exc}"}
            cache_put(key, payload)
            out_rows.append({
                "kind": "table",
                "bbox": list(bbox),
                "text": payload.get("text", ""),
                "prompt": prompt,
                "source_text": "",
                "error": payload.get("error"),
                "cached": False,
                "gated_by": "council:table_region",
            })

        pg["glm_ocr"] = out_rows

    return shared


def enrich_via_council_gate(
    shared: dict,
    pdf_path: Path,
    glm: dict | None,
    *,
    council_state: Any,
    table_candidates: list,
    render_scale: float = 2.0,
    threshold: float | None = None,
) -> dict:
    """Convenience: filter ``table_candidates`` by the Structure council
    then run GLM-OCR on the confirmed bboxes only.

    Equivalent to::

        from dart_semantic.council.cross_reranker import confirm_table_candidates
        confirmed = confirm_table_candidates(council_state, table_candidates)
        by_page = {}
        for c in confirmed:
            for p in c.pages:
                by_page.setdefault(int(p), []).append(c.bbox)
        enrich_confirmed_tables(shared, pdf_path, glm,
                                confirmed_bboxes_by_page=by_page)

    Use this from a v2-cascade-aware caller that has the council
    ``state`` and the original ``table_candidates`` (typically the
    return value of ``run_council`` plus ``feature_set.table_candidates``).
    """
    # Lazy import to avoid pulling council into the dataset-build path
    # when only the bbox-list API is needed.
    from .council.cross_reranker import confirm_table_candidates

    kwargs: dict[str, Any] = {}
    if threshold is not None:
        kwargs["threshold"] = float(threshold)
    confirmed = confirm_table_candidates(council_state, table_candidates, **kwargs)

    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for cand in confirmed:
        for pg_num in (cand.pages or []):
            by_page.setdefault(int(pg_num), []).append(tuple(cand.bbox))
    return enrich_confirmed_tables(
        shared, pdf_path, glm,
        confirmed_bboxes_by_page=by_page,
        render_scale=render_scale,
    )


def enrich_table_regions_with_glm_ocr(
    regions: "list[Any]",
    council_regions: "list[Any]",
    council_state: Any,
    shared: dict,
    pdf_path: Path,
    *,
    glm: dict | None = None,
    threshold: float | None = None,
    render_scale: float = 2.0,
) -> "list[Any]":
    """Stage 5b — optional GLM-OCR enrichment of confirmed table regions.

    Sits between Stage 5 (``build_structure_graph``) and Stage 6
    (``run_qwen_specialists``). For each Stage-5 ``Region`` with
    ``kind == "table"``:

      1. Find the underlying ``TableCandidate`` via
         ``region.source_region_id`` (an index into ``council_regions``).
      2. Filter to the subset confirmed by the Structure council's
         ``table_region`` head (via
         :func:`dart_semantic.council.cross_reranker.confirm_table_candidates`).
      3. Run GLM-OCR on confirmed bboxes only (via
         :func:`enrich_confirmed_tables`).
      4. Inject the matched OCR text into a NEW Region's
         ``payload["glm_ocr_text"]`` so Stage 6's table specialist can
         read it without needing access to ``shared``.

    Returns a new list of Regions; non-table regions and unmatched
    table regions are returned unchanged. Frozen-dataclass safe
    (uses ``dataclasses.replace`` rather than mutating in place).

    ``glm`` ``None`` → cache-only mode (worker-process safe). Cache
    misses are skipped; the affected table region simply won't carry
    ``glm_ocr_text``. Per the project no-silent-fallbacks preference,
    callers should log a banner upstream so the active mode is visible
    in the eval / smoke harness output.
    """
    import dataclasses

    from .council.cross_reranker import confirm_table_candidates
    from .region_detection import TableCandidate

    # Map source_region_id -> TableCandidate so we can look up the
    # bbox / pages for each Stage-5 table Region.
    table_cands_by_idx: dict[int, TableCandidate] = {}
    for idx, cand in enumerate(council_regions):
        if isinstance(cand, TableCandidate):
            table_cands_by_idx[idx] = cand

    if not table_cands_by_idx:
        return list(regions)

    # Filter by council agreement.
    confirm_kwargs: dict[str, Any] = {}
    if threshold is not None:
        confirm_kwargs["threshold"] = float(threshold)
    confirmed = confirm_table_candidates(
        council_state, list(table_cands_by_idx.values()), **confirm_kwargs,
    )
    if not confirmed:
        return list(regions)

    # Build per-page bbox dict and run gated OCR (mutates shared).
    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for cand in confirmed:
        for pg_num in (cand.pages or []):
            by_page.setdefault(int(pg_num), []).append(tuple(cand.bbox))
    enrich_confirmed_tables(
        shared, pdf_path, glm,
        confirmed_bboxes_by_page=by_page,
        render_scale=render_scale,
    )

    # Build (page, bbox) -> ocr_text lookup. Skip OCR errors and
    # entries with empty text — better to leave the payload empty than
    # to inject "" and have the prompt builder think OCR ran cleanly.
    ocr_by_key: dict[tuple[int, tuple[float, float, float, float]], str] = {}
    for pg in (shared.get("pages") or []):
        page_num = int(pg["page_num"])
        for entry in (pg.get("glm_ocr") or []):
            if entry.get("kind") != "table":
                continue
            bbox = tuple(entry.get("bbox") or ())
            text = (entry.get("text") or "").strip()
            if bbox and text and not entry.get("error"):
                ocr_by_key[(page_num, bbox)] = text

    if not ocr_by_key:
        return list(regions)

    out: list[Any] = []
    for region in regions:
        if region.kind != "table" or region.source_region_id is None:
            out.append(region)
            continue
        cand = table_cands_by_idx.get(region.source_region_id)
        if cand is None:
            out.append(region)
            continue
        ocr_text = None
        for pg_num in (cand.pages or []):
            ocr_text = ocr_by_key.get((int(pg_num), tuple(cand.bbox)))
            if ocr_text:
                break
        if ocr_text:
            new_payload = {**region.payload, "glm_ocr_text": ocr_text}
            out.append(dataclasses.replace(region, payload=new_payload))
        else:
            out.append(region)

    return out


def enrich_from_cache(shared: dict, pdf_path: Path) -> dict:
    """Cache-only enrichment — safe to call in worker processes.

    Equivalent to `enrich_shared(shared, pdf_path, glm=None)`.
    """
    return enrich_shared(shared, pdf_path, glm=None)


def glm_ocr_regions_as_feature_blocks(shared: dict):
    """Synthesize a FeatureBlock per GLM-OCR region so DistilBERT can
    classify the cleaned OCR text the same way it classifies any other
    block. Returns (feature_blocks, region_refs) — region_refs[i] is the
    original region dict on the page, so the caller can write the
    classification result back onto it.

    This is how stage 3a gets to "review" GLM-OCR output before it
    reaches Qwen (stage 3b).
    """
    from .types import FeatureBlock, RawBlock
    fbs = []
    refs = []
    for pg in shared.get("pages") or []:
        page_num = int(pg["page_num"])
        page_w = float(pg.get("width", 612))
        page_h = float(pg.get("height", 792))
        for region in pg.get("glm_ocr") or []:
            text = (region.get("text") or "").strip()
            if not text:
                continue
            bbox = tuple(float(v) for v in region.get("bbox") or [0, 0, 0, 0])
            raw = RawBlock(
                text=text,
                page=page_num,
                bbox=bbox,
                page_width=page_w,
                page_height=page_h,
                source="glm_ocr",
            )
            fb = FeatureBlock(
                raw=raw,
                size_bucket="md",
                gap_above=None,
                is_top_of_page=False,
                is_centered=False,
                caps=None,
                indent_bucket=0,
                relative_font_ratio=1.0,
                in_table=(region.get("kind") == "table"),
                in_header_row=False,
                in_widget=False,
                provenance="glm_ocr",
            )
            fbs.append(fb)
            refs.append(region)
    return fbs, refs
