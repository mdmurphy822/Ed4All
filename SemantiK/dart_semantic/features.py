"""Stage 2: Feature normalization. Deterministic. No model.

Contract:
    featurize(raw_blocks: list[RawBlock]) -> list[FeatureBlock]

Takes the raw-block stream from stage 1 and attaches normalized layout
features:
    size_bucket             font size relative to page median: xl/lg/md/sm
    gap_above               "lg" if vertical gap above is >= 2x page-median
                            line spacing, else None
    is_top_of_page          bbox top < 15% of page height
    is_centered             line is both short AND mid-x near page center
    caps                    "all"/"title"/None from text capitalization
    indent_bucket           left-edge quantized to 10 buckets 0..9
    relative_font_ratio     font_size / page_median_font_size (exact float)
    prev/next_size_bucket   contextual hints for rules and classifier

Still emits a serialized "flat" representation via features_to_prompt()
so scripts that feed a model a single string keep working.

This stage is unit-testable with hand-crafted RawBlock input — no files,
no network, no side effects.
"""
from __future__ import annotations

from pathlib import Path

from .extract import blocks_from_shared, extract_blocks
from .extract_shared import extract_shared
from .reading_order import (
    _DEFAULT_PAGE_WIDTH,
    column_ids_for_x0s,
    resolve_column_order_mode,
)
from .region_detection import (
    detect_image_region_candidates,
    detect_math_region_candidates,
    detect_table_region_candidates,
)
from .types import FeatureBlock, FeatureSet, RawBlock

# Note: we deliberately do NOT import pypdfium2 or pytesseract here —
# PDF I/O lives entirely in dart_semantic/extract{,_shared}.py. This
# module's only job is feature derivation from already-extracted blocks.


def featurize(raw_blocks: list[RawBlock]) -> list[FeatureBlock]:
    """Group blocks by page, compute per-page medians, emit FeatureBlocks."""
    if not raw_blocks:
        return []

    # Bucket by page to compute per-page medians.
    by_page: dict[int, list[RawBlock]] = {}
    for b in raw_blocks:
        by_page.setdefault(b.page, []).append(b)

    out: list[FeatureBlock] = []
    column_order = resolve_column_order_mode()
    for page_num in sorted(by_page.keys()):
        page_blocks = by_page[page_num]
        if column_order:
            # Column-major reading order (Fix A): cluster the page's blocks
            # into columns by left-edge gutter, then sort
            # (column_index, y0, x0) so a 2-column page reads left-column-
            # then-right-column instead of line-interleaved across the gutter.
            # Single-column -> one column -> byte-identical to the raster key.
            page_w = page_blocks[0].page_width if page_blocks else _DEFAULT_PAGE_WIDTH
            # W7.2 single-column guard: a stray wide gap (indent / centered
            # figure / page number) must NOT invent a column on the reorder
            # path (reorder only — never the council BERT feature).
            col = column_ids_for_x0s(
                [b.bbox[0] for b in page_blocks], page_w, single_column_guard=True,
            )
            page_blocks = [
                page_blocks[k]
                for k in sorted(
                    range(len(page_blocks)),
                    key=lambda k: (col[k], page_blocks[k].bbox[1], page_blocks[k].bbox[0]),
                )
            ]
        else:
            page_blocks = sorted(page_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

        sizes = [b.font_size or 1.0 for b in page_blocks if b.font_size]
        page_median_size = _median(sizes) if sizes else 1.0
        # Gaps between consecutive blocks (vertical).
        gaps = [0.0]
        for prev, cur in zip(page_blocks, page_blocks[1:]):
            gaps.append(max(0.0, cur.bbox[1] - prev.bbox[3]))
        positive_gaps = [g for g in gaps if g > 0]
        page_median_gap = _median(positive_gaps) if positive_gaps else page_median_size

        for i, b in enumerate(page_blocks):
            fb = _featurize_one(
                b, gap_above=gaps[i],
                page_median_size=page_median_size,
                page_median_gap=page_median_gap,
            )
            out.append(fb)

    _attach_neighbor_hints(out)
    return out


def featurize_pdf(pdf_path: Path) -> list[FeatureBlock]:
    """Convenience: stage 1 + stage 2 in one call."""
    return featurize(extract_blocks(pdf_path))


def featurize_from_shared(shared: dict) -> list[FeatureBlock]:
    """Build FeatureBlocks that carry the multi-extractor layout flags
    (in_table, in_header_row, in_widget, widget_kind, provenance).

    Uses the per-page `merged` blocks as the text stream (same as
    blocks_from_shared → extract_blocks), then enriches each block with
    pdfplumber's table bboxes and pikepdf's widget rects.
    """
    raw_blocks = blocks_from_shared(shared)
    feature_blocks = featurize(raw_blocks)

    # Build a quick lookup of tables + widgets per page.
    pages_by_num: dict[int, dict] = {p["page_num"]: p for p in shared.get("pages", [])}

    for fb in feature_blocks:
        page = pages_by_num.get(fb.raw.page)
        if page is None:
            continue
        x0, y0, x1, y1 = fb.raw.bbox
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        # in_table + in_header_row via pdfplumber tables.
        for t in page.get("pdfplumber", {}).get("tables", []):
            tb = t.get("bbox") or []
            if len(tb) == 4 and tb[0] <= mid_x <= tb[2] and tb[1] <= mid_y <= tb[3]:
                fb.in_table = True
                if tb[3] != tb[1]:
                    if (mid_y - tb[1]) / (tb[3] - tb[1]) < 0.25:
                        fb.in_header_row = True
                break

        # in_widget + widget_kind via pikepdf widgets.
        for w in page.get("pikepdf", {}).get("widgets", []):
            wb = w.get("bbox") or []
            if len(wb) == 4 and wb[0] <= mid_x <= wb[2] and wb[1] <= mid_y <= wb[3]:
                fb.in_widget = True
                ft = (w.get("field_type") or "").lstrip("/")
                if ft == "Tx":
                    fb.widget_kind = "text"
                elif ft == "Btn":
                    fb.widget_kind = "button"
                elif ft == "Ch":
                    fb.widget_kind = "select"
                elif ft == "Sig":
                    fb.widget_kind = "signature"
                break

        # provenance is stored on RawBlock.source; mirror it onto the feature block
        # for callers that consume FeatureBlock without reaching through to raw.
        fb.provenance = fb.raw.source

        # P2 — mirror the VLM structural hint from the raw block so the
        # heading-candidate consumers (structure_graph Pass-2, deterministic
        # sub-pass E) can read it without reaching through to ``raw``. None
        # when the VLM struct-hints flag is off -> byte-identical FeatureBlock.
        fb.vlm_hint = fb.raw.vlm_hint

    return feature_blocks


def featurize_with_regions(shared: dict) -> FeatureSet:
    """Phase 1 entry point: emit a FeatureSet with the flat FeatureBlock
    stream PLUS typed table/math region candidates.

    The FeatureBlock list is identical to what `featurize_from_shared`
    returns (calls into it directly — no re-implementation of layout
    feature extraction).

    Region candidates are derived from the same shared-extract JSON;
    pdfplumber output is reused, never re-run. `member_block_indices`
    on each candidate maps back into `feature_blocks`.

    v1 callers continue to use `featurize_from_shared` and get exactly
    the same FeatureBlock list they always did. This function is
    additive — never call it from the v1 path.

    Part F — when SEMANTIK_DETECT_FIGURES is on, image candidates
    (extracted PDF IMAGE page-objects) are detected and one SYNTHETIC
    image FeatureBlock per candidate is INTERLEAVED into the FeatureBlock
    stream in reading order (page, y0, x0). The candidate's
    `member_block_indices` then points at its synthetic FB so the
    Stage-5 coverage invariant (every FB index claimed by exactly one
    region) holds — the figure region claims its image FB. When the flag
    is off, `detect_image_region_candidates` returns `[]` (no `images`
    key on the shared pages) and the stream is byte-identical.
    """
    feature_blocks = featurize_from_shared(shared)

    # Detect image candidates FIRST (off the shared `images` lists), then
    # synthesize one image FB per candidate and merge into the stream in
    # reading order so downstream FB indices are stable. Done BEFORE the
    # table/math member-index pass so ALL candidates are indexed against
    # the FINAL (post-interleave) feature-block stream.
    image_candidates = detect_image_region_candidates(shared)
    if image_candidates:
        feature_blocks = _interleave_image_feature_blocks(
            feature_blocks, image_candidates
        )

    table_candidates = detect_table_region_candidates(
        shared, feature_blocks=feature_blocks)
    math_candidates = detect_math_region_candidates(
        shared, feature_blocks=feature_blocks)
    return FeatureSet(
        feature_blocks=feature_blocks,
        table_candidates=table_candidates,
        math_candidates=math_candidates,
        image_candidates=image_candidates,
    )


def _interleave_image_feature_blocks(
    feature_blocks: list[FeatureBlock],
    image_candidates: list,
) -> list[FeatureBlock]:
    """Synthesize one image FeatureBlock per image candidate and merge it
    into ``feature_blocks`` in reading order (page, y0, x0).

    Each synthetic FB carries empty text, the image bbox (top-left origin,
    same convention as text FBs), the page dims, ``source="pypdfium2:image"``
    and ``is_image=True``. After the merge, each candidate's
    ``member_block_indices`` is set to ``[that synthetic FB's index]`` so
    the figure region (Stage D) claims exactly that FB and the Stage-5
    coverage invariant holds.

    Image candidates with a degenerate bbox are skipped (their
    ``member_block_indices`` is left empty — they will be dropped by Stage
    D, which requires a claimable member FB).
    """
    # Resolve per-page dims from existing text FBs (fall back to LETTER).
    page_dims: dict[int, tuple[float, float]] = {}
    for fb in feature_blocks:
        raw = getattr(fb, "raw", None)
        if raw is None:
            continue
        page_dims.setdefault(
            int(raw.page),
            (float(raw.page_width), float(raw.page_height)),
        )

    # Build (sort_key, kind, payload) tuples for a single stable sort. Text
    # FBs keep their existing relative order via an enumerate tiebreak.
    entries: list[tuple] = []
    for orig_idx, fb in enumerate(feature_blocks):
        raw = getattr(fb, "raw", None)
        page = int(raw.page) if raw is not None else 0
        y0 = float(raw.bbox[1]) if raw is not None else 0.0
        x0 = float(raw.bbox[0]) if raw is not None else 0.0
        # (page, y0, x0, is_image=0 so text sorts before an image at the
        # same coordinate, original-index tiebreak to keep text order)
        entries.append(((page, y0, x0, 0, orig_idx), "text", fb))

    synth_for_cand: dict[int, FeatureBlock] = {}
    for ci, cand in enumerate(image_candidates):
        bbox = tuple(float(x) for x in (cand.bbox or ()))
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        page = int(cand.pages[0]) if cand.pages else 0
        pw, ph = page_dims.get(page, (612.0, 792.0))
        synth_raw = RawBlock(
            text="",
            page=page,
            bbox=bbox,
            page_width=pw,
            page_height=ph,
            source="pypdfium2:image",
        )
        synth_fb = FeatureBlock(
            raw=synth_raw,
            size_bucket="md",
            gap_above=None,
            is_top_of_page=False,
            is_centered=False,
            caps=None,
            indent_bucket=0,
            relative_font_ratio=1.0,
            provenance="pypdfium2:image",
            is_image=True,
        )
        synth_for_cand[ci] = synth_fb
        # is_image sorts AFTER text at the same coordinate (4th key=1).
        entries.append(((page, bbox[1], bbox[0], 1, ci), "image", synth_fb))

    if resolve_column_order_mode():
        # Column-major reading order (Fix A): insert a per-page column_index
        # after the page key so a 2-column page reads column-by-column. Cluster
        # over EVERY block on the page (text + synthetic image FBs) so an image
        # sorts within its column. Single-column -> one column -> the key
        # collapses to the original (page, y0, x0, is_image, idx).
        by_page_pos: dict[int, list[int]] = {}
        for pos, (key, _kind, _fb) in enumerate(entries):
            by_page_pos.setdefault(key[0], []).append(pos)
        col_by_pos: dict[int, int] = {}
        for page, positions in by_page_pos.items():
            page_w = page_dims.get(page, (_DEFAULT_PAGE_WIDTH, 792.0))[0]
            x0s = [entries[p][0][2] for p in positions]
            # W7.2 single-column guard (reorder path only).
            for p, c in zip(
                positions,
                column_ids_for_x0s(x0s, page_w, single_column_guard=True),
            ):
                col_by_pos[p] = c
        entries = [
            ((key[0], col_by_pos[pos], key[1], key[2], key[3], key[4]), kind, fb)
            for pos, (key, kind, fb) in enumerate(entries)
        ]

    entries.sort(key=lambda e: e[0])

    merged: list[FeatureBlock] = []
    fb_index_for_cand: dict[int, int] = {}
    for new_idx, (_key, kind, fb) in enumerate(entries):
        merged.append(fb)
        if kind == "image":
            # The 5th element of the sort key is the candidate index.
            ci = _key[4]
            fb_index_for_cand[ci] = new_idx

    # Re-point each surviving image candidate at its synthetic FB index.
    for ci, cand in enumerate(image_candidates):
        if ci in fb_index_for_cand:
            cand.member_block_indices = [fb_index_for_cand[ci]]
        else:
            cand.member_block_indices = []

    return merged


# ---------- per-block feature assignment ----------

def _featurize_one(b: RawBlock, *, gap_above: float,
                   page_median_size: float, page_median_gap: float) -> FeatureBlock:
    fs = b.font_size or page_median_size
    ratio = fs / page_median_size if page_median_size else 1.0
    if ratio >= 1.8:
        size_bucket = "xl"
    elif ratio >= 1.3:
        size_bucket = "lg"
    elif ratio < 0.8:
        size_bucket = "sm"
    else:
        size_bucket = "md"

    gap_flag = "lg" if page_median_gap and gap_above >= 2 * page_median_gap else None

    is_top = b.bbox[1] < 0.15 * b.page_height

    line_w = b.bbox[2] - b.bbox[0]
    mid = (b.bbox[0] + b.bbox[2]) / 2
    is_centered = (line_w < 0.5 * b.page_width
                   and abs(mid - b.page_width / 2) < 0.05 * b.page_width)

    caps = _caps_pattern(b.text)
    indent_bucket = int(min(9, max(0, b.bbox[0] / b.page_width * 10)))

    return FeatureBlock(
        raw=b,
        size_bucket=size_bucket,
        gap_above=gap_flag,
        is_top_of_page=is_top,
        is_centered=is_centered,
        caps=caps,
        indent_bucket=indent_bucket,
        relative_font_ratio=ratio,
    )


def _attach_neighbor_hints(fbs: list[FeatureBlock]) -> None:
    for i, fb in enumerate(fbs):
        fb.prev_size_bucket = fbs[i - 1].size_bucket if i > 0 else None
        fb.next_size_bucket = fbs[i + 1].size_bucket if i + 1 < len(fbs) else None


def _caps_pattern(text: str) -> str | None:
    alnum = [c for c in text if c.isalpha()]
    if len(alnum) < 3:
        return None
    if all(c.isupper() for c in alnum):
        return "all"
    if _is_title_case(text):
        return "title"
    return None


def _is_title_case(text: str) -> bool:
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if len(words) < 3:
        return False
    minor = {"a", "an", "the", "and", "or", "but", "of",
             "in", "on", "at", "to", "for"}
    for w in words:
        core = "".join(c for c in w if c.isalpha())
        if not core or core.lower() in minor:
            continue
        if not core[0].isupper():
            return False
    return True


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------- classifier input string ----------

# Provenance combinations that are too common to be worth tokens — treat as
# "default" and omit from the serialized flag string. Saves ~3 tokens/block
# on typical documents.
DEFAULT_PROVENANCE = {
    None, "pypdfium2", "pypdfium2+pdfplumber", "pdfplumber+pypdfium2",
}


def feature_block_to_classifier_input(fb: FeatureBlock) -> str:
    """Serialize one FeatureBlock into the classifier's input string.

    SINGLE SOURCE OF TRUTH for the format used at both training time
    (data/build_classifier_data_v2.py) and inference time
    (dart_semantic/classify.py). Keep these in lockstep — silent format
    drift between them would be a silent accuracy hit.
    """
    chunks: list[str] = []
    if fb.size_bucket != "md":
        chunks.append(f"size={fb.size_bucket}")
    if fb.gap_above == "lg":
        chunks.append("gap=lg")
    if fb.is_top_of_page:
        chunks.append("top")
    if fb.is_centered:
        chunks.append("centered")
    if fb.caps in ("all", "title"):
        chunks.append(f"caps={fb.caps}")
    if fb.in_table:
        chunks.append("in_table")
        if fb.in_header_row:
            chunks.append("header_row")
    if fb.in_widget:
        chunks.append("in_widget")
        if fb.widget_kind:
            chunks.append(f"widget={fb.widget_kind}")
    if fb.provenance not in DEFAULT_PROVENANCE:
        chunks.append(f"prov={fb.provenance}")
    if chunks:
        return f"[{' '.join(chunks)}] {fb.raw.text}"
    return fb.raw.text


# ---------- flat-string serialization (legacy path) ----------

def features_to_prompt(fbs: list[FeatureBlock]) -> str:
    """Emit the same PAGE N / [flags] text format used by older callers.

    Kept for backward compatibility with scripts that feed a single string
    to the model. New callers should consume FeatureBlock objects directly.
    """
    out: list[str] = []
    cur_page = -1
    for fb in fbs:
        if fb.raw.page != cur_page:
            out.append(f"PAGE {fb.raw.page}")
            cur_page = fb.raw.page
        flags = []
        if fb.size_bucket != "md":
            flags.append(f"size={fb.size_bucket}")
        if fb.gap_above:
            flags.append(f"gap={fb.gap_above}")
        if fb.is_top_of_page:
            flags.append("top")
        if fb.is_centered:
            flags.append("centered")
        if fb.caps:
            flags.append(f"caps={fb.caps}")
        prefix = f"[{' '.join(flags)}] " if flags else ""
        out.append(f"{prefix}{fb.raw.text}")
    return "\n".join(out)


# ---------- legacy compatibility: pdf_to_ocr_text ----------

def pdf_to_ocr_text(pdf_path: Path, *, page_range=None) -> str:
    """Legacy entry point: PDF -> flat layout-annotated string.

    Wraps the new featurize_pdf + features_to_prompt. Accepts an optional
    (start, end) page range for partial extraction; other modules rely on
    this signature.
    """
    fbs = featurize_pdf(pdf_path)
    if page_range is not None:
        start, end = page_range
        # page indices are 1-based here since RawBlock.page is 1-indexed
        start_p = start + 1
        end_p = end
        fbs = [fb for fb in fbs if start_p <= fb.raw.page <= end_p]
    return features_to_prompt(fbs)


# ---------- deprecated: left in place for any remaining callers ----------

def render_to_pdf(*args, **kwargs):
    """Deprecated. Moved responsibility into HtmlValidator.render_pdf."""
    from .validate import HtmlValidator
    raise RuntimeError(
        "features.render_to_pdf was removed. Use HtmlValidator.render_pdf "
        "inside a `with HtmlValidator()` block."
    )
