"""Build BERT-Structure (Phase 3b) training data.

Span-level multi-head classifier on PDF blocks. Replaces the v1
DistilBERT classifier (`train_classifier.py`) with a 4-head model on
the shared ModernBERT-base backbone.

Heads (all span-level):
    * structural_role     6-class     A NON-AUTHORITATIVE recommendation
                                      from BERT-Structure; downstream
                                      specialists override per their
                                      domain. Active classes only:
                                          paragraph, heading, list_item,
                                          form_label, blockquote,
                                          code_block
                                      td/th/caption/figcaption text
                                      content is labeled `paragraph`
                                      (its content shape) — table
                                      membership lives on the separate
                                      table_region binary head; figure
                                      captions will be handled by a
                                      future ImageSpecialist gated by
                                      an is_image_block head added to
                                      Structure later.
    * is_heading          binary      1 if span aligns to <h1>..<h6>;
                                      heading-level (h1..h6) is the
                                      HeadingSpecialist's job, gated on
                                      is_heading=1.
    * table_region        binary      1 if span sits inside a
                                      pdfplumber-detected table region
                                      OR has a <table> ancestor in the
                                      ground-truth HTML. Mirrors the
                                      is_heading→HeadingSpecialist
                                      pattern: table_region=1 hands the
                                      span over to the TableSpecialist
                                      (Phase 3e) for cell-level role +
                                      scope; this head only DETECTS
                                      table membership, it does not
                                      parse the table.
    * list_nesting        4-class     <ul>/<ol> ancestor count of <li>:
                                      {0=not in list, 1=top-level li,
                                       2=nested, 3=3+ deep}

Layout side-channel (numeric, fed to the model alongside text — same
pattern as Phase 3a v4 MergeOrSplit):
    fs_norm, fs_rel_page_median, bold, italic, width_norm, height_norm,
    lhr, x0_norm, x1_norm, y0_norm, top_5pct, bottom_5pct, is_artifact,
    in_table, ends_period, ends_colon, starts_upper, titlecase_frac,
    text_len_log, caps_ratio  (20 dims).

Source pipeline (mirrors data/build_classifier_data_v2.py):
    1. Walk pair files: ``data/pairs/<source>/*.json``.
    2. For each pair, run ``extract_shared`` on local_pdf (cached) OR
       render output_html via Playwright.
    3. Walk output_html DOM with metadata: per block emit
       ``(text, tag_name, role, list_depth)``.
    4. Align each pypdfium2 merged block to its HTML block via Jaccard.
    5. Compute layout vector + the 4 labels.
    6. Stratified 80/10/10 split by structural_role.

Outputs:
    data/structure_dataset/{train,val,test}.jsonl
    data/structure_dataset/coverage_report.json

Usage:
    python data/build_structure_data.py --workers 4
    python data/build_structure_data.py --pair-dirs data/synthetic/forms

Network usage: NONE — uses only local pair files + cached extractor
output. PDF rendering for synthetic HTML uses Playwright (already
present from v1 builder).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from dart_semantic.classify import Role
from dart_semantic.extract_shared import extract_shared, extract_shared_cached
from dart_semantic.text_utils import jaccard_overlap
from dart_semantic.worker_pool import run_in_pool
from data.balance import add_cap_args, apply_caps_and_report
from data.structure_align import (
    SCHEMA_VERSION as STRUCTURE_ALIGN_SCHEMA_VERSION,
)
from data.structure_align import (
    AlignRow,
    FBView,
    GoldView,
    LedgerIncompleteError,
    align_blocks,
    project_rows,
)

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

# 6-class structural_role head. Dropping the 15 Role enum values that
# get zero training data OR are handled by other signals/specialists:
#   TITLE, LIST, TABLE, TABLE_ROW, TABLE_HEADER_CELL, TABLE_DATA_CELL,
#   TABLE_CAPTION, FIGURE, FIGURE_CAPTION, FORM_FIELD, REFERENCE,
#   FOOTNOTE, METADATA, PAGE_HEADER, PAGE_FOOTER.
# Specifically: the entire TABLE_* family is dropped because the
# table_region BINARY head detects table membership, and the
# TableSpecialist (Phase 3e) parses cell-level role + scope downstream.
# FIGURE_CAPTION is dropped pending a future ImageSpecialist (gated by
# an is_image_block Structure head) that owns figure-caption emission.
# Page artifacts are captured by the layout side-channel
# (top_5pct/bottom_5pct/is_artifact) and post-processing. The 6
# remaining classes describe span CONTENT shape only.
ROLE_LIST = (
    Role.PARAGRAPH,
    Role.HEADING,
    Role.LIST_ITEM,
    Role.FORM_LABEL,
    Role.BLOCKQUOTE,
    Role.CODE_BLOCK,
)
ROLE_NAMES = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
)
ROLE_TO_ID = {r: i for i, r in enumerate(ROLE_LIST)}
NUM_ROLES = len(ROLE_LIST)

# list_nesting is 4-class for ALL spans (0 means "not in a list", which
# is true for most spans).
LIST_NESTING_BUCKETS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# HTML tag -> Role + per-span metadata
# ---------------------------------------------------------------------------

# Mirrors data/build_classifier_data_v2.py TAG_TO_ROLE but maps to the
# Role enum directly (not strings) so we can ID-encode for training.
TAG_TO_ROLE = {
    "h1": Role.HEADING,
    "h2": Role.HEADING,
    "h3": Role.HEADING,
    "h4": Role.HEADING,
    "h5": Role.HEADING,
    "h6": Role.HEADING,
    "p": Role.PARAGRAPH,
    "li": Role.LIST_ITEM,
    # Table cell text gets labeled by its content shape (paragraph) —
    # the "is in a table" signal lives on the separate table_region
    # binary head. The TableSpecialist (Phase 3e) parses the actual
    # cell-level role + scope downstream when table_region=1.
    "th": Role.PARAGRAPH,
    "td": Role.PARAGRAPH,
    "caption": Role.PARAGRAPH,
    # figcaption text content gets labeled paragraph for now. Phase 3f
    # will introduce an is_image_block Structure head + ImageSpecialist
    # that owns figure-caption emission (mirrors is_heading→
    # HeadingSpecialist and table_region→TableSpecialist gating).
    "figcaption": Role.PARAGRAPH,
    "blockquote": Role.BLOCKQUOTE,
    "pre": Role.CODE_BLOCK,
    "code": Role.CODE_BLOCK,
    "legend": Role.FORM_LABEL,
    "label": Role.FORM_LABEL,
    "input": Role.FORM_FIELD,
    "select": Role.FORM_FIELD,
    "textarea": Role.FORM_FIELD,
}

OUTER_TAG_ALLOW_NESTED = {
    "th",
    "td",
    "li",
    "caption",
    "figcaption",
    "legend",
    "label",
}

LIST_PARENT_TAGS = {"ul", "ol"}
HEADING_TAG_PATTERN = ("h1", "h2", "h3", "h4", "h5", "h6")


def _list_nesting_depth(el: Tag) -> int:
    """Count <ul>/<ol> ancestors of an HTML element. Bucket to {0,1,2,3+}."""
    depth = 0
    for p in el.parents:
        if isinstance(p, Tag) and p.name in LIST_PARENT_TAGS:
            depth += 1
    return min(depth, 3)


def _has_table_ancestor(el: Tag) -> bool:
    """True if any ancestor is a <table>. Backstop signal for the
    table_region binary head when pdfplumber's in_table detector misses
    a table region (e.g., a borderless layout table)."""
    for p in el.parents:
        if isinstance(p, Tag) and p.name == "table":
            return True
    return False


# Image-block ancestor classes — Phase 3f's is_image_block head needs
# to recognize HTML elements that live INSIDE a figure region. Mirror
# the table_region pattern: HTML truth via DOM ancestor walk, with
# class-name fallbacks for sources that don't use semantic <figure>.
#   * arXiv ar5iv: <figure class="ltx_figure">, .ltx_figure_panel
#   * Wikipedia (Parsoid): <figure>, .thumb, .thumbinner
#   * OpenStax: <figure>, .os-figure
#   * MathML images: skip (handled by Math, not Image)
_IMAGE_BLOCK_TAGS = {"figure", "picture", "figcaption"}
_IMAGE_BLOCK_CLASS_HINTS = (
    "ltx_figure",  # ar5iv
    "ltx_graphics",  # ar5iv embedded image
    "thumb",  # wikipedia
    "thumbinner",  # wikipedia
    "os-figure",  # openstax
    "image",  # generic
)


def _has_image_block_ancestor(el: Tag) -> bool:
    """True if the element itself or any ancestor is a recognized
    image-block container. Backstops: tag-based AND class-based —
    LaTeXML/Parsoid don't always emit semantic <figure>, but they do
    consistently use class names like ltx_figure / thumb."""
    cur: Tag | None = el
    while cur is not None and isinstance(cur, Tag):
        if cur.name in _IMAGE_BLOCK_TAGS:
            return True
        cls = cur.get("class") or []
        if isinstance(cls, str):
            cls = cls.split()
        if any(any(hint in c for hint in _IMAGE_BLOCK_CLASS_HINTS) for c in cls):
            return True
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return False


def _li_role_override(el: Tag) -> Role | None:
    """Wikiquote-style relabeling: if a <li> directly wraps a
    <blockquote> or <pre>, treat the li as that role instead of
    list_item. Wikiquote structures every quote as
    `<ul><li><blockquote>QUOTE</blockquote>...</li></ul>` — without
    this override the quote text gets labeled list_item.

    Same logic for <pre> inside <li> (some technical docs structure
    code samples this way). Direct child only — a deeply-nested
    blockquote inside a regular bulleted item should stay list_item.
    """
    if el.name != "li":
        return None
    if el.find("blockquote", recursive=False) is not None:
        return Role.BLOCKQUOTE
    if el.find("pre", recursive=False) is not None:
        return Role.CODE_BLOCK
    return None


def extract_html_blocks(html: str) -> list[dict]:
    """Return per-block records: ``{text, tag, role, list_nesting,
    in_table_html}`` in document order from output_html."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for el in soup.find_all(list(TAG_TO_ROLE.keys())):
        # Skip blocks nested inside another target block unless they're
        # natural-nest cells / list items / captions.
        if el.name not in OUTER_TAG_ALLOW_NESTED:
            has_outer = any(
                p.name in TAG_TO_ROLE and p is not el for p in el.parents if p.name is not None
            )
            if has_outer:
                continue
        text = " ".join((el.get_text() or "").split()).strip()
        if not text:
            continue
        role = TAG_TO_ROLE[el.name]
        override = _li_role_override(el)
        if override is not None:
            role = override
        out.append(
            {
                "text": text,
                "tag": el.name,
                "role": role,
                "list_nesting": _list_nesting_depth(el),
                "in_table_html": _has_table_ancestor(el),
                "in_image_block_html": _has_image_block_ancestor(el),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Span-level layout side-channel
# ---------------------------------------------------------------------------
#
# 20-dim numeric vector per span. Mirrors the pair-level layout pattern
# from Phase 3a v4 MergeOrSplit but adapted for single-span input.
# Output is clamped element-wise to ±10 to defend against pathological
# PDF coordinate transforms (we hit ~4 rows out of 120k in the
# Phase-3a build with y0=-4e74 from a broken transform; same defense
# applies here).

LAYOUT_FEATURE_NAMES = [
    "fs_norm",  # fs / 12
    "fs_rel_page_median",  # fs / page_median_fs (heading-vs-body signal)
    "bold",
    "italic",
    "width_norm",  # (x1 - x0) / page_width
    "height_norm",  # (y1 - y0) / 12
    "lhr",  # h / page_median_h
    "x0_norm",  # x0 / page_width  (left indent)
    "x1_norm",  # x1 / page_width  (right edge)
    "y0_norm",  # y0 / page_height
    "top_5pct",  # 1 if y0 < 5% of page height (page header territory)
    "bottom_5pct",  # 1 if y1 > 95% of page height (page footer territory)
    "is_artifact",  # short text in top/bottom 5% — same heuristic as classify.py
    "in_table",  # block center inside a pdfplumber-detected table bbox
    "ends_period",  # span ends with .!?
    "ends_colon",  # span ends with :
    "starts_upper",  # first char is uppercase
    "titlecase_frac",  # fraction of first-6 words starting with uppercase
    "text_len_log",  # log1p(len(text)) / 7
    "caps_ratio",  # fraction of uppercase letters in text
]

LAYOUT_FEATURE_DIM = len(LAYOUT_FEATURE_NAMES)


_LAYOUT_BBOX_CLAMP = 5000.0
_LAYOUT_FS_CLAMP = 200.0
_LAYOUT_OUTPUT_CLAMP = 10.0


def _safe_coord(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or abs(f) > _LAYOUT_BBOX_CLAMP:
        return 0.0
    return f


def _safe_fs(v) -> float:
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f < 0 or f > _LAYOUT_FS_CLAMP:
        return 0.0
    return f


def _clip_vec(vec: list[float]) -> list[float]:
    lo, hi = -_LAYOUT_OUTPUT_CLAMP, _LAYOUT_OUTPUT_CLAMP
    out = []
    for x in vec:
        if not math.isfinite(x):
            out.append(0.0)
        elif x < lo:
            out.append(lo)
        elif x > hi:
            out.append(hi)
        else:
            out.append(x)
    return out


def _ends_period(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _ends_colon(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] == ":"


def _starts_upper(text: str) -> bool:
    t = (text or "").lstrip()
    return bool(t) and t[0].isupper()


def _titlecase_frac(text: str, *, max_words: int = 6) -> float:
    words = (text or "").split()[:max_words]
    if not words:
        return 0.0
    return sum(1 for w in words if w and w[0].isupper()) / len(words)


def _caps_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _is_artifact(span: dict, *, page_h: float) -> bool:
    """Mirrors classify.py's _rule_page_header / _rule_page_footer:
    short text in top/bottom 5% of page."""
    bbox = span.get("bbox") or []
    if len(bbox) < 4 or page_h <= 0:
        return False
    text = (span.get("text") or "").strip()
    if not text or len(text) >= 100:
        return False
    top_frac = bbox[1] / page_h
    bot_frac = bbox[3] / page_h
    return top_frac < 0.05 or bot_frac > 0.95


def compute_span_layout_features(
    span: dict,
    *,
    page_w: float,
    page_h: float,
    page_median_fs: float,
    page_median_h: float,
    in_table: bool,
) -> list[float]:
    """Build the per-span layout vector. Order MUST match
    :data:`LAYOUT_FEATURE_NAMES`. Output clamped element-wise to ±10."""
    bbox = span.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    text = (span.get("text") or "").strip()
    fs = _safe_fs(span.get("font_size", 0.0) or 0.0)
    page_w = max(1.0, _safe_coord(page_w))
    page_h = max(1.0, _safe_coord(page_h))
    page_median_fs = max(1.0, _safe_fs(page_median_fs or 12.0))
    page_median_h = max(1.0, _safe_coord(page_median_h or 12.0))

    x0 = _safe_coord(bbox[0])
    y0 = _safe_coord(bbox[1])
    x1 = _safe_coord(bbox[2])
    y1 = _safe_coord(bbox[3])
    h = max(0.0, y1 - y0)
    w = max(0.0, x1 - x0)

    raw = [
        fs / 12.0,
        fs / page_median_fs,
        1.0 if span.get("is_bold") else 0.0,
        1.0 if span.get("is_italic") else 0.0,
        w / page_w,
        h / 12.0,
        h / page_median_h,
        x0 / page_w,
        x1 / page_w,
        y0 / page_h,
        1.0 if y0 < 0.05 * page_h else 0.0,
        1.0 if y1 > 0.95 * page_h else 0.0,
        1.0 if _is_artifact(span, page_h=page_h) else 0.0,
        1.0 if in_table else 0.0,
        1.0 if _ends_period(text) else 0.0,
        1.0 if _ends_colon(text) else 0.0,
        1.0 if _starts_upper(text) else 0.0,
        _titlecase_frac(text),
        math.log1p(len(text)) / 7.0,
        _caps_ratio(text),
    ]
    return _clip_vec(raw)


# ---------------------------------------------------------------------------
# Per-page in_table detection (cheap reuse of pdfplumber tables)
# ---------------------------------------------------------------------------


def _block_in_any_table(block: dict, page: dict) -> bool:
    bbox = block.get("bbox") or []
    if len(bbox) < 4:
        return False
    mid_x = (bbox[0] + bbox[2]) / 2
    mid_y = (bbox[1] + bbox[3]) / 2
    for t in page.get("pdfplumber", {}).get("tables", []):
        tb = t.get("bbox") or []
        if len(tb) == 4 and tb[0] <= mid_x <= tb[2] and tb[1] <= mid_y <= tb[3]:
            return True
    return False


# ---------------------------------------------------------------------------
# Worker — process one pair file
# ---------------------------------------------------------------------------


def process_pair(validator, work: tuple) -> dict:
    """Worker: align extract_shared blocks to HTML ground truth and emit
    one labeled row per aligned block."""
    pair_path_str, out_examples_dir_str = work
    pair_path = Path(pair_path_str)
    out_dir = Path(out_examples_dir_str)
    stats = {"pair": pair_path.name, "aligned": 0, "total_blocks": 0, "error": None}

    try:
        pair = json.loads(pair_path.read_text())
    except Exception as exc:
        stats["error"] = f"read pair: {exc}"
        return stats

    output_html = pair.get("output_html")
    if not output_html:
        stats["error"] = "no output_html"
        return stats

    html_blocks = extract_html_blocks(output_html)
    if not html_blocks:
        stats["error"] = "no html blocks"
        return stats

    # Source PDF: use local_pdf when available; otherwise render
    # output_html via Playwright (same pattern as v1 builder).
    local_pdf = pair.get("local_pdf")
    with tempfile.TemporaryDirectory() as tmp:
        if local_pdf and Path(local_pdf).exists():
            pdf_path = Path(local_pdf)
            try:
                shared = extract_shared_cached(pdf_path)
            except Exception as exc:
                stats["error"] = f"extract (cached): {exc}"
                return stats
        else:
            tmp_pdf = Path(tmp) / "rendered.pdf"
            try:
                validator.render_pdf(output_html, tmp_pdf)
            except Exception as exc:
                stats["error"] = f"render: {exc}"
                return stats
            try:
                shared = extract_shared(tmp_pdf)
            except Exception as exc:
                stats["error"] = f"extract: {exc}"
                return stats

    examples: list[dict] = []
    html_cursor = 0
    window = 8

    for page in shared.get("pages", []):
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue

        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))

        # Per-page medians for normalization.
        sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
        page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        heights = [
            b["bbox"][3] - b["bbox"][1]
            for b in merged
            if (b.get("bbox") and b["bbox"][3] > b["bbox"][1])
        ]
        page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0

        # Sort top-to-bottom, left-to-right (same as Phase 3a — keeps
        # alignment stable + preserves natural reading order for
        # context-windowed Jaccard).
        merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in merged_sorted:
            stats["total_blocks"] += 1
            in_table = _block_in_any_table(block, page)
            text = (block.get("text") or "").strip()
            if not text:
                continue

            # Align to HTML ground truth via Jaccard within a sliding
            # window (same pattern as v1 builder).
            best_idx = -1
            best_score = 0.0
            for j in range(
                max(0, html_cursor - 2),
                min(len(html_blocks), html_cursor + window),
            ):
                s = jaccard_overlap(text, html_blocks[j]["text"])
                if s > best_score:
                    best_score = s
                    best_idx = j
            if best_idx < 0 or best_score < 0.30:
                continue

            html_meta = html_blocks[best_idx]
            role = html_meta["role"]
            list_nesting = html_meta["list_nesting"]
            html_cursor = max(html_cursor, best_idx + 1)

            # Drop rows whose role isn't in the 7-class structural_role
            # vocabulary (e.g., a Role.FORM_FIELD slip-through from a
            # text-bearing <input>). The 14 dead Role values are
            # captured by other signals or downstream specialists, not
            # by this head.
            if role not in ROLE_TO_ID:
                continue

            is_heading = (
                1 if html_meta["tag"].startswith("h") and html_meta["tag"][1:].isdigit() else 0
            )

            # table_region BINARY label — independent of structural_role.
            # Positive if pdfplumber detected the block inside a table
            # region (authoritative inference-time signal) OR if the
            # ground-truth HTML has a <table> ancestor. Mirrors the
            # is_heading→HeadingSpecialist gating: 1 hands the span
            # over to the TableSpecialist; this head only DETECTS
            # table membership.
            table_region = 1 if (in_table or html_meta.get("in_table_html")) else 0

            # is_image_block BINARY label — Phase 3f gating signal.
            # HTML-only truth for now (no PDF-side image-region detector
            # yet — extract_shared currently only surfaces tables). The
            # ImageSpecialist (Phase 3f) consumes is_image_block=1 to
            # parse caption_role / caption_position / is_alt_candidate,
            # mirroring how table_region=1 hands spans to the
            # TableSpecialist.
            is_image_block = 1 if html_meta.get("in_image_block_html") else 0

            layout_vec = compute_span_layout_features(
                block,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )

            examples.append(
                {
                    "text": text,
                    "layout": layout_vec,
                    "labels": {
                        "structural_role": ROLE_TO_ID[role],
                        "is_heading": int(is_heading),
                        "table_region": int(table_region),
                        "is_image_block": int(is_image_block),
                        "list_nesting": int(list_nesting),
                    },
                    "html_tag": html_meta["tag"],
                    "source": pair.get("source", "unknown"),
                    "pair": pair_path.stem,
                }
            )
            stats["aligned"] += 1

    if examples:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return stats


# ---------------------------------------------------------------------------
# Global aligner path (structure_dataset/3) — opt-in, off by default
# ---------------------------------------------------------------------------
#
# The DEFAULT build path stays the greedy v2 loop above (byte-stable). Passing
# ``--aligner global`` (or ``SEMANTIK_STRUCTURE_ALIGNER=global``) routes the
# build through the lossless, split/merge-aware aligner in
# ``data/structure_align.py`` and writes the additive ``structure_dataset/3``
# rows to a SEPARATE ``data/structure_dataset_v3/`` tree — the greedy
# ``data/structure_dataset/`` (and ``_v2/``) outputs are never touched.
#
# The greedy loop silently DROPS ~25.6% of the gold supervision (a gold block
# no PDF block window-matches at >=0.30 just vanishes); the global path records
# every gold + pdf position in a ledger (lossless, fail-closed) and recovers
# 1-PDF->many-gold SPLIT supervision the greedy path cannot express.
#
# TODO(calibration): soft_floor=0.30 / max_run=6 are the frozen module
# defaults; calibration against a real corpus is DEFERRED — do not tune here.
GLOBAL_SOFT_FLOOR = 0.30
GLOBAL_MAX_RUN = 6


def resolve_aligner(cli_value: str | None) -> str:
    """Parse-with-fallback aligner selector (mirrors the SEMANTIK_* posture):
    explicit ``--aligner`` > ``SEMANTIK_STRUCTURE_ALIGNER`` env > ``greedy``.
    Any value other than ``global`` (case-insensitive) resolves to ``greedy``."""
    raw = cli_value if cli_value is not None else os.environ.get("SEMANTIK_STRUCTURE_ALIGNER", "")
    return "global" if str(raw).strip().lower() == "global" else "greedy"


def _pdf_views_for_pair(shared: dict) -> tuple[list[FBView], dict[int, bool]]:
    """Build :class:`FBView` instances in document reading order.

    Mirrors :func:`process_pair`'s per-page setup EXACTLY (per-page medians,
    top-to-bottom/left-to-right sort, :func:`_block_in_any_table`, and the
    SINGLE-source-of-truth :func:`compute_span_layout_features` 20-dim vector)
    and PASSES the layout vector into ``FBView`` so the aligner never recomputes
    geometry. Returns the FBView list plus an ``order_index -> in_table`` map
    (the pdf-side table-region signal ``FBView`` does not carry)."""
    views: list[FBView] = []
    in_table_by_order: dict[int, bool] = {}
    order_index = 0
    for page_no, page in enumerate(shared.get("pages", [])):
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue
        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))
        sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
        page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        heights = [
            b["bbox"][3] - b["bbox"][1]
            for b in merged
            if (b.get("bbox") and b["bbox"][3] > b["bbox"][1])
        ]
        page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
        merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        for block in merged_sorted:
            text = (block.get("text") or "").strip()
            if not text:
                continue
            in_table = _block_in_any_table(block, page)
            layout_vec = compute_span_layout_features(
                block,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )
            views.append(
                FBView.from_merged_text_block(
                    block, page=page_no, order_index=order_index, layout=layout_vec
                )
            )
            in_table_by_order[order_index] = in_table
            order_index += 1
    return views, in_table_by_order


def _resolve_gold_record(row: AlignRow, gold_records: list[dict]) -> dict | None:
    """Map an :class:`AlignRow` back to its source gold record for the
    secondary-head labels.

    MATCH/MERGE carry exactly one ``gold_index``; a SPLIT row carries the full
    member-index list, but per the frozen module's SPLIT-uses-gold-member-text
    contract its ``text`` IS the recovered member's text, so we resolve the
    precise member by text equality (falling back to the first index)."""
    gidx = [i for i in (row.align.get("gold_indices") or []) if 0 <= i < len(gold_records)]
    if not gidx:
        return None
    if len(gidx) > 1:
        for i in gidx:
            if gold_records[i].get("text") == row.text:
                return gold_records[i]
    return gold_records[gidx[0]]


def _enrich_v3_row(
    row: AlignRow, gold_records: list[dict], in_table_by_order: dict[int, bool]
) -> dict:
    """Serialize an ``AlignRow`` to its ``structure_dataset/3`` dict and
    backfill the secondary-head labels (``is_heading`` / ``table_region`` /
    ``is_image_block`` / ``list_nesting``).

    ``AlignRow.to_row()`` only fills ``labels.structural_role`` (at projection),
    but ``training/train_structure.py::load_split`` hard-subscripts the other
    head labels — so the glue enriches them here (NOT the frozen module),
    deriving ``is_heading`` from the html tag (== greedy) and the remaining
    heads from the resolved gold record + the pdf-side ``in_table`` signal
    (== greedy's ``table_region`` rule)."""
    rd = row.to_row()
    labels = rd["labels"]  # carries structural_role from project_rows
    tag = rd.get("html_tag") or ""
    labels["is_heading"] = 1 if (tag.startswith("h") and tag[1:].isdigit()) else 0
    gmeta = _resolve_gold_record(row, gold_records)
    pdf_oi = rd.get("provenance", {}).get("pdf_order_index")
    in_table_pdf = bool(in_table_by_order.get(pdf_oi, False))
    in_table_html = bool(gmeta.get("in_table_html")) if gmeta else False
    labels["table_region"] = 1 if (in_table_pdf or in_table_html) else 0
    labels["is_image_block"] = 1 if (gmeta and gmeta.get("in_image_block_html")) else 0
    labels["list_nesting"] = int(gmeta.get("list_nesting", 0)) if gmeta else 0
    return rd


def process_pair_global(validator, work: tuple) -> dict:
    """Worker (global path): globally align extract_shared blocks to the HTML
    ground truth via :func:`data.structure_align.align_blocks` and emit the
    lossless ``structure_dataset/3`` rows + a per-pair ledger sidecar."""
    pair_path_str, out_examples_dir_str = work
    pair_path = Path(pair_path_str)
    out_dir = Path(out_examples_dir_str)
    stats = {
        "pair": pair_path.name,
        "aligned": 0,
        "total_blocks": 0,
        "n_gold": 0,
        "gold_gap": 0,
        "error": None,
    }

    try:
        pair = json.loads(pair_path.read_text())
    except Exception as exc:
        stats["error"] = f"read pair: {exc}"
        return stats

    output_html = pair.get("output_html")
    if not output_html:
        stats["error"] = "no output_html"
        return stats

    gold_records = extract_html_blocks(output_html)
    if not gold_records:
        stats["error"] = "no html blocks"
        return stats

    local_pdf = pair.get("local_pdf")
    with tempfile.TemporaryDirectory() as tmp:
        if local_pdf and Path(local_pdf).exists():
            try:
                shared = extract_shared_cached(Path(local_pdf))
            except Exception as exc:
                stats["error"] = f"extract (cached): {exc}"
                return stats
        else:
            tmp_pdf = Path(tmp) / "rendered.pdf"
            try:
                validator.render_pdf(output_html, tmp_pdf)
            except Exception as exc:
                stats["error"] = f"render: {exc}"
                return stats
            try:
                shared = extract_shared(tmp_pdf)
            except Exception as exc:
                stats["error"] = f"extract: {exc}"
                return stats

    pdf_views, in_table_by_order = _pdf_views_for_pair(shared)
    gold_views = [GoldView.from_html_block(rec, i) for i, rec in enumerate(gold_records)]
    source = pair.get("source", "unknown")

    try:
        result = align_blocks(
            pdf_views,
            gold_views,
            max_run=GLOBAL_MAX_RUN,
            soft_floor=GLOBAL_SOFT_FLOOR,
            source=source,
            pair=pair_path.stem,
        )
    except LedgerIncompleteError as exc:
        stats["error"] = f"align not lossless: {exc}"
        return stats
    except Exception as exc:
        stats["error"] = f"align: {exc}"
        return stats

    kept = project_rows(result.rows, ROLE_NAMES, ledger=result.ledger)
    rows_out = [_enrich_v3_row(r, gold_records, in_table_by_order) for r in kept]

    n_pdf = len(pdf_views)
    n_gold = len(gold_views)
    summary = result.ledger.summary()
    stats["aligned"] = len(rows_out)
    stats["total_blocks"] = n_pdf
    stats["n_gold"] = n_gold
    stats["gold_gap"] = int(summary["counts"].get("gold_gap", 0))

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_out:
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in rows_out:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # Always write the ledger sidecar — even a 0-row pair consumed gold/pdf
    # positions the coverage rollup must count (and it is the resume key).
    sidecar = out_dir / f"{pair_path.stem}.ledger.json"
    sidecar.write_text(
        json.dumps(
            {
                "pair": pair_path.stem,
                "source": source,
                "n_pdf": n_pdf,
                "n_gold": n_gold,
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return stats


_LEDGER_SUFFIX = ".ledger.json"


def run_global_build(args) -> None:
    """Run the opt-in global-aligner build → ``data/structure_dataset_v3/``."""
    examples_dir: Path = args.examples_dir
    out_dir: Path = args.out_dir

    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[: args.limit_per_source]
        pair_paths.extend(files)
    print(
        f"[plan][global] {len(pair_paths)} pair files across {len(args.pair_dirs)} dirs",
        file=sys.stderr,
    )

    # Resume key = the always-written ledger sidecar (a 0-row pair has no
    # JSONL but still has a sidecar).
    done_stems = (
        {p.name[: -len(_LEDGER_SUFFIX)] for p in examples_dir.glob(f"*{_LEDGER_SUFFIX}")}
        if examples_dir.exists()
        else set()
    )
    pair_paths = [p for p in pair_paths if p.stem not in done_stems]
    print(f"[plan][global] {len(pair_paths)} to process ({len(done_stems)} done)", file=sys.stderr)

    work_items = [(str(p), str(examples_dir)) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {"aligned": 0, "total_blocks": 0, "n_gold": 0, "gold_gap": 0, "errors": 0}
    for stats in run_in_pool(process_pair_global, work_items, workers=args.workers):
        done += 1
        if stats.get("error"):
            totals["errors"] += 1
            if done % 25 == 1:
                print(
                    f"[{done}/{len(work_items)}] {stats['pair'][:50]} ERR {stats['error'][:70]}",
                    file=sys.stderr,
                )
        else:
            for k in ("aligned", "total_blocks", "n_gold", "gold_gap"):
                totals[k] += stats[k]
            if done % 25 == 0 or done == len(work_items):
                cov = 1.0 - totals["gold_gap"] / max(1, totals["n_gold"])
                print(
                    f"[{done}/{len(work_items)}] rows={totals['aligned']}  "
                    f"gold_cov={cov * 100:.1f}%  errors={totals['errors']}  "
                    f"elapsed={(time.time() - start) / 60:.1f}min",
                    file=sys.stderr,
                )

    # Merge per-pair v3 JSONL into one corpus (the .ledger.json sidecars are
    # NOT matched by *.jsonl).
    all_examples: list[dict] = []
    for p in sorted(examples_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                all_examples.append(json.loads(line))
    print(f"\n[merge][global] total v3 rows: {len(all_examples)}", file=sys.stderr)
    if not all_examples:
        raise SystemExit("no v3 examples produced")

    # Roll up every per-pair ledger (covers resumed pairs too).
    agg_counts: Counter = Counter()
    per_role: Counter = Counter()
    excluded_roles: Counter = Counter()
    projected_roles: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    total_gold = 0
    total_pdf = 0
    n_pairs = 0
    for sc in sorted(examples_dir.glob(f"*{_LEDGER_SUFFIX}")):
        d = json.loads(sc.read_text())
        n_pairs += 1
        summ = d["summary"]
        c = summ["counts"]
        agg_counts.update(c)
        per_role.update(summ["per_role"])
        excluded_roles.update(summ["excluded_roles"])
        projected_roles.update(summ["projected_roles"])
        src = d.get("source", "unknown")
        total_gold += int(d["n_gold"])
        total_pdf += int(d["n_pdf"])
        bs = by_source[src]
        bs["n_gold"] += int(d["n_gold"])
        bs["n_pdf"] += int(d["n_pdf"])
        for k in ("matched", "split", "merge", "gold_gap", "pdf_gap", "low_sim", "role_filtered"):
            bs[k] += c.get(k, 0)

    train, val, test = stratified_split(all_examples, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write][global] {path}  {len(rows)} rows", file=sys.stderr)

    # Confidence histogram + per-row tallies from the written rows.
    conf_hist = [0] * 10
    low_sim_rows = 0
    role_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for r in all_examples:
        al = r.get("align", {})
        conf = float(al.get("confidence", 0.0) or 0.0)
        bin_idx = 9 if conf >= 1.0 else max(0, min(9, int(conf * 10)))
        conf_hist[bin_idx] += 1
        if al.get("low_sim"):
            low_sim_rows += 1
        role_counts[r["labels"]["structural_role"]] += 1
        source_counts[r.get("source", "unknown")] += 1

    gold_gap = agg_counts.get("gold_gap", 0)
    gold_cov = 1.0 - gold_gap / max(1, total_gold)

    drop_ledger = {
        "by_reason": {
            "gold_gap": gold_gap,
            "pdf_gap": agg_counts.get("pdf_gap", 0),
            "role_filtered": agg_counts.get("role_filtered", 0),
        },
        "role_filtered_by_role": dict(excluded_roles),
        "by_source": {
            src: {
                "gold_total": bs["n_gold"],
                "gold_gap": bs["gold_gap"],
                "pdf_gap": bs["pdf_gap"],
                "role_filtered": bs["role_filtered"],
            }
            for src, bs in sorted(by_source.items())
        },
    }
    confidence_histogram = {
        f"{i / 10:.1f}-{(i + 1) / 10:.1f}": conf_hist[i] for i in range(10)
    }
    move_counts = {k: agg_counts.get(k, 0) for k in ("matched", "split", "merge", "gold_gap", "pdf_gap")}

    coverage = {
        "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
        "aligner": "global",
        "soft_floor": GLOBAL_SOFT_FLOOR,
        "max_run": GLOBAL_MAX_RUN,
        "calibration_note": "TODO(calibration): soft_floor=0.30 / max_run=6 are frozen defaults, not tuned",
        "n_pairs": n_pairs,
        "n_examples_total": len(all_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "gold_total_positions": total_gold,
        "pdf_total_positions": total_pdf,
        "gold_coverage_rate": round(gold_cov, 6),
        "move_counts": move_counts,
        "low_sim_moves": agg_counts.get("low_sim", 0),
        "low_sim_rows": low_sim_rows,
        "drop_ledger": drop_ledger,
        "confidence_histogram": confidence_histogram,
        "structural_role_counts": {ROLE_NAMES[k]: v for k, v in role_counts.items()},
        "source_counts": dict(source_counts),
        "projected_roles": dict(projected_roles),
        "excluded_roles": dict(excluded_roles),
        "by_source": {src: dict(bs) for src, bs in sorted(by_source.items())},
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_feature_names": LAYOUT_FEATURE_NAMES,
        "role_names": list(ROLE_NAMES),
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    print(f"[save][global] coverage report -> {out_dir / 'coverage_report.json'}", file=sys.stderr)
    print(
        f"[GLOBAL] gold_coverage_rate={gold_cov * 100:.2f}%  "
        f"rows={len(all_examples)}  pairs={n_pairs}  "
        f"matched={move_counts['matched']} split={move_counts['split']} "
        f"merge={move_counts['merge']} gold_gap={move_counts['gold_gap']} "
        f"pdf_gap={move_counts['pdf_gap']}  low_sim_rows={low_sim_rows}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Stratified split + main
# ---------------------------------------------------------------------------


def stratified_split(
    rows: list[dict],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratify by structural_role so rare classes appear in all splits."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["labels"]["structural_role"]].append(r)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(lst[:n_train])
        val.extend(lst[n_train : n_train + n_val])
        test.extend(lst[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pair-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/pairs/wikipedia"),
            Path("data/pairs/openstax"),
            Path("data/pairs/federal_register"),
            Path("data/pairs/gutenberg"),
            Path("data/pairs/forms"),
            Path("data/pairs/arxiv"),
            Path("data/pairs/synthetic_blockquote_code"),
        ],
    )
    # Defaults resolved AFTER --aligner so the greedy path keeps its exact
    # historical dirs (data/structure_dataset[/per_pair]) byte-for-byte while
    # the global path lands in a SEPARATE data/structure_dataset_v3/ tree.
    ap.add_argument("--examples-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--aligner",
        choices=["greedy", "global"],
        default=None,
        help="Block→gold aligner. 'greedy' (default) = the byte-stable v2 "
        "monotonic-Jaccard loop → data/structure_dataset/. 'global' = the "
        "lossless split/merge-aware aligner (data/structure_align.py) → "
        "data/structure_dataset_v3/. Env override: SEMANTIK_STRUCTURE_ALIGNER.",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Cap on pair files processed per source (smoke).",
    )
    ap.add_argument(
        "--include-legal-pseudo",
        action="store_true",
        help="Append the pre-balanced pseudo-labeled legal set "
        "(data/structure_dataset/per_pair_legal/courtlistener_pseudo.jsonl, built by "
        "data/build_courtlistener_pseudo_structure.py) AFTER source-capping, so its rare "
        "legal headings are not re-subsampled away (Plan 14).",
    )
    add_cap_args(ap)
    args = ap.parse_args()

    aligner = resolve_aligner(args.aligner)
    default_out = (
        Path("data/structure_dataset_v3") if aligner == "global" else Path("data/structure_dataset")
    )
    args.out_dir = args.out_dir or default_out
    args.examples_dir = args.examples_dir or (args.out_dir / "per_pair")

    if aligner == "global":
        run_global_build(args)
        return

    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[: args.limit_per_source]
        pair_paths.extend(files)
    print(f"[plan] {len(pair_paths)} pair files across {len(args.pair_dirs)} dirs", file=sys.stderr)

    existing = (
        {p.stem for p in args.examples_dir.glob("*.jsonl")} if args.examples_dir.exists() else set()
    )
    pair_paths = [p for p in pair_paths if p.stem not in existing]
    print(f"[plan] {len(pair_paths)} to process ({len(existing)} done)", file=sys.stderr)

    work_items = [(str(p), str(args.examples_dir)) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {"aligned": 0, "total_blocks": 0, "errors": 0}
    for stats in run_in_pool(process_pair, work_items, workers=args.workers):
        done += 1
        if stats.get("error"):
            totals["errors"] += 1
            if done % 25 == 1:
                print(
                    f"[{done}/{len(work_items)}] {stats['pair'][:50]} ERR {stats['error'][:60]}",
                    file=sys.stderr,
                )
        else:
            totals["aligned"] += stats["aligned"]
            totals["total_blocks"] += stats["total_blocks"]
            if done % 50 == 0 or done == len(work_items):
                rate = totals["aligned"] / max(1, totals["total_blocks"]) * 100
                print(
                    f"[{done}/{len(work_items)}] aligned={totals['aligned']}  "
                    f"align_rate={rate:.1f}%  "
                    f"elapsed={(time.time() - start) / 60:.1f}min",
                    file=sys.stderr,
                )

    # Merge per-pair JSONL into one corpus.
    all_examples: list[dict] = []
    for p in sorted(args.examples_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                all_examples.append(json.loads(line))
    print(f"\n[merge] total labeled examples: {len(all_examples)}", file=sys.stderr)

    if not all_examples:
        raise SystemExit("no examples produced")

    all_examples, cap_report = apply_caps_and_report(
        all_examples, args, label_key="structural_role"
    )

    # Append the pre-balanced pseudo-labeled legal set AFTER capping (Plan 14):
    # it is already balanced in its own builder (all headings/lists kept, body
    # paragraphs capped), so re-running it through apply_caps_and_report's
    # uniform per-source subsample would drop the rare legal headings we
    # specifically need. Opt-in via --include-legal-pseudo.
    if args.include_legal_pseudo:
        legal_path = args.examples_dir.parent / "per_pair_legal" / "courtlistener_pseudo.jsonl"
        if not legal_path.exists():
            raise SystemExit(
                f"--include-legal-pseudo set but {legal_path} missing; run "
                "data/build_courtlistener_pseudo_structure.py --build first."
            )
        legal = [json.loads(ln) for ln in legal_path.read_text().splitlines() if ln.strip()]
        all_examples.extend(legal)
        print(
            f"[merge] + {len(legal)} pre-balanced legal examples (uncapped) from {legal_path}",
            file=sys.stderr,
        )

    train, val, test = stratified_split(all_examples, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {path}  {len(rows)} rows", file=sys.stderr)

    # Per-head class prevalence reports.
    role_counts = Counter(r["labels"]["structural_role"] for r in all_examples)
    is_heading_counts = Counter(r["labels"]["is_heading"] for r in all_examples)
    table_region_counts = Counter(r["labels"]["table_region"] for r in all_examples)
    is_image_block_counts = Counter(r["labels"]["is_image_block"] for r in all_examples)
    list_nesting_counts = Counter(r["labels"]["list_nesting"] for r in all_examples)
    source_counts = Counter(r["source"] for r in all_examples)

    print("\n[balance] structural_role:")
    for cls_id, n in role_counts.most_common():
        pct = 100.0 * n / max(1, len(all_examples))
        print(f"  {ROLE_NAMES[cls_id]:24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] is_heading:")
    for k in (0, 1):
        n = is_heading_counts.get(k, 0)
        pct = 100.0 * n / max(1, len(all_examples))
        print(f"  {'heading' if k else 'not_heading':24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] table_region:")
    for k in (0, 1):
        n = table_region_counts.get(k, 0)
        pct = 100.0 * n / max(1, len(all_examples))
        label = "table_region" if k else "not_table_region"
        print(f"  {label:24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] is_image_block:")
    for k in (0, 1):
        n = is_image_block_counts.get(k, 0)
        pct = 100.0 * n / max(1, len(all_examples))
        label = "image_block" if k else "not_image_block"
        print(f"  {label:24} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] list_nesting:")
    for k in LIST_NESTING_BUCKETS:
        n = list_nesting_counts.get(k, 0)
        pct = 100.0 * n / max(1, len(all_examples))
        print(f"  depth={k:<18} {n:7}  ({pct:5.2f}%)")
    print("\n[balance] by source:")
    for src, n in source_counts.most_common():
        print(f"  {src:24} {n}")

    coverage = {
        "n_examples_total": len(all_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "structural_role_counts": {ROLE_NAMES[k]: v for k, v in role_counts.items()},
        "is_heading_counts": {str(k): v for k, v in is_heading_counts.items()},
        "table_region_counts": {str(k): v for k, v in table_region_counts.items()},
        "is_image_block_counts": {str(k): v for k, v in is_image_block_counts.items()},
        "list_nesting_counts": {
            str(k): list_nesting_counts.get(k, 0) for k in LIST_NESTING_BUCKETS
        },
        "source_counts": dict(source_counts),
        "source_caps": cap_report,
        "max_examples_per_source": args.max_examples_per_source,
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_feature_names": LAYOUT_FEATURE_NAMES,
        "role_names": ROLE_NAMES,
    }
    (args.out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
