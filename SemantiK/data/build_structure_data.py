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
from dart_semantic.reading_order import (
    column_ids_for_bboxes,
    resolve_column_order_mode,
)
from dart_semantic.text_utils import jaccard_overlap
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool
from data import structure_align
from data.render_augment import (
    augment_html,
    resolve_render_augment_mode,
    resolve_render_profile_for,
)
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
#
# AXIS-1 expansion (this wave): three roles were added so the head can EMIT
# what gold needs instead of collapsing the shape onto `paragraph`:
#   * definition_term / definition_def — <dt>/<dd> definition-list leaves
#     (previously had no role; the dl family was simply not extracted).
#   * caption — <caption>/<figcaption> text (previously collapsed onto
#     `paragraph`, erasing the "this is a caption" signal Phase-1 measured as
#     a real-PDF gap). The TABLE_* family + FIGURE stay dropped (table_region
#     / is_image_block binary heads + downstream specialists own them).
ROLE_LIST = (
    Role.PARAGRAPH,
    Role.HEADING,
    Role.LIST_ITEM,
    Role.FORM_LABEL,
    Role.BLOCKQUOTE,
    Role.CODE_BLOCK,
    Role.DEFINITION_TERM,
    Role.DEFINITION_DEF,
    Role.CAPTION,
)
ROLE_NAMES = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
    "definition_term",
    "definition_def",
    "caption",
)
ROLE_TO_ID = {r: i for i, r in enumerate(ROLE_LIST)}
NUM_ROLES = len(ROLE_LIST)

# list_nesting is 4-class for ALL spans (0 means "not in a list", which
# is true for most spans).
LIST_NESTING_BUCKETS = (0, 1, 2, 3)

# ---------------------------------------------------------------------------
# AXIS-2: pedagogical_role head vocabulary
# ---------------------------------------------------------------------------
#
# A NEW span-level head ORTHOGONAL to structural_role: structural_role says
# WHAT SHAPE a span is (paragraph / heading / …); pedagogical_role says WHAT
# PEDAGOGICAL FUNCTION it opens (an EXAMPLE / a Solution / a TRY IT / …). The
# council currently collapses these into code_block/heading, which Phase 1
# measured as a major source of the real-PDF gap (43/62 "code_blocks" are
# really exercises). Class 0 == `none` (the overwhelming majority of spans
# carry no pedagogical-label function).
#
# Gold labels are extracted by REUSING the single-source-of-truth
# `_PEDAGOGICAL_LABEL_CLASSES` regexes from
# `dart_semantic.qwen_specialists.deterministic_structure` (so the gold-side
# label predicate can never drift from the always-on clean pass) plus one
# supplementary `exercise_open` regex (OpenStax "Exercises" has no pedagogy-*
# CSS class of its own). Non-matches → `none`.
PEDAGOGICAL_ROLE_NAMES = (
    "none",
    "example_open",
    "exercise_open",
    "solution",
    "try_it",
    "how_to",
    "objectives",
    "be_prepared",
    "practice",
    "step",
)
PEDAGOGICAL_ROLE_TO_ID = {r: i for i, r in enumerate(PEDAGOGICAL_ROLE_NAMES)}
NUM_PEDAGOGICAL_ROLES = len(PEDAGOGICAL_ROLE_NAMES)

# Map the deterministic clean-pass CSS-class hints
# (`_PEDAGOGICAL_LABEL_CLASSES` values) onto pedagogical_role tokens. Every
# value is a member of PEDAGOGICAL_ROLE_NAMES.
_PEDAGOGY_CSS_TO_ROLE = {
    "pedagogy-example": "example_open",
    "pedagogy-solution": "solution",
    "pedagogy-try-it": "try_it",
    "pedagogy-how-to": "how_to",
    "pedagogy-step": "step",
    "pedagogy-objectives": "objectives",
    "pedagogy-be-prepared": "be_prepared",
    "pedagogy-practice": "practice",
}

# Supplementary gold-side regex for `exercise_open` — there is no
# `pedagogy-exercise` CSS class in `_PEDAGOGICAL_LABEL_CLASSES`, but an
# end-of-section "EXERCISES" / "Exercise N" block is a distinct pedagogical
# function the head should learn. Checked AFTER the reused clean-pass classes
# (which never match a bare "Exercises" label), so it only fires on a true
# miss of the reused set.
import re as _re  # noqa: E402  (local alias; module already imports re-free)

_EXERCISE_OPEN_RE = _re.compile(r"^\s*Exercises?\b(?:\s+\d+(?:\.\d+)*)?", _re.IGNORECASE)


def pedagogical_role_for(text: str) -> str:
    """Gold-side pedagogical_role for a span's ``text``; ``none`` on no match.

    Reuses the frozen ``_PEDAGOGICAL_LABEL_CLASSES`` regexes (via
    ``_pedagogical_class_for``) so the gold label predicate is byte-identical
    to the always-on deterministic clean pass, then falls back to the local
    ``exercise_open`` regex. Anti-fabrication: only labels a span whose text
    actually opens with a recognized pedagogical label."""
    if not text:
        return "none"
    # Import lazily so build_structure_data has no import-time dependency on the
    # qwen_specialists package (keeps the data builder importable in lean CI).
    from dart_semantic.qwen_specialists.deterministic_structure import (
        _pedagogical_class_for,
    )

    css = _pedagogical_class_for(text)
    if css and css in _PEDAGOGY_CSS_TO_ROLE:
        return _PEDAGOGY_CSS_TO_ROLE[css]
    if _EXERCISE_OPEN_RE.match(text):
        return "exercise_open"
    return "none"


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
    # AXIS-1: <caption>/<figcaption> are CAPTION shape now (was PARAGRAPH).
    # The text "is a caption" is a real structural signal gold needs; the
    # is_image_block binary head + the future ImageSpecialist still gate the
    # figure-caption EMISSION downstream, but the head can now recommend the
    # caption shape directly.
    "caption": Role.CAPTION,
    "figcaption": Role.CAPTION,
    # AXIS-1: definition-list leaves. <dt> = term, <dd> = definition. The <dl>
    # CONTAINER is intentionally NOT a target tag (adding it would emit a
    # duplicate whole-list block whose role can only be one of the two leaf
    # shapes); dt/dd are added to OUTER_TAG_ALLOW_NESTED so they survive the
    # nest-skip inside their <dl> parent and emit as the two distinct leaves.
    "dt": Role.DEFINITION_TERM,
    "dd": Role.DEFINITION_DEF,
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
    # AXIS-1: dt/dd are natural-nest leaves inside <dl> — emit them as the two
    # distinct definition-list shapes rather than skipping them under the dl.
    "dt",
    "dd",
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


def _reading_order_sorted(merged: list[dict], page_w: float) -> list[dict]:
    """Order one page's merged text blocks for alignment / feature extraction.

    Default (``SEMANTIK_COLUMN_ORDER`` off, ``resolve_column_order_mode()`` ->
    False) is the byte-identical legacy raster key ``(y0, x0)``. When the flag
    is on, the blocks are clustered into columns by left-edge ``x0`` (reusing
    the committed cascade core ``dart_semantic.reading_order.column_ids_for_bboxes``)
    and sorted column-major ``(column_index, y0, x0)`` so a two-column page is
    read down one column then the next instead of line-interleaved across the
    gutter. On a single-column page the clustering yields one column for every
    block, so the column-major key collapses to the raster key -> identical
    output on/off. Operates per page (the caller iterates pages), matching the
    cascade's per-page column detection."""
    if not resolve_column_order_mode():
        return sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    bboxes = [b.get("bbox") or (0.0, 0.0, 0.0, 0.0) for b in merged]
    col_ids = column_ids_for_bboxes(bboxes, page_w)
    order = sorted(
        range(len(merged)),
        key=lambda i: (col_ids[i], merged[i]["bbox"][1], merged[i]["bbox"][0]),
    )
    return [merged[i] for i in order]


# ---------------------------------------------------------------------------
# Realistic-render augmentation hook (presentation only; default OFF)
# ---------------------------------------------------------------------------
#
# When SEMANTIK_RENDER_AUGMENT is on, the gold output_html is rendered through a
# realistic PRESENTATION profile (2-column / serif / dense) BEFORE Playwright
# renders the PDF, so the extracted training/eval blocks look deploy-LIKE while
# the labels (read from the ORIGINAL output_html) stay clean. Default OFF =>
# augment_html is the identity => byte-identical render to today.


def _stable_doc_index(pair_stem: str) -> int:
    """Deterministic non-negative int per pair stem (the seed/profile key).

    Independent of process / iteration order so a worker-pool build assigns the
    same profile to a pair on every run (reproducible domain randomization)."""
    import hashlib

    h = hashlib.sha1((pair_stem or "").encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _augment_html_for_render(output_html: str, pair_stem: str) -> tuple[str, str, int]:
    """Resolve the render profile for one pair and return
    ``(html_to_render, profile, seed)``.

    Default (``SEMANTIK_RENDER_AUGMENT`` off) -> ``(output_html, "clean", 0)``,
    i.e. the input is returned UNCHANGED so the render is byte-identical."""
    mode = resolve_render_augment_mode()
    if mode == "off":
        return output_html, "clean", 0
    profile, seed = resolve_render_profile_for(_stable_doc_index(pair_stem), mode)
    return augment_html(output_html, profile, seed), profile, seed


# ---------------------------------------------------------------------------
# Worker — process one pair file
# ---------------------------------------------------------------------------


def _exact_render_extract_label(validator, output_html: str, pair_stem: str, stats: dict):
    """Shared EXACT render+extract+label core for both the greedy
    (:func:`_process_pair_exact`) and global (:func:`_process_pair_global_exact`)
    exact arms.

    Renders the gold HTML through the box-capturing single-tall-page Playwright
    path (REUSING the worker's existing Chromium context — sync Playwright
    cannot be nested), extracts the rendered PDF, and assigns each extracted PDF
    block the role of the gold element whose mapped bbox contains it. Applies
    the SEMANTIK_RENDER_AUGMENT profile to the rendered HTML exactly as the
    realistic-eval / process_pair exact path does (deploy-like renders).

    Returns the :class:`~data.render_capture.LabelResult` on success, or
    ``None`` with ``stats['error']`` set on render/extract failure."""
    from data.render_capture import label_blocks_by_overlap, render_with_boxes

    html_to_render, _profile, _seed = _augment_html_for_render(output_html, pair_stem)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            capture = render_with_boxes(
                html_to_render,
                context=getattr(validator, "_context", None),
                tmpdir=Path(tmp),
            )
        except Exception as exc:
            stats["error"] = f"render (exact): {exc}"
            return None
        try:
            shared = extract_shared(capture.pdf_path)
        except Exception as exc:
            stats["error"] = f"extract (exact): {exc}"
            return None
        result = label_blocks_by_overlap(shared, capture)
    # Surface the single-tall-page overflow flag (capture lives only inside the
    # tmpdir scope, so stash it on the result for the caller).
    result.overflowed_single_page = bool(capture.overflowed_single_page)
    return result


def _process_pair_exact(
    validator, pair: dict, pair_path: Path, out_dir: Path, output_html: str, stats: dict
) -> dict:
    """EXACT-labeling worker arm (SEMANTIK_EXACT_RENDER_LABELS).

    Replaces validator.render_pdf + the greedy text aligner with
    render_capture.render_with_boxes + label_blocks_by_overlap: the gold HTML is
    rendered through the box-capturing single-tall-page Playwright path
    (REUSING the worker's existing Chromium context — sync Playwright cannot be
    nested) and each extracted PDF block is assigned the role of the gold
    element whose mapped bbox contains it. Emits the SAME structure_dataset row
    shape (text + 20-dim ``compute_span_layout_features`` layout + the 5 hard
    head labels + pedagogical_role + html_tag/source/pair). UNMATCHED blocks
    (and blocks whose role is outside the head vocab) are EXCLUDED from emitted
    rows; the unmatched rate is reported on ``stats``."""
    result = _exact_render_extract_label(validator, output_html, pair_path.stem, stats)
    if result is None:
        return stats

    source = pair.get("source", "unknown")
    examples: list[dict] = []
    for r in result.rows:
        labels = r.get("labels")
        # Exclude unmatched blocks (labels is None) AND matched blocks whose
        # role falls outside the structural_role head vocab (structural_role is
        # None) — mirrors process_pair's `if role not in ROLE_TO_ID: continue`.
        if not labels or labels.get("structural_role") is None:
            continue
        examples.append(
            {
                "text": r["text"],
                "layout": r["layout"],
                "labels": {
                    "structural_role": labels["structural_role"],
                    "is_heading": int(labels["is_heading"]),
                    "table_region": int(labels["table_region"]),
                    "is_image_block": int(labels["is_image_block"]),
                    "list_nesting": int(labels["list_nesting"]),
                    "pedagogical_role": int(labels["pedagogical_role"]),
                },
                "html_tag": r["html_tag"],
                "source": source,
                "pair": pair_path.stem,
            }
        )

    stats["total_blocks"] = result.n_blocks
    stats["aligned"] = len(examples)
    stats["matched"] = result.n_matched
    stats["unmatched"] = result.n_unmatched
    stats["unmatched_rate"] = round(result.unmatched_rate, 6)
    stats["exact"] = True
    stats["overflowed_single_page"] = bool(getattr(result, "overflowed_single_page", False))

    if examples:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return stats


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

    # EXACT render-time bbox->role labeling (opt-in SEMANTIK_EXACT_RENDER_LABELS,
    # default OFF). In exact mode the gold HTML is ALWAYS rendered (exact needs a
    # render to capture element boxes), so the gold role of every extracted PDF
    # block is recoverable EXACTLY by geometry instead of fuzzily by text
    # overlap. The `not local_pdf` guard is DELIBERATELY DROPPED for the exact
    # path: arxiv pairs all carry a local_pdf, but the realistic EVAL renders
    # arxiv gold HTML — so to MATCH the eval, exact-mode training must render the
    # gold too (it does NOT consume local_pdf). Default off -> this branch is
    # skipped and the fuzzy path below (which still PREFERS local_pdf) is
    # byte-identical to today.
    from data.render_capture import exact_render_labels_enabled  # lazy: avoids circular import

    if exact_render_labels_enabled():
        return _process_pair_exact(validator, pair, pair_path, out_dir, output_html, stats)

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
            html_to_render, _profile, _seed = _augment_html_for_render(
                output_html, pair_path.stem
            )
            try:
                validator.render_pdf(html_to_render, tmp_pdf)
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
        # context-windowed Jaccard). Column-aware when SEMANTIK_COLUMN_ORDER
        # is on (byte-identical raster sort when off — the default).
        merged_sorted = _reading_order_sorted(merged, page_w)

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
        # Column-aware when SEMANTIK_COLUMN_ORDER is on; byte-identical raster
        # sort when off (the default) — closes the "eval build ignores the
        # flag" gap (the build previously raster-sorted unconditionally here).
        merged_sorted = _reading_order_sorted(merged, page_w)
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
    # AXIS-2: pedagogical_role label. Derived from the SAME text the model
    # reads at train time (``rd["text"]`` == ``_format_input``'s input) so the
    # label and the input are consistent across MATCH/SPLIT/MERGE; the gold
    # text is preferred where the row carries it (SPLIT members) but rd["text"]
    # already is that text by the frozen aligner's SPLIT contract.
    labels["pedagogical_role"] = PEDAGOGICAL_ROLE_TO_ID[
        pedagogical_role_for(rd.get("text") or "")
    ]
    return rd


def _process_pair_global_exact(
    validator,
    pair: dict,
    pair_path: Path,
    out_dir: Path,
    output_html: str,
    gold_records: list,
    stats: dict,
) -> dict:
    """EXACT-labeling arm of the GLOBAL build path (SEMANTIK_EXACT_RENDER_LABELS).

    Mirrors :func:`_process_pair_exact` but emits the GLOBAL per-pair shard
    contract verbatim — the ``structure_dataset/3`` row shape
    (``to_row``-shaped: text + 20-dim layout + the 6 head labels + html_tag +
    source + pair + schema_version + provenance + align) PLUS the always-written
    ``{stem}.ledger.json`` sidecar (the resume key + rollup input). The fuzzy
    aligner / ledger is REPLACED by exact geometric containment, so the rows
    carry exact provenance (``align.exact=True``, ``align.confidence=1.0``, NO
    ``low_sim`` key) instead of a similarity. ``load_split`` consumes the rows
    unchanged (it reads only ``labels`` / ``layout`` / ``text`` / ``source``)."""
    result = _exact_render_extract_label(validator, output_html, pair_path.stem, stats)
    if result is None:
        return stats

    source = pair.get("source", "unknown")
    rows_out: list[dict] = []
    per_role: Counter = Counter()
    for r in result.rows:
        labels = r.get("labels")
        # Exclude unmatched blocks (labels is None) AND matched blocks whose
        # role is outside the structural_role head vocab (structural_role is
        # None) — mirrors _process_pair_exact / process_pair's ROLE_TO_ID filter.
        if not labels or labels.get("structural_role") is None:
            continue
        match = r.get("match") or {}
        match_kind = match.get("kind") or "contained"
        rows_out.append(
            {
                "text": r["text"],
                "layout": r["layout"],
                "labels": {
                    "structural_role": labels["structural_role"],
                    "is_heading": int(labels["is_heading"]),
                    "table_region": int(labels["table_region"]),
                    "is_image_block": int(labels["is_image_block"]),
                    "list_nesting": int(labels["list_nesting"]),
                    "pedagogical_role": int(labels["pedagogical_role"]),
                },
                "html_tag": r["html_tag"],
                "source": source,
                "pair": pair_path.stem,
                "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
                "exact_label": True,
                "provenance": {"rc_id": r.get("rc_id")},
                # Exact provenance markers (NOT a fuzzy similarity): confidence
                # is 1.0 (containment IS the match) and there is deliberately NO
                # `low_sim` key — run_global_build's confidence histogram lands
                # every exact row in the 1.0 bin and low_sim_rows stays 0.
                "align": {
                    "kind": f"exact_{match_kind}",
                    "exact": True,
                    "confidence": 1.0,
                    "overlap": match.get("overlap"),
                },
            }
        )
        role_name = r.get("role")
        if role_name is not None:
            per_role[str(role_name)] += 1

    n_pdf = result.n_blocks
    n_gold = len(gold_records)
    stats["aligned"] = len(rows_out)
    stats["total_blocks"] = n_pdf
    stats["n_gold"] = n_gold
    stats["gold_gap"] = 0  # exact has no gold-side ledger; PDF-block centric
    stats["exact"] = True
    stats["unmatched"] = result.n_unmatched
    stats["unmatched_rate"] = round(result.unmatched_rate, 6)

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_out:
        out_path = out_dir / f"{pair_path.stem}.jsonl"
        with out_path.open("w") as f:
            for ex in rows_out:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Always write the ledger sidecar — it is the resume key AND the rollup
    # input in run_global_build (a 0-row pair still consumed positions). The
    # exact summary carries exact markers; the fuzzy buckets (split/merge/
    # gold_gap/pdf_gap/low_sim/role_filtered) are absent → counted as 0 by the
    # rollup's `.get(..., 0)`, which is the honest exact value.
    summary = {
        "counts": {
            "matched": result.n_matched,
            "exact": 1,
            "exact_contained": result.n_contained,
            "exact_overlap": result.n_overlap_only,
            "exact_unmatched": result.n_unmatched,
        },
        "per_role": dict(per_role),
        "per_source": {source: len(rows_out)} if rows_out else {},
        "projected_roles": {},
        "excluded_roles": {},
    }
    sidecar = out_dir / f"{pair_path.stem}.ledger.json"
    sidecar.write_text(
        json.dumps(
            {
                "pair": pair_path.stem,
                "source": source,
                "n_pdf": n_pdf,
                "n_gold": n_gold,
                "exact": True,
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    # Hand per-pair native allocator arenas back to the OS (mirrors the fuzzy
    # global path's malloc_trim).
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    return stats


def process_pair_global(validator, work: tuple) -> dict:
    """Worker (global path): globally align extract_shared blocks to the HTML
    ground truth via :func:`data.structure_align.align_blocks` and emit the
    lossless ``structure_dataset/3`` rows + a per-pair ledger sidecar."""
    # Backward-compatible unpack: a 3rd element carries the rendered-path block
    # cap (mirrors --realpdf-max-blocks); legacy 2-tuples disable the cap.
    pair_path_str, out_examples_dir_str, *rest = work
    max_blocks = int(rest[0]) if rest else 0
    pair_path = Path(pair_path_str)
    out_dir = Path(out_examples_dir_str)
    stats = {
        "pair": pair_path.name,
        "aligned": 0,
        "total_blocks": 0,
        "n_gold": 0,
        "gold_gap": 0,
        "too_large": 0,
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

    # EXACT render-time bbox->role labeling (opt-in SEMANTIK_EXACT_RENDER_LABELS,
    # default OFF). When on, the GLOBAL path routes through the exact arm — it
    # ALWAYS renders the gold HTML (exact needs a render; the `not local_pdf`
    # guard is dropped so arxiv renders gold+exact, matching the realistic eval)
    # and emits the SAME global per-pair shard contract (v3 rows + ledger
    # sidecar) with exact provenance instead of the fuzzy aligner's ledger.
    # Default off -> this branch is skipped and the fuzzy global path below
    # (which still PREFERS local_pdf) is byte-identical to today.
    from data.render_capture import exact_render_labels_enabled  # lazy: avoids circular import

    if exact_render_labels_enabled():
        return _process_pair_global_exact(
            validator, pair, pair_path, out_dir, output_html, gold_records, stats
        )

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
            html_to_render, _profile, _seed = _augment_html_for_render(
                output_html, pair_path.stem
            )
            try:
                validator.render_pdf(html_to_render, tmp_pdf)
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

    # Size guard (mirrors the real-PDF path): a freak huge page would spike the
    # aligner's O(n_pdf * n_gold * max_run) DP (a transient memory/CPU balloon,
    # not the monotonic leak). Record a too_large drop and skip.
    if max_blocks and (len(pdf_views) > max_blocks or len(gold_views) > max_blocks):
        stats["too_large"] = 1
        return stats

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
    finally:
        # The aligner's sim() lru_cache (maxsize=None) grows ~750B/entry with
        # ~0 cross-pair reuse — reset it to baseline each pair. The within-pair
        # DP memoization is preserved (cleared only after this pair's align).
        try:
            structure_align._sim_cached.cache_clear()
        except Exception:
            pass

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
    # Return per-pair native allocator arenas to the OS. pypdfium2/pdfplumber
    # free their per-page buffers, but glibc malloc keeps the freed arenas
    # mapped (the tracemalloc-invisible RSS ratchet). malloc_trim(0) hands them
    # back. Best-effort: a no-op on non-glibc / musl libc.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
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

    max_blocks = getattr(args, "global_max_blocks", 0) or 0
    work_items = [(str(p), str(examples_dir), max_blocks) for p in pair_paths]
    start = time.time()
    done = 0
    totals = {"aligned": 0, "total_blocks": 0, "n_gold": 0, "gold_gap": 0, "errors": 0, "too_large": 0}
    for stats in run_in_pool(process_pair_global, work_items, workers=args.workers):
        done += 1
        if stats.get("too_large"):
            totals["too_large"] += 1
        elif stats.get("error"):
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
                    f"too_large={totals['too_large']}  "
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
    ped_role_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for r in all_examples:
        al = r.get("align", {})
        conf = float(al.get("confidence", 0.0) or 0.0)
        bin_idx = 9 if conf >= 1.0 else max(0, min(9, int(conf * 10)))
        conf_hist[bin_idx] += 1
        if al.get("low_sim"):
            low_sim_rows += 1
        role_counts[r["labels"]["structural_role"]] += 1
        ped_role_counts[r["labels"].get("pedagogical_role", 0)] += 1
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
        "pedagogical_role_counts": {
            PEDAGOGICAL_ROLE_NAMES[k]: v for k, v in ped_role_counts.items()
        },
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
# Real-PDF eval-set build path (Phase 1) — opt-in, off by default
# ---------------------------------------------------------------------------
#
# Today's training + eval live on RENDERED-HTML-PDF gold (output_html -> a
# Playwright-rendered PDF -> extract_shared). The real deploy gap is therefore
# UNMEASURED. This path builds a real-PDF eval set: it emits ``FBView``s from a
# REAL source PDF (a born-digital arXiv paper on disk, an IRS form PDF, a
# Federal Register PDF) and aligns them against the EXISTING gold ``output_html``
# via the SAME frozen ``data/structure_align.py`` aligner the rendered path uses
# (nothing in the aligner changes — it consumes input-agnostic views). The v3
# rows it writes are then scored by the CURRENT committed structure head to
# measure real-PDF gold-reproduction F1 vs the in-distribution 0.894.
#
# Three load-bearing guarantees:
#   * LICENSE PARTITION — two physically separate roots. SHIPPABLE (PMC OA /
#     federal_register / CFR / gov_forms + arXiv papers whose cached license is
#     commercial-OK) vs INTERNAL-ONLY (OpenStax NC-SA + arXiv papers whose
#     license is non-commercial OR unknown). Gated per pair on source+license.
#   * ZERO LEAKAGE — ``--eval-holdout-by-docid`` holds the eval set out by
#     DOCUMENT id against the head's actual training corpus
#     (``data/structure_dataset/{train,val}.jsonl``): a doc whose rendered rows
#     touched train/val is EXCLUDED from the emitted (clean) test.jsonl and
#     routed to a clearly-named ``test_in_training.jsonl`` sidecar instead.
#   * ALIGNMENT-QUALITY GATE — a real-PDF doc whose aligned-gold fraction is
#     below ``--align-quality-floor`` is DROPPED (never mislabeled); drops are
#     counted per source/domain so a low head score is attributable to
#     aligner-vs-head, not silent contamination.

# Domain taxonomy for the per-domain real-PDF F1 table.
SOURCE_DOMAIN = {
    "arxiv": "scientific",
    "pmc": "scientific",
    "cfr": "regulatory_legal",
    "federal_register": "regulatory_legal",
    "pdf_form": "forms",
    "forms": "forms",
    "openstax": "math",
}

# Sources whose corpus license permits redistribution unconditionally. arXiv is
# gated PER PAPER on the cached license map (see ``_license_partition``).
_SHIPPABLE_SOURCES = {"pmc", "federal_register", "cfr", "pdf_form", "forms"}

# data/pairs/<dir> -> the canonical ``source`` field carried INSIDE the pair
# JSON (used for leakage keying; only ``forms`` differs from its dir name).
_DIR_TO_SOURCE = {"forms": "pdf_form"}

_ARXIV_LICENSE_MAP_PATH = Path("data/figure_images/_arxiv_license_map.json")
_RENDERED_TRAIN_CORPUS = Path("data/structure_dataset")  # the head's training data


def _domain_for(source: str) -> str:
    return SOURCE_DOMAIN.get(source, "other")


def _doc_id_for(source: str, pair_stem: str) -> str:
    """Stable per-document id. arXiv/OpenStax stems are
    ``<docid>__<NN>_<section_slug>`` (one doc -> many section pairs); the others
    are one pair per document."""
    if source in ("arxiv", "openstax"):
        return pair_stem.split("__")[0]
    return pair_stem


def _load_arxiv_license_map() -> dict:
    try:
        return json.loads(_ARXIV_LICENSE_MAP_PATH.read_text())
    except Exception:
        return {}


def _license_partition(source: str, doc_id: str, arxiv_lic: dict) -> str:
    """Return ``"shippable"`` or ``"internal"`` for one document."""
    if source in _SHIPPABLE_SOURCES:
        return "shippable"
    if source == "arxiv":
        rec = arxiv_lic.get(doc_id.replace("_", "."))
        if rec and rec.get("commercial_ok"):
            return "shippable"
        return "internal"  # non-commercial OR unknown -> cannot prove shippable
    return "internal"  # OpenStax NC-SA et al.


def _train_val_docids(corpus_dir: Path = _RENDERED_TRAIN_CORPUS) -> set:
    """The (source, doc_id) set the committed head actually trained on, read
    from the rendered train+val splits — the leakage exclusion set."""
    seen: set = set()
    for split in ("train", "val"):
        p = corpus_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            src = r.get("source", "?")
            seen.add((src, _doc_id_for(src, r.get("pair", ""))))
    return seen


def _federal_register_pdf_url(pair: dict, *, timeout: float = 30.0) -> str | None:
    """Resolve a Federal Register document's authoritative PDF url via the FR
    API (``pdf_url`` field). Best-effort — returns None on any failure."""
    docnum = pair.get("document_number")
    if not docnum:
        return None
    try:
        import requests

        api = f"https://www.federalregister.gov/api/v1/documents/{docnum}.json"
        r = requests.get(api, timeout=timeout, params={"fields[]": "pdf_url"})
        r.raise_for_status()
        return r.json().get("pdf_url")
    except Exception:
        return None


def _resolve_real_pdf(
    pair: dict, source: str, doc_id: str, cache_dir: Path, *, timeout: float = 60.0
) -> tuple[Path | None, tuple | None, str | None]:
    """Resolve the REAL source PDF for a pair. Returns
    ``(pdf_path, page_range_or_None, error_or_None)``.

    * On-disk ``local_pdf`` (arXiv) is used directly; its ``page_range`` (1-based
      inclusive) restricts extraction to the section's physical pages.
    * Downloadable PDFs (IRS forms, Federal Register) are fetched once into
      ``cache_dir`` (whole document; no page_range). Network failure -> deferred.
    """
    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        pr = pair.get("page_range")
        return Path(local_pdf), (tuple(pr) if pr else None), None

    url: str | None = None
    if source in ("pdf_form", "forms"):
        u = (pair.get("url") or "").strip()
        url = u if u.lower().endswith(".pdf") else None
    elif source == "federal_register":
        url = _federal_register_pdf_url(pair, timeout=timeout)
    if not url:
        return None, None, "no real PDF (no local_pdf, no resolvable download url)"

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{source}__{doc_id}.pdf"
    if not dest.exists():
        try:
            import requests

            r = requests.get(
                url, timeout=timeout, headers={"User-Agent": "SemantiK-realpdf-eval/1.0"}
            )
            r.raise_for_status()
            if not r.content[:5].startswith(b"%PDF"):
                return None, None, f"download is not a PDF: {url}"
            dest.write_bytes(r.content)
        except Exception as exc:
            return None, None, f"download failed ({url}): {exc}"
    return dest, None, None


class _ExtractTimeout(Exception):
    """A single PDF's extract_shared exceeded the per-pair wall-clock budget
    (pdfplumber can stall on pathological dense-vector pages)."""


def _extract_child(pdf_path_str: str) -> None:
    """Child-process entry: populate the on-disk extract cache for one PDF."""
    extract_shared_cached(Path(pdf_path_str))


def _extract_with_timeout(pdf_path: Path, seconds: int) -> dict:
    """Run ``extract_shared_cached`` under a HARD process-level wall-clock guard.

    pdfplumber/pdfminer can stall at the C level on pathological dense-vector
    pages, where a Python SIGALRM never fires. So the extraction runs in a
    forked child that writes the deterministic disk cache; on timeout the child
    is terminated and ``_ExtractTimeout`` is raised (the doc is skipped, not
    fatal). On success the parent re-reads via ``extract_shared_cached`` (now a
    cache hit). Disabled (``seconds<=0``) -> plain in-process call."""
    if seconds <= 0:
        return extract_shared_cached(pdf_path)
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_extract_child, args=(str(pdf_path),))
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise _ExtractTimeout()
    if proc.exitcode != 0:
        raise RuntimeError(f"extract child exited {proc.exitcode}")
    return extract_shared_cached(pdf_path)


def _filter_shared_pages(shared: dict, page_range: tuple | None) -> dict:
    """Restrict an ``extract_shared`` result to a 1-based-inclusive page range
    (used for arXiv section pairs whose gold covers only some pages of the whole
    paper PDF). No-op when ``page_range`` is None."""
    if not page_range:
        return shared
    s, e = int(page_range[0]), int(page_range[1])
    pages = [
        p
        for p in shared.get("pages", [])
        if s <= int(p.get("page_num", p.get("page", -1))) <= e
    ]
    return {**shared, "pages": pages}


def run_realpdf_build(args) -> None:
    """Build the real-PDF structure eval set (Phase 1) -> two license roots."""
    arxiv_lic = _load_arxiv_license_map()
    train_val = _train_val_docids() if args.eval_holdout_by_docid else set()
    floor = float(args.align_quality_floor)
    cache_root: Path = args.realpdf_cache
    soft_floor = GLOBAL_SOFT_FLOOR
    max_run = GLOBAL_MAX_RUN

    # Output roots: clean (held-out) test.jsonl per license partition; the
    # contaminated real-PDF rows land in a leakage-flagged sidecar so the
    # supplementary per-domain measurement stays reproducible without polluting
    # the zero-leakage eval set.
    roots = {
        "shippable": args.out_dir,
        "internal": args.internal_out_dir,
    }
    for r in roots.values():
        r.mkdir(parents=True, exist_ok=True)

    clean_rows: dict[str, list[dict]] = {"shippable": [], "internal": []}
    contam_rows: dict[str, list[dict]] = {"shippable": [], "internal": []}
    doc_records: list[dict] = []
    drops: Counter = Counter()
    per_domain_docs: Counter = Counter()
    total_kept = 0

    start = time.time()
    for source in args.realpdf_sources:
        src_dir = Path("data/pairs") / source
        if not src_dir.exists():
            print(f"[realpdf] source dir missing, skipping: {src_dir}", file=sys.stderr)
            continue
        cap = args.realpdf_limit_per_source
        kept_for_source = 0
        files = sorted(src_dir.glob("*.json"))
        # Process HELD-OUT (clean) docs FIRST so the per-source cap fills the
        # zero-leakage headline set before spending budget on contaminated docs
        # (which only matter for the supplementary leakage-flagged table). The
        # leakage key uses the pair's canonical source field (dir->source map).
        if train_val:
            canon_src = _DIR_TO_SOURCE.get(source, source)
            files = sorted(
                files,
                key=lambda pf: (
                    1 if (canon_src, _doc_id_for(canon_src, pf.stem)) in train_val else 0,
                    pf.name,
                ),
            )
        for pf in files:
            if cap and kept_for_source >= cap:
                break
            try:
                pair = json.loads(pf.read_text())
            except Exception as exc:
                drops["read_error"] += 1
                continue
            # The pair's INTERNAL source field is canonical for domain / license /
            # leakage keying (e.g. dir ``forms`` carries source ``pdf_form``);
            # the directory name (``source``) is only the glob root.
            psource = pair.get("source") or source
            domain = _domain_for(psource)
            doc_id = _doc_id_for(psource, pf.stem)
            contaminated = (psource, doc_id) in train_val
            if (
                args.eval_holdout_by_docid
                and contaminated
                and not args.include_contaminated
            ):
                drops["holdout_excluded"] += 1
                continue

            output_html = pair.get("output_html")
            if not output_html:
                drops["no_output_html"] += 1
                continue
            gold_records = extract_html_blocks(output_html)
            if not gold_records:
                drops["no_gold_blocks"] += 1
                continue

            pdf_path, page_range, err = _resolve_real_pdf(
                pair, psource, doc_id, cache_root / psource
            )
            if pdf_path is None:
                drops["no_real_pdf"] += 1
                continue
            try:
                shared = _extract_with_timeout(pdf_path, args.realpdf_extract_timeout)
            except _ExtractTimeout:
                drops["extract_timeout"] += 1
                continue
            except Exception as exc:
                drops["extract_error"] += 1
                continue
            shared = _filter_shared_pages(shared, page_range)
            pdf_views, in_table_by_order = _pdf_views_for_pair(shared)
            if not pdf_views:
                drops["no_pdf_blocks"] += 1
                continue

            gold_views = [
                GoldView.from_html_block(rec, i) for i, rec in enumerate(gold_records)
            ]
            # Size guard: the global aligner DP is O(n_pdf * n_gold * max_run);
            # an oversized whole-paper doc stalls the (single-process) align for
            # minutes. Skip + record rather than block the build.
            maxb = args.realpdf_max_blocks
            if maxb and (len(pdf_views) > maxb or len(gold_views) > maxb):
                drops["too_large"] += 1
                continue
            try:
                result = align_blocks(
                    pdf_views,
                    gold_views,
                    max_run=max_run,
                    soft_floor=soft_floor,
                    source=psource,
                    pair=pf.stem,
                )
            except LedgerIncompleteError as exc:
                drops["align_not_lossless"] += 1
                continue
            except Exception as exc:
                drops["align_error"] += 1
                continue

            n_gold = len(gold_views)
            gold_gap = int(result.ledger.counts.get("gold_gap", 0))
            aligned_frac = (n_gold - gold_gap) / max(1, n_gold)

            license_part = _license_partition(psource, doc_id, arxiv_lic)
            doc_rec = {
                "source": psource,
                "domain": domain,
                "doc_id": doc_id,
                "pair": pf.stem,
                "license": license_part,
                "holdout_clean": not contaminated,
                "page_range": list(page_range) if page_range else None,
                "n_gold": n_gold,
                "gold_gap": gold_gap,
                "aligned_gold_fraction": round(aligned_frac, 6),
                "n_pdf_blocks": len(pdf_views),
            }

            if aligned_frac < floor:
                drops["low_alignment"] += 1
                doc_rec["dropped"] = "low_alignment"
                doc_records.append(doc_rec)
                continue

            kept = project_rows(result.rows, ROLE_NAMES, ledger=result.ledger)
            rows = [_enrich_v3_row(r, gold_records, in_table_by_order) for r in kept]
            for rr in rows:
                rr["realpdf"] = True
                rr["domain"] = domain
                rr["license"] = license_part
                rr["doc_id"] = doc_id
                rr["holdout_clean"] = not contaminated
                rr["aligned_gold_fraction"] = round(aligned_frac, 6)

            bucket = clean_rows if not contaminated else contam_rows
            bucket[license_part].extend(rows)
            doc_rec["n_rows"] = len(rows)
            doc_rec["dropped"] = None
            doc_records.append(doc_rec)
            per_domain_docs[domain] += 1
            kept_for_source += 1
            total_kept += 1
            # Incremental flush (crash-safe for a long, slow extraction job —
            # extract is the bottleneck at minutes/paper on dense PDFs).
            if total_kept % 5 == 0:
                _flush_realpdf_outputs(
                    roots, clean_rows, contam_rows, doc_records, drops, args
                )
                print(
                    f"[realpdf] kept {total_kept} docs "
                    f"(last source={source}; elapsed {(time.time() - start) / 60:.1f}min)",
                    file=sys.stderr,
                )

    _flush_realpdf_outputs(roots, clean_rows, contam_rows, doc_records, drops, args)
    for part, root in roots.items():
        print(
            f"[realpdf][{part}] -> {root}  clean_rows={len(clean_rows[part])}  "
            f"contaminated_rows={len(contam_rows[part])}",
            file=sys.stderr,
        )
    print(
        f"[REALPDF] docs_kept={sum(per_domain_docs.values())}  "
        f"by_domain={dict(per_domain_docs)}  drops={dict(drops)}",
        file=sys.stderr,
    )


def _flush_realpdf_outputs(roots, clean_rows, contam_rows, doc_records, drops, args) -> None:
    """Write both license roots (clean test.jsonl + leakage-flagged sidecar +
    coverage report) and the flat per-doc ledger. Called periodically so a
    long-running build is crash-safe."""
    for part, root in roots.items():
        clean = clean_rows[part]
        contam = contam_rows[part]
        with (root / "test.jsonl").open("w") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (root / "test_in_training.jsonl").open("w") as f:
            for r in contam:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report = _realpdf_coverage_report(part, clean, contam, doc_records, drops, args)
        (root / "coverage_report.json").write_text(json.dumps(report, indent=2))
    args.out_dir.joinpath("realpdf_docs.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in doc_records) + "\n"
    )


def _realpdf_coverage_report(
    part: str,
    clean: list[dict],
    contam: list[dict],
    doc_records: list[dict],
    drops: Counter,
    args,
) -> dict:
    """Per-license-root coverage report with per-domain doc/row counts +
    alignment coverage and the drop ledger."""

    def _slice(rows, holdout):
        by_domain: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            if r.get("license") != part:
                continue
            d = r.get("domain", "other")
            by_domain[d]["rows"] += 1
        return by_domain

    # Per-domain doc-level alignment coverage (mean aligned_gold_fraction over
    # KEPT docs of this license partition).
    dom_cov: dict[str, list[float]] = defaultdict(list)
    dom_docs: dict[str, Counter] = defaultdict(Counter)
    for d in doc_records:
        if d.get("license") != part:
            continue
        dom = d.get("domain", "other")
        kind = "clean" if d.get("holdout_clean") else "contaminated"
        if d.get("dropped"):
            dom_docs[dom][f"dropped_{d['dropped']}"] += 1
            continue
        dom_docs[dom][f"docs_{kind}"] += 1
        dom_cov[dom].append(float(d.get("aligned_gold_fraction", 0.0)))

    per_domain = {}
    all_domains = set(dom_docs) | set(dom_cov)
    for dom in sorted(all_domains):
        covs = dom_cov.get(dom, [])
        per_domain[dom] = {
            **dict(dom_docs.get(dom, {})),
            "mean_aligned_gold_fraction": round(sum(covs) / len(covs), 6) if covs else None,
            "clean_rows": sum(
                1 for r in clean if r.get("domain") == dom and r.get("license") == part
            ),
            "contaminated_rows": sum(
                1 for r in contam if r.get("domain") == dom and r.get("license") == part
            ),
        }

    return {
        "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
        "build": "realpdf",
        "license_partition": part,
        "soft_floor": GLOBAL_SOFT_FLOOR,
        "max_run": GLOBAL_MAX_RUN,
        "align_quality_floor": float(args.align_quality_floor),
        "eval_holdout_by_docid": bool(args.eval_holdout_by_docid),
        "rendered_train_corpus": str(_RENDERED_TRAIN_CORPUS),
        "sources": list(args.realpdf_sources),
        "clean_rows_total": len(clean),
        "contaminated_rows_total": len(contam),
        "per_domain": per_domain,
        "drop_ledger": dict(drops),
        "role_names": list(ROLE_NAMES),
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
    }


# ---------------------------------------------------------------------------
# Realistic-render eval-set build path — opt-in, off by default
# ---------------------------------------------------------------------------
#
# Today's eval lives on either CLEAN rendered-HTML-PDF gold (in-distribution,
# the training render path) or REAL PDFs (--realpdf-only, where alignment is
# noisy: <0.30 per-block sim on the wild docs). This path is the third leg: it
# renders the held-out gold output_html through a REALISTIC presentation
# profile (2-column / serif / dense — data/render_augment.py) so the extracted
# blocks look deploy-LIKE, then aligns the extraction back to the SAME gold HTML
# (the labels are the HTML tags, unchanged by presentation) via the SAME frozen
# aligner. Because only PRESENTATION changed, the alignment stays clean (high
# per-block sim, >> the wild 0.30) -> FREE, correctly-aligned deploy-like labels
# with NO hand-labeling.
#
# The extraction MUST run with SEMANTIK_COLUMN_ORDER=1 so the column-aware
# reading-order re-sort fires on the 2-column renders.
#
# Output: a single root (default data/structure_dataset_realistic/) with a
# test.jsonl in the EXACT eval row schema eval/measure_realpdf_structure.py
# reads via --realpdf-roots (rows carry realpdf=True + domain + doc_id +
# aligned_gold_fraction, plus the additive render_profile / render_seed
# provenance). No eval-code change is needed.


def run_realistic_eval_build(args) -> None:
    """Build the realistic-render eval set: render held-out gold HTML through
    realistic presentation profiles, align back to the gold, emit deploy-like
    eval rows with clean labels + per-block alignment-similarity reporting."""
    mode = (getattr(args, "render_mode", None) or "").strip().lower() or None
    if mode is None:
        mode = resolve_render_augment_mode()
    if mode == "off":
        # The realistic build IS the augmentation; an off mode would render
        # clean (== the in-distribution path), defeating the point. Default to
        # the mixed domain-randomization distribution and say so.
        print(
            "[realistic] SEMANTIK_RENDER_AUGMENT off / no --render-mode; "
            "defaulting to 'mixed' (the realistic build needs a non-clean profile)",
            file=sys.stderr,
        )
        mode = "mixed"

    if not resolve_column_order_mode():
        print(
            "[realistic] WARNING: SEMANTIK_COLUMN_ORDER is OFF — the column-aware "
            "reading-order re-sort will NOT fire on 2-column renders. Set "
            "SEMANTIK_COLUMN_ORDER=1 for a faithful realistic build.",
            file=sys.stderr,
        )

    # EXACT render-time labeling (SEMANTIK_EXACT_RENDER_LABELS, default off):
    # replace the global text aligner with render_capture's bbox-overlap
    # labeler. The realistic eval renders gold HTML SemantiK controls, so the
    # exact-geometry path applies; off => the byte-identical global-aligner path.
    from data.render_capture import (
        RenderCaptureSession,
        exact_render_labels_enabled,
        label_blocks_by_overlap,
    )

    use_exact = exact_render_labels_enabled()

    train_val = _train_val_docids() if args.eval_holdout_by_docid else set()
    floor = float(args.align_quality_floor)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows: list[dict] = []
    contam_rows: list[dict] = []
    doc_records: list[dict] = []
    drops: Counter = Counter()
    profile_counts: Counter = Counter()
    per_domain_docs: Counter = Counter()
    block_sims: list[float] = []  # per-block align confidence over CLEAN rows
    total_kept = 0
    start = time.time()

    # Exact mode renders through RenderCaptureSession (one Chromium, batched);
    # fuzzy mode keeps the HtmlValidator (its render_pdf + Letter multi-page).
    render_cm = RenderCaptureSession() if use_exact else HtmlValidator()
    with render_cm as validator:
        for source in args.realpdf_sources:
            src_dir = Path("data/pairs") / source
            if not src_dir.exists():
                print(f"[realistic] source dir missing, skipping: {src_dir}", file=sys.stderr)
                continue
            cap = args.realpdf_limit_per_source
            kept_for_source = 0
            files = sorted(src_dir.glob("*.json"))
            if train_val:
                canon_src = _DIR_TO_SOURCE.get(source, source)
                files = sorted(
                    files,
                    key=lambda pf: (
                        1 if (canon_src, _doc_id_for(canon_src, pf.stem)) in train_val else 0,
                        pf.name,
                    ),
                )
            for pf in files:
                if cap and kept_for_source >= cap:
                    break
                try:
                    pair = json.loads(pf.read_text())
                except Exception:
                    drops["read_error"] += 1
                    continue
                psource = pair.get("source") or source
                domain = _domain_for(psource)
                doc_id = _doc_id_for(psource, pf.stem)
                contaminated = (psource, doc_id) in train_val
                if (
                    args.eval_holdout_by_docid
                    and contaminated
                    and not args.include_contaminated
                ):
                    drops["holdout_excluded"] += 1
                    continue

                output_html = pair.get("output_html")
                if not output_html:
                    drops["no_output_html"] += 1
                    continue
                gold_records = extract_html_blocks(output_html)
                if not gold_records:
                    drops["no_gold_blocks"] += 1
                    continue

                # Render the gold HTML through the realistic profile.
                profile, seed = resolve_render_profile_for(_stable_doc_index(pf.stem), mode)
                rendered_html = augment_html(output_html, profile, seed)

                if use_exact:
                    # EXACT render-time bbox->role labeling: render with boxes,
                    # extract, assign each PDF block its gold role by geometric
                    # containment (no fuzzy alignment). Emits the SAME eval row
                    # schema (text + layout + labels.structural_role + domain +
                    # doc_id + aligned_gold_fraction) measure_realpdf_structure
                    # reads; UNMATCHED blocks are EXCLUDED. aligned_gold_fraction
                    # is repurposed to the matched fraction (1 - unmatched_rate).
                    with tempfile.TemporaryDirectory() as tmp:
                        try:
                            capture = validator.render(rendered_html, tmpdir=Path(tmp))
                        except Exception:
                            drops["render_error"] += 1
                            continue
                        try:
                            shared = extract_shared(capture.pdf_path)
                        except Exception:
                            drops["extract_error"] += 1
                            continue
                    lr = label_blocks_by_overlap(shared, capture)
                    n_gold = len(gold_records)
                    n_pdf = lr.n_blocks
                    if n_pdf == 0:
                        drops["no_pdf_blocks"] += 1
                        continue
                    maxb = args.realpdf_max_blocks
                    if maxb and (n_pdf > maxb or n_gold > maxb):
                        drops["too_large"] += 1
                        continue
                    matched_frac = (n_pdf - lr.n_unmatched) / max(1, n_pdf)
                    doc_rec = {
                        "source": psource,
                        "domain": domain,
                        "doc_id": doc_id,
                        "pair": pf.stem,
                        "render_profile": profile,
                        "render_seed": seed,
                        "holdout_clean": not contaminated,
                        "n_gold": n_gold,
                        "n_pdf_blocks": n_pdf,
                        "n_unmatched": lr.n_unmatched,
                        "unmatched_rate": round(lr.unmatched_rate, 6),
                        "aligned_gold_fraction": round(matched_frac, 6),
                        "exact_label": True,
                        "overflowed_single_page": bool(capture.overflowed_single_page),
                    }
                    if matched_frac < floor:
                        drops["low_alignment"] += 1
                        doc_rec["dropped"] = "low_alignment"
                        doc_records.append(doc_rec)
                        continue
                    rows = []
                    for rr in lr.rows:
                        lbl = rr.get("labels")
                        if not lbl or lbl.get("structural_role") is None:
                            continue
                        rows.append(
                            {
                                "text": rr["text"],
                                "layout": rr["layout"],
                                "labels": {
                                    "structural_role": lbl["structural_role"],
                                    "is_heading": int(lbl["is_heading"]),
                                    "table_region": int(lbl["table_region"]),
                                    "is_image_block": int(lbl["is_image_block"]),
                                    "list_nesting": int(lbl["list_nesting"]),
                                    "pedagogical_role": int(lbl["pedagogical_role"]),
                                },
                                "html_tag": rr["html_tag"],
                                "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
                                "source": psource,
                                "pair": pf.stem,
                                "realpdf": True,
                                "realistic_render": True,
                                "exact_label": True,
                                "domain": domain,
                                "doc_id": doc_id,
                                "holdout_clean": not contaminated,
                                "aligned_gold_fraction": round(matched_frac, 6),
                                "render_profile": profile,
                                "render_seed": seed,
                            }
                        )
                    # Exact containment IS the match — no per-block text-sim. Use
                    # 1.0 so the mean-block-sim column stays well-defined.
                    doc_block_sims = [1.0] * len(rows)
                    if not contaminated:
                        clean_rows.extend(rows)
                        block_sims.extend(doc_block_sims)
                    else:
                        contam_rows.extend(rows)
                    profile_counts[profile] += 1
                    doc_rec["n_rows"] = len(rows)
                    doc_rec["mean_block_sim"] = 1.0 if rows else None
                    doc_rec["dropped"] = None
                    doc_records.append(doc_rec)
                    per_domain_docs[domain] += 1
                    kept_for_source += 1
                    total_kept += 1
                    print(
                        f"[realistic-exact] {pf.stem[:42]:42}  profile={profile:24}  "
                        f"rows={len(rows):4}  unmatched={lr.unmatched_rate:.3f}  "
                        f"matched_frac={matched_frac:.3f}",
                        file=sys.stderr,
                    )
                    continue

                with tempfile.TemporaryDirectory() as tmp:
                    tmp_pdf = Path(tmp) / "realistic.pdf"
                    try:
                        validator.render_pdf(rendered_html, tmp_pdf)
                    except Exception:
                        drops["render_error"] += 1
                        continue
                    try:
                        shared = extract_shared(tmp_pdf)
                    except Exception:
                        drops["extract_error"] += 1
                        continue

                pdf_views, in_table_by_order = _pdf_views_for_pair(shared)
                if not pdf_views:
                    drops["no_pdf_blocks"] += 1
                    continue
                gold_views = [
                    GoldView.from_html_block(rec, i) for i, rec in enumerate(gold_records)
                ]
                maxb = args.realpdf_max_blocks
                if maxb and (len(pdf_views) > maxb or len(gold_views) > maxb):
                    drops["too_large"] += 1
                    continue
                try:
                    result = align_blocks(
                        pdf_views,
                        gold_views,
                        max_run=GLOBAL_MAX_RUN,
                        soft_floor=GLOBAL_SOFT_FLOOR,
                        source=psource,
                        pair=pf.stem,
                    )
                except LedgerIncompleteError:
                    drops["align_not_lossless"] += 1
                    continue
                except Exception:
                    drops["align_error"] += 1
                    continue
                finally:
                    try:
                        structure_align._sim_cached.cache_clear()
                    except Exception:
                        pass

                n_gold = len(gold_views)
                gold_gap = int(result.ledger.counts.get("gold_gap", 0))
                aligned_frac = (n_gold - gold_gap) / max(1, n_gold)

                doc_rec = {
                    "source": psource,
                    "domain": domain,
                    "doc_id": doc_id,
                    "pair": pf.stem,
                    "render_profile": profile,
                    "render_seed": seed,
                    "holdout_clean": not contaminated,
                    "n_gold": n_gold,
                    "gold_gap": gold_gap,
                    "aligned_gold_fraction": round(aligned_frac, 6),
                    "n_pdf_blocks": len(pdf_views),
                }
                if aligned_frac < floor:
                    drops["low_alignment"] += 1
                    doc_rec["dropped"] = "low_alignment"
                    doc_records.append(doc_rec)
                    continue

                kept = project_rows(result.rows, ROLE_NAMES, ledger=result.ledger)
                rows = [_enrich_v3_row(r, gold_records, in_table_by_order) for r in kept]
                doc_block_sims: list[float] = []
                for rr in rows:
                    rr["realpdf"] = True  # eval reader buckets realpdf rows by domain
                    rr["realistic_render"] = True
                    rr["domain"] = domain
                    rr["doc_id"] = doc_id
                    rr["holdout_clean"] = not contaminated
                    rr["aligned_gold_fraction"] = round(aligned_frac, 6)
                    rr["render_profile"] = profile
                    rr["render_seed"] = seed
                    doc_block_sims.append(float(rr.get("align", {}).get("confidence", 0.0)))

                if not contaminated:
                    clean_rows.extend(rows)
                    block_sims.extend(doc_block_sims)
                else:
                    contam_rows.extend(rows)
                profile_counts[profile] += 1
                doc_rec["n_rows"] = len(rows)
                doc_rec["mean_block_sim"] = (
                    round(sum(doc_block_sims) / len(doc_block_sims), 6) if doc_block_sims else None
                )
                doc_rec["dropped"] = None
                doc_records.append(doc_rec)
                per_domain_docs[domain] += 1
                kept_for_source += 1
                total_kept += 1
                print(
                    f"[realistic] {pf.stem[:42]:42}  profile={profile:24}  "
                    f"rows={len(rows):4}  align_frac={aligned_frac:.3f}  "
                    f"mean_block_sim={doc_rec['mean_block_sim']}",
                    file=sys.stderr,
                )

    # Write the eval rows + coverage report.
    with (out_dir / "test.jsonl").open("w") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "test_in_training.jsonl").open("w") as f:
        for r in contam_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mean_block_sim = round(sum(block_sims) / len(block_sims), 6) if block_sims else None
    per_domain = {}
    dom_cov: dict[str, list[float]] = defaultdict(list)
    dom_docs: Counter = Counter()
    for d in doc_records:
        if d.get("dropped") or not d.get("holdout_clean"):
            continue
        dom_docs[d["domain"]] += 1
        dom_cov[d["domain"]].append(float(d.get("aligned_gold_fraction", 0.0)))
    for dom in sorted(set(dom_docs) | set(dom_cov)):
        covs = dom_cov.get(dom, [])
        per_domain[dom] = {
            "docs_clean": dom_docs.get(dom, 0),
            "clean_rows": sum(1 for r in clean_rows if r.get("domain") == dom),
            "mean_aligned_gold_fraction": round(sum(covs) / len(covs), 6) if covs else None,
        }

    report = {
        "schema_version": STRUCTURE_ALIGN_SCHEMA_VERSION,
        "build": "realistic",
        "render_mode": mode,
        "column_order_on": resolve_column_order_mode(),
        "soft_floor": GLOBAL_SOFT_FLOOR,
        "max_run": GLOBAL_MAX_RUN,
        "align_quality_floor": floor,
        "eval_holdout_by_docid": bool(args.eval_holdout_by_docid),
        "rendered_train_corpus": str(_RENDERED_TRAIN_CORPUS),
        "sources": list(args.realpdf_sources),
        "clean_rows_total": len(clean_rows),
        "contaminated_rows_total": len(contam_rows),
        "docs_kept": total_kept,
        # The headline: per-block alignment similarity on the CLEAN held-out
        # set vs the wild-PDF <0.30 baseline (proves the free labels are clean).
        "mean_per_block_align_sim": mean_block_sim,
        "wild_pdf_baseline_block_sim": 0.30,
        "render_profile_counts": dict(profile_counts),
        "per_domain": per_domain,
        "drop_ledger": dict(drops),
        "role_names": list(ROLE_NAMES),
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
    }
    # Exact-mode provenance — appended only when on, so a flag-OFF realistic
    # build's coverage_report.json stays byte-identical to today.
    if use_exact:
        report["build"] = "realistic_exact"
        report["exact_label"] = True
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=2))
    out_dir.joinpath("realistic_docs.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in doc_records) + "\n"
    )
    print(
        f"\n[REALISTIC] docs_kept={total_kept}  clean_rows={len(clean_rows)}  "
        f"mean_per_block_align_sim={mean_block_sim}  (wild-PDF baseline 0.30)\n"
        f"  profiles={dict(profile_counts)}  by_domain={dict(per_domain_docs)}  "
        f"drops={dict(drops)}  elapsed={(time.time() - start) / 60:.1f}min",
        file=sys.stderr,
    )
    print(f"[realistic] wrote {out_dir / 'test.jsonl'} + coverage_report.json", file=sys.stderr)


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
    # Cap glibc malloc arenas BEFORE any worker pool / ProcessPoolExecutor is
    # spawned so child workers inherit it. Fewer arenas = far less per-thread
    # arena fragmentation from the per-page pypdfium2/pdfplumber render+extract
    # churn (the native RSS ratchet). std glibc env, not a SemantiK flag;
    # setdefault so an operator can still override it.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

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
    # --- Real-PDF eval-set build path (Phase 1) ---
    ap.add_argument(
        "--realpdf-only",
        action="store_true",
        help="Build the REAL-PDF eval set (FBViews from real source PDFs aligned "
        "against the existing gold output_html) instead of the rendered-HTML-PDF "
        "training build. Emits two license roots (shippable / internal) each with "
        "a test.jsonl. See run_realpdf_build.",
    )
    ap.add_argument(
        "--realistic-eval",
        action="store_true",
        help="Build the REALISTIC-RENDER eval set: render held-out gold "
        "output_html through realistic presentation profiles (2-column / serif "
        "/ dense, data/render_augment.py), align back to the gold HTML, and emit "
        "deploy-like eval rows with CLEAN labels into --out-dir (default "
        "data/structure_dataset_realistic). Run with SEMANTIK_COLUMN_ORDER=1. "
        "See run_realistic_eval_build.",
    )
    ap.add_argument(
        "--render-mode",
        default=None,
        help="Render-augmentation mode for --realistic-eval (overrides "
        "SEMANTIK_RENDER_AUGMENT): 'mixed' (default), or a pinned profile "
        "('two_column' / 'serif_embed' / 'dense_layout' / "
        "'two_column+serif_embed' / 'clean').",
    )
    ap.add_argument(
        "--eval-holdout-by-docid",
        action="store_true",
        help="Hold the real-PDF eval set out by DOCUMENT id against the head's "
        "rendered train+val corpus (zero leakage): contaminated docs are excluded "
        "from test.jsonl and routed to test_in_training.jsonl.",
    )
    ap.add_argument(
        "--include-contaminated",
        action="store_true",
        help="With --eval-holdout-by-docid, STILL process train/val-contaminated "
        "docs (routed to the leakage-flagged test_in_training.jsonl) so the "
        "supplementary per-domain measurement is reproducible.",
    )
    ap.add_argument(
        "--realpdf-sources",
        nargs="+",
        default=["arxiv", "pdf_form", "federal_register"],
        help="Pair sources to source real PDFs from (data/pairs/<source>).",
    )
    ap.add_argument(
        "--realpdf-cache",
        type=Path,
        default=Path("data/realpdf_cache"),
        help="Where downloaded (forms / federal_register) real PDFs are cached.",
    )
    ap.add_argument(
        "--internal-out-dir",
        type=Path,
        default=Path("data/structure_dataset_realpdf_internal"),
        help="INTERNAL-ONLY license root (OpenStax NC-SA + non-commercial/unknown "
        "arXiv). Physically separate from the shippable --out-dir.",
    )
    ap.add_argument(
        "--align-quality-floor",
        type=float,
        default=0.50,
        help="Drop (never mislabel) a real-PDF doc whose aligned-gold fraction is "
        "below this floor.",
    )
    ap.add_argument(
        "--realpdf-limit-per-source",
        type=int,
        default=None,
        help="Cap kept docs per source (smoke / sampling the contaminated set).",
    )
    ap.add_argument(
        "--realpdf-extract-timeout",
        type=int,
        default=90,
        help="Per-PDF extract_shared wall-clock budget (SIGALRM); a pathological "
        "PDF that exceeds it is skipped (drop=extract_timeout), not fatal. 0=off.",
    )
    ap.add_argument(
        "--realpdf-max-blocks",
        type=int,
        default=500,
        help="Skip (drop=too_large) a doc with more than this many PDF or gold "
        "blocks — bounds the O(n*m*max_run) alignment DP. 0=off.",
    )
    ap.add_argument(
        "--global-max-blocks",
        type=int,
        default=500,
        help="Rendered-HTML global-aligner equivalent of --realpdf-max-blocks: "
        "skip (drop=too_large) a pair whose PDF or gold block count exceeds this, "
        "bounding the alignment DP on a freak huge page. 0=off.",
    )
    add_cap_args(ap)
    args = ap.parse_args()

    if args.realistic_eval:
        args.out_dir = args.out_dir or Path("data/structure_dataset_realistic")
        run_realistic_eval_build(args)
        return

    if args.realpdf_only:
        args.out_dir = args.out_dir or Path("data/structure_dataset_realpdf")
        run_realpdf_build(args)
        return

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
