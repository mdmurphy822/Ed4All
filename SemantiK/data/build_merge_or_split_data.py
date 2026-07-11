"""Build BERT-MergeOrSplit training data.

Phase 3a — 4-head multi-head classifier on adjacent span pairs.

Heads:
    * same_logical_block      binary       primary merge decision
    * join_type               3-class      space | newline_within_p |
                                           list_continuation
                                           (only meaningful for positive-primary
                                           pairs; masked with -100 when
                                           same_logical_block=0)
    * hyphen_repair           binary       "accessi-" + "bility" -> "accessibility"

(Heading detection used to live here as a fourth ``heading_body_boundary``
head; it's been moved to BERT-Structure (Phase 3b) as a span-level
``is_heading`` head — pair-level framing was the wrong shape.)

Deterministic per-span features (NOT learned heads, fed as input prefix):
    column_id, is_artifact, font_size, x_range, y_gap, line_height_ratio,
    ends_with_hyphen.

Source pipeline:
    1. Walk ``data/pairs/arxiv/*.json`` for ``(arxiv_id, local_pdf)`` pairs.
    2. For each unique paper that has both a cached ar5iv HTML and an
       on-disk PDF, run extract_shared on the PDF and pull
       ``pypdfium2.text_blocks`` per page.
    3. Walk ar5iv HTML to derive a list of ``(block_id, block_kind,
       block_text, line_text_per_line)`` records — one per paragraph,
       list item, table cell, heading, caption.
    4. Align each pypdfium2 span to one ar5iv block via Jaccard text
       overlap.
    5. For each adjacent pair within a page, derive 4 head labels.
    6. ~10% OCR-noise augmentation on labels-preserving text mutations
       (mark with ``noisy=true``).
    7. Stratified 80/10/10 split by ``same_logical_block``.

Outputs:
    data/merge_or_split_dataset/{train,val,test}.jsonl
    data/merge_or_split_dataset/coverage_report.json

Usage:
    python data/build_merge_or_split_data.py --limit 50      # smoke
    python data/build_merge_or_split_data.py                  # full

Network usage: NONE — uses only the cached ar5iv HTML at
``data/ar5iv_html_cache/``. The PDF and HTML cache are both prerequisites.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag


# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

import math

JOIN_TYPES = ["space", "newline_within_p", "list_continuation"]
JOIN_TYPE_TO_ID = {n: i for i, n in enumerate(JOIN_TYPES)}

# ---------------------------------------------------------------------------
# Layout side-channel feature spec
# ---------------------------------------------------------------------------
#
# A fixed-dim numeric vector per adjacent pair, fed into a small MLP at
# the model level alongside the BERT pooled output. The 5-bucket prefix
# tokens ("fs:xs/sm/md/lg/xl") were too lossy — they bucketed the most
# discriminative signal (font-size jump between adjacent spans) into 5
# coarse classes. This vector exposes the raw numeric signal directly.
#
# ORDER IS STABLE — the layout MLP weights are tied to this exact
# dimension order. Both the trainer and the runtime
# (semantik_structure/council/merge_or_split.py) MUST produce vectors in this
# order. To add a feature: append to the end and bump LAYOUT_FEATURE_DIM.
# Re-training is required because the MLP input dim changes.

LAYOUT_FEATURE_NAMES = [
    "fs_a_norm",          # fs_a / 12
    "fs_b_norm",          # fs_b / 12
    "fs_delta_norm",      # (fs_b - fs_a) / 12  (negative: heading-then-body)
    "fs_ratio_a_b",       # fs_a / max(fs_b, 1.0)
    "bold_a",
    "bold_b",
    "bold_transition",    # 1 if bold flips
    "italic_a",
    "italic_b",
    "y_gap_log1p",        # log1p(y_gap_pts)
    "y_gap_lines",        # y_gap / h_a (gap in line-heights of A)
    "lhr",                # h_a / median_page_h (line-height ratio of A)
    "x0_delta_dec",       # (x0_b - x0_a) / 100  (left-indent delta, deci-pts)
    "x1_delta_dec",       # (x1_b - x1_a) / 100
    "width_ratio_b_a",    # w_b / max(w_a, 1.0)
    "x_overlap_frac",     # 0..1
    "col_same",           # 1 if column_id_a == column_id_b
    "art_a",
    "art_b",
    "hy_a",               # span_a ends with a hyphen
    "ends_period_a",      # span_a ends with .!?
    "ends_colon_a",       # span_a ends with :
    "b_starts_upper",     # span_b starts with an uppercase letter
    "b_titlecase_frac",   # fraction of first-6 words of B that are title-cased
]

LAYOUT_FEATURE_DIM = len(LAYOUT_FEATURE_NAMES)


def _ends_with_period(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _ends_with_colon(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] == ":"


def _b_starts_upper(text: str) -> bool:
    t = (text or "").lstrip()
    return bool(t) and t[0].isupper()


def _b_titlecase_frac(text: str, *, max_words: int = 6) -> float:
    words = (text or "").split()[:max_words]
    if not words:
        return 0.0
    return sum(1 for w in words if w and w[0].isupper()) / len(words)


_LAYOUT_BBOX_CLAMP = 5000.0   # any abs(coord) > this is treated as junk
_LAYOUT_FS_CLAMP = 200.0      # any font size > this is junk
_LAYOUT_OUTPUT_CLAMP = 10.0   # final element-wise clamp on the vector


def _safe_coord(v: float) -> float:
    """Sanitize a bbox or coordinate value. Catches the rare PDFs that
    produce NaN/Inf/huge transforms (about 4 rows in our 120k arXiv set
    had y0=-4e74 from a broken transform matrix). The replacement is
    0.0 — the row's other features still compute, just without the
    geometric signal from this dim."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or abs(f) > _LAYOUT_BBOX_CLAMP:
        return 0.0
    return f


def _safe_fs(v: float) -> float:
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f < 0 or f > _LAYOUT_FS_CLAMP:
        return 0.0
    return f


def _clip_vec(vec: list[float], lo: float = -_LAYOUT_OUTPUT_CLAMP,
              hi: float = _LAYOUT_OUTPUT_CLAMP) -> list[float]:
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


def compute_pair_layout_features(
    *,
    fs_a: float, fs_b: float,
    bold_a: bool, bold_b: bool,
    italic_a: bool, italic_b: bool,
    x0_a: float, x1_a: float, y0_a: float, y1_a: float,
    x0_b: float, x1_b: float, y0_b: float, y1_b: float,
    col_a: int, col_b: int,
    art_a: bool, art_b: bool,
    hy_a: bool,
    text_a: str, text_b: str,
    median_h: float,
) -> list[float]:
    """Build the per-pair layout feature vector. Order MUST match
    :data:`LAYOUT_FEATURE_NAMES`. Output is clamped element-wise to
    +/-10 to defend against pathological PDF coordinate transforms."""
    fs_a = _safe_fs(fs_a)
    fs_b = _safe_fs(fs_b)
    median_h = max(1.0, _safe_coord(median_h))

    x0_a, x1_a = _safe_coord(x0_a), _safe_coord(x1_a)
    y0_a, y1_a = _safe_coord(y0_a), _safe_coord(y1_a)
    x0_b, x1_b = _safe_coord(x0_b), _safe_coord(x1_b)
    y0_b, y1_b = _safe_coord(y0_b), _safe_coord(y1_b)

    h_a = max(0.0, y1_a - y0_a)
    w_a = max(0.0, x1_a - x0_a)
    w_b = max(0.0, x1_b - x0_b)
    y_gap = max(0.0, y0_b - y1_a)

    # x-range overlap fraction between the two spans.
    overlap_lo = max(x0_a, x0_b)
    overlap_hi = min(x1_a, x1_b)
    overlap = max(0.0, overlap_hi - overlap_lo)
    union = max(x1_a, x1_b) - min(x0_a, x0_b)
    overlap_frac = (overlap / union) if union > 0 else 0.0

    raw = [
        fs_a / 12.0,
        fs_b / 12.0,
        (fs_b - fs_a) / 12.0,
        (fs_a / max(fs_b, 1.0)) if fs_b > 0 else 1.0,
        1.0 if bold_a else 0.0,
        1.0 if bold_b else 0.0,
        1.0 if bool(bold_a) != bool(bold_b) else 0.0,
        1.0 if italic_a else 0.0,
        1.0 if italic_b else 0.0,
        math.log1p(y_gap),
        y_gap / max(h_a, 1.0),
        h_a / median_h,
        (x0_b - x0_a) / 100.0,
        (x1_b - x1_a) / 100.0,
        w_b / max(w_a, 1.0),
        overlap_frac,
        1.0 if int(col_a) == int(col_b) else 0.0,
        1.0 if art_a else 0.0,
        1.0 if art_b else 0.0,
        1.0 if hy_a else 0.0,
        1.0 if _ends_with_period(text_a) else 0.0,
        1.0 if _ends_with_colon(text_a) else 0.0,
        1.0 if _b_starts_upper(text_b) else 0.0,
        _b_titlecase_frac(text_b),
    ]
    return _clip_vec(raw)


def layout_features_from_row(row: dict) -> list[float]:
    """Helper used by the trainer: build the layout vector from a row
    persisted by the data builder. Recovers ``median_h`` from
    ``line_height_ratio = h_a / median_h``.
    """
    feats = (row.get("det_features") or {})
    a = feats.get("a") or {}
    b = feats.get("b") or {}
    y_gap = float(feats.get("y_gap", 0.0) or 0.0)
    lhr = float(feats.get("line_height_ratio", 1.0) or 1.0)
    y0_a = float(a.get("y0", 0.0) or 0.0)
    y1_a = float(a.get("y1", 0.0) or 0.0)
    h_a = max(1.0, y1_a - y0_a)
    median_h = h_a / max(lhr, 0.01)
    return compute_pair_layout_features(
        fs_a=a.get("font_size", 0.0) or 0.0,
        fs_b=b.get("font_size", 0.0) or 0.0,
        bold_a=bool(a.get("is_bold", False)),
        bold_b=bool(b.get("is_bold", False)),
        italic_a=bool(a.get("is_italic", False)),
        italic_b=bool(b.get("is_italic", False)),
        x0_a=a.get("x0", 0.0), x1_a=a.get("x1", 0.0),
        y0_a=y0_a,             y1_a=y1_a,
        x0_b=b.get("x0", 0.0), x1_b=b.get("x1", 0.0),
        y0_b=b.get("y0", 0.0), y1_b=b.get("y1", 0.0),
        col_a=int(a.get("column_id", 0) or 0),
        col_b=int(b.get("column_id", 0) or 0),
        art_a=bool(a.get("is_artifact", False)),
        art_b=bool(b.get("is_artifact", False)),
        hy_a=bool(a.get("ends_with_hyphen", False)),
        text_a=row.get("span_a_text", "") or "",
        text_b=row.get("span_b_text", "") or "",
        median_h=median_h,
    )

# join_type is masked when same_logical_block=0. We use -100 (PyTorch
# ignore_index convention) so the loss is skipped cleanly per-pair.
# (Was 4-class with paragraph_break; the negative-primary side is fully
# defined by same_logical_block=0 + list_continuation membership, so a
# distinct paragraph_break class added no signal and never produced
# trainable positives — see Plans/HANDOFF.md.)
JOIN_TYPE_IGNORE = -100

DEFAULT_PAIRS_DIR = Path("data/pairs/arxiv")
DEFAULT_CACHE_DIR = Path("data/ar5iv_html_cache")
DEFAULT_OUT_DIR = Path("data/merge_or_split_dataset")
DEFAULT_EXTRACT_CACHE = Path("data/extract_cache")


# ---------------------------------------------------------------------------
# ar5iv HTML block walker
# ---------------------------------------------------------------------------


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
LIST_ITEM_TAGS = {"li"}
LIST_PARENT_TAGS = {"ul", "ol"}
PARA_TAGS = {"p"}
TABLE_CELL_TAGS = {"th", "td"}
CAPTION_TAGS = {"figcaption", "caption"}
QUOTE_TAGS = {"blockquote"}

BLOCK_KIND_TAGS = (
    HEADING_TAGS
    | LIST_ITEM_TAGS
    | PARA_TAGS
    | TABLE_CELL_TAGS
    | CAPTION_TAGS
    | QUOTE_TAGS
)

STRIP_AR5IV_SELECTORS = [
    ".ltx_page_header", ".ltx_page_footer",
    ".ltx_bibliography",
    ".ltx_role_footnote",
    ".ltx_note_content",
    ".ltx_pagination",
    ".ltx_dates",
    "style", "script", "link", "meta", "noscript",
]


def _strip_ar5iv_html(soup: BeautifulSoup) -> None:
    for sel in STRIP_AR5IV_SELECTORS:
        for el in soup.select(sel):
            el.decompose()


def _block_kind_for_tag(tag: Tag) -> str | None:
    name = tag.name
    if name in HEADING_TAGS:
        return f"h{name[-1]}"
    if name in PARA_TAGS:
        return "p"
    if name in LIST_ITEM_TAGS:
        return "li"
    if name in TABLE_CELL_TAGS:
        return name  # "th" or "td"
    if name in CAPTION_TAGS:
        return "caption"
    if name in QUOTE_TAGS:
        return "blockquote"
    return None


def _list_parent_id(tag: Tag) -> str | None:
    """Return a stable identifier for the nearest <ul>/<ol> ancestor.

    Used to group sibling <li>s into the same list for the
    ``list_continuation`` join_type. Returns ``None`` if no list ancestor.
    """
    for p in tag.parents:
        if isinstance(p, Tag) and p.name in LIST_PARENT_TAGS:
            return f"{p.name}@{id(p)}"
    return None


def _normalize_text(s: str) -> str:
    return " ".join((s or "").split())


def _block_lines(tag: Tag) -> list[str]:
    """Split the block's text into newline-delimited lines.

    LaTeXML wraps line breaks in ``<br>`` and uses preserved whitespace
    for ``<pre>``/``<code>`` blocks. For prose, the only "lines" inside
    a <p> are forced via <br> — there are usually none, which means the
    paragraph is a single logical line in the HTML even if it wraps in
    the rendered PDF. We use this to label join_type:
        * intra-line  (no <br> separating spans) -> ``space``
        * cross-line  (a <br> separates spans)   -> ``newline_within_p``
    """
    # Collect text segments, splitting at every <br>.
    segments: list[list[str]] = [[]]
    for desc in tag.descendants:
        if isinstance(desc, Tag):
            if desc.name == "br":
                segments.append([])
            continue
        s = str(desc)
        if s:
            segments[-1].append(s)
    lines = [_normalize_text("".join(seg)) for seg in segments]
    return [ln for ln in lines if ln]


def walk_ar5iv_blocks(html: str) -> list[dict]:
    """Walk an ar5iv HTML document and emit one record per block.

    Returns a list of dicts with:
        block_id    int      sequential, document order
        kind        str      "p" | "li" | "h1".."h6" | "td" | "th" | ...
        text        str      flat normalized text content
        lines       list[str]  per <br>-line slices of text
        list_id     str|None nearest ul/ol ancestor (for list_continuation)
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_ar5iv_html(soup)

    article = (
        soup.select_one(".ltx_document")
        or soup.select_one("article")
        or soup.body
        or soup
    )

    out: list[dict] = []
    seen_tags: set[int] = set()
    for tag in article.find_all(True, recursive=True):
        kind = _block_kind_for_tag(tag)
        if kind is None:
            continue
        # Skip blocks nested inside another block we already emitted —
        # e.g. a <p> inside a <figcaption>. The outer caption owns it.
        nested = False
        for parent in tag.parents:
            if id(parent) in seen_tags:
                nested = True
                break
        if nested:
            continue
        # Drop math-only blocks: alttext is enough; no useful prose signal.
        text = _normalize_text(tag.get_text(" "))
        if not text:
            continue
        seen_tags.add(id(tag))
        out.append({
            "block_id": len(out),
            "kind": kind,
            "text": text,
            "lines": _block_lines(tag) or [text],
            "list_id": _list_parent_id(tag),
        })
    return out


# ---------------------------------------------------------------------------
# Span -> ar5iv block alignment via Jaccard
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    """Fraction of A's tokens that appear in B. Useful when A is a tiny
    span and B is the whole paragraph: containment is high but Jaccard
    is small.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)


def align_spans_to_blocks(
    span_texts: list[str],
    blocks: list[dict],
    *,
    min_score: float = 0.30,
) -> list[int | None]:
    """For each span, return the block_id of the best-matching ar5iv block,
    or None if the best score is below ``min_score``.

    Score = max(jaccard, containment_in_block). Containment is the
    deciding factor for short spans (single line of a long paragraph).
    """
    block_token_sets = [_toks(b["text"]) for b in blocks]
    out: list[int | None] = []
    for span_text in span_texts:
        st = _toks(span_text)
        if not st:
            out.append(None)
            continue
        best_score = 0.0
        best_idx: int | None = None
        for i, bt in enumerate(block_token_sets):
            if not bt:
                continue
            j = _jaccard(st, bt)
            c = _containment(st, bt)
            s = max(j, c)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_score >= min_score:
            out.append(best_idx)
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Deterministic per-span features
# ---------------------------------------------------------------------------


def _column_ids(spans: list[dict], page_w: float) -> list[int]:
    """Cluster the spans on one page into column buckets by left-edge x.

    Returns one column_id per span (0 = leftmost column, then 1, 2, ...).

    Algorithm (cheap, deterministic):
        * Collect each span's left edge x0.
        * Sort, bin by gaps > 0.06 * page_w (~36pt for letter at 612pt).
        * Number bins left-to-right.
    """
    if not spans:
        return []
    if page_w <= 0:
        page_w = 612.0
    gap_thresh = 0.06 * page_w
    xs = sorted(((s["bbox"][0], i) for i, s in enumerate(spans)))
    bins: list[list[int]] = []
    cur_bin: list[int] = []
    last_x: float | None = None
    for x, i in xs:
        if last_x is None or x - last_x <= gap_thresh:
            cur_bin.append(i)
        else:
            bins.append(cur_bin)
            cur_bin = [i]
        last_x = x
    if cur_bin:
        bins.append(cur_bin)

    cid_by_idx = [0] * len(spans)
    for cid, b in enumerate(bins):
        for i in b:
            cid_by_idx[i] = cid
    return cid_by_idx


def _is_artifact(span: dict, *, page_h: float) -> bool:
    """Heuristic from `semantik_structure.classify`'s _rule_page_header /
    _rule_page_footer: short text in the top/bottom 5% of the page.
    """
    bbox = span.get("bbox") or []
    if len(bbox) < 4 or page_h <= 0:
        return False
    text = (span.get("text") or "").strip()
    if not text or len(text) >= 100:
        return False
    top_frac = bbox[1] / page_h
    bot_frac = bbox[3] / page_h
    return top_frac < 0.05 or bot_frac > 0.95


def _ends_with_hyphen(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    # Match plain hyphen or unicode hyphens at line-end.
    return t[-1] in {"-", "‐", "‑", "–", "—"}


def _det_features(
    span: dict,
    *,
    column_id: int,
    is_artifact: bool,
) -> dict:
    bbox = span.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    text = span.get("text") or ""
    return {
        "column_id": int(column_id),
        "is_artifact": bool(is_artifact),
        "font_size": float(span.get("font_size") or 0.0),
        "x0": float(bbox[0]),
        "x1": float(bbox[2]),
        "y0": float(bbox[1]),
        "y1": float(bbox[3]),
        "ends_with_hyphen": _ends_with_hyphen(text),
        "is_bold": bool(span.get("is_bold") or False),
        "is_italic": bool(span.get("is_italic") or False),
    }


# ---------------------------------------------------------------------------
# Pair label derivation
# ---------------------------------------------------------------------------


def _join_type_for_positive(
    block_a: dict,
    block_b: dict,
    span_a_text: str,
    span_b_text: str,
) -> str:
    """When same_logical_block=yes, decide WHICH join."""
    # Both spans align to the same block by construction (caller asserts).
    block = block_a  # == block_b
    # list_continuation: both inside an <li>, OR same list_id
    if block["kind"] == "li":
        return "list_continuation"
    # Heuristic: are span_a and span_b on the SAME ar5iv "line"?
    # We check if the two span texts both appear within a single line
    # of the block (i.e. neither line crosses a <br>).
    lines = block.get("lines") or [block.get("text", "")]
    a_low = (span_a_text or "").lower()
    b_low = (span_b_text or "").lower()
    if not a_low or not b_low:
        return "space"
    # Use first 12 chars of each as a signature so OCR-noisy mutations
    # still match.
    sig_a = a_low[:12]
    sig_b = b_low[:12]
    for line in lines:
        ll = line.lower()
        if sig_a in ll and sig_b in ll:
            return "space"
    # Different lines but same block (paragraph wrapped in PDF):
    return "newline_within_p"


def _join_type_diag_for_negative(
    block_a: dict | None,
    block_b: dict | None,
) -> str:
    """Diagnostic-only label string for negative-primary pairs.

    Two spans landing in the same list (same list_id) get tagged
    ``list_continuation``; everything else is ``paragraph_break``. This
    string is only persisted in the ``join_type_str`` diagnostic field;
    the trainable integer ``join_type`` for negative-primary pairs is
    always ``JOIN_TYPE_IGNORE`` so the loss never sees it.
    """
    if block_a is not None and block_b is not None:
        list_a = block_a.get("list_id")
        list_b = block_b.get("list_id")
        if list_a is not None and list_a == list_b:
            return "list_continuation"
    return "paragraph_break"


def _hyphen_repair_label(
    span_a_text: str,
    span_b_text: str,
    block_a: dict | None,
    block_b: dict | None,
) -> int:
    """1 iff span_a ends with a hyphen AND ar5iv text joins the two
    sides without that hyphen as one word.
    """
    if not _ends_with_hyphen(span_a_text):
        return 0
    if block_a is None or block_b is None or block_a is not block_b:
        # Hyphen-repair is only ground-truth-able when both halves come
        # from the same ar5iv block (so we can see the actual joined word).
        return 0
    # Strip the trailing hyphen + take the trailing letters of span_a.
    a_clean = (span_a_text or "").rstrip()
    if not a_clean or a_clean[-1] not in "-‐‑–—":
        return 0
    a_letters_end = re.search(r"([A-Za-z]+)\s*-\s*$", a_clean)
    if not a_letters_end:
        return 0
    pre = a_letters_end.group(1)
    b_clean = (span_b_text or "").lstrip()
    b_letters_start = re.match(r"([A-Za-z]+)", b_clean)
    if not b_letters_start:
        return 0
    post = b_letters_start.group(1)
    candidate_word = (pre + post).lower()
    block_text = (block_a.get("text") or "").lower()
    # Match as a whole word: the joined "accessibility" should appear
    # in the ar5iv text and the hyphenated form should NOT.
    word_re = re.compile(rf"\b{re.escape(candidate_word)}\b")
    hyphenated_form = f"{pre.lower()}-{post.lower()}"
    if word_re.search(block_text) and hyphenated_form not in block_text:
        return 1
    return 0


# ---------------------------------------------------------------------------
# OCR-noise augmentation
# ---------------------------------------------------------------------------


_VISUAL_SUBS = {
    "l": "1", "1": "l",
    "O": "0", "0": "O",
    "I": "l", "S": "5",
}


def _ocr_noise(text: str, rng: random.Random) -> str:
    """Apply 1-3 OCR-noise mutations to ``text``."""
    if len(text) < 4:
        return text
    out = list(text)
    n_ops = rng.randint(1, 3)
    for _ in range(n_ops):
        op = rng.choice(("drop", "visual", "space", "rn_m"))
        if op == "drop" and len(out) > 4:
            i = rng.randint(0, len(out) - 1)
            del out[i]
        elif op == "visual":
            for i, ch in enumerate(out):
                if ch in _VISUAL_SUBS and rng.random() < 0.10:
                    out[i] = _VISUAL_SUBS[ch]
                    break
        elif op == "space" and len(out) > 6:
            i = rng.randint(2, len(out) - 2)
            out.insert(i, " ")
        elif op == "rn_m":
            joined = "".join(out)
            if "rn" in joined and rng.random() < 0.5:
                joined = joined.replace("rn", "m", 1)
            elif "m" in joined and rng.random() < 0.5:
                joined = joined.replace("m", "rn", 1)
            out = list(joined)
    return "".join(out)


# ---------------------------------------------------------------------------
# Per-paper extraction
# ---------------------------------------------------------------------------


def _cache_path_candidates(cache_dir: Path, arxiv_id: str) -> list[Path]:
    safe = arxiv_id.replace("/", "_")
    return [cache_dir / f"{safe}.html"]


def load_cached_ar5iv(arxiv_id: str, cache_dir: Path) -> str | None:
    for p in _cache_path_candidates(cache_dir, arxiv_id):
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                return None
    return None


def per_page_pypdfium2_spans(shared: dict) -> list[tuple[int, list[dict], float, float]]:
    """Return ``[(page_num, [span dicts], page_w, page_h)]`` from the
    shared extract output.

    Operates on the LINE-LEVEL ``merged`` text stream — pypdfium2's raw
    ``count_rects`` output is glyph-level (one rect per character) for
    most arXiv PDFs, which doesn't carry the adjacency signal MergeOrSplit
    needs. The merged stream is the deterministic, line-clustered output
    of ``extract_shared`` (pdfplumber line blocks + clustered pypdfium2
    orphans) and is exactly what the v2 pipeline's stage-2 FeatureBlock
    stream consumes downstream. That's the input MergeOrSplit operates on
    in production, so it's also what we train on here.
    """
    out: list[tuple[int, list[dict], float, float]] = []
    for page in shared.get("pages", []):
        page_num = int(page.get("page_num", 1))
        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))
        merged = page.get("merged", {}).get("text_blocks", []) or []
        spans: list[dict] = []
        for b in merged:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            bbox = b.get("bbox") or [0, 0, 0, 0]
            spans.append({
                "bbox": [float(x) for x in bbox],
                "text": text,
                "font_size": (
                    float(b["font_size"]) if b.get("font_size") else None
                ),
                "font_name": b.get("font_name"),
                "is_bold": bool(b.get("is_bold") or False),
                "is_italic": bool(b.get("is_italic") or False),
            })
        # Sort top-to-bottom, left-to-right.
        spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        out.append((page_num, spans, page_w, page_h))
    return out


def build_pair_rows_for_paper(
    arxiv_id: str,
    pdf_path: Path,
    html: str,
    extract_cache_dir: Path,
    *,
    max_pairs_per_paper: int,
    rng: random.Random,
    noise_frac: float,
) -> tuple[list[dict], dict]:
    """Build labelled pair rows for one paper.

    Returns ``(rows, paper_stats)``. Stats include n_pages, n_spans,
    n_pairs, n_aligned_spans for the coverage report.
    """
    from semantik_structure.extract_shared import extract_shared_cached

    shared = extract_shared_cached(pdf_path, cache_dir=extract_cache_dir)
    blocks = walk_ar5iv_blocks(html)
    if not blocks:
        return [], {
            "arxiv_id": arxiv_id,
            "n_pages": 0, "n_spans": 0, "n_pairs": 0, "n_aligned_spans": 0,
        }

    pages = per_page_pypdfium2_spans(shared)
    rows: list[dict] = []
    n_total_spans = 0
    n_aligned_spans = 0
    for page_num, spans, page_w, page_h in pages:
        if len(spans) < 2:
            continue
        n_total_spans += len(spans)

        # Deterministic per-span features.
        col_ids = _column_ids(spans, page_w)
        artifact_flags = [_is_artifact(s, page_h=page_h) for s in spans]

        # Align spans to ar5iv blocks.
        span_texts = [s["text"] for s in spans]
        block_ids = align_spans_to_blocks(span_texts, blocks)
        n_aligned_spans += sum(1 for b in block_ids if b is not None)

        # Compute median line height for line_height_ratio feature.
        heights = [s["bbox"][3] - s["bbox"][1] for s in spans
                   if s["bbox"][3] > s["bbox"][1]]
        median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
        if median_h <= 0:
            median_h = 12.0

        for i in range(len(spans) - 1):
            span_a = spans[i]
            span_b = spans[i + 1]
            bid_a = block_ids[i]
            bid_b = block_ids[i + 1]
            block_a = blocks[bid_a] if bid_a is not None else None
            block_b = blocks[bid_b] if bid_b is not None else None

            if block_a is None and block_b is None:
                # Both spans unaligned — no useful supervision. Skip.
                continue

            # ---- labels ----
            same_block = int(
                bid_a is not None and bid_b is not None and bid_a == bid_b
            )
            if same_block:
                join_type_str = _join_type_for_positive(
                    block_a, block_b, span_a["text"], span_b["text"],
                )
                join_type_id = JOIN_TYPE_TO_ID[join_type_str]
            else:
                # Mask join_type for negative-primary pairs (loss skipped).
                # The string label is diagnostic-only — the int label is
                # always JOIN_TYPE_IGNORE so the model never learns it.
                join_type_str = _join_type_diag_for_negative(block_a, block_b)
                join_type_id = JOIN_TYPE_IGNORE
            hyphen_lbl = _hyphen_repair_label(
                span_a["text"], span_b["text"], block_a, block_b,
            )

            # ---- deterministic features ----
            det_a = _det_features(
                span_a, column_id=col_ids[i], is_artifact=artifact_flags[i],
            )
            det_b = _det_features(
                span_b, column_id=col_ids[i + 1], is_artifact=artifact_flags[i + 1],
            )
            y_gap = max(0.0, span_b["bbox"][1] - span_a["bbox"][3])
            line_height_ratio = (
                (span_a["bbox"][3] - span_a["bbox"][1]) / median_h
                if median_h > 0 else 1.0
            )

            # ---- OCR-noise augmentation ----
            noisy = rng.random() < noise_frac
            if noisy:
                txt_a = _ocr_noise(span_a["text"], rng)
                txt_b = _ocr_noise(span_b["text"], rng)
            else:
                txt_a = span_a["text"]
                txt_b = span_b["text"]

            row = {
                "document_id": f"arxiv:{arxiv_id}",
                "page": page_num,
                "span_a_text": txt_a,
                "span_b_text": txt_b,
                "det_features": {
                    "a": det_a,
                    "b": det_b,
                    "y_gap": float(y_gap),
                    "line_height_ratio": float(line_height_ratio),
                },
                "labels": {
                    "same_logical_block": int(same_block),
                    "join_type": int(join_type_id),
                    "hyphen_repair": int(hyphen_lbl),
                },
                "join_type_str": join_type_str,
                "noisy": bool(noisy),
            }
            rows.append(row)
            if max_pairs_per_paper and len(rows) >= max_pairs_per_paper:
                break
        if max_pairs_per_paper and len(rows) >= max_pairs_per_paper:
            break

    return rows, {
        "arxiv_id": arxiv_id,
        "n_pages": len(pages),
        "n_spans": n_total_spans,
        "n_pairs": len(rows),
        "n_aligned_spans": n_aligned_spans,
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    rows: list[dict],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["labels"]["same_logical_block"]].append(r)
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
        val.extend(lst[n_train:n_train + n_val])
        test.extend(lst[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--extract-cache-dir", type=Path,
                    default=DEFAULT_EXTRACT_CACHE)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many distinct arxiv_ids (0 = all).")
    ap.add_argument("--max-pairs-per-paper", type=int, default=400)
    ap.add_argument("--max-total-rows", type=int, default=120000)
    ap.add_argument("--noise-frac", type=float, default=0.10,
                    help="Fraction of pairs to mutate with OCR-noise.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Make repo importable so semantik_structure.extract_shared resolves.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    pair_files = sorted(args.pairs_dir.glob("*.json"))
    if not pair_files:
        sys.exit(f"no pairs under {args.pairs_dir}")

    seen_ids: dict[str, dict] = {}
    for p in pair_files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        aid = d.get("arxiv_id")
        if not aid or aid in seen_ids:
            continue
        local_pdf = d.get("local_pdf")
        if not local_pdf or not Path(local_pdf).exists():
            continue
        if not load_cached_ar5iv(aid, args.cache_dir):
            continue
        seen_ids[aid] = {"pdf": Path(local_pdf), "pair_file": p}

    print(f"[pairs] {len(seen_ids)} distinct arxiv_ids with PDF + cached HTML")

    if args.limit:
        ids = list(seen_ids)[: args.limit]
        seen_ids = {a: seen_ids[a] for a in ids}
        print(f"[pairs] truncated to {len(seen_ids)} ids (--limit)")

    rng = random.Random(args.seed)

    rows: list[dict] = []
    paper_stats: list[dict] = []
    n_papers_skipped = 0
    n_papers_with_rows = 0
    t0 = time.time()

    for i, (aid, meta) in enumerate(seen_ids.items()):
        pdf_path = meta["pdf"]
        html = load_cached_ar5iv(aid, args.cache_dir)
        if html is None:
            n_papers_skipped += 1
            continue
        try:
            paper_rows, stats = build_pair_rows_for_paper(
                aid, pdf_path, html, args.extract_cache_dir,
                max_pairs_per_paper=args.max_pairs_per_paper,
                rng=rng,
                noise_frac=args.noise_frac,
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[error] {aid}: {exc}\n")
            n_papers_skipped += 1
            continue
        rows.extend(paper_rows)
        paper_stats.append(stats)
        if paper_rows:
            n_papers_with_rows += 1
        if (i + 1) % 25 == 0 or args.limit and i + 1 == args.limit:
            dt = time.time() - t0
            print(f"[paper {i+1}/{len(seen_ids)}] rows={len(rows)} "
                  f"with_rows={n_papers_with_rows} skipped={n_papers_skipped} "
                  f"elapsed={dt:.1f}s")
        if args.max_total_rows and len(rows) >= args.max_total_rows:
            print(f"[stop] hit --max-total-rows={args.max_total_rows}")
            break

    if not rows:
        sys.exit("[fatal] no pair rows extracted")

    # Per-head class prevalence.
    same_counts = Counter(r["labels"]["same_logical_block"] for r in rows)
    join_counts = Counter(r["join_type_str"] for r in rows
                          if r["labels"]["same_logical_block"] == 1)
    hyphen_counts = Counter(r["labels"]["hyphen_repair"] for r in rows)
    noisy_n = sum(1 for r in rows if r["noisy"])

    print(f"\n[balance] same_logical_block ({len(rows)} rows):")
    for k in (0, 1):
        c = same_counts.get(k, 0)
        pct = 100.0 * c / max(1, len(rows))
        print(f"  {k}  {c:7}  ({pct:5.2f}%)")
    pos_total = same_counts.get(1, 0)
    print(f"[balance] join_type (only positive-primary pairs, n={pos_total}):")
    for jt in JOIN_TYPES:
        c = join_counts.get(jt, 0)
        pct = (100.0 * c / pos_total) if pos_total else 0.0
        print(f"  {jt:24} {c:7}  ({pct:5.2f}%)")
    print(f"[balance] hyphen_repair:")
    for k in (0, 1):
        c = hyphen_counts.get(k, 0)
        pct = 100.0 * c / max(1, len(rows))
        print(f"  {k}  {c:7}  ({pct:5.2f}%)")
    print(f"[balance] noisy={noisy_n} ({100.0 * noisy_n / max(1, len(rows)):5.2f}%)")

    train, val, test = stratified_split(rows, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")
    print(f"\n[split] train={len(train)} val={len(val)} test={len(test)} "
          f"-> {args.out_dir}")

    coverage = {
        "n_papers_attempted": len(seen_ids),
        "n_papers_with_rows": n_papers_with_rows,
        "n_papers_skipped": n_papers_skipped,
        "total_rows": len(rows),
        "noisy_rows": noisy_n,
        "noise_frac": args.noise_frac,
        "same_logical_block_counts": {str(k): v for k, v in same_counts.items()},
        "join_type_counts_positive_only": dict(join_counts),
        "hyphen_repair_counts": {str(k): v for k, v in hyphen_counts.items()},
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "join_types": JOIN_TYPES,
    }
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2)
    )
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
