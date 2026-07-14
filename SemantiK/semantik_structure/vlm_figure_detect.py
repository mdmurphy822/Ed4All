"""VLM figure-DETECTION lane (task #56) — the last WCAG 1.1.1 blocker.

WHY THIS EXISTS
---------------
On a page-raster scan (the page-arranger lane's usual input) there are **no
sub-page image page-objects at all**. Measured on a real 10-chapter scanned
textbook: 1,236 pages -> ZERO sub-page image objects; every image page-object is
the FULL-PAGE RASTER (612x792pt, 100% of the page). The figures — diagrams,
number lines, charts, algebra-tile arrays — are *pixels inside the page image*.
Object extraction cannot find them; only a VISION pass can.

So ``SEMANTIK_DETECT_FIGURES`` correctly yields ZERO figures there (every
candidate is a page raster and the mandatory page-raster guard excludes it), and
every figure in the book is silently lost — a WCAG 1.1.1 content loss.

WHAT THIS DOES
--------------
One tightly-scoped multimodal DETECT call per page on the SAME seat the
page-arranger already uses (:func:`page_arranger.resolve_arranger_seat`), asking
for the bounding boxes of the page's figures. The returned boxes are **proposals
only**: a deterministic accept gate (:func:`accept_figure_boxes`) decides, and
the boxes it accepts are injected into the extraction's per-page ``images`` list
as ordinary image dicts — after which **the entire existing figure chain runs
completely unchanged**::

    shared['pages'][i]['images']            <-- THIS MODULE INJECTS HERE
      -> region_detection.detect_image_region_candidates   (ImageCandidate)
      -> features._interleave_image_feature_blocks         (synthetic image FB,
                                                             spliced in reading order)
      -> page_arranger.build_figure_regions
           -> structure_graph.is_page_raster_candidate     (THE GUARD, still fires)
           -> structure_graph.build_figure_region_from_candidate  (SHARED builder)
      -> Stage 5c  image_extract.render_figure_regions_to_bytes   (PNG sidecar)
      -> Stage 6b  figure_captioner.caption_figure_regions        (alt text)
      -> <img src> + alt + <figcaption>

Not one line of that chain changes. This module supplies the bboxes it was
already waiting for.

WHY A SIBLING CALL, NOT A RIDER ON THE ARRANGE CALL
---------------------------------------------------
The arrange call fires inside ``ArrangerRoute.arrange_regions`` — i.e. AFTER
``featurize_with_regions`` has already frozen the FeatureBlock stream. Figure
FeatureBlocks must be INTERLEAVED into that stream in reading order (every
unit's ``fb_index`` is its exact position, and the coverage invariant counts
them), so bboxes discovered at arrange time would arrive too late to be placed
correctly. Detection therefore runs one stage EARLIER, at the extraction seam,
where injecting into ``images`` lets the existing interleave do the placement.
Same seat, same fan-out / sidecar / stop patterns — just an earlier, tighter
call with zero blast radius on the arrange contract.

DOCTRINE
--------
**The VLM PROPOSES bboxes; deterministic code + the guard DECIDE.** Identical
posture to the page-arranger (the model may only name *ids* / *boxes*, never
author output). Specifically:

* The **page-raster guard is KEPT as a backstop.** A VLM that returns the whole
  page must still be rejected. The ceiling is imported from
  ``structure_graph._PAGE_RASTER_MIN_COVERAGE`` so the two cannot drift, and it
  is enforced *twice*: once here at accept time (cheap, with a reason recorded)
  and again — unchanged — by ``is_page_raster_candidate`` downstream. Never
  trust the model not to do this.
* **Degenerate boxes are rejected**: zero/negative area, out-of-page, absurd
  aspect ratios.
* **Text is rejected — by TWO independent arms.** Cropping text ships a
  photograph of body text: a WCAG 1.4.5 images-of-text regression, strictly
  *worse* than the missing figure it pretends to fix. Arm 1 (AREA coverage)
  catches a prose COLUMN. Arm 2 (WORD COUNT) catches a procedure TABLE — which
  arm 1 provably cannot, because a table's coverage measures *below* a genuine
  figure's (see :data:`_MAX_TEXT_WORDS` for the measured numbers).

Default OFF (``SEMANTIK_VLM_FIGURE_DETECT``) -> this module is never imported by
the cascade, no POST fires, ``shared`` is untouched -> byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SEMANTIK_VLM_FIGURE_DETECT_ENV = "SEMANTIK_VLM_FIGURE_DETECT"
_CONCURRENCY_ENV = "SEMANTIK_VLM_FIGURE_DETECT_CONCURRENCY"
_CHECKPOINT_ENV = "SEMANTIK_VLM_FIGURE_DETECT_CHECKPOINT"
_CHECKPOINT_FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"
_TIMEOUT_ENV = "SEMANTIK_VLM_FIGURE_DETECT_TIMEOUT"
_MAX_PER_PAGE_ENV = "SEMANTIK_VLM_FIGURE_DETECT_MAX_PER_PAGE"
_MAX_WORDS_ENV = "SEMANTIK_VLM_FIGURE_DETECT_MAX_WORDS"
_DISABLE_THINKING_ENV = "SEMANTIK_VLM_FIGURE_DETECT_DISABLE_THINKING"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})

_DEFAULT_CONCURRENCY = 4
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_MAX_PER_PAGE = 8
# Thinking is ON by default (see resolve_detect_disable_thinking), so the budget
# must cover the reasoning block AND the JSON answer.
_DETECT_MAX_TOKENS = 4096
_CACHE_BASENAME = "vlm_figure_detect_cache"

# Prompt-contract version — salts the resume-sidecar key, so a prompt change
# never serves a stale verdict. v2: thinking-ON (v1 was thinking-off, which
# SUPPRESSED localization entirely — see resolve_detect_disable_thinking).
DETECT_PROMPT_VERSION = 2

# --- Deterministic accept-gate constants (calibration, NOT corpus targets) ---
# A box must cover at least this fraction of the page to be a figure worth
# cropping (below it is a glyph, a bullet, a rule, or noise).
# TODO(calibration) — domain-agnostic geometry, tuned on scanned-textbook pages.
_MIN_AREA_FRACTION = 0.010
# Absurd aspect ratios are not figures (a hairline rule, a margin strip).
_MIN_ASPECT = 0.06
_MAX_ASPECT = 16.0
# A box must be at least this fraction of the page in BOTH dimensions — a
# 1-pixel-tall full-width strip has a fine area but is a rule, not a figure.
_MIN_SIDE_FRACTION = 0.03
# TEXT-COLUMN GUARD (arm 1 — AREA coverage): a box whose area is this covered by
# extracted text blocks is a column of prose, not a figure. Cropping it = WCAG
# 1.4.5 images-of-text.
# TODO(calibration) — measured on a real scanned-textbook chapter: a dense prose
# column the model proposed measured 0.59 and was correctly rejected here.
_MAX_TEXT_COVERAGE = 0.35
# TEXT-DENSITY GUARD (arm 2 — WORD COUNT). Load-bearing, and the coverage arm
# provably CANNOT replace it. A procedure TABLE ("Step 1..4", all prose — an
# unambiguous images-of-text crop) has thin word boxes separated by cell
# whitespace, so its AREA coverage measures BELOW a genuine figure's and no
# coverage threshold can order the two. Word COUNT does.
#
# MEASURED on a real scanned-textbook chapter (production signal — words whose
# text-block centre falls inside the proposed box):
#     GENUINE chart / counters figure ....  1,  3 words   (coverage 0.08 / 0.22)
#     GENUINE chapter photo .............. 15 words       (coverage 0.02)
#     MIXED step-table w/ factor trees ... 22 words       (coverage 0.07)
#     PURE "Step 1..4" text table ........ 63 words       (coverage 0.07)  <-- the defect
#     prose COLUMN ........................ 0 words       (coverage 0.70)  <-- arm 1 catches
# 20 sits between the genuine photo (15) and the mixed table (22) and ERRS TOWARD
# REJECTING: per the standing doctrine we would rather MISS a figure than ship a
# crop of readable text (a WCAG 1.4.5 regression is strictly worse than the WCAG
# 1.1.1 gap it pretends to close). The mixed table is therefore rejected too —
# its factor trees are genuine, but they are inseparable from 22 words of
# procedural prose that would ship as pixels.
# TODO(calibration) — domain-agnostic ("a figure must not carry a paragraph of
# readable text"), NOT a corpus target; env-tunable via
# SEMANTIK_VLM_FIGURE_DETECT_MAX_WORDS (0 disables the arm).
_MAX_TEXT_WORDS = 20
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Two proposals overlapping this much are the same figure (keep the first).
_DEDUP_IOU = 0.55


def _page_raster_min_coverage() -> float:
    """The page-raster coverage ceiling — imported, never re-declared.

    Single source of truth: ``structure_graph._PAGE_RASTER_MIN_COVERAGE``. The
    accept gate and the downstream mandatory guard MUST use the same number or a
    box could pass here and be dropped there (or worse, vice versa).
    """
    try:
        from .structure_graph import _PAGE_RASTER_MIN_COVERAGE

        return float(_PAGE_RASTER_MIN_COVERAGE)
    except Exception:  # noqa: BLE001 — never crash the gate on an import
        return 0.90


# ---------------------------------------------------------------------------
# Flag resolution (parse-with-fallback, house posture).
# ---------------------------------------------------------------------------
def resolve_vlm_figure_detect_mode() -> bool:
    """``SEMANTIK_VLM_FIGURE_DETECT`` — default OFF (byte-identical).

    **DUAL-GATE: additionally requires ``SEMANTIK_DETECT_FIGURES``** (mirroring
    the house ``SEMANTIK_SEMANTIC_CLASS`` requires-``SEMANTIK_BLOCK_REVIEW``
    precedent). This is load-bearing, not decorative: ``features`` builds an
    ImageCandidate from ANY ``page['images']`` entry regardless of the figures
    flag, but Stage 5c (PNG render) and Stage 6b (captioner) are gated ON
    ``cascade.resolve_detect_figures``. Injecting bboxes with the figures flag
    off would therefore mint figure Regions that never get ``image_png_bytes``
    — and a ``kind="figure"`` Region without that payload is CHAPTER-FATAL one
    stage later (the Stage-6b captioner fails closed on it, the documented live
    ch09 second-order defect). So detection self-disables unless the chain that
    consumes it is on. A loud warning fires on the misconfiguration rather than
    a silent no-op.
    """
    raw = (os.environ.get(SEMANTIK_VLM_FIGURE_DETECT_ENV) or "").strip().lower()
    if raw not in _TRUTHY:
        return False
    try:
        from .cascade import resolve_detect_figures

        figures_on = bool(resolve_detect_figures())
    except Exception:  # noqa: BLE001 — never crash the resolver on an import
        figures_on = False
    if not figures_on:
        logger.warning(
            "SEMANTIK_VLM_FIGURE_DETECT is set but SEMANTIK_DETECT_FIGURES is OFF "
            "— figure detection is DISABLED. The Stage-5c PNG render + Stage-6b "
            "captioner are gated on SEMANTIK_DETECT_FIGURES, so injected bboxes "
            "would mint figure Regions with no image bytes (chapter-fatal at 6b). "
            "Set SEMANTIK_DETECT_FIGURES=1 to enable the figure lane."
        )
        return False
    return True


def resolve_detect_concurrency() -> int:
    raw = (os.environ.get(_CONCURRENCY_ENV) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CONCURRENCY
    return val if val > 0 else _DEFAULT_CONCURRENCY


def resolve_detect_timeout() -> float:
    raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    if val != val or val in (float("inf"), float("-inf")) or val <= 0:
        return _DEFAULT_TIMEOUT_SECONDS
    return val


def resolve_detect_max_per_page() -> int:
    raw = (os.environ.get(_MAX_PER_PAGE_ENV) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PER_PAGE
    return val if val > 0 else _DEFAULT_MAX_PER_PAGE


def resolve_detect_max_words() -> int:
    """Word ceiling for the TEXT-DENSITY guard (``0`` disables that arm)."""
    raw = (os.environ.get(_MAX_WORDS_ENV) or "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _MAX_TEXT_WORDS
    return val if val >= 0 else _MAX_TEXT_WORDS


def resolve_detect_disable_thinking() -> bool:
    """``SEMANTIK_VLM_FIGURE_DETECT_DISABLE_THINKING`` — default OFF ⇒ thinking ON.

    **Load-bearing, and measured.** Figure localization is a SPATIAL REASONING
    task, not a copy task. With ``thinking: false`` the seat returns
    ``{"figures": []}`` on EVERY page — including a page carrying a full-width
    construction photograph it can describe perfectly when asked in prose (so it
    SEES the figure; the non-thinking path simply will not ground it). Flipping
    thinking ON with the identical prompt returns the correct box on that page
    (`[0.1875, 0.125, 1.0, 0.4375], kind=photo, confidence=0.9`).

    This deliberately mirrors — and is the mirror IMAGE of — the extraction
    client's ``SEMANTIK_VLM_DISABLE_THINKING``: page TRANSCRIPTION is a copy task
    that wants thinking OFF (~10x faster), while this DETECTION pass wants it ON,
    exactly as ``SEMANTIK_REASONING_QC_DISABLE_THINKING`` defaults thinking ON for
    the QC judgment. The two are independent envs so a shared seat can run
    extraction thinking-off and detection thinking-on in the SAME process.

    Truthy → the operator has deliberately chosen a thinking-off detect run
    (which, on the measured seat, finds nothing — it exists as an escape hatch for
    a seat whose template rejects the field, not as a recommended mode).
    """
    raw = (os.environ.get(_DISABLE_THINKING_ENV) or "").strip().lower()
    return raw in _TRUTHY


def resolve_detect_checkpoint() -> bool:
    """Default ON, falling back to the ``ED4ALL_GENERATION_CHECKPOINT`` family."""
    raw = os.environ.get(_CHECKPOINT_ENV)
    if raw is not None and raw.strip():
        return raw.strip().lower() not in _FALSEY
    fam = os.environ.get(_CHECKPOINT_FAMILY_ENV)
    if fam is not None and fam.strip():
        return fam.strip().lower() not in _FALSEY
    return True


# ---------------------------------------------------------------------------
# The DETECT call.
# ---------------------------------------------------------------------------
DETECT_SYSTEM = """\
You locate FIGURES on a page from a scanned textbook.

A FIGURE is a piece of visual, non-prose content: a diagram, an illustration, a
photograph, a chart or graph, a number line, a coordinate plane, a geometric
shape, an array of manipulatives (algebra tiles, counters, blocks), or a
hand-drawn sketch.

The following are NOT figures. Never return a box for them:
  - paragraphs, sentences, or any run of body text
  - headings, titles, page numbers, running headers or footers
  - a column of text (even a narrow one)
  - equations, expressions, or lines of mathematical notation set as text
  - tables of numbers or words
  - exercise lists and answer keys

Return STRICT JSON only, no prose, in exactly this shape:

{"figures": [{"bbox": [x0, y0, x1, y1], "kind": "diagram", "confidence": 0.9}]}

Rules for bbox:
  - Coordinates are NORMALIZED to the page: 0.0 = left/top edge, 1.0 =
    right/bottom edge. They are fractions, never pixels.
  - x0 < x1 and y0 < y1. Origin is the TOP-LEFT of the page.
  - Box the figure's visual extent TIGHTLY. Do NOT include its caption, its
    surrounding paragraphs, or whitespace padding.
  - NEVER return a box covering the whole page or most of it. If the page is
    just text, return {"figures": []}.

"kind" is one of: diagram, illustration, photo, chart, number_line, geometry,
manipulative, sketch.
If the page has no figures, return {"figures": []}.
"""


def _extract_json(content: str) -> dict:
    """Tolerant JSON extraction (mirrors ``page_arranger_contract.extract_json``)."""
    text = (content or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class DetectNonJudgment(RuntimeError):
    """The seat produced NO usable verdict for a page (empty / unparseable body).

    Distinct from a genuine ``{"figures": []}`` ("this page has no figures"). The
    difference is load-bearing for the resume sidecar: a genuine verdict is
    CACHED, a non-judgment must NOT be — otherwise a transient runaway-thinker
    timeout on one page is persisted as "no figures here" and every later
    ``--resume`` serves it from cache, turning a transient blip into PERMANENT,
    SILENT figure loss (the exact WCAG 1.1.1 loss this lane exists to fix).
    Mirrors ``page_arranger`` (caches only ``status=='ok'``) and ``reasoning_qc``
    (never caches a ``_qc_incomplete`` verdict).
    """


def detect_page_figures(
    seat,
    image_b64: str,
    page_label: int,
    *,
    timeout: Optional[float] = None,
    requests_module=None,
) -> "tuple[list[dict], dict, float]":
    """One DETECT POST for one page. Returns ``(raw_proposals, usage, wall_s)``.

    ``raw_proposals`` is the model's UNFILTERED list — proposals only. The accept
    gate (:func:`accept_figure_boxes`) is what decides.

    Raises :class:`DetectNonJudgment` when the seat returned NO usable verdict (an
    empty completion — a runaway thinker exhausting the window — or an unparseable
    body). That is NOT the same as a genuine empty ``{"figures": []}``, and the
    caller must not cache it.
    """
    from .vlm_extract import _chat_completions_url

    requests = requests_module
    if requests is None:
        import requests as requests  # noqa: PLC0415

    timeout = resolve_detect_timeout() if timeout is None else timeout
    if not image_b64:
        # No page image -> nothing a vision pass can find. Never POST blind.
        return [], {}, 0.0

    user_text = (
        f"Page {page_label}. Locate every FIGURE on this page and return its "
        f'normalized bounding box. Return {{"figures": []}} if the page is '
        f"only text."
    )
    body = {
        "model": seat.model,
        "messages": [
            {"role": "system", "content": DETECT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": _DETECT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    # Thinking ON by default — localization is a spatial REASONING task; the
    # non-thinking path returns {"figures": []} on every page (see
    # resolve_detect_disable_thinking).
    if resolve_detect_disable_thinking():
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if getattr(seat, "api_key", None):
        headers["Authorization"] = f"Bearer {seat.api_key}"

    t0 = time.time()
    resp = requests.post(
        _chat_completions_url(seat.base_url), json=body, headers=headers, timeout=timeout
    )
    wall = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {}) or {}
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    finish = choice.get("finish_reason")
    if not content:
        # A runaway thinker exhausted the completion window. NEVER silent: a
        # suppressed page loses every figure on it, which is exactly the WCAG
        # 1.1.1 loss this lane exists to fix.
        logger.warning(
            "vlm-figure-detect page %s: EMPTY completion (finish_reason=%s, "
            "max_tokens=%s) — no figures recovered for this page. Raise "
            "%s or set %s=1 if this recurs.",
            page_label, finish, _DETECT_MAX_TOKENS,
            _TIMEOUT_ENV, _DISABLE_THINKING_ENV,
        )
        raise DetectNonJudgment(
            f"page {page_label}: empty completion (finish_reason={finish})"
        )
    try:
        parsed = _extract_json(content)
    except Exception as exc:  # noqa: BLE001 — tolerant parse
        logger.warning(
            "vlm-figure-detect page %s: UNPARSEABLE body (%s) — no figures "
            "recovered for this page (will re-run on resume, never cached).",
            page_label, exc,
        )
        raise DetectNonJudgment(f"page {page_label}: unparseable body") from exc
    figures = parsed.get("figures") if isinstance(parsed, dict) else None
    if not isinstance(figures, list):
        logger.warning(
            "vlm-figure-detect page %s: body carried no 'figures' list — no "
            "figures recovered for this page (will re-run on resume, never cached).",
            page_label,
        )
        raise DetectNonJudgment(f"page {page_label}: no 'figures' list")
    # A GENUINE verdict (possibly a legitimately empty one) — safe to cache.
    return [f for f in figures if isinstance(f, dict)], usage, wall


# ---------------------------------------------------------------------------
# THE DETERMINISTIC ACCEPT GATE — this is what DECIDES.
# ---------------------------------------------------------------------------
def _iou(a: "tuple[float, float, float, float]", b: "tuple[float, float, float, float]") -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _text_coverage(
    box: "tuple[float, float, float, float]",
    text_boxes: "list[tuple[float, float, float, float]]",
) -> float:
    """Fraction of ``box``'s area covered by extracted TEXT blocks (normalized).

    The text boxes are SAMPLED onto a coarse grid rather than area-summed, so
    OVERLAPPING text blocks (common on the OCR lane — a line block plus its word
    blocks) cannot inflate the coverage past 1.0 and false-reject a real figure.
    """
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    if bw <= 0 or bh <= 0 or not text_boxes:
        return 0.0
    n = 16  # 16x16 = 256 sample cells over the candidate box
    hit = 0
    for r in range(n):
        cy = by0 + (r + 0.5) * bh / n
        for c in range(n):
            cx = bx0 + (c + 0.5) * bw / n
            for tx0, ty0, tx1, ty1 in text_boxes:
                if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                    hit += 1
                    break
    return hit / float(n * n)


def _text_word_count(
    box: "tuple[float, float, float, float]",
    text_items: "list[tuple[tuple[float, float, float, float], str]]",
) -> int:
    """Readable WORDS carried INSIDE ``box`` (the text-density guard's signal).

    A text block counts when its CENTRE falls inside the box (centre-in-box, so a
    long line merely clipped by the figure's edge is not double-counted). Words
    are alphabetic runs of >=2 chars — digits alone never count, because a genuine
    chart axis / number line is all digits and must NOT be penalised for them.
    """
    bx0, by0, bx1, by1 = box
    n = 0
    for (tx0, ty0, tx1, ty1), text in text_items:
        cx, cy = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            n += len(_WORD_RE.findall(text or ""))
    return n


def accept_figure_boxes(
    proposals: "list[dict]",
    *,
    page_w: float,
    page_h: float,
    text_boxes_norm: "Optional[list[tuple[float, float, float, float]]]" = None,
    text_items_norm: "Optional[list[tuple[tuple[float, float, float, float], str]]]" = None,
    max_per_page: Optional[int] = None,
    max_words: Optional[int] = None,
) -> "tuple[list[dict], dict]":
    """Decide which proposed boxes become figures. Returns ``(accepted, stats)``.

    THIS is the decider — the VLM only proposed. Every rejection reason is
    counted into ``stats`` so a run can be audited (and so the DecisionCapture
    rationale can interpolate it).

    ``accepted`` entries are ready-to-inject extraction image dicts::

        {"bbox": [x0, y0, x1, y1],   # PDF-POINT space, TOP-LEFT origin
         "px_size": [w, h],          # nominal crop pixels (spacer-filter input)
         "index": i,
         "vlm_figure": True, "vlm_kind": ..., "vlm_confidence": ...}

    Rejections, in order:

    ``malformed``    bbox absent / not 4 numbers
    ``degenerate``   x1<=x0, y1<=y0, or NaN
    ``out_of_page``  coordinates outside the page
    ``page_raster``  >= the page-raster coverage ceiling — THE BACKSTOP. A VLM
                     that returns the whole page is rejected here, and would be
                     rejected AGAIN by ``is_page_raster_candidate`` downstream.
    ``too_small``    below the area floor or thinner than the side floor
    ``bad_aspect``   absurd aspect ratio (a rule / margin strip)
    ``text_column``  mostly covered by extracted text — prose, not a figure.
                     Cropping it would be a WCAG 1.4.5 images-of-text REGRESSION.
    ``duplicate``    same figure as an already-accepted box (IoU >= threshold)
    ``over_cap``     beyond ``max_per_page``
    """
    max_per_page = resolve_detect_max_per_page() if max_per_page is None else max_per_page
    max_words = resolve_detect_max_words() if max_words is None else max_words
    raster_ceiling = _page_raster_min_coverage()
    text_items = list(text_items_norm or ())
    # The coverage arm needs only the boxes; derive them from the items when the
    # caller supplied the richer shape, so the two arms can never disagree.
    if text_boxes_norm is None and text_items:
        text_boxes_norm = [bb for bb, _t in text_items]
    text_boxes_norm = list(text_boxes_norm or ())
    stats = {
        "proposed": len(proposals or ()),
        "accepted": 0,
        "malformed": 0,
        "degenerate": 0,
        "out_of_page": 0,
        "too_small": 0,
        "bad_aspect": 0,
        "page_raster": 0,
        "text_column": 0,
        "text_dense": 0,
        "duplicate": 0,
        "over_cap": 0,
    }
    if page_w <= 0 or page_h <= 0:
        stats["malformed"] = stats["proposed"]
        return [], stats

    kept: list[tuple[tuple[float, float, float, float], dict]] = []
    for prop in proposals or ():
        raw = prop.get("bbox")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            stats["malformed"] += 1
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in raw)
        except (TypeError, ValueError):
            stats["malformed"] += 1
            continue
        if any(v != v for v in (x0, y0, x1, y1)):  # NaN
            stats["degenerate"] += 1
            continue
        # COORDINATE-CONVENTION rescue. The prompt demands 0..1 fractions, but VLMs
        # commonly emit percentages (0..100) or — the Qwen-VL family convention,
        # observed live from this very seat under a direct grounding ask — a 0..1000
        # grid. Treating those as "out of page" would silently discard EVERY figure
        # on a model that switched convention, which is the exact WCAG 1.1.1 loss
        # this lane exists to fix. Rescale by the smallest convention that contains
        # the box; anything beyond is genuinely out of page.
        hi = max(x0, y0, x1, y1)
        if hi > 1.0:
            if hi <= 100.0:
                div = 100.0
            elif hi <= 1000.0:
                div = 1000.0
            else:
                stats["out_of_page"] += 1
                continue
            x0, y0, x1, y1 = x0 / div, y0 / div, x1 / div, y1 / div
        if x1 <= x0 or y1 <= y0:
            stats["degenerate"] += 1
            continue
        if min(x0, y0) < -0.02 or max(x1, y1) > 1.02:
            stats["out_of_page"] += 1
            continue
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, x1), min(1.0, y1)
        if x1 <= x0 or y1 <= y0:
            stats["degenerate"] += 1
            continue

        w, h = x1 - x0, y1 - y0
        area = w * h
        # THE PAGE-RASTER BACKSTOP — never trust the model not to return the page.
        if area >= raster_ceiling:
            stats["page_raster"] += 1
            continue
        if area < _MIN_AREA_FRACTION or w < _MIN_SIDE_FRACTION or h < _MIN_SIDE_FRACTION:
            stats["too_small"] += 1
            continue
        aspect = w / h if h > 0 else 0.0
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            stats["bad_aspect"] += 1
            continue
        # THE TEXT-COLUMN GUARD (arm 1) — a crop of a prose COLUMN is a 1.4.5
        # regression.
        if _text_coverage((x0, y0, x1, y1), text_boxes_norm) >= _MAX_TEXT_COVERAGE:
            stats["text_column"] += 1
            continue
        # THE TEXT-DENSITY GUARD (arm 2) — a crop carrying a PARAGRAPH of readable
        # words is text rendered as pixels, whatever its area coverage. This is the
        # arm that catches a procedure TABLE, which the coverage arm provably
        # cannot (its coverage measures BELOW a genuine figure's).
        if max_words > 0 and text_items:
            if _text_word_count((x0, y0, x1, y1), text_items) > max_words:
                stats["text_dense"] += 1
                continue
        box = (x0, y0, x1, y1)
        if any(_iou(box, prev) >= _DEDUP_IOU for prev, _ in kept):
            stats["duplicate"] += 1
            continue
        if len(kept) >= max_per_page:
            stats["over_cap"] += 1
            continue
        kept.append((box, prop))

    accepted: list[dict] = []
    for i, (box, prop) in enumerate(kept):
        x0, y0, x1, y1 = box
        # -> PDF-POINT space, TOP-LEFT origin: the ONE space an ImageCandidate
        # bbox lives in (see structure_graph.is_page_raster_candidate) and the
        # space Stage 5c's PIL crop reads off the synthetic image FB.
        px0, py0 = x0 * page_w, y0 * page_h
        px1, py1 = x1 * page_w, y1 * page_h
        # Nominal crop pixels at the Stage-5c 144-DPI render, so the extraction's
        # spacer/hairline filter (_is_spacer_image) sees a sane px_size.
        scale = 144.0 / 72.0
        accepted.append(
            {
                "bbox": [px0, py0, px1, py1],
                "px_size": [int((px1 - px0) * scale), int((py1 - py0) * scale)],
                "index": i,
                "vlm_figure": True,
                "vlm_kind": str(prop.get("kind") or "figure"),
                "vlm_confidence": prop.get("confidence"),
            }
        )
    stats["accepted"] = len(accepted)
    return accepted, stats


# ---------------------------------------------------------------------------
# Per-page text boxes (input to the text-column guard).
# ---------------------------------------------------------------------------
def page_text_items_norm(
    page: dict,
) -> "list[tuple[tuple[float, float, float, float], str]]":
    """Normalized [0,1] (bbox, text) items for a page's extracted TEXT blocks.

    Coordinate-space contract (the SAME trap ``is_page_raster_candidate``
    documents): on the OCR/scan lane the merged text blocks carry IMAGE-PIXEL
    bboxes (e.g. 1224x1584 at render scale 2.0) while the page's ``width`` /
    ``height`` are PDF POINTS (612x792). Normalizing by the WRONG dims would put
    every text box at ~25% of the page and silently defeat the text-column guard
    (letting body-text crops through — the exact WCAG 1.4.5 regression this guard
    exists to stop).

    So we normalize by the PIXEL dims when ``tesseract_width`` is present (the
    OCR lane), deriving the pixel height from the uniform render scale, and by
    the point dims otherwise (the born-digital lane).
    """
    pt_w = float(page.get("width") or 0.0)
    pt_h = float(page.get("height") or 0.0)
    if pt_w <= 0 or pt_h <= 0:
        return []
    tess_w = page.get("tesseract_width")
    norm_w, norm_h = pt_w, pt_h
    if tess_w:
        try:
            cand_w = float(tess_w)
            if cand_w > 0:
                norm_w = cand_w
                norm_h = pt_h * (cand_w / pt_w)  # uniform render scale
        except (TypeError, ValueError, ZeroDivisionError):
            norm_w, norm_h = pt_w, pt_h
    if norm_w <= 0 or norm_h <= 0:
        return []

    out: list[tuple[tuple[float, float, float, float], str]] = []
    blocks = ((page.get("merged") or {}).get("text_blocks")) or []
    for blk in blocks:
        bbox = (blk or {}).get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(
            (
                (
                    max(0.0, min(1.0, x0 / norm_w)),
                    max(0.0, min(1.0, y0 / norm_h)),
                    max(0.0, min(1.0, x1 / norm_w)),
                    max(0.0, min(1.0, y1 / norm_h)),
                ),
                str((blk or {}).get("text") or ""),
            )
        )
    return out


def page_text_boxes_norm(page: dict) -> "list[tuple[float, float, float, float]]":
    """Just the normalized text BOXES (the coverage arm's signal)."""
    return [bb for bb, _t in page_text_items_norm(page)]


# ---------------------------------------------------------------------------
# Resume sidecar + stop seam (mirrors page_arranger).
# ---------------------------------------------------------------------------
def _cache_root() -> Path:
    from . import paths as _paths

    return _paths.resolve_cache(_CACHE_BASENAME)


def _cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def _page_fingerprint(image_b64: str, *, model: str, page_label: int) -> str:
    """Content-address one page DETECT unit -> sha256 hex.

    Keys on the page IMAGE bytes (the actual model input), the model id, the
    prompt-contract version, the system-prompt sha, max_tokens and temperature —
    so a prompt / model / render change never serves a stale verdict. The accept
    gate is deliberately NOT in the key: it is deterministic and re-runs on every
    load, so a threshold change re-decides the CACHED proposals with no re-POST
    (the raw proposals are what is cached, not the verdict).
    """
    sys_sha = hashlib.sha256(DETECT_SYSTEM.encode("utf-8")).hexdigest()[:16]
    img_sha = hashlib.sha256((image_b64 or "").encode("utf-8")).hexdigest()
    # The thinking mode is in the key: it flips the verdict wholesale (thinking-off
    # returns [] on every page), so a cross-mode cache HIT would be a wrong verdict.
    nothink = int(resolve_detect_disable_thinking())
    raw = (
        f"{img_sha}␟{model}␟{DETECT_PROMPT_VERSION}␟{sys_sha}"
        f"␟{_DETECT_MAX_TOKENS}␟temp:0.0␟nothink:{nothink}␟page:{page_label}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(path: Path) -> "Optional[list]":
    try:
        if not path.exists():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt sidecar is a miss, never a crash
        return None
    props = doc.get("proposals") if isinstance(doc, dict) else None
    return props if isinstance(props, list) else None


def _cache_put(path: Path, proposals: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"proposals": proposals}, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 — cache infra never breaks the pass
        logger.debug("vlm-figure-detect cache write failed (non-fatal): %s", exc)


def _stop_requested() -> bool:
    try:
        from .stop_seam import stop_sentinel_present

        return bool(stop_sentinel_present())
    except Exception:  # noqa: BLE001 — a probe error is never a stop
        return False


def _check_stop(site_id: str) -> None:
    from .stop_seam import check_cascade_stop

    check_cascade_stop(site_id)


@dataclass
class _DetectPage:
    page_label: int
    image_b64: str
    page_w: float
    page_h: float
    text_items: "list[tuple[tuple[float, float, float, float], str]]"
    page_ref: dict


def _fan_out_detect(
    seat,
    pages: "list[_DetectPage]",
    *,
    log: Callable[[str], None],
    requests_module=None,
) -> "dict[int, list[dict]]":
    """Fan the per-page DETECT POSTs out concurrently — resume-cached, stop-aware.

    Mirrors ``page_arranger._fan_out_arrange`` exactly: submit-one-consume-one,
    the stop sentinel probed (non-raising) before EACH new submission; on a stop
    the fan-out stops submitting, drains in-flight (each persists its sidecar),
    then raises — worst-case loss <= concurrency in-flight page detections.

    Returns ``{page_label: raw_proposals}`` (UNFILTERED; the gate runs later).
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if not pages:
        return {}
    use_cache = resolve_detect_checkpoint()
    max_workers = max(1, min(resolve_detect_concurrency(), len(pages)))
    model = str(getattr(seat, "model", "") or "")

    def _work(dp: _DetectPage) -> list[dict]:
        cache_path: Optional[Path] = None
        if use_cache:
            try:
                key = _page_fingerprint(dp.image_b64, model=model, page_label=dp.page_label)
                cache_path = _cache_path(key, _cache_root())
                cached = _cache_get(cache_path)
                if cached is not None:
                    return cached
            except Exception as exc:  # noqa: BLE001 — cache infra never breaks detect
                logger.debug("vlm-figure-detect cache lookup failed (non-fatal): %s", exc)
                cache_path = None
        try:
            proposals, _usage, _wall = detect_page_figures(
                seat, dp.image_b64, dp.page_label, requests_module=requests_module
            )
        except DetectNonJudgment as exc:
            # NOT a verdict — degrade this page alone and DO NOT cache, so a
            # --resume re-POSTs it instead of serving a transient blip forever as
            # "no figures on this page".
            log(f"[cascade] vlm-figure-detect page {dp.page_label} no verdict: {exc}")
            return []
        except Exception as exc:  # noqa: BLE001 — per-page fail-soft
            log(f"[cascade] vlm-figure-detect page {dp.page_label} raised fail-soft: {exc}")
            return []
        if use_cache and cache_path is not None:
            _cache_put(cache_path, proposals)  # only a GENUINE verdict is persisted
        return proposals

    results: dict[int, list[dict]] = {}
    next_idx = 0
    n = len(pages)
    stop_requested = False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        in_flight: dict[Any, _DetectPage] = {}
        while next_idx < n and len(in_flight) < max_workers:
            if _stop_requested():
                stop_requested = True
                break
            dp = pages[next_idx]
            next_idx += 1
            in_flight[pool.submit(_work, dp)] = dp
        while in_flight:
            done, _pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                dp = in_flight.pop(future)
                try:
                    results[dp.page_label] = future.result()
                except BaseException as exc:  # noqa: BLE001 — belt: never propagate
                    log(
                        f"[cascade] vlm-figure-detect future raised fail-soft "
                        f"(page {dp.page_label}): {exc}"
                    )
                    results[dp.page_label] = []
            while not stop_requested and next_idx < n and len(in_flight) < max_workers:
                if _stop_requested():
                    stop_requested = True
                    break
                dp = pages[next_idx]
                next_idx += 1
                in_flight[pool.submit(_work, dp)] = dp

    if stop_requested:
        log(
            "[cascade] vlm-figure-detect: stop sentinel observed mid-fan-out — "
            "in-flight pages drained + checkpointed; pausing."
        )
        _check_stop("cascade:vlm-figure-detect-page")
    return results


# ---------------------------------------------------------------------------
# DecisionCapture (REQUIRED — this is a NEW LLM call site).
# ---------------------------------------------------------------------------
def _detect_course_code() -> str:
    """Course code for the capture — the ``vlm_extract`` / ``figure_captioner``
    resolution chain, so the event is attributable to a real course."""
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _emit_detect_capture(page_rows: "list[dict]", totals: dict, model: str) -> None:
    """One ``structure_detection`` DecisionCapture per converted document.

    Rides the EXISTING ``structure_detection`` enum (the page-arranger precedent)
    -> no ``decision_event`` schema change. Rationale is DYNAMIC and replayable:
    per-page proposal / accept counts, every guard-rejection tally (page-raster,
    text-column, degenerate, ...), the accepted bbox page-area fractions, the
    busiest pages, and the model id — so a post-hoc replay can attribute each
    accepted crop to its exact page + gate decision.

    Best-effort: a capture failure NEVER breaks conversion (the arranger /
    figure-captioner posture).
    """
    try:
        from lib.decision_capture import DecisionCapture
    except Exception as exc:  # noqa: BLE001 — the cross-venv bridge has no lib/
        logger.debug("vlm-figure-detect: DecisionCapture unavailable (non-fatal): %s", exc)
        return
    try:
        n_pages = len(page_rows)
        pages_with_figs = sum(1 for r in page_rows if r.get("accepted"))
        areas = [a for r in page_rows for a in (r.get("areas") or ())]
        top = sorted(
            (r for r in page_rows if r.get("accepted")),
            key=lambda r: -int(r.get("accepted", 0)),
        )[:5]
        rejected = sum(
            int(totals.get(k, 0))
            for k in (
                "page_raster", "text_column", "text_dense", "degenerate", "too_small",
                "bad_aspect", "out_of_page", "malformed", "duplicate", "over_cap",
            )
        )
        cap = DecisionCapture(
            course_code=_detect_course_code(),
            # ``dart-conversion`` (NOT ``semantik_conversion``) so this capture
            # lands in the SAME phase dir as every sibling SemantiK capture from
            # the same conversion — figure_captioner / vlm_extract / reasoning_qc /
            # page_arranger / subclassifier all use it, and it is the phase the
            # ``MIN_DECISIONS_PER_PHASE`` completeness bucket keys on. Renaming it
            # here alone would split one run's capture stream across two dirs.
            phase="dart-conversion",
            tool="dart",
        )
        cap.log_decision(
            decision_type="structure_detection",
            decision=(
                f"vlm_figure_detect: accepted {totals.get('accepted', 0)} figure bbox(es) "
                f"across {pages_with_figs}/{n_pages} pages"
            ),
            rationale=(
                f"VLM figure-detect (model={model}) PROPOSED {totals.get('proposed', 0)} bbox(es) "
                f"over {n_pages} page(s); the deterministic accept gate DECIDED — admitted "
                f"{totals.get('accepted', 0)} on {pages_with_figs} page(s), REJECTED {rejected}: "
                f"page_raster={totals.get('page_raster', 0)} (whole-page returns — the mandatory "
                f"backstop), text_column={totals.get('text_column', 0)} (prose crops that would be "
                f"a WCAG 1.4.5 images-of-text regression), text_dense={totals.get('text_dense', 0)} "
                f"(crops carrying a paragraph of readable words — a procedure table, not a "
                f"figure), too_small={totals.get('too_small', 0)}, "
                f"bad_aspect={totals.get('bad_aspect', 0)}, degenerate={totals.get('degenerate', 0)}, "
                f"out_of_page={totals.get('out_of_page', 0)}, malformed={totals.get('malformed', 0)}, "
                f"duplicate={totals.get('duplicate', 0)}, over_cap={totals.get('over_cap', 0)}. "
                f"Accepted page-area fractions={[round(a, 4) for a in areas[:24]]}; "
                f"busiest pages={[(r['page'], r['accepted']) for r in top]}."
            ),
            alternatives_considered=[
                "object extraction (impossible here: every image page-object is the full-page raster)",
                "accept every proposed bbox (rejected: would ship page rasters + body-text crops)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("vlm-figure-detect: decision capture failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# The one public seam the cascade calls.
# ---------------------------------------------------------------------------
def inject_figure_candidates(
    shared: dict,
    pdf_path: Any,
    *,
    seat=None,
    log: "Optional[Callable[[str], None]]" = None,
    requests_module=None,
    render_page_image=None,
) -> "Optional[dict]":
    """Detect figures on every page and INJECT them into ``shared``. Returns audit.

    Mutates ``shared['pages'][i]['images']`` **in memory only** — the extract disk
    cache was written before this runs and is never touched, so there is no
    stale-cache hazard; the per-page resume sidecars here carry the resume story.

    APPENDS to the existing ``images`` list rather than replacing it: on a
    page-raster scan that list already holds the full-page raster, and the
    mandatory downstream page-raster guard drops it exactly as it does today. The
    guard stays the backstop; we only add the sub-page boxes it was waiting for.

    Returns ``None`` when the flag is off (no POST, ``shared`` untouched ->
    byte-identical). Fail-soft throughout: any error degrades to zero injected
    figures rather than breaking the conversion.
    """
    if not resolve_vlm_figure_detect_mode():
        return None
    _log = log or (lambda _m: None)

    if seat is None:
        from .page_arranger import resolve_arranger_seat

        seat = resolve_arranger_seat()
    if render_page_image is None:
        from .page_arranger import _render_page_image_b64

        render_page_image = _render_page_image_b64

    pages = shared.get("pages") or []
    detect_pages: list[_DetectPage] = []
    # Sequential pre-pass: render page images (pypdfium2 is NOT thread-safe on one
    # document handle) — mirrors the arranger's own render pre-pass.
    for page in pages:
        try:
            page_num = int(page.get("page_num") or 0)
            page_w = float(page.get("width") or 0.0)
            page_h = float(page.get("height") or 0.0)
        except (TypeError, ValueError):
            continue
        if page_num <= 0 or page_w <= 0 or page_h <= 0:
            continue
        try:
            img_b64 = render_page_image(Path(pdf_path), page_num)
        except Exception as exc:  # noqa: BLE001 — a page that will not render has no figures
            _log(f"[cascade] vlm-figure-detect page {page_num} render failed: {exc}")
            continue
        detect_pages.append(
            _DetectPage(
                page_label=page_num,
                image_b64=img_b64,
                page_w=page_w,
                page_h=page_h,
                text_items=page_text_items_norm(page),
                page_ref=page,
            )
        )

    raw_by_page = _fan_out_detect(
        seat, detect_pages, log=_log, requests_module=requests_module
    )

    totals = {
        k: 0
        for k in (
            "proposed", "accepted", "malformed", "degenerate", "out_of_page",
            "too_small", "bad_aspect", "page_raster", "text_column", "text_dense",
            "duplicate", "over_cap",
        )
    }
    page_rows: list[dict] = []
    for dp in detect_pages:
        proposals = raw_by_page.get(dp.page_label) or []
        accepted, stats = accept_figure_boxes(
            proposals,
            page_w=dp.page_w,
            page_h=dp.page_h,
            text_items_norm=dp.text_items,
        )
        for key in totals:
            totals[key] += int(stats.get(key, 0))
        if accepted:
            existing = list(dp.page_ref.get("images") or [])
            base = len(existing)
            for i, img in enumerate(accepted):
                img["index"] = base + i
            dp.page_ref["images"] = existing + accepted
        page_rows.append(
            {
                "page": dp.page_label,
                "proposed": stats["proposed"],
                "accepted": stats["accepted"],
                "areas": [
                    round(
                        ((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
                        / (dp.page_w * dp.page_h),
                        4,
                    )
                    for b in accepted
                ],
                "rejected": {
                    k: v
                    for k, v in stats.items()
                    if k not in ("proposed", "accepted") and v
                },
            }
        )

    model = str(getattr(seat, "model", "") or "")
    _emit_detect_capture(page_rows, totals, model)
    _log(
        f"[cascade] vlm-figure-detect: {totals['accepted']} figure bbox(es) accepted "
        f"from {totals['proposed']} proposed over {len(detect_pages)} page(s) "
        f"(rejected: page_raster={totals['page_raster']} "
        f"text_column={totals['text_column']} text_dense={totals['text_dense']} "
        f"too_small={totals['too_small']} "
        f"bad_aspect={totals['bad_aspect']})"
    )
    return {
        "pass": "vlm_figure_detect",
        "prompt_version": DETECT_PROMPT_VERSION,
        "seat_model": model,
        "pages": len(detect_pages),
        "totals": totals,
        "page_rows": page_rows,
    }
