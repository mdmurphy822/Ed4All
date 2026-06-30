"""EXACT render-time element->bbox capture for the structure-data build.

PROTOTYPE (not yet wired into ``build_structure_data.py``). This module is the
proposed replacement for the fuzzy text aligner (``data/structure_align.py`` /
the greedy loop in ``build_structure_data.process_pair``) on the RENDERED-pair
track: because SemantiK renders gold HTML *it controls* to PDF, the gold role
of every extracted PDF block can be recovered EXACTLY by geometry instead of
fuzzily by text overlap.

Mechanism
---------
1. Render the gold HTML to PDF via Playwright **once**, in a layout that
   matches the PDF's print layout:
   * ``page.emulate_media(media="print")`` BEFORE measuring (the screen layout
     at the default 1280px viewport reflows differently — measured: a body
     ``max-width`` centers at 304px on screen but 72px in print);
   * viewport width pinned to the print page content width
     (``PAGE_W_PX = 816`` == 8.5in * 96dpi == 612pt at margin 0);
   * the PDF is emitted as ONE tall page (``height = scrollHeight``) so there
     is NO pagination to reverse — the CSS-px -> PDF-pt map is a single global
     affine ``pt = px * scale`` with ``scale`` derived empirically (== 0.75 =
     72/96 on every doc measured, but we read it from the rendered page width
     rather than assume it).
2. Before rendering the PDF, stamp every role-bearing element with a unique
   ``data-rc-id`` and capture its ``getBoundingClientRect`` (CSS px, document
   coords). ``querySelectorAll`` order == BeautifulSoup ``find_all`` order
   (both document order), so the captured set lines up with the gold-side
   extraction.
3. The labeler (:func:`label_blocks_by_overlap`) assigns each extracted PDF
   block the role of the element whose mapped (pt) box CONTAINS the block
   centroid; ties broken by smallest element area (deepest/tightest element);
   a block contained by no element falls back to max-overlap, else ``None``
   (reported as unmatched). Because every line-fragment of a paragraph falls
   inside that paragraph's element box, fragmentation is handled by
   construction — all fragments inherit the correct role regardless of 2-col
   reflow.

The gold-side role + secondary-head labels (``is_heading`` / ``table_region``
/ ``is_image_block`` / ``list_nesting`` / ``pedagogical_role``) are derived by
REUSING ``build_structure_data``'s ``TAG_TO_ROLE`` / ``pedagogical_role_for`` /
``_list_nesting_depth`` / ``_has_table_ancestor`` / ``_has_image_block_ancestor``
/ ``_li_role_override`` — the SAME logic the fuzzy gold path uses, so the emit
label space is identical (just the *assignment* is exact, not fuzzy).

Opt-in flag (prototype): :func:`exact_render_labels_enabled` reads
``SEMANTIK_EXACT_RENDER_LABELS`` (default OFF). Full build wiring is a
DELIBERATE follow-up after this prototype is validated.

Pure-ish: stdlib + Playwright + BeautifulSoup (via build_structure_data). No
torch, no model loads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# Reuse the SINGLE-source-of-truth gold-label logic so the exact path's label
# space can never drift from the fuzzy path's.
from data.build_structure_data import (
    PEDAGOGICAL_ROLE_TO_ID,
    ROLE_NAMES,
    ROLE_TO_ID,
    TAG_TO_ROLE,
    _block_in_any_table,
    _has_image_block_ancestor,
    _has_table_ancestor,
    _li_role_override,
    _list_nesting_depth,
    compute_span_layout_features,
    pedagogical_role_for,
)

__all__ = [
    "PAGE_W_PX",
    "RenderCapture",
    "RenderCaptureSession",
    "ElementBox",
    "exact_render_labels_enabled",
    "render_with_boxes",
    "label_blocks_by_overlap",
]

# 8.5in * 96 CSS dpi == the Letter content width in px at margin 0. At 72pt dpi
# this is 612pt, so the px->pt scale is 72/96 = 0.75 (derived empirically per
# doc, not hardcoded into the transform).
PAGE_W_PX = 816

# Chromium caps a single PDF page at ~200in (14400px). A doc taller than this
# cannot ship as one page; the prototype renders it anyway and flags it (the
# follow-up is per-page mapping). Generous headroom kept as a guard.
_MAX_SINGLE_PAGE_PX = 14000

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The role-bearing tag set captured at render time. Superset of
# ``TAG_TO_ROLE`` keys plus the structural containers ``table`` / ``figure``
# (captured for diagnostics / future use; only ``TAG_TO_ROLE`` tags become
# gold records). ``label`` / ``legend`` ARE captured — they carry the
# extractable form-field text and map to the in-vocab ``form_label`` role (the
# arxiv-only prototype omitted them, which left rendered PDF forms ~98%
# unmatched). ``input`` / ``select`` / ``textarea`` are dropped — a rendered
# blank control carries no extractable PDF text (its visible text is its
# ``label``).
_CAPTURE_TAGS = (
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "td", "th", "caption", "figcaption",
    "blockquote", "pre", "code", "dt", "dd", "label", "legend",
    "table", "figure",
)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class ElementBox:
    """One captured role-bearing element: its rc-id, tag, CSS-px box, text."""

    rc_id: int
    tag: str
    x: float
    y: float
    w: float
    h: float
    text: str

    @property
    def area_px(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class RenderCapture:
    """Result of :func:`render_with_boxes`.

    ``pdf_path`` lives under ``tmpdir`` (caller owns cleanup). ``attributed_html``
    is the post-load DOM serialization carrying the injected ``data-rc-id``
    attributes (parsed by the labeler to recover gold roles). ``scale`` is the
    px->pt factor read from the rendered PDF page width.
    """

    pdf_path: Path
    tmpdir: Path
    page_w_px: int
    scroll_height_px: float
    elements: list[ElementBox]
    attributed_html: str
    overflowed_single_page: bool = False


@dataclass
class LabelResult:
    """Per-block exact labels + an audit summary."""

    # one dict per extracted PDF block (document order), each:
    #   {text, labels:{...6 heads...}, html_tag, rc_id, role, match:{kind,overlap,elem_area}}
    rows: list[dict] = field(default_factory=list)
    n_blocks: int = 0
    n_matched: int = 0
    n_contained: int = 0
    n_overlap_only: int = 0
    n_unmatched: int = 0

    @property
    def unmatched_rate(self) -> float:
        return self.n_unmatched / self.n_blocks if self.n_blocks else 0.0


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------


def exact_render_labels_enabled() -> bool:
    """Resolve ``SEMANTIK_EXACT_RENDER_LABELS`` (default OFF). Parse-with-fallback
    (mirrors the SEMANTIK_* posture): truthy token -> on; anything else -> off.

    The full build wiring is a deliberate follow-up; for the prototype this
    only governs whether a caller routes through the exact labeler.
    """
    raw = (os.environ.get("SEMANTIK_EXACT_RENDER_LABELS") or "").strip().lower()
    return raw in _TRUTHY


# ---------------------------------------------------------------------------
# Render + capture
# ---------------------------------------------------------------------------


# JS run in the page: stamp every capture-tag element with a sequential
# data-rc-id (querySelectorAll == document order == soup.find_all order) and
# return its getBoundingClientRect in document coords (px).
_CAPTURE_JS = """
(tags) => {
  const els = document.querySelectorAll(tags.join(','));
  const out = [];
  let i = 0;
  for (const e of els) {
    e.setAttribute('data-rc-id', String(i));
    const r = e.getBoundingClientRect();
    out.push({
      rc_id: i,
      tag: e.tagName.toLowerCase(),
      x: r.x + window.scrollX,
      y: r.y + window.scrollY,
      w: r.width,
      h: r.height,
      text: (e.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 80),
    });
    i++;
  }
  return out;
}
"""


def _capture_on_page(page, html: str, *, page_w_px: int, pdf_path: Path) -> RenderCapture:
    """Render ``html`` to a single tall PDF page on an already-open Playwright
    ``page`` and capture per-element boxes. Shared by the single-shot and the
    batched/borrowed-context paths so the capture mechanics never drift."""
    # Print media BEFORE set_content/measure so getBoundingClientRect reflects
    # the SAME layout page.pdf() will rasterize.
    page.emulate_media(media="print")
    page.set_content(html, wait_until="networkidle")

    scroll_h = float(page.evaluate("() => document.documentElement.scrollHeight"))
    raw_boxes = page.evaluate(_CAPTURE_JS, list(_CAPTURE_TAGS))
    attributed_html = page.content()

    overflow = scroll_h > _MAX_SINGLE_PAGE_PX
    render_h = min(scroll_h, float(_MAX_SINGLE_PAGE_PX))

    page.pdf(
        path=str(pdf_path),
        width=f"{page_w_px}px",
        height=f"{render_h}px",
        print_background=True,
        margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        scale=1,
    )

    elements = [
        ElementBox(
            rc_id=int(b["rc_id"]),
            tag=str(b["tag"]),
            x=float(b["x"]),
            y=float(b["y"]),
            w=float(b["w"]),
            h=float(b["h"]),
            text=str(b.get("text") or ""),
        )
        for b in raw_boxes
    ]

    return RenderCapture(
        pdf_path=pdf_path,
        tmpdir=pdf_path.parent,
        page_w_px=page_w_px,
        scroll_height_px=scroll_h,
        elements=elements,
        attributed_html=attributed_html,
        overflowed_single_page=overflow,
    )


def render_with_boxes(
    html: str,
    *,
    page_w_px: int = PAGE_W_PX,
    context=None,
    tmpdir: Path | None = None,
) -> RenderCapture:
    """Render ``html`` to a single-tall-page PDF and capture per-element boxes.

    Two launch modes:

    * **Single-shot (default; ``context`` is None).** Spins up its own
      Playwright Chromium — byte-identical to the prototype. The caller owns
      ``RenderCapture.tmpdir`` cleanup.
    * **Borrowed context (``context`` is a Playwright ``BrowserContext``).**
      Reuses an already-open context so a batch build (or a caller that already
      holds a sync Playwright context — sync Playwright cannot be nested) does
      NOT relaunch Chromium per call. The page's viewport is pinned to
      ``page_w_px`` (the borrowed context's default viewport may differ), so the
      CSS-px -> PDF-pt affine is identical to the single-shot path.

    ``tmpdir`` (when given) is reused for ``rendered.pdf`` so a caller's own
    ``TemporaryDirectory`` owns cleanup; otherwise a fresh ``rc_*`` dir is made.
    """
    import tempfile

    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="rc_"))
    else:
        tmpdir = Path(tmpdir)
    pdf_path = tmpdir / "rendered.pdf"

    if context is not None:
        page = context.new_page()
        try:
            page.set_viewport_size({"width": page_w_px, "height": 1056})
            return _capture_on_page(page, html, page_w_px=page_w_px, pdf_path=pdf_path)
        finally:
            page.close()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            ctx = browser.new_context(viewport={"width": page_w_px, "height": 1056})
            page = ctx.new_page()
            try:
                return _capture_on_page(page, html, page_w_px=page_w_px, pdf_path=pdf_path)
            finally:
                ctx.close()
        finally:
            browser.close()


class RenderCaptureSession:
    """Reusable Chromium browser + context for BATCHED box-capture renders.

    Mirrors :class:`dart_semantik.validate.HtmlValidator`'s launch pattern (one
    Chromium for the whole build) so :func:`render_with_boxes` is not
    relaunching a browser per pair — the deliverable-2 batch refactor. Usage::

        with RenderCaptureSession() as session:
            for html in docs:
                capture = session.render(html, tmpdir=tmp)
    """

    def __init__(self, *, page_w_px: int = PAGE_W_PX):
        self.page_w_px = page_w_px
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self) -> "RenderCaptureSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._context = self._browser.new_context(
            viewport={"width": self.page_w_px, "height": 1056}
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def render(self, html: str, *, tmpdir: Path | None = None,
               page_w_px: int | None = None) -> RenderCapture:
        return render_with_boxes(
            html,
            page_w_px=page_w_px or self.page_w_px,
            context=self._context,
            tmpdir=tmpdir,
        )


# ---------------------------------------------------------------------------
# Gold-record recovery from the attributed DOM (reuses build_structure_data)
# ---------------------------------------------------------------------------


def _gold_records_with_rcid(attributed_html: str) -> list[dict]:
    """Mirror ``build_structure_data.extract_html_blocks`` but ALSO carry the
    injected ``data-rc-id`` so each gold record can be joined to its captured
    box. Reuses TAG_TO_ROLE / the nesting + override rules verbatim, so the
    role + secondary-head label space is identical to the fuzzy gold path."""
    soup = BeautifulSoup(attributed_html, "lxml")
    out: list[dict] = []
    for el in soup.find_all(list(TAG_TO_ROLE.keys())):
        if el.name not in _OUTER_ALLOW:
            has_outer = any(
                p.name in TAG_TO_ROLE and p is not el
                for p in el.parents
                if p.name is not None
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
        rc_raw = el.get("data-rc-id")
        rc_id = int(rc_raw) if (rc_raw is not None and str(rc_raw).isdigit()) else None
        out.append(
            {
                "rc_id": rc_id,
                "text": text,
                "tag": el.name,
                "role": role,
                "list_nesting": _list_nesting_depth(el),
                "in_table_html": _has_table_ancestor(el),
                "in_image_block_html": _has_image_block_ancestor(el),
            }
        )
    return out


# build_structure_data.OUTER_TAG_ALLOW_NESTED, imported lazily to avoid a hard
# name dependency if the constant is renamed.
from data.build_structure_data import OUTER_TAG_ALLOW_NESTED as _OUTER_ALLOW  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _contains(elem_pt: tuple[float, float, float, float], cx: float, cy: float,
              *, pad: float = 1.0) -> bool:
    x0, y0, x1, y1 = elem_pt
    return (x0 - pad) <= cx <= (x1 + pad) and (y0 - pad) <= cy <= (y1 + pad)


def _overlap_frac(block: list[float], elem: tuple[float, float, float, float]) -> float:
    """Fraction of the BLOCK area covered by the element box (block as the
    smaller object — line fragments are tiny vs the paragraph box)."""
    bx0, by0, bx1, by1 = block
    ex0, ey0, ex1, ey1 = elem
    ix0, iy0 = max(bx0, ex0), max(by0, ey0)
    ix1, iy1 = min(bx1, ex1), min(by1, ey1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    barea = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / barea


# ---------------------------------------------------------------------------
# The exact labeler
# ---------------------------------------------------------------------------


def label_blocks_by_overlap(
    shared: dict,
    capture: RenderCapture,
    *,
    min_overlap: float = 0.30,
) -> LabelResult:
    """Assign each extracted PDF block its gold role by bbox containment.

    ``shared`` is the ``extract_shared`` output for ``capture.pdf_path``
    (single page under the single-tall-page render). For every extracted block:

    1. map its centroid to compare against element boxes (both in PDF pt);
    2. among gold records whose mapped element box CONTAINS the centroid, take
       the SMALLEST-area one (deepest/tightest element) -> exact role;
    3. else take the gold record of max block-overlap (>= ``min_overlap``);
    4. else the block is UNMATCHED (no role; reported, not silently dropped).

    Returns a :class:`LabelResult` with one row per block + an audit summary.
    The px->pt ``scale`` is read from the rendered page width (== 0.75 in
    practice) rather than assumed.
    """
    pages = shared.get("pages", [])
    # scale derived from the rendered single page's width.
    page_w_pt = float(pages[0]["width"]) if pages else (capture.page_w_px * 0.75)
    scale = page_w_pt / float(capture.page_w_px)

    gold = _gold_records_with_rcid(capture.attributed_html)
    box_by_rcid = {e.rc_id: e for e in capture.elements}

    # Build the candidate list: (gold_record, mapped_pt_box, area_pt). Skip a
    # gold record with no captured box (display:none / zero-size) or a
    # degenerate (0-area) box.
    cands: list[tuple[dict, tuple[float, float, float, float], float]] = []
    for rec in gold:
        eb = box_by_rcid.get(rec["rc_id"])
        if eb is None or eb.w <= 0 or eb.h <= 0:
            continue
        pt = (eb.x * scale, eb.y * scale, (eb.x + eb.w) * scale, (eb.y + eb.h) * scale)
        cands.append((rec, pt, (pt[2] - pt[0]) * (pt[3] - pt[1])))

    res = LabelResult()
    for page in pages:
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue
        # Per-page normalization medians + page dims — mirrors
        # build_structure_data.process_pair so the 20-dim layout vector is
        # computed by the SINGLE source of truth (compute_span_layout_features).
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

        for blk in merged:
            text = (blk.get("text") or "").strip()
            if not text:
                continue
            res.n_blocks += 1
            in_table = _block_in_any_table(blk, page)
            layout_vec = compute_span_layout_features(
                blk,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )
            bbox = [float(c) for c in (blk.get("bbox") or [0, 0, 0, 0])[:4]]
            cx, cy = _centroid(bbox)

            # 1. containment, smallest area wins
            best_rec = None
            best_area = None
            best_kind = None
            best_overlap = 0.0
            for rec, pt, area in cands:
                if _contains(pt, cx, cy):
                    if best_area is None or area < best_area:
                        best_rec, best_area, best_kind = rec, area, "contained"

            # 2. fallback: max block-overlap
            if best_rec is None:
                for rec, pt, area in cands:
                    ov = _overlap_frac(bbox, pt)
                    if ov > best_overlap:
                        best_overlap, best_rec, best_area = ov, rec, area
                if best_rec is not None and best_overlap >= min_overlap:
                    best_kind = "overlap"
                else:
                    best_rec = None

            if best_rec is None:
                res.n_unmatched += 1
                res.rows.append(
                    {
                        "text": text,
                        "layout": layout_vec,
                        "labels": None,
                        "html_tag": None,
                        "rc_id": None,
                        "role": None,
                        "match": {"kind": "unmatched", "overlap": round(best_overlap, 4)},
                        "bbox": bbox,
                    }
                )
                continue

            role = best_rec["role"]
            role_name = getattr(role, "value", role)
            # Only emit a structural_role id for roles in the active head vocab
            # (mirrors the fuzzy path's ROLE_TO_ID filter); others keep role
            # text but no structural_role id.
            sr_id = ROLE_TO_ID.get(role) if role in ROLE_TO_ID else None
            tag = best_rec["tag"]
            labels = {
                "structural_role": sr_id,
                "is_heading": 1 if (tag.startswith("h") and tag[1:].isdigit()) else 0,
                # table_region: PDF-side pdfplumber detection OR gold <table>
                # ancestor — mirrors process_pair / _enrich_v3_row exactly.
                "table_region": 1 if (in_table or best_rec.get("in_table_html")) else 0,
                "is_image_block": 1 if best_rec.get("in_image_block_html") else 0,
                "list_nesting": int(best_rec.get("list_nesting", 0)),
                "pedagogical_role": PEDAGOGICAL_ROLE_TO_ID[pedagogical_role_for(text)],
            }
            res.n_matched += 1
            if best_kind == "contained":
                res.n_contained += 1
            else:
                res.n_overlap_only += 1
            res.rows.append(
                {
                    "text": text,
                    "layout": layout_vec,
                    "labels": labels,
                    "html_tag": tag,
                    "rc_id": best_rec["rc_id"],
                    "role": str(role_name),
                    "match": {
                        "kind": best_kind,
                        "overlap": round(best_overlap, 4),
                        "elem_area": round(best_area or 0.0, 2),
                    },
                    "bbox": bbox,
                }
            )

    return res
