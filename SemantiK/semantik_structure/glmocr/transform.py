"""Deterministic GLM-OCR layout → SemantiK ``region_provenance`` transform.

NO LLM. Ports the validated rules from ``scratchpad/glmocr/build_html.py`` into
the SemantiK wire contract: a list of ``region_provenance`` dicts (the shape
``lib/semantik/cascade_ir.py::_block_from_provenance`` consumes) + a
``heading_tree`` + a list of Super-escalation records.

Ported rules (all validated 2026-07-16):
  * 25-class ``native_label`` → ``region_kind`` (``region_map``)
  * heading LEVELS: ``doc_title``→1; ``N.M``→2; other title→3 (+ ``pending``)
  * apparatus-lexicon demotion: TRY IT / BE PREPARED / EXAMPLE / HOW TO /
    Solution / Step → paragraph carrying a ``pedagogy_class`` (data-driven)
  * box-title promotion: a short title-case line before prose → heading
  * spatial exercise pairing: ordinal text region ↔ display_formula by
    column/row bands → one ``exercise`` paragraph, row-major reading order
  * ordinal preservation: a literal "413. text" exercise keeps its number
  * table modes: md-pipe / embedded-HTML → ``cell_grid`` (+ ``data_table_mode``)
  * figure regions → ``figure`` with bbox + alt placeholder; a ``figure_title``
    attaches to the nearest preceding figure as extracted ``caption_text``
  * reading order from region ``index``; furniture dropped
  * cross-page continuation stitch (bbox+index heuristic) with escalation

Every emitted region carries a deterministic ``first_raw_block_index`` (a
document-monotonic counter over consumed GLM regions) so the same PDF → the
same ``data-semantik-block-id`` set across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import region_map as rm

# ── Input page shape (injectable; the SDK client produces these). ───────────


@dataclass
class GlmPage:
    """One GLM-OCR page: its 1-based number + the SDK region layout list.

    ``regions`` is the SDK ``json_result`` for the page — a list of dicts each
    with ``index`` (reading order), ``native_label`` (25-class), ``bbox_2d``
    (normalised 0-1000 ``[x0,y0,x1,y1]``), and ``content`` (OCR text / md).
    ``width``/``height`` are the render pixel dims (for the alt-text crop).
    ``error`` marks a page whose SDK OCR failed (fed to the escalations).
    """

    page_no: int
    regions: List[Dict[str, Any]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    image_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TransformResult:
    region_provenance: List[Dict[str, Any]]
    heading_tree: List[Tuple[int, str]]
    escalations: List[Dict[str, Any]]


# ── Ordinal / exercise shape predicates (ported from build_html.py). ────────
_ORD_ONLY_RE = re.compile(r"^(\d{1,4})\.\s*$")
_PROSE_ORD_RE = re.compile(r"^(\d{1,4})\.\s+(\S.*)$", re.S)
# Row-band tolerance in normalised 0-1000 bbox space (build_html used 12px on
# ~2600px pages ≈ 4.6 units; 15 is a conservative TODO(calibration) constant).
_ROW_TOL_NORM = 15


def _content(region: Dict[str, Any]) -> str:
    return str(region.get("content") or "").strip()


def _bbox(region: Dict[str, Any]) -> Optional[List[float]]:
    b = region.get("bbox_2d") or region.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            return [float(x) for x in b]
        except (TypeError, ValueError):
            return None
    return None


def _ends_open(text: str) -> bool:
    """True if a prose block ends WITHOUT terminal punctuation (continuation)."""
    t = text.rstrip()
    return bool(t) and t[-1] not in ".!?:;…)”\"'"


# ── Table parsing: md-pipe / embedded HTML → cell_grid. ─────────────────────
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SEP_ROW_RE = re.compile(r"^\s*:?-{2,}:?\s*$")


def _parse_table(content: str) -> Tuple[Optional[List[List[str]]], str]:
    """Parse a GLM table region → (cell_grid, data_table_mode).

    Returns ``(None, "none")`` when no grid is recoverable (the caller then
    keeps the region as flat text). Anti-fabrication: cells are split from the
    existing text only; a colspan/rowspan HTML table is FLATTENED to a best-
    effort rectangular grid (spans not represented — never invented).
    """
    inner = (content or "").strip()
    if "<table" in inner.lower() or "<tr" in inner.lower():
        rows: List[List[str]] = []
        for row_html in _HTML_ROW_RE.findall(inner):
            cells = [
                _TAG_RE.sub("", c).strip() for c in _HTML_CELL_RE.findall(row_html)
            ]
            if cells:
                rows.append(cells)
        return (rows or None), ("html" if rows else "none")
    # md pipe-table
    lines = [ln for ln in inner.splitlines() if ln.count("|") >= 2]
    if not lines:
        return None, "none"
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and all(_SEP_ROW_RE.match(c or "---") for c in cells):
            continue  # header-separator row — a topology hint, not data
        rows.append(cells)
    return (rows or None), ("md-pipe" if rows else "none")


def _wrap_math(content: str, display: bool) -> str:
    """Ensure a formula carries $-delimiters so SEMANTIK_LATEX_MATHML can lift
    it to MathML; keep an already-delimited span verbatim."""
    c = content.strip()
    if not c:
        return c
    if c.startswith("$") or c.startswith("\\(") or c.startswith("\\["):
        return c
    return f"$$ {c} $$" if display else f"${c}$"


# ── Region-provenance record builders. ──────────────────────────────────────


def _prov(
    *,
    idx: int,
    region_kind: str,
    raw_text: str = "",
    heading_text: Optional[str] = None,
    level: Optional[int] = None,
    page: int,
    role: Optional[str] = None,
    pedagogy_class: Optional[str] = None,
    native_label: str = "",
    bbox: Optional[Sequence[float]] = None,
    caption_text: Optional[str] = None,
    cell_grid: Optional[List[List[str]]] = None,
    data_table_mode: Optional[str] = None,
    heading_level_pending: bool = False,
    figure_alt: Optional[str] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "region_kind": region_kind,
        "raw_text": raw_text,
        "first_raw_block_index": idx,
        "pages": [page] if page else [],
        "confidence": 0.85,
        "wcag_status": "not_evaluated",
        # additive lane-provenance keys (ignored by the adapter, read by the
        # layout sidecar + escalations backbone):
        "native_label": native_label,
        "source_page": page,
    }
    if heading_text is not None:
        d["heading_text"] = heading_text
    if level is not None:
        d["level"] = level
    if role:
        d["role"] = role
    if pedagogy_class:
        d["pedagogy_class"] = pedagogy_class
    if bbox is not None:
        d["bbox"] = list(bbox)
    if caption_text:
        d["caption_text"] = caption_text
    if cell_grid is not None:
        d["cell_grid"] = cell_grid
        d["header_row_indices"] = [0]
    if data_table_mode:
        d["data_table_mode"] = data_table_mode
    if heading_level_pending:
        d["heading_level_pending"] = True
    if figure_alt is not None:
        d["figure_alt"] = figure_alt
    return d


def _emit_structural(
    region: Dict[str, Any],
    *,
    idx: int,
    page: int,
    next_text: str,
    next_kind: str,
    escalations: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Map one non-exercise GLM region → a region_provenance dict (or None to
    drop / defer to caption-attach)."""
    nl = str(region.get("native_label") or "text")
    content = _content(region)
    bbox = _bbox(region)
    base = rm.classify_native_label(nl)

    if base == "metadata_drop":
        return _prov(idx=idx, region_kind="metadata_drop", raw_text=content,
                     page=page, native_label=nl, bbox=bbox)

    if base == "heading":
        text = rm.strip_md_heading(content)
        # apparatus / table-caption demotions off the heading track.
        ped = rm.apparatus_pedagogy_class(text)
        if ped:
            return _prov(idx=idx, region_kind="paragraph", raw_text=text,
                         page=page, role="apparatus", pedagogy_class=ped,
                         native_label=nl, bbox=bbox)
        if rm.is_table_caption(text):
            return _prov(idx=idx, region_kind="paragraph", raw_text=text,
                         page=page, role="caption", native_label=nl, bbox=bbox)
        if nl == "doc_title":
            return _prov(idx=idx, region_kind="heading", heading_text=text,
                         level=1, page=page, native_label=nl, bbox=bbox)
        level, ambiguous = rm.heading_level_for_title(text)
        if ambiguous:
            escalations.append({
                "region_index": idx, "source_page": page, "native_label": nl,
                "reason": "heading_level_pending",
                "detail": "non-numbered title; level defaulted to 3",
                "text": text[:120],
            })
        return _prov(idx=idx, region_kind="heading", heading_text=text,
                     level=level, page=page, native_label=nl, bbox=bbox,
                     heading_level_pending=ambiguous)

    if base == "math":
        display = nl != "inline_formula"
        return _prov(idx=idx, region_kind="math",
                     raw_text=_wrap_math(content, display), page=page,
                     role="math", native_label=nl, bbox=bbox)

    if base == "table":
        grid, mode = _parse_table(content)
        return _prov(idx=idx, region_kind="table", raw_text=content, page=page,
                     native_label=nl, bbox=bbox, cell_grid=grid,
                     data_table_mode=mode)

    if base == "figure":
        return _prov(idx=idx, region_kind="figure", raw_text="", page=page,
                     role="figure", native_label=nl, bbox=bbox, figure_alt=None)

    if base == "caption":
        # handled by the caller (attach to preceding figure); emitted here only
        # when there is no figure to attach to.
        return _prov(idx=idx, region_kind="paragraph", raw_text=content,
                     page=page, role="caption", native_label=nl, bbox=bbox)

    if base == "footnote":
        return _prov(idx=idx, region_kind="paragraph", raw_text=content,
                     page=page, role="footnote", native_label=nl, bbox=bbox)

    if base == "aside":
        role = "reference" if nl == "reference" else "aside"
        return _prov(idx=idx, region_kind="paragraph", raw_text=content,
                     page=page, role=role, native_label=nl, bbox=bbox)

    # base == "paragraph" — apparatus / box-title / prose.
    ped = rm.apparatus_pedagogy_class(content)
    if ped:
        return _prov(idx=idx, region_kind="paragraph", raw_text=content,
                     page=page, role="apparatus", pedagogy_class=ped,
                     native_label=nl, bbox=bbox)
    if rm.is_box_title(content, next_text, next_kind):
        return _prov(idx=idx, region_kind="heading",
                     heading_text=rm.strip_md_heading(content), level=3,
                     page=page, native_label=nl, bbox=bbox,
                     heading_level_pending=True)
    return _prov(idx=idx, region_kind="paragraph", raw_text=content, page=page,
                 native_label=nl, bbox=bbox)


def _pair_exercises(
    regions: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], set, set]:
    """Spatially pair ordinal text regions with display_formula regions.

    Ported from ``build_html.build_page_exercise``: nearest unclaimed formula
    to the RIGHT sharing a row band. Returns
    ``(ordinal_region_id -> formula_region, used_formula_ids, unpaired_ord_ids)``.
    """
    formulas = [r for r in regions if str(r.get("native_label")) in rm.FORMULA_LABELS
                and _bbox(r) is not None]
    ordinals = [
        r for r in regions
        if str(r.get("native_label")) in rm.TEXTISH_LABELS
        and _ORD_ONLY_RE.match(_content(r)) and _bbox(r) is not None
    ]
    ordinals.sort(key=lambda r: (_bbox(r)[1], _bbox(r)[0]))
    used: set = set()
    pair: Dict[int, Dict[str, Any]] = {}
    unpaired: set = set()
    for o in ordinals:
        ox0, oy0, _ox1, oy1 = _bbox(o)
        best = None
        best_dx = None
        for f in formulas:
            if id(f) in used:
                continue
            fx0, fy0, _fx1, fy1 = _bbox(f)
            if fy1 < oy0 or oy1 < fy0:  # no row-band overlap
                continue
            if fx0 < ox0:  # formula must sit right of its number
                continue
            dx = fx0 - ox0
            if best_dx is None or dx < best_dx:
                best_dx, best = dx, f
        if best is not None:
            used.add(id(best))
            pair[id(o)] = best
        else:
            unpaired.add(id(o))
    return pair, used, unpaired


def transform_document(pages: Sequence[GlmPage]) -> TransformResult:
    """Transform an ordered list of GLM-OCR pages → the SemantiK wire contract."""
    prov: List[Dict[str, Any]] = []
    escalations: List[Dict[str, Any]] = []
    heading_tree: List[Tuple[int, str]] = []
    idx = 0

    for page in pages:
        pno = int(page.page_no)
        if page.error:
            escalations.append({
                "region_index": None, "source_page": pno, "native_label": "",
                "reason": "sdk_error", "detail": str(page.error),
            })
            continue
        regions = sorted(
            list(page.regions),
            key=lambda r: (int(r.get("index", 0)),),
        )
        has_exercises = any(
            str(r.get("native_label")) in rm.TEXTISH_LABELS
            and _ORD_ONLY_RE.match(_content(r)) for r in regions
        )
        pending_figure_id: Optional[int] = None  # prov index of a figure awaiting caption

        if has_exercises:
            pair, used, unpaired = _pair_exercises(regions)
            for oid in unpaired:
                escalations.append({
                    "region_index": None, "source_page": pno,
                    "native_label": "text", "reason": "unpaired_ordinal",
                    "detail": "exercise number with no paired formula",
                })
            for r in regions:
                nl = str(r.get("native_label") or "text")
                content = _content(r)
                if nl in rm.FORMULA_LABELS:
                    if id(r) in used:
                        continue  # consumed into an exercise unit
                    prov.append(_emit_structural(
                        r, idx=idx, page=pno, next_text="", next_kind="",
                        escalations=escalations))
                    idx += 1
                    continue
                if nl in rm.TEXTISH_LABELS:
                    m = _ORD_ONLY_RE.match(content)
                    if m:
                        f = pair.get(id(r))
                        body = _wrap_math(_content(f), True) if f is not None else ""
                        raw = f"{m.group(1)}. {body}".strip()
                        prov.append(_prov(
                            idx=idx, region_kind="paragraph", raw_text=raw,
                            page=pno, role="exercise", native_label=nl,
                            bbox=_bbox(r)))
                        idx += 1
                        continue
                    pm = _PROSE_ORD_RE.match(content)
                    if pm and "\n" not in content and int(pm.group(1)) > 1:
                        prov.append(_prov(
                            idx=idx, region_kind="paragraph", raw_text=content,
                            page=pno, role="exercise", native_label=nl,
                            bbox=_bbox(r)))
                        idx += 1
                        continue
                emitted = _emit_structural(
                    r, idx=idx, page=pno, next_text="", next_kind="",
                    escalations=escalations)
                if emitted is not None:
                    prov.append(emitted)
                    if emitted["region_kind"] == "heading":
                        heading_tree.append((int(emitted.get("level", 3)),
                                             str(emitted.get("heading_text", ""))))
                    idx += 1
            continue

        # ── single-column reading-order flow (figures merge with captions) ──
        for i, r in enumerate(regions):
            nl = str(r.get("native_label") or "text")
            base = rm.classify_native_label(nl)
            nxt = regions[i + 1] if i + 1 < len(regions) else None
            next_text = _content(nxt) if nxt else ""
            next_kind = rm.classify_native_label(str(nxt.get("native_label"))) if nxt else ""

            if base == "figure":
                emitted = _emit_structural(
                    r, idx=idx, page=pno, next_text=next_text, next_kind=next_kind,
                    escalations=escalations)
                prov.append(emitted)
                pending_figure_id = len(prov) - 1
                idx += 1
                continue
            if base == "caption":
                cap = rm.strip_md_heading(_content(r))
                if pending_figure_id is not None:
                    prov[pending_figure_id]["caption_text"] = cap
                    prov[pending_figure_id]["caption_source"] = "extracted"
                    pending_figure_id = None
                    # the caption is consumed into the figure; do not emit twice
                    continue
                emitted = _emit_structural(
                    r, idx=idx, page=pno, next_text=next_text, next_kind=next_kind,
                    escalations=escalations)
                prov.append(emitted)
                idx += 1
                continue

            pending_figure_id = None
            emitted = _emit_structural(
                r, idx=idx, page=pno, next_text=next_text, next_kind=next_kind,
                escalations=escalations)
            if emitted is None:
                continue
            prov.append(emitted)
            if emitted["region_kind"] == "heading":
                heading_tree.append((int(emitted.get("level", 3)),
                                     str(emitted.get("heading_text", ""))))
            idx += 1

        # flag caption-less figures for the Super/alt-text pass
        for p in prov:
            if p.get("region_kind") == "figure" and not p.get("caption_text") \
                    and not p.get("figure_alt") and p.get("source_page") == pno \
                    and not p.get("_escalated_caption"):
                escalations.append({
                    "region_index": p["first_raw_block_index"],
                    "source_page": pno, "native_label": p.get("native_label", "image"),
                    "reason": "caption_less_figure",
                    "detail": "figure with no extracted caption; needs alt-text",
                })
                p["_escalated_caption"] = True

    _stitch_cross_page(prov, escalations)
    # strip internal bookkeeping keys before returning the wire contract
    for p in prov:
        p.pop("_escalated_caption", None)
    return TransformResult(prov, heading_tree, escalations)


def _stitch_cross_page(
    prov: List[Dict[str, Any]], escalations: List[Dict[str, Any]]
) -> None:
    """Conservative cross-page continuation stitch (bbox+index heuristic).

    When a page's LAST paragraph ends open (no terminal punctuation) and the
    NEXT page's FIRST paragraph starts lowercase, they are the same broken
    sentence: merge the text into the earlier region and record an escalation
    (below-confidence stitches are Super's to confirm). Only plain paragraphs
    (no role / pedagogy) participate, so an exercise / apparatus / heading is
    never fused across a page.
    """
    def _plain(p: Dict[str, Any]) -> bool:
        return (p.get("region_kind") == "paragraph" and not p.get("role")
                and not p.get("pedagogy_class"))

    i = 0
    merged: List[bool] = [False] * len(prov)
    for a in range(len(prov) - 1):
        if merged[a] or not _plain(prov[a]):
            continue
        b = a + 1
        while b < len(prov) and merged[b]:
            b += 1
        if b >= len(prov) or not _plain(prov[b]):
            continue
        pa, pb = prov[a], prov[b]
        if pa.get("source_page") == pb.get("source_page"):
            continue  # same page — reading-order flow already handled it
        ta, tb = str(pa.get("raw_text", "")), str(pb.get("raw_text", ""))
        if _ends_open(ta) and tb[:1].islower():
            pa["raw_text"] = f"{ta.rstrip()} {tb.lstrip()}".strip()
            # union the page span
            pages = sorted(set(pa.get("pages", []) + pb.get("pages", [])))
            pa["pages"] = pages
            merged[b] = True
            escalations.append({
                "region_index": pa["first_raw_block_index"],
                "source_page": pa.get("source_page"),
                "native_label": pa.get("native_label", "text"),
                "reason": "cross_page_stitch",
                "detail": f"merged continuation from page {pb.get('source_page')}",
            })
    # drop merged-away regions
    prov[:] = [p for k, p in enumerate(prov) if not merged[k]]


__all__ = ["GlmPage", "TransformResult", "transform_document"]
