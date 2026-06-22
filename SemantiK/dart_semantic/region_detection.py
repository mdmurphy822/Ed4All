"""Heuristics for detecting math / table regions inside a shared-extract page.

Lives separate from `extract_shared.py` so both the smoke harness and
the production enrichment pass use the same detectors. Pure functions
over the shared-extract JSON — no PDF IO.

Phase 1 additions:
    `RegionCandidate` / `TableCandidate` / `MathCandidate` — typed
    dataclasses consumed by Stage 3 detector BERTs (in later phases).
    `detect_table_region_candidates(shared)` and
    `detect_math_region_candidates(shared)` wrap the legacy per-page
    detectors above and aggregate results across all pages, enriched
    with cell-grids, glyph-density features, and member block indices
    so detector heads do not need to re-derive them.

Phase 1 is deterministic — no torch, no transformers, no models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Latex math-specific font families. CMMI = math italic, CMSY = math
# symbols, CMEX = math extensions, MSBM/MSAM = AMS math, EUSM/EUFM =
# Euler math. CMR / CMBX / CMTT are regular body fonts, excluded.
MATH_FONT_TOKENS: tuple[str, ...] = (
    "CMMI", "CMSY", "CMEX", "MSBM", "MSAM", "EUSM", "EUFM", "MathPi",
    "cmmi", "cmsy", "cmex",
)
PURE_MATH_FONTS = MATH_FONT_TOKENS
CID_RE = re.compile(r"\(cid:\d+\)")
EQ_NUM_RE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)\s*$")
# Common math glyph characters / unicode ranges that appear in displayed
# equations (a permissive superset — Phase 1 is allowed to over-fire).
MATH_GLYPH_RE = re.compile(
    r"[=+\-−×÷≠≤≥≃∼≈"  # operators
    r"∂∇∑∏∫⋅±"  # calculus & misc
    r"α-ωΑ-Ω"  # greek letters
    r"→⇒⇔]"  # arrows
)
# A short trailing pattern that almost always indicates a numbered equation.
EQ_LABEL_TRAIL_RE = re.compile(r"\((?:\d+(?:\.\d+)?|[A-Z]\.\d+)\)\s*$")
FRACTION_BAR_RE = re.compile(r"[⁄∕─-━—―]")
SUBSUP_RE = re.compile(r"[⁰-₟²³¹]")


@dataclass
class Region:
    page_num: int
    kind: str                  # "math" | "table"
    bbox: tuple[float, float, float, float]  # pdf-space, top-left origin
    src_text: str              # the pre-existing extractor's best text


def _block_math_signal(tb: dict) -> int:
    """0 = no indicator, 2 = math (CID token OR short text in pure-math font).

    The length guard prevents pypdfium2's occasional whole-line mis-tag
    (e.g., an email labeled CMSY because it started with `∗`) from
    flooding the detector.
    """
    text = (tb.get("text") or "").strip()
    font = tb.get("font_name") or ""
    if CID_RE.search(text):
        return 2
    if any(tok in font for tok in PURE_MATH_FONTS) and len(text) <= 12:
        return 2
    return 0


def _merge_bboxes(bboxes: list[tuple[float, float, float, float]]
                  ) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def _cluster_by_line(blocks: list[dict],
                     y_tol: float = 6.0) -> list[list[dict]]:
    items: list[tuple[float, dict]] = []
    for tb in blocks:
        bx = tb.get("bbox")
        if not bx or len(bx) != 4:
            continue
        mid_y = (float(bx[1]) + float(bx[3])) / 2.0
        items.append((mid_y, tb))
    items.sort(key=lambda p: p[0])
    clusters: list[list[tuple[float, dict]]] = []
    for mid_y, tb in items:
        if clusters and abs(mid_y - clusters[-1][0][0]) <= y_tol:
            clusters[-1].append((mid_y, tb))
        else:
            clusters.append([(mid_y, tb)])
    return [[tb for _, tb in c] for c in clusters]


def _line_is_displayed_math(line: list[dict]) -> bool:
    if not line:
        return False
    math_count = sum(1 for tb in line if _block_math_signal(tb) >= 1)
    frac = math_count / len(line)
    if frac >= 0.30:
        return True
    if math_count >= 1 and frac >= 0.15:
        full_text = " ".join((tb.get("text") or "").strip() for tb in line)
        if EQ_NUM_RE.search(full_text):
            return True
    return False


def detect_math_regions(page: dict) -> list[Region]:
    """Return displayed-equation regions on this page.

    Line-level: clusters blocks by y-midpoint, keeps only lines whose
    block population is majority math-font / CID-heavy.
    """
    text_blocks = page.get("merged", {}).get("text_blocks", []) or []
    if not text_blocks:
        return []
    regions: list[Region] = []
    for line in _cluster_by_line(text_blocks):
        if not _line_is_displayed_math(line):
            continue
        bx_list = [tuple(float(x) for x in tb["bbox"]) for tb in line
                   if tb.get("bbox")]
        if not bx_list:
            continue
        bbox = _merge_bboxes(bx_list)
        text = " ".join((tb.get("text") or "").strip() for tb in line)
        regions.append(Region(page_num=page["page_num"], kind="math",
                              bbox=bbox, src_text=text))
    return regions


def detect_table_regions(page: dict) -> list[Region]:
    """Every pdfplumber-detected table is a GLM-OCR target."""
    tables = page.get("pdfplumber", {}).get("tables", []) or []
    out: list[Region] = []
    for t in tables:
        bbox = tuple(float(x) for x in (t.get("bbox") or []))
        if len(bbox) != 4:
            continue
        rows = t.get("rows") or []
        flat = " | ".join(
            ", ".join((c or "").strip() for c in row) for row in rows
        )
        out.append(Region(page_num=page["page_num"], kind="table",
                          bbox=bbox, src_text=flat))
    return out


DEFAULT_TESSERACT_CONF_THRESHOLD = 0.70


def detect_low_conf_tesseract_regions(
    page: dict,
    *,
    conf_threshold: float = DEFAULT_TESSERACT_CONF_THRESHOLD,
) -> list[Region]:
    """Tesseract blocks whose confidence is below the threshold get
    flagged for GLM-OCR re-OCR.

    Tesseract bboxes are stored in *image* pixel space (at the scale
    pypdfium2 used to render the page for OCR), while the rest of the
    pipeline works in PDF-point space. We divide by the
    tesseract_scale = tesseract_width / pdf_width so the returned
    Region.bbox is in PDF-space and feeds correctly into the shared
    `_crop` helper.
    """
    if "tesseract" not in (page.get("sources_used") or []):
        return []
    tess_w = float(page.get("tesseract_width") or 0.0)
    pdf_w = float(page.get("width") or 0.0)
    scale = (tess_w / pdf_w) if (tess_w and pdf_w) else 1.0
    if scale <= 0:
        scale = 1.0
    out: list[Region] = []
    for tb in page.get("tesseract", {}).get("text_blocks", []) or []:
        conf = float(tb.get("confidence") or 0.0)
        if conf >= conf_threshold:
            continue
        bbox_img = tb.get("bbox") or []
        if len(bbox_img) != 4:
            continue
        bbox = tuple(float(x) / scale for x in bbox_img)
        out.append(Region(page_num=page["page_num"], kind="tess_low_conf",
                          bbox=bbox, src_text=tb.get("text") or ""))
    return out


# ---------------------------------------------------------------------------
# Phase 1: typed RegionCandidate hierarchy
# ---------------------------------------------------------------------------

@dataclass
class RegionCandidate:
    """A geometric region candidate emitted by Stage 2.

    The `kind` attribute discriminates between subclasses. `bbox` is in
    PDF-point space (top-left origin, same convention as `Region`).
    `pages` lists every 1-indexed page the region touches — usually a
    single page, but kept as a list so multi-page tables / equations
    can be represented if the upstream detectors evolve to span pages.

    `evidence_features` is a free-form dict of which heuristic flags
    fired during detection (e.g. `{"cid_chars": 12, "math_font_frac":
    0.42}`); the detector BERTs in Phase 2/3 read this to gate their
    own confidences.

    `member_block_indices` are indices into the FeatureBlock stream
    (`featurize_with_regions(...).feature_blocks`) of any blocks whose
    centroid lies inside `bbox`. When the candidate covers no feature
    blocks (e.g. an OCR-only region in an otherwise text-layer doc),
    the list is empty.
    """
    kind: str
    bbox: tuple[float, float, float, float]
    pages: list[int]
    evidence_features: dict[str, Any] = field(default_factory=dict)
    member_block_indices: list[int] = field(default_factory=list)


@dataclass
class TableCandidate(RegionCandidate):
    """A table region candidate.

    `cell_grid` is the pdfplumber-extracted row/cell grid (list of
    rows, each a list of cell strings or `None` for empty cells).
    Forwarding it here avoids re-running pdfplumber from the
    TableSpecialist downstream (DP-1.2).

    `header_row_indices` lists row indices (0-based into `cell_grid`)
    that the geometric heuristic flagged as headers — typically just
    `[0]`, but pdfplumber can detect multi-row banded headers.

    `bordered` is a boolean indicating whether pdfplumber found
    explicit ruling lines (vs. spatial inference from whitespace).
    Bordered tables are higher-confidence positives.
    """
    cell_grid: list[list[Any]] = field(default_factory=list)
    header_row_indices: list[int] = field(default_factory=list)
    bordered: bool = False

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = "table"


@dataclass
class MathCandidate(RegionCandidate):
    """A math region candidate (typically a displayed equation).

    `glyph_density_features` precomputes signal scores the
    MathSpecialist (Phase 2) needs:

        * `math_symbol_density`     — count(math glyph chars) / total chars
        * `italic_letter_density`   — italic alpha chars / total alpha
        * `subsup_count`            — number of sub/super-script characters
        * `fraction_bar_count`      — number of fraction-bar style glyphs
        * `equation_number_trail`   — 1.0 if the line ends with `(N)`,
                                       0.0 otherwise
        * `math_font_frac`          — fraction of source blocks rendered
                                       in a CMMI/CMSY/CMEX-family font
        * `cid_token_count`         — number of `(cid:NNN)` placeholders
                                       in the merged text (these are
                                       always math glyphs that pdfplumber
                                       could not decode)
    """
    glyph_density_features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = "math"


# ---------------------------------------------------------------------------
# Phase 1: shared-extract-level detectors
# ---------------------------------------------------------------------------

def _block_centroid_in(bbox: tuple[float, float, float, float],
                       region_bbox: tuple[float, float, float, float]) -> bool:
    mx = (float(bbox[0]) + float(bbox[2])) / 2.0
    my = (float(bbox[1]) + float(bbox[3])) / 2.0
    return (region_bbox[0] <= mx <= region_bbox[2]
            and region_bbox[1] <= my <= region_bbox[3])


def _index_feature_blocks_by_page(feature_blocks) -> dict[int, list[tuple[int, Any]]]:
    """Map page_num -> list of (global_index, feature_block)."""
    by_page: dict[int, list[tuple[int, Any]]] = {}
    if not feature_blocks:
        return by_page
    for i, fb in enumerate(feature_blocks):
        page = getattr(fb.raw, "page", None) if hasattr(fb, "raw") else None
        if page is None:
            continue
        by_page.setdefault(page, []).append((i, fb))
    return by_page


def _members_for_region(
    page_num: int,
    bbox: tuple[float, float, float, float],
    by_page: dict[int, list[tuple[int, Any]]],
) -> list[int]:
    out: list[int] = []
    for idx, fb in by_page.get(page_num, []):
        if _block_centroid_in(fb.raw.bbox, bbox):
            out.append(idx)
    return out


def _table_is_bordered(table_dict: dict) -> bool:
    """Return True iff pdfplumber found explicit ruling lines.

    pdfplumber's per-table dict includes a `"strategy"` or `"settings"`
    key when configured; the format-of-record we observe in
    extract_shared has ruled tables come through with `cells` populated
    AND non-empty edges. We treat 'has cells' as the soft positive
    signal because the merged JSON reliably exposes it.
    """
    cells = table_dict.get("cells") or []
    edges = table_dict.get("edges") or []
    if edges:
        return True
    # Cells with explicit bbox geometry → ruling lines were detected.
    return any(isinstance(c, dict) and c.get("bbox") for c in cells)


def _glyph_density_features_from_line(
    line: list[dict],
) -> dict[str, float]:
    full_text = " ".join((tb.get("text") or "") for tb in line)
    total_chars = max(1, len(full_text))
    alpha = [c for c in full_text if c.isalpha()]
    n_alpha = max(1, len(alpha))
    math_glyphs = MATH_GLYPH_RE.findall(full_text)
    fraction_bars = FRACTION_BAR_RE.findall(full_text)
    sub_sup = SUBSUP_RE.findall(full_text)
    cid_tokens = CID_RE.findall(full_text)
    eq_trail = 1.0 if EQ_LABEL_TRAIL_RE.search(full_text.strip()) else 0.0

    n_math_font = sum(
        1 for tb in line
        if any(tok in (tb.get("font_name") or "") for tok in PURE_MATH_FONTS)
    )
    italic_count = 0
    for tb in line:
        if not tb.get("is_italic"):
            continue
        italic_count += sum(1 for c in (tb.get("text") or "") if c.isalpha())

    return {
        "math_symbol_density": len(math_glyphs) / total_chars,
        "italic_letter_density": italic_count / n_alpha,
        "subsup_count": float(len(sub_sup)),
        "fraction_bar_count": float(len(fraction_bars)),
        "equation_number_trail": eq_trail,
        "math_font_frac": (n_math_font / max(1, len(line))),
        "cid_token_count": float(len(cid_tokens)),
    }


def _evidence_features_for_math_line(line: list[dict]) -> dict[str, Any]:
    return {
        "n_blocks": len(line),
        "math_signal_blocks": sum(1 for tb in line if _block_math_signal(tb) >= 1),
        "fonts": sorted({(tb.get("font_name") or "") for tb in line}),
    }


def detect_table_region_candidates(
    shared: dict,
    *,
    feature_blocks: list | None = None,
) -> list[TableCandidate]:
    """Walk every page in `shared` and emit one TableCandidate per
    pdfplumber-detected table, enriched with cell-grid + bordered flag.

    Phase 1 contract: recall over precision (DP-1.1). Every pdfplumber
    table becomes a candidate; the detector BERT in Phase 3d filters.

    `feature_blocks` is optional. When provided, each candidate's
    `member_block_indices` is populated with the global indices of
    every FeatureBlock whose centroid lies inside the table bbox.
    """
    by_page = _index_feature_blocks_by_page(feature_blocks or [])
    out: list[TableCandidate] = []
    for page in shared.get("pages", []) or []:
        page_num = int(page.get("page_num") or 0)
        for t in page.get("pdfplumber", {}).get("tables", []) or []:
            bbox_raw = t.get("bbox") or []
            if len(bbox_raw) != 4:
                continue
            bbox = tuple(float(x) for x in bbox_raw)
            rows = t.get("rows") or []
            n_rows = len(rows)
            n_cols = max((len(r) for r in rows), default=0)
            bordered = _table_is_bordered(t)
            # Header row heuristic: the first row of every table is the
            # default header. Phase 1 over-fires (DP-1.1); the BERT
            # decides whether it's actually a header row.
            header_rows = [0] if n_rows > 0 else []
            evidence: dict[str, Any] = {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "bordered": bordered,
                "src_text_len": sum(
                    len((c or "")) for row in rows for c in row
                ),
            }
            members = _members_for_region(page_num, bbox, by_page)
            out.append(
                TableCandidate(
                    kind="table",
                    bbox=bbox,
                    pages=[page_num],
                    evidence_features=evidence,
                    member_block_indices=members,
                    cell_grid=[list(r) for r in rows],
                    header_row_indices=header_rows,
                    bordered=bordered,
                )
            )
    return out


def detect_math_region_candidates(
    shared: dict,
    *,
    feature_blocks: list | None = None,
) -> list[MathCandidate]:
    """Walk every page in `shared` and emit one MathCandidate per
    detected displayed-equation line (or contiguous run of lines).

    Internally re-uses `detect_math_regions` for the geometric pass,
    then enriches each emitted region with glyph-density features
    derived from the merged text blocks within the region's bbox.
    """
    by_page = _index_feature_blocks_by_page(feature_blocks or [])
    out: list[MathCandidate] = []
    for page in shared.get("pages", []) or []:
        page_num = int(page.get("page_num") or 0)
        regions = detect_math_regions(page)
        if not regions:
            continue
        text_blocks = page.get("merged", {}).get("text_blocks", []) or []
        for region in regions:
            # Find the merged blocks whose centroid is inside this region;
            # use them to compute glyph-density features.
            line_blocks = [
                tb for tb in text_blocks
                if (tb.get("bbox") and len(tb["bbox"]) == 4
                    and _block_centroid_in(tuple(float(x) for x in tb["bbox"]),
                                           region.bbox))
            ]
            features = _glyph_density_features_from_line(line_blocks)
            evidence = _evidence_features_for_math_line(line_blocks)
            evidence["src_text"] = region.src_text
            members = _members_for_region(page_num, region.bbox, by_page)
            out.append(
                MathCandidate(
                    kind="math",
                    bbox=region.bbox,
                    pages=[page_num],
                    evidence_features=evidence,
                    member_block_indices=members,
                    glyph_density_features=features,
                )
            )
    return out


__all__ = [
    "Region",
    "RegionCandidate",
    "TableCandidate",
    "MathCandidate",
    "detect_math_regions",
    "detect_table_regions",
    "detect_low_conf_tesseract_regions",
    "detect_table_region_candidates",
    "detect_math_region_candidates",
    "DEFAULT_TESSERACT_CONF_THRESHOLD",
    "MATH_FONT_TOKENS",
    "PURE_MATH_FONTS",
]
