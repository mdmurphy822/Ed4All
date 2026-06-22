"""Stage 1 core: run every applicable extractor and emit a standardized
per-page JSON document.

Rather than routing to a single best extractor per document, we run all
licence-compatible extractors that apply and expose their outputs side-by-
side in one JSON shape. Downstream stages (features, classify) read the
MERGED view but can also inspect each extractor's raw output for
disagreement signals or training augmentation.

Licensing inventory (all must be commercial-safe):
    pikepdf       MPL-2.0     metadata, encryption, tagged-PDF detection, widgets
    pypdfium2     Apache-2    primary text extraction w/ bboxes + rendering
    pdfplumber    MIT         table detection, char-level layout
    pdfminer.six  MIT         (transitive via pdfplumber; not called directly)
    Tesseract     Apache-2    OCR fallback when text layer missing/corrupt

Deliberately excluded: PyMuPDF/MuPDF (AGPL-3), Poppler (GPL-2).

Output JSON per document:
{
  "pdf_path": "...",
  "metadata": { producer, creator, title, pdf_version, is_encrypted,
                is_tagged, page_count },
  "pages": [
    {
      "page_num": 1,                 // 1-indexed
      "width": 612.0, "height": 792.0,
      "sources_used": ["pypdfium2", "pdfplumber"],   // or "tesseract", "tagged_pdf"
      "pikepdf": { "widgets": [...], "structure_nodes": [...] },  // present if applicable
      "pypdfium2": { "text_blocks": [...] },
      "pdfplumber": { "text_blocks": [...], "tables": [...] },
      "tesseract": { "text_blocks": [...] },          // only when we ran OCR
      "merged": {
        "text_blocks": [...],        // reconciled stream of {bbox, text, font_size, font_name, is_bold, is_italic, confidence, provenance}
        "tables": [...],             // from pdfplumber
        "widgets": [...]             // from pikepdf when fillable
      }
    }, ...
  ]
}

Where `provenance` is a list of source-names that contributed that block,
e.g. ["pypdfium2", "pdfplumber"] if both agreed on the line, or
["tesseract"] if only OCR produced it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import paths as _semantik_paths

logger = logging.getLogger(__name__)


# Minimum text-layer chars below which we assume the page is scanned.
TEXT_LAYER_MIN_CHARS = 10
# If >10% of text-layer chars are Unicode replacement, treat as corrupt.
REPLACEMENT_CHAR = "�"
REPLACEMENT_MAX_FRAC = 0.10
# pdfplumber word-break gap threshold as a fraction of font size. 0.15 (≈1.5pt
# at 10pt) recovers spaces on tight-kerned LaTeX PDFs without over-splitting
# large-font titles. See _pdfplumber_page / Plans/11 Stage 3.
_X_TOLERANCE_RATIO = 0.15


# ---------- entry point ----------


def extract_shared(pdf_path: Path) -> dict:
    """Run all applicable extractors and return the shared JSON."""
    pdf_path = Path(pdf_path)

    meta = _pikepdf_inspect(pdf_path)
    if meta["is_encrypted"]:
        raise EncryptedPDFError(pdf_path)

    # Per-page extraction.
    pages: list[dict] = []
    for page_num in range(1, meta["page_count"] + 1):
        pages.append(_extract_page(pdf_path, page_num, meta))

    return {
        "pdf_path": str(pdf_path),
        "metadata": meta,
        "pages": pages,
    }


# Bump whenever the extraction *output* changes for the same input PDF
# (new extractor settings, parser fixes, schema changes). Entries written
# under an older salt become unreachable, so stale extractions can never
# be served after a behavior change. v2: the 2026-06-08 pdfplumber
# x_tolerance_ratio=0.15 fix (Plans/11 Stage 3) — pre-fix caches glued
# sub-3pt LaTeX inter-word gaps into single words. v3: the 2026-06-17
# pypdfium2 line-segment re-extraction fix — pre-fix caches garbled
# letter-spaced small-caps ("M ARKWAYNE", "Before S YKES L EE").
EXTRACT_CACHE_VERSION = 3


def extract_shared_cached(pdf_path: Path, cache_dir: Path | str | None = None) -> dict:
    """extract_shared with a disk cache keyed on (version, path, size, mtime).

    Useful for batch workflows where the same PDF is consulted many times
    (e.g., arXiv section pairs — one arXiv PDF produces 4-15 section pairs,
    each of which would otherwise re-run pypdfium2 + pdfplumber + pikepdf
    from scratch).

    Cache hits are ~instant; cache misses run the full extractor and write
    the result to disk. The cache key includes mtime so a modified PDF
    auto-invalidates, and ``EXTRACT_CACHE_VERSION`` so an extractor
    behavior change auto-invalidates too. Safe across processes — each
    worker reads/writes its own cache files independently; last writer
    wins on collision.
    """
    import hashlib
    import json as _json

    pdf_path = Path(pdf_path)
    # Default cache_dir resolves under the SemantiK cache root (CWD-independent)
    # instead of the legacy CWD-relative ``data/extract_cache`` literal; an
    # explicit caller-supplied dir is honored unchanged. Byte-stable to the
    # historic ``<root>/data/extract_cache`` layout when no SEMANTIK_* env is set.
    cache_dir = (
        _semantik_paths.resolve_cache("extract_cache")
        if cache_dir is None
        else Path(cache_dir)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        st = pdf_path.stat()
        key_raw = f"v{EXTRACT_CACHE_VERSION}|{pdf_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        key = hashlib.sha256(key_raw.encode()).hexdigest()[:24]
    except OSError:
        # If we can't stat the file, skip the cache — let extract_shared
        # raise its normal error.
        return extract_shared(pdf_path)

    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            return _json.loads(cache_path.read_text())
        except Exception:
            # Corrupt cache file — fall through to fresh extraction.
            pass

    shared = extract_shared(pdf_path)
    try:
        cache_path.write_text(_json.dumps(shared, default=str))
    except Exception:
        # Failing to cache is non-fatal.
        pass
    return shared


class EncryptedPDFError(Exception):
    pass


# ---------- pikepdf inspection ----------


def _pikepdf_inspect(pdf_path: Path) -> dict:
    """Metadata + tagged-PDF check + encryption check via pikepdf."""
    import pikepdf  # type: ignore

    info = {
        "producer": None,
        "creator": None,
        "title": None,
        "pdf_version": None,
        "is_encrypted": False,
        "is_tagged": False,
        "page_count": 0,
    }
    try:
        # pikepdf refuses encrypted PDFs by default; we catch the specific error.
        pdf = pikepdf.open(str(pdf_path))
    except pikepdf.PasswordError:
        info["is_encrypted"] = True
        return info
    except Exception as exc:
        raise ExtractionError(f"pikepdf failed to open {pdf_path}: {exc}") from exc

    try:
        dm = pdf.docinfo
        for dm_key, out_key in (
            ("/Producer", "producer"),
            ("/Creator", "creator"),
            ("/Title", "title"),
        ):
            val = dm.get(dm_key) if dm is not None else None
            if val is not None:
                info[out_key] = str(val)
        info["pdf_version"] = str(pdf.pdf_version)
        info["page_count"] = len(pdf.pages)

        # Tagged PDF detection via StructTreeRoot.
        root = pdf.Root
        if "/StructTreeRoot" in root.keys():
            info["is_tagged"] = True
            info["structtree_ref"] = True
    finally:
        pdf.close()

    return info


def _pikepdf_page_widgets(pdf_path: Path, page_num: int) -> list[dict]:
    """Read AcroForm widget annotations on one page via pikepdf.

    Widget /Rect is in PDF default user space (bottom-left origin, y up).
    We normalize to top-left origin so bboxes are comparable with
    pypdfium2 text-block bboxes throughout the rest of the pipeline.
    """
    import pikepdf  # type: ignore

    out: list[dict] = []
    try:
        pdf = pikepdf.open(str(pdf_path))
    except Exception:
        return out
    try:
        if page_num < 1 or page_num > len(pdf.pages):
            return out
        page = pdf.pages[page_num - 1]

        # Page height for y-flip (bottom-left -> top-left origin).
        mediabox = page.get("/MediaBox")
        if mediabox is not None and len(mediabox) >= 4:
            page_h = float(mediabox[3]) - float(mediabox[1])
        else:
            page_h = 792.0  # Letter default

        annots = page.get("/Annots")
        if annots is None:
            return out
        for a in annots:
            try:
                subtype = str(a.get("/Subtype", ""))
                if subtype != "/Widget":
                    continue
                rect = a.get("/Rect")
                if rect is not None and len(rect) >= 4:
                    x0, y0_bl, x1, y1_bl = (
                        float(rect[0]),
                        float(rect[1]),
                        float(rect[2]),
                        float(rect[3]),
                    )
                    # Flip to top-left origin so downstream bbox-in-bbox
                    # checks work against pypdfium2 / pdfplumber blocks.
                    bbox = [x0, page_h - y1_bl, x1, page_h - y0_bl]
                else:
                    bbox = None
                field_type = str(a.get("/FT", "")).lstrip("/") or None
                name = str(a.get("/T", "")) or None
                tooltip = str(a.get("/TU", "")) or None
                flags = int(a.get("/Ff", 0) or 0)
                out.append(
                    {
                        "bbox": bbox,
                        "field_type": field_type,
                        "name": name,
                        "tooltip": tooltip,
                        "flags": flags,
                    }
                )
            except Exception as exc:
                logger.debug(f"widget parse error page {page_num}: {exc}")
                continue
    finally:
        pdf.close()
    return out


def _pikepdf_page_structure(pdf_path: Path, page_num: int) -> list[dict]:
    """Extract structure-tree nodes for a tagged page. Best-effort."""
    # Walking StructTreeRoot properly is complex. For v1 we just flag the
    # document as tagged in metadata and defer full StructTree parsing to
    # a later pass — the per-block classifier can still benefit from the
    # flag even without the tree details.
    return []


# ---------- pypdfium2 text extraction ----------


def _pypdfium2_page_blocks(pdf_path: Path, page_num: int) -> tuple[dict, float, float]:
    """Return ({text_blocks, chars_hint}, page_width, page_height) from pypdfium2."""
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_num - 1]
        page_w, page_h = float(page.get_size()[0]), float(page.get_size()[1])
        textpage = page.get_textpage()

        blocks: list[dict] = []
        try:
            n_rects = textpage.count_rects()
        except Exception:
            n_rects = 0

        # `count_rects()` usually returns one rect per visual line, but on
        # LETTER-SPACED small-caps (common in legal captions/headings —
        # "MARKWAYNE MULLIN", "Before SYKES, LEE, …") it over-splits: the leading
        # full-size capital becomes its own rect AND the rect reading order
        # scrambles ("M" … "ARKWAYNE" emitted apart). Emitting one block per rect
        # and space-joining them downstream yields garbled "M ARKWAYNE M ULLIN".
        # Instead: collect the rects, cluster them into line-segments (same line
        # by y-overlap; split on a big horizontal gap so columns / inline math
        # stay separate), then re-extract each segment's text with
        # ``get_text_bounded(union_bbox)`` — which pypdfium2 returns in correct
        # reading order ("MARKWAYNE MULLIN"). Single-rect lines are unchanged
        # (the union is the rect itself).
        raw: list[tuple[float, float, float, float]] = []
        for i in range(n_rects):
            try:
                left, bottom, right, top = textpage.get_rect(i)  # (l, b, r, t)
            except Exception:
                continue
            raw.append((float(left), float(bottom), float(right), float(top)))

        # Cluster into lines by vertical-midpoint proximity (native coords: y
        # increases upward, so read high-y first).
        raw.sort(key=lambda r: (-(r[1] + r[3]) / 2.0, r[0]))
        lines: list[list[tuple[float, float, float, float]]] = []
        for r in raw:
            mid = (r[1] + r[3]) / 2.0
            h = max(1.0, r[3] - r[1])
            if lines:
                prev = lines[-1]
                pmid = sum((p[1] + p[3]) / 2.0 for p in prev) / len(prev)
                ph = max(1.0, max(p[3] for p in prev) - min(p[1] for p in prev))
                if abs(mid - pmid) <= 0.6 * min(h, ph):
                    prev.append(r)
                    continue
            lines.append([r])

        # Within each line, split into segments on a big horizontal gap (a column
        # gutter or an inline-math gap), then one block per contiguous segment.
        for line in lines:
            line.sort(key=lambda r: r[0])
            segments: list[list[tuple[float, float, float, float]]] = [[line[0]]]
            for r in line[1:]:
                prev_r = segments[-1][-1]
                gap = r[0] - prev_r[2]
                h = max(1.0, r[3] - r[1])
                if gap > 3.0 * h:
                    segments.append([r])
                else:
                    segments[-1].append(r)
            for seg in segments:
                L = min(s[0] for s in seg)
                B = min(s[1] for s in seg)
                R = max(s[2] for s in seg)
                T = max(s[3] for s in seg)
                try:
                    text = " ".join(textpage.get_text_bounded(L, B, R, T).split())
                except Exception:
                    text = ""
                if not text:
                    continue
                # Convert to (x0, y0, x1, y1) with y0=top, y1=bottom (top-left).
                bbox = (L, float(page_h - T), R, float(page_h - B))
                blocks.append(
                    {
                        "bbox": list(bbox),
                        "text": text,
                        "font_size": None,  # pypdfium2 doesn't expose size per rect cheaply
                        "font_name": None,
                        "is_bold": None,
                        "is_italic": None,
                        "confidence": 1.0,
                    }
                )
        return {"text_blocks": blocks}, page_w, page_h
    finally:
        pdf.close()


def _pypdfium2_render_page_to_image(pdf_path: Path, page_num: int, scale: float = 2.0):
    """Render a page to a PIL image for Tesseract."""
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_num - 1]
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


# ---------- pdfplumber (tables + text with font info) ----------


def _pdfplumber_page(pdf_path: Path, page_num: int) -> dict:
    """Extract text blocks + tables via pdfplumber. Slower than pypdfium2
    but adds table detection and actual font metadata."""
    import pdfplumber  # type: ignore

    out = {"text_blocks": [], "tables": []}
    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as exc:
        logger.debug(f"pdfplumber open failed for {pdf_path}: {exc}")
        return out
    try:
        if page_num < 1 or page_num > len(pdf.pages):
            return out
        page = pdf.pages[page_num - 1]

        # Tables first so we can avoid double-counting their text.
        try:
            found_tables = page.find_tables()
        except Exception as exc:
            logger.debug(f"pdfplumber find_tables page {page_num}: {exc}")
            found_tables = []

        table_bboxes: list[tuple] = []
        for t in found_tables:
            try:
                rows = t.extract()
            except Exception:
                rows = None
            if not rows:
                continue
            bbox = t.bbox  # (x0, top, x1, bottom)
            table_bboxes.append(bbox)
            out["tables"].append(
                {
                    "bbox": [float(x) for x in bbox],
                    "rows": rows,
                }
            )

        # Extract line-level text via extract_words + group by line top.
        #
        # ``x_tolerance_ratio`` scales the word-break gap threshold by font
        # size so spaces are recovered on PDFs whose inter-word gaps are
        # smaller than pdfplumber's fixed 3pt default — common in LaTeX
        # output, which kerns words apart with NO space glyph. Without it,
        # ``extract_words`` returns whole lines as one token
        # ("frameworkforderivingvariousEPIs…"), which then poisons every
        # text feature downstream (heading detection, prose, tables). The
        # ratio (vs a fixed x_tolerance) avoids over-splitting large-font
        # titles. See Plans/11 Stage 3.
        words = []
        _word_kwargs = dict(
            keep_blank_chars=False,
            use_text_flow=True,
            x_tolerance_ratio=_X_TOLERANCE_RATIO,
            extra_attrs=["fontname", "size"],
        )
        try:
            words = page.extract_words(**_word_kwargs)
        except TypeError:
            # Older pdfplumber without x_tolerance_ratio — retry without it.
            _word_kwargs.pop("x_tolerance_ratio", None)
            try:
                words = page.extract_words(**_word_kwargs)
            except Exception as exc:
                logger.debug(f"pdfplumber extract_words page {page_num}: {exc}")
                words = []
        except Exception as exc:
            logger.debug(f"pdfplumber extract_words page {page_num}: {exc}")
            words = []

        # Bucket into lines by top-coordinate (within 2pt).
        words.sort(key=lambda w: (round(float(w.get("top", 0)), 0), float(w.get("x0", 0))))
        lines: list[list[dict]] = []
        for w in words:
            top = float(w.get("top", 0))
            if not lines or abs(top - float(lines[-1][0].get("top", 0))) > 3:
                lines.append([w])
            else:
                lines[-1].append(w)

        for line in lines:
            if not line:
                continue
            line.sort(key=lambda w: float(w.get("x0", 0)))
            x0 = min(float(w.get("x0", 0)) for w in line)
            x1 = max(float(w.get("x1", 0)) for w in line)
            top = min(float(w.get("top", 0)) for w in line)
            bot = max(float(w.get("bottom", 0)) for w in line)
            text = " ".join(w.get("text", "") for w in line).strip()
            if not text:
                continue
            # Skip lines that fall entirely inside a detected table bbox.
            mid_x = (x0 + x1) / 2
            mid_y = (top + bot) / 2
            if any(
                bx0 <= mid_x <= bx1 and by0 <= mid_y <= by1 for (bx0, by0, bx1, by1) in table_bboxes
            ):
                continue
            # Dominant font size (median).
            sizes = [float(w.get("size") or 0) for w in line if w.get("size")]
            font_name = line[0].get("fontname")
            size = sorted(sizes)[len(sizes) // 2] if sizes else None
            is_bold = bool(font_name and "Bold" in font_name)
            is_italic = bool(font_name and any(t in font_name for t in ("Italic", "Oblique")))
            out["text_blocks"].append(
                {
                    "bbox": [x0, top, x1, bot],
                    "text": text,
                    "font_size": size,
                    "font_name": font_name,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "confidence": 1.0,
                }
            )
    finally:
        pdf.close()
    return out


# ---------- tesseract OCR ----------


def _tesseract_page_blocks(pdf_path: Path, page_num: int) -> dict:
    """OCR one page via Tesseract. Rendered via pypdfium2 (Apache-2)."""
    import pytesseract

    image = _pypdfium2_render_page_to_image(pdf_path, page_num, scale=2.0)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    page_w = float(image.width)
    page_h = float(image.height)

    # Group words -> lines using (block_num, par_num, line_num).
    lines: dict[tuple, dict] = {}
    for j, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][j], data["par_num"][j], data["line_num"][j])
        ln = lines.setdefault(
            key,
            {
                "words": [],
                "heights": [],
                "confs": [],
                "lefts": [],
                "rights": [],
                "tops": [],
                "bottoms": [],
            },
        )
        ln["words"].append(word)
        ln["heights"].append(data["height"][j])
        conf = data.get("conf", [80])
        ln["confs"].append(float(conf[j] if j < len(conf) else 80))
        left = data["left"][j]
        top = data["top"][j]
        w = data["width"][j]
        h = data["height"][j]
        ln["lefts"].append(left)
        ln["rights"].append(left + w)
        ln["tops"].append(top)
        ln["bottoms"].append(top + h)

    blocks: list[dict] = []
    for ln in lines.values():
        heights = sorted(ln["heights"])
        est_size = heights[len(heights) // 2] if heights else 0.0
        conf = sum(ln["confs"]) / len(ln["confs"]) if ln["confs"] else 0.0
        blocks.append(
            {
                "bbox": [
                    float(min(ln["lefts"])),
                    float(min(ln["tops"])),
                    float(max(ln["rights"])),
                    float(max(ln["bottoms"])),
                ],
                "text": " ".join(ln["words"]),
                "font_size": float(est_size),
                "font_name": None,
                "is_bold": None,
                "is_italic": None,
                "confidence": conf / 100.0,
            }
        )
    # Tesseract's bboxes are in image-pixel space, not PDF-point space —
    # we expose page_w/page_h at the image scale too, so downstream
    # normalization (fractions of page) still works.
    return {"text_blocks": blocks, "page_width": page_w, "page_height": page_h}


# ---------- per-page orchestration ----------


def _extract_page(pdf_path: Path, page_num: int, meta: dict) -> dict:
    page: dict[str, Any] = {"page_num": page_num, "sources_used": []}

    # pypdfium2 first (primary text extractor)
    try:
        pfx, page_w, page_h = _pypdfium2_page_blocks(pdf_path, page_num)
        page["pypdfium2"] = pfx
        page["width"] = page_w
        page["height"] = page_h
        page["sources_used"].append("pypdfium2")
    except Exception as exc:
        page["pypdfium2"] = {"text_blocks": [], "error": str(exc)}
        page["width"] = 612.0
        page["height"] = 792.0

    pfx_blocks = page.get("pypdfium2", {}).get("text_blocks", [])
    pypdfium2_text = " ".join(b["text"] for b in pfx_blocks)
    is_text_ok = _text_layer_quality_ok(pypdfium2_text)

    # pdfplumber (tables + alternative text — always runs on text-layer pages)
    if is_text_ok:
        try:
            ppx = _pdfplumber_page(pdf_path, page_num)
            page["pdfplumber"] = ppx
            page["sources_used"].append("pdfplumber")
        except Exception as exc:
            page["pdfplumber"] = {"text_blocks": [], "tables": [], "error": str(exc)}

    # Tesseract: only if we don't have a clean text layer.
    if not is_text_ok:
        try:
            tes = _tesseract_page_blocks(pdf_path, page_num)
            page["tesseract"] = tes
            # Tesseract image pixel dims supersede the PDF point dims for
            # bbox normalization — store them separately.
            page["tesseract_width"] = tes.pop("page_width", None)
            page["tesseract_height"] = tes.pop("page_height", None)
            page["sources_used"].append("tesseract")
        except Exception as exc:
            page["tesseract"] = {"text_blocks": [], "error": str(exc)}

    # pikepdf: widgets + structure nodes (always check for widgets; cheap).
    try:
        widgets = _pikepdf_page_widgets(pdf_path, page_num)
    except Exception:
        widgets = []
    pike: dict[str, Any] = {"widgets": widgets}
    if meta.get("is_tagged"):
        pike["structure_nodes"] = _pikepdf_page_structure(pdf_path, page_num)
    page["pikepdf"] = pike
    if widgets or meta.get("is_tagged"):
        page["sources_used"].append("pikepdf")

    # Merge step.
    page["merged"] = _merge_page(page, is_text_ok)
    return page


# ---------- merge / reconcile ----------


def _merge_page(page: dict, text_layer_ok: bool) -> dict:
    """Produce the unified block list downstream stages consume.

    Strategy:
      - If text layer is clean: prefer pdfplumber line blocks (richer
        font info). For each pdfplumber block, suppress every pypdfium2
        block whose bbox is mostly contained — pypdfium2 enumerates math
        glyphs character-by-character and would otherwise flood the
        merged stream with single-char fragments.
      - Orphan pypdfium2 blocks (math equations pdfplumber missed) are
        clustered into per-line groups before emission.
      - If text layer is corrupt/missing: use tesseract blocks.
      - Attach pdfplumber tables + pikepdf widgets separately.
    """
    merged_blocks: list[dict] = []

    if text_layer_ok:
        ppx = page.get("pdfplumber", {}).get("text_blocks", [])
        pfx = page.get("pypdfium2", {}).get("text_blocks", [])
        used_pfx = [False] * len(pfx)
        for pp in ppx:
            covered = _all_contained_indices(pp["bbox"], pfx, used_pfx, frac=0.6)
            provenance = ["pdfplumber"]
            if covered:
                provenance.append("pypdfium2")
                for i in covered:
                    used_pfx[i] = True
            merged_blocks.append({**pp, "provenance": provenance})

        orphans = [pfx[i] for i, u in enumerate(used_pfx) if not u]
        for cluster in _cluster_blocks_by_line(orphans):
            merged_blocks.append({**cluster, "provenance": ["pypdfium2"]})
    else:
        for b in page.get("tesseract", {}).get("text_blocks", []):
            merged_blocks.append({**b, "provenance": ["tesseract"]})

    # Sort top-to-bottom, left-to-right.
    merged_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    return {
        "text_blocks": merged_blocks,
        "tables": page.get("pdfplumber", {}).get("tables", []),
        "widgets": page.get("pikepdf", {}).get("widgets", []),
    }


def _all_contained_indices(outer_bbox, candidates, used, *, frac: float) -> list[int]:
    """Indices of unused candidates whose bbox is at least `frac`-contained
    in `outer_bbox`. Used to dedupe pypdfium2's per-glyph rects against
    pdfplumber's line-level rects."""
    ax0, ay0, ax1, ay1 = outer_bbox
    out: list[int] = []
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        bx0, by0, bx1, by1 = c["bbox"]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        if inter / area >= frac:
            out.append(i)
    return out


def _cluster_blocks_by_line(
    blocks: list[dict], y_tol_ratio: float = 0.6, x_gap_ratio: float = 3.0
) -> list[dict]:
    """Cluster character-level blocks into line-level groups.

    Two blocks are on the same line if their y-midpoints differ by at
    most `y_tol_ratio * min_height`. Within a line, blocks are sorted by
    x and merged into a single block whose text concatenates parts with
    a space; bbox is the union; font metadata comes from the first part
    that exposed it.
    """
    if not blocks:
        return []
    items = []
    for b in blocks:
        bx = b.get("bbox")
        if not bx or len(bx) != 4:
            continue
        x0, y0, x1, y1 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
        items.append((x0, y0, x1, y1, b))
    items.sort(key=lambda it: ((it[1] + it[3]) / 2.0, it[0]))

    lines: list[list[tuple]] = []
    for it in items:
        x0, y0, x1, y1, _ = it
        my = (y0 + y1) / 2.0
        h = max(1.0, y1 - y0)
        if lines:
            prev = lines[-1]
            py0, py1 = min(p[1] for p in prev), max(p[3] for p in prev)
            pmy = (py0 + py1) / 2.0
            ph = max(1.0, py1 - py0)
            if abs(my - pmy) <= y_tol_ratio * min(h, ph):
                prev.append(it)
                continue
        lines.append([it])

    out: list[dict] = []
    for line in lines:
        line.sort(key=lambda it: it[0])
        # Within-line: split on big horizontal gaps so inline math equations
        # next to body text stay separate from the body text.
        groups: list[list[tuple]] = [[line[0]]]
        for it in line[1:]:
            prev = groups[-1][-1]
            gap = it[0] - prev[2]
            h = max(1.0, it[3] - it[1])
            if gap > x_gap_ratio * h:
                groups.append([it])
            else:
                groups[-1].append(it)
        for g in groups:
            xs0 = min(it[0] for it in g)
            ys0 = min(it[1] for it in g)
            xs1 = max(it[2] for it in g)
            ys1 = max(it[3] for it in g)
            text = " ".join(
                ((it[4].get("text") or "").strip()) for it in g if (it[4].get("text") or "").strip()
            )
            if not text:
                continue
            ref = g[0][4]
            font_size = next(
                (it[4].get("font_size") for it in g if it[4].get("font_size") is not None), None
            )
            font_name = next((it[4].get("font_name") for it in g if it[4].get("font_name")), None)
            out.append(
                {
                    "bbox": [xs0, ys0, xs1, ys1],
                    "text": text,
                    "font_size": font_size,
                    "font_name": font_name,
                    "is_bold": ref.get("is_bold"),
                    "is_italic": ref.get("is_italic"),
                    "confidence": ref.get("confidence", 1.0),
                }
            )
    return out


def _best_overlap_index(bbox_a, candidates, used) -> int | None:
    """Return index of candidate whose bbox overlaps bbox_a most, or None."""
    ax0, ay0, ax1, ay1 = bbox_a
    best_i = None
    best_iou = 0.0
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        bx0, by0, bx1, by1 = c["bbox"]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_i = i
    return best_i if best_iou > 0.3 else None


# ---------- text-layer quality check ----------

_WORD_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _text_layer_quality_ok(text: str) -> bool:
    """Heuristic: does the text layer look like real human-readable content?"""
    if len(text) < TEXT_LAYER_MIN_CHARS:
        return False
    if REPLACEMENT_CHAR in text:
        n_repl = text.count(REPLACEMENT_CHAR)
        if n_repl / max(1, len(text)) > REPLACEMENT_MAX_FRAC:
            return False
    # At least 30% of chars should be letters.
    word_chars = len(_WORD_CHAR_RE.findall(text))
    if word_chars / max(1, len(text)) < 0.30:
        return False
    return True


class ExtractionError(Exception):
    pass
