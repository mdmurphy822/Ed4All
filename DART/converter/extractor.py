"""Wave 16: unified PDF extraction layer.

Raw-text pdftotext has been the extraction floor for DART since the Wave
12 pipeline landed. That floor loses every structural signal: table
rows collapse into whitespace-aligned columns, figures disappear
entirely (pdftotext is a text extractor — it does not see raster
content), formulas degrade to ASCII approximations, and scanned /
image-only pages emit nothing at all.

This module adds a higher-floor extractor that preserves structure
when the upstream dependencies are available and degrades gracefully
when they are not. pdftotext remains the only hard requirement — every
other extractor is optional and contributes additively.

Output is a single :class:`ExtractedDocument` record consumed by the
next-phase segmenter
(:func:`DART.converter.block_segmenter.segment_extracted_document`)
plus the classifier + template layers unchanged.

Design invariants
-----------------

* **pdftotext is the only hard dependency.** Every other extractor
  degrades to ``None`` / ``[]`` on failure — no hard raise.
* **No Anthropic SDK imports here.** Claude calls for alt-text routing
  go through the injected :class:`LLMBackend` (passed down into
  :class:`AltTextGenerator`).
* **No side-effects on disk.** The extractor reads the source PDF but
  does not write image bytes anywhere; figure records carry the raw
  bytes (when available) so downstream callers can persist them into
  whatever staging layout they prefer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from DART.converter.page_chrome import (
    PageChrome,
    detect_page_chrome,
    strip_page_chrome,
)

if TYPE_CHECKING:  # pragma: no cover - type-check only imports
    from MCP.orchestrator.llm_backend import LLMBackend

logger = logging.getLogger(__name__)


# Wave 18: PyMuPDF-native date format ("D:20230315091500-07'00'"). The PDF
# spec date string starts with ``D:`` and uses 14-digit YYYYMMDDHHmmSS
# followed by optional timezone. We normalise to ISO 8601 date (YYYY-MM-DD)
# — good enough for Dublin Core / schema.org ``datePublished`` consumers.
_PDF_DATE_RE = re.compile(r"^\s*D?:?\s*(\d{4})(\d{2})(\d{2})")


# Caption detection — match ``Figure N[.M]:`` / ``Fig. N:`` / ``Image N:``
# / ``Figure N -`` patterns (case-insensitive). Keep the anchor loose so
# we accept both colon and dash separators; the full matched line becomes
# the caption (templates tolerate the leading label).
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?|image)\s+\d+(?:\.\d+)?\s*[:\-\u2013\u2014]\s*\S.*$",
    re.IGNORECASE,
)

# Known raster formats we round-trip through. Anything else falls back to
# ``.png`` so the file still lands on disk with a predictable extension.
_KNOWN_IMAGE_EXTS = {"png", "jpeg", "jpg", "gif", "bmp", "tiff", "tif", "webp"}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


BBox = Tuple[float, float, float, float]


@dataclass
class ExtractedTable:
    """A structured table extracted by pdfplumber (or equivalent).

    ``header_rows`` carries zero-to-many header rows so multi-row
    header tables are preserved. ``body_rows`` is the data region.
    Both are lists of stringified cells so downstream templates can
    ``html.escape`` without type-juggling. ``caption`` is a free-form
    string pulled from text immediately above / below the bbox when
    the extractor can identify one — often empty.

    Wave 18: ``source`` records which extractor produced the table
    (``"pdfplumber"`` or ``"pymupdf"``) so the reconciliation layer
    can prefer one over the other and downstream templates can surface
    a ``data-dart-table-extractor`` attribute for debuggability.
    """

    page: int
    bbox: BBox
    header_rows: List[List[str]] = field(default_factory=list)
    body_rows: List[List[str]] = field(default_factory=list)
    caption: Optional[str] = None
    source: str = "pdfplumber"


@dataclass
class ExtractedFigure:
    """A figure / image region extracted from the source PDF.

    ``image_path`` is a filesystem path where raw bytes were (or can
    be) written. The extractor leaves it empty when no image bytes
    were materialised — e.g. a vector-graphic region the upstream
    extractor only observed as a bounding box. ``alt_text`` is
    populated by the injected :class:`AltTextGenerator` when an LLM
    backend is provided; otherwise ``None``. ``caption`` mirrors the
    nearby caption pdfplumber / PyMuPDF found, when any.
    """

    page: int
    bbox: BBox
    image_path: str = ""
    alt_text: Optional[str] = None
    caption: Optional[str] = None


@dataclass
class ExtractedTOCEntry:
    """A single entry in the PDF's native outline (``doc.get_toc()``).

    Populated by :func:`_extract_toc_pymupdf` when PyMuPDF is available.
    ``level`` starts at 1 for top-level chapters. ``page`` is 1-indexed
    to match the rest of the extractor's page accounting.
    """

    level: int
    title: str
    page: int


@dataclass
class ExtractedTextSpan:
    """A contiguous text run with layout + font metadata from PyMuPDF.

    Produced by :func:`_extract_text_spans_pymupdf` via
    ``page.get_text("dict")``. Consumed by the heuristic classifier's
    font-size-based heading promotion (Wave 18): when a block's
    dominant span renders at >= 1.5x the document's median body font
    size, the block is promoted from ``PARAGRAPH`` to a heading role.
    """

    page: int
    bbox: BBox
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool


@dataclass
class ExtractedLink:
    """A hyperlink region from ``page.get_links()``.

    ``uri`` carries the external destination when the link points at a
    URL. ``dest_page`` carries the 1-indexed target page for internal
    (``goto``) links. Both are optional; at least one is typically
    populated.
    """

    page: int
    bbox: BBox
    uri: Optional[str] = None
    dest_page: Optional[int] = None


@dataclass
class ExtractedDocument:
    """Unified extraction artifact consumed by Phase 1 segmentation.

    Every field has a safe default so a pdftotext-only extraction
    still produces a valid document. Only ``raw_text`` is required —
    :func:`extract_document` raises when pdftotext itself fails.

    Wave 18 additions — all PyMuPDF-sourced, all empty by default so
    pre-Wave-18 callers stay compatible:

    * ``toc`` — native PDF bookmarks / outline.
    * ``pdf_metadata`` — normalised metadata dict (``title``,
      ``author``, ``subject``, ``creationDate`` in ISO 8601, etc.).
    * ``text_spans`` — font-size + bbox-tagged text runs used by the
      heuristic heading promoter.
    * ``links`` — hyperlink annotations (external URIs + internal
      ``goto`` destinations).
    """

    raw_text: str
    source_pdf: str
    pages_count: int = 0
    tables: List[ExtractedTable] = field(default_factory=list)
    figures: List[ExtractedFigure] = field(default_factory=list)
    ocr_text: Optional[str] = None
    # Wave 18 additions — all PyMuPDF-sourced.
    toc: List[ExtractedTOCEntry] = field(default_factory=list)
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)
    text_spans: List[ExtractedTextSpan] = field(default_factory=list)
    links: List[ExtractedLink] = field(default_factory=list)
    # Wave 20: running-header / running-footer / page-number chrome.
    # Populated by :func:`DART.converter.page_chrome.detect_page_chrome`
    # before ``raw_text`` is frozen. ``raw_text`` itself is already
    # chrome-stripped when this field is non-empty so downstream
    # segmentation never sees the polluting chrome lines; this field
    # preserves the detection result (headers / footers / per-page
    # page-number labels) for debuggability + sidecar emission.
    page_chrome: PageChrome = field(default_factory=PageChrome)


# ---------------------------------------------------------------------------
# pdftotext (hard dependency)
# ---------------------------------------------------------------------------


def _run_pdftotext(pdf_path: str) -> str:
    """Run ``pdftotext -layout`` and return stdout.

    Raises :class:`RuntimeError` on any failure so the caller can
    decide whether to abort the whole extraction — pdftotext is the
    only hard dependency of the extractor.
    """
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftotext binary not available; install poppler-utils"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"pdftotext timed out on {pdf_path}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"pdftotext failed with exit={proc.returncode}: {stderr.strip()}"
        )

    return proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""


def _count_pages(raw_text: str) -> int:
    """Estimate pages from form-feed markers in pdftotext output.

    pdftotext emits ``\\x0c`` between pages in layout mode. When no
    form-feeds are present (single-page or stripped), return 1 as a
    conservative lower bound.
    """
    if not raw_text:
        return 0
    form_feeds = raw_text.count("\x0c")
    return max(1, form_feeds + 1) if raw_text.strip() else 0


# ---------------------------------------------------------------------------
# pdfplumber (optional — tables)
# ---------------------------------------------------------------------------


def _stringify_row(row) -> List[str]:
    """Coerce a pdfplumber row into a list of plain strings.

    pdfplumber returns ``None`` for empty cells; we normalise to ""
    so templates can escape uniformly.
    """
    return ["" if cell is None else str(cell).strip() for cell in (row or [])]


def _find_caption_for_table(page_text: str, bbox: BBox) -> Optional[str]:
    """Best-effort caption scrape: look for a ``Table N:`` line in page text.

    We do not have per-line bbox matching without a full layout parse,
    so we return the first ``Table N:`` caption found on the page when
    available. Downstream templates tolerate missing captions.
    """
    if not page_text:
        return None
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("table ") and ":" in stripped:
            return stripped
    return None


def _extract_tables_pdfplumber(pdf_path: str) -> List[ExtractedTable]:
    """Return every table in ``pdf_path`` via pdfplumber.

    pdfplumber is optional; any failure (import, file open, per-page
    extraction) degrades to an empty list. The extractor never raises
    on table failure — downstream segmentation still produces prose
    blocks from the raw text.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.debug("pdfplumber not installed; skipping table extraction")
        return []

    tables: List[ExtractedTable] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.find_tables()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "pdfplumber find_tables failed on page %d: %s",
                        page_index,
                        exc,
                    )
                    continue

                if not page_tables:
                    continue

                # Text for the page is only pulled once, lazily.
                page_text = ""
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "pdfplumber extract_text failed on page %d: %s",
                        page_index,
                        exc,
                    )

                for tbl in page_tables:
                    try:
                        rows = tbl.extract() or []
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "pdfplumber table.extract failed on page %d: %s",
                            page_index,
                            exc,
                        )
                        continue

                    if not rows:
                        continue

                    # Treat the first row as the header by default.
                    # pdfplumber doesn't flag headers explicitly; a
                    # caller that knows its corpus can re-promote later.
                    header_rows = [_stringify_row(rows[0])]
                    body_rows = [_stringify_row(r) for r in rows[1:]]

                    bbox = tuple(getattr(tbl, "bbox", (0.0, 0.0, 0.0, 0.0)))
                    if len(bbox) != 4:
                        bbox = (0.0, 0.0, 0.0, 0.0)

                    tables.append(
                        ExtractedTable(
                            page=page_index,
                            bbox=bbox,  # type: ignore[arg-type]
                            header_rows=header_rows,
                            body_rows=body_rows,
                            caption=_find_caption_for_table(page_text, bbox),  # type: ignore[arg-type]
                        )
                    )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pdfplumber extraction failed: %s", exc)
        return []

    return tables


# ---------------------------------------------------------------------------
# PyMuPDF peer extractors (Wave 18): TOC, metadata, spans, links, tables
# ---------------------------------------------------------------------------
#
# Each helper accepts an already-open PyMuPDF ``fitz.Document`` and returns
# a list / dict. Every one is individually guarded so a failure in any
# single helper never takes the whole extraction down; PyMuPDF itself
# stays an optional dep. The public ``extract_document`` entry point opens
# the doc once and passes it into all of these helpers so we don't pay
# the open cost per call.


def _normalise_pdf_date(raw: Any) -> Optional[str]:
    """Normalise a PyMuPDF metadata date to ISO 8601 (``YYYY-MM-DD``).

    PyMuPDF surfaces PDF date strings like ``"D:20230315091500-07'00'"``
    (PDF spec format). We extract the ``YYYYMMDD`` prefix and render it
    as ``YYYY-MM-DD``. When the string doesn't match the PDF format, we
    return it unchanged (so ISO-8601 dates passed through the metadata
    dict stay intact). Non-string values degrade to ``None``.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    if not raw.strip():
        return None
    match = _PDF_DATE_RE.match(raw)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return f"{year}-{month}-{day}"
    # Not a PDF-spec date — pass through unchanged so ISO dates survive.
    return raw.strip()


def _extract_toc_pymupdf(doc: Any) -> List[ExtractedTOCEntry]:
    """Return the PDF's native outline / TOC as a list of entries.

    PyMuPDF's ``doc.get_toc(simple=True)`` returns a list of
    ``[level, title, page]`` tuples, 1-indexed pages. We map them into
    :class:`ExtractedTOCEntry` instances. Any failure — missing method
    on older versions, broken outline, etc. — degrades to ``[]``.
    """
    try:
        raw_toc = doc.get_toc(simple=True)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("PyMuPDF get_toc failed: %s", exc)
        return []
    if not raw_toc:
        return []

    entries: List[ExtractedTOCEntry] = []
    for row in raw_toc:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        try:
            level = int(row[0])
            title = str(row[1] or "").strip()
            page = int(row[2])
        except (TypeError, ValueError):
            continue
        if not title:
            continue
        entries.append(
            ExtractedTOCEntry(level=max(1, level), title=title, page=max(1, page))
        )
    return entries


def _extract_metadata_pymupdf(doc: Any) -> Dict[str, Any]:
    """Return a normalised metadata dict derived from ``doc.metadata``.

    PyMuPDF surfaces ``{"title": ..., "author": ..., "subject": ...,
    "keywords": ..., "creator": ..., "producer": ..., "creationDate":
    ..., "modDate": ..., "format": ..., "encryption": ...}``. We keep
    the keys that downstream consumers (Dublin Core emitter, document
    JSON-LD) read and normalise the date fields to ISO 8601.

    Empty / whitespace-only values are dropped so the merge step in
    :mod:`MCP.tools.pipeline_tools` can trivially check truthiness to
    decide whether to fill a blank in the caller's ``metadata`` dict.
    """
    try:
        raw = dict(doc.metadata or {})
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF metadata access failed: %s", exc)
        return {}

    result: Dict[str, Any] = {}
    for key in ("title", "author", "subject", "keywords", "creator", "producer"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()

    for date_key in ("creationDate", "modDate"):
        normalised = _normalise_pdf_date(raw.get(date_key))
        if normalised:
            result[date_key] = normalised

    return result


def _flag_is_bold(flags: int) -> bool:
    """Bit 4 (``2**4 == 16``) of a PyMuPDF span flag indicates bold."""
    try:
        return bool(int(flags) & 16)
    except (TypeError, ValueError):
        return False


def _flag_is_italic(flags: int) -> bool:
    """Bit 1 (``2**1 == 2``) of a PyMuPDF span flag indicates italic."""
    try:
        return bool(int(flags) & 2)
    except (TypeError, ValueError):
        return False


def _extract_text_spans_pymupdf(doc: Any) -> List[ExtractedTextSpan]:
    """Return every text span in ``doc`` with font-size + bbox metadata.

    Walks ``page.get_text("dict")`` and collects every ``span`` in every
    line of every text block. Spans with zero-length text or missing
    bbox are skipped. Any per-page failure degrades to skipping that
    page; missing-API failures (older PyMuPDF) degrade to ``[]``.
    """
    spans: List[ExtractedTextSpan] = []
    try:
        page_count = len(doc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF len(doc) failed: %s", exc)
        return []

    for page_index in range(page_count):
        try:
            page = doc[page_index]
            page_dict = page.get_text("dict")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PyMuPDF get_text('dict') failed on page %d: %s",
                page_index + 1,
                exc,
            )
            continue

        for block in (page_dict or {}).get("blocks", []) or []:
            # block type 0 is text (1 is image). Filter so we don't pick
            # up image-block stubs with no spans.
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    text = span.get("text") or ""
                    if not text.strip():
                        continue
                    bbox = tuple(span.get("bbox") or (0.0, 0.0, 0.0, 0.0))
                    if len(bbox) != 4:
                        bbox = (0.0, 0.0, 0.0, 0.0)
                    try:
                        font_size = float(span.get("size", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        font_size = 0.0
                    flags = span.get("flags", 0)
                    font_name = str(span.get("font") or "")
                    spans.append(
                        ExtractedTextSpan(
                            page=page_index + 1,
                            bbox=bbox,  # type: ignore[arg-type]
                            text=text,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=_flag_is_bold(flags),
                            is_italic=_flag_is_italic(flags),
                        )
                    )
    return spans


def _extract_links_pymupdf(doc: Any) -> List[ExtractedLink]:
    """Return every hyperlink in ``doc`` as an :class:`ExtractedLink`.

    PyMuPDF ``page.get_links()`` returns a list of dicts with ``kind``
    (``LINK_URI`` / ``LINK_GOTO`` / ...), ``from`` (rect), ``uri``,
    and ``page`` (target page, 0-indexed). We normalise page numbers to
    1-indexed to match the rest of the extractor.
    """
    links: List[ExtractedLink] = []
    try:
        page_count = len(doc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF len(doc) failed in links helper: %s", exc)
        return []

    for page_index in range(page_count):
        try:
            page = doc[page_index]
            raw_links = page.get_links() or []
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PyMuPDF get_links failed on page %d: %s",
                page_index + 1,
                exc,
            )
            continue

        for link in raw_links:
            rect = link.get("from")
            if rect is None:
                bbox: BBox = (0.0, 0.0, 0.0, 0.0)
            else:
                try:
                    bbox = (
                        float(rect[0]),
                        float(rect[1]),
                        float(rect[2]),
                        float(rect[3]),
                    )
                except (TypeError, ValueError, IndexError):
                    bbox = (0.0, 0.0, 0.0, 0.0)

            uri = link.get("uri")
            if isinstance(uri, str) and uri.strip():
                uri_value: Optional[str] = uri.strip()
            else:
                uri_value = None

            dest_raw = link.get("page")
            dest_page: Optional[int] = None
            if dest_raw is not None:
                try:
                    dest_int = int(dest_raw)
                    # PyMuPDF uses -1 for "no target"; 0-indexed internal
                    # page numbers get bumped to 1-indexed here.
                    if dest_int >= 0:
                        dest_page = dest_int + 1
                except (TypeError, ValueError):
                    dest_page = None

            if uri_value is None and dest_page is None:
                continue

            links.append(
                ExtractedLink(
                    page=page_index + 1,
                    bbox=bbox,
                    uri=uri_value,
                    dest_page=dest_page,
                )
            )
    return links


def _find_tables_pymupdf(doc: Any) -> List[ExtractedTable]:
    """Return tables found by PyMuPDF's ``page.find_tables()`` API.

    Used as a fallback when pdfplumber yields no tables. PyMuPDF's
    ``find_tables()`` is 1.23+; on older installs the method is missing
    and this helper degrades to ``[]``. Every extracted table is tagged
    with ``source="pymupdf"`` so the reconciliation layer knows where
    it came from.
    """
    tables: List[ExtractedTable] = []
    try:
        page_count = len(doc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF len(doc) failed in tables helper: %s", exc)
        return []

    for page_index in range(page_count):
        try:
            page = doc[page_index]
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PyMuPDF page access failed on %d: %s", page_index + 1, exc
            )
            continue

        find_tables = getattr(page, "find_tables", None)
        if find_tables is None:
            # Old PyMuPDF without the API — no tables, no crash.
            logger.debug(
                "PyMuPDF page.find_tables missing on page %d (version < 1.23?)",
                page_index + 1,
            )
            continue

        try:
            found = find_tables()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PyMuPDF find_tables raised on page %d: %s",
                page_index + 1,
                exc,
            )
            continue

        # ``find_tables`` may return a TableFinder object with a ``.tables``
        # attribute, or an iterable of tables directly. Handle both.
        table_iter = getattr(found, "tables", None)
        if table_iter is None:
            try:
                table_iter = list(found)
            except TypeError:
                table_iter = []

        for tbl in table_iter or []:
            try:
                rows = tbl.extract() or []
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "PyMuPDF table.extract raised on page %d: %s",
                    page_index + 1,
                    exc,
                )
                continue
            if not rows:
                continue

            header_rows = [_stringify_row(rows[0])]
            body_rows = [_stringify_row(r) for r in rows[1:]]

            bbox_raw = getattr(tbl, "bbox", (0.0, 0.0, 0.0, 0.0))
            try:
                bbox: BBox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except (TypeError, ValueError, IndexError):
                bbox = (0.0, 0.0, 0.0, 0.0)

            tables.append(
                ExtractedTable(
                    page=page_index + 1,
                    bbox=bbox,
                    header_rows=header_rows,
                    body_rows=body_rows,
                    caption=None,
                    source="pymupdf",
                )
            )
    return tables


def median_body_font_size(spans: List[ExtractedTextSpan]) -> Optional[float]:
    """Return the median ``font_size`` across all ``spans``.

    Used by the heuristic classifier's font-size-based heading promoter.
    Returns ``None`` when no usable spans are present so the promoter
    can no-op.
    """
    sizes = [s.font_size for s in spans if s.font_size and s.font_size > 0]
    if not sizes:
        return None
    sizes.sort()
    mid = len(sizes) // 2
    if len(sizes) % 2 == 1:
        return float(sizes[mid])
    return float((sizes[mid - 1] + sizes[mid]) / 2.0)


# ---------------------------------------------------------------------------
# Tesseract OCR (optional — full-page text fallback)
# ---------------------------------------------------------------------------


def _extract_ocr_text(pdf_path: str) -> Optional[str]:
    """Return OCR text for the whole PDF, or ``None`` when unavailable.

    OCR is only attempted when the ``tesseract`` binary is on PATH
    *and* PyMuPDF (``fitz``) is importable for per-page rasterisation.
    Any failure degrades to ``None``; OCR is purely additive.
    """
    if shutil.which("tesseract") is None:
        logger.debug("tesseract binary not on PATH; skipping OCR")
        return None

    try:
        import fitz  # type: ignore
    except ImportError:
        logger.debug("PyMuPDF not installed; skipping OCR")
        return None

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        logger.debug("pytesseract / Pillow missing; skipping OCR")
        return None

    try:
        import io

        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF could not open %s for OCR: %s", pdf_path, exc)
        return None

    chunks: List[str] = []
    try:
        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img)
                if text and text.strip():
                    chunks.append(text.strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("OCR failed on page %d: %s", page_index + 1, exc)
                continue
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass

    if not chunks:
        return None
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Figures (optional — PyMuPDF + existing AltTextGenerator)
# ---------------------------------------------------------------------------


def _image_ext_from_format(fmt: Optional[str]) -> str:
    """Normalise an upstream image format label to a filename extension.

    PyMuPDF reports the underlying image format on ``ExtractedImage.format``
    (``"png"``, ``"jpeg"``, etc.); a few odd corpora occasionally surface
    ``None`` or an unknown label. We default to ``png`` so every
    persisted figure still lands with a predictable extension.
    """
    if not fmt:
        return "png"
    label = str(fmt).strip().lower()
    if label == "jpg":
        return "jpg"
    if label in _KNOWN_IMAGE_EXTS:
        return label
    return "png"


def _figure_filename(page: int, image_bytes: bytes, fmt: Optional[str]) -> str:
    """Compute the stable ``{page:04d}-{hash8}.{ext}`` filename.

    ``page`` is 1-indexed (matches :class:`ExtractedFigure.page`).
    ``hash8`` is the first eight hex chars of ``sha256(image_bytes)`` so
    re-extracting the same bytes (same PDF, same page, same raster)
    produces the same filename — callers can idempotently re-run the
    pipeline without orphaning disk copies.
    """
    digest = hashlib.sha256(image_bytes or b"").hexdigest()[:8]
    ext = _image_ext_from_format(fmt)
    safe_page = max(0, int(page or 0))
    return f"{safe_page:04d}-{digest}.{ext}"


def _persist_image_bytes(
    image_bytes: bytes,
    figures_dir: Path,
    *,
    page: int,
    fmt: Optional[str],
) -> str:
    """Write ``image_bytes`` to ``figures_dir`` and return the relative path.

    Returns just the filename (no directory prefix) so the caller /
    assembler layer can control the relative path written into the
    final HTML ``<img src>`` attribute. Idempotent: if the target file
    already exists we skip the write and return the same filename.
    """
    filename = _figure_filename(page, image_bytes, fmt)
    figures_dir.mkdir(parents=True, exist_ok=True)
    target = figures_dir / filename
    if not target.exists():
        try:
            target.write_bytes(image_bytes)
        except OSError as exc:  # pragma: no cover - disk errors are rare
            logger.debug("Failed to persist figure to %s: %s", target, exc)
            return ""
    return filename


def _find_caption_for_figure(
    page_text: str,
    page_number: Optional[int],
    already_taken: Optional[set] = None,
) -> Optional[str]:
    """Best-effort figure caption scrape.

    Scans ``page_text`` for the first ``Figure N.M:`` / ``Fig. N:`` /
    ``Image N:`` / ``Figure N -`` line and returns the full line. Each
    caption line is "claimed" via ``already_taken`` so multiple figures
    on the same page bind to successive captions instead of all getting
    the first one. ``already_taken`` is a ``set`` of line numbers into
    ``page_text.splitlines()`` that the caller maintains per page.

    Returns ``None`` when no caption is found, so downstream templates
    can fall through to an alt-text-only or caption-less rendering.
    """
    if not page_text:
        return None
    lines = page_text.splitlines()
    for idx, line in enumerate(lines):
        if already_taken is not None and idx in already_taken:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if _FIGURE_CAPTION_RE.match(stripped):
            if already_taken is not None:
                already_taken.add(idx)
            return stripped
    return None


def _build_page_text_index(raw_text: str) -> List[str]:
    """Split pdftotext output into a list of per-page strings.

    pdftotext emits ``\\x0c`` between pages in ``-layout`` mode. When
    form-feeds are absent we return a single-element list carrying the
    whole document so the caption scraper still has something to scan.
    """
    if not raw_text:
        return []
    if "\x0c" in raw_text:
        return raw_text.split("\x0c")
    return [raw_text]


def _extract_figures(
    pdf_path: str,
    *,
    llm: Optional["LLMBackend"] = None,
    figures_dir: Optional[Path] = None,
    page_text_index: Optional[List[str]] = None,
) -> List[ExtractedFigure]:
    """Return every figure-like region in ``pdf_path``.

    Uses :class:`DART.pdf_converter.image_extractor.PDFImageExtractor`
    (PyMuPDF-backed) for raster + vector extraction. When ``llm`` is
    provided, each figure's alt-text is populated via
    :class:`DART.pdf_converter.alt_text_generator.AltTextGenerator`;
    when ``llm`` is ``None``, alt-text stays ``None`` so downstream
    templates can fall back to the caption.

    When ``figures_dir`` is provided, raw image bytes returned by the
    upstream extractor are persisted under ``figures_dir`` as
    ``{page:04d}-{hash8}.{ext}`` and ``ExtractedFigure.image_path`` is
    set to the relative filename (no directory prefix — the caller /
    assembler layer decides the relative path written into ``<img
    src>``). When ``figures_dir`` is ``None`` (the default), no disk
    I/O occurs and ``image_path`` stays empty, matching pre-Wave-17
    behaviour.

    ``page_text_index`` (optional) is a per-page list of pdftotext
    text used by the caption scraper; when absent we fall back to the
    upstream extractor's ``nearby_caption`` attribute.
    """
    try:
        from DART.pdf_converter.image_extractor import PDFImageExtractor
    except ImportError:
        logger.debug("image_extractor unavailable; skipping figure extraction")
        return []

    try:
        extractor = PDFImageExtractor(pdf_path)
        images = extractor.extract_all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDFImageExtractor failed on %s: %s", pdf_path, exc)
        return []

    if not images:
        return []

    alt_gen = None
    if llm is not None:
        try:
            from DART.pdf_converter.alt_text_generator import AltTextGenerator

            alt_gen = AltTextGenerator(llm=llm, use_ocr_fallback=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AltTextGenerator init failed: %s", exc)
            alt_gen = None

    # Track which caption lines have been claimed per page so multiple
    # figures on the same page bind to distinct captions rather than all
    # latching onto the first ``Figure N:`` line.
    caption_claims: dict = {}

    figures: List[ExtractedFigure] = []
    for img in images:
        page_number = int(getattr(img, "page", 0) or 0)
        alt_text: Optional[str] = None
        if alt_gen is not None:
            try:
                result = alt_gen.generate(img)
                if result.success and result.alt_text:
                    alt_text = result.alt_text
            except Exception as exc:  # noqa: BLE001
                logger.debug("Alt-text generation failed for image: %s", exc)

        # Caption: prefer the pdftotext-backed scraper (so "Figure N.M:"
        # lines bind to real captions); fall back to the upstream
        # extractor's nearby_caption only when we have nothing better.
        caption: Optional[str] = None
        if page_text_index and 0 < page_number <= len(page_text_index):
            page_text = page_text_index[page_number - 1]
            claims = caption_claims.setdefault(page_number, set())
            caption = _find_caption_for_figure(page_text, page_number, claims)
        if caption is None:
            nearby = getattr(img, "nearby_caption", "") or None
            if nearby:
                caption = nearby

        # Persist raw bytes when caller asked. ``img.data`` is populated
        # by PDFImageExtractor; ``img.format`` carries the raster format.
        image_path = ""
        if figures_dir is not None:
            data = getattr(img, "data", b"") or b""
            if data:
                fmt = getattr(img, "format", None)
                image_path = _persist_image_bytes(
                    data,
                    Path(figures_dir),
                    page=page_number,
                    fmt=fmt,
                )

        figures.append(
            ExtractedFigure(
                page=page_number,
                bbox=tuple(getattr(img, "bbox", (0.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                image_path=image_path,
                alt_text=alt_text,
                caption=caption,
            )
        )

    return figures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _open_pymupdf(pdf_path: str):
    """Return an open PyMuPDF ``fitz.Document`` or ``None`` when unavailable.

    Centralises the import + open dance so every Wave-18 PyMuPDF helper
    can share a single document handle. Any failure (missing dep, bad
    file) degrades to ``None`` so the calling code can no-op the
    PyMuPDF-dependent extractors.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.debug("PyMuPDF not installed; skipping PyMuPDF-backed extractors")
        return None
    try:
        return fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyMuPDF could not open %s: %s", pdf_path, exc)
        return None


def extract_document(
    pdf_path: str,
    *,
    llm: Optional["LLMBackend"] = None,
    figures_dir: Optional[Path] = None,
) -> ExtractedDocument:
    """Extract a :class:`ExtractedDocument` from ``pdf_path``.

    pdftotext is the only hard dependency; all other extractors are
    optional and contribute additively. Individual extractor failures
    are logged at DEBUG level and degrade to empty / ``None`` fields
    — the function never raises except when pdftotext itself fails.

    Parameters
    ----------
    pdf_path:
        Filesystem path to the source PDF.
    llm:
        Optional :class:`LLMBackend` used to populate figure alt-text
        via the existing :class:`AltTextGenerator`. When ``None`` (the
        default), figure alt-text stays ``None``.
    figures_dir:
        Optional directory for persisted figure image bytes. When
        provided, each :class:`ExtractedFigure` whose upstream
        extractor returned image bytes is written to
        ``figures_dir / {page:04d}-{hash8}.{ext}`` and its
        ``image_path`` set to the relative filename. When ``None`` (the
        default), no disk I/O occurs and ``image_path`` stays empty —
        matching pre-Wave-17 behaviour.

    Wave 18: opens PyMuPDF once (when available) and runs every
    PyMuPDF-backed peer extractor (TOC, metadata, text spans, links,
    and the ``find_tables`` fallback) against the single shared
    document handle. All outputs are additive — empty lists / dicts
    when PyMuPDF is unavailable, so existing callers see no change.
    """
    raw_text = _run_pdftotext(pdf_path)

    tables: List[ExtractedTable] = []
    try:
        tables = _extract_tables_pdfplumber(pdf_path)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("Table extraction raised unexpectedly: %s", exc)
        tables = []

    # Wave 20: detect + strip page-chrome before anything downstream sees
    # ``raw_text``. Detection is frequency-based on per-page top/bottom
    # lines and degrades to a no-op on short / form-feed-less input.
    # The stripped text is what every downstream consumer uses (figure
    # caption scraping, segmentation, etc.); the detected chrome record
    # is preserved on ``ExtractedDocument.page_chrome`` for sidecars.
    try:
        chrome = detect_page_chrome(raw_text)
    except Exception as exc:  # noqa: BLE001 — chrome detection never blocks
        logger.debug("Page-chrome detection raised unexpectedly: %s", exc)
        chrome = PageChrome()

    if chrome.headers or chrome.footers or chrome.page_number_lines:
        try:
            raw_text = strip_page_chrome(raw_text, chrome)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Page-chrome strip raised unexpectedly: %s; keeping raw", exc
            )

    page_text_index = _build_page_text_index(raw_text)

    figures: List[ExtractedFigure] = []
    try:
        figures = _extract_figures(
            pdf_path,
            llm=llm,
            figures_dir=figures_dir,
            page_text_index=page_text_index,
        )
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("Figure extraction raised unexpectedly: %s", exc)
        figures = []

    ocr_text: Optional[str] = None
    try:
        ocr_text = _extract_ocr_text(pdf_path)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("OCR extraction raised unexpectedly: %s", exc)
        ocr_text = None

    # Wave 18: PyMuPDF peer-extractor pass. One shared open handle.
    toc: List[ExtractedTOCEntry] = []
    pdf_metadata: Dict[str, Any] = {}
    text_spans: List[ExtractedTextSpan] = []
    links: List[ExtractedLink] = []

    fitz_doc = _open_pymupdf(pdf_path)
    if fitz_doc is not None:
        try:
            toc = _extract_toc_pymupdf(fitz_doc)
        except Exception as exc:  # noqa: BLE001 — defense in depth
            logger.debug("TOC extraction raised unexpectedly: %s", exc)
            toc = []

        try:
            pdf_metadata = _extract_metadata_pymupdf(fitz_doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("PyMuPDF metadata raised unexpectedly: %s", exc)
            pdf_metadata = {}

        try:
            text_spans = _extract_text_spans_pymupdf(fitz_doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Text-span extraction raised unexpectedly: %s", exc)
            text_spans = []

        try:
            links = _extract_links_pymupdf(fitz_doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Link extraction raised unexpectedly: %s", exc)
            links = []

        # Table reconciliation: if pdfplumber gave us nothing AND
        # PyMuPDF's find_tables has results, use PyMuPDF's instead. This
        # lets PyMuPDF pick up tables on text-heavy textbooks where
        # pdfplumber's structure detection fails to fire.
        if not tables:
            try:
                pymupdf_tables = _find_tables_pymupdf(fitz_doc)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "PyMuPDF find_tables raised unexpectedly: %s", exc
                )
                pymupdf_tables = []
            if pymupdf_tables:
                tables = pymupdf_tables

        try:
            fitz_doc.close()
        except Exception:  # noqa: BLE001
            pass

    return ExtractedDocument(
        raw_text=raw_text,
        source_pdf=pdf_path,
        pages_count=_count_pages(raw_text),
        tables=tables,
        figures=figures,
        ocr_text=ocr_text,
        toc=toc,
        pdf_metadata=pdf_metadata,
        text_spans=text_spans,
        links=links,
        page_chrome=chrome,
    )


__all__ = [
    "BBox",
    "ExtractedDocument",
    "ExtractedFigure",
    "ExtractedLink",
    "ExtractedTOCEntry",
    "ExtractedTable",
    "ExtractedTextSpan",
    "PageChrome",
    "extract_document",
    "median_body_font_size",
]
