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

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - type-check only imports
    from MCP.orchestrator.llm_backend import LLMBackend

logger = logging.getLogger(__name__)


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
    """

    page: int
    bbox: BBox
    header_rows: List[List[str]] = field(default_factory=list)
    body_rows: List[List[str]] = field(default_factory=list)
    caption: Optional[str] = None


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
class ExtractedDocument:
    """Unified extraction artifact consumed by Phase 1 segmentation.

    Every field has a safe default so a pdftotext-only extraction
    still produces a valid document. Only ``raw_text`` is required —
    :func:`extract_document` raises when pdftotext itself fails.
    """

    raw_text: str
    source_pdf: str
    pages_count: int = 0
    tables: List[ExtractedTable] = field(default_factory=list)
    figures: List[ExtractedFigure] = field(default_factory=list)
    ocr_text: Optional[str] = None


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


def _extract_figures(
    pdf_path: str,
    *,
    llm: Optional["LLMBackend"] = None,
) -> List[ExtractedFigure]:
    """Return every figure-like region in ``pdf_path``.

    Uses :class:`DART.pdf_converter.image_extractor.PDFImageExtractor`
    (PyMuPDF-backed) for raster + vector extraction. When ``llm`` is
    provided, each figure's alt-text is populated via
    :class:`DART.pdf_converter.alt_text_generator.AltTextGenerator`;
    when ``llm`` is ``None``, alt-text stays ``None`` so downstream
    templates can fall back to the caption.
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

    figures: List[ExtractedFigure] = []
    for img in images:
        alt_text: Optional[str] = None
        if alt_gen is not None:
            try:
                result = alt_gen.generate(img)
                if result.success and result.alt_text:
                    alt_text = result.alt_text
            except Exception as exc:  # noqa: BLE001
                logger.debug("Alt-text generation failed for image: %s", exc)

        figures.append(
            ExtractedFigure(
                page=int(getattr(img, "page", 0) or 0),
                bbox=tuple(getattr(img, "bbox", (0.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                image_path="",  # Raw bytes live on img.data; caller persists.
                alt_text=alt_text,
                caption=(getattr(img, "nearby_caption", "") or None),
            )
        )

    return figures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_document(
    pdf_path: str,
    *,
    llm: Optional["LLMBackend"] = None,
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
    """
    raw_text = _run_pdftotext(pdf_path)

    tables: List[ExtractedTable] = []
    try:
        tables = _extract_tables_pdfplumber(pdf_path)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("Table extraction raised unexpectedly: %s", exc)
        tables = []

    figures: List[ExtractedFigure] = []
    try:
        figures = _extract_figures(pdf_path, llm=llm)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("Figure extraction raised unexpectedly: %s", exc)
        figures = []

    ocr_text: Optional[str] = None
    try:
        ocr_text = _extract_ocr_text(pdf_path)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        logger.debug("OCR extraction raised unexpectedly: %s", exc)
        ocr_text = None

    return ExtractedDocument(
        raw_text=raw_text,
        source_pdf=pdf_path,
        pages_count=_count_pages(raw_text),
        tables=tables,
        figures=figures,
        ocr_text=ocr_text,
    )


__all__ = [
    "BBox",
    "ExtractedDocument",
    "ExtractedFigure",
    "ExtractedTable",
    "extract_document",
]
