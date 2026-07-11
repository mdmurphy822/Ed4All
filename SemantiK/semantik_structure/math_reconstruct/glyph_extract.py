"""Per-glyph geometry recovery for deterministic MathML reconstruction.

The Stage-2 math detector linearises every displayed equation to a flat
reading-order string (``region_detection.detect_math_regions`` ->
``src_text = " ".join(...)``), which destroys the 2-D layout the math
carries (super/subscripts, fraction stacks, bracket spans). The math
council BERT only *classifies* ``math_type``; the geometry signals it
computed (``subsup_count`` / ``fraction_bar_count`` / ``math_font_frac``)
are discarded after classification.

This module recovers per-glyph geometry so :mod:`.structure_infer` can
rebuild the 2-D structure deterministically. Two sources, in priority
order:

1. **Real geometry** — re-extract pdfplumber ``page.chars`` for the math
   region's page, filtered to its bbox. Each char carries precise
   ``x0/x1/top/bottom/size`` so baseline + size shifts (the super/sub and
   fraction signals) are measurable.
2. **Text fallback** — when no PDF / page.chars is available, build
   degenerate single-baseline glyph records straight from ``src_text``.
   Unicode super/subscript characters (``x²``, ``H₂``) still carry their
   2-D intent in the codepoint, so :mod:`.structure_infer` recovers those
   even without geometry; everything else linearises (and the policy
   routes a low-confidence linearisation to the generative fallback).

Pure-ish: source (1) re-opens the PDF read-only via pdfplumber; source
(2) is pure. Nothing here writes.
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class Glyph:
    """One recovered glyph in top-left PDF-point space.

    ``top``/``bottom`` follow pdfplumber's convention (distance from the
    page top, increasing downward). ``size`` is the font size in points.
    ``source`` records how the glyph was recovered (``"pdfplumber"`` |
    ``"src_text"``) for provenance / debugging.
    """

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    source: str = "pdfplumber"

    @property
    def baseline(self) -> float:
        """Approx baseline = bottom (pdfplumber's bottom ~ glyph baseline)."""
        return self.bottom

    @property
    def y_mid(self) -> float:
        return (self.top + self.bottom) / 2.0


# Unicode super/subscript maps — codepoints that carry 2-D intent even in
# a flat string. Normalised back to their base digit/letter so the emitter
# renders <msup>/<msub> rather than an exotic literal glyph.
_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
    "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-",
    "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n",
    "ⁱ": "i",
}
_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9", "₊": "+", "₋": "-",
    "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x",
}

SUPERSCRIPT_CHARS = frozenset(_SUPERSCRIPT_MAP)
SUBSCRIPT_CHARS = frozenset(_SUBSCRIPT_MAP)


def normalize_superscript(ch: str) -> str | None:
    return _SUPERSCRIPT_MAP.get(ch)


def normalize_subscript(ch: str) -> str | None:
    return _SUBSCRIPT_MAP.get(ch)


def _centroid_in_bbox(
    cx: float, cy: float, bbox: tuple[float, float, float, float]
) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 <= cx <= x1) and (y0 <= cy <= y1)


def extract_region_glyphs(
    pdf_path: str | Path,
    page_num: int,
    bbox: tuple[float, float, float, float],
) -> list[Glyph]:
    """Re-extract pdfplumber ``page.chars`` inside ``bbox`` on ``page_num``.

    ``page_num`` is 1-indexed (cascade convention). ``bbox`` is top-left
    PDF-point space ``(x0, y0, x1, y1)``. Returns glyphs sorted by reading
    order (top, then x0). Best-effort: any failure (missing PDF, page out
    of range, pdfplumber error) returns ``[]`` so the policy degrades to
    the text fallback rather than crashing the cascade.
    """
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:  # pragma: no cover - dep always present in venv
        logger.debug("pdfplumber unavailable for glyph extract: %s", exc)
        return []

    path = Path(pdf_path)
    if not path.exists():
        return []
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        logger.debug("pdfplumber open failed (%s): %s", path, exc)
        return []
    try:
        if page_num < 1 or page_num > len(pdf.pages):
            return []
        page = pdf.pages[page_num - 1]
        chars = page.chars or []
        out: list[Glyph] = []
        for ch in chars:
            try:
                x0 = float(ch.get("x0"))
                x1 = float(ch.get("x1"))
                top = float(ch.get("top"))
                bottom = float(ch.get("bottom"))
            except (TypeError, ValueError):
                continue
            cx = (x0 + x1) / 2.0
            cy = (top + bottom) / 2.0
            if not _centroid_in_bbox(cx, cy, bbox):
                continue
            text = ch.get("text") or ""
            if not text:
                continue
            size = ch.get("size")
            try:
                size_f = float(size) if size is not None else (bottom - top)
            except (TypeError, ValueError):
                size_f = bottom - top
            out.append(
                Glyph(
                    text=text,
                    x0=x0,
                    x1=x1,
                    top=top,
                    bottom=bottom,
                    size=max(0.1, size_f),
                    source="pdfplumber",
                )
            )
        out.sort(key=lambda g: (round(g.top, 1), g.x0))
        return out
    except Exception as exc:
        logger.debug("page.chars extract failed (%s p%s): %s", path, page_num, exc)
        return []
    finally:
        try:
            pdf.close()
        except Exception:
            pass


def glyphs_from_src_text(src_text: str) -> list[Glyph]:
    """Build degenerate single-baseline glyphs from a flat string.

    No real geometry — every glyph gets a synthetic monospace box on one
    baseline. This preserves the *sequence* (so the emitter still produces
    valid linear MathML) and, crucially, preserves the unicode super/sub
    codepoints so :mod:`.structure_infer` recovers ``x²`` / ``H₂`` even
    without pdfplumber geometry. Whitespace is dropped (it carries no math
    structure in the flat string).
    """
    src = (src_text or "").strip()
    glyphs: list[Glyph] = []
    x = 0.0
    advance = 6.0
    size = 10.0
    top = 0.0
    bottom = 10.0
    for ch in src:
        if ch.isspace():
            x += advance
            continue
        # Skip combining marks / zero-width — they don't tokenise cleanly.
        if unicodedata.category(ch).startswith("M"):
            continue
        glyphs.append(
            Glyph(
                text=ch,
                x0=x,
                x1=x + advance,
                top=top,
                bottom=bottom,
                size=size,
                source="src_text",
            )
        )
        x += advance
    return glyphs


def recover_glyphs(
    *,
    src_text: str,
    page_num: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pdf_path: str | Path | None = None,
) -> tuple[list[Glyph], str]:
    """Recover glyphs for a math region, preferring real geometry.

    Returns ``(glyphs, source)`` where ``source`` is ``"pdfplumber"`` when
    real geometry was used, else ``"src_text"``. Falls back to the flat
    string whenever the PDF re-extract yields nothing (no path, page out of
    range, empty bbox).
    """
    if pdf_path is not None and page_num is not None and bbox is not None:
        glyphs = extract_region_glyphs(pdf_path, page_num, bbox)
        if glyphs:
            return glyphs, "pdfplumber"
    return glyphs_from_src_text(src_text), "src_text"


def glyph_geometry_summary(glyphs: Iterable[Glyph]) -> dict[str, Any]:
    """Compact JSON-serialisable geometry digest for the region payload."""
    gl = list(glyphs)
    if not gl:
        return {"n_glyphs": 0}
    sizes = sorted(g.size for g in gl)
    bottoms = sorted(g.bottom for g in gl)
    return {
        "n_glyphs": len(gl),
        "median_size": sizes[len(sizes) // 2],
        "min_size": sizes[0],
        "max_size": sizes[-1],
        "baseline_span": round(bottoms[-1] - bottoms[0], 2),
        "source": gl[0].source,
    }


__all__ = [
    "Glyph",
    "extract_region_glyphs",
    "glyphs_from_src_text",
    "recover_glyphs",
    "glyph_geometry_summary",
    "normalize_superscript",
    "normalize_subscript",
    "SUPERSCRIPT_CHARS",
    "SUBSCRIPT_CHARS",
]
