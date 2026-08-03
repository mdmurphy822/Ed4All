"""Stage 6: Content enrichment. Multiple specialized **local** tools, one per subtask.

Contract:
    enrich(blocks: list[ResolvedBlock]) -> list[ResolvedBlock]

The pipeline is fully self-contained: no external LLM APIs (a standing
project constraint). Each enrichment subtask is handled by a small local
model or deterministic code.

Subtask routing:
    FIGURE                -> alt_text + extended_description (if complex)
    TABLE*                -> table_summary (aria-describedby target)
    Any text-bearing block -> language detection when len > 40 chars
    CODE_BLOCK            -> source-language tag for syntax
    math content          -> MathML passthrough or Pix2Tex

v1 implementations:
    language_detect:      lingua-language-detector (CPU, ~1ms/block)
    alt_text:             Florence-2-base image captioner (230M, Apache 2.0,
                          ~0.5GB VRAM). Runs on CPU for small batches.
    table_summary:        deterministic row/column summary fallback; future
                          upgrade to Florence-2 when row images are extracted.
    extended_description: Florence-2 with "caption_detailed" prompt for now;
                          future upgrade to a dedicated chart-understanding model.
    math:                 passthrough of MathML when present in the source
                          (e.g. ar5iv markup); Pix2Tex for image-only
                          equations (future).

All models above are Apache 2.0 or MIT. No Claude, OpenAI, Gemini, or
any other LLM-API dependency at runtime.
"""
from __future__ import annotations

from functools import lru_cache

from .classify import Role
from .types import Enrichment, ResolvedBlock


_MIN_LANG_DETECT_CHARS = 40


def enrich(blocks: list[ResolvedBlock]) -> list[ResolvedBlock]:
    """Walk blocks, attach enrichments in place. Returns the same list."""
    detector = _lingua_detector()
    for b in blocks:
        enr = b.enrichment or Enrichment()
        text = b.classified.features.raw.text
        # Language detection: every text-bearing block over a threshold.
        if text and len(text) >= _MIN_LANG_DETECT_CHARS:
            lang = _detect_language(detector, text)
            if lang:
                enr.language = lang

        role = b.classified.role
        if role == Role.FIGURE.value:
            # Figure alt-text is produced by the Stage-6b cascade path
            # (`figures.captioner.caption_figure_regions` →
            # `assembler/fallbacks.fallback_figure`), which has the image
            # bytes. The v1 `pipeline.py` path does not caption figures by
            # design, so it records a warning here instead.
            enr.warnings.append("figure alt_text not generated (stage-6 TODO)")
        elif role in (Role.TABLE.value, Role.TABLE_ROW.value,
                      Role.TABLE_HEADER_CELL.value, Role.TABLE_DATA_CELL.value):
            # Summary happens at table-group level; we skip per-cell.
            pass

        b.enrichment = enr
    return blocks


# ---------- language detection ----------

@lru_cache(maxsize=1)
def _lingua_detector():
    """Cache a single detector instance. First call downloads models."""
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return None
    # Build with all languages but the low-accuracy mode for speed;
    # documents are long enough that accuracy is fine.
    return (LanguageDetectorBuilder
            .from_all_languages()
            .with_low_accuracy_mode()
            .build())


def _detect_language(detector, text: str) -> str | None:
    if detector is None:
        return None
    detected = detector.detect_language_of(text)
    if detected is None:
        return None
    # lingua returns Language enum; use iso_code_639_1.name.lower() for BCP-47.
    try:
        return detected.iso_code_639_1.name.lower()
    except Exception:
        return None


# ---------- local enrichment helpers ----------

def summarize_table(rows: list[list[str]]) -> str:
    """Deterministic short description for aria-describedby.

    For v1, builds a summary like "Table with N rows and M columns.
    Column headers: ...". No model needed. Future upgrade to a
    table-aware LM once we have more sophisticated requirements.
    """
    if not rows:
        return "Empty table."
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows) if rows else 0
    header = rows[0] if rows else []
    header_text = ", ".join(header[:6])
    if len(header) > 6:
        header_text += f", and {len(header) - 6} more"
    return (f"Table with {n_rows} rows and {n_cols} columns. "
            f"Column headers: {header_text}.")
