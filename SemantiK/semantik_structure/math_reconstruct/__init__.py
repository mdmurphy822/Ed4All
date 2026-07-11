"""Deterministic MathML reconstruction (T4).

Recovers the 2-D structure a flat reading-order string destroyed at math
extraction and rebuilds presentation-MathML deterministically, with a
LOCAL QLoRA math-specialist generative fallback and an accessible-uniform
last resort — so the raw glyph-soup ``<span class="math-source">``
fallback is eliminated. Gated by ``SEMANTIK_MATH_RECONSTRUCT`` (default
OFF -> byte-identical).

Pipeline:
    glyph_extract  -> per-glyph geometry (pdfplumber chars / src_text)
    structure_infer -> deterministic 2-D -> structure tree
    mathml_emit    -> presentation-MathML <math> + alttext
    policy         -> deterministic-first -> local-QLoRA -> uniform chain
"""
from __future__ import annotations

from .policy import (
    DEFAULT_MIN_CONFIDENCE,
    MathReconstructResult,
    reconstruct_math,
    resolve_math_reconstruct_mode,
    stamp_math_reconstruction,
)

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "MathReconstructResult",
    "reconstruct_math",
    "resolve_math_reconstruct_mode",
    "stamp_math_reconstruction",
]
