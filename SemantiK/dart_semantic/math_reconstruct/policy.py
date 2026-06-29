"""Deterministic-first math-reconstruction policy.

Owner directive: **DETERMINISTIC-FIRST**. The reconstruction chain, in
strict priority order, is

    1. deterministic 2-D -> presentation-MathML  (:mod:`.structure_infer`
       + :mod:`.mathml_emit`)              [no model, geometry-driven]
    2. local QLoRA **math specialist** generative fallback
       (on-device, license-clean — NEVER the hosted 70B)
    3. accessible-uniform <math> last resort (:func:`mathml_emit
       .accessible_uniform_mathml`)        [zero raw glyph soup, ever]

EVERY candidate is validated through
:func:`dart_semantic.gates.mathml_check.validate_mathml` before it is
accepted; a candidate that fails the gate falls through to the next tier.
The last resort passes the gate by construction, so a region always ships
valid, screen-reader-addressable MathML — the raw ``<span class=
"math-source">`` glyph soup is eliminated.

Gated by ``SEMANTIK_MATH_RECONSTRUCT`` (default OFF -> byte-identical: no
stamping at Stage 5, the legacy soup fallback at assembly). Resolution is
parse-with-fallback, mirroring the SemantiK flag posture.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .glyph_extract import recover_glyphs, glyph_geometry_summary
from .mathml_emit import accessible_uniform_mathml, alttext_from_src, emit_mathml
from .structure_infer import infer_structure

logger = logging.getLogger(__name__)

_FLAG_ENV = "SEMANTIK_MATH_RECONSTRUCT"
_TRUTHY = {"1", "true", "yes", "on"}

# A deterministic candidate below this confidence is NOT trusted — the
# region routes to the local generative specialist instead. Tuned so the
# reconstructor's confident shapes (simple 0.88 / scripts 0.92 / fraction
# 0.85 / blind-linear 0.80) pass and its self-flagged "unsupported"
# linearisations (0.30-0.35) route on.
DEFAULT_MIN_CONFIDENCE = 0.6

# Generative-fallback type: src_text -> MathML fragment str (or None).
GenerativeFn = Callable[[str], "str | None"]


def resolve_math_reconstruct_mode() -> bool:
    """Return True when deterministic MathML reconstruction is enabled.

    Reads ``SEMANTIK_MATH_RECONSTRUCT``. **Default OFF** — only an explicit
    truthy value (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive)
    turns it on; unset / falsey / garbage -> off (parse-with-fallback,
    mirroring ``reviewer.resolve_block_review_mode``).
    """
    raw = (os.environ.get(_FLAG_ENV) or "").strip().lower()
    return raw in _TRUTHY


@dataclass
class MathReconstructResult:
    """The accepted candidate + provenance for the region payload."""

    mathml: str
    alttext: str
    confidence: float
    method: str  # "deterministic" | "generative_local" | "accessible_uniform"
    ok: bool
    geometry_source: str = "src_text"
    complexity: str = ""
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "mathml": self.mathml,
            "alttext": self.alttext,
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "geometry_source": self.geometry_source,
            "complexity": self.complexity,
            "notes": list(self.notes),
        }


def _validate(mathml: str) -> bool:
    """Run the Stage-7 MathML gate; True iff the fragment passes."""
    try:
        from ..gates.mathml_check import validate_mathml  # noqa: WPS433

        return bool(validate_mathml(mathml).passed)
    except Exception as exc:  # gate import / lxml failure -> reject candidate.
        logger.debug("mathml gate validation raised: %s", exc)
        return False


def reconstruct_math(
    *,
    src_text: str,
    display: bool = True,
    glyph_density_features: dict[str, Any] | None = None,
    page_num: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pdf_path: str | Path | None = None,
    generative_fn: GenerativeFn | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> MathReconstructResult:
    """Run the full deterministic-first -> generative -> uniform chain.

    ``generative_fn`` is the LOCAL QLoRA math-specialist callable (see
    :func:`build_local_math_generative_fn`); pass ``None`` to skip the
    generative tier (Stage-5 stamping does this — no model load at
    structure-graph time; the real generative tier is Stage 6's
    local-by-default specialist).
    """
    alt = alttext_from_src(src_text)

    # --- Tier 1: deterministic. -----------------------------------------
    glyphs, source = recover_glyphs(
        src_text=src_text, page_num=page_num, bbox=bbox, pdf_path=pdf_path
    )
    infer = infer_structure(
        glyphs,
        glyph_density_features=glyph_density_features,
        geometry_source=source,
    )
    if infer.can_reconstruct and infer.confidence >= min_confidence:
        mathml = emit_mathml(infer.root, alttext=alt, display=display)
        if _validate(mathml):
            return MathReconstructResult(
                mathml=mathml,
                alttext=alt,
                confidence=infer.confidence,
                method="deterministic",
                ok=True,
                geometry_source=source,
                complexity=infer.complexity,
                notes=infer.notes,
            )
        infer.notes.append("deterministic candidate failed mathml gate")

    # --- Tier 2: local QLoRA generative (never the hosted 70B). ----------
    if generative_fn is not None:
        try:
            cand = generative_fn(src_text)
        except Exception as exc:
            logger.debug("local math generative_fn raised: %s", exc)
            cand = None
        if cand and _validate(cand):
            return MathReconstructResult(
                mathml=cand,
                alttext=alt,
                confidence=0.5,
                method="generative_local",
                ok=True,
                geometry_source=source,
                complexity="generative",
                notes=[*infer.notes, "local generative specialist"],
            )

    # --- Tier 3: accessible-uniform last resort (NEVER raw soup). --------
    mathml = accessible_uniform_mathml(src_text, display=display)
    ok = _validate(mathml)
    return MathReconstructResult(
        mathml=mathml,
        alttext=alt,
        confidence=0.0,
        method="accessible_uniform",
        ok=ok,
        geometry_source=source,
        complexity="uniform",
        notes=[*infer.notes, "accessible-uniform last resort"],
    )


def stamp_math_reconstruction(
    payload: dict[str, Any],
    *,
    page_num: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pdf_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Run DETERMINISTIC-ONLY reconstruction and, when it produces a
    gate-valid candidate, stamp ``payload['math_reconstruct']``.

    Called at Stage 5 (structure-graph) where loading the generative model
    would be wrong — so ``generative_fn`` is omitted. Only a successful
    DETERMINISTIC candidate is stamped; a region the reconstructor cannot
    handle is left unstamped so Stage-6's local generative specialist runs
    and, failing that, ``fallbacks.fallback_math`` emits the
    accessible-uniform last resort. Returns the stamped dict or ``None``.

    Best-effort: any exception is swallowed (the cascade must never crash
    on an opt-in reconstruction pass).
    """
    try:
        src_text = payload.get("src_text") or ""
        if not str(src_text).strip():
            return None
        result = reconstruct_math(
            src_text=str(src_text),
            display=bool(payload.get("display", True)),
            glyph_density_features=payload.get("glyph_density_features") or {},
            page_num=page_num,
            bbox=bbox,
            pdf_path=pdf_path,
            generative_fn=None,
        )
        if result.method == "deterministic" and result.ok:
            stamped = result.to_payload()
            stamped["glyph_geometry"] = glyph_geometry_summary(
                recover_glyphs(
                    src_text=str(src_text),
                    page_num=page_num,
                    bbox=bbox,
                    pdf_path=pdf_path,
                )[0]
            )
            payload["math_reconstruct"] = stamped
            return stamped
        return None
    except Exception as exc:
        logger.debug("stamp_math_reconstruction failed: %s", exc)
        return None


def build_local_math_generative_fn() -> GenerativeFn | None:
    """Construct the LOCAL QLoRA math-specialist generative fallback.

    Honours the owner directive: the generative tier is the ON-DEVICE,
    license-clean QLoRA math specialist — **never** the hosted 70B. It
    forces the local GGUF runtime (``make_runtime("real", force_local=
    True)``) so an endpoint provider selection / displace flag cannot route
    math through the cloud seat. Returns ``None`` when the local specialist
    cannot be built (no GGUF weights) — the policy then falls straight to
    the accessible-uniform last resort.

    Best-effort + lazy: this is NOT invoked by the default Stage-5 stamping
    (which is deterministic-only); it is the wiring an operator-driven
    generative pass would use to keep the math fallback local.
    """
    try:
        from ..qwen_specialists.prompts import build_math_request  # noqa: WPS433
        from ..qwen_specialists.runtime import make_runtime  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("local math specialist imports unavailable: %s", exc)
        return None

    try:
        runtime = make_runtime("real", force_local=True)
    except Exception as exc:
        logger.info(
            "local math specialist unavailable (%s); generative tier off, "
            "policy falls to accessible-uniform last resort", exc,
        )
        return None

    def _generate(src_text: str) -> str | None:
        try:
            class _R:  # minimal region shim for build_math_request
                payload = {"src_text": src_text, "display": True,
                           "glyph_density_features": {}}
                feature_block_indices = ()

            req = build_math_request(_R(), [])
            prompt = (req.payload or {}).get("prompt") or ""
            if not prompt:
                return None
            outs = runtime.generate(
                prompt, n=1, temperature=0.0, top_p=1.0, max_tokens=512,
            )
            cand = (outs[0] if outs else "") or ""
            return _extract_math_fragment(cand)
        except Exception as exc:
            logger.debug("local math generate failed: %s", exc)
            return None

    return _generate


def _extract_math_fragment(text: str) -> str | None:
    """Pull the first ``<math ...>...</math>`` fragment from a completion."""
    lo = text.find("<math")
    hi = text.rfind("</math>")
    if lo == -1 or hi == -1 or hi < lo:
        return None
    return text[lo:hi + len("</math>")]


__all__ = [
    "resolve_math_reconstruct_mode",
    "MathReconstructResult",
    "reconstruct_math",
    "stamp_math_reconstruction",
    "build_local_math_generative_fn",
    "DEFAULT_MIN_CONFIDENCE",
]
