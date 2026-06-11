"""Learner answer service — wraps the grounded-answer pipeline (D4/D5).

The router's seam to the grounded-answer stack. ``ask()`` resolves the engine
("auto" → hybrid-rrf when a vector index exists, else lexical — a cheap fs check,
never a silent downgrade for an explicit semantic/hybrid request), wires one
``DecisionCapture`` handle per request (capture failure never blocks the answer),
calls ``answer_course_question`` with ``with_groundedness=False`` (the learner
path never loads the ~750 MB NLI model — groundedness is operator-only
calibration metadata), and returns ``GroundedAnswer.to_dict()``.

Heavy imports (LibV2 / Trainforge / the grounded-answer stack) are LAZY — kept
inside the function bodies so this module imports cleanly without those
dependencies (the retrieval-service precedent). Typed pipeline errors are NOT
caught here; they propagate to the router, which owns the
exception → HTTP-status map. The service returns a dict only on success.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT`` (test seam).

    A local 3-line copy of the env-honoring resolver — deliberately NOT imported
    from ``retrieval_service`` (that module is churned in the working tree). Read
    at call time so tests can redirect via the env var.
    """
    override = os.environ.get("ED4ALL_LIBV2_ROOT")
    if override:
        return Path(override)
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    return Path(LIBV2_PATH)


def _has_vector_index(libv2_root: Path, slug: str) -> bool:
    """Cheap fs check: does ``courses/<slug>/vector_index/manifest.json`` exist?

    The honest "auto" resolution signal (D5): a built semantic index leaves a
    ``manifest.json`` provenance file. No load, no torch — just a stat.
    """
    manifest = libv2_root / "courses" / slug / "vector_index" / "manifest.json"
    return manifest.is_file()


def _resolve_engine(engine: str, libv2_root: Path, slug: str) -> str:
    """Resolve the requested engine to a concrete engine BEFORE the pipeline call.

    ``"auto"`` → ``"hybrid-rrf"`` when a vector index manifest exists for the
    course, else ``"lexical"``. hybrid-rrf (BM25 fused with semantic via
    reciprocal-rank fusion) is the benchmark-selected default — pure semantic
    never beat the BM25 baseline, so ``auto`` routes through the fused engine
    when an index is available. Any explicit engine (``lexical`` / ``semantic``
    / ``hybrid-rrf``) passes through verbatim — an explicit ``semantic`` against
    a missing index surfaces as a typed 503 in the router
    (anti-silent-degradation contract), never a downgrade.
    """
    requested = (engine or "auto").strip().lower()
    if requested == "auto":
        return "hybrid-rrf" if _has_vector_index(libv2_root, slug) else "lexical"
    return requested


def ask(slug: str, query: str, engine: str = "auto") -> Dict[str, Any]:
    """Answer a learner question; return ``GroundedAnswer.to_dict()``.

    Resolves the engine, wires a per-request ``DecisionCapture`` (best-effort —
    the existing grounded-answer emit sites use it; a capture-construction
    failure is swallowed so it never blocks an answer), and calls
    ``answer_course_question(..., with_groundedness=False)`` (D4). Typed pipeline
    errors propagate to the router untouched.
    """
    from lib.retrieval.grounded_answer import (  # noqa: PLC0415
        answer_course_question,
    )

    libv2_root = _libv2_root()
    resolved_engine = _resolve_engine(engine, libv2_root, slug)

    capture = _build_capture(slug)

    result = answer_course_question(
        libv2_root,
        slug,
        query,
        engine=resolved_engine,
        capture=capture,
        with_groundedness=False,
    )
    return result.to_dict()


def _build_capture(slug: str) -> Any:
    """Construct a per-request DecisionCapture; never raise on failure.

    The capture handle is threaded through the grounded-answer stack's existing
    refusal / citation-gate / composer emit sites (no new emit sites here). A
    construction failure (e.g. an unwritable captures dir) must not block the
    learner's answer, so it degrades to ``None``.
    """
    try:
        from lib.decision_capture import DecisionCapture  # noqa: PLC0415

        return DecisionCapture(
            course_code=slug, phase="libv2-answer", tool="learner-ui"
        )
    except Exception:  # noqa: BLE001 — capture is advisory; never block the answer
        return None


__all__ = ["ask"]
