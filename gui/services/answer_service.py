"""Learner answer service — wraps the grounded-answer pipeline (D4/D5).

The router's seam to the grounded-answer stack. ``ask()`` resolves the engine
("auto" → hybrid-rrf when a vector index exists, else lexical — a cheap fs check,
never a silent downgrade for an explicit semantic/hybrid request), wires one
``DecisionCapture`` handle per request (capture failure never blocks the answer),
calls ``answer_library_question`` with ``with_groundedness=False`` (the learner
path never loads the ~750 MB NLI model — groundedness is operator-only
calibration metadata), and returns ``GroundedAnswer.to_dict()``.

``answer_library_question`` is the library-wide seam (W4.2): with
``ED4ALL_ANSWER_LIBRARY_WIDE`` off (default) it delegates VERBATIM to the
single-course ``answer_course_question`` on ``slug`` (byte-identical); with the
flag on it unions retrieval across the catalog's courses, keeping per-course
provenance on every citation.

Heavy imports (LibV2 / Trainforge / the grounded-answer stack) are LAZY — kept
inside the function bodies so this module imports cleanly without those
dependencies (the retrieval-service precedent). Typed pipeline errors are NOT
caught here; they propagate to the router, which owns the
exception → HTTP-status map. The service returns a dict only on success.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional


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
    ``manifest.json`` provenance file. No load, no torch — just a stat. Thin
    root+slug wrapper over the shared course-dir helper.
    """
    from lib.libv2_storage import has_vector_index  # noqa: PLC0415

    return has_vector_index(libv2_root / "courses" / slug)


def _resolve_engine(engine: str, libv2_root: Path, slug: str) -> str:
    """Resolve the requested engine to a concrete engine BEFORE the pipeline call.

    Delegates to ``lib.libv2_storage.resolve_auto_engine`` — the SINGLE resolver
    for this seam, shared with the ``libv2 answer-grounded`` CLI. This wrapper
    exists only to adapt (root, slug) to the shared helper's course-dir
    signature; the policy itself must not be re-stated here, because a local
    copy of it is exactly how the CLI and GUI drifted to different engines
    behind the same ``auto`` flag.

    ``"auto"`` → ``"hybrid-rrf"`` when a vector index manifest exists, else
    ``"lexical"``. Any explicit engine passes through verbatim — an explicit
    ``semantic`` against a missing index surfaces as a typed 503 in the router
    (anti-silent-degradation contract), never a downgrade.
    """
    from lib.libv2_storage import resolve_auto_engine  # noqa: PLC0415

    return resolve_auto_engine(engine, libv2_root / "courses" / slug)


def ask(
    slug: str,
    query: str,
    engine: str = "auto",
    *,
    library_wide: Optional[bool] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Answer a learner question; return ``GroundedAnswer.to_dict()``.

    Resolves the engine, wires a per-request ``DecisionCapture`` (best-effort —
    the existing grounded-answer emit sites use it; a capture-construction
    failure is swallowed so it never blocks an answer), and calls
    ``answer_course_question(..., with_groundedness=False)`` (D4). Typed pipeline
    errors propagate to the router untouched.

    ``library_wide`` (L3): an explicit request-level toggle that WINS over the
    ``ED4ALL_ANSWER_LIBRARY_WIDE`` env — ``None`` (default) resolves from the env
    flag (byte-identical to the prior behavior). ``on_progress`` (L4): an
    optional passages-first disclosure callback, fired once after the pre-LLM
    refusal gate with the retrieved passages, so a durable ask job can surface
    "here is what I found" while the compose call runs. Both default to the
    no-op path.

    ``ED4ALL_ANSWER_ASSESSMENT_GUARD`` (L2, three-valued off|shadow|on, default
    OFF ⇒ byte-identical): when ``on``, a question that matches a known course
    assessment stem short-circuits BEFORE the compose and returns a
    redirect-with-hint envelope (never a hard refusal — "the tutor won't do your
    homework"). ``shadow`` answers normally but records the would-have-matched
    signal. ``off`` leaves this path untouched.
    """
    from lib.retrieval import assessment_guard  # noqa: PLC0415
    from lib.retrieval.library_wide import (  # noqa: PLC0415
        answer_library_question,
    )

    libv2_root = _libv2_root()
    resolved_engine = _resolve_engine(engine, libv2_root, slug)

    capture = _build_capture(slug)

    # L2 assessment guard: OFF (default) leaves the answer path byte-identical
    # (no stem load, no capture, no signal). ``shadow`` / ``on`` evaluate the
    # incoming question against the course's assessment stems; ``on`` + a match
    # returns the redirect envelope WITHOUT dispatching the compose.
    guard_mode = assessment_guard.resolve_guard_mode()
    guard_outcome = None
    if guard_mode != assessment_guard.GUARD_OFF:
        try:
            guard_outcome = assessment_guard.evaluate_guard(
                libv2_root, slug, query, guard_mode, capture=capture
            )
        except Exception:  # noqa: BLE001 — guard is advisory; never block the answer
            guard_outcome = None
        if (
            guard_outcome is not None
            and guard_mode == assessment_guard.GUARD_ON
            and guard_outcome.matched
        ):
            return assessment_guard.build_redirect_envelope(
                libv2_root, slug, query, resolved_engine, guard_outcome
            )

    # ``answer_library_question`` is the library-wide seam: when
    # ``ED4ALL_ANSWER_LIBRARY_WIDE`` is off (default) AND no explicit course set
    # is passed, it delegates VERBATIM to ``answer_course_question`` on ``slug``
    # (byte-identical single-course path). When the flag is on it unions
    # retrieval across the catalog's courses, keeping per-course provenance on
    # every citation. ``library_wide`` threads the request-level override
    # (``None`` => resolve from the env flag).
    result = answer_library_question(
        libv2_root,
        slug,
        query,
        engine=resolved_engine,
        capture=capture,
        with_groundedness=False,
        library_wide=library_wide,
        on_progress=on_progress,
    )
    payload = result.to_dict()
    # ``shadow`` (and an ``on`` no-match) answers normally but stamps the
    # would-have-matched signal so an operator can measure guard hit-rate.
    if guard_outcome is not None:
        payload["assessment_guard"] = guard_outcome.signal()
    return payload


def source_materials_enabled(slug: str) -> bool:
    """Resolve the ``ED4ALL_SOURCE_MATERIALS`` toggle for ``slug`` (per-request).

    Threads the operator toggle (env flag + per-course ``manifest.json::viewer``
    override) into the answer renderer so the citation-side original-source + PDF
    deep links are suppressed when source materials are disabled (§2.5). Best-
    effort: a resolution failure (unknown course / unreadable manifest / the
    source-materials module unavailable) defaults to the documented default-on
    posture so a missing manifest never strips provenance links from an answer.
    """
    try:
        from gui.services import source_materials as _sm  # noqa: PLC0415

        libv2_root = _libv2_root()
        course_dir = libv2_root / "courses" / slug
        return _sm.is_enabled(course_dir)
    except Exception:  # noqa: BLE001 — toggle resolution is advisory; default on
        return True


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


__all__ = ["ask", "source_materials_enabled"]
