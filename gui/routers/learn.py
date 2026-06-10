"""Learner answer REST router — ``/api/learn/*`` (ask, courses, source).

Mounted at prefix ``/api/learn`` by ``gui.app``, so the paths here are relative.
Thin router: the grounded-answer wiring lives in ``gui.services.answer_service``
(lazy heavy imports), the server-side HTML rendering in
``gui.services.answer_render``, and the archived-source-page resolution +
sanitisation in ``gui.services.source_page`` (a sibling executor's module; this
router codes against its FROZEN signature and imports it lazily so the router
mounts cleanly even while that module is mid-build).

Endpoint contract (learner-facing):

* ``GET  "/courses"``        → ``[{slug, chunk_count}]`` (delegates the operator
  ``retrieval_service.list_courses``).
* ``POST "/ask"``            → ``{answer: GroundedAnswer.to_dict(), html: str}``
  with HTTP **200 for ALL six pipeline statuses** — a refusal or a citation-gate
  block is a successful pipeline *outcome*, i.e. data, not an HTTP error. Typed
  backend errors (engine down / index missing / compose failure) map to the
  503/502/500 contract below and ALSO carry a rendered learner-safe ``html``
  fragment so the JS swap path is uniform.
* ``GET  "/source/{slug}"``  → the archived course page wrapped in a viewer shell
  with a restrictive CSP, for citation-back navigation.

Errors surface as a typed ``{error, detail}`` body (plus ``html`` for the ask
path) with the right status — never a fabricated success / answer.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from gui.services import answer_render, answer_service

router = APIRouter()


# ----------------------------------------------------------------- request models


class AskRequest(BaseModel):
    """Body for ``POST "/ask"``.

    ``engine ∈ {auto, lexical, semantic, hybrid}`` (default ``"auto"``, resolved
    honestly in the service). Mirrors the operator ``QueryRequest`` permissive
    ``extra="allow"`` posture.
    """

    slug: str
    query: str
    engine: str = "auto"

    model_config = {"extra": "allow"}


# ------------------------------------------------------------------- error helper


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(
        status_code=status_code, content={"error": error, "detail": detail}
    )


def _error_with_fragment(
    status_code: int, error: str, copy_key: str, detail: str
) -> JSONResponse:
    """Typed ``{error, detail, html}`` error for the ask path.

    Carries a rendered learner-safe fragment (``copy_key`` → ``ERROR_COPY``) so
    the JS swap path is identical on success and failure: it always inserts
    ``body.html``; only the HTTP status drives the live-region message. The
    operator-actionable ``detail`` is JSON-only and never rendered.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "html": answer_render.render_error_fragment(copy_key),
        },
    )


# Exception ``__class__.__name__`` → (http_status, error_code, copy_key). Keyed
# by name so the router needs NO heavy imports (LibV2 / Trainforge) to match a
# typed error class — the WS3 § 3.4 typed-error contract.
_TYPED_ERROR_MAP: Dict[str, Any] = {
    "AnswerBackendUnavailable": (503, "answer_backend_unavailable", "error_backend_down"),
    "AnswerProviderNotLocal": (503, "answer_provider_not_local", "error_misconfigured"),
    "AnswerComposeError": (502, "answer_compose_failed", "error_compose"),
    "InvalidCitationError": (502, "answer_compose_failed", "error_compose"),
    "SemanticIndexMissing": (503, "semantic_index_unavailable", "error_index"),
    "SemanticIndexStale": (503, "semantic_index_unavailable", "error_index"),
    "EmbeddingBackendUnavailable": (503, "embedding_backend_unavailable", "error_index"),
}


def _course_exists(slug: str) -> bool:
    """Whether ``slug`` is a known LibV2 course (404 gate; never 404 a 0-chunk course).

    Keyed on ``retrieval_service.list_courses()`` (the operator-service precedent)
    — a known slug with zero chunks still exists.
    """
    from gui.services import retrieval_service  # noqa: PLC0415

    try:
        return any(c.get("slug") == slug for c in retrieval_service.list_courses())
    except Exception:  # noqa: BLE001 — unknown listing failure → treat as unknown
        return False


# ----------------------------------------------------------------------- endpoints


@router.get("/courses")
async def get_courses() -> Any:
    """List LibV2 courses with chunk counts (``[{slug, chunk_count}]``)."""
    from gui.services import retrieval_service  # noqa: PLC0415

    try:
        return retrieval_service.list_courses()
    except Exception as exc:  # noqa: BLE001 — unexpected filesystem/parse failure
        return _error(500, "list_courses_failed", str(exc))


@router.post("/ask")
async def ask(req: AskRequest) -> Any:
    """Answer a learner question; 200 + rendered fragment for all six statuses.

    Refusals and citation-gate blocks are 200s (the pipeline succeeded; the
    *answer* outcome is data). Typed backend errors map per ``_TYPED_ERROR_MAP``
    to 503/502, anything else to 500 — each error response still carries a
    rendered learner-safe ``html`` fragment.
    """
    if not req.query or not req.query.strip():
        return _error_with_fragment(
            422, "invalid_query", "error_generic", "query must be non-empty"
        )
    if not _course_exists(req.slug):
        return _error_with_fragment(
            404, "course_not_found", "error_generic", f"unknown course: {req.slug!r}"
        )

    try:
        answer = answer_service.ask(req.slug, req.query, req.engine)
    except Exception as exc:  # noqa: BLE001 — map typed pipeline errors by class name
        name = type(exc).__name__
        if name in _TYPED_ERROR_MAP:
            status_code, error_code, copy_key = _TYPED_ERROR_MAP[name]
            return _error_with_fragment(status_code, error_code, copy_key, str(exc))
        # A pre-E3 ``_retrieve`` guard raises a bare RuntimeError naming the
        # missing engine dependency — surface as an index-unavailable 503.
        if name == "RuntimeError":
            return _error_with_fragment(
                503, "engine_unavailable", "error_index", str(exc)
            )
        return _error_with_fragment(500, "ask_failed", "error_generic", str(exc))

    return {"answer": answer, "html": answer_render.render_answer_fragment(answer)}


@router.get("/source/{slug}")
async def get_source(slug: str, item_path: str = "", fragment: str = "") -> Any:
    """Serve an archived course page wrapped in the viewer shell (citation-back).

    Thin delegate to the frozen ``source_page.render_source_page`` contract; that
    service owns path-safety (sanitise → resolve → contain), cartridge HTML
    sanitisation, heading-id injection, and the CSP / nosniff headers. The import
    is lazy so the router mounts even while that module is mid-build.
    """
    try:
        from gui.services import source_page  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_viewer_unavailable", str(exc))

    if not item_path:
        return _error(422, "invalid_item_path", "item_path is required")

    from gui.services.answer_service import _libv2_root  # noqa: PLC0415

    try:
        result = source_page.render_source_page(
            slug,
            item_path,
            _libv2_root(),
            fragment=fragment or None,
        )
    except source_page.SourcePageError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected resolution/render failure
        return _error(500, "source_render_failed", str(exc))

    return Response(
        content=result.html,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


__all__ = ["router"]
