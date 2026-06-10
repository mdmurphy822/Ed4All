"""``/api/courses`` router — Courses & Topics vertical of the control-plane GUI.

Mounted at ``/api/courses`` (see ``gui/app.py::_ROUTER_MOUNTS``), so every route
path here is relative to that prefix. Thin web layer over
``gui.services.course_service`` (the import-safe, fastapi-free data layer).

Routes::

    GET   ""                          → list_courses
    GET   "/{course_id}"              → one course summary
    GET   "/{course_id}/objectives"  → objectives doc (id+text normalized)
    PUT   "/{course_id}/objectives"  → write reuse-compatible objectives JSON
    GET   "/{course_id}/classification"
    PATCH "/{course_id}/classification"

Errors are surfaced as ``{error, detail}`` with the right status (404 unknown
course / missing file, 422 validation). Never a fabricated success.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gui.services import course_service

router = APIRouter()


# --------------------------------------------------------------------------- #
# Pydantic models (inline per the build contract)
# --------------------------------------------------------------------------- #


class CourseSummary(BaseModel):
    """One row of the course list / a single course summary."""

    course_name: str
    project_id: Optional[str] = None
    slug: Optional[str] = None
    duration_weeks: Optional[int] = None
    terminal_count: int = 0
    chapter_count: int = 0
    topics: List[str] = Field(default_factory=list)
    subdomains: List[str] = Field(default_factory=list)


class ObjectiveItem(BaseModel):
    """A single learning objective as the frontend authors it.

    ``id`` must be a canonical LO id (``TO-NN`` / ``CO-NN``); ``text`` is the
    objective statement. Extra fields (``bloom_level`` etc.) are allowed and
    preserved on save.
    """

    id: str = Field(..., description="Canonical LO id, e.g. TO-01 / CO-03")
    text: str = ""

    model_config = {"extra": "allow"}


class ObjectivesPut(BaseModel):
    """Body for ``PUT /{course_id}/objectives``."""

    terminal_objectives: List[ObjectiveItem] = Field(default_factory=list)
    chapter_objectives: List[ObjectiveItem] = Field(default_factory=list)
    mint_method: Optional[str] = None
    duration_weeks: Optional[int] = None


class ClassificationPatch(BaseModel):
    """Body for ``PATCH /{course_id}/classification`` (all fields optional)."""

    division: Optional[str] = None
    primary_domain: Optional[str] = None
    secondary_domains: Optional[List[str]] = None
    subdomains: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    subtopics: Optional[List[str]] = None
    tags: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Error helper
# --------------------------------------------------------------------------- #


def _error(status: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error response."""
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get("")
def list_courses(limit: int = Query(default=200, ge=0)) -> List[Dict[str, Any]]:
    """List every known course across Courseforge exports + LibV2 courses.

    Empty corpora yield ``[]`` — never invented data. ``limit`` caps the number
    of rows returned (default 200, ``0`` returns none); the slice is applied in
    the router so the service layer stays unbounded and prior callers are
    unaffected by the default cap.
    """
    return course_service.list_courses()[:limit]


@router.get("/{course_id}", response_model=None)
def get_course(course_id: str) -> Dict[str, Any] | JSONResponse:
    """Return the summary row for one course (404 if unknown)."""
    target = course_id.casefold()
    for course in course_service.list_courses():
        if (
            str(course.get("project_id") or "").casefold() == target
            or str(course.get("course_name") or "").casefold() == target
            or str(course.get("slug") or "").casefold() == target
        ):
            return course
    return _error(404, "course_not_found", f"no course resolves to {course_id!r}")


@router.get("/{course_id}/objectives", response_model=None)
def get_objectives(course_id: str) -> Dict[str, Any] | JSONResponse:
    """Return the objectives doc (terminal/chapter normalized to id+text)."""
    try:
        return course_service.get_objectives(course_id)
    except FileNotFoundError as exc:
        return _error(404, "objectives_not_found", str(exc))


@router.put("/{course_id}/objectives", response_model=None)
def put_objectives(
    course_id: str, body: ObjectivesPut
) -> Dict[str, Any] | JSONResponse:
    """Write a reuse-compatible ``synthesized_objectives.json`` for the course.

    Maps ``text`` → ``statement`` and stamps
    ``mint_method="user_supplied_objectives_json"`` so
    ``ed4all run --reuse-objectives`` accepts the result. Invalid LO ids →
    422; a course with no Courseforge export to author into → 404.
    """
    doc = body.model_dump(exclude_none=True)
    try:
        return course_service.save_objectives(course_id, doc)
    except ValueError as exc:
        return _error(422, "invalid_objectives", str(exc))
    except FileNotFoundError as exc:
        return _error(404, "course_not_found", str(exc))


@router.get("/{course_id}/classification", response_model=None)
def get_classification(course_id: str) -> Dict[str, Any] | JSONResponse:
    """Return the LibV2 classification block (404 if no manifest)."""
    try:
        return course_service.get_classification(course_id)
    except FileNotFoundError as exc:
        return _error(404, "classification_not_found", str(exc))


@router.patch("/{course_id}/classification", response_model=None)
def patch_classification(
    course_id: str, body: ClassificationPatch
) -> Dict[str, Any] | JSONResponse:
    """Merge a classification patch into the LibV2 manifest.

    Only the supplied fields are updated; the merged block is schema-validated
    (best-effort) before write. Schema violation → 422; no manifest → 404.

    Uses ``exclude_unset`` (not ``exclude_none``) so an OMITTED field is left
    untouched while a field EXPLICITLY set to ``null`` is forwarded to
    ``save_classification`` to clear the stored value.
    """
    patch = body.model_dump(exclude_unset=True)
    try:
        return course_service.save_classification(course_id, patch)
    except ValueError as exc:
        return _error(422, "invalid_classification", str(exc))
    except FileNotFoundError as exc:
        return _error(404, "classification_not_found", str(exc))


__all__ = ["router"]
