"""Studio library + IMSCC course-viewer REST router (``/api/...``).

Mounted at the bare ``/api`` prefix by ``gui.app`` so the absolute paths declared
here (``/api/library``, ``/api/courses/{course_id}/...``) are authoritative. The
whole archived-course-serving surface lives in ``gui.services.imscc_service``
(lazy heavy imports); this router is a thin typed-error adapter over its frozen
contract, mirroring ``gui.routers.learn``'s relationship to ``source_page``.

Endpoint contract (Studio learner-facing, read-only):

* ``GET "/library"`` → ``[{slug, title, page_count, has_vector_index[,
  course_status]}]`` — course cards for every archived course carrying an IMSCC
  cartridge (``course_status`` present only when the course is governed).
* ``GET "/library/{course_id}/scorecard"`` → per-course quality scorecard
  (governance + eval sections, each present-if-available).
* ``GET "/courses/{course_id}/manifest"`` → ``{slug, title, items: [...]}`` — the
  course organization tree (recursive module/unit/page nodes).
* ``GET "/courses/{course_id}/page?item=<id>"`` → sanitised page HTML + CSP.
* ``GET "/courses/{course_id}/asset?path=<rel>"`` → whitelisted static asset
  (path-traversal-safe; serves user-uploaded archive bytes, sanitised posture).

Errors surface as a typed ``{error, detail}`` body with the mapped status —
never a fabricated page / tree.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

router = APIRouter()


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@router.get("/library")
async def get_library() -> Any:
    """List archived courses that carry a viewable IMSCC cartridge."""
    try:
        from gui.services import imscc_service  # noqa: PLC0415

        return imscc_service.list_library()
    except Exception as exc:  # noqa: BLE001 — unexpected fs/parse failure
        return _error(500, "library_failed", str(exc))


@router.get("/library/{course_id}/scorecard")
async def get_scorecard(course_id: str) -> Any:
    """Compose the per-course quality scorecard (T2).

    Reads the course's governance / eval artifacts request-time and returns a
    section-per-artifact scorecard; an un-evaluated course still returns 200
    with every section marked ``not yet evaluated`` (never a fabricated number).
    404 unknown course, 422 malformed slug.
    """
    try:
        from gui.services import scorecard_service  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "scorecard_unavailable", str(exc))

    try:
        return scorecard_service.build_scorecard(course_id)
    except scorecard_service.ScorecardServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected fs/parse failure
        return _error(500, "scorecard_failed", str(exc))


@router.delete("/library/{course_id}")
async def delete_course(course_id: str, confirm: str = "") -> Any:
    """Permanently delete an archived course (Marketable-v1 D5).

    DESTRUCTIVE + confirmation-gated: requires a ``confirm=<slug>`` query param
    echoing ``course_id`` exactly (a 400 ``confirm_required`` otherwise) so a
    stray DELETE can never wipe a course without the caller re-stating the
    target. Delegates to the shared ``LibV2.tools.libv2.remove.remove_course``
    service (same logic the ``libv2 remove`` CLI runs) — containment-checked,
    scoped strictly to the resolved course dir.

    When an operator token is configured, the ``OperatorTokenMiddleware``
    classifies this DELETE as operator-gated (GET ``/api/library`` stays open);
    see ``gui.auth``.
    """
    if confirm != course_id:
        return _error(
            400,
            "confirm_required",
            "destructive delete requires ?confirm=<slug> echoing the course id",
        )
    try:
        from LibV2.tools.libv2 import remove as _remove  # noqa: PLC0415
        from gui.services.imscc_service import _libv2_root  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "remove_unavailable", str(exc))

    try:
        result = _remove.remove_course(_libv2_root(), course_id)
    except _remove.CourseRemovalError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected fs failure
        return _error(500, "remove_failed", str(exc))

    # Drop the deleted course's cached disk size so a fresh /api/library load
    # doesn't keep reporting it (it's gone from the listing anyway, but be tidy).
    try:
        from gui.services.imscc_service import _invalidate_disk_cache  # noqa: PLC0415

        _invalidate_disk_cache(result.course_dir)
    except Exception:  # noqa: BLE001 — cache hygiene is best-effort
        pass

    return {"removed": True, **result.to_dict()}


@router.get("/courses/{course_id}/manifest")
async def get_manifest(course_id: str) -> Any:
    """Return the course organization tree (manifest) as JSON."""
    try:
        from gui.services import imscc_service  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "viewer_unavailable", str(exc))

    try:
        return imscc_service.get_manifest(course_id)
    except imscc_service.IMSCCServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected resolution/parse failure
        return _error(500, "manifest_failed", str(exc))


@router.get("/courses/{course_id}/page")
async def get_page(course_id: str, item: str = "") -> Any:
    """Serve one sanitised cartridge page resolved via its manifest resource id."""
    try:
        from gui.services import imscc_service  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "viewer_unavailable", str(exc))

    try:
        result = imscc_service.get_page(course_id, item)
    except imscc_service.IMSCCServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected render failure
        return _error(500, "page_failed", str(exc))

    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


@router.get("/courses/{course_id}/source-pdf")
async def get_source_pdf(course_id: str, file: str = "", page: int = 1) -> Any:
    """Serve one archived source-PDF page (B4 provenance chain final hop).

    Locates the raw PDF archived under ``courses/<id>/source/pdf/`` (resolving a
    sourceId-derived ``file`` hint via the archived source sidecars, whitelisted
    against the dir listing), and returns the SINGLE requested page as
    ``application/pdf`` when a PDF lib is available — otherwise the whole PDF
    (the browser deep-links to ``#page=N``). 404s with an explanation when no
    source PDFs are archived or the mapping is unresolvable.
    """
    try:
        from gui.services import source_pdf as _svc  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_pdf_unavailable", str(exc))

    try:
        result = _svc.serve_source_pdf_page(course_id, file, page)
    except _svc.SourcePdfError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected resolution/read failure
        return _error(500, "source_pdf_failed", str(exc))

    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


@router.get("/courses/{course_id}/source-materials")
async def get_source_materials(course_id: str) -> Any:
    """List the course's original archived source materials (accessible HTML + PDFs).

    ``{slug, enabled, dart_docs: [...], pdfs: [...]}``. When the operator toggle
    (``ED4ALL_SOURCE_MATERIALS`` / the per-course ``manifest.json::viewer`` key)
    is off, ``enabled`` is ``false`` and both lists are empty. 404 unknown course,
    422 invalid slug.
    """
    try:
        from gui.services import source_materials as _svc  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_materials_unavailable", str(exc))

    try:
        return _svc.list_source_materials(course_id)
    except _svc.SourceMaterialsError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected fs/parse failure
        return _error(500, "source_materials_failed", str(exc))


@router.get("/courses/{course_id}/source-doc")
async def get_source_doc(
    course_id: str, doc: str = "", ref: str = "", page: str = ""
) -> Any:
    """Serve one sanitized, block-anchored archived source document.

    ``doc`` is whitelisted against the inventory stem listing (never a path);
    serve-time passes inject ``id="semantik-{block_id}"`` block anchors, per-page
    ``id="page-{N}"`` anchors, + heading ids and rewrite ``{stem}_figures/``
    image srcs to the ``/source-doc-asset`` endpoint. ``ref`` is the requested
    anchor; ``page`` is the requested physical page (both informational — the
    ``#fragment`` is built by the caller; the deep links append
    ``#semantik-<anchor>`` or fall back to ``#page-N``). The ``id="page-N"``
    anchors are injected
    unconditionally, so a ``#page-N`` fragment resolves whether or not ``?page``
    rides along; accepting it here keeps the param a documented part of the
    contract (and absorbs it rather than 422-ing on an unexpected query arg).
    403 when source materials are disabled; 404 when the source doc isn't archived
    (the emit-then-resolve citation hop); restrictive CSP.
    """
    try:
        from gui.services import source_materials as _svc  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_materials_unavailable", str(exc))

    # Resolve ``?page=N`` to the informational ``page-N`` ref when the caller
    # supplied no usable ``#semantik-<anchor>`` (``ref``), so the served banner /
    # provenance reflects the requested page. The actual scroll target is the
    # serve-time-injected ``id="page-N"`` anchor (unconditional), reached via the
    # ``#page-N`` fragment the deep links build. Anti-fabrication: only a
    # positive integer page is honored; garbage is ignored (byte-identical to
    # the no-page request).
    resolved_ref = ref or None
    if not resolved_ref and page:
        try:
            pg = int(str(page).strip())
        except (TypeError, ValueError):
            pg = 0
        if pg > 0:
            resolved_ref = "page-{}".format(pg)

    try:
        result = _svc.serve_source_doc(course_id, doc, resolved_ref)
    except _svc.SourceMaterialsError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected render failure
        return _error(500, "source_doc_failed", str(exc))

    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


@router.get("/courses/{course_id}/source-doc-asset")
async def get_source_doc_asset(course_id: str, doc: str = "", path: str = "") -> Any:
    """Serve a whitelisted figure asset from a source doc's ``{stem}_figures/`` dir.

    Path-traversal-safe (pre-rejection + extension whitelist + commonpath
    containment under the figures dir). 403 when disabled, 422 bad path /
    disallowed type, 404 unknown course / doc / member.
    """
    try:
        from gui.services import source_materials as _svc  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "source_materials_unavailable", str(exc))

    try:
        result = _svc.serve_source_doc_asset(course_id, doc, path)
    except _svc.SourceMaterialsError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected read failure
        return _error(500, "source_doc_asset_failed", str(exc))

    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


@router.get("/courses/{course_id}/asset")
async def get_asset(course_id: str, path: str = "") -> Any:
    """Serve a whitelisted static asset (image/css/font) from the cartridge."""
    try:
        from gui.services import imscc_service  # noqa: PLC0415
    except ImportError as exc:
        return _error(503, "viewer_unavailable", str(exc))

    try:
        result = imscc_service.get_asset(course_id, path)
    except imscc_service.IMSCCServiceError as exc:
        return _error(exc.status, exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 — unexpected read failure
        return _error(500, "asset_failed", str(exc))

    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=dict(result.headers),
    )


__all__ = ["router"]
