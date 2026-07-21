"""UPLOADS router for the Ed4All control-plane GUI.

Mounted at ``/api/uploads`` (see ``gui/app.py::_ROUTER_MOUNTS``). Saves uploaded
corpora (PDF / IMSCC) under ``state/gui/uploads/<upload_id>/`` so the saved file
paths can feed ``--corpus`` on a subsequent pipeline launch.

Real, no stubs: files are validated by extension and (when available) by
``lib.secure_paths.validate_path_within_root`` so a crafted filename can't
escape the uploads root. Bytes are streamed to disk; nothing is faked.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Query, UploadFile
from fastapi.responses import JSONResponse

from gui import shared_state
from lib.secure_paths import PathTraversalError

logger = logging.getLogger("gui.routers.uploads")

router = APIRouter()

# Accepted corpus extensions. PDFs (SemantiK conversion intake) + IMSCC (rag_training).
# Lower-cased for comparison.
_ALLOWED_EXTENSIONS = {".pdf", ".imscc"}


class _UploadError(Exception):
    """Internal signal carrying a flat ``{error, detail}`` body + status code.

    Lets the validation loop bail out (and trigger rollback) while still
    emitting the FLAT wire shape ``{"error": ..., "detail": ...}`` that the
    other GUI routers use, rather than ``HTTPException``'s nested
    ``{"detail": {...}}`` body.
    """

    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def _kind_for(name: str) -> str:
    """Map a filename to a corpus kind label (``pdf`` / ``imscc``)."""
    return Path(name).suffix.lower().lstrip(".")


def _validate_within_uploads(path: Path, root: Path) -> Path:
    """Confirm ``path`` stays within ``root`` (uploads dir).

    Prefers ``lib.secure_paths.validate_path_within_root`` when importable;
    falls back to a ``commonpath`` check so the guard is always enforced.
    """
    try:
        from lib.secure_paths import validate_path_within_root  # noqa: PLC0415

        return validate_path_within_root(path, root)
    except ImportError:
        import os  # noqa: PLC0415

        resolved = path.resolve()
        resolved_root = root.resolve()
        common = os.path.commonpath([str(resolved), str(resolved_root)])
        if common != str(resolved_root):
            raise ValueError(f"path escapes uploads root: {path}") from None
        return resolved


@router.post("")
async def create_upload(files: List[UploadFile]) -> Any:
    """Save one or more uploaded corpus files under a fresh ``upload_id``.

    Validates each file's extension is in ``{.pdf, .imscc}`` and that the
    resolved destination stays within the uploads root. Returns the saved paths
    so they can feed ``--corpus`` on a launch.
    """
    if not files:
        return _error(422, "no_files", "no files in request")

    upload_id = shared_state.new_run_id("UPL")
    dest_dir = shared_state.uploads_dir() / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    try:
        for upload in files:
            name = Path(upload.filename or "").name  # strip any path components
            if not name:
                raise _UploadError(422, "bad_filename", "empty filename")
            suffix = Path(name).suffix.lower()
            if suffix not in _ALLOWED_EXTENSIONS:
                raise _UploadError(
                    422,
                    "unsupported_extension",
                    f"{name}: {suffix!r} not in {sorted(_ALLOWED_EXTENSIONS)}",
                )

            target = dest_dir / name
            try:
                _validate_within_uploads(target, shared_state.uploads_dir())
            except (ValueError, PathTraversalError) as exc:
                raise _UploadError(422, "path_traversal", str(exc)) from exc

            with target.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved.append(
                {
                    "name": name,
                    "path": str(target),
                    "size": target.stat().st_size,
                    "kind": _kind_for(name),
                }
            )
    except _UploadError as exc:
        # Roll back the partial upload dir so a rejected batch leaves no trace.
        shutil.rmtree(dest_dir, ignore_errors=True)
        return _error(exc.status_code, exc.error, exc.detail)
    except Exception as exc:  # noqa: BLE001 — surface the real I/O error
        shutil.rmtree(dest_dir, ignore_errors=True)
        logger.exception("upload save failed")
        return _error(500, "upload_failed", str(exc))

    shared_state.append_event(
        "gui", "upload_created", {"upload_id": upload_id, "files": [f["name"] for f in saved]}
    )
    return {"upload_id": upload_id, "files": saved}


@router.get("")
async def list_uploads(limit: int = Query(default=200, ge=0)) -> Dict[str, Any]:
    """List prior uploads (one entry per ``upload_id`` directory).

    ``limit`` caps the number of returned uploads (newest first); default 200,
    ``0`` returns none. Slicing is applied in the router after the directory
    scan so the unbounded default keeps prior callers working.
    """
    root = shared_state.uploads_dir()
    uploads: List[Dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        files = []
        for fp in sorted(entry.iterdir()):
            if fp.is_file() and not fp.name.startswith("."):
                files.append(
                    {
                        "name": fp.name,
                        "path": str(fp),
                        "size": fp.stat().st_size,
                        "kind": _kind_for(fp.name),
                    }
                )
        uploads.append({"upload_id": entry.name, "files": files})
    return {"uploads": uploads[:limit]}


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str) -> Any:
    """Delete an upload directory and its files."""
    # Reject path-separator / traversal ids up front so a crafted segment never
    # reaches the resolver (a single-segment ``..`` resolves cleanly but escapes).
    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return _error(422, "bad_upload_id", "id may not contain path separators")
    try:
        # Reuse shared_state's id guard against path-separator escapes.
        target = shared_state.uploads_dir() / upload_id
        _validate_within_uploads(target, shared_state.uploads_dir())
    except (ValueError, PathTraversalError) as exc:
        return _error(422, "bad_upload_id", str(exc))
    if not target.is_dir():
        return _error(404, "unknown_upload", upload_id)
    shutil.rmtree(target, ignore_errors=True)
    shared_state.append_event("gui", "upload_deleted", {"upload_id": upload_id})
    return {"deleted": upload_id}


__all__: List[str] = ["router"]
