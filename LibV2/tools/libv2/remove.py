"""Shared course-removal service (Marketable-v1 D5).

One implementation, two entry points: the ``libv2 remove <slug>`` CLI and the
GUI ``DELETE /api/library/{course_id}`` endpoint both call
:func:`remove_course`. The removal is strictly scoped to the resolved course dir
under ``<libv2_root>/courses/<slug>`` — a ``commonpath`` containment check
(mirroring the source-pdf endpoint) refuses any slug that escapes the courses
root, and the ``shutil.rmtree`` only ever touches the contained dir.

Catalog hygiene: the LibV2 catalog (``catalog/master_catalog.json``,
``course_index.json``, ``by_division/*.json``, ``by_domain/*.json``,
``by_subdomain/**/*.json``) is *derived* — the importer never appends an
entry on import; ``libv2 index rebuild`` regenerates the whole catalog by
scanning ``courses/``. So the symmetric "remove the entry the same way it was
added" is to drop the deleted slug from every derived catalog file that still
references it (so the catalog never points at a vanished course before the next
rebuild). :func:`prune_catalog_entries` does exactly that, generically.

The disk-usage helper (:func:`dir_size_bytes` / :func:`human_size`) is shared by
the CLI removal summary, the GUI library cards (with a TTL cache), and the
removal report.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Course-slug shape — identical to ``gui.services.imscc_service._SLUG_RE`` and
# ``source_page._SLUG_RE`` so all three surfaces share one slug contract.
_SLUG_RE = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")


class CourseRemovalError(Exception):
    """Typed removal failure carrying a (status, code, detail) triple.

    ``status`` mirrors the HTTP status the GUI endpoint maps it to (422 bad
    slug / escaping containment, 404 missing course); the CLI renders ``detail``
    and exits non-zero. One error type for both entry points.
    """

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class RemovalResult:
    """What :func:`remove_course` deleted + the pre-deletion summary."""

    slug: str
    course_dir: Path
    disk_bytes: int
    top_level: List[str] = field(default_factory=list)
    catalog_files_pruned: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "course_dir": str(self.course_dir),
            "disk_bytes": self.disk_bytes,
            "top_level": self.top_level,
            "catalog_files_pruned": self.catalog_files_pruned,
        }


# --------------------------------------------------------------------------- #
# Disk-usage helpers (shared by the CLI summary + GUI cards + removal report)
# --------------------------------------------------------------------------- #


def dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path`` via ``os.scandir``.

    Robust to broken symlinks / races (a file vanishing mid-walk is skipped).
    Does not follow directory symlinks (avoids double-counting / escaping the
    tree). Returns 0 for a missing path.
    """
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_symlink():
                        # Count the link's own size, never follow it.
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += dir_size_bytes(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
                except (OSError, ValueError):
                    continue
    except (OSError, ValueError):
        return total
    return total


def human_size(n: int) -> str:
    """Render an integer byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


# --------------------------------------------------------------------------- #
# Containment-checked slug resolution
# --------------------------------------------------------------------------- #


def resolve_course_dir(libv2_root: Path, slug: str) -> Path:
    """Resolve + containment-check ``<libv2_root>/courses/<slug>``.

    Raises :class:`CourseRemovalError`:

    * 422 ``invalid_slug`` — bare/empty slug or one that fails the slug regex.
    * 422 ``slug_escapes_root`` — the resolved dir is not contained under
      ``courses/`` (``..`` traversal / absolute path).
    * 404 ``course_not_found`` — no such course dir.

    The containment check uses ``os.path.commonpath`` over the *resolved* paths,
    exactly like ``gui.services.source_pdf._validate_slug``.
    """
    if not slug or not slug.strip():
        raise CourseRemovalError(422, "invalid_slug", "course slug must not be empty")
    if not _SLUG_RE.match(slug):
        raise CourseRemovalError(422, "invalid_slug", f"invalid course slug: {slug!r}")

    courses_dir = (Path(libv2_root) / "courses").resolve()
    course_dir = (courses_dir / slug).resolve()
    try:
        common = os.path.commonpath([str(course_dir), str(courses_dir)])
    except ValueError:
        # Different drives (Windows) — definitionally not contained.
        raise CourseRemovalError(
            422, "slug_escapes_root", f"slug escapes courses root: {slug!r}"
        ) from None
    if common != str(courses_dir) or course_dir == courses_dir:
        raise CourseRemovalError(
            422, "slug_escapes_root", f"slug escapes courses root: {slug!r}"
        )
    if not course_dir.is_dir():
        raise CourseRemovalError(404, "course_not_found", f"course not found: {slug}")
    return course_dir


# --------------------------------------------------------------------------- #
# Pre-deletion summary
# --------------------------------------------------------------------------- #


def summarize_course(course_dir: Path) -> tuple[int, List[str]]:
    """Return ``(disk_bytes, sorted_top_level_names)`` for a course dir."""
    disk = dir_size_bytes(course_dir)
    try:
        top = sorted(p.name for p in course_dir.iterdir())
    except OSError:
        top = []
    return disk, top


# --------------------------------------------------------------------------- #
# Derived-catalog hygiene
# --------------------------------------------------------------------------- #


def _prune_slug_from_payload(payload: object, slug: str) -> bool:
    """Drop ``slug`` from a loaded catalog payload in place. Returns mutated?.

    Handles the two derived catalog shapes:

    * a ``{"courses": [{"slug": ...}, ...]}`` envelope (master_catalog +
      by_division/by_domain/by_subdomain indexes), where ``total_courses`` /
      ``count`` are re-derived from the surviving list, and
    * a slug-keyed ``{slug: {...}}`` map (course_index.json).
    """
    mutated = False
    if isinstance(payload, dict) and isinstance(payload.get("courses"), list):
        before = payload["courses"]
        after = [c for c in before if not (isinstance(c, dict) and c.get("slug") == slug)]
        if len(after) != len(before):
            payload["courses"] = after
            if "total_courses" in payload:
                payload["total_courses"] = len(after)
            if "count" in payload:
                payload["count"] = len(after)
            mutated = True
    elif isinstance(payload, dict) and slug in payload and "courses" not in payload:
        # Slug-keyed map (course_index.json).
        del payload[slug]
        mutated = True
    return mutated


def prune_catalog_entries(libv2_root: Path, slug: str) -> List[str]:
    """Remove ``slug`` from every derived catalog file that references it.

    Returns the list of catalog file paths (relative to ``libv2_root``) that
    were rewritten. Best-effort per file: a malformed / unreadable catalog file
    is skipped (it will be regenerated wholesale on the next ``index rebuild``).
    """
    catalog_dir = Path(libv2_root) / "catalog"
    if not catalog_dir.is_dir():
        return []
    pruned: List[str] = []
    for json_path in sorted(catalog_dir.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _prune_slug_from_payload(payload, slug):
            try:
                json_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            except OSError:
                continue
            try:
                rel = str(json_path.relative_to(libv2_root))
            except ValueError:
                rel = str(json_path)
            pruned.append(rel)
    return pruned


# --------------------------------------------------------------------------- #
# The removal itself
# --------------------------------------------------------------------------- #


def remove_course(
    libv2_root: Path,
    slug: str,
    *,
    prune_catalog: bool = True,
) -> RemovalResult:
    """Delete ``<libv2_root>/courses/<slug>`` and prune its catalog entries.

    Containment-checked + fail-closed: resolves the course dir through
    :func:`resolve_course_dir` (which refuses an empty / malformed / escaping
    slug and a missing course) BEFORE any deletion, captures a pre-deletion
    summary, then ``shutil.rmtree``s exactly the resolved dir. With
    ``prune_catalog`` (default) the deleted slug is dropped from the derived
    catalog files so the catalog never points at a vanished course.

    Raises :class:`CourseRemovalError` for any refusal; never deletes anything
    outside the contained course dir.
    """
    course_dir = resolve_course_dir(libv2_root, slug)
    disk_bytes, top_level = summarize_course(course_dir)

    # Defence in depth: re-assert containment immediately before the rmtree so a
    # symlink swap between resolve + delete can't redirect the deletion.
    courses_dir = (Path(libv2_root) / "courses").resolve()
    resolved = course_dir.resolve()
    if (
        os.path.commonpath([str(resolved), str(courses_dir)]) != str(courses_dir)
        or resolved == courses_dir
    ):
        raise CourseRemovalError(
            422, "slug_escapes_root", f"slug escapes courses root: {slug!r}"
        )

    shutil.rmtree(course_dir)

    pruned: List[str] = []
    if prune_catalog:
        pruned = prune_catalog_entries(libv2_root, slug)

    return RemovalResult(
        slug=slug,
        course_dir=course_dir,
        disk_bytes=disk_bytes,
        top_level=top_level,
        catalog_files_pruned=pruned,
    )


__all__ = [
    "CourseRemovalError",
    "RemovalResult",
    "dir_size_bytes",
    "human_size",
    "resolve_course_dir",
    "summarize_course",
    "prune_catalog_entries",
    "remove_course",
]
