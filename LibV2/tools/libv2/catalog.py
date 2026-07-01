"""Catalog management for LibV2."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models.catalog import CatalogEntry, MasterCatalog
from .models.course import CourseManifest

logger = logging.getLogger(__name__)

# W0.2: cross-process advisory lock around the catalog read-modify-write.
# Reuse the shared ``lib/file_lock.py`` helper when reachable; degrade to a
# no-op context manager in CLI-only environments where the repo-root ``lib``
# package isn't on ``sys.path`` (the helper itself already degrades gracefully
# on non-POSIX / unsupported filesystems).
try:  # pragma: no cover - import wiring
    from lib.file_lock import file_lock as _file_lock  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - lib not importable (CLI-only install)
    @contextmanager
    def _file_lock(_lock_path):  # type: ignore[misc]
        yield

# Sentinel filename for the catalog RMW lock. One lock guards both
# master_catalog.json and course_index.json so a concurrent archive can never
# half-update one while another reads the other.
_CATALOG_LOCK_FILENAME = ".catalog.lock"


def load_course_manifest(course_dir: Path) -> Optional[CourseManifest]:
    """Load a course manifest from a course directory."""
    manifest_path = course_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        data = json.load(f)

    return CourseManifest.from_dict(data)


def build_catalog_entry(course_dir: Path) -> Optional[CatalogEntry]:
    """Build a catalog entry from a course directory."""
    manifest = load_course_manifest(course_dir)
    if manifest is None:
        return None

    return CatalogEntry.from_manifest(manifest)


def generate_master_catalog(repo_root: Path) -> MasterCatalog:
    """Generate the master catalog from all courses."""
    courses_dir = repo_root / "courses"
    entries = []

    if courses_dir.exists():
        for course_dir in sorted(courses_dir.iterdir()):
            if course_dir.is_dir() and not course_dir.name.startswith("."):
                entry = build_catalog_entry(course_dir)
                if entry:
                    entries.append(entry)

    return MasterCatalog(
        version="1.0.0",
        generated_at=datetime.now().isoformat(),
        total_courses=len(entries),
        courses=entries,
    )


def save_master_catalog(catalog: MasterCatalog, repo_root: Path) -> None:
    """Save the master catalog to disk."""
    catalog_dir = repo_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = catalog_dir / "master_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog.to_dict(), f, indent=2)


def load_master_catalog(repo_root: Path) -> Optional[MasterCatalog]:
    """Load the master catalog from disk."""
    catalog_path = repo_root / "catalog" / "master_catalog.json"
    if not catalog_path.exists():
        return None

    with open(catalog_path) as f:
        data = json.load(f)

    return MasterCatalog.from_dict(data)


def generate_course_index(catalog: MasterCatalog) -> dict:
    """Generate a quick lookup index from slug to basic info."""
    return {
        entry.slug: {
            "path": f"courses/{entry.slug}",
            "title": entry.title,
            "division": entry.division,
        }
        for entry in catalog.courses
    }


def save_course_index(index: dict, repo_root: Path) -> None:
    """Save the course index to disk."""
    catalog_dir = repo_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    index_path = catalog_dir / "course_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def search_catalog(
    catalog: MasterCatalog,
    division: Optional[str] = None,
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    difficulty: Optional[str] = None,
    query: Optional[str] = None,
) -> list[CatalogEntry]:
    """Search the catalog with various filters."""
    results = catalog.courses

    if division:
        results = [c for c in results if c.division == division.upper()]

    if domain:
        results = [
            c for c in results
            if c.primary_domain == domain.lower()
            or domain.lower() in c.secondary_domains
        ]

    if subdomain:
        results = [c for c in results if subdomain.lower() in c.subdomains]

    if difficulty:
        results = [c for c in results if c.difficulty_primary == difficulty.lower()]

    if query:
        query_lower = query.lower()
        results = [
            c for c in results
            if query_lower in c.title.lower()
            or query_lower in c.slug
            or query_lower in c.primary_domain
            or any(query_lower in s for s in c.subdomains)
        ]

    return results


def _entry_from_manifest(course_slug: str, manifest: dict) -> CatalogEntry:
    """Build a :class:`CatalogEntry` from a course slug + its manifest dict."""
    classification = manifest.get("classification", {}) or {}
    return CatalogEntry(
        slug=course_slug,
        title=manifest.get("title") or course_slug,
        division=classification.get("division", "STEM"),
        primary_domain=classification.get("primary_domain", "general"),
        secondary_domains=classification.get("secondary_domains", []),
        subdomains=classification.get("subdomains", []),
        # content_profile fields are not yet present in the pipeline manifest;
        # leave defaults (0) so a later index rebuild can fill them in.
    )


def _load_existing_catalog(catalog_path: Path) -> MasterCatalog:
    """Load the master catalog from disk, falling back to an empty one."""
    if catalog_path.exists():
        try:
            with open(catalog_path) as fh:
                raw = json.load(fh)
            return MasterCatalog.from_dict(raw)
        except Exception:
            pass
    return MasterCatalog(
        version="1.0.0",
        generated_at=datetime.now().isoformat(),
        total_courses=0,
        courses=[],
    )


def _write_catalog_and_index(
    catalog_dir: Path,
    catalog: MasterCatalog,
) -> None:
    """Atomically persist the master catalog + slug→path quick-lookup index.

    Caller is responsible for holding the catalog lock (see
    :func:`_register_course_in_catalog` / :func:`backfill_master_catalog`).
    """
    import os
    import tempfile

    catalog_path = catalog_dir / "master_catalog.json"

    # Atomic write via tmpfile + rename so a concurrent reader never sees a
    # half-written catalog.
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=catalog_dir,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(catalog.to_dict(), tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, catalog_path)

    # Rebuild course_index.json from the catalog so the two stay in lockstep.
    index = {
        entry.slug: {
            "path": f"courses/{entry.slug}",
            "title": entry.title,
            "division": entry.division,
        }
        for entry in catalog.courses
    }
    index_path = catalog_dir / "course_index.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=catalog_dir,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(index, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, index_path)


def _register_course_in_catalog(
    course_slug: str,
    manifest: dict,
    libv2_root: Path,
) -> None:
    """Register or update a course entry in the master catalog (atomic, idempotent).

    Called from ``_archive_to_libv2`` / ``archive_to_libv2`` immediately after
    the per-course ``manifest.json`` is written so that ``libv2 catalog list``
    and ``libv2 info <slug>`` see the course without a manual ``index rebuild``.

    W0.2: the whole read-modify-write (load catalog → upsert entry → write
    catalog + index) runs UNDER a cross-process file lock so two concurrent
    archives can't lost-update each other (the classic read-both-then-each-
    write-its-own-superset race that drops the other archive's new entry).

    Args:
        course_slug: Canonical slug for the archived course.
        manifest: The dict that was written to ``manifest.json``.
        libv2_root: Absolute path to the LibV2 root directory.
    """
    catalog_dir = libv2_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "master_catalog.json"

    with _file_lock(catalog_dir / _CATALOG_LOCK_FILENAME):
        existing = _load_existing_catalog(catalog_path)
        new_entry = _entry_from_manifest(course_slug, manifest)

        # Idempotent: replace any existing entry with the same slug.
        courses = [c for c in existing.courses if c.slug != course_slug]
        courses.append(new_entry)

        updated = MasterCatalog(
            version=existing.version,
            generated_at=datetime.now().isoformat(),
            total_courses=len(courses),
            courses=courses,
        )
        _write_catalog_and_index(catalog_dir, updated)


def backfill_master_catalog(libv2_root: Path) -> dict:
    """Enumerate EVERY archived course dir into the master catalog (repair).

    W0.8: the live ``_register_course_in_catalog`` path only ever adds the one
    course currently being archived, so a catalog that pre-dates that wiring (or
    that was lost-updated before W0.2) can enumerate only a handful of the ~90
    courses actually on disk under ``courses/``. This backfill walks every
    ``courses/<slug>/manifest.json`` and upserts a catalog entry for each,
    MERGING with (never truncating) any existing catalog so a course already
    carrying richer ``content_profile`` data keeps it.

    Idempotent (re-running yields the same catalog) and lock-guarded via the
    same W0.2 catalog lock as the live registration path. Returns a summary
    dict ``{"discovered", "added", "updated", "total"}``.

    Anti-fabrication: only directories carrying a readable ``manifest.json`` are
    enrolled; a bare/empty course skeleton with no manifest is skipped.
    """
    libv2_root = Path(libv2_root)
    catalog_dir = libv2_root / "catalog"
    courses_dir = libv2_root / "courses"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "master_catalog.json"

    discovered = 0
    added = 0
    updated = 0

    with _file_lock(catalog_dir / _CATALOG_LOCK_FILENAME):
        existing = _load_existing_catalog(catalog_path)
        by_slug = {c.slug: c for c in existing.courses}

        if courses_dir.exists():
            for course_dir in sorted(courses_dir.iterdir()):
                if not course_dir.is_dir() or course_dir.name.startswith("."):
                    continue
                manifest_path = course_dir / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    with open(manifest_path) as fh:
                        manifest = json.load(fh)
                except Exception as exc:
                    logger.warning(
                        "backfill_master_catalog: unreadable manifest for %s "
                        "(%s); skipping.",
                        course_dir.name,
                        exc,
                    )
                    continue
                if not isinstance(manifest, dict):
                    continue
                discovered += 1
                slug = course_dir.name
                was_present = slug in by_slug
                by_slug[slug] = _entry_from_manifest(slug, manifest)
                if was_present:
                    updated += 1
                else:
                    added += 1

        courses = [by_slug[s] for s in sorted(by_slug)]
        merged = MasterCatalog(
            version=existing.version,
            generated_at=datetime.now().isoformat(),
            total_courses=len(courses),
            courses=courses,
        )
        _write_catalog_and_index(catalog_dir, merged)

    return {
        "discovered": discovered,
        "added": added,
        "updated": updated,
        "total": len(by_slug),
    }


def get_catalog_statistics(catalog: MasterCatalog) -> dict:
    """Get statistics about the catalog."""
    stats = {
        "total_courses": catalog.total_courses,
        "by_division": {},
        "by_domain": {},
        "by_difficulty": {},
        "total_chunks": 0,
        "total_tokens": 0,
    }

    for entry in catalog.courses:
        # By division
        div = entry.division
        stats["by_division"][div] = stats["by_division"].get(div, 0) + 1

        # By domain
        dom = entry.primary_domain
        stats["by_domain"][dom] = stats["by_domain"].get(dom, 0) + 1

        # By difficulty
        diff = entry.difficulty_primary
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        # Totals
        stats["total_chunks"] += entry.chunk_count
        stats["total_tokens"] += entry.token_count

    return stats
