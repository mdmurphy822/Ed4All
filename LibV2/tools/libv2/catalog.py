"""Catalog management for LibV2."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models.catalog import CatalogEntry, MasterCatalog
from .models.course import CourseManifest


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


def _register_course_in_catalog(
    course_slug: str,
    manifest: dict,
    libv2_root: Path,
) -> None:
    """Register or update a course entry in the master catalog (atomic, idempotent).

    Called from ``_archive_to_libv2`` / ``archive_to_libv2`` immediately after
    the per-course ``manifest.json`` is written so that ``libv2 catalog list``
    and ``libv2 info <slug>`` see the course without a manual ``index rebuild``.

    Args:
        course_slug: Canonical slug for the archived course.
        manifest: The dict that was written to ``manifest.json``.
        libv2_root: Absolute path to the LibV2 root directory.
    """
    import os
    import tempfile

    catalog_dir = libv2_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "master_catalog.json"

    # Load existing catalog or create an empty one.
    if catalog_path.exists():
        try:
            with open(catalog_path) as fh:
                raw = json.load(fh)
            existing = MasterCatalog.from_dict(raw)
        except Exception:
            existing = MasterCatalog(
                version="1.0.0",
                generated_at=datetime.now().isoformat(),
                total_courses=0,
                courses=[],
            )
    else:
        existing = MasterCatalog(
            version="1.0.0",
            generated_at=datetime.now().isoformat(),
            total_courses=0,
            courses=[],
        )

    classification = manifest.get("classification", {})
    new_entry = CatalogEntry(
        slug=course_slug,
        title=manifest.get("title") or course_slug,
        division=classification.get("division", "STEM"),
        primary_domain=classification.get("primary_domain", "general"),
        secondary_domains=classification.get("secondary_domains", []),
        subdomains=classification.get("subdomains", []),
        # content_profile fields are not yet present in the pipeline manifest;
        # leave defaults (0) so a later index rebuild can fill them in.
    )

    # Idempotent: replace any existing entry with the same slug.
    courses = [c for c in existing.courses if c.slug != course_slug]
    courses.append(new_entry)

    updated = MasterCatalog(
        version=existing.version,
        generated_at=datetime.now().isoformat(),
        total_courses=len(courses),
        courses=courses,
    )

    # Atomic write via tmpfile + rename so a concurrent reader never sees a
    # half-written catalog.
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=catalog_dir,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(updated.to_dict(), tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, catalog_path)

    # Also keep course_index.json in sync (slug → quick-lookup dict).
    index_path = catalog_dir / "course_index.json"
    if index_path.exists():
        try:
            with open(index_path) as fh:
                index = json.load(fh)
        except Exception:
            index = {}
    else:
        index = {}

    index[course_slug] = {
        "path": f"courses/{course_slug}",
        "title": new_entry.title,
        "division": new_entry.division,
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=catalog_dir,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(index, tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, index_path)


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
