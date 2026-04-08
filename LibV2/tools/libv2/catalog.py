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
