"""Index generation for LibV2."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .catalog import generate_master_catalog, load_course_manifest, save_master_catalog
from .models.catalog import MasterCatalog


def generate_division_indexes(catalog: MasterCatalog, repo_root: Path) -> None:
    """Generate indexes for each division (STEM, ARTS)."""
    by_division_dir = repo_root / "catalog" / "by_division"
    by_division_dir.mkdir(parents=True, exist_ok=True)

    divisions = defaultdict(list)
    for entry in catalog.courses:
        divisions[entry.division].append(entry.to_dict())

    for division, courses in divisions.items():
        index_path = by_division_dir / f"{division}.json"
        with open(index_path, "w") as f:
            json.dump({
                "division": division,
                "generated_at": datetime.now().isoformat(),
                "count": len(courses),
                "courses": courses,
            }, f, indent=2)


def generate_domain_indexes(catalog: MasterCatalog, repo_root: Path) -> None:
    """Generate indexes for each domain."""
    by_domain_dir = repo_root / "catalog" / "by_domain"
    by_domain_dir.mkdir(parents=True, exist_ok=True)

    # Group by primary domain
    by_domain = defaultdict(list)
    for entry in catalog.courses:
        by_domain[entry.primary_domain].append(entry.to_dict())
        # Also add to secondary domains
        for domain in entry.secondary_domains:
            by_domain[domain].append(entry.to_dict())

    for domain, courses in by_domain.items():
        index_path = by_domain_dir / f"{domain}.json"
        with open(index_path, "w") as f:
            json.dump({
                "domain": domain,
                "generated_at": datetime.now().isoformat(),
                "count": len(courses),
                "courses": courses,
            }, f, indent=2)


def generate_subdomain_indexes(catalog: MasterCatalog, repo_root: Path) -> None:
    """Generate indexes for each subdomain."""
    by_subdomain_dir = repo_root / "catalog" / "by_subdomain"
    by_subdomain_dir.mkdir(parents=True, exist_ok=True)

    # Group by domain and subdomain
    by_subdomain = defaultdict(lambda: defaultdict(list))
    for entry in catalog.courses:
        domain = entry.primary_domain
        for subdomain in entry.subdomains:
            by_subdomain[domain][subdomain].append(entry.to_dict())

    for domain, subdomains in by_subdomain.items():
        domain_dir = by_subdomain_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        for subdomain, courses in subdomains.items():
            index_path = domain_dir / f"{subdomain}.json"
            with open(index_path, "w") as f:
                json.dump({
                    "domain": domain,
                    "subdomain": subdomain,
                    "generated_at": datetime.now().isoformat(),
                    "count": len(courses),
                    "courses": courses,
                }, f, indent=2)


def generate_cross_references(repo_root: Path) -> None:
    """Generate cross-reference indexes for shared concepts."""
    courses_dir = repo_root / "courses"
    xref_dir = repo_root / "catalog" / "cross_references"
    xref_dir.mkdir(parents=True, exist_ok=True)

    concept_to_courses = defaultdict(list)

    if courses_dir.exists():
        for course_dir in courses_dir.iterdir():
            if not course_dir.is_dir() or course_dir.name.startswith("."):
                continue

            manifest = load_course_manifest(course_dir)
            if manifest is None:
                continue

            slug = manifest.slug

            # Read concept graph
            graph_path = course_dir / "graph" / "concept_graph.json"
            if graph_path.exists():
                with open(graph_path) as f:
                    graph = json.load(f)

                nodes = graph.get("nodes", [])
                for node in nodes:
                    concept_id = node.get("id", "")
                    if concept_id:
                        concept_to_courses[concept_id].append({
                            "slug": slug,
                            "frequency": node.get("frequency", 1),
                            "centrality": node.get("centrality", 0),
                        })

    # Filter to concepts appearing in multiple courses
    shared_concepts = {
        concept: courses
        for concept, courses in concept_to_courses.items()
        if len(courses) > 1
    }

    # Save concept_to_courses index
    with open(xref_dir / "concept_to_courses.json", "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_concepts": len(concept_to_courses),
            "concepts": dict(concept_to_courses),
        }, f, indent=2)

    # Save shared concepts index
    with open(xref_dir / "shared_concepts.json", "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_shared": len(shared_concepts),
            "concepts": shared_concepts,
        }, f, indent=2)


def generate_statistics(catalog: MasterCatalog, repo_root: Path) -> None:
    """Generate repository-wide statistics."""
    stats_dir = repo_root / "catalog" / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Overall stats
    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_courses": catalog.total_courses,
        "by_division": {},
        "by_domain": {},
        "by_difficulty": {},
        "totals": {
            "chunks": 0,
            "tokens": 0,
            "concepts": 0,
        },
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
        stats["totals"]["chunks"] += entry.chunk_count
        stats["totals"]["tokens"] += entry.token_count
        stats["totals"]["concepts"] += entry.concept_count

    with open(stats_dir / "repository_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def rebuild_all_indexes(repo_root: Path) -> dict:
    """Rebuild all indexes from scratch."""
    results = {
        "master_catalog": False,
        "division_indexes": False,
        "domain_indexes": False,
        "subdomain_indexes": False,
        "cross_references": False,
        "statistics": False,
    }

    # Generate master catalog
    catalog = generate_master_catalog(repo_root)
    save_master_catalog(catalog, repo_root)
    results["master_catalog"] = True

    # Generate course index
    from .catalog import generate_course_index, save_course_index
    index = generate_course_index(catalog)
    save_course_index(index, repo_root)

    # Generate division indexes
    generate_division_indexes(catalog, repo_root)
    results["division_indexes"] = True

    # Generate domain indexes
    generate_domain_indexes(catalog, repo_root)
    results["domain_indexes"] = True

    # Generate subdomain indexes
    generate_subdomain_indexes(catalog, repo_root)
    results["subdomain_indexes"] = True

    # Generate cross-references
    generate_cross_references(repo_root)
    results["cross_references"] = True

    # Generate statistics
    generate_statistics(catalog, repo_root)
    results["statistics"] = True

    return results
