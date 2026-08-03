"""Discover courses and concept relationships from the library-wide index.

The cross-package index aggregates the concept node IDs found in each course
and records typed relationships that cross course-package boundaries. This
module provides the deterministic, filesystem-only read surface for that
artifact. Given a topic query, it identifies which courses teach matching
concepts so a library-wide question can be routed to per-course ``libv2 ask``
or ``answer-grounded`` calls. It requires no LLM, network, or retrieval engine.

Anti-fabrication contract: every concept / course / edge returned is present
verbatim in the on-disk index. This helper NEVER invents a concept, a course
slug, or a cross-course association — it only surfaces (with provenance: slug +
frequency + label) associations the index already recorded. Read-only: it does
not mutate the index or any course data.

Discovery is read-only and runs only when invoked through
``libv2 cross-discover`` or imported directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CrossPackageIndexError(RuntimeError):
    """Base class for cross-package discovery failures."""


class CrossPackageIndexMissing(CrossPackageIndexError):
    """Raised when the cross-package concept index has not been built yet."""


class CrossPackageIndexUnreadable(CrossPackageIndexError):
    """Raised when the index file exists but cannot be parsed as JSON."""


def default_index_path(repo_root: Path) -> Path:
    """Return the canonical on-disk location of the cross-package index.

    Uses the writer default from ``cli.py::cross_index`` and
    ``cross_package.indexer.write_cross_package_index`` so both surfaces resolve
    the same artifact location.
    """
    return Path(repo_root) / "LibV2" / "catalog" / "cross_package_concepts.json"


def load_cross_package_index(
    repo_root: Optional[Path] = None,
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the cross-package concept index artifact.

    Parameters
    ----------
    repo_root:
        Repository root that contains ``LibV2/catalog/``. Used to derive the
        default index path when *path* is not supplied.
    path:
        Explicit path to the index JSON (wins over *repo_root*).

    Raises
    ------
    CrossPackageIndexMissing:
        The index file does not exist (build it with ``libv2 cross-index``).
    CrossPackageIndexUnreadable:
        The file exists but is not valid JSON / not an object.
    """
    if path is not None:
        index_path = Path(path)
    elif repo_root is not None:
        index_path = default_index_path(repo_root)
    else:  # pragma: no cover - guarded by callers
        raise ValueError("load_cross_package_index requires repo_root or path")

    if not index_path.exists():
        raise CrossPackageIndexMissing(
            f"cross-package concept index not found at {index_path}; "
            "build it with `libv2 cross-index`"
        )
    try:
        with index_path.open(encoding="utf-8") as f:
            artifact = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossPackageIndexUnreadable(
            f"cross-package concept index at {index_path} is unreadable: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise CrossPackageIndexUnreadable(
            f"cross-package concept index at {index_path} is not a JSON object"
        )
    return artifact


def _concepts(index: Dict[str, Any]) -> Dict[str, Any]:
    concepts = index.get("concepts")
    return concepts if isinstance(concepts, dict) else {}


def _matches(query: str, concept_id: str, entry: Dict[str, Any]) -> bool:
    """Case-insensitive substring match against the concept id or label."""
    needle = query.strip().lower()
    if not needle:
        return True  # empty query -> list everything (bounded by limit)
    if needle in concept_id.lower():
        return True
    label = entry.get("label")
    return isinstance(label, str) and needle in label.lower()


def search_concepts(
    index: Dict[str, Any],
    query: str,
    *,
    limit: int = 10,
    min_courses: int = 1,
) -> List[Dict[str, Any]]:
    """Return concepts whose id/label matches *query*, richest-coverage first.

    Each result carries the concept id, label, the courses that teach it (slug
    + frequency, provenance-preserved), and the count of cross-package edges the
    index recorded for it. Deterministic ordering:
    ``(total_courses desc, concept_id asc)`` — a tie-break identical to the
    builder's own concept ordering so the discovery view lines up with the
    artifact.
    """
    matched: List[Dict[str, Any]] = []
    for concept_id, entry in _concepts(index).items():
        if not isinstance(entry, dict):
            continue
        total = int(entry.get("total_courses", len(entry.get("courses", []) or [])) or 0)
        if total < max(1, min_courses):
            continue
        if not _matches(query, concept_id, entry):
            continue
        courses = [
            {
                "slug": c.get("slug"),
                "frequency": int(c.get("frequency", 0) or 0),
                "label": c.get("label", entry.get("label", concept_id)),
            }
            for c in (entry.get("courses") or [])
            if isinstance(c, dict) and c.get("slug")
        ]
        matched.append(
            {
                "concept_id": concept_id,
                "label": entry.get("label", concept_id),
                "total_courses": total,
                "courses": courses,
                "cross_package_edge_count": len(entry.get("cross_package_edges") or []),
            }
        )
    matched.sort(key=lambda r: (-r["total_courses"], r["concept_id"]))
    if limit is not None and limit >= 0:
        matched = matched[:limit]
    return matched


def courses_for_concept(
    index: Dict[str, Any],
    concept_id: str,
) -> List[Dict[str, Any]]:
    """Return the courses that teach *concept_id* (exact id lookup).

    Empty list when the concept is unknown — never fabricates a course.
    """
    entry = _concepts(index).get(concept_id)
    if not isinstance(entry, dict):
        return []
    return [
        {
            "slug": c.get("slug"),
            "frequency": int(c.get("frequency", 0) or 0),
            "label": c.get("label", entry.get("label", concept_id)),
        }
        for c in (entry.get("courses") or [])
        if isinstance(c, dict) and c.get("slug")
    ]


def related_concepts(
    index: Dict[str, Any],
    concept_id: str,
) -> List[Dict[str, Any]]:
    """Return the typed cross-package edges recorded on *concept_id*.

    These are the boundary-crossing relationships the builder surfaced (both
    endpoints shared across >=2 courses). Empty list when unknown.
    """
    entry = _concepts(index).get(concept_id)
    if not isinstance(entry, dict):
        return []
    edges = entry.get("cross_package_edges") or []
    return [e for e in edges if isinstance(e, dict)]


def discover_courses(
    index: Dict[str, Any],
    query: str,
    *,
    limit: int = 10,
    min_courses: int = 1,
    per_concept_limit: int = 50,
) -> Dict[str, Any]:
    """Route a topic *query* to the candidate courses a library-wide ask spans.

    Composes with the per-course ask/answer path: to answer a library-wide
    question, first discover WHICH courses to fan out over (this function), then
    run ``libv2 ask`` / ``answer-grounded`` per candidate slug.

    A course is a candidate iff it teaches at least one concept matching the
    query. Candidates are ranked by ``matched_concept_count`` (how many distinct
    matched concepts the course teaches) then by summed frequency (how strongly)
    — richer, more-central coverage first. Deterministic tie-break on slug.

    Returns a JSON-serialisable dict::

        {
          "query": <str>,
          "matched_concepts": [ {concept_id, label, total_courses}, ... ],
          "courses": [ {slug, matched_concept_count, matched_concepts,
                        total_frequency}, ... ],
        }

    Anti-fabrication: every slug/concept is drawn from the index verbatim.
    """
    matched = search_concepts(
        index, query, limit=per_concept_limit, min_courses=min_courses
    )

    per_course: Dict[str, Dict[str, Any]] = {}
    for concept in matched:
        for course in concept["courses"]:
            slug = course["slug"]
            if not slug:
                continue
            rec = per_course.setdefault(
                slug,
                {
                    "slug": slug,
                    "matched_concepts": [],
                    "total_frequency": 0,
                },
            )
            rec["matched_concepts"].append(concept["concept_id"])
            rec["total_frequency"] += int(course.get("frequency", 0) or 0)

    courses: List[Dict[str, Any]] = []
    for rec in per_course.values():
        concept_ids = sorted(set(rec["matched_concepts"]))
        courses.append(
            {
                "slug": rec["slug"],
                "matched_concept_count": len(concept_ids),
                "matched_concepts": concept_ids,
                "total_frequency": rec["total_frequency"],
            }
        )
    courses.sort(
        key=lambda r: (
            -r["matched_concept_count"],
            -r["total_frequency"],
            r["slug"],
        )
    )
    if limit is not None and limit >= 0:
        courses = courses[:limit]

    return {
        "query": query,
        "matched_concepts": [
            {
                "concept_id": c["concept_id"],
                "label": c["label"],
                "total_courses": c["total_courses"],
            }
            for c in matched
        ],
        "courses": courses,
    }
