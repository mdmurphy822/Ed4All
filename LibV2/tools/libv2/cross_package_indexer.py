"""Cross-package concept index builder (Worker G).

Scans every course under ``<repo_root>/LibV2/courses/*/graph/`` and aggregates
which concept node IDs appear in which courses. When a course additionally
carries a Worker F ``concept_graph_semantic.json``, typed edges whose endpoints
are shared with at least one *other* course are surfaced as
``cross_package_edges`` so downstream tools can navigate typed relationships
that genuinely cross package boundaries.

The output is written to ``<repo_root>/LibV2/catalog/cross_package_concepts.json``
and intentionally has no LLM, no network, and no retrieval-engine dependencies:
it is a pure filesystem + JSON aggregation pass.

Contract version: ``catalog_version = 1``. Any shape change bumps this integer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CATALOG_VERSION = 1
"""Artifact schema version. Bump on any breaking shape change."""


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from *path*, returning ``None`` if the file is missing or
    unreadable. We deliberately swallow parse errors so a single corrupt file
    cannot sink the whole index build; callers can detect degradation via the
    ``course_count`` field in the emitted artifact.
    """
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _iter_course_graph_dirs(courses_root: Path) -> List[Path]:
    """Return the list of course directories that have a ``graph/`` subdir
    containing ``concept_graph.json``. Deterministic (sorted by slug).
    """
    if not courses_root.exists() or not courses_root.is_dir():
        return []
    results: List[Path] = []
    for child in sorted(courses_root.iterdir()):
        if not child.is_dir():
            continue
        graph_dir = child / "graph"
        if not graph_dir.is_dir():
            continue
        if not (graph_dir / "concept_graph.json").is_file():
            continue
        results.append(child)
    return results


def build_cross_package_index(repo_root: Path) -> Dict[str, Any]:
    """Build the cross-package concept index for *repo_root*.

    Parameters
    ----------
    repo_root:
        Repository root that contains a ``LibV2/courses/`` directory. The
        function tolerates a repo with no courses (returns an empty index).

    Returns
    -------
    dict
        A JSON-serialisable artifact following the ``catalog_version=1``
        shape. Concepts are sorted by ``total_courses`` descending, then
        alphabetically by concept id.
    """
    repo_root = Path(repo_root).resolve()
    courses_root = repo_root / "LibV2" / "courses"
    course_dirs = _iter_course_graph_dirs(courses_root)

    # Per-course data we assemble in a single pass so later steps can be
    # driven off an in-memory view rather than re-reading graphs.
    course_records: List[Dict[str, Any]] = []
    concepts_by_id: Dict[str, Dict[str, Any]] = {}

    for course_dir in course_dirs:
        slug = course_dir.name
        untyped = _load_json(course_dir / "graph" / "concept_graph.json") or {}
        typed = _load_json(course_dir / "graph" / "concept_graph_semantic.json")

        # Build lookup of concept id -> (label, frequency) for this course.
        nodes: Dict[str, Dict[str, Any]] = {}
        for node in untyped.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue
            nodes[node_id] = {
                "label": node.get("label", node_id),
                "frequency": int(node.get("frequency", 0) or 0),
            }

        # Accumulate per-concept per-course presence.
        for node_id, info in nodes.items():
            concept = concepts_by_id.setdefault(
                node_id,
                {
                    "label": info["label"],
                    "courses": {},  # slug -> {frequency, label}
                    "typed_edges": [],  # raw per-course typed edges; filtered later
                },
            )
            # Prefer the first non-empty label we saw; keep concepts_by_id[id]["label"]
            # stable across runs by only overwriting a missing/empty label.
            if not concept.get("label"):
                concept["label"] = info["label"]
            concept["courses"][slug] = {
                "frequency": info["frequency"],
                "label": info["label"],
            }

        course_records.append({
            "slug": slug,
            "nodes": nodes,
            "typed": typed,
        })

    # Determine the set of concepts shared across >=2 courses. Only shared
    # concepts can carry cross_package_edges (an edge where both endpoints
    # are present in at least one OTHER course's untyped graph).
    shared_concept_ids = {
        cid for cid, c in concepts_by_id.items() if len(c["courses"]) >= 2
    }

    # For each course that has a typed semantic graph, collect edges whose
    # BOTH endpoints are shared concepts AND at least one endpoint appears in
    # a DIFFERENT course (guaranteed true when both endpoints are in >=2
    # courses, since that includes the current course plus at least one more).
    for record in course_records:
        typed = record["typed"]
        if not typed:
            continue
        slug = record["slug"]
        for edge in typed.get("edges", []) or []:
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue
            if source not in shared_concept_ids or target not in shared_concept_ids:
                continue
            entry = {
                "source_concept": source,
                "target_concept": target,
                "type": edge.get("type"),
                "course_slug": slug,
            }
            if "confidence" in edge:
                entry["confidence"] = edge["confidence"]
            if "weight" in edge:
                entry["weight"] = edge["weight"]
            # Attach to the source-concept's bucket so a reader can pivot by
            # the concept they are investigating.
            concepts_by_id[source]["typed_edges"].append(entry)

    # Shape the final per-concept payload with deterministic ordering.
    out_concepts: Dict[str, Dict[str, Any]] = {}
    # Sort concept ids by (total_courses desc, id asc).
    ordered_ids = sorted(
        concepts_by_id.keys(),
        key=lambda cid: (-len(concepts_by_id[cid]["courses"]), cid),
    )
    for cid in ordered_ids:
        concept = concepts_by_id[cid]
        courses_list = [
            {
                "slug": slug,
                "frequency": info["frequency"],
                "label": info["label"],
            }
            for slug, info in sorted(concept["courses"].items())
        ]
        # Deterministic edge ordering: by (target, type, course_slug).
        edges_sorted = sorted(
            concept["typed_edges"],
            key=lambda e: (
                e.get("target_concept") or "",
                e.get("type") or "",
                e.get("course_slug") or "",
            ),
        )
        out_concepts[cid] = {
            "label": concept["label"],
            "total_courses": len(courses_list),
            "courses": courses_list,
            "cross_package_edges": edges_sorted,
        }

    # ``generated_at`` is the one non-deterministic field; tests that need
    # byte-stable output should ignore it (see
    # ``test_deterministic_ordering`` for the canonicalisation helper).
    return {
        "catalog_version": CATALOG_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "course_count": len(course_records),
        "concept_count": len(out_concepts),
        "concepts": out_concepts,
    }


def write_cross_package_index(
    repo_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build the index for *repo_root* and write it to *output_path*.

    Returns the in-memory artifact so callers can summarise it without a
    round-trip read.
    """
    artifact = build_cross_package_index(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=False)
        f.write("\n")
    return artifact


def canonical_payload(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *artifact* with non-deterministic fields removed.

    Useful for equality-style tests that want byte stability across runs.
    """
    stripped = dict(artifact)
    stripped.pop("generated_at", None)
    stripped.pop("repo_root", None)
    return stripped
