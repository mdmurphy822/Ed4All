"""Tests for the Worker-G cross-package concept index builder."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make the repo root importable so ``LibV2.tools.libv2.*`` resolves regardless
# of where pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LibV2.tools.libv2.cross_package_indexer import (  # noqa: E402
    CATALOG_VERSION,
    build_cross_package_index,
    canonical_payload,
    write_cross_package_index,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_course(
    repo_root: Path,
    slug: str,
    untyped: dict,
    typed: dict | None = None,
) -> Path:
    """Write a synthetic course with the given graph files and return its dir."""
    course_dir = repo_root / "LibV2" / "courses" / slug
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "concept_graph.json").write_text(
        json.dumps(untyped), encoding="utf-8"
    )
    if typed is not None:
        (graph_dir / "concept_graph_semantic.json").write_text(
            json.dumps(typed), encoding="utf-8"
        )
    return course_dir


def _untyped(nodes: list[tuple[str, str, int]], edges=None) -> dict:
    return {
        "nodes": [
            {"id": nid, "label": label, "frequency": freq}
            for (nid, label, freq) in nodes
        ],
        "edges": edges or [],
        "generated_at": "2026-04-17T12:00:00+00:00",
    }


def _semantic(nodes: list[tuple[str, str, int]], edges: list[dict]) -> dict:
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-04-17T12:00:00+00:00",
        "rule_versions": {"related_from_cooccurrence": 1},
        "nodes": [
            {"id": nid, "label": label, "frequency": freq}
            for (nid, label, freq) in nodes
        ],
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concept_in_both_courses_lists_both_slugs(tmp_path: Path) -> None:
    """Concept observed in two courses must list both slugs in ``courses``."""
    _write_course(
        tmp_path,
        "course-a",
        _untyped([("accessibility", "Accessibility", 10), ("unique-a", "Unique A", 3)]),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([("accessibility", "Accessibility", 7), ("unique-b", "Unique B", 2)]),
    )

    artifact = build_cross_package_index(tmp_path)

    assert artifact["catalog_version"] == CATALOG_VERSION
    assert artifact["course_count"] == 2
    acc = artifact["concepts"]["accessibility"]
    assert acc["total_courses"] == 2
    slugs = sorted(c["slug"] for c in acc["courses"])
    assert slugs == ["course-a", "course-b"]
    # Per-course frequency is preserved verbatim.
    by_slug = {c["slug"]: c for c in acc["courses"]}
    assert by_slug["course-a"]["frequency"] == 10
    assert by_slug["course-b"]["frequency"] == 7
    # No semantic graphs supplied anywhere -> empty edge list.
    assert acc["cross_package_edges"] == []


def test_concept_in_one_course_lists_only_that_slug(tmp_path: Path) -> None:
    """Concept observed in only one course must not be attributed to others."""
    _write_course(
        tmp_path,
        "course-a",
        _untyped([("shared-concept", "Shared", 5), ("solo", "Solo", 9)]),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([("shared-concept", "Shared", 2)]),
    )

    artifact = build_cross_package_index(tmp_path)

    solo = artifact["concepts"]["solo"]
    assert solo["total_courses"] == 1
    assert [c["slug"] for c in solo["courses"]] == ["course-a"]

    shared = artifact["concepts"]["shared-concept"]
    assert shared["total_courses"] == 2


def test_semantic_graph_populates_cross_package_edges(tmp_path: Path) -> None:
    """When a semantic graph is present and endpoints are shared, its edges
    are surfaced on the source concept."""
    _write_course(
        tmp_path,
        "course-a",
        _untyped([
            ("accessibility", "Accessibility", 10),
            ("udl", "UDL", 6),
        ]),
        typed=_semantic(
            [("accessibility", "Accessibility", 10), ("udl", "UDL", 6)],
            [
                {
                    "source": "accessibility",
                    "target": "udl",
                    "type": "related-to",
                    "confidence": 0.6,
                    "weight": 3,
                    "provenance": {
                        "rule": "related_from_cooccurrence",
                        "rule_version": 1,
                    },
                },
                # An edge whose endpoints are NOT shared across courses must
                # be filtered out.
                {
                    "source": "accessibility",
                    "target": "only-in-a",
                    "type": "related-to",
                    "provenance": {"rule": "x", "rule_version": 1},
                },
            ],
        ),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([
            ("accessibility", "Accessibility", 4),
            ("udl", "UDL", 2),
        ]),
    )

    artifact = build_cross_package_index(tmp_path)

    edges = artifact["concepts"]["accessibility"]["cross_package_edges"]
    assert len(edges) == 1, f"expected exactly one cross-package edge, got {edges}"
    edge = edges[0]
    assert edge["source_concept"] == "accessibility"
    assert edge["target_concept"] == "udl"
    assert edge["type"] == "related-to"
    assert edge["course_slug"] == "course-a"
    assert edge["confidence"] == 0.6
    assert edge["weight"] == 3


def test_deterministic_ordering(tmp_path: Path) -> None:
    """Two runs on identical input produce byte-identical canonical output."""
    _write_course(
        tmp_path,
        "course-a",
        _untyped([("beta", "Beta", 2), ("alpha", "Alpha", 5)]),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([("alpha", "Alpha", 3), ("gamma", "Gamma", 1)]),
    )

    first = canonical_payload(build_cross_package_index(tmp_path))
    second = canonical_payload(build_cross_package_index(tmp_path))

    first_blob = json.dumps(first, indent=2, sort_keys=False)
    second_blob = json.dumps(second, indent=2, sort_keys=False)
    assert first_blob == second_blob

    # Ordering: shared concepts (2 courses) before singletons; ties broken
    # alphabetically by id.
    ids_in_order = list(first["concepts"].keys())
    assert ids_in_order[0] == "alpha"  # total_courses=2, wins
    # The two singletons ("beta", "gamma") follow in alphabetical order.
    assert ids_in_order[1:] == ["beta", "gamma"]


def test_missing_semantic_graph_degrades_to_untyped(tmp_path: Path) -> None:
    """No course has a semantic graph -> cross_package_edges is empty on every
    concept; this must NOT be an error."""
    _write_course(
        tmp_path,
        "course-a",
        _untyped([("shared", "Shared", 4)]),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([("shared", "Shared", 2)]),
    )
    artifact = build_cross_package_index(tmp_path)
    for concept in artifact["concepts"].values():
        assert concept["cross_package_edges"] == []


def test_write_produces_file_matching_in_memory_payload(tmp_path: Path) -> None:
    """``write_cross_package_index`` emits the same payload it returns."""
    _write_course(tmp_path, "course-a", _untyped([("x", "X", 1)]))
    output_path = tmp_path / "LibV2" / "catalog" / "cross_package_concepts.json"

    artifact = write_cross_package_index(tmp_path, output_path)

    assert output_path.is_file()
    with output_path.open(encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == artifact


def test_empty_repo_yields_empty_index(tmp_path: Path) -> None:
    """A repo with no courses produces a well-formed empty artifact."""
    (tmp_path / "LibV2" / "courses").mkdir(parents=True)
    artifact = build_cross_package_index(tmp_path)
    assert artifact["course_count"] == 0
    assert artifact["concept_count"] == 0
    assert artifact["concepts"] == {}


# ---------------------------------------------------------------------------
# Staleness check (lives in lib.libv2_fsck; exercised here because it depends
# on the catalog shape decided by the indexer).
# ---------------------------------------------------------------------------


def test_staleness_check_returns_issue_when_graph_newer(tmp_path: Path) -> None:
    from lib.libv2_fsck import check_cross_package_index_freshness

    course_a = _write_course(
        tmp_path,
        "course-a",
        _untyped([("concept-1", "Concept 1", 3)]),
    )
    catalog_path = tmp_path / "LibV2" / "catalog" / "cross_package_concepts.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a catalog with an explicit old ``generated_at`` timestamp.
    catalog_path.write_text(
        json.dumps({
            "catalog_version": CATALOG_VERSION,
            "generated_at": "2000-01-01T00:00:00+00:00",
            "repo_root": str(tmp_path),
            "course_count": 1,
            "concept_count": 1,
            "concepts": {},
        }),
        encoding="utf-8",
    )
    # Touch the course graph AFTER writing the catalog so its mtime is newer.
    graph_file = course_a / "graph" / "concept_graph.json"
    now = time.time()
    os.utime(graph_file, (now, now))

    issue = check_cross_package_index_freshness(tmp_path)
    assert issue is not None
    assert issue.severity in {"warning", "error"}
    assert "stale" in issue.message.lower() or "newer" in issue.message.lower()
    assert issue.category == "stale_catalog"


def test_staleness_check_returns_none_when_catalog_absent(tmp_path: Path) -> None:
    from lib.libv2_fsck import check_cross_package_index_freshness

    _write_course(tmp_path, "course-a", _untyped([("x", "X", 1)]))
    # No catalog file at all.
    assert check_cross_package_index_freshness(tmp_path) is None


def test_staleness_check_returns_none_when_catalog_fresh(tmp_path: Path) -> None:
    from lib.libv2_fsck import check_cross_package_index_freshness

    _write_course(tmp_path, "course-a", _untyped([("x", "X", 1)]))
    catalog_path = tmp_path / "LibV2" / "catalog" / "cross_package_concepts.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # Catalog generated "now" — definitely newer than the graph files we just
    # wrote.
    catalog_path.write_text(
        json.dumps({
            "catalog_version": CATALOG_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(tmp_path),
            "course_count": 1,
            "concept_count": 1,
            "concepts": {},
        }),
        encoding="utf-8",
    )

    issue = check_cross_package_index_freshness(tmp_path)
    assert issue is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
