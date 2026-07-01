"""Tests for the cross-package concept discovery consumer.

Exercises the READ side that turns Worker-G's (previously dead) cross-package
index into a library-wide course-routing surface. Pure filesystem/JSON — no
LLM, no network, no retrieval engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LibV2.tools.libv2.cross_package_discovery import (  # noqa: E402
    CrossPackageIndexMissing,
    CrossPackageIndexUnreadable,
    courses_for_concept,
    default_index_path,
    discover_courses,
    load_cross_package_index,
    related_concepts,
    search_concepts,
)
from LibV2.tools.libv2.cross_package_indexer import (  # noqa: E402
    write_cross_package_index,
)


def _write_course(repo_root: Path, slug: str, untyped: dict, typed: dict | None = None):
    graph_dir = repo_root / "LibV2" / "courses" / slug / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "concept_graph.json").write_text(json.dumps(untyped), encoding="utf-8")
    if typed is not None:
        (graph_dir / "concept_graph_semantic.json").write_text(
            json.dumps(typed), encoding="utf-8"
        )


def _untyped(nodes):
    return {
        "nodes": [{"id": n, "label": lab, "frequency": f} for (n, lab, f) in nodes],
        "edges": [],
    }


def _semantic(nodes, edges):
    return {
        "kind": "concept_semantic",
        "nodes": [{"id": n, "label": lab, "frequency": f} for (n, lab, f) in nodes],
        "edges": edges,
    }


def _build_index(tmp_path: Path) -> dict:
    _write_course(
        tmp_path,
        "course-a",
        _untyped([("accessibility", "Accessibility", 10), ("udl", "UDL", 6)]),
        typed=_semantic(
            [("accessibility", "Accessibility", 10), ("udl", "UDL", 6)],
            [{"source": "accessibility", "target": "udl", "type": "related-to",
              "confidence": 0.6}],
        ),
    )
    _write_course(
        tmp_path,
        "course-b",
        _untyped([("accessibility", "Accessibility", 4), ("udl", "UDL", 2),
                  ("solo-b", "Solo B", 1)]),
    )
    out = tmp_path / "LibV2" / "catalog" / "cross_package_concepts.json"
    return write_cross_package_index(tmp_path, out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(CrossPackageIndexMissing):
        load_cross_package_index(tmp_path)


def test_load_unreadable_index_raises(tmp_path: Path) -> None:
    path = default_index_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CrossPackageIndexUnreadable):
        load_cross_package_index(tmp_path)


def test_load_default_path_matches_writer(tmp_path: Path) -> None:
    _build_index(tmp_path)
    # Loading via repo_root resolves the same path the writer used.
    index = load_cross_package_index(tmp_path)
    assert index["concept_count"] == 3
    assert default_index_path(tmp_path).is_file()


# ---------------------------------------------------------------------------
# search_concepts
# ---------------------------------------------------------------------------


def test_search_matches_id_and_label_case_insensitive(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    # Match by label substring, different case.
    hits = search_concepts(index, "ACCESS")
    assert [h["concept_id"] for h in hits] == ["accessibility"]
    acc = hits[0]
    assert acc["total_courses"] == 2
    assert sorted(c["slug"] for c in acc["courses"]) == ["course-a", "course-b"]
    assert acc["cross_package_edge_count"] == 1


def test_search_ordering_shared_before_singleton(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    # Empty query lists all, richest-coverage first, tie-break by id.
    hits = search_concepts(index, "", limit=10)
    ids = [h["concept_id"] for h in hits]
    # accessibility + udl share 2 courses (sorted alpha), solo-b is a singleton.
    assert ids == ["accessibility", "udl", "solo-b"]


def test_search_min_courses_filters_singletons(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    hits = search_concepts(index, "", min_courses=2)
    assert {h["concept_id"] for h in hits} == {"accessibility", "udl"}


def test_search_no_fabrication_on_unknown_query(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    assert search_concepts(index, "no-such-topic") == []


# ---------------------------------------------------------------------------
# courses_for_concept / related_concepts
# ---------------------------------------------------------------------------


def test_courses_for_concept_exact_lookup(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    courses = courses_for_concept(index, "udl")
    assert sorted(c["slug"] for c in courses) == ["course-a", "course-b"]
    assert courses_for_concept(index, "does-not-exist") == []


def test_related_concepts_surfaces_cross_package_edges(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    edges = related_concepts(index, "accessibility")
    assert len(edges) == 1
    assert edges[0]["target_concept"] == "udl"
    assert related_concepts(index, "does-not-exist") == []


# ---------------------------------------------------------------------------
# discover_courses (library-wide routing)
# ---------------------------------------------------------------------------


def test_discover_routes_query_to_candidate_courses(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    result = discover_courses(index, "udl")
    slugs = [c["slug"] for c in result["courses"]]
    assert set(slugs) == {"course-a", "course-b"}
    # course-a teaches udl with higher frequency -> ranked first on the tie.
    assert slugs[0] == "course-a"
    assert result["matched_concepts"][0]["concept_id"] == "udl"


def test_discover_ranks_by_matched_concept_count(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    # Both accessibility and udl match "a"? Use a query hitting both concepts:
    result = discover_courses(index, "", min_courses=2)
    # Both courses teach both shared concepts -> matched_concept_count == 2.
    for c in result["courses"]:
        assert c["matched_concept_count"] == 2
    # course-a has higher summed frequency (16 vs 6) -> first.
    assert result["courses"][0]["slug"] == "course-a"


def test_discover_empty_on_unknown_query(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    result = discover_courses(index, "quantum-chromodynamics")
    assert result["courses"] == []
    assert result["matched_concepts"] == []


def test_discover_deterministic(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    a = discover_courses(index, "access")
    b = discover_courses(index, "access")
    assert json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
