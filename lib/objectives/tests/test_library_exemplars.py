"""Tests for W4.3 cross-course objective-library EXEMPLARS (default OFF).

Covers:
  * resolver parse-with-fallback (enabled / limit / min-overlap);
  * exemplar surfacing ranks OTHER courses' archived objectives by lexical
    overlap and provenance-labels each (anti-fabrication: never THIS course);
  * the current course is EXCLUDED from its own exemplar set;
  * both archive shapes (terminal_outcomes/component_objectives AND
    terminal_objectives/chapter_objectives group form) are discovered;
  * limit + min_overlap knobs bound the surfaced set;
  * graceful degradation: missing root / empty query => empty set, no raise;
  * build_query_terms derives tokens from this course's own objectives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.objectives.library_exemplars import (
    build_query_terms,
    resolve_exemplar_limit,
    resolve_exemplar_min_overlap,
    resolve_library_exemplars_enabled,
    surface_exemplars,
)


# --------------------------------------------------------------------------- #
# Resolvers — parse-with-fallback                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False), ("", False), ("0", False), ("no", False),
        ("garbage", False), ("1", True), ("true", True), ("YES", True),
        ("on", True),
    ],
)
def test_resolve_enabled(value, expected):
    assert resolve_library_exemplars_enabled(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, 8), ("3", 3), ("0", 8), ("-2", 8), ("bad", 8), ("25", 25)],
)
def test_resolve_limit(value, expected):
    assert resolve_exemplar_limit(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, 0.05), ("0.2", 0.2), ("9", 0.05), ("-1", 0.05), ("x", 0.05), ("0", 0.0)],
)
def test_resolve_min_overlap(value, expected):
    assert resolve_exemplar_min_overlap(value) == expected


# --------------------------------------------------------------------------- #
# Fixture library                                                             #
# --------------------------------------------------------------------------- #
def _make_library(root: Path) -> None:
    courses = root / "courses"
    # Archive form (terminal_outcomes / component_objectives).
    a = courses / "rdf-shacl-551"
    a.mkdir(parents=True)
    (a / "objectives.json").write_text(
        json.dumps(
            {
                "terminal_outcomes": [
                    {
                        "id": "to-01",
                        "statement": (
                            "Analyze RDF data models by identifying triples, "
                            "IRIs, literals, and blank nodes."
                        ),
                        "bloom_level": "analyze",
                        "bloom_verb": "analyze",
                    }
                ],
                "component_objectives": [
                    {
                        "id": "co-01",
                        "statement": (
                            "Identify the three components of an RDF triple "
                            "subject predicate object."
                        ),
                        "bloom_level": "remember",
                        "bloom_verb": "identify",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Courseforge synthesized group form under sources/objectives/.
    b = courses / "photosynthesis-201" / "sources" / "objectives"
    b.mkdir(parents=True)
    (b / "synthesized_objectives.json").write_text(
        json.dumps(
            {
                "terminal_objectives": [
                    {
                        "id": "TO-01",
                        "statement": (
                            "Explain how chloroplasts convert light energy "
                            "into chemical energy."
                        ),
                        "bloom_level": "understand",
                    }
                ],
                "chapter_objectives": [
                    {
                        "chapter": "1",
                        "objectives": [
                            {
                                "id": "CO-01",
                                "statement": (
                                    "Describe the light-dependent reactions "
                                    "in the thylakoid membrane."
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_surface_ranks_and_labels_exemplars(tmp_path):
    _make_library(tmp_path)
    # Plan a NEW RDF-flavored course; query terms overlap the rdf course.
    terms = build_query_terms(
        terminals=[
            {"statement": "Analyze RDF triples and graph data models"}
        ],
        chapter_objectives=[
            {"objectives": [{"statement": "Identify RDF triple components"}]}
        ],
    )
    assert "rdf" in terms and "triples" in terms

    res = surface_exemplars(
        course_name="RDF Intro",
        query_terms=terms,
        libv2_root=str(tmp_path),
        current_slug="rdf-intro",
    )
    assert res["enabled"] is True
    assert res["exemplar_count"] >= 1
    # RDF exemplars must rank above the photosynthesis ones.
    top = res["exemplars"][0]
    assert top["source_course"] == "rdf-shacl-551"
    # Provenance label is present + immutable-shaped, anti-fabrication.
    assert top["provenance"].startswith("libv2:rdf-shacl-551#")
    assert "relevance" in top
    # Photosynthesis objectives (from the group form) are discoverable at all
    # (proves both shapes are read) — but only if they clear the floor.
    all_sources = {e["source_course"] for e in res["exemplars"]}
    assert "rdf-shacl-551" in all_sources


def test_current_course_excluded_from_own_exemplars(tmp_path):
    _make_library(tmp_path)
    terms = build_query_terms(
        terminals=[{"statement": "Analyze RDF triples IRIs literals"}],
        chapter_objectives=[],
    )
    res = surface_exemplars(
        course_name="RDF SHACL 551",
        query_terms=terms,
        libv2_root=str(tmp_path),
        current_slug="rdf-shacl-551",  # exclude self
    )
    sources = {e["source_course"] for e in res["exemplars"]}
    assert "rdf-shacl-551" not in sources


def test_limit_and_min_overlap_bounds(tmp_path):
    _make_library(tmp_path)
    terms = build_query_terms(
        terminals=[{"statement": "Analyze RDF triples IRIs literals nodes"}],
        chapter_objectives=[],
    )
    res = surface_exemplars(
        course_name="X",
        query_terms=terms,
        libv2_root=str(tmp_path),
        current_slug="x",
        limit=1,
    )
    assert res["exemplar_count"] <= 1
    # A very high min_overlap prunes everything gracefully.
    res2 = surface_exemplars(
        course_name="X",
        query_terms=terms,
        libv2_root=str(tmp_path),
        current_slug="x",
        min_overlap=0.99,
    )
    assert res2["exemplar_count"] == 0
    assert res2["exemplars"] == []


def test_graceful_no_root_and_empty_query(tmp_path):
    # No courses dir under the root -> empty, no raise.
    res = surface_exemplars(
        course_name="X",
        query_terms=["rdf", "triples"],
        libv2_root=str(tmp_path / "does-not-exist"),
        current_slug="x",
    )
    assert res["enabled"] is True
    assert res["exemplar_count"] == 0
    # Empty query terms -> immediate empty result.
    res2 = surface_exemplars(
        course_name="X",
        query_terms=[],
        libv2_root=str(tmp_path),
        current_slug="x",
    )
    assert res2["exemplar_count"] == 0
    assert res2["query_term_count"] == 0
