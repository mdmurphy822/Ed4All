"""Tests for ``MCP/tools/tutoring_tools.py`` (Wave 77).

Smoke tests against the real ``rdf-shacl-550`` LibV2 archive — that's
the canonical fixture this module is built around (67 unique
misconception/correction pairs anchored in 76% of chunks). The archive
is checked into the repo under ``LibV2/courses/rdf-shacl-550-rdf-shacl-550/``,
so these tests don't need network or generated fixtures.
"""

from __future__ import annotations

import pytest

from MCP.tools.tutoring_tools import (
    cluster_misconceptions,
    load_misconception_index,
    match_misconception,
    preemptive_misconception_guardrails,
)


SLUG = "rdf-shacl-550"


# ---------------------------------------------------------------------- #
# match_misconception
# ---------------------------------------------------------------------- #


def test_match_misconception_relational_table_smoke():
    """Real-archive smoke: 'RDF triple is like a row in a relational
    table' should top-1 match the editorial misconception of the same
    framing with similarity > 0.5 and contain 'row' or 'relational' in
    the matched misconception OR correction."""
    results = match_misconception(
        SLUG,
        "An RDF triple is like a row in a relational table",
        top_k=5,
    )
    assert results, "expected at least one match"
    top = results[0]
    assert top["score"] > 0.5, top
    haystack = (top["misconception"] + " " + top["correction"]).lower()
    assert "row" in haystack or "relational" in haystack, top


def test_match_misconception_envelope_shape():
    """Each match record carries the documented keys."""
    results = match_misconception(SLUG, "RDF triple row table", top_k=3)
    assert results
    expected_keys = {
        "misconception", "correction", "chunk_id",
        "source_references", "concept_tags", "score", "backend",
    }
    for r in results:
        assert set(r.keys()) >= expected_keys, r
        assert isinstance(r["misconception"], str)
        assert isinstance(r["correction"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["source_references"], list)
        assert isinstance(r["concept_tags"], list)
        assert r["backend"] in {"embedding", "tfidf", "bm25", "jaccard"}


def test_match_misconception_empty_text_returns_empty():
    """Empty student input yields empty result list (no spurious matches)."""
    assert match_misconception(SLUG, "", top_k=5) == []
    assert match_misconception(SLUG, "   ", top_k=5) == []


def test_match_misconception_unknown_slug_returns_empty():
    """Slug with no archive (or no misconceptions) yields empty list."""
    out = match_misconception("does-not-exist-slug", "anything", top_k=5)
    assert out == []


def test_match_misconception_top_k_respected():
    """``top_k=2`` returns at most 2 results."""
    results = match_misconception(
        SLUG, "An RDF triple is like a row", top_k=2,
    )
    assert len(results) <= 2


# ---------------------------------------------------------------------- #
# preemptive_misconception_guardrails
# ---------------------------------------------------------------------- #


def test_guardrails_rdf_graph_has_at_least_one():
    """Real-archive smoke: ``rdf-graph`` has interferes_with edges from
    at least one misconception in pedagogy_graph.json (Wave 76 prune
    kept these because rdf-graph is a DomainConcept)."""
    out = preemptive_misconception_guardrails(SLUG, "rdf-graph")
    assert len(out) >= 1, out


def test_guardrails_concept_prefix_strip():
    """``concept:rdf-graph`` and ``rdf-graph`` both resolve."""
    a = preemptive_misconception_guardrails(SLUG, "concept:rdf-graph")
    b = preemptive_misconception_guardrails(SLUG, "rdf-graph")
    assert {x["misconception"] for x in a} == {x["misconception"] for x in b}


def test_guardrails_unknown_concept_returns_empty():
    """Concept that has no interferes_with edges yields empty list."""
    assert preemptive_misconception_guardrails(SLUG, "no-such-concept") == []
    assert preemptive_misconception_guardrails(SLUG, "") == []


def test_guardrails_envelope_shape():
    """Each guardrail record carries the documented keys."""
    out = preemptive_misconception_guardrails(SLUG, "rdf-graph")
    assert out
    expected_keys = {
        "misconception", "correction", "chunk_id",
        "source_references", "concept_tags", "concept_slug",
    }
    for r in out:
        assert set(r.keys()) >= expected_keys, r
        assert r["concept_slug"] == "rdf-graph"


# ---------------------------------------------------------------------- #
# cluster_misconceptions
# ---------------------------------------------------------------------- #


def test_cluster_misconceptions_total_members_is_67():
    """Real-archive smoke: 67 unique misconceptions -> all 67 placed
    into clusters (no statement is dropped)."""
    clusters = cluster_misconceptions(SLUG, n_clusters=4)
    assert clusters, "expected at least one cluster"
    total = sum(c["size"] for c in clusters)
    assert total == 67, total


def test_cluster_misconceptions_n_clusters_4_yields_4():
    """``n_clusters=4`` returns 4 clusters, each non-empty."""
    clusters = cluster_misconceptions(SLUG, n_clusters=4)
    assert len(clusters) == 4
    for c in clusters:
        assert c["size"] >= 1


def test_cluster_misconceptions_envelope_shape():
    """Each cluster record carries the documented keys."""
    clusters = cluster_misconceptions(SLUG, n_clusters=4)
    expected_keys = {"label", "members", "size", "canonical_correction", "backend"}
    for c in clusters:
        assert set(c.keys()) >= expected_keys, c
        assert isinstance(c["label"], str) and c["label"]
        assert isinstance(c["members"], list) and c["members"]
        assert isinstance(c["size"], int) and c["size"] == len(c["members"])
        assert isinstance(c["canonical_correction"], str)
        assert c["backend"] in {"kmeans", "greedy"}


def test_cluster_misconceptions_unknown_slug_returns_empty():
    """No archive -> empty cluster list."""
    assert cluster_misconceptions("does-not-exist-slug", n_clusters=4) == []


# ---------------------------------------------------------------------- #
# load_misconception_index (cache)
# ---------------------------------------------------------------------- #


def test_load_index_cached_by_mtime():
    """Repeated calls with the same slug return the same cached object
    (mtime hasn't changed)."""
    a = load_misconception_index(SLUG)
    b = load_misconception_index(SLUG)
    assert a is b
    # And we got 67 unique misconceptions.
    assert len(a) == 67
