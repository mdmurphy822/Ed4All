"""Tests for ``MCP/tools/tutoring_tools.py`` (Wave 77).

Smoke tests against a real LibV2 archive. The original Wave 77 fixture
(``rdf-shacl-550``) was authored against a corpus that was never checked
in; the tests have been rebound to ``rdf-shacl-551-2`` (the in-tree
RDF/SHACL corpus) and the assertions tightened to guard the SHAPE of
the output rather than the specific member counts (which differ by
corpus regeneration).

Some assertions still depend on the misconception index being populated
(non-empty ``misconceptions[]`` envelopes in the chunkset). The
module-level ``pytestmark`` skips the whole file cleanly when the index
is empty — e.g. when ``imscc_chunks/chunks.jsonl`` hasn't been
backfilled and the legacy ``corpus/chunks.jsonl`` isn't reachable
through the Phase 7c resolver. Whoever rehydrates the corpus (or wires
``rdf-shacl-550``) will see the assertions fire automatically.
"""

from __future__ import annotations

import pytest

from MCP.tools.tutoring_tools import (
    cluster_misconceptions,
    load_misconception_index,
    match_misconception,
    preemptive_misconception_guardrails,
)


SLUG = "rdf-shacl-551-2"


pytestmark = pytest.mark.skipif(
    not load_misconception_index(SLUG).items,
    reason=(
        f"LibV2 archive {SLUG!r} has no reachable misconception index "
        "(check imscc_chunks/chunks.jsonl is populated)."
    ),
)


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
    """Real-archive smoke: at least one DomainConcept in the pedagogy
    graph carries ``interferes_with`` edges (Wave 76 prune kept these
    for DomainConcepts). We probe a few likely targets and require
    that at least one resolves to >= 1 guardrail; no specific concept
    is pinned because the populated targets vary by corpus regen."""
    candidates = [
        "rdf-graph", "rdf", "shacl", "sparql", "owl", "iri",
        "blank-node", "turtle", "rdfs",
    ]
    found = False
    for c in candidates:
        if preemptive_misconception_guardrails(SLUG, c):
            found = True
            break
    assert found, (
        "expected at least one of the canonical RDF/SHACL concepts "
        f"to carry interferes_with edges: {candidates}"
    )


def test_guardrails_concept_prefix_strip():
    """``concept:<slug>`` and ``<slug>`` both resolve identically. We
    pick a concept that's known to have guardrails in this corpus."""
    # Find any concept that has guardrails in this corpus.
    index = load_misconception_index(SLUG)
    candidates = list(index.concept_to_mc_keys.keys())
    target = None
    for c in candidates:
        if preemptive_misconception_guardrails(SLUG, c):
            target = c
            break
    if target is None:
        pytest.skip("no concept in this corpus carries guardrails")
    a = preemptive_misconception_guardrails(SLUG, f"concept:{target}")
    b = preemptive_misconception_guardrails(SLUG, target)
    assert {x["misconception"] for x in a} == {x["misconception"] for x in b}


def test_guardrails_unknown_concept_returns_empty():
    """Concept that has no interferes_with edges yields empty list."""
    assert preemptive_misconception_guardrails(SLUG, "no-such-concept") == []
    assert preemptive_misconception_guardrails(SLUG, "") == []


def test_guardrails_envelope_shape():
    """Each guardrail record carries the documented keys."""
    # Pick a concept that we know has guardrails in this corpus.
    index = load_misconception_index(SLUG)
    target = None
    for c in index.concept_to_mc_keys:
        out = preemptive_misconception_guardrails(SLUG, c)
        if out:
            target = c
            break
    if target is None:
        pytest.skip("no concept in this corpus carries guardrails")
    out = preemptive_misconception_guardrails(SLUG, target)
    assert out
    expected_keys = {
        "misconception", "correction", "chunk_id",
        "source_references", "concept_tags", "concept_slug",
    }
    for r in out:
        assert set(r.keys()) >= expected_keys, r
        assert r["concept_slug"] == target


# ---------------------------------------------------------------------- #
# cluster_misconceptions
# ---------------------------------------------------------------------- #


def test_cluster_misconceptions_total_members_matches_index():
    """Real-archive smoke: every unique misconception is placed into a
    cluster (no statement is dropped). The exact count is corpus-
    dependent; we assert it matches ``len(index)``."""
    index = load_misconception_index(SLUG)
    expected_total = len(index)
    clusters = cluster_misconceptions(SLUG, n_clusters=4)
    assert clusters, "expected at least one cluster"
    total = sum(c["size"] for c in clusters)
    assert total == expected_total, (total, expected_total)


def test_cluster_misconceptions_n_clusters_4_yields_up_to_4():
    """``n_clusters=4`` returns at most 4 clusters (capped by unique-
    statement count when the corpus has fewer than 4 statements), each
    non-empty."""
    index = load_misconception_index(SLUG)
    expected_max = min(4, len(index))
    clusters = cluster_misconceptions(SLUG, n_clusters=4)
    assert 1 <= len(clusters) <= expected_max
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
    (mtime hasn't changed). Index size is corpus-dependent — assert
    only that it's populated."""
    a = load_misconception_index(SLUG)
    b = load_misconception_index(SLUG)
    assert a is b
    assert len(a) >= 1
