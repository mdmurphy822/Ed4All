"""M4 — degenerate-grouping fallback for the cooccurrence pair-counting unit.

Root-cause regression net for the 7B-build KG dead-backbone bug: a DART
converter that stamps a SINGLE ``lesson_id`` over an entire multi-chapter PDF
folds every chunk into ONE page-group. With ``group_by="page"`` every pair
then lands ``weight == 1`` (one group), and the downstream
``related_from_cooccurrence`` rule (``weight >= 3``) emits ZERO ``related-to``
edges — the KG backbone dies (measured 0 vs 556 on the Sonnet baseline).

The fix: when the coarse grouping level resolves into fewer than
``_MIN_GROUPS_FOR_THRESHOLD`` distinct REAL groups, step DOWN the ladder
(page -> section -> chunk) for pair-counting. This is real co-occurrence at a
valid finer window — not fabricated edges. Gated by
``TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK`` (default-off at the bare-library
level for W3 byte-stability; turned on for pipeline runs by
``_apply_corpus_generalization_defaults``).
"""

from __future__ import annotations

import pytest

from lib.ontology.cooccurrence_graph import build_cooccurrence_graph
from Trainforge.rag.inference_rules import related_from_cooccurrence as R


@pytest.fixture(autouse=True)
def _single_occurrence(monkeypatch):
    # Drop the 2-chunk node-mint floor so the tiny fixtures mint every tag as a
    # node; the cooccurrence pair structure is then observable without padding.
    monkeypatch.setenv("TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", "true")
    yield


def _edge_pairs(graph) -> dict:
    return {
        tuple(sorted([e["source"], e["target"]])): e["weight"]
        for e in graph["edges"]
    }


def _node_signature(graph) -> list:
    return sorted(
        (n["id"], n["frequency"], tuple(n.get("occurrences", [])))
        for n in graph["nodes"]
    )


# Six chunks that ALL share a single lesson_id (the DART single-page-collapse
# failure mode) but live in distinct SECTIONS. Every adjacent pair of chunks
# overlaps on a concept so a pair co-occurs in >= 3 distinct sections.
def _collapsed_page_corpus():
    # concepts chained so a, b, c each appear across >= 3 sections:
    #   s0 {a,b}  s1 {a,b}  s2 {a,b}  s3 {b,c}  s4 {b,c}  s5 {b,c}
    # pair (a,b) co-occurs in sections s0,s1,s2 -> weight 3 at section level.
    # pair (b,c) co-occurs in sections s3,s4,s5 -> weight 3 at section level.
    tagsets = [
        ["alpha", "beta"], ["alpha", "beta"], ["alpha", "beta"],
        ["beta", "gamma"], ["beta", "gamma"], ["beta", "gamma"],
    ]
    return [
        {
            "id": f"c{i}",
            "concept_tags": ts,
            # ALL chunks share one lesson_id (the collapse); sections differ.
            "source": {"lesson_id": "whole_pdf", "section_heading": f"sec_{i}"},
        }
        for i, ts in enumerate(tagsets)
    ]


# --------------------------------------------------------------------------
# The core regression: page-collapse kills the rule WITHOUT the fallback.
# --------------------------------------------------------------------------
def test_collapsed_page_emits_zero_without_fallback(monkeypatch):
    monkeypatch.delenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", raising=False)
    chunks = _collapsed_page_corpus()
    cog = build_cooccurrence_graph(chunks, group_by="page")
    # One page-group -> every pair weight == 1.
    assert all(w == 1 for w in _edge_pairs(cog).values())
    # The related-to rule (weight >= 3) therefore emits NOTHING — the bug.
    assert R.infer([], None, cog, threshold=R.DEFAULT_THRESHOLD) == []


def test_collapsed_page_recovers_with_fallback(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", "true")
    chunks = _collapsed_page_corpus()
    cog = build_cooccurrence_graph(chunks, group_by="page")
    pairs = _edge_pairs(cog)
    # Fell through page (1 group) -> section (6 groups). Both real pairs reach
    # weight 3 (each co-occurs in 3 distinct sections).
    assert pairs.get(("alpha", "beta")) == 3
    assert pairs.get(("beta", "gamma")) == 3
    # And the related-to rule now emits the genuine co-occurrence edges.
    emitted = R.infer([], None, cog, threshold=R.DEFAULT_THRESHOLD)
    emitted_pairs = {tuple(sorted([e["source"], e["target"]])) for e in emitted}
    assert ("alpha", "beta") in emitted_pairs
    assert ("beta", "gamma") in emitted_pairs


def test_fallback_preserves_node_provenance(monkeypatch):
    # Nodes + frequency + occurrences[] are chunk-level and MUST be identical
    # whether the fallback fires or not (only pair-counting moves).
    chunks = _collapsed_page_corpus()
    monkeypatch.delenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", raising=False)
    no_fb = build_cooccurrence_graph(chunks, group_by="page")
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", "true")
    fb = build_cooccurrence_graph(chunks, group_by="page")
    assert _node_signature(no_fb) == _node_signature(fb)


def test_fallback_noop_when_enough_groups(monkeypatch):
    # A corpus that resolves into >= 3 page-groups must be BYTE-IDENTICAL with
    # the fallback on vs off (the guard only fires in the degenerate case).
    chunks = []
    for page in range(4):
        chunks.append({
            "id": f"p{page}_a",
            "concept_tags": ["alpha", "beta"],
            "source": {"lesson_id": f"page_{page}", "section_heading": f"s{page}"},
        })
    monkeypatch.delenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", raising=False)
    off = _edge_pairs(build_cooccurrence_graph(chunks, group_by="page"))
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", "true")
    on = _edge_pairs(build_cooccurrence_graph(chunks, group_by="page"))
    assert off == on
    # 4 pages share (alpha,beta) -> weight 4 either way.
    assert on.get(("alpha", "beta")) == 4


def test_genuine_no_cooccurrence_still_emits_zero(monkeypatch):
    # The guard must NOT fabricate edges. Concepts that genuinely never share
    # ANY window (each its own chunk AND its own section) stay disconnected even
    # after the page->section->chunk fall-through.
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", "true")
    chunks = [
        {
            "id": "c0",
            "concept_tags": ["alpha"],
            "source": {"lesson_id": "whole_pdf", "section_heading": "s0"},
        },
        {
            "id": "c1",
            "concept_tags": ["beta"],
            "source": {"lesson_id": "whole_pdf", "section_heading": "s1"},
        },
    ]
    cog = build_cooccurrence_graph(chunks, group_by="page")
    # alpha and beta never co-occur in any chunk OR section -> no edge.
    assert ("alpha", "beta") not in _edge_pairs(cog)
    assert R.infer([], None, cog, threshold=R.DEFAULT_THRESHOLD) == []


def test_disabled_value_keeps_strict_coarse(monkeypatch):
    # An explicit falsey value forces the strict page-only behaviour even on the
    # degenerate corpus (operator override).
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK", "false")
    chunks = _collapsed_page_corpus()
    cog = build_cooccurrence_graph(chunks, group_by="page")
    assert all(w == 1 for w in _edge_pairs(cog).values())
    assert R.infer([], None, cog, threshold=R.DEFAULT_THRESHOLD) == []
