"""Tests for Wave 75 Worker C — real pedagogy graph builder.

Asserts node-class invariants, per-relation edge counts on a small
synthetic corpus, plus a regression that the rdf-shacl-550 archive
regenerates with >= 800 edges (the floor stated in the task).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from Trainforge.pedagogy_graph_builder import build_pedagogy_graph

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Synthetic fixture: 5 chunks + 3 objectives + 2 modules.
# ---------------------------------------------------------------------------


def _objectives() -> Dict[str, Any]:
    return {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Analyze RDF triples.",
                "bloom_level": "analyze",
            },
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "statement": "Identify subject/predicate/object.",
                "parent_to": "TO-01",
                "bloom_level": "remember",
                "week": 1,
            },
            {
                "id": "CO-02",
                "statement": "Distinguish IRIs and literals.",
                "parent_to": "TO-01",
                "bloom_level": "understand",
                "week": 2,
            },
        ],
    }


def _chunks() -> List[Dict[str, Any]]:
    return [
        # 1) explanation chunk teaching CO-01
        {
            "id": "ck_001",
            "chunk_type": "explanation",
            "concept_tags": ["triples", "subject"],
            "learning_outcome_refs": ["co-01"],
            "source": {
                "module_id": "week_01",
                "item_path": "week_01/content_01.html",
            },
            "misconceptions": [
                {
                    "misconception": "Triples are like rows in a table.",
                    "correction": "They are graph statements.",
                }
            ],
        },
        # 2) example chunk exemplifying triples + IRIs
        {
            "id": "ck_002",
            "chunk_type": "example",
            "concept_tags": ["triples", "iri"],
            "learning_outcome_refs": ["co-01", "co-02"],
            "source": {
                "module_id": "week_01",
                "item_path": "week_01/content_02.html",
            },
        },
        # 3) exercise chunk practicing CO-02
        {
            "id": "ck_003",
            "chunk_type": "exercise",
            "concept_tags": ["iri"],
            "learning_outcome_refs": ["co-02"],
            "source": {
                "module_id": "week_02",
                "item_path": "week_02/application.html",
            },
        },
        # 4) assessment_item chunk assessing CO-01
        {
            "id": "ck_004",
            "chunk_type": "assessment_item",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "source": {
                "module_id": "week_01_self_check",
                "item_path": "week_01/self_check.html",
            },
        },
        # 5) explanation chunk on quiz page → routes through assesses
        {
            "id": "ck_005",
            "chunk_type": "explanation",
            "concept_tags": ["literal"],
            "learning_outcome_refs": ["co-02"],
            "source": {
                "module_id": "week_02_self_check",
                "item_path": "week_02/self_check.html",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_node_class_counts_match_inputs():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    counts = g["stats"]["nodes_by_class"]
    # 1 TO + 2 CO = 3 objective nodes
    assert counts["Outcome"] == 1
    assert counts["ComponentObjective"] == 2
    assert counts["Chunk"] == 5
    # Two top-level weeks: week_01 + week_02
    assert counts["Module"] == 2
    # Six fixed Bloom levels.
    assert counts["BloomLevel"] == 6
    # Concept nodes: triples, subject, iri, literal — at least 3 are
    # referenced by exemplifies/interferes_with edges and must be emitted.
    assert counts.get("Concept", 0) >= 3
    # One unique misconception.
    assert counts["Misconception"] == 1


def test_teaches_edge_count_matches_lo_refs_for_non_assessment():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    teaches = [e for e in g["edges"] if e["relation_type"] == "teaches"]
    # ck_001 -> co-01      (1)
    # ck_002 -> co-01,co-02 (2; example chunks still teach)
    # ck_003 is exercise → practices, not teaches
    # ck_004 is assessment_item → assesses, not teaches
    # ck_005 is explanation but on quiz page → assesses, not teaches
    assert len(teaches) == 3
    pairs = {(e["source"], e["target"]) for e in teaches}
    assert ("ck_001", "CO-01") in pairs
    assert ("ck_002", "CO-01") in pairs
    assert ("ck_002", "CO-02") in pairs


def test_assesses_edges_only_for_assessment_chunks_and_quiz_pages():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    assesses = [e for e in g["edges"] if e["relation_type"] == "assesses"]
    sources = {e["source"] for e in assesses}
    # ck_004 is assessment_item; ck_005 is explanation on a self_check
    # page → both should count as assesses.
    assert sources == {"ck_004", "ck_005"}
    # And nothing else should claim to assess.
    explanation_ids = {"ck_001", "ck_002", "ck_003"}
    assert sources.isdisjoint(explanation_ids)


def test_practices_edge_for_exercise_chunk():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    practices = [e for e in g["edges"] if e["relation_type"] == "practices"]
    assert len(practices) == 1
    assert practices[0]["source"] == "ck_003"
    assert practices[0]["target"] == "CO-02"


def test_supports_outcome_edge_for_every_co():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    supports = [e for e in g["edges"] if e["relation_type"] == "supports_outcome"]
    # 2 COs, both parented to TO-01.
    assert len(supports) == 2
    targets = {e["target"] for e in supports}
    assert targets == {"TO-01"}
    sources = {e["source"] for e in supports}
    assert sources == {"CO-01", "CO-02"}


def test_at_bloom_level_edges_for_each_objective():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    bloom_edges = [e for e in g["edges"] if e["relation_type"] == "at_bloom_level"]
    # 1 TO + 2 CO with valid bloom levels.
    assert len(bloom_edges) == 3
    pairs = {(e["source"], e["target"]) for e in bloom_edges}
    assert ("TO-01", "bloom:analyze") in pairs
    assert ("CO-01", "bloom:remember") in pairs
    assert ("CO-02", "bloom:understand") in pairs


def test_follows_edges_chain_modules_in_order():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    follows = [e for e in g["edges"] if e["relation_type"] == "follows"]
    # 2 modules → 1 transition.
    assert len(follows) == 1
    assert follows[0]["source"] == "module:week_01"
    assert follows[0]["target"] == "module:week_02"


def test_belongs_to_module_one_per_chunk():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    belongs = [e for e in g["edges"] if e["relation_type"] == "belongs_to_module"]
    # One per chunk; quiz pages collapse to their week_NN slice.
    assert len(belongs) == 5
    targets = {e["target"] for e in belongs}
    assert targets == {"module:week_01", "module:week_02"}


def test_exemplifies_only_for_example_chunks():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    exempl = [e for e in g["edges"] if e["relation_type"] == "exemplifies"]
    # ck_002 is the only example; emits one edge per concept_tag.
    sources = {e["source"] for e in exempl}
    assert sources == {"ck_002"}
    targets = {e["target"] for e in exempl}
    assert targets == {"concept:triples", "concept:iri"}


def test_misconception_node_and_interferes_with_edge():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    mcs = [n for n in g["nodes"] if n["class"] == "Misconception"]
    assert len(mcs) == 1
    interferes = [e for e in g["edges"] if e["relation_type"] == "interferes_with"]
    assert len(interferes) >= 1
    # All interferes_with edges should originate from the misconception
    # node id.
    mc_id = mcs[0]["id"]
    assert all(e["source"] == mc_id for e in interferes)


def test_every_node_has_class_field():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    for n in g["nodes"]:
        assert "class" in n, f"node missing class: {n}"


def test_every_edge_has_relation_type():
    g = build_pedagogy_graph(_chunks(), _objectives(), course_id="T_001")
    for e in g["edges"]:
        assert "relation_type" in e, f"edge missing relation_type: {e}"


def test_empty_input_no_crash():
    g = build_pedagogy_graph([], {})
    # Bloom levels are intrinsic structure; no chunks / objectives /
    # modules / concepts emitted from empty input. Crucially: no crash.
    counts = g["stats"]["nodes_by_class"]
    assert counts == {"BloomLevel": 6}
    assert g["edges"] == []


def test_lo_ref_normalization_handles_compound_refs():
    """A chunk with 'co-01,co-02,co-03' as a single ref entry should split."""
    chunks = [
        {
            "id": "ck_x",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01,co-02"],
            "source": {"module_id": "week_01", "item_path": "week_01/x.html"},
        }
    ]
    g = build_pedagogy_graph(chunks, _objectives())
    teaches = [e for e in g["edges"] if e["relation_type"] == "teaches"]
    targets = {e["target"] for e in teaches}
    assert "CO-01" in targets
    assert "CO-02" in targets


# ---------------------------------------------------------------------------
# Regression: real archive must re-build with >= 800 edges.
# ---------------------------------------------------------------------------


CORPUS_CHUNKS = (
    ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
    / "corpus"
    / "chunks.jsonl"
)
SYNTH_OBJECTIVES = (
    ROOT
    / "Courseforge"
    / "exports"
    / "PROJ-RDF_SHACL_550-20260424135037"
    / "01_learning_objectives"
    / "synthesized_objectives.json"
)


@pytest.mark.skipif(
    not (CORPUS_CHUNKS.exists() and SYNTH_OBJECTIVES.exists()),
    reason="rdf-shacl-550 archive missing — regression skipped",
)
def test_real_archive_regen_has_at_least_800_edges():
    chunks = []
    with open(CORPUS_CHUNKS, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    with open(SYNTH_OBJECTIVES, encoding="utf-8") as f:
        objectives = json.load(f)

    g = build_pedagogy_graph(chunks, objectives, course_id="RDF_SHACL_550")

    # Floor-asserts from the task spec.
    assert g["stats"]["edge_count"] >= 800, (
        f"expected >= 800 edges, got {g['stats']['edge_count']}"
    )
    er = g["stats"]["edges_by_relation"]
    assert er.get("teaches", 0) >= 219, er.get("teaches", 0)
    assert er.get("belongs_to_module", 0) == 219, er.get("belongs_to_module", 0)
    assert er.get("supports_outcome", 0) == 29, er.get("supports_outcome", 0)
    assert er.get("at_bloom_level", 0) == 36, er.get("at_bloom_level", 0)
    assert er.get("follows", 0) == 11, er.get("follows", 0)

    # Node counts: floor of (36 objectives + 219 chunks + 12 modules
    # + 6 bloom levels) ~ 273.
    assert g["stats"]["node_count"] >= 273, g["stats"]["node_count"]
    nc = g["stats"]["nodes_by_class"]
    assert nc.get("Outcome", 0) == 7
    assert nc.get("ComponentObjective", 0) == 29
    assert nc.get("Chunk", 0) == 219
    assert nc.get("Module", 0) == 12
    assert nc.get("BloomLevel", 0) == 6
