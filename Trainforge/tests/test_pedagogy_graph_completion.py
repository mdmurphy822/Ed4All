"""Wave 78 Worker B — pedagogy graph relation-set completion.

Asserts the four new edge types added to ``build_pedagogy_graph`` so
the strict validator (Worker A) and intent router (Worker C) have
complete substrate to operate on:

* ``derived_from_objective`` — Chunk → Objective provenance, mirrored
  from concept_graph_semantic into the pedagogy graph.
* ``concept_supports_outcome`` — DomainConcept → Outcome rollup,
  derived from concept ∩ chunk LO refs (CO refs rolled up to parent
  TO before counting). Filtered to DomainConcept-classified sources.
* ``assessment_validates_outcome`` — AssessmentItem chunk → Outcome
  rollup via the parent_terminal CO chain. Distinct from
  ``assesses`` (which targets the direct LO ref).
* ``chunk_at_difficulty`` — Chunk → DifficultyLevel typed node
  (foundational / intermediate / advanced).

A regression on the real rdf-shacl-550 archive asserts the post-Wave-
78 relation-type count (10 → 14) and the four edge counts land in
sensible envelopes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from Trainforge.pedagogy_graph_builder import build_pedagogy_graph

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Shared minimal objectives — TO-01 has parent CO-01 + CO-02; TO-04 has
# parent CO-25 (the "co-25 → to-04 rollup" assessment fixture below
# anchors the assessment_validates_outcome rollup test).
# ---------------------------------------------------------------------------


def _objectives_basic() -> Dict[str, Any]:
    return {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Reason about RDF triples.",
                "bloom_level": "analyze",
            },
            {
                "id": "TO-04",
                "statement": "Validate SHACL shapes.",
                "bloom_level": "evaluate",
            },
            {
                "id": "TO-07",
                "statement": "Apply named-graph patterns.",
                "bloom_level": "apply",
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
            {
                "id": "CO-25",
                "statement": "Validate SHACL node shapes.",
                "parent_to": "TO-07",
                "bloom_level": "apply",
                "week": 7,
            },
        ],
    }


# ---------------------------------------------------------------------------
# 1. derived_from_objective: per-chunk × per-ref provenance edge.
# ---------------------------------------------------------------------------


def test_derived_from_objective_emits_one_edge_per_chunk_per_ref():
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_a",
            "chunk_type": "explanation",
            "concept_tags": ["triples"],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "ck_b",
            "chunk_type": "example",
            "concept_tags": ["iri"],
            "learning_outcome_refs": ["co-01", "co-02"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "ck_c",
            "chunk_type": "assessment_item",
            "concept_tags": [],
            "learning_outcome_refs": ["co-25", "to-07"],
            "source": {"module_id": "week_07", "item_path": "week_07/q.html"},
        },
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    derived = [e for e in g["edges"] if e["relation_type"] == "derived_from_objective"]
    # Count = sum of (deduped) refs per chunk: 1 + 2 + 2 = 5.
    assert len(derived) == 5
    pairs = {(e["source"], e["target"]) for e in derived}
    assert ("ck_a", "CO-01") in pairs
    assert ("ck_b", "CO-01") in pairs
    assert ("ck_b", "CO-02") in pairs
    assert ("ck_c", "CO-25") in pairs
    assert ("ck_c", "TO-07") in pairs


def test_derived_from_objective_emits_for_assessment_chunks_too():
    """Distinct from teaches/assesses split — derived_from is uniform."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_q",
            "chunk_type": "assessment_item",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/q.html"},
        }
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    derived = [e for e in g["edges"] if e["relation_type"] == "derived_from_objective"]
    # The chunk also emits an assesses edge, but derived_from must
    # still fire — they are distinct semantic relations.
    assert len(derived) == 1
    assert derived[0]["source"] == "ck_q"
    assert derived[0]["target"] == "CO-01"
    assesses = [e for e in g["edges"] if e["relation_type"] == "assesses"]
    assert len(assesses) == 1


# ---------------------------------------------------------------------------
# 2. assessment_validates_outcome: assessment_item chunk → parent_TO rollup.
# ---------------------------------------------------------------------------


def test_assessment_validates_outcome_rolls_co_up_to_parent_to():
    """A co-25 ref on an assessment_item chunk emits an
    assessment_validates_outcome edge to to-07 (parent_terminal)."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_q",
            "chunk_type": "assessment_item",
            "concept_tags": [],
            "learning_outcome_refs": ["co-25"],
            "source": {"module_id": "week_07", "item_path": "week_07/q.html"},
        }
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    validates = [
        e for e in g["edges"] if e["relation_type"] == "assessment_validates_outcome"
    ]
    assert len(validates) == 1
    assert validates[0]["source"] == "ck_q"
    assert validates[0]["target"] == "TO-07"


def test_assessment_validates_outcome_dedupes_when_multiple_cos_share_parent():
    """An assessment chunk citing co-01 + co-02 (both rolling up to to-01)
    must emit exactly ONE validates edge to to-01."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_qmulti",
            "chunk_type": "assessment_item",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01", "co-02"],
            "source": {"module_id": "week_02", "item_path": "week_02/q.html"},
        }
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    validates = [
        e for e in g["edges"] if e["relation_type"] == "assessment_validates_outcome"
    ]
    assert len(validates) == 1
    assert validates[0]["target"] == "TO-01"


def test_assessment_validates_outcome_only_for_assessment_item_chunks():
    """Explanation / exercise / example chunks must NOT emit validates."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_exp",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "ck_ex",
            "chunk_type": "exercise",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/x.html"},
        },
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    validates = [
        e for e in g["edges"] if e["relation_type"] == "assessment_validates_outcome"
    ]
    assert validates == []


# ---------------------------------------------------------------------------
# 3. chunk_at_difficulty: Chunk → DifficultyLevel typed node.
# ---------------------------------------------------------------------------


def test_chunk_at_difficulty_emits_one_edge_per_foundational_chunk():
    """Three chunks all flagged 'foundational' → 3 edges, all to the
    same DifficultyLevel node id."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": f"ck_{i}",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "foundational",
            "source": {
                "module_id": "week_01",
                "item_path": f"week_01/p_{i}.html",
            },
        }
        for i in range(3)
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    diff_edges = [e for e in g["edges"] if e["relation_type"] == "chunk_at_difficulty"]
    assert len(diff_edges) == 3
    assert all(e["target"] == "difficulty:foundational" for e in diff_edges)
    sources = {e["source"] for e in diff_edges}
    assert sources == {"ck_0", "ck_1", "ck_2"}


def test_chunk_at_difficulty_routes_to_correct_level_node():
    """Mixed difficulties should each route to their level-specific node."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_f",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "foundational",
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "ck_i",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "intermediate",
            "source": {"module_id": "week_01", "item_path": "week_01/p2.html"},
        },
        {
            "id": "ck_a",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "advanced",
            "source": {"module_id": "week_01", "item_path": "week_01/p3.html"},
        },
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    diff_edges = {
        e["source"]: e["target"]
        for e in g["edges"]
        if e["relation_type"] == "chunk_at_difficulty"
    }
    assert diff_edges == {
        "ck_f": "difficulty:foundational",
        "ck_i": "difficulty:intermediate",
        "ck_a": "difficulty:advanced",
    }


def test_chunk_at_difficulty_skips_chunks_with_missing_or_invalid_difficulty():
    """A chunk without difficulty (legacy) emits no edge — fail-soft."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_legacy",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            # no 'difficulty' key
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "ck_garbage",
            "chunk_type": "explanation",
            "concept_tags": [],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "expert",  # not in the canonical enum
            "source": {"module_id": "week_01", "item_path": "week_01/p2.html"},
        },
    ]
    g = build_pedagogy_graph(chunks, _objectives_basic())
    diff_edges = [e for e in g["edges"] if e["relation_type"] == "chunk_at_difficulty"]
    assert diff_edges == []


def test_difficulty_level_nodes_emitted_unconditionally():
    """The 3 DifficultyLevel typed nodes must always exist (parity with
    the BloomLevel-node convention) — even on empty input."""
    g = build_pedagogy_graph([], {})
    diff_nodes = [n for n in g["nodes"] if n["class"] == "DifficultyLevel"]
    assert len(diff_nodes) == 3
    ids = {n["id"] for n in diff_nodes}
    assert ids == {
        "difficulty:foundational",
        "difficulty:intermediate",
        "difficulty:advanced",
    }


# ---------------------------------------------------------------------------
# 4. concept_supports_outcome: DomainConcept → Outcome rollup with weight.
# ---------------------------------------------------------------------------


def test_concept_supports_outcome_weight_is_count_of_supporting_chunks():
    """Concept 'triples' appears in 2 chunks both citing co-01 (which
    rolls up to to-01) → one concept_supports_outcome(triples, TO-01)
    edge with confidence == 2."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_1",
            "chunk_type": "explanation",
            "concept_tags": ["triples"],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p1.html"},
        },
        {
            "id": "ck_2",
            "chunk_type": "explanation",
            "concept_tags": ["triples"],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p2.html"},
        },
    ]
    g = build_pedagogy_graph(
        chunks,
        _objectives_basic(),
        concept_classes={"triples": "DomainConcept"},
    )
    supports = [
        e
        for e in g["edges"]
        if e["relation_type"] == "concept_supports_outcome"
        and e["source"] == "concept:triples"
        and e["target"] == "TO-01"
    ]
    assert len(supports) == 1
    assert supports[0]["confidence"] == 2


def test_concept_supports_outcome_filters_non_domain_concept_sources():
    """Pedagogical scaffolding ('key-takeaway') as concept_tag must NOT
    emit a supports edge even when present in a chunk citing co-01."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_1",
            "chunk_type": "explanation",
            "concept_tags": ["triples", "key-takeaway"],
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        }
    ]
    g = build_pedagogy_graph(
        chunks,
        _objectives_basic(),
        concept_classes={
            "triples": "DomainConcept",
            "key-takeaway": "PedagogicalMarker",
        },
    )
    supports = [
        e for e in g["edges"] if e["relation_type"] == "concept_supports_outcome"
    ]
    sources = {e["source"] for e in supports}
    assert "concept:triples" in sources
    assert "concept:key-takeaway" not in sources


def test_concept_supports_outcome_rolls_co_to_parent_to():
    """A chunk citing co-25 (parent_to == to-07) must produce a
    concept_supports_outcome edge to to-07, not co-25."""
    chunks: List[Dict[str, Any]] = [
        {
            "id": "ck_x",
            "chunk_type": "explanation",
            "concept_tags": ["shacl"],
            "learning_outcome_refs": ["co-25"],
            "source": {"module_id": "week_07", "item_path": "week_07/p.html"},
        }
    ]
    g = build_pedagogy_graph(
        chunks,
        _objectives_basic(),
        concept_classes={"shacl": "DomainConcept"},
    )
    supports = [
        e for e in g["edges"] if e["relation_type"] == "concept_supports_outcome"
    ]
    assert len(supports) == 1
    assert supports[0]["source"] == "concept:shacl"
    # Must point at the parent terminal, not at the CO directly.
    assert supports[0]["target"] == "TO-07"


# ---------------------------------------------------------------------------
# Regression on the real rdf-shacl-550 archive: relation-type count
# must land at 14 with all four new edge types present in non-trivial
# counts.
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
CONCEPT_GRAPH = (
    ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
    / "graph"
    / "concept_graph.json"
)


@pytest.mark.skipif(
    not (
        CORPUS_CHUNKS.exists()
        and SYNTH_OBJECTIVES.exists()
        and CONCEPT_GRAPH.exists()
    ),
    reason="rdf-shacl-550 archive missing — regression skipped",
)
def test_real_archive_has_14_distinct_relation_types_after_wave78():
    chunks = []
    with open(CORPUS_CHUNKS, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    with open(SYNTH_OBJECTIVES, encoding="utf-8") as f:
        objectives = json.load(f)
    with open(CONCEPT_GRAPH, encoding="utf-8") as f:
        cg = json.load(f)
    classes = {
        n["id"]: n.get("class")
        for n in cg.get("nodes", [])
        if isinstance(n, dict) and isinstance(n.get("id"), str)
    }

    g = build_pedagogy_graph(
        chunks,
        objectives,
        course_id="RDF_SHACL_550",
        concept_classes=classes,
    )
    er = g["stats"]["edges_by_relation"]
    # All 10 pre-Wave-78 relation types must still be present.
    pre_wave78 = {
        "teaches",
        "assesses",
        "practices",
        "exemplifies",
        "prerequisite_of",
        "interferes_with",
        "belongs_to_module",
        "supports_outcome",
        "at_bloom_level",
        "follows",
    }
    for rel in pre_wave78:
        assert er.get(rel, 0) > 0, f"missing pre-Wave-78 relation: {rel}"
    # All 4 new relation types must be present in non-trivial counts.
    assert er.get("derived_from_objective", 0) >= 400, er.get(
        "derived_from_objective", 0
    )
    assert er.get("concept_supports_outcome", 0) >= 100, er.get(
        "concept_supports_outcome", 0
    )
    assert er.get("assessment_validates_outcome", 0) >= 10, er.get(
        "assessment_validates_outcome", 0
    )
    # chunk_at_difficulty == chunk count exactly (one edge per chunk;
    # all 219 chunks in this archive have a canonical difficulty).
    assert er.get("chunk_at_difficulty", 0) == 219, er.get("chunk_at_difficulty", 0)
    # Total relation_type count must be 14.
    assert len(er) == 14, sorted(er.keys())
    # DifficultyLevel typed nodes must be present.
    nbc = g["stats"]["nodes_by_class"]
    assert nbc.get("DifficultyLevel", 0) == 3
