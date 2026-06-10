"""Tests for the duplicate-concept node-merge pass.

Covers Pass A (plural/alias fold, unconditional), Pass B (prefix fold, guarded
by shared occurrence-chunk OR direct edge), guard-rejection of false positives,
edge integrity (no dangling refs, self-loop drop, parallel-edge collapse), and
determinism (stable across runs and input reordering).
"""

from __future__ import annotations

import copy

from lib.ontology.concept_node_merge import merge_duplicate_concept_nodes


def _concept(node_id, frequency, *, label=None, occurrences=None, source_refs=None):
    node = {
        "id": node_id,
        "label": label if label is not None else node_id.replace("-", " ").title(),
        "frequency": frequency,
        "class": "DomainConcept",
    }
    if occurrences is not None:
        node["occurrences"] = list(occurrences)
    if source_refs is not None:
        node["source_refs"] = list(source_refs)
    return node


def _edge(source, target, *, type="related-to", confidence=0.5):
    return {"source": source, "target": target, "type": type, "confidence": confidence}


# ---------------------------------------------------------------------------
# Pass A — plural / alias fold (unconditional)
# ---------------------------------------------------------------------------


def test_plural_merge_knowledge_base():
    graph = {
        "nodes": [
            _concept("knowledge-base", 5, occurrences=["c_chunk_00001"]),
            _concept("knowledge-bases", 3, occurrences=["c_chunk_00002"]),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["knowledge-base"]
    winner = out["nodes"][0]
    assert winner["frequency"] == 8  # summed
    assert winner["occurrences"] == ["c_chunk_00001", "c_chunk_00002"]  # unioned + sorted
    assert out["concept_node_merges"] == 1


def test_plural_merge_agent():
    graph = {
        "nodes": [
            _concept("agent", 10),
            _concept("agents", 2),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["agent"]  # higher-frequency singular survives
    assert out["nodes"][0]["frequency"] == 12


def test_plural_winner_is_higher_frequency_even_if_plural():
    # The plural form has higher frequency, so it wins (winner = higher freq).
    graph = {
        "nodes": [
            _concept("agent", 1),
            _concept("agents", 9),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    assert [n["id"] for n in out["nodes"]] == ["agents"]
    assert out["nodes"][0]["frequency"] == 10


# ---------------------------------------------------------------------------
# Pass B — prefix fold (guarded)
# ---------------------------------------------------------------------------


def test_one_word_prefix_affix_not_merged_pydantic():
    # pydantic + pydantic-validator-guide: "pydantic" is a 1-WORD prefix of
    # the longer label. A single-token affix is too weak a merge signal
    # (Guard 1: >=2-word affix required), so the head node ``pydantic`` is NOT
    # demoted even though they share a chunk. Both survive.
    graph = {
        "nodes": [
            _concept("pydantic", 4, label="Pydantic", occurrences=["x_chunk_00010"]),
            _concept(
                "pydantic-validator-guide",
                2,
                label="Pydantic Validator Guide",
                occurrences=["x_chunk_00010", "x_chunk_00011"],
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["pydantic", "pydantic-validator-guide"]
    assert out["concept_node_merges"] == 0


def test_two_word_prefix_merge_shared_chunk():
    # A genuine >=2-word prefix dup STILL folds: "pydantic validator" is a
    # 2-word prefix of "pydantic validator guide"; share a chunk -> MERGE onto
    # the shorter head concept.
    graph = {
        "nodes": [
            _concept(
                "pydantic-validator", 4, label="Pydantic Validator",
                occurrences=["x_chunk_00010"],
            ),
            _concept(
                "pydantic-validator-guide",
                2,
                label="Pydantic Validator Guide",
                occurrences=["x_chunk_00010", "x_chunk_00011"],
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["pydantic-validator"]  # winner = shorter (head) label
    assert out["nodes"][0]["frequency"] == 6
    assert out["nodes"][0]["occurrences"] == ["x_chunk_00010", "x_chunk_00011"]
    assert out["concept_node_merges"] == 1


def test_prefix_merge_shared_edge_running_state():
    graph = {
        "nodes": [
            _concept("running-state", 6, label="Running State"),
            _concept("running-state-chain", 3, label="Running State Chain"),
        ],
        "edges": [
            _edge("running-state", "running-state-chain", type="related-to", confidence=0.7),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["running-state"]  # winner = head concept
    assert out["nodes"][0]["frequency"] == 9
    # The related-to edge between the two merged nodes becomes a self-loop -> dropped.
    assert out["edges"] == []


# ---------------------------------------------------------------------------
# Guard rejects false positives
# ---------------------------------------------------------------------------


def test_prefix_guard_rejects_unrelated_llm_call():
    # llm + llm-call: prefix relation but NO shared chunk and NO edge -> NOT merged.
    graph = {
        "nodes": [
            _concept("llm", 8, label="LLM", occurrences=["a_chunk_00001"]),
            _concept("llm-call", 4, label="LLM Call", occurrences=["a_chunk_00099"]),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["llm", "llm-call"]
    assert out["concept_node_merges"] == 0


def test_no_prefix_relation_embedding_vs_foundation_models():
    # embedding-models vs foundation-models: share a head NOUN but neither
    # label is a prefix or suffix of the other (equal length, differing first
    # word) -> NOT merged (affix-only, no interior/partial folding).
    graph = {
        "nodes": [
            _concept("embedding-models", 5, label="Embedding Models", occurrences=["m_chunk_1"]),
            _concept("foundation-models", 5, label="Foundation Models", occurrences=["m_chunk_1"]),
        ],
        "edges": [
            _edge("embedding-models", "foundation-models", confidence=0.9),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["embedding-models", "foundation-models"]
    assert out["concept_node_merges"] == 0


def test_suffix_merge_shared_chunk_faiss_vector_store():
    # faiss-vector-store + vector-store: "vector store" is a SUFFIX of "faiss
    # vector store"; share a chunk -> MERGE to the shorter head concept.
    graph = {
        "nodes": [
            _concept(
                "faiss-vector-store",
                2,
                label="Faiss Vector Store",
                occurrences=["f_chunk_1"],
            ),
            _concept(
                "vector-store", 5, label="Vector Store", occurrences=["f_chunk_1"]
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["vector-store"]  # winner = shorter (head) label
    assert out["nodes"][0]["frequency"] == 7
    assert out["concept_node_merges"] == 1


def test_one_word_suffix_affix_not_merged_langchain_family():
    # langchain + refine-langchain + retrieval-langchain: "langchain" is a
    # 1-WORD suffix of both. A single-token affix is too weak a merge signal
    # (Guard 1: >=2-word affix required), so NOTHING folds even though edges +
    # a shared chunk corroborate co-location. langchain survives intact.
    graph = {
        "nodes": [
            _concept("langchain", 7, label="LangChain", occurrences=["s_chunk_1"]),
            _concept(
                "refine-langchain", 3, label="Refine LangChain", occurrences=["s_chunk_1"]
            ),
            _concept(
                "retrieval-langchain", 2, label="Retrieval LangChain"
            ),
        ],
        "edges": [
            _edge("langchain", "refine-langchain", confidence=0.8),
            _edge("langchain", "retrieval-langchain", confidence=0.6),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["langchain", "refine-langchain", "retrieval-langchain"]
    assert out["concept_node_merges"] == 0


def test_suffix_guard_rejects_unrelated_pair():
    # Suffix relation but NO shared chunk and NO edge -> NOT merged.
    graph = {
        "nodes": [
            _concept("langchain", 7, label="LangChain", occurrences=["s_chunk_1"]),
            _concept(
                "refine-langchain", 3, label="Refine LangChain", occurrences=["s_chunk_99"]
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["langchain", "refine-langchain"]
    assert out["concept_node_merges"] == 0


def test_interior_substring_not_merged():
    # "embedding" appears in the INTERIOR of "deep embedding models", neither
    # prefix nor suffix -> NOT merged even with shared chunk + edge.
    graph = {
        "nodes": [
            _concept("embedding", 6, label="Embedding", occurrences=["i_chunk_1"]),
            _concept(
                "deep-embedding-models",
                2,
                label="Deep Embedding Models",
                occurrences=["i_chunk_1"],
            ),
        ],
        "edges": [
            _edge("embedding", "deep-embedding-models", confidence=0.9),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["deep-embedding-models", "embedding"]
    assert out["concept_node_merges"] == 0


# ---------------------------------------------------------------------------
# Over-merge regression — flagship-concept protection (Pass B guards)
# ---------------------------------------------------------------------------


def test_high_frequency_langchain_survives_variant_cascade():
    # langchain (freq 62) with *-langchain variants sharing chunks: the only
    # affix shared between langchain and a variant is the 1-word "langchain"
    # (Guard 1 rejects); even if it weren't, langchain is a tech-anchor + a
    # high-frequency single-token head (Guards 2 & 3). langchain is NEVER the
    # loser.
    graph = {
        "nodes": [
            _concept("langchain", 62, label="LangChain", occurrences=["s_chunk_1"]),
            _concept("stuff-langchain", 3, label="Stuff LangChain", occurrences=["s_chunk_1"]),
            _concept("refine-langchain", 2, label="Refine LangChain", occurrences=["s_chunk_1"]),
        ],
        "edges": [
            _edge("langchain", "stuff-langchain", confidence=0.8),
            _edge("langchain", "refine-langchain", confidence=0.7),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = {n["id"] for n in out["nodes"]}
    assert "langchain" in ids  # flagship survives
    lc = next(n for n in out["nodes"] if n["id"] == "langchain")
    assert lc["frequency"] == 62  # not demoted / not rolled into anything


def test_faiss_tech_anchor_survives_with_vector_store():
    # faiss (freq 16, tech-anchor) + faiss-vector-store + vector-store. The
    # genuine 2-word affix dup (vector-store ⊂ faiss-vector-store) folds, but
    # faiss stays its own node: faiss⊂faiss-vector-store is only a 1-word affix
    # (Guard 1), and faiss is a tech-anchor (Guard 2).
    graph = {
        "nodes": [
            _concept("faiss", 16, label="FAISS", occurrences=["f_chunk_1"]),
            _concept(
                "faiss-vector-store", 2, label="Faiss Vector Store",
                occurrences=["f_chunk_1"],
            ),
            _concept("vector-store", 5, label="Vector Store", occurrences=["f_chunk_1"]),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert "faiss" in ids  # tech-anchor survives as its own node
    assert "vector-store" in ids  # head of the genuine 2-word dup
    assert "faiss-vector-store" not in ids  # folded into vector-store
    faiss = next(n for n in out["nodes"] if n["id"] == "faiss")
    assert faiss["frequency"] == 16  # untouched


def test_agentic_rag_not_folded_into_rag():
    # agentic-rag (tech-anchor) shares the 1-word suffix "rag" with rag.
    # Guard 1 (1-word affix) + Guard 2 (agentic-rag is a tech-anchor) keep it
    # distinct.
    graph = {
        "nodes": [
            _concept("rag", 30, label="RAG", occurrences=["r_chunk_1"]),
            _concept("agentic-rag", 8, label="Agentic RAG", occurrences=["r_chunk_1"]),
        ],
        "edges": [
            _edge("rag", "agentic-rag", confidence=0.9),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["agentic-rag", "rag"]
    assert out["concept_node_merges"] == 0


def test_genuine_two_word_affix_dup_still_merges():
    # running-state + running-state-chain share the 2-word affix "running
    # state"; neither is a tech-anchor -> genuine dup STILL folds (no
    # regression on real merges).
    graph = {
        "nodes": [
            _concept("running-state", 6, label="Running State", occurrences=["q_chunk_1"]),
            _concept(
                "running-state-chain", 3, label="Running State Chain",
                occurrences=["q_chunk_1"],
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["running-state"]
    assert out["nodes"][0]["frequency"] == 9
    assert out["concept_node_merges"] == 1


def test_one_word_affix_pair_no_other_signal_not_merged():
    # A 1-word affix pair with NO other signal does not merge. (Belt-and-
    # suspenders: even with a shared chunk, the 1-word affix is rejected.)
    graph = {
        "nodes": [
            _concept("retriever", 9, label="Retriever", occurrences=["z_chunk_1"]),
            _concept(
                "hybrid-retriever", 3, label="Hybrid Retriever", occurrences=["z_chunk_1"]
            ),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["hybrid-retriever", "retriever"]
    assert out["concept_node_merges"] == 0


# ---------------------------------------------------------------------------
# Edge integrity
# ---------------------------------------------------------------------------


def test_chunk_nodes_never_merged():
    # Two chunk nodes whose ids would singularize identically must NOT merge.
    graph = {
        "nodes": [
            {"id": "rdf_chunk_00001", "label": "c1", "frequency": 1, "class": "DomainConcept"},
            {"id": "rdf_chunk_00002", "label": "c2", "frequency": 1, "class": "DomainConcept"},
            _concept("agent", 3),
            _concept("agents", 2),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert "rdf_chunk_00001" in ids and "rdf_chunk_00002" in ids
    assert "agent" in ids and "agents" not in ids
    assert out["concept_node_merges"] == 1


def test_edge_integrity_no_dangling_refs_and_parallel_collapse():
    graph = {
        "nodes": [
            _concept("agent", 5, occurrences=["g_chunk_1"]),
            _concept("agents", 2, occurrences=["g_chunk_2"]),
            _concept("planner", 4),
            _concept("memory", 3),
        ],
        "edges": [
            # references loser "agents" -> must re-point to winner "agent"
            _edge("agents", "planner", type="related-to", confidence=0.4),
            _edge("agent", "planner", type="related-to", confidence=0.6),
            # parallel edge after re-point: agent->planner related-to twice;
            # keep the max-confidence one (0.6).
            _edge("memory", "agents", type="depends-on", confidence=0.5),
        ],
        "edges_extra": None,
    }
    out = merge_duplicate_concept_nodes(graph)
    valid = {n["id"] for n in out["nodes"]}
    # No edge references a dropped id.
    for e in out["edges"]:
        assert e["source"] in valid and e["target"] in valid
        assert "agents" not in (e["source"], e["target"])
    # Parallel agent->planner related-to collapsed to the 0.6 edge.
    ap = [e for e in out["edges"] if e["source"] == "agent" and e["target"] == "planner" and e["type"] == "related-to"]
    assert len(ap) == 1
    assert ap[0]["confidence"] == 0.6
    # The depends-on edge re-pointed memory->agent.
    md = [e for e in out["edges"] if e["type"] == "depends-on"]
    assert len(md) == 1
    assert md[0]["source"] == "memory" and md[0]["target"] == "agent"


def test_self_loop_dropped_after_merge():
    graph = {
        "nodes": [
            _concept("knowledge-base", 4, occurrences=["k_chunk_1"]),
            _concept("knowledge-bases", 2, occurrences=["k_chunk_2"]),
        ],
        "edges": [
            _edge("knowledge-base", "knowledge-bases", confidence=0.9),
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    assert [n["id"] for n in out["nodes"]] == ["knowledge-base"]
    assert out["edges"] == []  # self-loop dropped


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _sample_graph():
    return {
        "nodes": [
            _concept("agent", 5, occurrences=["d_chunk_1"]),
            _concept("agents", 3, occurrences=["d_chunk_2"]),
            _concept("knowledge-base", 7, occurrences=["d_chunk_3"]),
            _concept("knowledge-bases", 1, occurrences=["d_chunk_4"]),
            _concept("running-state", 4, label="Running State"),
            _concept("running-state-chain", 2, label="Running State Chain"),
            _concept("planner", 6),
        ],
        "edges": [
            _edge("running-state", "running-state-chain", confidence=0.7),
            _edge("agent", "planner", confidence=0.5),
            _edge("agents", "planner", confidence=0.8),
        ],
    }


def test_determinism_two_runs_identical():
    g1 = _sample_graph()
    g2 = copy.deepcopy(g1)
    out1 = merge_duplicate_concept_nodes(g1)
    out2 = merge_duplicate_concept_nodes(g2)
    assert out1 == out2


def test_input_not_mutated():
    g = _sample_graph()
    snapshot = copy.deepcopy(g)
    merge_duplicate_concept_nodes(g)
    assert g == snapshot


def test_winner_stable_under_input_reordering():
    g = _sample_graph()
    out_a = merge_duplicate_concept_nodes(g)

    g_rev = copy.deepcopy(g)
    g_rev["nodes"] = list(reversed(g_rev["nodes"]))
    g_rev["edges"] = list(reversed(g_rev["edges"]))
    out_b = merge_duplicate_concept_nodes(g_rev)

    surviving_a = sorted(n["id"] for n in out_a["nodes"])
    surviving_b = sorted(n["id"] for n in out_b["nodes"])
    assert surviving_a == surviving_b
    assert out_a["concept_node_merges"] == out_b["concept_node_merges"]
    # Frequency roll-up is identical regardless of input order.
    freq_a = {n["id"]: n["frequency"] for n in out_a["nodes"]}
    freq_b = {n["id"]: n["frequency"] for n in out_b["nodes"]}
    assert freq_a == freq_b


# ---------------------------------------------------------------------------
# External-endpoint pass-through (LO targets-concept materializer interplay)
# ---------------------------------------------------------------------------


def test_targets_concept_edge_to_non_node_target_survives_merge():
    """An edge whose target was NEVER an input node (a federation-by-convention
    endpoint such as an LO-authored targets-concept phantom) passes through
    verbatim with external_endpoint_edges_preserved == 1 — it is not dropped as
    a dangling edge."""
    graph = {
        "nodes": [
            _concept("framework", 5, occurrences=["c_chunk_00001"]),
        ],
        "edges": [
            # target "phantom-concept" is not a node and was never an input
            # node; source is the LO ID (also never a node).
            {
                "source": "to-01",
                "target": "phantom-concept",
                "type": "targets-concept",
                "confidence": 1.0,
            },
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    surviving = [
        e
        for e in out["edges"]
        if e["source"] == "to-01" and e["target"] == "phantom-concept"
    ]
    assert len(surviving) == 1
    assert surviving[0]["type"] == "targets-concept"
    assert surviving[0]["confidence"] == 1.0
    assert out["external_endpoint_edges_preserved"] == 1


def test_materialized_plural_folds_and_edge_repoints_no_drop():
    """A materialized DomainConcept "vector-stores" (freq 0) + a chunk-derived
    "vector-store" (freq > 0) Pass-A fold onto the singular winner; the
    targets-concept edge onto "vector-stores" re-points to "vector-store"
    rather than being dropped."""
    graph = {
        "nodes": [
            {
                "id": "vector-stores",
                "label": "vector-stores",
                "frequency": 0,
                "class": "DomainConcept",
                "node_provenance": "lo_key_concept",
            },
            _concept("vector-store", 5, occurrences=["c_chunk_00001"]),
        ],
        "edges": [
            {
                "source": "to-01",
                "target": "vector-stores",
                "type": "targets-concept",
                "confidence": 1.0,
            },
        ],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    assert ids == ["vector-store"]  # plural folded into singular winner
    assert out["concept_node_merges"] == 1
    repointed = [e for e in out["edges"] if e["type"] == "targets-concept"]
    assert len(repointed) == 1
    assert repointed[0]["source"] == "to-01"
    assert repointed[0]["target"] == "vector-store"  # re-pointed onto winner
    # The edge survives (not dropped). Its source "to-01" was never an input
    # node (an LO ID), so the edge is counted as an external pass-through; the
    # target's fold-and-repoint is orthogonal to that count.
    assert out["external_endpoint_edges_preserved"] == 1


def test_edge_to_folded_loser_still_repoints_regression():
    """Regression: an edge whose endpoint was an input node that got folded into
    a winner still re-points to the winner (the input_ids guard does not break
    existing fold-and-repoint semantics)."""
    graph = {
        "nodes": [
            _concept("agent", 10),
            _concept("agents", 2),
        ],
        "edges": [
            # edge onto the loser "agents" must re-point to winner "agent".
            _edge("agents", "framework", confidence=0.6),
            _concept and _edge("framework", "agent", confidence=0.4),
        ],
    }
    # Add the framework node so its endpoints resolve.
    graph["nodes"].append(_concept("framework", 4))
    out = merge_duplicate_concept_nodes(graph)
    surviving_ids = sorted(n["id"] for n in out["nodes"])
    assert "agents" not in surviving_ids
    assert "agent" in surviving_ids
    # The (agents, framework) edge re-points to (agent, framework); the
    # (framework, agent) edge stays. Both endpoints resolve → no drop.
    edge_pairs = {(e["source"], e["target"]) for e in out["edges"]}
    assert ("agent", "framework") in edge_pairs or ("framework", "agent") in edge_pairs
    # No edge dangles onto the folded loser.
    assert all(e["source"] != "agents" and e["target"] != "agents" for e in out["edges"])
    # Both endpoints were input nodes (agent/agents folded, framework survived),
    # so nothing is counted as an external pass-through.
    assert out["external_endpoint_edges_preserved"] == 0
