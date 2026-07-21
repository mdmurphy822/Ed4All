"""Fix-2 — `_run_concept_extraction` emits the genuine semantic graph.

Locks the contract that ``MCP/tools/pipeline_tools.py::_run_concept_extraction``
now emits the genuine ``kind: "concept_semantic"`` graph (``DomainConcept``
nodes + typed edges) via ``build_semantic_graph``, replacing the prior
lightweight ``kind: "pedagogy"`` graph from ``build_pedagogy_graph``.

The phase consumes the upstream SemantiK chunkset; this test feeds a
tagged SemantiK chunkset fixture whose chunks carry hand-set
``concept_tags`` — the test does NOT depend on the chunk-tagging path
(a parallel worker owns that).

Contract assertions:

1. **`kind: "concept_semantic"`.** The emitted file is the genuine
   semantic graph, not the pedagogy graph.
2. **`DomainConcept` nodes.** ≥1 node carries a domain-concept class
   so ``concept_objective_linker`` has something to link.
3. **Node-count floor.** ≥10 nodes — the ``ConceptGraphValidator``
   ``min_nodes`` default floor.
4. **Typed edges carry `type`.** Semantic-graph edges expose ``type``.
5. **sha256 chain integrity.** ``concept_graph_sha256`` re-hashes equal
   to the on-disk file bytes.
6. **Empty-input shell.** A chunkset with empty ``concept_tags`` still
   emits a ``kind: "concept_semantic"`` shell (not pedagogy).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Tagged SemantiK chunkset fixture — hand-set concept_tags, no tagging path.
# ---------------------------------------------------------------------------

_CONCEPTS = [
    "vector-space", "linear-map", "eigenvalue", "eigenvector",
    "matrix-rank", "determinant", "basis-set", "orthogonality",
    "inner-product", "subspace", "kernel-space", "image-space",
]


def _tagged_chunkset() -> List[Dict[str, Any]]:
    """A SemantiK chunkset whose chunks carry populated ``concept_tags``.

    Each concept appears in ≥2 chunks so it survives the default
    ``min_freq=2`` co-occurrence floor and clears the validator's
    ``min_nodes`` floor.
    """
    chunks: List[Dict[str, Any]] = []
    # Emit 24 chunks: each concept tagged on 4 consecutive chunks via a
    # sliding window, so every concept clears min_freq and co-occurs.
    for i in range(24):
        window = [
            _CONCEPTS[(i + j) % len(_CONCEPTS)] for j in range(3)
        ]
        chunks.append({
            "id": f"linalg_chunk_{i:05d}",
            "text": (
                f"This passage explains {window[0]} in relation to "
                f"{window[1]} and {window[2]} for a linear-algebra course."
            ),
            "chunk_type": "explanation",
            "concept_tags": window,
            "learning_outcome_refs": [],
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": f"week_{(i // 4) + 1:02d}",
                "item_path": f"week_{(i // 4) + 1:02d}/page_{i:03d}.html",
            },
        })
    return chunks


def _empty_tag_chunkset() -> List[Dict[str, Any]]:
    """A SemantiK chunkset with empty ``concept_tags`` — the pre-Fix-2
    chunk shape. Exercises the empty-graph shell path.
    """
    return [
        {
            "id": f"empty_chunk_{i:05d}",
            "text": f"section {i}",
            "chunk_type": "content",
            "concept_tags": [],
            "learning_outcome_refs": [],
            "source": {"module_id": "week_01", "item_path": f"w1/p{i}.html"},
        }
        for i in range(3)
    ]


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Invoke ``_run_concept_extraction`` against a tagged chunkset
    fixture and return the parsed result payload + the on-disk graph.
    """
    chunks_path = tmp_path / "semantik_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    custom_libv2 = tmp_path / "libv2"

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LINALG_FIX2",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
        )
    )
    payload = json.loads(result)
    graph_path = Path(payload["concept_graph_path"])
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["_graph"] = graph
    payload["_graph_path"] = graph_path
    return payload


def test_emits_concept_semantic_kind(tmp_path: Path) -> None:
    """The emitted graph file is ``kind: "concept_semantic"`` — the
    genuine semantic graph, not the pedagogy graph.
    """
    payload = _run(tmp_path, _tagged_chunkset())
    assert payload.get("success") is True, f"payload={payload!r}"
    assert payload["_graph"].get("kind") == "concept_semantic", (
        "concept_extraction must emit the genuine semantic graph kind, "
        f"got kind={payload['_graph'].get('kind')!r}."
    )


def test_emits_domain_concept_nodes(tmp_path: Path) -> None:
    """≥1 node carries a domain-concept class so the
    ``concept_objective_linker`` has nodes to link into
    ``LearningObjective.keyConcepts[]``.
    """
    payload = _run(tmp_path, _tagged_chunkset())
    nodes = payload["_graph"].get("nodes") or []
    concept_classes = {"Concept", "DomainConcept", "concept", "domain_concept"}
    domain_nodes = [
        n for n in nodes if n.get("class") in concept_classes
    ]
    assert domain_nodes, (
        "Expected ≥1 DomainConcept-class node from a tagged SemantiK "
        f"chunkset; node classes seen: {sorted({n.get('class') for n in nodes})}."
    )


def test_clears_validator_node_floor(tmp_path: Path) -> None:
    """≥10 nodes — the ``ConceptGraphValidator`` ``min_nodes`` floor."""
    payload = _run(tmp_path, _tagged_chunkset())
    nodes = payload["_graph"].get("nodes") or []
    assert len(nodes) >= 10, (
        f"Semantic graph should clear the min_nodes=10 floor; got {len(nodes)}."
    )
    assert payload["node_count"] == len(nodes)


# Canonical node-class vocabulary — single source of truth is
# lib/validators/libv2/_packet_integrity_severity.py::EDGE_CLASS_SYNONYMS
# (structural / pedagogical targets) + lib/ontology/concept_classifier.py
# (concept sub-classes). Mirrored by the additive ``class`` enum in
# schemas/knowledge/concept_graph_semantic.schema.json.
_VALID_NODE_CLASSES = {
    "DomainConcept",
    "AssessmentOption",
    "InstructionalArtifact",
    "LowSignal",
    "Chunk",
    "Outcome",
    "ComponentObjective",
    "Misconception",
    "Module",
    "BloomLevel",
}


def test_every_node_carries_valid_class(tmp_path: Path) -> None:
    """Every emitted semantic-graph node carries a non-empty ``class``
    drawn from the canonical vocabulary.

    Closes the KG-quality "classless node" gap: the coverage denominator
    (``lib/validators/kg_quality.py``), completeness, and
    ``packet_integrity``'s node-class checks all read ``node['class']``.
    Concept nodes are stamped by ``lib/ontology/cooccurrence_graph.py``
    via ``classify_concept``; chunk / LO / misconception endpoint nodes
    are stamped by ``Trainforge/rag/typed_edge_inference.py::
    _materialize_endpoint_nodes`` at authoring time.
    """
    payload = _run(tmp_path, _tagged_chunkset())
    nodes = payload["_graph"].get("nodes") or []
    assert nodes, "Expected ≥1 node from a tagged chunkset."
    classless = [n.get("id") for n in nodes if not n.get("class")]
    assert not classless, (
        "Every emitted node must carry a non-empty 'class' at authoring "
        f"time; classless node ids: {classless}."
    )
    bad = {
        n.get("id"): n.get("class")
        for n in nodes
        if n.get("class") not in _VALID_NODE_CLASSES
    }
    assert not bad, (
        "Every node 'class' must be in the canonical vocabulary "
        f"{sorted(_VALID_NODE_CLASSES)}; out-of-vocab: {bad}."
    )


def test_endpoint_nodes_carry_structural_class(tmp_path: Path) -> None:
    """Chunk-endpoint nodes materialized from chunk-anchored edges carry
    ``class == 'Chunk'`` (NOT a concept class), so the kg_quality
    coverage denominator excludes them and ``packet_integrity``'s
    edge-endpoint typing contract sees a real class.
    """
    payload = _run(tmp_path, _tagged_chunkset())
    nodes = payload["_graph"].get("nodes") or []
    chunk_nodes = [n for n in nodes if "chunk_" in (n.get("id") or "")]
    assert chunk_nodes, (
        "Expected ≥1 materialized chunk-endpoint node (chunk-anchored "
        "edges reference chunk IDs)."
    )
    for n in chunk_nodes:
        assert n.get("class") == "Chunk", (
            f"Chunk-endpoint node {n.get('id')!r} must carry "
            f"class='Chunk', got {n.get('class')!r}."
        )


def test_typed_edges_carry_type(tmp_path: Path) -> None:
    """Semantic-graph edges expose a ``type`` field (the typed-edge
    discriminator the ``ConceptGraphValidator`` reads).
    """
    payload = _run(tmp_path, _tagged_chunkset())
    edges = payload["_graph"].get("edges") or []
    assert edges, "Expected ≥1 typed edge from a tagged chunkset."
    for edge in edges:
        assert "type" in edge, (
            f"Semantic-graph edge missing 'type': {edge!r}."
        )


def test_sha256_chain_integrity(tmp_path: Path) -> None:
    """``concept_graph_sha256`` re-hashes equal to the on-disk file
    bytes — the manifest-validator chain stays coherent.
    """
    payload = _run(tmp_path, _tagged_chunkset())
    on_disk = payload["_graph_path"].read_bytes()
    recomputed = hashlib.sha256(on_disk).hexdigest()
    assert payload["concept_graph_sha256"] == recomputed, (
        "Returned concept_graph_sha256 does not match the on-disk bytes."
    )
    # Sibling manifest also carries the same hash.
    manifest = json.loads(
        Path(payload["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["concept_graph_sha256"] == recomputed


def test_empty_tag_chunkset_emits_semantic_shell(tmp_path: Path) -> None:
    """A SemantiK chunkset with empty ``concept_tags`` still emits a
    ``kind: "concept_semantic"`` graph (an empty shell), NOT a
    pedagogy graph — the empty-shell ``kind`` was updated by Fix-2.
    """
    payload = _run(tmp_path, _empty_tag_chunkset())
    assert payload.get("success") is True
    assert payload["_graph"].get("kind") == "concept_semantic", (
        "Empty-input shell must still be kind='concept_semantic'."
    )
