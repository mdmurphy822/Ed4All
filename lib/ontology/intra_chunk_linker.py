"""Intra-chunk concept-linking pass for the semantic concept graph.

The co-occurrence ``related-to`` inference rule emits a ``related-to`` edge only
when two concepts co-occur across at least three chunks (weight >= 3). A topic
confined to a single notebook section — e.g. a state-machine vocabulary
(``State Graph``, ``Conditional Routing``, ``Terminal States``,
``Node Implementation``, ...) — never reaches that floor,
so its concepts are each anchored to a SINGLE chunk by one ``defined-by`` edge
and carry ZERO ``related-to`` edges between them. The topic then shatters into
many one-node connected components.

This pass closes that gap. For every chunk node it collects the
``DomainConcept`` nodes co-located in that chunk (concepts with a
``defined-by``/``exemplifies`` edge to the chunk) and adds a low-confidence
``related-to`` edge between every co-located pair that does not already have a
``related-to`` edge in either direction. The new edges are explicitly stamped
(``confidence=0.35``, ``provenance.origin="intra_chunk"``) so they read as
*weaker* than real cross-chunk co-occurrence ties and remain distinguishable to
downstream consumers (the fan-out cap, KG-quality, edge-consensus).

Design contract (mirrors ``concept_node_merge`` / ``related_edge_cap``):

* Pure ``dict -> dict``. The input graph is never mutated; a new graph dict is
  returned with the same nodes and a superset of edges.
* Reads no environment variables — the opt-in flag gate lives at the call site.
* Deterministic: pairs are emitted in sorted ``(source, target)`` order so two
  runs over the same input produce byte-identical output.
* Connectivity guarantee: EVERY co-located concept gains at least one
  ``related-to`` edge so it is never left a degree-1 singleton. When a chunk is
  sparse enough that its full pairwise clique fits under ``max_pairs_per_chunk``
  the clique is emitted; when it is dense the pass falls back to a *star* —
  every concept is linked to the single highest-frequency hub concept in the
  chunk (N-1 edges). The earlier "top-k clique" guard silently DROPPED the
  lowest-ranked concepts past the cap, re-stranding the very singletons this
  pass exists to fix; the star keeps every concept connected with a bounded
  edge count.
* Never creates self-loops, never duplicates an existing ``related-to`` edge,
  never touches non-concept nodes or non-``related-to`` edges.

Stamps ``graph["intra_chunk_links_added"]`` with the number of edges added.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

__all__ = ["RULE_NAME", "RULE_VERSION", "link_intra_chunk_concepts"]

# Rule identity for the edges this pass emits. Mirrors the per-module
# ``RULE_NAME`` / ``RULE_VERSION`` constant convention used by the
# ``Trainforge.rag.inference_rules`` rule modules. The orchestrator stamps
# ``provenance = {"rule": RULE_NAME, "rule_version": RULE_VERSION}`` on each
# rule-emitted edge; this pass does the same so its edges carry an integer
# ``rule_version >= 1`` and validate against the ``FallbackProvenance`` arm of
# ``schemas/knowledge/concept_graph_semantic.schema.json`` (which requires
# ``[rule, rule_version]`` with ``rule_version`` an integer >= 1). The version
# is also reflected into the graph's top-level ``rule_versions`` map so
# ``kg_quality_report``'s ``rule_outputs`` surfaces the integer rather than a
# null. Bump ``RULE_VERSION`` whenever the emitted edge shape / topology
# changes.
RULE_NAME = "intra_chunk_link"
RULE_VERSION = 1

# Edge types that anchor a concept to the chunk that defines/exemplifies it.
_CHUNK_ANCHOR_TYPES = frozenset({"defined-by", "exemplifies"})
_RELATED_TYPE = "related-to"

# A chunk node carries a ``chunk_NNNNN`` token in its id (full corpus form
# ``demo_course_chunk_00270`` or the short ``chunk_00270``).
_CHUNK_ID_RE = re.compile(r"chunk_\d+")


def _edge_type(edge: Dict[str, Any]) -> Any:
    """Edge type, honoring the ``relation_type`` legacy alias."""
    return edge.get("type", edge.get("relation_type"))


def _is_chunk_id(node_id: Any) -> bool:
    return isinstance(node_id, str) and bool(_CHUNK_ID_RE.search(node_id))


def _is_concept_node(node: Dict[str, Any]) -> bool:
    """A ``DomainConcept`` node that is not itself a chunk node."""
    if node.get("class") != "DomainConcept":
        return False
    if _is_chunk_id(node.get("id")):
        return False
    return True


def _frequency(node: Dict[str, Any]) -> int:
    freq = node.get("frequency", 0)
    return freq if isinstance(freq, int) else 0


def link_intra_chunk_concepts(
    graph: Dict[str, Any],
    *,
    max_pairs_per_chunk: int = 20,
    confidence: float = 0.35,
) -> Dict[str, Any]:
    """Add low-confidence ``related-to`` edges between co-located concepts.

    Args:
        graph: A concept/semantic graph dict with ``nodes`` and ``edges``
            lists. Concept nodes carry ``id``, ``class == "DomainConcept"`` and
            an optional ``frequency``. Chunk nodes carry a ``chunk_NNNNN`` id.
            Edges carry ``source`` / ``target`` and ``type`` (or the legacy
            ``relation_type`` alias).
        max_pairs_per_chunk: Per-chunk ceiling on the number of pairs linked.
            When a chunk's full pairwise clique would exceed this ceiling the
            pass falls back to a *star* topology — every co-located concept is
            linked to the single highest-frequency hub concept (N-1 edges) so
            no concept is dropped or left a singleton. Keep conservative — the
            downstream fan-out cap prunes further.
        confidence: Confidence stamped on every added edge. Defaults to a
            deliberately low ``0.35`` so the intra-chunk ties rank below real
            cross-chunk co-occurrence ``related-to`` edges.

    Returns:
        A new graph dict with the same nodes, the original edges, and the added
        intra-chunk ``related-to`` edges appended (each stamped
        ``confidence=<confidence>``, ``weight=1``, ``provenance={"rule":
        "intra_chunk_link", "rule_version": 1, "origin": "intra_chunk"}``).
        ``graph["intra_chunk_links_added"]`` is stamped with the count of added
        edges, and ``graph["rule_versions"]["intra_chunk_link"]`` is set to
        ``RULE_VERSION`` (only when edges were added) so the top-level map stays
        in sync with the per-edge provenance and ``kg_quality_report``'s
        ``rule_outputs`` surfaces an integer rather than a null. The input is
        not mutated.
    """
    nodes_in = graph.get("nodes")
    if not isinstance(nodes_in, list):
        nodes_in = []
    edges_in = graph.get("edges")
    if not isinstance(edges_in, list):
        edges_in = []

    # Index concept nodes and their frequency for the dense-chunk top-N guard.
    concept_freq: Dict[str, int] = {}
    for node in nodes_in:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        if _is_concept_node(node) and nid not in concept_freq:
            concept_freq[nid] = _frequency(node)

    # Existing undirected ``related-to`` adjacency, so we never duplicate an
    # edge that exists in EITHER direction.
    existing_related: set = set()
    for edge in edges_in:
        if not isinstance(edge, dict):
            continue
        if _edge_type(edge) != _RELATED_TYPE:
            continue
        s, t = edge.get("source"), edge.get("target")
        if isinstance(s, str) and isinstance(t, str):
            existing_related.add((s, t))
            existing_related.add((t, s))

    # Group co-located concepts per chunk: a concept is co-located in a chunk
    # when a ``defined-by``/``exemplifies`` edge connects the two (either
    # orientation — concept may be source or target of the anchor edge).
    chunk_to_concepts: Dict[str, set] = {}
    for edge in edges_in:
        if not isinstance(edge, dict):
            continue
        if _edge_type(edge) not in _CHUNK_ANCHOR_TYPES:
            continue
        s, t = edge.get("source"), edge.get("target")
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        # Identify which endpoint is the chunk and which is the concept.
        if _is_chunk_id(s) and t in concept_freq:
            chunk_id, concept_id = s, t
        elif _is_chunk_id(t) and s in concept_freq:
            chunk_id, concept_id = t, s
        else:
            continue
        chunk_to_concepts.setdefault(chunk_id, set()).add(concept_id)

    # Build the new edges. Iterate chunks in deterministic id order; within a
    # chunk, order concepts by (-frequency, id) so the highest-frequency member
    # is the deterministic star hub when the clique would overflow the cap.
    new_edges: List[Dict[str, Any]] = []
    added_pairs: set = set()  # de-dupes across chunks too
    cap = max_pairs_per_chunk if isinstance(max_pairs_per_chunk, int) and max_pairs_per_chunk > 0 else 0

    def _emit(a: str, b: str) -> None:
        """Append a deduped, sorted-orientation intra-chunk related-to edge."""
        if a == b:
            return
        key = (a, b) if a <= b else (b, a)
        if key in added_pairs or key in existing_related:
            return
        added_pairs.add(key)
        added_pairs.add((key[1], key[0]))
        new_edges.append({
            "source": key[0],
            "target": key[1],
            "type": _RELATED_TYPE,
            "confidence": confidence,
            "weight": 1,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "origin": "intra_chunk",
            },
        })

    for chunk_id in sorted(chunk_to_concepts):
        concepts = chunk_to_concepts[chunk_id]
        if len(concepts) < 2:
            continue
        # Rank by frequency desc, id asc — deterministic ordering. The first
        # element is the star hub when the full clique would overflow the cap.
        ranked = sorted(concepts, key=lambda c: (-concept_freq.get(c, 0), c))
        full_pairs = len(ranked) * (len(ranked) - 1) // 2

        if not cap or full_pairs <= cap:
            # Sparse chunk: emit the full pairwise clique.
            ordered = sorted(ranked)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    _emit(a, b)
        else:
            # Dense chunk: fall back to a star so EVERY concept gains an edge
            # (N-1 edges) rather than dropping the lowest-ranked concepts and
            # re-stranding them as singletons. Hub = highest-frequency concept.
            hub = ranked[0]
            for spoke in sorted(ranked[1:]):
                _emit(hub, spoke)

    result = dict(graph)
    result["nodes"] = list(nodes_in)
    result["edges"] = list(edges_in) + new_edges
    result["intra_chunk_links_added"] = len(new_edges)
    # Reflect the rule version into the graph's top-level ``rule_versions``
    # map so it stays consistent with the per-edge provenance and
    # ``kg_quality_report._summarize_rule_outputs`` (which keys
    # ``rule_outputs[].rule_version`` off this map) surfaces the integer
    # rather than ``None``. Only stamp when edges were actually added — a
    # no-op pass leaves the map (and thus the graph_build_hash inputs)
    # byte-identical. Copy the map so the input graph is never mutated.
    if new_edges:
        existing_versions = graph.get("rule_versions")
        versions = dict(existing_versions) if isinstance(existing_versions, dict) else {}
        versions[RULE_NAME] = RULE_VERSION
        result["rule_versions"] = versions
    return result
