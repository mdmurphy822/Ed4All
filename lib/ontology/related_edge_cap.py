"""Per-node top-K fan-out cap for ``related-to`` edges.

The ``related_from_cooccurrence`` inference rule emits ``related-to`` edges that
dominate the semantic concept graph (roughly 78% of all edges on real corpora).
Many are spurious co-occurrence artifacts (e.g. ``advanced`` <-> ``mnist-digits``,
``pros`` <-> ``cons``). PMI/Jaccard normalization does *not* separate signal from
noise here: the spurious pairs have perfect (Jaccard == 1.0) association because
they always co-occur. The effective lever is a per-node fan-out cap — keep only
the strongest few ``related-to`` edges per concept.

``cap_related_fanout`` is a pure ``dict -> dict`` transform over a
``concept_semantic`` graph. It is deterministic (stable tie-breaks) so two runs
produce byte-identical output. It does not read the environment; the cap value
``k`` is supplied by the call site (a separate integration step wires this into
the pipeline).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

__all__ = ["cap_related_fanout"]

_RELATED_TYPE = "related-to"


def _edge_sort_key(edge: Dict[str, Any]) -> tuple:
    """Rank key for an incident edge: (confidence, weight) desc, stable tie-break.

    Higher ``confidence`` then higher ``weight`` rank first. Ties break on the
    ``(source, target, type)`` string ascending so ranking is deterministic and
    independent of input edge ordering. Missing numeric fields fall back to 0.
    """

    confidence = edge.get("confidence")
    weight = edge.get("weight")
    confidence = confidence if isinstance(confidence, (int, float)) else 0
    weight = weight if isinstance(weight, (int, float)) else 0
    tie = "\x00".join(
        str(edge.get(field, "")) for field in ("source", "target", "type")
    )
    # Negate the numeric ranks so a plain ascending sort yields desc order while
    # the ascending string tie-break is preserved.
    return (-confidence, -weight, tie)


def cap_related_fanout(graph: dict, k: int) -> dict:
    """Cap per-node fan-out of ``related-to`` edges at top-K (UNION semantics).

    Only edges with ``type == "related-to"`` are considered for capping; every
    other edge type (``defined-by``, ``exemplifies``, ``is-a``,
    ``prerequisite``, ...) is preserved verbatim. For each node, its incident
    ``related-to`` edges are ranked by ``(confidence, weight)`` descending with a
    stable ``(source, target, type)`` tie-break. A ``related-to`` edge SURVIVES
    if it is within the top-K of EITHER endpoint (union) — so a peripheral but
    genuine concept whose strongest link is its neighbor's 5th-best is not
    starved. Edges that are top-K for neither endpoint are dropped.

    Returns a NEW graph dict (deep-copied) with the filtered edge list. Nodes and
    non-``related-to`` edges are preserved exactly. Stamps
    ``graph["related_fanout_cap_k"] = k`` for observability. ``rule_versions`` is
    left untouched. The function does not mutate its input.
    """

    out: Dict[str, Any] = copy.deepcopy(graph)
    edges: List[Dict[str, Any]] = out.get("edges") or []

    related: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for edge in edges:
        if edge.get("type") == _RELATED_TYPE:
            related.append(edge)
        else:
            other.append(edge)

    # Bucket related-to edges by each incident node.
    incident: Dict[Any, List[Dict[str, Any]]] = {}
    for edge in related:
        for node in (edge.get("source"), edge.get("target")):
            incident.setdefault(node, []).append(edge)

    # Collect the set of edges that survive (top-K for at least one endpoint).
    survivors: set[int] = set()
    if k > 0:
        for _node, node_edges in incident.items():
            ranked = sorted(node_edges, key=_edge_sort_key)
            for edge in ranked[:k]:
                survivors.add(id(edge))

    # Reassemble preserving original relative order (other edges interleaved as
    # they appeared, related survivors in their original order).
    new_edges: List[Dict[str, Any]] = []
    for edge in edges:
        if edge.get("type") == _RELATED_TYPE:
            if id(edge) in survivors:
                new_edges.append(edge)
        else:
            new_edges.append(edge)

    out["edges"] = new_edges
    out["related_fanout_cap_k"] = k
    return out
