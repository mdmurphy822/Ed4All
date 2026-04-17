"""Rule: derive ``related-to`` edges from co-occurrence weight.

The existing ``concept_graph.json`` (built by
``Trainforge.process_course._build_tag_graph``) already stores every
concept-tag pair that co-occurs in at least one chunk, with a ``weight``
count. This rule does NOT recompute co-occurrence — it consumes
``concept_graph["edges"]`` and re-emits the high-weight pairs as typed
``related-to`` edges.

Threshold is configurable; default is 3 (pairs co-occurring in >=3 chunks).
``related-to`` is the lowest-precedence typed edge — the orchestrator drops
it when either ``is-a`` or ``prerequisite`` covers the same (source, target).

Deterministic: output sorted by (source, target).
"""

from __future__ import annotations

from typing import Any, Dict, List

RULE_NAME = "related_from_cooccurrence"
RULE_VERSION = 1
EDGE_TYPE = "related-to"
DEFAULT_THRESHOLD = 3


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``related-to`` edges for concept pairs with co-occurrence >= threshold.

    Args:
        chunks: Unused (we reuse concept_graph.edges which is already computed).
            Kept for interface parity.
        course: Unused; interface parity.
        concept_graph: The co-occurrence graph dict. Edges should have
            ``weight`` (from the base generator) or be filtered by caller.
        threshold: Minimum weight to keep. Defaults to 3.

    Returns:
        A deterministically-ordered list of edge dicts. ``related-to`` edges
        are undirected in semantic intent; we always emit with
        ``source < target`` lexicographically so dedup is trivial.
    """
    del chunks, course  # unused; interface parity

    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    edges: List[Dict[str, Any]] = []

    for base_edge in concept_graph.get("edges", []) or []:
        src = base_edge.get("source")
        tgt = base_edge.get("target")
        weight = base_edge.get("weight", 0)
        if not src or not tgt or src == tgt:
            continue
        if src not in node_ids or tgt not in node_ids:
            continue
        if weight < threshold:
            continue
        a, b = sorted([src, tgt])
        edges.append({
            "source": a,
            "target": b,
            "type": EDGE_TYPE,
            "confidence": min(1.0, 0.4 + 0.05 * weight),
            "weight": weight,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": {
                    "cooccurrence_weight": weight,
                    "threshold": threshold,
                },
            },
        })

    return sorted(edges, key=lambda e: (e["source"], e["target"]))
