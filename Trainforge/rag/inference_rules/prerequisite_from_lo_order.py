"""Rule: derive ``prerequisite`` edges from learning-outcome order.

Heuristic: if concept A first appears in a chunk tagged to an earlier
learning outcome than concept B's first chunk, then B depends on A —
edge ``B --prerequisite--> A``.

"Order" is the position of the outcome id in ``course.json::learning_outcomes``.
Chunks that reference no known outcome are skipped. Concepts that both first
appear at the same LO position produce no edge (no signal).

Deterministic: LO positions are looked up from a frozen ordering; output is
sorted by (source, target) before return. No randomness, no LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

RULE_NAME = "prerequisite_from_lo_order"
RULE_VERSION = 1
EDGE_TYPE = "prerequisite"


def _lo_order_map(course: Dict[str, Any] | None) -> Dict[str, int]:
    """Return {lo_id -> position} from ``course.json::learning_outcomes``."""
    if not course:
        return {}
    outcomes = course.get("learning_outcomes") or []
    # Normalize to lowercase — chunks store refs lowercased
    # (see process_course._extract_objective_refs).
    return {
        (o.get("id") or "").lower(): idx
        for idx, o in enumerate(outcomes)
        if o.get("id")
    }


def _earliest_lo_position(
    refs: List[str],
    lo_order: Dict[str, int],
) -> Optional[Tuple[int, str]]:
    """Return (position, lo_id) for the earliest-ordered LO in ``refs``."""
    best: Optional[Tuple[int, str]] = None
    for ref in refs or []:
        pos = lo_order.get((ref or "").lower())
        if pos is None:
            continue
        if best is None or pos < best[0]:
            best = (pos, ref.lower())
    return best


def _first_positions_by_concept(
    chunks: List[Dict[str, Any]],
    node_ids: set,
    lo_order: Dict[str, int],
) -> Dict[str, Tuple[int, str, str]]:
    """For each concept, record (earliest_lo_pos, lo_id, chunk_id)."""
    first: Dict[str, Tuple[int, str, str]] = {}
    for chunk in chunks:
        refs = chunk.get("learning_outcome_refs") or []
        pos_info = _earliest_lo_position(refs, lo_order)
        if pos_info is None:
            continue
        position, lo_id = pos_info
        for tag in chunk.get("concept_tags") or []:
            if tag not in node_ids:
                continue
            prior = first.get(tag)
            if prior is None or position < prior[0]:
                first[tag] = (position, lo_id, chunk.get("id") or "")
    return first


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``prerequisite`` edges for concept pairs with LO-order skew.

    Args:
        chunks: Pipeline chunk dicts with ``concept_tags`` and
            ``learning_outcome_refs``.
        course: ``course.json`` dict — the source of outcome ordering.
        concept_graph: The co-occurrence graph dict; only concepts that are
            also nodes in this graph are eligible for edges.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    lo_order = _lo_order_map(course)
    if not node_ids or not lo_order:
        return []

    first_by_concept = _first_positions_by_concept(chunks, node_ids, lo_order)
    if len(first_by_concept) < 2:
        return []

    # Collect co-occurring pairs (only infer prerequisite for pairs that
    # share a chunk — otherwise the signal is too thin).
    co_occurring: set = set()
    for chunk in chunks:
        tags = [t for t in chunk.get("concept_tags") or [] if t in node_ids]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                co_occurring.add(tuple(sorted((a, b))))

    edges: List[Dict[str, Any]] = []
    for a, b in sorted(co_occurring):
        info_a = first_by_concept.get(a)
        info_b = first_by_concept.get(b)
        if info_a is None or info_b is None:
            continue
        pos_a, lo_a, _ = info_a
        pos_b, lo_b, _ = info_b
        if pos_a == pos_b:
            continue  # same position — no prerequisite signal
        # Earlier concept is the prerequisite; later concept depends on it.
        if pos_a < pos_b:
            source, target = b, a
            evidence = {
                "target_first_lo": lo_a,
                "target_first_lo_position": pos_a,
                "source_first_lo": lo_b,
                "source_first_lo_position": pos_b,
            }
        else:
            source, target = a, b
            evidence = {
                "target_first_lo": lo_b,
                "target_first_lo_position": pos_b,
                "source_first_lo": lo_a,
                "source_first_lo_position": pos_a,
            }
        edges.append({
            "source": source,
            "target": target,
            "type": EDGE_TYPE,
            "confidence": 0.6,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": evidence,
            },
        })

    return sorted(edges, key=lambda e: (e["source"], e["target"]))
