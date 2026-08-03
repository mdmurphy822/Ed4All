"""Rule: derive ``prerequisite`` edges from learning-outcome order.

Heuristic: if concept A first appears in a chunk tagged to an earlier
learning outcome than concept B's first chunk, then B depends on A —
edge ``B --prerequisite--> A``.

"Order" is the position of the outcome id in ``course.json::learning_outcomes``.
Chunks that reference no known outcome are skipped. Concepts that both first
appear at the same LO position produce no edge (no signal).

Deterministic: LO positions are looked up from a frozen ordering; output is
sorted by (source, target) before return. No randomness, no LLM.

Quadratic-closure mitigation (opt-in, ``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY``)
----------------------------------------------------------------------------
The default behaviour emits an edge for every co-occurring concept pair whose
earliest-LO positions differ. On a dense corpus this is an O(n^2) transitive
closure: a linear LO chain A(pos0) -> B(pos1) -> C(pos2) where all three
co-occur emits A<-B, A<-C, B<-C — the redundant A<-C "skip" edge bloats the
graph (measured: 61% of the densest real graph; the sole source of
``lo_order_vs_text_order_conflict`` contradictions). When
``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY=true``:

* **Transitive reduction** — a pair (later, earlier) is suppressed when a
  third concept co-occurs with BOTH endpoints and sits strictly between them
  in LO position. Reachability over the co-occurrence subgraph is preserved
  (B still reaches A through C); only the redundant skip edge disappears. The
  chain A<-B<-C replaces the closure A<-B, A<-C, B<-C.
* **Text-order confidence demotion** — the rule also tracks each concept's
  first-occurrence chunk *index* (document/text order). When LO order and
  text order disagree for a surviving pair (the prerequisite-by-LO concept
  appears LATER in the text than its dependent), the edge's confidence is
  demoted from ``0.6`` to ``0.3`` and ``lo_order_vs_text_order_conflict`` is
  stamped on the evidence, so the downstream edge-consensus pass starts the
  edge weaker instead of discovering the conflict later.

The flag defaults off so existing corpora regenerate byte-identically (the
``concept_graph_semantic.json`` and its sha256 are unchanged). ``RULE_VERSION``
intentionally stays ``1`` — the flag-off path is the version-1 contract;
the reduced output is an operator-selected variant, not a new rule revision.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

RULE_NAME = "prerequisite_from_lo_order"
RULE_VERSION = 1
EDGE_TYPE = "prerequisite"

# Confidence for an LO-order prerequisite edge with no text-order conflict.
_BASE_CONFIDENCE = 0.6
# Demoted confidence when LO order disagrees with first-mention text order.
_CONFLICT_CONFIDENCE = 0.3


def _adjacent_only() -> bool:
    """Return True when ``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY`` is truthy.

    Read per-call (not at import) so tests can toggle via
    ``monkeypatch.setenv`` without reloading the module.
    """
    return os.getenv("TRAINFORGE_PREREQ_LO_ADJACENT_ONLY", "").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


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
) -> Dict[str, Tuple[int, str, str, int]]:
    """For each concept, record (earliest_lo_pos, lo_id, chunk_id, text_idx).

    ``text_idx`` is the index of the concept's earliest-LO chunk in the input
    ``chunks`` list — the document/text order proxy used for the text-order
    confidence-demotion mitigation. Chunks arrive in document order, so a
    lower ``text_idx`` means the concept is introduced earlier in the text.

    When ``TRAINFORGE_SCOPE_CONCEPT_IDS`` is on, graph node IDs are composite
    ``{course_id}:{slug}``. Chunks store
    raw (unscoped) slugs in ``concept_tags``; we scope each tag via the
    chunk's ``source.course_id`` before node-id lookup. Flag-off path is
    identity — behaviour unchanged.
    """
    from Trainforge.rag.typed_edge_inference import _make_concept_id

    first: Dict[str, Tuple[int, str, str, int]] = {}
    for text_idx, chunk in enumerate(chunks):
        refs = chunk.get("learning_outcome_refs") or []
        pos_info = _earliest_lo_position(refs, lo_order)
        if pos_info is None:
            continue
        position, lo_id = pos_info
        course_id = (chunk.get("source") or {}).get("course_id")
        for tag in chunk.get("concept_tags") or []:
            scoped = _make_concept_id(tag, course_id)
            if scoped not in node_ids:
                continue
            prior = first.get(scoped)
            if prior is None or position < prior[0]:
                first[scoped] = (position, lo_id, chunk.get("id") or "", text_idx)
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
        A deterministically-ordered list of edge dicts. When
        ``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY`` is set, the transitive
        closure is reduced to adjacent (consecutive-in-LO-order) pairs and
        text-order conflicts are confidence-demoted; see the module docstring.
    """
    from Trainforge.rag.typed_edge_inference import _make_concept_id

    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    lo_order = _lo_order_map(course)
    if not node_ids or not lo_order:
        return []

    first_by_concept = _first_positions_by_concept(chunks, node_ids, lo_order)
    if len(first_by_concept) < 2:
        return []

    adjacent_only = _adjacent_only()

    # Collect co-occurring pairs (only infer prerequisite for pairs that
    # share a chunk — otherwise the signal is too thin). Per-chunk course
    # scoping matches the scoped node IDs produced upstream when the
    # TRAINFORGE_SCOPE_CONCEPT_IDS flag is on.
    co_occurring: set = set()
    for chunk in chunks:
        course_id = (chunk.get("source") or {}).get("course_id")
        tags = [
            _make_concept_id(t, course_id)
            for t in chunk.get("concept_tags") or []
        ]
        tags = [t for t in tags if t in node_ids]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                co_occurring.add(tuple(sorted((a, b))))

    edges: List[Dict[str, Any]] = []
    for a, b in sorted(co_occurring):
        info_a = first_by_concept.get(a)
        info_b = first_by_concept.get(b)
        if info_a is None or info_b is None:
            continue
        pos_a, lo_a, _, text_a = info_a
        pos_b, lo_b, _, text_b = info_b
        if pos_a == pos_b:
            continue  # same position — no prerequisite signal

        # Transitive reduction (opt-in): suppress the redundant skip edge
        # A<-C when a third concept co-occurs with BOTH endpoints and sits
        # strictly between them in LO order. The chain (A<-mid<-C) preserves
        # reachability; only the closure edge is dropped.
        if adjacent_only and _has_intermediate(
            a, b, pos_a, pos_b, first_by_concept, co_occurring
        ):
            continue

        # Earlier concept is the prerequisite; later concept depends on it.
        if pos_a < pos_b:
            source, target = b, a
            evidence = {
                "target_first_lo": lo_a,
                "target_first_lo_position": pos_a,
                "source_first_lo": lo_b,
                "source_first_lo_position": pos_b,
            }
            prereq_text_idx, dependent_text_idx = text_a, text_b
        else:
            source, target = a, b
            evidence = {
                "target_first_lo": lo_b,
                "target_first_lo_position": pos_b,
                "source_first_lo": lo_a,
                "source_first_lo_position": pos_a,
            }
            prereq_text_idx, dependent_text_idx = text_b, text_a

        confidence = _BASE_CONFIDENCE
        # Text-order demotion (opt-in): LO order says the target is the
        # prerequisite (introduced earlier by LO), but the text introduces it
        # LATER than the dependent — the orderings disagree. Demote and
        # annotate so consensus starts the edge weaker.
        if adjacent_only and prereq_text_idx > dependent_text_idx:
            confidence = _CONFLICT_CONFIDENCE
            evidence["lo_order_vs_text_order_conflict"] = True
            evidence["target_first_text_index"] = prereq_text_idx
            evidence["source_first_text_index"] = dependent_text_idx

        edges.append({
            "source": source,
            "target": target,
            "type": EDGE_TYPE,
            "confidence": confidence,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": evidence,
            },
        })

    return sorted(edges, key=lambda e: (e["source"], e["target"]))


def _has_intermediate(
    a: str,
    b: str,
    pos_a: int,
    pos_b: int,
    first_by_concept: Dict[str, Tuple[int, str, str, int]],
    co_occurring: set,
) -> bool:
    """Return True if a co-occurring concept sits strictly between a and b.

    Transitive-reduction predicate: there exists a third concept ``mid``
    (distinct from a/b) whose earliest-LO position lies strictly between
    ``pos_a`` and ``pos_b`` AND that co-occurs with BOTH ``a`` and ``b``.
    When such a ``mid`` exists, the direct a<->b prerequisite edge is the
    redundant "skip" edge of an A<-mid<-B chain and is suppressed.

    Reachability is preserved: the (a, mid) and (mid, b) pairs are
    themselves co-occurring, so each yields its own (possibly further
    reduced) edge, leaving the later endpoint reachable to the earlier one.
    """
    lo, hi = (pos_a, pos_b) if pos_a < pos_b else (pos_b, pos_a)
    for mid, info_mid in first_by_concept.items():
        if mid == a or mid == b:
            continue
        pos_mid = info_mid[0]
        if not (lo < pos_mid < hi):
            continue
        pair_am = tuple(sorted((a, mid)))
        pair_mb = tuple(sorted((mid, b)))
        if pair_am in co_occurring and pair_mb in co_occurring:
            return True
    return False
