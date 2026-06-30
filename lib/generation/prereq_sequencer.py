"""Prerequisite-DAG-driven terminal-objective sequencing (deterministic).

Today the planner orders terminal objectives (TOs) by their WS1.1 Ward-cluster
list order; the concept ``prerequisite`` edges in ``concept_graph_semantic.json``
are consumed only by KG-quality / edge-consensus, never to SEQUENCE course
content. This module projects those concept-level ``prerequisite`` edges onto a
TO->TO dependency graph and topologically re-sorts the TOs so prerequisite
topics precede dependents.

Design (PURE, deterministic, partition-invariant)
-------------------------------------------------
  * concept<->TO binding comes ONLY from objective ``key_concepts`` (the slugs
    already populated by ``lib/ontology/concept_objective_linker.py``),
    ``canonical_slug``-normalized. The OWNING TO of a concept is the lowest
    original-index TO that teaches it.
  * a prerequisite edge ``B --prerequisite--> A`` (source=B dependent,
    target=A prerequisite) is projected onto a TO edge ``owner(A) -> owner(B)``
    (prerequisite TO precedes dependent TO) when both owners resolve and differ.
  * Kahn topological sort with the ORIGINAL Ward index as a stable min-heap
    tie-break (independent TOs keep their original order).
  * deterministic cycle-break: drop the LOWEST-confidence dependency edge that
    points against the original order, repeating until acyclic.
  * PARTITION INVARIANT: the output TO id-set equals the input id-set (no TO
    added / dropped / re-id'd). Anti-fabrication: cycle-break only DROPS an
    existing edge; no concept / edge / TO is invented.

KNOWN LIMIT (documented honestly): today's prerequisite edges are DERIVED from
LO order (``prerequisite_from_lo_order.py``), so the topo sort largely reproduces
the existing order unless paired with an independent signal
(``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY`` text-order demotion, or a future
content-based prereq rule). The calibration gate measures actual reorder
magnitude.

Default OFF — :func:`resolve_prereq_sequencing` gates the call site; when unset
the caller keeps the WS1.1 order verbatim (byte-identical).
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)

ENV_PREREQ_SEQUENCING = "ED4ALL_PREREQ_SEQUENCING"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Default confidence for a prerequisite edge that carries none.
_DEFAULT_EDGE_CONFIDENCE = 0.6

__all__ = [
    "ENV_PREREQ_SEQUENCING",
    "resolve_prereq_sequencing",
    "sequence_terminal_objectives",
]


def resolve_prereq_sequencing(value: object = None) -> bool:
    """Return True iff prerequisite sequencing is enabled (parse-with-fallback).

    ``value`` overrides the env when not ``None``. Falsey / garbage / unset →
    False.
    """
    if value is None:
        raw = os.environ.get(ENV_PREREQ_SEQUENCING, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Concept-id normalization
# ---------------------------------------------------------------------------


def _norm_concept(raw: Any) -> str:
    """Normalize a concept id / slug to a canonical comparison key.

    Concept graph endpoints may be a flat slug or ``{course_id}:{slug}``; key
    concepts are slugs. Strip a single ``:`` namespace prefix, then
    canonical_slug so both surfaces compare equal.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    s = raw.split(":", 1)[1] if ":" in raw else raw
    return canonical_slug(s)


def _to_id(to: Any) -> str:
    return str(to.get("id") or "") if isinstance(to, dict) else ""


def _concept_owner_map(
    terminal_objectives: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Map normalized concept -> owning TO id (lowest original index wins)."""
    owner: Dict[str, str] = {}
    for to in terminal_objectives:
        tid = _to_id(to)
        if not tid:
            continue
        for kc in (to.get("key_concepts") or []):
            key = _norm_concept(kc)
            if key and key not in owner:
                owner[key] = tid
    return owner


def _prereq_edges(concept_graph: Any) -> List[Tuple[str, str, float]]:
    """Extract (dependent_concept, prereq_concept, confidence) prerequisite edges."""
    out: List[Tuple[str, str, float]] = []
    if not isinstance(concept_graph, dict):
        return out
    edges = concept_graph.get("edges")
    if not isinstance(edges, list):
        return out
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "prerequisite":
            continue
        # source = dependent (B), target = prerequisite (A) per
        # prerequisite_from_lo_order: edge B --prerequisite--> A.
        dep = _norm_concept(e.get("source"))
        pre = _norm_concept(e.get("target"))
        if not dep or not pre or dep == pre:
            continue
        conf = e.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            conf = _DEFAULT_EDGE_CONFIDENCE
        out.append((dep, pre, float(conf)))
    return out


# ---------------------------------------------------------------------------
# TO->TO dependency graph + topo sort
# ---------------------------------------------------------------------------


def _build_to_edges(
    terminal_objectives: List[Dict[str, Any]],
    concept_graph: Any,
) -> Tuple[List[Tuple[str, str, float]], int, int]:
    """Project concept prereq edges onto TO edges (prereq_to -> dep_to).

    Returns (to_edges, n_prereq_concept_edges, n_ignored_unmapped). A TO edge's
    confidence is the MAX confidence across contributing concept edges (the
    strongest evidence). Self-edges and edges whose endpoints map to no TO are
    ignored (anti-fabrication — never invents a TO dependency).
    """
    owner = _concept_owner_map(terminal_objectives)
    concept_edges = _prereq_edges(concept_graph)
    best: Dict[Tuple[str, str], float] = {}
    ignored = 0
    for dep_c, pre_c, conf in concept_edges:
        dep_to = owner.get(dep_c)
        pre_to = owner.get(pre_c)
        if not dep_to or not pre_to or dep_to == pre_to:
            ignored += 1
            continue
        key = (pre_to, dep_to)  # prereq precedes dependent
        if conf > best.get(key, -1.0):
            best[key] = conf
    to_edges = [(p, d, c) for (p, d), c in best.items()]
    return to_edges, len(concept_edges), ignored


def _topo_sort(
    ordered_ids: List[str],
    to_edges: List[Tuple[str, str, float]],
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Kahn topo sort with original-index tie-break + deterministic cycle-break.

    ``ordered_ids`` is the original (Ward) TO order. ``to_edges`` are
    (prereq, dependent, confidence). Returns (sorted_ids, broken_edges).
    """
    orig_index = {tid: i for i, tid in enumerate(ordered_ids)}
    active = list(to_edges)
    broken: List[Tuple[str, str, float]] = []

    while True:
        indeg: Dict[str, int] = {tid: 0 for tid in ordered_ids}
        adj: Dict[str, List[str]] = {tid: [] for tid in ordered_ids}
        seen_pairs = set()
        for pre, dep, _conf in active:
            if pre not in indeg or dep not in indeg:
                continue
            if (pre, dep) in seen_pairs:
                continue
            seen_pairs.add((pre, dep))
            adj[pre].append(dep)
            indeg[dep] += 1

        # Min-heap keyed by original index → stable tie-break.
        heap = [orig_index[t] for t in ordered_ids if indeg[t] == 0]
        heapq.heapify(heap)
        order: List[str] = []
        local_indeg = dict(indeg)
        while heap:
            idx = heapq.heappop(heap)
            tid = ordered_ids[idx]
            order.append(tid)
            for succ in adj[tid]:
                local_indeg[succ] -= 1
                if local_indeg[succ] == 0:
                    heapq.heappush(heap, orig_index[succ])

        if len(order) == len(ordered_ids):
            return order, broken

        # Cycle: drop the lowest-confidence back-pointing edge among the
        # unprocessed nodes (deterministic). "Back-pointing" = the prerequisite
        # currently sits AFTER the dependent in the original order.
        processed = set(order)
        remaining = [
            (p, d, c) for (p, d, c) in active
            if p not in processed and d not in processed
        ]
        candidates = [
            e for e in remaining
            if orig_index.get(e[0], 0) > orig_index.get(e[1], 0)
        ] or remaining
        if not candidates:
            # No removable edge (shouldn't happen) — bail to original order.
            logger.warning("prereq_sequencer: cycle with no breakable edge")
            return list(ordered_ids), broken
        victim = min(
            candidates,
            key=lambda e: (e[2], orig_index.get(e[0], 0), orig_index.get(e[1], 0)),
        )
        active.remove(victim)
        broken.append(victim)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def _graph_sha(concept_graph: Any) -> str:
    try:
        payload = json.dumps(
            concept_graph.get("edges") if isinstance(concept_graph, dict) else None,
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return ""


def sequence_terminal_objectives(
    terminal_objectives: List[Dict[str, Any]],
    concept_graph: Any,
    *,
    capture: Any = None,
    course_code: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Topologically re-sort ``terminal_objectives`` by concept prereq edges.

    PURE reorder of the existing TO list (partition invariant). NEVER raises —
    any error / empty graph / no edges → the original list + a no-op signal.
    Returns ``(reordered_terminal_objectives, signals)``.
    """
    tos = [t for t in (terminal_objectives or []) if isinstance(t, dict)]
    signals: Dict[str, Any] = {
        "reordered": False,
        "n_terminal_objectives": len(tos),
        "n_reordered": 0,
        "to_edges_used": 0,
        "prereq_concept_edges": 0,
        "ignored_unmapped_edges": 0,
        "cycles_broken": 0,
        "graph_sha": "",
    }
    try:
        ordered_ids = [_to_id(t) for t in tos]
        # Guard: missing / duplicate ids make the partition invariant unsafe.
        if any(not i for i in ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
            logger.warning(
                "prereq_sequencer: missing/duplicate TO ids — keeping original order"
            )
            _emit(capture, course_code, signals)
            return list(terminal_objectives or []), signals

        signals["graph_sha"] = _graph_sha(concept_graph)
        to_edges, n_concept_edges, ignored = _build_to_edges(tos, concept_graph)
        signals["prereq_concept_edges"] = n_concept_edges
        signals["ignored_unmapped_edges"] = ignored
        signals["to_edges_used"] = len(to_edges)

        if not to_edges:
            _emit(capture, course_code, signals)
            return list(terminal_objectives or []), signals

        sorted_ids, broken = _topo_sort(ordered_ids, to_edges)
        signals["cycles_broken"] = len(broken)

        # PARTITION INVARIANT — never trust a reorder that lost/gained an id.
        if set(sorted_ids) != set(ordered_ids) or len(sorted_ids) != len(ordered_ids):
            logger.error(
                "prereq_sequencer: partition invariant violated — keeping order"
            )
            _emit(capture, course_code, signals)
            return list(terminal_objectives or []), signals

        by_id = {_to_id(t): t for t in tos}
        reordered = [by_id[i] for i in sorted_ids]
        n_moved = sum(1 for a, b in zip(ordered_ids, sorted_ids) if a != b)
        signals["n_reordered"] = n_moved
        signals["reordered"] = n_moved > 0
        _emit(capture, course_code, signals)
        return reordered, signals
    except Exception as exc:  # noqa: BLE001 — never break the build
        logger.warning("prereq_sequencer: failed (%s) — keeping original order", exc)
        _emit(capture, course_code, signals)
        return list(terminal_objectives or []), signals


def _emit(capture: Any, course_code: str, signals: Dict[str, Any]) -> None:
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type="content_selection",
            decision=(
                f"prereq_sequencing reordered={signals['reordered']} "
                f"n_reordered={signals['n_reordered']}/"
                f"{signals['n_terminal_objectives']}"
            ),
            rationale=(
                "Projected concept prerequisite edges onto a TO->TO dependency "
                f"graph for course {course_code or '<unknown>'}: "
                f"{signals['prereq_concept_edges']} prereq concept edge(s) -> "
                f"{signals['to_edges_used']} TO edge(s) "
                f"({signals['ignored_unmapped_edges']} ignored as unmapped); "
                f"topologically re-sorted {signals['n_terminal_objectives']} TOs "
                f"(moved {signals['n_reordered']}), broke "
                f"{signals['cycles_broken']} cycle edge(s) (graph "
                f"{signals['graph_sha'] or 'n/a'}). Pure partition-invariant "
                "reorder; prerequisite topics precede dependents."
            ),
            ml_features={
                "n_terminal_objectives": int(signals["n_terminal_objectives"]),
                "n_reordered": int(signals["n_reordered"]),
                "to_edges_used": int(signals["to_edges_used"]),
                "prereq_concept_edges": int(signals["prereq_concept_edges"]),
                "cycles_broken": int(signals["cycles_broken"]),
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("prereq_sequencer: decision capture raised: %s", exc)
