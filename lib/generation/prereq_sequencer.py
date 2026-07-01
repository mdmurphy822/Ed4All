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

KNOWN LIMIT (documented honestly): the LEGACY prerequisite edges are DERIVED
from LO order (``prerequisite_from_lo_order.py``), so the topo sort largely
reproduces the existing order unless paired with an independent signal
(``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY`` text-order demotion, or the W3.1
content-based ``prerequisite_from_definition_mention`` rule). The calibration
gate measures actual reorder magnitude.

W3 additions (this module)
--------------------------
  * **W3.1 Kahn topo-union** — :func:`_build_to_edges` now consumes BOTH the
    concept-projected prerequisite edges (concept endpoints, mapped to their
    owning TO) AND *federation* ``TO -> TO`` prerequisite edges whose endpoints
    are already TO ids (emitted by
    ``lib/generation/prerequisite_from_definition_mention.py``). The two edge
    sources are unioned (max-confidence per TO pair) before the topo sort.
  * **W3.3 transitive reduction** (``ED4ALL_PREREQ_TRANSITIVE_REDUCTION``,
    default OFF) — a stdlib DFS-reachability pass drops a projected TO edge
    ``u -> v`` when ``v`` is still reachable from ``u`` without it, removing
    edges implied by transitivity. Reachability-preserving, so it never
    changes the topo/cycle structure; deterministic (edges processed in sorted
    order).
  * **W3.4 concept-centrality tie-break** (``ED4ALL_PREREQ_CENTRALITY_TIEBREAK``,
    default OFF) — concept centrality (in-degree, or PageRank via
    ``ED4ALL_PREREQ_CENTRALITY_METHOD=pagerank``) is aggregated onto each TO and
    used as a STABLE tie-break in the Kahn heap (front-load high-centrality
    prerequisites), with the original Ward index as the final tie-break. The
    concept + TO centrality maps are surfaced on the returned signals for Lane
    K to expose.

Default OFF — :func:`resolve_prereq_sequencing` gates the call site; when unset
the caller keeps the WS1.1 order verbatim (byte-identical). The W3.3 / W3.4
passes are independently default-OFF, so with the sequencer on but those flags
unset the output is byte-identical to the pre-W3 sequencer.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)

ENV_PREREQ_SEQUENCING = "ED4ALL_PREREQ_SEQUENCING"
#: W3.3 — transitive reduction of the projected TO prereq graph (default OFF).
ENV_PREREQ_TRANSITIVE_REDUCTION = "ED4ALL_PREREQ_TRANSITIVE_REDUCTION"
#: W3.4 — concept-centrality stable tie-break in the topo order (default OFF).
ENV_PREREQ_CENTRALITY_TIEBREAK = "ED4ALL_PREREQ_CENTRALITY_TIEBREAK"
#: W3.4 — centrality method: "in_degree" (default) or "pagerank".
ENV_PREREQ_CENTRALITY_METHOD = "ED4ALL_PREREQ_CENTRALITY_METHOD"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Default confidence for a prerequisite edge that carries none.
_DEFAULT_EDGE_CONFIDENCE = 0.6

#: Stdlib PageRank parameters (deterministic power iteration).
_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 40

__all__ = [
    "ENV_PREREQ_SEQUENCING",
    "ENV_PREREQ_TRANSITIVE_REDUCTION",
    "ENV_PREREQ_CENTRALITY_TIEBREAK",
    "ENV_PREREQ_CENTRALITY_METHOD",
    "resolve_prereq_sequencing",
    "resolve_transitive_reduction",
    "resolve_centrality_tiebreak",
    "sequence_terminal_objectives",
]


def _resolve_flag(env_name: str, value: object = None) -> bool:
    if value is None:
        raw = os.environ.get(env_name, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


def resolve_prereq_sequencing(value: object = None) -> bool:
    """Return True iff prerequisite sequencing is enabled (parse-with-fallback).

    ``value`` overrides the env when not ``None``. Falsey / garbage / unset →
    False.
    """
    return _resolve_flag(ENV_PREREQ_SEQUENCING, value)


def resolve_transitive_reduction(value: object = None) -> bool:
    """Return True iff W3.3 transitive reduction is enabled (default OFF)."""
    return _resolve_flag(ENV_PREREQ_TRANSITIVE_REDUCTION, value)


def resolve_centrality_tiebreak(value: object = None) -> bool:
    """Return True iff W3.4 centrality tie-break is enabled (default OFF)."""
    return _resolve_flag(ENV_PREREQ_CENTRALITY_TIEBREAK, value)


def _resolve_centrality_method(value: object = None) -> str:
    """Return the centrality method ("in_degree" default / "pagerank")."""
    if value is None:
        raw = os.environ.get(ENV_PREREQ_CENTRALITY_METHOD, "")
    else:
        raw = str(value)
    method = raw.strip().lower()
    return method if method in {"in_degree", "pagerank"} else "in_degree"


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


def _lo_key(raw: Any) -> str:
    """Uppercase-normalize an LO/TO id for case-insensitive federation match.

    Federation ``TO -> TO`` prerequisite edges (from
    ``prerequisite_from_definition_mention``) may carry TO ids in a different
    case than the sequencer's ``terminal_objectives`` ids (chunks store refs
    lowercased). Uppercasing both sides makes them compare equal.
    """
    return raw.strip().upper() if isinstance(raw, str) else ""


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


def _prereq_edges_raw(concept_graph: Any) -> List[Tuple[str, str, float]]:
    """Extract RAW (dependent, prerequisite, confidence) prerequisite edges.

    Endpoints are returned VERBATIM (not concept-normalized) so
    :func:`_build_to_edges` can distinguish a *federation* ``TO -> TO`` edge
    (endpoints already TO ids) from a concept edge (endpoints concept slugs).
    ``source`` = dependent (B), ``target`` = prerequisite (A) per the
    ``prerequisite_from_lo_order`` / ``prerequisite_from_definition_mention``
    direction convention (edge ``B --prerequisite--> A``).
    """
    out: List[Tuple[str, str, float]] = []
    if not isinstance(concept_graph, dict):
        return out
    edges = concept_graph.get("edges")
    if not isinstance(edges, list):
        return out
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "prerequisite":
            continue
        src = e.get("source")
        tgt = e.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str) or not src or not tgt:
            continue
        conf = e.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            conf = _DEFAULT_EDGE_CONFIDENCE
        out.append((src, tgt, float(conf)))
    return out


# ---------------------------------------------------------------------------
# TO->TO dependency graph + topo sort
# ---------------------------------------------------------------------------


def _build_to_edges(
    terminal_objectives: List[Dict[str, Any]],
    concept_graph: Any,
    *,
    transitive_reduction: bool = False,
) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
    """Project + union prereq edges onto TO edges (prereq_to -> dep_to).

    Kahn topo-UNION of two edge sources (W3.1):

      1. *Concept* prerequisite edges — endpoints are concept slugs, mapped to
         their owning TO via ``key_concepts`` (legacy path).
      2. *Federation* ``TO -> TO`` prerequisite edges — endpoints are already
         TO ids (case-insensitive match against the TO set), used directly.
         These carry the content-dependency signal from
         ``prerequisite_from_definition_mention``.

    A TO edge's confidence is the MAX confidence across contributing edges (the
    strongest evidence). Self-edges and edges whose endpoints map to no TO are
    ignored (anti-fabrication — never invents a TO dependency).

    When ``transitive_reduction`` is True (W3.3), edges implied by transitivity
    are removed via a deterministic stdlib DFS-reachability pass.

    Returns ``(to_edges, stats)`` where ``stats`` counts the contributing
    signals for the decision capture.
    """
    owner = _concept_owner_map(terminal_objectives)
    to_key_map = {
        _lo_key(_to_id(t)): _to_id(t)
        for t in terminal_objectives
        if _to_id(t)
    }
    raw_edges = _prereq_edges_raw(concept_graph)

    best: Dict[Tuple[str, str], float] = {}
    ignored = 0
    federation_edges = 0
    concept_projected = 0
    for src_raw, tgt_raw, conf in raw_edges:
        # 1. Federation TO->TO edge: both endpoints resolve to known TO ids.
        dep_to = to_key_map.get(_lo_key(src_raw))
        pre_to = to_key_map.get(_lo_key(tgt_raw))
        if dep_to and pre_to:
            if dep_to == pre_to:
                ignored += 1
                continue
            key = (pre_to, dep_to)  # prereq precedes dependent
            if conf > best.get(key, -1.0):
                best[key] = conf
            federation_edges += 1
            continue
        # 2. Concept edge: map each endpoint's slug to its owning TO.
        dep_owner = owner.get(_norm_concept(src_raw))
        pre_owner = owner.get(_norm_concept(tgt_raw))
        if not dep_owner or not pre_owner or dep_owner == pre_owner:
            ignored += 1
            continue
        key = (pre_owner, dep_owner)
        if conf > best.get(key, -1.0):
            best[key] = conf
        concept_projected += 1

    to_edges = [(p, d, c) for (p, d), c in best.items()]

    reduced = 0
    if transitive_reduction and to_edges:
        to_edges, reduced = _transitive_reduction(to_edges)

    stats = {
        "prereq_concept_edges": len(raw_edges),
        "ignored_unmapped_edges": ignored,
        "federation_edges": federation_edges,
        "concept_projected_edges": concept_projected,
        "transitively_reduced": reduced,
    }
    return to_edges, stats


def _reachable(adj: Dict[str, set], src: str, dst: str) -> bool:
    """Return True iff ``dst`` is reachable from ``src`` in ``adj`` (stdlib DFS)."""
    if src == dst:
        return True
    stack = [src]
    seen = {src}
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, ()):  # membership boolean is order-independent
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _transitive_reduction(
    to_edges: List[Tuple[str, str, float]],
) -> Tuple[List[Tuple[str, str, float]], int]:
    """Remove TO edges implied by transitivity (W3.3, deterministic).

    ``to_edges`` are ``(pre, dep, confidence)`` meaning ``pre -> dep`` (pre
    precedes dep). An edge ``u -> v`` is redundant when ``v`` is still
    reachable from ``u`` via the OTHER edges; such edges are dropped. The pass
    is reachability-preserving, so the reachable-set (and therefore every valid
    topo order + the cycle structure) is unchanged — it only thins redundant
    skip edges. Processed in sorted order for determinism; safe on cyclic
    inputs (a 2-cycle keeps both edges — neither is reachable without itself).
    """
    ordered = sorted(to_edges, key=lambda e: (e[0], e[1]))
    adj: Dict[str, set] = defaultdict(set)
    for pre, dep, _c in ordered:
        adj[pre].add(dep)
    kept: List[Tuple[str, str, float]] = []
    removed = 0
    for pre, dep, conf in ordered:
        adj[pre].discard(dep)
        if _reachable(adj, pre, dep):
            removed += 1  # redundant — leave dropped from adjacency
        else:
            adj[pre].add(dep)  # restore
            kept.append((pre, dep, conf))
    return kept, removed


# ---------------------------------------------------------------------------
# W3.4 — concept centrality (stable tie-break, exposed for Lane K)
# ---------------------------------------------------------------------------


def _concept_centrality(
    concept_graph: Any,
    method: str,
    to_ids: Optional[set] = None,
) -> Dict[str, float]:
    """Return ``{norm_concept: centrality}`` over the concept graph edges.

    ``in_degree`` (default): a concept's centrality is the count of edges that
    point AT it (target endpoint) — how many concepts reference/depend on it,
    a proxy for foundationality. ``pagerank``: deterministic power-iteration
    PageRank over the directed edge set. Both are stdlib-only. Endpoints are
    ``_norm_concept``-normalized so they align with the ``key_concepts`` slugs.

    Federation ``TO -> TO`` edges are EXCLUDED: their endpoints are terminal
    objectives, not concepts (detected via ``to_ids`` — the uppercase-keyed TO
    set — or the canonical LO-id pattern as a fallback).
    """
    if not isinstance(concept_graph, dict):
        return {}
    edges = concept_graph.get("edges")
    if not isinstance(edges, list):
        return {}
    from lib.ontology.learning_objectives import validate_lo_id

    to_keys = to_ids or set()

    def _is_lo(raw: Any) -> bool:
        if not isinstance(raw, str):
            return False
        key = _lo_key(raw)
        return key in to_keys or validate_lo_id(key)

    nodes: set = set()
    directed: List[Tuple[str, str]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        raw_src = e.get("source")
        raw_tgt = e.get("target")
        # Skip federation edges whose endpoints are LO/TO ids — centrality is a
        # CONCEPT-graph measure; TO endpoints are not concepts.
        if _is_lo(raw_src) or _is_lo(raw_tgt):
            continue
        src = _norm_concept(raw_src)
        tgt = _norm_concept(raw_tgt)
        if not src or not tgt:
            continue
        nodes.add(src)
        nodes.add(tgt)
        directed.append((src, tgt))
    # Seed every declared node so isolated concepts carry a defined score.
    for n in concept_graph.get("nodes") or []:
        if isinstance(n, dict):
            key = _norm_concept(n.get("id"))
            if key:
                nodes.add(key)
    if not nodes:
        return {}

    if method == "pagerank":
        return _pagerank(nodes, directed)

    # in_degree
    indeg: Dict[str, float] = {n: 0.0 for n in nodes}
    for _src, tgt in directed:
        indeg[tgt] = indeg.get(tgt, 0.0) + 1.0
    return indeg


def _pagerank(nodes: set, directed: List[Tuple[str, str]]) -> Dict[str, float]:
    """Deterministic stdlib PageRank power iteration over a directed edge set."""
    n = len(nodes)
    if n == 0:
        return {}
    out_adj: Dict[str, List[str]] = defaultdict(list)
    for src, tgt in directed:
        out_adj[src].append(tgt)
    rank: Dict[str, float] = {node: 1.0 / n for node in nodes}
    base = (1.0 - _PAGERANK_DAMPING) / n
    for _ in range(_PAGERANK_ITERATIONS):
        nxt: Dict[str, float] = {node: base for node in nodes}
        dangling = 0.0
        for node in nodes:
            outs = out_adj.get(node)
            if outs:
                share = _PAGERANK_DAMPING * rank[node] / len(outs)
                for tgt in outs:
                    nxt[tgt] += share
            else:
                dangling += _PAGERANK_DAMPING * rank[node] / n
        if dangling:
            for node in nodes:
                nxt[node] += dangling
        rank = nxt
    return rank


def _to_centrality(
    terminal_objectives: List[Dict[str, Any]],
    concept_centrality: Dict[str, float],
) -> Dict[str, float]:
    """Aggregate concept centrality onto each TO (sum over its key_concepts)."""
    out: Dict[str, float] = {}
    for to in terminal_objectives:
        tid = _to_id(to)
        if not tid:
            continue
        score = 0.0
        for kc in (to.get("key_concepts") or []):
            score += concept_centrality.get(_norm_concept(kc), 0.0)
        out[tid] = score
    return out


def _topo_sort(
    ordered_ids: List[str],
    to_edges: List[Tuple[str, str, float]],
    *,
    priority: Optional[Dict[str, Tuple[Any, ...]]] = None,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Kahn topo sort with a stable tie-break + deterministic cycle-break.

    ``ordered_ids`` is the original (Ward) TO order. ``to_edges`` are
    (prereq, dependent, confidence). ``priority`` maps each TO id to a
    heap-key tuple WHOSE LAST ELEMENT IS THE ORIGINAL INDEX (guaranteeing a
    total order — the original index is unique). When omitted the key is
    ``(orig_index,)`` — byte-identical to the pre-W3.4 original-index-only
    tie-break. The W3.4 centrality tie-break passes
    ``(-to_centrality, orig_index)`` so high-centrality prerequisites are
    front-loaded while ties fall back to the original order.
    Returns (sorted_ids, broken_edges).
    """
    orig_index = {tid: i for i, tid in enumerate(ordered_ids)}
    prio: Dict[str, Tuple[Any, ...]] = priority or {
        tid: (i,) for i, tid in enumerate(ordered_ids)
    }
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

        # Min-heap keyed by the priority tuple → stable tie-break. The tuple's
        # last element is the original index, so tids are recovered from it.
        heap = [prio[t] for t in ordered_ids if indeg[t] == 0]
        heapq.heapify(heap)
        order: List[str] = []
        local_indeg = dict(indeg)
        while heap:
            key = heapq.heappop(heap)
            tid = ordered_ids[key[-1]]
            order.append(tid)
            for succ in adj[tid]:
                local_indeg[succ] -= 1
                if local_indeg[succ] == 0:
                    heapq.heappush(heap, prio[succ])

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
    transitive_reduction = resolve_transitive_reduction()
    centrality_tiebreak = resolve_centrality_tiebreak()
    signals: Dict[str, Any] = {
        "reordered": False,
        "n_terminal_objectives": len(tos),
        "n_reordered": 0,
        "to_edges_used": 0,
        "prereq_concept_edges": 0,
        "ignored_unmapped_edges": 0,
        "federation_edges": 0,
        "concept_projected_edges": 0,
        "transitively_reduced": 0,
        "cycles_broken": 0,
        "transitive_reduction_enabled": transitive_reduction,
        "centrality_tiebreak_enabled": centrality_tiebreak,
        "centrality_method": "",
        "concept_centrality": {},
        "to_centrality": {},
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
        to_edges, edge_stats = _build_to_edges(
            tos, concept_graph, transitive_reduction=transitive_reduction
        )
        signals["prereq_concept_edges"] = edge_stats["prereq_concept_edges"]
        signals["ignored_unmapped_edges"] = edge_stats["ignored_unmapped_edges"]
        signals["federation_edges"] = edge_stats["federation_edges"]
        signals["concept_projected_edges"] = edge_stats["concept_projected_edges"]
        signals["transitively_reduced"] = edge_stats["transitively_reduced"]
        signals["to_edges_used"] = len(to_edges)

        # W3.4 — compute concept centrality + its TO aggregation. Always
        # surfaced on the signals (for Lane K) once the sequencer runs; only
        # USED as a tie-break when the flag is on (byte-identical otherwise).
        priority: Optional[Dict[str, Tuple[Any, ...]]] = None
        if centrality_tiebreak:
            method = _resolve_centrality_method()
            to_id_keys = {_lo_key(tid) for tid in ordered_ids}
            concept_cent = _concept_centrality(concept_graph, method, to_id_keys)
            to_cent = _to_centrality(tos, concept_cent)
            signals["centrality_method"] = method
            signals["concept_centrality"] = concept_cent
            signals["to_centrality"] = to_cent
            priority = {
                tid: (-to_cent.get(tid, 0.0), i)
                for i, tid in enumerate(ordered_ids)
            }

        if not to_edges:
            _emit(capture, course_code, signals)
            return list(terminal_objectives or []), signals

        sorted_ids, broken = _topo_sort(ordered_ids, to_edges, priority=priority)
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
                "Unioned concept + federation prerequisite edges onto a TO->TO "
                f"dependency graph for course {course_code or '<unknown>'}: "
                f"{signals['prereq_concept_edges']} prereq edge(s) -> "
                f"{signals['to_edges_used']} TO edge(s) "
                f"({signals['federation_edges']} federation TO->TO, "
                f"{signals['concept_projected_edges']} concept-projected, "
                f"{signals['ignored_unmapped_edges']} ignored as unmapped, "
                f"{signals['transitively_reduced']} transitively reduced); "
                f"topologically re-sorted {signals['n_terminal_objectives']} TOs "
                f"(moved {signals['n_reordered']}), broke "
                f"{signals['cycles_broken']} cycle edge(s) (graph "
                f"{signals['graph_sha'] or 'n/a'}; centrality tie-break="
                f"{signals['centrality_tiebreak_enabled']}). Pure partition-"
                "invariant reorder; prerequisite topics precede dependents."
            ),
            ml_features={
                "n_terminal_objectives": int(signals["n_terminal_objectives"]),
                "n_reordered": int(signals["n_reordered"]),
                "to_edges_used": int(signals["to_edges_used"]),
                "prereq_concept_edges": int(signals["prereq_concept_edges"]),
                "federation_edges": int(signals["federation_edges"]),
                "concept_projected_edges": int(signals["concept_projected_edges"]),
                "transitively_reduced": int(signals["transitively_reduced"]),
                "cycles_broken": int(signals["cycles_broken"]),
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("prereq_sequencer: decision capture raised: %s", exc)
