#!/usr/bin/env python3
"""
Trainforge — Prerequisite-Aware Curriculum Ordering (Wave 79 Worker B)

Reads ``pedagogy_graph.json`` from a LibV2 course archive and provides:

  * :func:`build_concept_topo_order` — Kahn's-algorithm topological sort over
    ``prerequisite_of`` edges (Concept → Concept). Cycles are broken
    deterministically by ``(first_seen_week, concept_id)`` ascending so two
    runs over the same archive return the same ordering.

  * :func:`order_pairs_by_curriculum` — given a topo order and a list of
    training pairs, returns the pairs sorted by the *latest* concept their
    chunk references in topo order. Pairs whose chunks reference no graph
    concepts go to the end (preserving their input order amongst themselves).

  * :func:`build_prereq_recap` — for a chunk's concepts, look up depth-1
    ``prerequisite_of`` predecessors and pull the first sentence from the
    chunk where each predecessor was first introduced. Caps the recap at
    ``context_tokens`` (whitespace-token approximation, default 200).

  * :func:`build_curriculum_manifest` — returns the JSON-shaped manifest
    document this stage writes to disk alongside the standard outputs.

The module is self-contained: no I/O against the corpus directly. All graph
loading happens in :func:`load_pedagogy_graph` so callers can inject test
fixtures (``Trainforge/tests/test_prereq_curriculum.py``) without writing a
real archive on disk.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Default ceiling on prerequisite-recap token count. The CLI exposes this as
# ``--prereq-context-tokens``.
DEFAULT_PREREQ_CONTEXT_TOKENS = 200


# ---------------------------------------------------------------------------
# Graph loading + concept-tag normalization
# ---------------------------------------------------------------------------


def load_pedagogy_graph(path: Path) -> Dict[str, Any]:
    """Load a pedagogy_graph.json document from disk.

    Returns the raw dict unchanged so callers can read whatever fields they
    need without forcing a schema. Missing file -> FileNotFoundError, malformed
    JSON -> ValueError (re-raised by ``json.load``).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _concept_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return Concept-class nodes from a pedagogy graph dict.

    Matches both the production class name (``Concept``) and the literal
    ``DomainConcept`` aliasing the task description uses; tolerant lookup so
    fixtures wired with either label work without translation.
    """
    out: List[Dict[str, Any]] = []
    for node in graph.get("nodes", []) or []:
        cls = str(node.get("class") or "").strip()
        if cls in ("Concept", "DomainConcept"):
            out.append(node)
    return out


def _prerequisite_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ``prerequisite_of`` edges from a pedagogy graph dict.

    Pedagogy graphs key the relation under ``relation_type`` (production) or
    ``relation`` (some older fixtures). Both shapes accepted.
    """
    out: List[Dict[str, Any]] = []
    for edge in graph.get("edges", []) or []:
        rel = edge.get("relation_type") or edge.get("relation")
        if str(rel) == "prerequisite_of":
            out.append(edge)
    return out


def _slug_to_concept_id(concepts: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Map concept slug -> canonical concept id.

    Chunks store ``concept_tags`` as bare slugs (``"rdf-graph"``) but the
    pedagogy graph keys concepts as ``"concept:rdf-graph"``. Both forms are
    resolved against this lookup so the caller doesn't have to choose.
    """
    out: Dict[str, str] = {}
    for c in concepts:
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        slug = str(c.get("slug") or "").strip()
        if slug and slug not in out:
            out[slug] = cid
        # Self-mapping so callers that pass the canonical id pass through.
        out[cid] = cid
    return out


def normalize_concept_refs(
    refs: Sequence[str],
    concept_lookup: Dict[str, str],
) -> List[str]:
    """Translate a list of concept slugs / IDs into canonical concept IDs.

    Refs not present in the graph are silently dropped (they wouldn't have a
    topo position anyway). Order is preserved; duplicates are collapsed.
    """
    seen: set = set()
    out: List[str] = []
    for r in refs or []:
        key = str(r).strip()
        if not key:
            continue
        canon = concept_lookup.get(key)
        if canon is None:
            # Allow a reverse "concept:foo" -> "foo" pass too, in case the
            # caller passed canonical IDs but the graph indexed only slugs.
            if key.startswith("concept:"):
                canon = concept_lookup.get(key[len("concept:") :])
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


# ---------------------------------------------------------------------------
# Topo sort (Kahn's algorithm) with deterministic cycle-breaking
# ---------------------------------------------------------------------------


@dataclass
class TopoResult:
    """Output of :func:`build_concept_topo_order`."""

    order: List[str] = field(default_factory=list)
    method: str = "kahn"
    cycles_broken: List[List[str]] = field(default_factory=list)
    # Map concept_id -> integer position (0-based) in ``order``. Built once so
    # the pair-ordering pass is O(1) per concept lookup.
    position: Dict[str, int] = field(default_factory=dict)


def build_concept_topo_order(graph: Dict[str, Any]) -> TopoResult:
    """Topologically sort concept nodes by ``prerequisite_of`` edges.

    Algorithm: Kahn's algorithm. At each step pop the node with the smallest
    ``(first_seen_week, concept_id)`` from the zero-indegree set. This makes
    the ordering fully deterministic: equally-ready concepts always emerge in
    week-then-id order. Cycle break is the same rule applied to remaining
    nodes (smallest week-then-id first), and each broken cycle (the offending
    edge's source/target chain) is recorded in ``cycles_broken`` so the
    manifest can surface it for human inspection.

    Returns a :class:`TopoResult` with the order, method tag, and
    cycles-broken record. ``position`` indexes order for O(1) lookup later.
    """
    concepts = _concept_nodes(graph)
    edges = _prerequisite_edges(graph)
    concept_ids = {str(c.get("id") or "").strip() for c in concepts}
    concept_ids.discard("")

    # Sort key: (first_seen_week or +inf, id) — concepts without a week land
    # last among their indegree-0 cohort, behind those with a known week.
    week_by_id: Dict[str, int] = {}
    for c in concepts:
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        wk = c.get("first_seen_week")
        if isinstance(wk, (int, float)):
            week_by_id[cid] = int(wk)
        else:
            # Sentinel chosen to sort after any reasonable course week count.
            week_by_id[cid] = 10**9

    indeg: Dict[str, int] = {cid: 0 for cid in concept_ids}
    succ: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if src not in concept_ids or tgt not in concept_ids:
            continue
        # Each source is a prerequisite of target; target's indegree grows.
        succ[src].append(tgt)
        indeg[tgt] += 1

    # Sort successors deterministically too so iteration order is stable
    # across Python dict-insertion accidents.
    for src in succ:
        succ[src].sort(key=lambda x: (week_by_id.get(x, 10**9), x))

    def _key(cid: str) -> Tuple[int, str]:
        return (week_by_id.get(cid, 10**9), cid)

    # Indegree-0 frontier kept as a sorted list; we pop the smallest-key
    # element each iteration. A heapq would be faster but the corpora are
    # small (sub-1k concepts) and the sorted-list version is easier to audit.
    ready = sorted([cid for cid, d in indeg.items() if d == 0], key=_key)

    order: List[str] = []
    cycles_broken: List[List[str]] = []

    while ready or any(d > 0 for d in indeg.values()):
        if ready:
            cid = ready.pop(0)
            order.append(cid)
            for nxt in succ.get(cid, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    # Insert preserving sort order.
                    _insort(ready, nxt, _key)
        else:
            # Cycle: every remaining node has indegree >= 1. Pick the
            # smallest-key remaining concept, force its indegree to 0, and
            # record the cycle for the manifest. This loops until every
            # remaining node has been emitted.
            remaining = sorted(
                [cid for cid, d in indeg.items() if d > 0 or cid not in set(order)],
                key=_key,
            )
            # Filter out nodes we've already emitted (could happen if a
            # successor's indegree dropped to 0 mid-iteration).
            emitted = set(order)
            remaining = [cid for cid in remaining if cid not in emitted]
            if not remaining:
                break
            broken = remaining[0]
            # Reconstruct the cycle path involving ``broken`` for the report:
            # walk forward from broken following successors until we either
            # cycle back or hit a frontier node. This is best-effort; the
            # manifest just needs to surface "a cycle was broken here".
            cycles_broken.append(_trace_cycle(broken, succ, emitted))
            # Force-emit and decrement successors.
            order.append(broken)
            indeg[broken] = -1  # mark consumed
            for nxt in succ.get(broken, ()):
                if nxt in emitted:
                    continue
                indeg[nxt] = max(0, indeg[nxt] - 1)
                if indeg[nxt] == 0:
                    _insort(ready, nxt, _key)

    position = {cid: i for i, cid in enumerate(order)}
    return TopoResult(
        order=order,
        method="kahn",
        cycles_broken=cycles_broken,
        position=position,
    )


def _insort(seq: List[str], item: str, key) -> None:
    """Insert ``item`` into ``seq`` preserving sort order under ``key``."""
    k = key(item)
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(seq[mid]) <= k:
            lo = mid + 1
        else:
            hi = mid
    seq.insert(lo, item)


def _trace_cycle(
    start: str,
    succ: Dict[str, List[str]],
    emitted: set,
    max_steps: int = 32,
) -> List[str]:
    """Best-effort reconstruction of a cycle path starting at ``start``.

    Walks forward by successor (skipping already-emitted nodes) until we
    return to ``start`` or exceed ``max_steps``. Emits the visited path as a
    list; if no proper cycle can be traced the singleton ``[start]`` is
    returned and the manifest still records that the node was force-emitted.
    """
    path = [start]
    seen = {start}
    cur = start
    for _ in range(max_steps):
        nxts = [n for n in succ.get(cur, ()) if n not in emitted]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            path.append(nxt)
            return path
        if nxt in seen:
            path.append(nxt)
            return path
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


# ---------------------------------------------------------------------------
# Pair ordering against a topo result
# ---------------------------------------------------------------------------


def order_pairs_by_curriculum(
    pairs: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    topo: TopoResult,
    concept_lookup: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[str], int]:
    """Order ``pairs`` by the latest topo position of any concept their
    chunk references.

    Pairs whose chunk references no graph concept go to the end (in their
    input order). Tie-break inside a topo bucket: chunk_id ascending, then
    seed ascending, so the ordering is byte-stable across runs at the same
    seed.

    Returns:
        ordered_pairs: list of pair dicts in emit order.
        pairs_by_concept_position: ``concept_id -> [pair_summary]`` for the
            curriculum manifest. ``pair_summary`` is a dict with
            ``pair_id``, ``chunk_id``, ``extraction_method``, and ``seed``
            so consumers can join back to the JSONL outputs.
        concepts_without_pairs: concept ids in topo order that had no pair
            anchor onto them.
        pairs_without_concepts: count of pairs whose chunk had zero graph
            concepts.
    """
    pairs_by_concept_position: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    bucketed: List[Tuple[int, str, int, int, Dict[str, Any]]] = []
    no_concept: List[Tuple[int, Dict[str, Any]]] = []
    pairs_without_concepts = 0

    for input_idx, pair in enumerate(pairs):
        chunk_id = str(pair.get("chunk_id") or "")
        chunk = chunks_by_id.get(chunk_id) or {}
        tags = chunk.get("concept_tags") or []
        canon = normalize_concept_refs(tags, concept_lookup)
        # The pair anchors at the LATEST concept its chunk uses (the deepest
        # in topo order). This is the "earliest position you could safely
        # introduce this pair" criterion from the task spec.
        latest_pos = -1
        latest_cid = ""
        for cid in canon:
            pos = topo.position.get(cid)
            if pos is None:
                continue
            if pos > latest_pos:
                latest_pos = pos
                latest_cid = cid
        if latest_pos < 0:
            no_concept.append((input_idx, pair))
            pairs_without_concepts += 1
            continue
        seed_val = int(pair.get("seed") or 0)
        bucketed.append((latest_pos, chunk_id, seed_val, input_idx, pair))
        pairs_by_concept_position[latest_cid].append(
            {
                "pair_id": _pair_identifier(pair),
                "chunk_id": chunk_id,
                "extraction_method": _extraction_method(pair),
                "seed": seed_val,
            }
        )

    bucketed.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    ordered = [p for _, _, _, _, p in bucketed]
    no_concept.sort(key=lambda t: t[0])
    ordered.extend(p for _, p in no_concept)

    concepts_without_pairs = [
        cid for cid in topo.order if cid not in pairs_by_concept_position
    ]
    return (
        ordered,
        {k: v for k, v in pairs_by_concept_position.items()},
        concepts_without_pairs,
        pairs_without_concepts,
    )


def _pair_identifier(pair: Dict[str, Any]) -> str:
    """Resolve a stable pair identifier for the manifest report.

    Prefers an explicit ``id`` field (misconception-DPO pairs carry one);
    falls back to ``"<chunk_id>:<seed>"`` for instruction / preference pairs.
    """
    pid = pair.get("id")
    if pid:
        return str(pid)
    return f"{pair.get('chunk_id', '')}:{pair.get('seed', 0)}"


def _extraction_method(pair: Dict[str, Any]) -> str:
    """Derive an extraction-method tag for the manifest.

    Order of precedence:
      * ``provider`` (instruction pairs) — e.g. ``mock`` or ``anthropic``;
      * ``source`` (preference pairs)   — e.g. ``misconception_editorial``;
      * fallback ``unknown``.
    """
    return str(pair.get("provider") or pair.get("source") or "unknown")


# ---------------------------------------------------------------------------
# Prerequisite recap (used by --prereq-windowed)
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    sentences = _SENTENCE_SPLIT_RE.split(text, maxsplit=1)
    first = sentences[0].strip()
    # Truncate the absolute monster sentences so the recap stays usable
    # even on chunks whose first sentence runs to a paragraph.
    if len(first) > 320:
        first = first[:320].rstrip() + "..."
    return first


def _token_count(text: str) -> int:
    """Whitespace-token approximation — good enough for recap budgeting."""
    return len([t for t in (text or "").split() if t])


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = (text or "").split()
    if len(tokens) <= max_tokens:
        return text or ""
    return " ".join(tokens[:max_tokens]).rstrip() + "..."


def build_prereq_predecessors_index(
    graph: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Map concept_id -> list of depth-1 ``prerequisite_of`` predecessors.

    A predecessor is a concept that must be learned BEFORE the target —
    i.e. an edge ``predecessor --prerequisite_of--> target``. Predecessors
    are returned in insertion order; tie-break the recap output by topo
    position when consumers need a deterministic emit.
    """
    edges = _prerequisite_edges(graph)
    out: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if not src or not tgt:
            continue
        if src in out[tgt]:
            continue
        out[tgt].append(src)
    return dict(out)


def build_first_seen_chunk_index(
    chunks: Sequence[Dict[str, Any]],
    concept_lookup: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Return concept_id -> the FIRST chunk dict that tags that concept.

    "First" is determined by chunk position in the input sequence, which
    matches the corpus emit order (chronological / module-aligned). The
    chunk dict is returned by reference so the caller can reach the text
    and source fields for the recap.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        for tag in chunk.get("concept_tags") or []:
            cid = concept_lookup.get(str(tag).strip())
            if cid is None and str(tag).startswith("concept:"):
                cid = concept_lookup.get(str(tag)[len("concept:") :])
            if cid is None or cid in out:
                continue
            out[cid] = chunk
    return out


def build_prereq_recap(
    pair: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    concept_lookup: Dict[str, str],
    predecessors: Dict[str, List[str]],
    first_seen_chunk: Dict[str, Dict[str, Any]],
    *,
    context_tokens: int = DEFAULT_PREREQ_CONTEXT_TOKENS,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build the prerequisites recap block for one pair.

    For each concept the pair's chunk uses, look up its depth-1
    ``prerequisite_of`` predecessors. For each predecessor pull the first
    sentence from the chunk where it was first introduced. Concatenate as
    a "Prerequisites recap:" block, capped at ``context_tokens`` total
    tokens. Returns the empty string when there is nothing to recap.
    """
    chunk_id = str(pair.get("chunk_id") or "")
    chunk = chunks_by_id.get(chunk_id) or {}
    canon = normalize_concept_refs(
        chunk.get("concept_tags") or [], concept_lookup
    )
    if not canon:
        return ""

    seen: set = set()
    items: List[str] = []
    for cid in canon:
        for pred in predecessors.get(cid, []) or []:
            if pred in seen:
                continue
            seen.add(pred)
            src = first_seen_chunk.get(pred)
            if not src:
                continue
            # Don't recap predecessors the pair's own chunk introduces —
            # the learner is already in the right neighborhood.
            if str(src.get("id") or src.get("chunk_id") or "") == chunk_id:
                continue
            sent = _first_sentence(src.get("text") or "")
            if not sent:
                continue
            label = (label_lookup or {}).get(pred) or _label_from_id(pred)
            items.append(f"- {label}: {sent}")
    if not items:
        return ""

    block = "Prerequisites recap:\n" + "\n".join(items)
    return _truncate_to_tokens(block, context_tokens)


def _label_from_id(concept_id: str) -> str:
    """Cheap label fallback when no label_lookup is supplied."""
    raw = concept_id
    if raw.startswith("concept:"):
        raw = raw[len("concept:") :]
    return raw.replace("-", " ").replace("_", " ")


def build_concept_label_lookup(graph: Dict[str, Any]) -> Dict[str, str]:
    """Map concept_id -> human label for recap output."""
    out: Dict[str, str] = {}
    for c in _concept_nodes(graph):
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        label = str(c.get("label") or "").strip() or _label_from_id(cid)
        out[cid] = label
    return out


# ---------------------------------------------------------------------------
# Curriculum manifest (written next to instruction_pairs.jsonl)
# ---------------------------------------------------------------------------


def build_curriculum_manifest(
    *,
    slug: str,
    topo: TopoResult,
    pairs_by_concept_position: Dict[str, List[Dict[str, Any]]],
    concepts_without_pairs: List[str],
    pairs_without_concepts: int,
) -> Dict[str, Any]:
    """Compose the JSON manifest the stage writes to disk.

    Schema (informal):
        {
            "slug": str,
            "topo_order": [concept_id, ...],
            "topo_method": "kahn",
            "cycles_broken": [[concept_id, ...], ...],
            "pairs_by_concept_position": {
                concept_id: [{"pair_id":..., "chunk_id":..., "extraction_method":...}]
            },
            "concepts_without_pairs": [concept_id, ...],
            "pairs_without_concepts": int
        }
    """
    return {
        "slug": slug,
        "topo_order": list(topo.order),
        "topo_method": topo.method,
        "cycles_broken": [list(c) for c in topo.cycles_broken],
        "pairs_by_concept_position": {
            k: list(v) for k, v in pairs_by_concept_position.items()
        },
        "concepts_without_pairs": list(concepts_without_pairs),
        "pairs_without_concepts": int(pairs_without_concepts),
    }


# ---------------------------------------------------------------------------
# Top-level convenience: load + topo in one call
# ---------------------------------------------------------------------------


@dataclass
class CurriculumContext:
    """Cached state needed for both pair ordering and recap building."""

    graph: Dict[str, Any]
    topo: TopoResult
    concept_lookup: Dict[str, str]
    predecessors: Dict[str, List[str]]
    label_lookup: Dict[str, str]
    first_seen_chunk: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def build_curriculum_context(
    graph: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
) -> CurriculumContext:
    """Bundle every derived index a synthesis run needs.

    Computed once per run; reused by the pair ordering and the per-pair
    recap construction.
    """
    concepts = _concept_nodes(graph)
    lookup = _slug_to_concept_id(concepts)
    topo = build_concept_topo_order(graph)
    preds = build_prereq_predecessors_index(graph)
    label_lookup = build_concept_label_lookup(graph)
    first_seen = build_first_seen_chunk_index(chunks, lookup)
    return CurriculumContext(
        graph=graph,
        topo=topo,
        concept_lookup=lookup,
        predecessors=preds,
        label_lookup=label_lookup,
        first_seen_chunk=first_seen,
    )


__all__ = [
    "CurriculumContext",
    "DEFAULT_PREREQ_CONTEXT_TOKENS",
    "TopoResult",
    "build_concept_label_lookup",
    "build_concept_topo_order",
    "build_curriculum_context",
    "build_curriculum_manifest",
    "build_first_seen_chunk_index",
    "build_prereq_predecessors_index",
    "build_prereq_recap",
    "load_pedagogy_graph",
    "normalize_concept_refs",
    "order_pairs_by_curriculum",
]
