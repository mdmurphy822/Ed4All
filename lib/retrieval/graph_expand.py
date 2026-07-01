"""W5.3 — concept-graph passage expansion for the grounded-answer path.

Opt-in (``ED4ALL_ANSWER_GRAPH_EXPAND``, default OFF), fail-open, verdict-safe.

After retrieval, expand the candidate passage set with graph-reachable NEIGHBOR
chunks of the retrieved chunks, read from the archived
``concept_graph_semantic.json``. Two chunks are graph-reachable neighbors when
the graph connects them within one hop through a shared node (a DomainConcept
whose ``occurrences`` list both, or an objective / concept node adjacent to both
chunk nodes).

Contract (mirrors the decompose / HyDE / rerank arms):

* **Anti-fabrication** — every neighbor is a REAL chunk loaded from the course
  chunkset; no id / citation is invented. A reachable id with no loadable chunk
  record is skipped. The kept set is a strict SUBSET of ``retrieved ∪
  graph-reachable``.
* **Verdict-safe / never-worsens** — neighbors are APPENDED after the retrieved
  passages (a retrieved passage is never evicted or reordered) with ``score
  0.0``. The refusal verdict reads ``top_score`` / ``n_above_floor`` over the
  retrieved scores, and a 0.0 neighbor clears no floor, so the verdict is
  byte-identical to the no-expand path. Because the composer selects the first
  ``max_passages`` in order, neighbors only fill RESIDUAL composer slots left by
  a thin retrieval — they enrich a sparse retrieval without ever displacing a
  real retrieved passage.
* **Fail-open** — a missing / unreadable graph, an id scheme that does not match
  the chunkset, or any load error returns the input passages unchanged.

Emits one ``grounded_answer_graph_expand`` decision-capture event with a
dynamic, replayable rationale.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "ENV_GRAPH_EXPAND",
    "ENV_GRAPH_EXPAND_MAX",
    "DECISION_TYPE_GRAPH_EXPAND",
    "graph_expand_enabled",
    "resolve_graph_expand_max",
    "expand_passages_via_graph",
    "graph_reachable_chunk_ids",
]

ENV_GRAPH_EXPAND = "ED4ALL_ANSWER_GRAPH_EXPAND"
ENV_GRAPH_EXPAND_MAX = "ED4ALL_ANSWER_GRAPH_EXPAND_MAX"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Registered in schemas/events/decision_event.schema.json by the Integrate
#: agent — this module only EMITS it (DECISION_VALIDATION_STRICT off ⇒ an unknown
#: type is accepted).
DECISION_TYPE_GRAPH_EXPAND = "grounded_answer_graph_expand"

#: Default cap on graph-reachable neighbors added per answer. Small on purpose —
#: graph expansion fills residual composer slots, it does not flood the pool.
_DEFAULT_MAX_ADD = 4

#: chunkset_kind -> chunks.jsonl relative path (mirrors citation_anchor).
_CHUNKS_REL_BY_KIND: Dict[str, str] = {
    "dart": "dart_chunks/chunks.jsonl",
    "imscc": "imscc_chunks/chunks.jsonl",
    "corpus": "corpus/chunks.jsonl",
}

#: Candidate archived-graph locations under a course dir (mirrors
#: MCP/tools/pipeline_tools.py::_graph_candidates).
_GRAPH_REL_CANDIDATES: Tuple[str, ...] = (
    "graph/concept_graph_semantic.json",
    "imscc_chunks/concept_graph_semantic.json",
    "dart_chunks/concept_graph_semantic.json",
    "corpus/concept_graph_semantic.json",
)


# --------------------------------------------------------------------------- #
# Flag / cap resolvers (parse-with-fallback; default OFF)
# --------------------------------------------------------------------------- #
def graph_expand_enabled() -> bool:
    """``ED4ALL_ANSWER_GRAPH_EXPAND`` truthy? Default OFF."""
    return os.environ.get(ENV_GRAPH_EXPAND, "").strip().lower() in _TRUTHY


def resolve_graph_expand_max(default: int = _DEFAULT_MAX_ADD) -> int:
    """Resolve the per-answer neighbor cap (``ED4ALL_ANSWER_GRAPH_EXPAND_MAX``).

    Garbage / non-positive → ``default`` (parse-with-fallback, mirroring the
    other answer-path caps).
    """
    raw = os.environ.get(ENV_GRAPH_EXPAND_MAX)
    if raw is None:
        return default
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


# --------------------------------------------------------------------------- #
# Decision capture (never breaks the answer path)
# --------------------------------------------------------------------------- #
def _safe_log(capture: Optional[Any], **kwargs: Any) -> None:
    if capture is None:
        return
    try:
        capture.log_decision(**kwargs)
    except Exception:  # pragma: no cover - capture must never break the path
        pass


# --------------------------------------------------------------------------- #
# Graph loading + indexing
# --------------------------------------------------------------------------- #
def _load_semantic_graph(course_dir: Path) -> Optional[Dict[str, Any]]:
    """First readable ``concept_graph_semantic.json`` under ``course_dir``, or None."""
    for rel in _GRAPH_REL_CANDIDATES:
        path = course_dir / rel
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("nodes"), list):
            return data
    return None


def _build_index(
    graph: Dict[str, Any],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build ``(chunk_to_nodes, node_to_chunks, adjacency)`` from a semantic graph.

    Handles both archived shapes:

    * DomainConcept nodes carrying an ``occurrences`` list of chunk ids
      (the co-occurrence graph shape) → each occurrence chunk is bound to the
      concept node.
    * ``Chunk``-class nodes whose id IS a chunk id, wired to objective / concept
      endpoints via edges (the typed-edge graph shape) → the chunk node id is
      bound to itself.

    ``adjacency`` is the undirected node graph derived from ``edges``.
    """
    node_to_chunks: Dict[str, Set[str]] = {}
    chunk_to_nodes: Dict[str, Set[str]] = {}
    adjacency: Dict[str, Set[str]] = {}

    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not nid:
            continue
        nid = str(nid)
        occ = node.get("occurrences")
        if isinstance(occ, list) and occ:
            for cid in occ:
                if not cid:
                    continue
                cid = str(cid)
                node_to_chunks.setdefault(nid, set()).add(cid)
                chunk_to_nodes.setdefault(cid, set()).add(nid)
        if str(node.get("class") or "") == "Chunk":
            # A chunk node id is itself a chunk id (typed-edge graph shape).
            node_to_chunks.setdefault(nid, set()).add(nid)
            chunk_to_nodes.setdefault(nid, set()).add(nid)

    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            continue
        src, tgt = str(src), str(tgt)
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)

    return chunk_to_nodes, node_to_chunks, adjacency


def graph_reachable_chunk_ids(
    seed_ids: Sequence[str],
    chunk_to_nodes: Dict[str, Set[str]],
    node_to_chunks: Dict[str, Set[str]],
    adjacency: Dict[str, Set[str]],
    *,
    max_add: int,
) -> List[str]:
    """Rank graph-reachable neighbor chunk ids for the retrieved ``seed_ids``.

    Neighbors are chunks bound to a seed node OR to a 1-hop neighbor of a seed
    node. Already-retrieved ids are excluded (the return is a strict subset of
    the graph-reachable set minus the seeds). Ranked by how many distinct seed-
    reachable nodes touch the candidate (stronger graph support first), tie-
    broken by chunk id for determinism, then capped to ``max_add``.
    """
    seed_set = {str(s) for s in seed_ids if s}
    if not seed_set:
        return []

    # Seed nodes = every graph node a retrieved chunk is bound to.
    seed_nodes: Set[str] = set()
    for cid in seed_set:
        seed_nodes |= chunk_to_nodes.get(cid, set())
    if not seed_nodes:
        return []

    # Reachable nodes = seed nodes plus their 1-hop neighbors.
    reachable_nodes: Set[str] = set(seed_nodes)
    for n in seed_nodes:
        reachable_nodes |= adjacency.get(n, set())

    # Candidate chunks = chunks bound to any reachable node, minus the seeds.
    support: Dict[str, int] = {}
    for n in reachable_nodes:
        for cid in node_to_chunks.get(n, set()):
            if cid in seed_set:
                continue
            support[cid] = support.get(cid, 0) + 1
    if not support:
        return []

    ranked = sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in ranked[: max(1, int(max_add))]]


# --------------------------------------------------------------------------- #
# Chunk-record → RetrievedPassage loading
# --------------------------------------------------------------------------- #
def _chunks_path(course_dir: Path, chunkset_kind: str) -> Optional[Path]:
    rel = _CHUNKS_REL_BY_KIND.get(chunkset_kind)
    if rel is None:
        # Unknown kind: probe the known chunkset dirs, first that exists wins.
        for cand in _CHUNKS_REL_BY_KIND.values():
            p = course_dir / cand
            if p.is_file():
                return p
        return None
    p = course_dir / rel
    return p if p.is_file() else None


def _load_neighbor_passages(
    chunk_ids: Sequence[str],
    course_dir: Path,
    chunkset_kind: str,
    *,
    engine: str,
) -> List[Any]:
    """Load ``chunk_ids`` from the course chunkset as score-0.0 RetrievedPassages.

    Preserves the requested order. A requested id absent from the chunkset is
    skipped (anti-fabrication: only real chunk records become passages).
    """
    from lib.retrieval.answer_composer import RetrievedPassage

    path = _chunks_path(course_dir, chunkset_kind)
    if path is None:
        return []
    wanted = {str(c) for c in chunk_ids if c}
    if not wanted:
        return []
    by_id: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("id") or rec.get("chunk_id")
                if cid is None:
                    continue
                cid = str(cid)
                if cid in wanted and cid not in by_id:
                    by_id[cid] = rec
                    if len(by_id) == len(wanted):
                        break
    except OSError:
        return []

    passages: List[Any] = []
    for cid in chunk_ids:
        rec = by_id.get(str(cid))
        if rec is None:
            continue
        source = rec.get("source") if isinstance(rec.get("source"), dict) else {}
        text = str(rec.get("text") or rec.get("content") or "")

        def _pick(name: str) -> Any:
            val = rec.get(name)
            if val is None:
                val = (source or {}).get(name)
            return val

        passages.append(
            RetrievedPassage(
                chunk_id=str(cid),
                text=text,
                score=0.0,
                engine=engine,
                item_path=str(_pick("item_path") or ""),
                section_heading=(
                    str(_pick("section_heading"))
                    if _pick("section_heading")
                    else None
                ),
                module_id=str(_pick("module_id")) if _pick("module_id") else None,
                source=source or {},
            )
        )
    return passages


# --------------------------------------------------------------------------- #
# Top-level arm
# --------------------------------------------------------------------------- #
def expand_passages_via_graph(
    passages: Sequence[Any],
    course_dir: Path,
    chunkset_kind: str,
    *,
    engine: str,
    max_add: Optional[int] = None,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> List[Any]:
    """Append graph-reachable neighbor chunks to ``passages`` (verdict-safe).

    Fully fail-open: a missing / unreadable graph, an id scheme that matches no
    retrieved chunk, or any load error returns ``passages`` unchanged. Neighbors
    are appended AFTER the retrieved passages with score 0.0 and deduped against
    the retrieved set, so the retrieved order + scores (and therefore the refusal
    verdict) are byte-identical.
    """
    flat = list(passages)
    if not flat:
        return flat
    cap = resolve_graph_expand_max() if max_add is None else max(1, int(max_add))
    try:
        graph = _load_semantic_graph(Path(course_dir))
    except Exception:  # noqa: BLE001 - fail OPEN
        graph = None
    if graph is None:
        _emit(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:no_graph", n_flat=len(flat), added_ids=[], reason="graph_absent",
        )
        return flat

    try:
        chunk_to_nodes, node_to_chunks, adjacency = _build_index(graph)
        seed_ids = [str(getattr(p, "chunk_id", "") or "") for p in flat]
        reachable = graph_reachable_chunk_ids(
            seed_ids, chunk_to_nodes, node_to_chunks, adjacency, max_add=cap
        )
    except Exception as exc:  # noqa: BLE001 - fail OPEN
        _emit(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:index_error", n_flat=len(flat), added_ids=[],
            reason=type(exc).__name__,
        )
        return flat

    if not reachable:
        _emit(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:no_reachable", n_flat=len(flat), added_ids=[], reason=None,
        )
        return flat

    try:
        neighbors = _load_neighbor_passages(
            reachable, Path(course_dir), chunkset_kind, engine=engine
        )
    except Exception as exc:  # noqa: BLE001 - fail OPEN
        _emit(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:load_error", n_flat=len(flat), added_ids=[],
            reason=type(exc).__name__,
        )
        return flat

    flat_ids = {str(getattr(p, "chunk_id", "") or "") for p in flat}
    added = [p for p in neighbors if str(getattr(p, "chunk_id", "") or "") not in flat_ids]
    if not added:
        _emit(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:no_loadable_neighbors", n_flat=len(flat), added_ids=[],
            reason=None,
        )
        return flat

    added_ids = [str(getattr(p, "chunk_id", "") or "") for p in added]
    _emit(
        capture, course_slug=course_slug, query_sha=query_sha,
        outcome="expanded", n_flat=len(flat), added_ids=added_ids, reason=None,
    )
    # APPEND-only: retrieved passages keep their order + scores → verdict-safe.
    return flat + added


def _emit(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    outcome: str,
    n_flat: int,
    added_ids: Sequence[str],
    reason: Optional[str],
) -> None:
    reason_str = f" reason={reason}" if reason else ""
    rationale = (
        f"graph-expand {outcome} for query {query_sha or '?'} "
        f"(course={course_slug or '?'}){reason_str}; n_retrieved={n_flat} "
        f"n_added={len(added_ids)} added_chunk_ids={list(added_ids)}; "
        f"neighbors are real chunkset records appended at score 0.0 after the "
        f"retrieved set (no eviction/reorder) → refusal verdict byte-identical; "
        f"kept set ⊆ retrieved ∪ graph-reachable (never invents a chunk/citation)"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_GRAPH_EXPAND,
        decision=f"graph_expand:{outcome}",
        rationale=rationale,
        context=f"added={len(added_ids)} n_flat={n_flat}",
    )
