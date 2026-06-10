"""Duplicate-concept node-merge pass for the semantic concept graph.

Concept nodes are keyed by their raw tag/slug, so near-duplicate surface forms
land as *distinct* nodes that fragment the knowledge graph:

* ``Knowledge Base`` / ``Knowledge Bases`` (regular plural)
* ``Agent`` / ``Agents`` (regular plural)
* ``Running State`` / ``Running State Chain`` (head-concept prefix)
* ``Pydantic`` / ``Pydantic Validator Guide`` (head-concept prefix)

This module folds those duplicates into a single canonical node via two
passes:

* **Pass A — plural fold (unconditional).** Two concept nodes whose ids
  singularize (``lib.ontology.lexical_concept_seeds._singular``) to the same
  slug, or where one folds onto the other through
  :data:`lib.ontology.lexical_concept_seeds.ALIAS_FOLD`, are merged. This is a
  high-precision lexical equivalence, so no co-occurrence guard is required.

* **Pass B — affix fold (guarded).** When one node's whitespace-tokenized
  label is a contiguous *prefix* OR *suffix* of another's (``running state`` ⊂
  ``running state chain``; ``vector store`` ⊂ ``faiss vector store``), the
  longer (more specific) label is folded into the shorter (head-concept) label
  — but ONLY when the two nodes are corroborated to be the same concept, i.e.
  they share at least one ``occurrences[]`` chunk back-reference OR there is a
  direct edge between them. Affix-only (prefix or suffix at an end): an
  *interior* head-noun is never folded (``a embedding b`` does NOT fold into
  ``embedding``), since interior-substring folding is too lossy.

  Three over-merge guards keep flagship concepts intact (a 1-word affix is too
  weak a merge signal — co-located course concepts trivially share a chunk so
  the corroboration guard never blocks, and path-compression cascades unrelated
  nodes into a super-node):

  1. **≥2-word affix.** The shorter (head) label must be ≥2 words AND a
     contiguous prefix/suffix of the longer. A single-token affix fold
     (``langchain`` ⊂ ``refine-langchain``; ``rag`` ⊂ ``agentic-rag``;
     ``faiss`` ⊂ ``faiss-vector-store``) is rejected. Genuine multi-word
     affix dups (``running state`` ⊂ ``running state chain``) still merge.
  2. **Tech-anchor merge-immunity (as LOSER).** A node whose id is a
     :func:`lib.ontology.tech_anchors.anchor_slugs` slug is never demoted to a
     loser via affix fold (it may still be a winner). Protects ``langchain``,
     ``faiss``, ``ragas``, ``langgraph``, ``nim``, ``rag``, etc.
  3. **High-frequency head-node protection.** A single-token node with
     frequency ≥ :data:`_HEAD_NODE_FREQ_FLOOR` (or ≥ the would-be winner's
     frequency) is never demoted to a loser via affix fold.

The function is pure ``dict -> dict``, deterministic, and reads no environment
variables — the opt-in flag gate lives at the (future) call site. The input
graph is not mutated destructively; a new graph dict is returned.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from lib.ontology.lexical_concept_seeds import ALIAS_FOLD, _singular
from lib.ontology.tech_anchors import anchor_slugs

__all__ = ["merge_duplicate_concept_nodes"]

# A single-token node with frequency at or above this floor is treated as a
# flagship head concept and is never demoted to a loser via the guarded
# affix fold (Pass B). A high-frequency standalone term (e.g. ``langchain``
# at freq 62) is the canonical concept, not a folding candidate.
_HEAD_NODE_FREQ_FLOOR = 5

# A chunk node carries a ``chunk_NNNNN`` token in its id (full corpus form
# ``demo_course_chunk_00270`` or the short ``chunk_00270``). Chunk nodes are
# NEVER folded — only ``DomainConcept`` concept nodes are.
_CHUNK_ID_RE = re.compile(r"chunk_\d+")


def _is_chunk_id(node_id: Any) -> bool:
    return isinstance(node_id, str) and bool(_CHUNK_ID_RE.search(node_id))


def _is_concept_node(node: Dict[str, Any]) -> bool:
    """Foldable iff it is a ``DomainConcept`` and not a chunk node."""
    if node.get("class") != "DomainConcept":
        return False
    if _is_chunk_id(node.get("id")):
        return False
    return True


def _label_words(node: Dict[str, Any]) -> List[str]:
    """Whitespace-tokenized lowercase words of a node's label.

    Falls back to the slug-derived id (hyphens -> spaces) when no ``label`` is
    present, so prefix folding still works on label-less graphs.
    """
    label = node.get("label")
    if not isinstance(label, str) or not label.strip():
        nid = node.get("id")
        label = nid.replace("-", " ") if isinstance(nid, str) else ""
    return [w for w in label.lower().split() if w]


def _frequency(node: Dict[str, Any]) -> int:
    freq = node.get("frequency", 0)
    return freq if isinstance(freq, int) else 0


# Tech-anchor slugs are flagship foundational-technology concepts; an affix
# fold must never demote one to a loser. Computed once at import.
_TECH_ANCHOR_SLUGS = frozenset(anchor_slugs())


def _is_loser_immune(node_id: str, words: List[str], freq: int, winner_freq: int) -> bool:
    """True when ``node_id`` must NOT be demoted to a Pass-B affix loser.

    Two protections:

    * **Tech-anchor immunity** — the id is a canonical tech-anchor slug.
    * **High-frequency head node** — a single-token node whose frequency
      meets the head-node floor or matches/exceeds the would-be winner's.
    """
    if node_id in _TECH_ANCHOR_SLUGS:
        return True
    if len(words) <= 1 and (freq >= _HEAD_NODE_FREQ_FLOOR or freq >= winner_freq):
        return True
    return False


def _resolve(merge_into: Dict[str, str], node_id: str) -> str:
    """Path-compress ``node_id`` to its terminal winner."""
    root = node_id
    seen = []
    while root in merge_into and merge_into[root] != root:
        seen.append(root)
        root = merge_into[root]
    for nid in seen:
        merge_into[nid] = root
    return root


def _pick_plural_winner(a: str, b: str, freq: Dict[str, int]) -> Tuple[str, str]:
    """Return ``(winner, loser)`` for a Pass-A plural/alias merge.

    Winner = higher frequency; tie-break shorter id then lexicographically
    smaller id (fully deterministic, order-independent).
    """
    fa, fb = freq.get(a, 0), freq.get(b, 0)
    if fa != fb:
        return (a, b) if fa > fb else (b, a)
    if len(a) != len(b):
        return (a, b) if len(a) < len(b) else (b, a)
    return (a, b) if a <= b else (b, a)


def merge_duplicate_concept_nodes(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Fold near-duplicate ``DomainConcept`` nodes into canonical nodes.

    Args:
        graph: A concept/semantic graph dict with ``nodes`` and ``edges``
            lists. Concept nodes carry ``id``, ``label``, ``frequency``,
            ``class`` and optional ``occurrences[]`` / ``source_refs[]``.
            Edges carry ``source`` / ``target`` and optional ``type`` /
            ``confidence`` (or ``relation_type``).

    Returns:
        A new graph dict with duplicate concept nodes folded, edges re-pointed
        onto winners (self-loops dropped, parallel edges collapsed keeping the
        highest ``confidence``), and ``concept_node_merges`` stamped with the
        number of folded loser nodes. ``rule_versions`` / ``rulepack_version``
        are left untouched. The input is not mutated.

        Edge-endpoint policy: an edge is dropped only when an endpoint that WAS
        an input node got folded into nothing (a hard guard — folded-loser
        endpoints re-point to their winner, never dangle). An edge whose
        endpoint was NEVER an input node (e.g. an LO-authored ``targets-concept``
        target the upstream materializer left as a phantom, or any other
        federation-by-convention endpoint resolved by ID-namespace) is passed
        through verbatim with re-pointed endpoints rather than dropped —
        input-dangling edges are the materializer's / orphan-gate's problem,
        not the merge pass's. The count of such pass-through edges is stamped on
        ``external_endpoint_edges_preserved``.
    """
    nodes_in = graph.get("nodes")
    if not isinstance(nodes_in, list):
        nodes_in = []
    edges_in = graph.get("edges")
    if not isinstance(edges_in, list):
        edges_in = []

    # Index foldable concept nodes by id; preserve original order.
    concept_nodes: Dict[str, Dict[str, Any]] = {}
    concept_order: List[str] = []
    for node in nodes_in:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        if _is_concept_node(node) and nid not in concept_nodes:
            concept_nodes[nid] = node
            concept_order.append(nid)

    freq: Dict[str, int] = {nid: _frequency(concept_nodes[nid]) for nid in concept_order}
    occ: Dict[str, set] = {}
    for nid in concept_order:
        raw = concept_nodes[nid].get("occurrences")
        occ[nid] = set(raw) if isinstance(raw, list) else set()

    # Direct adjacency (undirected) among concept nodes, for the Pass-B guard.
    adjacency: Dict[str, set] = {nid: set() for nid in concept_order}
    for edge in edges_in:
        if not isinstance(edge, dict):
            continue
        s, t = edge.get("source"), edge.get("target")
        if s in adjacency and t in adjacency and s != t:
            adjacency[s].add(t)
            adjacency[t].add(s)

    merge_into: Dict[str, str] = {}

    def _record_merge(winner: str, loser: str) -> None:
        w = _resolve(merge_into, winner)
        l = _resolve(merge_into, loser)
        if w == l:
            return
        # Re-pick the survivor between the two CURRENT roots so chained merges
        # still land on the globally-best winner deterministically.
        new_w, new_l = _pick_plural_winner(w, l, freq)
        merge_into[new_l] = new_w
        # Roll frequency/occurrences up onto the survivor so subsequent
        # tie-breaks and guards see the merged signal.
        freq[new_w] = freq.get(new_w, 0) + freq.get(new_l, 0)
        occ[new_w] = occ.get(new_w, set()) | occ.get(new_l, set())
        adjacency[new_w] = adjacency.get(new_w, set()) | adjacency.get(new_l, set())

    # --- Pass A: plural / alias fold (unconditional) ----------------------
    # Two concept ids are plural/alias equivalent when ANY of the following
    # surface relations hold (all built from the imported, dependency-free
    # ``_singular`` / ``ALIAS_FOLD`` — no new pluralization logic):
    #
    #   * ``_singular(x) == _singular(y)`` — both singularize to the same slug.
    #   * one is the singular of the other (``_singular(x) == y``).
    #   * stripping a bare trailing ``s`` from one yields the other. This
    #     covers the regular ``base``/``bases`` case where ``_singular`` is
    #     asymmetric ("bases" -> "bas" but "base" -> "base"); we link the two
    #     surface forms directly rather than through a non-word canonical key.
    #   * ALIAS_FOLD maps one onto the other (irregulars / known variants).
    #
    # Equivalence is closed transitively via the deterministic anchor merge,
    # so order does not matter.
    id_set = set(concept_order)

    def _bare_singular(slug: str) -> str:
        head, sep, tail = slug.rpartition("-")
        if tail.endswith("s") and not tail.endswith(("ss", "us", "is", "as")) and len(tail) > 3:
            tail = tail[:-1]
        return f"{head}{sep}{tail}" if sep else tail

    def _plural_equiv(a: str, b: str) -> bool:
        sa, sb = _singular(a), _singular(b)
        if sa == sb:
            return True
        if sa == b or sb == a:
            return True
        if _bare_singular(a) == b or _bare_singular(b) == a:
            return True
        if _bare_singular(a) == _bare_singular(b):
            return True
        if ALIAS_FOLD.get(a) == b or ALIAS_FOLD.get(b) == a:
            return True
        if ALIAS_FOLD.get(a, a) == ALIAS_FOLD.get(b, b):
            return True
        return False

    # Deterministic pairwise scan over the sorted id list. Every equivalent
    # pair is merged; ``_record_merge`` + path compression closes transitivity.
    sorted_ids = sorted(id_set)
    for i, a in enumerate(sorted_ids):
        for b in sorted_ids[i + 1:]:
            if _plural_equiv(a, b):
                _record_merge(a, b)

    # --- Pass B: affix fold (guarded) -------------------------------------
    # Candidate pairs: distinct concept nodes where one's label words are a
    # contiguous PREFIX or SUFFIX of the other's. Iterate over a deterministic
    # ordering. Interior-substring folding is deliberately NOT done (too lossy).
    words = {nid: _label_words(concept_nodes[nid]) for nid in concept_order}
    ordered = sorted(concept_order)
    for i, a in enumerate(ordered):
        wa = words[a]
        if not wa:
            continue
        for b in ordered[i + 1:]:
            wb = words[b]
            if not wb:
                continue
            # Strict affix: the shorter word-list is a prefix OR suffix of the
            # longer.
            if len(wa) < len(wb):
                short, long_ = a, b
                short_w, long_w = wa, wb
            elif len(wb) < len(wa):
                short, long_ = b, a
                short_w, long_w = wb, wa
            else:
                continue  # equal length -> not a strict affix
            k = len(short_w)
            # Guard 1: a 1-word affix is too weak a merge signal (co-located
            # course concepts trivially share a chunk). Require the shorter
            # (head) label to be >=2 words AND a contiguous affix of the longer.
            if k < 2:
                continue
            is_prefix = long_w[:k] == short_w
            is_suffix = long_w[-k:] == short_w
            if not (is_prefix or is_suffix):
                continue
            ra = _resolve(merge_into, short)
            rb = _resolve(merge_into, long_)
            if ra == rb:
                continue
            # Guard: corroborate same-concept via shared occurrence chunk OR a
            # direct edge between the (resolved) endpoints.
            shared_chunk = bool(occ.get(ra, set()) & occ.get(rb, set()))
            direct_edge = (rb in adjacency.get(ra, set())) or (ra in adjacency.get(rb, set()))
            if not (shared_chunk or direct_edge):
                continue
            # Guards 2 & 3: the loser (the LONGER, more-specific node ``long_``)
            # is never demoted if it is a tech-anchor slug or a protected
            # high-frequency single-token head node. ``short`` (the head) is the
            # winner; ``long_`` would be the loser. Re-check immunity against the
            # CURRENT resolved long-node + winner frequency.
            if _is_loser_immune(rb, words.get(rb, words[long_]), freq.get(rb, 0), freq.get(ra, 0)):
                continue
            # Winner is the SHORTER label (head concept) regardless of freq.
            # Force the merge direction explicitly.
            merge_into[rb] = ra
            freq[ra] = freq.get(ra, 0) + freq.get(rb, 0)
            occ[ra] = occ.get(ra, set()) | occ.get(rb, set())
            adjacency[ra] = adjacency.get(ra, set()) | adjacency.get(rb, set())

    # Path-compress every chain to its terminal winner.
    for nid in list(merge_into):
        _resolve(merge_into, nid)
    final_winner: Dict[str, str] = {nid: _resolve(merge_into, nid) for nid in concept_order}

    losers = {nid for nid in concept_order if final_winner[nid] != nid}

    # --- Rewrite nodes -----------------------------------------------------
    # Accumulate folded-in signal per winner (deterministic union/sort/dedup).
    fold_freq: Dict[str, int] = {}
    fold_occ: Dict[str, set] = {}
    fold_srcs: Dict[str, List[Dict[str, Any]]] = {}
    fold_src_seen: Dict[str, set] = {}

    def _src_key(ref: Any) -> Tuple:
        if isinstance(ref, dict):
            return tuple(sorted((str(k), str(v)) for k, v in ref.items()))
        return (repr(ref),)

    # Process winners first, then losers, in stable original order so
    # first-seen-wins source_refs is deterministic.
    order_for_fold = [nid for nid in concept_order if final_winner[nid] == nid] + [
        nid for nid in concept_order if final_winner[nid] != nid
    ]
    for nid in order_for_fold:
        win = final_winner[nid]
        node = concept_nodes[nid]
        fold_freq[win] = fold_freq.get(win, 0) + _frequency(node)
        raw_occ = node.get("occurrences")
        if isinstance(raw_occ, list):
            fold_occ.setdefault(win, set()).update(raw_occ)
        else:
            fold_occ.setdefault(win, set())
        seen = fold_src_seen.setdefault(win, set())
        bucket = fold_srcs.setdefault(win, [])
        raw_src = node.get("source_refs")
        if isinstance(raw_src, list):
            for ref in raw_src:
                k = _src_key(ref)
                if k not in seen:
                    seen.add(k)
                    bucket.append(ref)

    new_nodes: List[Dict[str, Any]] = []
    for node in nodes_in:
        if not isinstance(node, dict):
            new_nodes.append(node)
            continue
        nid = node.get("id")
        if nid in losers:
            continue  # dropped
        if nid in concept_nodes and final_winner.get(nid) == nid:
            merged = dict(node)
            merged["frequency"] = fold_freq.get(nid, _frequency(node))
            occ_union = fold_occ.get(nid)
            if occ_union:
                merged["occurrences"] = sorted(occ_union)
            srcs = fold_srcs.get(nid)
            if srcs:
                merged["source_refs"] = srcs
            new_nodes.append(merged)
        else:
            new_nodes.append(node)

    # --- Re-point edges ----------------------------------------------------
    valid_ids = {n["id"] for n in new_nodes if isinstance(n, dict) and isinstance(n.get("id"), str)}
    # Every id that WAS an input node (regardless of whether it survived the
    # fold). Lets us distinguish "endpoint dropped during fold" (hard-drop) from
    # "endpoint was never a node" (federation-by-convention pass-through).
    input_ids = {n.get("id") for n in nodes_in if isinstance(n, dict) and isinstance(n.get("id"), str)}

    def _edge_type(edge: Dict[str, Any]) -> Any:
        return edge.get("type", edge.get("relation_type"))

    def _edge_conf(edge: Dict[str, Any]) -> float:
        c = edge.get("confidence")
        try:
            return float(c)
        except (TypeError, ValueError):
            return 0.0

    external_passthrough = 0
    best_edge: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    edge_order: List[Tuple[Any, Any, Any]] = []
    for edge in edges_in:
        if not isinstance(edge, dict):
            continue
        s = final_winner.get(edge.get("source"), edge.get("source"))
        t = final_winner.get(edge.get("target"), edge.get("target"))
        if s == t:
            continue  # self-loop
        if s not in valid_ids or t not in valid_ids:
            # An endpoint isn't a surviving node. Two cases:
            #   (a) it WAS an input node that got folded into nothing — a hard
            #       guard so we never emit a genuinely dangling edge.
            #   (b) it was NEVER an input node (federation-by-convention
            #       endpoint: an LO-authored targets-concept phantom target, a
            #       chunk/LO/misconception ID, etc.) — pass it through with the
            #       re-pointed endpoints. Input-dangling edges are the
            #       materializer's / orphan-gate's problem, not the merge pass's.
            s_dropped = s not in valid_ids and s in input_ids
            t_dropped = t not in valid_ids and t in input_ids
            if s_dropped or t_dropped:
                continue
            external_passthrough += 1
        new_edge = dict(edge)
        new_edge["source"] = s
        new_edge["target"] = t
        key = (s, t, _edge_type(new_edge))
        prev = best_edge.get(key)
        if prev is None:
            best_edge[key] = new_edge
            edge_order.append(key)
        elif _edge_conf(new_edge) > _edge_conf(prev):
            best_edge[key] = new_edge

    new_edges = [best_edge[k] for k in edge_order]

    result = dict(graph)
    result["nodes"] = new_nodes
    result["edges"] = new_edges
    result["concept_node_merges"] = len(losers)
    result["external_endpoint_edges_preserved"] = external_passthrough
    return result
