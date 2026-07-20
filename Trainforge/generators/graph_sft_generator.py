"""Concept-graph -> SFT pair generator (SFT data program Phase 2 / S5).

Deterministic, LLM-free emitter that turns ``concept_graph_semantic.json`` +
its prerequisite DAG into open-book, verbalized instruction pairs for the
course-pinned 1.5B LoRA adapter.

Design contract (``scratchpad/sft_data_program.md`` §A rows 8-10 + §D-Phase-2):

* **Open-book, verbalized.** Every pair is grounded in the graph frame itself
  (the concept labels + typed edges supplied in the prompt), never free-form
  traversal from weights — memorizing arbitrary string mappings goes stale on
  re-slice.  Three families:

  1. ``relation_qa``          — "how does X relate to Y" over a typed edge.
  2. ``prereq_study_path``    — 1-2 hop study sequence over prerequisite edges.
  3. ``concept_verbalization``— present a concept from its graph neighborhood.

* **Consensus-filtered.** Edges whose ``edge_status`` is ``contradicted`` /
  ``retracted`` (stamped by ``lib/aggregators/edge_consensus.py``) are excluded
  BEFORE any pair is built — the low-consensus / contradiction-flagged edges
  never become training signal.

* **Graph-frame sampling.** Every surviving node AND edge yields >= 1 pair;
  emit order is INVERSE-degree weighted (lowest-degree first) so a global
  ``max_pairs`` cap covers long-tail concepts before graph hubs.

* **Navigation cap.** "which week covers X" style pairs are capped at <= 2%
  of the emitted total (arbitrary, re-slice-fragile string mappings — sourced
  from retrieval at serve time, not memorized).

* **Holdout-reduced graph.** The caller feeds this generator the
  holdout-REDUCED concept graph (withheld edges removed via the pedagogy-graph
  holdout split) so a withheld edge can never surface in a pair — the design
  rule for ALL graph->pair generators (SFT data program S4).

* **Per-pair provenance** (§B): ``generation_method`` (``deterministic_template``),
  ``generating_seat`` + ``seat_license``, ``verifier_results``
  (``edge_consensus_status``), ``source_chunk_ids``, ``template_id``, ``seed``,
  ``decision_capture_id``, ``holdout_safe``, ``decontam_checked``.

This module NEVER writes files — it yields pair dicts the Trainforge synthesis
runner appends to ``instruction_pairs.jsonl`` (staying inside the
``instruction_pairs_hash`` provenance chain).
"""
from __future__ import annotations

import logging
import re
import zlib
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# instruction_pair.schema.json floors/ceilings (prompt 40-400, completion
# 50-600). A pair that cannot meet a floor is SKIPPED — never padded.
_PROMPT_MIN, _PROMPT_MAX = 40, 400
_COMPLETION_MIN, _COMPLETION_MAX = 50, 600

_CONTENT_TYPE = "graph_sft"
_PROVIDER = "local"

# Edge-status verdicts that exclude an edge from ANY pair (consensus filter).
_EXCLUDED_EDGE_STATUS: Set[str] = {"contradicted", "retracted"}

# Emit-order family list (deterministic).
_FAMILIES: Tuple[str, ...] = (
    "relation_qa",
    "prereq_study_path",
    "concept_verbalization",
)

# Typed-edge -> natural-language phrase (edge source ROLE first). Direction
# matters for prerequisite: source=dependent, target=prerequisite (matches
# prerequisite_from_lo_order + prereq_sequencer).
_REL_PHRASE: Dict[str, str] = {
    "prerequisite": "has as a prerequisite",
    "is-a": "is a kind of",
    "related-to": "is related to",
    "assesses": "assesses",
    "exemplifies": "is exemplified by",
    "misconception-of": "is a common misconception of",
    "derived-from-objective": "is derived from the objective",
    "defined-by": "is defined by",
    "targets-concept": "targets the concept",
    "broader-than": "is broader than",
    "narrower-than": "is narrower than",
    "corrected-by-chunk": "is corrected by",
    "detected-by-question": "is detected by the question",
    "interferes-with-outcome": "interferes with the outcome",
}

_WS_RE = re.compile(r"\s+")


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _seed_for(key: str) -> int:
    return int(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF)


def _bound(text: str, lo: int, hi: int) -> Optional[str]:
    t = _norm_text(text)
    if len(t) < lo:
        return None
    if len(t) > hi:
        cut = t[:hi]
        sp = cut.rfind(" ")
        if sp >= lo:
            cut = cut[:sp]
        t = cut.rstrip()
    return t if len(t) >= lo else None


def _rel_phrase(edge_type: str) -> str:
    return _REL_PHRASE.get(edge_type, f"is connected ('{edge_type}') to")


# --------------------------------------------------------------------------- #
# Graph parsing
# --------------------------------------------------------------------------- #

def _node_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in graph.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if nid:
            out[str(nid)] = n
    return out


def _label_for(node: Optional[Dict[str, Any]], node_id: str) -> str:
    if isinstance(node, dict):
        lbl = _norm_text(node.get("label"))
        if lbl:
            return lbl
    # Fall back to a de-slugged id (never the raw chunk-ish literal).
    return _norm_text(str(node_id).replace("-", " ").replace("_", " ")) or str(node_id)


def _surviving_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Consensus-filtered edge list: drop contradicted / retracted edges and
    edges missing an endpoint. Deterministically ordered."""
    out: List[Dict[str, Any]] = []
    for e in graph.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        s = e.get("source")
        t = e.get("target")
        et = e.get("type") or e.get("relation_type")
        if not (s and t and et):
            continue
        if str(e.get("edge_status") or "").strip().lower() in _EXCLUDED_EDGE_STATUS:
            continue
        out.append(e)
    out.sort(key=lambda e: (
        str(e.get("source")),
        str(e.get("target")),
        str(e.get("type") or e.get("relation_type")),
    ))
    return out


def _degree(edges: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for e in edges:
        for ep in (str(e.get("source")), str(e.get("target"))):
            deg[ep] = deg.get(ep, 0) + 1
    return deg


def _edge_source_chunks(edge: Dict[str, Any], node_index: Dict[str, Dict[str, Any]]) -> List[str]:
    """Best-effort chunk-evidence for an edge: the source node's occurrences[]."""
    src = str(edge.get("source"))
    node = node_index.get(src)
    if not isinstance(node, dict):
        return []
    occ = node.get("occurrences")
    out: List[str] = []
    seen: Set[str] = set()
    for c in occ or []:
        cid = str(c).strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out[:5]


# --------------------------------------------------------------------------- #
# Per-family builders — return (prompt, completion) or None
# --------------------------------------------------------------------------- #

def _fmt_relation_qa(
    edge: Dict[str, Any], node_index: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    src = str(edge.get("source"))
    tgt = str(edge.get("target"))
    et = str(edge.get("type") or edge.get("relation_type"))
    sl = _label_for(node_index.get(src), src)
    tl = _label_for(node_index.get(tgt), tgt)
    if not sl or not tl or sl == tl:
        return None
    prompt = (
        "Using the course concept map, explain how the concept "
        f"'{sl}' relates to '{tl}'."
    )
    completion = (
        f"In this course, '{sl}' {_rel_phrase(et)} '{tl}'. This relationship "
        f"is part of the course's concept structure and connects the two topics."
    )
    return prompt, completion


def _fmt_prereq_path(
    node_id: str,
    prereq_adj: Dict[str, List[str]],
    node_index: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    """1-2 hop study path over prerequisite edges (source depends on target)."""
    first = prereq_adj.get(node_id) or []
    if not first:
        return None
    p1 = first[0]
    label = _label_for(node_index.get(node_id), node_id)
    p1l = _label_for(node_index.get(p1), p1)
    # Try a second hop (prerequisite of the prerequisite).
    second = prereq_adj.get(p1) or []
    p2 = next((x for x in second if x not in (node_id, p1)), None)
    if p2:
        p2l = _label_for(node_index.get(p2), p2)
        completion = (
            f"To learn '{label}', follow the prerequisite path in the course: "
            f"first study '{p2l}', then '{p1l}', and finally '{label}'."
        )
    else:
        completion = (
            f"To learn '{label}', first study its prerequisite '{p1l}', then "
            f"proceed to '{label}'."
        )
    prompt = (
        "Based on the course prerequisite structure, what should a learner "
        f"study before '{label}', and in what order?"
    )
    return prompt, completion


def _fmt_concept_verbalization(
    node_id: str,
    neighbors: Sequence[Tuple[str, str]],
    node_index: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    label = _label_for(node_index.get(node_id), node_id)
    if not label:
        return None
    if neighbors:
        rel_bits = [f"{_rel_phrase(et)} '{nl}'" for et, nl in neighbors[:3]]
        body = "; it " + "; ".join(rel_bits)
    else:
        body = " and is a distinct topic in the course's concept structure"
    completion = (
        f"'{label}' is a concept covered in this course{body}."
    )
    prompt = (
        "Using the course concept map, describe the concept "
        f"'{label}' and how it fits into the course."
    )
    return prompt, completion


# --------------------------------------------------------------------------- #
# Pair assembly
# --------------------------------------------------------------------------- #

def _build_pair(
    *,
    family: str,
    prompt: str,
    completion: str,
    seed_key: str,
    decision_capture_id: str,
    generating_seat: str,
    seat_license: str,
    source_chunk_ids: Sequence[str],
    lo_ref: str,
    bloom_level: str,
    edge_status: Optional[str],
) -> Optional[Dict[str, Any]]:
    bp = _bound(prompt, _PROMPT_MIN, _PROMPT_MAX)
    bc = _bound(completion, _COMPLETION_MIN, _COMPLETION_MAX)
    if bp is None or bc is None:
        return None
    chunk_ids = [str(c) for c in source_chunk_ids if str(c).strip()]
    anchor = chunk_ids[0] if chunk_ids else f"graph:{lo_ref}"
    return {
        "prompt": bp,
        "completion": bc,
        "chunk_id": anchor,
        "lo_refs": [lo_ref],
        "bloom_level": bloom_level,
        "content_type": _CONTENT_TYPE,
        "seed": _seed_for(seed_key),
        "decision_capture_id": decision_capture_id,
        "template_id": f"graph_sft.{family}",
        "provider": _PROVIDER,
        "schema_version": "v1",
        "requires_source_citation": False,
        # Per-pair provenance (§B).
        "generation_method": "deterministic_template",
        "generating_seat": generating_seat,
        "seat_license": seat_license,
        "verifier_results": {"edge_consensus_status": edge_status or "unfiltered"},
        "source_chunk_ids": list(chunk_ids),
        "holdout_safe": True,
        "decontam_checked": False,
        "pair_format": family,
        "question_type": "short_answer",
    }


def _log_batch(capture: Any, family: str, rationale: str) -> str:
    capture.log_decision(
        decision_type="kg_metadata_generation",
        decision=f"graph_sft_batch:{family}",
        rationale=rationale,
    )
    return _last_event_id(capture)


def _last_event_id(capture: Any) -> str:
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        raise RuntimeError(
            "graph_sft_generator: capture has no logged decisions; log a batch "
            "decision before anchoring a pair's decision_capture_id."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def generate_graph_sft_pairs(
    concept_graph: Dict[str, Any],
    capture: Any,
    *,
    max_pairs: Optional[int] = None,
    seed: int = 19,
    generating_seat: str = "local",
    seat_license: str = "deterministic-template",
    navigation_items: Optional[Sequence[Tuple[str, str]]] = None,
    navigation_cap_fraction: float = 0.02,
) -> Iterator[Dict[str, Any]]:
    """Yield concept-graph-derived SFT instruction pairs.

    Args:
        concept_graph: The holdout-REDUCED ``concept_graph_semantic.json``
            payload (dict with ``nodes`` + ``edges``). The caller is
            responsible for removing withheld edges (S4 design rule).
        capture: A ``DecisionCapture``-shaped object (``log_decision`` +
            ``decisions`` list). **Required** — one ``kg_metadata_generation``
            event fires per family batch and anchors each pair's
            ``decision_capture_id``.
        max_pairs: Optional global cap. Emit order is inverse-degree weighted
            so a cap covers long-tail concepts first.
        seed: Deterministic seed base (folded into each pair's ``seed``).
        generating_seat / seat_license: Provenance tags per pair.
        navigation_items: OPTIONAL ``(concept_label, location_label)`` pairs
            for the capped navigation family. Omitted -> no navigation pairs.
        navigation_cap_fraction: Hard ceiling on navigation share (default 2%).

    Yields:
        Instruction-pair dicts (``instruction_pair.schema.json`` shape +
        additive provenance). Deterministic: same graph -> same pairs/order.
    """
    if capture is None:
        raise ValueError(
            "graph_sft_generator requires a DecisionCapture (got None); it logs "
            "one kg_metadata_generation event per family batch and anchors each "
            "pair's decision_capture_id from it."
        )
    if not isinstance(concept_graph, dict):
        raise TypeError("concept_graph must be a dict")

    node_index = _node_index(concept_graph)
    edges = _surviving_edges(concept_graph)
    deg = _degree(edges)

    # Prerequisite adjacency (source depends on target).
    prereq_adj: Dict[str, List[str]] = {}
    neighbors_by_node: Dict[str, List[Tuple[str, str]]] = {}
    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        et = str(e.get("type") or e.get("relation_type"))
        if et == "prerequisite":
            prereq_adj.setdefault(s, []).append(t)
        tl = _label_for(node_index.get(t), t)
        neighbors_by_node.setdefault(s, []).append((et, tl))

    # Deterministic emit order: inverse-degree (lowest first), then id.
    node_ids = sorted(node_index.keys(), key=lambda nid: (deg.get(nid, 0), nid))
    # edges already sorted; re-order by min-endpoint-degree ascending.
    ordered_edges = sorted(
        edges,
        key=lambda e: (
            min(deg.get(str(e.get("source")), 0), deg.get(str(e.get("target")), 0)),
            str(e.get("source")),
            str(e.get("target")),
            str(e.get("type") or e.get("relation_type")),
        ),
    )

    total_emitted = 0
    emitted_by_family: Dict[str, int] = {f: 0 for f in _FAMILIES}
    nav_emitted = 0

    def _capped() -> bool:
        return max_pairs is not None and total_emitted >= max_pairs

    # --- Batch decisions (anchor each family's pairs). --------------------- #
    rel_event = _log_batch(
        capture, "relation_qa",
        rationale=(
            f"Deterministic concept-graph->SFT relation_qa batch: "
            f"{len(edges)} consensus-surviving edges, {len(node_index)} nodes, "
            f"max_pairs={max_pairs}, seat={generating_seat}/{seat_license}. "
            f"Contradicted/retracted edges excluded; open-book verbalized."
        ),
    )
    prereq_event = _log_batch(
        capture, "prereq_study_path",
        rationale=(
            f"Deterministic concept-graph->SFT prereq_study_path batch: "
            f"{len(prereq_adj)} nodes with prerequisites, 1-2 hop paths over the "
            f"prerequisite DAG, max_pairs={max_pairs}, seed={seed}."
        ),
    )
    verbal_event = _log_batch(
        capture, "concept_verbalization",
        rationale=(
            f"Deterministic concept-graph->SFT concept_verbalization batch: "
            f"{len(node_index)} nodes, inverse-degree weighted so long-tail "
            f"concepts are covered first under max_pairs={max_pairs}."
        ),
    )

    # --- 1. relation_qa: one pair per surviving edge. --------------------- #
    for e in ordered_edges:
        if _capped():
            return
        built = _fmt_relation_qa(e, node_index)
        if not built:
            continue
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        et = str(e.get("type") or e.get("relation_type"))
        pair = _build_pair(
            family="relation_qa",
            prompt=built[0], completion=built[1],
            seed_key=f"{seed}:relation_qa:{src}:{tgt}:{et}",
            decision_capture_id=rel_event,
            generating_seat=generating_seat, seat_license=seat_license,
            source_chunk_ids=_edge_source_chunks(e, node_index),
            lo_ref="graph", bloom_level="understand",
            edge_status=str(e.get("edge_status") or "") or None,
        )
        if pair is None:
            continue
        emitted_by_family["relation_qa"] += 1
        total_emitted += 1
        yield pair

    # --- 2. prereq_study_path: one path per node with a prerequisite. ----- #
    for nid in sorted(prereq_adj.keys(), key=lambda n: (deg.get(n, 0), n)):
        if _capped():
            return
        # keep the adjacency deterministic
        prereq_adj[nid] = sorted(set(prereq_adj[nid]))
    for nid in sorted(prereq_adj.keys(), key=lambda n: (deg.get(n, 0), n)):
        if _capped():
            return
        built = _fmt_prereq_path(nid, prereq_adj, node_index)
        if not built:
            continue
        pair = _build_pair(
            family="prereq_study_path",
            prompt=built[0], completion=built[1],
            seed_key=f"{seed}:prereq:{nid}",
            decision_capture_id=prereq_event,
            generating_seat=generating_seat, seat_license=seat_license,
            source_chunk_ids=_edge_source_chunks(
                {"source": nid}, node_index,
            ),
            lo_ref="graph", bloom_level="apply",
            edge_status=None,
        )
        if pair is None:
            continue
        emitted_by_family["prereq_study_path"] += 1
        total_emitted += 1
        yield pair

    # --- 3. concept_verbalization: one pair per node (long-tail first). --- #
    for nid in node_ids:
        if _capped():
            return
        neighbors = neighbors_by_node.get(nid, [])
        built = _fmt_concept_verbalization(nid, neighbors, node_index)
        if not built:
            continue
        pair = _build_pair(
            family="concept_verbalization",
            prompt=built[0], completion=built[1],
            seed_key=f"{seed}:verbalize:{nid}",
            decision_capture_id=verbal_event,
            generating_seat=generating_seat, seat_license=seat_license,
            source_chunk_ids=[str(c) for c in (node_index.get(nid, {}).get("occurrences") or [])][:5],
            lo_ref="graph", bloom_level="understand",
            edge_status=None,
        )
        if pair is None:
            continue
        emitted_by_family["concept_verbalization"] += 1
        total_emitted += 1
        yield pair

    # --- 4. navigation: capped at <= navigation_cap_fraction of total. ---- #
    if navigation_items:
        # Cap is a fraction of the (already emitted) non-navigation total.
        base = max(total_emitted, 1)
        nav_cap = int(base * max(0.0, navigation_cap_fraction))
        nav_event = _log_batch(
            capture, "navigation",
            rationale=(
                f"Deterministic concept-graph->SFT navigation batch capped at "
                f"{navigation_cap_fraction:.0%} ({nav_cap} of {total_emitted} "
                f"emitted): re-slice-fragile 'which week covers X' mappings, "
                f"open-book (location supplied), never memorized traversal."
            ),
        )
        for concept_label, location_label in navigation_items:
            if nav_emitted >= nav_cap or _capped():
                break
            cl = _norm_text(concept_label)
            ll = _norm_text(location_label)
            if not cl or not ll:
                continue
            prompt = f"According to the course structure, where is '{cl}' covered?"
            completion = (
                f"In this course, '{cl}' is covered in {ll}. Consult the course "
                f"outline for the exact section."
            )
            pair = _build_pair(
                family="navigation",
                prompt=prompt, completion=completion,
                seed_key=f"{seed}:navigation:{cl}:{ll}",
                decision_capture_id=nav_event,
                generating_seat=generating_seat, seat_license=seat_license,
                source_chunk_ids=[], lo_ref="graph", bloom_level="remember",
                edge_status=None,
            )
            if pair is None:
                continue
            nav_emitted += 1
            total_emitted += 1
            yield pair


__all__ = ["generate_graph_sft_pairs"]
