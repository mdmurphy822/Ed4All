"""W3.1 — content-dependency prerequisite edges (deterministic, no-LLM).

Today every ``prerequisite`` edge in ``concept_graph_semantic.json`` is
produced by ``Trainforge/rag/inference_rules/prerequisite_from_lo_order.py``
(``edge_kind`` = ``inferred``), which reads *learning-outcome ORDER*. Because
the LO order is itself derived from the authored objective sequence,
``lib/generation/prereq_sequencer.py`` self-documents that the resulting
topo-sort "largely reproduces the existing order". There is no CONTENT signal.

This module adds one: a deterministic rule that emits a ``TO -> TO``
prerequisite edge when a concept is DEFINED in one terminal objective's
chunks and then appears ASSUMED-BUT-UNDEFINED in another terminal objective's
chunks. The definitional anchor reuses the same substrate as
``defined_by_from_first_mention`` — a concept node's ``occurrences[0]`` (first
mention by ascending chunk-ID sort) is its definition chunk. If that chunk
belongs to ``TO_a`` (via its ``learning_outcome_refs``) and the concept later
surfaces in a chunk owned by a *different* ``TO_b`` that does NOT define it,
then ``TO_b`` depends on ``TO_a`` — you must teach ``TO_a`` first.

The emitted edge is a federation-by-convention edge (``source`` / ``target``
are TO ids, per the ``inference_rules`` package docstring, which explicitly
permits LO ids as endpoints). ``source`` is the dependent (``TO_b``);
``target`` is the prerequisite (``TO_a``) — matching the direction convention
of ``prerequisite_from_lo_order`` (``B --prerequisite--> A``).

Anti-fabrication contract
-------------------------
  * Only REAL ids are ever emitted — concept slugs that are actual graph
    nodes, chunk ids that are actual ``occurrences[]`` entries, and TO ids
    that are actual ``learning_outcome_refs`` (canonical ``TO-NN`` form).
  * The definition anchor (``occurrences[0]``) guards against term-collision:
    a concept is treated as "defined" only at its single first-mention chunk;
    every other appearance is an "assumed" mention. Two distinct concepts
    never collapse because concept nodes are keyed by canonical slug and carry
    their own ``occurrences[]``.
  * A concept whose definition chunk carries no terminal-objective ref is
    SKIPPED (we cannot attribute a defining TO — never guessed).

Default OFF — :func:`resolve_prereq_definition_mention` gates the rule
(``TRAINFORGE_PREREQ_DEFINITION_MENTION``). When off, :func:`infer` returns an
empty list, so a caller that leaves the rule unregistered (or registers it but
gated on the flag) regenerates ``concept_graph_semantic.json`` — and its
``concept_graph_sha256`` — byte-identically.

Contract for downstream consumers (Lanes K / V)
-----------------------------------------------
  * Edge-kind constant: register ``RULE_NAME`` as
    ``lib.ontology.edge_kind.EDGE_KIND_INFERRED`` (done in this wave). Every
    emitted edge is stamped ``edge_kind="inferred"`` by the orchestrator's
    ``_stamp_edge_kind`` in lockstep with the other inference rules.
  * Module API: :func:`infer` (standard
    ``(chunks, course, concept_graph, **kwargs)`` rule signature, plus an
    optional ``terminal_objectives=`` kwarg to pin the canonical TO id form)
    and :func:`resolve_prereq_definition_mention`.
  * Flag: ``TRAINFORGE_PREREQ_DEFINITION_MENTION`` (default OFF). The dispatch
    registration MUST be gated on this flag so the graph's ``rule_versions``
    map is unchanged when off (an unconditional registration would add a
    ``rule_versions`` key even on a zero-edge run, changing the sha).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)

ENV_PREREQ_DEFINITION_MENTION = "TRAINFORGE_PREREQ_DEFINITION_MENTION"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

RULE_NAME = "prerequisite_from_definition_mention"
RULE_VERSION = 1
EDGE_TYPE = "prerequisite"

#: Confidence for a definition-mention prerequisite edge. Mirrors the
#: ``defined_by_from_first_mention`` 0.7 (first-mention-by-chunk-ID-sort is a
#: stable proxy for the pedagogical definition, but not a certainty).
_EDGE_CONFIDENCE = 0.7

__all__ = [
    "ENV_PREREQ_DEFINITION_MENTION",
    "RULE_NAME",
    "RULE_VERSION",
    "EDGE_TYPE",
    "resolve_prereq_definition_mention",
    "infer",
]


def resolve_prereq_definition_mention(value: object = None) -> bool:
    """Return True iff the definition-mention prereq rule is enabled.

    ``value`` overrides the env when not ``None``. Falsey / garbage / unset →
    False (parse-with-fallback, mirroring the other ``TRAINFORGE_*`` toggles).
    """
    if value is None:
        raw = os.environ.get(ENV_PREREQ_DEFINITION_MENTION, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Concept / LO normalization
# ---------------------------------------------------------------------------


def _norm_concept(raw: Any) -> str:
    """Normalize a concept id / tag / slug to a canonical comparison key.

    Concept-graph node ids may be a flat slug or ``{course_id}:{slug}`` (Worker
    O scoped-id mode); chunk ``concept_tags`` are bare slugs. Strip a single
    ``:`` namespace prefix then ``canonical_slug`` so both surfaces compare
    equal — this avoids a ``lib -> Trainforge`` import for ``_make_concept_id``.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    s = raw.split(":", 1)[1] if ":" in raw else raw
    return canonical_slug(s)


def _lo_key(raw: Any) -> str:
    """Uppercase-normalize an LO ref for case-insensitive TO matching.

    Chunks store refs lowercased (see ``process_course._extract_objective_refs``)
    while synthesized ``terminal_objectives`` ids are canonical uppercase
    ``TO-NN``. Uppercasing both sides makes them compare equal.
    """
    return raw.strip().upper() if isinstance(raw, str) else ""


def _is_terminal_lo(ref: str) -> bool:
    """Return True iff ``ref`` is a canonical TERMINAL-objective id (``TO-NN``).

    Uses the single source of truth for LO identity. Non-canonical /
    unrecognized-prefix ids degrade to False (never raises).
    """
    from lib.ontology.learning_objectives import hierarchy_from_id, validate_lo_id

    key = _lo_key(ref)
    if not validate_lo_id(key):
        return False
    try:
        return hierarchy_from_id(key) == "terminal"
    except ValueError:
        return False


def _build_to_resolver(
    terminal_objectives: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, str], bool]:
    """Return ``({lo_key: canonical_to_id}, have_explicit_set)``.

    When ``terminal_objectives`` is provided, the map pins the CANONICAL id
    form (so emitted edges match the sequencer's ``terminal_objectives`` ids
    exactly). Otherwise the map is empty and the caller falls back to
    :func:`_is_terminal_lo` pattern detection (emitting the uppercased ref).
    """
    resolver: Dict[str, str] = {}
    if not terminal_objectives:
        return resolver, False
    for to in terminal_objectives:
        if not isinstance(to, dict):
            continue
        tid = to.get("id")
        if isinstance(tid, str) and tid.strip():
            resolver.setdefault(_lo_key(tid), tid.strip())
    return resolver, bool(resolver)


def _resolve_to_id(
    ref: Any,
    resolver: Dict[str, str],
    have_explicit_set: bool,
) -> Optional[str]:
    """Map a raw chunk LO ref to its canonical TO id, or None if not a TO.

    * With an explicit ``terminal_objectives`` set: the ref must resolve to a
      known TO id (uppercase-keyed) — anything else (a CO ref, a stray id) is
      dropped.
    * Without one: any canonical ``TO-NN`` ref passes (returned uppercased).
    """
    if not isinstance(ref, str) or not ref.strip():
        return None
    key = _lo_key(ref)
    if have_explicit_set:
        return resolver.get(key)
    return key if _is_terminal_lo(ref) else None


def _definition_chunk_by_concept(
    concept_graph: Any,
) -> Dict[str, Tuple[str, str]]:
    """Map ``norm_slug -> (definition_chunk_id, display_slug)``.

    The definition chunk is ``occurrences[0]`` (first mention by ascending
    chunk-ID sort — the same anchor ``defined_by_from_first_mention`` uses).
    Nodes without ``occurrences`` produce no entry.
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not isinstance(concept_graph, dict):
        return out
    for node in concept_graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        occurrences = node.get("occurrences") or []
        real = [o for o in occurrences if isinstance(o, str) and o]
        if not real:
            continue
        key = _norm_concept(node_id)
        if not key or key in out:
            continue
        first_chunk = sorted(real)[0]
        # Display slug: strip a single namespace prefix, keep the raw slug.
        display = node_id.split(":", 1)[1] if ":" in node_id else node_id
        out[key] = (first_chunk, display)
    return out


# ---------------------------------------------------------------------------
# Public rule entrypoint
# ---------------------------------------------------------------------------


def infer(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]] = None,
    concept_graph: Optional[Dict[str, Any]] = None,
    *,
    terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    capture: Any = None,
    course_code: str = "",
    enabled: object = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``TO -> TO`` prerequisite edges from the definition-mention signal.

    Args:
        chunks: Pipeline chunk dicts carrying ``id``, ``concept_tags`` and
            ``learning_outcome_refs``.
        course: Unused for ownership (interface parity with the other rules).
        concept_graph: The concept graph; nodes supply ``occurrences[]`` (the
            definition anchor). Only concepts that are graph nodes are eligible.
        terminal_objectives: Optional — pins the canonical TO id form. When
            omitted, TO ids are detected from the ``TO-NN`` LO pattern.
        capture: Optional ``DecisionCapture`` — one ``typed_edge_inference``
            event is logged per invocation that produces edges.
        course_code: Course slug for the decision rationale.
        enabled: Optional flag override for tests; defaults to the env.

    Returns:
        A deterministically-ordered list of edge dicts (empty when the flag is
        off, or when the signal produces nothing). NEVER raises — any error
        degrades to an empty list (best-effort, never breaks the graph build).
    """
    del course  # unused; interface parity
    if not resolve_prereq_definition_mention(enabled):
        return []

    try:
        return _infer_impl(
            chunks or [],
            concept_graph,
            terminal_objectives=terminal_objectives,
            capture=capture,
            course_code=course_code,
        )
    except Exception as exc:  # noqa: BLE001 — never break the graph build
        logger.warning("prerequisite_from_definition_mention failed: %s", exc)
        return []


def _infer_impl(
    chunks: List[Dict[str, Any]],
    concept_graph: Optional[Dict[str, Any]],
    *,
    terminal_objectives: Optional[List[Dict[str, Any]]],
    capture: Any,
    course_code: str,
) -> List[Dict[str, Any]]:
    def_by_concept = _definition_chunk_by_concept(concept_graph)
    if not def_by_concept:
        return []

    resolver, have_explicit = _build_to_resolver(terminal_objectives)

    # Index chunks: id -> (normalized concept_tag set, resolved-TO-id set).
    chunk_index: Dict[str, Tuple[frozenset, frozenset]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("id")
        if not isinstance(cid, str) or not cid:
            continue
        tags = {
            _norm_concept(t)
            for t in (chunk.get("concept_tags") or [])
            if isinstance(t, str)
        }
        tags.discard("")
        tos = set()
        for ref in chunk.get("learning_outcome_refs") or []:
            resolved = _resolve_to_id(ref, resolver, have_explicit)
            if resolved:
                tos.add(resolved)
        chunk_index[cid] = (frozenset(tags), frozenset(tos))

    # Reverse index: normalized concept -> set of TO ids that mention it.
    mention_tos_by_concept: Dict[str, set] = {}
    for _cid, (tags, tos) in chunk_index.items():
        if not tos:
            continue
        for tag in tags:
            if tag in def_by_concept:
                mention_tos_by_concept.setdefault(tag, set()).update(tos)

    # Build (TO_b dependent, TO_a prereq) -> evidence, deduped, deterministic.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    concepts_scanned = 0
    concepts_bridged = 0
    for concept, (def_chunk, display) in sorted(def_by_concept.items()):
        entry = chunk_index.get(def_chunk)
        if entry is None:
            continue  # definition chunk not in the chunk set — cannot attribute
        _def_tags, defining_tos = entry
        if not defining_tos:
            continue  # no terminal-objective owns the definition — skip
        concepts_scanned += 1
        mention_tos = mention_tos_by_concept.get(concept, set())
        # Assuming TOs = TOs that mention the concept but do NOT define it.
        assuming_tos = mention_tos - defining_tos
        if not assuming_tos:
            continue
        bridged = False
        for to_b in sorted(assuming_tos):  # dependent
            for to_a in sorted(defining_tos):  # prerequisite
                if to_a == to_b:
                    continue
                key = (to_b, to_a)  # source=dependent, target=prereq
                if key not in best:
                    best[key] = {
                        "bridge_concept": display,
                        "bridge_concept_slug": concept,
                        "definition_chunk": def_chunk,
                        "defining_lo": to_a,
                        "assuming_lo": to_b,
                        "bridge_concept_count": 1,
                    }
                    bridged = True
                else:
                    best[key]["bridge_concept_count"] += 1
        if bridged:
            concepts_bridged += 1

    edges: List[Dict[str, Any]] = []
    for (source, target) in sorted(best.keys()):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": EDGE_TYPE,
                "confidence": _EDGE_CONFIDENCE,
                "provenance": {
                    "rule": RULE_NAME,
                    "rule_version": RULE_VERSION,
                    "evidence": best[(source, target)],
                },
            }
        )

    _emit(
        capture,
        course_code,
        concepts_scanned=concepts_scanned,
        concepts_bridged=concepts_bridged,
        n_edges=len(edges),
        n_concepts_total=len(def_by_concept),
    )
    return edges


def _emit(
    capture: Any,
    course_code: str,
    *,
    concepts_scanned: int,
    concepts_bridged: int,
    n_edges: int,
    n_concepts_total: int,
) -> None:
    if capture is None or n_edges == 0:
        return
    try:
        capture.log_decision(
            decision_type="typed_edge_inference",
            decision=(
                f"{RULE_NAME}: emitted {n_edges} TO->TO prerequisite edge(s)"
            ),
            rationale=(
                f"Content-dependency prerequisite rule for course "
                f"{course_code or '<unknown>'}: scanned {n_concepts_total} "
                f"concept node(s) with a definition anchor, {concepts_scanned} "
                f"had a terminal-objective-owned definition, {concepts_bridged} "
                f"bridged a defining TO to an assuming (uses-but-undefines) TO, "
                f"yielding {n_edges} deterministic TO->TO prerequisite edge(s) "
                f"(confidence {_EDGE_CONFIDENCE}). Anti-fabrication: only real "
                f"concept slugs, occurrences[0] definition chunks, and "
                f"canonical TO ids are emitted."
            ),
            ml_features={
                "concepts_total": int(n_concepts_total),
                "concepts_scanned": int(concepts_scanned),
                "concepts_bridged": int(concepts_bridged),
                "edges_emitted": int(n_edges),
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("%s: decision capture raised: %s", RULE_NAME, exc)
