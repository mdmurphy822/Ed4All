"""Orchestrator for typed-edge concept-graph inference.

Consumes:
    - chunks (list of chunk dicts, same shape as ``chunks.jsonl`` entries)
    - course (``course.json`` dict — optional; needed for prerequisite rule)
    - concept_graph (``concept_graph.json`` dict — the co-occurrence base)

Emits:
    - A dict matching ``schemas/knowledge/concept_graph_semantic.schema.json``:
        {"kind": "concept_semantic", "nodes": [...], "edges": [...], ...}

Precedence on (source, target) collisions: ``is-a`` > ``prerequisite`` > ``related-to``.
The lower-precedence edge is dropped; the kept edge's provenance is unchanged.

LLM escalation: OFF by default. When ``llm_enabled=True``, an optional pass
can propose additional edges for "uncertain" pairs (pairs that appear in the
co-occurrence graph but no rule assigned a type). The LLM path is gated
behind an injected callable so that unit tests can stub it out and so that
the default runtime is fully deterministic — no LLM call means byte-identical
output across runs.

Decision capture: when the LLM path fires, each inferred edge logs a
``typed_edge_inference`` decision via ``lib.decision_capture``.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401

from Trainforge.rag.inference_rules import assesses_from_question_lo as _assesses_mod
from Trainforge.rag.inference_rules import (
    corrected_by_from_chunk_misconception as _corrected_by_chunk_mod,
)
from Trainforge.rag.inference_rules import defined_by_from_first_mention as _defined_by_mod
from Trainforge.rag.inference_rules import derived_from_lo_ref as _derived_lo_mod
from Trainforge.rag.inference_rules import (
    detected_by_from_distractor_misconception_id as _detected_by_question_mod,
)
from Trainforge.rag.inference_rules import exemplifies_from_example_chunks as _exemplifies_mod
from Trainforge.rag.inference_rules import (
    RULEPACK_VERSION,
    infer_assesses,
    infer_corrected_by_chunk,
    infer_defined_by,
    infer_derived_from_objective,
    infer_detected_by_question,
    infer_exemplifies,
    infer_interferes_with_outcome,
    infer_is_a,
    infer_misconception_of,
    infer_prerequisite,
    infer_related,
    infer_targets_concept,
)
from Trainforge.rag.inference_rules import (
    interferes_with_outcome_from_misconception_lo as _interferes_with_outcome_mod,
)
from Trainforge.rag.inference_rules import is_a_from_key_terms as _is_a_mod
from Trainforge.rag.inference_rules import (
    misconception_of_from_misconception_ref as _misconception_mod,
)
from Trainforge.rag.inference_rules import prerequisite_from_lo_order as _prereq_mod
from Trainforge.rag.inference_rules import related_from_cooccurrence as _related_mod
from Trainforge.rag.inference_rules import targets_concept_from_lo as _targets_concept_mod
from Trainforge.rag import shacl_rule_runner as _shacl_runner

from lib.generation import (
    prerequisite_from_definition_mention as _prereq_def_mention_mod,
)
from lib.ontology.edge_kind import edge_kind_for_rule

logger = logging.getLogger(__name__)

ARTIFACT_KIND = "concept_semantic"

# Opt-in course-scoped concept IDs.
# When TRAINFORGE_SCOPE_CONCEPT_IDS=true, every concept-node ID is emitted as
# ``f"{course_id}:{slug}"`` instead of the flat slug. Default off → legacy
# behaviour. The flag is captured at import time; tests that need to toggle
# behaviour should monkeypatch ``SCOPE_CONCEPT_IDS`` directly (or
# ``importlib.reload`` this module).
SCOPE_CONCEPT_IDS = os.getenv("TRAINFORGE_SCOPE_CONCEPT_IDS", "").lower() == "true"


def _make_concept_id(slug: str, course_id: Optional[str]) -> str:
    """Return the scoped concept ID when the flag is on, else the flat slug.

    When ``SCOPE_CONCEPT_IDS`` is True and ``course_id`` is truthy, returns
    ``f"{course_id}:{slug}"``. Otherwise returns ``slug`` unchanged. Exposed
    as a module-level helper so rule modules (and the co-occurrence graph
    builder in ``Trainforge.process_course``) can produce node IDs that
    match the graph's scoped namespace.

    Cross-course behaviour: two courses carrying the same concept slug
    produce two distinct scoped IDs — no silent merge.
    """
    if SCOPE_CONCEPT_IDS and course_id:
        return f"{course_id}:{slug}"
    return slug

# Precedence: higher wins. The orchestrator drops lower-precedence edges
# whose (source, target) pair is already claimed by a higher-precedence
# type.
#
# The pedagogical edge types sit at tier 2 (same as ``prerequisite``). In
# practice they don't collide with taxonomic edges because their endpoint
# namespaces differ (concept↔chunk, concept↔LO, chunk↔LO,
# misconception↔concept, question↔LO vs the concept↔concept taxonomic
# edges). Tier 2 assignment is defensive — ties among tier-2 rules break by
# fixed rule-invocation order.
_PRECEDENCE: Dict[str, int] = {
    "is-a": 3,
    # SKOS hierarchy slugs share the top tier with ``is-a``: both
    # express directional taxonomic subsumption between two
    # cf:Concept instances. ``broader-than`` is what
    # ``is_a_from_key_terms`` emits when both endpoints are concept
    # nodes (the canonical case under the W3C-canonical SKOS pattern);
    # ``narrower-than`` is reserved for the inverse-direction
    # emit-side and held at the same tier so a future emitter that
    # produces it doesn't get silently dropped against a
    # ``related-to`` collision.
    "broader-than": 3,
    "narrower-than": 3,
    "assesses": 2,
    # Misconception-anchored materializers, also tier 2. Their endpoint
    # namespaces (misconception↔chunk, misconception↔question,
    # misconception↔LO) don't collide with the concept↔concept taxonomic
    # edges, so the tier assignment is defensive.
    "corrected-by-chunk": 2,
    "defined-by": 2,
    "derived-from-objective": 2,
    "detected-by-question": 2,
    "exemplifies": 2,
    "interferes-with-outcome": 2,
    "misconception-of": 2,
    "prerequisite": 2,
    "targets-concept": 2,
    "related-to": 1,
}

# For ``related-to`` we treat the pair as undirected when deciding
# collisions; ``is-a`` and ``prerequisite`` are directed and only collide on
# exact (source, target).
_UNDIRECTED_TYPES = {"related-to"}


def _key(edge: Dict[str, Any]) -> Tuple[str, str, bool]:
    """Return the collision key. Third element marks undirected edges so the
    dedupe step can canonicalize sorted pairs without losing direction for
    directed edges.
    """
    if edge["type"] in _UNDIRECTED_TYPES:
        a, b = sorted([edge["source"], edge["target"]])
        return (a, b, True)
    return (edge["source"], edge["target"], False)


def _stamp_provenance(
    obj: Dict[str, Any],
    run_id: Optional[str],
    created_at: str,
) -> Dict[str, Any]:
    """Stamp ``run_id`` + ``created_at`` onto a node or edge dict in-place.

    ``run_id`` is omitted when ``None`` so callers that construct graphs
    without a DecisionCapture instance still work. ``created_at`` is always
    stamped — it's produced by the orchestrator, not the rule modules, so
    it's always available.

    Both fields are OPTIONAL per ``concept_graph_semantic.schema.json`` —
    legacy artifacts without them still validate.
    """
    if run_id:
        obj["run_id"] = run_id
    obj["created_at"] = created_at
    return obj


def _stamp_edge_kind(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp ``edge_kind`` (asserted | inferred) onto an edge dict in-place.

    Looks up the edge's ``provenance.rule`` in the canonical
    :mod:`lib.ontology.edge_kind` registry and writes the classification
    onto the edge as a top-level ``edge_kind`` field. Unknown rules
    silently skip the stamp, so legacy graphs validate without the field
    and a future rule landing without a registry update doesn't crash the
    build; ``Trainforge/tests/test_edge_kind_classification.py`` is what
    fails loudly on that drift instead.

    ``edge_kind`` is OPTIONAL per ``concept_graph_semantic.schema.json`` —
    legacy edges without the field validate untouched.
    """
    prov = edge.get("provenance") or {}
    rule = prov.get("rule") if isinstance(prov, dict) else None
    kind = edge_kind_for_rule(rule) if isinstance(rule, str) else None
    if kind is not None:
        edge["edge_kind"] = kind
    return edge


def _apply_precedence(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the precedence policy.

    For each collision key, keep the edge with the highest ``_PRECEDENCE``
    value. Among equal-precedence collisions (shouldn't happen with the
    three built-in rules, but guard against it for extensibility), keep the
    first one the rule list produced — deterministic because rules are
    invoked in a fixed order.
    """
    best: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
    # Also track the "directed" slot for a pair so that when a directed
    # edge fires we can drop any undirected ``related-to`` that would
    # otherwise duplicate semantically.
    directed_pairs: set = set()
    for edge in edges:
        if edge["type"] not in _UNDIRECTED_TYPES:
            directed_pairs.add(tuple(sorted([edge["source"], edge["target"]])))

    for edge in edges:
        key = _key(edge)
        # If an undirected edge matches a pair already claimed by a directed
        # higher-precedence type, drop it.
        if key[2] and (key[0], key[1]) in directed_pairs:
            continue
        prev = best.get(key)
        if prev is None:
            best[key] = edge
            continue
        if _PRECEDENCE[edge["type"]] > _PRECEDENCE[prev["type"]]:
            best[key] = edge

    return sorted(
        best.values(),
        key=lambda e: (e["type"], e["source"], e["target"]),
    )


def _build_nodes(
    concept_graph: Dict[str, Any],
    run_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Copy nodes verbatim from the co-occurrence graph.

    The semantic graph shares its node set with the co-occurrence graph —
    no typed edge can reference a node that didn't already qualify for the
    base graph.

    When ``run_id`` / ``created_at`` are provided, they are stamped onto
    each node so the semantic graph is time- and run-addressable.
    """
    # Defensive backfill — an upstream concept_graph built before classifier
    # wiring landed carries no ``class``, so classify on the fly and keep the
    # field present on every node. Imported lazily so the rule library stays
    # decoupled from lib/ontology at module import time.
    from lib.ontology.concept_classifier import classify_concept

    nodes: List[Dict[str, Any]] = []
    for n in concept_graph.get("nodes", []):
        node = {
            "id": n["id"],
            "label": n.get("label", n["id"]),
            "frequency": n.get("frequency", 0),
        }
        # Carry ``occurrences[]`` from the co-occurrence graph node into the
        # semantic graph node so downstream consumers (e.g. the ``defined-by``
        # rule) don't have to re-derive the inverted index from chunks. The
        # invariant to preserve: ``occurrences[]`` is available on every
        # concept node that has chunks referencing it.
        occurrences = n.get("occurrences")
        if occurrences:
            node["occurrences"] = list(occurrences)
        # Carry ``class`` through so retrieval can filter pedagogical /
        # assessment / low-signal nodes uniformly across both graph
        # artifacts. Backfill via the classifier when the source node
        # lacks the field.
        klass = n.get("class")
        if not klass:
            # Strip course_id prefix when scoping is on so the classifier
            # sees the bare slug it was designed against.
            slug_for_class = node["id"].split(":", 1)[-1]
            klass = classify_concept(slug_for_class, label=n.get("label"))
        node["class"] = klass
        if created_at is not None:
            _stamp_provenance(node, run_id, created_at)
        nodes.append(node)
    return nodes


# --------------------------------------------------------------------------- #
# Endpoint-node materialization (typed-endpoint contract — packet_integrity).
# --------------------------------------------------------------------------- #
#
# The pedagogical rules (``derived_from_lo_ref``, ``assesses_from_question_lo``,
# the misconception-anchored materializers, etc.) emit edges whose endpoints are
# NOT concept-graph nodes: chunk IDs, synthetic question IDs (``q_<chunk>_<lo>``),
# learning-objective IDs (``to-NN`` / ``co-NN``), and misconception IDs
# (``mc_<hex>``). Each rule documents this as "federation-by-convention:
# consumers resolve the endpoints by ID-namespace prefix; no new node types are
# added to the concept graph."
#
# Relying on that convention alone is not enough. Whenever the upstream
# co-occurrence ``concept_graph`` is degenerate (empty ``concept_tags`` → no
# DomainConcept nodes), the graph carries typed edges with zero materialized
# nodes for those endpoints; the LibV2 ``packet_integrity`` gate's
# ``edge_endpoint_typing`` rule then cannot classify them (``node_class``
# lookup → ``None``) and fails the ``assesses`` contract (source must be
# ``Chunk``, target must be ``Outcome``/``ComponentObjective``), blocking
# archival with ``EDGE_ENDPOINT_TYPE_MISMATCH``.
#
# So after edge resolution we synthesize a typed node for every edge endpoint
# that isn't already a node, classifying it by its ID namespace using the SAME
# class names ``pedagogy_graph_builder`` uses (``Chunk`` / ``Outcome`` /
# ``ComponentObjective`` / ``Misconception``) and that
# ``EDGE_TYPING_CONTRACT`` allows. This is additive + backward-compatible:
# graphs whose endpoints already resolve to concept nodes are untouched, and a
# graph with no pedagogical edges materializes no extra nodes.
#
# A second materialization arm covers ``targets-concept`` edges: their
# ``target`` is an LO-authored concept slug (objectives ``key_concepts``
# vocabulary), NOT a chunk-derived concept-tag slug. When such a target has no
# co-occurrence DomainConcept node (the LO named a concept the chunks never
# tagged) it is materialized as a provenance-flagged DomainConcept node
# (``node_provenance="lo_key_concept"``, ``frequency=0``). Without this the
# downstream ``merge_duplicate_concept_nodes`` pass — and the flagless-build
# orphan gate — drop the edge because its endpoint is a phantom, silently
# losing the LO's targetedConcepts. Namespace classification keeps precedence
# (a key_concept that slugifies to ``to-01`` still classifies as ``Outcome``),
# so only genuinely concept-shaped targets take the DomainConcept arm. Both
# arms are unconditional, not flag-gated.

# Compiled once: a corpus chunk ID carries a ``chunk_`` token (matches
# ``Trainforge.eval.retrieval.chunk_ids.is_chunk_id``). A synthetic question ID is
# ``q_<chunk_id>_<lo_id>`` — it ALSO carries a ``chunk_`` token, so the chunk
# check below covers it (a question authored from an assessment_item chunk is a
# ``Chunk`` endpoint for the ``assesses`` contract). A misconception ID is
# ``mc_<16 hex>``.
_MISCONCEPTION_ID_RE = re.compile(r"^mc_[0-9a-f]{16}$")

# Whole-string LO-ID matcher, case-insensitive, mirroring the canonical
# ``lib.ontology.learning_objectives.LO_ID_PATTERN`` (``^[A-Z]{2,}-\d{2,}$``)
# but tolerant of the lowercased form the rules see.
_LO_ENDPOINT_RE = re.compile(r"^[a-z]{2,}-\d{2,}$")

# Edge types whose target endpoint is an LO-authored concept slug (the
# objectives ``key_concepts`` vocabulary, canonical_slug-normalized) rather
# than a chunk-derived concept-tag slug. When such a target does not resolve
# to a co-occurrence DomainConcept node (the LO author named a concept the
# corpus chunks never tagged), it is materialized as a provenance-flagged
# DomainConcept node instead of being left dangling — otherwise the merge pass
# / orphan gate drops the edge and the LO's targetedConcepts silently vanish.
_LO_CONCEPT_TARGET_EDGE_TYPES = frozenset({"targets-concept"})


def _classify_endpoint_id(node_id: str) -> Optional[str]:
    """Classify an edge-endpoint ID into a canonical node class by namespace.

    Returns the canonical class name (one allowed by
    ``EDGE_TYPING_CONTRACT`` / ``EDGE_CLASS_SYNONYMS``) or ``None`` when the
    ID doesn't match a known pedagogical namespace (e.g. a concept slug — which
    is already carried as a node by ``_build_nodes`` and must not be
    re-materialized with a guessed class).

    Resolution order (most specific first):

    * ``mc_<hex>``              -> ``Misconception``
    * contains a ``chunk_`` token (raw chunk IDs AND synthetic
      ``q_<chunk>_<lo>`` question IDs) -> ``Chunk``
    * LO ID (``to-NN`` / ``co-NN`` / any ``[A-Z]{2,}-\\d{2,}`` form,
      case-insensitive) -> ``Outcome`` for terminal prefixes, else
      ``ComponentObjective``
    """
    if not isinstance(node_id, str) or not node_id:
        return None

    if _MISCONCEPTION_ID_RE.match(node_id):
        return "Misconception"

    # Chunk IDs and synthetic question IDs both carry a ``chunk_`` token.
    # ``q_<chunk_id>_<lo_id>`` is the assessment-question endpoint for the
    # ``assesses`` contract whose source must be a ``Chunk``.
    if "chunk_" in node_id:
        return "Chunk"

    # Learning-objective IDs. process_course lowercases LO IDs before they
    # reach the rules (``to-01``), so classify case-insensitively. Terminal
    # objectives (``to-``) map to ``Outcome``; chapter/component objectives
    # (``co-``) and any other LO prefix map to ``ComponentObjective``.
    lowered = node_id.strip().lower()
    if _LO_ENDPOINT_RE.match(lowered):
        return "Outcome" if lowered.startswith("to-") else "ComponentObjective"

    return None


def _materialize_endpoint_nodes(
    existing_nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    run_id: Optional[str],
    created_at: str,
) -> List[Dict[str, Any]]:
    """Return ``existing_nodes`` plus a synthesized typed node per unresolved
    edge endpoint.

    For every ``source`` / ``target`` referenced by ``edges`` that is not
    already a node ID, classify it via :func:`_classify_endpoint_id` and emit a
    minimal typed node ``{id, label, class}`` (+ run/created_at provenance).

    Two materialization arms run, in precedence order:

    1. **Pedagogical-namespace endpoints** — chunk / question / objective /
       misconception IDs that :func:`_classify_endpoint_id` resolves to a
       canonical class. Namespace classification keeps precedence: a
       ``key_concept`` that happens to slugify to ``to-01`` still classifies
       as ``Outcome`` here, NOT as an LO-concept DomainConcept.
    2. **LO-authored concept targets** — the unresolved ``target`` of a
       ``targets-concept`` edge (see ``_LO_CONCEPT_TARGET_EDGE_TYPES``). These
       are objectives ``key_concepts`` slugs the corpus chunks never tagged,
       so they have no co-occurrence node. They are materialized as
       provenance-flagged ``DomainConcept`` nodes (``node_provenance =
       "lo_key_concept"``, ``frequency = 0``) so the merge pass / orphan gate
       can fold or segment them instead of dropping the edge.

    Endpoints that match neither arm (genuinely unknown slugs, non-
    targets-concept dangling targets) are left alone so the
    ``graph_edges_resolve`` rule still surfaces a genuine dangling edge rather
    than this step papering over it.

    Deterministic: new nodes are appended in sorted ID order so the artifact is
    byte-stable across runs for fixed inputs.
    """
    existing_ids = {
        n.get("id") for n in existing_nodes if isinstance(n, dict) and n.get("id")
    }

    # LO-authored concept targets: the target endpoint of a ``targets-concept``
    # edge that isn't already a node. These come from objectives key_concepts.
    lo_concept_targets = {
        e["target"]
        for e in edges
        if isinstance(e, dict)
        and e.get("type") in _LO_CONCEPT_TARGET_EDGE_TYPES
        and isinstance(e.get("target"), str)
        and e["target"]
        and e["target"] not in existing_ids
    }

    new_ids: set = set()
    for edge in edges:
        for side in ("source", "target"):
            endpoint = edge.get(side)
            if (
                isinstance(endpoint, str)
                and endpoint
                and endpoint not in existing_ids
            ):
                new_ids.add(endpoint)

    synthesized: List[Dict[str, Any]] = []
    for endpoint in sorted(new_ids):
        klass = _classify_endpoint_id(endpoint)
        if klass is not None:
            node = {"id": endpoint, "label": endpoint, "class": klass}
        elif endpoint in lo_concept_targets:
            # Unresolved targets-concept target → LO-authored DomainConcept.
            node = {
                "id": endpoint,
                "label": endpoint,
                "class": "DomainConcept",
                "frequency": 0,
                "node_provenance": "lo_key_concept",
            }
        else:
            continue
        if created_at is not None:
            _stamp_provenance(node, run_id, created_at)
        synthesized.append(node)

    if not synthesized:
        return existing_nodes
    return existing_nodes + synthesized


def _llm_escalate(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    rule_edges: List[Dict[str, Any]],
    llm_callable: Callable[..., List[Dict[str, Any]]],
    decision_capture: Any = None,
    run_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Invoke the LLM callable to propose extra edges and log decisions.

    The callable must return a list of edge dicts with the same shape as
    the rule modules. Provenance rule name is forced to ``llm_typed_edge``
    regardless of what the callable returned. If the callable raises, we
    swallow and return an empty list — LLM is advisory, never required.
    """
    try:
        proposed = llm_callable(
            chunks=chunks,
            course=course,
            concept_graph=concept_graph,
            existing=rule_edges,
        ) or []
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("LLM typed-edge escalation failed: %s", exc)
        return []

    normalized: List[Dict[str, Any]] = []
    node_ids = {n["id"] for n in concept_graph.get("nodes", [])}
    for edge in proposed:
        src = edge.get("source")
        tgt = edge.get("target")
        typ = edge.get("type")
        if not src or not tgt or typ not in _PRECEDENCE:
            continue
        if src not in node_ids or tgt not in node_ids:
            continue
        record = {
            "source": src,
            "target": tgt,
            "type": typ,
            "confidence": float(edge.get("confidence", 0.5)),
            "provenance": {
                "rule": "llm_typed_edge",
                "rule_version": 1,
                "evidence": dict(edge.get("evidence") or {}),
            },
        }
        if created_at is not None:
            _stamp_provenance(record, run_id, created_at)
        # LLM-escalated edges classify as `inferred` via the canonical
        # registry (rule name forced to `llm_typed_edge` above), keeping the
        # stamp symmetric with the deterministic rule loop.
        _stamp_edge_kind(record)
        normalized.append(record)
        if decision_capture is not None:
            try:
                decision_capture.log_decision(
                    decision_type="typed_edge_inference",
                    decision=f"{src} --{typ}--> {tgt}",
                    rationale=(
                        f"LLM escalation proposed a '{typ}' edge with confidence "
                        f"{record['confidence']:.2f}; rule-based pass produced no "
                        f"typed edge for this pair."
                    ),
                    confidence=record["confidence"],
                    context="typed_edge_inference.llm_escalate",
                )
            except Exception as exc:  # pragma: no cover — capture optional
                logger.debug("Decision capture failed for LLM edge: %s", exc)
    return normalized


def build_semantic_graph_with_dataset(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    *,
    llm_enabled: bool = False,
    llm_callable: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    decision_capture: Any = None,
    related_threshold: int = _related_mod.DEFAULT_THRESHOLD,
    now: Optional[datetime] = None,
    run_id: Optional[str] = None,
    misconceptions: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    objectives_metadata: Optional[List[Dict[str, Any]]] = None,
    terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    emit_trig: Optional[bool] = None,
    course_package_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Sibling of ``build_semantic_graph`` that additionally composes an
    ``rdflib.Dataset`` of per-rule named graphs.

    Returns ``(json_dict, dataset)``. ``json_dict`` is byte-identical to
    what ``build_semantic_graph`` returns for the same inputs (the JSON
    contract is preserved). ``dataset`` is an ``rdflib.Dataset`` when the
    flag is on and rdflib is importable; ``None`` otherwise.

    Flag resolution: ``emit_trig`` kwarg overrides the module flag when
    set. When ``None`` (default), falls back to
    ``Trainforge.rag.named_graph_writer.EMIT_TRIG``. Tests can either
    pass ``emit_trig=True`` directly or monkeypatch the module flag.

    All other kwargs are forwarded verbatim to the underlying rule
    pipeline; see ``build_semantic_graph`` for full documentation.
    """
    from Trainforge.rag import named_graph_writer

    if emit_trig is None:
        emit_trig = named_graph_writer.EMIT_TRIG

    json_dict, rule_outputs = _build_semantic_graph_internal(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        llm_enabled=llm_enabled,
        llm_callable=llm_callable,
        decision_capture=decision_capture,
        related_threshold=related_threshold,
        now=now,
        run_id=run_id,
        misconceptions=misconceptions,
        questions=questions,
        objectives_metadata=objectives_metadata,
        terminal_objectives=terminal_objectives,
        course_package_version=course_package_version,
    )

    if not emit_trig:
        return json_dict, None

    try:
        dataset = named_graph_writer.build_dataset(
            rule_outputs,
            run_id=run_id
            or (
                getattr(decision_capture, "run_id", None)
                if decision_capture is not None
                else None
            ),
            generated_at=json_dict["generated_at"],
            input_chunk_count=len(chunks),
        )
    except ImportError as exc:  # pragma: no cover — rdflib missing
        logger.warning(
            "TRAINFORGE_EMIT_TRIG is on but rdflib is unavailable: %s",
            exc,
        )
        return json_dict, None

    return json_dict, dataset


def build_semantic_graph(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    *,
    llm_enabled: bool = False,
    llm_callable: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    decision_capture: Any = None,
    related_threshold: int = _related_mod.DEFAULT_THRESHOLD,
    now: Optional[datetime] = None,
    run_id: Optional[str] = None,
    misconceptions: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    objectives_metadata: Optional[List[Dict[str, Any]]] = None,
    terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    course_package_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the typed-edge concept graph.

    Args:
        chunks: Pipeline chunks (see ``chunks.jsonl``).
        course: ``course.json`` dict; may be ``None`` if unavailable, in
            which case the prerequisite rule is skipped.
        concept_graph: Co-occurrence graph dict from ``concept_graph.json``.
        llm_enabled: When True and ``llm_callable`` is provided, call the
            callable for LLM-based escalation. Default off.
        llm_callable: Callable used for LLM escalation. Injected so tests
            and fallback paths are deterministic.
        decision_capture: Optional ``DecisionCapture`` instance. Used only
            when LLM escalation fires (every LLM decision must be logged).
            When ``run_id`` is not explicitly provided,
            ``decision_capture.run_id`` (if present) is used as the source
            for per-node/per-edge provenance.
        related_threshold: Minimum co-occurrence weight for ``related-to``.
        now: Override for ``generated_at``. When supplied, makes the
            artifact byte-identical across runs.
        run_id: Pipeline run identifier stamped on every emitted node +
            edge. When ``None``, falls back to ``decision_capture.run_id``
            if available; otherwise no ``run_id`` field is stamped. The
            per-node/per-edge ``created_at`` is always stamped with the
            artifact-level timestamp (``now`` or
            ``datetime.now(timezone.utc)``).
        misconceptions: Optional list of misconception entities (see
            ``schemas/knowledge/misconception.schema.json``). Used by the
            ``misconception-of`` rule to emit
            ``misconception_id -> concept_id`` edges when the upstream
            ``concept_id`` field is populated. Current call sites pass
            ``None`` → rule emits empty.
        questions: Optional list of assessment-question dicts carrying at
            minimum ``id`` + ``objective_id`` (and optional
            ``source_chunk_id``). Used by the ``assesses`` rule to emit
            ``question_id -> objective_id`` edges. Current call sites pass
            ``None`` → rule emits empty.

    Returns:
        Dict matching ``schemas/knowledge/concept_graph_semantic.schema.json``.
    """
    json_dict, _ = _build_semantic_graph_internal(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        llm_enabled=llm_enabled,
        llm_callable=llm_callable,
        decision_capture=decision_capture,
        related_threshold=related_threshold,
        now=now,
        run_id=run_id,
        misconceptions=misconceptions,
        questions=questions,
        objectives_metadata=objectives_metadata,
        terminal_objectives=terminal_objectives,
        course_package_version=course_package_version,
    )
    return json_dict


def _compute_graph_build_hash(
    *,
    course_id: Any,
    concept_graph: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    rule_versions: Dict[str, int],
    rulepack_version: str,
) -> str:
    """Deterministic build-hash over the graph's inputs.

    SHA-256 over a canonicalised JSON payload of the inputs that produced
    this graph. ``generated_at`` is excluded so two byte-identical runs
    against the same upstream inputs hash to the same value; any input
    change flips the hash loudly. See
    ``schemas/knowledge/concept_graph_semantic.schema.json::properties.graph_build_hash``
    for the consumer contract.
    """
    import hashlib
    import json as _json

    # Pull node IDs from the upstream co-occurrence concept_graph. Use the
    # raw `nodes[].id` field (the same ID the rules see), sorted for
    # determinism. Falling back to an empty list keeps a None-input graph
    # deterministic instead of crashing.
    raw_nodes = concept_graph.get("nodes") if isinstance(concept_graph, dict) else None
    node_ids = sorted(
        n.get("id", "")
        for n in (raw_nodes or [])
        if isinstance(n, dict) and isinstance(n.get("id"), str)
    )
    chunk_ids = sorted(
        c.get("id", "")
        for c in chunks
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    )
    payload = {
        "course_id": course_id or "",
        "node_ids": node_ids,
        "chunk_ids": chunk_ids,
        "rule_versions": dict(sorted(rule_versions.items())),
        "rulepack_version": rulepack_version,
    }
    canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_semantic_graph_internal(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    *,
    llm_enabled: bool,
    llm_callable: Optional[Callable[..., List[Dict[str, Any]]]],
    decision_capture: Any,
    related_threshold: int,
    now: Optional[datetime],
    run_id: Optional[str],
    misconceptions: Optional[List[Dict[str, Any]]],
    questions: Optional[List[Dict[str, Any]]],
    objectives_metadata: Optional[List[Dict[str, Any]]],
    terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    course_package_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Compute the JSON artifact AND the per-rule output list
    (``RuleOutput`` records) so the TriG writer can emit even-zero-edge
    named graphs.

    Returns ``(json_dict, rule_outputs)``. ``rule_outputs`` is a list of
    ``named_graph_writer.RuleOutput`` records — one per rule invoked,
    in fixed invocation order. The list is *pre-precedence*: each rule's
    emit is preserved exactly as the rule produced it (post-stamp). The
    JSON layer drops collisions; the named-graph layer must keep the raw
    per-rule emit so SPARQL can diff per-rule edge counts across runs and
    detect a rule that silently went to zero.
    """
    # Lazy import to keep the rule-only callers (``build_semantic_graph``)
    # free of rdflib at import time.
    from Trainforge.rag.named_graph_writer import RuleOutput

    # Resolve effective run_id / created_at once so every node
    # and edge in the artifact shares the same stamp. ``created_at`` equals
    # the artifact-level ``generated_at`` deliberately — the graph is an
    # atomic snapshot; per-element timestamps would drift only by sub-ms
    # jitter and break determinism tests that pin ``now``.
    effective_run_id = run_id
    if effective_run_id is None and decision_capture is not None:
        effective_run_id = getattr(decision_capture, "run_id", None)
    effective_now = now or datetime.now(timezone.utc)
    created_at = effective_now.isoformat()

    nodes = _build_nodes(concept_graph, run_id=effective_run_id, created_at=created_at)

    rule_edges: List[Dict[str, Any]] = []
    rule_versions: Dict[str, int] = {}
    rule_outputs: List[Any] = []

    # When TRAINFORGE_USE_SHACL_RULES=true, route the ``defined-by`` slot
    # through the SHACL-AF rule runner instead of the Python rule. The
    # runner exposes the same
    # ``(chunks, course, concept_graph) -> list[edge dict]`` signature
    # so the dispatch loop is otherwise unchanged. Equivalence with
    # the Python rule is pinned by
    # ``Trainforge/tests/test_shacl_rules_defined_by.py``.
    defined_by_fn = (
        _shacl_runner.shacl_defined_by_edges
        if _shacl_runner.USE_SHACL_RULES
        else infer_defined_by
    )

    # Rules are invoked in a fixed order so that equal-precedence ties
    # break deterministically. Taxonomic rules (is-a, prerequisite,
    # related-to) fire first; the pedagogical rules follow, ordered
    # alphabetically by EDGE_TYPE. Reordering this list changes which edge
    # survives a tie, so treat the order as part of the output contract.
    rule_specs: List[Tuple[Any, Any, Dict[str, Any]]] = [
        (infer_is_a, _is_a_mod, {}),
        (infer_prerequisite, _prereq_mod, {}),
        (infer_related, _related_mod, {"threshold": related_threshold}),
        (infer_assesses, _assesses_mod, {"questions": questions}),
        (defined_by_fn, _defined_by_mod, {}),
        (infer_derived_from_objective, _derived_lo_mod, {}),
        (infer_exemplifies, _exemplifies_mod, {}),
        (infer_misconception_of, _misconception_mod, {"misconceptions": misconceptions}),
        (
            infer_targets_concept,
            _targets_concept_mod,
            {"objectives_metadata": objectives_metadata},
        ),
        # Misconception-anchored materializers, in the same
        # alphabetical-by-EDGE_TYPE order as the pedagogical rules above.
        (infer_corrected_by_chunk, _corrected_by_chunk_mod, {}),
        (
            infer_detected_by_question,
            _detected_by_question_mod,
            {"questions": questions},
        ),
        (
            infer_interferes_with_outcome,
            _interferes_with_outcome_mod,
            {"misconceptions": misconceptions},
        ),
    ]

    # Content-dependency prerequisite rule
    # (``prerequisite_from_definition_mention``). CONDITIONALLY registered —
    # ONLY when ``TRAINFORGE_PREREQ_DEFINITION_MENTION`` is on. The rule name is
    # folded into ``rule_versions`` (below), which is serialized into
    # ``graph_build_hash`` / the downstream ``concept_graph_sha256``; an
    # UNCONDITIONAL registration would add a ``rule_versions`` key even on a
    # zero-edge run and change the hash. Gating the registration itself keeps
    # a flag-off (and even a flag-on-but-zero-edge) run byte-identical unless
    # the flag is on AND the rule produces edges. Emits federation TO->TO
    # ``prerequisite`` edges; ``infer`` self-gates on the same flag and
    # degrades to [] on any error, so appending it is safe.
    if _prereq_def_mention_mod.resolve_prereq_definition_mention():
        rule_specs.append(
            (
                _prereq_def_mention_mod.infer,
                _prereq_def_mention_mod,
                {
                    "terminal_objectives": terminal_objectives,
                    "capture": decision_capture,
                    "course_code": (
                        course.get("course_id")
                        if isinstance(course, dict)
                        else ""
                    )
                    or "",
                },
            )
        )

    for fn, rule_mod, kwargs in rule_specs:
        try:
            produced = fn(chunks, course, concept_graph, **kwargs) or []
        except Exception as exc:
            logger.warning("Rule %s failed: %s", rule_mod.RULE_NAME, exc)
            produced = []
        # Stamp each rule-produced edge with run provenance before precedence
        # resolution. Rule modules stay pure (they don't know about run_id);
        # the orchestrator decorates their output. edge_kind is stamped from
        # the canonical rule-classification registry in lockstep so every
        # emitted edge carries the asserted/inferred discriminator alongside
        # its run / created_at provenance.
        for edge in produced:
            _stamp_provenance(edge, effective_run_id, created_at)
            _stamp_edge_kind(edge)
        rule_edges.extend(produced)
        rule_versions[rule_mod.RULE_NAME] = rule_mod.RULE_VERSION
        # Capture the per-rule emit verbatim, even when empty, so the
        # named-graph writer can register a zero-edge graph — that is how a
        # rule that silently stopped producing edges becomes detectable.
        rule_outputs.append(
            RuleOutput(
                rule_name=rule_mod.RULE_NAME,
                rule_version=rule_mod.RULE_VERSION,
                edges=list(produced),
            )
        )

    # Apply precedence over the rule-based edges first so the LLM pass only
    # sees what the deterministic layer produced.
    rule_resolved = _apply_precedence(rule_edges)

    if llm_enabled and llm_callable is not None:
        extra = _llm_escalate(
            chunks=chunks,
            course=course,
            concept_graph=concept_graph,
            rule_edges=rule_resolved,
            llm_callable=llm_callable,
            decision_capture=decision_capture,
            run_id=effective_run_id,
            created_at=created_at,
        )
        if extra:
            rule_versions["llm_typed_edge"] = 1
            resolved = _apply_precedence(rule_resolved + extra)
            # Capture LLM-escalated edges as their own pseudo-rule output
            # so the TriG dataset reflects them too. Distinct rule_name
            # keeps it from colliding with deterministic rules.
            rule_outputs.append(
                RuleOutput(
                    rule_name="llm_typed_edge",
                    rule_version=1,
                    edges=list(extra),
                )
            )
        else:
            resolved = rule_resolved
    else:
        resolved = rule_resolved

    # Materialize a typed node for every resolved-edge endpoint that the
    # co-occurrence node set didn't cover (chunk / question / objective /
    # misconception IDs emitted by the pedagogical rules). This satisfies the
    # LibV2 ``packet_integrity`` ``edge_endpoint_typing`` contract — the
    # ``assesses`` source resolves to ``Chunk`` and its target to
    # ``Outcome``/``ComponentObjective`` — instead of leaving the endpoints
    # unclassified. Additive: when every endpoint already resolves (the rich
    # concept-graph case) no nodes are added.
    nodes = _materialize_endpoint_nodes(
        nodes, resolved, effective_run_id, created_at
    )

    generated_at = effective_now.isoformat()

    # Aggregate lineage fields.
    sorted_rule_versions = dict(sorted(rule_versions.items()))
    course_id_for_hash = course.get("course_id") if isinstance(course, dict) else None
    resolved_course_package_version = course_package_version
    if resolved_course_package_version is None and isinstance(course, dict):
        # Best-effort pickup from course.json. Courseforge doesn't emit
        # a package_version today; the field stays null when absent.
        cv = course.get("package_version")
        if isinstance(cv, str) and cv:
            resolved_course_package_version = cv
    graph_build_hash = _compute_graph_build_hash(
        course_id=course_id_for_hash,
        concept_graph=concept_graph,
        chunks=chunks,
        rule_versions=sorted_rule_versions,
        rulepack_version=RULEPACK_VERSION,
    )

    json_dict = {
        "kind": ARTIFACT_KIND,
        "generated_at": generated_at,
        "rule_versions": sorted_rule_versions,
        # Consumer contract: schemas/knowledge/concept_graph_semantic.schema.json
        "rulepack_version": RULEPACK_VERSION,
        "graph_build_hash": graph_build_hash,
        "course_package_version": resolved_course_package_version,
        "nodes": nodes,
        "edges": resolved,
    }
    return json_dict, rule_outputs


__all__ = [
    "ARTIFACT_KIND",
    "SCOPE_CONCEPT_IDS",
    "_make_concept_id",
    "build_semantic_graph",
    "build_semantic_graph_with_dataset",
]
