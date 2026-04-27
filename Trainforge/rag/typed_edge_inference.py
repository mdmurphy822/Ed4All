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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401

from Trainforge.rag.inference_rules import assesses_from_question_lo as _assesses_mod
from Trainforge.rag.inference_rules import defined_by_from_first_mention as _defined_by_mod
from Trainforge.rag.inference_rules import derived_from_lo_ref as _derived_lo_mod
from Trainforge.rag.inference_rules import exemplifies_from_example_chunks as _exemplifies_mod
from Trainforge.rag.inference_rules import (
    infer_assesses,
    infer_defined_by,
    infer_derived_from_objective,
    infer_exemplifies,
    infer_is_a,
    infer_misconception_of,
    infer_prerequisite,
    infer_related,
    infer_targets_concept,
)
from Trainforge.rag.inference_rules import is_a_from_key_terms as _is_a_mod
from Trainforge.rag.inference_rules import (
    misconception_of_from_misconception_ref as _misconception_mod,
)
from Trainforge.rag.inference_rules import prerequisite_from_lo_order as _prereq_mod
from Trainforge.rag.inference_rules import related_from_cooccurrence as _related_mod
from Trainforge.rag.inference_rules import targets_concept_from_lo as _targets_concept_mod
from Trainforge.rag import shacl_rule_runner as _shacl_runner

logger = logging.getLogger(__name__)

ARTIFACT_KIND = "concept_semantic"

# REC-ID-02 (Wave 4, Worker O): opt-in course-scoped concept IDs.
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
    produce two distinct scoped IDs — no silent merge. Wave 5 adds explicit
    ``aliases[]`` / equivalence edges for cross-course reconciliation.
    """
    if SCOPE_CONCEPT_IDS and course_id:
        return f"{course_id}:{slug}"
    return slug

# Precedence: higher wins. The orchestrator drops lower-precedence edges
# whose (source, target) pair is already claimed by a higher-precedence
# type.
#
# REC-LNK-04 (Wave 5.2, Worker U): 5 new pedagogical edge types slot at
# tier 2 (same as ``prerequisite``). In practice they don't collide with
# taxonomic edges because their endpoint namespaces differ (concept↔chunk,
# concept↔LO, chunk↔LO, misconception↔concept, question↔LO vs the
# concept↔concept taxonomic edges). Tier 2 assignment is defensive —
# ties among tier-2 rules break by fixed rule-invocation order.
_PRECEDENCE: Dict[str, int] = {
    "is-a": 3,
    "assesses": 2,
    "defined-by": 2,
    "derived-from-objective": 2,
    "exemplifies": 2,
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

    REC-PRV-01 (Worker P Wave 4.1). ``run_id`` is omitted when ``None`` so
    tests that construct graphs without a DecisionCapture instance keep
    passing. ``created_at`` is always stamped — it's produced by the
    orchestrator, not the rule modules, so it's always available.

    Schema side: both fields are OPTIONAL per ``concept_graph_semantic.schema.json``
    — legacy artifacts without them still validate.
    """
    if run_id:
        obj["run_id"] = run_id
    obj["created_at"] = created_at
    return obj


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

    REC-PRV-01 (Worker P Wave 4.1): when ``run_id`` / ``created_at`` are
    provided, stamp them onto each node so the semantic graph is
    time- and run-addressable.
    """
    # Wave 75: defensive backfill — if the upstream concept_graph was
    # built before classifier wiring landed, classify on the fly so the
    # semantic graph still ships ``class`` on every node. Imported lazily
    # so the rule library stays decoupled from lib/ontology at module
    # import time.
    from lib.ontology.concept_classifier import classify_concept

    nodes: List[Dict[str, Any]] = []
    for n in concept_graph.get("nodes", []):
        node = {
            "id": n["id"],
            "label": n.get("label", n["id"]),
            "frequency": n.get("frequency", 0),
        }
        # REC-LNK-04 (Wave 5.2, Worker U) / Worker S handoff: carry
        # ``occurrences[]`` (Wave 5.1, REC-LNK-01) from the co-occurrence
        # graph node into the semantic graph node so downstream consumers
        # (e.g. the ``defined-by`` rule) don't have to re-derive the
        # inverted index from chunks. Preserves Worker S's invariant that
        # ``occurrences[]`` is available on every concept node that has
        # chunks referencing it.
        occurrences = n.get("occurrences")
        if occurrences:
            node["occurrences"] = list(occurrences)
        # Wave 75: carry ``class`` through to the semantic graph so
        # retrieval can filter pedagogical / assessment / low-signal
        # nodes uniformly across both graph artifacts. Backfill via the
        # classifier when the source node lacks the field (legacy
        # graphs).
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
    emit_trig: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Phase 3 sibling of ``build_semantic_graph`` that additionally
    composes an ``rdflib.Dataset`` of per-rule named graphs.

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
            when LLM escalation fires (per CLAUDE.md: "ALL Claude decisions
            MUST be logged"). When ``run_id`` is not explicitly provided,
            ``decision_capture.run_id`` (if present) is used as the source
            for per-node/per-edge provenance (REC-PRV-01, Worker P).
        related_threshold: Minimum co-occurrence weight for ``related-to``.
        now: Override for ``generated_at``. When supplied, makes the
            artifact byte-identical across runs.
        run_id: REC-PRV-01 (Worker P Wave 4.1). Pipeline run identifier
            stamped on every emitted node + edge. When ``None``, falls
            back to ``decision_capture.run_id`` if available; otherwise
            no ``run_id`` field is stamped. The per-node/per-edge
            ``created_at`` is always stamped with the artifact-level
            timestamp (``now`` or ``datetime.now(timezone.utc)``).
        misconceptions: REC-LNK-04 (Wave 5.2, Worker U). Optional list of
            misconception entities (see ``schemas/knowledge/misconception.schema.json``).
            Used by the ``misconception-of`` rule to emit
            ``misconception_id -> concept_id`` edges when the upstream
            ``concept_id`` field is populated. Current call sites pass
            ``None`` → rule emits empty. Signal wiring deferred to a
            future wave.
        questions: REC-LNK-04 (Wave 5.2, Worker U). Optional list of
            assessment-question dicts carrying at minimum ``id`` +
            ``objective_id`` (and optional ``source_chunk_id``). Used by
            the ``assesses`` rule to emit ``question_id -> objective_id``
            edges. Current call sites pass ``None`` → rule emits empty.
            Signal wiring deferred to a future wave.

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
    )
    return json_dict


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
) -> Tuple[Dict[str, Any], List[Any]]:
    """Phase 3 internal: compute the JSON artifact AND the per-rule
    output list (``RuleOutput`` records) so the TriG writer can emit
    even-zero-edge named graphs.

    Returns ``(json_dict, rule_outputs)``. ``rule_outputs`` is a list of
    ``named_graph_writer.RuleOutput`` records — one per rule invoked,
    in fixed invocation order. The list is *pre-precedence*: each
    rule's emit is preserved exactly as the rule produced it
    (post-stamp), which is what Wave 82 self-detection needs (the
    JSON layer drops collisions; the named-graph layer keeps the raw
    per-rule emit so SPARQL can diff per-rule edge counts across runs).
    """
    # Lazy import to keep the rule-only callers (legacy
    # ``build_semantic_graph``) free of rdflib at import time.
    from Trainforge.rag.named_graph_writer import RuleOutput

    # REC-PRV-01: resolve effective run_id / created_at once so every node
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

    # Phase 5: when TRAINFORGE_USE_SHACL_RULES=true, route the
    # ``defined-by`` slot through the SHACL-AF rule runner instead of
    # the Python rule. The runner exposes the same
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
    # related-to) fire first to preserve Wave 4 behaviour on their output
    # shape; Wave 5.2 pedagogical rules (REC-LNK-04, Worker U) follow
    # alphabetically by EDGE_TYPE.
    for fn, rule_mod, kwargs in (
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
    ):
        try:
            produced = fn(chunks, course, concept_graph, **kwargs) or []
        except Exception as exc:
            logger.warning("Rule %s failed: %s", rule_mod.RULE_NAME, exc)
            produced = []
        # REC-PRV-01: stamp each rule-produced edge with run provenance
        # before precedence resolution. Rule modules stay pure (they don't
        # know about run_id); the orchestrator decorates their output.
        for edge in produced:
            _stamp_provenance(edge, effective_run_id, created_at)
        rule_edges.extend(produced)
        rule_versions[rule_mod.RULE_NAME] = rule_mod.RULE_VERSION
        # Phase 3: capture the per-rule emit verbatim (even when empty)
        # so the named-graph writer can register a zero-edge graph for
        # Wave 82 self-detection.
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

    generated_at = effective_now.isoformat()

    json_dict = {
        "kind": ARTIFACT_KIND,
        "generated_at": generated_at,
        "rule_versions": dict(sorted(rule_versions.items())),
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
