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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from Trainforge.rag.inference_rules import (
    infer_is_a,
    infer_prerequisite,
    infer_related,
)
from Trainforge.rag.inference_rules import is_a_from_key_terms as _is_a_mod
from Trainforge.rag.inference_rules import prerequisite_from_lo_order as _prereq_mod
from Trainforge.rag.inference_rules import related_from_cooccurrence as _related_mod

logger = logging.getLogger(__name__)

ARTIFACT_KIND = "concept_semantic"

# Precedence: higher wins. The orchestrator drops lower-precedence edges
# whose (source, target) pair is already claimed by a higher-precedence
# type.
_PRECEDENCE: Dict[str, int] = {
    "is-a": 3,
    "prerequisite": 2,
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


def _build_nodes(concept_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Copy nodes verbatim from the co-occurrence graph.

    The semantic graph shares its node set with the co-occurrence graph —
    no typed edge can reference a node that didn't already qualify for the
    base graph.
    """
    return [
        {
            "id": n["id"],
            "label": n.get("label", n["id"]),
            "frequency": n.get("frequency", 0),
        }
        for n in concept_graph.get("nodes", [])
    ]


def _llm_escalate(
    chunks: List[Dict[str, Any]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    rule_edges: List[Dict[str, Any]],
    llm_callable: Callable[..., List[Dict[str, Any]]],
    decision_capture: Any = None,
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
            MUST be logged").
        related_threshold: Minimum co-occurrence weight for ``related-to``.
        now: Override for ``generated_at``. When supplied, makes the
            artifact byte-identical across runs.

    Returns:
        Dict matching ``schemas/knowledge/concept_graph_semantic.schema.json``.
    """
    nodes = _build_nodes(concept_graph)

    rule_edges: List[Dict[str, Any]] = []
    rule_versions: Dict[str, int] = {}

    # Rules are invoked in a fixed order so that equal-precedence ties
    # break deterministically. (The three built-in rules never collide on
    # precedence; the ordering is for forward-compatibility.)
    for fn, rule_mod, kwargs in (
        (infer_is_a, _is_a_mod, {}),
        (infer_prerequisite, _prereq_mod, {}),
        (infer_related, _related_mod, {"threshold": related_threshold}),
    ):
        try:
            produced = fn(chunks, course, concept_graph, **kwargs) or []
        except Exception as exc:
            logger.warning("Rule %s failed: %s", rule_mod.RULE_NAME, exc)
            produced = []
        rule_edges.extend(produced)
        rule_versions[rule_mod.RULE_NAME] = rule_mod.RULE_VERSION

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
        )
        if extra:
            rule_versions["llm_typed_edge"] = 1
            resolved = _apply_precedence(rule_resolved + extra)
        else:
            resolved = rule_resolved
    else:
        resolved = rule_resolved

    generated_at = (now or datetime.now(timezone.utc)).isoformat()

    return {
        "kind": ARTIFACT_KIND,
        "generated_at": generated_at,
        "rule_versions": dict(sorted(rule_versions.items())),
        "nodes": nodes,
        "edges": resolved,
    }


__all__ = [
    "ARTIFACT_KIND",
    "build_semantic_graph",
]
