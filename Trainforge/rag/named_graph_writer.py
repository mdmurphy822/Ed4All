"""Phase 3 of plans/rdf-shacl-enrichment-2026-04-26.md — TriG / named-graph
writer for typed-edge inference.

Builds an ``rdflib.Dataset`` where each inference rule's derived edges live
in their own named graph IRI, scoped by ``(run_id, rule_name)``. Per-graph
metadata (rule_version, generated_at, input_chunk_count, edge_count) is
attached to the graph IRI in the dataset's *default* graph so SPARQL queries
can detect regressions like Wave 82's zero-edge bug — a named graph with
``ed4all:edgeCount 0`` and an unchanged ``ed4all:ruleVersionApplied`` is an
obvious diff.

Behaviour flag: ``TRAINFORGE_EMIT_TRIG`` (default off). When off, this
module is never invoked from the orchestrator's emit path; ``EMIT_TRIG``
is captured at import time. Tests that need to toggle should monkeypatch
``EMIT_TRIG`` directly (or ``importlib.reload`` this module).

IRI scheme (sub-plan § 2): ``https://ed4all.io/run/<run_id>/rule/<rule_name>``.

   * Re-run-stable: same ``(run_id, rule_name)`` -> same graph IRI.
   * Distinct ``run_id`` -> disjoint graph IRIs, so two runs of the same
     fixture produce a queryable diff in the dataset.
   * Mirrors the project's other ``https://ed4all.io/`` IRI surfaces
     (``/concept/<slug>``, ``/lo/<id>``).

Metadata predicates (sub-plan § 3): reuse ``ed4all:rule``,
``ed4all:ruleVersionApplied``, ``dcterms:created``, ``prov:wasGeneratedBy``
from the existing concept_graph_semantic_v1.jsonld @context. Mints two
new: ``ed4all:edgeCount`` and ``ed4all:inputChunkCount`` (flagged
``_phase2_followup`` until folded into the @context). Graph IRI typed
``ed4all:RuleProvenanceGraph a prov:Bundle``.

Per-edge serialization inside each named graph (sub-plan § 4): dual emit.
Bare ``<source> <pred> <target>`` triple where ``<pred>`` is resolved via
``lib.ontology.edge_predicates.SLUG_TO_IRI`` (the asserted form a
reasoner consumes), plus a reified ``ed4all:TypedEdge`` blank node
carrying the per-edge provenance dict (the surface SPARQL queries can
join on rule, evidence, confidence). Mirrors Worker A's
concept_graph_semantic_v1.jsonld convention exactly.

Citations: Q3 (q_20260426_205702_83cd5b5d), Q5 (q_20260426_205702_6d4302e5),
Q49 (q_20260426_205724_4b21cb83), fresh-retrieve
q_20260426_230212_b9be9116 (this session).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger_name = __name__

# Captured at module-import time; tests monkeypatch this attribute directly.
EMIT_TRIG: bool = os.getenv("TRAINFORGE_EMIT_TRIG", "").lower() in {"true", "1"}
"""Phase 3 opt-in flag. Default off keeps existing JSON consumers
byte-identical. When on, the orchestrator routes through
``build_dataset`` and emits a sibling ``concept_graph_semantic.trig``."""

# IRI scheme — kept here so any caller (writer, tests, future SPARQL
# consumers) can reconstruct graph IRIs from ``(run_id, rule_name)``
# without parsing strings.
RULE_GRAPH_BASE: str = "https://ed4all.io/run/"
"""Base for per-rule graph IRIs. Full scheme:
``{RULE_GRAPH_BASE}<run_id>/rule/<rule_name>``."""

# Namespaces — must match concept_graph_semantic_v1.jsonld (Phase 1.1).
ED4ALL_NS: str = "https://ed4all.io/vocab/"
ED4ALL_CONCEPT_BASE: str = "https://ed4all.io/concept/"
PROV_NS: str = "http://www.w3.org/ns/prov#"
DCTERMS_NS: str = "http://purl.org/dc/terms/"
RDF_NS: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS: str = "http://www.w3.org/2000/01/rdf-schema#"
XSD_NS: str = "http://www.w3.org/2001/XMLSchema#"


@dataclass(frozen=True)
class RuleOutput:
    """One inference rule's emit summary, fed into ``build_dataset``.

    The orchestrator records per-rule output even when the rule produced
    zero edges — that's the Wave 82 self-detection mechanism. ``edges``
    carries the post-stamp pre-precedence edge dicts (each already
    decorated with ``run_id`` / ``created_at`` by the orchestrator's
    ``_stamp_provenance``).
    """

    rule_name: str
    rule_version: int
    edges: List[Dict[str, Any]]


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug_for_run_id(run_id: Optional[str], generated_at: str) -> str:
    """Return a URL-path-safe run identifier.

    When ``run_id`` is missing (legacy callers, tests without a
    ``DecisionCapture``), derive a deterministic local id from
    ``generated_at`` so the IRI scheme is always satisfiable. Tests pin
    ``run_id="test"`` to lock golden output.
    """
    if run_id:
        cleaned = _SLUG_RE.sub("-", str(run_id)).strip("-")
        if cleaned:
            return cleaned
    digest = hashlib.sha1(generated_at.encode("utf-8")).hexdigest()[:12]
    return f"local-{digest}"


def mint_rule_graph_iri(run_id: Optional[str], rule_name: str, generated_at: str) -> str:
    """Compute the named-graph IRI for ``(run_id, rule_name)``.

    Pure function; no rdflib import required. Useful for tests that
    want to assert IRI shape without constructing a dataset.
    """
    safe_run = _slug_for_run_id(run_id, generated_at)
    safe_rule = _SLUG_RE.sub("-", rule_name).strip("-")
    return f"{RULE_GRAPH_BASE}{safe_run}/rule/{safe_rule}"


def _resolve_edge_predicate_iri(slug: str) -> Optional[str]:
    """Resolve an edge slug to its registered predicate IRI.

    Imported lazily to avoid a runtime dependency on
    ``lib.ontology.edge_predicates`` at module import time (matches the
    convention used by ``typed_edge_inference._build_nodes`` for
    ``concept_classifier``).
    """
    from lib.ontology.edge_predicates import lookup_iri

    return lookup_iri(slug)


def _node_iri(graph_module: Any, node_id: str) -> Any:
    """Build an rdflib URIRef for an edge endpoint.

    Endpoints in the typed-edge graph are concept slugs by default
    (resolved against the concept-graph @base) but can also be LO IDs
    (``TO-NN``/``CO-NN``), chunk IDs, misconception IDs (``mc_*``), or
    question IDs per the federation-by-convention note in
    ``Trainforge/rag/inference_rules/__init__.py``. We resolve via
    the concept @base for bare slugs; an absolute IRI (``http://``,
    ``https://``) is returned verbatim.
    """
    URIRef = graph_module.URIRef
    if node_id.startswith("http://") or node_id.startswith("https://"):
        return URIRef(node_id)
    return URIRef(ED4ALL_CONCEPT_BASE + node_id)


def _add_edge_to_graph(
    rdflib_mod: Any,
    graph: Any,  # rdflib.Graph (named-graph context inside Dataset)
    edge: Dict[str, Any],
) -> int:
    """Materialize one edge inside a named graph.

    Returns the count of triples added (always >= 0). Emits both the
    bare asserted triple ``<src> <pred> <tgt>`` (when the slug resolves
    via SLUG_TO_IRI; we skip the bare triple if the slug is unregistered
    rather than fabricate a predicate) AND a reified ``ed4all:TypedEdge``
    blank node mirroring the JSON-LD context shape.
    """
    URIRef = rdflib_mod.URIRef
    BNode = rdflib_mod.BNode
    Literal = rdflib_mod.Literal

    src = edge.get("source")
    tgt = edge.get("target")
    slug = edge.get("type")
    if not src or not tgt or not slug:
        return 0

    src_iri = _node_iri(rdflib_mod, str(src))
    tgt_iri = _node_iri(rdflib_mod, str(tgt))
    pred_iri_str = _resolve_edge_predicate_iri(str(slug))

    added = 0

    # 1. Bare asserted triple (Q5 — the form a reasoner consumes).
    if pred_iri_str:
        graph.add((src_iri, URIRef(pred_iri_str), tgt_iri))
        added += 1

    # 2. Reified ed4all:TypedEdge blank node (per-edge provenance reachable
    #    as a subgraph; mirrors concept_graph_semantic_v1.jsonld § "Edges
    #    materialize as blank nodes typed ed4all:TypedEdge").
    edge_node = BNode()
    graph.add((edge_node, URIRef(RDF_NS + "type"), URIRef(ED4ALL_NS + "TypedEdge")))
    graph.add((edge_node, URIRef(ED4ALL_NS + "edgeSource"), src_iri))
    graph.add((edge_node, URIRef(ED4ALL_NS + "edgeTarget"), tgt_iri))
    # Slug expressed as a typed literal (the JSON-LD context uses @vocab
    # to land it as an IRI; here we keep it as a literal slug to preserve
    # round-trip fidelity even when the slug isn't yet registered).
    graph.add((edge_node, URIRef(ED4ALL_NS + "edgeType"), Literal(str(slug))))
    added += 4

    confidence = edge.get("confidence")
    if confidence is not None:
        graph.add(
            (
                edge_node,
                URIRef(ED4ALL_NS + "confidence"),
                Literal(float(confidence), datatype=URIRef(XSD_NS + "decimal")),
            )
        )
        added += 1

    prov = edge.get("provenance") or {}
    if prov:
        prov_node = BNode()
        graph.add((edge_node, URIRef(ED4ALL_NS + "hasProvenance"), prov_node))
        added += 1
        rule = prov.get("rule")
        if rule:
            graph.add(
                (prov_node, URIRef(ED4ALL_NS + "rule"), Literal(str(rule)))
            )
            added += 1
        rule_version = prov.get("rule_version")
        if rule_version is not None:
            graph.add(
                (
                    prov_node,
                    URIRef(ED4ALL_NS + "ruleVersionApplied"),
                    Literal(int(rule_version), datatype=URIRef(XSD_NS + "integer")),
                )
            )
            added += 1

    return added


def build_dataset(
    rule_outputs: List[RuleOutput],
    *,
    run_id: Optional[str],
    generated_at: str,
    input_chunk_count: int,
) -> Any:
    """Compose an ``rdflib.Dataset`` containing one named graph per rule.

    Each rule's named graph carries its derived edges (dual emit per
    ``_add_edge_to_graph``). Graph metadata (``ed4all:RuleProvenanceGraph``
    type, ``prov:Bundle`` type, rule, rule_version, generated_at, run_id,
    edge_count, input_chunk_count) is written to the dataset's *default*
    graph with the named-graph IRI as subject.

    Deterministic: rule_outputs are processed in input order; rdflib
    blank nodes use the default skolem strategy. Tests pinning a fixed
    ``run_id`` + ``generated_at`` get a stable triple set modulo
    blank-node identifier noise (sub-plan § 8 open question 1).

    Raises ``ImportError`` if rdflib isn't available — callers should
    not invoke this when ``EMIT_TRIG`` is off.
    """
    import rdflib

    URIRef = rdflib.URIRef
    Literal = rdflib.Literal
    Dataset = rdflib.Dataset

    ds = Dataset(default_union=False)
    # Bind common prefixes so TriG output is human-readable.
    ds.bind("ed4all", ED4ALL_NS)
    ds.bind("prov", PROV_NS)
    ds.bind("dcterms", DCTERMS_NS)
    ds.bind("rdf", RDF_NS)
    ds.bind("rdfs", RDFS_NS)
    ds.bind("xsd", XSD_NS)
    ds.bind("concept", ED4ALL_CONCEPT_BASE)

    default_graph = ds.default_context

    for output in rule_outputs:
        graph_iri_str = mint_rule_graph_iri(run_id, output.rule_name, generated_at)
        graph_iri = URIRef(graph_iri_str)

        # Register the named graph even when edge_count == 0 — that's the
        # Wave 82 self-detection mechanism. rdflib creates the context on
        # first .graph() call.
        named_graph = ds.graph(graph_iri)

        edge_count = 0
        for edge in output.edges:
            edge_count += 1 if _add_edge_to_graph(rdflib, named_graph, edge) else 0

        # Per-graph metadata in the *default* graph (sub-plan § 3).
        default_graph.add(
            (graph_iri, URIRef(RDF_NS + "type"), URIRef(ED4ALL_NS + "RuleProvenanceGraph"))
        )
        default_graph.add((graph_iri, URIRef(RDF_NS + "type"), URIRef(PROV_NS + "Bundle")))
        default_graph.add(
            (graph_iri, URIRef(ED4ALL_NS + "rule"), Literal(output.rule_name))
        )
        default_graph.add(
            (
                graph_iri,
                URIRef(ED4ALL_NS + "ruleVersionApplied"),
                Literal(int(output.rule_version), datatype=URIRef(XSD_NS + "integer")),
            )
        )
        default_graph.add(
            (
                graph_iri,
                URIRef(DCTERMS_NS + "created"),
                Literal(generated_at, datatype=URIRef(XSD_NS + "dateTime")),
            )
        )
        if run_id:
            default_graph.add(
                (
                    graph_iri,
                    URIRef(PROV_NS + "wasGeneratedBy"),
                    Literal(str(run_id)),
                )
            )
        default_graph.add(
            (
                graph_iri,
                URIRef(ED4ALL_NS + "edgeCount"),
                Literal(int(edge_count), datatype=URIRef(XSD_NS + "integer")),
            )
        )
        default_graph.add(
            (
                graph_iri,
                URIRef(ED4ALL_NS + "inputChunkCount"),
                Literal(int(input_chunk_count), datatype=URIRef(XSD_NS + "integer")),
            )
        )

    return ds


def serialize_trig(dataset: Any) -> str:
    """Serialize an rdflib Dataset as TriG (UTF-8 string).

    Thin wrapper kept here so callers don't import rdflib directly.
    """
    return dataset.serialize(format="trig")


__all__ = [
    "EMIT_TRIG",
    "RULE_GRAPH_BASE",
    "ED4ALL_NS",
    "ED4ALL_CONCEPT_BASE",
    "PROV_NS",
    "DCTERMS_NS",
    "RuleOutput",
    "mint_rule_graph_iri",
    "build_dataset",
    "serialize_trig",
]
