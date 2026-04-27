"""SHACL-AF rule runner for Phase 5 of the RDF/SHACL enrichment plan.

The Python rules under ``Trainforge/rag/inference_rules/`` derive typed
concept-graph edges procedurally. Phase 5 converts a subset of those
rules to SHACL-AF rules that live alongside the validation shapes in
``schemas/context/courseforge_v1.shacl-rules.ttl``. Co-locating
derivation with validation is the design point — see corpus query
``q_20260426_205719_89306a21`` (Q38 — SHACL Rules vs RDFS/OWL
entailment).

This module is the bridge: it takes a Trainforge concept-graph dict,
converts the relevant slice (concept nodes + ``occurrences[]``
back-references) to an in-memory rdflib Graph, runs the SHACL-AF rule
set via pyshacl with ``advanced=True`` and ``inplace=True``, and
projects the inferred ``ed4all:isDefinedBy`` triples back into the same
edge-dict shape that ``defined_by_from_first_mention.py`` emits.

Activation is gated by the ``TRAINFORGE_USE_SHACL_RULES`` env var:

* OFF (default): callers should dispatch the Python rule. This module
  returns an empty list when the flag is off so that calling it
  unconditionally is safe.
* ON: returns the SHACL-derived edge list, byte-identical to the
  Python rule's output (modulo deterministic ordering). Equivalence is
  pinned by ``Trainforge/tests/test_shacl_rules_defined_by.py``.

The Python rule under ``inference_rules/defined_by_from_first_mention.py``
is NOT modified or deleted — it remains the default until the SHACL
path proves out across the project test suite + corpus.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the canonical rule constants so confidence / RULE_NAME /
# RULE_VERSION stay in lockstep with the Python implementation. If
# the Python module bumps RULE_VERSION, the SHACL-derived edges
# automatically pick up the new value — no second source of truth.
from Trainforge.rag.inference_rules import defined_by_from_first_mention as _defined_by_mod

__all__ = [
    "USE_SHACL_RULES",
    "SHACL_RULES_PATH",
    "ED4ALL_NS",
    "CONCEPT_IRI_BASE",
    "CHUNK_IRI_BASE",
    "shacl_defined_by_edges",
]

#: Activation flag. Captured at import time; tests that need to flip
#: behavior should monkeypatch the module attribute (or
#: ``importlib.reload`` this module) rather than re-reading os.environ
#: at every call site.
USE_SHACL_RULES = os.getenv("TRAINFORGE_USE_SHACL_RULES", "").lower() == "true"

# Path to the SHACL-AF rules turtle file. Resolved lazily so test
# fixtures can monkeypatch the module attribute if they ever need to
# point at a sibling rule set.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SHACL_RULES_PATH: Path = _REPO_ROOT / "schemas" / "context" / "courseforge_v1.shacl-rules.ttl"

#: Namespace IRIs used to materialize the concept-graph slice as RDF.
#: ed4all: matches the canonical Phase 2.1 vocabulary
#: (schemas/context/courseforge_v1.vocabulary.ttl).
ED4ALL_NS = "https://ed4all.dev/ns/courseforge/v1#"

#: Per Phase 1.1 / 1.2c, concept node IRIs live under
#: ``https://ed4all.io/concept/<slug>`` so that pedagogy_graph and
#: concept_graph IRIs join cleanly. Chunk IRIs follow the same
#: convention under ``/chunk/<id>``. These bases are SHACL-rule-run
#: scoped — they don't leak into the emit dict, where we strip them
#: back to bare slugs / chunk IDs for parity with the Python rule.
CONCEPT_IRI_BASE = "https://ed4all.io/concept/"
CHUNK_IRI_BASE = "https://ed4all.io/chunk/"


def _concept_iri(node_id: str) -> str:
    """Return the IRI form of a concept node ID for the SHACL run."""
    return f"{CONCEPT_IRI_BASE}{node_id}"


def _chunk_iri(chunk_id: str) -> str:
    """Return the IRI form of a chunk ID for the SHACL run."""
    return f"{CHUNK_IRI_BASE}{chunk_id}"


def _strip_concept_iri(iri: str) -> str:
    """Inverse of ``_concept_iri`` — produce the bare node ID."""
    if iri.startswith(CONCEPT_IRI_BASE):
        return iri[len(CONCEPT_IRI_BASE):]
    return iri


def _strip_chunk_iri(iri: str) -> str:
    """Inverse of ``_chunk_iri`` — produce the bare chunk ID."""
    if iri.startswith(CHUNK_IRI_BASE):
        return iri[len(CHUNK_IRI_BASE):]
    return iri


def _build_data_graph(concept_graph: Dict[str, Any]):
    """Materialize the concept-graph slice the SHACL rule needs.

    For every node carrying ``occurrences``, emit:

        <concept_iri> a ed4all:Concept .
        <concept_iri> ed4all:occurrence "<chunk_iri>" .   (one per occurrence)

    Nodes without ``occurrences`` still get the ``rdf:type`` triple so
    the rule's ``targetClass ed4all:Concept`` selects them — but the
    rule's WHERE clause finds no ``ed4all:occurrence`` triples and
    therefore produces no inferred edge. That mirrors the Python rule's
    "no occurrences -> no edge" semantics for free.

    Lazy import of rdflib so this module stays importable even when the
    optional pyshacl/rdflib stack isn't installed.
    """
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import RDF

    g = Graph()
    ed4all = Namespace(ED4ALL_NS)
    g.bind("ed4all", ed4all)

    concept_class = URIRef(f"{ED4ALL_NS}Concept")
    occurrence_pred = URIRef(f"{ED4ALL_NS}occurrence")

    for node in concept_graph.get("nodes", []) or []:
        node_id = node.get("id")
        if not node_id:
            continue
        subj = URIRef(_concept_iri(node_id))
        g.add((subj, RDF.type, concept_class))
        for chunk_id in node.get("occurrences") or []:
            # Use a string Literal for the occurrence value. The SHACL
            # rule's MIN() then yields lex-min over the string form,
            # which is byte-identical to Python's sorted() on the
            # original chunk_id strings — no IRI/datatype mismatch
            # affecting the ordering.
            g.add((subj, occurrence_pred, Literal(chunk_id)))
    return g


def _run_pyshacl(data_graph) -> Any:
    """Run the SHACL-AF rule set against the data graph in place.

    Imports pyshacl lazily. Caller catches ImportError and degrades.
    """
    import pyshacl
    from rdflib import Graph

    shapes_graph = Graph()
    shapes_graph.parse(SHACL_RULES_PATH, format="turtle")

    # advanced=True activates SHACL-AF (rules + node expressions).
    # inplace=True merges inferred triples back into data_graph so we
    # can read them out below.
    pyshacl.validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",        # no RDFS/OWL — rules only
        advanced=True,
        inplace=True,
        abort_on_first=False,
        meta_shacl=False,
        js=False,
        debug=False,
    )
    return data_graph


def _project_defined_by_edges(
    data_graph,
    *,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert inferred ed4all:isDefinedBy triples to the edge-dict shape.

    Identical shape (and field set) as
    ``defined_by_from_first_mention.infer`` so the orchestrator's
    precedence + provenance-stamping passes are agnostic to which
    backend produced the edge.

    The ``chunks`` arg is accepted for interface parity with the Python
    rule. It's intentionally NOT consulted here — the SHACL pre-PoC
    skips the Wave 11 ``source_references[]`` evidence-arm path,
    deferring it to a follow-up wave once the equivalence baseline is
    proven.
    """
    del chunks  # interface parity; see docstring

    from rdflib import URIRef

    is_defined_by = URIRef(f"{ED4ALL_NS}isDefinedBy")

    edges: List[Dict[str, Any]] = []
    for s, _p, o in data_graph.triples((None, is_defined_by, None)):
        source = _strip_concept_iri(str(s))
        target = _strip_chunk_iri(str(o))
        evidence: Dict[str, Any] = {
            "chunk_id": target,
            "concept_slug": _concept_slug(source),
            "first_mention_position": 0,
        }
        edges.append({
            "source": source,
            "target": target,
            "type": _defined_by_mod.EDGE_TYPE,
            "confidence": 0.7,
            "provenance": {
                "rule": _defined_by_mod.RULE_NAME,
                "rule_version": _defined_by_mod.RULE_VERSION,
                "evidence": evidence,
            },
        })

    # Mirror the Python rule's deterministic emit order.
    return sorted(edges, key=lambda e: (e["source"], e["target"]))


def _concept_slug(node_id: str) -> str:
    """Mirror of ``defined_by_from_first_mention._concept_slug``.

    Vendored locally to avoid a private-name import. When
    ``TRAINFORGE_SCOPE_CONCEPT_IDS=true`` is in effect, node IDs are
    ``{course_id}:{slug}``; otherwise the ID is already a flat slug.
    """
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id


def shacl_defined_by_edges(
    chunks: Optional[List[Dict[str, Any]]],
    course: Optional[Dict[str, Any]],
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """SHACL-AF analogue of ``infer_defined_by``.

    Same ``(chunks, course, concept_graph) -> list[edge dict]`` signature
    as the Python rule modules. Returns ``[]`` when
    ``TRAINFORGE_USE_SHACL_RULES`` is off, when pyshacl/rdflib aren't
    importable, or when the input concept_graph has zero nodes — all
    three cases are silent fall-through to keep the Python rule the
    canonical default.
    """
    del course  # interface parity

    if not USE_SHACL_RULES:
        return []
    if not concept_graph.get("nodes"):
        return []

    try:
        data_graph = _build_data_graph(concept_graph)
        _run_pyshacl(data_graph)
        return _project_defined_by_edges(data_graph, chunks=chunks)
    except ImportError:
        # pyshacl / rdflib not installed. Return empty; the orchestrator
        # falls back to the Python rule path. The flag is opt-in, so a
        # missing dep on the SHACL path is not a hard fail.
        return []
