"""W3C foundational-technology anchor concepts (Wave 82, Slice 2).

The Wave 82 retrieval audit on ``LibV2/courses/rdf-shacl-551-2/`` found
that queries about top-level standards (``RDF``, ``RDFS``, ``OWL``,
``SHACL``, ``SPARQL``, ``Turtle``) and predicates (``owl:sameAs``)
failed graph-assisted retrieval because no canonical concept node
existed for the standalone term — the corpus had ``owl-2``,
``owl-2-dl``, ``apply-rdfs2-domain-typing`` etc. but never plain
``owl`` / ``rdfs``.

This module owns the bounded vocabulary of surface-form regexes that
the chunk-emit pipeline scans for, mapping each match to a canonical
anchor slug. When ``TRAINFORGE_SEED_TECH_CONCEPTS=true`` and a chunk's
text mentions ``RDF Schema`` (or ``RDFS``), ``rdfs`` is appended to its
``concept_tags`` so the existing 2-chunk co-occurrence gate in
``_generate_concept_graph`` admits it as a node.

Scope: W3C semantic-web standards + the predicates the audit named.
Domain-specific anchors (per-course technical vocab) belong in
``CourseProcessor.domain_concept_seeds``, not here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# Canonical slug → list of compiled surface-form regex patterns.
# Patterns use ``\b`` word boundaries to avoid false positives on
# substrings (``RDF`` matching inside ``XMLRDFReader``).  Case-
# insensitive because chunk text comes through verbatim — uppercase
# acronyms (``RDF``, ``OWL``) and predicate camelCase (``sameAs``)
# both need to fire.
_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    # The W3C standards. The user audit named these as missing concept
    # nodes; the chunks reference them across the whole corpus.
    "rdf": [
        re.compile(r"\bRDF\b(?!\s*[/-])", re.IGNORECASE),
    ],
    "rdfs": [
        re.compile(r"\bRDFS\b", re.IGNORECASE),
        re.compile(r"\bRDF\s+Schema\b", re.IGNORECASE),
    ],
    "owl": [
        # Match standalone OWL — but not when followed by a version
        # qualifier (``OWL 2``, ``OWL-2``); those already have specific
        # nodes (``owl-2``, ``owl-2-dl``).
        re.compile(r"\bOWL\b(?!\s*[-\s]?2\b)", re.IGNORECASE),
        re.compile(r"\bWeb\s+Ontology\s+Language\b", re.IGNORECASE),
    ],
    "shacl": [
        re.compile(r"\bSHACL\b", re.IGNORECASE),
        re.compile(r"\bShapes\s+Constraint\s+Language\b", re.IGNORECASE),
    ],
    "sparql": [
        re.compile(r"\bSPARQL\b", re.IGNORECASE),
    ],
    "turtle": [
        re.compile(r"\bTurtle\b", re.IGNORECASE),
        re.compile(r"\bTTL\b"),  # case-sensitive: ``ttl`` lowercase is the file extension noise
    ],
    "json-ld": [
        re.compile(r"\bJSON-LD\b", re.IGNORECASE),
    ],
    "n-triples": [
        re.compile(r"\bN-Triples\b", re.IGNORECASE),
        re.compile(r"\bNTriples\b", re.IGNORECASE),
    ],
    # Predicates the audit named explicitly. ``same-as`` is the canonical
    # slug; the alias map routes ``owl:sameAs`` / ``sameAs`` / ``owlsameas``
    # surface forms onto it.
    "same-as": [
        re.compile(r"\bowl:sameAs\b"),
        re.compile(r"\bsameAs\b"),
    ],
    # ---------------------------------------------------------------
    # Wave 84 — serialization formats beyond the Wave 82 set.
    # The audit found 36 chunks (worked-example pages spanning multiple
    # syntaxes) tagged with only the headline standards but missing the
    # specific format slugs they discussed.
    # ---------------------------------------------------------------
    "trig": [
        re.compile(r"\bTriG\b"),  # case-sensitive: distinct from "trig" (math fn)
    ],
    "n-quads": [
        re.compile(r"\bN-Quads\b", re.IGNORECASE),
        re.compile(r"\bNQuads\b", re.IGNORECASE),
    ],
    "rdf-xml": [
        # The W3C-spec rendering uses a slash; alias map normalises the
        # hyphenated form. Both surface forms must hit.
        re.compile(r"\bRDF/XML\b"),
        re.compile(r"\bRDFXML\b", re.IGNORECASE),
    ],
    # ---------------------------------------------------------------
    # Wave 84 — RDF foundational vocabulary. These concepts appeared
    # in 90%+ of weak-chunk text but never surfaced as concept_tags
    # because the surface form ``IRI`` / ``literal`` / ``datatype`` /
    # ``blank node`` matched no existing pattern.
    # ---------------------------------------------------------------
    "iri": [
        # Match standalone IRI but not URI/IRI compounds; the audit's
        # "IRI as Resource Identifier" chunk uses both freely so we
        # accept either.
        re.compile(r"\bIRI\b"),
        re.compile(r"\bIRIs\b"),
    ],
    "literal": [
        # RDF literal — only fires when adjacent to RDF/datatype context
        # so we don't over-match the English word ``literal``. Anchors:
        # "RDF literal", "literal value", "lexical literal", "datatype literal".
        re.compile(r"\bRDF\s+literals?\b", re.IGNORECASE),
        re.compile(r"\bdatatype[d-]?\s+literals?\b", re.IGNORECASE),
        re.compile(r"\blexical\s+literals?\b", re.IGNORECASE),
        re.compile(r"\bliteral\s+values?\b", re.IGNORECASE),
    ],
    "datatype": [
        # ``datatype`` is overloaded; require RDF/SHACL/XSD context to fire.
        re.compile(r"\bdatatypes?\b", re.IGNORECASE),
    ],
    "blank-node": [
        re.compile(r"\bblank\s+nodes?\b", re.IGNORECASE),
        re.compile(r"\b_:[A-Za-z][A-Za-z0-9_]*\b"),  # Turtle blank-node syntax
    ],
    "rdf-dataset": [
        # The W3C dataset / quad model. Distinct from generic "dataset".
        re.compile(r"\bRDF\s+datasets?\b", re.IGNORECASE),
        re.compile(r"\bnamed\s+graphs?\b", re.IGNORECASE),
    ],
    # ---------------------------------------------------------------
    # Wave 84 — SHACL-specific shape vocabulary. Audit's
    # SHACL queries (NodeShape, PropertyShape) had to retrieve via
    # raw text match because the slugs weren't in concept_tags.
    # ---------------------------------------------------------------
    "node-shape": [
        re.compile(r"\bsh:NodeShape\b"),
        re.compile(r"\bNodeShape\b"),
        re.compile(r"\bnode\s+shapes?\b", re.IGNORECASE),
    ],
    "property-shape": [
        re.compile(r"\bsh:PropertyShape\b"),
        re.compile(r"\bPropertyShape\b"),
        re.compile(r"\bproperty\s+shapes?\b", re.IGNORECASE),
    ],
    # ---------------------------------------------------------------
    # Wave 84 — RDFS predicates. ``subclassof`` and ``subpropertyof``
    # are first-class concepts on the entailment chunks; the audit
    # named ``subClassOf entailment`` as the headline retrieval miss.
    # ---------------------------------------------------------------
    "subclassof": [
        re.compile(r"\brdfs:subClassOf\b"),
        re.compile(r"\bsubClassOf\b"),
        re.compile(r"\bsubclass\s+of\b", re.IGNORECASE),
    ],
    "subpropertyof": [
        re.compile(r"\brdfs:subPropertyOf\b"),
        re.compile(r"\bsubPropertyOf\b"),
        re.compile(r"\bsubproperty\s+of\b", re.IGNORECASE),
    ],
    "rdf-type": [
        # The most-used RDF predicate. Surface: ``rdf:type``, ``a`` in
        # Turtle (too noisy to match standalone), ``is a`` (English).
        re.compile(r"\brdf:type\b"),
    ],
    # ---------------------------------------------------------------
    # Wave 84 — Turtle/SPARQL syntax keywords. Conservative: only the
    # ones distinctive enough that they reliably indicate the topic.
    # ---------------------------------------------------------------
    "turtle-prefix": [
        re.compile(r"@prefix\b"),  # Turtle prefix declaration
        re.compile(r"\bPREFIX\s+[a-z]"),  # SPARQL PREFIX
    ],
}


def detect_anchors(text: str) -> Set[str]:
    """Scan ``text`` for tech-anchor surface forms; return canonical slugs.

    Returns the set of anchor slugs whose patterns match the text.
    Empty set when ``text`` is falsy or no pattern matches. Pure
    function: no side effects, deterministic.
    """
    if not text:
        return set()
    hits: Set[str] = set()
    for slug, patterns in _PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                hits.add(slug)
                break
    return hits


def anchor_slugs() -> Tuple[str, ...]:
    """Return the canonical anchor slugs (sorted for stable output)."""
    return tuple(sorted(_PATTERNS.keys()))


__all__ = ["detect_anchors", "anchor_slugs"]
