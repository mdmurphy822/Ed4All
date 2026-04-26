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
