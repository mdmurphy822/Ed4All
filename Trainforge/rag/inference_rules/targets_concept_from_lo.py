"""Rule: derive ``targets-concept`` edges from LO targetedConcepts[] (Wave 66).

Wave 57 added the ``targetedConcepts[]`` field on Courseforge-emitted
LearningObjectives: a Bloom-qualified LO→concept edge list carrying
``{concept, bloomLevel}`` per entry. This rule materializes those
entries as first-class typed edges in the concept graph, one edge per
``(lo_id, concept_slug)`` pair, with the Bloom level recorded on the
evidence provenance.

Federation-by-convention (REC-LNK-04): ``source`` is an LO ID
(``TO-NN`` / ``CO-NN``, lowercased to match the Trainforge normalization
used elsewhere) and ``target`` is a concept slug. No new node types are
added — consumers resolve endpoints by ID-namespace prefix.

Input contract: ``objectives_metadata`` kwarg — a list of dicts shaped
like the JSON-LD ``learningObjectives[]`` emitted by
``Courseforge/scripts/generate_course.py::_build_objectives_metadata``.
Each entry may have:

    {
      "id": "<canonical LO ID>",
      "targetedConcepts": [
        {"concept": "<slug>", "bloomLevel": "<canonical level>"},
        ...
      ],
    }

Missing fields silently skip: LOs without ``targetedConcepts`` produce
no edges; entries missing ``concept`` or ``bloomLevel`` are dropped with
a logged warning. This matches the defensive pattern of the other
inference rules so a legacy corpus without the Wave 57 emit produces
an empty edge list rather than a crash.

Confidence is ``1.0`` — the edge is explicit in the emit, not inferred.

Deterministic: output sorted by (source, target); duplicates within
the same LO's targetedConcepts (same concept slug appearing twice) are
collapsed to a single edge. When the same (lo_id, concept_id) appears
across multiple LOs, the first wins — matching the dedup pattern of
the Wave 5.2 ``derived_from_lo_ref`` rule.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

RULE_NAME = "targets_concept_from_lo"
RULE_VERSION = 1
EDGE_TYPE = "targets-concept"

# Wave 11 convention: opt-in provenance-source flag. When true, copy
# page-level sourceReferences[] from the emitting LO onto the evidence.
SOURCE_PROVENANCE = os.getenv("TRAINFORGE_SOURCE_PROVENANCE", "").lower() == "true"

# Canonical Bloom levels — keeping this local to avoid a cross-package
# import from lib.ontology.bloom. The six values match
# schemas/taxonomies/bloom_verbs.json which is the single source of truth.
_CANONICAL_BLOOM_LEVELS = frozenset(
    {"remember", "understand", "apply", "analyze", "evaluate", "create"}
)


def _lo_source_references(lo: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return deep-copied sourceReferences[] from an LO dict, if any."""
    refs = lo.get("sourceReferences") if isinstance(lo, dict) else None
    if not isinstance(refs, list):
        return []
    return [dict(r) for r in refs if isinstance(r, dict)]


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Emit one edge per (lo_id, concept_slug) pair from targetedConcepts[].

    Args:
        chunks: Unused; the emit predates chunking so edges come from the
            LO metadata directly. Interface parity with other rules.
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.
        **kwargs: Expects ``objectives_metadata`` — a list of LO dicts
            each potentially carrying a ``targetedConcepts[]`` field.
            Missing ⇒ no edges (legacy corpus without Wave 57 emit).

    Returns:
        Deterministically-ordered list of edge dicts. Empty when no
        ``objectives_metadata`` supplied or when no LO has
        ``targetedConcepts``.
    """
    del chunks, course, concept_graph  # interface parity

    objectives_metadata = kwargs.get("objectives_metadata") or []
    if not isinstance(objectives_metadata, list):
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for lo in objectives_metadata:
        if not isinstance(lo, dict):
            continue
        lo_id = lo.get("id")
        if not isinstance(lo_id, str) or not lo_id:
            continue
        # Normalize LO IDs to lowercase to match Trainforge's
        # process_course convention (ref-validation is case-insensitive
        # and the canonical downstream form is lowercase).
        lo_id_norm = lo_id.lower()

        targets = lo.get("targetedConcepts") or []
        if not isinstance(targets, list):
            continue

        for entry in targets:
            if not isinstance(entry, dict):
                continue
            concept = entry.get("concept")
            bloom = entry.get("bloomLevel")
            if not isinstance(concept, str) or not concept:
                logger.warning(
                    "targets_concept_from_lo: skipping entry on LO %r with "
                    "missing/empty concept: %r",
                    lo_id_norm,
                    entry,
                )
                continue
            if bloom not in _CANONICAL_BLOOM_LEVELS:
                logger.warning(
                    "targets_concept_from_lo: skipping entry on LO %r with "
                    "non-canonical bloomLevel %r (expected one of %r)",
                    lo_id_norm,
                    bloom,
                    sorted(_CANONICAL_BLOOM_LEVELS),
                )
                continue

            key = (lo_id_norm, concept)
            if key in seen:
                continue

            evidence: Dict[str, Any] = {
                "lo_id": lo_id_norm,
                "concept_id": concept,
                "bloom_level": bloom,
            }
            if SOURCE_PROVENANCE:
                src_refs = _lo_source_references(lo)
                if src_refs:
                    evidence["source_references"] = src_refs

            seen[key] = {
                "source": lo_id_norm,
                "target": concept,
                "type": EDGE_TYPE,
                "confidence": 1.0,
                "provenance": {
                    "rule": RULE_NAME,
                    "rule_version": RULE_VERSION,
                    "evidence": evidence,
                },
            }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
