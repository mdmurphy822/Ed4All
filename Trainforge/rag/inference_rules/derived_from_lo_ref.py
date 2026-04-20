"""Rule: derive ``derived-from-objective`` edges from chunk LO references.

Every chunk in the pipeline carries ``learning_outcome_refs: List[str]`` —
the set of LO IDs the chunk is anchored to (populated by
``process_course._extract_objective_refs`` from JSON-LD, data-cf-*, or
heading heuristics). This rule materializes that existing pointer as a
first-class typed edge: ``chunk_id --derived-from-objective--> lo_id``.

Confidence is ``1.0`` — the reference is explicit, not inferred.

Federation-by-convention: ``source`` is a chunk ID (the raw chunk.id) and
``target`` is an LO ID (format ``TO-NN`` / ``CO-NN`` in current pipelines,
lowercased by the process_course normalization). Consumers resolve the
endpoints by ID-namespace prefix; no new node types are added to the
concept graph.

Deterministic: output sorted by (source, target); duplicates within the
same chunk's refs list are collapsed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

RULE_NAME = "derived_from_lo_ref"
# Wave 11 (Worker cc): bumped from 1 -> 2 to expose the optional
# source_references[] emit shape on DerivedFromObjectiveEvidence.
RULE_VERSION = 2
EDGE_TYPE = "derived-from-objective"

# Wave 11: opt-in flag gates the evidence-arm source_references[] emission.
SOURCE_PROVENANCE = os.getenv("TRAINFORGE_SOURCE_PROVENANCE", "").lower() == "true"


def _chunk_source_references(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a deep-copied list of source_references from a chunk's source block."""
    source = chunk.get("source") if isinstance(chunk, dict) else None
    if not isinstance(source, dict):
        return []
    refs = source.get("source_references")
    if not isinstance(refs, list):
        return []
    return [dict(r) for r in refs if isinstance(r, dict)]


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit one edge per (chunk_id, lo_id) reference.

    Args:
        chunks: Pipeline chunk dicts. Each may have ``id`` and
            ``learning_outcome_refs`` (list of LO IDs).
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces
            (chunks + LOs), not concept-graph nodes.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course, concept_graph  # unused; interface parity

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue
        refs = chunk.get("learning_outcome_refs") or []
        for ref in refs:
            if not ref:
                continue
            key = (chunk_id, ref)
            if key in seen:
                continue
            evidence: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "objective_id": ref,
            }
            # Wave 11: flag-gated source_references emit.
            if SOURCE_PROVENANCE:
                src_refs = _chunk_source_references(chunk)
                if src_refs:
                    evidence["source_references"] = src_refs
            seen[key] = {
                "source": chunk_id,
                "target": ref,
                "type": EDGE_TYPE,
                "confidence": 1.0,
                "provenance": {
                    "rule": RULE_NAME,
                    "rule_version": RULE_VERSION,
                    "evidence": evidence,
                },
            }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
