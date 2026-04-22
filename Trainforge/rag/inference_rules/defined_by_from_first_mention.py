"""Rule: derive ``defined-by`` edges from Worker S's ``occurrences[]``.

Worker S (Wave 5.1, REC-LNK-01) added ``occurrences: List[str]`` to
every concept-graph node — the set of chunk IDs that reference the
concept, sorted ASC at emit time. This rule materializes the canonical
"first mention" as a typed edge:

    concept_id --defined-by--> chunk_id

where ``chunk_id`` is ``occurrences[0]`` (first entry by ASC chunk-ID
sort).

Federation-by-convention: ``source`` is a concept node ID (flat slug or
``{course_id}:{slug}`` under Worker O's scoped-ID mode); ``target`` is a
raw chunk ID. No new node types are added.

Confidence is ``0.7`` — first-mention-by-chunk-ID-sort-order isn't
necessarily the pedagogical first-definition, but it's a stable,
reasonable proxy until a dedicated "definition detector" lands.

Stability caveat (inherited from Worker S): under position-based chunk
IDs (default), re-chunking invalidates ``occurrences[]`` entries and
therefore this rule's output. Under ``TRAINFORGE_CONTENT_HASH_IDS=true``
(Worker N's Wave 4 flag), occurrences survive re-chunks and this rule's
edges are stable.

Deterministic: output sorted by (source, target); concepts without
``occurrences`` produce no edge.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

RULE_NAME = "defined_by_from_first_mention"
# Wave 11 (Worker cc): bumped from 1 -> 2 to expose the optional
# source_references[] emit shape on DefinedByEvidence.
RULE_VERSION = 2
EDGE_TYPE = "defined-by"

# Wave 11: opt-in flag gates the evidence-arm source_references[] emission.
SOURCE_PROVENANCE = os.getenv("TRAINFORGE_SOURCE_PROVENANCE", "").lower() == "true"


def _build_chunk_index(chunks: List[Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    """Return a {chunk_id: chunk} lookup map for flag-on source_references
    resolution. Returns an empty dict when chunks is None or empty."""
    if not chunks:
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("id")
        if not cid:
            continue
        idx[cid] = chunk
    return idx


def _lookup_source_references(
    chunk_id: str, chunk_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return deep-copied source_references[] for the given chunk_id.

    Returns [] when the chunk isn't in the index (legacy) or has no refs.
    """
    chunk = chunk_index.get(chunk_id)
    if not isinstance(chunk, dict):
        return []
    source = chunk.get("source")
    if not isinstance(source, dict):
        return []
    refs = source.get("source_references")
    if not isinstance(refs, list):
        return []
    return [dict(r) for r in refs if isinstance(r, dict)]


def _concept_slug(node_id: str) -> str:
    """Strip the ``{course_id}:`` prefix from a scoped node ID.

    When ``TRAINFORGE_SCOPE_CONCEPT_IDS`` is on, node IDs are
    ``{course_id}:{slug}``; the slug portion is a suffix after the
    first colon. Off → the ID is already a flat slug.
    """
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``defined-by`` edges from each concept's first ``occurrences`` entry.

    Args:
        chunks: Unused; signal source is ``concept_graph.nodes[].occurrences``
            (populated by Worker S's Wave 5.1 change to ``_build_tag_graph``).
            Kept for interface parity.
        course: Unused; interface parity.
        concept_graph: The co-occurrence graph dict. Nodes may carry
            optional ``occurrences: List[str]`` already sorted ASC.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course  # unused; interface parity

    # Wave 11: build chunk lookup so the flag-on path can find source refs
    # for each first-mention chunk. Pre-Wave-11, chunks were deleted here —
    # now retained for the lookup. No behavioral change when the flag is off
    # (the index is simply not consulted).
    chunk_index = _build_chunk_index(chunks) if SOURCE_PROVENANCE else {}

    edges: List[Dict[str, Any]] = []
    for node in concept_graph.get("nodes", []) or []:
        node_id = node.get("id")
        if not node_id:
            continue
        occurrences = node.get("occurrences") or []
        if not occurrences:
            continue
        # Worker S sorts occurrences ASC at emit time; defensive re-sort
        # here in case an upstream fixture passes an unsorted list.
        first_chunk = sorted(occurrences)[0]
        evidence: Dict[str, Any] = {
            "chunk_id": first_chunk,
            "concept_slug": _concept_slug(node_id),
            "first_mention_position": 0,
        }
        # Wave 11: flag-gated source_references emit. Looks up the first-
        # mention chunk in the index and copies its source_references[].
        if SOURCE_PROVENANCE:
            refs = _lookup_source_references(first_chunk, chunk_index)
            if refs:
                evidence["source_references"] = refs
        edges.append({
            "source": node_id,
            "target": first_chunk,
            "type": EDGE_TYPE,
            "confidence": 0.7,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": evidence,
            },
        })

    return sorted(edges, key=lambda e: (e["source"], e["target"]))
