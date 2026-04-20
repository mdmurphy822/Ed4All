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

from typing import Any, Dict, List

RULE_NAME = "defined_by_from_first_mention"
RULE_VERSION = 1
EDGE_TYPE = "defined-by"


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
    del chunks, course  # unused; interface parity

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
        edges.append({
            "source": node_id,
            "target": first_chunk,
            "type": EDGE_TYPE,
            "confidence": 0.7,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": {
                    "chunk_id": first_chunk,
                    "concept_slug": _concept_slug(node_id),
                    "first_mention_position": 0,
                },
            },
        })

    return sorted(edges, key=lambda e: (e["source"], e["target"]))
