"""Rule: derive ``exemplifies`` edges from chunks flagged as examples.

A chunk is treated as an "example" when either of two fields marks it:

1. ``chunk_type == "example"`` (process_course's structural classification)
2. ``content_type_label == "example"`` (JSON-LD / data-cf-content-type
   signal preserved by process_course.)

For each such chunk, emit one ``chunk_id --exemplifies--> concept_id``
edge per ``concept_tags`` entry that resolves to a node in the
concept graph (after scoped-ID canonicalization via ``_make_concept_id``).

Federation-by-convention: ``source`` is a chunk ID; ``target`` is a
concept node ID (flat slug or scoped ``{course_id}:{slug}``). No new
node types are added.

Confidence is ``0.8`` — structural flag + non-empty concept_tags is a
strong but not perfectly reliable signal.

Deterministic: output sorted by (source, target); ties broken within a
chunk by sorted concept tag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

RULE_NAME = "exemplifies_from_example_chunks"
RULE_VERSION = 1
EDGE_TYPE = "exemplifies"


def _is_example(chunk: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (is_example, content_type) where content_type describes
    which signal triggered the match. Prefer ``content_type_label`` when
    both are "example" — it's the higher-fidelity JSON-LD/data-cf signal.
    """
    label = (chunk.get("content_type_label") or "").strip().lower()
    ctype = (chunk.get("chunk_type") or "").strip().lower()
    if label == "example":
        return True, "content_type_label"
    if ctype == "example":
        return True, "chunk_type"
    return False, ""


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``exemplifies`` edges for example chunks' concept tags.

    Args:
        chunks: Pipeline chunk dicts with ``id``, optional ``chunk_type``,
            optional ``content_type_label``, ``concept_tags``.
        course: Unused; interface parity.
        concept_graph: Used to filter concept_tags to those that are
            actually nodes in the graph.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course  # unused; interface parity

    # Late import to avoid circular dependency (typed_edge_inference imports
    # this module via inference_rules.__init__).
    from Trainforge.rag.typed_edge_inference import _make_concept_id

    node_ids = {n["id"] for n in concept_graph.get("nodes", []) if n.get("id")}
    if not node_ids:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue
        is_example, content_type = _is_example(chunk)
        if not is_example:
            continue
        course_id = (chunk.get("source") or {}).get("course_id")
        tags = chunk.get("concept_tags") or []
        for tag in sorted(set(tags)):
            if not tag:
                continue
            concept_id = _make_concept_id(tag, course_id)
            if concept_id not in node_ids:
                continue
            key = (chunk_id, concept_id)
            if key in seen:
                continue
            seen[key] = {
                "source": chunk_id,
                "target": concept_id,
                "type": EDGE_TYPE,
                "confidence": 0.8,
                "provenance": {
                    "rule": RULE_NAME,
                    "rule_version": RULE_VERSION,
                    "evidence": {
                        "chunk_id": chunk_id,
                        "concept_slug": tag,
                        "content_type": content_type,
                    },
                },
            }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
