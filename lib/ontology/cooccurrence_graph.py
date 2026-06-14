"""Instance-free co-occurrence concept-graph builder.

Extracted from ``Trainforge/process_course.py::CourseProcessor._build_tag_graph``
so the co-occurrence graph can be built without a ``CourseProcessor``
instance — specifically by the ``textbook_to_course::concept_extraction``
workflow phase (``MCP/tools/pipeline_tools.py::_run_concept_extraction``),
which needs a ``kind: "concept"`` co-occurrence graph to feed
``Trainforge.rag.typed_edge_inference.build_semantic_graph`` (whose
``_build_nodes`` copies its node set verbatim from this graph).

The single source of truth for the co-occurrence build lives here;
``CourseProcessor._build_tag_graph`` delegates to ``build_cooccurrence_graph``
so the IMSCC path and the DART ``concept_extraction`` path produce
byte-equivalent graphs from the same chunk inputs.

v0.1.0 semantics preserved verbatim from the original ``_build_tag_graph``:

* Nodes are unique ``concept_tags`` appearing in ``>= min_freq`` chunks
  (default 2; drops to 1 under ``TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE``).
* Edges carry ``relation_type = "co-occurs"`` — the only type produced.
* Every node is stamped with a ``class`` (``lib.ontology.concept_classifier``).
* ``occurrences[]`` (sorted chunk-ID back-refs) + ``source_refs[]`` are
  emitted when non-empty.
* ``TRAINFORGE_SCOPE_CONCEPT_IDS`` course-scopes node IDs / endpoints.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["build_cooccurrence_graph"]

# Canonical group_by levels for the W3 page-aggregation knob. ``chunk`` is the
# byte-stable legacy default; ``page`` / ``section`` regroup pair-counting.
_GROUP_BY_LEVELS = frozenset({"chunk", "page", "section"})


def _group_key_for_chunk(chunk: Dict[str, Any], group_by: str) -> Optional[str]:
    """Resolve the co-occurrence GROUP key for a chunk (W3 page-aggregation).

    ``group_by="page"`` keys on ``source.lesson_id`` (the page identity),
    falling back to ``module_id`` then ``section_heading`` when ``lesson_id``
    is absent — all three live in ``chunk_v4.schema.json::$defs.Source``.
    ``group_by="section"`` keys on ``source.section_heading``. Returns ``None``
    when no key resolves (the caller folds such chunks into their own per-chunk
    group so the aggregation degrades gracefully rather than collapsing
    everything into one all-chunks group).
    """
    source = chunk.get("source")
    if not isinstance(source, dict):
        return None
    if group_by == "section":
        heading = source.get("section_heading")
        return heading if isinstance(heading, str) and heading.strip() else None
    # group_by == "page"
    for field in ("lesson_id", "module_id", "section_heading"):
        value = source.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_cooccurrence_graph(
    chunks: List[Dict[str, Any]],
    course_id: str = "",
    *,
    include_tags: Optional[Set[str]] = None,
    exclude_tags: Optional[Set[str]] = None,
    graph_kind: str = "concept",
    group_by: str = "chunk",
) -> Dict[str, Any]:
    """Build the domain-concept co-occurrence graph from chunk ``concept_tags``.

    Args:
        chunks: Pipeline chunks. Each chunk's ``concept_tags`` list drives
            both the node-frequency map and the co-occurrence pairs; a
            chunk with empty ``concept_tags`` contributes zero nodes.
        course_id: Course identifier; empty string ("no course_id") emits
            flat slugs even when ``TRAINFORGE_SCOPE_CONCEPT_IDS`` is on.
        include_tags: When set, only tags in this set are admitted.
        exclude_tags: When set, tags in this set are dropped (e.g. the
            pedagogy / logistics tag sets the IMSCC path filters out).
        graph_kind: Value stamped into the ``kind`` field of the result.
        group_by: W3 page-aggregation knob — the cooccurrence PAIR-COUNTING
            unit. ``"chunk"`` (default, byte-stable legacy): a pair gets an
            edge only when the two tags appear in the SAME chunk. ``"page"`` /
            ``"section"``: pair-counting regroups to the page
            (``source.lesson_id``) / section (``source.section_heading``) level
            BEFORE the pairwise loop, so two concepts that never share a chunk
            but co-occur on the same page get an edge. The KG-fragmentation
            fix — chunk-local tags fragment the graph; page-level connects it.
            The ``tag_frequency`` / ``occurrences[]`` node-provenance accounting
            stays CHUNK-level regardless of this value (only the pair counting
            moves to the group level), so node frequency + chunk back-refs are
            byte-identical across ``group_by`` values. An unrecognized value
            falls back to ``"chunk"`` with a logged warning. Chunks whose
            group key doesn't resolve (no lesson/module/section) fall back to
            per-chunk grouping so the aggregation degrades gracefully.

    Returns:
        ``{"kind": graph_kind, "nodes": [...], "edges": [...],
        "generated_at": ...}`` — the canonical ``concept_graph.json`` shape.
    """
    # REC-ID-02: opt-in course-scoped concept IDs. Imported here (not at
    # module scope) so this helper has no import-time dependency on the
    # typed-edge-inference module beyond the call path.
    from Trainforge.rag.typed_edge_inference import (
        SCOPE_CONCEPT_IDS,
        _make_concept_id,
    )
    from lib.ontology.concept_classifier import (
        classify_concept,
        is_scaffolding_noise,
    )
    from lib.ontology.labels import slug_to_label

    import logging

    course_id = course_id or ""
    if group_by not in _GROUP_BY_LEVELS:
        logging.getLogger(__name__).warning(
            "build_cooccurrence_graph: unrecognized group_by=%r; "
            "falling back to 'chunk' (byte-stable legacy).",
            group_by,
        )
        group_by = "chunk"
    # Change B: default-OFF scaffolding-noise prune. When ON, tags that
    # ``is_scaffolding_noise`` flags (pros/cons/creating/advanced/...) are
    # dropped before they become nodes or enter co-occurrence pairs.
    prune_scaffolding = (
        os.getenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", "").lower()
        == "true"
    )
    tag_frequency: Dict[str, int] = defaultdict(int)
    co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    # REC-LNK-01: inverted index node_id -> {chunk_id, ...}.
    concept_to_chunks: Dict[str, set] = defaultdict(set)
    # Wave 10: chunk id -> source.source_references[].
    chunk_source_refs: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        cid = chunk.get("id")
        if not cid:
            continue
        src = chunk.get("source") or {}
        refs = src.get("source_references") if isinstance(src, dict) else None
        if isinstance(refs, list) and refs:
            chunk_source_refs[cid] = refs

    def _accept(tag: str) -> bool:
        if include_tags is not None and tag not in include_tags:
            return False
        if exclude_tags is not None and tag in exclude_tags:
            return False
        if prune_scaffolding and is_scaffolding_noise(tag):
            return False
        return True

    # Node-frequency + occurrences accounting stays CHUNK-level regardless of
    # ``group_by`` (node provenance is per-chunk; only pair-counting regroups).
    # When ``group_by != "chunk"`` we fold the per-chunk tag lists into per-GROUP
    # tag SETS, then run the pairwise loop over the per-group sets.
    #   * group key resolves -> "group:<key>" bucket (pages/sections aggregate).
    #   * group key is None   -> a unique per-chunk bucket ("chunk:<idx>") so a
    #     chunk lacking lesson/module/section still pairs its OWN tags (graceful
    #     degradation; never collapses into one all-chunks group).
    group_tag_sets: Dict[str, Set[str]] = defaultdict(set)
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("id")
        tags = [t for t in chunk.get("concept_tags", []) if _accept(t)]
        for tag in tags:
            tag_frequency[tag] += 1
            if chunk_id:
                concept_to_chunks[_make_concept_id(tag, course_id)].add(chunk_id)
        if group_by == "chunk":
            # Legacy path: pair-counting per chunk, weighted by # chunks the
            # pair co-occurs in. Byte-identical to the pre-W3 behavior.
            for i, a in enumerate(tags):
                for b in tags[i + 1:]:
                    key = tuple(sorted([a, b]))
                    co_occurrence[key] += 1
        else:
            group_key = _group_key_for_chunk(chunk, group_by)
            bucket = f"group:{group_key}" if group_key is not None else f"chunk:{idx}"
            group_tag_sets[bucket].update(tags)

    if group_by != "chunk":
        # Pair-count over per-GROUP tag SETS — weight = # groups the pair
        # co-occurs in. Sorted for deterministic emit (the per-chunk path is
        # already deterministic in chunk order).
        for bucket in sorted(group_tag_sets):
            grouped = sorted(group_tag_sets[bucket])
            for i, a in enumerate(grouped):
                for b in grouped[i + 1:]:
                    key = tuple(sorted([a, b]))
                    co_occurrence[key] += 1

    sorted_tags = sorted(tag_frequency.items(), key=lambda x: -x[1])
    nodes: List[Dict[str, Any]] = []
    # Wave 82: single-occurrence concepts survive under the opt-in flag.
    min_freq = (
        1
        if os.getenv(
            "TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", ""
        ).lower() == "true"
        else 2
    )
    for tag, freq in sorted_tags:
        if freq < min_freq:
            continue
        node_id = _make_concept_id(tag, course_id)
        label = slug_to_label(tag)
        node: Dict[str, Any] = {
            "id": node_id,
            "label": label,
            "frequency": freq,
            "class": classify_concept(tag, label=label),
        }
        if SCOPE_CONCEPT_IDS and course_id:
            node["course_id"] = course_id
        occurrences = concept_to_chunks.get(node_id)
        if occurrences:
            sorted_occurrences = sorted(occurrences)
            node["occurrences"] = sorted_occurrences
            first_chunk_refs = chunk_source_refs.get(sorted_occurrences[0])
            if first_chunk_refs:
                node["source_refs"] = [dict(r) for r in first_chunk_refs]
        nodes.append(node)
    node_ids = {n["id"] for n in nodes}

    edges = []
    for (a, b), weight in co_occurrence.items():
        scoped_a = _make_concept_id(a, course_id)
        scoped_b = _make_concept_id(b, course_id)
        if scoped_a in node_ids and scoped_b in node_ids:
            edges.append({
                "source": scoped_a,
                "target": scoped_b,
                "weight": weight,
                "relation_type": "co-occurs",
            })

    return {
        "kind": graph_kind,
        "nodes": nodes,
        "edges": edges,
        "generated_at": datetime.now().isoformat(),
    }
