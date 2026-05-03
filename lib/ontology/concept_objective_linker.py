"""Phase 6 Subtask 15 — concept-objective linker.

Two-stage linker (per ``plans/courseforge_architecture_roadmap.md`` §6.6
recommendation, refined in ``plans/phase6_abcd_concept_extractor.md``
Subtask 15) that runs as an explicit deterministic pass between the
concept-extraction phase output and objective-synthesizer persistence
in ``MCP/tools/pipeline_tools.py::_plan_course_structure``.

Contract
========

:func:`link_concepts_to_objectives` walks every learning objective and
populates its ``key_concepts`` / ``keyConcepts`` field from the concept
graph emitted by :func:`Trainforge.pedagogy_graph_builder.build_pedagogy_graph`
(the same concept-graph schema that
``schemas/knowledge/concept_graph_semantic.schema.json`` validates). The
goal is to make every LO carry a non-empty list of concept slugs so
downstream consumers (content_generation, training_synthesis) have a
deterministic concept-set to anchor against. Pre-Phase-6 corpora only
got concept anchors when the original outliner happened to author them;
Phase 6 backfills systematically.

Two passes
----------

1. **Slug-aware enrichment.** For each LO that already carries
   ``key_concepts`` / ``keyConcepts`` slugs, expand the set with any
   concept-graph node whose canonical slug contains one of the existing
   slugs as a substring (or vice-versa). This catches the very common
   case where the outliner emitted ``"property-paths"`` and the concept
   graph has ``"property-paths-co-15"`` (the LO-ref-suffixed variant
   that ``slug_to_label`` produces upstream).
2. **Statement-text matching.** For each concept-graph node not yet
   linked, check whether the LO ``statement`` text contains the concept
   slug verbatim or its deslugified surface form (kebab → space)
   case-insensitively. If yes, add the concept slug to the LO.

Defensive rules
---------------

- LO field name is preserved on emit: if the source LO uses
  ``key_concepts`` (snake_case, the runtime form
  :func:`MCP.tools._content_gen_helpers._normalize_objective_entry`
  emits) we write back to ``key_concepts``; if it uses ``keyConcepts``
  (camelCase, the JSON-LD ``$defs.LearningObjective`` form) we write
  back to ``keyConcepts``. When the LO has neither, we default to
  ``key_concepts`` (the runtime form) since :func:`_plan_course_structure`
  is the canonical caller.
- User-supplied concepts are NEVER overwritten — we only ADD. Order is
  preserved (existing concepts first, new concepts in deterministic
  sorted order).
- Duplicate slugs are collapsed.
- Empty / malformed concept graphs (missing ``nodes``, non-dict nodes,
  missing ``id``) are silently no-ops on the LO list, matching the
  warning-only severity of the upstream
  :class:`lib.validators.concept_graph.ConceptGraphValidator` gate.
- Concept-node ID prefixes like ``"concept:"`` (the pedagogy-graph form
  emitted by :func:`Trainforge.pedagogy_graph_builder.build_pedagogy_graph`)
  are stripped before matching. Bloom / DifficultyLevel / Outcome /
  ComponentObjective / Module / Chunk classes are filtered out — only
  ``Concept`` and ``DomainConcept`` (the two concept-class node labels
  the upstream emits) participate. Nodes lacking a ``class`` field are
  permissively included if their ID looks like a concept slug
  (``[a-z0-9-]+`` with no namespace prefix).

Cross-references
================

* ``plans/phase6_abcd_concept_extractor.md`` Subtask 15 — function spec.
* ``plans/phase6_abcd_concept_extractor.md`` Subtask 16 — wiring point in
  :func:`MCP.tools.pipeline_tools._plan_course_structure`.
* ``schemas/knowledge/concept_graph_semantic.schema.json`` — concept-
  graph node shape.
* :mod:`lib.ontology.slugs` — canonical slug helper.
* :mod:`Trainforge.pedagogy_graph_builder` — upstream concept-graph
  emitter (concept-class nodes use ``"concept:{slug}"`` IDs and class
  ``"Concept"`` per :func:`build_pedagogy_graph`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)


__all__ = ["link_concepts_to_objectives"]


# Concept-class labels the pedagogy-graph builder emits for nodes that
# represent learnable concepts. Other node classes (BloomLevel,
# DifficultyLevel, Outcome, ComponentObjective, Module, Chunk,
# Misconception, etc.) are filtered out — they are not "concepts" the
# LO can be anchored to.
_CONCEPT_CLASS_LABELS: frozenset = frozenset(
    {"Concept", "DomainConcept", "concept", "domain_concept"}
)


# ID-namespace prefix the pedagogy-graph builder uses for concept nodes.
# We strip this so the slug matched against LO text is the bare concept
# slug, not the prefixed form. Mirror :func:`Trainforge.pedagogy_graph_builder`
# which emits ``"concept:{slug}"`` for concept-class nodes (line 786).
_CONCEPT_ID_PREFIX: str = "concept:"


def _strip_concept_prefix(node_id: str) -> str:
    """Strip the ``"concept:"`` namespace prefix if present."""
    if node_id.startswith(_CONCEPT_ID_PREFIX):
        return node_id[len(_CONCEPT_ID_PREFIX):]
    return node_id


def _is_concept_node(node: Mapping[str, Any]) -> bool:
    """Return True iff ``node`` is a concept-class graph node."""
    cls = node.get("class")
    if isinstance(cls, str) and cls.strip() in _CONCEPT_CLASS_LABELS:
        return True
    # Permissive fallback: a node with no class but a concept-shaped ID
    # (no namespace prefix, kebab-case slug) is treated as a concept.
    # This matches future emit paths that drop the class field.
    if cls is None:
        nid = node.get("id")
        if isinstance(nid, str) and nid.strip():
            slug = nid.strip()
            if ":" not in slug and slug == canonical_slug(slug):
                return True
    return False


def _extract_concept_index(
    concept_graph: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    """Return a deterministic list of ``(slug, deslugified_label)`` pairs.

    Walks ``concept_graph["nodes"]``, filters to concept-class nodes,
    strips the ``"concept:"`` ID prefix, and yields each concept node's
    canonical slug paired with the human-readable label used for
    statement-text matching.

    Empty / malformed graphs return an empty list (no exception).
    """
    nodes_raw = concept_graph.get("nodes") if isinstance(concept_graph, Mapping) else None
    if not isinstance(nodes_raw, list):
        return []

    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for n in nodes_raw:
        if not isinstance(n, Mapping):
            continue
        if not _is_concept_node(n):
            continue
        nid = n.get("id")
        if not isinstance(nid, str) or not nid.strip():
            continue
        slug = _strip_concept_prefix(nid.strip())
        # Re-canonicalise so we tolerate non-canonical IDs (e.g. an
        # upstream emit that used uppercase or whitespace).
        slug = canonical_slug(slug)
        if not slug or slug in seen:
            continue
        seen.add(slug)

        # Prefer the explicit ``label`` field for statement-text
        # matching (it's the human-readable surface form). Fall back to
        # the deslugified slug.
        label = n.get("label")
        if not (isinstance(label, str) and label.strip()):
            label = slug.replace("-", " ")
        out.append((slug, label.strip().lower()))

    # Deterministic ordering — sort by slug so test fixtures stay stable.
    out.sort(key=lambda pair: pair[0])
    return out


def _existing_concepts(lo: Mapping[str, Any]) -> Tuple[List[str], str]:
    """Return ``(existing_slugs, field_name)`` for ``lo``.

    Tolerates both ``key_concepts`` (snake_case runtime form) and
    ``keyConcepts`` (camelCase JSON-LD form). When neither is present
    returns ``([], "key_concepts")`` since the runtime caller in
    :func:`_plan_course_structure` is the canonical write target.
    """
    raw = lo.get("key_concepts")
    if raw is not None:
        return ([str(s).strip() for s in raw if str(s).strip()], "key_concepts")
    raw = lo.get("keyConcepts")
    if raw is not None:
        return ([str(s).strip() for s in raw if str(s).strip()], "keyConcepts")
    return ([], "key_concepts")


def _statement_text(lo: Mapping[str, Any]) -> str:
    """Return the LO's statement-text-equivalent for verbatim matching.

    Tolerates ``statement`` (Courseforge runtime form) and
    ``description`` (alternate alias used in some legacy fixtures).
    Lowercased + whitespace-trimmed for case-insensitive matching.
    """
    for key in ("statement", "description", "text"):
        raw = lo.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return ""


def _slug_substring_match(existing: str, candidate: str) -> bool:
    """Return True iff ``existing`` and ``candidate`` are substring-related.

    A candidate concept-graph slug is considered a match for an existing
    LO slug when either is a substring of the other AFTER both have been
    re-canonicalised. Mirrors the spec's "substring match" wording — the
    common case is ``existing="property-paths"`` matching
    ``candidate="property-paths-co-15"`` (the LO-ref-suffixed variant
    upstream emits).
    """
    if not existing or not candidate:
        return False
    if existing == candidate:
        return True
    return existing in candidate or candidate in existing


def _statement_contains_concept(statement_text: str, slug: str, label: str) -> bool:
    """Return True iff ``statement_text`` contains the concept verbatim.

    Matches on the deslugified label (e.g. ``"property paths"`` for slug
    ``"property-paths"``) OR the bare slug. Case-insensitive (caller
    pre-lowers ``statement_text``). The deslugified form is the natural
    surface form a course outliner would have emitted; the bare-slug
    form catches edge-cases where the outliner kept the kebab-case ID.
    """
    if not statement_text:
        return False
    if label and label in statement_text:
        return True
    if slug and slug in statement_text:
        return True
    return False


def link_concepts_to_objectives(
    objectives: List[Dict[str, Any]],
    concept_graph: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Populate ``keyConcepts`` on each LO from a concept graph.

    Phase 6 Subtask 15 deterministic linker. Given a list of learning
    objectives and a concept graph, return a NEW list where each LO
    carries an enriched ``key_concepts`` / ``keyConcepts`` slug list
    drawn from concept-graph nodes that match the LO via:

    1. **Slug-aware substring match** against any concept the LO
       already lists.
    2. **Statement-text verbatim match** for any concept not yet
       linked, using the concept's deslugified label or bare slug.

    The function is a pure transform — it does NOT mutate the input
    list nor any LO dict. User-supplied concepts on the input LO are
    preserved verbatim and ordered first; newly linked concepts follow
    in deterministic sorted order.

    Args:
        objectives: List of LO dicts. Each carries ``id``,
            ``statement``, optional ``key_concepts`` / ``keyConcepts``
            slug list, plus other Phase 6 fields. Empty list or
            non-list returns an empty list.
        concept_graph: The dict loaded from
            ``concept_graph_semantic.json`` (or compatible schema).
            ``None`` / empty / malformed returns the input list with
            each LO shallow-copied but unchanged.

    Returns:
        New list with the same length / order as ``objectives``. Each
        LO is shallow-copied; the ``key_concepts`` / ``keyConcepts``
        field is replaced with the enriched list. The chosen field
        name preserves the source LO's casing (``key_concepts`` for
        runtime LOs, ``keyConcepts`` for JSON-LD LOs).

    Examples:
        >>> los = [{"id": "TO-01", "statement": "Identify cell parts"}]
        >>> graph = {"nodes": [
        ...     {"id": "concept:cell-parts", "class": "Concept",
        ...      "label": "Cell Parts"}
        ... ]}
        >>> out = link_concepts_to_objectives(los, graph)
        >>> out[0]["key_concepts"]
        ['cell-parts']
    """
    if not isinstance(objectives, list):
        return []

    # Shallow-copy every LO so mutation is contained even when the
    # concept graph is empty / malformed.
    cloned: List[Dict[str, Any]] = [
        dict(lo) if isinstance(lo, Mapping) else lo  # passthrough for malformed entries
        for lo in objectives
    ]

    if not concept_graph or not isinstance(concept_graph, Mapping):
        return cloned

    concept_index = _extract_concept_index(concept_graph)
    if not concept_index:
        return cloned

    # The set of all concept slugs from the graph — used by Pass 1.
    all_slugs: List[str] = [pair[0] for pair in concept_index]

    for idx, lo in enumerate(cloned):
        if not isinstance(lo, Mapping):
            continue

        existing, field_name = _existing_concepts(lo)
        # Use a list (not set) for the result so we preserve insertion
        # order: existing first, then sorted-deterministic additions.
        result: List[str] = []
        seen: Set[str] = set()
        for slug in existing:
            cs = canonical_slug(slug) or slug
            if cs and cs not in seen:
                result.append(cs)
                seen.add(cs)

        # Pass 1 — slug-aware substring enrichment of existing concepts.
        # For each existing concept slug, pull in any graph slug that
        # substring-matches.
        pass1_additions: Set[str] = set()
        for existing_slug in list(result):
            for candidate in all_slugs:
                if candidate in seen or candidate in pass1_additions:
                    continue
                if _slug_substring_match(existing_slug, candidate):
                    pass1_additions.add(candidate)

        # Pass 2 — statement-text verbatim match for everything not
        # already linked.
        statement_text = _statement_text(lo)
        pass2_additions: Set[str] = set()
        if statement_text:
            for slug, label in concept_index:
                if slug in seen or slug in pass1_additions:
                    continue
                if _statement_contains_concept(statement_text, slug, label):
                    pass2_additions.add(slug)

        # Append in deterministic sorted order so two runs over the same
        # inputs produce byte-identical output.
        for slug in sorted(pass1_additions):
            if slug not in seen:
                result.append(slug)
                seen.add(slug)
        for slug in sorted(pass2_additions):
            if slug not in seen:
                result.append(slug)
                seen.add(slug)

        # Preserve the source field-name casing.
        cloned[idx] = dict(lo)
        cloned[idx][field_name] = result

    return cloned
