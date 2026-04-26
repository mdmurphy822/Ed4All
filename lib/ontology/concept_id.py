"""Wave 82 concept-ID prefix helper.

The Ed4All pipeline carries concepts in two ID conventions:

* **Pedagogy graph** (``Trainforge/pedagogy_graph_builder.py``) — emits
  ``concept:{slug}`` because nodes are typed-class entities and the
  prefix names the class (e.g. ``concept:rdf-graph``,
  ``module:week-3``, ``bloom:apply``).
* **Concept graphs** (``concept_graph.json``,
  ``concept_graph_semantic.json``) — emit bare ``{slug}`` because every
  node is a concept; the typing is implicit at the graph level.

Cross-graph joins (e.g. "for each pedagogy concept, look up its
co-occurrence frequency") need to bridge the two forms. Pre-Wave-82
this happened ad-hoc: ``Trainforge/curriculum.py`` strips the prefix
inline at four call sites; other consumers copied the pattern. This
helper centralizes the rule so callers don't reinvent it.

The audit (rdf-shacl-551-2 deep review, Section D) flagged the format
mismatch as a should-fix because it adds friction to every cross-graph
join. Wave 82 picks "keep both forms, document the join" over "rewrite
one side" — cheaper, no schema migration, no manifest churn.
"""

from __future__ import annotations

CONCEPT_PREFIX = "concept:"


def strip_concept_prefix(node_id: str) -> str:
    """Return the bare concept slug from a possibly-prefixed node id.

    Pass-through for IDs that don't start with ``concept:``. Empty input
    returns ``""``. Used at every concept-graph ↔ pedagogy-graph join
    boundary so consumers don't have to reinvent the prefix-strip logic.

    Examples:
        >>> strip_concept_prefix("concept:rdf-graph")
        'rdf-graph'
        >>> strip_concept_prefix("rdf-graph")
        'rdf-graph'
        >>> strip_concept_prefix("")
        ''
    """
    if not node_id:
        return ""
    if node_id.startswith(CONCEPT_PREFIX):
        return node_id[len(CONCEPT_PREFIX):]
    return node_id


def add_concept_prefix(slug: str) -> str:
    """Return ``concept:{slug}`` form for a bare slug; idempotent.

    Used when a consumer needs to look up a concept-graph slug in the
    pedagogy graph. Pass-through when the input already carries the
    prefix. Empty input returns ``""``.

    Examples:
        >>> add_concept_prefix("rdf-graph")
        'concept:rdf-graph'
        >>> add_concept_prefix("concept:rdf-graph")
        'concept:rdf-graph'
        >>> add_concept_prefix("")
        ''
    """
    if not slug:
        return ""
    if slug.startswith(CONCEPT_PREFIX):
        return slug
    return f"{CONCEPT_PREFIX}{slug}"


__all__ = ["CONCEPT_PREFIX", "strip_concept_prefix", "add_concept_prefix"]
