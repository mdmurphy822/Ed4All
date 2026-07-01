"""Canonical inference-rule -> edge_kind classifier (GPT feedback 12 May 2026, item 1).

Every typed-edge inference rule in the Trainforge concept-graph pipeline
emits edges that fall into one of two categories:

* ``asserted`` — the edge materializes an explicit upstream pointer
  (chunk -> LO reference, question -> objective_id, JSON-LD
  ``targetedConcepts[]``, misconception -> concept_id, concept ->
  first-mention chunk). The rule does NOT infer a relationship; it
  promotes an explicit declaration that already exists in the corpus.

* ``inferred`` — the edge is produced by a heuristic, statistical,
  co-occurrence, regex-pattern, or LLM-driven rule. The relationship
  is NOT explicit in the corpus; the rule is making an educated guess
  from structural signals.

Surfacing this distinction as a first-class ``edge_kind`` field on the
edge (see ``schemas/knowledge/concept_graph_semantic.schema.json``)
lets downstream consumers (retrieval, RDF export, KG-quality reports,
eval probes, training-pair filters) gate on "only consider explicitly-
asserted edges" without each maintaining its own rule-name allowlist.

The classification is orthogonal to ``confidence``: a materializer
may carry a sub-1.0 confidence to reflect pedagogical uncertainty
about WHICH explicit pointer is canonical (e.g.
``defined_by_from_first_mention`` confidence=0.7 reflects "first
chunk by ID sort is a reasonable proxy for first-pedagogical-
definition" — but the chunk it points at IS explicitly recorded in
``occurrences[]``).

Single source of truth: this module. Every new inference rule that
lands in ``Trainforge/rag/inference_rules/`` MUST add a registry
entry here; the unit test in
``Trainforge/tests/test_edge_kind_classification.py`` fails loudly if
a rule lands without an entry.

Public surface:

* :data:`EDGE_KIND_ASSERTED` / :data:`EDGE_KIND_INFERRED` — the two
  enum constants.
* :func:`edge_kind_for_rule` — registry lookup. Returns ``None`` on
  unknown rule names (caller decides whether to treat absence as
  warning or hard fail).
"""

from __future__ import annotations

from typing import Dict, Final, Optional

EDGE_KIND_ASSERTED: Final[str] = "asserted"
EDGE_KIND_INFERRED: Final[str] = "inferred"


# Rationale per rule lives in the rule module's docstring. The
# classification below mirrors what those docstrings declare:
#
#   asserted ("the edge is explicit in the emit, not inferred",
#             "Materializes the explicit ... reference",
#             "materializes the canonical first mention as a typed edge")
#
#   inferred (regex pattern match on ``key_terms[].definition``,
#             structural-flag heuristic on ``chunk_type == "example"``,
#             pedagogical-ordering inference, co-occurrence weight,
#             LLM proposal)
_RULE_KIND_REGISTRY: Final[Dict[str, str]] = {
    # ----- asserted: rule materializes an explicit upstream pointer
    "assesses_from_question_lo": EDGE_KIND_ASSERTED,
    "derived_from_lo_ref": EDGE_KIND_ASSERTED,
    "misconception_of_from_misconception_ref": EDGE_KIND_ASSERTED,
    "targets_concept_from_lo": EDGE_KIND_ASSERTED,
    "defined_by_from_first_mention": EDGE_KIND_ASSERTED,
    # GPT feedback 12 May 2026 item 4 — three new misconception-anchored
    # materializers. Each promotes an existing explicit pointer in the
    # corpus (chunk.misconceptions[].correction; question.misconception_id;
    # misconception.lo_id) into a typed edge.
    "corrected_by_from_chunk_misconception": EDGE_KIND_ASSERTED,
    "detected_by_from_distractor_misconception_id": EDGE_KIND_ASSERTED,
    "interferes_with_outcome_from_misconception_lo": EDGE_KIND_ASSERTED,
    # ----- inferred: heuristic / statistical / pattern-match / LLM
    "is_a_from_key_terms": EDGE_KIND_INFERRED,
    "exemplifies_from_example_chunks": EDGE_KIND_INFERRED,
    "prerequisite_from_lo_order": EDGE_KIND_INFERRED,
    # W3.1 — content-dependency (definition-in-TO_a assumed-in-TO_b) TO->TO
    # prerequisite edges. A deterministic no-LLM CONTENT signal (concept
    # first-mention/definition anchor), NOT an explicit upstream pointer, so
    # it classifies as inferred alongside prerequisite_from_lo_order. Producer:
    # lib/generation/prerequisite_from_definition_mention.py (gated by
    # TRAINFORGE_PREREQ_DEFINITION_MENTION). NB: this rule module lives under
    # lib/generation/ (a lib-side edge producer feeding the concept graph),
    # not Trainforge/rag/inference_rules/, so the inference_rules-discovery
    # test does not auto-cover it; the entry is asserted directly here.
    "prerequisite_from_definition_mention": EDGE_KIND_INFERRED,
    "related_from_cooccurrence": EDGE_KIND_INFERRED,
    "llm_typed_edge": EDGE_KIND_INFERRED,
}


def edge_kind_for_rule(rule_name: str) -> Optional[str]:
    """Return the canonical ``edge_kind`` for a given inference rule.

    Args:
        rule_name: The rule name as emitted on
            ``edge.provenance.rule`` by Trainforge inference rules.

    Returns:
        ``"asserted"`` when the rule materializes an explicit upstream
        pointer; ``"inferred"`` when the rule is heuristic / statistical
        / LLM-based; ``None`` when the rule is not in the registry
        (legacy / future-rule path — caller decides on graceful
        degradation vs fail-closed).
    """
    if not isinstance(rule_name, str):
        return None
    return _RULE_KIND_REGISTRY.get(rule_name)


def known_rule_names() -> frozenset[str]:
    """Return the set of rule names this registry knows about.

    Exposed so test suites can assert coverage against the set of
    ``RULE_NAME`` values exported by ``Trainforge/rag/inference_rules/``
    modules; any new rule landing without a registry entry trips the
    classification unit test before the silent-default behavior ships.
    """
    return frozenset(_RULE_KIND_REGISTRY.keys())


__all__ = [
    "EDGE_KIND_ASSERTED",
    "EDGE_KIND_INFERRED",
    "edge_kind_for_rule",
    "known_rule_names",
]
