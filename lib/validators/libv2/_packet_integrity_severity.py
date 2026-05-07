"""W-D7 T7.1 — Severity tables + class synonyms for the libv2
packet-integrity validator.

Extracted from :mod:`lib.validators.libv2.packet_integrity` so the rule
catalog + the typing-contract maps live in one auditable file.
Re-exported through the canonical module so existing imports
(``from lib.validators.libv2.packet_integrity import RULE_SEVERITY``)
keep resolving.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.1.
"""
from __future__ import annotations

import os
from typing import Dict, Set, Tuple

__all__ = [
    "ASSESSMENT_CHUNK_TYPE",
    "COVERAGE_RULES",
    "EDGE_CLASS_SYNONYMS",
    "EDGE_TYPING_CONTRACT",
    "PEDAGOGICAL_EDGE_TYPES",
    "RELAX_ENV_VAR",
    "RULE_SEVERITY",
    "SCAFFOLDING_CLASSES",
    "TEACHING_CHUNK_TYPES",
    "TYPING_RULES",
    "_env_relax",
]


RULE_SEVERITY: Dict[str, str] = {
    "unique_chunk_ids": "critical",
    "refs_resolve": "critical",
    "co_has_parent": "critical",
    "no_comma_refs": "critical",
    "graph_edges_resolve": "critical",
    "assessment_has_objective": "warning",
    "to_has_teaching_and_assessment": "warning",
    "domain_concept_has_chunk": "warning",
    "scaffolding_not_assessed": "warning",
    # Wave 78 — coverage rules (default warning, critical under
    # --strict-coverage).
    "every_objective_has_teaching": "warning",
    "every_objective_has_assessment": "warning",
    # Wave 78 — typing rule (default warning, critical under
    # --strict-typing).
    "edge_endpoint_typing": "warning",
}

# Wave 78 — rules promoted to critical when the caller passes
# ``strict=True`` or the more granular flags. The CLI surfaces
# ``--strict-coverage`` and ``--strict-typing``; ``--strict`` implies
# both.
COVERAGE_RULES: Set[str] = {
    "to_has_teaching_and_assessment",
    "every_objective_has_teaching",
    "every_objective_has_assessment",
    "domain_concept_has_chunk",
}
TYPING_RULES: Set[str] = {
    "edge_endpoint_typing",
}

# Chunk types that *teach* their referenced LOs (vs. assess them or
# scaffold them as exercise prompts).
TEACHING_CHUNK_TYPES: Set[str] = {
    "explanation",
    "overview",
    "summary",
    "example",
}
ASSESSMENT_CHUNK_TYPE = "assessment_item"

# Concept-graph node classes that scaffold rather than represent
# domain content. ``scaffolding_not_assessed`` flags them when they
# appear as edge targets of pedagogical edges.
SCAFFOLDING_CLASSES: Set[str] = {
    "PedagogicalMarker",
    "AssessmentOption",
    "LowSignal",
    "InstructionalArtifact",
}

# Edge ``type`` values in concept_graph_semantic.json that should
# point at *content* (DomainConcept / Misconception / LearningObjective)
# rather than scaffolding.
PEDAGOGICAL_EDGE_TYPES: Set[str] = {
    "derived-from-objective",
    "assesses",
}


# Wave 78 — typed-endpoint contract for ``edge_endpoint_typing``.
EDGE_TYPING_CONTRACT: Dict[str, Tuple[Set[str], Set[str]]] = {
    "teaches": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "assesses": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "practices": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "exemplifies": ({"Chunk"}, {"DomainConcept"}),
    "supports_outcome": ({"ComponentObjective"}, {"Outcome"}),
    "interferes_with": ({"Misconception"}, {"DomainConcept"}),
    "prerequisite_of": ({"DomainConcept"}, {"DomainConcept"}),
    "belongs_to_module": ({"Chunk"}, {"Module"}),
    "at_bloom_level": (
        {"Outcome", "ComponentObjective"},
        {"BloomLevel"},
    ),
    "follows": ({"Module"}, {"Module"}),
}

# Class synonyms — pedagogy_graph emits ``Concept`` for what the
# concept_graph emits as ``DomainConcept``; ``TerminalOutcome`` and
# ``Outcome`` are synonyms in different worker emits.
EDGE_CLASS_SYNONYMS: Dict[str, str] = {
    "Concept": "DomainConcept",
    "DomainConcept": "DomainConcept",
    "TerminalOutcome": "Outcome",
    "Outcome": "Outcome",
    "LearningObjective": "Outcome",
    "ComponentObjective": "ComponentObjective",
    "Chunk": "Chunk",
    "Module": "Module",
    "BloomLevel": "BloomLevel",
    "Misconception": "Misconception",
}


# Wave 78 — env var that downgrades all gated rules to warnings,
# regardless of strict-mode flags. Legacy archives only.
RELAX_ENV_VAR = "LIBV2_RELAX_PACKET_INTEGRITY"


def _env_relax() -> bool:
    """Return True iff ``LIBV2_RELAX_PACKET_INTEGRITY=true`` is set."""
    return (os.getenv(RELAX_ENV_VAR, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
