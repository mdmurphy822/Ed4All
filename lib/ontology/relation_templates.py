"""Single source of truth for KG-relation training/eval templates.

The kg_metadata generator (`Trainforge/generators/kg_metadata_generator.py`)
emits ``(positive, negative)`` tuples per relation; the faithfulness
evaluator (`Trainforge/eval/faithfulness.py`) probes the positive form.

Drift between train-time and eval-time relation strings would desync
the adapter's training signal from the eval probe, so both consumers
import from this canonical map. Wave 132a consolidated the two
duplicated copies (faithfulness.py:51-68 + kg_metadata_generator.py:62-79)
into this module.

The ``negative_template`` slot is reserved for future expansion: the
kg_metadata generator currently uses it as a structural placeholder
(question text is identical to the positive form; polarity is encoded
via the answer "Yes." / "No.", not the prompt). New consumers that
want a prompt-level negation surface can populate it without breaking
existing call sites.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["RELATION_TEMPLATES"]


# (positive_template, negative_template) keyed by relation name.
# Bytewise-aligned across train (kg_metadata_generator) + eval
# (faithfulness) — see module docstring for rationale.
#
# The current 12 entries are the union of:
#   - faithfulness.py:51-68 (12 templates: prerequisite_of, teaches,
#     interferes_with, concept_supports_outcome,
#     derived_from_objective, exemplifies, assesses, supports_outcome,
#     follows, belongs_to_module, at_bloom_level,
#     assessment_validates_outcome).
#   - kg_metadata_generator.py:62-79 (3 templates: assesses,
#     belongs_to_module, at_bloom_level — all subsumed by faithfulness).
#
# Wave 108 / Phase B dropped ``chunk_at_difficulty`` (trivially-true
# probe). Held-out edges of that type fall through to the generic
# template at both call sites.
RELATION_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "prerequisite_of": (
        "Is the concept '{source}' a prerequisite for the concept '{target}'?",
        "Is the concept '{source}' a prerequisite for the concept '{target}'?",
    ),
    "teaches": (
        "Does the chunk '{source}' teach the concept '{target}'?",
        "Does the chunk '{source}' teach the concept '{target}'?",
    ),
    "interferes_with": (
        "Does the misconception '{source}' interfere with the concept '{target}'?",
        "Does the misconception '{source}' interfere with the concept '{target}'?",
    ),
    "concept_supports_outcome": (
        "Does the concept '{source}' support the learning outcome '{target}'?",
        "Does the concept '{source}' support the learning outcome '{target}'?",
    ),
    "derived_from_objective": (
        "Is the concept '{source}' derived from the objective '{target}'?",
        "Is the concept '{source}' derived from the objective '{target}'?",
    ),
    "exemplifies": (
        "Does the chunk '{source}' exemplify the concept '{target}'?",
        "Does the chunk '{source}' exemplify the concept '{target}'?",
    ),
    "assesses": (
        "Does the assessment '{source}' assess the concept '{target}'?",
        "Does the assessment '{source}' assess the concept '{target}'?",
    ),
    "supports_outcome": (
        "Does the component objective '{source}' support the terminal outcome '{target}'?",
        "Does the component objective '{source}' support the terminal outcome '{target}'?",
    ),
    "follows": (
        "Does '{source}' follow '{target}' in the curriculum order?",
        "Does '{source}' follow '{target}' in the curriculum order?",
    ),
    "belongs_to_module": (
        "Does the chunk '{source}' belong to the module '{target}'?",
        "Does the chunk '{source}' belong to the module '{target}'?",
    ),
    "at_bloom_level": (
        "Is the chunk '{source}' at Bloom level '{target}'?",
        "Is the chunk '{source}' at Bloom level '{target}'?",
    ),
    "assessment_validates_outcome": (
        "Does the assessment '{source}' validate the outcome '{target}'?",
        "Does the assessment '{source}' validate the outcome '{target}'?",
    ),
}
