"""
Textbook Objective Generator Package

Generates learning objectives from textbook structure,
following the Equal Treatment Principle.
"""

from .bloom_taxonomy_mapper import (
    BloomLevel,
    BloomTaxonomyMapper,
    BloomVerb,
    BLOOM_VERBS,
    get_bloom_verbs,
    suggest_bloom_level
)

from .objective_formatter import (
    ObjectiveFormatter,
    LearningObjective
)

from .textbook_objective_generator import (
    TextbookObjectiveGenerator,
    ChapterObjectives,
    SectionObjectives,
    generate_objectives
)

__version__ = "1.0.0"

__all__ = [
    # Bloom's Taxonomy
    "BloomLevel",
    "BloomTaxonomyMapper",
    "BloomVerb",
    "BLOOM_VERBS",
    "get_bloom_verbs",
    "suggest_bloom_level",
    # Objective Formatter
    "ObjectiveFormatter",
    "LearningObjective",
    # Generator
    "TextbookObjectiveGenerator",
    "ChapterObjectives",
    "SectionObjectives",
    "generate_objectives",
]
