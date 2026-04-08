"""
Textbook Objective Generator Package

Generates learning objectives from textbook structure,
following the Equal Treatment Principle.
"""

from .bloom_taxonomy_mapper import (
    BLOOM_VERBS,
    BloomLevel,
    BloomTaxonomyMapper,
    BloomVerb,
    get_bloom_verbs,
    suggest_bloom_level,
)
from .objective_formatter import LearningObjective, ObjectiveFormatter
from .textbook_objective_generator import (
    ChapterObjectives,
    SectionObjectives,
    TextbookObjectiveGenerator,
    generate_objectives,
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
