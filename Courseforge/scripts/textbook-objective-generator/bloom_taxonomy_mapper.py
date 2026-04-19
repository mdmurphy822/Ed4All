"""
Bloom's Taxonomy Mapper Module

Maps content types and patterns to Bloom's taxonomy levels.
Provides action verbs and objective templates for each level.

Equal Treatment Principle: This module does NOT filter or rank importance.
All extracted content is treated equally and mapped to appropriate Bloom's levels.
"""

import random
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is importable so lib.ontology.bloom resolves when
# this module is run from inside Courseforge/ (often invoked directly by
# script name, not as a package).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.ontology.bloom import get_verb_objects as _get_canonical_verb_objects  # noqa: E402


class BloomLevel(Enum):
    """Bloom's taxonomy cognitive levels (revised)."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

    @property
    def order(self) -> int:
        """Cognitive complexity order (1=lowest, 6=highest)."""
        order_map = {
            BloomLevel.REMEMBER: 1,
            BloomLevel.UNDERSTAND: 2,
            BloomLevel.APPLY: 3,
            BloomLevel.ANALYZE: 4,
            BloomLevel.EVALUATE: 5,
            BloomLevel.CREATE: 6,
        }
        return order_map[self]


@dataclass
class BloomVerb:
    """An action verb associated with a Bloom's level."""
    verb: str
    level: BloomLevel
    usage_context: str  # brief description of when to use
    example_template: str  # template for generating objectives


# Comprehensive verb mappings with usage contexts.
#
# Source of truth: schemas/taxonomies/bloom_verbs.json (loaded via
# lib.ontology.bloom). This module's local BloomVerb dataclass carries an
# extra `level` field that the ontology BloomVerb omits (redundant with the
# dict key). The builder below bridges the two shapes so downstream imports
# (see objective_formatter.py, __init__.py) continue to see
# Dict[BloomLevel, List[BloomVerb]] with the richer local dataclass.
# Migrated in Wave 1.2 / Worker H (REC-BL-01).
def _build_bloom_verbs() -> Dict[BloomLevel, List[BloomVerb]]:
    canonical = _get_canonical_verb_objects()
    return {
        BloomLevel(level): [
            BloomVerb(
                verb=entry.verb,
                level=BloomLevel(level),
                usage_context=entry.usage_context,
                example_template=entry.example_template,
            )
            for entry in canonical[level]
        ]
        for level in ("remember", "understand", "apply", "analyze", "evaluate", "create")
    }


BLOOM_VERBS: Dict[BloomLevel, List[BloomVerb]] = _build_bloom_verbs()


class BloomTaxonomyMapper:
    """
    Maps content to Bloom's taxonomy levels.

    Equal Treatment: All content is mapped without filtering.
    The mapper determines appropriate cognitive levels but does not
    exclude any content based on perceived importance.
    """

    # Content type to default Bloom's level mapping
    CONTENT_TYPE_DEFAULTS: Dict[str, BloomLevel] = {
        # Definitions default to Remember
        "definition": BloomLevel.REMEMBER,
        "term": BloomLevel.REMEMBER,
        "glossary": BloomLevel.REMEMBER,

        # Explanations default to Understand
        "explanation": BloomLevel.UNDERSTAND,
        "description": BloomLevel.UNDERSTAND,
        "concept": BloomLevel.UNDERSTAND,
        "summary": BloomLevel.UNDERSTAND,

        # Procedures default to Apply
        "procedure": BloomLevel.APPLY,
        "steps": BloomLevel.APPLY,
        "how_to": BloomLevel.APPLY,
        "example": BloomLevel.APPLY,

        # Analysis content defaults to Analyze
        "comparison": BloomLevel.ANALYZE,
        "relationship": BloomLevel.ANALYZE,
        "structure": BloomLevel.ANALYZE,

        # Assessment content defaults to Evaluate
        "evaluation": BloomLevel.EVALUATE,
        "criteria": BloomLevel.EVALUATE,
        "judgment": BloomLevel.EVALUATE,

        # Creative content defaults to Create
        "design": BloomLevel.CREATE,
        "solution": BloomLevel.CREATE,
        "synthesis": BloomLevel.CREATE,
    }

    # Patterns that suggest higher-order thinking
    HIGHER_ORDER_PATTERNS = {
        BloomLevel.ANALYZE: [
            r'\b(relationship|structure|component|element|factor|cause|effect)\b',
            r'\b(how|why)\s+(?:does|do|is|are)\b',
            r'\b(compare|contrast|analyze)\b',
        ],
        BloomLevel.EVALUATE: [
            r'\b(best|worst|optimal|effective|efficient)\b',
            r'\b(advantage|disadvantage|pro|con|benefit|drawback)\b',
            r'\b(should|recommend|prefer)\b',
        ],
        BloomLevel.CREATE: [
            r'\b(design|develop|create|build|construct)\b',
            r'\b(plan|strategy|approach)\b',
            r'\b(new|novel|innovative)\b',
        ],
    }

    def __init__(self):
        # Build a flat list of all verbs for quick lookup
        self._verb_to_level: Dict[str, BloomLevel] = {}
        for level, verbs in BLOOM_VERBS.items():
            for verb in verbs:
                self._verb_to_level[verb.verb.lower()] = level

    def map_content_type(self, content_type: str) -> BloomLevel:
        """
        Map a content type to its default Bloom's level.

        Args:
            content_type: Type of content (e.g., "definition", "procedure")

        Returns:
            Appropriate BloomLevel
        """
        return self.CONTENT_TYPE_DEFAULTS.get(
            content_type.lower(),
            BloomLevel.UNDERSTAND  # Default
        )

    def analyze_text_complexity(self, text: str) -> BloomLevel:
        """
        Analyze text to determine suggested Bloom's level.

        Uses pattern matching to detect indicators of cognitive complexity.
        Does NOT filter content - only suggests appropriate level.

        Args:
            text: Text content to analyze

        Returns:
            Suggested BloomLevel
        """
        text_lower = text.lower()

        # Check for higher-order patterns first
        for level in [BloomLevel.CREATE, BloomLevel.EVALUATE, BloomLevel.ANALYZE]:
            patterns = self.HIGHER_ORDER_PATTERNS.get(level, [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level

        # Check for explicit verbs
        words = text_lower.split()
        for word in words[:10]:  # Check first 10 words
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self._verb_to_level:
                return self._verb_to_level[clean_word]

        # Default based on text characteristics
        if len(words) < 10:
            return BloomLevel.REMEMBER
        elif len(words) < 30:
            return BloomLevel.UNDERSTAND
        else:
            return BloomLevel.UNDERSTAND

    def get_verbs_for_level(self, level: BloomLevel) -> List[BloomVerb]:
        """Get all action verbs for a Bloom's level."""
        return BLOOM_VERBS.get(level, [])

    def get_verb(self, level: BloomLevel, context: Optional[str] = None) -> BloomVerb:
        """
        Get an appropriate verb for a Bloom's level.

        Args:
            level: The Bloom's taxonomy level
            context: Optional context hint to select best verb

        Returns:
            A BloomVerb object
        """
        verbs = self.get_verbs_for_level(level)

        if not verbs:
            # Fallback
            return BloomVerb("understand", BloomLevel.UNDERSTAND, "general", "Understand {concept}")

        if context:
            # Try to match context
            context_lower = context.lower()
            for verb in verbs:
                if verb.usage_context.lower() in context_lower or context_lower in verb.usage_context.lower():
                    return verb

        # Return a random verb for variety
        return random.choice(verbs)

    def suggest_level_for_definition(self) -> BloomLevel:
        """Suggest Bloom's level for a definition."""
        return BloomLevel.REMEMBER

    def suggest_level_for_concept(self, has_example: bool = False) -> BloomLevel:
        """
        Suggest Bloom's level for a concept.

        Args:
            has_example: Whether the concept includes an example

        Returns:
            BloomLevel
        """
        if has_example:
            return BloomLevel.UNDERSTAND
        return BloomLevel.UNDERSTAND

    def suggest_level_for_procedure(self, step_count: int) -> BloomLevel:
        """
        Suggest Bloom's level for a procedure.

        Args:
            step_count: Number of steps in the procedure

        Returns:
            BloomLevel
        """
        return BloomLevel.APPLY

    def suggest_level_for_review_question(self, question_text: str) -> BloomLevel:
        """
        Suggest Bloom's level for a review question.

        Analyzes the question text to determine cognitive level.

        Args:
            question_text: The review question text

        Returns:
            BloomLevel
        """
        return self.analyze_text_complexity(question_text)

    def get_level_distribution_recommendation(
        self,
        total_objectives: int
    ) -> Dict[BloomLevel, int]:
        """
        Get recommended distribution of objectives across Bloom's levels.

        Based on educational best practices:
        - Remember/Understand: ~30% (foundational)
        - Apply/Analyze: ~50% (core)
        - Evaluate/Create: ~20% (advanced)

        Args:
            total_objectives: Total number of objectives to distribute

        Returns:
            Dictionary mapping levels to recommended counts
        """
        distribution = {
            BloomLevel.REMEMBER: 0.10,
            BloomLevel.UNDERSTAND: 0.20,
            BloomLevel.APPLY: 0.30,
            BloomLevel.ANALYZE: 0.20,
            BloomLevel.EVALUATE: 0.12,
            BloomLevel.CREATE: 0.08,
        }

        result = {}
        remaining = total_objectives
        for level, ratio in distribution.items():
            count = int(total_objectives * ratio)
            result[level] = count
            remaining -= count

        # Distribute remaining to Apply
        result[BloomLevel.APPLY] += remaining

        return result


def get_bloom_verbs(level: str) -> List[str]:
    """
    Convenience function to get verb strings for a level.

    Args:
        level: Bloom's level name (e.g., "remember", "understand")

    Returns:
        List of verb strings
    """
    try:
        bloom_level = BloomLevel(level.lower())
        return [v.verb for v in BLOOM_VERBS.get(bloom_level, [])]
    except ValueError:
        return []


def suggest_bloom_level(content_type: str, text: str = "") -> str:
    """
    Convenience function to suggest a Bloom's level.

    Args:
        content_type: Type of content
        text: Optional text to analyze

    Returns:
        Bloom's level name string
    """
    mapper = BloomTaxonomyMapper()

    if text:
        level = mapper.analyze_text_complexity(text)
    else:
        level = mapper.map_content_type(content_type)

    return level.value
