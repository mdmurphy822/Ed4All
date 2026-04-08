"""Query decomposition dataclasses for multi-query retrieval.

This module defines the data structures for breaking complex queries
into targeted sub-queries. All processing is zero-LLM using pattern matching.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryIntent(Enum):
    """Classification of query intent for targeted retrieval."""
    DEFINITION = "definition"       # "What is X?"
    EXPLANATION = "explanation"     # "Why does X work?"
    EXAMPLE = "example"             # "Show me examples of X"
    PROCEDURE = "procedure"         # "How do I do X?"
    COMPARISON = "comparison"       # "Compare X and Y"
    APPLICATION = "application"     # "When should I use X?"
    ANALYSIS = "analysis"           # "Analyze X"
    SYNTHESIS = "synthesis"         # "Combine X with Y"


class QueryAspect(Enum):
    """Aspect-specific sub-query dimensions."""
    WHAT = "what"           # Core definition/concept
    WHY = "why"             # Rationale, motivation
    HOW = "how"             # Process, procedure
    WHEN = "when"           # Timing, conditions
    EXAMPLES = "examples"   # Practical illustrations
    RELATED = "related"     # Connected concepts


@dataclass
class SubQuery:
    """A decomposed sub-query with metadata.

    Attributes:
        text: The sub-query string to execute
        aspect: Which aspect this sub-query addresses
        intent: Intent classification inherited from parent query
        weight: Importance weight for result fusion (0.0-1.0)
        chunk_types: Preferred chunk types to filter for
        bloom_level: Target Bloom's taxonomy level if applicable
    """
    text: str
    aspect: QueryAspect
    intent: QueryIntent
    weight: float = 1.0
    chunk_types: list[str] = field(default_factory=list)
    bloom_level: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "aspect": self.aspect.value,
            "intent": self.intent.value,
            "weight": self.weight,
            "chunk_types": self.chunk_types,
            "bloom_level": self.bloom_level,
        }


@dataclass
class DecomposedQuery:
    """Result of query decomposition.

    Attributes:
        original_query: The original user query
        primary_intent: Main query intent detected
        sub_queries: List of decomposed sub-queries
        detected_concepts: Key concepts extracted from query
        bloom_level: Detected Bloom's taxonomy level
        domain_hints: Suggested domains for filtering
    """
    original_query: str
    primary_intent: QueryIntent
    sub_queries: list[SubQuery]
    detected_concepts: list[str] = field(default_factory=list)
    bloom_level: Optional[str] = None
    domain_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "original_query": self.original_query,
            "primary_intent": self.primary_intent.value,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "detected_concepts": self.detected_concepts,
            "bloom_level": self.bloom_level,
            "domain_hints": self.domain_hints,
        }

    @property
    def query_count(self) -> int:
        """Number of sub-queries generated."""
        return len(self.sub_queries)

    def get_queries_by_aspect(self, aspect: QueryAspect) -> list[SubQuery]:
        """Get all sub-queries for a specific aspect."""
        return [sq for sq in self.sub_queries if sq.aspect == aspect]

    def get_weighted_queries(self) -> list[tuple[SubQuery, float]]:
        """Get sub-queries with their weights, sorted by weight descending."""
        return sorted(
            [(sq, sq.weight) for sq in self.sub_queries],
            key=lambda x: x[1],
            reverse=True
        )


# Bloom's taxonomy level order for validation
BLOOM_LEVELS = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']


# Intent to preferred chunk types mapping
INTENT_CHUNK_TYPES = {
    QueryIntent.DEFINITION: ['definition', 'concept', 'overview'],
    QueryIntent.EXPLANATION: ['explanation', 'concept', 'theory'],
    QueryIntent.EXAMPLE: ['example', 'case_study', 'application'],
    QueryIntent.PROCEDURE: ['procedure', 'steps', 'how_to', 'tutorial'],
    QueryIntent.COMPARISON: ['comparison', 'concept', 'analysis'],
    QueryIntent.APPLICATION: ['application', 'example', 'case_study'],
    QueryIntent.ANALYSIS: ['analysis', 'evaluation', 'concept'],
    QueryIntent.SYNTHESIS: ['synthesis', 'project', 'design'],
}


# Aspect to sub-query weight defaults
ASPECT_WEIGHTS = {
    QueryAspect.WHAT: 0.8,
    QueryAspect.WHY: 0.6,
    QueryAspect.HOW: 0.7,
    QueryAspect.WHEN: 0.4,
    QueryAspect.EXAMPLES: 0.5,
    QueryAspect.RELATED: 0.3,
}


# Intent to aspect generation rules
INTENT_ASPECT_RULES = {
    QueryIntent.DEFINITION: [
        (QueryAspect.WHAT, 0.8),
        (QueryAspect.RELATED, 0.2),
    ],
    QueryIntent.EXPLANATION: [
        (QueryAspect.WHY, 0.6),
        (QueryAspect.WHAT, 0.3),
        (QueryAspect.EXAMPLES, 0.1),
    ],
    QueryIntent.EXAMPLE: [
        (QueryAspect.EXAMPLES, 0.7),
        (QueryAspect.HOW, 0.2),
        (QueryAspect.WHAT, 0.1),
    ],
    QueryIntent.PROCEDURE: [
        (QueryAspect.HOW, 0.6),
        (QueryAspect.WHAT, 0.2),
        (QueryAspect.EXAMPLES, 0.2),
    ],
    QueryIntent.COMPARISON: [
        (QueryAspect.WHAT, 0.4),  # Applied to first concept
        (QueryAspect.WHAT, 0.4),  # Applied to second concept
        (QueryAspect.RELATED, 0.2),
    ],
    QueryIntent.APPLICATION: [
        (QueryAspect.WHEN, 0.4),
        (QueryAspect.HOW, 0.3),
        (QueryAspect.EXAMPLES, 0.3),
    ],
    QueryIntent.ANALYSIS: [
        (QueryAspect.WHAT, 0.3),
        (QueryAspect.WHY, 0.3),
        (QueryAspect.HOW, 0.2),
        (QueryAspect.RELATED, 0.2),
    ],
    QueryIntent.SYNTHESIS: [
        (QueryAspect.HOW, 0.4),
        (QueryAspect.WHAT, 0.3),
        (QueryAspect.EXAMPLES, 0.3),
    ],
}
