"""Query decomposition for multi-query retrieval.

This module provides zero-LLM query decomposition using pattern matching
and rule-based systems. Complex queries are broken into targeted sub-queries
for improved retrieval coverage.
"""

import re
from typing import Optional

from ._bloom_verbs import detect_bloom_level as _vendored_detect_bloom_level
from .query_decomposition import (
    INTENT_ASPECT_RULES,
    INTENT_CHUNK_TYPES,
    DecomposedQuery,
    QueryAspect,
    QueryIntent,
    SubQuery,
)


class QueryDecomposer:
    """Decompose complex queries into targeted sub-queries.

    Zero-LLM implementation using:
    - Regex pattern matching for intent detection
    - Rule-based aspect extraction
    - Bloom's taxonomy verb detection
    - Domain keyword heuristics

    Example:
        >>> decomposer = QueryDecomposer()
        >>> result = decomposer.decompose("How does UDL improve accessibility?")
        >>> result.primary_intent
        QueryIntent.PROCEDURE
        >>> len(result.sub_queries)
        3
    """

    # Intent detection patterns (case-insensitive)
    INTENT_PATTERNS = {
        QueryIntent.DEFINITION: [
            r'^what\s+is\s+',
            r'^define\s+',
            r'definition\s+of',
            r'^explain\s+what\s+',
            r'meaning\s+of',
            r'^describe\s+what\s+',
        ],
        QueryIntent.EXPLANATION: [
            r'^why\s+',
            r'^explain\s+why',
            r'reason\s+for',
            r'how\s+come',
            r'purpose\s+of',
            r'rationale\s+(for|behind)',
        ],
        QueryIntent.EXAMPLE: [
            r'^show\s+me',
            r'example\s+of',
            r'give\s+(an\s+)?example',
            r'illustrate',
            r'demonstrate',
            r'case\s+study',
        ],
        QueryIntent.PROCEDURE: [
            r'^how\s+(?:do|can|should|would|to)',
            r'steps\s+to',
            r'process\s+for',
            r'implement',
            r'procedure\s+for',
            r'method\s+(?:for|to)',
            r'^create\s+',
            r'^build\s+',
        ],
        QueryIntent.COMPARISON: [
            r'compare',
            r'difference\s+between',
            r'contrast',
            r'\bversus\b',
            r'\bvs\.?\b',
            r'similarities?\s+between',
            r'distinguish\s+between',
        ],
        QueryIntent.APPLICATION: [
            r'when\s+(?:should|to|would)',
            r'use\s+case',
            r'apply\s+',
            r'best\s+practice',
            r'in\s+practice',
        ],
        QueryIntent.ANALYSIS: [
            r'^analyze',
            r'^analyse',
            r'break\s+down',
            r'components?\s+of',
            r'examine',
            r'investigate',
        ],
        QueryIntent.SYNTHESIS: [
            r'combine',
            r'integrate',
            r'synthesize',
            r'design\s+a',
            r'develop\s+a',
            r'create\s+a\s+new',
        ],
    }

    # Domain keyword hints
    DOMAIN_KEYWORDS = {
        'instructional-design': [
            'addie', 'sam', 'instructional', 'learning design', 'curriculum',
            'assessment', 'objectives', 'outcomes',
        ],
        'accessibility': [
            'wcag', 'accessibility', 'accessible', 'a11y', 'screen reader',
            'alt text', 'aria', 'disability',
        ],
        'udl': [
            'udl', 'universal design', 'multiple means', 'representation',
            'engagement', 'action and expression',
        ],
        'pedagogy': [
            'pedagogy', 'teaching', 'bloom', 'taxonomy', 'scaffolding',
            'metacognition', 'cognitive load',
        ],
        'design': [
            'ux', 'ui', 'user experience', 'interface', 'visual design',
            'typography', 'layout', 'wireframe',
        ],
    }

    # Aspect-specific sub-query templates
    ASPECT_TEMPLATES = {
        QueryAspect.WHAT: [
            "{concept} definition",
            "what is {concept}",
            "{concept} meaning",
            "{concept} overview",
        ],
        QueryAspect.WHY: [
            "importance of {concept}",
            "purpose of {concept}",
            "why {concept} matters",
            "benefits of {concept}",
        ],
        QueryAspect.HOW: [
            "how to {concept}",
            "{concept} implementation",
            "{concept} process",
            "{concept} steps",
        ],
        QueryAspect.WHEN: [
            "when to use {concept}",
            "{concept} use cases",
            "{concept} applications",
            "appropriate use of {concept}",
        ],
        QueryAspect.EXAMPLES: [
            "{concept} examples",
            "{concept} case study",
            "{concept} in practice",
            "practical {concept}",
        ],
        QueryAspect.RELATED: [
            "{concept} related concepts",
            "concepts similar to {concept}",
            "{concept} connections",
            "{concept} dependencies",
        ],
    }

    def __init__(self):
        """Initialize the query decomposer."""
        # Compile regex patterns for efficiency
        self._compiled_patterns = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def decompose(self, query: str) -> DecomposedQuery:
        """Decompose a complex query into sub-queries.

        Args:
            query: Original user query

        Returns:
            DecomposedQuery with intent, sub-queries, and metadata
        """
        # Normalize query
        query = query.strip()

        # Detect primary intent
        primary_intent = self._detect_intent(query)

        # Detect Bloom's level
        bloom_level = self._detect_bloom_level(query)

        # Extract concepts from query
        concepts = self._extract_concepts(query)

        # Detect domain hints
        domain_hints = self._detect_domains(query)

        # Generate sub-queries based on intent and concepts
        sub_queries = self._generate_sub_queries(
            query=query,
            intent=primary_intent,
            concepts=concepts,
            bloom_level=bloom_level,
        )

        return DecomposedQuery(
            original_query=query,
            primary_intent=primary_intent,
            sub_queries=sub_queries,
            detected_concepts=concepts,
            bloom_level=bloom_level,
            domain_hints=domain_hints,
        )

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect primary intent using pattern matching.

        Args:
            query: User query

        Returns:
            Detected QueryIntent (defaults to EXPLANATION if no match)
        """
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent

        # Default to EXPLANATION for general queries
        return QueryIntent.EXPLANATION

    def _detect_bloom_level(self, query: str) -> Optional[str]:
        """Detect Bloom's taxonomy level from verb usage.

        Wave 55: delegates to the vendored canonical matcher
        (``LibV2/tools/libv2/_bloom_verbs.py::detect_bloom_level``) and
        discards the verb. The pre-Wave-55 local implementation used word-
        token set intersection, which lacked the longest-verb-first tie-
        breaking of the canonical matcher. Vendoring (rather than importing
        ``lib.ontology.bloom``) preserves LibV2's cross-package sandbox.

        Args:
            query: User query

        Returns:
            Bloom level string or None if not detected
        """
        level, _verb = _vendored_detect_bloom_level(query)
        return level

    def _extract_concepts(self, query: str) -> list[str]:
        """Extract key concepts/terms from query.

        Uses simple heuristics:
        - Quoted phrases
        - Capitalized terms
        - Known domain terms
        - Noun phrases after question words

        Args:
            query: User query

        Returns:
            List of extracted concept strings
        """
        concepts = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        concepts.extend(quoted)

        # Extract capitalized multi-word terms (e.g., "Universal Design for Learning")
        # But skip common question words
        skip_words = {'What', 'How', 'Why', 'When', 'Where', 'Which', 'Who', 'Give', 'Show', 'Compare', 'Define', 'Explain'}
        cap_phrases = re.findall(r'[A-Z][a-z]+(?:\s+(?:for|of|and|the|in)\s+[A-Z]?[a-z]+)*(?:\s+[A-Z][a-z]+)*', query)
        cap_phrases = [p for p in cap_phrases if p not in skip_words]
        concepts.extend(cap_phrases)

        # Extract terms after common question patterns
        after_patterns = [
            r'what\s+is\s+(.+?)(?:\?|$)',
            r'how\s+does\s+(.+?)\s+(?:work|function|operate)',
            r'compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
        ]

        for pattern in after_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.extend(m.strip() for m in match if m.strip())
                else:
                    concepts.append(match.strip())

        # Clean and deduplicate
        cleaned = []
        seen = set()
        for concept in concepts:
            concept = concept.strip().strip('?.,!')
            concept_lower = concept.lower()
            if concept and concept_lower not in seen and len(concept) > 2:
                cleaned.append(concept)
                seen.add(concept_lower)

        # If no concepts found, use significant words from query
        if not cleaned:
            words = query.split()
            # Filter out common words and question words
            stop_words = {
                'what', 'is', 'are', 'how', 'does', 'do', 'why', 'when',
                'where', 'which', 'the', 'a', 'an', 'and', 'or', 'but',
                'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                'can', 'could', 'should', 'would', 'will', 'may', 'might',
            }
            significant = [w for w in words if w.lower() not in stop_words and len(w) > 2]
            # Take up to 3 significant words as concept
            if significant:
                cleaned.append(' '.join(significant[:3]))

        return cleaned[:5]  # Limit to 5 concepts

    def _detect_domains(self, query: str) -> list[str]:
        """Detect relevant domains from query keywords.

        Args:
            query: User query

        Returns:
            List of domain hint strings
        """
        query_lower = query.lower()
        domains = []

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    domains.append(domain)
                    break

        return domains

    def _generate_sub_queries(
        self,
        query: str,
        intent: QueryIntent,
        concepts: list[str],
        bloom_level: Optional[str],
    ) -> list[SubQuery]:
        """Generate aspect-specific sub-queries.

        Args:
            query: Original query
            intent: Detected intent
            concepts: Extracted concepts
            bloom_level: Detected Bloom level

        Returns:
            List of SubQuery objects
        """
        sub_queries = []
        aspect_rules = INTENT_ASPECT_RULES.get(intent, [(QueryAspect.WHAT, 1.0)])

        # Get preferred chunk types for this intent
        chunk_types = INTENT_CHUNK_TYPES.get(intent, [])

        # Handle comparison intent specially (needs two concepts)
        if intent == QueryIntent.COMPARISON and len(concepts) >= 2:
            # First concept WHAT query
            sub_queries.append(SubQuery(
                text=f"{concepts[0]} definition overview",
                aspect=QueryAspect.WHAT,
                intent=intent,
                weight=0.4,
                chunk_types=['definition', 'concept', 'overview'],
                bloom_level=bloom_level,
            ))
            # Second concept WHAT query
            sub_queries.append(SubQuery(
                text=f"{concepts[1]} definition overview",
                aspect=QueryAspect.WHAT,
                intent=intent,
                weight=0.4,
                chunk_types=['definition', 'concept', 'overview'],
                bloom_level=bloom_level,
            ))
            # Comparison query
            sub_queries.append(SubQuery(
                text=f"{concepts[0]} {concepts[1]} comparison difference",
                aspect=QueryAspect.RELATED,
                intent=intent,
                weight=0.2,
                chunk_types=['comparison', 'analysis'],
                bloom_level=bloom_level,
            ))
            return sub_queries

        # For other intents, generate sub-queries per aspect
        primary_concept = concepts[0] if concepts else self._extract_fallback_concept(query)

        for aspect, weight in aspect_rules:
            # Get template for this aspect
            templates = self.ASPECT_TEMPLATES.get(aspect, ["{concept}"])
            template = templates[0]  # Use first template

            # Generate sub-query text
            sub_query_text = template.format(concept=primary_concept)

            # Determine chunk types for this aspect
            aspect_chunk_types = self._get_chunk_types_for_aspect(aspect, chunk_types)

            sub_queries.append(SubQuery(
                text=sub_query_text,
                aspect=aspect,
                intent=intent,
                weight=weight,
                chunk_types=aspect_chunk_types,
                bloom_level=bloom_level,
            ))

        return sub_queries

    def _extract_fallback_concept(self, query: str) -> str:
        """Extract a fallback concept when no concepts detected.

        Args:
            query: Original query

        Returns:
            A reasonable concept string from the query
        """
        # Remove question words and common verbs
        cleaned = re.sub(
            r'\b(what|is|are|how|does|do|why|when|where|which|the|a|an|'
            r'can|could|should|would|will|may|might)\b',
            '',
            query,
            flags=re.IGNORECASE
        )
        cleaned = cleaned.strip().strip('?.,!')

        # Take first 4 words
        words = cleaned.split()[:4]
        return ' '.join(words) if words else query[:50]

    def _get_chunk_types_for_aspect(
        self,
        aspect: QueryAspect,
        intent_chunk_types: list[str],
    ) -> list[str]:
        """Get appropriate chunk types for an aspect.

        Args:
            aspect: Query aspect
            intent_chunk_types: Chunk types from intent

        Returns:
            Combined list of chunk types
        """
        aspect_types = {
            QueryAspect.WHAT: ['definition', 'concept', 'overview'],
            QueryAspect.WHY: ['explanation', 'rationale', 'theory'],
            QueryAspect.HOW: ['procedure', 'steps', 'tutorial', 'how_to'],
            QueryAspect.WHEN: ['application', 'use_case', 'context'],
            QueryAspect.EXAMPLES: ['example', 'case_study', 'illustration'],
            QueryAspect.RELATED: ['concept', 'overview', 'connection'],
        }

        aspect_specific = aspect_types.get(aspect, [])

        # Combine with intent types, prioritizing aspect-specific
        combined = list(aspect_specific)
        for ct in intent_chunk_types:
            if ct not in combined:
                combined.append(ct)

        return combined[:5]  # Limit to 5 types
