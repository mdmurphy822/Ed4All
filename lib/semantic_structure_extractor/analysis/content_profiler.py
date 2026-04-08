"""
Content Profiler for Semantic Structure Extraction

Analyzes content blocks for difficulty level, concept extraction, and
educational metadata. Implements RAG library patterns for content profiling:
- Difficulty assessment (vocabulary, sentence complexity, concept density)
- Concept extraction with frequency tracking
- Readability scoring (Flesch-Kincaid)
- Pedagogical pattern detection
- Content type classification
"""

import re
import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import Counter, defaultdict
from enum import Enum


class ContentType(Enum):
    """Types of educational content."""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    PROCEDURE = "procedure"
    EXERCISE = "exercise"
    ASSESSMENT = "assessment"
    NARRATIVE = "narrative"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class DifficultyLevel(Enum):
    """Content difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PedagogicalPattern(Enum):
    """Pedagogical patterns in content structure."""
    PROBLEM_BASED = "problem_based"
    SCAFFOLDED = "scaffolded"
    SPIRAL_REVIEW = "spiral_review"
    ASSESSMENT_DRIVEN = "assessment_driven"
    DIRECT_INSTRUCTION = "direct_instruction"
    UNKNOWN = "unknown"


@dataclass
class ConceptReference:
    """A concept extracted from content."""
    term: str
    normalized_term: str
    frequency: int = 1
    positions: List[int] = field(default_factory=list)
    context_snippets: List[str] = field(default_factory=list)
    is_key_concept: bool = False
    section_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "normalizedTerm": self.normalized_term,
            "frequency": self.frequency,
            "positions": self.positions,
            "contextSnippets": self.context_snippets[:3],  # Limit snippets
            "isKeyConcept": self.is_key_concept,
            "sectionIds": self.section_ids
        }


@dataclass
class ContentProfile:
    """Profile of a content block or section."""
    difficulty_score: float = 0.0
    difficulty_level: DifficultyLevel = DifficultyLevel.BEGINNER
    concepts: List[ConceptReference] = field(default_factory=list)
    token_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    readability_score: float = 0.0
    content_type: ContentType = ContentType.UNKNOWN
    pedagogical_pattern: PedagogicalPattern = PedagogicalPattern.UNKNOWN
    technical_term_density: float = 0.0
    concept_density: float = 0.0
    bloom_level_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "difficultyScore": round(self.difficulty_score, 3),
            "difficultyLevel": self.difficulty_level.value,
            "concepts": [c.to_dict() for c in self.concepts],
            "tokenCount": self.token_count,
            "wordCount": self.word_count,
            "sentenceCount": self.sentence_count,
            "avgSentenceLength": round(self.avg_sentence_length, 2),
            "avgWordLength": round(self.avg_word_length, 2),
            "readabilityScore": round(self.readability_score, 2),
            "contentType": self.content_type.value,
            "pedagogicalPattern": self.pedagogical_pattern.value,
            "technicalTermDensity": round(self.technical_term_density, 3),
            "conceptDensity": round(self.concept_density, 3),
            "bloomLevelDistribution": self.bloom_level_distribution
        }


@dataclass
class SectionProfile:
    """Aggregated profile for a section."""
    section_id: str
    block_profiles: List[ContentProfile] = field(default_factory=list)
    aggregate_profile: Optional[ContentProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sectionId": self.section_id,
            "blockProfiles": [p.to_dict() for p in self.block_profiles],
            "aggregateProfile": self.aggregate_profile.to_dict() if self.aggregate_profile else None
        }


class ContentProfiler:
    """
    Analyzes content for difficulty, concepts, and educational metadata.

    Uses multiple signals to assess content complexity and extract
    meaningful concepts for presentation optimization.
    """

    # Common English stopwords (extended)
    STOPWORDS: Set[str] = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'however', 'therefore', 'thus', 'hence', 'then', 'here', 'there'
    }

    # Content type indicator patterns
    CONTENT_TYPE_PATTERNS: Dict[ContentType, List[str]] = {
        ContentType.DEFINITION: [
            r'\b(is defined as|means|refers to|is a|are a)\b',
            r'\b(definition|meaning|concept)\b',
            r':\s*[A-Z]',  # Colon followed by definition
        ],
        ContentType.EXPLANATION: [
            r'\b(because|therefore|thus|as a result|consequently)\b',
            r'\b(explains?|understanding|reason|cause)\b',
        ],
        ContentType.EXAMPLE: [
            r'\b(for example|for instance|such as|e\.g\.|i\.e\.)\b',
            r'\b(example|illustration|case|sample)\b',
        ],
        ContentType.PROCEDURE: [
            r'\b(step \d+|first|second|third|finally|then|next)\b',
            r'\b(procedure|process|method|instructions?|how to)\b',
        ],
        ContentType.EXERCISE: [
            r'\b(practice|try|exercise|activity|task)\b',
            r'\b(complete|solve|answer|work through)\b',
        ],
        ContentType.ASSESSMENT: [
            r'\b(quiz|test|exam|assessment|check)\b',
            r'\b(question|evaluate|measure|score)\b',
        ],
    }

    # Bloom's taxonomy verb patterns for difficulty estimation
    BLOOM_PATTERNS: Dict[str, List[str]] = {
        'remember': ['define', 'list', 'recall', 'identify', 'name', 'state', 'label', 'match', 'recognize'],
        'understand': ['explain', 'describe', 'summarize', 'classify', 'compare', 'interpret', 'discuss'],
        'apply': ['demonstrate', 'implement', 'solve', 'use', 'execute', 'apply', 'compute', 'calculate'],
        'analyze': ['analyze', 'differentiate', 'examine', 'distinguish', 'organize'],
        'evaluate': ['evaluate', 'assess', 'critique', 'justify', 'judge', 'argue', 'defend'],
        'create': ['create', 'design', 'construct', 'develop', 'formulate', 'compose', 'plan']
    }

    # Bloom level weights for difficulty calculation
    BLOOM_DIFFICULTY_WEIGHTS: Dict[str, float] = {
        'remember': 0.1,
        'understand': 0.25,
        'apply': 0.5,
        'analyze': 0.7,
        'evaluate': 0.85,
        'create': 1.0
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the profiler.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        self._extend_stopwords()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        config_path = Path(__file__).parent / "config" / "extractor_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _extend_stopwords(self) -> None:
        """Extend stopwords from config."""
        extend = self.config.get('concept_extraction', {}).get('stopwords_extend', [])
        self.STOPWORDS.update(extend)

    def profile_text(self, text: str, section_id: str = "") -> ContentProfile:
        """
        Profile a text block.

        Args:
            text: Text content to profile
            section_id: Optional section identifier

        Returns:
            ContentProfile with analysis results
        """
        profile = ContentProfile()

        # Basic metrics
        words = self._tokenize(text)
        sentences = self._split_sentences(text)

        profile.word_count = len(words)
        profile.token_count = len(text.split())
        profile.sentence_count = len(sentences)

        if words:
            profile.avg_word_length = sum(len(w) for w in words) / len(words)
        if sentences:
            profile.avg_sentence_length = profile.word_count / len(sentences)

        # Readability score
        profile.readability_score = self._calculate_readability(text, words, sentences)

        # Technical term density
        technical_terms = self._extract_technical_terms(text)
        if profile.word_count > 0:
            profile.technical_term_density = len(technical_terms) / profile.word_count

        # Concept extraction
        profile.concepts = self._extract_concepts(text, section_id)
        if profile.word_count > 0:
            profile.concept_density = len(profile.concepts) / (profile.word_count / 100)

        # Bloom's level distribution
        profile.bloom_level_distribution = self._analyze_bloom_levels(text)

        # Content type detection
        profile.content_type = self._detect_content_type(text)

        # Difficulty calculation
        profile.difficulty_score = self._calculate_difficulty(profile)
        profile.difficulty_level = self._score_to_level(profile.difficulty_score)

        return profile

    def profile_block(self, block: Dict[str, Any], section_id: str = "") -> ContentProfile:
        """
        Profile a content block from semantic extractor.

        Args:
            block: Content block dictionary
            section_id: Section identifier

        Returns:
            ContentProfile with analysis results
        """
        # Extract text from various block structures
        text = ""
        if isinstance(block, dict):
            text = block.get('content', '') or block.get('text', '')
            if not text and 'items' in block:
                text = ' '.join(block['items'])
        elif isinstance(block, str):
            text = block

        return self.profile_text(text, section_id)

    def profile_section(self, section: Dict[str, Any]) -> SectionProfile:
        """
        Profile an entire section including all content blocks.

        Args:
            section: Section dictionary with content blocks

        Returns:
            SectionProfile with block and aggregate profiles
        """
        section_id = section.get('id', '')
        section_profile = SectionProfile(section_id=section_id)

        # Profile each content block
        blocks = section.get('contentBlocks', [])
        for block in blocks:
            block_profile = self.profile_block(block, section_id)
            section_profile.block_profiles.append(block_profile)

        # Calculate aggregate profile
        section_profile.aggregate_profile = self._aggregate_profiles(
            section_profile.block_profiles,
            section_id
        )

        return section_profile

    def detect_pedagogical_pattern(self, sections: List[Dict[str, Any]]) -> PedagogicalPattern:
        """
        Detect the overall pedagogical pattern across sections.

        Args:
            sections: List of section dictionaries

        Returns:
            Detected pedagogical pattern
        """
        config = self.config.get('pedagogical_patterns', {})
        if not config.get('enabled', True):
            return PedagogicalPattern.UNKNOWN

        patterns = config.get('patterns', {})
        pattern_scores: Dict[str, int] = defaultdict(int)

        # Combine all text
        all_text = ""
        for section in sections:
            all_text += section.get('headingText', '') + " "
            for block in section.get('contentBlocks', []):
                if isinstance(block, dict):
                    all_text += block.get('content', '') + " "

        all_text = all_text.lower()

        # Score each pattern
        for pattern_name, pattern_config in patterns.items():
            indicators = pattern_config.get('indicators', [])
            for indicator in indicators:
                if indicator.lower() in all_text:
                    pattern_scores[pattern_name] += 1

        # Find highest scoring pattern
        if pattern_scores:
            best_pattern = max(pattern_scores.keys(), key=lambda k: pattern_scores[k])
            if pattern_scores[best_pattern] >= 2:  # Minimum threshold
                try:
                    return PedagogicalPattern(best_pattern)
                except ValueError:
                    pass

        return PedagogicalPattern.DIRECT_INSTRUCTION

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """
        Calculate Flesch-Kincaid readability score.

        Score interpretation:
        - 90-100: Very easy (5th grade)
        - 80-90: Easy (6th grade)
        - 70-80: Fairly easy (7th grade)
        - 60-70: Standard (8th-9th grade)
        - 50-60: Fairly difficult (10th-12th grade)
        - 30-50: Difficult (college)
        - 0-30: Very difficult (professional)
        """
        if not words or not sentences:
            return 0.0

        # Count syllables
        total_syllables = sum(self._count_syllables(word) for word in words)

        word_count = len(words)
        sentence_count = len(sentences)

        # Flesch Reading Ease formula
        asl = word_count / sentence_count  # Average sentence length
        asw = total_syllables / word_count  # Average syllables per word

        score = 206.835 - (1.015 * asl) - (84.6 * asw)

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1

        return max(1, count)

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        technical_terms = []

        patterns = self.config.get('content_profiling', {}).get('technical_term_patterns', [
            r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b',  # CamelCase
            r'\b[a-z]+_[a-z]+\b',  # snake_case
            r'\b[A-Z]{2,}\b',  # ACRONYMS
        ])

        for pattern in patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)

        return technical_terms

    def _extract_concepts(self, text: str, section_id: str = "") -> List[ConceptReference]:
        """
        Extract concepts from text.

        Args:
            text: Text to analyze
            section_id: Section identifier

        Returns:
            List of ConceptReference objects
        """
        config = self.config.get('concept_extraction', {})
        min_freq = config.get('min_term_frequency', 2)
        max_concepts = config.get('max_concepts', 50)

        # Tokenize and filter
        words = self._tokenize(text)
        filtered_words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]

        # Count frequencies
        word_freq = Counter(filtered_words)

        # Extract n-grams (bigrams for now)
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in self.STOPWORDS and words[i+1] not in self.STOPWORDS:
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)

        bigram_freq = Counter(bigrams)

        # Combine and create concepts
        concepts = []

        # Single word concepts
        for word, freq in word_freq.most_common(max_concepts):
            if freq >= min_freq:
                # Find context
                contexts = self._find_context_snippets(text, word)
                concepts.append(ConceptReference(
                    term=word,
                    normalized_term=word.lower(),
                    frequency=freq,
                    context_snippets=contexts,
                    section_ids=[section_id] if section_id else []
                ))

        # Bigram concepts (if significant)
        for bigram, freq in bigram_freq.most_common(max_concepts // 2):
            if freq >= min_freq:
                contexts = self._find_context_snippets(text, bigram)
                concepts.append(ConceptReference(
                    term=bigram,
                    normalized_term=bigram.lower(),
                    frequency=freq,
                    context_snippets=contexts,
                    section_ids=[section_id] if section_id else []
                ))

        # Sort by frequency
        concepts.sort(key=lambda c: c.frequency, reverse=True)

        # Mark key concepts (top 20%)
        key_count = max(1, len(concepts) // 5)
        for concept in concepts[:key_count]:
            concept.is_key_concept = True

        return concepts[:max_concepts]

    def _find_context_snippets(self, text: str, term: str, max_snippets: int = 3) -> List[str]:
        """Find context snippets for a term."""
        snippets = []
        pattern = re.compile(r'[^.]*\b' + re.escape(term) + r'\b[^.]*\.?', re.IGNORECASE)

        for match in pattern.finditer(text):
            snippet = match.group().strip()
            if len(snippet) > 20 and len(snippet) < 200:
                snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    break

        return snippets

    def _analyze_bloom_levels(self, text: str) -> Dict[str, int]:
        """Analyze Bloom's taxonomy verb distribution."""
        distribution = defaultdict(int)
        text_lower = text.lower()

        for level, verbs in self.BLOOM_PATTERNS.items():
            for verb in verbs:
                pattern = r'\b' + verb + r'\w*\b'
                matches = re.findall(pattern, text_lower)
                distribution[level] += len(matches)

        return dict(distribution)

    def _detect_content_type(self, text: str) -> ContentType:
        """Detect the type of content."""
        text_lower = text.lower()
        scores = defaultdict(int)

        for content_type, patterns in self.CONTENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                scores[content_type] += len(matches)

        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            if scores[best_type] > 0:
                return best_type

        return ContentType.EXPLANATION  # Default

    def _calculate_difficulty(self, profile: ContentProfile) -> float:
        """
        Calculate overall difficulty score.

        Combines:
        - Vocabulary complexity (word length, technical terms)
        - Sentence complexity (sentence length)
        - Concept density
        - Bloom's level distribution
        """
        config = self.config.get('content_profiling', {})
        weights = config.get('difficulty_weights', {
            'vocabulary_complexity': 0.25,
            'sentence_complexity': 0.25,
            'concept_density': 0.25,
            'bloom_level': 0.25
        })

        # Vocabulary complexity (0-1)
        # Average word length: 4-5 = easy, 6-7 = medium, 8+ = hard
        vocab_score = min(1.0, max(0.0, (profile.avg_word_length - 4) / 4))
        vocab_score = (vocab_score + profile.technical_term_density * 5) / 2

        # Sentence complexity (0-1)
        # Average sentence length: 10-15 = easy, 15-20 = medium, 20+ = hard
        sent_score = min(1.0, max(0.0, (profile.avg_sentence_length - 10) / 15))

        # Concept density (0-1)
        # Concepts per 100 words: 1-2 = easy, 3-5 = medium, 5+ = hard
        concept_score = min(1.0, max(0.0, profile.concept_density / 5))

        # Bloom level (0-1)
        bloom_score = 0.0
        total_bloom = sum(profile.bloom_level_distribution.values())
        if total_bloom > 0:
            for level, count in profile.bloom_level_distribution.items():
                bloom_score += self.BLOOM_DIFFICULTY_WEIGHTS.get(level, 0.5) * count
            bloom_score /= total_bloom

        # Weighted combination
        difficulty = (
            weights.get('vocabulary_complexity', 0.25) * vocab_score +
            weights.get('sentence_complexity', 0.25) * sent_score +
            weights.get('concept_density', 0.25) * concept_score +
            weights.get('bloom_level', 0.25) * bloom_score
        )

        return min(1.0, max(0.0, difficulty))

    def _score_to_level(self, score: float) -> DifficultyLevel:
        """Convert difficulty score to level."""
        thresholds = self.config.get('content_profiling', {}).get('difficulty_thresholds', {
            'beginner': 0.33,
            'intermediate': 0.66,
            'advanced': 1.0
        })

        if score <= thresholds.get('beginner', 0.33):
            return DifficultyLevel.BEGINNER
        elif score <= thresholds.get('intermediate', 0.66):
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.ADVANCED

    def _aggregate_profiles(self, profiles: List[ContentProfile], section_id: str) -> ContentProfile:
        """Aggregate multiple profiles into one."""
        if not profiles:
            return ContentProfile()

        aggregate = ContentProfile()

        # Sum counts
        aggregate.word_count = sum(p.word_count for p in profiles)
        aggregate.token_count = sum(p.token_count for p in profiles)
        aggregate.sentence_count = sum(p.sentence_count for p in profiles)

        # Average metrics (weighted by word count)
        total_words = aggregate.word_count or 1
        aggregate.avg_sentence_length = sum(
            p.avg_sentence_length * p.word_count for p in profiles
        ) / total_words
        aggregate.avg_word_length = sum(
            p.avg_word_length * p.word_count for p in profiles
        ) / total_words
        aggregate.readability_score = sum(
            p.readability_score * p.word_count for p in profiles
        ) / total_words
        aggregate.technical_term_density = sum(
            p.technical_term_density * p.word_count for p in profiles
        ) / total_words
        aggregate.concept_density = sum(
            p.concept_density * p.word_count for p in profiles
        ) / total_words
        aggregate.difficulty_score = sum(
            p.difficulty_score * p.word_count for p in profiles
        ) / total_words

        # Combine Bloom distribution
        for p in profiles:
            for level, count in p.bloom_level_distribution.items():
                aggregate.bloom_level_distribution[level] = (
                    aggregate.bloom_level_distribution.get(level, 0) + count
                )

        # Merge concepts
        concept_map: Dict[str, ConceptReference] = {}
        for p in profiles:
            for concept in p.concepts:
                key = concept.normalized_term
                if key in concept_map:
                    concept_map[key].frequency += concept.frequency
                    concept_map[key].context_snippets.extend(concept.context_snippets)
                    if section_id and section_id not in concept_map[key].section_ids:
                        concept_map[key].section_ids.append(section_id)
                else:
                    concept_map[key] = ConceptReference(
                        term=concept.term,
                        normalized_term=concept.normalized_term,
                        frequency=concept.frequency,
                        context_snippets=concept.context_snippets.copy(),
                        is_key_concept=concept.is_key_concept,
                        section_ids=[section_id] if section_id else []
                    )

        aggregate.concepts = sorted(
            concept_map.values(),
            key=lambda c: c.frequency,
            reverse=True
        )

        # Most common content type
        type_counts = Counter(p.content_type for p in profiles)
        aggregate.content_type = type_counts.most_common(1)[0][0]

        # Difficulty level from score
        aggregate.difficulty_level = self._score_to_level(aggregate.difficulty_score)

        return aggregate


# Convenience function
def profile_content(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Profile content and return dictionary representation.

    Args:
        text: Text content to profile
        config: Optional configuration

    Returns:
        Dictionary with profile data
    """
    profiler = ContentProfiler(config)
    profile = profiler.profile_text(text)
    return profile.to_dict()
