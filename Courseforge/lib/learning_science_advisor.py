"""
Learning Science Advisor Middleware

Provides research-backed pedagogical guidance during Courseforge content generation
by querying the Learning Science RAG corpus (1,144 chunks, 16 domains).

Query Orchestration | Two-level Caching | Token Budget Management
"""

import hashlib
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path for LibV2 imports
ED4ALL_ROOT = Path(__file__).resolve().parents[2]  # Courseforge/lib/learning_science_advisor.py → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

logger = logging.getLogger(__name__)

# Configuration
CORPUS_PATH = ED4ALL_ROOT / "LibV2" / "courses" / "learning-science-for-instructional-designers"
CHUNKS_FILE = CORPUS_PATH / "corpus" / "chunks.jsonl"
CACHE_DIR = ED4ALL_ROOT / "state" / "learning_science_cache"
MEMORY_CACHE_TTL_SECONDS = 900  # 15 minutes
FILE_CACHE_TTL_HOURS = 24
DEFAULT_LIMIT = 10
MAX_LIMIT = 25
MAX_TOKENS_PER_QUERY = 10000  # ~10K tokens target

# Stop words for TF-IDF
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these',
    'those', 'it', 'its', 'as', 'if', 'when', 'where', 'how', 'what',
    'which', 'who', 'whom', 'why', 'can', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'any', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'while', 'your', 'you', 'we', 'our', 'they', 'their',
    'learning', 'learners', 'students', 'design', 'instructional',  # domain-common
}

# Query pattern keywords by context type
CONTEXT_KEYWORDS = {
    "cognitive_load": ["cognitive load", "Sweller", "worked examples", "split attention",
                       "intrinsic load", "extraneous load", "germane load", "element interactivity"],
    "multimedia": ["multimedia learning", "Mayer", "dual coding", "modality principle",
                   "contiguity", "redundancy", "signaling", "coherence principle"],
    "retrieval_practice": ["retrieval practice", "testing effect", "Bjork", "spacing effect",
                          "interleaving", "desirable difficulties", "distributed practice"],
    "motivation": ["motivation", "self-determination", "Deci", "Ryan", "intrinsic motivation",
                  "autonomy", "competence", "relatedness", "Dweck", "growth mindset"],
    "metacognition": ["metacognition", "self-regulation", "monitoring", "planning",
                     "evaluation", "strategy use", "metacognitive awareness"],
    "transfer": ["transfer of learning", "near transfer", "far transfer", "analogical reasoning",
                "situated cognition", "generalization"],
    "feedback": ["feedback", "formative assessment", "immediate feedback", "delayed feedback",
                "corrective feedback", "knowledge of results", "scaffolding"],
    "emotion": ["emotion", "affective learning", "anxiety", "engagement", "flow state",
               "curiosity", "interest", "achievement emotions"],
    "social": ["social learning", "collaborative learning", "peer learning", "Vygotsky",
              "zone of proximal development", "scaffolding", "community of practice"],
    "expertise": ["expertise development", "deliberate practice", "chunking", "automaticity",
                 "pattern recognition", "skilled performance"],
    "schema": ["schema theory", "prior knowledge", "conceptual change", "mental models",
              "knowledge organization", "misconceptions"],
    "individual_differences": ["individual differences", "learning styles", "aptitude",
                               "prior knowledge", "working memory capacity", "preferences"],
    "technology": ["educational technology", "e-learning", "online learning", "MOOC",
                  "learning management system", "adaptive learning", "intelligent tutoring"],
}


@dataclass
class CachedResult:
    """Cached retrieval result with timestamp."""
    results: list[dict]
    timestamp: datetime
    query: str
    context_type: str

    def is_expired(self, ttl_seconds: int) -> bool:
        return datetime.now() - self.timestamp > timedelta(seconds=ttl_seconds)


@dataclass
class PedagogicalContext:
    """Formatted pedagogical context for agent consumption."""
    principles: list[str]
    strategies: list[str]
    citations: list[str]
    raw_chunks: list[dict]
    context_type: str
    query: str
    token_estimate: int

    def to_prompt_injection(self) -> str:
        """Format for injection into agent prompts."""
        lines = [
            f"## Pedagogical Research Context ({self.context_type})",
            "",
            "### Key Principles",
        ]
        for p in self.principles[:5]:
            lines.append(f"- {p}")

        lines.extend(["", "### Recommended Strategies"])
        for s in self.strategies[:5]:
            lines.append(f"- {s}")

        lines.extend(["", "### Research Citations"])
        for c in self.citations[:8]:
            lines.append(f"- {c}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "principles": self.principles,
            "strategies": self.strategies,
            "citations": self.citations,
            "context_type": self.context_type,
            "query": self.query,
            "token_estimate": self.token_estimate,
            "chunk_count": len(self.raw_chunks),
        }


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for TF-IDF."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


class LearningScienceAdvisor:
    """
    Middleware for querying Learning Science corpus.

    Provides:
    - Direct TF-IDF retrieval on Learning Science corpus
    - Two-level caching (memory 15min, file 24hr)
    - Prompt injection formatting
    - Token budget management

    Example:
        >>> advisor = LearningScienceAdvisor()
        >>> context = advisor.query(
        ...     topic="teaching database normalization",
        ...     context_type="cognitive_load",
        ...     bloom_level="apply"
        ... )
        >>> print(context.to_prompt_injection())
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_ttl: int = MEMORY_CACHE_TTL_SECONDS,
        file_ttl_hours: int = FILE_CACHE_TTL_HOURS,
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_ttl = memory_ttl
        self.file_ttl_hours = file_ttl_hours

        # Memory cache: dict[cache_key, CachedResult]
        self._memory_cache: dict[str, CachedResult] = {}

        # Corpus state (lazy loaded)
        self._chunks: Optional[list[dict]] = None
        self._idf: Optional[dict[str, float]] = None

    def _load_corpus(self) -> None:
        """Lazy load the corpus and build IDF index."""
        if self._chunks is not None:
            return

        self._chunks = []
        doc_freq: Counter = Counter()

        if not CHUNKS_FILE.exists():
            logger.error(f"Corpus file not found: {CHUNKS_FILE}")
            return

        with open(CHUNKS_FILE, 'r') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    text = chunk.get("text", "")
                    tokens = set(_tokenize(text))
                    chunk["_tokens"] = tokens
                    self._chunks.append(chunk)
                    for token in tokens:
                        doc_freq[token] += 1
                except json.JSONDecodeError:
                    continue

        # Compute IDF
        n_docs = len(self._chunks)
        self._idf = {}
        for token, freq in doc_freq.items():
            self._idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1

        logger.info(f"Loaded {len(self._chunks)} chunks, {len(self._idf)} unique terms")

    def _tfidf_search(self, query: str, limit: int) -> list[dict]:
        """Perform TF-IDF search on corpus."""
        self._load_corpus()

        if not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Score each chunk
        scores = []
        for i, chunk in enumerate(self._chunks):
            chunk_tokens = chunk.get("_tokens", set())
            score = 0.0
            for token in query_tokens:
                if token in chunk_tokens:
                    # TF-IDF score
                    tf = 1  # Binary TF
                    idf = self._idf.get(token, 1.0)
                    score += tf * idf
            if score > 0:
                scores.append((score, i))

        # Sort by score descending
        scores.sort(reverse=True)

        # Return top results
        results = []
        for score, idx in scores[:limit]:
            chunk = self._chunks[idx].copy()
            chunk.pop("_tokens", None)  # Remove internal field
            chunk["score"] = score
            results.append(chunk)

        return results

    def _cache_key(self, topic: str, context_type: str, bloom_level: Optional[str] = None) -> str:
        """Generate cache key from query parameters."""
        key_str = f"{topic}:{context_type}:{bloom_level or 'any'}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_from_memory_cache(self, cache_key: str) -> Optional[CachedResult]:
        """Check memory cache for valid entry."""
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if not cached.is_expired(self.memory_ttl):
                logger.debug(f"Memory cache hit: {cache_key}")
                return cached
            else:
                del self._memory_cache[cache_key]
        return None

    def _get_from_file_cache(self, cache_key: str) -> Optional[CachedResult]:
        """Check file cache for valid entry."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                timestamp = datetime.fromisoformat(data["timestamp"])
                if datetime.now() - timestamp < timedelta(hours=self.file_ttl_hours):
                    logger.debug(f"File cache hit: {cache_key}")
                    return CachedResult(
                        results=data["results"],
                        timestamp=timestamp,
                        query=data["query"],
                        context_type=data["context_type"],
                    )
                else:
                    cache_file.unlink()  # Expired, remove
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid cache file {cache_file}: {e}")
                cache_file.unlink()
        return None

    def _save_to_cache(self, cache_key: str, cached: CachedResult) -> None:
        """Save to both memory and file cache."""
        # Memory cache
        self._memory_cache[cache_key] = cached

        # File cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            "results": cached.results,
            "timestamp": cached.timestamp.isoformat(),
            "query": cached.query,
            "context_type": cached.context_type,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

    def _build_query(self, topic: str, context_type: str) -> str:
        """Build enhanced query with context keywords."""
        keywords = CONTEXT_KEYWORDS.get(context_type, [])
        # Add 2-3 most relevant keywords to query
        keyword_boost = " ".join(keywords[:3]) if keywords else ""
        return f"{topic} {keyword_boost} learning theory instructional design"

    def _extract_principles(self, chunks: list[dict]) -> list[str]:
        """Extract key principles from chunk texts."""
        principles = []
        principle_patterns = [
            "principle", "finding", "research shows", "evidence suggests",
            "studies demonstrate", "key insight", "fundamental"
        ]

        for chunk in chunks:
            text = chunk.get("text", "")
            # Look for sentences containing principle indicators
            sentences = text.split(". ")
            for sent in sentences:
                sent_lower = sent.lower()
                if any(p in sent_lower for p in principle_patterns):
                    if len(sent) > 30 and len(sent) < 300:
                        principles.append(sent.strip() + ".")
                        if len(principles) >= 8:
                            return principles

        # Fallback: first sentence of each chunk
        if len(principles) < 3:
            for chunk in chunks[:5]:
                text = chunk.get("text", "")
                first_sent = text.split(". ")[0]
                if len(first_sent) > 30:
                    principles.append(first_sent.strip() + ".")

        return principles[:8]

    def _extract_strategies(self, chunks: list[dict]) -> list[str]:
        """Extract actionable strategies from chunk texts."""
        strategies = []
        strategy_patterns = [
            "should", "can be used", "effective for", "strategy",
            "technique", "approach", "method", "implement", "apply"
        ]

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = text.split(". ")
            for sent in sentences:
                sent_lower = sent.lower()
                if any(p in sent_lower for p in strategy_patterns):
                    if len(sent) > 30 and len(sent) < 250:
                        strategies.append(sent.strip() + ".")
                        if len(strategies) >= 8:
                            return strategies

        return strategies[:8]

    def _extract_citations(self, chunks: list[dict]) -> list[str]:
        """Extract researcher citations from chunk texts."""
        import re
        citations = set()

        # Pattern for APA-style citations: (Author, Year) or Author (Year)
        patterns = [
            r'\(([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?,\s*\d{4}[a-z]?)\)',
            r'([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?)\s*\((\d{4}[a-z]?)\)',
        ]

        for chunk in chunks:
            text = chunk.get("text", "")
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        citation = f"{match[0]} ({match[1]})"
                    else:
                        citation = match
                    citations.add(citation)

        return sorted(list(citations))[:15]

    def _estimate_tokens(self, chunks: list[dict]) -> int:
        """Estimate token count for chunks (rough: 4 chars = 1 token)."""
        total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
        return total_chars // 4

    def _trim_to_budget(self, chunks: list[dict], max_tokens: int = MAX_TOKENS_PER_QUERY) -> list[dict]:
        """Trim results to fit token budget."""
        trimmed = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(chunk.get("text", "")) // 4
            if current_tokens + chunk_tokens > max_tokens:
                break
            trimmed.append(chunk)
            current_tokens += chunk_tokens

        return trimmed

    def query(
        self,
        topic: str,
        context_type: str = "general",
        bloom_level: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        use_cache: bool = True,
    ) -> PedagogicalContext:
        """
        Query the Learning Science corpus for pedagogical guidance.

        Args:
            topic: The instructional topic (e.g., "teaching database normalization")
            context_type: Type of pedagogical context needed:
                - cognitive_load, multimedia, retrieval_practice, motivation,
                - metacognition, transfer, feedback, emotion, social,
                - expertise, schema, individual_differences, technology, general
            bloom_level: Optional Bloom's taxonomy level filter
            limit: Maximum results (default 10, max 25)
            use_cache: Whether to use caching (default True)

        Returns:
            PedagogicalContext with formatted principles, strategies, and citations
        """
        limit = min(limit, MAX_LIMIT)
        cache_key = self._cache_key(topic, context_type, bloom_level)

        # Check caches
        if use_cache:
            cached = self._get_from_memory_cache(cache_key)
            if cached:
                return self._format_context(cached.results, context_type, topic)

            cached = self._get_from_file_cache(cache_key)
            if cached:
                self._memory_cache[cache_key] = cached  # Promote to memory
                return self._format_context(cached.results, context_type, topic)

        # Build enhanced query
        query = self._build_query(topic, context_type)

        # Execute TF-IDF search on corpus
        start_time = time.time()
        try:
            results = self._tfidf_search(query, limit)
            logger.info(f"Learning science query completed in {time.time() - start_time:.2f}s: {len(results)} results")
        except Exception as e:
            logger.error(f"Learning science retrieval failed: {e}")
            results = []

        # Trim to token budget
        results = self._trim_to_budget(results)

        # Cache results
        if use_cache and results:
            cached = CachedResult(
                results=results,
                timestamp=datetime.now(),
                query=query,
                context_type=context_type,
            )
            self._save_to_cache(cache_key, cached)

        return self._format_context(results, context_type, topic)

    def _format_context(self, results: list[dict], context_type: str, topic: str) -> PedagogicalContext:
        """Format raw results into PedagogicalContext."""
        return PedagogicalContext(
            principles=self._extract_principles(results),
            strategies=self._extract_strategies(results),
            citations=self._extract_citations(results),
            raw_chunks=results,
            context_type=context_type,
            query=topic,
            token_estimate=self._estimate_tokens(results),
        )

    def get_pedagogical_strategy(
        self,
        topic: str,
        objective: str,
        bloom_level: str = "apply",
    ) -> PedagogicalContext:
        """
        Get specific pedagogical strategies for a learning objective.

        Args:
            topic: The content topic
            objective: The learning objective
            bloom_level: Bloom's taxonomy level

        Returns:
            PedagogicalContext focused on strategies
        """
        # Map bloom level to appropriate context types
        bloom_context_map = {
            "remember": "retrieval_practice",
            "understand": "schema",
            "apply": "cognitive_load",
            "analyze": "metacognition",
            "evaluate": "feedback",
            "create": "transfer",
        }

        context_type = bloom_context_map.get(bloom_level.lower(), "general")
        combined_query = f"{topic} {objective}"

        return self.query(
            topic=combined_query,
            context_type=context_type,
            bloom_level=bloom_level,
            limit=15,
        )

    def validate_with_research(
        self,
        content_summary: str,
        aspects: list[str],
    ) -> dict:
        """
        Validate content design against learning science research.

        Args:
            content_summary: Brief summary of the content being validated
            aspects: List of aspects to check (e.g., ["cognitive_load", "engagement"])

        Returns:
            Dict with validation results per aspect
        """
        validation = {}

        for aspect in aspects:
            if aspect in CONTEXT_KEYWORDS:
                context = self.query(
                    topic=content_summary,
                    context_type=aspect,
                    limit=5,
                )
                validation[aspect] = {
                    "relevant_principles": context.principles[:3],
                    "recommended_strategies": context.strategies[:3],
                    "supporting_citations": context.citations[:5],
                    "chunk_count": len(context.raw_chunks),
                }
            else:
                validation[aspect] = {"error": f"Unknown aspect: {aspect}"}

        return validation

    def invalidate_cache(self) -> int:
        """
        Invalidate all cached results.

        Call this when the corpus is updated.

        Returns:
            Number of cache entries removed
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        # Remove file cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Invalidated {count} cache entries")
        return count


# Singleton instance for reuse
_advisor_instance: Optional[LearningScienceAdvisor] = None


def get_advisor() -> LearningScienceAdvisor:
    """Get or create singleton advisor instance."""
    global _advisor_instance
    if _advisor_instance is None:
        _advisor_instance = LearningScienceAdvisor()
    return _advisor_instance


# Convenience functions
def learning_science_query(
    topic: str,
    context_type: str = "general",
    limit: int = DEFAULT_LIMIT,
) -> PedagogicalContext:
    """Query learning science corpus (convenience function)."""
    return get_advisor().query(topic, context_type, limit=limit)


def get_pedagogical_strategy(
    topic: str,
    objective: str,
    bloom_level: str = "apply",
) -> PedagogicalContext:
    """Get pedagogical strategies (convenience function)."""
    return get_advisor().get_pedagogical_strategy(topic, objective, bloom_level)


def validate_with_research(
    content_summary: str,
    aspects: list[str],
) -> dict:
    """Validate with research (convenience function)."""
    return get_advisor().validate_with_research(content_summary, aspects)
