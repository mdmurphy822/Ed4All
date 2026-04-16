"""Streaming retrieval module for LibV2.

This module provides sample-based querying without loading the entire corpus.
It filters courses by metadata first, then streams chunks line-by-line from
chunks.jsonl files, applying filters and ranking with BM25 + character
n-gram boosting for improved semantic resilience.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .catalog import load_master_catalog, search_catalog
from .models.catalog import CatalogEntry


@dataclass
class ChunkFilter:
    """Filter criteria for chunks."""
    chunk_type: Optional[str] = None
    difficulty: Optional[str] = None
    concept_tags: Optional[list[str]] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    learning_outcome_refs: Optional[list[str]] = None
    bloom_level: Optional[str] = None


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    chunk_id: str
    text: str
    score: float
    course_slug: str
    domain: str
    chunk_type: str
    difficulty: Optional[str]
    concept_tags: list[str]
    source: dict
    tokens_estimate: int = 0
    learning_outcome_refs: list[str] = field(default_factory=list)
    bloom_level: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "course_slug": self.course_slug,
            "domain": self.domain,
            "chunk_type": self.chunk_type,
            "difficulty": self.difficulty,
            "concept_tags": self.concept_tags,
            "source": self.source,
            "tokens_estimate": self.tokens_estimate,
            "learning_outcome_refs": self.learning_outcome_refs,
            "bloom_level": self.bloom_level,
        }


# Retrieval scoring utilities

# Default minimum relevance threshold.
# Results below this threshold are filtered. If no results meet the
# threshold, an empty list is returned rather than low-quality fallbacks.
# This ensures downstream consumers (Trainforge) only receive content
# with sufficient relevance for high-quality question generation.
DEFAULT_MIN_RELEVANCE = 0.5

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
}

# Backward compatibility alias
MIN_RELEVANCE_THRESHOLD = DEFAULT_MIN_RELEVANCE


def tokenize(text: str) -> list[str]:
    """Simple tokenization for search."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _char_trigrams(text: str) -> set[str]:
    """Extract character trigrams from text for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    trigrams: set[str] = set()
    for word in text.split():
        if len(word) >= 3:
            for i in range(len(word) - 2):
                trigrams.add(word[i:i + 3])
    return trigrams


class LazyBM25:
    """BM25 index with character n-gram boosting.

    Replaces the previous LazyTFIDF implementation with Okapi BM25
    for better term saturation and document length normalization.
    Character trigram overlap provides lightweight fuzzy matching
    for morphological variants and partial word matches.

    Args:
        chunks: List of chunk dicts with "text" field.
        k1: BM25 term frequency saturation parameter (default 1.5).
        b: BM25 document length normalization (0=none, 1=full, default 0.75).
        ngram_weight: Weight for character trigram score blending (default 0.15).
    """

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
        ngram_weight: float = 0.15,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.ngram_weight = ngram_weight
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self._build_index()

    def _build_index(self):
        """Build BM25 index from chunks."""
        total_length = 0

        for chunk in self.chunks:
            text = chunk.get("text", "")
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            for token in set(tokens):
                self.doc_freq[token] += 1

        n = len(self.chunks)
        self.avgdl = total_length / n if n > 0 else 1.0

    def _idf(self, term: str) -> float:
        """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_relevance: Optional[float] = None,
    ) -> list[tuple[dict, float]]:
        """Search using BM25 scoring with optional n-gram boosting.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            min_relevance: Minimum score threshold (default: DEFAULT_MIN_RELEVANCE).

        Returns:
            List of (chunk, score) tuples sorted by descending score.
        """
        if min_relevance is None:
            min_relevance = DEFAULT_MIN_RELEVANCE

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # BM25 scoring
        results: list[tuple[dict, float]] = []
        for i, chunk in enumerate(self.chunks):
            doc_len = self.doc_lengths[i]
            tf_counts = Counter(self.doc_tokens[i])

            bm25_score = 0.0
            for term in query_tokens:
                if term not in tf_counts:
                    continue
                tf = tf_counts[term]
                idf = self._idf(term)
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                bm25_score += idf * (numerator / denominator)

            if bm25_score > 0:
                results.append((chunk, bm25_score))

        if not results:
            return []

        # Character trigram boosting for fuzzy matching
        if self.ngram_weight > 0:
            query_trigrams = _char_trigrams(query)
            if query_trigrams:
                for idx, (chunk, bm25_score) in enumerate(results):
                    chunk_text = chunk.get("text", "")
                    # Only compute trigrams on first 500 chars for efficiency
                    chunk_trigrams = _char_trigrams(chunk_text[:500])
                    if chunk_trigrams:
                        intersection = len(query_trigrams & chunk_trigrams)
                        union = len(query_trigrams | chunk_trigrams)
                        jaccard = intersection / union if union > 0 else 0.0
                    else:
                        jaccard = 0.0

                    blended = (
                        (1 - self.ngram_weight) * bm25_score
                        + self.ngram_weight * jaccard * bm25_score
                    )
                    results[idx] = (chunk, blended)

        results.sort(key=lambda x: x[1], reverse=True)

        # Apply relevance threshold — no fallback.
        # Returning low-relevance chunks degrades downstream quality.
        above_threshold = [
            (chunk, score) for chunk, score in results
            if score >= min_relevance
        ]
        return above_threshold[:limit]


# Backward compatibility alias
LazyTFIDF = LazyBM25


def _matches_filter(chunk: dict, chunk_filter: ChunkFilter) -> bool:
    """Check if a chunk matches the filter criteria."""
    if chunk_filter.chunk_type:
        if chunk.get("chunk_type") != chunk_filter.chunk_type:
            return False

    if chunk_filter.difficulty:
        if chunk.get("difficulty") != chunk_filter.difficulty:
            return False

    if chunk_filter.concept_tags:
        chunk_tags = set(chunk.get("concept_tags", []))
        filter_tags = set(chunk_filter.concept_tags)
        if not filter_tags & chunk_tags:  # No overlap
            return False

    if chunk_filter.min_tokens:
        if chunk.get("tokens_estimate", 0) < chunk_filter.min_tokens:
            return False

    if chunk_filter.max_tokens:
        if chunk.get("tokens_estimate", 0) > chunk_filter.max_tokens:
            return False

    if chunk_filter.learning_outcome_refs:
        chunk_refs = set(chunk.get("learning_outcome_refs", []))
        filter_refs = set(chunk_filter.learning_outcome_refs)
        if not filter_refs & chunk_refs:  # No overlap
            return False

    if chunk_filter.bloom_level:
        # Match bloom level from linked outcomes or chunk metadata
        chunk_bloom = chunk.get("bloom_level")
        if chunk_bloom != chunk_filter.bloom_level:
            return False

    return True


def stream_chunks_from_course(
    course_dir: Path,
    course_slug: str,
    domain: str,
    chunk_filter: Optional[ChunkFilter] = None,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Stream chunks line-by-line from chunks.jsonl, applying filters.

    Yields chunks one at a time without loading the entire file.
    """
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    if not chunks_path.exists():
        return

    count = 0
    with open(chunks_path) as f:
        for line in f:
            if limit and count >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Add course metadata
            chunk["_course_slug"] = course_slug
            chunk["_domain"] = domain

            # Apply filter
            if chunk_filter and not _matches_filter(chunk, chunk_filter):
                continue

            count += 1
            yield chunk


def _collect_filtered_chunks(
    courses: list[CatalogEntry],
    repo_root: Path,
    chunk_filter: Optional[ChunkFilter],
    budget: int,
    per_course_budget: Optional[int] = None,
) -> list[dict]:
    """Collect chunks from filtered courses until budget is reached.

    Uses round-robin across courses to get diverse results.
    """
    collected = []
    course_iterators = {}
    course_counts = {}

    # Initialize iterators for each course
    for entry in courses:
        course_dir = repo_root / "courses" / entry.slug
        chunks_path = course_dir / "corpus" / "chunks.jsonl"
        if chunks_path.exists():
            course_iterators[entry.slug] = {
                "iterator": stream_chunks_from_course(
                    course_dir, entry.slug, entry.primary_domain, chunk_filter
                ),
                "domain": entry.primary_domain,
            }
            course_counts[entry.slug] = 0

    # Round-robin collection
    while len(collected) < budget and course_iterators:
        exhausted = []
        for slug, info in list(course_iterators.items()):
            if len(collected) >= budget:
                break

            if per_course_budget and course_counts[slug] >= per_course_budget:
                exhausted.append(slug)
                continue

            try:
                chunk = next(info["iterator"])
                collected.append(chunk)
                course_counts[slug] += 1
            except StopIteration:
                exhausted.append(slug)

        for slug in exhausted:
            del course_iterators[slug]

    return collected


def retrieve_chunks(
    repo_root: Path,
    query: str,
    domain: Optional[str] = None,
    division: Optional[str] = None,
    subdomain: Optional[str] = None,
    course_slug: Optional[str] = None,
    chunk_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    concept_tags: Optional[list[str]] = None,
    learning_outcome_refs: Optional[list[str]] = None,
    bloom_level: Optional[str] = None,
    limit: int = 10,
    sample_per_course: Optional[int] = None,
    min_relevance: Optional[float] = None,
) -> list[RetrievalResult]:
    """Retrieve chunks matching query and filters.

    Two-phase retrieval:
    1. Filter courses by metadata (no chunk loading)
    2. Stream chunks from filtered courses, apply chunk filters
    3. Rank with TF-IDF on filtered candidates only

    Args:
        repo_root: Path to LibV2 repository root
        query: Search query string
        domain: Filter by domain
        division: Filter by division (STEM/ARTS)
        subdomain: Filter by subdomain
        course_slug: Limit to specific course
        chunk_type: Filter by chunk type (explanation, example, etc.)
        difficulty: Filter by difficulty
        concept_tags: Filter by concept tags (any match)
        learning_outcome_refs: Filter by learning outcome refs (any match)
        bloom_level: Filter by Bloom's taxonomy level
        limit: Maximum results to return
        sample_per_course: Max chunks per course for cross-course search
        min_relevance: Minimum relevance score (default: DEFAULT_MIN_RELEVANCE)

    Returns:
        List of RetrievalResult sorted by relevance score
    """
    # Phase 1: Filter courses by metadata
    if course_slug:
        # Single course mode
        course_dir = repo_root / "courses" / course_slug
        if not course_dir.exists():
            return []

        # Load manifest for domain info
        manifest_path = course_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            domain_info = manifest.get("classification", {}).get("primary_domain", "unknown")
        else:
            domain_info = "unknown"

        courses = [CatalogEntry(
            slug=course_slug,
            title="",
            division="",
            primary_domain=domain_info,
        )]
    else:
        # Cross-course mode - use catalog
        catalog = load_master_catalog(repo_root)
        if catalog is None:
            return []

        courses = search_catalog(
            catalog,
            division=division,
            domain=domain,
            subdomain=subdomain,
        )

        if not courses:
            return []

    # Phase 2: Stream and filter chunks
    chunk_filter = ChunkFilter(
        chunk_type=chunk_type,
        difficulty=difficulty,
        concept_tags=concept_tags,
        learning_outcome_refs=learning_outcome_refs,
        bloom_level=bloom_level,
    )

    # Collect enough candidates for ranking
    # We want more candidates than limit to rank well
    candidate_budget = max(limit * 10, 100)

    candidates = _collect_filtered_chunks(
        courses=courses,
        repo_root=repo_root,
        chunk_filter=chunk_filter,
        budget=candidate_budget,
        per_course_budget=sample_per_course,
    )

    if not candidates:
        return []

    # Phase 3: Rank with BM25 + n-gram boosting
    index = LazyBM25(candidates)
    scored = index.search(query, limit=limit, min_relevance=min_relevance)

    # Convert to RetrievalResult
    results = []
    for chunk, score in scored:
        result = RetrievalResult(
            chunk_id=chunk.get("id", ""),
            text=chunk.get("text", ""),
            score=score,
            course_slug=chunk.get("_course_slug", ""),
            domain=chunk.get("_domain", ""),
            chunk_type=chunk.get("chunk_type", ""),
            difficulty=chunk.get("difficulty"),
            concept_tags=chunk.get("concept_tags", []),
            source=chunk.get("source", {}),
            tokens_estimate=chunk.get("tokens_estimate", 0),
            learning_outcome_refs=chunk.get("learning_outcome_refs", []),
            bloom_level=chunk.get("bloom_level"),
        )
        results.append(result)

    return results
