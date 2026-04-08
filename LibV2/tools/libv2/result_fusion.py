"""Result fusion module for multi-query retrieval.

This module provides Reciprocal Rank Fusion (RRF) for combining results
from multiple retrieval strategies, along with deduplication and
coherence scoring.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .retriever import RetrievalResult

# Add lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # LibV2/tools/libv2/result_fusion.py → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture


@dataclass
class FusedResult:
    """A result from the fusion process with aggregated scores.

    Attributes:
        chunk_id: Unique identifier for the chunk
        text: Chunk text content
        fused_score: RRF-normalized score
        source_scores: Scores from each retrieval strategy
        course_slug: Source course identifier
        domain: Content domain
        chunk_type: Type of chunk (definition, example, etc.)
        difficulty: Difficulty level
        concept_tags: Associated concept tags
        source: Source metadata dictionary
        tokens_estimate: Estimated token count
        contributing_queries: Which sub-queries returned this result
        coherence_score: Cross-query coherence score
    """
    chunk_id: str
    text: str
    fused_score: float
    source_scores: dict[str, float]
    course_slug: str
    domain: str
    chunk_type: str
    difficulty: Optional[str]
    concept_tags: list[str]
    source: dict
    tokens_estimate: int = 0
    contributing_queries: list[str] = field(default_factory=list)
    coherence_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "fused_score": self.fused_score,
            "source_scores": self.source_scores,
            "course_slug": self.course_slug,
            "domain": self.domain,
            "chunk_type": self.chunk_type,
            "difficulty": self.difficulty,
            "concept_tags": self.concept_tags,
            "source": self.source,
            "tokens_estimate": self.tokens_estimate,
            "contributing_queries": self.contributing_queries,
            "coherence_score": self.coherence_score,
        }

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
        strategy_name: str,
    ) -> "FusedResult":
        """Create FusedResult from a RetrievalResult.

        Args:
            result: Original retrieval result
            strategy_name: Name of the retrieval strategy

        Returns:
            FusedResult initialized from the retrieval result
        """
        return cls(
            chunk_id=result.chunk_id,
            text=result.text,
            fused_score=result.score,
            source_scores={strategy_name: result.score},
            course_slug=result.course_slug,
            domain=result.domain,
            chunk_type=result.chunk_type,
            difficulty=result.difficulty,
            concept_tags=result.concept_tags,
            source=result.source,
            tokens_estimate=result.tokens_estimate,
            contributing_queries=[strategy_name],
        )


@dataclass
class FusionResult:
    """Complete result of multi-query fusion.

    Attributes:
        results: Fused and ranked results
        query_coverage: Number of results per sub-query
        deduplication_stats: Statistics on deduplication
        coherence_metrics: Cross-result coherence scores
        fusion_method: Method used for fusion (e.g., "rrf")
        intent_coverage: Phase I.3 - Dict of {aspect: result_count} for each sub-query
        all_intents_covered: Phase I.3 - True if all sub-queries returned ≥1 result
    """
    results: list[FusedResult]
    query_coverage: dict[str, int] = field(default_factory=dict)
    deduplication_stats: dict = field(default_factory=dict)
    coherence_metrics: dict = field(default_factory=dict)
    fusion_method: str = "rrf"
    # Phase I.3: Intent coverage validation
    intent_coverage: dict = field(default_factory=dict)
    all_intents_covered: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query_coverage": self.query_coverage,
            "deduplication_stats": self.deduplication_stats,
            "coherence_metrics": self.coherence_metrics,
            "fusion_method": self.fusion_method,
            # Phase I.3: Intent coverage
            "intent_coverage": self.intent_coverage,
            "all_intents_covered": self.all_intents_covered,
        }

    @property
    def result_count(self) -> int:
        """Number of fused results."""
        return len(self.results)

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens across all results."""
        return sum(r.tokens_estimate for r in self.results)

    def get_top_results(self, n: int = 10) -> list[FusedResult]:
        """Get top N results by fused score."""
        return sorted(self.results, key=lambda r: r.fused_score, reverse=True)[:n]


class ResultFuser:
    """Merge results from multiple retrieval strategies.

    Implements:
    - Reciprocal Rank Fusion (RRF) for score normalization
    - Similarity-based deduplication using Jaccard index
    - Cross-query coherence checking

    Example:
        >>> fuser = ResultFuser()
        >>> result_sets = {
        ...     "query_1": [RetrievalResult(...), ...],
        ...     "query_2": [RetrievalResult(...), ...],
        ... }
        >>> fusion = fuser.fuse(result_sets)
        >>> print(fusion.result_count)
    """

    # RRF constant (standard value from literature)
    RRF_K = 60

    # Deduplication similarity threshold (Jaccard index)
    # Phase I.2: Lowered from 0.85 to 0.75 to compensate for stricter bigram matching
    DEDUP_THRESHOLD = 0.75

    # Phase I.4: Intent-specific coherence weights
    # Tuple format: (domain_consistency, concept_overlap, multi_query_ratio)
    INTENT_WEIGHTS = {
        "COMPARISON": (0.2, 0.5, 0.3),   # Prioritize concept overlap for comparisons
        "DEFINITION": (0.5, 0.3, 0.2),   # Prioritize domain consistency for definitions
        "PROCEDURE": (0.4, 0.2, 0.4),    # Balance domain + multi-query for procedures
        "EXPLORATION": (0.4, 0.3, 0.3),  # Default balanced weights
    }

    def __init__(
        self,
        rrf_k: int = 60,
        dedup_threshold: float = 0.85,
        capture: Optional["DecisionCapture"] = None,
    ):
        """Initialize the result fuser.

        Args:
            rrf_k: RRF constant (higher = more weight to lower ranks)
            dedup_threshold: Jaccard similarity threshold for deduplication
            capture: Optional DecisionCapture for logging fusion decisions
        """
        self.rrf_k = rrf_k
        self.dedup_threshold = dedup_threshold
        self.capture = capture

    def fuse(
        self,
        result_sets: dict[str, list[RetrievalResult]],
        strategy_weights: Optional[dict[str, float]] = None,
        limit: int = 50,
        primary_intent: Optional[str] = None,
    ) -> FusionResult:
        """Fuse multiple result sets using RRF.

        Args:
            result_sets: Dict mapping strategy name to results
            strategy_weights: Optional weights for each strategy
            limit: Maximum number of results to return
            primary_intent: Phase I.4 - Query intent for adaptive coherence weights

        Returns:
            FusionResult with merged, deduplicated, ranked results
        """
        if not result_sets:
            return FusionResult(results=[])

        # Default equal weights
        if strategy_weights is None:
            strategy_weights = {k: 1.0 for k in result_sets.keys()}

        # Normalize weights
        total_weight = sum(strategy_weights.values())
        if total_weight > 0:
            strategy_weights = {
                k: v / total_weight for k, v in strategy_weights.items()
            }

        # Calculate RRF scores
        chunk_data, rrf_scores = self._calculate_rrf_scores(
            result_sets, strategy_weights
        )

        # Build fused results
        fused_results = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            data = chunk_data[chunk_id]
            fused_results.append(FusedResult(
                chunk_id=chunk_id,
                text=data["text"],
                fused_score=score,
                source_scores=data["source_scores"],
                course_slug=data["course_slug"],
                domain=data["domain"],
                chunk_type=data["chunk_type"],
                difficulty=data["difficulty"],
                concept_tags=data["concept_tags"],
                source=data["source"],
                tokens_estimate=data["tokens_estimate"],
                contributing_queries=data["contributing_queries"],
            ))

        # Deduplicate
        deduped_results, dedup_stats = self._deduplicate(fused_results)

        # Calculate coherence (Phase I.4: with intent-specific weights)
        coherence_metrics = self._calculate_coherence(deduped_results, primary_intent)

        # Apply coherence scores to results
        for result in deduped_results:
            result.coherence_score = self._calculate_result_coherence(
                result, deduped_results
            )

        # Track query coverage
        query_coverage = {}
        for strategy_name in result_sets:
            query_coverage[strategy_name] = len(result_sets[strategy_name])

        # Limit results
        final_results = deduped_results[:limit]

        # Log fusion and deduplication decisions
        if self.capture:
            total_before_dedup = len(fused_results)
            total_after_dedup = len(deduped_results)
            duplicates_removed = total_before_dedup - total_after_dedup

            self.capture.log_decision(
                decision_type="result_fusion",
                decision=f"Fused {len(result_sets)} result sets into {len(final_results)} results",
                rationale=(
                    f"RRF k={self.rrf_k}, "
                    f"Strategy weights: {strategy_weights}, "
                    f"Intent: {primary_intent or 'unspecified'}"
                ),
            )

            if duplicates_removed > 0:
                self.capture.log_decision(
                    decision_type="chunk_deduplication",
                    decision=f"Removed {duplicates_removed} duplicate chunks",
                    rationale=(
                        f"Dedup threshold: {self.dedup_threshold}, "
                        f"Before: {total_before_dedup}, After: {total_after_dedup}"
                    ),
                )

        return FusionResult(
            results=final_results,
            query_coverage=query_coverage,
            deduplication_stats=dedup_stats,
            coherence_metrics=coherence_metrics,
            fusion_method="rrf",
        )

    def _calculate_rrf_scores(
        self,
        result_sets: dict[str, list[RetrievalResult]],
        weights: dict[str, float],
    ) -> tuple[dict, dict[str, float]]:
        """Calculate RRF scores for all unique chunks.

        RRF formula: score(d) = sum(weight_i / (k + rank_i(d)))
        where k=60 is the RRF constant.

        Args:
            result_sets: Results from each strategy
            weights: Strategy weights

        Returns:
            Tuple of (chunk_data dict, rrf_scores dict)
        """
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for strategy_name, results in result_sets.items():
            weight = weights.get(strategy_name, 1.0)

            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # Calculate RRF contribution
                rrf_contribution = weight / (self.rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

                # Store or update chunk data
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "text": result.text,
                        "course_slug": result.course_slug,
                        "domain": result.domain,
                        "chunk_type": result.chunk_type,
                        "difficulty": result.difficulty,
                        "concept_tags": result.concept_tags,
                        "source": result.source,
                        "tokens_estimate": result.tokens_estimate,
                        "source_scores": {},
                        "contributing_queries": [],
                    }

                # Track source scores and contributing queries
                chunk_data[chunk_id]["source_scores"][strategy_name] = result.score
                if strategy_name not in chunk_data[chunk_id]["contributing_queries"]:
                    chunk_data[chunk_id]["contributing_queries"].append(strategy_name)

        return chunk_data, rrf_scores

    def _deduplicate(
        self,
        results: list[FusedResult],
    ) -> tuple[list[FusedResult], dict]:
        """Remove near-duplicate results using Jaccard similarity.

        Args:
            results: List of fused results

        Returns:
            Tuple of (deduplicated results, deduplication stats)
        """
        if not results:
            return [], {"removed": 0, "original": 0}

        deduplicated = []
        removed_count = 0

        for result in results:
            is_duplicate = False

            for existing in deduplicated:
                similarity = self._calculate_jaccard_similarity(
                    result.text, existing.text
                )
                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    removed_count += 1
                    # Merge contributing queries into existing result
                    for query in result.contributing_queries:
                        if query not in existing.contributing_queries:
                            existing.contributing_queries.append(query)
                    # Update source scores
                    existing.source_scores.update(result.source_scores)
                    # Keep higher score
                    existing.fused_score = max(existing.fused_score, result.fused_score)
                    break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated, {
            "removed": removed_count,
            "original": len(results),
            "deduplicated": len(deduplicated),
            "dedup_threshold": self.dedup_threshold,
        }

    def _get_bigrams(self, text: str) -> set[str]:
        """Extract bigrams from text for improved similarity detection.

        Phase I.2: Bigrams capture word pairs, making similarity detection
        more robust against simple word reordering.

        Args:
            text: Input text

        Returns:
            Set of bigram strings (word pairs)
        """
        words = text.lower().split()
        if len(words) < 2:
            return set(words)  # Fall back to single words if too short
        return set(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate length-normalized Jaccard similarity between two texts.

        Phase I.2: Uses bigrams instead of words, plus length normalization
        to prevent short texts from matching long texts too easily.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Length-normalized Jaccard similarity coefficient (0.0 to 1.0)
        """
        # Phase I.2: Use bigrams instead of words for better matching
        bigrams1 = self._get_bigrams(text1)
        bigrams2 = self._get_bigrams(text2)

        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        if union == 0:
            return 0.0

        jaccard_score = intersection / union

        # Phase I.2: Apply length normalization
        # This prevents short excerpts from matching long texts too easily
        len1 = len(text1)
        len2 = len(text2)
        length_factor = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        return jaccard_score * length_factor

    def _calculate_coherence(
        self,
        results: list[FusedResult],
        primary_intent: Optional[str] = None,
    ) -> dict:
        """Calculate cross-result coherence metrics.

        Coherence is measured by:
        - Concept overlap between results
        - Topic consistency (same domain/subdomain)
        - Difficulty progression appropriateness

        Phase I.4: Uses adaptive weights based on query intent.

        Args:
            results: List of fused results
            primary_intent: Query intent for weight selection (e.g., "COMPARISON")

        Returns:
            Dictionary of coherence metrics
        """
        if not results:
            return {"overall": 0.0}

        # Domain consistency
        domains = [r.domain for r in results if r.domain]
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        if domains:
            most_common_domain = max(domain_counts.values())
            domain_consistency = most_common_domain / len(domains)
        else:
            domain_consistency = 0.0

        # Concept overlap (average pairwise tag overlap)
        concept_overlaps = []
        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                tags1 = set(r1.concept_tags)
                tags2 = set(r2.concept_tags)
                if tags1 or tags2:
                    overlap = len(tags1 & tags2) / len(tags1 | tags2) if (tags1 | tags2) else 0
                    concept_overlaps.append(overlap)

        avg_concept_overlap = (
            sum(concept_overlaps) / len(concept_overlaps)
            if concept_overlaps else 0.0
        )

        # Contributing query distribution (how many results come from multiple queries)
        multi_query_results = sum(
            1 for r in results if len(r.contributing_queries) > 1
        )
        multi_query_ratio = multi_query_results / len(results) if results else 0.0

        # Phase I.4: Get intent-specific weights (default to EXPLORATION)
        intent_key = primary_intent.upper() if primary_intent else "EXPLORATION"
        weights = self.INTENT_WEIGHTS.get(intent_key, self.INTENT_WEIGHTS["EXPLORATION"])
        domain_weight, concept_weight, multiquery_weight = weights

        # Overall coherence score (Phase I.4: uses adaptive weights)
        overall = (
            domain_consistency * domain_weight +
            avg_concept_overlap * concept_weight +
            multi_query_ratio * multiquery_weight
        )

        return {
            "overall": overall,
            "domain_consistency": domain_consistency,
            "concept_overlap": avg_concept_overlap,
            "multi_query_ratio": multi_query_ratio,
            "result_count": len(results),
            # Phase I.4: Include weight info for transparency
            "intent_used": intent_key,
            "weights_applied": {
                "domain": domain_weight,
                "concept": concept_weight,
                "multi_query": multiquery_weight,
            },
        }

    def _calculate_result_coherence(
        self,
        result: FusedResult,
        all_results: list[FusedResult],
    ) -> float:
        """Calculate coherence score for a single result.

        Args:
            result: The result to score
            all_results: All results for context

        Returns:
            Coherence score (0.0 to 1.0)
        """
        if len(all_results) <= 1:
            return 1.0

        # How many queries contributed to this result
        query_coverage = len(result.contributing_queries) / max(
            max(len(r.contributing_queries) for r in all_results), 1
        )

        # Concept overlap with other results
        result_tags = set(result.concept_tags)
        overlaps = []
        for other in all_results:
            if other.chunk_id != result.chunk_id:
                other_tags = set(other.concept_tags)
                if result_tags or other_tags:
                    overlap = len(result_tags & other_tags) / len(result_tags | other_tags) if (result_tags | other_tags) else 0
                    overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Combined score
        return (query_coverage * 0.5 + avg_overlap * 0.5)
