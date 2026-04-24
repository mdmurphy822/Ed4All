"""Multi-query retrieval orchestration.

This module provides the MultiQueryRetriever class that orchestrates
query decomposition, parallel sub-query execution, and result fusion.
"""

import concurrent.futures
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .query_decomposer import QueryDecomposer
from .query_decomposition import DecomposedQuery, SubQuery
from .result_fusion import FusionResult, ResultFuser
from .retriever import (
    RetrievalResult,
    retrieve_chunks,
)

# Add lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # LibV2/tools/libv2/multi_retriever.py → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture


class MultiQueryRetriever:
    """Execute multiple queries in parallel and fuse results.

    Extends the existing LibV2 retriever with:
    - Query decomposition integration
    - Parallel sub-query execution
    - Result fusion using RRF

    Example:
        >>> retriever = MultiQueryRetriever(repo_root=Path("/path/to/LibV2"))
        >>> results = retriever.retrieve(
        ...     query="Compare ADDIE and SAM instructional design models",
        ...     limit=15,
        ... )
        >>> print(results.result_count)
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        max_workers: int = 4,
        rrf_k: int = 60,
        dedup_threshold: float = 0.85,
        capture: Optional["DecisionCapture"] = None,
    ):
        """Initialize the multi-query retriever.

        Args:
            repo_root: Root path of LibV2 repository
            max_workers: Maximum parallel threads for sub-query execution
            rrf_k: RRF constant for score fusion
            dedup_threshold: Jaccard threshold for deduplication
            capture: Optional DecisionCapture for logging retrieval decisions
        """
        self.repo_root = repo_root or self._auto_detect_repo_root()
        self.max_workers = max_workers
        self.decomposer = QueryDecomposer()
        self.fuser = ResultFuser(rrf_k=rrf_k, dedup_threshold=dedup_threshold, capture=capture)
        self.capture = capture

    def _auto_detect_repo_root(self) -> Path:
        """Auto-detect LibV2 repository root."""
        # Try relative to this file
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent.parent,  # LibV2/tools/libv2 -> LibV2
            ED4ALL_ROOT / "LibV2",
        ]
        for candidate in candidates:
            if (candidate / "courses").exists():
                return candidate
        return candidates[0]

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        division: Optional[str] = None,
        decompose: bool = True,
        per_query_limit: int = 20,
        chunk_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        strategy_weights: Optional[dict[str, float]] = None,
        # Wave 70 — RDF-aligned filter axes. Additive; fusion behavior
        # untouched (filters fire per-sub-query in ``_execute_single_query``).
        cognitive_domain: Optional[str] = None,
        hierarchy_level: Optional[str] = None,
    ) -> FusionResult:
        """Retrieve chunks using query decomposition and fusion.

        Args:
            query: User query
            limit: Maximum total results after fusion
            domain: Domain filter (e.g., "instructional-design")
            division: Division filter ("STEM" or "ARTS")
            decompose: If True, decompose query into sub-queries
            per_query_limit: Max results per sub-query
            chunk_type: Chunk type filter
            difficulty: Difficulty filter
            strategy_weights: Optional weights for sub-queries

        Returns:
            FusionResult with fused, ranked results
        """
        if not decompose:
            # Single query mode (no decomposition)
            results = self._execute_single_query(
                query=query,
                limit=limit,
                domain=domain,
                division=division,
                chunk_type=chunk_type,
                difficulty=difficulty,
                cognitive_domain=cognitive_domain,
                hierarchy_level=hierarchy_level,
            )
            return FusionResult(
                results=[],
                query_coverage={"original": len(results)},
                fusion_method="single",
            )

        # Decompose query
        decomposed = self.decomposer.decompose(query)

        # Log query decomposition decision
        if self.capture:
            sub_query_texts = [sq.text for sq in decomposed.sub_queries]
            self.capture.log_decision(
                decision_type="query_decomposition",
                decision=f"Decomposed into {len(decomposed.sub_queries)} sub-queries",
                rationale=(
                    f"Primary intent: {decomposed.primary_intent.value if hasattr(decomposed.primary_intent, 'value') else decomposed.primary_intent}, "
                    f"Bloom level: {decomposed.bloom_level}, "
                    f"Detected concepts: {decomposed.detected_concepts}"
                ),
                inputs_ref=[{"type": "query", "content": query}],
                alternatives_considered=[
                    "Single query without decomposition",
                    f"Alternative decomposition with {len(sub_query_texts) + 1} queries",
                ],
            )

        # Execute sub-queries in parallel
        result_sets = self._execute_sub_queries(
            decomposed=decomposed,
            per_query_limit=per_query_limit,
            domain=domain,
            division=division,
            chunk_type=chunk_type,
            difficulty=difficulty,
            cognitive_domain=cognitive_domain,
            hierarchy_level=hierarchy_level,
        )

        # Also execute original query for coverage
        original_results = self._execute_single_query(
            query=query,
            limit=per_query_limit,
            domain=domain,
            division=division,
            chunk_type=chunk_type,
            difficulty=difficulty,
            cognitive_domain=cognitive_domain,
            hierarchy_level=hierarchy_level,
        )
        result_sets["original"] = original_results

        # Build strategy weights from sub-query weights
        if strategy_weights is None:
            strategy_weights = {"original": 1.0}
            for sq in decomposed.sub_queries:
                strategy_weights[sq.text] = sq.weight

        # Phase I.3: Validate intent coverage before fusion
        intent_coverage, all_covered = self._validate_intent_coverage(
            decomposed, result_sets
        )

        # Fuse results (Phase I.4: pass primary intent for adaptive coherence)
        # Extract intent value for the fuser
        primary_intent_str = (
            decomposed.primary_intent.value
            if hasattr(decomposed.primary_intent, 'value')
            else str(decomposed.primary_intent)
        )
        fusion_result = self.fuser.fuse(
            result_sets=result_sets,
            strategy_weights=strategy_weights,
            limit=limit,
            primary_intent=primary_intent_str,
        )

        # Phase I.3: Attach intent coverage to fusion result
        fusion_result.intent_coverage = intent_coverage
        fusion_result.all_intents_covered = all_covered

        # Log retrieval ranking decision
        if self.capture:
            total_candidates = sum(len(rs) for rs in result_sets.values())
            self.capture.log_decision(
                decision_type="retrieval_ranking",
                decision=f"Ranked {len(fusion_result.results)} results from {total_candidates} candidates",
                rationale=(
                    f"Fusion method: {fusion_result.fusion_method}, "
                    f"Intent coverage: {intent_coverage}, "
                    f"All intents covered: {all_covered}"
                ),
            )

        return fusion_result

    def retrieve_with_decomposition(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> tuple[FusionResult, DecomposedQuery]:
        """Retrieve and return both results and decomposition details.

        Args:
            query: User query
            limit: Maximum results
            **kwargs: Additional arguments passed to retrieve()

        Returns:
            Tuple of (FusionResult, DecomposedQuery)
        """
        decomposed = self.decomposer.decompose(query)
        results = self.retrieve(query=query, limit=limit, decompose=True, **kwargs)
        return results, decomposed

    def _validate_intent_coverage(
        self,
        decomposed: DecomposedQuery,
        result_sets: dict[str, list[RetrievalResult]],
    ) -> tuple[dict[str, int], bool]:
        """Validate that all query intents received results.

        Phase I.3: Tracks which sub-query aspects received results,
        enabling detection of coverage gaps in retrieval.

        Args:
            decomposed: DecomposedQuery with sub-queries
            result_sets: Dict mapping sub-query text to results

        Returns:
            Tuple of (intent_coverage dict, all_covered boolean)
            - intent_coverage: {aspect_name: result_count} for each sub-query
            - all_covered: True if all sub-queries got ≥1 result
        """
        intent_coverage = {}
        all_covered = True

        for sq in decomposed.sub_queries:
            # Get result count for this sub-query
            results = result_sets.get(sq.text, [])
            count = len(results)

            # Track by aspect name for readability
            aspect_name = sq.aspect.value if hasattr(sq.aspect, 'value') else str(sq.aspect)
            intent_coverage[aspect_name] = count

            if count == 0:
                all_covered = False

        # Also track original query coverage
        if "original" in result_sets:
            intent_coverage["ORIGINAL"] = len(result_sets["original"])

        return intent_coverage, all_covered

    def _execute_sub_queries(
        self,
        decomposed: DecomposedQuery,
        per_query_limit: int,
        domain: Optional[str],
        division: Optional[str],
        chunk_type: Optional[str],
        difficulty: Optional[str],
        cognitive_domain: Optional[str] = None,
        hierarchy_level: Optional[str] = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Execute sub-queries in parallel.

        Args:
            decomposed: Decomposed query with sub-queries
            per_query_limit: Max results per sub-query
            domain: Domain filter
            division: Division filter
            chunk_type: Chunk type filter
            difficulty: Difficulty filter
            cognitive_domain: Cognitive domain filter (Wave 70)
            hierarchy_level: LO hierarchy level filter (Wave 70)

        Returns:
            Dict mapping sub-query text to results
        """
        result_sets = {}

        def execute_one(sub_query: SubQuery) -> tuple[str, list[RetrievalResult]]:
            # Prefer sub-query chunk types, fall back to filter
            sq_chunk_type = (
                sub_query.chunk_types[0] if sub_query.chunk_types
                else chunk_type
            )
            # Prefer sub-query bloom level for difficulty
            sq_difficulty = sub_query.bloom_level or difficulty

            results = self._execute_single_query(
                query=sub_query.text,
                limit=per_query_limit,
                domain=domain,
                division=division,
                chunk_type=sq_chunk_type,
                difficulty=sq_difficulty,
                cognitive_domain=cognitive_domain,
                hierarchy_level=hierarchy_level,
            )
            return sub_query.text, results

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(execute_one, sq)
                for sq in decomposed.sub_queries
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    query_text, results = future.result()
                    result_sets[query_text] = results
                except Exception as e:
                    # Log error but continue with other queries
                    print(f"Sub-query failed: {e}")

        return result_sets

    def _execute_single_query(
        self,
        query: str,
        limit: int,
        domain: Optional[str],
        division: Optional[str],
        chunk_type: Optional[str],
        difficulty: Optional[str],
        cognitive_domain: Optional[str] = None,
        hierarchy_level: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Execute a single query using the base retriever.

        Args:
            query: Query string
            limit: Maximum results
            domain: Domain filter
            division: Division filter
            chunk_type: Chunk type filter
            difficulty: Difficulty filter
            cognitive_domain: Cognitive domain filter (Wave 70)
            hierarchy_level: LO hierarchy level filter (Wave 70)

        Returns:
            List of RetrievalResult objects
        """
        # Execute retrieval with individual filter parameters
        results = retrieve_chunks(
            query=query,
            repo_root=self.repo_root,
            limit=limit,
            domain=domain,
            division=division,
            chunk_type=chunk_type,
            difficulty=difficulty,
            cognitive_domain=cognitive_domain,
            hierarchy_level=hierarchy_level,
        )

        return results

    def explain_decomposition(self, query: str) -> dict:
        """Explain how a query would be decomposed.

        Useful for debugging and understanding the decomposition process.

        Args:
            query: User query

        Returns:
            Dictionary with decomposition explanation
        """
        decomposed = self.decomposer.decompose(query)

        return {
            "original_query": query,
            "detected_intent": decomposed.primary_intent.value,
            "detected_bloom_level": decomposed.bloom_level,
            "extracted_concepts": decomposed.detected_concepts,
            "domain_hints": decomposed.domain_hints,
            "sub_queries": [
                {
                    "text": sq.text,
                    "aspect": sq.aspect.value,
                    "weight": sq.weight,
                    "chunk_types": sq.chunk_types,
                }
                for sq in decomposed.sub_queries
            ],
            "total_sub_queries": len(decomposed.sub_queries),
        }


def multi_retrieve(
    query: str,
    repo_root: Optional[Path] = None,
    limit: int = 10,
    decompose: bool = True,
    **kwargs,
) -> FusionResult:
    """Convenience function for multi-query retrieval.

    Args:
        query: User query
        repo_root: LibV2 repository root
        limit: Maximum results
        decompose: Whether to decompose the query
        **kwargs: Additional arguments

    Returns:
        FusionResult with fused results
    """
    retriever = MultiQueryRetriever(repo_root=repo_root)
    return retriever.retrieve(query=query, limit=limit, decompose=decompose, **kwargs)
