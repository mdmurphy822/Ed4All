"""Retrieval evaluation harness for LibV2.

Provides evaluation of retrieval quality with standard IR metrics:
- Hit@k: Did a relevant chunk appear in top k?
- MRR: Mean Reciprocal Rank
- MAP@k: Mean Average Precision at k
- Latency: Query response time

Usage:
    evaluator = RetrievalEvaluator(repo_root, course_slug)
    report = evaluator.run_evaluation(eval_set_path)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .retriever import retrieve_chunks

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with expected results."""
    query_id: str
    query_text: str
    expected_chunk_ids: List[str]
    domain: Optional[str] = None
    chunk_type: Optional[str] = None
    difficulty: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalQuery":
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            expected_chunk_ids=data["expected_chunk_ids"],
            domain=data.get("domain"),
            chunk_type=data.get("chunk_type"),
            difficulty=data.get("difficulty"),
            notes=data.get("notes"),
        )

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EvalResult:
    """Result for a single evaluation query."""
    query_id: str
    query_text: str
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    reciprocal_rank: float
    precision_at_10: float
    latency_ms: float
    retrieved_chunk_ids: List[str]
    expected_chunk_ids: List[str]
    matched_chunks: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    course_slug: str
    eval_timestamp: str
    total_queries: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr: float  # Mean Reciprocal Rank
    map_at_10: float  # Mean Average Precision at 10
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    query_results: List[EvalResult] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "course_slug": self.course_slug,
            "eval_timestamp": self.eval_timestamp,
            "summary": {
                "total_queries": self.total_queries,
                "hit_at_1": round(self.hit_at_1, 4),
                "hit_at_5": round(self.hit_at_5, 4),
                "hit_at_10": round(self.hit_at_10, 4),
                "mrr": round(self.mrr, 4),
                "map_at_10": round(self.map_at_10, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "min_latency_ms": round(self.min_latency_ms, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
            },
            "query_results": [r.to_dict() for r in self.query_results],
            "metadata": self.metadata,
        }


@dataclass
class EvalSet:
    """A set of evaluation queries for a course."""
    course_slug: str
    created_timestamp: str
    queries: List[EvalQuery]
    description: Optional[str] = None
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalSet":
        return cls(
            course_slug=data["course_slug"],
            created_timestamp=data["created_timestamp"],
            queries=[EvalQuery.from_dict(q) for q in data["queries"]],
            description=data.get("description"),
            version=data.get("version", "1.0"),
        )

    def to_dict(self) -> Dict:
        return {
            "course_slug": self.course_slug,
            "created_timestamp": self.created_timestamp,
            "version": self.version,
            "description": self.description,
            "queries": [q.to_dict() for q in self.queries],
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality against an eval set."""

    def __init__(
        self,
        repo_root: Path,
        course_slug: str,
        retrieval_limit: int = 10,
    ):
        """
        Initialize evaluator.

        Args:
            repo_root: Path to LibV2 repository root
            course_slug: Course to evaluate
            retrieval_limit: Max results per query (default 10)
        """
        self.repo_root = Path(repo_root)
        self.course_slug = course_slug
        self.retrieval_limit = retrieval_limit

    def load_eval_set(self, eval_set_path: Path) -> EvalSet:
        """Load evaluation set from JSON file."""
        with open(eval_set_path) as f:
            data = json.load(f)
        return EvalSet.from_dict(data)

    def evaluate_query(self, query: EvalQuery) -> EvalResult:
        """
        Evaluate a single query.

        Args:
            query: EvalQuery to evaluate

        Returns:
            EvalResult with metrics
        """
        expected_set = set(query.expected_chunk_ids)

        # Time the retrieval
        start_time = time.perf_counter()

        results = retrieve_chunks(
            repo_root=self.repo_root,
            query=query.query_text,
            course_slug=self.course_slug,
            chunk_type=query.chunk_type,
            difficulty=query.difficulty,
            limit=self.retrieval_limit,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract retrieved chunk IDs
        retrieved_ids = [r.chunk_id for r in results]
        retrieved_set = set(retrieved_ids)

        # Calculate metrics
        matched_chunks = list(expected_set & retrieved_set)

        # Hit@k: Did any expected chunk appear in top k?
        hit_at_1 = len(expected_set & set(retrieved_ids[:1])) > 0
        hit_at_5 = len(expected_set & set(retrieved_ids[:5])) > 0
        hit_at_10 = len(expected_set & set(retrieved_ids[:10])) > 0

        # Reciprocal Rank: 1/rank of first relevant result
        reciprocal_rank = 0.0
        for i, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in expected_set:
                reciprocal_rank = 1.0 / i
                break

        # Precision@10: Fraction of top 10 that are relevant
        relevant_in_top_10 = len(expected_set & set(retrieved_ids[:10]))
        precision_at_10 = relevant_in_top_10 / min(10, len(retrieved_ids)) if retrieved_ids else 0.0

        return EvalResult(
            query_id=query.query_id,
            query_text=query.query_text,
            hit_at_1=hit_at_1,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            reciprocal_rank=reciprocal_rank,
            precision_at_10=precision_at_10,
            latency_ms=latency_ms,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=query.expected_chunk_ids,
            matched_chunks=matched_chunks,
        )

    def run_evaluation(
        self,
        eval_set: EvalSet,
        verbose: bool = False,
    ) -> EvalReport:
        """
        Run evaluation on an eval set.

        Args:
            eval_set: EvalSet to evaluate
            verbose: If True, log progress for each query

        Returns:
            EvalReport with aggregate metrics
        """
        results: List[EvalResult] = []
        latencies: List[float] = []

        for i, query in enumerate(eval_set.queries):
            if verbose:
                logger.info(f"Evaluating query {i+1}/{len(eval_set.queries)}: {query.query_id}")

            result = self.evaluate_query(query)
            results.append(result)
            latencies.append(result.latency_ms)

        # Aggregate metrics
        n = len(results)

        hit_at_1 = sum(1 for r in results if r.hit_at_1) / n if n else 0
        hit_at_5 = sum(1 for r in results if r.hit_at_5) / n if n else 0
        hit_at_10 = sum(1 for r in results if r.hit_at_10) / n if n else 0
        mrr = sum(r.reciprocal_rank for r in results) / n if n else 0
        map_at_10 = sum(r.precision_at_10 for r in results) / n if n else 0

        return EvalReport(
            course_slug=self.course_slug,
            eval_timestamp=datetime.now().isoformat(),
            total_queries=n,
            hit_at_1=hit_at_1,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            mrr=mrr,
            map_at_10=map_at_10,
            avg_latency_ms=sum(latencies) / n if n else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            query_results=results,
            metadata={
                "retrieval_limit": self.retrieval_limit,
                "eval_set_version": eval_set.version,
            },
        )


def run_course_evaluation(
    course_dir: Path,
    repo_root: Path,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> EvalReport:
    """
    Run evaluation for a course using its eval set.

    Args:
        course_dir: Path to course directory
        repo_root: Path to repository root
        output_path: Optional path to save report
        verbose: If True, show progress

    Returns:
        EvalReport
    """
    eval_set_path = course_dir / "quality" / "eval_set.json"

    if not eval_set_path.exists():
        raise FileNotFoundError(f"No eval set found at {eval_set_path}")

    course_slug = course_dir.name
    evaluator = RetrievalEvaluator(repo_root, course_slug)

    eval_set = evaluator.load_eval_set(eval_set_path)
    report = evaluator.run_evaluation(eval_set, verbose=verbose)

    # Save report if output path specified
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation report to {output_path}")
    else:
        # Save to quality/eval_results/
        results_dir = course_dir / "quality" / "eval_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"eval_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation report to {result_path}")

    return report


def evaluate_retrieval(
    course_slug: str,
    repo_root: Path,
    gold_queries_path: Optional[Path] = None,
    include_rationale: bool = True,
    metadata_scoring: bool = True,
    k_values: tuple = (1, 5, 10),
    retrieval_limit: int = 10,
    output_path: Optional[Path] = None,
) -> Dict:
    """Run hand-curated gold queries against retrieve_chunks and compute
    recall@k + MRR + per-query rationale.

    This wraps the lower-level ``RetrievalEvaluator`` so Worker J's reference
    retrieval implementation has a single entry point for the
    ``libv2 retrieval-eval`` CLI.  Gold queries live at
    ``LibV2/courses/<slug>/retrieval/gold_queries.jsonl`` by default (one
    JSON record per line).  Each record shape:

        {"id": str, "query": str, "relevant_chunk_ids": [str],
         "kind": "hand-curated" | "lo-derived", "notes": str (optional)}

    Returns a dict with per-query entries and aggregate metrics
    (MRR, recall@1, recall@5, recall@10).  Writes the report JSON to
    ``output_path`` if provided, else to
    ``LibV2/courses/<slug>/retrieval/evaluation_results.json``.

    Deterministic: pure function of gold_queries + course contents.
    """
    repo_root = Path(repo_root)
    if gold_queries_path is None:
        gold_queries_path = (
            repo_root / "courses" / course_slug / "retrieval" / "gold_queries.jsonl"
        )
    if not gold_queries_path.exists():
        raise FileNotFoundError(f"Gold queries file not found: {gold_queries_path}")

    # Load gold queries (JSONL, one record per line)
    queries: List[Dict] = []
    with open(gold_queries_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{gold_queries_path}:{line_num} — invalid JSON: {e}"
                ) from e

    if not queries:
        raise ValueError(f"No gold queries found in {gold_queries_path}")

    per_query: List[Dict] = []
    latencies: List[float] = []

    for q in queries:
        qid = q.get("id") or q.get("query_id") or ""
        qtext = q.get("query") or q.get("query_text") or ""
        relevant = [str(r) for r in (q.get("relevant_chunk_ids") or [])]
        relevant_set = set(relevant)
        if not qtext or not relevant:
            continue

        t0 = time.perf_counter()
        results = retrieve_chunks(
            repo_root=repo_root,
            query=qtext,
            course_slug=course_slug,
            limit=retrieval_limit,
            include_rationale=include_rationale,
            metadata_scoring=metadata_scoring,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r.chunk_id for r in results]
        retrieved_set = set(retrieved_ids)
        matched = list(relevant_set & retrieved_set)

        # Rank-of-first-relevant
        rank_of_first_relevant: Optional[int] = None
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_set:
                rank_of_first_relevant = rank
                break
        reciprocal_rank = (1.0 / rank_of_first_relevant) if rank_of_first_relevant else 0.0

        # Recall@k = fraction of relevant chunks found in top-k
        recall_at_k: Dict[int, float] = {}
        for k in k_values:
            found = len(relevant_set & set(retrieved_ids[:k]))
            recall_at_k[k] = found / len(relevant_set) if relevant_set else 0.0

        entry = {
            "id": qid,
            "query": qtext,
            "kind": q.get("kind"),
            "notes": q.get("notes"),
            "relevant_chunk_ids": relevant,
            "retrieved_chunk_ids": retrieved_ids,
            "matched_chunk_ids": matched,
            "rank_of_first_relevant": rank_of_first_relevant,
            "reciprocal_rank": round(reciprocal_rank, 4),
            **{f"recall_at_{k}": round(recall_at_k[k], 4) for k in k_values},
            "latency_ms": round(latency_ms, 2),
        }
        if include_rationale:
            # Include rationale for the top-ranked result only (keeps the
            # report small; full per-result rationale is accessible via the
            # retrieve CLI).
            entry["top_result_rationale"] = results[0].rationale if results else None
        per_query.append(entry)

    # Aggregate
    n = len(per_query)
    aggregate: Dict[str, float] = {
        "mrr": round(sum(q["reciprocal_rank"] for q in per_query) / n, 4) if n else 0.0,
        "total_queries": n,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }
    for k in k_values:
        key = f"recall_at_{k}"
        aggregate[key] = round(sum(q[key] for q in per_query) / n, 4) if n else 0.0

    report = {
        "course_slug": course_slug,
        "eval_timestamp": datetime.now().isoformat(),
        "gold_queries_path": str(gold_queries_path),
        "aggregate": aggregate,
        "per_query": per_query,
        "config": {
            "retrieval_limit": retrieval_limit,
            "include_rationale": include_rationale,
            "metadata_scoring": metadata_scoring,
            "k_values": list(k_values),
        },
    }

    # Write
    if output_path is None:
        output_path = (
            repo_root / "courses" / course_slug / "retrieval" / "evaluation_results.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def compare_reports(
    report1_path: Path,
    report2_path: Path,
) -> Dict:
    """
    Compare two evaluation reports.

    Args:
        report1_path: Path to first report (baseline)
        report2_path: Path to second report (comparison)

    Returns:
        Dict with comparison metrics
    """
    with open(report1_path) as f:
        report1 = json.load(f)
    with open(report2_path) as f:
        report2 = json.load(f)

    s1 = report1["summary"]
    s2 = report2["summary"]

    comparison = {
        "baseline": {
            "path": str(report1_path),
            "timestamp": report1["eval_timestamp"],
        },
        "comparison": {
            "path": str(report2_path),
            "timestamp": report2["eval_timestamp"],
        },
        "changes": {
            "hit_at_1": {
                "baseline": s1["hit_at_1"],
                "comparison": s2["hit_at_1"],
                "delta": s2["hit_at_1"] - s1["hit_at_1"],
            },
            "hit_at_5": {
                "baseline": s1["hit_at_5"],
                "comparison": s2["hit_at_5"],
                "delta": s2["hit_at_5"] - s1["hit_at_5"],
            },
            "hit_at_10": {
                "baseline": s1["hit_at_10"],
                "comparison": s2["hit_at_10"],
                "delta": s2["hit_at_10"] - s1["hit_at_10"],
            },
            "mrr": {
                "baseline": s1["mrr"],
                "comparison": s2["mrr"],
                "delta": s2["mrr"] - s1["mrr"],
            },
            "map_at_10": {
                "baseline": s1["map_at_10"],
                "comparison": s2["map_at_10"],
                "delta": s2["map_at_10"] - s1["map_at_10"],
            },
            "avg_latency_ms": {
                "baseline": s1["avg_latency_ms"],
                "comparison": s2["avg_latency_ms"],
                "delta": s2["avg_latency_ms"] - s1["avg_latency_ms"],
            },
        },
        "regression_detected": (
            s2["hit_at_10"] < s1["hit_at_10"] - 0.05 or
            s2["mrr"] < s1["mrr"] - 0.05
        ),
    }

    return comparison


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python eval_harness.py <course_dir> <repo_root>")
        sys.exit(1)

    course_dir = Path(sys.argv[1])
    repo_root = Path(sys.argv[2])

    try:
        report = run_course_evaluation(course_dir, repo_root, verbose=True)
        print(f"\nEvaluation Results for {report.course_slug}")
        print("=" * 50)
        print(f"Total queries: {report.total_queries}")
        print(f"Hit@1: {report.hit_at_1:.1%}")
        print(f"Hit@5: {report.hit_at_5:.1%}")
        print(f"Hit@10: {report.hit_at_10:.1%}")
        print(f"MRR: {report.mrr:.4f}")
        print(f"MAP@10: {report.map_at_10:.4f}")
        print(f"Avg Latency: {report.avg_latency_ms:.1f}ms")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
