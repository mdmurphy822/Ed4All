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

import inspect
import json
import logging
import resource
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    k_values: tuple = (1, 3, 5, 10),
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


# ---------------------------------------------------------------------------
# Wave 84: A/B comparison across retrieval method presets
# ---------------------------------------------------------------------------


def _evaluate_query_with_method(
    query: EvalQuery,
    repo_root: Path,
    course_slug: str,
    method: str,
    retrieval_limit: int,
) -> EvalResult:
    """Evaluate a single query under a specific retrieval method preset.

    Wraps ``retrieve_chunks`` with ``method=method`` and computes the same
    metrics ``RetrievalEvaluator.evaluate_query`` does. Lifted out so the
    comparison harness can iterate methods without instantiating an
    Evaluator per method.
    """
    expected_set = set(query.expected_chunk_ids)
    t0 = time.perf_counter()
    results = retrieve_chunks(
        repo_root=repo_root,
        query=query.query_text,
        course_slug=course_slug,
        chunk_type=query.chunk_type,
        difficulty=query.difficulty,
        limit=retrieval_limit,
        method=method,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    retrieved_ids = [r.chunk_id for r in results]
    hit_at_1 = bool(expected_set & set(retrieved_ids[:1]))
    hit_at_5 = bool(expected_set & set(retrieved_ids[:5]))
    hit_at_10 = bool(expected_set & set(retrieved_ids[:10]))

    rr = 0.0
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in expected_set:
            rr = 1.0 / i
            break
    relevant_in_top_10 = len(expected_set & set(retrieved_ids[:10]))
    p10 = relevant_in_top_10 / min(10, len(retrieved_ids)) if retrieved_ids else 0.0

    return EvalResult(
        query_id=query.query_id,
        query_text=query.query_text,
        hit_at_1=hit_at_1,
        hit_at_5=hit_at_5,
        hit_at_10=hit_at_10,
        reciprocal_rank=rr,
        precision_at_10=p10,
        latency_ms=latency_ms,
        retrieved_chunk_ids=retrieved_ids,
        expected_chunk_ids=query.expected_chunk_ids,
        matched_chunks=list(expected_set & set(retrieved_ids)),
    )


def compare_retrieval_methods(
    repo_root: Path,
    course_slug: str,
    probe_path: Path,
    methods: List[str],
    retrieval_limit: int = 10,
) -> Dict:
    """Run a probe set under every named retrieval-method preset.

    Returns a dict of shape::

        {
          "course_slug": ...,
          "probe_path": ...,
          "total_queries": N,
          "methods": ["bm25", "hybrid", ...],
          "aggregate": {
            "bm25":   {"hit_at_1": .., "hit_at_5": .., "hit_at_10": ..,
                       "mrr": .., "map_at_10": .., "avg_latency_ms": ..},
            "hybrid": {...},
          },
          "per_query": [
            {"query_id": "...", "query_text": "...",
             "expected_chunk_ids": [...],
             "results": {
                "bm25":   {"hit_at_1": bool, "rr": float, "top1": "..."},
                "hybrid": {...},
             }},
            ...
          ],
        }

    Pure: no I/O beyond reading the probe file and dispatching retrievals
    through ``retrieve_chunks`` with the given method preset. Caller can
    write the result wherever they like.
    """
    repo_root = Path(repo_root)
    probe_path = Path(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe set not found: {probe_path}")
    with open(probe_path) as f:
        data = json.load(f)
    eval_set = EvalSet.from_dict(
        {
            "course_slug": data.get("course_slug", course_slug),
            "created_timestamp": data.get(
                "created_timestamp", datetime.now().isoformat()
            ),
            "version": data.get("version", "1.0"),
            "queries": data.get("queries", []),
            "description": data.get("description"),
        }
    )
    if not eval_set.queries:
        raise ValueError(f"No queries in probe set: {probe_path}")

    aggregate: Dict[str, Dict[str, float]] = {}
    per_query: List[Dict] = []

    # Run each query under each method. Outer loop = query so we can build
    # the per-query diff table inline.
    for q in eval_set.queries:
        row: Dict = {
            "query_id": q.query_id,
            "query_text": q.query_text,
            "expected_chunk_ids": q.expected_chunk_ids,
            "results": {},
        }
        for method in methods:
            r = _evaluate_query_with_method(
                q, repo_root, course_slug, method, retrieval_limit
            )
            row["results"][method] = {
                "hit_at_1": r.hit_at_1,
                "hit_at_5": r.hit_at_5,
                "hit_at_10": r.hit_at_10,
                "rr": round(r.reciprocal_rank, 4),
                "precision_at_10": round(r.precision_at_10, 4),
                "latency_ms": round(r.latency_ms, 2),
                "top1": r.retrieved_chunk_ids[0] if r.retrieved_chunk_ids else None,
                "matched_in_top_10": r.matched_chunks,
            }
            agg = aggregate.setdefault(
                method,
                {
                    "_n": 0,
                    "_h1": 0,
                    "_h5": 0,
                    "_h10": 0,
                    "_rr": 0.0,
                    "_p10": 0.0,
                    "_lat": 0.0,
                },
            )
            agg["_n"] += 1
            agg["_h1"] += int(r.hit_at_1)
            agg["_h5"] += int(r.hit_at_5)
            agg["_h10"] += int(r.hit_at_10)
            agg["_rr"] += r.reciprocal_rank
            agg["_p10"] += r.precision_at_10
            agg["_lat"] += r.latency_ms
        per_query.append(row)

    # Finalize aggregates
    final_aggregate: Dict[str, Dict[str, float]] = {}
    for method, agg in aggregate.items():
        n = max(agg["_n"], 1)
        final_aggregate[method] = {
            "hit_at_1": round(agg["_h1"] / n, 4),
            "hit_at_5": round(agg["_h5"] / n, 4),
            "hit_at_10": round(agg["_h10"] / n, 4),
            "mrr": round(agg["_rr"] / n, 4),
            "map_at_10": round(agg["_p10"] / n, 4),
            "avg_latency_ms": round(agg["_lat"] / n, 2),
        }

    return {
        "course_slug": course_slug,
        "probe_path": str(probe_path),
        "eval_timestamp": datetime.now().isoformat(),
        "total_queries": len(eval_set.queries),
        "methods": list(methods),
        "aggregate": final_aggregate,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# WS2 — BM25-vs-semantic benchmark harness
# ---------------------------------------------------------------------------
#
# ``benchmark_retrieval_engines`` runs each retrieval engine over a course's
# gold set and emits Recall@{1,3,5,10} + MRR + latency percentiles + memory +
# per-question detail to ``retrieval_eval/benchmark_<ts>.json``. It is the
# milestone-gate evidence that semantic retrieval beats the BM25 baseline.
#
# Engine invocation flows through the orthogonal ``engine=`` axis on
# ``retrieve_chunks`` (the semantic retrieval path). That axis lands with a
# parallel executor; this harness is import-guarded so it *fails honestly*
# (a clear ``NotImplementedError`` naming the missing dependency) rather than
# silently returning fake results when the semantic path is not yet wired.
# The ``bm25`` engine needs no axis — it routes through the existing
# ``method="bm25"`` boost preset.
#
# The GUI ``llm_rerank`` mode is LLM-backed and MUST NOT enter this harness
# (the benchmark is a deterministic retrieval-vs-retrieval comparison with no
# LLM-judge step); the engine allow-list below excludes it by construction.

# Engines the benchmark accepts. ``llm_rerank`` is deliberately absent — it is
# an LLM-backed mode and must not enter a deterministic retrieval benchmark.
_BENCHMARK_ENGINES = ("bm25", "semantic", "hybrid-rrf")
_LEXICAL_ENGINES = frozenset({"bm25", "lexical"})


def _supports_engine_axis() -> Tuple[bool, str]:
    """Return ``(available, reason)`` for the semantic ``engine=`` axis.

    Import-guard: the semantic retrieval path (``engine="semantic"`` /
    ``engine="hybrid-rrf"`` on ``retrieve_chunks``) lands with a parallel
    executor. We probe two signals:

    1. ``retrieve_chunks`` exposes an ``engine`` parameter, AND
    2. ``LibV2.tools.libv2.semantic_retriever`` is importable.

    When either is absent the harness raises ``NotImplementedError`` rather
    than fabricating results (anti-silent-degradation).
    """
    try:
        sig = inspect.signature(retrieve_chunks)
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return False, "could not introspect retrieve_chunks signature"
    if "engine" not in sig.parameters:
        return (
            False,
            "retrieve_chunks does not yet accept an `engine` parameter "
            "(the semantic retrieval path has not landed)",
        )
    try:
        import importlib

        importlib.import_module("LibV2.tools.libv2.semantic_retriever")
    except Exception as exc:  # noqa: BLE001 — any import failure means unavailable
        return (
            False,
            f"LibV2.tools.libv2.semantic_retriever is not importable: {exc}",
        )
    return True, ""


def _retrieve_for_engine(
    *,
    repo_root: Path,
    query: str,
    course_slug: str,
    engine: str,
    limit: int,
) -> Tuple[List[str], float, Optional[float]]:
    """Run one query under ``engine``; return ``(chunk_ids, embed_ms, search_ms)``.

    The lexical (``bm25``) arm routes through the long-standing
    ``method="bm25"`` preset and reports a single wall-clock latency
    (``embed_ms`` is the total, ``search_ms`` is ``None``). The semantic /
    hybrid arms route through the ``engine=`` axis; their latency is reported
    as a single wall-clock figure too (the embed/search split is exposed by
    the retrieval path's rationale when available, but the harness does not
    depend on it being present).
    """
    if engine in _LEXICAL_ENGINES:
        t0 = time.perf_counter()
        results = retrieve_chunks(
            repo_root=repo_root,
            query=query,
            course_slug=course_slug,
            limit=limit,
            method="bm25",
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return [r.chunk_id for r in results], wall_ms, None

    # Semantic / hybrid — gated by the import guard.
    available, reason = _supports_engine_axis()
    if not available:
        raise NotImplementedError(
            f"benchmark engine {engine!r} requires the semantic retrieval "
            f"`engine=` axis, which is unavailable: {reason}. Build the "
            f"semantic index and ensure the semantic retrieval path is wired "
            f"before benchmarking non-lexical engines. (No fake results are "
            f"emitted.)"
        )
    t0 = time.perf_counter()
    results = retrieve_chunks(
        repo_root=repo_root,
        query=query,
        course_slug=course_slug,
        limit=limit,
        engine=engine,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return [r.chunk_id for r in results], wall_ms, None


def _gold_questions_from_ws1(
    gold_doc: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Project a WS1 gold-set doc into the harness's per-question shape.

    Splits ``relevant_passages[]`` into a primary set (``relevance ==
    "primary"`` — the headline-metric ground truth) and an any-relevant set
    (primary + supporting). The chunk_id join is valid because the gold set
    and the index both pin the same chunkset sha (enforced by the loader's
    pin check upstream).
    """
    out: List[Dict[str, Any]] = []
    for q in gold_doc.get("questions", []) or []:
        if not isinstance(q, dict):
            continue
        passages = q.get("relevant_passages", []) or []
        primary = [
            str(p.get("chunk_id"))
            for p in passages
            if isinstance(p, dict) and p.get("relevance") == "primary"
            and p.get("chunk_id")
        ]
        any_rel = [
            str(p.get("chunk_id"))
            for p in passages
            if isinstance(p, dict) and p.get("chunk_id")
        ]
        out.append(
            {
                "question_id": q.get("question_id") or "",
                "question_text": q.get("question_text") or "",
                "question_type": q.get("question_type"),
                "primary_chunk_ids": primary,
                "any_relevant_chunk_ids": any_rel,
            }
        )
    return out


def _gold_questions_from_legacy(
    gold_queries_path: Path,
) -> List[Dict[str, Any]]:
    """Project the legacy ``retrieval/gold_queries.jsonl`` shape (deprecated).

    Each JSONL record: ``{"id", "query", "relevant_chunk_ids", "kind",
    "notes"?}``. The legacy format has no primary/supporting distinction, so
    every relevant chunk is treated as both primary and any-relevant.
    """
    out: List[Dict[str, Any]] = []
    with gold_queries_path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{gold_queries_path}:{line_num} — invalid JSON: {exc}"
                ) from exc
            relevant = [str(c) for c in (rec.get("relevant_chunk_ids") or [])]
            out.append(
                {
                    "question_id": rec.get("id") or rec.get("query_id") or "",
                    "question_text": rec.get("query") or rec.get("query_text") or "",
                    "question_type": rec.get("kind"),
                    "primary_chunk_ids": relevant,
                    "any_relevant_chunk_ids": relevant,
                }
            )
    return out


def _resolve_gold_questions(
    repo_root: Path,
    course_slug: str,
    gold_set_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], str]:
    """Resolve gold questions, preferring the WS1 gold set, then legacy JSONL.

    Returns ``(questions, source_label)``. Hard-errors on any critical WS1
    gold-set issue (incl. the chunkset-sha pin mismatch WS2 is contracted to
    enforce). Falls back to the legacy JSONL path with a deprecation warning
    only when the WS1 gold set is absent / its loader is unavailable.
    """
    course_dir = repo_root / "courses" / course_slug

    # Explicit override path: if the caller passed a JSON gold_set.json-shaped
    # file, load it directly; if it's a .jsonl, treat as legacy.
    if gold_set_path is not None:
        gold_set_path = Path(gold_set_path)
        if gold_set_path.suffix == ".jsonl":
            return _gold_questions_from_legacy(gold_set_path), str(gold_set_path)

    # Prefer the WS1 gold set (retrieval_eval/gold_set.json) via the loader.
    try:
        from lib.retrieval.gold_set import (  # type: ignore
            has_critical_issues,
            load_gold_set,
        )

        ws1_available = True
    except Exception:  # noqa: BLE001 — WS1-B not merged yet
        ws1_available = False

    if ws1_available:
        gold_doc, issues = load_gold_set(course_dir, verify=True)
        if gold_doc:
            if has_critical_issues(issues):
                crit = [
                    f"{i.code}: {i.message}"
                    for i in issues
                    if i.severity == "critical"
                ]
                raise ValueError(
                    "WS1 gold set has critical issues — refusing to benchmark "
                    "against a drifted/invalid gold set (the chunkset-sha pin "
                    "and chunk_id joins must hold). Issues:\n  - "
                    + "\n  - ".join(crit)
                )
            return (
                _gold_questions_from_ws1(gold_doc),
                str(course_dir / "retrieval_eval" / "gold_set.json"),
            )
        # gold_doc == {} → not found / unparseable; fall through to legacy.

    # Legacy fallback (deprecated).
    legacy_path = course_dir / "retrieval" / "gold_queries.jsonl"
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"no gold set found for {course_slug}: neither "
            f"{course_dir / 'retrieval_eval' / 'gold_set.json'} (WS1) nor "
            f"{legacy_path} (legacy) exists."
        )
    warnings.warn(
        f"falling back to legacy gold queries at {legacy_path}; author a "
        f"retrieval_eval/gold_set.json (WS1 schema) to use the primary/"
        f"supporting relevance distinction.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _gold_questions_from_legacy(legacy_path), str(legacy_path)


def _recall_at_k(retrieved: List[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of ``relevant`` chunk ids appearing in the top-``k`` retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant_set & top_k) / len(relevant_set)


def _reciprocal_rank(retrieved: List[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation — deterministic)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, int(round((pct / 100.0) * len(sorted_values))))
    rank = min(rank, len(sorted_values))
    return sorted_values[rank - 1]


def _index_size_bytes(repo_root: Path, course_slug: str) -> int:
    """Total on-disk bytes of the course's vector_index/ artifacts (0 if absent)."""
    index_dir = repo_root / "courses" / course_slug / "vector_index"
    if not index_dir.exists():
        return 0
    total = 0
    for p in index_dir.iterdir():
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def benchmark_retrieval_engines(
    repo_root: Path,
    course_slug: str,
    gold_set_path: Optional[Path] = None,
    engines: Sequence[str] = ("bm25", "semantic"),
    models: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
    retrieval_limit: int = 10,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Benchmark each retrieval engine over a course's gold set.

    Runs every engine in ``engines`` over the gold questions and emits
    Recall@k (primary + any_relevant), MRR, latency percentiles, memory delta,
    index size, and per-question detail. Writes ``retrieval_eval/
    benchmark_<ts>.json`` (WS2 writes only ``benchmark_*`` siblings there —
    never ``gold_set.json``).

    Gold input resolution: the WS1 gold set
    (``retrieval_eval/gold_set.json``) FIRST — via ``lib.retrieval.gold_set.
    load_gold_set(verify=True)``, hard-erroring on any critical issue
    (including the chunkset-sha pin mismatch) — then the legacy
    ``retrieval/gold_queries.jsonl`` with a deprecation warning.

    Engine invocation routes through the ``engine=`` axis (semantic / hybrid)
    or the ``method="bm25"`` preset (lexical). Non-lexical engines raise
    ``NotImplementedError`` (naming the missing dependency) when the semantic
    retrieval path is not yet wired — no fake results are ever produced. The
    GUI ``llm_rerank`` mode is excluded by construction.

    Args:
        models: when the semantic arm is benchmarked over multiple models, the
            caller (CLI) is responsible for pre-building one index per model
            into a temp suffix dir. The harness records the requested model
            list in the report config; it does not itself orchestrate the
            per-model index builds (that is wave-C CLI scope).

    Returns the report dict (also written to ``output_path``).
    """
    repo_root = Path(repo_root)
    k_values = tuple(int(k) for k in k_values)

    bad_engines = [e for e in engines if e not in _BENCHMARK_ENGINES]
    if bad_engines:
        raise ValueError(
            f"unknown benchmark engine(s) {bad_engines}; allowed: "
            f"{list(_BENCHMARK_ENGINES)}. (The LLM-backed 'llm_rerank' mode is "
            f"intentionally excluded from the deterministic benchmark.)"
        )

    questions, gold_source = _resolve_gold_questions(
        repo_root, course_slug, gold_set_path
    )
    if not questions:
        raise ValueError(
            f"gold set for {course_slug} ({gold_source}) has no questions."
        )

    per_engine: Dict[str, Any] = {}
    per_query_rows: List[Dict[str, Any]] = []

    # Per-query rows accumulate one block per engine; build per-engine
    # aggregates in a parallel pass.
    engine_query_metrics: Dict[str, List[Dict[str, Any]]] = {e: [] for e in engines}

    for q in questions:
        qid = q["question_id"]
        qtext = q["question_text"]
        primary = q["primary_chunk_ids"]
        any_rel = q["any_relevant_chunk_ids"]
        row: Dict[str, Any] = {
            "question_id": qid,
            "question_text": qtext,
            "question_type": q.get("question_type"),
            "primary_chunk_ids": primary,
            "any_relevant_chunk_ids": any_rel,
            "engines": {},
        }
        for engine in engines:
            retrieved, latency_ms, search_ms = _retrieve_for_engine(
                repo_root=repo_root,
                query=qtext,
                course_slug=course_slug,
                engine=engine,
                limit=retrieval_limit,
            )
            primary_recall = {
                k: _recall_at_k(retrieved, primary, k) for k in k_values
            }
            any_recall = {
                k: _recall_at_k(retrieved, any_rel, k) for k in k_values
            }
            rr = _reciprocal_rank(retrieved, primary)
            row["engines"][engine] = {
                "retrieved_chunk_ids": retrieved,
                "recall_at_k_primary": {
                    str(k): round(primary_recall[k], 4) for k in k_values
                },
                "recall_at_k_any_relevant": {
                    str(k): round(any_recall[k], 4) for k in k_values
                },
                "reciprocal_rank": round(rr, 4),
                "latency_ms": round(latency_ms, 3),
            }
            engine_query_metrics[engine].append(
                {
                    "primary_recall": primary_recall,
                    "any_recall": any_recall,
                    "rr": rr,
                    "latency_ms": latency_ms,
                    "search_ms": search_ms,
                }
            )
        per_query_rows.append(row)

    n = len(questions)
    for engine in engines:
        rows = engine_query_metrics[engine]
        latencies = sorted(r["latency_ms"] for r in rows)
        recall_primary = {
            k: round(sum(r["primary_recall"][k] for r in rows) / n, 4)
            for k in k_values
        }
        recall_any = {
            k: round(sum(r["any_recall"][k] for r in rows) / n, 4)
            for k in k_values
        }
        mrr = round(sum(r["rr"] for r in rows) / n, 4) if n else 0.0
        per_engine[engine] = {
            "recall_at_k_primary": {str(k): recall_primary[k] for k in k_values},
            "recall_at_k_any_relevant": {str(k): recall_any[k] for k in k_values},
            "mrr": mrr,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3)
            if latencies
            else 0.0,
            "p50_latency_ms": round(_percentile(latencies, 50), 3),
            "p95_latency_ms": round(_percentile(latencies, 95), 3),
            "index_size_bytes": _index_size_bytes(repo_root, course_slug)
            if engine not in _LEXICAL_ENGINES
            else 0,
        }

    # Winner block: each semantic/hybrid arm vs the bm25 baseline.
    winner: Dict[str, Any] = {}
    if "bm25" in per_engine:
        baseline = per_engine["bm25"]
        for engine in engines:
            if engine == "bm25":
                continue
            arm = per_engine[engine]
            winner[engine] = {
                "delta_recall_at_k_primary": {
                    str(k): round(
                        arm["recall_at_k_primary"][str(k)]
                        - baseline["recall_at_k_primary"][str(k)],
                        4,
                    )
                    for k in k_values
                },
                "delta_mrr": round(arm["mrr"] - baseline["mrr"], 4),
            }

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    report: Dict[str, Any] = {
        "course_slug": course_slug,
        "benchmark_timestamp": datetime.now().isoformat(),
        "gold_source": gold_source,
        "config": {
            "engines": list(engines),
            "models": list(models) if models else None,
            "k_values": list(k_values),
            "retrieval_limit": retrieval_limit,
            "total_questions": n,
        },
        "per_engine": per_engine,
        "winner": winner,
        "peak_rss_bytes": int(peak_rss),
        "per_query": per_query_rows,
    }

    if output_path is None:
        results_dir = repo_root / "courses" / course_slug / "retrieval_eval"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"benchmark_{ts}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)
    report["output_path"] = str(output_path)
    return report


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
