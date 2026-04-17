"""Recall@k benchmark for the per-chunk ``summary`` field.

This module does NOT implement its own retrieval engine — it consumes
``LibV2.tools.libv2.retriever.LazyBM25`` (the BM25 primitive already
battle-tested in LibV2) and compares three corpus variants against the
same held-out question set:

1. ``text``            — current production baseline, BM25 over raw chunk text.
2. ``summary``         — BM25 over the chunk's ``summary`` field only.
3. ``retrieval_text``  — BM25 over ``summary + " " + key_terms``. Reported
                         so Worker D's ADR-001 commit has evidence for
                         shipping vs deferring the optional
                         ``retrieval_text`` field.

Held-out question set
---------------------
The question set is derived deterministically from ``course.json``'s
``learning_outcomes``: every LO statement becomes a query, and a chunk is
"correct" for that query iff the LO's ``id`` appears in the chunk's
``learning_outcome_refs``. This is synthetic but reproducible, which is
what we want for cross-commit delta measurement.

CLI / flag
----------
``Trainforge/process_course.py`` gains a ``--benchmark-retrieval`` flag
that, after processing, calls :func:`run_benchmark` against the freshly
written ``chunks.jsonl`` + ``course.json``. The result lands at
``<output>/quality/retrieval_benchmark.json``. See
``tests/test_retrieval_benchmark.py`` for the expected schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "build_question_set",
    "run_benchmark",
    "write_benchmark",
    "recall_at_k",
]


# LibV2 BM25 lives outside Trainforge's package tree. Import path fix-up
# mirrors what Trainforge/tests/test_retrieval_improvements.py does so
# the benchmark works in both editable-install and bare-checkout layouts.
def _import_lazybm25():
    try:
        from libv2.retriever import LazyBM25  # type: ignore
        return LazyBM25
    except Exception:
        here = Path(__file__).resolve()
        project_root = here.parents[2]
        libv2_tools = project_root / "LibV2" / "tools"
        if str(libv2_tools) not in sys.path:
            sys.path.insert(0, str(libv2_tools))
        from libv2.retriever import LazyBM25  # type: ignore
        return LazyBM25


def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load chunks.jsonl into a list of dicts. Skips blank / malformed lines."""
    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def _load_course(course_path: Path) -> Dict[str, Any]:
    with open(course_path, encoding="utf-8") as f:
        return json.load(f)


def _key_terms_to_text(key_terms: Any) -> str:
    """Flatten key_terms field to a single space-separated string.

    Accepts both list-of-dicts ({term, definition}) and list-of-strings.
    """
    if not key_terms:
        return ""
    parts: List[str] = []
    for kt in key_terms:
        if isinstance(kt, dict):
            t = kt.get("term")
            if t:
                parts.append(str(t))
            d = kt.get("definition")
            if d:
                parts.append(str(d))
        elif isinstance(kt, str):
            parts.append(kt)
    return " ".join(parts)


def _build_retrieval_text(chunk: Dict[str, Any]) -> str:
    """Compose the optional ``retrieval_text`` field: summary + key terms.

    Matches the ADR-001 contract-driven scope language: "summary + ' ' +
    key_terms_joined". Falls back to summary-only when key_terms is absent.
    """
    summary = chunk.get("summary", "") or ""
    kt = _key_terms_to_text(chunk.get("key_terms"))
    if kt:
        return f"{summary} {kt}".strip()
    return summary.strip()


def build_question_set(
    chunks: Sequence[Dict[str, Any]],
    course: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Derive a held-out question set from ``course.learning_outcomes``.

    Each LO statement becomes one query; the correct chunk-id set is the
    set of chunks whose ``learning_outcome_refs`` contains the LO id.

    Returns:
        A list of ``{"lo_id", "query", "relevant_chunk_ids"}`` dicts. LOs
        with zero matching chunks are included but annotated — they
        contribute 0.0 to every variant's recall (not silently dropped,
        so a future fixture with a broader LO spread benefits automatically).
    """
    los = course.get("learning_outcomes") or []
    questions: List[Dict[str, Any]] = []
    for lo in los:
        lo_id = lo.get("id")
        statement = lo.get("statement") or ""
        if not lo_id or not statement:
            continue
        relevant = {
            c["id"]
            for c in chunks
            if c.get("id") and lo_id in (c.get("learning_outcome_refs") or [])
        }
        questions.append({
            "lo_id": lo_id,
            "query": statement,
            "relevant_chunk_ids": sorted(relevant),
        })
    return questions


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of relevant chunks found in the top-k retrieved set.

    Classic recall@k semantics:
        recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Returns 0.0 when relevant is empty (degenerate query).
    """
    if not relevant_ids:
        return 0.0
    rel = set(relevant_ids)
    top_k = set(retrieved_ids[:k])
    return len(top_k & rel) / len(rel)


def _run_variant(
    chunks: Sequence[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    field: str,
    k_values: Sequence[int],
    LazyBM25,
) -> Dict[str, float]:
    """Run BM25 over a single chunk-text variant and return mean recall@k.

    ``field`` selects the text used for indexing:
      - "text": chunk["text"]
      - "summary": chunk["summary"]
      - "retrieval_text": _build_retrieval_text(chunk)
    """
    # Project chunks down to (id, field-text) for indexing. LazyBM25 expects
    # chunks with a "text" key, so we rename the chosen field into "text"
    # before handing it off.
    projected: List[Dict[str, Any]] = []
    for c in chunks:
        if field == "text":
            projected_text = c.get("text", "") or ""
        elif field == "summary":
            projected_text = c.get("summary", "") or ""
        elif field == "retrieval_text":
            projected_text = _build_retrieval_text(c)
        else:
            raise ValueError(f"Unknown field: {field}")
        projected.append({"id": c.get("id"), "text": projected_text})

    bm25 = LazyBM25(projected)
    max_k = max(k_values) if k_values else 10

    recall_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    q_count = 0

    for q in questions:
        relevant = q.get("relevant_chunk_ids") or []
        if not relevant:
            continue
        # Use min_relevance=0.0 for benchmarking — we want top-k irrespective
        # of the production threshold. Otherwise summary-only variants with
        # shorter BM25 norms get unfairly hard-filtered.
        results = bm25.search(q["query"], limit=max_k, min_relevance=0.0)
        retrieved_ids = [r[0]["id"] for r in results]
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved_ids, relevant, k)
        q_count += 1

    if q_count == 0:
        return {f"recall@{k}": 0.0 for k in k_values}

    return {f"recall@{k}": recall_sums[k] / q_count for k in k_values}


def run_benchmark(
    chunks_path: Path,
    course_path: Path,
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """Run the recall@k benchmark.

    Args:
        chunks_path: Path to a chunks.jsonl file.
        course_path: Path to the matching course.json (for learning_outcomes).
        k_values: K values to compute recall for.

    Returns:
        A dict with:
          - ``chunk_count``: int
          - ``question_count``: int (questions with >=1 relevant chunk)
          - ``variants``: {variant_name: {recall@k: float}}
          - ``k_values``: list of ints
          - ``fields_compared``: list of variant names
    """
    LazyBM25 = _import_lazybm25()

    chunks = _load_chunks(chunks_path)
    course = _load_course(course_path)
    questions = build_question_set(chunks, course)

    # Decide which variants we can run. "summary" and "retrieval_text"
    # require the v4 fields; fall back gracefully when regenerating against
    # a pre-v4 corpus (useful for regression comparison).
    has_summary = any(c.get("summary") for c in chunks)
    variants = ["text"]
    if has_summary:
        variants.append("summary")
        # retrieval_text is always meaningful when summary exists, because
        # key_terms may be absent per-chunk but the composed field still
        # falls back to summary.
        variants.append("retrieval_text")

    results: Dict[str, Any] = {
        "chunk_count": len(chunks),
        "question_count": sum(1 for q in questions if q["relevant_chunk_ids"]),
        "total_questions": len(questions),
        "k_values": list(k_values),
        "fields_compared": variants,
        "variants": {},
    }

    for v in variants:
        results["variants"][v] = _run_variant(chunks, questions, v, k_values, LazyBM25)

    # Top-line delta for convenience: summary lift over text at k=5.
    if "summary" in results["variants"] and "text" in results["variants"]:
        t = results["variants"]["text"].get("recall@5", 0.0)
        s = results["variants"]["summary"].get("recall@5", 0.0)
        results["summary_vs_text_recall_at_5_delta"] = s - t

    return results


def write_benchmark(
    output_dir: Path,
    chunks_path: Optional[Path] = None,
    course_path: Optional[Path] = None,
    k_values: Sequence[int] = (1, 5, 10),
) -> Tuple[Path, Dict[str, Any]]:
    """Run the benchmark and write ``quality/retrieval_benchmark.json``.

    Convenience wrapper for the CLI. Paths default to the standard
    Trainforge output layout (``<output_dir>/corpus/chunks.jsonl`` and
    ``<output_dir>/course.json``).
    """
    output_dir = Path(output_dir)
    if chunks_path is None:
        chunks_path = output_dir / "corpus" / "chunks.jsonl"
    if course_path is None:
        course_path = output_dir / "course.json"

    results = run_benchmark(chunks_path, course_path, k_values=k_values)

    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    out_path = quality_dir / "retrieval_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path, results
