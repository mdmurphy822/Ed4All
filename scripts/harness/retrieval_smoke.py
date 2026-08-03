#!/usr/bin/env python3
"""GAP 2 — deterministic, corpus-agnostic retrieval smoke harness.

Answers one question the validation net never asked before: *given a
freshly-built LibV2 course, can the retrieval stack actually find its own
content?* A course can pass every schema / gate check and still be
un-askable if the vector index drifted, the embedder mis-fired, or the
chunkset numbering diverged from the index. This harness catches that.

What it does (all deterministic given ``--seed``):

1. Loads the course chunkset (the SAME file the live retriever serves,
   via ``lib.libv2_storage.resolve_chunks_path_for_query``).
2. Samples ``N`` *content-rich* chunks — prefers instructional prose
   (worked_example / statement / example / explanation / overview) and
   excludes apparatus (exercise / exercise_set / assessment_item /
   answer keys), using whatever pedagogical metadata the chunk carries
   (``chunk_type`` / ``content_type_label`` / ``unit_roles`` /
   ``source_block_role`` / ``composite_unit``). Falls back to length
   when no pedagogical metadata is present.
3. Forms a deterministic query per sampled chunk: ``heading + first
   sentence`` of the chunk (seeded RNG only governs *which* chunks are
   sampled, never the query text — so a re-run over the same corpus is
   byte-identical).
4. Queries the canonical retriever (``LibV2.tools.libv2.retriever.
   retrieve_chunks`` — the hybrid-rrf / bge-large path the grounded-answer
   backend uses) and measures **self-retrieval hit@k + MRR**: does the
   source chunk resurface in its own query's top-k?
5. Fires ``M`` fixed **out-of-domain refusal probes** (generic cooking /
   astrology / sports strings — corpus-agnostic) and reports the top-1
   score + top1-minus-top2 margin so an operator can see whether the
   index cleanly separates in-domain from out-of-domain.

PASS/FAIL: ``hit@k >= --hit-threshold`` (default 0.8). Exit 0 pass,
1 fail, 2 when the embedding extras / vector index are missing (with an
actionable message — the harness never silently falls back to lexical
when a semantic engine was requested).

The scoring + sampling + query-forming logic is factored into pure
functions (``select_sample`` / ``form_query`` / ``score_self_retrieval``
/ ``probe_margins`` / ``build_report``) so it is unit-testable with a
mocked retriever and no LibV2 course on disk.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Repo root on path so ``LibV2`` / ``lib`` import when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ------------------------------------------------------------------ #
# Corpus-agnostic pedagogical-preference vocabulary. Substring match
# against whatever metadata the chunk exposes — no corpus can be
# assumed to use a fixed taxonomy, so we match liberally.
# ------------------------------------------------------------------ #

#: Roles/types that mark a chunk as content-rich instructional prose.
_PREFER_SUBSTR: Tuple[str, ...] = (
    "worked_example",
    "worked-example",
    "statement",
    "example",
    "explanation",
    "overview",
    "definition",
    "concept",
    "summary",
)

#: Roles/types that mark a chunk as apparatus (poor self-retrieval
#: queries — exercises restate little unique content).
_EXCLUDE_SUBSTR: Tuple[str, ...] = (
    "exercise_set",
    "exercise",
    "assessment_item",
    "assessment",
    "answer",
    "solution",
    "apparatus",
    "quiz",
)

#: Fixed out-of-domain probes. Deliberately generic + off-topic for any
#: educational corpus so a well-separated index scores them low.
DEFAULT_OOD_PROBES: Tuple[str, ...] = (
    "What is the best recipe for a chocolate souffle?",
    "Which zodiac sign is most compatible with a Scorpio?",
    "How do I change the oil filter on a 1998 pickup truck?",
    "What time does the next commuter train to the airport leave?",
    "Who won the championship football match last weekend?",
)


# ------------------------------------------------------------------ #
# Pure functions (unit-tested).
# ------------------------------------------------------------------ #


def _chunk_id(chunk: Dict[str, Any]) -> str:
    """Return the chunk's canonical id (``id`` or legacy ``chunk_id``)."""
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _metadata_signal(chunk: Dict[str, Any]) -> str:
    """Concatenate every pedagogical-metadata field into one lowered blob.

    Corpus-agnostic: we don't know which field a given corpus populates,
    so we fold ``chunk_type`` / ``content_type_label`` / ``unit_roles`` /
    ``source_block_role`` / ``composite_unit`` into a single string and
    substring-match against it. The multi-ontology rename landed
    ``composite_unit`` / ``unit_roles`` (formerly ``pedagogical_unit`` /
    ``pedagogical_roles``); the legacy names are still read as a fallback
    so not-yet-migrated archives keep sampling correctly.
    """
    parts: List[str] = []
    for key in (
        "chunk_type",
        "content_type_label",
        "source_block_role",
        "composite_unit",
        "pedagogical_unit",  # legacy fallback (pre-rename archives)
    ):
        val = chunk.get(key)
        if isinstance(val, str):
            parts.append(val)
    for roles_key in ("unit_roles", "pedagogical_roles"):
        roles = chunk.get(roles_key)
        if isinstance(roles, list):
            parts.extend(str(r) for r in roles)
    return " ".join(parts).lower()


def _content_length(chunk: Dict[str, Any]) -> int:
    """Best-effort content length: word_count, else token estimate, else text."""
    wc = chunk.get("word_count")
    if isinstance(wc, int) and wc > 0:
        return wc
    tok = chunk.get("tokens_estimate")
    if isinstance(tok, int) and tok > 0:
        return tok
    text = chunk.get("text")
    if isinstance(text, str):
        return len(text.split())
    return 0


def richness_score(chunk: Dict[str, Any], *, min_words: int = 25) -> float:
    """Score a chunk's suitability as a self-retrieval probe.

    Returns a float; higher = better probe. Returns ``-1.0`` for chunks
    that are ineligible (apparatus, or too short to form a meaningful
    query). Preference is multiplicative on length so a rich instructional
    chunk beats a long exercise dump.
    """
    signal = _metadata_signal(chunk)
    if any(bad in signal for bad in _EXCLUDE_SUBSTR):
        return -1.0
    length = _content_length(chunk)
    if length < min_words:
        return -1.0
    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        return -1.0
    boost = 1.5 if any(good in signal for good in _PREFER_SUBSTR) else 1.0
    return float(length) * boost


def select_sample(
    chunks: Sequence[Dict[str, Any]],
    *,
    n: int,
    seed: int,
    min_words: int = 25,
) -> List[Dict[str, Any]]:
    """Deterministically select up to ``n`` content-rich chunks.

    Eligible chunks (richness_score >= 0) are ranked by (score desc, id
    asc). To keep a content-rich bias while still exercising variety, we
    draw the sample from the top pool (``max(n * 3, n)``) using a seeded
    RNG, then return them sorted by id for a stable, reproducible order.
    Same corpus + same seed ⇒ identical sample.
    """
    eligible = [
        (richness_score(c, min_words=min_words), _chunk_id(c), c)
        for c in chunks
    ]
    eligible = [e for e in eligible if e[0] >= 0.0 and e[1]]
    # Rank: richest first, id as deterministic tie-break.
    eligible.sort(key=lambda e: (-e[0], e[1]))
    if not eligible:
        return []
    if len(eligible) <= n:
        return [e[2] for e in eligible]
    pool_size = min(len(eligible), max(n * 3, n))
    pool = eligible[:pool_size]
    rng = random.Random(seed)
    picked = rng.sample(range(len(pool)), n)
    chosen = [pool[i] for i in sorted(picked)]
    chosen.sort(key=lambda e: e[1])
    return [e[2] for e in chosen]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str, *, max_chars: int = 240) -> str:
    """Return the first sentence of ``text`` (bounded)."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    parts = _SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)
    first = parts[0].strip()
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0]
    return first


def _heading(chunk: Dict[str, Any]) -> str:
    """Extract a heading for the chunk from its source metadata, if any."""
    source = chunk.get("source")
    if isinstance(source, dict):
        for key in ("section_heading", "lesson_title", "module_title", "heading"):
            val = source.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for key in ("section_title", "heading", "title"):
        val = chunk.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def form_query(chunk: Dict[str, Any]) -> str:
    """Form the deterministic self-retrieval query: heading + first sentence."""
    heading = _heading(chunk)
    sentence = _first_sentence(str(chunk.get("text") or ""))
    if heading and sentence:
        return f"{heading}. {sentence}"
    return heading or sentence


@dataclass
class SelfRetrievalOutcome:
    chunk_id: str
    query: str
    hit: bool
    rank: Optional[int]  # 1-based rank of the source chunk, None if absent
    reciprocal_rank: float
    top_result_ids: List[str] = field(default_factory=list)


def score_self_retrieval(
    sample: Sequence[Dict[str, Any]],
    retriever: Callable[[str, int], Sequence[Any]],
    *,
    k: int,
) -> List[SelfRetrievalOutcome]:
    """Query the retriever per sampled chunk and score hit@k + MRR.

    ``retriever(query, k)`` must return an ordered sequence of results;
    each result's chunk id is read via ``.chunk_id`` / ``.id`` attribute
    OR a ``chunk_id`` / ``id`` mapping key (so both ``RetrievalResult``
    and plain dicts work — the latter for tests).
    """
    outcomes: List[SelfRetrievalOutcome] = []
    for chunk in sample:
        cid = _chunk_id(chunk)
        query = form_query(chunk)
        results = list(retriever(query, k))
        result_ids = [_result_id(r) for r in results]
        rank: Optional[int] = None
        for idx, rid in enumerate(result_ids, start=1):
            if rid == cid:
                rank = idx
                break
        hit = rank is not None
        rr = (1.0 / rank) if rank is not None else 0.0
        outcomes.append(SelfRetrievalOutcome(
            chunk_id=cid,
            query=query,
            hit=hit,
            rank=rank,
            reciprocal_rank=rr,
            top_result_ids=result_ids[:k],
        ))
    return outcomes


def _result_id(result: Any) -> str:
    """Read a chunk id from a RetrievalResult-like object or a dict."""
    if isinstance(result, dict):
        return str(result.get("chunk_id") or result.get("id") or "")
    return str(
        getattr(result, "chunk_id", None) or getattr(result, "id", "") or ""
    )


def _result_score(result: Any) -> float:
    if isinstance(result, dict):
        return float(result.get("score", 0.0) or 0.0)
    return float(getattr(result, "score", 0.0) or 0.0)


@dataclass
class ProbeOutcome:
    query: str
    top_score: float
    margin: float  # top1 - top2 (0.0 if <2 results)
    top_result_ids: List[str] = field(default_factory=list)


def probe_margins(
    probes: Sequence[str],
    retriever: Callable[[str, int], Sequence[Any]],
    *,
    k: int,
) -> List[ProbeOutcome]:
    """Run out-of-domain probes and report top score + top1-top2 margin."""
    outcomes: List[ProbeOutcome] = []
    for probe in probes:
        results = list(retriever(probe, k))
        scores = [_result_score(r) for r in results]
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        outcomes.append(ProbeOutcome(
            query=probe,
            top_score=round(top, 6),
            margin=round(top - second, 6),
            top_result_ids=[_result_id(r) for r in results[:k]],
        ))
    return outcomes


def build_report(
    *,
    course_code: str,
    engine: str,
    k: int,
    sample_requested: int,
    self_outcomes: Sequence[SelfRetrievalOutcome],
    probe_outcomes: Sequence[ProbeOutcome],
    hit_threshold: float,
    seed: int,
) -> Dict[str, Any]:
    """Assemble the JSON report + pass/fail verdict."""
    n = len(self_outcomes)
    hits = sum(1 for o in self_outcomes if o.hit)
    hit_at_k = (hits / n) if n else 0.0
    mrr = (sum(o.reciprocal_rank for o in self_outcomes) / n) if n else 0.0
    # PASS requires at least one sampled chunk AND hit@k over threshold.
    passed = bool(n > 0 and hit_at_k >= hit_threshold)
    return {
        "course_code": course_code,
        "engine": engine,
        "k": k,
        "seed": seed,
        "sample_requested": sample_requested,
        "sample_scored": n,
        "hit_at_k": round(hit_at_k, 4),
        "mrr": round(mrr, 4),
        "hit_threshold": hit_threshold,
        "passed": passed,
        "self_retrieval": [
            {
                "chunk_id": o.chunk_id,
                "query": o.query,
                "hit": o.hit,
                "rank": o.rank,
                "reciprocal_rank": round(o.reciprocal_rank, 4),
                "top_result_ids": o.top_result_ids,
            }
            for o in self_outcomes
        ],
        "ood_probes": [
            {
                "query": p.query,
                "top_score": p.top_score,
                "margin": p.margin,
                "top_result_ids": p.top_result_ids,
            }
            for p in probe_outcomes
        ],
    }


# ------------------------------------------------------------------ #
# I/O + live retriever wiring (not unit-tested — exercised live).
# ------------------------------------------------------------------ #


class MissingRetrievalStack(RuntimeError):
    """Raised when the embedding extras / vector index are unavailable."""


def _load_chunkset(course_dir: Path) -> List[Dict[str, Any]]:
    """Stream the course chunkset the live retriever serves."""
    from lib.libv2_storage import resolve_chunks_path_for_query

    chunks_path, _resolution = resolve_chunks_path_for_query(
        course_dir, "chunks.jsonl"
    )
    if not chunks_path.exists():
        raise MissingRetrievalStack(
            f"No chunkset found for course at {course_dir} "
            f"(resolved to {chunks_path}, which does not exist). "
            f"Import/build the course before running the smoke harness."
        )
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def _make_live_retriever(
    libv2_root: Path, course_slug: str, engine: str
) -> Callable[[str, int], Sequence[Any]]:
    """Bind the canonical LibV2 retriever into a ``(query, k)`` callable.

    Fail-closed: a missing semantic index / embedding backend surfaces as
    :class:`MissingRetrievalStack` (exit 2) rather than a silent lexical
    fallback.
    """
    from LibV2.tools.libv2.retriever import retrieve_chunks

    def _retrieve(query: str, k: int) -> Sequence[Any]:
        try:
            return retrieve_chunks(
                libv2_root,
                query,
                course_slug=course_slug,
                engine=engine,
                limit=k,
            )
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if name in {
                "SemanticIndexMissing",
                "SemanticIndexStale",
                "SemanticBackendUnavailable",
                "EmbeddingBackendUnavailable",
                "ModuleNotFoundError",
                "ImportError",
            }:
                raise MissingRetrievalStack(
                    f"Retrieval stack unavailable for engine={engine!r} on "
                    f"course {course_slug!r}: {name}: {exc}. "
                    f"Build the vector index (`libv2 vector-index build "
                    f"--course {course_slug}`) and install the [embedding] "
                    f"extras, or re-run with --engine lexical."
                ) from exc
            raise

    return _retrieve


def _resolve_libv2_root(cli_root: Optional[str]) -> Path:
    if cli_root:
        return Path(cli_root).resolve()
    import os

    env = os.environ.get("ED4ALL_LIBV2_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parents[2] / "LibV2").resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic retrieval smoke harness for a LibV2 course."
    )
    parser.add_argument("--course-code", required=True,
                        help="LibV2 course slug (directory under courses/).")
    parser.add_argument("--libv2-root", default=None,
                        help="LibV2 root (default: repo LibV2/ or "
                             "$ED4ALL_LIBV2_ROOT).")
    parser.add_argument("--index-dir", default=None,
                        help="Optional explicit vector_index dir (recorded "
                             "for provenance; retrieval resolves the course's "
                             "own index).")
    parser.add_argument("--sample", type=int, default=25,
                        help="Number of content-rich chunks to probe.")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k cutoff for hit@k + MRR.")
    parser.add_argument("--engine", default="hybrid-rrf",
                        choices=["hybrid-rrf", "semantic", "lexical"],
                        help="Retrieval engine (default hybrid-rrf).")
    parser.add_argument("--hit-threshold", type=float, default=0.8,
                        help="PASS floor for hit@k (default 0.8).")
    parser.add_argument("--seed", type=int, default=1729,
                        help="Sampling RNG seed (reproducibility).")
    parser.add_argument("--min-words", type=int, default=25,
                        help="Minimum chunk word count to be eligible.")
    parser.add_argument("--json-out", default=None,
                        help="Write the JSON report to this path.")
    args = parser.parse_args(argv)

    libv2_root = _resolve_libv2_root(args.libv2_root)
    course_dir = libv2_root / "courses" / args.course_code

    try:
        chunks = _load_chunkset(course_dir)
        retriever = _make_live_retriever(
            libv2_root, args.course_code, args.engine
        )
        sample = select_sample(
            chunks, n=args.sample, seed=args.seed, min_words=args.min_words
        )
        if not sample:
            print(
                f"ERROR: no content-rich chunks eligible in {args.course_code} "
                f"(min_words={args.min_words}). Nothing to probe.",
                file=sys.stderr,
            )
            return 2
        self_outcomes = score_self_retrieval(sample, retriever, k=args.k)
        probe_outcomes = probe_margins(
            DEFAULT_OOD_PROBES, retriever, k=args.k
        )
    except MissingRetrievalStack as exc:
        print(f"ERROR (retrieval stack unavailable): {exc}", file=sys.stderr)
        return 2

    report = build_report(
        course_code=args.course_code,
        engine=args.engine,
        k=args.k,
        sample_requested=args.sample,
        self_outcomes=self_outcomes,
        probe_outcomes=probe_outcomes,
        hit_threshold=args.hit_threshold,
        seed=args.seed,
    )
    if args.index_dir:
        report["index_dir"] = str(Path(args.index_dir).resolve())

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    # Human summary.
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"Retrieval smoke [{args.course_code}] engine={args.engine} "
          f"k={args.k}: {verdict}")
    print(f"  hit@{args.k} = {report['hit_at_k']:.3f} "
          f"(threshold {args.hit_threshold}) over "
          f"{report['sample_scored']} chunks; MRR = {report['mrr']:.3f}")
    misses = [o for o in self_outcomes if not o.hit]
    if misses:
        print(f"  {len(misses)} miss(es):")
        for o in misses[:10]:
            print(f"    - {o.chunk_id}: {o.query[:70]!r}")
    ood_max = max((p.top_score for p in probe_outcomes), default=0.0)
    print(f"  OOD refusal probes: max top-score = {ood_max:.4f} "
          f"across {len(probe_outcomes)} probes")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
