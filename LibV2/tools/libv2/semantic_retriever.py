"""Real semantic retrieval over the on-device vector index (WS2 E3).

This is the honest semantic-retrieval path that retires the mock. A query
is embedded with an OFFLINE embedding client (so a query can never trigger
a model download), searched against the per-course vector index built by
``LibV2.tools.libv2.vector_index``, and hydrated back into the same
``RetrievalResult`` shape the lexical retriever emits — so production
consumers (``Trainforge/rag/libv2_bridge.py``) read the result objects
identically regardless of engine.

Anti-silent-degradation (the contract this whole work-stream enforces):
every typed failure propagates. There is **NO BM25 fallback** anywhere in
this module. The typed errors are:

* :class:`~LibV2.tools.libv2.vector_index.SemanticIndexMissing` — no index
  for the course (operator must run ``libv2 vector-index build``).
* :class:`~LibV2.tools.libv2.vector_index.SemanticIndexStale` — the index
  no longer matches the live chunkset (or on-disk sha drift).
* :class:`~LibV2.tools.libv2.vector_index.FakeIndexRefused` — the manifest
  provider is ``fake`` and ``ED4ALL_EMBEDDING_ALLOW_FAKE`` is unset.
* :class:`~lib.embedding.providers.EmbeddingBackendUnavailable` — the
  embedding backend (weights / server) is unavailable.

The query path PRE-FLIGHTS ``load_vector_index`` before any ThreadPool
dispatch so a semantic failure surfaces from the caller rather than being
swallowed by ``MultiQueryRetriever``'s ``print("Sub-query failed: ...")``
arm. See ``multi_retriever.py``'s engine pre-flight.

Score domain: ``RetrievalResult.score`` carries the raw cosine similarity
in ``[-1, 1]``. Cosine scores are NOT comparable to BM25 scores; the
lexical ``DEFAULT_MIN_RELEVANCE=0.5`` floor does NOT apply here — use the
explicit ``min_similarity`` knob (default ``0.0``) instead. The hybrid arm
fuses the two engines in RANK domain (RRF), never in score domain, exactly
because the score domains are incommensurable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from lib.libv2_storage import resolve_imscc_chunks_path

from .retriever import (
    ChunkFilter,
    RetrievalResult,
    _build_outcomes_lookup,
    _matches_filter,
    retrieve_chunks,
)

if TYPE_CHECKING:  # pragma: no cover — type-only
    import numpy as np

    from lib.embedding.providers import EmbeddingClient


# Default RRF constant — reuses the standard value pinned on
# ``result_fusion.ResultFuser.RRF_K`` so lexical-fusion and hybrid-fusion
# share one constant (D8).
DEFAULT_RRF_K = 60


def _resolve_course_domain(course_dir: Path) -> str:
    """Resolve a course's primary domain from manifest.json (best-effort).

    Mirrors the single-course branch of ``retrieve_chunks`` so a semantic
    result carries the same ``domain`` field a lexical result would.
    """
    manifest_path = course_dir / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return "unknown"
    return manifest.get("classification", {}).get("primary_domain", "unknown")


def _load_chunks_by_id(course_dir: Path) -> Dict[str, dict]:
    """Build an ``{chunk_id: chunk_dict}`` join table for hydration.

    Streams the same chunkset the index was built from (via the canonical
    ``resolve_imscc_chunks_path`` shim: imscc_chunks/ -> dart_chunks/ ->
    legacy corpus/). Accepts both ``id`` (canonical) and ``chunk_id``
    (legacy / fixture) as the key — matching ``vector_index._chunk_id``.
    """
    chunks_path = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    table: Dict[str, dict] = {}
    if not chunks_path.exists():
        return table
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = chunk.get("id") or chunk.get("chunk_id")
            if cid:
                table[str(cid)] = chunk
    return table


def _hydrate_result(
    chunk: dict,
    *,
    chunk_id: str,
    cosine: float,
    course_slug: str,
    domain: str,
    include_rationale: bool,
    model_id: str,
) -> RetrievalResult:
    """Project a chunk dict + cosine score into a ``RetrievalResult``.

    Field selection mirrors the lexical assembly in
    ``retriever.retrieve_chunks`` exactly so ``to_dict()`` is byte-compatible
    (same keys, same order) with the lexical consumer contract. ``score`` is
    the raw cosine (NOT a BM25-comparable value).
    """
    result = RetrievalResult(
        chunk_id=chunk.get("id", chunk.get("chunk_id", chunk_id)),
        text=chunk.get("text", ""),
        score=float(cosine),
        course_slug=course_slug,
        domain=domain,
        chunk_type=chunk.get("chunk_type", ""),
        difficulty=chunk.get("difficulty"),
        concept_tags=chunk.get("concept_tags", []),
        source=chunk.get("source", {}),
        tokens_estimate=chunk.get("tokens_estimate", 0),
        learning_outcome_refs=chunk.get("learning_outcome_refs", []),
        bloom_level=chunk.get("bloom_level"),
    )
    if include_rationale:
        result.rationale = {
            "engine": "semantic",
            "cosine": round(float(cosine), 6),
            "model_id": model_id,
        }
    return result


def _embed_query(query: str, client: "EmbeddingClient") -> "np.ndarray":
    """Embed a single query string, applying the provider's query prefix.

    Nomic-style task prefixes (``search_query: ``) live on the resolved
    provider; we apply it to QUERIES only (documents got the document
    prefix at index-build time). Returns a 1-D float32 vector. Raises
    ``EmbeddingBackendUnavailable`` (propagated from the client) on backend
    failure — NEVER a fallback vector.
    """
    prefix = getattr(getattr(client, "resolved", None), "query_prefix", "") or ""
    text = f"{prefix}{query}" if prefix else query
    matrix = client.encode_batch([text])
    return matrix[0]


def _client_for_index(index: Any) -> "EmbeddingClient":
    """Build the OFFLINE query-path embedding client that MATCHES the index.

    The query MUST be embedded with the same model the index was built
    against — embedding a query with a different model produces cosines
    against a foreign vector space (silently meaningless ranks). So instead
    of falling through to the registry default (which may be a different,
    uncached model and raise ``EmbeddingBackendUnavailable``), we pin the
    client to the manifest's recorded ``embedding_provider`` +
    ``embedding_model_id``.

    Offline so a query can never trigger a model download (the build path
    is the only place a download may happen). Imported lazily so this
    module has no import-time coupling to ``lib.embedding.providers``.
    """
    from lib.embedding.providers import build_embedding_client

    manifest = index.manifest
    provider_name = str(getattr(manifest, "embedding_provider", "") or "") or None
    model_id = str(getattr(manifest, "embedding_model_id", "") or "") or None
    return build_embedding_client(
        provider_name=provider_name,
        model_id=model_id,
        offline=True,
    )


def _verify_client_matches_index(client: "EmbeddingClient", index: Any) -> None:
    """Fail closed when an explicit client doesn't match the index's model.

    Querying an index built with model ``A`` using a client embedding with
    model ``B`` yields cosines across incommensurable vector spaces — a
    silent correctness bug. We raise ``SemanticModelMismatch`` rather than
    serve foreign-model ranks. Best-effort: a client that doesn't expose a
    ``model_fingerprint()`` is left un-verified (the fake/duck-typed test
    clients without one still work).
    """
    from .vector_index import SemanticModelMismatch

    fingerprint_fn = getattr(client, "model_fingerprint", None)
    if not callable(fingerprint_fn):
        return
    try:
        client_model_id = dict(fingerprint_fn()).get("model_id")
    except Exception:  # noqa: BLE001 — a broken fingerprint must not crash retrieval
        return
    if client_model_id is None:
        return
    index_model_id = str(getattr(index.manifest, "embedding_model_id", "") or "")
    if str(client_model_id) != index_model_id:
        raise SemanticModelMismatch(
            f"explicit embedding client model_id {client_model_id!r} does "
            f"not match the vector index's embedding_model_id "
            f"{index_model_id!r}; querying a foreign-model index produces "
            f"meaningless cosine scores. Pass a client built for "
            f"{index_model_id!r} or omit `client` to auto-match the index."
        )


def semantic_retrieve_chunks(
    repo_root: Path,
    query: str,
    *,
    course_slug: str,
    limit: int = 10,
    min_similarity: float = 0.0,
    chunk_filter: Optional[ChunkFilter] = None,
    include_rationale: bool = False,
    client: Optional["EmbeddingClient"] = None,
    allow_fake: Optional[bool] = None,
    verify_chunkset: bool = True,
) -> List[RetrievalResult]:
    """Real nearest-neighbor semantic retrieval for one course.

    Pipeline: embed the query (offline client, query_prefix applied) ->
    pre-flight ``load_vector_index`` (fail-closed; D7) -> exact cosine
    ``idx.search`` -> hydrate chunks via an id->chunk join over the same
    chunkset the index embedded -> apply ``ChunkFilter`` post-search ->
    truncate to ``limit``.

    Filter semantics (§ 3): over-fetch ``top_k = max(limit * 5, 50)`` so a
    filtered query doesn't starve, then filter, then truncate to ``limit``.

    Score domain: ``RetrievalResult.score`` is the raw cosine in
    ``[-1, 1]`` — NOT comparable to BM25. ``min_similarity`` (default 0.0)
    is the cosine floor; the lexical ``DEFAULT_MIN_RELEVANCE`` does NOT
    apply.

    HARD-ERROR contract: ``SemanticIndexMissing`` / ``SemanticIndexStale``
    / ``FakeIndexRefused`` / ``EmbeddingBackendUnavailable`` all propagate.
    There is NO BM25 fallback.
    """
    from .vector_index import load_vector_index

    repo_root = Path(repo_root)
    course_dir = repo_root / "courses" / course_slug

    # Pre-flight the index load FIRST so a missing/stale/fake index raises
    # before we spend an embedding call (and, in the multi-query path,
    # before any ThreadPool dispatch can swallow the exception).
    index = load_vector_index(
        course_dir,
        verify_chunkset=verify_chunkset,
        allow_fake=allow_fake,
    )

    # Align the query embedder with the index's manifest (E3 fix): when no
    # client is passed, build one pinned to the manifest's recorded model
    # instead of the registry default (which may be a different, uncached
    # model that raises EmbeddingBackendUnavailable). When an explicit client
    # IS passed, verify it matches the index rather than silently querying a
    # foreign-model index.
    if client is None:
        client = _client_for_index(index)
    else:
        _verify_client_matches_index(client, index)

    query_vec = _embed_query(query, client)

    # Over-fetch so post-search filtering doesn't starve the result set.
    over_fetch = max(limit * 5, 50)
    hits = index.search(query_vec, top_k=over_fetch)
    if not hits:
        return []

    # Apply the cosine floor before hydration (cheap short-circuit).
    hits = [(cid, score) for cid, score in hits if score >= min_similarity]
    if not hits:
        return []

    chunk_table = _load_chunks_by_id(course_dir)
    domain = _resolve_course_domain(course_dir)
    model_id = str(index.manifest.embedding_model_id)

    # Build the outcomes lookup once if the hierarchy_level filter needs it
    # (mirrors the lexical streaming path's lazy build).
    outcomes_by_id: Optional[dict] = None
    if chunk_filter and chunk_filter.hierarchy_level:
        outcomes_by_id = _build_outcomes_lookup(course_dir)

    results: List[RetrievalResult] = []
    for chunk_id, cosine in hits:
        chunk = chunk_table.get(chunk_id)
        if chunk is None:
            # Index references a chunk that's no longer in the chunkset.
            # verify_chunkset=True would already have raised on a sha drift;
            # this only fires when the caller disabled that check. Skip
            # rather than emit a hollow result.
            continue
        if chunk_filter and not _matches_filter(
            chunk, chunk_filter, outcomes_by_id=outcomes_by_id
        ):
            continue
        results.append(
            _hydrate_result(
                chunk,
                chunk_id=chunk_id,
                cosine=cosine,
                course_slug=course_slug,
                domain=domain,
                include_rationale=include_rationale,
                model_id=model_id,
            )
        )
        if len(results) >= limit:
            break

    return results


def _rrf_fuse(
    lexical: List[RetrievalResult],
    semantic: List[RetrievalResult],
    *,
    limit: int,
    rrf_k: int,
) -> List[RetrievalResult]:
    """Rank-domain RRF fusion of two ranked result lists.

    RRF score(d) = sum over arms of ``1 / (rrf_k + rank_arm(d))`` (1-based
    rank). Rank-domain (NOT score-domain) so the incommensurable cosine and
    BM25 score scales never get summed directly. Ties broken by chunk_id asc
    for determinism. The fused ``RetrievalResult.score`` carries the RRF
    score; the per-arm contributions land in ``rationale`` when either input
    carried a rationale.
    """
    by_id: Dict[str, RetrievalResult] = {}
    rrf: Dict[str, float] = {}
    lex_rank: Dict[str, int] = {}
    sem_rank: Dict[str, int] = {}

    for rank, r in enumerate(lexical, start=1):
        rrf[r.chunk_id] = rrf.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        lex_rank[r.chunk_id] = rank
        by_id.setdefault(r.chunk_id, r)
    for rank, r in enumerate(semantic, start=1):
        rrf[r.chunk_id] = rrf.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        sem_rank[r.chunk_id] = rank
        # Prefer the semantic hydration when a chunk appears in both arms
        # (identical fields except score/rationale; either is fine, but
        # being explicit keeps the merge deterministic).
        by_id.setdefault(r.chunk_id, r)

    ordered_ids = sorted(
        rrf.keys(), key=lambda cid: (-rrf[cid], cid)
    )

    fused: List[RetrievalResult] = []
    for cid in ordered_ids[:limit]:
        base = by_id[cid]
        merged = RetrievalResult(
            chunk_id=base.chunk_id,
            text=base.text,
            score=float(rrf[cid]),
            course_slug=base.course_slug,
            domain=base.domain,
            chunk_type=base.chunk_type,
            difficulty=base.difficulty,
            concept_tags=base.concept_tags,
            source=base.source,
            tokens_estimate=base.tokens_estimate,
            learning_outcome_refs=base.learning_outcome_refs,
            bloom_level=base.bloom_level,
        )
        if base.rationale is not None or cid in lex_rank or cid in sem_rank:
            merged.rationale = {
                "engine": "hybrid-rrf",
                "rrf_score": round(float(rrf[cid]), 6),
                "rrf_k": rrf_k,
                "lexical_rank": lex_rank.get(cid),
                "semantic_rank": sem_rank.get(cid),
            }
        fused.append(merged)
    return fused


def hybrid_rrf_retrieve(
    repo_root: Path,
    query: str,
    *,
    course_slug: str,
    limit: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
    chunk_filter: Optional[ChunkFilter] = None,
    include_rationale: bool = False,
    client: Optional["EmbeddingClient"] = None,
    allow_fake: Optional[bool] = None,
    min_similarity: float = 0.0,
    **lexical_kwargs: Any,
) -> List[RetrievalResult]:
    """Hybrid retrieval: RRF-fuse the lexical and semantic arms (D8).

    Runs BOTH the lexical BM25 arm and the semantic arm, then fuses their
    RANKINGS via RRF (rank-domain — never score-domain, since cosine and
    BM25 scores are incommensurable). Fail-closed on a semantic-arm failure:
    the typed vector-index / embedding errors propagate exactly as for
    ``engine="semantic"`` — the hybrid contingency is an EXPLICIT operator
    choice, not a silent BM25 fallback.

    The semantic arm runs FIRST so its index pre-flight fails fast before
    the (potentially expensive) lexical streaming pass.

    ``**lexical_kwargs`` are threaded into the lexical ``retrieve_chunks``
    arm (e.g. ``method=``, boost toggles); they are NOT applied to the
    semantic arm (boosts are N/A there).
    """
    # Semantic arm first — fail-closed pre-flight before lexical work.
    semantic = semantic_retrieve_chunks(
        repo_root,
        query,
        course_slug=course_slug,
        # Over-fetch each arm so the fusion pool is rich (RRF rewards
        # agreement across arms; a tiny per-arm cut starves it).
        limit=max(limit * 2, limit),
        min_similarity=min_similarity,
        chunk_filter=chunk_filter,
        include_rationale=include_rationale,
        client=client,
        allow_fake=allow_fake,
    )

    # Lexical arm — same filter axes, threaded through the public surface so
    # the BM25 path is byte-identical to a direct lexical call.
    lexical_filter_kwargs: Dict[str, Any] = {}
    if chunk_filter is not None:
        lexical_filter_kwargs = {
            "chunk_type": chunk_filter.chunk_type,
            "difficulty": chunk_filter.difficulty,
            "concept_tags": chunk_filter.concept_tags,
            "learning_outcome_refs": chunk_filter.learning_outcome_refs,
            "bloom_level": chunk_filter.bloom_level,
            "teaching_role": chunk_filter.teaching_role,
            "content_type_label": chunk_filter.content_type_label,
            "module_id": chunk_filter.module_id,
            "week_num": chunk_filter.week_num,
            "cognitive_domain": chunk_filter.cognitive_domain,
            "hierarchy_level": chunk_filter.hierarchy_level,
        }
    lexical = retrieve_chunks(
        repo_root,
        query,
        course_slug=course_slug,
        limit=max(limit * 2, limit),
        include_rationale=include_rationale,
        **lexical_filter_kwargs,
        **lexical_kwargs,
    )

    return _rrf_fuse(lexical, semantic, limit=limit, rrf_k=rrf_k)


__all__ = [
    "semantic_retrieve_chunks",
    "hybrid_rrf_retrieve",
    "DEFAULT_RRF_K",
]
