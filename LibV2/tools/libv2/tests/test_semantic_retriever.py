"""Tests for the semantic retrieval path + engine axis (WS2 E3).

All coverage runs on the deterministic ``fake`` embedding provider (E1's
registry entry; ``ED4ALL_EMBEDDING_ALLOW_FAKE`` set in-test) so there are
zero model weights and zero network calls. Covers:

* semantic roundtrip over a tmp fake-built course (self-retrieval);
* ChunkFilter post-search filtering + over-fetch;
* HARD-ERROR propagation (missing index, stale index, backend unavailable)
  with NO BM25 fallback — including through the MultiQueryRetriever
  ThreadPool (the pre-flight must surface the error, not swallow it);
* hybrid-rrf rank-domain fusion correctness on a constructed case;
* lexical byte-identity regression (engine="lexical" == pre-change);
* result objects byte-compatible with the lexical ``to_dict()`` consumer
  contract.
"""

from __future__ import annotations

import json
from pathlib import Path

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from lib.embedding.providers import (
    EmbeddingBackendUnavailable,
    build_embedding_client,
)

from LibV2.tools.libv2.multi_retriever import MultiQueryRetriever
from LibV2.tools.libv2.retriever import ChunkFilter, retrieve_chunks
from LibV2.tools.libv2.semantic_retriever import (
    _rrf_fuse,
    hybrid_rrf_retrieve,
    semantic_retrieve_chunks,
)
from LibV2.tools.libv2.retrieval.vector_index import (
    SemanticIndexMissing,
    SemanticIndexStale,
    SemanticModelMismatch,
    build_vector_index,
)


@pytest.fixture(autouse=True)
def _allow_fake(monkeypatch):
    """Every test loads a fake-built index; allow it (test-only)."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")


def _fake_client():
    return build_embedding_client(provider_name="fake")


def _write_course(repo_root: Path, slug: str = "sem-101", n: int = 6) -> Path:
    """Build a tmp LibV2 repo course with a semantik_chunks/chunks.jsonl."""
    course_dir = repo_root / "courses" / slug
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "id": f"sem_101_chunk_{i:05d}",
                    "text": f"This is chunk number {i} about topic {i}.",
                    "chunk_type": "explanation" if i % 2 == 0 else "example",
                    "concept_tags": [f"topic-{i}"],
                    "tokens_estimate": 10 + i,
                    "learning_outcome_refs": [],
                    "source": {"section_heading": f"Section {i}"},
                }
            )
        )
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return course_dir


def _build_index(repo_root: Path, slug: str = "sem-101", n: int = 6):
    course_dir = _write_course(repo_root, slug, n)
    build_vector_index(course_dir, client=_fake_client())
    return course_dir


# --------------------------------------------------------------------------
# Semantic roundtrip
# --------------------------------------------------------------------------


def test_semantic_self_retrieval(tmp_path):
    _build_index(tmp_path, n=6)
    client = _fake_client()
    # The exact text+heading of chunk 3 must rank #1 (cosine ~1.0).
    target = "Section 3\nThis is chunk number 3 about topic 3."
    # Force the query string so it embeds to the same vector the index stored.
    results = semantic_retrieve_chunks(
        tmp_path,
        target,
        course_slug="sem-101",
        limit=3,
        client=client,
        include_rationale=True,
    )
    assert results[0].chunk_id == "sem_101_chunk_00003"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert len(results) == 3
    # rationale carries the semantic engine marker + cosine + model_id.
    assert results[0].rationale["engine"] == "semantic"
    assert results[0].rationale["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert results[0].rationale["model_id"] == "fake-deterministic-v1"


def test_semantic_result_to_dict_byte_compatible(tmp_path):
    """A semantic RetrievalResult.to_dict() carries the same keys/shape as a
    lexical one (consumer contract w/ Trainforge/rag/libv2_bridge.py)."""
    _build_index(tmp_path, n=4)
    results = semantic_retrieve_chunks(
        tmp_path, "topic 1", course_slug="sem-101", limit=2, client=_fake_client(),
    )
    assert results
    d = results[0].to_dict()
    # Same keys the lexical path emits when include_rationale=False.
    assert set(d.keys()) == {
        "chunk_id", "text", "score", "course_slug", "domain", "chunk_type",
        "difficulty", "concept_tags", "source", "tokens_estimate",
        "learning_outcome_refs", "bloom_level",
    }
    # No `rationale` key leaks when not requested.
    assert "rationale" not in d


def test_semantic_chunk_filter(tmp_path):
    _build_index(tmp_path, n=6)
    # Filter to chunk_type=example (odd-indexed chunks).
    results = semantic_retrieve_chunks(
        tmp_path,
        "topic",
        course_slug="sem-101",
        limit=10,
        chunk_filter=ChunkFilter(chunk_type="example"),
        client=_fake_client(),
    )
    assert results
    assert all(r.chunk_type == "example" for r in results)


def test_semantic_min_similarity_floor(tmp_path):
    _build_index(tmp_path, n=6)
    # A high cosine floor drops everything except (near-)exact matches.
    results = semantic_retrieve_chunks(
        tmp_path,
        "completely unrelated gibberish xyzzy",
        course_slug="sem-101",
        limit=10,
        min_similarity=0.99,
        client=_fake_client(),
    )
    # Random fake vectors are ~orthogonal -> nothing clears a 0.99 floor.
    assert results == []


# --------------------------------------------------------------------------
# HARD-ERROR propagation — NO BM25 fallback
# --------------------------------------------------------------------------


def test_missing_index_raises_no_fallback(tmp_path):
    # Course with chunks but no built index.
    _write_course(tmp_path, n=3)
    with pytest.raises(SemanticIndexMissing):
        semantic_retrieve_chunks(
            tmp_path, "topic 1", course_slug="sem-101", client=_fake_client(),
        )


def test_stale_index_raises_no_fallback(tmp_path):
    course_dir = _build_index(tmp_path, n=4)
    # Mutate the chunkset so the index is stale.
    chunks = course_dir / "semantik_chunks" / "chunks.jsonl"
    chunks.write_text(
        chunks.read_text(encoding="utf-8") + '{"id":"x","text":"new"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SemanticIndexStale):
        semantic_retrieve_chunks(
            tmp_path, "topic 1", course_slug="sem-101", client=_fake_client(),
        )


def test_backend_unavailable_propagates(tmp_path):
    _build_index(tmp_path, n=3)

    class _DeadClient:
        resolved = None

        def encode_batch(self, texts):
            raise EmbeddingBackendUnavailable("weights absent")

    with pytest.raises(EmbeddingBackendUnavailable):
        semantic_retrieve_chunks(
            tmp_path, "topic 1", course_slug="sem-101", client=_DeadClient(),
        )


def test_engine_semantic_via_retrieve_chunks(tmp_path):
    _build_index(tmp_path, n=5)
    results = retrieve_chunks(
        tmp_path,
        "Section 2\nThis is chunk number 2 about topic 2.",
        course_slug="sem-101",
        engine="semantic",
        limit=3,
    )
    assert results[0].chunk_id == "sem_101_chunk_00002"


def test_engine_semantic_missing_index_propagates_through_retrieve_chunks(tmp_path):
    _write_course(tmp_path, n=3)
    with pytest.raises(SemanticIndexMissing):
        retrieve_chunks(
            tmp_path, "topic", course_slug="sem-101", engine="semantic",
        )


def test_engine_rejects_unknown(tmp_path):
    _build_index(tmp_path, n=2)
    with pytest.raises(ValueError):
        retrieve_chunks(tmp_path, "q", course_slug="sem-101", engine="bogus")


def test_engine_semantic_requires_course(tmp_path):
    with pytest.raises(ValueError):
        retrieve_chunks(tmp_path, "q", engine="semantic")


def test_engine_semantic_rejects_method(tmp_path):
    _build_index(tmp_path, n=2)
    with pytest.raises(ValueError):
        retrieve_chunks(
            tmp_path, "q", course_slug="sem-101", engine="semantic",
            method="bm25+graph",
        )


# --------------------------------------------------------------------------
# MultiQueryRetriever pre-flight — error must SURFACE, not be swallowed
# --------------------------------------------------------------------------


def test_multi_retriever_preflight_surfaces_missing_index(tmp_path):
    """The non-lexical engine pre-flights the index load BEFORE ThreadPool
    dispatch, so a missing index raises from retrieve() instead of being
    swallowed by the sub-query ``print('Sub-query failed: ...')`` arm."""
    _write_course(tmp_path, n=3)  # chunks, no index
    retriever = MultiQueryRetriever(repo_root=tmp_path, course_slug="sem-101")
    with pytest.raises(SemanticIndexMissing):
        retriever.retrieve(query="topic 1", engine="semantic", decompose=True)


def test_multi_retriever_semantic_happy_path(tmp_path):
    _build_index(tmp_path, n=6)
    retriever = MultiQueryRetriever(repo_root=tmp_path, course_slug="sem-101")
    fusion = retriever.retrieve(
        query="topic 2", engine="semantic", decompose=False,
    )
    assert fusion.result_count >= 1


def test_multi_retriever_unknown_engine_raises(tmp_path):
    _build_index(tmp_path, n=2)
    retriever = MultiQueryRetriever(repo_root=tmp_path, course_slug="sem-101")
    with pytest.raises(ValueError):
        retriever.retrieve(query="q", engine="bogus")


# --------------------------------------------------------------------------
# E3 fix: client=None aligns the query embedder with the index's manifest
# --------------------------------------------------------------------------


def _build_index_with_model(repo_root: Path, model_id: str, slug="sem-101", n=6):
    """Build a fake-provider index whose manifest records a custom model id."""
    course_dir = _write_course(repo_root, slug, n)
    client = build_embedding_client(provider_name="fake", model_id=model_id)
    manifest = build_vector_index(course_dir, client=client)
    assert manifest.embedding_model_id == model_id
    return course_dir


def test_client_none_builds_manifest_model_no_env_dependency(tmp_path, monkeypatch):
    """With client=None the query embedder is pinned to the index manifest's
    recorded model — NOT the registry default — so an index built with model
    'X' is queried with model 'X' regardless of the ambient env default.

    Fail-without-fix: pre-fix client=None fell through to build_embedding_client
    (registry default model), which would NOT match a manifest model of 'X'
    and would mis-route the query embedder.
    """
    custom_model = "fake-model-for-this-index"
    # Ambient env model default deliberately differs from the manifest's model.
    monkeypatch.setenv("ED4ALL_EMBEDDING_MODEL", "some-other-default")
    _build_index_with_model(tmp_path, custom_model, n=6)

    captured = {}
    import lib.embedding.providers as prov_mod
    real_build = prov_mod.build_embedding_client

    def _spy_build(provider_name=None, model_id=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["model_id"] = model_id
        return real_build(provider_name=provider_name, model_id=model_id, **kwargs)

    monkeypatch.setattr(prov_mod, "build_embedding_client", _spy_build)

    target = "Section 3\nThis is chunk number 3 about topic 3."
    results = semantic_retrieve_chunks(
        tmp_path, target, course_slug="sem-101", limit=3, client=None,
    )
    # The auto-built client was pinned to the MANIFEST's model, not the env.
    assert captured["model_id"] == custom_model
    assert captured["provider_name"] == "fake"
    # And retrieval actually worked (self-retrieval ranks the exact chunk #1).
    assert results[0].chunk_id == "sem_101_chunk_00003"


def test_explicit_client_mismatch_raises_typed_error(tmp_path):
    """An explicit client whose model_id differs from the index manifest's
    raises SemanticModelMismatch rather than silently querying a foreign-model
    index.

    Fail-without-fix: pre-fix the explicit client was used verbatim with no
    manifest cross-check, returning meaningless foreign-model cosines.
    """
    _build_index_with_model(tmp_path, "index-built-with-this-model", n=4)
    foreign_client = build_embedding_client(
        provider_name="fake", model_id="a-totally-different-model"
    )
    with pytest.raises(SemanticModelMismatch):
        semantic_retrieve_chunks(
            tmp_path, "topic 1", course_slug="sem-101", client=foreign_client,
        )


def test_explicit_client_matching_model_works(tmp_path):
    """An explicit client whose model_id matches the manifest is accepted."""
    model_id = "the-matching-model"
    _build_index_with_model(tmp_path, model_id, n=6)
    matching_client = build_embedding_client(
        provider_name="fake", model_id=model_id
    )
    target = "Section 2\nThis is chunk number 2 about topic 2."
    results = semantic_retrieve_chunks(
        tmp_path, target, course_slug="sem-101", limit=3, client=matching_client,
    )
    assert results[0].chunk_id == "sem_101_chunk_00002"


# --------------------------------------------------------------------------
# Hybrid RRF fusion correctness (rank-domain)
# --------------------------------------------------------------------------


def test_rrf_fuse_rank_domain():
    """Hand-computed RRF on constructed rankings. A chunk ranked highly in
    both arms must win over a chunk top-ranked in only one."""
    from LibV2.tools.libv2.retriever import RetrievalResult

    def _r(cid, score):
        return RetrievalResult(
            chunk_id=cid, text=cid, score=score, course_slug="c", domain="d",
            chunk_type="explanation", difficulty=None, concept_tags=[],
            source={}, tokens_estimate=0,
        )

    # lexical: A(1) B(2) C(3) ; semantic: B(1) A(2) D(3)
    lexical = [_r("A", 9.0), _r("B", 8.0), _r("C", 7.0)]
    semantic = [_r("B", 0.9), _r("A", 0.8), _r("D", 0.7)]
    fused = _rrf_fuse(lexical, semantic, limit=10, rrf_k=60)
    ids = [r.chunk_id for r in fused]
    # B: 1/62 + 1/61 ; A: 1/61 + 1/62 (equal to B) -> tie broken by id asc.
    # Both A and B appear in both arms with symmetric ranks -> equal RRF;
    # A < B lexicographically so A first.
    assert ids[0] == "A"
    assert ids[1] == "B"
    # C (lex rank3 only) and D (sem rank3 only) have equal single-arm RRF;
    # tie-break by id asc -> C before D.
    assert ids[2:] == ["C", "D"]
    # rationale carries both ranks.
    a = fused[0]
    assert a.rationale["engine"] == "hybrid-rrf"
    assert a.rationale["lexical_rank"] == 1
    assert a.rationale["semantic_rank"] == 2


def test_hybrid_rrf_end_to_end(tmp_path):
    _build_index(tmp_path, n=6)
    results = hybrid_rrf_retrieve(
        tmp_path,
        "Section 3\nThis is chunk number 3 about topic 3.",
        course_slug="sem-101",
        limit=4,
        client=_fake_client(),
        include_rationale=True,
    )
    assert results
    # The exact-match chunk should surface near the top via the semantic arm.
    ids = [r.chunk_id for r in results]
    assert "sem_101_chunk_00003" in ids
    assert all(r.rationale["engine"] == "hybrid-rrf" for r in results)


def test_hybrid_fails_closed_on_missing_index(tmp_path):
    _write_course(tmp_path, n=3)  # no index
    with pytest.raises(SemanticIndexMissing):
        hybrid_rrf_retrieve(
            tmp_path, "topic", course_slug="sem-101", client=_fake_client(),
        )


# --------------------------------------------------------------------------
# Lexical byte-identity regression — engine="lexical" == pre-change
# --------------------------------------------------------------------------


def test_lexical_engine_byte_identical_default(tmp_path):
    """retrieve_chunks(engine="lexical") produces byte-identical to_dict()
    output vs the default (no engine kwarg) call. Proves the additive engine
    axis is zero-behavior-change for existing lexical callers."""
    _write_course(tmp_path, n=6)
    # NB: no index built — the lexical path must not touch it.
    kwargs = dict(course_slug="sem-101", limit=5, min_relevance=0.0)

    baseline = retrieve_chunks(tmp_path, "topic chunk", **kwargs)
    with_engine = retrieve_chunks(tmp_path, "topic chunk", engine="lexical", **kwargs)

    base_json = json.dumps([r.to_dict() for r in baseline], sort_keys=True)
    eng_json = json.dumps([r.to_dict() for r in with_engine], sort_keys=True)
    assert base_json == eng_json
    # And it returned real lexical results (not an accidental empty match).
    assert baseline


def test_lexical_engine_with_rationale_byte_identical(tmp_path):
    _write_course(tmp_path, n=6)
    kwargs = dict(
        course_slug="sem-101", limit=5, min_relevance=0.0, include_rationale=True,
    )
    baseline = retrieve_chunks(tmp_path, "topic chunk", **kwargs)
    with_engine = retrieve_chunks(tmp_path, "topic chunk", engine="lexical", **kwargs)
    assert json.dumps([r.to_dict() for r in baseline], sort_keys=True) == json.dumps(
        [r.to_dict() for r in with_engine], sort_keys=True
    )
