"""Tests for the on-device vector index (WS2 E2).

All unit coverage runs on a LOCAL fake-embedding stub honoring the frozen
``EmbeddingClient`` protocol (§ 3 of the WS2 plan) — no model weights, no
network, no import-time coupling to E1's ``lib.embedding.providers`` (which
may not have landed yet). Covers:

* build -> load round-trip on a tiny fixture course;
* double-build BYTE-IDENTITY of embeddings.npy + id_map.json + manifest
  content (minus ``generated_at``) — the determinism contract (D4);
* nearest-neighbor query correctness + deterministic tie-break;
* staleness (mutate chunks.jsonl -> SemanticIndexStale);
* missing index -> SemanticIndexMissing;
* fake-provider refusal -> FakeIndexRefused unless allowed;
* on-disk sha tamper detection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from LibV2.tools.libv2.vector_index import (
    FakeIndexRefused,
    SemanticIndexMissing,
    SemanticIndexStale,
    VectorIndexManifest,
    build_vector_index,
    load_vector_index,
)


# --------------------------------------------------------------------------
# Local fake EmbeddingClient honoring the frozen § 3 protocol.
# --------------------------------------------------------------------------


class _FakeEmbeddingClient:
    """Deterministic hash-seeded unit vectors (dim 32), mirroring the plan's
    'fake' provider semantics WITHOUT importing E1's module."""

    def __init__(self, *, dim: int = 32, provider_name: str = "fake") -> None:
        self._dim = dim
        self._provider_name = provider_name
        self.batch_size = 8
        self.device = "cpu"

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return v / n

    def encode_batch(self, texts):
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def model_fingerprint(self) -> dict:
        return {
            "model_id": "fake-deterministic-v1",
            "revision": None,
            "provider_name": self._provider_name,
            "kind": "fake",
            "dim": self._dim,
            "batch_size": self.batch_size,
            "device": self.device,
        }


def _write_course(tmp_path: Path, n: int = 5) -> Path:
    """Create a tmp LibV2 course dir with a semantik_chunks/chunks.jsonl."""
    course_dir = tmp_path / "courses" / "fake-101"
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "id": f"fake_101_chunk_{i:05d}",
                    "text": f"This is chunk number {i} about topic {i}.",
                    "source": {"section_heading": f"Section {i}"},
                }
            )
        )
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return course_dir


# --------------------------------------------------------------------------
# Build / load round-trip
# --------------------------------------------------------------------------


def test_build_then_load_roundtrip(tmp_path):
    course_dir = _write_course(tmp_path, n=5)
    manifest = build_vector_index(course_dir, client=_FakeEmbeddingClient())

    assert isinstance(manifest, VectorIndexManifest)
    assert manifest.chunks_count == 5
    assert manifest.embedding_dim == 32
    assert manifest.embedding_provider == "fake"
    assert manifest.chunkset_kind == "semantik"
    assert manifest.normalized is True
    assert manifest.index_type == "exact-numpy"

    # vector_index/ dir materialized with all three artifacts.
    index_dir = course_dir / "vector_index"
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "id_map.json").exists()
    assert (index_dir / "manifest.json").exists()

    index = load_vector_index(course_dir, allow_fake=True)
    assert index.matrix.shape == (5, 32)
    assert index.matrix.dtype == np.float32
    assert len(index.chunk_ids) == 5
    assert index.chunk_ids[0] == "fake_101_chunk_00000"
    # rows are L2-normalized.
    norms = np.linalg.norm(index.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_manifest_from_file_roundtrip(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    built = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    loaded = VectorIndexManifest.from_file(
        course_dir / "vector_index" / "manifest.json"
    )
    assert loaded.source_chunks_sha256 == built.source_chunks_sha256
    assert loaded.embeddings_sha256 == built.embeddings_sha256
    assert loaded.to_dict()["chunks_count"] == 3


# --------------------------------------------------------------------------
# Determinism (D4) — double-build byte identity
# --------------------------------------------------------------------------


def test_double_build_byte_identical(tmp_path):
    course_a = _write_course(tmp_path / "a", n=5)
    course_b = _write_course(tmp_path / "b", n=5)

    m_a = build_vector_index(course_a, client=_FakeEmbeddingClient())
    m_b = build_vector_index(course_b, client=_FakeEmbeddingClient())

    emb_a = (course_a / "vector_index" / "embeddings.npy").read_bytes()
    emb_b = (course_b / "vector_index" / "embeddings.npy").read_bytes()
    assert emb_a == emb_b, "embeddings.npy not byte-identical across builds"

    idm_a = (course_a / "vector_index" / "id_map.json").read_bytes()
    idm_b = (course_b / "vector_index" / "id_map.json").read_bytes()
    assert idm_a == idm_b, "id_map.json not byte-identical across builds"

    # manifest content (everything except generated_at) identical.
    assert m_a.content_dict() == m_b.content_dict()
    assert m_a.embeddings_sha256 == m_b.embeddings_sha256
    assert m_a.id_map_sha256 == m_b.id_map_sha256


def test_rebuild_in_place_force_byte_identical(tmp_path):
    course_dir = _write_course(tmp_path, n=4)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    first = (course_dir / "vector_index" / "embeddings.npy").read_bytes()
    # rebuild with force (source sha unchanged -> needs force)
    build_vector_index(course_dir, client=_FakeEmbeddingClient(), force=True)
    second = (course_dir / "vector_index" / "embeddings.npy").read_bytes()
    assert first == second


def test_generated_at_excluded_from_content(tmp_path):
    course_dir = _write_course(tmp_path, n=2)
    m = build_vector_index(
        course_dir,
        client=_FakeEmbeddingClient(),
        generated_at="2026-06-09T00:00:00Z",
    )
    assert m.generated_at == "2026-06-09T00:00:00Z"
    assert "generated_at" not in m.content_dict()


# --------------------------------------------------------------------------
# Query correctness + deterministic tie-break
# --------------------------------------------------------------------------


def test_search_self_retrieval(tmp_path):
    course_dir = _write_course(tmp_path, n=6)
    client = _FakeEmbeddingClient()
    build_vector_index(course_dir, client=client)
    index = load_vector_index(course_dir, allow_fake=True)

    # Embed the exact text+heading of chunk 2 -> it must rank #1.
    target_text = "Section 2\nThis is chunk number 2 about topic 2."
    q = client.encode_batch([target_text])[0]
    results = index.search(q, top_k=3)
    assert results[0][0] == "fake_101_chunk_00002"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(results) == 3


def test_search_tie_break_by_chunk_id(tmp_path):
    # Two identical rows -> equal scores -> ordered by chunk_id asc.
    course_dir = tmp_path / "courses" / "ties"
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    rows = [
        {"id": "zzz", "text": "identical", "source": {"section_heading": "H"}},
        {"id": "aaa", "text": "identical", "source": {"section_heading": "H"}},
    ]
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    client = _FakeEmbeddingClient()
    build_vector_index(course_dir, client=client)
    index = load_vector_index(course_dir, allow_fake=True)
    q = client.encode_batch(["H\nidentical"])[0]
    results = index.search(q, top_k=2)
    # equal cosine -> 'aaa' before 'zzz'
    assert [cid for cid, _ in results] == ["aaa", "zzz"]


def test_search_empty_topk(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _FakeEmbeddingClient()
    build_vector_index(course_dir, client=client)
    index = load_vector_index(course_dir, allow_fake=True)
    assert index.search(client.encode_batch(["x"])[0], top_k=0) == []


# --------------------------------------------------------------------------
# Fail-closed read semantics (D7)
# --------------------------------------------------------------------------


def test_missing_index_raises(tmp_path):
    course_dir = _write_course(tmp_path, n=2)
    # no build -> missing
    with pytest.raises(SemanticIndexMissing):
        load_vector_index(course_dir, allow_fake=True)


def test_staleness_on_chunk_mutation(tmp_path):
    course_dir = _write_course(tmp_path, n=4)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    # load fine first
    load_vector_index(course_dir, allow_fake=True)
    # mutate chunks.jsonl -> stale
    chunks = course_dir / "semantik_chunks" / "chunks.jsonl"
    chunks.write_text(
        chunks.read_text(encoding="utf-8") + '{"id":"x","text":"new"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SemanticIndexStale):
        load_vector_index(course_dir, allow_fake=True)
    # verify_chunkset=False skips the staleness check.
    idx = load_vector_index(
        course_dir, allow_fake=True, verify_chunkset=False
    )
    assert idx.matrix.shape[0] == 4


def test_fake_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_ALLOW_FAKE", raising=False)
    course_dir = _write_course(tmp_path, n=2)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    with pytest.raises(FakeIndexRefused):
        load_vector_index(course_dir)  # allow_fake None -> env -> refused


def test_fake_allowed_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    course_dir = _write_course(tmp_path, n=2)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    idx = load_vector_index(course_dir)  # env allows fake
    assert idx.matrix.shape[0] == 2


def test_embeddings_tamper_detected(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    emb = course_dir / "vector_index" / "embeddings.npy"
    raw = bytearray(emb.read_bytes())
    raw[-1] ^= 0xFF  # flip a byte
    emb.write_bytes(bytes(raw))
    with pytest.raises(SemanticIndexStale):
        load_vector_index(course_dir, allow_fake=True)


# --------------------------------------------------------------------------
# Overwrite protection
# --------------------------------------------------------------------------


def test_fresh_index_refuses_overwrite_without_force(tmp_path):
    course_dir = _write_course(tmp_path, n=2)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    with pytest.raises(FileExistsError):
        build_vector_index(course_dir, client=_FakeEmbeddingClient())


def test_stale_index_rebuilds_without_force(tmp_path):
    course_dir = _write_course(tmp_path, n=2)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    # mutate chunks so the on-disk index is now stale
    chunks = course_dir / "semantik_chunks" / "chunks.jsonl"
    chunks.write_text(
        chunks.read_text(encoding="utf-8") + '{"id":"y","text":"z"}\n',
        encoding="utf-8",
    )
    # rebuild allowed without force because source sha differs
    m = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert m.chunks_count == 3


def test_empty_chunkset_builds_zero_rows(tmp_path):
    course_dir = tmp_path / "courses" / "empty-101"
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    m = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert m.chunks_count == 0
    idx = load_vector_index(course_dir, allow_fake=True)
    assert idx.matrix.shape[0] == 0
    assert idx.search(np.zeros(32, dtype=np.float32), top_k=5) == []


# --------------------------------------------------------------------------
# Asymmetric-retrieval prefixes (D6) — document_prefix applied at build,
# both prefixes recorded in the manifest.
# --------------------------------------------------------------------------


class _PrefixCapturingClient(_FakeEmbeddingClient):
    """Fake client exposing a ``resolved`` with prefixes (mirrors the
    registry-backed EmbeddingClient surface) + capturing the texts it
    receives so the build-time passage-prefix application is observable."""

    def __init__(self, *, document_prefix="", query_prefix="", dim=32):
        super().__init__(dim=dim)
        self.captured_texts = []

        class _Resolved:
            pass

        self.resolved = _Resolved()
        self.resolved.document_prefix = document_prefix
        self.resolved.query_prefix = query_prefix

    def encode_batch(self, texts):
        self.captured_texts = list(texts)
        return super().encode_batch(texts)


def test_document_prefix_applied_to_passages_at_build(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _PrefixCapturingClient(
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    build_vector_index(course_dir, client=client)
    # Every embedded passage carries the document prefix.
    assert client.captured_texts, "no texts were embedded"
    for text in client.captured_texts:
        assert text.startswith("search_document: "), text


def test_prefixes_recorded_in_manifest(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _PrefixCapturingClient(
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    m = build_vector_index(course_dir, client=client)
    assert m.document_prefix == "search_document: "
    assert m.query_prefix == "search_query: "
    # Round-trips through the on-disk manifest.
    loaded = VectorIndexManifest.from_file(
        course_dir / "vector_index" / "manifest.json"
    )
    assert loaded.document_prefix == "search_document: "
    assert loaded.query_prefix == "search_query: "


def test_query_only_prefix_leaves_passages_unprefixed(tmp_path):
    """bge-style: query-side prefix only; passages get NO prefix."""
    course_dir = _write_course(tmp_path, n=3)
    client = _PrefixCapturingClient(
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    m = build_vector_index(course_dir, client=client)
    for text in client.captured_texts:
        assert not text.startswith("Represent this sentence"), text
    assert m.document_prefix == ""
    assert m.query_prefix.startswith("Represent this sentence")


def test_no_resolved_attr_defaults_to_empty_prefixes(tmp_path):
    """Duck-typed clients without a ``resolved`` attr stay symmetric."""
    course_dir = _write_course(tmp_path, n=3)
    m = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert m.document_prefix == ""
    assert m.query_prefix == ""


# --------------------------------------------------------------------------
# W1b.2 — ED4ALL_EMBED_OVERFLOW_GUARD count-and-stamp arm.
# --------------------------------------------------------------------------


def test_embed_overflow_explicitly_off_byte_identical(tmp_path, monkeypatch):
    """Guard explicitly opted OUT: manifest carries no embed_overflow block,
    and the serialized manifest bytes never mention it (byte-identical
    off-path). The guard now DEFAULTS ON (report-only), so 'off' is an
    explicit falsey token rather than an unset env."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "0")
    course_dir = _write_course(tmp_path, n=5)
    m = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert m.embed_overflow is None
    assert "embed_overflow" not in m.to_dict()
    assert "embed_overflow" not in m.content_dict()
    raw = (course_dir / "vector_index" / "manifest.json").read_text("utf-8")
    assert "embed_overflow" not in raw


def test_embed_overflow_on_stamps_block(tmp_path, monkeypatch):
    """Guard ON with a tight token ceiling: every default chunk overflows, and
    the manifest carries the embed_overflow accounting block (count + stamp)."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "5")
    course_dir = _write_course(tmp_path, n=5)
    m = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert isinstance(m.embed_overflow, dict)
    assert m.embed_overflow["max_seq_tokens"] == 5
    assert m.embed_overflow["records_scanned"] == 5
    assert m.embed_overflow["overflow_count"] >= 1
    # It IS part of the determinism content hash when present.
    assert "embed_overflow" in m.content_dict()
    # Round-trips through the on-disk manifest.
    loaded = VectorIndexManifest.from_file(
        course_dir / "vector_index" / "manifest.json"
    )
    assert loaded.embed_overflow == m.embed_overflow


def test_embed_overflow_manifest_passes_validator(tmp_path, monkeypatch):
    """A guard-on manifest (with the embed_overflow block) still passes the
    VectorIndexManifestValidator — the schema admits the optional field."""
    from lib.validators.vector_index_manifest import VectorIndexManifestValidator

    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "5")
    course_dir = _write_course(tmp_path, n=4)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    manifest_path = course_dir / "vector_index" / "manifest.json"
    result = VectorIndexManifestValidator().validate(
        {"vector_index_manifest_path": str(manifest_path)}
    )
    schema_violations = [
        i for i in result.issues
        if i.code == "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION"
    ]
    assert schema_violations == [], schema_violations
    assert result.passed, [i.code for i in result.issues]
