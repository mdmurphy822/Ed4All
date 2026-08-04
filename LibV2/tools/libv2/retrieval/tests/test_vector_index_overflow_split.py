"""W1b.2 SPLIT arm — index-build wiring + parent-resolving retrieval.

The count arm (ED4ALL_EMBED_OVERFLOW_GUARD) only reports over-window chunks;
this arm acts on the report by slicing an over-window chunk into contiguous
sub-window pieces BEFORE the encode, so its tail earns a vector instead of
being silently dropped by the encoder. The contract under test:

* split ON  -> id_map carries ``<parent>#pN`` sub-ids, one embedding row per
  id, and the manifest records the parent-level split stats;
* split OFF -> byte-identical to the count-only path (no new manifest keys,
  no id rewriting), including when the arm is armed over an in-window corpus;
* query side -> a sub-piece hit resolves to its PARENT chunk id, several
  sub-hits of one parent collapse to one hit keeping the best cosine, and no
  caller-visible surface ever emits a ``#pN`` id.

All coverage runs on the deterministic fake client (no weights, no network,
no GPU) with a stub-tight ``ED4ALL_EMBED_MAX_SEQ_TOKENS`` so the fixture text
overflows without needing real 512-token passages.
"""

from __future__ import annotations

import json
from pathlib import Path

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from LibV2.tools.libv2.retrieval.vector_index import (
    VectorIndexManifest,
    build_vector_index,
    collapse_sub_piece_hits,
    load_vector_index,
    parent_chunk_id,
)

from .test_vector_index import _FakeEmbeddingClient


_LONG_TEXT = (
    "Chunk {i} explains the derivative of a polynomial term by term and then "
    "applies the same rule to a longer expression so the passage comfortably "
    "exceeds the stubbed serving window used by this test fixture."
)


def _write_course(
    tmp_path: Path, n: int = 3, *, slug: str = "split-101", long_text: bool = True
) -> Path:
    course_dir = tmp_path / "courses" / slug
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    lines = []
    for i in range(n):
        text = _LONG_TEXT.format(i=i) if long_text else f"Short chunk {i}."
        lines.append(
            json.dumps(
                {
                    "id": f"split_101_chunk_{i:05d}",
                    "text": text,
                    "chunk_type": "explanation",
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


def _id_map(course_dir: Path) -> list:
    doc = json.loads(
        (course_dir / "vector_index" / "id_map.json").read_text("utf-8")
    )
    return list(doc["chunk_ids"])


# --------------------------------------------------------------------------
# Build side — split ON
# --------------------------------------------------------------------------


def test_split_on_emits_sub_pieces_with_one_row_each(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")
    course_dir = _write_course(tmp_path, n=3)

    manifest = build_vector_index(course_dir, client=_FakeEmbeddingClient())

    ids = _id_map(course_dir)
    sub_ids = [cid for cid in ids if "#p" in cid]
    assert sub_ids, "the over-window fixture must produce sub-pieces"
    # Every sub-id resolves back to a real parent chunk (anti-fabrication).
    assert {parent_chunk_id(cid) for cid in sub_ids} == {
        f"split_101_chunk_{i:05d}" for i in range(3)
    }
    # Contiguous 0-based numbering per parent.
    assert f"split_101_chunk_{0:05d}#p0" in sub_ids

    # One embedding row per id_map entry — the load-path invariant.
    matrix = np.load(course_dir / "vector_index" / "embeddings.npy")
    assert matrix.shape[0] == len(ids)
    assert manifest.chunks_count == len(ids)
    assert manifest.parent_chunks_count == 3
    assert manifest.chunks_count > manifest.parent_chunks_count

    # The index still loads (chunks_count / row-count cross-checks hold).
    index = load_vector_index(course_dir, allow_fake=True)
    assert len(index.chunk_ids) == matrix.shape[0]


def test_split_on_stamps_parent_level_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")
    course_dir = _write_course(tmp_path, n=3)

    manifest = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    block = manifest.embed_overflow
    assert isinstance(block, dict)
    assert block["split_enabled"] is True
    assert block["parent_records_scanned"] == 3
    assert block["pre_split_overflow_count"] == 3
    assert block["sub_pieces_created"] == manifest.chunks_count
    assert block["split_window_tokens"] == 12  # no document_prefix on the fake
    # The count fields describe the EMBEDDED rows: after slicing, nothing is
    # still over-window, which is the whole point of the arm.
    assert block["records_scanned"] == manifest.chunks_count
    assert block["overflow_count"] == 0
    assert block["max_observed_tokens"] <= 12

    # Round-trips on disk and stays inside the determinism content hash.
    loaded = VectorIndexManifest.from_file(
        course_dir / "vector_index" / "manifest.json"
    )
    assert loaded.embed_overflow == block
    assert loaded.parent_chunks_count == 3
    assert "parent_chunks_count" in manifest.content_dict()


def test_split_on_manifest_passes_validator(tmp_path, monkeypatch):
    from lib.validators.vector_index_manifest import VectorIndexManifestValidator

    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")
    course_dir = _write_course(tmp_path, n=3)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())

    result = VectorIndexManifestValidator().validate(
        {
            "vector_index_manifest_path": str(
                course_dir / "vector_index" / "manifest.json"
            )
        }
    )
    assert [
        i for i in result.issues
        if i.code == "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION"
    ] == []
    assert result.passed, [i.code for i in result.issues]


def test_split_uses_the_exact_counter_when_the_client_exposes_one(
    tmp_path, monkeypatch
):
    """A client offering ``token_counter()`` drives BOTH the split and the
    report — the estimate is a fallback, not the contract."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "8")

    class _CountingClient(_FakeEmbeddingClient):
        def token_counter(self):
            # 2 tokens per whitespace word — deliberately unlike the words x
            # 1.3 estimate, so the emitted piece sizes prove which one ran.
            return lambda text: 2 * len(str(text).split())

    course_dir = _write_course(tmp_path, n=1)
    manifest = build_vector_index(course_dir, client=_CountingClient())

    assert manifest.embed_overflow["max_observed_tokens"] <= 8
    # 4 words/piece under the exact counter vs 6 under the estimate.
    for cid in _id_map(course_dir):
        assert parent_chunk_id(cid) != cid
    assert manifest.chunks_count == manifest.embed_overflow["sub_pieces_created"]


# --------------------------------------------------------------------------
# Build side — split OFF stays byte-identical
# --------------------------------------------------------------------------


def _artifact_bytes(course_dir: Path) -> dict:
    index_dir = course_dir / "vector_index"
    return {
        name: (index_dir / name).read_bytes()
        for name in ("embeddings.npy", "id_map.json", "manifest.json")
    }


def test_split_unset_matches_explicit_off_byte_for_byte(tmp_path, monkeypatch):
    """Guard ON + split OFF is the count-and-stamp path, unchanged: no split
    keys, no parent_chunks_count, no id rewriting."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")

    monkeypatch.delenv("ED4ALL_EMBED_OVERFLOW_SPLIT", raising=False)
    unset_dir = _write_course(tmp_path / "a", n=3)
    unset = build_vector_index(unset_dir, client=_FakeEmbeddingClient())

    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "off")
    off_dir = _write_course(tmp_path / "b", n=3)
    off = build_vector_index(off_dir, client=_FakeEmbeddingClient())

    assert _artifact_bytes(unset_dir) == _artifact_bytes(off_dir)
    assert unset.content_dict() == off.content_dict()
    assert off.parent_chunks_count is None
    assert set(off.embed_overflow) == {
        "max_seq_tokens",
        "records_scanned",
        "overflow_count",
        "overflow_rate",
        "max_observed_tokens",
        "overflow_chunk_ids",
    }
    raw = (off_dir / "vector_index" / "manifest.json").read_text("utf-8")
    assert "split_enabled" not in raw
    assert "parent_chunks_count" not in raw
    assert all("#p" not in cid for cid in _id_map(off_dir))
    # The count arm still reports the over-window rows it did NOT split.
    assert off.embed_overflow["overflow_count"] == 3


def test_split_on_over_an_in_window_corpus_is_byte_identical(
    tmp_path, monkeypatch
):
    """Arming the split arm must not perturb a corpus that fits the window —
    ``split_overflow_records`` passes non-overflow records through unchanged."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "512")

    monkeypatch.delenv("ED4ALL_EMBED_OVERFLOW_SPLIT", raising=False)
    off_dir = _write_course(tmp_path / "a", n=3, long_text=False)
    off = build_vector_index(off_dir, client=_FakeEmbeddingClient())

    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    on_dir = _write_course(tmp_path / "b", n=3, long_text=False)
    on = build_vector_index(on_dir, client=_FakeEmbeddingClient())

    assert (off_dir / "vector_index" / "embeddings.npy").read_bytes() == (
        on_dir / "vector_index" / "embeddings.npy"
    ).read_bytes()
    assert _id_map(off_dir) == _id_map(on_dir)
    assert on.parent_chunks_count is None
    # The arm still records that it ran (and found nothing to slice); the
    # off-path block never grows the key at all.
    assert on.embed_overflow["sub_pieces_created"] == 0
    assert on.embed_overflow["pre_split_overflow_count"] == 0
    assert "sub_pieces_created" not in off.embed_overflow


def test_guard_off_never_splits(tmp_path, monkeypatch):
    """The split satellite is a no-op without its master guard."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "0")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")
    course_dir = _write_course(tmp_path, n=3)

    manifest = build_vector_index(course_dir, client=_FakeEmbeddingClient())
    assert manifest.embed_overflow is None
    assert manifest.parent_chunks_count is None
    assert _id_map(course_dir) == [f"split_101_chunk_{i:05d}" for i in range(3)]
    raw = (course_dir / "vector_index" / "manifest.json").read_text("utf-8")
    assert "embed_overflow" not in raw
    assert "parent_chunks_count" not in raw


# --------------------------------------------------------------------------
# Read side — sub-id -> parent resolution + best-score dedup
# --------------------------------------------------------------------------


def test_parent_chunk_id_only_strips_the_minted_shape():
    assert parent_chunk_id("c_00001#p0") == "c_00001"
    assert parent_chunk_id("c_00001#p12") == "c_00001"
    # A legitimate id that merely contains '#' (or '#p' + non-digits) is NOT
    # a sub-piece and must survive untouched.
    assert parent_chunk_id("week_01#block_02") == "week_01#block_02"
    assert parent_chunk_id("c_1#part") == "c_1#part"
    assert parent_chunk_id("c_1#p") == "c_1#p"
    assert parent_chunk_id("#p3") == "#p3"  # no parent to resolve to
    assert parent_chunk_id("plain_id") == "plain_id"


def test_collapse_keeps_the_best_score_per_parent():
    hits = [
        ("a#p0", 0.4),
        ("b#p2", 0.9),
        ("a#p3", 0.7),
        ("c", 0.5),
        ("b#p0", 0.2),
    ]
    assert collapse_sub_piece_hits(hits) == [
        ("b", 0.9),
        ("a", 0.7),
        ("c", 0.5),
    ]


def test_collapse_is_identity_without_sub_ids():
    hits = [("a", 0.9), ("b", 0.5)]
    assert collapse_sub_piece_hits(hits) == hits


def test_index_search_never_emits_a_sub_id(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")
    course_dir = _write_course(tmp_path, n=3)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())

    index = load_vector_index(course_dir, allow_fake=True)
    assert any("#p" in cid for cid in index.chunk_ids), "fixture must be split"

    # Query the exact vector of one sub-piece: it ranks #1 at cosine ~1.0 and
    # must surface as its PARENT id.
    sub_id = next(cid for cid in index.chunk_ids if cid.endswith("#p1"))
    row = index.chunk_ids.index(sub_id)
    hits = index.search(index.matrix[row], top_k=10)

    assert all("#p" not in cid for cid, _ in hits)
    assert hits[0][0] == parent_chunk_id(sub_id)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    # One hit per parent — the 3 fixture chunks, never their many pieces.
    assert len(hits) == len({parent_chunk_id(c) for c in index.chunk_ids})


def test_semantic_retrieval_hydrates_parents_of_split_rows(
    tmp_path, monkeypatch
):
    """End-to-end query path: sub-piece rows retrieve, hydrate against the
    UNSPLIT chunkset, and emit parent chunk ids."""
    from lib.embedding.providers import build_embedding_client
    from LibV2.tools.libv2.semantic_retriever import semantic_retrieve_chunks

    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_SPLIT", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "12")

    client = build_embedding_client(provider_name="fake")
    course_dir = _write_course(tmp_path, n=3)
    build_vector_index(course_dir, client=client)

    results = semantic_retrieve_chunks(
        tmp_path,
        "derivative of a polynomial term by term",
        course_slug="split-101",
        limit=3,
        client=client,
    )
    assert results, "the split index must still retrieve"
    ids = [r.chunk_id for r in results]
    assert all("#p" not in cid for cid in ids)
    assert len(ids) == len(set(ids))  # collapsed, never one slot per piece
    assert set(ids) <= {f"split_101_chunk_{i:05d}" for i in range(3)}
