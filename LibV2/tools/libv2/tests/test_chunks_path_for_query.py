"""Retrieval-stack fix: serve the chunkset the live vector index was built on.

A LibV2 course can carry several chunksets at once — e.g. an ``imscc_chunks/``
chunkset AND a ``corpus/`` UNION that is a superset of it. The on-device vector
index is built over exactly ONE of them and records which in
``vector_index/manifest.json::chunkset_kind``. The QUERY path (semantic
hydration + BM25 streaming) must read back the SAME chunkset, or every id the
index has that a precedence-resolved file lacks is silently DROPPED — a
union/corpus-legacy index becomes unserveable.

These tests pin ``resolve_chunks_path_for_query`` (lib/libv2_storage.py) and the
two retriever call sites that consume it (semantic hydration via
``_load_chunks_by_id`` and BM25 streaming via ``stream_chunks_from_course``).

All synthetic fixtures — no real course slugs / machine paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "numpy", reason="[embedding] extras absent — semantic-retrieval tests need numpy"
)

from lib.libv2_storage import (
    CHUNKSET_KIND_TO_DIRNAME,
    resolve_chunks_path_for_query,
    resolve_imscc_chunks_path,
)
from lib.embedding.providers import build_embedding_client

from LibV2.tools.libv2.retriever import stream_chunks_from_course
from LibV2.tools.libv2.semantic_retriever import (
    _load_chunks_by_id,
    semantic_retrieve_chunks,
)
from LibV2.tools.libv2.vector_index import build_vector_index


# --------------------------------------------------------------------------
# Synthetic fixture builders
# --------------------------------------------------------------------------


def _chunk(i: int) -> dict:
    return {
        "id": f"chunk_{i:05d}",
        "text": f"This is chunk number {i} about topic {i}.",
        "chunk_type": "explanation" if i % 2 == 0 else "example",
        "concept_tags": [f"topic-{i}"],
        "tokens_estimate": 10 + i,
        "learning_outcome_refs": [],
        "source": {"section_heading": f"Section {i}"},
    }


def _write_chunkset(course_dir: Path, dirname: str, ids: range) -> Path:
    chunks_dir = course_dir / dirname
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_chunk(i)) for i in ids]
    path = chunks_dir / "chunks.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stamp_manifest_kind(course_dir: Path, kind: str) -> None:
    """Write a minimal vector_index/manifest.json carrying chunkset_kind.

    Used by the 'mapped file missing' test where we want the manifest to point
    at a chunkset that doesn't exist on disk without building a real index.
    """
    index_dir = course_dir / "vector_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "manifest.json").write_text(
        json.dumps({"chunkset_kind": kind}), encoding="utf-8"
    )


def _fake_client():
    return build_embedding_client(provider_name="fake")


@pytest.fixture(autouse=True)
def _allow_fake(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")


# --------------------------------------------------------------------------
# (a) union-index scenario: corpus-legacy index serves the superset
# --------------------------------------------------------------------------


def test_union_index_resolves_to_corpus_file(tmp_path):
    """imscc_chunks has N ids; corpus has the N+M union; the index is stamped
    corpus-legacy -> the query resolver must return the corpus file (so the M
    extra ids are reachable), NOT the precedence-resolved imscc_chunks file."""
    course_dir = tmp_path / "courses" / "union-101"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))       # N=4
    corpus = _write_chunkset(course_dir, "corpus", range(0, 10))   # N+M=10
    # Build a real corpus-legacy index so the manifest is genuine.
    build_vector_index(course_dir, client=_fake_client(), chunkset="corpus-legacy")

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "manifest"
    assert path == corpus

    # Precedence (the OLD behavior) would have picked imscc_chunks — confirm
    # the fix actually diverges from precedence here.
    prec = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    assert prec == course_dir / "imscc_chunks" / "chunks.jsonl"
    assert path != prec


def test_union_index_hydration_reads_extra_ids(tmp_path):
    """_load_chunks_by_id (semantic hydration) reads the corpus union, so the M
    ids absent from imscc_chunks are present in the join table."""
    course_dir = tmp_path / "courses" / "union-102"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    _write_chunkset(course_dir, "corpus", range(0, 10))
    build_vector_index(course_dir, client=_fake_client(), chunkset="corpus-legacy")

    table = _load_chunks_by_id(course_dir)
    # All 10 union ids hydrate (not just the 4 imscc ids).
    assert set(table) == {f"chunk_{i:05d}" for i in range(10)}
    # The extra ids (5..9) that imscc_chunks lacks are present.
    assert "chunk_00009" in table


def test_union_index_bm25_streaming_reads_extra_ids(tmp_path):
    """BM25 streaming (stream_chunks_from_course) reads the corpus union too,
    so lexical + hybrid arms see the same superset the semantic arm does."""
    course_dir = tmp_path / "courses" / "union-103"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    _write_chunkset(course_dir, "corpus", range(0, 10))
    build_vector_index(course_dir, client=_fake_client(), chunkset="corpus-legacy")

    streamed = list(
        stream_chunks_from_course(course_dir, "union-103", "unknown")
    )
    ids = {c["id"] for c in streamed}
    assert ids == {f"chunk_{i:05d}" for i in range(10)}


def test_union_index_semantic_retrieves_extra_id(tmp_path):
    """End-to-end: a chunk that exists ONLY in the corpus union (not in
    imscc_chunks) is retrievable via the semantic engine — the previously
    dropped id now hydrates."""
    repo_root = tmp_path
    course_dir = repo_root / "courses" / "union-104"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    _write_chunkset(course_dir, "corpus", range(0, 10))
    build_vector_index(course_dir, client=_fake_client(), chunkset="corpus-legacy")

    # Self-retrieve the exact text+heading of an id only in the corpus union.
    target = "Section 8\nThis is chunk number 8 about topic 8."
    results = semantic_retrieve_chunks(
        repo_root, target, course_slug="union-104", limit=3, client=_fake_client(),
    )
    assert results
    assert results[0].chunk_id == "chunk_00008"


# --------------------------------------------------------------------------
# (b) no-index course -> precedence unchanged
# --------------------------------------------------------------------------


def test_no_index_falls_back_to_precedence(tmp_path):
    """No vector_index/ -> resolver returns the precedence file, byte-identical
    to resolve_imscc_chunks_path. Lexical-only courses keep working."""
    course_dir = tmp_path / "courses" / "noindex-101"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    _write_chunkset(course_dir, "corpus", range(0, 10))
    # No index built.

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "precedence"
    assert path == resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    assert path == course_dir / "imscc_chunks" / "chunks.jsonl"


def test_no_index_legacy_chunks_only_course_legacy_compat(tmp_path):
    """Legacy-compat: a course carrying only the pre-SemantiK
    ``dart_chunks/`` read-fallback dir (no index) resolves via precedence
    to that legacy dir — the dual-read path un-migrated corpora depend on."""
    course_dir = tmp_path / "courses" / "legacy-only-101"
    _write_chunkset(course_dir, "dart_chunks", range(0, 5))

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "precedence"
    assert path == course_dir / "dart_chunks" / "chunks.jsonl"


# --------------------------------------------------------------------------
# (c) manifest kind maps to a missing file -> warning + precedence fallback
# --------------------------------------------------------------------------


def test_manifest_kind_missing_file_warns_and_falls_back(tmp_path, caplog):
    """Manifest pins corpus-legacy but corpus/chunks.jsonl is absent -> warn +
    fall back to precedence (the semantic engine's sha check fails closed on
    this mismatch anyway; the warning makes the lexical fallback observable)."""
    course_dir = tmp_path / "courses" / "missing-101"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    # NO corpus/ dir written.
    _stamp_manifest_kind(course_dir, "corpus-legacy")
    assert "corpus-legacy" in CHUNKSET_KIND_TO_DIRNAME  # sanity

    with caplog.at_level("WARNING"):
        path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")

    assert resolution == "precedence"
    assert path == course_dir / "imscc_chunks" / "chunks.jsonl"
    assert any("is missing" in rec.message for rec in caplog.records)


def test_manifest_unknown_kind_falls_back(tmp_path):
    """An unknown / absent chunkset_kind -> precedence fallback, no crash."""
    course_dir = tmp_path / "courses" / "unknown-kind-101"
    _write_chunkset(course_dir, "imscc_chunks", range(0, 4))
    _stamp_manifest_kind(course_dir, "not-a-real-kind")

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "precedence"
    assert path == course_dir / "imscc_chunks" / "chunks.jsonl"


# --------------------------------------------------------------------------
# (d) imscc-kind manifest behaves identically to today
# --------------------------------------------------------------------------


def test_imscc_kind_manifest_resolves_to_imscc(tmp_path):
    """An imscc-pinned index resolves to imscc_chunks/ — same file precedence
    would pick. Zero behavior change for the common imscc case."""
    course_dir = tmp_path / "courses" / "imscc-101"
    imscc = _write_chunkset(course_dir, "imscc_chunks", range(0, 6))
    build_vector_index(course_dir, client=_fake_client(), chunkset="imscc")

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "manifest"
    assert path == imscc
    # Identical to precedence for this single-chunkset course.
    assert path == resolve_imscc_chunks_path(course_dir, "chunks.jsonl")


def test_legacy_kind_manifest_resolves_to_legacy_dir_legacy_compat(tmp_path):
    """Legacy-compat: an index pinned to the legacy ``dart`` chunkset_kind
    resolves to the pre-SemantiK ``dart_chunks/`` dir via the manifest —
    the dual-read path un-migrated corpora depend on."""
    course_dir = tmp_path / "courses" / "legacy-101"
    legacy = _write_chunkset(course_dir, "dart_chunks", range(0, 6))
    build_vector_index(course_dir, client=_fake_client(), chunkset="dart")

    path, resolution = resolve_chunks_path_for_query(course_dir, "chunks.jsonl")
    assert resolution == "manifest"
    assert path == legacy
