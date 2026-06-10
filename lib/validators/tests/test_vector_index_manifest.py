"""Tests for VectorIndexManifestValidator (WS2 E2).

Mirrors the surface of ``test_chunkset_manifest.py``: happy path,
missing-input / not-found / invalid-JSON critical paths, schema-violation
paths, sibling-artifact existence, SHA re-verification (the load-bearing
tamper-detection signal), chunks_count cross-check, and the (warning)
staleness cross-check when a ``course_dir`` is supplied.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.vector_index_manifest import (  # noqa: E402
    VectorIndexManifestValidator,
)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(p.read_bytes())


def _build_index_dir(
    tmp_path: Path,
    *,
    n: int = 3,
    dim: int = 4,
    provider: str = "st",
    chunkset_kind: str = "imscc",
    source_chunks_sha: str | None = None,
    perturb: Dict[str, Any] | None = None,
) -> Path:
    """Write a self-consistent vector_index/ dir and return the manifest
    path. ``perturb`` overrides manifest fields after the consistent build
    (for negative paths)."""
    index_dir = tmp_path / "vector_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = [f"c{i}" for i in range(n)]
    id_map_bytes = (
        json.dumps(
            {"schema_version": "1.0", "chunk_ids": chunk_ids},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    id_map_path = index_dir / "id_map.json"
    id_map_path.write_bytes(id_map_bytes)

    matrix = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        matrix[i, i % dim] = 1.0
    emb_path = index_dir / "embeddings.npy"
    with emb_path.open("wb") as fh:
        np.save(fh, matrix, allow_pickle=False)

    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "embedding_provider": provider,
        "embedding_kind": provider if provider in {"st", "fake"} else "st",
        "embedding_model_id": "test-model",
        "embedding_model_revision": None,
        "embedding_dim": dim,
        "normalized": True,
        "index_type": "exact-numpy",
        "chunker_version": "v4",
        "chunkset_kind": chunkset_kind,
        "source_chunks_sha256": source_chunks_sha or ("a" * 64),
        "embeddings_sha256": _sha_file(emb_path),
        "id_map_sha256": _sha_file(id_map_path),
        "chunks_count": n,
        "text_field_policy": "text+heading",
        "batch_size": 8,
        "device": "cpu",
    }
    if perturb:
        manifest.update(perturb)

    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _validate(manifest_path: Path, **extra: Any):
    inputs: Dict[str, Any] = {"vector_index_manifest_path": str(manifest_path)}
    inputs.update(extra)
    return VectorIndexManifestValidator().validate(inputs)


def test_happy_path(tmp_path):
    manifest_path = _build_index_dir(tmp_path)
    result = _validate(manifest_path)
    assert result.passed, [i.code for i in result.issues]
    assert result.action is None
    assert result.score == 1.0


def test_missing_input():
    result = VectorIndexManifestValidator().validate({})
    assert not result.passed
    assert result.action == "block"
    assert result.issues[0].code == "VECTOR_INDEX_MANIFEST_MISSING_INPUT"


def test_not_found(tmp_path):
    result = _validate(tmp_path / "vector_index" / "manifest.json")
    assert not result.passed
    assert result.issues[0].code == "VECTOR_INDEX_MANIFEST_NOT_FOUND"


def test_invalid_json(tmp_path):
    index_dir = tmp_path / "vector_index"
    index_dir.mkdir(parents=True)
    bad = index_dir / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    result = _validate(bad)
    assert not result.passed
    assert result.issues[0].code == "VECTOR_INDEX_MANIFEST_INVALID_JSON"


def test_schema_violation_missing_field(tmp_path):
    manifest_path = _build_index_dir(tmp_path)
    doc = json.loads(manifest_path.read_text())
    doc.pop("embedding_dim")
    manifest_path.write_text(json.dumps(doc), encoding="utf-8")
    result = _validate(manifest_path)
    assert not result.passed
    assert any(
        i.code == "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION" for i in result.issues
    )


def test_schema_violation_additional_property(tmp_path):
    manifest_path = _build_index_dir(tmp_path, perturb={"surprise": "x"})
    result = _validate(manifest_path)
    assert not result.passed
    assert any(
        i.code == "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION" for i in result.issues
    )


def test_embeddings_sha_mismatch(tmp_path):
    manifest_path = _build_index_dir(
        tmp_path, perturb={"embeddings_sha256": "d" * 64}
    )
    result = _validate(manifest_path)
    assert not result.passed
    assert any(i.code == "VECTOR_INDEX_HASH_MISMATCH" for i in result.issues)


def test_id_map_sha_mismatch(tmp_path):
    manifest_path = _build_index_dir(
        tmp_path, perturb={"id_map_sha256": "e" * 64}
    )
    result = _validate(manifest_path)
    assert not result.passed
    assert any(i.code == "VECTOR_INDEX_HASH_MISMATCH" for i in result.issues)


def test_missing_sibling_artifact(tmp_path):
    manifest_path = _build_index_dir(tmp_path)
    (manifest_path.parent / "embeddings.npy").unlink()
    result = _validate(manifest_path)
    assert not result.passed
    assert any(
        i.code == "VECTOR_INDEX_ARTIFACT_NOT_FOUND" for i in result.issues
    )


def test_count_mismatch(tmp_path):
    manifest_path = _build_index_dir(tmp_path, n=3, perturb={"chunks_count": 99})
    result = _validate(manifest_path)
    assert not result.passed
    assert any(i.code == "VECTOR_INDEX_COUNT_MISMATCH" for i in result.issues)


def test_staleness_warning_with_course_dir(tmp_path):
    # course_dir = tmp_path; chunkset at tmp_path/imscc_chunks/chunks.jsonl.
    chunks_dir = tmp_path / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    chunks_jsonl = chunks_dir / "chunks.jsonl"
    chunks_jsonl.write_text('{"id": "c0", "text": "hi"}\n', encoding="utf-8")
    live_sha = _sha_file(chunks_jsonl)

    # Manifest pins a DIFFERENT source sha => stale (warning).
    manifest_path = _build_index_dir(
        tmp_path, source_chunks_sha="f" * 64, chunkset_kind="imscc"
    )
    result = _validate(manifest_path, course_dir=str(tmp_path))
    assert result.passed  # staleness is warning-only at the gate
    assert any(i.code == "VECTOR_INDEX_STALE" for i in result.issues)
    # And a matching sha => no staleness warning.
    manifest_path2 = _build_index_dir(
        tmp_path / "fresh", source_chunks_sha=live_sha, chunkset_kind="imscc"
    )
    # point the fresh index's course_dir at the same chunkset
    result2 = _validate(manifest_path2, course_dir=str(tmp_path))
    assert not any(i.code == "VECTOR_INDEX_STALE" for i in result2.issues)
