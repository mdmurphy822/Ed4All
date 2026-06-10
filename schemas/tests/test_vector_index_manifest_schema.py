"""Regression tests for vector_index_manifest.schema.json (WS2 E2).

Canonical contract for the provenance + integrity manifest emitted at
``LibV2/courses/<slug>/vector_index/manifest.json`` alongside
``embeddings.npy`` + ``id_map.json``. Mirrors the accept/reject surface of
``test_chunkset_manifest_schema.py`` (the closest sibling library-manifest
schema) and additionally exercises the on-disk accept/reject fixtures under
``schemas/tests/fixtures/vector_index/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "library" / "vector_index_manifest.schema.json"
)
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "vector_index"


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _minimal_doc() -> dict:
    return {
        "schema_version": "1.0",
        "embedding_provider": "st",
        "embedding_kind": "st",
        "embedding_model_id": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "normalized": True,
        "index_type": "exact-numpy",
        "chunker_version": "v4",
        "chunkset_kind": "imscc",
        "source_chunks_sha256": "a" * 64,
        "embeddings_sha256": "b" * 64,
        "id_map_sha256": "c" * 64,
        "chunks_count": 295,
        "text_field_policy": "text+heading",
        "batch_size": 16,
        "device": "cpu",
    }


@pytest.mark.unit
def test_schema_file_exists_and_is_valid_json_schema():
    assert SCHEMA_PATH.exists(), f"schema missing: {SCHEMA_PATH}"
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.Draft7Validator.check_schema(schema)
    assert schema.get("title") == "Vector Index Manifest"


@pytest.mark.unit
def test_minimal_manifest_validates():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_minimal_doc(), _load_schema())


@pytest.mark.unit
def test_full_manifest_with_optionals_validates():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["embedding_model_revision"] = "f" * 40
    doc["generated_at"] = "2026-06-09T12:00:00Z"
    jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_null_revision_validates():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["embedding_model_revision"] = None
    jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
@pytest.mark.parametrize("field", sorted(_minimal_doc().keys()))
def test_missing_required_field_fails(field):
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc.pop(field)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_sha_pattern_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["source_chunks_sha256"] = "NOTAHEX"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_uppercase_sha_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["embeddings_sha256"] = "B" * 64
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_embedding_kind_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["embedding_kind"] = "tfidf"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_chunkset_kind_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["chunkset_kind"] = "pdf"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_index_type_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["index_type"] = "faiss-flat"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_text_field_policy_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["text_field_policy"] = "title_only"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_normalized_false_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["normalized"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_schema_version_not_one_zero_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["schema_version"] = "2.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_negative_chunks_count_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["chunks_count"] = -1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_additional_properties_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["surprise"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_bad_chunker_version_fails():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _minimal_doc()
    doc["chunker_version"] = "v0.1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


# ----------------------------------------------------------- fixture files


@pytest.mark.unit
@pytest.mark.parametrize(
    "fixture_name",
    ["valid_st_manifest.json", "valid_fake_manifest_null_revision.json"],
)
def test_accept_fixtures(fixture_name):
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    jsonschema.validate(doc, _load_schema())


@pytest.mark.unit
def test_reject_fixture():
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads(
        (FIXTURE_DIR / "invalid_bad_sha.json").read_text(encoding="utf-8")
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())
