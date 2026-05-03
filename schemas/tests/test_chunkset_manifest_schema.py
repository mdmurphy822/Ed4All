"""Phase 7b Subtask 12 — regression tests for chunkset_manifest.schema.json.

The schema is the canonical contract for the per-chunkset sidecar
manifest emitted alongside ``chunks.jsonl`` at:

  - ``LibV2/courses/<slug>/dart_chunks/manifest.json``  (Phase 7b)
  - ``LibV2/courses/<slug>/imscc_chunks/manifest.json`` (Phase 7c)

Symmetric across both chunkset kinds (decision #6 from the Phase 7b-prep
investigation): one file, one schema, discriminated by ``chunkset_kind``
plus a conditional ``source_*_sha256`` requirement enforced via
``allOf / if / then``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "library" / "chunkset_manifest.schema.json"
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _minimal_dart_doc() -> dict:
    return {
        "chunks_sha256": "a" * 64,
        "chunker_version": "0.1.0",
        "chunkset_kind": "dart",
        "source_dart_html_sha256": "b" * 64,
    }


def _minimal_imscc_doc() -> dict:
    return {
        "chunks_sha256": "a" * 64,
        "chunker_version": "0.1.0",
        "chunkset_kind": "imscc",
        "source_imscc_sha256": "c" * 64,
    }


@pytest.mark.unit
def test_schema_file_exists_and_is_valid_json_schema():
    """The schema itself must be parseable JSON Schema."""
    assert SCHEMA_PATH.exists(), f"schema missing: {SCHEMA_PATH}"
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.Draft7Validator.check_schema(schema)
    assert schema.get("title") == "Chunkset Manifest"


@pytest.mark.unit
def test_minimal_dart_manifest_validates():
    """A minimal dart-chunkset doc with required fields must validate."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.validate(_minimal_dart_doc(), schema)


@pytest.mark.unit
def test_minimal_imscc_manifest_validates():
    """A minimal imscc-chunkset doc with required fields must validate."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.validate(_minimal_imscc_doc(), schema)


@pytest.mark.unit
def test_full_dart_manifest_with_optionals_validates():
    """A doc carrying all optional fields (chunks_count, generated_at)
    must validate."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc.update({
        "chunks_count": 42,
        "generated_at": "2026-05-03T11:25:00Z",
    })
    jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_chunker_version_fallback_sentinel_validates():
    """The '0.0.0+missing' sentinel emitted by
    MCP/tools/pipeline_tools.py::_resolve_chunker_version when
    ed4all-chunker isn't importable must be schema-valid."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunker_version"] = "0.0.0+missing"
    jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_chunker_version_prerelease_validates():
    """semver -prerelease suffixes must be schema-valid."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunker_version"] = "1.0.0-rc1"
    jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_dart_kind_without_source_dart_html_sha256_fails():
    """chunkset_kind=='dart' without source_dart_html_sha256 must fail."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc.pop("source_dart_html_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_imscc_kind_without_source_imscc_sha256_fails():
    """chunkset_kind=='imscc' without source_imscc_sha256 must fail."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_imscc_doc()
    doc.pop("source_imscc_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_missing_chunks_sha256_fails():
    """chunks_sha256 is unconditionally required."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc.pop("chunks_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_missing_chunker_version_fails():
    """chunker_version is unconditionally required."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc.pop("chunker_version")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_missing_chunkset_kind_fails():
    """chunkset_kind is unconditionally required."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc.pop("chunkset_kind")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_invalid_chunkset_kind_enum_fails():
    """chunkset_kind values outside {dart, imscc} must fail."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunkset_kind"] = "pdf"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_invalid_sha_pattern_fails():
    """SHA fields must be 64-char lowercase hex."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunks_sha256"] = "NOTAHEX"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_invalid_chunker_version_pattern_fails():
    """chunker_version must match the semver(+local|-prerelease) pattern."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunker_version"] = "v0.1"  # missing patch + bad 'v' prefix
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_negative_chunks_count_fails():
    """chunks_count must be >= 0."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["chunks_count"] = -1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_additional_properties_rejected():
    """additionalProperties: false — unknown fields must fail closed."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_dart_doc()
    doc["unexpected_field"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
