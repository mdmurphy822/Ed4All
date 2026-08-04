"""Regression tests for GPT Feedback (May 12) item 3: chunk-level
``source.source_document_sha256``.

Validates:
  - Trainforge's ``CourseProcessor._create_chunk`` propagates
    ``self._source_document_sha256`` onto each chunk's source block
    when set, omits it when absent.
  - Chunks carrying the new field validate against ``chunk_v4``.
  - Legacy chunks without the field continue to validate.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_validator():
    """Mirror Trainforge/pipeline/tests/test_chunk_strict_validation.py loader."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schema = json.loads(CHUNK_SCHEMA_PATH.read_text())
    id_to_schema: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def _base_chunk() -> Dict[str, Any]:
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "week_01_overview",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "apply",
    }


def test_chunk_with_source_document_sha256_validates():
    """A chunk carrying the new field validates against chunk_v4 schema."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source"]["source_document_sha256"] = "a" * 64
    errors = sorted(validator.iter_errors(chunk), key=lambda e: e.path)
    assert not errors, [str(e) for e in errors]


def test_chunk_without_source_document_sha256_still_validates():
    """Legacy chunks (no new field) keep validating."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "source_document_sha256" not in chunk["source"]
    errors = sorted(validator.iter_errors(chunk), key=lambda e: e.path)
    assert not errors, [str(e) for e in errors]


def test_chunk_with_malformed_source_document_sha256_rejected():
    """Strict schema rejects a non-64-hex value."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source"]["source_document_sha256"] = "not-a-hash"
    errors = list(validator.iter_errors(chunk))
    assert errors, "Expected schema rejection of malformed SHA"


def _build_minimal_processor() -> Any:
    """Construct a bare ``CourseProcessor`` instance bypassing __init__.

    Mirrors the test_provenance.py pattern: only sets the fields
    ``_create_chunk`` reads + monkey-patches the metadata extractors
    that depend on full __init__ state.
    """
    from Trainforge.pipeline.process_course import CourseProcessor

    cp = CourseProcessor.__new__(CourseProcessor)
    cp.course_code = "TEST_101"
    cp.capture = None
    cp.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": collections.defaultdict(int),
        "difficulty_distribution": collections.defaultdict(int),
    }
    cp._all_concept_tags = set()
    # Monkey-patch extractors that read deeper __init__ state.
    cp._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    cp._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    cp._extract_objective_refs = (
        lambda i, section_heading=None: []
    )  # type: ignore[method-assign]
    cp._extract_section_metadata = lambda i, h, **kw: (
        None,
        None,
        [],
        {"key_claims": [], "objective_alignment": []},
        {
            "content_type_label": "none",
            "key_terms": "none",
            "key_claims": "none",
            "objective_alignment": "none",
        },
    )  # type: ignore[method-assign]
    cp._resolve_chunk_source_references = (
        lambda *, item, section_heading, section_source_ids, merged_headings=None: []
    )  # type: ignore[method-assign]
    return cp


def _minimal_item() -> Dict[str, Any]:
    return {
        "module_id": "week_01_intro",
        "module_title": "Week 1 — Intro",
        "item_id": "intro",
        "title": "Intro",
        "resource_type": "webcontent",
        "item_path": "week_01/intro.html",
        "learning_objectives": [],
    }


def test_trainforge_create_chunk_threads_source_document_sha256():
    """When ``self._source_document_sha256`` is set on a CourseProcessor
    instance, ``_create_chunk`` stamps the field onto the chunk's source.
    """
    cp = _build_minimal_processor()
    cp._source_document_sha256 = "b" * 64

    chunk = cp._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Hello world.",
        html="<p>Hello world.</p>",
        item=_minimal_item(),
        section_heading="Intro",
        chunk_type="explanation",
        position_in_module=0,
    )
    assert chunk["source"]["source_document_sha256"] == "b" * 64


def test_trainforge_create_chunk_omits_field_when_not_set():
    """When ``self._source_document_sha256`` is absent, field is omitted."""
    cp = _build_minimal_processor()
    # Attribute deliberately NOT set.

    chunk = cp._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Hello world.",
        html="<p>Hello world.</p>",
        item=_minimal_item(),
        section_heading="Intro",
        chunk_type="explanation",
        position_in_module=0,
    )
    assert "source_document_sha256" not in chunk["source"]
