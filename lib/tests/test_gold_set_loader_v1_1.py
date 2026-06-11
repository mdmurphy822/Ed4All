"""retrieval-answer-eval-set P0 — dual-version loader tests for
lib/retrieval/gold_set.py.

Hermetic: builds a tiny course layout in tmp_path (chunks.jsonl + gold_set.json)
and exercises:
  - v1.1 doc validates against the v1.1 schema (procedural + multi_part).
  - v1.0 doc still validates against the v1.0 schema (back-compat).
  - the loader selects the schema by the doc's schema_version.
  - content_sha256 round-trip (match -> clean; mismatch -> GOLD_SET_CONTENT_SHA_MISMATCH).
  - the chunk_content_sha256 helper is stable + matches the validator's check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lib.retrieval._text import normalize_ws
from lib.retrieval.gold_set import (
    chunk_content_sha256,
    critical_issues,
    doc_schema_version,
    load_gold_set,
)
from lib.utils import sha256_file


def _codes(issues):
    return {i.code for i in issues}


_CHUNK_TEXT_1 = (
    "A vector store is a database that indexes high-dimensional embedding "
    "vectors for nearest-neighbor retrieval."
)
_CHUNK_TEXT_2 = (
    "To build the index, embed each chunk, normalize the vectors, then "
    "persist embeddings.npy alongside the id_map and the manifest."
)


def _write_chunks(course_dir: Path, rel: str = "corpus/chunks.jsonl") -> Path:
    path = course_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        {"id": "c001", "text": _CHUNK_TEXT_1,
         "source": {"item_path": "week_01/intro.html", "section_heading": "Vector Stores"},
         "learning_outcome_refs": ["to-01", "co-01"]},
        {"id": "c002", "text": _CHUNK_TEXT_2,
         "source": {"item_path": "week_02/build.html", "section_heading": "Building"},
         "learning_outcome_refs": ["co-02"]},
    ]
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    return path


def _csha(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).lower().encode("utf-8")).hexdigest()


def _v1_1_gold(course_dir: Path, *, content_sha: str | None = None) -> dict:
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    anchor = {
        "item_path": "week_01/intro.html",
        "section_heading": "Vector Stores",
        "text_quote": "A vector store is a database that indexes high-dimensional embedding vectors",
    }
    if content_sha is not None:
        anchor["content_sha256"] = content_sha
    return {
        "schema_version": "1.1",
        "course_slug": "demo-101",
        "chunkset": {
            "kind": "corpus",
            "chunks_path": "corpus/chunks.jsonl",
            "chunks_sha256": sha256_file(chunks_path),
        },
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-demo-101-0001",
                "question_type": "procedural",
                "question_text": "How do you build a vector store index?",
                "difficulty": "medium",
                "objective_refs": ["TO-01"],
                "relevant_passages": [
                    {"chunk_id": "c001", "relevance": "primary", "anchor": anchor}
                ],
                "authoring": {
                    "method": "manual", "author": "@t",
                    "reviewed_by": "PENDING_REVIEW", "status": "seed",
                },
            },
            {
                "question_id": "gq-demo-101-0002",
                "question_type": "multi_part",
                "question_text": "List the build steps and one uncovered detail.",
                "parts": [
                    {"part_id": "a", "part_text": "first step", "covered": True,
                     "relevant_passage_refs": ["c002"]},
                    {"part_id": "b", "part_text": "the GPU memory budget formula",
                     "covered": False,
                     "absence_note": "The corpus never states a GPU memory budget formula; dry-run confirmed."},
                ],
                "relevant_passages": [
                    {"chunk_id": "c002", "relevance": "primary",
                     "anchor": {"item_path": "week_02/build.html",
                                "text_quote": "To build the index, embed each chunk, normalize the vectors"}}
                ],
                "authoring": {
                    "method": "manual", "author": "@t",
                    "reviewed_by": "PENDING_REVIEW", "status": "seed",
                },
            },
        ],
    }


def _write_gold(course_dir: Path, doc: dict) -> None:
    p = course_dir / "retrieval_eval" / "gold_set.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------- dual version


def test_v1_1_doc_loads_clean(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    _write_gold(course, _v1_1_gold(course))
    gold, issues = load_gold_set(course, verify=True)
    assert doc_schema_version(gold) == "1.1"
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]


def test_v1_0_doc_still_loads(tmp_path: Path):
    """A v1.0 doc validates against the v1.0 schema (back-compat preserved)."""
    course = tmp_path / "demo-101"
    _write_chunks(course, rel="dart_chunks/chunks.jsonl")
    chunks_path = course / "dart_chunks" / "chunks.jsonl"
    doc = {
        "schema_version": "1.0",
        "course_slug": "demo-101",
        "chunkset": {
            "kind": "dart",
            "chunks_path": "dart_chunks/chunks.jsonl",
            "chunks_sha256": sha256_file(chunks_path),
        },
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-demo-101-0001",
                "question_type": "factual_recall",
                "question_text": "What does a vector store index?",
                "relevant_passages": [
                    {"chunk_id": "c001", "relevance": "primary",
                     "anchor": {"item_path": "week_01/intro.html",
                                "text_quote": "A vector store is a database that indexes high-dimensional embedding vectors"}}
                ],
                "authoring": {"method": "manual", "author": "@t",
                              "reviewed_by": "PENDING_REVIEW", "status": "seed"},
            }
        ],
    }
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert doc_schema_version(gold) == "1.0"
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]


def test_v1_0_doc_with_v1_1_type_rejected(tmp_path: Path):
    """A v1.0-declared doc using a v1.1-only question_type must be rejected by
    the v1.0 schema (the version pin is load-bearing)."""
    course = tmp_path / "demo-101"
    _write_chunks(course, rel="dart_chunks/chunks.jsonl")
    chunks_path = course / "dart_chunks" / "chunks.jsonl"
    doc = {
        "schema_version": "1.0",
        "course_slug": "demo-101",
        "chunkset": {"kind": "dart", "chunks_path": "dart_chunks/chunks.jsonl",
                     "chunks_sha256": sha256_file(chunks_path)},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-demo-101-0001",
                "question_type": "procedural",  # v1.1-only under a v1.0 pin
                "question_text": "How do you build a vector store index?",
                "relevant_passages": [
                    {"chunk_id": "c001", "relevance": "primary",
                     "anchor": {"item_path": "week_01/intro.html",
                                "text_quote": "A vector store is a database that indexes high-dimensional embedding vectors"}}
                ],
                "authoring": {"method": "manual", "author": "@t",
                              "reviewed_by": "PENDING_REVIEW", "status": "seed"},
            }
        ],
    }
    _write_gold(course, doc)
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_SCHEMA_VIOLATION" in _codes(issues)


# ---------------------------------------------------------------- content_sha256


def test_content_sha_match_is_clean(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    _write_gold(course, _v1_1_gold(course, content_sha=_csha(_CHUNK_TEXT_1)))
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_CONTENT_SHA_MISMATCH" not in _codes(issues)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]


def test_content_sha_mismatch_detected(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    _write_gold(course, _v1_1_gold(course, content_sha="0" * 64))
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_CONTENT_SHA_MISMATCH" in _codes(issues)


def test_chunk_content_sha256_helper_matches(tmp_path: Path):
    chunk = {"id": "c001", "text": "  Mixed   WHITESPACE\nand Case  "}
    assert chunk_content_sha256(chunk) == _csha("Mixed WHITESPACE and Case")
    # stable across whitespace-only differences
    chunk2 = {"id": "c001", "text": "Mixed WHITESPACE and Case"}
    assert chunk_content_sha256(chunk) == chunk_content_sha256(chunk2)
