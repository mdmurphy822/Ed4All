"""retrieval-answer-eval-set (gold v1.2 foundation) — loader tests for the
three additive v1.2 question axes in lib/retrieval/gold_set.py.

Hermetic: builds a tiny course layout in tmp_path (chunks.jsonl + gold_set.json)
and exercises:
  - a v1.2 doc validates against the v1.2 schema (learner_intent /
    expected_behavior / prior_turns present and valid).
  - each new field: valid value loads clean; invalid enum / malformed shape →
    GOLD_SET_SCHEMA_VIOLATION; absent → clean + accessor default.
  - the public accessors question_learner_intent / question_expected_behavior /
    question_prior_turns return the authored value (or the documented default).
  - v1.1 backward-compat: a v1.1 doc still loads byte-identically (the new
    fields are v1.2-only; a v1.1 schema rejects them via additionalProperties).
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_set import (
    critical_issues,
    doc_schema_version,
    load_gold_set,
    question_expected_behavior,
    question_learner_intent,
    question_prior_turns,
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


def _write_gold(course_dir: Path, doc: dict) -> None:
    p = course_dir / "retrieval_eval" / "gold_set.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def _v1_2_gold(course_dir: Path) -> dict:
    """A minimal v1.2 gold doc carrying the fully-authored v1.1 metadata (so no
    metadata-completeness warnings) and NONE of the new v1.2 axes — tests add
    the axes onto the returned questions as needed."""
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    return {
        "schema_version": "1.2",
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
                "question_type": "factual_recall",
                "question_text": "What does a vector store index?",
                "difficulty": "easy",
                "expected_citation_population": "source",
                "expected_key_points": [
                    "A vector store indexes high-dimensional embedding vectors.",
                    "It supports nearest-neighbor retrieval.",
                ],
                "relevant_passages": [
                    {"chunk_id": "c001", "relevance": "primary",
                     "anchor": {"item_path": "week_01/intro.html",
                                "text_quote": "A vector store is a database that indexes high-dimensional embedding vectors"}}
                ],
                "authoring": {"method": "manual", "author": "@t",
                              "reviewed_by": "PENDING_REVIEW", "status": "seed"},
            },
        ],
    }


# ---------------------------------------------------------------- version dispatch


def test_v1_2_doc_loads_clean(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["learner_intent"] = "conceptual_why"
    doc["questions"][0]["expected_behavior"] = "answer_grounded"
    doc["questions"][0]["prior_turns"] = [
        {"role": "user", "text": "Tell me about databases."},
        {"role": "assistant", "text": "Sure — which kind?"},
    ]
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert doc_schema_version(gold) == "1.2"
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]


def test_v1_1_doc_still_loads_byte_identically(tmp_path: Path):
    """A v1.1 doc loads unchanged under the v1.1 schema (v1.2 is additive)."""
    course = tmp_path / "demo-101"
    _write_chunks(course, rel="dart_chunks/chunks.jsonl")
    chunks_path = course / "dart_chunks" / "chunks.jsonl"
    doc = {
        "schema_version": "1.1",
        "course_slug": "demo-101",
        "chunkset": {"kind": "dart", "chunks_path": "dart_chunks/chunks.jsonl",
                     "chunks_sha256": sha256_file(chunks_path)},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-demo-101-0001",
                "question_type": "factual_recall",
                "question_text": "What does a vector store index?",
                "difficulty": "easy",
                "expected_citation_population": "source",
                "expected_key_points": [
                    "A vector store indexes high-dimensional embedding vectors.",
                    "It supports nearest-neighbor retrieval.",
                ],
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
    assert doc_schema_version(gold) == "1.1"
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]


def test_v1_1_doc_with_v1_2_field_rejected(tmp_path: Path):
    """A v1.1-declared doc carrying a v1.2-only field is rejected by the v1.1
    schema (additionalProperties:false) — the version pin is load-bearing."""
    course = tmp_path / "demo-101"
    _write_chunks(course, rel="dart_chunks/chunks.jsonl")
    chunks_path = course / "dart_chunks" / "chunks.jsonl"
    doc = {
        "schema_version": "1.1",
        "course_slug": "demo-101",
        "chunkset": {"kind": "dart", "chunks_path": "dart_chunks/chunks.jsonl",
                     "chunks_sha256": sha256_file(chunks_path)},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-demo-101-0001",
                "question_type": "factual_recall",
                "question_text": "What does a vector store index?",
                "learner_intent": "conceptual_why",  # v1.2-only under a v1.1 pin
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


# ---------------------------------------------------------------- learner_intent


def test_learner_intent_valid_loads_and_reads(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["learner_intent"] = "error_diagnosis"
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_learner_intent(gold["questions"][0]) == "error_diagnosis"


def test_learner_intent_invalid_rejected(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["learner_intent"] = "vibes"  # not in the enum
    _write_gold(course, doc)
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_SCHEMA_VIOLATION" in _codes(issues)


def test_learner_intent_absent_reads_none(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)  # no learner_intent
    assert "learner_intent" not in doc["questions"][0]
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_learner_intent(gold["questions"][0]) is None
    # accessor defaults a garbage value / missing dict to None
    assert question_learner_intent({"learner_intent": 7}) is None
    assert question_learner_intent({}) is None


# ---------------------------------------------------------------- expected_behavior


def test_expected_behavior_valid_loads_and_reads(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["expected_behavior"] = "correct_premise"
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_expected_behavior(gold["questions"][0]) == "correct_premise"


def test_expected_behavior_invalid_rejected(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["expected_behavior"] = "deflect"  # not in the enum
    _write_gold(course, doc)
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_SCHEMA_VIOLATION" in _codes(issues)


def test_expected_behavior_absent_defaults_to_answer_grounded(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)  # no expected_behavior
    assert "expected_behavior" not in doc["questions"][0]
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_expected_behavior(gold["questions"][0]) == "answer_grounded"
    assert question_expected_behavior({"expected_behavior": 0}) == "answer_grounded"
    assert question_expected_behavior({}) == "answer_grounded"


# ---------------------------------------------------------------- prior_turns


def test_prior_turns_valid_loads_and_reads(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    turns = [
        {"role": "user", "text": "What is a vector store?"},
        {"role": "assistant", "text": "A database for embeddings."},
    ]
    doc["questions"][0]["prior_turns"] = turns
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_prior_turns(gold["questions"][0]) == turns


def test_prior_turns_invalid_entry_rejected(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)
    doc["questions"][0]["prior_turns"] = [
        {"role": "system", "text": "bad role"},  # role not in the enum
    ]
    _write_gold(course, doc)
    _, issues = load_gold_set(course, verify=True)
    assert "GOLD_SET_SCHEMA_VIOLATION" in _codes(issues)


def test_prior_turns_absent_reads_empty(tmp_path: Path):
    course = tmp_path / "demo-101"
    _write_chunks(course)
    doc = _v1_2_gold(course)  # no prior_turns
    assert "prior_turns" not in doc["questions"][0]
    _write_gold(course, doc)
    gold, issues = load_gold_set(course, verify=True)
    assert not critical_issues(issues), [(i.code, i.message) for i in issues]
    assert question_prior_turns(gold["questions"][0]) == []
    # accessor is lenient on an already-validated doc: drops malformed entries
    assert question_prior_turns({"prior_turns": "nope"}) == []
    assert question_prior_turns(
        {"prior_turns": [{"role": "user", "text": "hi"}, {"role": "x"}]}
    ) == [{"role": "user", "text": "hi"}]
    assert question_prior_turns({}) == []
