"""retrieval-answer-eval-set P1 — tests for lib/retrieval/gold_repin.py.

Exercises the §3 resolution ladder per branch:
  - content_sha256 exact match (survives a pure renumber).
  - text_quote unique containment (survives a re-split / renumber w/o sha).
  - text_quote scoped by item_path (disambiguates an ambiguous quote).
  - unresolved -> question parked status:draft + reported orphan.
  - frozen-set fail-closed (and the --unfreeze override).
  - chunkset pin rewrite + content_sha256 backfill.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lib.retrieval._text import normalize_ws
from lib.retrieval.gold_repin import (
    GoldRepinError,
    repin_gold_doc,
    repin_gold_set,
)
from lib.retrieval.gold_set import chunk_content_sha256
from lib.utils import sha256_file


def _csha(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).lower().encode("utf-8")).hexdigest()


_QUOTE_A = "A vector store indexes high-dimensional embedding vectors for retrieval."
_QUOTE_B = "To build the index you embed each chunk and normalize the vectors first."
_SHARED = "Common boilerplate sentence that appears in two different chunks here."


def _new_chunks():
    """The NEW (re-pinned) chunkset: same content as authoring, but the ids
    are RENUMBERED (the union-rebuild case) and one quote appears twice."""
    return {
        # quote A content lives under a renumbered id
        "new_500": {"id": "new_500", "text": f"intro. {_QUOTE_A} outro.",
                    "source": {"item_path": "week_01/a.html", "section_heading": "A"}},
        # quote B content under another renumbered id
        "new_600": {"id": "new_600", "text": f"steps: {_QUOTE_B} end.",
                    "source": {"item_path": "week_02/b.html", "section_heading": "B"}},
        # shared quote appears in two chunks with DIFFERENT item_paths
        "new_700": {"id": "new_700", "text": f"x {_SHARED} y",
                    "source": {"item_path": "week_03/c.html", "section_heading": "C"}},
        "new_701": {"id": "new_701", "text": f"p {_SHARED} q",
                    "source": {"item_path": "week_04/d.html", "section_heading": "D"}},
    }


def _gold(*, with_content_sha=False, ambiguous=False, orphan=False, frozen=False):
    passages_q1 = [{
        "chunk_id": "old_001", "relevance": "primary",
        "anchor": {"item_path": "week_01/a.html", "section_heading": "A",
                   "text_quote": _QUOTE_A,
                   **({"content_sha256": _csha(f"intro. {_QUOTE_A} outro.")} if with_content_sha else {})},
    }]
    questions = [
        {"question_id": "gq-demo-0001", "question_type": "factual_recall",
         "question_text": "what does a vector store index?", "relevant_passages": passages_q1,
         "authoring": {"method": "manual", "author": "@t",
                       "reviewed_by": "PENDING_REVIEW", "status": "seed"}},
        {"question_id": "gq-demo-0002", "question_type": "factual_recall",
         "question_text": "how do you build the index?",
         "relevant_passages": [{
             "chunk_id": "old_002", "relevance": "primary",
             "anchor": {"item_path": "week_02/b.html", "text_quote": _QUOTE_B}}],
         "authoring": {"method": "manual", "author": "@t",
                       "reviewed_by": "PENDING_REVIEW", "status": "seed"}},
    ]
    if ambiguous:
        questions.append({
            "question_id": "gq-demo-0003", "question_type": "factual_recall",
            "question_text": "the ambiguous boilerplate question",
            "relevant_passages": [{
                "chunk_id": "old_003", "relevance": "primary",
                "anchor": {"item_path": "week_04/d.html", "section_heading": "D",
                           "text_quote": _SHARED}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "PENDING_REVIEW", "status": "seed"}})
    if orphan:
        questions.append({
            "question_id": "gq-demo-0004", "question_type": "factual_recall",
            "question_text": "a question whose quote vanished",
            "relevant_passages": [{
                "chunk_id": "old_004", "relevance": "primary",
                "anchor": {"item_path": "week_09/z.html",
                           "text_quote": "this exact sentence is absent from every new chunk entirely now"}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "PENDING_REVIEW", "status": "seed"}})
    return {
        "schema_version": "1.1",
        "course_slug": "demo",
        "chunkset": {"kind": "dart", "chunks_path": "dart_chunks/chunks.jsonl",
                     "chunks_sha256": "f" * 64},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": frozen,
        "questions": questions,
    }


# ---------------------------------------------------------------- ladder branches


def test_renumber_resolved_by_content_sha():
    new_gold, report = repin_gold_doc(
        _gold(with_content_sha=True), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    q1 = new_gold["questions"][0]
    # old_001 -> new_500 by content_sha256
    assert q1["relevant_passages"][0]["chunk_id"] == "new_500"
    methods = {r.method for r in report.resolutions if r.question_id == "gq-demo-0001"}
    assert "content_sha256" in methods


def test_renumber_resolved_by_text_quote_when_no_sha():
    new_gold, report = repin_gold_doc(
        _gold(with_content_sha=False), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    # q1 has no content_sha -> resolved by unique text_quote
    assert new_gold["questions"][0]["relevant_passages"][0]["chunk_id"] == "new_500"
    assert new_gold["questions"][1]["relevant_passages"][0]["chunk_id"] == "new_600"
    methods = {r.method for r in report.resolutions}
    assert methods <= {"text_quote", "content_sha256"}


def test_ambiguous_quote_resolved_by_item_path_scope():
    new_gold, report = repin_gold_doc(
        _gold(ambiguous=True), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    # the shared quote is in new_700 + new_701; scoped to week_04/d.html -> new_701
    q3 = next(q for q in new_gold["questions"] if q["question_id"] == "gq-demo-0003")
    assert q3["relevant_passages"][0]["chunk_id"] == "new_701"
    method = next(r.method for r in report.resolutions if r.question_id == "gq-demo-0003")
    assert method == "text_quote_scoped"


def test_orphan_parks_question_as_draft():
    new_gold, report = repin_gold_doc(
        _gold(orphan=True), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    q4 = next(q for q in new_gold["questions"] if q["question_id"] == "gq-demo-0004")
    assert q4["authoring"]["status"] == "draft"
    assert "gq-demo-0004" in report.parked_question_ids
    orphans = [r for r in report.resolutions if r.method == "unresolved"]
    assert any(r.question_id == "gq-demo-0004" for r in orphans)


def test_content_sha_backfilled_on_resolve():
    new_gold, _ = repin_gold_doc(
        _gold(), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    anchor = new_gold["questions"][0]["relevant_passages"][0]["anchor"]
    assert anchor["content_sha256"] == chunk_content_sha256(_new_chunks()["new_500"])


def test_chunkset_pin_rewritten():
    new_gold, _ = repin_gold_doc(
        _gold(), _new_chunks(),
        kind="corpus", chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    assert new_gold["chunkset"] == {
        "kind": "corpus", "chunks_path": "corpus/chunks.jsonl", "chunks_sha256": "a" * 64,
    }


def test_input_doc_not_mutated():
    original = _gold()
    snapshot = json.loads(json.dumps(original))
    repin_gold_doc(original, _new_chunks(), kind="corpus",
                   chunks_path="corpus/chunks.jsonl", new_sha256="a" * 64)
    assert original == snapshot


# ---------------------------------------------------------------- I/O + fail-closed


def _write_course(tmp_path: Path, gold: dict) -> Path:
    course = tmp_path / "demo"
    (course / "retrieval_eval").mkdir(parents=True)
    (course / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    corpus = course / "corpus"
    corpus.mkdir()
    with (corpus / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in _new_chunks().values():
            fh.write(json.dumps(c) + "\n")
    return course


def test_repin_gold_set_writes_files(tmp_path: Path):
    course = _write_course(tmp_path, _gold(orphan=True))
    report, written = repin_gold_set(course, kind="corpus")
    assert written and written.exists()
    on_disk = json.loads(written.read_text())
    assert on_disk["chunkset"]["kind"] == "corpus"
    assert on_disk["chunkset"]["chunks_sha256"] == sha256_file(course / "corpus" / "chunks.jsonl")
    # a repin report sidecar was written
    reports = list((course / "retrieval_eval").glob("gold_repin_report_*.json"))
    assert reports


def test_repin_dry_run_writes_nothing(tmp_path: Path):
    course = _write_course(tmp_path, _gold())
    before = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    report, written = repin_gold_set(course, kind="corpus", dry_run=True)
    assert written is None
    after = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    assert before == after
    assert not list((course / "retrieval_eval").glob("gold_repin_report_*.json"))


def test_frozen_set_refused(tmp_path: Path):
    course = _write_course(tmp_path, _gold(frozen=True))
    with pytest.raises(GoldRepinError) as exc:
        repin_gold_set(course, kind="corpus")
    assert exc.value.code == "GOLD_SET_FROZEN"


def test_frozen_set_repinned_with_unfreeze(tmp_path: Path):
    course = _write_course(tmp_path, _gold(frozen=True))
    report, written = repin_gold_set(course, kind="corpus", unfreeze_for_repin=True)
    assert written and written.exists()


def test_missing_chunkset_refused(tmp_path: Path):
    course = tmp_path / "demo"
    (course / "retrieval_eval").mkdir(parents=True)
    (course / "retrieval_eval" / "gold_set.json").write_text(json.dumps(_gold()))
    with pytest.raises(GoldRepinError) as exc:
        repin_gold_set(course, kind="corpus")
    assert exc.value.code == "GOLD_SET_CHUNKSET_NOT_FOUND"
