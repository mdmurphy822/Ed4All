"""retrieval-answer-eval-set — gold-question metadata completeness gate
(lib/retrieval/gold_set.py::GOLD_QUESTION_METADATA_INCOMPLETE) + the rule-based
backfill PROPOSAL builder (lib/retrieval/gold_metadata_backfill.py).

Hermetic: synthetic gold docs + a tiny chunkset in tmp_path; no real course
data, no network. The gate is WARNING-severity by contract (a partially
authored set must still load + eval — the frozen v1.1 set has seed questions
missing difficulty / population), so the tests assert the warning fires without
ever flipping has_critical_issues.
"""
from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_metadata_backfill import (
    build_backfill_proposals,
    generate_backfill_proposal,
    propose_difficulty,
    propose_population,
)
from lib.retrieval.gold_set import (
    critical_issues,
    has_critical_issues,
    validate_gold_set,
)
from lib.utils import sha256_file


# --------------------------------------------------------------- fixtures


def _chunk(cid, text, *, item_path, source_refs=None):
    src = {"item_path": item_path}
    if source_refs:
        src["source_references"] = source_refs
    return {"id": cid, "text": text, "source": src}


_TEXT = "A vector store indexes high-dimensional embedding vectors for retrieval."


def _chunks():
    return {
        # source population (top-level *.html + source_references)
        "c_src": _chunk("c_src", _TEXT, item_path="textbook.html",
                        source_refs=[{"sourceId": "dart:tb#1"}]),
        # course population (week_NN page, no source_refs)
        "c_crs": _chunk("c_crs", _TEXT, item_path="week_02/page.html"),
    }


def _question(qid, *, chunk_id, difficulty=None, population=None,
              key_points=None, qtype="factual_recall"):
    q = {
        "question_id": qid,
        "question_type": qtype,
        "question_text": "what does a vector store index for retrieval?",
        "relevant_passages": [{
            "chunk_id": chunk_id, "relevance": "primary",
            "anchor": {"item_path": "week_02/page.html",
                       "text_quote": _TEXT}}],
        "authoring": {"method": "manual", "author": "@t",
                      "reviewed_by": "@r", "status": "reviewed"},
    }
    if difficulty is not None:
        q["difficulty"] = difficulty
    if population is not None:
        q["expected_citation_population"] = population
    if key_points is not None:
        q["expected_key_points"] = key_points
    return q


def _gold(questions):
    return {
        "schema_version": "1.1", "course_slug": "demo-101",
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": "0" * 64},
        "authored_at": "2026-06-11T00:00:00Z", "frozen": False,
        "questions": questions,
    }


def _codes(issues):
    return [i.code for i in issues]


# --------------------------------------------------------------- gate


def test_metadata_gate_flags_missing_fields_warning_only():
    chunks = _chunks()
    # missing difficulty + population + key_points
    gold = _gold([_question("gq-demo-101-0001", chunk_id="c_crs")])
    issues = validate_gold_set(gold, chunks, "0" * 64)
    meta = [i for i in issues if i.code == "GOLD_QUESTION_METADATA_INCOMPLETE"]
    assert len(meta) == 1
    assert meta[0].severity == "warning"
    assert "difficulty" in meta[0].message
    assert "expected_citation_population" in meta[0].message
    assert "expected_key_points" in meta[0].message
    # warning never escalates to critical
    assert not has_critical_issues(meta)


def test_metadata_gate_clean_when_complete():
    chunks = _chunks()
    gold = _gold([_question(
        "gq-demo-101-0001", chunk_id="c_crs", difficulty="easy",
        population="course", key_points=["indexes embeddings", "for retrieval"])])
    issues = validate_gold_set(gold, chunks, "0" * 64)
    assert "GOLD_QUESTION_METADATA_INCOMPLETE" not in _codes(issues)


def test_metadata_gate_refusal_question_exempt_from_key_points():
    chunks = _chunks()
    q = _question("gq-demo-101-0001", chunk_id="c_crs",
                  difficulty="easy", population="course")
    q["is_refusal"] = True  # refusal entry => no expected_key_points required
    issues = validate_gold_set(_gold([q]), chunks, "0" * 64)
    assert "GOLD_QUESTION_METADATA_INCOMPLETE" not in _codes(issues)


def test_metadata_gate_does_not_break_load_path():
    chunks = _chunks()
    gold = _gold([_question("gq-demo-101-0001", chunk_id="c_crs")])
    issues = validate_gold_set(gold, chunks, "0" * 64)
    # the only criticals (if any) must NOT be the metadata code.
    assert all(i.code != "GOLD_QUESTION_METADATA_INCOMPLETE"
               for i in critical_issues(issues))


# --------------------------------------------------------------- backfill rules


def test_propose_population_source_course_both():
    chunks = _chunks()
    assert propose_population([chunks["c_src"]])[0] == "source"
    assert propose_population([chunks["c_crs"]])[0] == "course"
    assert propose_population([chunks["c_src"], chunks["c_crs"]])[0] == "both"
    assert propose_population([]) is None


def test_propose_difficulty_uses_heuristic():
    chunks = _chunks()
    q = _question("gq-1", chunk_id="c_crs", qtype="factual_recall")
    value, rationale = propose_difficulty(q, [chunks["c_crs"]])
    assert value in ("easy", "medium", "hard")
    assert "heuristic" in rationale


def test_build_backfill_proposals_skips_complete_questions():
    chunks = _chunks()
    gold = _gold([
        _question("gq-demo-101-0001", chunk_id="c_crs"),  # missing both
        _question("gq-demo-101-0002", chunk_id="c_src",
                  difficulty="easy", population="source"),  # complete
    ])
    proposals = build_backfill_proposals(gold, chunks)
    ids = {p.question_id for p in proposals}
    assert "gq-demo-101-0001" in ids
    assert "gq-demo-101-0002" not in ids
    p = next(p for p in proposals if p.question_id == "gq-demo-101-0001").to_dict()
    assert p["proposed_difficulty"] in ("easy", "medium", "hard")
    assert p["proposed_population"] == "course"
    assert p["source"] == "rule"


# --------------------------------------------------------------- end-to-end


def _write_course(tmp_path: Path, questions):
    cdir = tmp_path / "courses" / "demo-101"
    (cdir / "corpus").mkdir(parents=True)
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in _chunks().values():
            fh.write(json.dumps(c) + "\n")
    sha = sha256_file(chunks_path)
    gold = _gold(questions)
    gold["chunkset"]["chunks_sha256"] = sha
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    return cdir


def test_generate_backfill_proposal_writes_artifact(tmp_path):
    cdir = _write_course(tmp_path, [
        _question("gq-demo-101-0001", chunk_id="c_crs"),
        _question("gq-demo-101-0002", chunk_id="c_src"),
    ])
    doc, out_path = generate_backfill_proposal(cdir)
    assert out_path and out_path.name == "gold_metadata_backfill_proposal.json"
    assert out_path.exists()
    assert doc["n_questions"] == 2
    assert doc["method"] == "rule-based"
    # populations inferred per the seed chunk population
    by_id = {p["question_id"]: p for p in doc["proposals"]}
    assert by_id["gq-demo-101-0001"]["proposed_population"] == "course"
    assert by_id["gq-demo-101-0002"]["proposed_population"] == "source"
    # the gold set itself is never mutated
    gold_after = json.loads(
        (cdir / "retrieval_eval" / "gold_set.json").read_text())
    assert "difficulty" not in gold_after["questions"][0]


def test_generate_backfill_proposal_no_write(tmp_path):
    cdir = _write_course(tmp_path, [_question("gq-demo-101-0001", chunk_id="c_crs")])
    doc, out_path = generate_backfill_proposal(cdir, write=False)
    assert out_path is None
    assert doc["n_questions"] == 1
