"""retrieval-answer-eval-set P2 — CliRunner tests for the candidate-authoring CLI.

Covers ``libv2 gold-candidates``, ``libv2 probe-candidates``, ``libv2 gold-promote``.
Builds a synthetic union-corpus course in tmp_path; the local drafting client +
the retriever are monkeypatched to deterministic fakes so the tests never touch
Ollama or a vector index.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402
from lib.utils import sha256_file  # noqa: E402

_SLUG = "demo-union-101"

_TEXT_A = "A vector store indexes high-dimensional embedding vectors for fast retrieval."
_TEXT_B = "To add or subtract fractions, they must first share a common denominator value."


class _FakeClient:
    model = "qwen2.5:14b-instruct-q4_K_M"

    def __init__(self, response):
        self._response = response
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=512, temperature=0.2, **kw):
        self.calls.append(messages)
        return self._response


class _Result:
    def __init__(self, cid, score, text):
        self.chunk_id, self.score, self.text = cid, score, text


def _make_course(repo_root: Path, *, frozen=False) -> Path:
    cdir = repo_root / "courses" / _SLUG
    (cdir / "corpus").mkdir(parents=True)
    chunks = [
        {"id": "c001", "text": _TEXT_A, "source": {"item_path": "textbook.html",
         "source_references": [{"sourceId": "dart:tb#1"}]},
         "learning_outcome_refs": ["to-01", "co-01"],
         "concept_tags": ["vector-store", "embeddings"]},
        {"id": "c002", "text": _TEXT_B, "source": {"item_path": "week_01/p.html"},
         "learning_outcome_refs": ["co-02"], "concept_tags": ["fractions"]},
    ]
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    sha = sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1", "course_slug": _SLUG,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha},
        "authored_at": "2026-06-11T00:00:00Z", "frozen": frozen,
        "questions": [{
            "question_id": f"gq-{_SLUG}-0001", "question_type": "factual_recall",
            "question_text": "seed: what does a vector store index here?",
            "relevant_passages": [{"chunk_id": "c001", "relevance": "primary",
                "anchor": {"item_path": "textbook.html", "text_quote": _TEXT_A}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    (cdir / "manifest.json").write_text(json.dumps({"classification": {}}))
    return cdir


def _invoke(repo_root: Path, *args: str):
    return CliRunner().invoke(libv2_main, ["--repo", str(repo_root), *args])


def _patch_client(monkeypatch, response):
    fake = _FakeClient(response)
    monkeypatch.setattr(
        "lib.retrieval.answer_backend.build_answer_client",
        lambda **kw: fake,
    )
    return fake


def _draft(q="Explain in detail what this passage describes about the topic here?",
           quote=_TEXT_A, kps=("p1", "p2")):
    return json.dumps({"question_text": q, "expected_key_points": list(kps),
                       "quote": quote})


# --------------------------------------------------------------- gold-candidates


def test_gold_candidates_writes_artifact(tmp_path, monkeypatch):
    course = _make_course(tmp_path)
    _patch_client(monkeypatch, _draft())
    res = _invoke(tmp_path, "gold-candidates", "--course", _SLUG, "--n", "4")
    assert res.exit_code == 0, res.output
    assert "gold-candidates" in res.output
    art = course / "retrieval_eval" / "gold_candidates.json"
    assert art.exists()
    doc = json.loads(art.read_text())
    # only 2 chunks in the corpus -> sampler caps at 2 even though n=4 requested
    assert doc["authoring_run"]["n_requested"] == 4
    assert doc["authoring_run"]["n_drafted"] == 2
    assert all("prescreen" in c for c in doc["candidates"])


def test_gold_candidates_no_write(tmp_path, monkeypatch):
    course = _make_course(tmp_path)
    _patch_client(monkeypatch, _draft())
    res = _invoke(tmp_path, "gold-candidates", "--course", _SLUG, "--n", "2", "--no-write")
    assert res.exit_code == 0, res.output
    assert not (course / "retrieval_eval" / "gold_candidates.json").exists()


def test_gold_candidates_unknown_course(tmp_path, monkeypatch):
    _patch_client(monkeypatch, _draft())
    res = _invoke(tmp_path, "gold-candidates", "--course", "no-such")
    assert res.exit_code == 1
    assert "course not found" in res.output


_GLOSSARY_TEXT = (
    "Integer Any whole number, its negative counterpart, or zero. Absolute "
    "Value The distance between a number and zero on the number line. Opposite "
    "The number the same distance from zero but on the other side of zero. "
    "Number Line A horizontal line on which every real number has a position."
)
_WORKED_TEXT = (
    "Problem 1 — Scheduling. The chess club meets every 6 days. On what day do "
    "they next meet? Solution. We need the least common multiple of 6 and 8. "
    "Step 1: list multiples. Step 2: pick the smallest common multiple."
)


def _make_course_with_shapes(repo_root: Path) -> Path:
    """A course whose corpus carries a glossary (summary) chunk + a worked
    example chunk so the definition / worked_example arms have seeds."""
    cdir = repo_root / "courses" / _SLUG
    (cdir / "corpus").mkdir(parents=True)
    chunks = [
        {"id": "g001", "text": _GLOSSARY_TEXT, "chunk_type": "summary",
         "source": {"item_path": "week_03/summary.html",
                    "section_heading": "Week 3 Summary"},
         "learning_outcome_refs": ["to-03"]},
        {"id": "w001", "text": _WORKED_TEXT, "chunk_type": "exercise",
         "source": {"item_path": "week_01/work.html"},
         "learning_outcome_refs": ["to-01"]},
    ]
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    sha = sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1", "course_slug": _SLUG,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha},
        "authored_at": "2026-06-11T00:00:00Z", "frozen": False,
        "questions": [{
            "question_id": f"gq-{_SLUG}-0001", "question_type": "factual_recall",
            "question_text": "seed: define the term integer here?",
            "relevant_passages": [{"chunk_id": "g001", "relevance": "primary",
                "anchor": {"item_path": "week_03/summary.html",
                           "text_quote": _GLOSSARY_TEXT[:60]}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    (cdir / "manifest.json").write_text(json.dumps({"classification": {}}))
    return cdir


def test_gold_candidates_template_arms(tmp_path, monkeypatch):
    course = _make_course_with_shapes(tmp_path)
    # quote pulled verbatim from the glossary so the definition arm pre-screens.
    _patch_client(monkeypatch, _draft(
        q="What is the definition of the term integer in this glossary?",
        quote="Any whole number, its negative counterpart, or zero"))
    res = _invoke(tmp_path, "gold-candidates", "--course", _SLUG,
                  "--templates", "definition,worked_example")
    assert res.exit_code == 0, res.output
    doc = json.loads((course / "retrieval_eval" / "gold_candidates.json").read_text())
    bt = doc["authoring_run"]["by_template"]
    assert "definition" in bt and "worked_example" in bt
    assert "stratified" not in bt
    templates = {c["authoring"].get("template") for c in doc["candidates"]}
    assert templates <= {"definition", "worked_example"}


# --------------------------------------------------------------- gold-metadata-backfill


def test_gold_metadata_backfill_writes_proposal(tmp_path):
    course = _make_course_with_shapes(tmp_path)
    # add a question missing difficulty + population (the seed has neither).
    res = _invoke(tmp_path, "gold-metadata-backfill", "--course", _SLUG)
    assert res.exit_code == 0, res.output
    art = course / "retrieval_eval" / "gold_metadata_backfill_proposal.json"
    assert art.exists()
    doc = json.loads(art.read_text())
    assert doc["method"] == "rule-based"
    assert doc["n_questions"] >= 1
    # gold set is NOT mutated
    gold = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    assert "difficulty" not in gold["questions"][0]


def test_gold_metadata_backfill_unknown_course(tmp_path):
    res = _invoke(tmp_path, "gold-metadata-backfill", "--course", "no-such")
    assert res.exit_code == 1
    assert "course not found" in res.output


# --------------------------------------------------------------- probe-candidates


def test_probe_candidates_writes_artifact(tmp_path, monkeypatch):
    course = _make_course(tmp_path)
    _patch_client(monkeypatch, json.dumps([
        "How do graph databases store embeddings differently from vector stores?"]))
    # patch the retriever the CLI imports
    monkeypatch.setattr(
        "LibV2.tools.libv2.retriever.retrieve_chunks",
        lambda root, q, *, course_slug, engine, limit: (
            [_Result("c001", 0.1, "body")] if engine == "lexical" else []
        ),
    )
    res = _invoke(tmp_path, "probe-candidates", "--course", _SLUG)
    assert res.exit_code == 0, res.output
    art = course / "retrieval_eval" / "refusal_probe_candidates.json"
    assert art.exists()
    doc = json.loads(art.read_text())
    cats = doc["authoring_run"]["by_category"]
    assert cats.get("off_topic", 0) >= 1
    assert cats.get("out_of_scope_detail", 0) >= 1
    for p in doc["probe_candidates"]:
        assert len(p["dry_runs"]) == 3


# --------------------------------------------------------------- gold-promote


def _accept_candidates(course: Path, *, quote=_TEXT_B):
    """Write a candidates file with one accepted candidate for c002."""
    cand = {
        "question_id": "gqc-c002", "question_type": "factual_recall",
        "question_text": "what must fractions share before they can be added?",
        "difficulty": "easy",
        "relevant_passages": [{"chunk_id": "c002", "relevance": "primary",
            "anchor": {"item_path": "week_01/p.html", "text_quote": quote}}],
        "authoring": {"method": "llm_assisted", "author": "m/v1",
                      "reviewed_by": "@operator", "status": "reviewed"},
        "prescreen": {"passed": True, "reasons": []},
    }
    doc = {"schema_version": "1.1", "course_slug": _SLUG,
           "candidates": [cand]}
    (course / "retrieval_eval" / "gold_candidates.json").write_text(json.dumps(doc))


def test_gold_promote_merges_and_renumbers(tmp_path):
    course = _make_course(tmp_path)
    _accept_candidates(course)
    res = _invoke(tmp_path, "gold-promote", "--course", _SLUG)
    assert res.exit_code == 0, res.output
    assert "promoted 1" in res.output
    gold = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    qids = [q["question_id"] for q in gold["questions"]]
    assert qids == [f"gq-{_SLUG}-0001", f"gq-{_SLUG}-0002"]
    # validates clean afterwards
    res2 = _invoke(tmp_path, "gold-validate", "--course", _SLUG)
    assert res2.exit_code == 0, res2.output


def test_gold_promote_freeze(tmp_path):
    course = _make_course(tmp_path)
    _accept_candidates(course)
    res = _invoke(tmp_path, "gold-promote", "--course", _SLUG, "--freeze")
    assert res.exit_code == 0, res.output
    gold = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    assert gold["frozen"] is True
    assert "FROZEN" in res.output


def test_gold_promote_dry_run(tmp_path):
    course = _make_course(tmp_path)
    _accept_candidates(course)
    before = (course / "retrieval_eval" / "gold_set.json").read_text()
    res = _invoke(tmp_path, "gold-promote", "--course", _SLUG, "--dry-run")
    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output
    assert (course / "retrieval_eval" / "gold_set.json").read_text() == before


def test_gold_promote_refuses_on_frozen(tmp_path):
    course = _make_course(tmp_path, frozen=True)
    _accept_candidates(course)
    res = _invoke(tmp_path, "gold-promote", "--course", _SLUG)
    assert res.exit_code == 1
    assert "GOLD_SET_FROZEN" in res.output


def test_gold_promote_refuses_bad_quote(tmp_path):
    course = _make_course(tmp_path)
    _accept_candidates(course, quote="A quote that does not appear in chunk c002 anywhere.")
    res = _invoke(tmp_path, "gold-promote", "--course", _SLUG)
    assert res.exit_code == 0, res.output
    assert "refused" in res.output or "promoted 0" in res.output
    gold = json.loads((course / "retrieval_eval" / "gold_set.json").read_text())
    # nothing promoted
    assert len(gold["questions"]) == 1
