"""Difficulty-regrade proposal tests — synthetic, no live LLM.

Scope (easy-only), rubric grading via a FAKE client, the honest distribution
(no quota forcing), before/after projection + band report, parse-failure
recording, the decision-capture regression, and the end-to-end artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_difficulty_regrade import (
    DIFFICULTY_REGRADE_PROMPT_VERSION,
    build_regrade_rows,
    generate_difficulty_regrade_proposal,
    projected_distribution,
    regrade_question,
)


class FakeRegradeClient:
    """Scripted client. ``script`` maps a marker substring of the question text
    to a response; falls back to the default."""

    def __init__(self, by_marker=None, default=None, *, model="qwen2.5:7b-instruct-q4_K_M"):
        self.by_marker = by_marker or {}
        self.default = default
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=256, temperature=0.1, **kw):
        self.calls.append(messages)
        user = messages[-1]["content"]
        for marker, resp in self.by_marker.items():
            if marker in user:
                if isinstance(resp, BaseException):
                    raise resp
                return resp
        if isinstance(self.default, BaseException):
            raise self.default
        return self.default


class SpyCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.events.append({"decision_type": decision_type, "decision": decision,
                            "rationale": rationale, **kw})


def _chunk(cid, text, item_path):
    return {"id": cid, "text": text, "source": {"item_path": item_path},
            "learning_outcome_refs": [], "concept_tags": []}


def _q(qid, text, *, difficulty="easy", qtype="factual_recall", refs=("c001",)):
    return {
        "question_id": qid, "question_type": qtype, "question_text": text,
        "difficulty": difficulty,
        "relevant_passages": [
            {"chunk_id": c, "relevance": "primary",
             "anchor": {"item_path": "week_01/a.html", "text_quote": "x" * 40}}
            for c in refs
        ],
    }


def _grade(diff, rationale="because"):
    return json.dumps({"difficulty": diff, "rationale": rationale})


_CB = {"c001": _chunk("c001", "Body about absolute value.", "week_01/a.html"),
       "c002": _chunk("c002", "Body about LCM steps.", "week_02/b.html")}


# ------------------------------------------------------------- scope


def test_regrade_only_targets_easy():
    gold = {"course_slug": "d", "questions": [
        _q("gq-d-0001", "easy lookup question here?", difficulty="easy"),
        _q("gq-d-0002", "a hard one here?", difficulty="hard"),
    ]}
    rows = build_regrade_rows(
        gold, _CB, client=FakeRegradeClient(default=_grade("medium")))
    assert len(rows) == 1
    assert rows[0].question_id == "gq-d-0001"


# ------------------------------------------------------------- grading


def test_regrade_question_changed_and_unchanged():
    changed = regrade_question(
        _q("gq-d-0001", "synth across two weeks here?", refs=("c001", "c002")),
        _CB, client=FakeRegradeClient(default=_grade("hard")))
    assert changed.proposed == "hard" and changed.changed is True

    kept = regrade_question(
        _q("gq-d-0002", "simple lookup here?"),
        _CB, client=FakeRegradeClient(default=_grade("easy")))
    assert kept.proposed == "easy" and kept.changed is False  # honest: stays easy


def test_regrade_records_parse_failure():
    row = regrade_question(
        _q("gq-d-0003", "weird one here?"),
        _CB, client=FakeRegradeClient(default="not parseable"))
    assert row.error is not None
    assert row.proposed == row.current  # unchanged on failure


# ------------------------------------------------------------- distribution


def test_projected_distribution_applies_changes():
    gold = {"course_slug": "d", "questions": [
        _q("gq-d-0001", "q one here?", difficulty="easy"),
        _q("gq-d-0002", "q two here?", difficulty="easy"),
        _q("gq-d-0003", "q three here?", difficulty="hard"),
    ]}
    rows = build_regrade_rows(gold, _CB, client=FakeRegradeClient(
        by_marker={"q one": _grade("medium"), "q two": _grade("easy")}))
    before, after = projected_distribution(gold, rows)
    assert before == {"easy": 2, "medium": 0, "hard": 1}
    assert after == {"easy": 1, "medium": 1, "hard": 1}


# ------------------------------------------------------------- capture + e2e


def test_capture_fires_with_dynamic_rationale():
    gold = {"course_slug": "demo", "questions": [
        _q("gq-demo-0001", "easy q here?", difficulty="easy")]}
    cap = SpyCapture()
    build_regrade_rows(gold, _CB, client=FakeRegradeClient(default=_grade("medium")),
                       capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_difficulty_regrade"]
    assert len(ev) == 1
    assert DIFFICULTY_REGRADE_PROMPT_VERSION in ev[0]["rationale"]
    assert "before=" in ev[0]["rationale"] and "after=" in ev[0]["rationale"]
    assert len(ev[0]["rationale"]) >= 20


def test_generate_proposal_writes_artifact(tmp_path):
    slug = "demo-union-101"
    cdir = tmp_path / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in _CB.values():
            fh.write(json.dumps(c) + "\n")
    from lib.utils import sha256_file
    sha = sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1", "course_slug": slug,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha},
        "authored_at": "2026-06-12T00:00:00Z", "frozen": True,
        "questions": [
            _q(f"gq-{slug}-0001", "easy lookup here?", difficulty="easy"),
            _q(f"gq-{slug}-0002", "another easy here?", difficulty="easy"),
        ],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))

    doc, out_path = generate_difficulty_regrade_proposal(
        cdir, client=FakeRegradeClient(default=_grade("medium")), capture=SpyCapture())
    assert out_path is not None and out_path.exists()
    assert doc["n_regraded"] == 2
    assert doc["distribution"]["after"]["medium"] == 2
    assert "rubric" in doc
    assert "after_bands" in doc["distribution"]
    # never touches gold_set.json
    written = json.loads((cdir / "retrieval_eval" / "gold_set.json").read_text())
    assert written["questions"][0]["difficulty"] == "easy"
