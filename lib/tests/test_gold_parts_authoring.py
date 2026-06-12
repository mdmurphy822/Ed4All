"""multi_part parts[] authoring proposal tests — synthetic, no live LLM.

Decomposition + parent-passage binding, conservative covered:false +
absence_note when retrieval is empty, schema-legal part objects, the
decision-capture regression, and the end-to-end artifact. Includes a fake
retrieve_fn (the probe_authoring precedent).
"""
from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_parts_authoring import (
    PARTS_AUTHORING_PROMPT_VERSION,
    author_parts,
    build_parts_proposals,
    generate_parts_proposal,
    _parse_parts,
)


def test_parse_parts_degenerate_and_wrapped():
    # keys-as-labels degenerate object (7B-Q4 / json_mode artifact)
    deg = '{"part a": "Define the GCF", "part b": "Define the LCM"}'
    assert _parse_parts(deg) == ["Define the GCF", "Define the LCM"]
    # wrapped-array envelope
    assert _parse_parts('{"parts": [{"part_text": "A"}, {"part_text": "B"}]}') == ["A", "B"]
    # bare array of objects
    assert _parse_parts('[{"part_text": "A"}, {"part_text": "B"}]') == ["A", "B"]


class FakeClient:
    """Scripted client keyed by marker substring of the user message."""

    def __init__(self, by_marker=None, default=None, *, model="qwen2.5:7b-instruct-q4_K_M"):
        self.by_marker = by_marker or {}
        self.default = default
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=512, temperature=0.2, **kw):
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


class _Result:
    def __init__(self, chunk_id, score):
        self.chunk_id = chunk_id
        self.score = score
        self.text = ""


_BODY_A = ("The greatest common factor of two integers is the largest integer "
           "that divides both without a remainder.")
_BODY_B = ("The least common multiple of two integers is the smallest positive "
           "integer divisible by both.")


def _chunk(cid, text):
    return {"id": cid, "text": text, "source": {"item_path": "week_01/a.html"},
            "learning_outcome_refs": [], "concept_tags": []}


_CB = {"c001": _chunk("c001", _BODY_A), "c002": _chunk("c002", _BODY_B)}


def _mp_q(qid="gq-x-0001", refs=("c001", "c002")):
    return {
        "question_id": qid, "question_type": "multi_part",
        "question_text": "a) Define the greatest common factor. "
                         "b) Define the least common multiple.",
        "relevant_passages": [
            {"chunk_id": c, "relevance": "primary" if i == 0 else "supporting",
             "anchor": {"item_path": "week_01/a.html", "text_quote": "x" * 40}}
            for i, c in enumerate(refs)
        ],
        "parts": [
            {"part_id": "a", "part_text": "PART A — operator: fill in",
             "covered": True, "relevant_passage_refs": refs[:1]},
            {"part_id": "b", "part_text": "PART B — operator: fill in",
             "covered": True, "relevant_passage_refs": refs[:1]},
        ],
    }


def _decomp_json():
    return json.dumps([
        {"part_text": "Define the greatest common factor of two integers."},
        {"part_text": "Define the least common multiple of two integers."},
    ])


# ------------------------------------------------------------- decomposition + binding


def test_author_parts_decomposes_and_binds_to_parents():
    cands = author_parts(
        _mp_q(), _CB, client=FakeClient(default=_decomp_json()))
    assert cands.error is None
    assert len(cands.parts) == 2
    # both parts bind to a parent passage by 4-shingle overlap.
    for p in cands.parts:
        assert p["covered"] is True
        assert p["relevant_passage_refs"]  # non-empty
        # schema-legal keys only
        assert set(p.keys()) <= {"part_id", "part_text", "covered",
                                 "relevant_passage_refs"}
    # part a binds to the GCF chunk, part b to the LCM chunk.
    assert cands.parts[0]["relevant_passage_refs"] == ["c001"]
    assert cands.parts[1]["relevant_passage_refs"] == ["c002"]


def test_author_parts_uncovered_when_no_binding():
    """A part that overlaps neither a parent passage nor a retrieval hit is
    flagged covered:false with an absence_note (>= 20 chars)."""
    decomp = json.dumps([
        {"part_text": "Define the greatest common factor of two integers."},
        {"part_text": "Prove the Riemann hypothesis using contour integration."},
    ])
    note = json.dumps({"absence_note":
                       "The course corpus does not cover this sub-part; the dry-run confirmed it."})
    client = FakeClient(by_marker={"Riemann": note}, default=decomp)
    # retrieve_fn returns nothing eligible (empty-ish).
    def fake_retrieve(root, q, **kw):
        return []
    cands = author_parts(
        _mp_q(), _CB, client=client, retrieve_fn=fake_retrieve,
        libv2_root=Path("/x"), course_slug="x")
    covered = [p for p in cands.parts if p["covered"]]
    uncovered = [p for p in cands.parts if not p["covered"]]
    assert len(covered) == 1 and len(uncovered) == 1
    assert len(uncovered[0]["absence_note"]) >= 20
    assert set(uncovered[0].keys()) <= {"part_id", "part_text", "covered", "absence_note"}


def test_author_parts_records_decompose_failure():
    cands = author_parts(_mp_q(), _CB, client=FakeClient(default="not json"))
    assert cands.error is not None
    assert cands.parts == []


def test_author_parts_too_few_parts_errors():
    one = json.dumps([{"part_text": "only one part here"}])
    cands = author_parts(_mp_q(), _CB, client=FakeClient(default=one))
    assert cands.error is not None and "< 2" in cands.error


# ------------------------------------------------------------- capture + e2e


def test_capture_fires_with_dynamic_rationale():
    gold = {"course_slug": "demo", "questions": [_mp_q("gq-demo-0001")]}
    cap = SpyCapture()
    build_parts_proposals(
        gold, _CB, client=FakeClient(default=_decomp_json()), capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_parts_authoring"]
    assert len(ev) == 1
    assert PARTS_AUTHORING_PROMPT_VERSION in ev[0]["rationale"]
    assert "gq-demo-0001" in ev[0]["rationale"]
    assert len(ev[0]["rationale"]) >= 20


def test_only_multi_part_questions_in_scope():
    gold = {"course_slug": "demo", "questions": [
        _mp_q("gq-demo-0001"),
        {"question_id": "gq-demo-0002", "question_type": "factual_recall",
         "question_text": "not multi part?", "relevant_passages": []},
    ]}
    out = build_parts_proposals(gold, _CB, client=FakeClient(default=_decomp_json()))
    assert len(out) == 1 and out[0].question_id == "gq-demo-0001"


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
        "questions": [_mp_q(f"gq-{slug}-0001")],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))

    doc, out_path = generate_parts_proposal(
        cdir, client=FakeClient(default=_decomp_json()), capture=SpyCapture())
    assert out_path is not None and out_path.exists()
    assert doc["n_questions"] == 1
    prop = doc["proposals"][0]
    assert prop["n_parts"] == 2 and prop["n_covered"] == 2
    # never touches gold_set.json (placeholders intact)
    written = json.loads((cdir / "retrieval_eval" / "gold_set.json").read_text())
    assert "operator: fill in" in written["questions"][0]["parts"][0]["part_text"]
