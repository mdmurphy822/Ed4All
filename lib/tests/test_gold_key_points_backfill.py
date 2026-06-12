"""expected_key_points backfill proposal tests — synthetic, no live LLM.

Selection (non-refusal + < 2 key points), grounded drafting via a FAKE local
client, parse-failure recording, the decision-capture regression, and the
end-to-end proposal artifact. No course slugs, no network, no Ollama.
"""
from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_key_points_backfill import (
    KEY_POINTS_BACKFILL_PROMPT_VERSION,
    build_key_points_proposals,
    draft_key_points,
    generate_key_points_proposal,
    _needs_key_points,
    _parse_key_points,
)


def test_parse_degenerate_keys_as_labels_object():
    """The 7B-Q4 / json_mode envelope re-emits the example array as object KEYS;
    the real points are the VALUES. Harvest them (the observed live artifact)."""
    deg = ('{"key point one": "Absolute value is distance from zero", '
           '"key point two": "always non-negative"}')
    assert _parse_key_points(deg) == [
        "Absolute value is distance from zero", "always non-negative"]


def test_parse_wrapped_array_envelope():
    assert _parse_key_points('{"points": ["a", "b"]}') == ["a", "b"]
    assert _parse_key_points('["a", "b", "c"]') == ["a", "b", "c"]


class FakeDraftClient:
    def __init__(self, responses, *, model="qwen2.5:7b-instruct-q4_K_M"):
        self._responses = list(responses)
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=384, temperature=0.2, **kw):
        idx = len(self.calls)
        self.calls.append({"messages": messages})
        resp = self._responses[min(idx, len(self._responses) - 1)]
        if isinstance(resp, BaseException):
            raise resp
        return resp


class SpyCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.events.append({"decision_type": decision_type, "decision": decision,
                            "rationale": rationale, **kw})


_BODY = ("Absolute value is the distance of a number from zero on the number "
         "line, and is therefore always non-negative.")


def _chunk(cid="c001"):
    return {"id": cid, "text": _BODY, "source": {"item_path": "week_01/a.html"},
            "learning_outcome_refs": [], "concept_tags": []}


def _q(qid="gq-x-0001", *, kps=None, qtype="factual_recall", category=None,
       refs=("c001",)):
    q = {
        "question_id": qid, "question_type": qtype,
        "question_text": "What is the absolute value of a number?",
        "relevant_passages": [
            {"chunk_id": c, "relevance": "primary",
             "anchor": {"item_path": "week_01/a.html", "text_quote": _BODY[:50]}}
            for c in refs
        ],
    }
    if kps is not None:
        q["expected_key_points"] = list(kps)
    if category is not None:
        q["category"] = category
    return q


def _kp_json(points=("distance from zero", "always non-negative")):
    return json.dumps(list(points))


# ------------------------------------------------------------- selection


def test_needs_key_points_selects_missing_non_refusal():
    assert _needs_key_points(_q(kps=None)) is True
    assert _needs_key_points(_q(kps=["one"])) is True  # < 2
    assert _needs_key_points(_q(kps=["one", "two"])) is False
    # refusal-style is exempt even with no key points.
    assert _needs_key_points(_q(kps=None, category="off_topic")) is False


# ------------------------------------------------------------- drafting


def test_draft_key_points_grounded_shape():
    points, rationale, error = draft_key_points(
        _q(), {"c001": _chunk()}, client=FakeDraftClient([_kp_json()]))
    assert error is None
    assert 2 <= len(points) <= 4
    assert "c001" in rationale


def test_draft_key_points_records_parse_failure():
    points, rationale, error = draft_key_points(
        _q(), {"c001": _chunk()}, client=FakeDraftClient(["nope"]))
    assert error is not None
    assert points == []


def test_draft_key_points_too_few_flagged():
    points, rationale, error = draft_key_points(
        _q(), {"c001": _chunk()}, client=FakeDraftClient([json.dumps(["only one"])]))
    assert error is not None and "< 2" in error


def test_draft_key_points_no_passages_errors():
    points, rationale, error = draft_key_points(
        _q(refs=()), {}, client=FakeDraftClient([_kp_json()]))
    assert error is not None and "no resolvable" in error


# ------------------------------------------------------------- capture + e2e


def test_capture_fires_with_dynamic_rationale():
    gold = {"course_slug": "demo", "questions": [_q("gq-demo-0001", kps=None)]}
    cap = SpyCapture()
    build_key_points_proposals(
        gold, {"c001": _chunk()}, client=FakeDraftClient([_kp_json()]), capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_key_points_backfill"]
    assert len(ev) == 1
    assert KEY_POINTS_BACKFILL_PROMPT_VERSION in ev[0]["rationale"]
    assert "gq-demo-0001" in ev[0]["rationale"]
    assert len(ev[0]["rationale"]) >= 20


def test_generate_proposal_writes_artifact(tmp_path):
    slug = "demo-union-101"
    cdir = tmp_path / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    chunks_path.write_text(json.dumps(_chunk()) + "\n")
    from lib.utils import sha256_file
    sha = sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1", "course_slug": slug,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha},
        "authored_at": "2026-06-12T00:00:00Z", "frozen": True,
        "questions": [
            _q("gq-demo-union-101-0001", kps=None),
            _q("gq-demo-union-101-0002", kps=["a", "b"]),  # already has => skipped
        ],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))

    doc, out_path = generate_key_points_proposal(
        cdir, client=FakeDraftClient([_kp_json()]), capture=SpyCapture())
    assert out_path is not None and out_path.exists()
    assert doc["n_questions"] == 1  # only the missing one
    assert doc["proposals"][0]["question_id"] == "gq-demo-union-101-0001"
    assert 2 <= len(doc["proposals"][0]["proposed_key_points"]) <= 4
    # never touches gold_set.json
    written = json.loads((cdir / "retrieval_eval" / "gold_set.json").read_text())
    assert "expected_key_points" not in written["questions"][0]
