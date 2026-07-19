"""retrieval-answer-eval-set P2 — refusal-probe candidate authoring tests.

off_topic bank determinism, out_of_scope_detail perturbation determinism +
gold-id linkage, adjacent_domain drafting via a FAKE local provider, the
per-engine dry-run evidence shape (fake retriever — no live index), and the
probe-authoring decision-capture regression. All synthetic; no network, no
Ollama, no vector index.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval.probe_authoring import (
    ILL_POSED_BANK,
    OFF_TOPIC_BANK,
    OOS_DETAIL_TEMPLATES,
    PROBE_AUTHORING_PROMPT_VERSION,
    adjacent_domain_candidates,
    attach_dry_runs,
    build_probe_candidates_doc,
    generate_probe_candidates,
    ill_posed_candidates,
    off_topic_candidates,
    off_topic_llm_candidates,
    out_of_scope_detail_candidates,
    run_dry_runs,
)


# --------------------------------------------------------------- doubles


class FakeProbeClient:
    def __init__(self, responses, *, model="qwen2.5:14b-instruct-q4_K_M"):
        self._responses = list(responses)
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=512, temperature=0.4, **kw):
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
        self.events.append({"decision_type": decision_type, "rationale": rationale})


class _Result:
    """Score-bearing retrieval result stand-in (.score / .chunk_id / .text)."""

    def __init__(self, chunk_id, score, text):
        self.chunk_id = chunk_id
        self.score = score
        self.text = text


def _fake_retrieve(scores_by_engine):
    """Build a retrieve_fn returning per-engine scripted results."""

    def _fn(libv2_root, query, *, course_slug, engine, limit):
        items = scores_by_engine.get(engine, [])
        return [_Result(cid, sc, txt) for (cid, sc, txt) in items][:limit]

    return _fn


def _chunk(cid, text, tags=()):
    return {"id": cid, "text": text, "concept_tags": list(tags),
            "source": {"item_path": f"week_01/{cid}.html"}}


# --------------------------------------------------------------- off_topic


def test_off_topic_deterministic_prefix():
    a = off_topic_candidates(8)
    b = off_topic_candidates(8)
    assert [c["question_text"] for c in a] == [c["question_text"] for c in b]
    assert all(c["category"] == "off_topic" for c in a)
    assert a[0]["question_text"] == OFF_TOPIC_BANK[0]


def test_off_topic_caps_at_bank_size():
    out = off_topic_candidates(999)
    assert len(out) == len(OFF_TOPIC_BANK)


def test_off_topic_no_llm_method_manual():
    out = off_topic_candidates(3)
    assert all(c["authoring"]["method"] == "manual" for c in out)


# --------------------------------------------------------------- oos detail


def _gold_questions():
    return [
        {"question_id": "gq-c-0002", "question_text": "What is a derivative?"},
        {"question_id": "gq-c-0001", "question_text": "How do fractions add together?"},
    ]


def test_oos_detail_deterministic_and_linked():
    a = out_of_scope_detail_candidates(_gold_questions(), 2)
    b = out_of_scope_detail_candidates(_gold_questions(), 2)
    assert a == b
    # id-sorted, so the first twin derives from gq-c-0001
    assert a[0]["derived_from_gold_question_id"] == "gq-c-0001"
    assert all(c["category"] == "out_of_scope_detail" for c in a)


def test_oos_detail_uses_templates():
    out = out_of_scope_detail_candidates(_gold_questions(), 2)
    # the perturbation template body appears in the produced text
    assert any(OOS_DETAIL_TEMPLATES[0].split("{q}")[1][:10] in c["question_text"]
               for c in out)


def test_oos_detail_caps_at_n():
    out = out_of_scope_detail_candidates(_gold_questions(), 1)
    assert len(out) == 1


# --------------------------------------------------------------- adjacent


def test_adjacent_domain_drafts_from_vocab():
    chunks = {"c001": _chunk("c001", "vector store text", tags=["vector-store", "embeddings"]),
              "c002": _chunk("c002", "more text", tags=["embeddings", "retrieval"])}
    resp = json.dumps(["How do graph databases index embeddings differently?",
                       "What is approximate nearest neighbour search complexity?"])
    cands = adjacent_domain_candidates(chunks, client=FakeProbeClient([resp]), n=2)
    assert len(cands) == 2
    assert all(c["category"] == "adjacent_domain" for c in cands)
    assert all(c["authoring"]["method"] == "llm_assisted" for c in cands)


def test_adjacent_domain_empty_vocab_no_draft():
    chunks = {"c001": _chunk("c001", "text", tags=[])}
    cands = adjacent_domain_candidates(chunks, client=FakeProbeClient(["[]"]), n=2)
    assert cands == []


def test_adjacent_domain_capture_fires():
    chunks = {"c001": _chunk("c001", "t", tags=["vector-store"])}
    resp = json.dumps(["A vocabulary-overlapping but uncovered question here?"])
    cap = SpyCapture()
    adjacent_domain_candidates(chunks, client=FakeProbeClient([resp]), n=1, capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "probe_candidate_authoring"]
    assert len(ev) == 1
    assert len(ev[0]["rationale"]) >= 20
    assert "vector-store" in ev[0]["rationale"]


def test_adjacent_domain_parse_failure_yields_empty():
    chunks = {"c001": _chunk("c001", "t", tags=["vector-store"])}
    cands = adjacent_domain_candidates(chunks, client=FakeProbeClient(["not json, no questions"]), n=2)
    assert cands == []


# --------------------------------------------------------------- off_topic_llm


def test_off_topic_llm_drafts_from_model():
    resp = json.dumps([
        "What temperature should you preheat an oven for sourdough bread?",
        "Which country has Canberra as its capital city?",
    ])
    cands = off_topic_llm_candidates(client=FakeProbeClient([resp]), n=2)
    assert len(cands) == 2
    assert all(c["category"] == "off_topic_llm" for c in cands)
    assert all(c["authoring"]["method"] == "llm_assisted" for c in cands)
    assert all(PROBE_AUTHORING_PROMPT_VERSION in c["authoring"]["author"] for c in cands)


def test_off_topic_llm_zero_n_no_draft():
    cands = off_topic_llm_candidates(client=FakeProbeClient(["[]"]), n=0)
    assert cands == []


def test_off_topic_llm_parse_failure_yields_empty():
    cands = off_topic_llm_candidates(
        client=FakeProbeClient(["no questions, just prose"]), n=3)
    assert cands == []


def test_off_topic_llm_capture_fires():
    resp = json.dumps(["Who directed the film that won best picture in 1994?"])
    cap = SpyCapture()
    off_topic_llm_candidates(client=FakeProbeClient([resp]), n=1, capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "probe_candidate_authoring"]
    assert len(ev) == 1
    assert len(ev[0]["rationale"]) >= 20
    assert "off-topic" in ev[0]["rationale"].lower()


# --------------------------------------------------------------- ill_posed


def _algebra_chunks():
    return {
        "c001": _chunk("c001", "least common multiple text",
                       tags=["least common multiple", "greatest common factor"]),
        "c002": _chunk("c002", "discriminant text",
                       tags=["discriminant", "quadratic equation"]),
    }


def test_ill_posed_shape_and_expected_outcome():
    out = ill_posed_candidates(_algebra_chunks(), 6)
    assert out, "algebra vocab should yield near-domain ill_posed probes"
    for c in out:
        assert c["category"] == "ill_posed"
        assert c["expected_outcome"] == "correct_premise"
        assert c["authoring"]["method"] == "manual"
        assert PROBE_AUTHORING_PROMPT_VERSION in c["authoring"]["author"]


def test_ill_posed_deterministic():
    a = ill_posed_candidates(_algebra_chunks(), 6)
    b = ill_posed_candidates(_algebra_chunks(), 6)
    assert [c["question_text"] for c in a] == [c["question_text"] for c in b]


def test_ill_posed_near_domain_gate_filters_off_domain():
    # A course whose vocabulary shares NO tokens with the algebra false-premise
    # bank yields no ill_posed probes (they would not be near-domain traps).
    off = {"c001": _chunk("c001", "text", tags=["photosynthesis", "chloroplast"])}
    assert ill_posed_candidates(off, 6) == []


def test_ill_posed_empty_vocab_uses_whole_bank():
    # No concept_tags => the whole bank is eligible (no near-domain gate).
    novocab = {"c001": _chunk("c001", "text", tags=[])}
    out = ill_posed_candidates(novocab, 3)
    assert len(out) == 3


def test_ill_posed_zero_n_empty():
    assert ill_posed_candidates(_algebra_chunks(), 0) == []


def test_ill_posed_bank_entries_carry_needles():
    for question, why, needles in ILL_POSED_BANK:
        assert question.endswith("?")
        assert len(why) >= 20
        assert needles and all(isinstance(t, str) for t in needles)


def test_generate_probe_candidates_includes_ill_posed(tmp_path):
    cdir = _write_course(tmp_path)
    # Give the course algebra vocabulary so the near-domain gate fires.
    import json as _json
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        fh.write(_json.dumps({"id": "c000", "text": "lcm",
                              "concept_tags": ["least common multiple",
                                               "greatest common factor",
                                               "discriminant", "quadratic equation"],
                              "source": {"item_path": "week_01/c000.html"}}) + "\n")
    from lib.utils import sha256_file
    sha = sha256_file(chunks_path)
    gold = _json.loads((cdir / "retrieval_eval" / "gold_set.json").read_text())
    gold["chunkset"]["chunks_sha256"] = sha
    (cdir / "retrieval_eval" / "gold_set.json").write_text(_json.dumps(gold))

    retrieve = _fake_retrieve({"lexical": [("c000", 0.1, "b")],
                               "semantic": [], "hybrid-rrf": []})
    doc, _ = generate_probe_candidates(
        cdir, client=FakeProbeClient(["[]"]), retrieve_fn=retrieve,
        libv2_root=tmp_path,
        targets={"off_topic": 0, "off_topic_llm": 0, "adjacent_domain": 0,
                 "out_of_scope_detail": 0, "ill_posed": 3},
    )
    ill = [p for p in doc["probe_candidates"] if p["category"] == "ill_posed"]
    assert len(ill) == 3
    assert all(p["expected_outcome"] == "correct_premise" for p in ill)
    assert all(len(p["dry_runs"]) == 3 for p in ill)
    assert doc["authoring_run"]["by_category"]["ill_posed"] == 3


# --------------------------------------------------------------- dry-runs


def test_run_dry_runs_per_engine_shape():
    retrieve = _fake_retrieve({
        "lexical": [("c001", 0.4, "lexical body")],
        "semantic": [("c002", 0.6, "semantic body")],
        "hybrid-rrf": [("c003", 0.5, "hybrid body")],
    })
    ev = run_dry_runs("some probe?", retrieve_fn=retrieve, libv2_root=Path("/x"),
                      course_slug="demo")
    assert [e.engine for e in ev] == ["lexical", "semantic", "hybrid-rrf"]
    d = ev[1].to_dict()
    assert d["engine"] == "semantic"
    assert d["top_chunk_id"] == "c002"
    assert d["n_results"] == 1
    assert d["top_score"] == 0.6
    assert "semantic body" in d["top_passage_excerpt"]
    # top_passage_answers left null for the operator (R5)
    assert d["top_passage_answers"] is None


def test_run_dry_runs_missing_index_records_zero():
    def _raises(libv2_root, query, *, course_slug, engine, limit):
        if engine == "semantic":
            raise RuntimeError("no index")
        return [_Result("c001", 0.3, "body")]

    ev = run_dry_runs("q?", retrieve_fn=_raises, libv2_root=Path("/x"),
                      course_slug="demo")
    sem = next(e for e in ev if e.engine == "semantic")
    assert sem.n_results == 0
    assert sem.top_score == 0.0
    # lexical still recorded
    lex = next(e for e in ev if e.engine == "lexical")
    assert lex.n_results == 1


def test_attach_dry_runs_adds_block():
    retrieve = _fake_retrieve({"lexical": [("c001", 0.2, "b")],
                               "semantic": [], "hybrid-rrf": []})
    cands = off_topic_candidates(2)
    out = attach_dry_runs(cands, retrieve_fn=retrieve, libv2_root=Path("/x"),
                          course_slug="demo")
    assert all(len(c["dry_runs"]) == 3 for c in out)


# --------------------------------------------------------------- end-to-end


def _write_course(tmp_path: Path, slug="demo-union-101"):
    cdir = tmp_path / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    chunks = {f"c{i:03d}": _chunk(f"c{i:03d}", f"body {i}",
                                  tags=["vector-store", "embeddings"]) for i in range(5)}
    chunks_path = cdir / "corpus" / "chunks.jsonl"
    with chunks_path.open("w") as fh:
        for c in chunks.values():
            fh.write(json.dumps(c) + "\n")
    from lib.utils import sha256_file
    sha = sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1", "course_slug": slug,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha},
        "authored_at": "2026-06-11T00:00:00Z", "frozen": False,
        "questions": [{
            "question_id": f"gq-{slug}-0001", "question_type": "factual_recall",
            "question_text": "what does a vector store index here?",
            "relevant_passages": [{"chunk_id": "c000", "relevance": "primary",
                "anchor": {"item_path": "week_01/c000.html", "text_quote": "body 0"}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    return cdir


def test_generate_probe_candidates_end_to_end(tmp_path):
    cdir = _write_course(tmp_path)
    client = FakeProbeClient([json.dumps([
        "How do graph databases differ in embedding storage strategy?",
        "What is the recall curve for product quantization at scale?",
    ])])
    retrieve = _fake_retrieve({"lexical": [("c001", 0.1, "b")],
                               "semantic": [], "hybrid-rrf": []})
    cap = SpyCapture()
    doc, out_path = generate_probe_candidates(
        cdir, client=client, retrieve_fn=retrieve, libv2_root=tmp_path,
        targets={"off_topic": 3, "off_topic_llm": 2,
                 "adjacent_domain": 2, "out_of_scope_detail": 1},
        capture=cap,
    )
    assert out_path and out_path.exists()
    by_cat = doc["authoring_run"]["by_category"]
    assert by_cat["off_topic"] == 3
    assert by_cat["off_topic_llm"] == 2
    assert by_cat["adjacent_domain"] == 2
    assert by_cat["out_of_scope_detail"] == 1
    # every probe candidate carries per-engine dry-runs + a probe_id
    for p in doc["probe_candidates"]:
        assert len(p["dry_runs"]) == 3
        assert p["probe_id"].startswith("rpc-")
    # capture fired for the adjacent-domain drafting
    assert any(e["decision_type"] == "probe_candidate_authoring" for e in cap.events)


def test_parse_adjacent_degenerate_keys_as_questions_shape():
    """Observed live with a 7B-Q4: a valid-JSON OBJECT whose keys are the
    escaped question strings (with a stray quote-echo before the final mark)
    and empty values. The parser harvests genuine questions and rejects
    statement-shaped drafts (no trailing '?')."""
    from lib.retrieval.probe_authoring import _parse_adjacent
    degenerate = (
        '{"\\"What is the prime factorization of the fraction 36/48?\\"?":"",'
        ' "\\"Convert the decimal 0.75 to an integer without rounding.\\"?":"",'
        ' "\\"How do eigenvalues relate to systems of equations?\\"?":""}'
    )
    parsed = _parse_adjacent(degenerate)
    assert "What is the prime factorization of the fraction 36/48?" in parsed
    assert "How do eigenvalues relate to systems of equations?" in parsed
    assert all(q.endswith("?") for q in parsed)
    assert not any("Convert the decimal" in q for q in parsed)


def test_parse_adjacent_fenced_and_object_shapes():
    from lib.retrieval.probe_authoring import _parse_adjacent
    assert _parse_adjacent('```json\n["What is a derivative of a function?"]\n```') == [
        "What is a derivative of a function?"
    ]
    assert _parse_adjacent('{"questions": ["What is linear regression used for?"]}') == [
        "What is linear regression used for?"
    ]
