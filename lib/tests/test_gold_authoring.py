"""retrieval-answer-eval-set P2 — gold-candidate authoring tests.

Sampler determinism + quota allocation, the pre-screen reject matrix, drafting
via a FAKE local provider (no live LLM — follows the FakeAnswerClient
precedent), the decision-capture regression, and the promote merge/renumber/
refuse paths. All synthetic; no real course data, no network, no Ollama.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from lib.retrieval.gold_authoring import (
    BOTH_POPULATION_TEMPLATE,
    DEFINITION_TEMPLATE,
    GOLD_AUTHORING_PROMPT_VERSION,
    WORKED_EXAMPLE_TEMPLATE,
    GoldPromoteError,
    PrescreenVerdict,
    build_candidates_doc,
    draft_both_population_candidates,
    draft_candidates,
    draft_definition_candidates,
    draft_worked_example_candidates,
    generate_gold_candidates,
    mine_both_population_pairs,
    mine_glossary_terms,
    mine_worked_example_chunks,
    prescreen_candidate,
    prescreen_candidates,
    promote_candidates_into_gold,
    sample_chunks,
    _build_type_plan,
    _difficulty_heuristic,
)
from lib.retrieval.gold_set import chunk_content_sha256


# --------------------------------------------------------------- doubles


class FakeDraftClient:
    """Duck-typed ``chat_completion`` returning scripted JSON drafts.

    Mirrors the FakeAnswerClient precedent. ``script`` maps a chunk's text
    substring marker -> the JSON string to return, OR a single default string.
    """

    def __init__(self, responses, *, model="qwen2.5:7b-instruct-q4_K_M"):
        self._responses = list(responses)
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=512, temperature=0.2, **kw):
        idx = len(self.calls)
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        resp = self._responses[min(idx, len(self._responses) - 1)]
        if isinstance(resp, BaseException):
            raise resp
        return resp


class SpyCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.events.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale, **kw}
        )


_TEXT_A = "A vector store indexes high-dimensional embedding vectors for fast retrieval."
_TEXT_B = "To add or subtract fractions, they must first share a common denominator value."
_TEXT_C = "The chain rule lets you differentiate a composition of two functions cleanly."


def _chunk(cid, text, *, item_path, los=(), tags=(), role="explanation",
           source_refs=None, chunk_type=None, section_heading=None):
    src = {"item_path": item_path}
    if source_refs:
        src["source_references"] = source_refs
    if section_heading:
        src["section_heading"] = section_heading
    chunk = {
        "id": cid, "text": text, "source": src,
        "learning_outcome_refs": list(los), "concept_tags": list(tags),
        "teaching_role": role,
    }
    if chunk_type:
        chunk["chunk_type"] = chunk_type
    return chunk


def _draft_json(q="What does a vector store index for retrieval?",
                quote=_TEXT_A, kps=("indexes embeddings", "for retrieval")):
    return json.dumps({"question_text": q, "expected_key_points": list(kps),
                       "quote": quote})


def _synthetic_chunkset(n=24):
    """A union-ish chunkset across weeks/populations/roles for the sampler."""
    chunks = {}
    roles = ["explanation", "example", "procedure"]
    for i in range(n):
        wk = (i % 6) + 1
        if i % 3 == 0:
            # source population (textbook + source_references)
            ch = _chunk(f"c{i:03d}", f"{_TEXT_A} Variation {i}.",
                        item_path="textbook.html", los=["to-01", "co-01"],
                        tags=["vector-store", "embeddings"], role=roles[i % 3],
                        source_refs=[{"sourceId": f"dart:tb#{i}"}])
        else:
            ch = _chunk(f"c{i:03d}", f"{_TEXT_B} Variation {i}.",
                        item_path=f"week_{wk:02d}/p.html",
                        los=[f"co-{wk:02d}"], tags=["fractions", "denominator"],
                        role=roles[i % 3])
        chunks[ch["id"]] = ch
    return chunks


# --------------------------------------------------------------- sampler


def test_type_plan_matches_taxonomy_at_50():
    plan = _build_type_plan(50)
    counts = Counter(plan)
    assert counts == {"factual_recall": 15, "procedural": 10,
                      "conceptual_synthesis": 12, "multi_part": 8,
                      "where_covered": 5}


def test_type_plan_scales_and_sums_exactly():
    for n in (10, 37, 50, 100):
        plan = _build_type_plan(n)
        assert len(plan) == n


def test_sampler_is_deterministic():
    chunks = _synthetic_chunkset(24)
    a = sample_chunks(chunks, n=12, seed=7)
    b = sample_chunks(chunks, n=12, seed=7)
    assert [s.chunk_id for s in a] == [s.chunk_id for s in b]
    assert [s.question_type for s in a] == [s.question_type for s in b]


def test_sampler_seed_changes_order():
    chunks = _synthetic_chunkset(24)
    a = sample_chunks(chunks, n=12, seed=0)
    b = sample_chunks(chunks, n=12, seed=5)
    # Different rotation should change at least the leading slot ordering.
    assert [s.chunk_id for s in a] != [s.chunk_id for s in b]


def test_sampler_no_duplicate_chunks():
    chunks = _synthetic_chunkset(24)
    slots = sample_chunks(chunks, n=24, seed=0)
    ids = [s.chunk_id for s in slots]
    assert len(ids) == len(set(ids))


def test_sampler_spreads_population_on_union():
    chunks = _synthetic_chunkset(30)
    slots = sample_chunks(chunks, n=20, seed=0, is_union=True)
    pops = Counter(s.population for s in slots)
    # both source + course populations represented (quota bias pulls source in).
    assert pops.get("source", 0) >= 1
    assert pops.get("course", 0) >= 1


def test_sampler_caps_at_chunk_count():
    chunks = _synthetic_chunkset(6)
    slots = sample_chunks(chunks, n=50, seed=0)
    assert len(slots) == 6


# --------------------------------------------------------------- difficulty


def test_difficulty_heuristic():
    assert _difficulty_heuristic("factual_recall", 1, 1) == "easy"
    assert _difficulty_heuristic("conceptual_synthesis", 2, 1) == "medium"
    assert _difficulty_heuristic("conceptual_synthesis", 3, 1) == "hard"
    assert _difficulty_heuristic("multi_part", 1, 1) == "hard"
    assert _difficulty_heuristic("factual_recall", 1, 2) == "hard"  # across weeks


# --------------------------------------------------------------- drafting


def test_draft_candidates_builds_v1_1_shape():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html",
                             los=["to-01", "co-01"])}
    slots = sample_chunks(chunks, n=1, seed=0)
    client = FakeDraftClient([_draft_json()])
    cands = draft_candidates(slots, chunks, client=client)
    assert len(cands) == 1
    c = cands[0]
    assert c["question_type"] in ("factual_recall", "procedural",
                                  "conceptual_synthesis", "multi_part", "where_covered")
    p = c["relevant_passages"][0]
    assert p["chunk_id"] == "c001"
    assert p["relevance"] == "primary"
    assert p["anchor"]["text_quote"] == _TEXT_A
    assert p["anchor"]["content_sha256"] == chunk_content_sha256(chunks["c001"])
    assert c["objective_refs"] == ["co-01", "to-01"]
    assert c["authoring"]["method"] == "llm_assisted"
    assert GOLD_AUTHORING_PROMPT_VERSION in c["authoring"]["author"]
    assert c["authoring"]["status"] == "draft"
    assert 2 <= len(c["expected_key_points"]) <= 4


def test_draft_records_parse_failure_not_dropped():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    client = FakeDraftClient(["this is not json at all"])
    cands = draft_candidates(slots, chunks, client=client)
    assert len(cands) == 1
    assert "draft_error" in cands[0]


def test_draft_capture_fires_with_dynamic_rationale():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    client = FakeDraftClient([_draft_json()])
    cap = SpyCapture()
    draft_candidates(slots, chunks, client=client, capture=cap)
    auth_events = [e for e in cap.events
                   if e["decision_type"] == "gold_candidate_authoring"]
    assert len(auth_events) == 1
    ev = auth_events[0]
    assert len(ev["rationale"]) >= 20
    # dynamic signals interpolated
    assert "c001" in ev["rationale"]
    assert client.model in ev["rationale"]


# --------------------------------------------------------------- prescreen


def test_prescreen_accepts_clean_candidate():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([_draft_json()]))[0]
    v = prescreen_candidate(cand, chunks)
    assert v.passed, v.reasons


def test_prescreen_rejects_quote_not_contained():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    bad = _draft_json(quote="This sentence is nowhere in the source chunk text at all.")
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([bad]))[0]
    v = prescreen_candidate(cand, chunks)
    assert not v.passed
    assert "QUOTE_NOT_IN_CHUNK" in v.reasons


def test_prescreen_rejects_short_quote():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([_draft_json(quote="short")]))[0]
    v = prescreen_candidate(cand, chunks)
    assert "QUOTE_TOO_SHORT" in v.reasons


def test_prescreen_rejects_short_question():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([_draft_json(q="why?")]))[0]
    v = prescreen_candidate(cand, chunks)
    assert "QUESTION_TOO_SHORT" in v.reasons


def test_prescreen_rejects_ambiguous_quote():
    # The shared sentence appears in 4 chunks => ambiguous (>3).
    shared = "This shared sentence appears verbatim in several different chunks here."
    chunks = {f"c{i:03d}": _chunk(f"c{i:03d}", f"{shared} Item {i}.",
                                 item_path=f"week_0{i}/p.html") for i in range(4)}
    slots = sample_chunks(chunks, n=1, seed=0)
    cid = slots[0].chunk_id
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([_draft_json(quote=shared)]))[0]
    v = prescreen_candidate(cand, chunks)
    assert "QUOTE_AMBIGUOUS" in v.reasons


def test_prescreen_rejects_near_duplicate():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html")}
    slots = sample_chunks(chunks, n=1, seed=0)
    q = "What does a vector store index for retrieval in this system?"
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([_draft_json(q=q)]))[0]
    v = prescreen_candidate(cand, chunks, prior_questions=[q])
    assert "NEAR_DUPLICATE" in v.reasons


def test_prescreen_batch_accumulates_near_dups():
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html"),
              "c002": _chunk("c002", _TEXT_A, item_path="week_02/b.html")}
    same_q = "What does a vector store index for retrieval purposes here?"
    cands = [
        draft_candidates([s], chunks, client=FakeDraftClient([_draft_json(q=same_q)]))[0]
        for s in sample_chunks(chunks, n=2, seed=0)
    ]
    screened = prescreen_candidates(cands, chunks)
    # second occurrence of the same question flags near-dup
    flags = [v.passed for _, v in screened]
    assert flags[0] is True
    assert flags[1] is False


# --------------------------------------------------------------- promote


def _gold_doc(slug, chunks_sha, *, questions=None, frozen=False):
    return {
        "schema_version": "1.1",
        "course_slug": slug,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": chunks_sha},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": frozen,
        "questions": questions or [{
            "question_id": f"gq-{slug}-0001", "question_type": "factual_recall",
            "question_text": "what does a vector store index?",
            "relevant_passages": [{"chunk_id": "c001", "relevance": "primary",
                "anchor": {"item_path": "week_01/a.html", "text_quote": _TEXT_A}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }


def _accepted_candidate(slug, chunks, chunk_id="c002", quote=_TEXT_B):
    slots = [s for s in sample_chunks(chunks, n=len(chunks), seed=0)
             if s.chunk_id == chunk_id]
    cand = draft_candidates(slots, chunks, client=FakeDraftClient([
        _draft_json(q="what must fractions share before adding them?", quote=quote)
    ]))[0]
    cand["question_type"] = "factual_recall"
    cand["authoring"]["status"] = "reviewed"
    cand["authoring"]["reviewed_by"] = "@operator"
    return cand


def test_promote_merges_and_renumbers():
    slug = "demo-union-101"
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html"),
              "c002": _chunk("c002", _TEXT_B, item_path="textbook.html")}
    import hashlib
    sha = hashlib.sha256(b"x").hexdigest()
    gold = _gold_doc(slug, sha)
    cand = _accepted_candidate(slug, chunks)
    cand_doc = {"candidates": [{**cand, "prescreen": {"passed": True, "reasons": []}}]}
    new_gold, report = promote_candidates_into_gold(
        gold, cand_doc, chunks, sha, course_slug=slug)
    assert report.promoted_ids == [f"gq-{slug}-0002"]
    qids = [q["question_id"] for q in new_gold["questions"]]
    assert qids == [f"gq-{slug}-0001", f"gq-{slug}-0002"]
    # candidate-only fields stripped
    assert "prescreen" not in new_gold["questions"][1]


def test_promote_skips_unaccepted():
    slug = "demo-union-101"
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html"),
              "c002": _chunk("c002", _TEXT_B, item_path="textbook.html")}
    import hashlib
    sha = hashlib.sha256(b"x").hexdigest()
    gold = _gold_doc(slug, sha)
    cand = _accepted_candidate(slug, chunks)
    cand["authoring"]["status"] = "draft"  # not accepted
    cand_doc = {"candidates": [{**cand, "prescreen": {"passed": True}}]}
    new_gold, report = promote_candidates_into_gold(
        gold, cand_doc, chunks, sha, course_slug=slug)
    assert report.promoted_ids == []
    assert report.skipped and report.skipped[0]["reason"] == "not_accepted"


def test_promote_refuses_failing_prescreen():
    slug = "demo-union-101"
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html"),
              "c002": _chunk("c002", _TEXT_B, item_path="textbook.html")}
    import hashlib
    sha = hashlib.sha256(b"x").hexdigest()
    gold = _gold_doc(slug, sha)
    # accepted but the (edited) quote is no longer contained in its chunk
    cand = _accepted_candidate(slug, chunks, quote="A quote not present anywhere in the chunk text here.")
    cand_doc = {"candidates": [{**cand, "prescreen": {"passed": True}}]}
    new_gold, report = promote_candidates_into_gold(
        gold, cand_doc, chunks, sha, course_slug=slug)
    assert report.promoted_ids == []
    assert report.refused and report.refused[0]["reason"] == "prescreen_failed"


def test_promote_freeze_sets_frozen():
    slug = "demo-union-101"
    chunks = {"c001": _chunk("c001", _TEXT_A, item_path="week_01/a.html"),
              "c002": _chunk("c002", _TEXT_B, item_path="textbook.html")}
    import hashlib
    sha = hashlib.sha256(b"x").hexdigest()
    gold = _gold_doc(slug, sha)
    cand = _accepted_candidate(slug, chunks)
    cand_doc = {"candidates": [{**cand, "prescreen": {"passed": True}}]}
    new_gold, report = promote_candidates_into_gold(
        gold, cand_doc, chunks, sha, course_slug=slug, freeze=True)
    assert new_gold["frozen"] is True
    assert report.frozen is True


# --------------------------------------------------------------- end-to-end


def _write_union_course(tmp_path: Path, slug="demo-union-101"):
    import hashlib
    from lib.retrieval._text import normalize_ws
    cdir = tmp_path / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    chunks = _synthetic_chunkset(18)
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
            "question_text": "seed question about vector stores here?",
            "relevant_passages": [{"chunk_id": "c000", "relevance": "primary",
                "anchor": {"item_path": "textbook.html", "text_quote": chunks["c000"]["text"]}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))
    return cdir, chunks


def test_generate_gold_candidates_end_to_end(tmp_path):
    cdir, chunks = _write_union_course(tmp_path)
    # one good draft repeated; quote pulled from c000's text so it contains.
    client = FakeDraftClient([_draft_json(
        q="Explain what this passage describes about the topic in detail please?",
        quote=chunks["c000"]["text"])])
    cap = SpyCapture()
    doc, out_path = generate_gold_candidates(cdir, client=client, n=8, seed=0, capture=cap)
    assert out_path and out_path.exists()
    assert doc["authoring_run"]["n_drafted"] == 8
    assert doc["schema_version"] == "1.1"
    # capture fired
    assert any(e["decision_type"] == "gold_candidate_authoring" for e in cap.events)
    # every candidate carries a prescreen verdict
    assert all("prescreen" in c for c in doc["candidates"])


# --------------------------------------------------------------- definition arm


# A synthetic glossary chunk: "Term Definition. Term Definition." layout with
# >= 4 pairs, marked chunk_type=summary (the structural glossary signal).
_GLOSSARY_TEXT = (
    "Integer Any whole number, its negative counterpart, or zero. Fractions "
    "and decimals are not integers. Absolute Value The distance between a "
    "number and zero on the number line, always non-negative. Opposite The "
    "number the same distance from zero but on the other side of zero. Number "
    "Line A horizontal line on which every real number has a unique position. "
    "Signed Number Any number that may be positive or negative in arithmetic."
)


def _glossary_chunk(cid="g001", **kw):
    kw.setdefault("item_path", "week_03/summary.html")
    kw.setdefault("chunk_type", "summary")
    kw.setdefault("section_heading", "Week 3 Summary")
    return _chunk(cid, _GLOSSARY_TEXT, **kw)


def test_mine_glossary_terms_extracts_pairs():
    chunks = {"g001": _glossary_chunk()}
    seeds = mine_glossary_terms(chunks)
    terms = {s.term for s in seeds}
    assert "Integer" in terms or "Absolute Value" in terms
    assert all(s.chunk_id == "g001" for s in seeds)
    assert all(len(s.definition) >= 25 for s in seeds)


def test_mine_glossary_terms_deterministic():
    chunks = {"g001": _glossary_chunk()}
    a = mine_glossary_terms(chunks)
    b = mine_glossary_terms(chunks)
    assert [(s.term, s.definition) for s in a] == [(s.term, s.definition) for s in b]


def test_mine_glossary_terms_skips_non_glossary():
    # A worked-example / prose chunk (no summary marker) must be skipped even if
    # it packs Title-Case phrases — the structural gate rejects it.
    chunks = {"x001": _chunk("x001", _GLOSSARY_TEXT, item_path="week_01/a.html",
                             chunk_type="explanation")}
    seeds = mine_glossary_terms(chunks)
    assert seeds == []


def test_draft_definition_candidates_shape_and_metadata():
    chunks = {"g001": _glossary_chunk(los=["to-03", "co-07"],
                                      source_refs=[{"sourceId": "dart:tb#1"}])}
    seeds = mine_glossary_terms(chunks)[:1]
    quote = "The distance between a number and zero on the number line, always non-negative"
    client = FakeDraftClient([_draft_json(
        q="What is the definition of absolute value?", quote=quote,
        kps=("distance from zero", "always non-negative"))])
    cands = draft_definition_candidates(seeds, chunks, client=client)
    assert len(cands) == 1
    c = cands[0]
    assert c["question_type"] == "factual_recall"
    assert c["authoring"]["template"] == DEFINITION_TEMPLATE
    assert c["authoring"]["method"] == "llm_assisted"
    # full metadata: difficulty + population + key points + objective_refs.
    assert c["difficulty"] in ("easy", "medium", "hard")
    assert c["expected_citation_population"] == "source"  # source_refs => source
    assert 2 <= len(c["expected_key_points"]) <= 4
    assert c["objective_refs"] == ["co-07", "to-03"]
    p = c["relevant_passages"][0]
    assert p["chunk_id"] == "g001" and p["relevance"] == "primary"


def test_draft_definition_capture_fires_dynamic():
    chunks = {"g001": _glossary_chunk()}
    seeds = mine_glossary_terms(chunks)[:1]
    quote = "The distance between a number and zero on the number line, always non-negative"
    cap = SpyCapture()
    draft_definition_candidates(
        seeds, chunks, client=FakeDraftClient([_draft_json(quote=quote)]), capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_candidate_authoring"]
    assert len(ev) == 1
    assert len(ev[0]["rationale"]) >= 20
    assert DEFINITION_TEMPLATE in ev[0]["rationale"]
    assert "g001" in ev[0]["rationale"]


def test_draft_definition_records_parse_failure():
    chunks = {"g001": _glossary_chunk()}
    seeds = mine_glossary_terms(chunks)[:1]
    cands = draft_definition_candidates(
        seeds, chunks, client=FakeDraftClient(["not json"]))
    assert cands and "draft_error" in cands[0]
    assert cands[0]["authoring"]["template"] == DEFINITION_TEMPLATE


# --------------------------------------------------------------- worked-example arm


_WORKED_TEXT = (
    "Problem 1 — Event Scheduling. The chess club meets every 6 days. On what "
    "day will they next meet? Solution. We need the least common multiple of 6 "
    "and 8. Step 1: list multiples. Step 2: find the smallest common one."
)


def test_mine_worked_example_chunks_matches_problem_solution():
    chunks = {
        "w001": _chunk("w001", _WORKED_TEXT, item_path="week_01/work.html"),
        "p001": _chunk("p001", _TEXT_A, item_path="week_01/a.html"),
    }
    ids = mine_worked_example_chunks(chunks)
    assert "w001" in ids
    assert "p001" not in ids  # no Problem/Solution structure


def test_draft_worked_example_candidates_shape_and_metadata():
    chunks = {"w001": _chunk("w001", _WORKED_TEXT, item_path="week_01/work.html",
                             los=["to-01"])}
    ids = mine_worked_example_chunks(chunks)
    quote = "We need the least common multiple of 6 and 8. Step 1: list multiples."
    client = FakeDraftClient([_draft_json(
        q="How do you solve the event scheduling problem step by step?",
        quote=quote, kps=("find the LCM", "list multiples"))])
    cands = draft_worked_example_candidates(ids, chunks, client=client)
    assert len(cands) == 1
    c = cands[0]
    assert c["question_type"] == "procedural"
    assert c["authoring"]["template"] == WORKED_EXAMPLE_TEMPLATE
    assert c["difficulty"] in ("easy", "medium", "hard")
    # course population (week_NN path, no source_refs)
    assert c["expected_citation_population"] == "course"
    assert 2 <= len(c["expected_key_points"]) <= 4


def test_draft_worked_example_capture_fires_dynamic():
    chunks = {"w001": _chunk("w001", _WORKED_TEXT, item_path="week_01/work.html")}
    ids = mine_worked_example_chunks(chunks)
    quote = "We need the least common multiple of 6 and 8. Step 1: list multiples."
    cap = SpyCapture()
    draft_worked_example_candidates(
        ids, chunks, client=FakeDraftClient([_draft_json(quote=quote)]), capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_candidate_authoring"]
    assert len(ev) == 1
    assert WORKED_EXAMPLE_TEMPLATE in ev[0]["rationale"]
    assert "w001" in ev[0]["rationale"]


# --------------------------------------------------------------- arms end-to-end


def test_generate_gold_candidates_template_arms_only(tmp_path):
    """Selecting only the new arms runs them + records per-template counts."""
    slug = "demo-union-101"
    cdir = tmp_path / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    chunks = {
        "g001": _glossary_chunk(los=["to-03"]),
        "w001": _chunk("w001", _WORKED_TEXT, item_path="week_01/work.html",
                       los=["to-01"]),
    }
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
            "question_text": "seed question about integers here?",
            "relevant_passages": [{"chunk_id": "g001", "relevance": "primary",
                "anchor": {"item_path": "week_03/summary.html",
                           "text_quote": chunks["g001"]["text"][:60]}}],
            "authoring": {"method": "manual", "author": "@t",
                          "reviewed_by": "@r", "status": "reviewed"}}],
    }
    (cdir / "retrieval_eval").mkdir(parents=True)
    (cdir / "retrieval_eval" / "gold_set.json").write_text(json.dumps(gold))

    quote_def = "Any whole number, its negative counterpart, or zero"
    quote_we = "We need the least common multiple of 6 and 8. Step 1: list multiples."
    # one def draft + one worked-example draft (client cycles last response).
    client = FakeDraftClient([
        _draft_json(q="Define the term integer in this glossary please?", quote=quote_def),
        _draft_json(q="Walk through solving the scheduling problem step by step?", quote=quote_we),
    ])
    doc, out_path = generate_gold_candidates(
        cdir, client=client, capture=SpyCapture(),
        templates=[DEFINITION_TEMPLATE, WORKED_EXAMPLE_TEMPLATE],
    )
    bt = doc["authoring_run"]["by_template"]
    assert DEFINITION_TEMPLATE in bt and WORKED_EXAMPLE_TEMPLATE in bt
    assert "stratified" not in bt  # not selected
    templates_seen = {c["authoring"].get("template") for c in doc["candidates"]}
    assert templates_seen <= {DEFINITION_TEMPLATE, WORKED_EXAMPLE_TEMPLATE}


# --------------------------------------------------------------- both-population arm


def _both_chunks():
    """A union pair: one course chunk + one source chunk on absolute value."""
    course = _chunk(
        "crs1",
        "The course page explains absolute value as the distance of a number "
        "from zero on the number line for integers.",
        item_path="week_02/abs.html", los=["co-02"],
        section_heading="Absolute Value",
    )
    source = _chunk(
        "src1",
        "In the original textbook, absolute value measures the distance from "
        "zero on the number line; worked integer examples follow this idea.",
        item_path="algebra-ch1.html", los=["to-01"],
        source_refs=[{"sourceId": "dart:tb#1"}], section_heading="Absolute Value",
    )
    return {"crs1": course, "src1": source}


def _both_draft_json(
    q="How does the course page's treatment of absolute value relate to the textbook's?",
    course_quote="absolute value as the distance of a number from zero on the number line",
    source_quote="absolute value measures the distance from zero on the number line",
    kps=("distance from zero", "number line"),
):
    return json.dumps({
        "question_text": q, "expected_key_points": list(kps),
        "course_quote": course_quote, "source_quote": source_quote,
    })


def test_mine_both_population_pairs_pairs_on_shared_vocab():
    pairs = mine_both_population_pairs(_both_chunks())
    assert len(pairs) == 1
    p = pairs[0]
    assert p.course_chunk_id == "crs1" and p.source_chunk_id == "src1"
    assert "absolute" in p.shared_terms and "distance" in p.shared_terms


def test_mine_both_population_no_pair_when_no_shared_vocab():
    chunks = {
        "crs1": _chunk("crs1", _TEXT_B, item_path="week_01/frac.html"),
        "src1": _chunk("src1", _TEXT_C, item_path="calc.html",
                       source_refs=[{"sourceId": "x"}]),
    }
    assert mine_both_population_pairs(chunks) == []


def test_draft_both_population_builds_two_passage_candidate():
    chunks = _both_chunks()
    pairs = mine_both_population_pairs(chunks)
    cands = draft_both_population_candidates(
        pairs, chunks, client=FakeDraftClient([_both_draft_json()]))
    assert len(cands) == 1
    c = cands[0]
    assert c["question_type"] == "conceptual_synthesis"
    assert c["authoring"]["template"] == BOTH_POPULATION_TEMPLATE
    assert c["expected_citation_population"] == "both"
    assert c["difficulty"] == "medium"  # source chunk has no week => within_week
    assert c["synthesis_scope"] == "within_week"
    refs = [(p["chunk_id"], p["relevance"]) for p in c["relevant_passages"]]
    assert ("crs1", "primary") in refs and ("src1", "supporting") in refs
    assert c["objective_refs"] == ["co-02", "to-01"]
    assert 2 <= len(c["expected_key_points"]) <= 4
    # both quotes anchor (prescreen verifies primary + supporting).
    v = prescreen_candidate(c, chunks)
    assert v.passed, v.reasons


def test_draft_both_population_supporting_quote_must_anchor():
    chunks = _both_chunks()
    pairs = mine_both_population_pairs(chunks)
    bad = _both_draft_json(source_quote="this string is not in the source chunk at all")
    cands = draft_both_population_candidates(
        pairs, chunks, client=FakeDraftClient([bad]))
    v = prescreen_candidate(cands[0], chunks)
    assert not v.passed
    assert "SUPPORTING_QUOTE_NOT_IN_CHUNK" in v.reasons


def test_draft_both_population_records_parse_failure():
    chunks = _both_chunks()
    pairs = mine_both_population_pairs(chunks)
    cands = draft_both_population_candidates(
        pairs, chunks, client=FakeDraftClient(["not json"]))
    assert cands and "draft_error" in cands[0]
    assert cands[0]["authoring"]["template"] == BOTH_POPULATION_TEMPLATE
    assert len(cands[0]["relevant_passages"]) == 2


def test_draft_both_population_capture_fires_dynamic():
    chunks = _both_chunks()
    pairs = mine_both_population_pairs(chunks)
    cap = SpyCapture()
    draft_both_population_candidates(
        pairs, chunks, client=FakeDraftClient([_both_draft_json()]), capture=cap)
    ev = [e for e in cap.events if e["decision_type"] == "gold_candidate_authoring"]
    assert len(ev) == 1
    assert BOTH_POPULATION_TEMPLATE in ev[0]["rationale"]
    assert "crs1+src1" in ev[0]["rationale"]
    assert len(ev[0]["rationale"]) >= 20
