"""gold-enrichment-2026-06 — multi-passage relevant_passages tooling.

Covers the four deliverable concerns from plans/gold-enrichment-2026-06.md §2:

  1. The chunk-vs-expected_key_points SUPPORT SCORER supports / doesn't a key
     point at the REUSED citation-attribution floors (4-shingle >= 0.25 OR
     token-coverage >= 0.80) — no new thresholds invented.
  2. CANDIDATE SELECTION excludes the existing primary (and any already-pinned
     chunk), keeps only genuinely-supporting unpinned candidates.
  3. The PROPOSAL artifact shape (per question: existing_primary +
     proposed_additional_passages[] w/ support + rationale; primary untouched).
  4. The PROMOTE path is GATED: fails closed on a frozen set without unfreeze,
     re-validates fail-closed after applying, and never replaces the primary.

Hermetic: synthetic gold docs + a tiny on-disk chunkset in tmp_path. No real
course data, no network, no LLM, no hardcoded slugs/paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval.citation_attribution import (
    DEFAULT_MIN_OVERLAP,
    TOKEN_COVERAGE_FLOOR,
)
from lib.retrieval.gold_passage_enrichment import (
    GOLD_PASSAGE_ENRICHMENT_FILENAME,
    GoldEnrichError,
    apply_enrichment_into_gold,
    build_enrichment_proposals,
    chunk_supports_question,
    gather_candidates,
    generate_enrichment_proposal,
    promote_enrichment,
    propose_enrichment_for_question,
)
from lib.retrieval.gold_set import (
    GOLD_SET_FILENAME,
    RETRIEVAL_EVAL_SUBDIR,
    has_critical_issues,
    load_gold_set,
)
from lib.utils import sha256_file


# --------------------------------------------------------------- fixtures

# A verbatim sentence that both names the key point's tokens AND is >= 40 chars
# (so it can anchor) and is distinctive (so it isn't quote-ambiguous).
_PRIMARY_TEXT = (
    "A polynomial is an expression of variables and coefficients. "
    "Polynomials are combined using addition and multiplication only."
)
_SUPPORTING_TEXT = (
    "Like terms are terms that share the same variable raised to the same power. "
    "Combining like terms simplifies a polynomial expression substantially today."
)
_OFFTOPIC_TEXT = (
    "The water cycle describes evaporation and precipitation over the oceans. "
    "It has nothing whatsoever to do with algebra or polynomial expressions."
)
_NEAR_DUP_TEXT = _PRIMARY_TEXT  # identical content => near-duplicate of primary


def _chunk(cid, text, *, item_path, section_heading=None):
    src = {"item_path": item_path}
    if section_heading:
        src["section_heading"] = section_heading
    return {"id": cid, "text": text, "source": src}


def _chunks():
    return {
        "c_primary": _chunk("c_primary", _PRIMARY_TEXT,
                            item_path="week_02/polynomials.html",
                            section_heading="Polynomials"),
        "c_support": _chunk("c_support", _SUPPORTING_TEXT,
                            item_path="week_03/like_terms.html",
                            section_heading="Like Terms"),
        "c_offtopic": _chunk("c_offtopic", _OFFTOPIC_TEXT,
                             item_path="week_09/water.html",
                             section_heading="Water Cycle"),
        "c_neardup": _chunk("c_neardup", _NEAR_DUP_TEXT,
                            item_path="week_02/polynomials_copy.html",
                            section_heading="Polynomials (review)"),
    }


def _write_chunkset(tmp_path: Path, chunks) -> tuple[Path, str]:
    course_dir = tmp_path / "courses" / "demo-101"
    corpus_dir = course_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = corpus_dir / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks.values()) + "\n", encoding="utf-8"
    )
    return course_dir, sha256_file(chunks_path)


def _question(qid="gq-demo-101-0001", *, key_points, primary_cid="c_primary",
              extra_passages=None):
    passages = [{
        "chunk_id": primary_cid,
        "relevance": "primary",
        "anchor": {
            "item_path": "week_02/polynomials.html",
            "text_quote": "A polynomial is an expression of variables and coefficients.",
        },
    }]
    if extra_passages:
        passages.extend(extra_passages)
    return {
        "question_id": qid,
        "question_type": "factual_recall",
        "question_text": "what is a polynomial and what are like terms?",
        "difficulty": "easy",
        "expected_citation_population": "course",
        "expected_key_points": key_points,
        "relevant_passages": passages,
        "authoring": {"method": "manual", "author": "@t",
                      "reviewed_by": "@r", "status": "reviewed"},
    }


def _gold(course_dir: Path, chunks_sha, questions, *, frozen=False):
    return {
        "schema_version": "1.1",
        "course_slug": "demo-101",
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": chunks_sha},
        "authored_at": "2026-06-13T00:00:00Z",
        "frozen": frozen,
        "questions": questions,
    }


def _write_gold(course_dir: Path, gold) -> Path:
    eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / GOLD_SET_FILENAME
    p.write_text(json.dumps(gold, indent=2) + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------- 1) support scorer


def test_support_scorer_supports_key_point_at_reused_floors():
    """A chunk that names a key point's tokens supports it via a REUSED arm."""
    support = chunk_supports_question(
        _SUPPORTING_TEXT,
        ["like terms share the same variable raised to the same power"],
    )
    assert support.supports is True
    assert support.best is not None
    # The arm that fired is one of the reused citation-attribution arms.
    assert support.best.arm in ("shingle", "token_coverage")
    # And it cleared a reused floor (no new threshold).
    assert (support.best.shingle >= DEFAULT_MIN_OVERLAP
            or support.best.coverage >= TOKEN_COVERAGE_FLOOR)


def test_support_scorer_rejects_offtopic_chunk():
    """An off-topic chunk supports no key point => supports=False."""
    support = chunk_supports_question(
        _OFFTOPIC_TEXT,
        ["like terms share the same variable raised to the same power",
         "a polynomial is an expression of variables and coefficients"],
    )
    assert support.supports is False
    assert support.best is None
    assert support.supported_key_point_count == 0


def test_support_scorer_no_key_points_is_unsupported():
    """No expected_key_points => nothing to score => supports=False."""
    support = chunk_supports_question(_SUPPORTING_TEXT, [])
    assert support.supports is False


def test_support_scorer_nli_arm_uses_groundedness_floor():
    """When an NLI classifier is injected, the entailment arm fires at the
    SAME NLI_ADD_ENTAILMENT_FLOOR the groundedness/NLI-ADD arms use."""

    class _FakeNLIScore:
        def __init__(self, ent):
            self.entailment = ent

    class _FakeNLI:
        def score_batch(self, *, pairs):
            return [_FakeNLIScore(0.95) for _ in pairs]

    # A chunk whose lexical overlap is below the deterministic floors but which
    # the NLI entails at >= 0.70 still supports the key point via the nli arm.
    support = chunk_supports_question(
        "Monomials and binomials are special cases discussed elsewhere.",
        ["a polynomial is an expression of variables and coefficients"],
        nli=_FakeNLI(),
    )
    assert support.supports is True
    assert support.best is not None
    assert support.best.arm == "nli"
    assert support.best.entailment is not None and support.best.entailment >= 0.70


# ------------------------------------------------ 2) candidate selection


def test_candidate_selection_excludes_existing_primary():
    """The pinned primary chunk is NEVER proposed, even if it would score as a
    supporter. Only the genuinely-supporting unpinned chunk is proposed."""
    chunks = _chunks()
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
        "like terms share the same variable raised to the same power",
    ])
    # Candidate pool deliberately includes the primary + a supporter + offtopic.
    enrichment = propose_enrichment_for_question(
        q, chunks, ["c_primary", "c_support", "c_offtopic"]
    )
    proposed_ids = {p.chunk_id for p in enrichment.proposed_additional_passages}
    assert "c_primary" not in proposed_ids       # the primary is excluded
    assert "c_offtopic" not in proposed_ids      # off-topic supports nothing
    assert "c_support" in proposed_ids           # the real supporter is proposed
    # Every proposed passage is relevance:supporting (never primary).
    assert all(p.relevance == "supporting"
               for p in enrichment.proposed_additional_passages)


def test_candidate_selection_excludes_already_pinned_supporting():
    """A chunk already pinned (as supporting) is not re-proposed."""
    chunks = _chunks()
    q = _question(
        key_points=["like terms share the same variable raised to the same power"],
        extra_passages=[{
            "chunk_id": "c_support", "relevance": "supporting",
            "anchor": {"item_path": "week_03/like_terms.html",
                       "text_quote": "Like terms are terms that share the same variable raised to the same power."},
        }],
    )
    enrichment = propose_enrichment_for_question(q, chunks, ["c_support"])
    assert enrichment.proposed_additional_passages == []


def test_candidate_selection_excludes_near_duplicate_of_pinned():
    """A near-duplicate of an already-pinned passage is not new support."""
    chunks = _chunks()
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
    ])
    # c_neardup is identical content to the pinned primary => NEAR_DUPLICATE.
    enrichment = propose_enrichment_for_question(q, chunks, ["c_neardup"])
    assert enrichment.proposed_additional_passages == []


def test_gather_candidates_unions_retrieval_and_cited_drops_unknown():
    chunks = _chunks()
    q = _question(key_points=["x"])

    class _R:
        def __init__(self, cid):
            self.chunk_id = cid

    def fake_retrieve(root, query, *, course_slug=None, engine="lexical", limit=10):
        return [_R("c_support"), _R("c_offtopic"), _R("c_not_in_set")]

    pool = gather_candidates(
        q, chunks, retrieve_fn=fake_retrieve, libv2_root=Path("/x"),
        cited_chunk_ids=["c_primary", "c_ghost"],
    )
    # retrieval first (rank order, unknown dropped), then cited-only (sorted),
    # unknown ids dropped from both arms.
    assert pool == ["c_support", "c_offtopic", "c_primary"]
    assert "c_not_in_set" not in pool
    assert "c_ghost" not in pool


# ------------------------------------------------ 3) proposal artifact shape


def test_proposal_doc_shape_and_primary_untouched(tmp_path):
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
        "like terms share the same variable raised to the same power",
    ])
    gold = _gold(course_dir, sha, [q])
    _write_gold(course_dir, gold)

    class _R:
        def __init__(self, cid):
            self.chunk_id = cid

    def fake_retrieve(root, query, *, course_slug=None, engine="lexical", limit=10):
        return [_R("c_support"), _R("c_offtopic")]

    doc, out_path = generate_enrichment_proposal(
        course_dir, retrieve_fn=fake_retrieve, libv2_root=course_dir,
    )
    # Doc-level shape.
    assert doc["schema_version"] == "1.0"
    assert doc["course_slug"] == "demo-101"
    assert "_operator_note" in doc and "PROPOSAL ONLY" in doc["_operator_note"]
    assert doc["support_floors"]["min_overlap_shingle"] == DEFAULT_MIN_OVERLAP
    assert doc["support_floors"]["token_coverage_floor"] == TOKEN_COVERAGE_FLOOR
    assert doc["n_questions"] == 1
    assert doc["n_proposed_passages"] == 1

    entry = doc["questions"][0]
    assert entry["question_id"] == "gq-demo-101-0001"
    # existing_primary recorded (for review) but never modified.
    assert entry["existing_primary"]["chunk_id"] == "c_primary"
    assert entry["existing_primary"]["relevance"] == "primary"
    # per-passage shape.
    [passage] = entry["proposed_additional_passages"]
    assert passage["chunk_id"] == "c_support"
    assert passage["relevance"] == "supporting"
    assert passage["status"] == "proposed"
    assert passage["anchor"]["text_quote"]
    assert passage["anchor"]["content_sha256"]
    assert passage["supported_key_point"]
    assert passage["support"]["arm"] in ("shingle", "token_coverage", "nli")
    assert "rationale" in passage and len(passage["rationale"]) >= 20

    # The artifact was written; the gold set itself is byte-for-byte UNCHANGED
    # (no mutation of gold_set.json).
    assert out_path.name == GOLD_PASSAGE_ENRICHMENT_FILENAME
    gold_after = json.loads(
        (course_dir / RETRIEVAL_EVAL_SUBDIR / GOLD_SET_FILENAME).read_text()
    )
    assert gold_after == gold


def test_build_proposals_records_scored_question_with_no_proposals(tmp_path):
    """A question whose candidates all fail support still appears (n_proposed=0)
    so the operator sees it WAS scored, not silently skipped."""
    chunks = _chunks()
    q = _question(key_points=["a polynomial is an expression of variables and coefficients"])
    gold = _gold(tmp_path, "0" * 64, [q])
    enrichments = build_enrichment_proposals(gold, chunks, retrieve_fn=None,
                                             cited_by_question={"gq-demo-101-0001": ["c_offtopic"]})
    assert len(enrichments) == 1
    assert enrichments[0].proposed_additional_passages == []
    assert enrichments[0].n_candidates_scored == 1


# ------------------------------------------------ 4) promote path is gated


def _proposal_doc_with_accepted(chunk_id="c_support",
                                qid="gq-demo-101-0001", quote=None, status="accepted"):
    quote = quote or "Like terms are terms that share the same variable raised to the same power."
    return {
        "schema_version": "1.0",
        "course_slug": "demo-101",
        "questions": [{
            "question_id": qid,
            "proposed_additional_passages": [{
                "chunk_id": chunk_id,
                "relevance": "supporting",
                "anchor": {"item_path": "week_03/like_terms.html", "text_quote": quote},
                "status": status,
            }],
        }],
    }


def _write_proposal(course_dir: Path, doc) -> Path:
    eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / GOLD_PASSAGE_ENRICHMENT_FILENAME
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return p


def test_promote_refuses_frozen_without_unfreeze(tmp_path):
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=["like terms share the same variable raised to the same power"])
    _write_gold(course_dir, _gold(course_dir, sha, [q], frozen=True))
    _write_proposal(course_dir, _proposal_doc_with_accepted())

    with pytest.raises(GoldEnrichError) as exc:
        promote_enrichment(course_dir, unfreeze_for_enrich=False)
    assert exc.value.code == "GOLD_SET_FROZEN"
    # The gold set is untouched after a refused promote.
    gold_after = json.loads(
        (course_dir / RETRIEVAL_EVAL_SUBDIR / GOLD_SET_FILENAME).read_text()
    )
    assert len(gold_after["questions"][0]["relevant_passages"]) == 1


def test_promote_applies_accepted_and_validates(tmp_path):
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
        "like terms share the same variable raised to the same power",
    ])
    _write_gold(course_dir, _gold(course_dir, sha, [q], frozen=True))
    _write_proposal(course_dir, _proposal_doc_with_accepted())

    report, written = promote_enrichment(course_dir, unfreeze_for_enrich=True)
    assert report.n_applied == 1
    assert written is not None
    # The supporting passage was appended; the primary is preserved verbatim.
    gold_after, issues = load_gold_set(course_dir, verify=True)
    assert not has_critical_issues(issues)  # re-validated fail-closed: clean
    passages = gold_after["questions"][0]["relevant_passages"]
    assert len(passages) == 2
    primaries = [p for p in passages if p["relevance"] == "primary"]
    supporting = [p for p in passages if p["relevance"] == "supporting"]
    assert len(primaries) == 1 and primaries[0]["chunk_id"] == "c_primary"
    assert len(supporting) == 1 and supporting[0]["chunk_id"] == "c_support"


def test_promote_skips_non_accepted_passages(tmp_path):
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
        "like terms share the same variable raised to the same power",
    ])
    _write_gold(course_dir, _gold(course_dir, sha, [q]))
    # status still "proposed" => not accepted => skipped.
    _write_proposal(course_dir, _proposal_doc_with_accepted(status="proposed"))

    report, _ = promote_enrichment(course_dir, dry_run=True)
    assert report.n_applied == 0
    assert any(s["reason"] == "not_accepted" for s in report.skipped)


def test_apply_fails_closed_on_quote_not_in_chunk(tmp_path):
    """An accepted passage whose quote isn't in its chunk is caught by the
    fail-closed validate_gold_set re-validation (the WHOLE promote is refused)."""
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=["like terms share the same variable raised to the same power"])
    gold = _gold(course_dir, sha, [q])
    bad = _proposal_doc_with_accepted(
        quote="this verbatim sentence does not appear anywhere in the cited chunk text."
    )
    with pytest.raises(GoldEnrichError) as exc:
        apply_enrichment_into_gold(gold, bad, chunks, sha, course_slug="demo-101")
    assert exc.value.code == "GOLD_ENRICH_VALIDATION_FAILED"
    assert "GOLD_SET_QUOTE_NOT_IN_CHUNK" in exc.value.detail


def test_apply_skips_unknown_question_and_already_pinned(tmp_path):
    chunks = _chunks()
    course_dir, sha = _write_chunkset(tmp_path, chunks)
    q = _question(key_points=[
        "a polynomial is an expression of variables and coefficients",
        "like terms share the same variable raised to the same power",
    ])
    gold = _gold(course_dir, sha, [q])
    doc = {
        "questions": [
            # unknown question id => skipped.
            _proposal_doc_with_accepted(qid="gq-demo-101-9999")["questions"][0],
            # accepted but the chunk is the already-pinned primary => skipped.
            {
                "question_id": "gq-demo-101-0001",
                "proposed_additional_passages": [{
                    "chunk_id": "c_primary", "relevance": "supporting",
                    "anchor": {"item_path": "week_02/polynomials.html",
                               "text_quote": "A polynomial is an expression of variables and coefficients."},
                    "status": "accepted",
                }],
            },
        ]
    }
    new_gold, report = apply_enrichment_into_gold(
        gold, doc, chunks, sha, course_slug="demo-101"
    )
    assert report.n_applied == 0
    reasons = {s["reason"] for s in report.skipped}
    assert "unknown_question_id" in reasons
    assert "already_pinned" in reasons
    # Primary untouched, no supporting added.
    assert len(new_gold["questions"][0]["relevant_passages"]) == 1
