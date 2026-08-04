"""End-to-end tests for the grounded-answer pipeline + citation gate.

Deterministic, CI-safe: no model, no server, no network. The LLM call site is
driven by the shared ``FakeAnswerClient`` from ``test_answer_composer``;
retrieval runs the REAL lexical BM25 path over the mini-course fixture
materialised into a tmp LibV2 layout. The offline-guard arm proves the whole
query path needs nothing beyond loopback.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lib.retrieval.answer_backend import AnswerBackendUnavailable
from lib.retrieval.grounded_answer import (
    DECISION_TYPE_COMPLETENESS_RECHECK,
    STATUS_ANSWERED,
    STATUS_ANSWERED_WITH_WARNINGS,
    STATUS_BLOCKED_CITATION_GATE,
    STATUS_BLOCKED_INVALID_CITATION,
    STATUS_REFUSED_LOW_CONFIDENCE,
    STATUS_REFUSED_NOT_IN_COURSE,
    Citation,
    GroundedAnswer,
    answer_course_question,
    page_label_for_source,
)
from lib.retrieval.refusal import (
    REASON_LOW_CONFIDENCE,
    REASON_NOT_IN_COURSE_MODEL,
    RefusalPolicy,
)

# A permissive semantic policy: cosine floor 0 so the citation gate under test
# is reached regardless of the fake embedder's cosine.
_PERMISSIVE_SEMANTIC = RefusalPolicy(
    engine="semantic",
    min_top_score=-1.0,
    score_floor=-1.0,
    min_passages_above_floor=1,
    policy_version="test-permissive-semantic",
)
from lib.testing.no_network import no_network

# Shared test doubles.
from lib.tests.test_answer_composer import FakeAnswerClient, SpyCapture

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)
COURSE_SLUG = "mini-retrieval-101"

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas" / "events" / "decision_event.schema.json"
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def mini_libv2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialise the mini-course fixture into a tmp LibV2 layout.

    Returns the LibV2 root; points ``ED4ALL_LIBV2_ROOT`` at it so the retriever
    + anchor resolver resolve the course exactly as a real archived course.
    """
    if not FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {FIXTURE_ROOT}")
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / COURSE_SLUG
    shutil.copytree(FIXTURE_ROOT, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return libv2_root


def _append_chunk(libv2_root: Path, chunk: dict) -> None:
    path = libv2_root / "courses" / COURSE_SLUG / "semantik_chunks" / "chunks.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(chunk) + "\n")


def _envelope(answer, citations, not_in_course=False):
    return json.dumps(
        {"answer": answer, "citations": citations, "not_in_course": not_in_course}
    )


def _enum_members() -> set:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return set(schema["properties"]["decision_type"]["enum"])


# A deliberately STRICT lexical policy used to force a pre-LLM refusal on a
# low-score retrieval (the v0-uncalibrated default is permissive by design).
_STRICT_LEXICAL = RefusalPolicy(
    engine="lexical",
    min_top_score=100.0,  # nothing clears this — guarantees refusal
    score_floor=0.5,
    min_passages_above_floor=1,
    policy_version="test-strict",
)

# A permissive lexical policy. By default the pipeline resolves the calibrated
# (lexical, None) pin, whose min_top_score sits above this small fixture's BM25
# scores. Tests exercising the POST-confidence surfaces (citation gate, answer
# shape) pass this policy explicitly so they clear confidence regardless of the
# fixture's corpus-specific scores.
_PERMISSIVE_LEXICAL = RefusalPolicy(
    engine="lexical",
    min_top_score=0.0,
    score_floor=0.0,
    min_passages_above_floor=1,
    policy_version="test-permissive-lexical",
)


# --------------------------------------------------------------------------- #
# Happy path: answer + resolving citations over the fixture
# --------------------------------------------------------------------------- #


def test_happy_path_answers_with_resolved_citation(mini_libv2: Path):
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_001"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert isinstance(result, GroundedAnswer)
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "A vector store indexes embedding vectors."
    assert len(result.citations) == 1
    cit = result.citations[0]
    assert cit.chunk_id == "mini_alpha_chunk_001"
    # chunk_001 anchors resolved_exact in the fixture.
    assert cit.anchor_status == "resolved_exact"
    assert cit.page_label  # human link text present
    assert cit.link_target["kind"] == "course_page"
    # char_span forwarded only for resolved_exact.
    assert cit.link_target["char_span"] is not None
    assert result.model_id == "qwen2.5:7b-instruct-q4_K_M"
    assert result.prompt_version == "ws3.v5"


def test_resolved_normalized_citation_passes_without_char_span(mini_libv2: Path):
    # The _fabricated chunk resolves via normalized substring (bad span), so it
    # PASSES the gate but must NOT forward a char_span (only resolved_exact does).
    client = FakeAnswerClient([
        _envelope("FAISS does similarity search.",
                  ["mini_alpha_chunk_007_fabricated"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What is FAISS used for similarity search?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert result.status == STATUS_ANSWERED
    cit = result.citations[0]
    assert cit.anchor_status == "resolved_normalized"
    assert cit.link_target["char_span"] is None


# --------------------------------------------------------------------------- #
# Provenance chain: source_references → citation.source_block + pdf_pages
# --------------------------------------------------------------------------- #


def _load_chunk(libv2_root: Path, chunk_id: str) -> dict:
    path = libv2_root / "courses" / COURSE_SLUG / "semantik_chunks" / "chunks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") == chunk_id:
            return rec
    raise AssertionError(f"chunk {chunk_id} not in fixture")


def test_source_references_thread_to_citation(mini_libv2: Path):
    """A chunk carrying source_references → citation.source_block + pdf_pages."""
    base = _load_chunk(mini_libv2, "mini_alpha_chunk_001")
    prov_chunk = dict(base)
    prov_chunk["id"] = "mini_alpha_chunk_prov"
    prov_chunk["text"] = (
        "Provenance marker xenon: a vector store indexes embedding vectors "
        "for nearest-neighbor lookup."
    )
    src = dict(prov_chunk["source"])
    src["source_references"] = [
        {
            "sourceId": "semantik:mini_alpha#s3_c0",
            "role": "primary",
            "extractor": "synthesized",
            "pages": [12, 12, 7],
        }
    ]
    prov_chunk["source"] = src
    _append_chunk(mini_libv2, prov_chunk)

    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_prov"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "provenance marker xenon vector store",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        validate_citations=False,  # this synthetic chunk need not anchor
    )
    assert result.status == STATUS_ANSWERED
    cit = result.citations[0]
    assert cit.source_block == "semantik:mini_alpha#s3_c0"
    # de-duplicated + sorted page list.
    assert cit.pdf_pages == [7, 12]
    d = cit.to_dict()
    assert d["source_block"] == "semantik:mini_alpha#s3_c0"
    assert d["pdf_pages"] == [7, 12]


def test_contributing_role_used_when_no_primary(mini_libv2: Path):
    """No 'primary' ref → fall back to the first reference (never empty)."""
    base = _load_chunk(mini_libv2, "mini_alpha_chunk_001")
    chunk = dict(base)
    chunk["id"] = "mini_alpha_chunk_contrib"
    chunk["text"] = "Krypton contributing marker: embeddings power retrieval."
    src = dict(chunk["source"])
    src["source_references"] = [
        {"sourceId": "semantik:mini_alpha#s9_c2", "role": "contributing", "pages": [4]},
    ]
    chunk["source"] = src
    _append_chunk(mini_libv2, chunk)

    client = FakeAnswerClient([
        _envelope("Embeddings power retrieval.", ["mini_alpha_chunk_contrib"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "krypton contributing marker embeddings",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        validate_citations=False,
    )
    cit = result.citations[0]
    assert cit.source_block == "semantik:mini_alpha#s9_c2"
    assert cit.pdf_pages == [4]


# --------------------------------------------------------------------------- #
# Pre-LLM refusal (low confidence) — zero client calls
# --------------------------------------------------------------------------- #


def test_pre_llm_refusal_does_not_call_client(mini_libv2: Path):
    client = FakeAnswerClient([_envelope("should never be used", ["x"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_STRICT_LEXICAL,
    )
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert result.answer_text is None
    assert result.citations == []
    assert result.refusal["reason_code"] == REASON_LOW_CONFIDENCE
    # The composer was NEVER called (the whole point of pre-LLM refusal).
    assert len(client.calls) == 0


def test_pre_llm_refusal_emits_capture_with_dynamic_rationale(mini_libv2: Path):
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope("x", ["x"])])
    answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_STRICT_LEXICAL, capture=spy,
    )
    refusals = [e for e in spy.events
                if e["decision_type"] == "grounded_answer_refusal"]
    assert len(refusals) == 1
    rationale = refusals[0]["rationale"]
    assert len(rationale) >= 20
    # Dynamic signal interpolation: policy version + reason code present.
    assert "test-strict" in rationale
    assert REASON_LOW_CONFIDENCE in rationale
    assert refusals[0]["decision_type"] in _enum_members()


# --------------------------------------------------------------------------- #
# Cross-encoder reranker hook (ED4ALL_RERANK_PROVIDER, default OFF)
# --------------------------------------------------------------------------- #


def test_rerank_off_byte_identical(mini_libv2: Path, monkeypatch: pytest.MonkeyPatch):
    """Unset ED4ALL_RERANK_PROVIDER → byte-identical happy-path result and NO
    grounded_answer_rerank capture (the byte-identical-when-off contract)."""
    monkeypatch.delenv("ED4ALL_RERANK_PROVIDER", raising=False)
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_001"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "A vector store indexes embedding vectors."
    assert result.citations[0].chunk_id == "mini_alpha_chunk_001"
    assert not [e for e in spy.events
                if e["decision_type"] == "grounded_answer_rerank"]


def test_rerank_on_emits_capture(mini_libv2: Path, monkeypatch: pytest.MonkeyPatch):
    """With the fake reranker on (+ allow-fake), a grounded_answer_rerank
    DecisionCapture row fires with a dynamic >=20-char rationale, and the answer
    still resolves (reorder+trim never drops the only candidates here)."""
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_RERANK_ALLOW_FAKE", "1")
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_001"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == "grounded_answer_rerank"]
    assert len(rows) == 1
    rationale = rows[0]["rationale"]
    assert len(rationale) >= 20
    assert "provider=fake" in rationale
    assert "top_k=" in rationale
    assert "fallback_used=" in rationale
    assert rows[0]["decision_type"] in _enum_members()


def test_rerank_does_not_change_refusal_verdict_for_same_set(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Reorder-only (pool==limit so no over-fetch/trim) never rescues a
    refusal: native scores are preserved so the verdict is identical on/off."""
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_RERANK_ALLOW_FAKE", "1")
    monkeypatch.setenv("ED4ALL_RERANK_CANDIDATE_POOL", "8")
    client = FakeAnswerClient([_envelope("never used", ["x"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_STRICT_LEXICAL, limit=8,
    )
    # Strict policy refuses regardless of reorder (top score unchanged).
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert len(client.calls) == 0


# --------------------------------------------------------------------------- #
# Model-side not_in_course refusal
# --------------------------------------------------------------------------- #


def test_model_not_in_course_refuses(mini_libv2: Path):
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope("", [], not_in_course=True)])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert result.status == STATUS_REFUSED_NOT_IN_COURSE
    assert result.answer_text is None
    assert result.citations == []
    assert result.refusal["reason_code"] == REASON_NOT_IN_COURSE_MODEL
    # The client WAS called (model-side refusal is post-compose).
    assert len(client.calls) == 1
    refusals = [e for e in spy.events
                if e["decision_type"] == "grounded_answer_refusal"]
    assert len(refusals) == 1
    assert REASON_NOT_IN_COURSE_MODEL in refusals[0]["rationale"]


# --------------------------------------------------------------------------- #
# Citation-gate block path
# --------------------------------------------------------------------------- #


def test_citation_gate_blocks_unresolvable_citation(mini_libv2: Path):
    # Inject a ghost chunk whose item_path points at a non-existent page so its
    # anchor resolves to source_page_missing. It is keyword-rich so BM25 ranks
    # it; the model then cites it and the gate must block the answer.
    _append_chunk(mini_libv2, {
        "id": "mini_ghost_chunk_999",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "concept_tags": ["chunking", "overlap"],
        "text": ("Chunking strategies split documents into overlapping "
                 "passages before embedding for retrieval."),
        "learning_outcome_refs": [],
        "source": {
            "item_path": "ghost.html",
            "html_xpath": "/html[1]/body[1]",
            "char_span": [0, 10],
            "section_heading": "Ghost Section",
            "module_id": "ghost",
            "course_id": "MINI_RETRIEVAL_101",
        },
    })
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("Chunking splits documents.", ["mini_ghost_chunk_999"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "chunking strategies overlapping passages",
        client=client, capture=spy,
    )
    assert result.status == STATUS_BLOCKED_CITATION_GATE
    # Answer text WITHHELD; citations empty (contract: non-empty only for
    # answered*). The blocked id rides the warnings + capture stream.
    assert result.to_dict()["answer_text"] is None
    assert result.citations == []
    assert any("mini_ghost_chunk_999" in w for w in result.warnings)
    gate_events = [e for e in spy.events
                   if e["decision_type"] == "grounded_answer_citation_gate"]
    assert len(gate_events) == 1
    rationale = gate_events[0]["rationale"]
    assert "mini_ghost_chunk_999" in rationale
    assert "source_page_missing" in rationale
    assert gate_events[0]["decision_type"] in _enum_members()


def test_citation_gate_passes_emits_capture(mini_libv2: Path):
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("Answer.", ["mini_alpha_chunk_001"])
    ])
    answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    gate_events = [e for e in spy.events
                   if e["decision_type"] == "grounded_answer_citation_gate"]
    assert len(gate_events) == 1
    assert "passed" in gate_events[0]["rationale"]


def test_empty_citations_blocks_as_invalid(mini_libv2: Path):
    # A grounded answer with zero citations and not_in_course=false is a
    # contradiction in terms -> blocked_invalid_citation.
    client = FakeAnswerClient([_envelope("Bare answer.", [])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert result.status == STATUS_BLOCKED_INVALID_CITATION
    assert result.answer_text is None
    assert result.citations == []


def test_validate_citations_false_bypasses_gate(mini_libv2: Path):
    # Same ghost chunk that would normally block, but the test-only bypass
    # emits an answer with the (unresolved) citation surfaced.
    _append_chunk(mini_libv2, {
        "id": "mini_ghost_chunk_999",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "concept_tags": ["chunking"],
        "text": ("Chunking strategies split documents into overlapping "
                 "passages before embedding for retrieval."),
        "learning_outcome_refs": [],
        "source": {"item_path": "ghost.html", "html_xpath": "/html[1]/body[1]",
                   "char_span": [0, 10], "section_heading": "Ghost"},
    })
    client = FakeAnswerClient([
        _envelope("Chunking splits documents.", ["mini_ghost_chunk_999"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "chunking strategies overlapping passages",
        client=client, validate_citations=False,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "Chunking splits documents."
    assert result.citations[0].anchor_status == "source_page_missing"


# --------------------------------------------------------------------------- #
# Attribution-driven citation prune + add
# --------------------------------------------------------------------------- #
#
# The pipeline runs claim-attribution over ALL gate-eligible passages strictly
# post-citation-gate. Default code mode is `shadow` (compute + capture + excerpt,
# mutate nothing); these tests drive `prune_mode="on"` explicitly. Verdicts never
# change: pruning keeps >= 1 citation, additions only add anchor-resolved
# uncited passages.

# A definitional answer near-verbatim from mini_alpha_chunk_001.
_DEF_ANSWER = (
    "A vector store is a database that indexes high-dimensional embedding "
    "vectors so that approximate nearest-neighbour search can retrieve "
    "semantically similar passages."
)


def _prune_events(spy):
    return [e for e in spy.events
            if e["decision_type"] == "grounded_answer_citation_prune"]


def test_prune_drops_claimless_cited_citation(mini_libv2: Path):
    """Two cited chunks; one backs the answer, one (recall@k) backs nothing →
    on-mode prunes the claim-less one, keeps the supporter, fires the capture."""
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER,
                  ["mini_alpha_chunk_001", "mini_alpha_chunk_003"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store recall at k embedding nearest neighbour passages",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="on",
    )
    assert result.status == STATUS_ANSWERED
    kept_ids = [c.chunk_id for c in result.citations]
    assert "mini_alpha_chunk_001" in kept_ids
    assert "mini_alpha_chunk_003" not in kept_ids  # claim-less → pruned
    assert any("pruned_claimless_citation:mini_alpha_chunk_003" in w
               for w in result.warnings)
    # The kept supporter carries its attribution surfacing.
    kept = next(c for c in result.citations if c.chunk_id == "mini_alpha_chunk_001")
    assert kept.supported_claim_count >= 1
    assert kept.supporting_excerpt
    # Capture fired with dynamic rationale.
    ev = _prune_events(spy)
    assert len(ev) == 1
    rat = ev[0]["rationale"]
    assert "mini_alpha_chunk_003" in rat and "min_overlap" in rat
    assert ev[0]["decision_type"] in _enum_members()
    assert ev[0]["decision"] == "citation_prune:pruned"


def test_prune_all_claimless_empties_sources_with_advisory(mini_libv2: Path):
    """Policy: no sources beats a misleading one.

    When EVERY cited citation is claim-less and no uncited supporter rescues
    the answer, ALL citations are pruned; the answer ships with zero sources
    and flips to answered_with_warnings (the unverified-support advisory).
    The answered-family verdict never becomes a refusal/block."""
    spy = SpyCapture()
    # An answer with no lexical overlap to either cited chunk → both claim-less.
    client = FakeAnswerClient([
        _envelope("Photosynthesis stores chemical energy in glucose molecules.",
                  ["mini_alpha_chunk_001", "mini_alpha_chunk_002"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "vector store faiss similarity search",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="on",
    )
    assert result.status == STATUS_ANSWERED_WITH_WARNINGS
    assert result.citations == []
    assert any("pruned_all_claimless_citations" in w for w in result.warnings)
    assert _prune_events(spy)[0]["decision"] == "citation_prune:pruned_all_claimless"


def test_prune_all_claimless_shadow_mode_mutates_nothing(mini_libv2: Path):
    """Shadow mode computes + captures but never mutates: the all-claimless
    case keeps every citation and the answered status."""
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("Photosynthesis stores chemical energy in glucose molecules.",
                  ["mini_alpha_chunk_001", "mini_alpha_chunk_002"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "vector store faiss similarity search",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="shadow",
    )
    assert result.status == STATUS_ANSWERED
    assert {c.chunk_id for c in result.citations} == {
        "mini_alpha_chunk_001", "mini_alpha_chunk_002"
    }


def test_add_credits_uncited_supporter(mini_libv2: Path):
    """The model cites topical-mention chunks but NOT the definitional supporter
    that backs the answer; on-mode ADDS the uncited supporter (it out-supports
    the cited set and anchor-resolves)."""
    spy = SpyCapture()
    # Cite only FAISS + recall@k (topical), answer is the vector-store definition
    # (mini_alpha_chunk_001 — uncited but retrieved + anchorable).
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER,
                  ["mini_alpha_chunk_002", "mini_alpha_chunk_003"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors faiss recall at k",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="on",
    )
    assert result.status == STATUS_ANSWERED
    kept_ids = [c.chunk_id for c in result.citations]
    assert "mini_alpha_chunk_001" in kept_ids  # the uncited supporter, ADDED
    added = next(c for c in result.citations if c.chunk_id == "mini_alpha_chunk_001")
    assert added.anchor_status.startswith("resolved")  # passed the anchor gate
    assert added.supported_claim_count >= 1
    assert any("added_supporting_citation:mini_alpha_chunk_001" in w
               for w in result.warnings)
    ev = _prune_events(spy)
    assert "added=[mini_alpha_chunk_001]" in ev[0]["rationale"]


def test_added_citation_leads_when_strongest(mini_libv2: Path):
    """Final citations sort strongest-supporter-first: the added definitional
    supporter (backs the answer) leads the weaker cited mentions."""
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER,
                  ["mini_alpha_chunk_002", "mini_alpha_chunk_003"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors faiss recall at k",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, prune_mode="on",
    )
    # Strongest supporter (by supported_claim_count) is first.
    counts = [c.supported_claim_count for c in result.citations]
    assert counts == sorted(counts, reverse=True)
    assert result.citations[0].chunk_id == "mini_alpha_chunk_001"


def test_addition_never_changes_verdict_blocked_stays_blocked(mini_libv2: Path):
    """An unresolvable CITED citation still blocks — attribution add/prune runs
    only AFTER the gate passes and never rescues a blocked answer."""
    _append_chunk(mini_libv2, {
        "id": "mini_ghost_chunk_999",
        "schema_version": "v4", "chunk_type": "explanation",
        "concept_tags": ["chunking"],
        "text": ("Chunking strategies split documents into overlapping "
                 "passages before embedding for retrieval."),
        "learning_outcome_refs": [],
        "source": {"item_path": "ghost.html", "html_xpath": "/html[1]/body[1]",
                   "char_span": [0, 10], "section_heading": "Ghost"},
    })
    client = FakeAnswerClient([
        _envelope("Chunking splits documents into overlapping passages.",
                  ["mini_ghost_chunk_999"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "chunking strategies overlapping passages",
        client=client, prune_mode="on",
    )
    assert result.status == STATUS_BLOCKED_CITATION_GATE
    assert result.citations == []


def test_shadow_mode_captures_and_excerpts_but_does_not_mutate(mini_libv2: Path):
    """Shadow mode: the claim-less citation is NOT removed, but the capture +
    warnings + supporting_excerpt are still produced (the audit/UX surfaces)."""
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER,
                  ["mini_alpha_chunk_001", "mini_alpha_chunk_003"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store recall at k embedding nearest neighbour passages",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="shadow",
    )
    assert result.status == STATUS_ANSWERED
    kept_ids = {c.chunk_id for c in result.citations}
    # Shadow: BOTH cited chunks remain (no prune, no add mutation).
    assert kept_ids == {"mini_alpha_chunk_001", "mini_alpha_chunk_003"}
    # But the excerpt + count still stamp on the supporter.
    supporter = next(c for c in result.citations
                     if c.chunk_id == "mini_alpha_chunk_001")
    assert supporter.supporting_excerpt
    assert supporter.supported_claim_count >= 1
    # And the capture fired (the claim-less id surfaces in the rationale).
    ev = _prune_events(spy)
    assert len(ev) == 1
    assert "mini_alpha_chunk_003" in ev[0]["rationale"]
    assert "mode=shadow" in ev[0]["rationale"]


def test_off_mode_skips_attribution_entirely(mini_libv2: Path):
    """off mode: no prune, no add, no capture, no attribution warnings."""
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER,
                  ["mini_alpha_chunk_001", "mini_alpha_chunk_003"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store recall at k embedding nearest neighbour passages",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        prune_mode="off",
    )
    assert result.status == STATUS_ANSWERED
    assert {c.chunk_id for c in result.citations} == {
        "mini_alpha_chunk_001", "mini_alpha_chunk_003"}
    assert _prune_events(spy) == []
    assert not any("claimless" in w or "added_supporting" in w
                   for w in result.warnings)


def test_to_dict_carries_attribution_keys(mini_libv2: Path):
    client = FakeAnswerClient([
        _envelope(_DEF_ANSWER, ["mini_alpha_chunk_001"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, prune_mode="on",
    )
    cit = result.to_dict()["citations"][0]
    assert "supporting_excerpt" in cit and "supported_claim_count" in cit
    assert cit["supported_claim_count"] >= 1
    json.dumps(result.to_dict())


# --------------------------------------------------------------------------- #
# Typed-error propagation
# --------------------------------------------------------------------------- #


def test_backend_unavailable_propagates(mini_libv2: Path):
    import httpx

    client = FakeAnswerClient([httpx.ConnectError("connection refused")])
    with pytest.raises(AnswerBackendUnavailable):
        answer_course_question(
            mini_libv2, COURSE_SLUG, "What does a vector store index?",
            client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        )


def test_semantic_engine_missing_index_propagates(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    # Monkeypatch retrieve_chunks to raise a SemanticIndexMissing-shaped typed
    # error for engine="semantic"; the pipeline must NOT swallow or downgrade.
    import LibV2.tools.libv2.retriever as retr_mod

    class SemanticIndexMissing(RuntimeError):
        pass

    def _fake_retrieve(repo_root, query, *, course_slug=None, limit=10,
                       engine="lexical", **kwargs):
        if engine != "lexical":
            raise SemanticIndexMissing("no on-device vector index for course")
        return []

    monkeypatch.setattr(retr_mod, "retrieve_chunks", _fake_retrieve)
    client = FakeAnswerClient([_envelope("x", ["x"])])
    with pytest.raises(SemanticIndexMissing):
        answer_course_question(
            mini_libv2, COURSE_SLUG, "q", engine="semantic", client=client,
        )
    # The composer must never have been reached.
    assert len(client.calls) == 0


def test_semantic_engine_no_engine_param_raises_runtime_error(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    # Simulate a pre-E3 tree whose retrieve_chunks lacks the `engine` kwarg:
    # a non-lexical engine must raise (never silently downgrade to lexical).
    import LibV2.tools.libv2.retriever as retr_mod

    def _legacy_retrieve(repo_root, query, *, course_slug=None, limit=10,
                         **kwargs):
        return []

    monkeypatch.setattr(retr_mod, "retrieve_chunks", _legacy_retrieve)
    client = FakeAnswerClient([_envelope("x", ["x"])])
    with pytest.raises(RuntimeError, match="engine"):
        answer_course_question(
            mini_libv2, COURSE_SLUG, "q", engine="semantic", client=client,
        )


# --------------------------------------------------------------------------- #
# Offline guard (master-plan constraint)
# --------------------------------------------------------------------------- #


def test_full_pipeline_runs_under_offline_guard(mini_libv2: Path):
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_001"])
    ])
    with no_network(allow_loopback=True):
        result = answer_course_question(
            mini_libv2, COURSE_SLUG, "What does a vector store index?",
            client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        )
    assert result.status == STATUS_ANSWERED


def test_pipeline_has_no_hidden_non_loopback_dependency(mini_libv2: Path):
    # With loopback disallowed too, the ONLY thing that could raise is a real
    # network touch — the fake client never opens a socket, so the run must
    # succeed, proving retrieve/anchor/refusal code paths are network-free.
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.",
                  ["mini_alpha_chunk_001"])
    ])
    with no_network(allow_loopback=False):
        result = answer_course_question(
            mini_libv2, COURSE_SLUG, "What does a vector store index?",
            client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        )
    assert result.status == STATUS_ANSWERED


# --------------------------------------------------------------------------- #
# Contract shape (to_dict keys frozen)
# --------------------------------------------------------------------------- #


def test_grounded_answer_to_dict_keys_frozen(mini_libv2: Path):
    client = FakeAnswerClient([
        _envelope("Answer.", ["mini_alpha_chunk_001"])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    d = result.to_dict()
    assert set(d.keys()) == {
        "status", "query", "course_slug", "engine", "answer_text",
        "citations", "refusal", "confidence", "groundedness", "warnings",
        "model_id", "prompt_version", "generated_at", "latency_ms",
        # NLI-ADD shadow diagnostics (additive, optional; None on the default
        # off path — present only when ED4ALL_ANSWER_NLI_ADD is shadow/on).
        "nli_citation_add",
    }
    # Default (off) path: the additive block is present-but-None.
    assert d["nli_citation_add"] is None
    # Citation dict shape (the WS4 rendering contract).
    cit = d["citations"][0]
    assert set(cit.keys()) == {
        "chunk_id", "item_path", "section_heading", "module_id", "page_label",
        "anchor_status", "source_path", "text_quote", "link_target",
        # Provenance-chain fields (additive, optional).
        "source_block", "pdf_pages",
        # Display title (additive, optional): renderers prefer it over the
        # filename-stem module_id, which repeats across weeks.
        "module_title",
        # Claim-attribution surfacing (additive, optional).
        "supporting_excerpt", "supported_claim_count",
    }
    # Legacy fixture chunk carries no source_references → provenance absent.
    assert cit["source_block"] is None
    assert cit["pdf_pages"] == []
    assert set(cit["link_target"].keys()) == {
        "kind", "item_path", "fragment", "char_span",
    }
    # to_dict is JSON-serializable (the CLI/GUI emit it verbatim).
    json.dumps(d)


def test_refused_payload_shape(mini_libv2: Path):
    client = FakeAnswerClient([_envelope("x", ["x"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_STRICT_LEXICAL,
    )
    d = result.to_dict()
    assert d["answer_text"] is None
    assert d["citations"] == []
    assert d["refusal"]["reason_code"] == REASON_LOW_CONFIDENCE
    assert "policy_version" in d["refusal"]
    json.dumps(d)


# --------------------------------------------------------------------------- #
# page_label_for_source helper
# --------------------------------------------------------------------------- #


def test_page_label_prefers_section_heading():
    assert page_label_for_source(
        {"section_heading": "Vector Stores", "item_path": "alpha.html"}
    ) == "Vector Stores"


def test_page_label_falls_back_to_prettified_stem():
    assert page_label_for_source(
        {"item_path": "module/intro_to_chunking.html"}
    ) == "Intro To Chunking"


def test_page_label_empty_source():
    assert page_label_for_source({}) == "Source"


# --------------------------------------------------------------------------- #
# Source PDF page surfaced in the citation label
# --------------------------------------------------------------------------- #


def test_page_label_appends_pdf_page_from_source_references():
    """A source whose refs carry pages → the label appends the page citation."""
    from lib.page_label import page_citation

    source = {
        "section_heading": "Vector Stores",
        "source_references": [
            {"sourceId": "semantik:x#s1_c0", "role": "primary", "pages": [12]},
        ],
    }
    label = page_label_for_source(source)
    # Physical pages (no pages_kind on the ref) → honest "PDF p. N".
    assert label == "Vector Stores (PDF p. 12)"
    # Reuses the shared formatter verbatim (no reimplemented formatting).
    assert page_citation([12], kind="physical") in label


def test_page_label_aggregates_pages_across_multiple_refs():
    """Pages union across ALL refs, de-duplicated + sorted, before formatting."""
    from lib.page_label import page_citation

    source = {
        "item_path": "module/intro_to_chunking.html",
        "source_references": [
            {"sourceId": "semantik:x#s1_c0", "role": "primary", "pages": [7, 12, 7]},
            {"sourceId": "semantik:x#s2_c1", "role": "contributing", "pages": [3, 12]},
        ],
    }
    label = page_label_for_source(source)
    assert label == "Intro To Chunking ({})".format(
        page_citation([3, 7, 12], kind="physical")
    )
    assert label == "Intro To Chunking (PDF pp. 3, 7, 12)"


def test_page_label_no_pages_is_byte_identical():
    """No pages present → label is byte-identical to the pre-change behavior."""
    # Empty / absent source_references, and a ref with an empty page list.
    assert page_label_for_source(
        {"section_heading": "Vector Stores"}
    ) == "Vector Stores"
    assert page_label_for_source(
        {
            "section_heading": "Vector Stores",
            "source_references": [
                {"sourceId": "semantik:x#s1_c0", "role": "primary", "pages": []},
            ],
        }
    ) == "Vector Stores"
    assert page_label_for_source(
        {"item_path": "module/intro_to_chunking.html"}
    ) == "Intro To Chunking"


# --------------------------------------------------------------------------- #
# Semantic engine + chunkset_kind=None aligns the citation gate with the vector
# index's manifest (NOT the directory-presence heuristic).
# --------------------------------------------------------------------------- #


# Heading + body of the imscc chunk. The query below is the index's exact
# text+heading projection so the deterministic fake embedder
# yields cosine ~1.0 (clearing the default min_similarity floor) — isolating
# the chunkset-kind/citation-gate behavior from fake-cosine luck.
_IMSCC_CHUNK_HEADING = "Vector Stores and Embeddings"
_IMSCC_CHUNK_TEXT = (
    "A vector store is a database that indexes high-dimensional "
    "embedding vectors so that approximate nearest-neighbour search "
    "can retrieve semantically similar passages."
)
# Matches vector_index._embed_text default policy "text+heading": heading\ntext.
_IMSCC_INDEX_QUERY = f"{_IMSCC_CHUNK_HEADING}\n{_IMSCC_CHUNK_TEXT}"


def _build_imscc_index_and_chunks(libv2_root: Path) -> str:
    """Set up the multi-chunkset misalignment: semantik_chunks/ present (from
    the fixture) AND an imscc-pinned vector index over imscc_chunks/.

    Returns the imscc chunk_id the model will cite. The imscc chunk's
    item_path (``alpha.html``) resolves as an imscc member but NOT under the
    semantik roots (we drop ``source/html`` so a semantik-kind gate fails with
    source_page_missing). So:
      * directory heuristic (_infer_chunkset_kind) -> "semantik" -> gate BLOCKS
      * vector-index manifest (the fix)            -> "imscc"    -> gate PASSES
    """
    from lib.embedding.providers import build_embedding_client
    from LibV2.tools.libv2.retrieval.vector_index import build_vector_index

    course_dir = libv2_root / "courses" / COURSE_SLUG

    # Drop the semantik HTML sources so a semantik-kind anchor resolution fails. The
    # imscc archive (source/imscc/mini.imscc) still carries alpha.html.
    shutil.rmtree(course_dir / "source" / "html")

    imscc_chunk_id = "mini_imscc_chunk_001"
    imscc_chunk = {
        "id": imscc_chunk_id,
        "schema_version": "v4",
        "chunk_type": "explanation",
        "concept_tags": ["vector-store", "embeddings"],
        "text": _IMSCC_CHUNK_TEXT,
        "learning_outcome_refs": [],
        "source": {
            "item_path": "alpha.html",
            "html_xpath": "/html[1]/body[1]",
            "section_heading": _IMSCC_CHUNK_HEADING,
            "module_id": "alpha",
            "course_id": "MINI_RETRIEVAL_101",
        },
    }
    imscc_chunks_dir = course_dir / "imscc_chunks"
    imscc_chunks_dir.mkdir(parents=True)
    (imscc_chunks_dir / "chunks.jsonl").write_text(
        json.dumps(imscc_chunk) + "\n", encoding="utf-8"
    )

    # Build the vector index PINNED to the imscc chunkset (manifest records
    # chunkset_kind="imscc"). Fake provider => no weights / no network.
    client = build_embedding_client(provider_name="fake")
    manifest = build_vector_index(course_dir, client=client, chunkset="imscc")
    assert manifest.chunkset_kind == "imscc"
    return imscc_chunk_id


def test_semantic_engine_chunkset_kind_from_index_manifest(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """semantic engine + chunkset_kind=None reads the chunkset from the vector
    index manifest, so the citation gate resolves against imscc (the index's
    kind) even though semantik_chunks/ is also present.

    Inferring chunkset_kind from directory presence instead would pick semantik
    (semantik_chunks/ wins) and resolve the gate against semantik ->
    source_page_missing -> blocked_citation_gate, despite the answer + citation
    being correct against the imscc-built index.
    """
    pytest.importorskip("numpy")  # builds a real vector index (needs [embedding])
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    imscc_chunk_id = _build_imscc_index_and_chunks(mini_libv2)

    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.", [imscc_chunk_id])
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _IMSCC_INDEX_QUERY,
        engine="semantic", client=client, refusal_policy=_PERMISSIVE_SEMANTIC,
    )
    # Gate resolved against imscc (the manifest's kind) -> answered.
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "A vector store indexes embedding vectors."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == imscc_chunk_id
    assert result.citations[0].anchor_status.startswith("resolved")


def test_semantic_engine_directory_heuristic_would_misroute(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pin the misalignment: the directory heuristic returns 'semantik' (so an
    explicit chunkset_kind='semantik' BLOCKS) — the index-manifest read in the
    previous test is what unblocks the correct answer."""
    pytest.importorskip("numpy")  # builds a real vector index (needs [embedding])
    from lib.retrieval.grounded_answer import _infer_chunkset_kind

    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    imscc_chunk_id = _build_imscc_index_and_chunks(mini_libv2)

    # The directory heuristic still picks semantik (semantik_chunks/ present).
    assert _infer_chunkset_kind(mini_libv2, COURSE_SLUG) == "semantik"

    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.", [imscc_chunk_id])
    ])
    # Forcing chunkset_kind='semantik' (what the heuristic would pick) BLOCKS,
    # because the semantik HTML source was dropped -> source_page_missing.
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _IMSCC_INDEX_QUERY,
        engine="semantic", client=client, chunkset_kind="semantik",
        refusal_policy=_PERMISSIVE_SEMANTIC,
    )
    assert result.status == STATUS_BLOCKED_CITATION_GATE


# --------------------------------------------------------------------------- #
# Refusal-policy pins wired onto the answer path
# --------------------------------------------------------------------------- #
#
# The pipeline resolves its refusal policy through resolve_policy((engine,
# embedding_model_id)) — NOT the bare engine default. These tests assert the
# pinned (engine, model) pairs land on the answer path, unknown models fall back
# to the v0-uncalibrated default, and lexical resolves model-agnostic (None).
# All three force a PRE-LLM refusal (zero client calls) so the resolved policy's
# version + embedding_model_id surface in result.refusal without an LLM.

from lib.retrieval.refusal import (  # noqa: E402
    PINNED_POLICIES,
    POLICY_VERSION_PINNED,
    POLICY_VERSION_UNCALIBRATED,
)

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _stub_low_score_retrieval(monkeypatch, score: float):
    """Stub retrieve_chunks to return one low-scoring passage for any engine.

    Keeps the resolved-policy assertions independent of the fake embedder's
    cosine luck: the single low score is below every pinned min_top_score, so
    the pipeline refuses PRE-LLM and the resolved policy surfaces in
    result.refusal.
    """
    import LibV2.tools.libv2.retriever as retr_mod

    class _Result:
        def __init__(self, s):
            self.score = s
            self.chunk_id = "mini_alpha_chunk_001"

    def _fake(repo_root, query, *, course_slug=None, limit=10, engine="lexical",
              **kwargs):
        return [_Result(score)]

    monkeypatch.setattr(retr_mod, "retrieve_chunks", _fake)


def test_answer_path_resolves_pinned_semantic_policy(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """A (semantic, <pinned model>) live index resolves the MEASURED pin on the
    answer path — not the permissive v0-uncalibrated semantic default."""
    import lib.retrieval.grounded_answer as ga

    # The live index manifest reports the pinned MiniLM embedder.
    monkeypatch.setattr(
        ga, "_vector_index_embedding_model_id", lambda root, slug: _MINILM
    )
    # Low cosine (below the pinned 0.369377 min_top_score) → pre-LLM refusal.
    _stub_low_score_retrieval(monkeypatch, 0.10)
    client = FakeAnswerClient([_envelope("never used", ["x"])])

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "q", engine="semantic", client=client,
    )
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert len(client.calls) == 0  # pre-LLM refusal
    pinned = PINNED_POLICIES[("semantic", _MINILM)]
    assert result.refusal["policy_version"] == POLICY_VERSION_PINNED
    assert result.refusal["policy_version"] == pinned.policy_version
    assert result.refusal["embedding_model_id"] == _MINILM


def test_answer_path_unknown_semantic_model_falls_back_to_uncalibrated(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """A semantic engine whose live index reports an UNKNOWN embedder must NOT
    reuse a pinned cosine — it falls back to the v0-uncalibrated default."""
    import lib.retrieval.grounded_answer as ga

    monkeypatch.setattr(
        ga, "_vector_index_embedding_model_id", lambda root, slug: "other/embed-v9"
    )
    # Below the v0-uncalibrated semantic min_top_score (0.30) → refusal.
    _stub_low_score_retrieval(monkeypatch, 0.05)
    client = FakeAnswerClient([_envelope("never used", ["x"])])

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "q", engine="semantic", client=client,
    )
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert result.refusal["policy_version"] == POLICY_VERSION_UNCALIBRATED
    # Uncalibrated default carries no embedder pin.
    assert result.refusal["embedding_model_id"] is None


def test_answer_path_lexical_resolves_pinned_model_none(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """The lexical engine is model-agnostic: resolve_policy(lexical) → the
    (lexical, None) pin on the answer path, never an embedder-keyed lookup."""
    # Below the pinned lexical 4.455352 min_top_score → pre-LLM refusal.
    _stub_low_score_retrieval(monkeypatch, 1.0)
    client = FakeAnswerClient([_envelope("never used", ["x"])])

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "q", engine="lexical", client=client,
    )
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert len(client.calls) == 0
    pinned = PINNED_POLICIES[("lexical", None)]
    assert result.refusal["policy_version"] == POLICY_VERSION_PINNED
    assert result.refusal["policy_version"] == pinned.policy_version
    assert result.refusal["embedding_model_id"] is None


def test_answer_path_hybrid_rrf_reads_embedder_and_applies_bge_large_pin(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """hybrid-rrf resolves the live index's embedder (its fused score depends on
    the semantic arm) and applies the (hybrid-rrf, bge-large) pin — a FUSED-RRF
    threshold, NOT the (semantic, bge-large) cosine pin. The two live on
    different score scales, so a pin must never leak across engines.

    Asserts the embedder IS consulted for hybrid-rrf AND that the
    engine-correct pin is applied with the embedder recorded on the verdict."""
    import lib.retrieval.grounded_answer as ga
    from lib.retrieval.refusal import PINNED_POLICIES

    seen = {}

    def _fake_model(root, slug):
        seen["called"] = True
        return _BGE_LARGE

    monkeypatch.setattr(ga, "_vector_index_embedding_model_id", _fake_model)
    # Avoid the index-manifest chunkset read picking a wrong kind; pin it.
    monkeypatch.setattr(
        ga, "_vector_index_chunkset_kind", lambda root, slug: "semantik"
    )
    # Below the pinned hybrid-rrf min_top_score (0.029643 on the RRF scale)
    # → pre-LLM refusal, so the resolved policy surfaces.
    hybrid_pin = PINNED_POLICIES[("hybrid-rrf", _BGE_LARGE)]
    _stub_low_score_retrieval(monkeypatch, 0.005)
    client = FakeAnswerClient([_envelope("never used", ["x"])])

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "q", engine="hybrid-rrf", client=client,
    )
    assert seen.get("called") is True  # the embedder WAS resolved for hybrid-rrf
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    # Pinned → the engine-correct measured pin (NOT the semantic cosine pin).
    assert result.refusal["policy_version"] == POLICY_VERSION_PINNED
    assert result.refusal["embedding_model_id"] == _BGE_LARGE
    # The applied threshold is the FUSED-RRF pin, not the semantic cosine pin.
    assert hybrid_pin.min_top_score < PINNED_POLICIES[
        ("semantic", _BGE_LARGE)
    ].min_top_score


_BGE_LARGE = "BAAI/bge-large-en-v1.5"


# --------------------------------------------------------------------------- #
# Default-path NLI never imported (DeBERTa stays off the query path)
# --------------------------------------------------------------------------- #


def test_default_path_does_not_import_transformers(mini_libv2: Path):
    import sys

    had_transformers = "transformers" in sys.modules
    client = FakeAnswerClient([
        _envelope("Answer.", ["mini_alpha_chunk_001"])
    ])
    answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client,  # with_groundedness defaults False
    )
    # The default path must not have newly imported transformers.
    if not had_transformers:
        assert "transformers" not in sys.modules


# --------------------------------------------------------------------------- #
# Groundedness — status-flip warning policy (_score_groundedness)
#
# The "contradicted_claim" warning is what flips an answer to
# answered_with_warnings (answer_course_question step 7). The scorer computes
# contradiction with whole-chunk false contradictions removed (windowed /
# single-topic semantics). These tests drive _score_groundedness directly with a
# fake NLI (injected via the process singleton) so the contradiction/rescue
# semantics are exercised without the heavy DeBERTa stack.
# --------------------------------------------------------------------------- #


def _gp_passage(chunk_id, text):
    from lib.retrieval.answer_composer import RetrievedPassage

    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text,
        score=1.0,
        engine="lexical",
        item_path="alpha.html",
        section_heading="Sec",
        module_id="m1",
        source={"item_path": "alpha.html"},
    )


class _FlipFakeNli:
    """Substring-entailment fake with a CONTRADICTS sentinel + a length budget
    that lets a short window rescue a claim a long premise cannot resolve.
    """

    _revision = "flip-fake-0"
    MAX_RESOLVABLE_CHARS = 130

    def score_batch(self, *, pairs, batch_size=8):
        from lib.tests.test_groundedness import _FakeNliScore

        out = []
        for premise, hypothesis in pairs:
            h = " ".join(hypothesis.lower().split())
            p = " ".join(premise.lower().split())
            if "contradicts" in h:
                out.append(_FakeNliScore(0.1, 0.3, 0.6))
            elif h and h in p and len(premise) <= self.MAX_RESOLVABLE_CHARS:
                out.append(_FakeNliScore(0.9, 0.05, 0.05))
            else:
                out.append(_FakeNliScore(0.1, 0.8, 0.1))
        return out


def _patch_singleton_nli(monkeypatch, fake):
    from lib.classifiers import nli_classifier

    monkeypatch.setattr(
        nli_classifier.NliClassifier,
        "get_or_load",
        classmethod(lambda cls: fake),
    )


def test_score_groundedness_emits_contradicted_warning_on_v2_contradiction(
    monkeypatch,
):
    from lib.retrieval.grounded_answer import _score_groundedness

    _patch_singleton_nli(monkeypatch, _FlipFakeNli())
    passage = _gp_passage("c1", "Embeddings are dense numerical vectors here.")
    report, warnings, report_obj = _score_groundedness(
        "This statement CONTRADICTS the cited passage about embeddings.",
        [passage],
        cited_chunk_ids={"c1"},
    )
    assert report is not None
    assert report_obj is not None  # additive 3rd return (the live report object)
    assert report["contradicted_count"] == 1
    assert "contradicted_claim" in warnings


def test_score_groundedness_no_warning_when_window_rescues(monkeypatch):
    from lib.retrieval.grounded_answer import _score_groundedness

    _patch_singleton_nli(monkeypatch, _FlipFakeNli())
    # Glossary-style multi-topic chunk: the whole-chunk premise cannot resolve
    # the claim (over the char budget) but a 3-sentence window can → stage-2
    # rescue entails it, so no contradicted/unsupported claim and no warning.
    glossary = (
        "Embeddings are dense numerical vectors. "
        "A vector store indexes them for search. "
        "Recall at k measures retrieval completeness. "
        "Precision at k measures retrieval correctness. "
        "Latency is the time to answer a query."
    )
    passage = _gp_passage("c1", glossary)
    report, warnings, _report_obj = _score_groundedness(
        "Recall at k measures retrieval completeness.",
        [passage],
        cited_chunk_ids={"c1"},
    )
    assert report is not None
    assert report["contradicted_count"] == 0
    assert report["groundedness_rate"] == 1.0
    assert report["claims"][0]["windowed"] is True
    assert "contradicted_claim" not in warnings


def test_status_flips_to_warnings_on_genuine_v2_contradiction(
    mini_libv2: Path, monkeypatch
):
    # End-to-end through answer_course_question: a genuine contradicted claim
    # flips status to answered_with_warnings.
    _patch_singleton_nli(monkeypatch, _FlipFakeNli())
    client = FakeAnswerClient([
        _envelope(
            "This statement CONTRADICTS the cited passage about vectors here.",
            ["mini_alpha_chunk_001"],
        )
    ])
    result = answer_course_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        client=client,
        refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True,
    )
    assert result.status == STATUS_ANSWERED_WITH_WARNINGS
    assert "contradicted_claim" in result.warnings


# =========================================================================== #
# NLI-based citation ADD (off by default; opt in to shadow|on)
# =========================================================================== #
#
# These tests drive the NLI-ADD arm (ED4ALL_ANSWER_NLI_ADD) which derives ADD
# candidates from the groundedness scorer's per-claim verdicts (NLI is never run
# twice) and gates them through the composite criterion. Unit-level tests call
# the pure helpers with fake verdict objects + fake passages (no model); the
# integration tests drive answer_course_question with a fake NLI singleton.

from dataclasses import dataclass as _dataclass
from typing import List as _List, Optional as _Optional

from lib.retrieval.answer_composer import RetrievedPassage as _RP
from lib.retrieval.grounded_answer import (
    DECISION_TYPE_NLI_CITATION_ADD,
    _apply_nli_citation_add,
    _nli_add_candidate_signals,
    Citation as _Citation,
)


@_dataclass
class _FakeVerdict:
    claim_text: str
    verdict: str
    entailment: float
    best_chunk_id: _Optional[str]
    best_chunk_cited: _Optional[bool]


@_dataclass
class _FakeGroundReport:
    available: bool
    claims: _List[_FakeVerdict]


def _nli_passage(chunk_id, text, item_path="alpha.html"):
    return _RP(
        chunk_id=chunk_id, text=text, score=1.0, engine="lexical",
        item_path=item_path, section_heading="Sec", module_id="m1",
        source={"item_path": item_path},
    )


def _cit(chunk_id):
    return _Citation(
        chunk_id=chunk_id, item_path="alpha.html", section_heading="Sec",
        module_id="m1", page_label="Sec", anchor_status="resolved_exact",
        source_path=None, text_quote=None,
    )


# A chunk text whose content tokens cover the claim well (>= 0.65) with NO
# numeric mismatch; the verdict entails it from an UNCITED chunk.
_COVER_CLAIM = "Add the numerators and place the result over the common denominator."
_COVER_CHUNK = (
    "To add fractions: add the numerators and place the result over the "
    "common denominator, then simplify."
)


def test_candidate_signals_composite_passes_all_legs():
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.82, "u1", False)],
    )
    by_id = {"u1": _nli_passage("u1", _COVER_CHUNK)}
    cands = _nli_add_candidate_signals(
        report, cited_chunk_ids=set(), gate_eligible_by_id=by_id
    )
    assert len(cands) == 1
    c = cands[0]
    assert c["chunk_id"] == "u1"
    assert c["entailment_ok"] and c["coverage_ok"] and c["numerics_present"]
    assert c["passes_composite"] is True


def test_candidate_signals_entailment_leg_rejects():
    # ent 0.65 < the 0.70 floor: the NLI leg fails even with perfect coverage.
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.65, "u1", False)],
    )
    by_id = {"u1": _nli_passage("u1", _COVER_CHUNK)}
    cands = _nli_add_candidate_signals(report, cited_chunk_ids=set(), gate_eligible_by_id=by_id)
    assert cands[0]["entailment_ok"] is False
    assert cands[0]["passes_composite"] is False


def test_candidate_signals_coverage_leg_rejects():
    # High entailment but the chunk shares almost none of the claim's tokens.
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.95, "u1", False)],
    )
    by_id = {"u1": _nli_passage("u1", "An unrelated paragraph about photosynthesis and sunlight.")}
    cands = _nli_add_candidate_signals(report, cited_chunk_ids=set(), gate_eligible_by_id=by_id)
    assert cands[0]["coverage_ok"] is False
    assert cands[0]["passes_composite"] is False


def test_candidate_signals_numeric_leg_rejects():
    # High entailment + high token coverage, but the claim's "$1.50" appears
    # nowhere in the chunk → the numeric leg rejects. Entailment alone is
    # number-agnostic, so without this leg the add would be a false credit.
    claim = "The price of one pen is $1.50 in the store."
    chunk = "The price of one pen is computed in the store exercise here."
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(claim, "entailed", 0.90, "u1", False)],
    )
    by_id = {"u1": _nli_passage("u1", chunk)}
    cands = _nli_add_candidate_signals(report, cited_chunk_ids=set(), gate_eligible_by_id=by_id)
    assert cands[0]["entailment_ok"] and cands[0]["coverage_ok"]
    assert cands[0]["numerics_present"] is False
    assert cands[0]["passes_composite"] is False


def test_candidate_signals_skips_cited_and_unentailed():
    report = _FakeGroundReport(
        available=True,
        claims=[
            _FakeVerdict(_COVER_CLAIM, "entailed", 0.90, "cited1", True),   # cited
            _FakeVerdict(_COVER_CLAIM, "unsupported", 0.50, "u1", False),    # not entailed
        ],
    )
    by_id = {"cited1": _nli_passage("cited1", _COVER_CHUNK), "u1": _nli_passage("u1", _COVER_CHUNK)}
    cands = _nli_add_candidate_signals(
        report, cited_chunk_ids={"cited1"}, gate_eligible_by_id=by_id
    )
    assert cands == []


def test_apply_off_mode_is_total_noop():
    cits = [_cit("a")]
    report = _FakeGroundReport(available=True, claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.9, "u1", False)])
    out, warns, diag = _apply_nli_citation_add(
        citations=cits, groundedness_report=report, cited_chunk_ids={"a"},
        gate_eligible_passages=[_nli_passage("u1", _COVER_CHUNK)],
        answer_text=_COVER_CLAIM, course_dir=Path("/nonexistent"),
        chunkset_kind="semantik", containment_threshold=0.85, mode="off",
        capture=None, course_slug="s", query_sha="qs", engine="lexical",
    )
    assert out is cits and warns == [] and diag is None


def test_apply_shadow_mode_unavailable_report_logs_reason():
    spy = SpyCapture()
    out, warns, diag = _apply_nli_citation_add(
        citations=[_cit("a")], groundedness_report=None, cited_chunk_ids={"a"},
        gate_eligible_passages=[_nli_passage("u1", _COVER_CHUNK)],
        answer_text=_COVER_CLAIM, course_dir=Path("/nonexistent"),
        chunkset_kind="semantik", containment_threshold=0.85, mode="shadow",
        capture=spy, course_slug="s", query_sha="qs", engine="lexical",
    )
    assert diag["outcome"] == "skipped_no_nli"
    assert any("nli_citation_add_skipped:" in w for w in warns)
    ev = [e for e in spy.events if e["decision_type"] == DECISION_TYPE_NLI_CITATION_ADD]
    assert len(ev) == 1 and "reason=" in ev[0]["rationale"]


# --------------------------------------------------------------------------- #
# Integration: full answer_course_question with a fake NLI singleton + fixture
# --------------------------------------------------------------------------- #

# The model cites a topical mention (chunk_003 = recall@k); the answer is the
# vector-store definition near-verbatim from the UNCITED chunk_001. A fake NLI
# that entails any answer-sentence whose content tokens are mostly contained in
# the premise makes chunk_001 the entailing uncited supporter.


class _SubstringEntailNli:
    """Entails (0.92) when >= 60% of the hypothesis content tokens appear in the
    premise; else low entailment. Number-agnostic (like a real NLI)."""

    _revision = "subnli-0"

    def score_batch(self, *, pairs, batch_size=8):
        from lib.tests.test_groundedness import _FakeNliScore
        import re as _re
        tok = lambda s: {t.lower() for t in _re.findall(r"[A-Za-z]{2,}", s)}
        out = []
        for premise, hypothesis in pairs:
            h = tok(hypothesis)
            p = tok(premise)
            frac = (len(h & p) / len(h)) if h else 0.0
            if frac >= 0.6:
                out.append(_FakeNliScore(0.92, 0.05, 0.03))
            else:
                out.append(_FakeNliScore(0.1, 0.8, 0.1))
        return out


def _nli_add_events(spy):
    return [e for e in spy.events
            if e["decision_type"] == DECISION_TYPE_NLI_CITATION_ADD]


def test_nli_shadow_mutates_nothing_but_captures_would_add(mini_libv2, monkeypatch):
    _patch_singleton_nli(monkeypatch, _SubstringEntailNli())
    monkeypatch.setenv("ED4ALL_ANSWER_NLI_ADD", "shadow")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_003"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors recall at k",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True, prune_mode="off",
    )
    # Shadow mutates NOTHING: the only citation remains the model's chunk_003.
    assert [c.chunk_id for c in result.citations] == ["mini_alpha_chunk_003"]
    # The would-add list surfaced chunk_001 (uncited entailed supporter).
    assert any("nli_would_add_citation:mini_alpha_chunk_001" in w
               for w in result.warnings)
    diag = result.nli_citation_add
    assert diag is not None and diag["mode"] == "shadow"
    assert "mini_alpha_chunk_001" in diag["would_add_ids"]
    assert diag["added_ids"] == []
    # The to_dict carries the additive block + the capture fired.
    assert result.to_dict()["nli_citation_add"]["mode"] == "shadow"
    ev = _nli_add_events(spy)
    assert len(ev) == 1
    assert ev[0]["decision"] == "nli_citation_add:would_add"
    assert "mini_alpha_chunk_001" in ev[0]["rationale"]
    assert ev[0]["decision_type"] in _enum_members()


def test_nli_on_mode_adds_with_anchor_and_cap(mini_libv2, monkeypatch):
    _patch_singleton_nli(monkeypatch, _SubstringEntailNli())
    monkeypatch.setenv("ED4ALL_ANSWER_NLI_ADD", "on")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_003"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors recall at k",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True, prune_mode="off",
    )
    ids = [c.chunk_id for c in result.citations]
    # On-mode appended the anchor-resolved uncited supporter AFTER the cited set.
    assert "mini_alpha_chunk_001" in ids
    added = next(c for c in result.citations if c.chunk_id == "mini_alpha_chunk_001")
    assert added.anchor_status.startswith("resolved")
    assert ids[0] == "mini_alpha_chunk_003"  # existing citation leads; add trails
    assert len(ids) <= 1 + 2  # cap of 2 NLI adds
    assert any("nli_added_citation:mini_alpha_chunk_001" in w for w in result.warnings)
    assert result.nli_citation_add["added_ids"] == ["mini_alpha_chunk_001"]


def test_nli_off_default_skips_entirely(mini_libv2, monkeypatch):
    _patch_singleton_nli(monkeypatch, _SubstringEntailNli())
    monkeypatch.delenv("ED4ALL_ANSWER_NLI_ADD", raising=False)  # default off
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_003"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors recall at k",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True, prune_mode="off",
    )
    assert result.nli_citation_add is None
    assert _nli_add_events(spy) == []
    assert not any("nli_" in w for w in result.warnings)


def test_nli_shadow_without_groundedness_logs_skip_reason(mini_libv2, monkeypatch):
    # NLI-ADD shadow on, but with_groundedness OFF → no per-claim verdicts to
    # reuse; the arm does nothing and logs WHY (it never runs NLI itself).
    monkeypatch.setenv("ED4ALL_ANSWER_NLI_ADD", "shadow")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_001"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "vector store database indexes embedding vectors",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=False, prune_mode="off",
    )
    diag = result.nli_citation_add
    assert diag is not None and diag["outcome"] == "skipped_no_nli"
    assert any("nli_citation_add_skipped:with_groundedness_off" in w
               for w in result.warnings)
    ev = _nli_add_events(spy)
    assert len(ev) == 1 and ev[0]["decision"] == "nli_citation_add:skipped_no_nli"


def test_apply_zero_citation_answer_exclusion_is_warning_class(monkeypatch):
    """PRUNE-TO-EMPTY EXCLUSION: on a zero-citation (prune-to-empty) answer,
    would-adds are treated as a distinct warning class — shadow surfaces them
    under their own outcome/prefix, and ON mode NEVER restores a citation to
    such an answer (no sources beats a misleading source)."""
    import lib.retrieval.grounded_answer as ga
    from lib.retrieval.citation_anchor import AnchorStatus, CitationAnchor

    def _resolved_anchor(record, course_dir, **kwargs):
        return CitationAnchor(
            chunk_id=str(record.get("id", "u1")),
            status=AnchorStatus.RESOLVED_EXACT,
            source_path=None,
            item_path="alpha.html",
            html_xpath="/html[1]/body[1]",
            char_span=(0, 10),
            containment_rate=1.0,
            normalized_match=False,
        )

    monkeypatch.setattr(ga, "resolve_citation_anchor", _resolved_anchor)
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.9, "u1", False)],
    )
    for mode in ("shadow", "on"):
        out, warns, diag = _apply_nli_citation_add(
            citations=[], groundedness_report=report, cited_chunk_ids=set(),
            gate_eligible_passages=[_nli_passage("u1", _COVER_CHUNK)],
            answer_text=_COVER_CLAIM, course_dir=Path("/nonexistent"),
            chunkset_kind="semantik", containment_threshold=0.85, mode=mode,
            capture=None, course_slug="s", query_sha="qs", engine="lexical",
        )
        assert out == []  # never restores a citation, even in ON mode
        assert diag["zero_citation_answer"] is True
        assert diag["outcome"] == "would_add_on_zero_citation_answer"
        assert diag["added_ids"] == []
        assert any(
            w.startswith("nli_would_add_on_zero_citation_answer:") for w in warns
        )


# --------------------------------------------------------------------------- #
# Flip-gating guardrail invariants: the apply path
# must (a) cap adds at NLI_ADD_MAX_ADDED_CITATIONS, (b) never add an
# un-anchorable candidate (recorded anchor_resolved=False), and (c) never change
# an answer's status/verdict (it only ever APPENDS to the citation list).
# --------------------------------------------------------------------------- #


def _three_passing_report():
    """A groundedness report with THREE distinct uncited entailing supporters,
    each clearing the composite criterion (ent>=0.70, cov>=0.80, no numerics)."""
    return _FakeGroundReport(
        available=True,
        claims=[
            _FakeVerdict(_COVER_CLAIM, "entailed", 0.95, "u1", False),
            _FakeVerdict(_COVER_CLAIM, "entailed", 0.90, "u2", False),
            _FakeVerdict(_COVER_CLAIM, "entailed", 0.85, "u3", False),
        ],
    )


def _always_resolved_anchor(monkeypatch):
    import lib.retrieval.grounded_answer as ga
    from lib.retrieval.citation_anchor import AnchorStatus, CitationAnchor

    def _resolved(record, course_dir, **kwargs):
        return CitationAnchor(
            chunk_id=str(record.get("id", "u1")),
            status=AnchorStatus.RESOLVED_EXACT,
            source_path=None, item_path="alpha.html",
            html_xpath="/html[1]/body[1]", char_span=(0, 10),
            containment_rate=1.0, normalized_match=False,
        )

    monkeypatch.setattr(ga, "resolve_citation_anchor", _resolved)


def test_apply_on_mode_caps_added_citations_at_two(monkeypatch):
    """Three composite-passing anchor-resolved candidates → exactly 2 added
    (NLI_ADD_MAX_ADDED_CITATIONS), never 3. The would-add list is also capped so
    the shadow signal reports what ON WOULD do, not an unbounded count."""
    from lib.retrieval.citation_attribution import NLI_ADD_MAX_ADDED_CITATIONS

    _always_resolved_anchor(monkeypatch)
    report = _three_passing_report()
    passages = [
        _nli_passage("u1", _COVER_CHUNK),
        _nli_passage("u2", _COVER_CHUNK),
        _nli_passage("u3", _COVER_CHUNK),
    ]
    out, warns, diag = _apply_nli_citation_add(
        citations=[_cit("seed")], groundedness_report=report,
        cited_chunk_ids={"seed"}, gate_eligible_passages=passages,
        answer_text=_COVER_CLAIM, course_dir=Path("/nonexistent"),
        chunkset_kind="semantik", containment_threshold=0.85, mode="on",
        capture=None, course_slug="s", query_sha="qs", engine="lexical",
    )
    assert NLI_ADD_MAX_ADDED_CITATIONS == 2
    assert len(diag["added_ids"]) == 2
    assert len(diag["would_add_ids"]) == 2
    # The seed citation is preserved and leads; exactly 2 adds trail it.
    added = [c.chunk_id for c in out if c.chunk_id != "seed"]
    assert out[0].chunk_id == "seed"
    assert len(added) == 2
    assert len(out) == 3


def test_apply_on_mode_anchor_fail_records_false_and_never_adds(monkeypatch):
    """A candidate that passes the composite legs but FAILS to anchor is recorded
    anchor_resolved=False and is NEVER added (an add can never introduce an
    unresolvable citation the gate would otherwise have blocked)."""
    import lib.retrieval.grounded_answer as ga
    from lib.retrieval.citation_anchor import AnchorStatus, CitationAnchor

    def _unresolved(record, course_dir, **kwargs):
        # SPAN_FABRICATED is a NON-resolved status (not in _RESOLVED_STATUSES),
        # so anchor_ok is False — the candidate must never be added.
        return CitationAnchor(
            chunk_id=str(record.get("id", "u1")),
            status=AnchorStatus.SPAN_FABRICATED,
            source_path=None, item_path="alpha.html",
            html_xpath=None, char_span=None,
            containment_rate=0.0, normalized_match=False,
        )

    monkeypatch.setattr(ga, "resolve_citation_anchor", _unresolved)
    report = _FakeGroundReport(
        available=True,
        claims=[_FakeVerdict(_COVER_CLAIM, "entailed", 0.95, "u1", False)],
    )
    out, warns, diag = _apply_nli_citation_add(
        citations=[_cit("seed")], groundedness_report=report,
        cited_chunk_ids={"seed"}, gate_eligible_passages=[_nli_passage("u1", _COVER_CHUNK)],
        answer_text=_COVER_CLAIM, course_dir=Path("/nonexistent"),
        chunkset_kind="semantik", containment_threshold=0.85, mode="on",
        capture=None, course_slug="s", query_sha="qs", engine="lexical",
    )
    # Composite passed, but anchor failed → recorded False, not added.
    cand = next(c for c in diag["candidates"] if c["chunk_id"] == "u1")
    assert cand["passes_composite"] is True
    assert cand["anchor_resolved"] is False
    assert diag["added_ids"] == []
    assert diag["would_add_ids"] == []
    assert [c.chunk_id for c in out] == ["seed"]  # citation list unchanged


def test_on_mode_never_changes_answer_status(mini_libv2, monkeypatch):
    """The flip-gating invariant: ON mode only ever APPENDS citations — it must
    not change the answer's status/verdict vs the same run with NLI-ADD off."""
    client_off = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_003"])])
    _patch_singleton_nli(monkeypatch, _SubstringEntailNli())
    monkeypatch.delenv("ED4ALL_ANSWER_NLI_ADD", raising=False)  # default off
    baseline = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors recall at k",
        client=client_off, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True, prune_mode="off",
    )

    client_on = FakeAnswerClient([_envelope(_DEF_ANSWER, ["mini_alpha_chunk_003"])])
    _patch_singleton_nli(monkeypatch, _SubstringEntailNli())
    monkeypatch.setenv("ED4ALL_ANSWER_NLI_ADD", "on")
    on = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "vector store database indexes embedding vectors recall at k",
        client=client_on, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True, prune_mode="off",
    )
    # Status + answer text are byte-identical; ON only grew the citation list.
    assert on.status == baseline.status
    assert on.answer_text == baseline.answer_text
    assert len(on.citations) >= len(baseline.citations)
    # The model's original citation is still present (an add appends, never drops).
    assert "mini_alpha_chunk_003" in [c.chunk_id for c in on.citations]


# --------------------------------------------------------------------------- #
# Completeness recheck — a single bounded re-ask when a multi-part question
# leaves a GROUNDED sub-question unanswered.
# validate_citations=False isolates the recheck (it runs pre-gate) from the
# anchor resolver, so these synthetic geo chunks need no backing HTML.
# --------------------------------------------------------------------------- #

_GEO_PERIMETER_CHUNK = {
    "id": "mini_geo_perimeter",
    "schema_version": "v4",
    "chunk_type": "explanation",
    "concept_tags": ["perimeter", "rectangle"],
    "text": (
        "Perimeter of a rectangle. The perimeter of a rectangle is the distance "
        "around it: add all four sides. Use the formula P = 2L + 2W where L is "
        "the length and W is the width of the rectangle."
    ),
    "learning_outcome_refs": [],
    "source": {
        "item_path": "geo/perimeter.html",
        "section_heading": "Perimeter",
        "module_id": "geo",
        "course_id": "MINI_RETRIEVAL_101",
    },
}

_GEO_CIRCLE_CHUNK = {
    "id": "mini_geo_circle",
    "schema_version": "v4",
    "chunk_type": "explanation",
    "concept_tags": ["circumference", "circle"],
    "text": (
        "Circumference of a circle. The circumference of a circle is the distance "
        "around the circle. Use the formula C = 2 pi r where r is the radius of "
        "the circle, or equivalently C = pi d with diameter d."
    ),
    "learning_outcome_refs": [],
    "source": {
        "item_path": "geo/circle.html",
        "section_heading": "Circumference",
        "module_id": "geo",
        "course_id": "MINI_RETRIEVAL_101",
    },
}

_MULTIPART_GEO_QUERY = (
    "how do I find the perimeter of a rectangle? the circumference of a circle?"
)


def _recheck_events(spy):
    return [e for e in spy.events
            if e["decision_type"] == DECISION_TYPE_COMPLETENESS_RECHECK]


def test_completeness_recheck_reasks_and_merges_missing_part(mini_libv2: Path):
    """First pass answers only the rectangle half; the circumference half is
    grounded in a passage, so the recheck fires ONE re-ask and merges it in."""
    _append_chunk(mini_libv2, _GEO_PERIMETER_CHUNK)
    _append_chunk(mini_libv2, _GEO_CIRCLE_CHUNK)
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("To find the perimeter of a rectangle, use P = 2L + 2W.",
                  ["mini_geo_perimeter"]),
        _envelope("The circumference of a circle is C = 2 pi r.",
                  ["mini_geo_circle"]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _MULTIPART_GEO_QUERY,
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        validate_citations=False,
    )
    assert result.status == STATUS_ANSWERED
    # The merged answer now covers BOTH parts.
    assert "perimeter" in result.answer_text.lower()
    assert "circumference" in result.answer_text.lower()
    # Two LLM calls: the initial compose + the single bounded re-ask.
    assert len(client.calls) == 2
    assert "completeness_reasked" in result.warnings
    # Unioned citations carry both parts' supporters.
    cited = {c.chunk_id for c in result.citations}
    assert {"mini_geo_perimeter", "mini_geo_circle"} <= cited
    # Capture fired once with a dynamic, replayable rationale.
    ev = _recheck_events(spy)
    assert len(ev) == 1
    assert ev[0]["decision_type"] in _enum_members()
    assert ev[0]["decision"] == "completeness_recheck:reask:covered"
    rat = ev[0]["rationale"]
    assert len(rat) >= 20
    assert "circumference" in rat.lower()          # names the uncovered part
    assert "reasked=True" in rat and "adopted=True" in rat


def test_completeness_recheck_noop_when_both_parts_addressed(mini_libv2: Path):
    """A two-part question answered completely on the first pass: the recheck
    runs, finds nothing uncovered-but-grounded, and does NOT re-ask."""
    _append_chunk(mini_libv2, _GEO_PERIMETER_CHUNK)
    _append_chunk(mini_libv2, _GEO_CIRCLE_CHUNK)
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope(
            "To find the perimeter of a rectangle use P = 2L + 2W. "
            "The circumference of a circle is C = 2 pi r.",
            ["mini_geo_perimeter", "mini_geo_circle"]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _MULTIPART_GEO_QUERY,
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        validate_citations=False,
    )
    assert result.status == STATUS_ANSWERED
    assert len(client.calls) == 1            # no re-ask
    assert "completeness_reasked" not in result.warnings
    ev = _recheck_events(spy)
    assert len(ev) == 1
    assert ev[0]["decision"].startswith("completeness_recheck:noop")
    assert "reasked=False" in ev[0]["rationale"]


def test_completeness_recheck_noop_when_uncovered_part_ungrounded(mini_libv2: Path):
    """Only the perimeter chunk exists: the unanswered circumference part has no
    supporting passage, so the recheck does NOT re-ask (re-asking would refuse)."""
    _append_chunk(mini_libv2, _GEO_PERIMETER_CHUNK)  # NO circle chunk
    spy = SpyCapture()
    client = FakeAnswerClient([
        _envelope("To find the perimeter of a rectangle, use P = 2L + 2W.",
                  ["mini_geo_perimeter"]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _MULTIPART_GEO_QUERY,
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        validate_citations=False,
    )
    assert result.status == STATUS_ANSWERED
    assert len(client.calls) == 1            # no re-ask
    assert "completeness_reasked" not in result.warnings
    ev = _recheck_events(spy)
    assert len(ev) == 1
    assert ev[0]["decision"] == "completeness_recheck:noop:all_addressed_or_ungrounded"


# --------------------------------------------------------------------------- #
# groundedness_computational_check capture threads through the
# PRODUCTION grounded-answer path (not just the eval path). The capture is
# threaded into ``_score_groundedness`` -> ``score_groundedness(capture=...)``.
# --------------------------------------------------------------------------- #

_COMP_ANSWER = "The computed sum is 3 plus 10 equals 13 for this vector problem."


def _groundedness_comp_events(spy: SpyCapture) -> list:
    return [e for e in spy.events
            if e["decision_type"] == "groundedness_computational_check"]


def test_groundedness_capture_threads_on_production_path(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """With ED4ALL_GROUNDEDNESS_COMPUTATIONAL on + a computational claim in the
    answer, the production answer path threads the DecisionCapture into
    score_groundedness so exactly one groundedness_computational_check decision
    fires with a dynamic >=20-char rationale."""
    from lib.tests.test_groundedness import FakeNli

    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_COMPUTATIONAL", "1")
    # Inject a deterministic NLI so the scorer runs (and reaches comp_check)
    # without the ~750MB DeBERTa stack.
    monkeypatch.setattr(
        "lib.retrieval.groundedness._resolve_nli", lambda nli: FakeNli()
    )
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_COMP_ANSWER, ["mini_alpha_chunk_001"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True,
    )
    assert result.status in (STATUS_ANSWERED, STATUS_ANSWERED_WITH_WARNINGS)
    ev = _groundedness_comp_events(spy)
    assert len(ev) == 1
    assert len(ev[0]["rationale"]) >= 20


def test_groundedness_capture_silent_when_flag_off(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """Flag OFF ⇒ no groundedness_computational_check decision fires even though
    groundedness scoring runs."""
    from lib.tests.test_groundedness import FakeNli

    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_COMPUTATIONAL", raising=False)
    monkeypatch.setattr(
        "lib.retrieval.groundedness._resolve_nli", lambda nli: FakeNli()
    )
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_COMP_ANSWER, ["mini_alpha_chunk_001"])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, capture=spy, refusal_policy=_PERMISSIVE_LEXICAL,
        with_groundedness=True,
    )
    assert result.status in (STATUS_ANSWERED, STATUS_ANSWERED_WITH_WARNINGS)
    assert _groundedness_comp_events(spy) == []


# --------------------------------------------------------------------------- #
# Grounded-answer chunk_type exclusion filter + tunable citation-gate
# anchor-containment floor. Both exist for courses whose index carries
# QTI-harvested assessment_item chunks: their text lives in QTI XML, not in any
# archived HTML page, so they can never pass the anchor gate and a model
# citation of one would withhold the whole answer.
# --------------------------------------------------------------------------- #

from lib.retrieval.answer_composer import ComposedAnswer  # noqa: E402
from lib.retrieval import grounded_answer as _ga  # noqa: E402
from lib.retrieval.grounded_answer import (  # noqa: E402
    overfetch_for_exclude,
    resolve_anchor_containment,
    resolve_exclude_chunk_types,
)


class _FakeCTResult:
    """Duck-typed LibV2 RetrievalResult carrying a first-class ``chunk_type``."""

    def __init__(self, idx: int, chunk_type: str):
        self.chunk_id = f"c{idx}"
        self.text = f"passage {idx} about vector stores and embeddings"
        self.score = 5.0  # clears the permissive lexical policy
        self.chunk_type = chunk_type
        self.item_path = "page.html"
        self.section_heading = None
        self.module_id = None
        self.source = {"item_path": "page.html"}


def _interleaved_pool(n: int = 24):
    # even idx -> assessment_item (the QTI-harvested span-fabricated kind),
    # odd idx -> explanation (genuine instructional prose).
    return [
        _FakeCTResult(i, "assessment_item" if i % 2 == 0 else "explanation")
        for i in range(n)
    ]


def _spy_compose_factory(captured: dict):
    def _spy(query, passages, *, client, capture, course_code, max_passages):
        captured["passages"] = list(passages)
        cid = passages[0].chunk_id if passages else "none"
        return ComposedAnswer(
            answer_text="A.",
            cited_chunk_ids=[cid],
            not_in_course=False,
            model_id="fake",
            prompt_version="v-test",
            attempts=1,
            latency_ms=0.0,
            raw_response_len=2,
            allowed_chunk_ids=[cid],
        )

    return _spy


def test_fix_ab_env_unset_no_filtering_byte_identical(mini_libv2, monkeypatch):
    """(1) Env unset → 0.85 floor + NO filtering: the composer sees exactly the
    first `limit` candidates (assessment_items included), no over-fetch."""
    monkeypatch.delenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", raising=False)
    monkeypatch.delenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", raising=False)
    assert resolve_anchor_containment() == 0.85
    assert resolve_exclude_chunk_types() == frozenset()

    pool = _interleaved_pool(24)
    fetched = {}

    def _fake_retrieve(root, slug, q, *, engine, limit):
        fetched["limit"] = limit
        return pool[:limit]

    monkeypatch.setattr(_ga, "_retrieve", _fake_retrieve)
    captured = {}
    monkeypatch.setattr(_ga, "compose_answer", _spy_compose_factory(captured))

    _ga.answer_course_question(
        mini_libv2, COURSE_SLUG, "vector stores",
        client=object(), refusal_policy=_PERMISSIVE_LEXICAL,
        completeness_recheck=False,
    )
    # Byte-identical: retriever asked for exactly `limit` (8), no over-fetch.
    assert fetched["limit"] == 8
    seen = [p.chunk_id for p in captured["passages"]]
    assert seen == [f"c{i}" for i in range(8)]
    # The assessment_item candidates are RETAINED on the default path.
    assert "c0" in seen


def test_fix_a_resolve_exclude_chunk_types_parsing(monkeypatch):
    """(2) Exclude-list parsing incl. whitespace / blank / garbage tokens."""
    monkeypatch.delenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", raising=False)
    assert resolve_exclude_chunk_types() == frozenset()
    monkeypatch.setenv(
        "ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", "  assessment_item , ,Summary ,, "
    )
    assert resolve_exclude_chunk_types() == frozenset({"assessment_item", "summary"})
    # All-blank / whitespace → empty set (no filtering).
    monkeypatch.setenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", "   ")
    assert resolve_exclude_chunk_types() == frozenset()
    monkeypatch.setenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", " , , ")
    assert resolve_exclude_chunk_types() == frozenset()
    # Explicit arg wins over env.
    monkeypatch.setenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", "assessment_item")
    assert resolve_exclude_chunk_types("x , Y") == frozenset({"x", "y"})
    # Over-fetch helper: inactive → verbatim; active → min(limit*3, 50) >= limit.
    assert overfetch_for_exclude(8, frozenset()) == 8
    assert overfetch_for_exclude(8, frozenset({"a"})) == 24
    assert overfetch_for_exclude(30, frozenset({"a"})) == 50


def test_fix_a_filter_removes_assessment_items_and_overfetch_keeps_count(
    mini_libv2, monkeypatch
):
    """(3) Filter drops assessment_item candidates; over-fetch keeps the
    top-`limit` window full; the audit warning records the drop count."""
    monkeypatch.setenv("ED4ALL_ANSWER_EXCLUDE_CHUNK_TYPES", "assessment_item")
    pool = _interleaved_pool(24)
    fetched = {}

    def _fake_retrieve(root, slug, q, *, engine, limit):
        fetched["limit"] = limit
        return pool[:limit]

    monkeypatch.setattr(_ga, "_retrieve", _fake_retrieve)
    captured = {}
    monkeypatch.setattr(_ga, "compose_answer", _spy_compose_factory(captured))

    result = _ga.answer_course_question(
        mini_libv2, COURSE_SLUG, "vector stores",
        client=object(), refusal_policy=_PERMISSIVE_LEXICAL,
        completeness_recheck=False,
    )
    # Over-fetched min(8*3, 50) = 24 from the retriever.
    assert fetched["limit"] == 24
    seen = [p.chunk_id for p in captured["passages"]]
    # Window stays FULL at `limit` (8) after dropping every assessment_item.
    assert len(seen) == 8
    # Only the odd-index (explanation) chunks survive, in rank order.
    assert seen == [f"c{i}" for i in range(1, 16, 2)]
    # 8 assessment_items were consumed+dropped to reach 8 explanations.
    assert any(
        w.startswith("excluded_chunk_types:assessment_item:8")
        for w in result.warnings
    )


def test_fix_b_resolve_anchor_containment_clamp_and_garbage(monkeypatch):
    """(4) Containment resolver: clamp to [0.5, 1.0]; garbage/out-of-range →
    0.85; explicit arg respected within range."""
    monkeypatch.delenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", raising=False)
    assert resolve_anchor_containment() == 0.85
    monkeypatch.setenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", "0.80")
    assert resolve_anchor_containment() == 0.80
    for edge in ("0.5", "1.0"):
        monkeypatch.setenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", edge)
        assert resolve_anchor_containment() == float(edge)
    for bad in ("0.4", "1.5", "-1", "garbage", ""):
        monkeypatch.setenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", bad)
        assert resolve_anchor_containment() == 0.85, bad
    # Explicit arg wins verbatim within range; out-of-range explicit → default.
    assert resolve_anchor_containment(0.9) == 0.9
    assert resolve_anchor_containment(0.4) == 0.85


def test_fix_b_threshold_threads_into_citation_gate(mini_libv2, monkeypatch):
    """FIX B end-to-end: the resolved floor threads into every answer-path
    ``resolve_citation_anchor`` call (gate + attribution)."""
    monkeypatch.setenv("ED4ALL_ANSWER_ANCHOR_CONTAINMENT", "0.80")
    seen_thresholds = []
    real = _ga.resolve_citation_anchor

    def _spy_anchor(chunk_record, course_dir, *, chunkset_kind, containment_threshold):
        seen_thresholds.append(containment_threshold)
        return real(
            chunk_record, course_dir, chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
        )

    monkeypatch.setattr(_ga, "resolve_citation_anchor", _spy_anchor)
    client = FakeAnswerClient([_envelope("A.", ["mini_alpha_chunk_001"])])
    _ga.answer_course_question(
        mini_libv2, COURSE_SLUG, "What does a vector store index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL,
        completeness_recheck=False,
    )
    assert seen_thresholds
    assert all(t == 0.80 for t in seen_thresholds)
