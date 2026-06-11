"""End-to-end tests for the grounded-answer pipeline + citation gate (E6b).

Deterministic, CI-safe: no model, no server, no network. The LLM call site is
driven by the shared ``FakeAnswerClient`` (published by E5 in
``test_answer_composer``); retrieval runs the REAL lexical BM25 path over the
mini-course fixture materialised into a tmp LibV2 layout (the WS1/WS2 fixture
pattern). The offline-guard arm proves the whole query path needs nothing
beyond loopback.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lib.retrieval.answer_backend import AnswerBackendUnavailable
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
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

# A permissive semantic policy: cosine floor 0 so the citation gate (the E6/E7
# surface under test) is reached regardless of the fake embedder's cosine.
_PERMISSIVE_SEMANTIC = RefusalPolicy(
    engine="semantic",
    min_top_score=-1.0,
    score_floor=-1.0,
    min_passages_above_floor=1,
    policy_version="test-permissive-semantic",
)
from lib.testing.no_network import no_network

# Shared test doubles published by E5.
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
    path = libv2_root / "courses" / COURSE_SLUG / "dart_chunks" / "chunks.jsonl"
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

# A permissive lexical policy. The pipeline now resolves the MEASURED (lexical,
# None) pin by default (min_top_score=4.455352, calibrated on a different
# corpus); the mini-course fixture's BM25 scores fall below it, so the
# citation-gate / answer-shape tests below — which exercise the post-confidence
# surfaces, not the threshold — pass this permissive policy explicitly to clear
# confidence regardless of the fixture's corpus-specific scores.
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
    assert result.model_id == "qwen2.5:14b-instruct-q4_K_M"
    assert result.prompt_version == "ws3.v1"


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
# B4 provenance chain: source_references → citation.source_block + pdf_pages
# --------------------------------------------------------------------------- #


def _load_chunk(libv2_root: Path, chunk_id: str) -> dict:
    path = libv2_root / "courses" / COURSE_SLUG / "dart_chunks" / "chunks.jsonl"
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
            "sourceId": "dart:mini_alpha#s3_c0",
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
    assert cit.source_block == "dart:mini_alpha#s3_c0"
    # de-duplicated + sorted page list.
    assert cit.pdf_pages == [7, 12]
    d = cit.to_dict()
    assert d["source_block"] == "dart:mini_alpha#s3_c0"
    assert d["pdf_pages"] == [7, 12]


def test_contributing_role_used_when_no_primary(mini_libv2: Path):
    """No 'primary' ref → fall back to the first reference (never empty)."""
    base = _load_chunk(mini_libv2, "mini_alpha_chunk_001")
    chunk = dict(base)
    chunk["id"] = "mini_alpha_chunk_contrib"
    chunk["text"] = "Krypton contributing marker: embeddings power retrieval."
    src = dict(chunk["source"])
    src["source_references"] = [
        {"sourceId": "dart:mini_alpha#s9_c2", "role": "contributing", "pages": [4]},
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
    assert cit.source_block == "dart:mini_alpha#s9_c2"
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
# Attribution-driven citation prune + add (2026-06 extension)
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


def test_prune_all_claimless_is_noop(mini_libv2: Path):
    """When EVERY cited citation is claim-less, prune is a no-op (verdict + all
    citations preserved) with the skipped-all warning."""
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
    assert result.status == STATUS_ANSWERED
    kept_ids = {c.chunk_id for c in result.citations}
    assert kept_ids == {"mini_alpha_chunk_001", "mini_alpha_chunk_002"}
    assert any("claimless_prune_skipped_all_below_threshold" in w
               for w in result.warnings)
    assert _prune_events(spy)[0]["decision"] == "citation_prune:skipped_all_claimless"


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
# WS4 contract shape (to_dict keys frozen)
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
    }
    # Citation dict shape (the WS4 rendering contract).
    cit = d["citations"][0]
    assert set(cit.keys()) == {
        "chunk_id", "item_path", "section_heading", "module_id", "page_label",
        "anchor_status", "source_path", "text_quote", "link_target",
        # B4 provenance-chain fields (additive, optional).
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
# E6/E7 fix: semantic engine + chunkset_kind=None aligns the citation gate
# with the vector index's manifest (NOT the directory-presence heuristic).
# --------------------------------------------------------------------------- #


# Heading + body of the imscc chunk. The query in the E6/E7 tests is the
# index's exact text+heading projection so the deterministic fake embedder
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
    """Set up the multi-chunkset misalignment: dart_chunks/ present (from the
    fixture) AND an imscc-pinned vector index over imscc_chunks/.

    Returns the imscc chunk_id the model will cite. The imscc chunk's
    item_path (``alpha.html``) resolves as an imscc member but NOT under the
    dart roots (we drop ``source/html`` so a dart-kind gate fails with
    source_page_missing). So:
      * directory heuristic (_infer_chunkset_kind) -> "dart"  -> gate BLOCKS
      * vector-index manifest (the fix)            -> "imscc" -> gate PASSES
    """
    from lib.embedding.providers import build_embedding_client
    from LibV2.tools.libv2.vector_index import build_vector_index

    course_dir = libv2_root / "courses" / COURSE_SLUG

    # Drop the dart HTML sources so a dart-kind anchor resolution fails. The
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
    kind) even though dart_chunks/ is also present.

    Fail-without-fix: the pre-fix pipeline inferred chunkset_kind from
    directory presence (dart_chunks/ wins) and the gate resolved against dart
    -> source_page_missing -> blocked_citation_gate, despite the answer +
    citation being correct against the imscc-built index.
    """
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
    """Pin the misalignment: the directory heuristic returns 'dart' (so an
    explicit chunkset_kind='dart' BLOCKS), proving the index-manifest read in
    the previous test is what unblocks the correct answer."""
    from lib.retrieval.grounded_answer import _infer_chunkset_kind

    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    imscc_chunk_id = _build_imscc_index_and_chunks(mini_libv2)

    # The directory heuristic still picks dart (dart_chunks/ present).
    assert _infer_chunkset_kind(mini_libv2, COURSE_SLUG) == "dart"

    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.", [imscc_chunk_id])
    ])
    # Forcing chunkset_kind='dart' (what the heuristic would pick) BLOCKS,
    # because the dart HTML source was dropped -> source_page_missing.
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _IMSCC_INDEX_QUERY,
        engine="semantic", client=client, chunkset_kind="dart",
        refusal_policy=_PERMISSIVE_SEMANTIC,
    )
    assert result.status == STATUS_BLOCKED_CITATION_GATE


# --------------------------------------------------------------------------- #
# Refusal-policy pins wired onto the answer path (Workstream 0)
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


def test_answer_path_hybrid_rrf_reads_embedder_but_falls_back_uncalibrated(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    """hybrid-rrf resolves the live index's embedder (its fused score depends on
    the semantic arm) but, ABSENT a pin, falls back to the v0-uncalibrated
    default — never a stale semantic cosine pin for the same model.

    The wiring reads the manifest embedder for hybrid-rrf; this test confirms it
    is consulted (the embedder is keyed) yet does not leak the (semantic, model)
    pin onto the hybrid path."""
    import lib.retrieval.grounded_answer as ga

    seen = {}

    def _fake_model(root, slug):
        seen["called"] = True
        return _BGE_LARGE

    monkeypatch.setattr(ga, "_vector_index_embedding_model_id", _fake_model)
    # Avoid the index-manifest chunkset read picking a wrong kind; pin it.
    monkeypatch.setattr(
        ga, "_vector_index_chunkset_kind", lambda root, slug: "dart"
    )
    # Below the v0-uncalibrated hybrid-rrf min_top_score (1/(RRF_K+1) on the
    # RRF scale) → pre-LLM refusal, so the resolved policy surfaces.
    _stub_low_score_retrieval(monkeypatch, 0.005)
    client = FakeAnswerClient([_envelope("never used", ["x"])])

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "q", engine="hybrid-rrf", client=client,
    )
    assert seen.get("called") is True  # the embedder WAS resolved for hybrid-rrf
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    # Unpinned → v0-uncalibrated; the (semantic, bge-large) pin did NOT leak in.
    assert result.refusal["policy_version"] == POLICY_VERSION_UNCALIBRATED
    assert result.refusal["embedding_model_id"] is None


_BGE_LARGE = "BAAI/bge-large-en-v1.5"


# --------------------------------------------------------------------------- #
# Default-path NLI never imported (R10 — DeBERTa stays off the query path)
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
