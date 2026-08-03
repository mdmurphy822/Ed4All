"""Tests for the three-arm eval scorecard (``lib.retrieval.eval_arms``).

CI-safe: a fake local-model client + fake retriever (and the injected
``answer_fn`` for the grounded arm) are used throughout, so no network / no
model / no real LLM is ever touched. The deterministic key-point machinery
(``score_key_point_coverage``) is the real one — that's the whole point of the
honest cross-arm comparison.

Two gold-set strategies:
  * synthetic in-memory gold (``gold=`` injection) for the per-arm scoring
    tests — lets us pin expected_key_points + relevant_passages precisely;
  * the real mini-course fixture (copied into a tmp LibV2 root) for the
    scorecard-assembly tests that exercise ``_load_verified_gold``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lib.retrieval.eval_arms import (
    ALL_ARMS,
    ARM_BASE,
    ARM_GROUNDED,
    ARM_RETRIEVAL,
    SCORECARD_SCHEMA_VERSION,
    format_scorecard_table,
    run_base_arm,
    run_retrieval_arm,
    run_scorecard,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)


# ===========================================================================
# Synthetic gold + fakes
# ===========================================================================

def _synthetic_gold():
    """A 2-question gold carrying expected_key_points + a primary passage."""
    return {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "What does a vector store index?",
                "question_type": "short_answer",
                "expected_key_points": [
                    "high dimensional embedding vectors",
                    "similarity search",
                ],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"},
                ],
            },
            {
                "question_id": "gq-0002",
                "question_text": "How is retrieval quality measured?",
                "question_type": "short_answer",
                "expected_key_points": ["recall at k", "mean reciprocal rank"],
                "relevant_passages": [
                    {"chunk_id": "chunk_b", "relevance": "primary"},
                ],
            },
        ],
    }


class _FakeClient:
    """Duck-types ``chat_completion`` + ``model``. Maps question text → answer
    text; an entry of ``RAISE`` raises to exercise per-question isolation."""

    def __init__(self, answers, *, model="fake-qwen-7b"):
        self._answers = answers
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=800, temperature=0.4):
        user = messages[-1]["content"]
        self.calls.append(user)
        out = self._answers.get(user)
        if out == "RAISE":
            raise RuntimeError("simulated backend hiccup")
        return out if out is not None else ""


class _Result:
    """Duck-types the live retrieval-result shape (``chunk_id`` + ``text``)."""

    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


class _NliScore:
    """Duck-types ``NliScore`` (``.entailment`` + ``.contradiction``)."""

    def __init__(self, entailment, contradiction):
        self.entailment = float(entailment)
        self.contradiction = float(contradiction)


class _FakeNli:
    """Deterministic fake NLI singleton for groundedness scoring.

    ``score_batch(pairs=[(premise, hypothesis), ...])`` returns one
    :class:`_NliScore` per pair. A claim (hypothesis) is "entailed" iff its
    ``support`` substring appears in the premise; otherwise unsupported (or
    contradicted when its ``contradict`` substring appears). Lets a test pin
    exactly which answer sentences ground against which passages.
    """

    _revision = "fake-nli-rev"

    def __init__(self, *, entailed_if=(), contradicted_if=()):
        # Tuples of (premise_substr, hypothesis_substr) → high entailment /
        # contradiction. First match wins.
        self._entailed_if = list(entailed_if)
        self._contradicted_if = list(contradicted_if)
        self.batches = 0

    def score_batch(self, *, pairs):
        self.batches += 1
        out = []
        for premise, hypothesis in pairs:
            ent, con = 0.10, 0.10
            for p_sub, h_sub in self._entailed_if:
                if p_sub in premise and h_sub in hypothesis:
                    ent = 0.95
                    break
            for p_sub, h_sub in self._contradicted_if:
                if p_sub in premise and h_sub in hypothesis:
                    con = 0.95
                    break
            out.append(_NliScore(ent, con))
        return out


class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


# ===========================================================================
# BASE arm
# ===========================================================================

def test_base_arm_scores_key_points_with_shared_scorer():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            # First answer covers BOTH key points verbatim → 2/2.
            "What does a vector store index?": (
                "A vector store indexes high dimensional embedding vectors and "
                "supports similarity search over them."
            ),
            # Second answer covers ONE key point → 1/2.
            "How is retrieval quality measured?": (
                "Retrieval quality is often measured by recall at k."
            ),
        }
    )
    result = run_base_arm(Path("/unused"), "course-x", client=client, gold=gold)

    assert result["arm"] == ARM_BASE
    assert result["retrieval"] is False
    assert result["refusal_scaffolding"] is False
    assert result["citations"] == "n/a"
    assert result["answers_everything"] is True
    assert result["declined"] == 0
    assert result["answered"] == 2
    assert result["errored"] == 0
    # 2 + 1 of 4 total key points covered.
    kp = result["key_point_coverage"]
    assert kp["total_key_points"] == 4
    assert kp["covered_key_points"] == 3
    assert kp["coverage_rate"] == pytest.approx(0.75)
    # No system grounding / passages were injected — base prompt only.
    assert all("vector store indexes" not in c for c in client.calls[1:2])


def test_base_arm_error_isolation_does_not_kill_arm():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            "What does a vector store index?": "RAISE",  # one question explodes
            "How is retrieval quality measured?": (
                "Measured by recall at k and mean reciprocal rank."
            ),
        }
    )
    result = run_base_arm(Path("/unused"), "course-x", client=client, gold=gold)
    # The arm still completed: one errored, one answered + scored.
    assert result["errored"] == 1
    assert result["answered"] == 1
    rows = {r["question_id"]: r for r in result["questions"]}
    assert rows["gq-0001"]["answered"] is False
    assert "error" in rows["gq-0001"]
    assert rows["gq-0002"]["answered"] is True
    # Only the surviving question's key points are in the denominator.
    assert result["key_point_coverage"]["total_key_points"] == 2
    assert result["key_point_coverage"]["covered_key_points"] == 2


# --- BASE-arm hallucination axis -------------------------------------------

def _base_groundedness_setup():
    """A 1-question gold + answer whose two sentences split into one grounded
    and one ungrounded claim against a single retrieved passage."""
    gold = {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "What does a vector store index?",
                "question_type": "short_answer",
                "expected_key_points": ["embedding vectors"],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"}
                ],
            }
        ],
    }
    # Two scorable sentences (≥4 content tokens each): the first grounds in the
    # passage (carries the GROUNDED marker), the second does not (FABRICATED).
    answer = (
        "A vector store indexes dense GROUNDED embedding vectors carefully. "
        "The vector store also performs FABRICATED nightly database backups."
    )
    client = _FakeClient({"What does a vector store index?": answer})
    # Retrieved passage entails ONLY the GROUNDED sentence.
    retrieve_fn = _retrieve_fn_factory(
        {
            "What does a vector store index?": [
                _Result(
                    "chunk_a",
                    "A vector store indexes dense GROUNDED embedding vectors "
                    "for similarity search.",
                )
            ]
        }
    )
    nli = _FakeNli(entailed_if=[("GROUNDED", "GROUNDED")])
    return gold, client, retrieve_fn, nli


def test_base_arm_scores_hallucination_against_corpus():
    gold, client, retrieve_fn, nli = _base_groundedness_setup()
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=gold,
        retrieve_fn=retrieve_fn,
        nli=nli,
    )
    g = result["groundedness"]
    assert g["available"] is True
    # 2 scorable claims, 1 entailed → 0.5 grounded, 0.5 unsupported.
    assert g["claims_scored"] == 2
    assert g["groundedness_rate_mean"] == pytest.approx(0.5)
    assert g["unsupported_claim_rate"] == pytest.approx(0.5)
    # Semantics note present + flags the corpus-bound definition.
    assert "unsupported-vs-COURSE-CORPUS" in g["_note"]
    assert "factual-error rate" in g["_note"]
    # The per-question row carries its own groundedness sub-block.
    row = result["questions"][0]
    assert row["groundedness"]["available"] is True
    assert row["groundedness"]["scored_count"] == 2


def test_base_arm_hallucination_na_when_nli_unavailable():
    """NLI unavailable → axis reads n/a with a reason, NEVER fabricated rates."""
    gold, client, retrieve_fn, _ = _base_groundedness_setup()
    # nli=None AND monkeypatch the singleton resolver to return None so the real
    # ~750MB model is never touched; the arm degrades honestly.
    import lib.retrieval.groundedness as gmod

    orig = gmod._resolve_nli
    gmod._resolve_nli = lambda nli=None: None
    try:
        result = run_base_arm(
            Path("/unused"),
            "course-x",
            client=client,
            gold=gold,
            retrieve_fn=retrieve_fn,
            nli=None,
        )
    finally:
        gmod._resolve_nli = orig
    g = result["groundedness"]
    assert g["available"] is False
    # No fabricated numbers — the rates are None and a reason is recorded.
    assert g["unsupported_claim_rate"] is None
    assert g["groundedness_rate_mean"] is None
    assert g["reason"] == "nli_unavailable"
    # The key-point axis is unaffected (deterministic, no NLI).
    assert result["key_point_coverage"]["coverage_rate"] is not None


def test_base_arm_free_text_answer_is_scored_not_artifact_filtered():
    """Regression: the BASE arm requests FREE TEXT (json_mode=False), so a prose
    answer's claims reach the groundedness scorer. A whole-answer JSON envelope
    — what json_mode=True would have forced raw qwen to emit — is a single
    fully-bracketed literal that scorer-v2's artifact filter discards, scoring
    ZERO claims and hiding the very fabrication this arm measures. This proves
    the json_mode=False fix matters: prose scores, a JSON blob does not.
    """
    gold = {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "What is the national dish of Peru?",
                "question_type": "short_answer",
                "expected_key_points": ["ceviche"],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"}
                ],
            }
        ],
    }
    retrieve_fn = _retrieve_fn_factory(
        {
            "What is the national dish of Peru?": [
                _Result(
                    "chunk_a",
                    "Ceviche is widely regarded as the national dish of Peru.",
                )
            ]
        }
    )
    nli = _FakeNli(entailed_if=[("Ceviche", "Ceviche")])

    # FREE-TEXT (what json_mode=False yields): two prose sentences, each ≥4
    # content tokens → both reach the scorer as real claims.
    prose_client = _FakeClient(
        {
            "What is the national dish of Peru?": (
                "Ceviche is the national dish of Peru today. "
                "It also pairs nicely with imported French wine."
            )
        }
    )
    prose = run_base_arm(
        Path("/unused"), "course-x", client=prose_client, gold=gold,
        retrieve_fn=retrieve_fn, nli=nli,
    )["groundedness"]
    assert prose["available"] is True
    # The fix's whole point: the base arm scores real claims on free text.
    assert prose["claims_scored"] > 0
    assert prose["filtered_count"] == 0

    # JSON ENVELOPE (what json_mode=True would force raw qwen to emit): one
    # fully-bracketed literal → the artifact filter discards it whole → ZERO
    # scored claims. Same gold/corpus/NLI; only the answer SHAPE differs.
    json_client = _FakeClient(
        {
            "What is the national dish of Peru?": (
                '{"national_dish": "Ceviche", "description": "a dish of '
                'marinated raw fish cured in citrus juices"}'
            )
        }
    )
    blob = run_base_arm(
        Path("/unused"), "course-x", client=json_client, gold=gold,
        retrieve_fn=retrieve_fn, nli=nli,
    )["groundedness"]
    # The JSON blob was artifact-filtered → nothing scored. Were the base arm
    # to (wrongly) keep json_mode=True, this is the silent near-zero metric: the
    # arm scores ZERO claims (everything filtered) yet still credits a 0.0
    # unsupported rate, hiding the ungrounded invention inside the envelope.
    assert blob["claims_scored"] == 0
    assert blob["filtered_count"] >= 1
    # The contrast is the whole point: prose had real scored claims, the JSON
    # blob had none — same gold / corpus / NLI, only the answer SHAPE differs.
    assert prose["claims_scored"] > blob["claims_scored"]


def test_base_arm_decision_rationale_carries_groundedness_signal():
    gold, client, retrieve_fn, nli = _base_groundedness_setup()
    cap = _RecordingCapture()
    run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=gold,
        retrieve_fn=retrieve_fn,
        nli=nli,
        capture=cap,
    )
    events = [
        e for e in cap.events if e["decision_type"] == "base_model_eval_call"
    ]
    assert events
    # Extended (not a new call site): the existing base decision's rationale now
    # interpolates the hallucination-axis signal.
    assert any("hallucination" in e["rationale"] for e in events)
    assert any("score_groundedness" in e["rationale"] for e in events)


def test_base_arm_emits_decision_capture_per_question():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            "What does a vector store index?": "embedding vectors",
            "How is retrieval quality measured?": "recall at k",
        }
    )
    cap = _RecordingCapture()
    run_base_arm(
        Path("/unused"), "course-x", client=client, gold=gold, capture=cap
    )
    base_events = [
        e for e in cap.events if e["decision_type"] == "base_model_eval_call"
    ]
    # One capture per scored question.
    assert len(base_events) == 2
    for e in base_events:
        # Rationale interpolates dynamic per-call signals (not boilerplate).
        assert len(e["rationale"]) >= 20
        assert "course-x" in e["rationale"]
        assert "fake-qwen-7b" in e["rationale"]
        alternatives = e["alternatives_considered"]
        assert len(alternatives) == 2
        assert all(
            set(item) == {"option", "reason_rejected"} for item in alternatives
        )
        assert all(
            any(
                question_id in item["reason_rejected"]
                for question_id in ("gq-0001", "gq-0002")
            )
            for item in alternatives
        )
    # Each question id appears in exactly one decision string.
    decided = " ".join(e["decision"] for e in base_events)
    assert "gq-0001" in decided and "gq-0002" in decided


# ===========================================================================
# RETRIEVAL arm
# ===========================================================================

def _retrieve_fn_factory(by_query):
    def _fn(_root, _slug, query, *, engine, limit):
        return by_query.get(query, [])[:limit]

    return _fn


def test_retrieval_arm_extractive_ceiling_and_hits():
    gold = _synthetic_gold()
    by_query = {
        # Q1: primary chunk_a is top-1; passages cover BOTH key points.
        "What does a vector store index?": [
            _Result(
                "chunk_a",
                "A vector store indexes high dimensional embedding vectors "
                "for similarity search.",
            ),
            _Result("chunk_z", "unrelated passage"),
        ],
        # Q2: primary chunk_b is present but NOT top-1; passages cover ONE
        # key point.
        "How is retrieval quality measured?": [
            _Result("chunk_y", "Some passage about recall at k only."),
            _Result("chunk_b", "Mean reciprocal rank is one measure."),
        ],
    }
    result = run_retrieval_arm(
        Path("/unused"),
        "course-x",
        engine="semantic",
        limit=8,
        retrieve_fn=_retrieve_fn_factory(by_query),
        gold=gold,
    )
    assert result["arm"] == ARM_RETRIEVAL
    assert result["retrieval"] is True
    assert result["model_id"] is None  # no LLM

    # Extractive ceiling: Q1 covers 2/2, Q2 covers 2/2 (recall at k + MRR both
    # appear across the concatenated passages) → 4/4.
    kp = result["key_point_coverage"]
    assert kp["total_key_points"] == 4
    assert kp["covered_key_points"] == 4

    hit = result["primary_relevant_hit"]
    assert hit["questions"] == 2
    # Both primaries present in top-k.
    assert hit["hit_at_k"] == 2
    assert hit["hit_at_k_rate"] == pytest.approx(1.0)
    # Only Q1's primary is top-1.
    assert hit["hit_top1"] == 1
    assert hit["hit_top1_rate"] == pytest.approx(0.5)


def test_retrieval_arm_error_isolation():
    gold = _synthetic_gold()

    def _fn(_root, _slug, query, *, engine, limit):
        if query == "What does a vector store index?":
            raise RuntimeError("index unavailable for this query")
        return [_Result("chunk_b", "recall at k and mean reciprocal rank")]

    result = run_retrieval_arm(
        Path("/unused"), "course-x", retrieve_fn=_fn, gold=gold
    )
    rows = {r["question_id"]: r for r in result["questions"]}
    assert "error" in rows["gq-0001"]
    assert rows["gq-0001"]["hit_at_k"] is False
    # The other question still scored.
    assert rows["gq-0002"]["hit_at_k"] is True


# ===========================================================================
# Scorecard assembly — uses the real fixture for _load_verified_gold
# ===========================================================================

@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


# --- a fake grounded pipeline, shaped like answer_course_question -----------

class _FakeCitation:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "anchor_status": self.anchor_status,
            "page_label": self.page_label,
            "text_quote": self.text_quote,
        }


class _FakeAnswer:
    def __init__(self, *, status, answer_text, citations):
        self.status = status
        self.answer_text = answer_text
        self.citations = citations
        self.groundedness = None
        self.latency_ms = 10.0
        self.model_id = "fake-grounded-model"
        self.prompt_version = "ws3.v1"
        self.confidence = {"policy_version": "ws3.v0"}


def _fake_grounded_fn():
    answers = {
        "What does a vector store index?": "mini_alpha_chunk_001",
        "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
        "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
    }

    def _fn(repo_root, course_slug, query, *, engine="lexical", limit=8,
            client=None, refusal_policy=None, with_groundedness=False,
            capture=None, **kwargs):
        cid = answers.get(query)
        if cid is None:
            return _FakeAnswer(
                status="refused_low_confidence", answer_text=None, citations=[]
            )
        return _FakeAnswer(
            status="answered",
            answer_text=f"Answer about {cid}.",
            citations=[_FakeCitation(cid)],
        )

    return _fn


def _fake_retrieve_for_fixture():
    # Map fixture gold question text → a top-k list whose top-1 is the gold
    # primary chunk (so hit_top1_rate == 1.0).
    by_query = {
        "What does a vector store index?": [_Result("mini_alpha_chunk_001", "x")],
        "How is retrieval quality commonly measured?": [
            _Result("mini_alpha_chunk_003", "y")
        ],
        "Where does the course cover chunking strategies?": [
            _Result("mini_beta_chunk_005", "z")
        ],
    }
    return _retrieve_fn_factory(by_query)


def test_scorecard_default_is_grounded_only(libv2_course):
    """Default arms=(grounded,) — only the grounded arm runs; the grounded
    report is still written (staleness test contract)."""
    repo_root, slug, course_dir = libv2_course
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical", write=True,
        grounded_kwargs={"answer_fn": _fake_grounded_fn()},
    )
    assert scorecard["schema_version"] == SCORECARD_SCHEMA_VERSION
    assert scorecard["arms_run"] == [ARM_GROUNDED]
    assert set(scorecard["arms"]) == {ARM_GROUNDED}
    # Grounded arm still wrote its own grounded_answer_eval report.
    eval_dir = course_dir / "retrieval_eval"
    assert list(eval_dir.glob("grounded_answer_eval_*.json"))
    # Scorecard file also written.
    assert list(eval_dir.glob("eval_scorecard_*.json"))
    # Comparison carries the grounded column + the top-level refusal_safety
    # block (schema 1.4; populated from the grounded arm alone).
    assert set(scorecard["comparison"]) == {ARM_GROUNDED, "refusal_safety"}


def test_scorecard_three_arms_side_by_side(libv2_course):
    repo_root, slug, course_dir = libv2_course
    base_client = _FakeClient(
        {
            "What does a vector store index?": "an answer",
            "How is retrieval quality commonly measured?": "another",
            "Where does the course cover chunking strategies?": "third",
        },
        model="fake-base-7b",
    )
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical",
        arms=[ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED],
        base_client=base_client,
        retrieve_fn=_fake_retrieve_for_fixture(),
        grounded_kwargs={"answer_fn": _fake_grounded_fn()},
        write=True,
    )
    assert scorecard["arms_run"] == [ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED]
    # Per-arm blocks all present + carry their identity.
    assert scorecard["arms"][ARM_BASE]["model_id"] == "fake-base-7b"
    assert scorecard["arms"][ARM_RETRIEVAL]["retrieval"] is True
    assert scorecard["arms"][ARM_GROUNDED]["model_id"] == "fake-grounded-model"

    # Comparison block: all three columns on the shared axes + the derived
    # hallucination_reduction entry (BASE→GROUNDED).
    comp = scorecard["comparison"]
    assert set(comp) == {
        ARM_BASE,
        ARM_RETRIEVAL,
        ARM_GROUNDED,
        "hallucination_reduction",
        "refusal_safety",
    }
    for arm in (ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED):
        assert "key_point_coverage_rate" in comp[arm]
        assert "latency_ms" in comp[arm]
        # Hallucination axis present on every arm column.
        assert "unsupported_claim_rate" in comp[arm]
    # RETRIEVAL has no answer claims → the sentinel "—" (not None).
    assert comp[ARM_RETRIEVAL]["unsupported_claim_rate"] == "—"
    # Derived reduction entry carries all four fields + the dilution-convention
    # note (schema 1.2, additive).
    red = comp["hallucination_reduction"]
    assert set(red) == {
        "base_rate",
        "grounded_rate",
        "absolute_reduction",
        "relative_reduction",
        "_note",
    }
    assert "per-question mean" in red["_note"]
    # Base answers everything → declined 0; retrieval has no answered notion.
    assert comp[ARM_BASE]["declined"] == 0
    assert comp[ARM_RETRIEVAL]["answered"] is None
    # Retrieval hit@k surfaced in its comparison column (fixture: all top-1).
    assert comp[ARM_RETRIEVAL]["primary_relevant_hit_at_k_rate"] == pytest.approx(
        1.0
    )

    # Grounded-only axes stay inside the grounded ARM block, NOT in comparison.
    assert "refusal" in scorecard["arms"][ARM_GROUNDED]["headline"]
    assert "refusal" not in comp[ARM_GROUNDED]

    # Aligned table renders all three columns.
    table = format_scorecard_table(scorecard)
    assert "BASE (qwen)" in table and "RETRIEVAL" in table and "GROUNDED" in table
    assert "key_point_coverage" in table


def test_scorecard_two_arms_base_and_retrieval(libv2_course):
    """A 2-arm scorecard (no grounded) assembles cleanly + writes no grounded
    report."""
    repo_root, slug, course_dir = libv2_course
    base_client = _FakeClient(
        {
            "What does a vector store index?": "a",
            "How is retrieval quality commonly measured?": "b",
            "Where does the course cover chunking strategies?": "c",
        }
    )
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical",
        arms=[ARM_BASE, ARM_RETRIEVAL],
        base_client=base_client,
        retrieve_fn=_fake_retrieve_for_fixture(),
        write=True,
    )
    assert scorecard["arms_run"] == [ARM_BASE, ARM_RETRIEVAL]
    assert ARM_GROUNDED not in scorecard["arms"]
    # No grounded arm → no grounded_answer_eval report written by this run.
    eval_dir = course_dir / "retrieval_eval"
    assert not list(eval_dir.glob("grounded_answer_eval_*.json"))
    # Scorecard still written.
    assert list(eval_dir.glob("eval_scorecard_*.json"))


def test_scorecard_single_base_arm(libv2_course):
    repo_root, slug, _ = libv2_course
    base_client = _FakeClient({})  # all questions → "" (answered, 0 coverage)
    scorecard = run_scorecard(
        repo_root, slug, arms=[ARM_BASE], base_client=base_client, write=False
    )
    assert scorecard["arms_run"] == [ARM_BASE]
    # Base column + the top-level refusal_safety block (schema 1.4).
    assert set(scorecard["comparison"]) == {ARM_BASE, "refusal_safety"}


def test_scorecard_unknown_arm_raises(libv2_course):
    repo_root, slug, _ = libv2_course
    with pytest.raises(ValueError) as exc:
        run_scorecard(repo_root, slug, arms=["base", "bogus"], write=False)
    assert "bogus" in str(exc.value)


def test_scorecard_critical_gold_issue_raises(libv2_course):
    """A tampered (sha-mismatched) gold set fails closed BEFORE any arm runs."""
    repo_root, slug, course_dir = libv2_course
    chunks = course_dir / "semantik_chunks" / "chunks.jsonl"
    chunks.write_text(chunks.read_text() + '\n{"id":"x","text":"tamper"}\n')
    with pytest.raises(RuntimeError) as exc:
        run_scorecard(
            repo_root, slug, arms=[ARM_BASE],
            base_client=_FakeClient({}), write=False,
        )
    assert "critical" in str(exc.value).lower()


# ===========================================================================
# Comparison / hallucination-reduction math + table rendering
# ===========================================================================

from lib.retrieval.eval_arms import (  # noqa: E402
    _build_comparison,
    _hallucination_reduction,
)


def test_hallucination_reduction_math():
    red = _hallucination_reduction(0.50, 0.10)
    assert red["base_rate"] == 0.50
    assert red["grounded_rate"] == 0.10
    assert red["absolute_reduction"] == pytest.approx(0.40)
    assert red["relative_reduction"] == pytest.approx(0.80)


def test_hallucination_reduction_guards():
    # base == 0 → relative undefined (no division), absolute still computed.
    red = _hallucination_reduction(0.0, 0.0)
    assert red["absolute_reduction"] == pytest.approx(0.0)
    assert red["relative_reduction"] is None
    # base None (NLI n/a) → both derived fields None, never fabricated.
    red_none = _hallucination_reduction(None, 0.10)
    assert red_none["absolute_reduction"] is None
    assert red_none["relative_reduction"] is None
    # grounded None → likewise None (can't compute a delta).
    red_g = _hallucination_reduction(0.30, None)
    assert red_g["absolute_reduction"] is None
    assert red_g["relative_reduction"] is None


def _arm_blocks_for_comparison(base_rate, grounded_rate):
    """Minimal arm-result dicts shaped enough for _build_comparison."""
    base = {
        "arm": ARM_BASE,
        "key_point_coverage": {"coverage_rate": 0.4},
        "answered": 2,
        "declined": 0,
        "latency_ms": {"p50": 1.0, "p95": 2.0},
        "groundedness": {
            "available": base_rate is not None,
            "unsupported_claim_rate": base_rate,
        },
    }
    retr = {
        "arm": ARM_RETRIEVAL,
        "key_point_coverage": {"coverage_rate": 0.9},
        "primary_relevant_hit": {"hit_at_k_rate": 1.0},
        "latency_ms": {"p50": 0.5, "p95": 1.0},
    }
    grounded = {
        "arm": ARM_GROUNDED,
        "headline": {
            "key_point_coverage": {"coverage_rate": 0.6},
            "unsupported_claim_rate": grounded_rate,
            "latency_ms": {"p50": 3.0, "p95": 5.0},
            "refusal": {},
        },
        "questions": [],
    }
    return {ARM_BASE: base, ARM_RETRIEVAL: retr, ARM_GROUNDED: grounded}


def test_comparison_carries_hallucination_axis_and_reduction():
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    comp = _build_comparison(arms)
    assert comp[ARM_BASE]["unsupported_claim_rate"] == pytest.approx(0.40)
    assert comp[ARM_RETRIEVAL]["unsupported_claim_rate"] == "—"
    assert comp[ARM_GROUNDED]["unsupported_claim_rate"] == pytest.approx(0.05)
    red = comp["hallucination_reduction"]
    assert red["absolute_reduction"] == pytest.approx(0.35)
    assert red["relative_reduction"] == pytest.approx(0.875)


def test_comparison_reduction_na_when_base_axis_unavailable():
    # Base NLI unavailable (rate None) → reduction guards to None.
    arms = _arm_blocks_for_comparison(None, 0.05)
    comp = _build_comparison(arms)
    assert comp[ARM_BASE]["unsupported_claim_rate"] is None
    red = comp["hallucination_reduction"]
    assert red["base_rate"] is None
    assert red["absolute_reduction"] is None
    assert red["relative_reduction"] is None


def test_table_renders_hallucination_row_and_reduction_summary():
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    scorecard = {
        "course_slug": "course-x",
        "engine": "semantic",
        "arms": arms,
        "comparison": _build_comparison(arms),
    }
    table = format_scorecard_table(scorecard)
    # The hallucination row label + the per-arm values render.
    assert "hallucination (unsupported-vs-corpus)" in table
    assert "0.4000" in table  # base rate
    assert "0.0500" in table  # grounded rate
    # The one-line reduction summary renders with the relative percentage.
    assert "hallucination reduction (BASE→GROUNDED" in table
    assert "87.5%" in table
    # RETRIEVAL column shows the "—" sentinel on the hallucination row.
    assert "—" in table


def test_schema_version_bumped_to_1_4():
    # Semantic-refinement bump: refusal-safety hoisted to a TOP-LEVEL
    # comparison.refusal_safety block (per-arm rows), answered_probe_rate renamed
    # answered_not_refused_rate, BASE probe-answer NLI dropped (category error),
    # BASE probe_fabrication_rate intrinsic (= answered_not_refused_rate).
    assert SCORECARD_SCHEMA_VERSION == "1.4"


def test_all_arms_constant():
    assert ALL_ARMS == (ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED)


# ===========================================================================
# GAP 1 — out-of-scope confabulation axis (probe pass)
# ===========================================================================

def _probe_set():
    """A 3-probe refusal set: 2 the raw model answers (confabulates), 1 it
    declines (its answer trips a does-not-cover disclaimer phrase)."""
    return [
        {
            "probe_id": "rp-0001",
            "question_text": "How does ATP synthesis occur in mitochondria?",
            "category": "off_topic",
        },
        {
            "probe_id": "rp-0002",
            "question_text": "What ef_search does the FAISS HNSW index use?",
            "category": "out_of_scope_detail",
        },
        {
            "probe_id": "rp-0003",
            "question_text": "How do you fine-tune the embedding weights?",
            "category": "adjacent_domain",
        },
    ]


def _base_probe_client():
    """Fake client: answers 2 probes with invention, declines the 3rd via a
    disclaimer phrase. Also answers the one gold question."""
    return _FakeClient(
        {
            # Gold question — answered normally.
            "What does a vector store index?": (
                "A vector store indexes high dimensional embedding vectors."
            ),
            # Probe 1 + 2: pure invention (no disclaimer) → confabulated.
            "How does ATP synthesis occur in mitochondria?": (
                "ATP synthase pumps protons across the inner membrane to make ATP."
            ),
            "What ef_search does the FAISS HNSW index use?": (
                "It uses an ef_search of 64 by default for HNSW queries."
            ),
            # Probe 3: declines — answer carries a does-not-cover disclaimer.
            "How do you fine-tune the embedding weights?": (
                "The course does not cover model training; this is out of scope."
            ),
        }
    )


def _one_question_gold():
    return {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "What does a vector store index?",
                "question_type": "short_answer",
                "expected_key_points": ["embedding vectors"],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"}
                ],
            }
        ],
    }


def test_base_probe_pass_counts_answered_declined_errored():
    client = _base_probe_client()
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=_one_question_gold(),
        probes=_probe_set(),
    )
    probe = result["probe_confabulation"]
    assert probe["n_probes"] == 3
    # 2 confabulated (answered), 1 declined (disclaimer), 0 errored.
    assert probe["answered"] == 2
    assert probe["declined"] == 1
    assert probe["errored"] == 0
    # confabulation_rate = answered / (answered + declined) = 2/3.
    assert probe["confabulation_rate"] == pytest.approx(2 / 3)
    # Per-probe rows carry id / category / answered.
    rows = {r["probe_id"]: r for r in probe["probes"]}
    assert rows["rp-0001"]["answered"] is True
    assert rows["rp-0001"]["category"] == "off_topic"
    assert rows["rp-0003"]["answered"] is False


def test_base_probe_pass_error_isolation():
    """A client exception on one probe is recorded (errored) + excluded from the
    confabulation denominator; the pass keeps going."""
    probes = _probe_set()
    client = _FakeClient(
        {
            "What does a vector store index?": "embedding vectors",
            # Probe 1 explodes.
            "How does ATP synthesis occur in mitochondria?": "RAISE",
            # Probe 2 answered (confabulated).
            "What ef_search does the FAISS HNSW index use?": "ef_search is 64.",
            # Probe 3 declined.
            "How do you fine-tune the embedding weights?": (
                "Not covered by this course."
            ),
        }
    )
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=_one_question_gold(),
        probes=probes,
    )
    probe = result["probe_confabulation"]
    assert probe["errored"] == 1
    assert probe["answered"] == 1
    assert probe["declined"] == 1
    # Errored probe excluded from the denominator: 1 / (1 + 1) = 0.5.
    assert probe["confabulation_rate"] == pytest.approx(0.5)
    rows = {r["probe_id"]: r for r in probe["probes"]}
    assert rows["rp-0001"]["answered"] is None
    assert "error" in rows["rp-0001"]


def test_base_probe_pass_empty_answer_counts_as_declined():
    """An empty/whitespace answer is a decline (the model produced no answer)."""
    client = _FakeClient(
        {"What does a vector store index?": "embedding vectors"}
    )  # all probe queries unmapped → "" → declined
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=_one_question_gold(),
        probes=_probe_set(),
    )
    probe = result["probe_confabulation"]
    assert probe["answered"] == 0
    assert probe["declined"] == 3
    assert probe["confabulation_rate"] == pytest.approx(0.0)


def test_base_probe_pass_na_when_no_probes():
    """No probes loaded → confabulation_rate None (axis n/a), never fabricated."""
    client = _FakeClient(
        {"What does a vector store index?": "embedding vectors"}
    )
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=_one_question_gold(),
        probes=[],
    )
    probe = result["probe_confabulation"]
    assert probe["n_probes"] == 0
    assert probe["confabulation_rate"] is None


def test_base_decision_rationale_carries_probe_counts():
    """No new call site: the probe-pass counts ride on the existing per-question
    base_model_eval_call rationale."""
    cap = _RecordingCapture()
    run_base_arm(
        Path("/unused"),
        "course-x",
        client=_base_probe_client(),
        gold=_one_question_gold(),
        probes=_probe_set(),
        capture=cap,
    )
    events = [
        e for e in cap.events if e["decision_type"] == "base_model_eval_call"
    ]
    assert events
    # Only the existing decision type — no new call site introduced.
    assert {e["decision_type"] for e in cap.events} == {"base_model_eval_call"}
    # The rationale interpolates the probe-pass answered/declined/errored counts.
    assert any("out-of-scope confabulation" in e["rationale"] for e in events)
    assert any("answered 2" in e["rationale"] for e in events)


def test_grounded_out_of_scope_rate_derived_from_refusal_block():
    """GROUNDED out-of-scope rate = 1 - refusal_recall from the headline."""
    from lib.retrieval.eval_arms import _grounded_out_of_scope_rate

    grounded = {
        "headline": {
            "refusal": {"n_probes": 44, "refusal_recall": 0.75},
        }
    }
    assert _grounded_out_of_scope_rate(grounded) == pytest.approx(0.25)
    # No probes → None (n/a, never fabricated).
    assert _grounded_out_of_scope_rate({"headline": {"refusal": {}}}) is None
    assert (
        _grounded_out_of_scope_rate(
            {"headline": {"refusal": {"n_probes": 0, "refusal_recall": 1.0}}}
        )
        is None
    )


def test_comparison_carries_out_of_scope_confabulation_axis():
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    # Add a probe block to the base arm + a refusal block to the grounded arm.
    arms[ARM_BASE]["probe_confabulation"] = {"confabulation_rate": 0.95}
    arms[ARM_GROUNDED]["headline"]["refusal"] = {
        "n_probes": 44,
        "refusal_recall": 0.64,
    }
    comp = _build_comparison(arms)
    # BASE = its confabulation_rate; GROUNDED = 1 - refusal_recall; RETRIEVAL "—".
    assert comp[ARM_BASE]["out_of_scope_confabulation_rate"] == pytest.approx(0.95)
    assert comp[ARM_GROUNDED]["out_of_scope_confabulation_rate"] == pytest.approx(
        0.36
    )
    assert comp[ARM_RETRIEVAL]["out_of_scope_confabulation_rate"] == "—"


def test_table_renders_out_of_scope_confabulation_row():
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    arms[ARM_BASE]["probe_confabulation"] = {"confabulation_rate": 0.95}
    arms[ARM_GROUNDED]["headline"]["refusal"] = {
        "n_probes": 44,
        "refusal_recall": 0.64,
    }
    scorecard = {
        "course_slug": "course-x",
        "engine": "semantic",
        "arms": arms,
        "comparison": _build_comparison(arms),
    }
    table = format_scorecard_table(scorecard)
    assert "out_of_scope_confab (answered-instead-of-refused)" in table
    assert "0.9500" in table  # base confabulation rate
    assert "0.3600" in table  # grounded 1 - refusal_recall


# ===========================================================================
# GAP 2 — dilution transparency (undiluted claim-level fields)
# ===========================================================================

def _dilution_setup():
    """A 2-question gold + a crafted answer mix:
      * Q1 answer is ALL computational → 0 scorable claims (the per-question
        mean convention credits it as 0.0 unsupported).
      * Q2 answer is half-unsupported → 1 of 2 claims unsupported.
    The diluted per-question mean is (0.0 + 0.5)/2 = 0.25; the undiluted
    claim-level rate is 1 unsupported / 2 total scored = 0.50.
    """
    gold = {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "Compute 2 plus 2.",
                "question_type": "short_answer",
                "expected_key_points": ["four"],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"}
                ],
            },
            {
                "question_id": "gq-0002",
                "question_text": "What does a vector store index?",
                "question_type": "short_answer",
                "expected_key_points": ["embedding vectors"],
                "relevant_passages": [
                    {"chunk_id": "chunk_b", "relevance": "primary"}
                ],
            },
        ],
    }
    client = _FakeClient(
        {
            # Q1: a bare arithmetic statement → the v2 scorer exempts it as
            # computational, leaving 0 scorable claims.
            "Compute 2 plus 2.": "2 + 2 = 4.",
            # Q2: two scorable sentences — one grounds, one fabricated.
            "What does a vector store index?": (
                "A vector store indexes dense GROUNDED embedding vectors here. "
                "The vector store also performs FABRICATED nightly backups daily."
            ),
        }
    )
    retrieve_fn = _retrieve_fn_factory(
        {
            "Compute 2 plus 2.": [_Result("chunk_a", "Some arithmetic context.")],
            "What does a vector store index?": [
                _Result(
                    "chunk_b",
                    "A vector store indexes dense GROUNDED embedding vectors "
                    "for similarity search.",
                )
            ],
        }
    )
    nli = _FakeNli(entailed_if=[("GROUNDED", "GROUNDED")])
    return gold, client, retrieve_fn, nli


def test_base_groundedness_undiluted_claim_level_vs_diluted_mean():
    gold, client, retrieve_fn, nli = _dilution_setup()
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=gold,
        retrieve_fn=retrieve_fn,
        nli=nli,
    )
    g = result["groundedness"]
    # Q1 contributed 0 scored claims; Q2 contributed 2 (1 unsupported).
    assert g["claims_scored"] == 2
    # Only Q2 had a scorable claim.
    assert g["questions_with_scorable_claims"] == 1
    # Diluted per-question mean: (0.0 + 0.5)/2 = 0.25 — KEPT convention.
    assert g["unsupported_claim_rate"] == pytest.approx(0.25)
    # Undiluted claim-level: 1 unsupported / 2 total scored = 0.50.
    assert g["claim_level_unsupported_rate"] == pytest.approx(0.50)


def test_base_groundedness_claim_level_none_when_no_scored_claims():
    """All-computational arm → no scored claims → claim-level rate None."""
    gold = {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "Compute 2 plus 2.",
                "question_type": "short_answer",
                "expected_key_points": ["four"],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"}
                ],
            }
        ],
    }
    client = _FakeClient({"Compute 2 plus 2.": "2 + 2 = 4."})
    retrieve_fn = _retrieve_fn_factory(
        {"Compute 2 plus 2.": [_Result("chunk_a", "arithmetic context")]}
    )
    nli = _FakeNli()
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=gold,
        retrieve_fn=retrieve_fn,
        nli=nli,
    )
    g = result["groundedness"]
    assert g["claims_scored"] == 0
    assert g["questions_with_scorable_claims"] == 0
    assert g["claim_level_unsupported_rate"] is None


def test_comparison_grounded_claim_level_none_with_reason():
    """The grounded claim-level aggregate is NOT derivable from the persisted
    report → None with a reason string (anti-silent-degradation)."""
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    arms[ARM_BASE]["groundedness"]["claim_level_unsupported_rate"] = 0.6
    comp = _build_comparison(arms)
    # BASE carries the undiluted rate from its groundedness block.
    assert comp[ARM_BASE]["claim_level_unsupported_rate"] == pytest.approx(0.6)
    # GROUNDED claim-level is None + carries a reason (not approximated).
    assert comp[ARM_GROUNDED]["claim_level_unsupported_rate"] is None
    assert "not derivable" in comp[ARM_GROUNDED][
        "claim_level_unsupported_rate_reason"
    ]
    # RETRIEVAL has no claims → sentinel "—".
    assert comp[ARM_RETRIEVAL]["claim_level_unsupported_rate"] == "—"


# ===========================================================================
# REFUSAL-SAFETY axis (schema 1.4) — reframed.
#
# A refusal probe is unanswerable-from-the-course BY CONSTRUCTION, so any base
# answer to it is unsupported-by-course (membership already establishes that;
# no NLI). The HEADLINE base safety metric is the ANSWERED-NOT-REFUSED rate.
# NLI "unsupported-when-answered" is GROUNDED-ARM-ONLY (near-miss answers from a
# real cited passage). The comparison.refusal_safety block is TOP-LEVEL.
# ===========================================================================

def _refusal_safety_setup():
    """A 3-probe set + crafted base answers:
      * rp-0001 ANSWERED (no disclaimer).
      * rp-0002 ANSWERED (no disclaimer).
      * rp-0003 DECLINED (does-not-cover disclaimer).
    The BASE arm NO LONGER NLI-scores these answers (category error), so no
    corpus / NLI fixtures are needed.
    """
    probes = [
        {
            "probe_id": "rp-0001",
            "question_text": "Probe one answered?",
            "category": "near_miss",
        },
        {
            "probe_id": "rp-0002",
            "question_text": "Probe two answered?",
            "category": "off_topic",
        },
        {
            "probe_id": "rp-0003",
            "question_text": "Probe three declined?",
            "category": "off_topic",
        },
    ]
    client = _FakeClient(
        {
            "What does a vector store index?": (
                "A vector store indexes embedding vectors."
            ),
            "Probe one answered?": "Some invented probe-one answer claim.",
            "Probe two answered?": "Some invented probe-two answer claim.",
            "Probe three declined?": (
                "The course does not cover this; out of scope."
            ),
        }
    )
    return probes, client


def test_base_probe_pass_does_not_nli_score_answered_probes():
    """Category-error fix: the BASE probe pass NEVER NLI-scores answered probe
    answers. A spy on _score_base_groundedness asserts it is NOT invoked on any
    probe text; per-probe unsupported_rate is null with the category-error
    reason; the dropped roll-up fields are absent."""
    import lib.retrieval.eval_arms as ea

    probes, client = _refusal_safety_setup()
    probe_texts = {p["question_text"] for p in probes}

    orig = ea._score_base_groundedness
    scored_texts = []

    def _spy(answer_text, *, qtext, **kwargs):
        scored_texts.append(qtext)
        return orig(answer_text, qtext=qtext, **kwargs)

    ea._score_base_groundedness = _spy
    try:
        result = run_base_arm(
            Path("/unused"),
            "course-x",
            client=client,
            gold=_one_question_gold(),
            probes=probes,
        )
    finally:
        ea._score_base_groundedness = orig

    probe = result["probe_confabulation"]
    assert probe["answered"] == 2
    assert probe["declined"] == 1
    # No PROBE text was ever sent to the groundedness scorer (only the gold
    # question may have been — and only if NLI was available).
    assert not (probe_texts & set(scored_texts))
    # The dropped roll-up fields are GONE (no longer fabricated for BASE).
    assert "unsupported_answer_rate_on_probes" not in probe
    assert "claim_level_unsupported_rate_on_probes" not in probe
    assert "answered_probe_claims_scored" not in probe
    # Per-probe rows: unsupported_rate is null + carries the category-error
    # reason, for every answered probe.
    rows = {r["probe_id"]: r for r in probe["probes"]}
    for pid in ("rp-0001", "rp-0002"):
        assert rows[pid]["answered"] is True
        assert rows[pid]["unsupported_rate"] is None
        assert rows[pid]["unsupported_rate_reason"] == "category_error_offtopic_nli"
    assert rows["rp-0003"]["answered"] is False


def test_base_decision_rationale_carries_category_error_signal():
    """The base decision rationale carries the headline confabulation signal +
    notes the probe-answer NLI is skipped (category error); no new call site."""
    probes, client = _refusal_safety_setup()
    cap = _RecordingCapture()
    run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=_one_question_gold(),
        probes=probes,
        capture=cap,
    )
    events = [
        e for e in cap.events if e["decision_type"] == "base_model_eval_call"
    ]
    assert events
    assert {e["decision_type"] for e in cap.events} == {"base_model_eval_call"}
    assert any("out-of-scope confabulation" in e["rationale"] for e in events)
    # The reframe is reflected: answered base probe answers are NOT NLI-scored.
    assert any("NOT NLI-scored" in e["rationale"] for e in events)


# --- composite probe_fabrication_rate math + guards (GROUNDED-only) --------

from lib.retrieval.eval_arms import (  # noqa: E402
    _composite_probe_fabrication_rate,
    _base_refusal_safety,
    _grounded_refusal_safety,
    _build_refusal_safety,
)


def test_composite_probe_fabrication_rate_math_and_guards():
    # GROUNDED-only formula: answered 0.95, 0.40 of those unsupported → 0.38.
    assert _composite_probe_fabrication_rate(0.95, 0.40) == pytest.approx(0.38)
    # Either factor None / sentinel → None (never fabricated).
    assert _composite_probe_fabrication_rate(None, 0.40) is None
    assert _composite_probe_fabrication_rate(0.95, None) is None
    assert _composite_probe_fabrication_rate("—", 0.40) is None
    # Zero answered → 0.0 (a true zero, not n/a).
    assert _composite_probe_fabrication_rate(0.0, 0.40) == pytest.approx(0.0)


def test_base_refusal_safety_row_shape():
    base = {
        "arm": ARM_BASE,
        "probe_confabulation": {"confabulation_rate": 0.90},
    }
    row = _base_refusal_safety(base)
    # BASE refuses nothing — recall 0.0, precision None (honest, not 1.0).
    assert row["refusal_recall"] == 0.0
    assert row["refusal_precision"] is None
    # THE HEADLINE: answered-not-refused = the confabulation rate.
    assert row["answered_not_refused_rate"] == pytest.approx(0.90)
    # GROUNDED-only NLI sub-metric is the category-error sentinel for BASE.
    assert row["unsupported_answer_rate_on_answered_probes"] == "—"
    # INTRINSIC: base fabrication == answered_not_refused_rate (NOT NLI product).
    assert row["probe_fabrication_rate"] == pytest.approx(0.90)


def test_grounded_refusal_safety_row_from_headline():
    grounded = {
        "headline": {
            "refusal": {
                "n_probes": 44,
                "refusal_recall": 0.75,
                "refusal_precision": 0.96,
                "unsupported_answer_rate_on_answered_probes": 0.20,
            }
        }
    }
    row = _grounded_refusal_safety(grounded)
    assert row["refusal_recall"] == pytest.approx(0.75)
    assert row["refusal_precision"] == pytest.approx(0.96)
    # answered_not_refused_rate = 1 - recall = 0.25.
    assert row["answered_not_refused_rate"] == pytest.approx(0.25)
    assert row["unsupported_answer_rate_on_answered_probes"] == pytest.approx(0.20)
    # composite = 0.25 × 0.20 = 0.05 (NLI-derived; valid for grounded).
    assert row["probe_fabrication_rate"] == pytest.approx(0.05)


def test_comparison_refusal_safety_is_top_level_and_populates():
    """Regression: comparison.refusal_safety is a TOP-LEVEL block (peer of
    hallucination_reduction) that POPULATES — it is NOT empty, NOT nested per-arm
    column. (The empty-{} regression was that no top-level key existed at all.)"""
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    arms[ARM_BASE]["probe_confabulation"] = {"confabulation_rate": 0.95}
    arms[ARM_GROUNDED]["headline"]["refusal"] = {
        "n_probes": 44,
        "refusal_recall": 0.64,
        "refusal_precision": 0.96,
        "unsupported_answer_rate_on_answered_probes": 0.10,
    }
    comp = _build_comparison(arms)

    # TOP-LEVEL block exists and is non-empty.
    assert "refusal_safety" in comp
    safety = comp["refusal_safety"]
    assert safety and isinstance(safety, dict)
    assert "_note" in safety and "category error" in safety["_note"]

    # The per-arm column no longer nests a refusal_safety sub-key.
    assert "refusal_safety" not in comp[ARM_BASE]
    assert "refusal_safety" not in comp[ARM_GROUNDED]
    assert "refusal_safety" not in comp[ARM_RETRIEVAL]

    # BASE: recall 0.0 / precision None; HEADLINE answered-not-refused 0.95;
    # fabrication INTRINSIC (= answered_not_refused_rate, NOT NLI product);
    # unsupported-when-answered is the category-error sentinel.
    base_s = safety[ARM_BASE]
    assert base_s["refusal_recall"] == 0.0
    assert base_s["refusal_precision"] is None
    assert base_s["answered_not_refused_rate"] == pytest.approx(0.95)
    assert base_s["probe_fabrication_rate"] == pytest.approx(0.95)
    assert base_s["unsupported_answer_rate_on_answered_probes"] == "—"

    # GROUNDED: HEADLINE answered-not-refused = 1 - 0.64 = 0.36; fabrication =
    # 0.36 × 0.10 (NLI-derived).
    grounded_s = safety[ARM_GROUNDED]
    assert grounded_s["refusal_recall"] == pytest.approx(0.64)
    assert grounded_s["answered_not_refused_rate"] == pytest.approx(0.36)
    assert grounded_s["unsupported_answer_rate_on_answered_probes"] == pytest.approx(
        0.10
    )
    assert grounded_s["probe_fabrication_rate"] == pytest.approx(0.36 * 0.10)

    # RETRIEVAL = "—" throughout.
    assert safety[ARM_RETRIEVAL]["refusal_recall"] == "—"
    assert safety[ARM_RETRIEVAL]["probe_fabrication_rate"] == "—"


def test_build_refusal_safety_subset_of_arms():
    """The block populates from whichever arms ran (base-only here)."""
    arms = {
        ARM_BASE: {
            "arm": ARM_BASE,
            "probe_confabulation": {"confabulation_rate": 1.0},
        }
    }
    safety = _build_refusal_safety(arms)
    assert set(safety) == {ARM_BASE, "_note"}
    assert safety[ARM_BASE]["answered_not_refused_rate"] == pytest.approx(1.0)
    assert safety[ARM_BASE]["probe_fabrication_rate"] == pytest.approx(1.0)


def test_table_renders_reframed_safety_block_and_summary():
    arms = _arm_blocks_for_comparison(0.40, 0.05)
    arms[ARM_BASE]["probe_confabulation"] = {"confabulation_rate": 1.00}
    arms[ARM_GROUNDED]["headline"]["refusal"] = {
        "n_probes": 44,
        "refusal_recall": 0.64,
        "refusal_precision": 0.96,
        "unsupported_answer_rate_on_answered_probes": 0.10,
    }
    scorecard = {
        "course_slug": "course-x",
        "engine": "semantic",
        "arms": arms,
        "comparison": _build_comparison(arms),
    }
    table = format_scorecard_table(scorecard)
    # The clearly-headed SAFETY block + its reframed rows render.
    assert "SAFETY (refusal probes" in table
    assert "refusal_recall" in table
    assert "answered-not-refused (HEADLINE)" in table
    assert "unsupported-when-answered" in table
    assert "probe_fabrication_rate" in table
    # The unsupported-when-answered row shows the BASE category-error sentinel.
    assert "off-topic NLI category error" in table
    # The one-line reframed summary.
    assert "answered a question it should have refused" in table
    assert "of grounded's answered probes, NLI-ungrounded share" in table
    # BASE answered-not-refused 1.0000 (HEADLINE + intrinsic fabrication);
    # GROUNDED answered-not-refused = 1 - 0.64 = 0.3600; fabrication 0.0360.
    assert "1.0000" in table
    assert "0.3600" in table
    assert "0.0360" in table


# ===========================================================================
# Incremental progress wiring (lib.retrieval.eval_progress.EvalProgressWriter)
# ===========================================================================

import json as _json  # noqa: E402 — local to the progress tests

from lib.retrieval.eval_progress import (  # noqa: E402
    PROGRESS_JSONL_FILENAME,
    PROGRESS_SNAPSHOT_FILENAME,
    EvalProgressWriter,
)


def _base_setup_simple():
    """A 2-question base-arm setup with both answered (no groundedness axis)."""
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            "What does a vector store index?": (
                "A vector store indexes high dimensional embedding vectors and "
                "supports similarity search over them."
            ),
            "How is retrieval quality measured?": (
                "Retrieval quality is often measured by recall at k."
            ),
        }
    )
    return gold, client


def _strip_latency(arm_result):
    """Deep-copy an arm result with all ``latency_ms`` fields nulled.

    Latency is measured from a live monotonic clock, so it varies run-to-run
    independent of any code path — strip it to compare behavioral identity.
    """
    out = _json.loads(_json.dumps(arm_result))
    out.pop("latency_ms", None)
    for row in out.get("questions", []) or []:
        row.pop("latency_ms", None)
    return out


def test_base_arm_progress_none_is_byte_identical_regression():
    """progress=None must reproduce the legacy result EXACTLY (no-op path)."""
    gold, client = _base_setup_simple()
    without = run_base_arm(Path("/unused"), "course-x", client=client, gold=gold)
    gold2, client2 = _base_setup_simple()
    explicit_none = run_base_arm(
        Path("/unused"), "course-x", client=client2, gold=gold2, progress=None
    )
    # Identical modulo the inherently non-deterministic wall-clock latencies
    # (measured live; they vary run-to-run regardless of the progress arg).
    assert _strip_latency(without) == _strip_latency(explicit_none)


def test_retrieval_arm_progress_none_is_byte_identical_regression():
    gold = _synthetic_gold()
    by_query = {
        "What does a vector store index?": [
            _Result("chunk_a", "high dimensional embedding vectors similarity"),
        ],
        "How is retrieval quality measured?": [
            _Result("chunk_b", "recall at k mean reciprocal rank"),
        ],
    }
    fn = _retrieve_fn_factory(by_query)
    without = run_retrieval_arm(
        Path("/unused"), "course-x", retrieve_fn=fn, gold=gold
    )
    explicit_none = run_retrieval_arm(
        Path("/unused"), "course-x", retrieve_fn=fn, gold=gold, progress=None
    )
    assert _strip_latency(without) == _strip_latency(explicit_none)


def test_base_arm_progress_appends_one_line_per_item(tmp_path):
    gold, client = _base_setup_simple()
    w = EvalProgressWriter(
        tmp_path, run_id="ARM-RUN", arm_totals={ARM_BASE: 2}
    )
    result = run_base_arm(
        Path("/unused"), "course-x", client=client, gold=gold, progress=w
    )
    w.close()
    # The arm still returns its normal result (progress is additive).
    assert result["answered"] == 2
    lines = [
        _json.loads(x)
        for x in (tmp_path / PROGRESS_JSONL_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    # One line per scored gold question (the synthetic gold has no probes).
    assert len(lines) == 2
    assert all(line["arm"] == ARM_BASE for line in lines)
    assert [line["index"] for line in lines] == [1, 2]
    snap = _json.loads(
        (tmp_path / PROGRESS_SNAPSHOT_FILENAME).read_text(encoding="utf-8")
    )
    assert snap["state"] == "complete"
    assert snap["arms"][ARM_BASE]["done"] == 2
    assert snap["arms"][ARM_BASE]["finished"] is True


def test_retrieval_arm_progress_records_every_question(tmp_path):
    gold = _synthetic_gold()
    by_query = {
        "What does a vector store index?": [
            _Result("chunk_a", "high dimensional embedding vectors similarity"),
        ],
        "How is retrieval quality measured?": [
            _Result("chunk_b", "recall at k mean reciprocal rank"),
        ],
    }
    w = EvalProgressWriter(
        tmp_path, run_id="RETR-RUN", arm_totals={ARM_RETRIEVAL: 2}
    )
    run_retrieval_arm(
        Path("/unused"),
        "course-x",
        retrieve_fn=_retrieve_fn_factory(by_query),
        gold=gold,
        progress=w,
    )
    w.close()
    lines = [
        _json.loads(x)
        for x in (tmp_path / PROGRESS_JSONL_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    assert len(lines) == 2  # retrieval scores every gold question
    assert all(line["arm"] == ARM_RETRIEVAL for line in lines)


def test_scorecard_progress_off_writes_no_progress_files(libv2_course):
    """progress=False (the --no-progress path) constructs no writer → no files."""
    repo_root, slug, course_dir = libv2_course
    eval_dir = course_dir / "retrieval_eval"
    run_scorecard(
        repo_root, slug, engine="lexical", write=False, progress=False,
        grounded_kwargs={"answer_fn": _fake_grounded_fn()},
    )
    assert not (eval_dir / PROGRESS_JSONL_FILENAME).exists()
    assert not (eval_dir / PROGRESS_SNAPSHOT_FILENAME).exists()


def test_scorecard_progress_on_lays_down_run_scoped_files(libv2_course):
    """progress=True (default) constructs ONE writer, finalizes it complete."""
    repo_root, slug, course_dir = libv2_course
    eval_dir = course_dir / "retrieval_eval"
    run_scorecard(
        repo_root, slug, arms=[ARM_RETRIEVAL], engine="lexical", write=False,
        progress=True, progress_run_id="SC-RUN",
        retrieve_fn=_fake_retrieve_for_fixture(),
    )
    snap = _json.loads(
        (eval_dir / PROGRESS_SNAPSHOT_FILENAME).read_text(encoding="utf-8")
    )
    assert snap["run_id"] == "SC-RUN"
    assert snap["state"] == "complete"  # finalized after the run
    assert ARM_RETRIEVAL in snap["arms"]
    lines = [
        x
        for x in (eval_dir / PROGRESS_JSONL_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    assert lines  # at least one per-item line landed during the run


class _BoomWriter:
    """A progress writer whose every method raises — must NOT abort the eval."""

    def start_arm(self, *a, **k):
        raise RuntimeError("boom start")

    def record_item(self, *a, **k):
        raise RuntimeError("boom record")

    def finish_arm(self, *a, **k):
        raise RuntimeError("boom finish")

    def close(self, *a, **k):
        raise RuntimeError("boom close")


def test_base_arm_progress_errors_are_swallowed(tmp_path):
    """A progress method that raises must not propagate into the eval logic."""
    gold, client = _base_setup_simple()
    # No exception escapes despite every progress call raising.
    result = run_base_arm(
        Path("/unused"),
        "course-x",
        client=client,
        gold=gold,
        progress=_BoomWriter(),
    )
    # The eval still produced its normal, correct result.
    assert result["answered"] == 2
    assert result["key_point_coverage"]["covered_key_points"] == 3
