"""Tests for the schema-1.7 grounded-eval scoring upgrades.

Covers the two-axis abstention scorer + premise-correction judge, Wilson CIs,
the flag-config stamp, the re-semantics'd citation metrics (macro precision +
recall), and micro groundedness. CI-safe: the grounded-answer pipeline is
injected via ``answer_fn`` and the premise-correction judge via a STUBBED
``judge_client`` (no network, no model weights) exactly like the P4 suite.

Law 3 (decision-capture on every NEW LLM call site): the judge is a new LLM
call site, so one test asserts a ``fresh_eval_invocation`` decision fires on a
real ``DecisionCapture`` when the judge runs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lib.decision_capture import DecisionCapture
from lib.retrieval.grounded_eval import (
    ENV_EVAL_JUDGE_MODEL,
    ENV_EVAL_JUDGE_PROVIDER,
    run_grounded_eval,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)

#: A false-premise (ill_posed) probe text answered by the fake pipeline below.
FALSE_PREMISE_TEXT = "Why does a vector store delete embeddings after each query?"

#: The three answerable gold questions in the fixture, mapped to their primary
#: relevant chunk (mirrors the P4/base suite fixture).
_GOLD_MAP = {
    "What does a vector store index?": "mini_alpha_chunk_001",
    "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
    "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
}


# ===========================================================================
# Fixtures + fakes
# ===========================================================================

@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


class _FakeCitation:
    def __init__(self, chunk_id, anchor_status="resolved_exact",
                 page_label="P", text_quote="q"):
        self.chunk_id = chunk_id
        self.anchor_status = anchor_status
        self.page_label = page_label
        self.text_quote = text_quote

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "anchor_status": self.anchor_status,
            "page_label": self.page_label,
            "text_quote": self.text_quote,
        }


class _FakeAnswer:
    def __init__(self, *, status, answer_text, citations, groundedness=None,
                 latency_ms=10.0):
        self.status = status
        self.answer_text = answer_text
        self.citations = citations
        self.groundedness = groundedness
        self.latency_ms = latency_ms
        self.model_id = "fake-model"
        self.prompt_version = "ws3.v1"
        self.confidence = {"policy_version": "ws3.v0"}


class _StubJudge:
    """A stub OpenAICompatibleClient — records calls, returns a canned verdict.

    ``corrected`` controls the JSON verdict; ``raise_exc`` simulates a backend
    failure/timeout; ``raw`` forces a raw (possibly unparseable) response."""

    def __init__(self, *, corrected=True, confidence=0.9, raise_exc=False,
                 raw=None):
        self.corrected = corrected
        self.confidence = confidence
        self.raise_exc = raise_exc
        self.raw = raw
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=256, temperature=0.0,
                        **kwargs):
        self.calls.append(messages)
        if self.raise_exc:
            raise RuntimeError("judge backend down")
        if self.raw is not None:
            return self.raw
        return json.dumps(
            {
                "reasoning": "the answer disputes the premise",
                "flags_false_premise": self.corrected,
                "confidence": self.confidence,
            }
        )


def _fp_probes_file(course_dir, *, expected_outcome="correct_premise",
                    category="ill_posed"):
    """Write a single false-premise probe file and return its path."""
    path = course_dir / "retrieval_eval" / "fp_probes.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "course_slug": "mini-retrieval-101",
                "probes": [
                    {
                        "probe_id": "ip-mini-0001",
                        "question_text": FALSE_PREMISE_TEXT,
                        "category": category,
                        "expected_outcome": expected_outcome,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fp_answer_fn(*, answer_the_fp=True):
    """Fake pipeline: answers the gold questions, ANSWERS the false-premise
    probe (so it becomes a judge candidate) unless ``answer_the_fp`` is False."""

    def _fn(repo_root, course_slug, query, *, engine="lexical", limit=8,
            client=None, refusal_policy=None, with_groundedness=False,
            capture=None, **kwargs):
        if query == FALSE_PREMISE_TEXT:
            if answer_the_fp:
                return _FakeAnswer(
                    status="answered",
                    answer_text="A vector store does not delete embeddings.",
                    citations=[_FakeCitation("mini_alpha_chunk_001")],
                )
            return _FakeAnswer(status="refused_low_confidence",
                               answer_text=None, citations=[])
        cid = _GOLD_MAP.get(query)
        if cid is not None:
            return _FakeAnswer(status="answered",
                               answer_text=f"About {cid}.",
                               citations=[_FakeCitation(cid)])
        return _FakeAnswer(status="refused_low_confidence",
                           answer_text=None, citations=[])

    return _fn


# ===========================================================================
# Two-axis abstention block shape + defaults (no false-premise items)
# ===========================================================================

def test_abstention_block_present_default_no_false_premise(libv2_course):
    """With the stock fixture (no false-premise items) the abstention block is
    present, the judge never resolves (enabled False, zero candidates), and all
    three gold questions land on the answerable axis / answered outcome."""
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    assert ab["axis_totals"]["answerable"] == 3
    assert ab["axis_totals"]["false_premise"] == 0
    # 3 stock probes are unanswerable + refused.
    assert ab["axis_totals"]["unanswerable"] == 3
    assert ab["outcome_matrix"]["answerable"]["answered"] == 3
    assert ab["outcome_matrix"]["unanswerable"]["refused"] == 3
    # No false-premise items → premise_correction rate is None (not a fake 0.0).
    assert ab["premise_correction"]["rate"] is None
    assert ab["premise_correction"]["false_premise_items"] == 0
    # Judge never ran (no candidates, no client resolution attempted).
    assert ab["judge"]["enabled"] is False
    assert ab["judge"]["candidates"] == 0
    # Legacy refusal block preserved + marked legacy.
    assert "_note" in report["headline"]["refusal"]
    assert report["headline"]["refusal"]["refusal_recall"] == 1.0


# ===========================================================================
# Premise-correction judge — credited outcome + capture fires (law 3)
# ===========================================================================

def test_judge_credits_premise_correction_and_capture_fires(libv2_course):
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=True, confidence=0.88)
    capture = DecisionCapture("FIX_001", "grounded_eval_judge", "trainforge")

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        capture=capture, with_groundedness=False, write=False,
    )

    ab = report["headline"]["abstention"]
    # The one false-premise probe was answered → judged → corrected.
    assert ab["axis_totals"]["false_premise"] == 1
    assert ab["outcome_matrix"]["false_premise"]["premise_corrected"] == 1
    assert ab["outcome_matrix"]["false_premise"]["answered"] == 0
    assert ab["premise_correction"]["corrected"] == 1
    assert ab["premise_correction"]["rate"] == 1.0
    # Judge metadata records it ran on the injected client.
    assert ab["judge"]["enabled"] is True
    assert ab["judge"]["candidates"] == 1
    assert ab["judge"]["judged"] == 1
    assert ab["judge"]["degraded"] == 0
    assert ab["judge"]["provider"] == "injected"
    # The judge actually got called once.
    assert len(judge.calls) == 1

    # Law 3: a fresh_eval_invocation decision fired on the capture at the judge
    # call site (the fake pipeline logs nothing, so this decision is the judge's).
    judge_decisions = [
        d for d in capture.decisions
        if d["decision_type"] == "fresh_eval_invocation"
    ]
    assert len(judge_decisions) == 1
    rat = judge_decisions[0]["rationale"]
    assert len(rat) >= 20
    assert "flags_false_premise=True" in rat
    assert "ip-mini-0001" in rat  # dynamic — interpolates the item id
    capture.close()


def test_judge_answered_uncorrected_is_failure(libv2_course):
    """A false-premise item answered WITHOUT flagging the premise stays
    'answered' (the failure outcome) — premise_corrected is NOT credited."""
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=False)

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    assert ab["outcome_matrix"]["false_premise"]["premise_corrected"] == 0
    assert ab["outcome_matrix"]["false_premise"]["answered"] == 1
    assert ab["premise_correction"]["rate"] == 0.0
    assert ab["premise_correction"]["answered_uncorrected"] == 1
    assert ab["judge"]["judged"] == 1
    assert ab["judge"]["degraded"] == 0


def test_judge_failure_degrades_to_answered_without_crashing(libv2_course):
    """A judge exception / unparseable verdict degrades the item to 'answered'
    with a warning — the eval never crashes, degraded count reflects it."""
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(raise_exc=True)
    capture = DecisionCapture("FIX_002", "grounded_eval_judge", "trainforge")

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        capture=capture, with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    # Degraded: outcome falls back to answered, premise_corrected not credited.
    assert ab["outcome_matrix"]["false_premise"]["answered"] == 1
    assert ab["outcome_matrix"]["false_premise"]["premise_corrected"] == 0
    assert ab["judge"]["judged"] == 0
    assert ab["judge"]["degraded"] == 1
    # A decision STILL fires on the failed judge call (records the degrade).
    judge_decisions = [
        d for d in capture.decisions
        if d["decision_type"] == "fresh_eval_invocation"
    ]
    assert len(judge_decisions) == 1
    assert "degraded_to_answered" in judge_decisions[0]["rationale"]
    capture.close()


def test_unparseable_judge_response_degrades(libv2_course):
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(raw="not json at all")

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    assert ab["judge"]["degraded"] == 1
    assert ab["outcome_matrix"]["false_premise"]["answered"] == 1


def test_refused_false_premise_probe_is_not_judged(libv2_course):
    """A false-premise probe the pipeline REFUSED is acceptable — it is not
    answered, so the judge never runs on it (candidates == 0)."""
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=True)

    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_fp_answer_fn(answer_the_fp=False),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    assert ab["outcome_matrix"]["false_premise"]["refused"] == 1
    assert ab["judge"]["candidates"] == 0
    assert len(judge.calls) == 0
    # A refusal on a false-premise item is an acceptable decline → recall 1.0.
    assert ab["corrected_refusal"]["recall"] == 1.0


# ===========================================================================
# Corrected-refusal recall/precision math
# ===========================================================================

def test_corrected_refusal_math_credits_correction_and_refusal(libv2_course):
    """Refusable universe = false_premise + unanswerable. A corrected
    false-premise item + refused unanswerable probes all count as correct
    declines; a false refusal on answerable gold is the only wrong decline."""
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=True)

    # Pipeline: gold answered (no false refusals), 3 stock unanswerable probes
    # refused, 1 false-premise probe answered+corrected.
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=False,
    )
    cr = report["headline"]["abstention"]["corrected_refusal"]
    # refusable = 1 false_premise + 0 unanswerable-probes... note: the injected
    # probes file REPLACES the stock probes, so only the 1 false-premise probe
    # is present. refusable = 1; it was corrected → recall 1.0.
    assert cr["refusable_items"] == 1
    assert cr["recall"] == 1.0
    # declines = corrections (1) + refusals (0) = 1; all correct → precision 1.0
    assert cr["correct_declines"] == 1
    assert cr["precision"] == 1.0


# ===========================================================================
# Gold false-premise item (overwrite gold_set.json)
# ===========================================================================

def _make_gold_false_premise(course_dir):
    """Rewrite the fixture gold_set.json → schema 1.2 with the first question
    flagged expected_behavior=correct_premise (chunkset pin preserved)."""
    gp = course_dir / "retrieval_eval" / "gold_set.json"
    doc = json.loads(gp.read_text(encoding="utf-8"))
    doc["schema_version"] = "1.2"
    doc["questions"][0]["expected_behavior"] = "correct_premise"
    gp.write_text(json.dumps(doc), encoding="utf-8")


def test_gold_false_premise_item_judged_and_credited(libv2_course):
    repo_root, slug, course_dir = libv2_course
    _make_gold_false_premise(course_dir)
    judge = _StubJudge(corrected=True)

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        judge_client=judge, with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    # One gold question is now a false-premise item; the other two answerable.
    assert ab["axis_totals"]["false_premise"] == 1
    assert ab["axis_totals"]["answerable"] == 2
    assert ab["outcome_matrix"]["false_premise"]["premise_corrected"] == 1
    assert ab["judge"]["candidates"] == 1


# ===========================================================================
# Wilson CIs
# ===========================================================================

def test_wilson_cis_present_and_small_buckets_are_diagnostic(libv2_course):
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=True)
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=False,
    )
    # Phrasing breakdown CI.
    for bucket in report["headline"]["phrasing_breakdown"].values():
        ci = bucket["answered_rate_ci"]
        assert set(ci) == {"lo", "hi", "n", "basis"}
        assert ci["basis"] == "diagnostic"  # tiny fixture, n<30

    ab = report["headline"]["abstention"]
    assert ab["premise_correction"]["rate_ci"]["basis"] == "diagnostic"
    assert ab["corrected_refusal"]["recall_ci"]["basis"] == "diagnostic"
    assert ab["corrected_refusal"]["precision_ci"]["basis"] == "diagnostic"

    # Per-category probe stats CI.
    by_cat = report["headline"]["refusal"]["by_category"]
    assert "ill_posed" in by_cat
    assert by_cat["ill_posed"]["refused_rate_ci"]["basis"] == "diagnostic"
    assert by_cat["ill_posed"]["n"] == 1


def test_wilson_ci_math():
    from lib.retrieval.grounded_eval import _wilson_ci

    empty = _wilson_ci(0, 0)
    assert empty["lo"] is None and empty["hi"] is None
    assert empty["basis"] == "diagnostic"

    full = _wilson_ci(50, 50)
    assert full["basis"] == "sufficient"  # n >= 30
    assert 0.0 <= full["lo"] <= full["hi"] <= 1.0
    assert full["hi"] == 1.0  # all successes → upper bound pinned at 1

    half = _wilson_ci(15, 30)
    assert half["basis"] == "sufficient"
    assert half["lo"] < 0.5 < half["hi"]


# ===========================================================================
# Judge-client resolution — env override routing + loopback + graceful degrade
# ===========================================================================

def test_resolve_judge_client_honors_env_override(monkeypatch):
    """The judge routes through answer_backend with the NEW override pair; the
    model override wins and the resolved base_url is loopback-enforced."""
    monkeypatch.setenv(ENV_EVAL_JUDGE_PROVIDER, "local")
    monkeypatch.setenv(ENV_EVAL_JUDGE_MODEL, "custom-judge-model")
    from lib.retrieval.grounded_eval import _resolve_judge_client

    client, resolved = _resolve_judge_client(capture=None)
    assert resolved is not None
    assert resolved.model_id == "custom-judge-model"
    assert "localhost" in resolved.base_url or "127.0.0.1" in resolved.base_url
    assert client is not None


def test_resolve_judge_client_unknown_provider_degrades(monkeypatch):
    """An unresolvable judge provider degrades to (None, None) — the eval then
    falls back to 'answered' for every candidate rather than crashing."""
    monkeypatch.setenv(ENV_EVAL_JUDGE_PROVIDER, "definitely-not-a-provider")
    from lib.retrieval.grounded_eval import _resolve_judge_client

    client, resolved = _resolve_judge_client(capture=None)
    assert client is None
    assert resolved is None


def test_unresolvable_judge_degrades_all_candidates(libv2_course, monkeypatch):
    """With false-premise candidates but no injected client AND an unresolvable
    judge backend, every candidate degrades to 'answered' (judge disabled)."""
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    monkeypatch.setenv(ENV_EVAL_JUDGE_PROVIDER, "definitely-not-a-provider")

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, with_groundedness=False, write=False,
    )
    ab = report["headline"]["abstention"]
    assert ab["judge"]["enabled"] is False
    assert ab["judge"]["candidates"] == 1
    assert ab["judge"]["degraded"] == 1
    assert ab["outcome_matrix"]["false_premise"]["answered"] == 1
    assert ab["outcome_matrix"]["false_premise"]["premise_corrected"] == 0


# ===========================================================================
# Flag-config stamp
# ===========================================================================

def test_flag_config_stamp_records_answer_and_judge_env(libv2_course, monkeypatch):
    repo_root, slug, _ = libv2_course
    monkeypatch.setenv("ED4ALL_ANSWER_NUM_CTX", "8192")
    monkeypatch.setenv("ED4ALL_ANSWER_NLI_ADD", "shadow")
    monkeypatch.setenv(ENV_EVAL_JUDGE_PROVIDER, "local")
    monkeypatch.setenv(ENV_EVAL_JUDGE_MODEL, "qwen2.5:7b")

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        with_groundedness=False, write=False,
    )
    fc = report["flag_config"]
    assert fc["answer_env"]["ED4ALL_ANSWER_NUM_CTX"] == "8192"
    assert fc["answer_env"]["ED4ALL_ANSWER_NLI_ADD"] == "shadow"
    # An unset documented flag is recorded as None (attributable either way).
    assert "ED4ALL_ANSWER_PROVIDER" in fc["answer_env"]
    assert fc["eval_judge"][ENV_EVAL_JUDGE_PROVIDER] == "local"
    assert fc["eval_judge"][ENV_EVAL_JUDGE_MODEL] == "qwen2.5:7b"


# ===========================================================================
# Citation metrics re-semantics (macro precision + recall) + micro groundedness
# ===========================================================================

_GROUNDED_OK = {
    "available": True,
    "groundedness_rate": 1.0,
    "scored_count": 4,
    "unsupported_count": 1,
    "contradicted_count": 0,
    "claims": [{"verdict": "entailed"}],
}


def _grounded_answer_fn():
    def _fn(repo_root, course_slug, query, *, engine="lexical", limit=8,
            client=None, refusal_policy=None, with_groundedness=False,
            capture=None, **kwargs):
        cid = _GOLD_MAP.get(query)
        if cid is None:
            return _FakeAnswer(status="refused_low_confidence",
                               answer_text=None, citations=[])
        grounded = _GROUNDED_OK if with_groundedness else None
        return _FakeAnswer(status="answered", answer_text=f"About {cid}.",
                           citations=[_FakeCitation(cid)], groundedness=grounded)
    return _fn


def test_citation_precision_is_macro_and_legacy_pooled_kept(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_grounded_answer_fn(),
        with_groundedness=False, write=False,
    )
    h = report["headline"]
    # Every gold answer cites exactly its 1 relevant chunk → per-question
    # precision 1.0 each → macro 1.0; pooled legacy also 3/3 = 1.0.
    assert h["citation_precision"] == 1.0
    assert h["citation_precision_legacy"] == 1.0
    # Recall is the multi-passage-aware signal: Q1/Q2 pin one passage each (cited
    # → 1.0), Q3 pins TWO (primary+supporting) but the answer cites only the
    # primary → recall 0.5. macro = (1 + 1 + 0.5) / 3.
    assert h["citation_recall"] == pytest.approx(5 / 6)
    assert h["citation_recall_basis"]["answered_questions_with_pinned_passage"] == 3
    # Exactly Q3 pins >1 relevant passage.
    assert h["citation_recall_basis"]["multi_passage_questions"] == 1


def test_groundedness_rate_micro_headline(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_grounded_answer_fn(),
        with_groundedness=True, write=False,
    )
    h = report["headline"]
    # Each answered question scored 4 claims, entailed = 4-1-0 = 3.
    # micro = total entailed / total scored = (3*3)/(4*3) = 0.75.
    assert h["groundedness_rate_micro"] == pytest.approx(0.75)
    # macro mean of per-question groundedness_rate (all 1.0) stays 1.0.
    assert h["groundedness_rate_mean"] == 1.0


def test_micro_groundedness_none_when_unavailable(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_grounded_answer_fn(),
        with_groundedness=False, write=False,
    )
    assert report["headline"]["groundedness_rate_micro"] is None


# ===========================================================================
# Report JSON round-trips with the new blocks
# ===========================================================================

def test_report_round_trips_with_new_blocks(libv2_course):
    repo_root, slug, course_dir = libv2_course
    probes = _fp_probes_file(course_dir)
    judge = _StubJudge(corrected=True)
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fp_answer_fn(),
        refusal_probes_path=probes, judge_client=judge,
        with_groundedness=False, write=True,
    )
    doc = json.loads(Path(report["_written"]["report_path"]).read_text("utf-8"))
    assert doc["schema_version"] == "1.8"
    assert "abstention" in doc["headline"]
    assert "flag_config" in doc
    assert "citation_recall" in doc["headline"]
    assert "groundedness_rate_micro" in doc["headline"]
    assert "by_category" in doc["headline"]["refusal"]
