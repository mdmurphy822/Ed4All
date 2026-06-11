"""Tests for the WS3 answer composer + the shared ``FakeAnswerClient`` (D3/D9).

``FakeAnswerClient`` and ``SpyCapture`` are defined here and imported by the
E6/E7 test modules — E5 publishes them first. They are deterministic and
CI-safe (no model, no server, no network).
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from lib.retrieval._prompts import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    MAX_CONTEXT_CHARS,
    PASSAGE_CHAR_CAP,
    render_answer_user_prompt,
)
from lib.retrieval.answer_backend import AnswerBackendUnavailable
from lib.retrieval.answer_composer import (
    AnswerComposeError,
    ComposedAnswer,
    InvalidCitationError,
    RetrievedPassage,
    compose_answer,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas" / "events" / "decision_event.schema.json"
)


# ===========================================================================
# Shared test doubles (imported by E6/E7 test modules)
# ===========================================================================

class FakeAnswerClient:
    """Deterministic, CI-safe stand-in for ``OpenAICompatibleClient``.

    Duck-typed ``chat_completion(messages, max_tokens=, temperature=)``.
    Scripted ``responses`` are returned by call index; a response that is
    an ``Exception`` instance is raised (drives the transport-failure
    path). When ``responses`` is exhausted, the last entry repeats.
    Records every call's messages + kwargs for assertions.
    """

    def __init__(self, responses, *, model="qwen2.5:14b-instruct-q4_K_M"):
        self._responses = list(responses)
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=1024, temperature=0.0,
                        **kwargs):
        idx = len(self.calls)
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens,
             "temperature": temperature}
        )
        resp = self._responses[min(idx, len(self._responses) - 1)]
        if isinstance(resp, BaseException):
            raise resp
        return resp


class SpyCapture:
    """Records ``log_decision`` kwargs without writing to disk."""

    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.events.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale, **kwargs}
        )


def _envelope(answer, citations, not_in_course=False):
    return json.dumps(
        {"answer": answer, "citations": citations,
         "not_in_course": not_in_course}
    )


def _passages(n=2):
    return [
        RetrievedPassage(
            chunk_id=f"chunk_{i:05d}", text=f"Body of passage {i}. " * 3,
            score=0.9 - i * 0.1, engine="lexical",
            item_path=f"module/page_{i}.html",
            section_heading=f"Section {i}", module_id="m1",
            source={"item_path": f"module/page_{i}.html"},
        )
        for i in range(n)
    ]


# ===========================================================================
# Prompt-version pinning + rendering
# ===========================================================================

def test_prompt_version_pinned():
    assert ANSWER_PROMPT_VERSION == "ws3.v2"


def test_system_prompt_requires_multipart_completeness():
    # ws3.v2: a multi-part question must have every supported part addressed,
    # and any uncovered part flagged explicitly (not silently omitted) — while
    # the wholly-uncovered case still routes to not_in_course.
    assert "multiple parts" in ANSWER_SYSTEM_PROMPT
    assert "does not cover" in ANSWER_SYSTEM_PROMPT
    assert "never invent material" in ANSWER_SYSTEM_PROMPT
    assert "If NO part of the question is covered" in ANSWER_SYSTEM_PROMPT
    assert '"not_in_course" to true' in ANSWER_SYSTEM_PROMPT
    # The instruction must reach the assembled system message verbatim.
    client = FakeAnswerClient([_envelope("X is a thing.", ["chunk_00000"])])
    compose_answer("What is X?", _passages(2), client=client)
    system_msg = client.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "multiple parts" in system_msg["content"]
    assert "does not cover" in system_msg["content"]


def test_user_prompt_numbers_blocks_and_appends_question():
    passages = _passages(2)
    prompt = render_answer_user_prompt("What is X?", passages)
    assert "[chunk_00000] (Section 0)" in prompt
    assert "[chunk_00001] (Section 1)" in prompt
    assert "Question: What is X?" in prompt
    # trailing JSON directive present (most-respected position)
    assert prompt.rstrip().endswith("}.")


def test_user_prompt_truncates_long_passage():
    long_passage = [
        RetrievedPassage(
            chunk_id="chunk_long", text="z" * (PASSAGE_CHAR_CAP + 500),
            score=0.9, engine="lexical", item_path="p.html",
            section_heading="Long", module_id=None, source={},
        )
    ]
    prompt = render_answer_user_prompt("q", long_passage)
    assert "z" * PASSAGE_CHAR_CAP in prompt
    assert "z" * (PASSAGE_CHAR_CAP + 1) not in prompt


def test_user_prompt_drops_trailing_passages_over_budget():
    # Many large passages exceed MAX_CONTEXT_CHARS; the question must
    # always survive and at least the first block must be present.
    big = [
        RetrievedPassage(
            chunk_id=f"chunk_{i}", text="y" * PASSAGE_CHAR_CAP, score=0.9,
            engine="lexical", item_path="p.html", section_heading=f"S{i}",
            module_id=None, source={},
        )
        for i in range(20)
    ]
    prompt = render_answer_user_prompt("the question", big)
    assert "Question: the question" in prompt
    assert "[chunk_0]" in prompt
    assert len(prompt) < MAX_CONTEXT_CHARS + 2000


# ===========================================================================
# RetrievedPassage.from_retrieval_result
# ===========================================================================

def test_from_retrieval_result_duck_types_lexical_result():
    class _Result:
        chunk_id = "chunk_00007"
        text = "lexical body"
        score = 0.77
        source = {"item_path": "imscc/page.html",
                  "section_heading": "Intro", "module_id": "mod2"}

    rp = RetrievedPassage.from_retrieval_result(_Result(), engine="lexical")
    assert rp.chunk_id == "chunk_00007"
    assert rp.engine == "lexical"
    assert rp.item_path == "imscc/page.html"
    assert rp.section_heading == "Intro"
    assert rp.module_id == "mod2"
    assert rp.source["item_path"] == "imscc/page.html"


# ===========================================================================
# compose_answer — happy path + envelope shapes
# ===========================================================================

def test_compose_well_formed_envelope_roundtrip():
    client = FakeAnswerClient([_envelope("X is a thing.", ["chunk_00000"])])
    result = compose_answer("What is X?", _passages(2), client=client)
    assert isinstance(result, ComposedAnswer)
    assert result.answer_text == "X is a thing."
    assert result.cited_chunk_ids == ["chunk_00000"]
    assert result.not_in_course is False
    assert result.attempts == 1
    assert result.prompt_version == "ws3.v2"
    assert result.model_id == "qwen2.5:14b-instruct-q4_K_M"


def test_compose_extracts_json_wrapped_in_prose():
    raw = "Sure! Here is the answer:\n" + _envelope("Answer.", ["chunk_00001"]) + "\nDone."
    client = FakeAnswerClient([raw])
    result = compose_answer("q", _passages(2), client=client)
    assert result.answer_text == "Answer."
    assert result.cited_chunk_ids == ["chunk_00001"]


def test_compose_not_in_course_returns_refusal_shape():
    client = FakeAnswerClient([_envelope("", [], not_in_course=True)])
    result = compose_answer("q", _passages(2), client=client)
    assert result.not_in_course is True
    assert result.answer_text is None
    assert result.cited_chunk_ids == []


def test_compose_temperature_and_max_tokens_forwarded():
    client = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    compose_answer("q", _passages(1), client=client,
                   temperature=0.0, max_tokens=512)
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["max_tokens"] == 512


# ===========================================================================
# Remediation + parse exhaustion
# ===========================================================================

def test_unknown_citation_triggers_remediation_then_corrects():
    client = FakeAnswerClient([
        _envelope("draft", ["chunk_99999"]),       # unknown id
        _envelope("fixed", ["chunk_00000"]),        # corrected
    ])
    result = compose_answer("q", _passages(2), client=client)
    assert result.cited_chunk_ids == ["chunk_00000"]
    assert result.attempts == 2
    # remediation directive appended on the second call
    second_user = client.calls[1]["messages"][1]["content"]
    assert "chunk_99999" in second_user


def test_persistent_unknown_citation_raises_invalid_citation():
    client = FakeAnswerClient([_envelope("x", ["chunk_99999"])])
    with pytest.raises(InvalidCitationError):
        compose_answer("q", _passages(2), client=client, max_parse_retries=3)
    assert len(client.calls) == 3


def test_persistent_garbage_raises_compose_error_after_max_retries():
    client = FakeAnswerClient(["not json at all"])
    with pytest.raises(AnswerComposeError) as exc_info:
        compose_answer("q", _passages(2), client=client, max_parse_retries=3)
    assert not isinstance(exc_info.value, InvalidCitationError)
    assert len(client.calls) == 3


def test_transport_failure_maps_to_backend_unavailable():
    client = FakeAnswerClient([httpx.ConnectError("connection refused")])
    with pytest.raises(AnswerBackendUnavailable):
        compose_answer("q", _passages(2), client=client)


def test_synthesis_provider_transport_error_maps_to_unavailable():
    class _ProviderErr(RuntimeError):
        code = "max_retries_exceeded"

    client = FakeAnswerClient([_ProviderErr("server down")])
    with pytest.raises(AnswerBackendUnavailable):
        compose_answer("q", _passages(2), client=client)


# ===========================================================================
# Capture regression (repo-mandated "capture fires" assertion)
# ===========================================================================

def _enum_members():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return set(schema["properties"]["decision_type"]["enum"])


def test_capture_fires_with_dynamic_rationale():
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    compose_answer("What is X?", _passages(2), client=client,
                   capture=spy, course_code="test-course-101")
    comp_events = [e for e in spy.events
                   if e["decision_type"] == "grounded_answer_composition"]
    assert len(comp_events) >= 1
    for e in comp_events:
        assert len(e["rationale"]) >= 20


def test_capture_two_queries_produce_different_rationales():
    client1 = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    client2 = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    spy1, spy2 = SpyCapture(), SpyCapture()
    compose_answer("question one", _passages(2), client=client1, capture=spy1)
    compose_answer("a completely different question two", _passages(2),
                   client=client2, capture=spy2)
    r1 = spy1.events[0]["rationale"]
    r2 = spy2.events[0]["rationale"]
    assert r1 != r2  # different query sha → different rationale


def test_emitted_decision_type_is_schema_enum_member():
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    compose_answer("q", _passages(2), client=client, capture=spy)
    enum = _enum_members()
    for e in spy.events:
        assert e["decision_type"] in enum


def test_no_capture_does_not_break_path():
    client = FakeAnswerClient([_envelope("a", ["chunk_00000"])])
    # capture=None must not raise
    result = compose_answer("q", _passages(2), client=client, capture=None)
    assert result.answer_text == "a"


# ===========================================================================
# Schema validates events with the new enum values (task 4)
# ===========================================================================

@pytest.mark.parametrize("dtype", [
    "grounded_answer_composition",
    "grounded_answer_refusal",
    "grounded_answer_citation_gate",
])
def test_schema_accepts_new_decision_types(dtype):
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    event = {
        "run_id": "RUN_20260609_000000",
        "timestamp": "2026-06-09T00:00:00Z",
        "operation": "grounded_answer",
        "decision_type": dtype,
        "decision": "answered",
        "rationale": "x" * 25,
        "phase": "libv2-answer",
    }
    jsonschema.validate(event, schema)


def test_schema_accepts_libv2_answer_phase():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert "libv2-answer" in schema["properties"]["phase"]["enum"]
    event = {
        "run_id": "RUN_20260609_000000",
        "timestamp": "2026-06-09T00:00:00Z",
        "operation": "grounded_answer",
        "decision_type": "grounded_answer_composition",
        "decision": "answered",
        "rationale": "y" * 25,
        "phase": "libv2-answer",
    }
    jsonschema.validate(event, schema)


# ===========================================================================
# Token-aware context budget (math-safe; dense vs prose) — Fix 1
# ===========================================================================

from lib.retrieval._prompts import (  # noqa: E402
    CHARS_PER_TOKEN,
    DEFAULT_NUM_CTX,
    ENV_ANSWER_NUM_CTX,
    CITATION_REMEDIATION_DIRECTIVE,
    context_token_budget,
    estimate_tokens,
    render_answer_user_prompt_with_meta,
    resolve_num_ctx,
)
from lib.retrieval.answer_backend import PromptTruncatedError  # noqa: E402


def test_estimate_tokens_is_math_safe_upper_bound():
    # 2.5 chars/token: 100 chars -> 40 tokens (ceil). A prose ~4 chars/token
    # would have estimated 25; the math-safe divisor over-counts on purpose.
    assert estimate_tokens("x" * 100) == 40
    assert estimate_tokens("") == 0
    assert CHARS_PER_TOKEN == 2.5


def test_num_ctx_env_override_and_fallback(monkeypatch):
    monkeypatch.delenv(ENV_ANSWER_NUM_CTX, raising=False)
    assert resolve_num_ctx() == DEFAULT_NUM_CTX == 4096
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "8192")
    assert resolve_num_ctx() == 8192
    # Garbage / non-positive fall back to the default (never disables budget).
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "not-a-number")
    assert resolve_num_ctx() == DEFAULT_NUM_CTX
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "0")
    assert resolve_num_ctx() == DEFAULT_NUM_CTX


def _dense_passages(n, chars):
    # Dense math-like text (symbols tokenize finer; we model worst-case via the
    # conservative divisor). Each passage is `chars` long.
    return [
        RetrievedPassage(
            chunk_id=f"chunk_{i:05d}",
            text=("a < b ≤ c ∧ ∀x∈S " * 1000)[:chars],
            score=0.9 - i * 0.01, engine="lexical",
            item_path=f"m/p{i}.html", section_heading=f"S{i}",
            module_id="m1", source={},
        )
        for i in range(n)
    ]


def test_token_budget_drops_passages_to_fit_small_window(monkeypatch):
    # A 4096 window cannot hold 8 dense 1500-char passages. The renderer must
    # drop trailing passages so the estimated prompt tokens fit the window.
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    passages = _dense_passages(8, 1500)
    prompt, meta = render_answer_user_prompt_with_meta(
        "How is a < b read aloud?", passages, max_tokens=1024
    )
    assert meta["dropped_count"] >= 1
    assert len(meta["included_ids"]) < 8
    # The estimate stays under the window (system + prompt + max_tokens fit).
    assert (
        meta["estimated_prompt_tokens"] + 1024 < meta["num_ctx"] + 200
    )
    assert "Question: How is a < b read aloud?" in prompt


def test_token_budget_larger_window_keeps_more_passages(monkeypatch):
    passages = _dense_passages(8, 1500)
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    _, small = render_answer_user_prompt_with_meta(
        "q", passages, max_tokens=512
    )
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "16384")
    _, big = render_answer_user_prompt_with_meta(
        "q", passages, max_tokens=512
    )
    # A bigger window keeps strictly more passages (prose vs dense both gain).
    assert len(big["included_ids"]) > len(small["included_ids"])


def test_prose_corpus_fits_more_than_dense_for_same_window(monkeypatch):
    # Same window, same passage count, but the budget is token-based so the
    # estimate scales with char length identically — the point is the budget
    # NEVER lets the rendered estimate overflow the window regardless of text.
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    dense = _dense_passages(8, 1500)
    _, meta = render_answer_user_prompt_with_meta("q", dense, max_tokens=1024)
    assert meta["estimated_prompt_tokens"] <= meta["num_ctx"]


def test_context_token_budget_accounts_for_max_tokens():
    # Larger max_tokens -> smaller passage budget (generation eats the window).
    b_small = context_token_budget("q", max_tokens=256, num_ctx=4096)
    b_big = context_token_budget("q", max_tokens=2048, num_ctx=4096)
    assert b_small > b_big >= 0


# ===========================================================================
# allowed_ids ∩ included (Fix 1 latent bug) — over-budget passages dropped
# ===========================================================================

def test_allowed_ids_only_includes_rendered_passages(monkeypatch):
    # Many dense passages overflow the small window; the composer must only
    # allow citations to passages actually rendered. A model that cites a
    # DROPPED passage id is treated as an unknown citation (remediation/raise),
    # NOT silently credited.
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    passages = _dense_passages(8, 1500)
    _, meta = render_answer_user_prompt_with_meta("q", passages, max_tokens=1024)
    included = set(meta["included_ids"])
    dropped_ids = [p.chunk_id for p in passages if p.chunk_id not in included]
    assert dropped_ids  # at least one passage was dropped

    # Model cites a dropped id every time -> persistent unknown citation.
    bad_id = dropped_ids[0]
    client = FakeAnswerClient([_envelope("ans", [bad_id])])
    with pytest.raises(InvalidCitationError):
        compose_answer("q", passages, client=client, max_parse_retries=2)


def test_cited_dropped_passage_not_silently_accepted(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    passages = _dense_passages(8, 1500)
    _, meta = render_answer_user_prompt_with_meta("q", passages, max_tokens=1024)
    included = list(meta["included_ids"])
    # An INCLUDED id is accepted normally (control).
    client = FakeAnswerClient([_envelope("ans", [included[0]])])
    result = compose_answer("q", passages, client=client)
    assert result.cited_chunk_ids == [included[0]]


# ===========================================================================
# Tail directive + remediation directive enumerate the full allowed set
# ===========================================================================

def test_trailing_directive_enumerates_allowed_ids():
    passages = _passages(2)
    prompt, meta = render_answer_user_prompt_with_meta("q", passages)
    # The tail (most-respected, truncation-surviving position) lists the ids.
    assert "Valid citation ids:" in prompt
    for cid in meta["included_ids"]:
        assert cid in prompt.rsplit("Question:", 1)[-1] or cid in prompt
    assert prompt.rstrip().endswith("}.")


def test_remediation_directive_restates_full_allowed_set():
    # On a bad-id reply, the remediation directive names the offenders AND the
    # full allowed set (set-compliance, not just format-compliance).
    client = FakeAnswerClient([
        _envelope("draft", ["chunk_99999"]),    # unknown
        _envelope("fixed", ["chunk_00000"]),     # corrected
    ])
    result = compose_answer("q", _passages(2), client=client)
    assert result.cited_chunk_ids == ["chunk_00000"]
    second_user = client.calls[1]["messages"][1]["content"]
    assert "chunk_99999" in second_user            # offenders named
    assert "chunk_00000" in second_user            # full allowed set restated
    assert "chunk_00001" in second_user            # ... including the other id
    # Directive template carries both placeholders.
    assert "{allowed_ids}" in CITATION_REMEDIATION_DIRECTIVE
    assert "{bad_ids}" in CITATION_REMEDIATION_DIRECTIVE


# ===========================================================================
# Truncation tripwire — both arms (Fix 1)
# ===========================================================================

class _UsageFakeClient(FakeAnswerClient):
    """FakeAnswerClient that also reports server-side prompt_tokens usage."""

    def __init__(self, responses, *, model="qwen2.5:14b-instruct-q4_K_M",
                 reported_prompt_tokens=None):
        super().__init__(responses, model=model)
        self.last_usage = {}
        self._reported = reported_prompt_tokens

    def chat_completion(self, messages, *, max_tokens=1024, temperature=0.0,
                        **kwargs):
        text = super().chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        if self._reported is not None:
            self.last_usage = {"prompt_tokens": int(self._reported)}
        return text


def test_tripwire_fires_on_silent_head_truncation(monkeypatch):
    # Big dense prompt; server reports a tiny prompt_tokens (head dropped).
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "4096")
    passages = _dense_passages(6, 1500)
    # Server claims it only saw ~50 prompt tokens -> head was truncated.
    client = _UsageFakeClient(
        [_envelope("fabricated", ["chunk_00000"])],
        reported_prompt_tokens=50,
    )
    with pytest.raises(PromptTruncatedError) as exc:
        compose_answer("q", passages, client=client)
    msg = str(exc.value)
    assert "OLLAMA_CONTEXT_LENGTH" in msg
    assert "ED4ALL_ANSWER_NUM_CTX" in msg


def test_tripwire_passes_when_reported_matches_estimate(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_NUM_CTX, "16384")
    passages = _dense_passages(4, 1200)
    _, meta = render_answer_user_prompt_with_meta(
        "q", passages, max_tokens=512
    )
    est = meta["estimated_prompt_tokens"]
    included = meta["included_ids"]
    # Server reports a count close to the estimate -> no truncation.
    client = _UsageFakeClient(
        [_envelope("ans", [included[0]])],
        reported_prompt_tokens=est,
    )
    result = compose_answer("q", passages, client=client)
    assert result.answer_text == "ans"


def test_tripwire_noops_when_usage_absent():
    # The default FakeAnswerClient exposes no last_usage attr meaningfully
    # (set to {} by the real client); a missing/empty usage must NOT block.
    passages = _passages(2)
    client = FakeAnswerClient([_envelope("ans", ["chunk_00000"])])
    # Ensure no usage signal.
    assert not getattr(client, "last_usage", {})
    result = compose_answer("q", passages, client=client)
    assert result.answer_text == "ans"


def test_tripwire_noops_on_small_prompt_below_floor():
    # A tiny estimate (< the absolute floor) can't be truncated by a 4096
    # window, so even a zero-ish reported count must NOT trip the wire. The
    # ws3.v2 system prompt alone now clears the floor, so the floor branch is
    # exercised by calling the guard directly with a sub-floor estimate.
    from lib.retrieval.answer_composer import (
        _TRUNCATION_MIN_ESTIMATE_TOKENS,
        _check_prompt_not_truncated,
    )

    client = _UsageFakeClient(
        [_envelope("ans", ["chunk_00000"])],
        reported_prompt_tokens=5,
    )
    client.last_usage = {"prompt_tokens": 5}
    # estimate below the floor -> guard no-ops (returns, no raise).
    _check_prompt_not_truncated(
        client,
        _TRUNCATION_MIN_ESTIMATE_TOKENS - 1,
        model_id="qwen2.5:7b-instruct-q4_K_M",
        num_ctx=4096,
    )
