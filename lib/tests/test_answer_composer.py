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
    assert ANSWER_PROMPT_VERSION == "ws3.v1"


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
    assert result.prompt_version == "ws3.v1"
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
