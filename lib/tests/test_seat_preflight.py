"""Tests for the E7b seat-coherence preflight (vLLM restart-mode-collapse guard).

Deterministic, CI-safe: no model, no server, no network. The seat is driven by
a duck-typed fake ``chat_completion`` client; a ``SpyCapture`` records the
decision-capture emit so the LAW (every LLM call site fires a DecisionCapture)
is asserted.
"""
from __future__ import annotations

import pytest

from lib.retrieval.seat_preflight import (
    DECISION_TYPE_SEAT_PREFLIGHT,
    PROBE_USER_PROMPT,
    SeatCoherenceError,
    SeatCoherenceResult,
    preflight_or_raise,
    probe_resolved_backend,
    probe_seat_coherence,
    seat_label_for_backend,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeSeatClient:
    """Duck-typed ``chat_completion`` seat. Returns ``response`` or raises it."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=64, temperature=0.0, **kw):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens,
             "temperature": temperature}
        )
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class SpyCapture:
    """Records ``log_decision`` kwargs without writing to disk."""

    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.events.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale, **kwargs}
        )


# --------------------------------------------------------------------------- #
# Coherent path
# --------------------------------------------------------------------------- #


def test_coherent_response_passes():
    client = FakeSeatClient("The answer is 4.")
    result = probe_seat_coherence(client, seat_label="local:m@loop")
    assert isinstance(result, SeatCoherenceResult)
    assert result.coherent is True
    assert result.reason == "coherent"
    assert result.seat == "local:m@loop"


def test_probe_uses_the_arithmetic_content_prompt():
    client = FakeSeatClient("4")
    probe_seat_coherence(client, seat_label="s")
    user_msg = client.calls[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == PROBE_USER_PROMPT


def test_expected_token_word_form_accepted():
    client = FakeSeatClient("Two plus two is four.")
    assert probe_seat_coherence(client, seat_label="s").coherent is True


# --------------------------------------------------------------------------- #
# Incoherent shapes
# --------------------------------------------------------------------------- #


def test_empty_response_is_incoherent():
    result = probe_seat_coherence(FakeSeatClient("   "), seat_label="s")
    assert result.coherent is False
    assert result.reason == "empty_response"


def test_none_response_is_incoherent():
    result = probe_seat_coherence(FakeSeatClient(None), seat_label="s")
    assert result.coherent is False
    assert result.reason == "empty_response"


def test_word_repetition_soup_is_degenerate():
    result = probe_seat_coherence(
        FakeSeatClient("the the the the the the the the"), seat_label="s"
    )
    assert result.coherent is False
    assert result.reason == "degenerate_repetition"


def test_single_glyph_soup_is_degenerate():
    result = probe_seat_coherence(
        FakeSeatClient("aaaaaaaaaaaaaaaa"), seat_label="s"
    )
    assert result.coherent is False
    assert result.reason == "degenerate_repetition"


def test_non_degenerate_but_wrong_answer_is_incoherent():
    # Real words, not repetitive, but the answer token is absent → collapse-ish.
    result = probe_seat_coherence(FakeSeatClient("banana"), seat_label="s")
    assert result.coherent is False
    assert result.reason == "expected_token_absent"


def test_transport_failure_folds_into_incoherent_verdict():
    client = FakeSeatClient(ConnectionError("seat down"))
    result = probe_seat_coherence(client, seat_label="dead-seat")
    assert result.coherent is False
    assert result.reason.startswith("probe_call_failed:")
    assert "ConnectionError" in result.reason


def test_expected_tokens_disabled_skips_that_leg():
    # With no expected tokens, a coherent non-empty non-degenerate reply passes.
    result = probe_seat_coherence(
        FakeSeatClient("Some prose about geometry."),
        seat_label="s",
        expected_tokens=(),
    )
    assert result.coherent is True


# --------------------------------------------------------------------------- #
# DecisionCapture (the LAW)
# --------------------------------------------------------------------------- #


def test_capture_fires_on_coherent_verdict():
    cap = SpyCapture()
    probe_seat_coherence(FakeSeatClient("4"), seat_label="local:m@loop", capture=cap)
    assert len(cap.events) == 1
    ev = cap.events[0]
    assert ev["decision_type"] == DECISION_TYPE_SEAT_PREFLIGHT
    assert ev["decision"] == "seat_preflight:coherent"
    assert len(ev["rationale"]) >= 20
    # Dynamic, replayable signals in the rationale.
    assert "local:m@loop" in ev["rationale"]
    assert "distinct_word_ratio" in ev["rationale"]


def test_capture_fires_on_incoherent_verdict():
    cap = SpyCapture()
    probe_seat_coherence(
        FakeSeatClient("the the the the the the the"), seat_label="bad", capture=cap
    )
    assert len(cap.events) == 1
    assert cap.events[0]["decision"] == "seat_preflight:incoherent"
    assert "degenerate" in cap.events[0]["rationale"]


def test_capture_fires_even_on_transport_failure():
    cap = SpyCapture()
    probe_seat_coherence(
        FakeSeatClient(RuntimeError("boom")), seat_label="dead", capture=cap
    )
    assert len(cap.events) == 1
    assert cap.events[0]["decision"] == "seat_preflight:incoherent"


# --------------------------------------------------------------------------- #
# preflight_or_raise gate
# --------------------------------------------------------------------------- #


def test_preflight_or_raise_returns_result_when_coherent():
    result = preflight_or_raise(FakeSeatClient("4"), seat_label="s")
    assert result.coherent is True


def test_preflight_or_raise_raises_and_names_the_seat():
    with pytest.raises(SeatCoherenceError) as exc:
        preflight_or_raise(
            FakeSeatClient(""), seat_label="super:nemotron@127.0.0.1:8001"
        )
    msg = str(exc.value)
    assert "super:nemotron@127.0.0.1:8001" in msg
    assert "empty_response" in msg


def test_preflight_or_raise_still_captures_before_raising():
    cap = SpyCapture()
    with pytest.raises(SeatCoherenceError):
        preflight_or_raise(FakeSeatClient(""), seat_label="s", capture=cap)
    assert len(cap.events) == 1
    assert cap.events[0]["decision"] == "seat_preflight:incoherent"


# --------------------------------------------------------------------------- #
# probe_resolved_backend + seat_label_for_backend
# --------------------------------------------------------------------------- #


def test_seat_label_for_backend_shape():
    from lib.retrieval.answer_backend import ResolvedAnswerBackend

    resolved = ResolvedAnswerBackend(
        provider_name="local", model_id="qwen", base_url="http://localhost:8001/v1",
        api_key=None, timeout=120.0,
    )
    assert seat_label_for_backend(resolved) == "local:qwen@http://localhost:8001/v1"


def test_probe_resolved_backend_builds_client_and_gates(monkeypatch):
    from lib.retrieval.answer_backend import ResolvedAnswerBackend
    import lib.retrieval.answer_backend as ab

    resolved = ResolvedAnswerBackend(
        provider_name="local", model_id="qwen", base_url="http://localhost:8001/v1",
        api_key=None, timeout=120.0,
    )
    fake = FakeSeatClient("4")
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: fake)
    result = probe_resolved_backend(resolved)
    assert result.coherent is True
    assert fake.calls  # the built client was actually probed


def test_probe_resolved_backend_raises_on_incoherent(monkeypatch):
    from lib.retrieval.answer_backend import ResolvedAnswerBackend
    import lib.retrieval.answer_backend as ab

    resolved = ResolvedAnswerBackend(
        provider_name="local", model_id="qwen", base_url="http://localhost:8001/v1",
        api_key=None, timeout=120.0,
    )
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: FakeSeatClient(""))
    with pytest.raises(SeatCoherenceError):
        probe_resolved_backend(resolved)


def test_probe_resolved_backend_no_raise_mode(monkeypatch):
    from lib.retrieval.answer_backend import ResolvedAnswerBackend
    import lib.retrieval.answer_backend as ab

    resolved = ResolvedAnswerBackend(
        provider_name="local", model_id="qwen", base_url="http://localhost:8001/v1",
        api_key=None, timeout=120.0,
    )
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: FakeSeatClient(""))
    result = probe_resolved_backend(resolved, raise_on_fail=False)
    assert result.coherent is False
