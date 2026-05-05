"""H2 regression test — ``QualitativeJudge`` decision-capture wiring.

The Anthropic-backed scoring path in
``Trainforge/eval/qualitative_judge.py::_score_anthropic`` previously
issued ``client.messages.create(...)`` with no decision-capture wiring,
which let a degenerate stub-response run (e.g. local model server quirk
returning ``"1"`` ten times in a row) influence ``EvalGatingValidator``
promotion decisions without any audit trail.

These tests assert:

- One ``decision_type="llm_chat_call"`` event fires per scoring call
  when ``capture`` is wired.
- Field names are LLM-agnostic — provider lands as a VALUE in
  ``ml_features["provider"]``; no ``claude_*`` / ``anthropic_*`` field
  shapes leak.
- Rationale is dynamic (>=20 chars, references the model name and
  parsed score).
- ``capture=None`` is a silent no-op (no events, no exceptions).
- A capture-side exception during ``log_decision`` does not crash the
  scoring call (defensive try/except).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------- #
# Mocks                                                                   #
# ---------------------------------------------------------------------- #


class _FakeMsgBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeMsgBlock(text)]


class _FakeMessages:
    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self.replies.pop(0))


class _FakeAnthropicClient:
    def __init__(self, replies: List[str]):
        self.messages = _FakeMessages(replies)


class _RecordingCapture:
    """Minimal ``DecisionCapture`` substitute — records kwargs only."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


class _ExplodingCapture:
    """``log_decision`` raises — used to verify defensive try/except."""

    def __init__(self):
        self.calls = 0

    def log_decision(self, **kwargs):
        self.calls += 1
        raise RuntimeError("capture backend offline")


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_anthropic_score_emits_llm_chat_call_when_capture_wired():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    capture = _RecordingCapture()
    fake = _FakeAnthropicClient(replies=["4"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
        capture=capture,
    )
    score = judge.score(
        "What is X?", "X is Y.", "X is Y.", probe_id="probe_42",
    )

    assert score == 4.0
    # Exactly one event emitted for one scoring call.
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "llm_chat_call"
    # Rationale is dynamic + non-trivial.
    rationale = event["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    assert "claude-sonnet-4-6" in rationale
    assert "score=4.0" in rationale
    # Decision string also references model + probe.
    assert "probe_42" in event["decision"]
    assert "claude-sonnet-4-6" in event["decision"]
    # ml_features carries provider as a VALUE — no provider-named field.
    ml = event["ml_features"]
    assert ml["provider"] == "anthropic"
    assert ml["model"] == "claude-sonnet-4-6"
    assert ml["probe_id"] == "probe_42"
    assert ml["max_tokens"] == 8
    assert ml["score"] == 4.0
    assert isinstance(ml["latency_ms"], int)
    assert ml["latency_ms"] >= 0
    # No provider-named fields leaked into the top-level event.
    forbidden = {k for k in event.keys() if "claude" in k.lower() or "anthropic" in k.lower()}
    assert forbidden == set(), f"LLM-agnostic contract violated: {forbidden}"


def test_anthropic_score_no_capture_emits_nothing_and_does_not_crash():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    fake = _FakeAnthropicClient(replies=["3"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
        capture=None,
    )
    # No capture instance to inspect; success is "did not raise".
    score = judge.score("p", "o", "g")
    assert score == 3.0
    # And exactly one SDK call still happened.
    assert len(fake.messages.calls) == 1


def test_anthropic_score_capture_exception_does_not_crash_scoring():
    """A capture-side failure must NOT propagate — observability is
    non-load-bearing for scoring correctness."""
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    capture = _ExplodingCapture()
    fake = _FakeAnthropicClient(replies=["5"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
        capture=capture,
    )
    # Should not raise even though log_decision raises.
    score = judge.score("p", "o", "g")
    assert score == 5.0
    assert capture.calls == 1


def test_per_call_probe_id_overrides_instance_probe_id():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    capture = _RecordingCapture()
    fake = _FakeAnthropicClient(replies=["2"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
        capture=capture,
        probe_id="instance_probe",
    )
    judge.score("p", "o", "g", probe_id="per_call_probe")
    assert capture.events[0]["ml_features"]["probe_id"] == "per_call_probe"


def test_instance_probe_id_used_when_per_call_probe_omitted():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    capture = _RecordingCapture()
    fake = _FakeAnthropicClient(replies=["2"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
        capture=capture,
        probe_id="instance_probe",
    )
    judge.score("p", "o", "g")
    assert capture.events[0]["ml_features"]["probe_id"] == "instance_probe"


def test_local_nli_path_does_not_emit_llm_chat_call():
    """The local-NLI scorer is not an LLM chat call — it must NOT
    emit ``llm_chat_call`` even when capture is wired."""
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    def _pipe(_text):
        return [[
            {"label": "ENTAILMENT", "score": 0.97},
        ]]

    capture = _RecordingCapture()
    judge = QualitativeJudge(
        provider="local_nli",
        nli_pipeline=_pipe,
        capture=capture,
    )
    score = judge.score("p", "model output", "ground truth")
    assert score == 5.0
    # No llm_chat_call event — local NLI is not a chat-style LLM call.
    chat_events = [
        e for e in capture.events
        if e.get("decision_type") == "llm_chat_call"
    ]
    assert chat_events == []
