"""Wave 102 - QualitativeJudge tests.

Asserts:

* ``provider="none"`` returns None (no scoring).
* ``provider="anthropic"`` raises a clear error when ANTHROPIC_API_KEY
  is unset and no client is injected.
* The Anthropic backend, given a mocked client, parses 1-5 scores.
* The local-NLI backend, given a mocked pipeline, maps ENTAIL prob to
  the 1-5 band.
* Provider env-var routing works (ED4ALL_LLM_JUDGE_PROVIDER).
* Score range is clamped to [1, 5].
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
# Anthropic client mock                                                   #
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


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_none_provider_returns_none():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    judge = QualitativeJudge(provider="none")
    assert judge.enabled is False
    assert judge.score("p", "o", "g") is None


def test_unknown_provider_raises():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    with pytest.raises(ValueError):
        QualitativeJudge(provider="not-a-thing")


def test_provider_env_var_routes(monkeypatch):
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    monkeypatch.setenv("ED4ALL_LLM_JUDGE_PROVIDER", "none")
    judge = QualitativeJudge()
    assert judge.provider == "none"


def test_anthropic_provider_missing_key_raises(monkeypatch):
    """No client + no env var -> clear runtime error."""
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # We don't want the lazy `import anthropic` to actually happen, so
    # inject a stub module that lets the constructor get past the
    # import line; the API-key check should still fire.
    import types
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: None  # noqa: E731
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    judge = QualitativeJudge(provider="anthropic")
    with pytest.raises(RuntimeError) as exc:
        judge.score("p", "o", "g")
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_anthropic_provider_parses_score():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    fake = _FakeAnthropicClient(replies=["4"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
        model="claude-sonnet-4-6",
    )
    score = judge.score("What is X?", "X is Y.", "X is Y.")
    assert score == 4.0
    # The system prompt must be cache-controlled (procurement-friendly
    # cost ceiling).
    assert fake.messages.calls
    sys_block = fake.messages.calls[0]["system"][0]
    assert sys_block["cache_control"]["type"] == "ephemeral"


def test_anthropic_provider_clamps_to_range():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    fake = _FakeAnthropicClient(replies=["The score is 3 stars"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
    )
    score = judge.score("p", "o", "g")
    assert 1.0 <= score <= 5.0


def test_anthropic_provider_default_when_no_digit():
    """Reply without a 1-5 digit -> defaults to mid-band 3.0."""
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    fake = _FakeAnthropicClient(replies=["I refuse"])
    judge = QualitativeJudge(
        provider="anthropic",
        anthropic_client=fake,
    )
    assert judge.score("p", "o", "g") == 3.0


def test_local_nli_high_entail_maps_to_5():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    def _pipe(_text):
        return [[
            {"label": "ENTAILMENT", "score": 0.97},
            {"label": "NEUTRAL", "score": 0.02},
            {"label": "CONTRADICTION", "score": 0.01},
        ]]

    judge = QualitativeJudge(
        provider="local_nli",
        nli_pipeline=_pipe,
    )
    assert judge.score("p", "model output", "ground truth") == 5.0


def test_local_nli_mid_entail_maps_to_band():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    def _pipe(_text):
        return [[
            {"label": "ENTAILMENT", "score": 0.65},
            {"label": "NEUTRAL", "score": 0.30},
            {"label": "CONTRADICTION", "score": 0.05},
        ]]

    judge = QualitativeJudge(
        provider="local_nli",
        nli_pipeline=_pipe,
    )
    # 0.65 falls in the 0.55-0.80 band -> 3
    assert judge.score("p", "o", "g") == 3.0


def test_local_nli_low_entail_floors_to_1():
    from Trainforge.eval.qualitative_judge import QualitativeJudge

    def _pipe(_text):
        return [[
            {"label": "ENTAILMENT", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.85},
        ]]

    judge = QualitativeJudge(
        provider="local_nli",
        nli_pipeline=_pipe,
    )
    assert judge.score("p", "o", "g") == 1.0
