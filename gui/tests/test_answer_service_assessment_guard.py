"""Integration of the L2 assessment guard into ``answer_service.ask``.

Verifies the three-valued rollout at the ask seam:
* OFF (unset): byte-identical to the prior path — no guard evaluation, no
  ``assessment_guard`` key, ``answer_library_question`` called unchanged.
* shadow: answers normally but stamps the would-have-matched signal.
* on + match: returns the redirect envelope WITHOUT dispatching the compose.
* missing assessments: graceful no-op (answers normally).

The pipeline call + guard evaluation are stubbed so no model / network / real
course is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gui.services import answer_service
from lib.retrieval import assessment_guard as ag


class _FakeResult:
    def __init__(self, slug, query, engine):
        self._d = {
            "status": "answered",
            "query": query,
            "course_slug": slug,
            "engine": engine,
            "answer_text": "A composed answer.",
            "citations": [],
            "refusal": None,
            "confidence": {},
            "groundedness": None,
            "warnings": [],
            "model_id": "m",
            "prompt_version": "v",
            "generated_at": "now",
            "latency_ms": 1.0,
        }

    def to_dict(self):
        return dict(self._d)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub the library-wide answer seam; record calls."""
    calls = {"n": 0, "last": None}

    def _fake_answer_library_question(libv2_root, slug, query, **kwargs):
        calls["n"] += 1
        calls["last"] = kwargs
        return _FakeResult(slug, query, kwargs.get("engine"))

    import lib.retrieval.library_wide as lw
    monkeypatch.setattr(lw, "answer_library_question", _fake_answer_library_question)
    return calls


def _libv2(tmp_path: Path, slug: str) -> Path:
    root = tmp_path / "LibV2"
    (root / "courses" / slug).mkdir(parents=True)
    return root


def test_off_mode_byte_identical(tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.delenv(ag.ENV_ASSESSMENT_GUARD, raising=False)
    slug = "c-off"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(_libv2(tmp_path, slug)))

    # A landmine: if the guard were evaluated in OFF mode this would raise.
    def _boom(*a, **k):
        raise AssertionError("guard evaluated in OFF mode")
    monkeypatch.setattr(ag, "evaluate_guard", _boom)

    out = answer_service.ask(slug, "any question", engine="lexical")
    assert stub_pipeline["n"] == 1
    assert "assessment_guard" not in out  # additive key absent when OFF
    assert out["answer_text"] == "A composed answer."


def test_shadow_answers_normally_with_signal(tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD, "shadow")
    slug = "c-shadow"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(_libv2(tmp_path, slug)))

    outcome = ag.GuardOutcome(
        mode=ag.GUARD_SHADOW, evaluated=True, matched=True, score=0.88,
        threshold=0.75, quiz_id="quiz-1", question_id="q2", method="lexical",
    )
    monkeypatch.setattr(ag, "evaluate_guard", lambda *a, **k: outcome)

    out = answer_service.ask(slug, "a matching question", engine="lexical")
    # Compose still ran (normal answer), but the signal is stamped.
    assert stub_pipeline["n"] == 1
    assert out["answer_text"] == "A composed answer."
    assert out["assessment_guard"]["matched"] is True
    assert out["assessment_guard"]["redirected"] is False


def test_on_match_redirects_without_dispatch(tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD, "on")
    slug = "c-on"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(_libv2(tmp_path, slug)))

    outcome = ag.GuardOutcome(
        mode=ag.GUARD_ON, evaluated=True, matched=True, score=0.95,
        threshold=0.75, quiz_id="quiz-1", question_id="q1", method="lexical",
    )
    monkeypatch.setattr(ag, "evaluate_guard", lambda *a, **k: outcome)
    # No real chunkset — stub concept retrieval.
    monkeypatch.setattr(ag, "_retrieve_passages", lambda *a, **k: [])

    out = answer_service.ask(slug, "a homework question", engine="lexical")
    # The compose was NEVER dispatched.
    assert stub_pipeline["n"] == 0
    assert out["status"] == "answered"
    assert out["refusal"] is None
    assert "won't answer it directly" in out["answer_text"]
    assert out["assessment_guard"]["redirected"] is True


def test_on_no_match_answers_normally(tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD, "on")
    slug = "c-on-nomatch"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(_libv2(tmp_path, slug)))

    outcome = ag.GuardOutcome(
        mode=ag.GUARD_ON, evaluated=True, matched=False, score=0.1,
        threshold=0.75, quiz_id="", question_id="", method="lexical",
    )
    monkeypatch.setattr(ag, "evaluate_guard", lambda *a, **k: outcome)

    out = answer_service.ask(slug, "an unrelated question", engine="lexical")
    assert stub_pipeline["n"] == 1  # composed normally
    assert out["answer_text"] == "A composed answer."
    assert out["assessment_guard"]["redirected"] is False


def test_missing_assessments_graceful_noop(tmp_path, monkeypatch, stub_pipeline):
    # 'on' mode but the real evaluate_guard runs over a course with NO
    # assessment sources → no match → normal answer.
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD, "on")
    ag._STEM_CACHE.clear()
    slug = "c-empty"
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(_libv2(tmp_path, slug)))

    out = answer_service.ask(slug, "any question at all", engine="lexical")
    assert stub_pipeline["n"] == 1
    assert out["answer_text"] == "A composed answer."
    assert out["assessment_guard"]["matched"] is False
