"""Unit tests for the L2 assessment-aware answering guard.

Covers flag/threshold resolution, stem loading (QTI + specs, cache-by-mtime),
lexical matching + threshold boundary, the decision-capture emit, and the
redirect-envelope shape. Import-light: injects stems directly / redirects the
LibV2 root via ``ED4ALL_LIBV2_ROOT`` — no model, no network, no real course.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval import assessment_guard as ag


# --------------------------------------------------------------------------- #
# Flag + threshold resolution
# --------------------------------------------------------------------------- #


def test_mode_default_off(monkeypatch):
    monkeypatch.delenv(ag.ENV_ASSESSMENT_GUARD, raising=False)
    assert ag.resolve_guard_mode() == ag.GUARD_OFF


@pytest.mark.parametrize("raw,expected", [
    ("off", ag.GUARD_OFF),
    ("shadow", ag.GUARD_SHADOW),
    ("on", ag.GUARD_ON),
    ("ON", ag.GUARD_ON),
    (" Shadow ", ag.GUARD_SHADOW),
    ("garbage", ag.GUARD_OFF),  # typo never silently enables enforcement
])
def test_mode_resolution(monkeypatch, raw, expected):
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD, raw)
    assert ag.resolve_guard_mode() == expected


def test_threshold_default(monkeypatch):
    monkeypatch.delenv(ag.ENV_ASSESSMENT_GUARD_THRESHOLD, raising=False)
    assert ag.resolve_guard_threshold() == ag.DEFAULT_THRESHOLD


@pytest.mark.parametrize("raw,expected", [
    ("0.9", 0.9),
    ("0", 0.0),
    ("1", 1.0),
    ("", ag.DEFAULT_THRESHOLD),
    ("nope", ag.DEFAULT_THRESHOLD),
    ("1.5", ag.DEFAULT_THRESHOLD),  # out of band → default
    ("-0.2", ag.DEFAULT_THRESHOLD),
])
def test_threshold_resolution(monkeypatch, raw, expected):
    monkeypatch.setenv(ag.ENV_ASSESSMENT_GUARD_THRESHOLD, raw)
    assert ag.resolve_guard_threshold() == expected


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _stems(*pairs):
    return [ag.AssessmentStem(quiz_id="quiz-1", question_id=qid, stem=stem)
            for qid, stem in pairs]


def test_match_verbatim_paste():
    stems = _stems(
        ("q1", "Calculate the derivative of the polynomial function f of x"),
        ("q2", "Describe the water cycle and its major stages"),
    )
    q = "Calculate the derivative of the polynomial function f of x"
    res = ag.match_question(q, stems, threshold=0.75)
    assert res.matched is True
    assert res.quiz_id == "quiz-1"
    assert res.question_id == "q1"
    assert res.method == "lexical"
    assert res.score >= 0.75


def test_no_match_unrelated_query():
    stems = _stems(("q1", "Calculate the derivative of the polynomial function"))
    res = ag.match_question(
        "What time does the library open on weekends", stems, threshold=0.75
    )
    assert res.matched is False


def test_threshold_boundary():
    # 4 shared content tokens out of a 4-token stem => containment 1.0 for stem,
    # but min() is over the smaller set. Build a controlled overlap.
    stems = _stems(("q1", "alpha beta gamma delta"))
    # query shares 3 of 4 stem tokens => containment 3/4 = 0.75 exactly.
    q = "alpha beta gamma epsilon"
    at = ag.match_question(q, stems, threshold=0.75)
    assert at.score == pytest.approx(0.75)
    assert at.matched is True  # >= boundary
    above = ag.match_question(q, stems, threshold=0.7501)
    assert above.matched is False  # just above the score


def test_match_empty_inputs():
    assert ag.match_question("", _stems(("q1", "anything here")), 0.75).matched is False
    assert ag.match_question("a real query here", [], 0.75).matched is False


def test_cosine_arm_used_when_client_present():
    stems = _stems(("q1", "totally different lexical wording entirely"))

    class _FakeClient:
        def encode_batch(self, texts):
            import numpy as np
            # query row identical to the single stem row => cosine 1.0.
            v = np.array([1.0, 0.0], dtype=np.float32)
            return np.stack([v] * len(texts))

    res = ag.match_question(
        "unrelated question", stems, threshold=0.75, embedding_client=_FakeClient()
    )
    assert res.matched is True
    assert res.method == "cosine"
    assert res.score == pytest.approx(1.0)


def test_cosine_backend_failure_degrades_to_lexical():
    stems = _stems(("q1", "alpha beta gamma delta"))

    class _BrokenClient:
        def encode_batch(self, texts):
            raise RuntimeError("backend down")

    # cosine unavailable → lexical still scores the verbatim overlap.
    res = ag.match_question(
        "alpha beta gamma delta", stems, threshold=0.75, embedding_client=_BrokenClient()
    )
    assert res.matched is True
    assert res.method == "lexical"


# --------------------------------------------------------------------------- #
# Stem loading (specs + cache) with a tmp LibV2 root
# --------------------------------------------------------------------------- #


def _course_dir(tmp_path: Path, slug: str) -> Path:
    d = tmp_path / "LibV2" / "courses" / slug
    d.mkdir(parents=True)
    return d


def test_load_stems_from_specs(tmp_path, monkeypatch):
    ag._STEM_CACHE.clear()
    slug = "specs-course"
    cdir = _course_dir(tmp_path, slug)
    specs = cdir / "training_specs"
    specs.mkdir()
    (specs / "assessments.json").write_text(json.dumps({
        "assessments": [{
            "assessment_id": "a1",
            "questions": [
                {"question_id": "q1", "stem": "What is a covalent bond?", "answer": "SECRET"},
                {"id": "q2", "prompt": "Define electronegativity"},
            ],
        }],
    }), encoding="utf-8")
    libv2 = tmp_path / "LibV2"
    stems = ag.load_assessment_stems(libv2, slug)
    texts = {s.stem for s in stems}
    assert "What is a covalent bond?" in texts
    assert "Define electronegativity" in texts
    # Answer text is never harvested.
    assert not any("SECRET" in s.stem for s in stems)


def test_load_stems_missing_course_graceful(tmp_path):
    ag._STEM_CACHE.clear()
    libv2 = tmp_path / "LibV2"
    (libv2 / "courses").mkdir(parents=True)
    assert ag.load_assessment_stems(libv2, "no-such-course") == []


def test_stem_cache_invalidates_on_mtime(tmp_path):
    ag._STEM_CACHE.clear()
    slug = "cache-course"
    cdir = _course_dir(tmp_path, slug)
    specs = cdir / "training_specs"
    specs.mkdir()
    f = specs / "assessments.json"
    f.write_text(json.dumps([{"stem": "first question here"}]), encoding="utf-8")
    libv2 = tmp_path / "LibV2"
    first = ag.load_assessment_stems(libv2, slug)
    assert {s.stem for s in first} == {"first question here"}

    # Rewrite with a bumped mtime → cache must refresh.
    import os
    f.write_text(json.dumps([{"stem": "second question added"}]), encoding="utf-8")
    os.utime(f, ns=(f.stat().st_atime_ns + 10_000, f.stat().st_mtime_ns + 10_000_000))
    second = ag.load_assessment_stems(libv2, slug)
    assert "second question added" in {s.stem for s in second}


# --------------------------------------------------------------------------- #
# Decision-capture emit
# --------------------------------------------------------------------------- #


class _SpyCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def test_capture_fires_on_match(tmp_path, monkeypatch):
    ag._STEM_CACHE.clear()
    slug = "cap-course"
    cdir = _course_dir(tmp_path, slug)
    specs = cdir / "training_specs"
    specs.mkdir()
    (specs / "assessments.json").write_text(
        json.dumps([{"question_id": "q9", "stem": "alpha beta gamma delta epsilon"}]),
        encoding="utf-8",
    )
    libv2 = tmp_path / "LibV2"
    spy = _SpyCapture()
    outcome = ag.evaluate_guard(
        libv2, slug, "alpha beta gamma delta epsilon", ag.GUARD_ON,
        capture=spy, use_embeddings=False,
    )
    assert outcome.matched is True
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["decision_type"] == ag.DECISION_TYPE_GUARD
    assert call["decision"] == "redirect_to_study_sections"
    # Dynamic rationale interpolates real signals.
    assert "qhash=" in call["rationale"]
    assert "score=" in call["rationale"]
    assert "mode=on" in call["rationale"]
    assert len(call["rationale"]) >= 20


def test_capture_fires_on_no_match_and_shadow(tmp_path):
    ag._STEM_CACHE.clear()
    slug = "cap-course-2"
    cdir = _course_dir(tmp_path, slug)
    specs = cdir / "training_specs"
    specs.mkdir()
    (specs / "assessments.json").write_text(
        json.dumps([{"stem": "alpha beta gamma delta"}]), encoding="utf-8"
    )
    libv2 = tmp_path / "LibV2"

    # No match: still emits (pass_through).
    spy = _SpyCapture()
    out = ag.evaluate_guard(
        libv2, slug, "library opening hours weekend", ag.GUARD_ON,
        capture=spy, use_embeddings=False,
    )
    assert out.matched is False
    assert spy.calls[0]["decision"] == "pass_through"

    # Shadow + match: shadow_would_redirect.
    spy2 = _SpyCapture()
    out2 = ag.evaluate_guard(
        libv2, slug, "alpha beta gamma delta", ag.GUARD_SHADOW,
        capture=spy2, use_embeddings=False,
    )
    assert out2.matched is True
    assert spy2.calls[0]["decision"] == "shadow_would_redirect"


def test_capture_none_never_raises(tmp_path):
    ag._STEM_CACHE.clear()
    libv2 = tmp_path / "LibV2"
    (libv2 / "courses").mkdir(parents=True)
    # capture=None must be a clean no-op.
    out = ag.evaluate_guard(libv2, "x", "a query", ag.GUARD_ON, capture=None,
                            use_embeddings=False)
    assert out.evaluated is True


# --------------------------------------------------------------------------- #
# Redirect envelope shape
# --------------------------------------------------------------------------- #


def test_redirect_envelope_shape(tmp_path, monkeypatch):
    ag._STEM_CACHE.clear()
    slug = "redir-course"
    _course_dir(tmp_path, slug)
    libv2 = tmp_path / "LibV2"
    # Stub retrieval so we don't need a real chunkset.
    monkeypatch.setattr(ag, "_retrieve_passages", lambda *a, **k: [])
    outcome = ag.GuardOutcome(
        mode=ag.GUARD_ON, evaluated=True, matched=True, score=0.91,
        threshold=0.75, quiz_id="quiz-1", question_id="q3", method="lexical",
    )
    env = ag.build_redirect_envelope(libv2, slug, "the question", "lexical", outcome)
    # GroundedAnswer-shaped keys the renderer/drawer consume.
    for key in ("status", "query", "course_slug", "engine", "answer_text",
                "citations", "refusal", "confidence", "warnings",
                "generated_at", "latency_ms"):
        assert key in env
    assert env["status"] == "answered"  # renders answer_text + Sources
    assert env["refusal"] is None       # NEVER a hard refusal
    assert "won't answer it directly" in env["answer_text"]
    assert env["assessment_guard"]["redirected"] is True
    assert env["assessment_guard"]["quiz_id"] == "quiz-1"
    assert any("redirected" in w for w in env["warnings"])


def test_redirect_citations_from_stubbed_passages(tmp_path, monkeypatch):
    ag._STEM_CACHE.clear()
    slug = "redir-course-2"
    _course_dir(tmp_path, slug)
    libv2 = tmp_path / "LibV2"

    class _P:
        chunk_id = "c1"
        text = "The covalent bond forms when atoms share electron pairs. " * 8
        item_path = "week_01/content_01.html"
        section_heading = "Covalent Bonds"
        module_id = "content_01"
        source = {}

    monkeypatch.setattr(ag, "_retrieve_passages", lambda *a, **k: [_P()])
    outcome = ag.GuardOutcome(
        mode=ag.GUARD_ON, evaluated=True, matched=True, score=0.9,
        threshold=0.75, quiz_id="q", question_id="1", method="lexical",
    )
    env = ag.build_redirect_envelope(libv2, slug, "covalent bond", "lexical", outcome)
    cits = env["citations"]
    assert len(cits) == 1
    c = cits[0]
    assert c["item_path"] == "week_01/content_01.html"
    assert c["link_target"]["fragment"] == {"kind": "heading", "value": "covalent-bonds"}
    assert c["text_quote"] and c["text_quote"].endswith("…")
