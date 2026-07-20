"""E7a diagnostic-composer arm + E7b preflight wiring in answer_course_question.

Deterministic, CI-safe: reuses the committed mini-course LibV2 fixture (real
lexical BM25 retrieval) and drives every LLM seat with the shared
``FakeAnswerClient``. Proves:

* default (no eval backend, no env) → production composition, NO preflight;
* an explicit ``eval_composer_backend`` PREFLIGHTS the seat then routes ONLY
  composition through it (retrieval + gate unchanged);
* a seat that fails the preflight raises ``SeatCoherenceError`` (loud), never
  composing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import lib.retrieval.answer_backend as ab
from lib.retrieval.answer_backend import ResolvedAnswerBackend
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    answer_course_question,
)
from lib.retrieval.seat_preflight import (
    DECISION_TYPE_SEAT_PREFLIGHT,
    SeatCoherenceError,
)
from lib.retrieval.refusal import RefusalPolicy

# Shared test doubles + fixture identity from the pipeline suite.
from lib.tests.test_answer_composer import FakeAnswerClient, SpyCapture, _envelope

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)
COURSE_SLUG = "mini-retrieval-101"

_PERMISSIVE_LEXICAL = RefusalPolicy(
    engine="lexical",
    min_top_score=0.0,
    score_floor=0.0,
    min_passages_above_floor=1,
    policy_version="test-permissive-lexical",
)

_STRONG_SEAT = ResolvedAnswerBackend(
    provider_name="local",
    model_id="super-120b",
    base_url="http://localhost:8001/v1",
    api_key=None,
    timeout=120.0,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        ab.ENV_EVAL_COMPOSER_PROVIDER, ab.ENV_EVAL_COMPOSER_MODEL,
        ab.ENV_ANSWER_PROVIDER, ab.ENV_ANSWER_MODEL,
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture()
def mini_libv2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {FIXTURE_ROOT}")
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(FIXTURE_ROOT, libv2_root / "courses" / COURSE_SLUG)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return libv2_root


_QUERY = "What does a vector store index?"
_ENVELOPE = _envelope(
    "A vector store indexes embedding vectors.", ["mini_alpha_chunk_001"]
)


# --------------------------------------------------------------------------- #
# Default (arm OFF) → byte-identical production composition, no preflight
# --------------------------------------------------------------------------- #


def test_default_no_eval_backend_uses_production_client(mini_libv2: Path):
    prod = FakeAnswerClient([_ENVELOPE])
    cap = SpyCapture()
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=prod, refusal_policy=_PERMISSIVE_LEXICAL, capture=cap,
    )
    assert result.status == STATUS_ANSWERED
    # The production client composed once; no preflight probe ran.
    assert len(prod.calls) == 1
    assert not any(
        e["decision_type"] == DECISION_TYPE_SEAT_PREFLIGHT for e in cap.events
    )


# --------------------------------------------------------------------------- #
# Arm ON → preflight the stronger seat, compose on it, retrieval unchanged
# --------------------------------------------------------------------------- #


def test_eval_backend_preflights_then_composes_on_strong_seat(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    # call 0 = preflight probe ("4"), call 1 = compose (envelope).
    strong = FakeAnswerClient(["4", _ENVELOPE])
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: strong)
    cap = SpyCapture()

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        refusal_policy=_PERMISSIVE_LEXICAL, capture=cap,
        eval_composer_backend=_STRONG_SEAT,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "A vector store indexes embedding vectors."
    # Two calls on the strong seat: preflight probe + composition.
    assert len(strong.calls) == 2
    # The preflight fired a decision capture naming the seat.
    preflights = [
        e for e in cap.events
        if e["decision_type"] == DECISION_TYPE_SEAT_PREFLIGHT
    ]
    assert len(preflights) == 1
    assert "super-120b" in preflights[0]["rationale"]


def test_eval_backend_from_env_activates_the_arm(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ab.ENV_EVAL_COMPOSER_MODEL, "super-120b")
    strong = FakeAnswerClient(["4", _ENVELOPE])
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: strong)

    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY, refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert result.status == STATUS_ANSWERED
    assert len(strong.calls) == 2  # preflight + compose on the env-selected seat


# --------------------------------------------------------------------------- #
# Preflight failure → loud refusal to measure (no composition)
# --------------------------------------------------------------------------- #


def test_collapsed_seat_raises_before_composing(
    mini_libv2: Path, monkeypatch: pytest.MonkeyPatch
):
    # Mode-collapsed seat: empty content on the probe → SeatCoherenceError, and
    # the compose call (index 1) is NEVER reached.
    collapsed = FakeAnswerClient(["", _ENVELOPE])
    monkeypatch.setattr(ab, "build_answer_client", lambda **kw: collapsed)

    with pytest.raises(SeatCoherenceError) as exc:
        answer_course_question(
            mini_libv2, COURSE_SLUG, _QUERY,
            refusal_policy=_PERMISSIVE_LEXICAL,
            eval_composer_backend=_STRONG_SEAT,
        )
    assert "super-120b" in str(exc.value)
    # Only the probe ran; composition was refused.
    assert len(collapsed.calls) == 1
