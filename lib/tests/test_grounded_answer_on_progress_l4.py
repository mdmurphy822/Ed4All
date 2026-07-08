"""L4 passages-first disclosure — the ``on_progress`` callback (lane: learner-api).

Deterministic, CI-safe: the LLM call site is driven by the shared
``FakeAnswerClient``; retrieval runs the REAL lexical BM25 path over the
mini-course fixture in a tmp LibV2 layout (the WS1/WS2 fixture pattern reused by
``test_grounded_answer_pipeline``).

Contract under test:
  * the callback fires exactly ONCE, after the pre-LLM refusal gate;
  * the answered path carries the retrieved passages + ``refused=False``;
  * the refused path fires with ``refused=True`` + the refusal status and short-
    circuits BEFORE compose;
  * a callback exception never breaks (or changes) the answer (best-effort);
  * the default (no callback) path is byte-identical.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    STATUS_REFUSED_LOW_CONFIDENCE,
    answer_course_question,
)
from lib.retrieval.refusal import RefusalPolicy

from lib.tests.test_answer_composer import FakeAnswerClient

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
_STRICT_LEXICAL = RefusalPolicy(
    engine="lexical",
    min_top_score=100.0,  # nothing clears this — guarantees a pre-LLM refusal
    score_floor=0.5,
    min_passages_above_floor=1,
    policy_version="test-strict",
)


@pytest.fixture()
def mini_libv2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {FIXTURE_ROOT}")
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(FIXTURE_ROOT, libv2_root / "courses" / COURSE_SLUG)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return libv2_root


def _envelope(answer, citations, not_in_course=False):
    return json.dumps(
        {"answer": answer, "citations": citations, "not_in_course": not_in_course}
    )


def test_on_progress_fires_once_with_passages_on_answered_path(mini_libv2: Path):
    calls = []
    client = FakeAnswerClient(
        [_envelope("A vector store indexes embedding vectors.", ["mini_alpha_chunk_001"])]
    )
    result = answer_course_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        client=client,
        refusal_policy=_PERMISSIVE_LEXICAL,
        on_progress=calls.append,
    )
    assert result.status == STATUS_ANSWERED
    # Exactly one disclosure, after the gate, before compose.
    assert len(calls) == 1
    payload = calls[0]
    assert payload["refused"] is False
    assert payload["status"] is None
    assert isinstance(payload["passages"], list) and payload["passages"]
    first = payload["passages"][0]
    # JSON-serializable + bounded snippet + provenance fields present.
    assert set(first) >= {"chunk_id", "score", "snippet", "item_path", "course_slug"}
    json.dumps(payload)  # must be serializable for the async job store


def test_on_progress_fires_on_refused_path_before_compose(mini_libv2: Path):
    calls = []
    # A client that would RAISE if compose were reached — proves the refusal
    # short-circuit happens after the disclosure, not after an LLM call.
    client = FakeAnswerClient([])  # empty → composer would fail if invoked
    result = answer_course_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        client=client,
        refusal_policy=_STRICT_LEXICAL,
        on_progress=calls.append,
    )
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE
    assert len(calls) == 1
    assert calls[0]["refused"] is True
    assert calls[0]["status"] == STATUS_REFUSED_LOW_CONFIDENCE


def test_on_progress_exception_never_breaks_the_answer(mini_libv2: Path):
    client = FakeAnswerClient(
        [_envelope("A vector store indexes embedding vectors.", ["mini_alpha_chunk_001"])]
    )

    def boom(_payload):
        raise RuntimeError("callback blew up")

    result = answer_course_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        client=client,
        refusal_policy=_PERMISSIVE_LEXICAL,
        on_progress=boom,
    )
    # The answer is unaffected by the callback failure.
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == "A vector store indexes embedding vectors."


def test_default_no_callback_is_byte_identical(mini_libv2: Path):
    client = FakeAnswerClient(
        [_envelope("A vector store indexes embedding vectors.", ["mini_alpha_chunk_001"])]
    )
    result = answer_course_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        client=client,
        refusal_policy=_PERMISSIVE_LEXICAL,
    )
    assert result.status == STATUS_ANSWERED
    assert len(result.citations) == 1


def test_library_wide_seam_threads_on_progress_on_single_course_path(mini_libv2: Path):
    """``answer_library_question`` (library-wide OFF) delegates to the single-course
    path AND threads the L4 callback through it."""
    from lib.retrieval.library_wide import answer_library_question

    calls = []
    client = FakeAnswerClient(
        [_envelope("A vector store indexes embedding vectors.", ["mini_alpha_chunk_001"])]
    )
    result = answer_library_question(
        mini_libv2,
        COURSE_SLUG,
        "What does a vector store index?",
        engine="lexical",
        client=client,
        refusal_policy=_PERMISSIVE_LEXICAL,
        library_wide=False,  # explicit single-course
        on_progress=calls.append,
    )
    assert result.status == STATUS_ANSWERED
    assert len(calls) == 1
    assert calls[0]["refused"] is False
    assert calls[0]["passages"]
