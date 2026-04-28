"""Tests for CurriculumAlignmentProvider.

Exercises the LLM-agnostic teaching-role classifier that replaces the
Anthropic-pinned LLM call inside ``Trainforge.align_chunks``. Coverage:

- Together-backed happy path (httpx.MockTransport).
- Local-backed happy path (different base URL, no API key required).
- Anthropic-backed happy path (mock SDK client).
- Invalid response (model returns a word outside the four allowed
  roles) raises ``SynthesisProviderError(code="invalid_role_response")``
  so the caller's fallback path fires.
- DecisionCapture fires per call with chunk_id + chosen_role +
  provider in the rationale.
- ``CURRICULUM_ALIGNMENT_PROVIDER`` env var honored at construction.
- align_chunks integration: when no curriculum_provider is injected,
  the pre-existing LLMBackend / mock path still drives classification
  (backward compat).
- align_chunks integration: when curriculum_provider IS injected, it
  routes through that provider for ambiguous chunks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._curriculum_provider import (  # noqa: E402
    CurriculumAlignmentProvider,
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    SUPPORTED_PROVIDERS,
    SynthesisProviderError,
    VALID_ROLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-model") -> dict:
    return {
        "id": "cmpl-curr-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 1,
            "total_tokens": 201,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chunk_text() -> str:
    return (
        "This section introduces the central concept of topic X. "
        "Learners will encounter this idea repeatedly in later "
        "chapters; the foundational framing here anchors all "
        "subsequent examples."
    )


def _neighbors() -> List[Dict[str, Any]]:
    return [
        {
            "id": "chunk_000",
            "concept_tags": ["intro", "topic-x"],
            "text": "Course overview and welcome.",
        },
        {
            "id": "chunk_002",
            "concept_tags": ["topic-x"],
            "text": "Worked example using topic X.",
        },
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        CurriculumAlignmentProvider(provider="bogus")


def test_default_provider_is_anthropic_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    p = CurriculumAlignmentProvider(anthropic_client=object())
    assert p._provider == "anthropic"
    assert DEFAULT_PROVIDER == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    """CURRICULUM_ALIGNMENT_PROVIDER overrides the default."""
    monkeypatch.setenv(ENV_PROVIDER, "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = CurriculumAlignmentProvider(
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("introduce"))
        )
    )
    assert p._provider == "together"


def test_supported_providers_set_is_three():
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "together", "local"}


# ---------------------------------------------------------------------------
# Happy paths per backend
# ---------------------------------------------------------------------------


def test_together_backend_returns_one_of_four_roles(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("introduce"))

    p = CurriculumAlignmentProvider(
        provider="together",
        client=_make_client(handler),
    )
    role = p.classify_teaching_role(
        _chunk_text(),
        chunk_id="chunk_001",
        neighbors=_neighbors(),
    )
    assert role == "introduce"
    assert role in VALID_ROLES
    # Endpoint built off Together's base URL.
    assert "api.together.xyz/v1/chat/completions" in str(seen[0].url)


def test_local_backend_routes_to_local_base_url(monkeypatch):
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("elaborate"))

    p = CurriculumAlignmentProvider(
        provider="local",
        client=_make_client(handler),
    )
    role = p.classify_teaching_role(
        _chunk_text(),
        chunk_id="chunk_001",
        neighbors=_neighbors(),
    )
    assert role == "elaborate"
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"


def test_anthropic_backend_returns_role(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    class _FakeMessages:
        def create(self, **kwargs):
            return {
                "content": [
                    {"type": "text", "text": "synthesize"},
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    p = CurriculumAlignmentProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
    )
    role = p.classify_teaching_role(
        _chunk_text(),
        chunk_id="chunk_001",
        neighbors=_neighbors(),
    )
    assert role == "synthesize"


def test_anthropic_backend_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        CurriculumAlignmentProvider(provider="anthropic")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_together_backend_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        CurriculumAlignmentProvider(provider="together")
    assert "TOGETHER_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Invalid-response handling
# ---------------------------------------------------------------------------


def test_invalid_role_response_raises_synthesis_provider_error(monkeypatch):
    """A response of "hello" must raise the typed error code."""
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = CurriculumAlignmentProvider(
        provider="together",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("hello"))
        ),
    )
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.classify_teaching_role(
            _chunk_text(), chunk_id="chunk_001", neighbors=_neighbors()
        )
    assert excinfo.value.code == "invalid_role_response"
    assert excinfo.value.chunk_id == "chunk_001"


def test_empty_response_raises_invalid_role(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = CurriculumAlignmentProvider(
        provider="together",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(""))
        ),
    )
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.classify_teaching_role(
            _chunk_text(), chunk_id="cX", neighbors=_neighbors()
        )
    assert excinfo.value.code == "invalid_role_response"


def test_role_with_trailing_punctuation_still_matches(monkeypatch):
    """``introduce.`` should classify as ``introduce`` (tolerant parse)."""
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = CurriculumAlignmentProvider(
        provider="together",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("introduce."))
        ),
    )
    role = p.classify_teaching_role(
        _chunk_text(), chunk_id="cX", neighbors=_neighbors()
    )
    assert role == "introduce"


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def test_decision_capture_fires_with_chunk_id_and_chosen_role(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    cap = _FakeCapture()
    p = CurriculumAlignmentProvider(
        provider="together",
        capture=cap,
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("reinforce"))
        ),
    )
    p.classify_teaching_role(
        _chunk_text(),
        chunk_id="chunk_042",
        neighbors=_neighbors(),
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "curriculum_alignment_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    assert "chunk_id=chunk_042" in rationale
    assert "reinforce" in rationale
    assert "provider=together" in rationale
    decision = event["decision"]
    assert "chunk_042" in decision
    assert "role=reinforce" in decision


# ---------------------------------------------------------------------------
# align_chunks.py integration
# ---------------------------------------------------------------------------


def test_align_chunks_without_curriculum_provider_keeps_legacy_path():
    """Backward compat: no curriculum_provider → existing mock path runs."""
    from Trainforge.align_chunks import classify_teaching_roles

    chunks = [
        {
            "id": "c_legacy",
            "_position": 0,
            "chunk_type": "content",
            "concept_tags": ["fresh"],
            "text": "freshly introduced concept",
            "source": {"resource_type": "content"},
        },
    ]
    classify_teaching_roles(chunks, llm_provider="mock", verbose=False)
    # Mock fallback assigns "introduce" + source "mock" — exactly what
    # test_align_chunks_mock_fallback_preserved already locks in.
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "mock"


def test_align_chunks_with_curriculum_provider_injection():
    """Injection: ambiguous chunks route through the curriculum provider."""
    from Trainforge.align_chunks import classify_teaching_roles

    class _FakeCurriculumProvider:
        """Stand-in: just returns a fixed role without any HTTP."""

        def __init__(self) -> None:
            self.calls: List[str] = []

        def classify_teaching_role(
            self, chunk_text, *, chunk_id, neighbors, course_outcomes=None
        ) -> str:
            self.calls.append(chunk_id)
            return "synthesize"

    fake = _FakeCurriculumProvider()
    chunks = [
        {
            "id": "c_amb",
            "_position": 0,
            "chunk_type": "content",
            "concept_tags": ["fresh"],
            "text": "A passage that needs LLM-driven classification.",
            "source": {"resource_type": "content"},
        },
    ]
    classify_teaching_roles(
        chunks,
        llm_provider="mock",
        verbose=False,
        curriculum_provider=fake,
    )
    assert chunks[0]["teaching_role"] == "synthesize"
    assert chunks[0]["teaching_role_source"] == "llm"
    assert fake.calls == ["c_amb"]


def test_align_chunks_curriculum_provider_failure_falls_back_to_mock():
    """Provider error → mock fallback (pipeline keeps moving)."""
    from Trainforge.align_chunks import classify_teaching_roles

    class _FailingProvider:
        def classify_teaching_role(self, *_a, **_k):
            raise SynthesisProviderError(
                "boom", code="invalid_role_response"
            )

    chunks = [
        {
            "id": "c_fail",
            "_position": 0,
            "chunk_type": "content",
            "concept_tags": ["topic"],
            "text": "Ambiguous chunk text.",
            "source": {"resource_type": "content"},
        },
    ]
    classify_teaching_roles(
        chunks,
        llm_provider="mock",
        verbose=False,
        curriculum_provider=_FailingProvider(),
    )
    # Mock fallback should still produce one of the legitimate roles.
    assert chunks[0]["teaching_role"] in {
        "introduce", "elaborate", "reinforce", "assess",
        "transfer", "synthesize",
    }
