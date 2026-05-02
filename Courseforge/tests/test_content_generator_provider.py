"""Tests for ContentGeneratorProvider.

Exercises the LLM-agnostic Courseforge content-generator provider that
opens a Phase-1 in-process LLM seam alongside the existing Wave-74
subagent dispatch path. Coverage:

- Construction: unknown provider raises, default provider is anthropic,
  ``COURSEFORGE_PROVIDER`` env honored, supported providers set.
- Anthropic / Together missing-API-key fail-loud RuntimeError.
- Local-backed happy path (httpx.MockTransport on the Ollama default
  endpoint).
- Together-backed happy path (different base URL).
- Anthropic-backed happy path (mock SDK client).
- DecisionCapture fires per call with page_id + course_code +
  week_number + provider + model in the rationale.
- ``page_id`` / ``course_code`` empty raises ``ValueError``.

Mirrors ``Trainforge/tests/test_curriculum_alignment_provider.py`` for
import-path + helper conventions so the two LLM call-site test surfaces
stay parallel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._provider import (  # noqa: E402
    ContentGeneratorProvider,
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    SUPPORTED_PROVIDERS,
    SynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-model") -> dict:
    return {
        "id": "cmpl-cf-test",
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
            "completion_tokens": 50,
            "total_tokens": 250,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _sample_page_context() -> Dict[str, Any]:
    return {
        "objectives": [
            {"id": "TO-01", "statement": "Define the central concept."},
            {"id": "CO-01", "statement": "Explain the introductory framing."},
        ],
        "key_terms": [
            {"term": "central concept", "definition": "the anchoring idea"},
            {"term": "framing", "definition": "the introductory context"},
        ],
        "section_headings": [
            "Introduction",
            "Core Concept",
            "Worked Example",
        ],
        "primary_topic": {
            "title": "Introductory Framing",
            "summary": "Sets the foundational vocabulary for the course.",
        },
    }


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        ContentGeneratorProvider(provider="bogus")


def test_default_provider_is_anthropic_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    p = ContentGeneratorProvider(anthropic_client=object())
    assert p._provider == "anthropic"
    assert DEFAULT_PROVIDER == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    """COURSEFORGE_PROVIDER overrides the default."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = ContentGeneratorProvider()
    assert p._provider == "local"


def test_supported_providers_set_is_three():
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "together", "local"}


def test_anthropic_backend_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        ContentGeneratorProvider(provider="anthropic")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_together_backend_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        ContentGeneratorProvider(provider="together")
    assert "TOGETHER_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Happy paths per backend
# ---------------------------------------------------------------------------


def test_local_backend_routes_to_local_base_url(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = "<section><h2>Topic</h2><p>" + ("alpha " * 100) + "</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = ContentGeneratorProvider(
        provider="local",
        client=_make_client(handler),
    )
    out = p.generate_page(
        course_code="DEMO_101",
        week_number=1,
        page_id="week_01_content_01_intro",
        page_template="<!--TEMPLATE-->",
        page_context=_sample_page_context(),
    )
    assert "<section>" in out
    assert len(seen) == 1
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"


def test_together_backend_returns_html(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("<p>Body</p>"))

    p = ContentGeneratorProvider(
        provider="together",
        client=_make_client(handler),
    )
    out = p.generate_page(
        course_code="DEMO_101",
        week_number=2,
        page_id="week_02_content_01_topic",
        page_template="<!--TEMPLATE-->",
        page_context=_sample_page_context(),
    )
    assert "<p>" in out
    assert "api.together.xyz/v1/chat/completions" in str(seen[0].url)


def test_anthropic_backend_returns_html(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            return {
                "content": [
                    {"type": "text", "text": "<p>Body</p>"},
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    p = ContentGeneratorProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
    )
    out = p.generate_page(
        course_code="DEMO_101",
        week_number=4,
        page_id="week_04_content_01_topic",
        page_template="<!--TEMPLATE-->",
        page_context=_sample_page_context(),
    )
    assert "<p>" in out


# ---------------------------------------------------------------------------
# Decision capture + input validation
# ---------------------------------------------------------------------------


def test_decision_capture_fires_with_page_id_and_provider_in_rationale(
    monkeypatch,
):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body("<section><h2>X</h2><p>body</p></section>"),
        )

    p = ContentGeneratorProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    p.generate_page(
        course_code="DEMO_101",
        week_number=3,
        page_id="week_03_content_01_topic",
        page_template="<!--TEMPLATE-->",
        page_context=_sample_page_context(),
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "content_generator_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    assert "course_code=DEMO_101" in rationale
    assert "week_number=3" in rationale
    assert "week_03_content_01_topic" in rationale
    assert "provider=local" in rationale
    assert "model=" in rationale
    decision = event["decision"]
    assert "week_03_content_01_topic" in decision


def test_empty_page_id_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = ContentGeneratorProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("<p>x</p>"))
        ),
    )
    with pytest.raises(ValueError):
        p.generate_page(
            course_code="DEMO_101",
            week_number=1,
            page_id="",
            page_template="<!--TEMPLATE-->",
            page_context=_sample_page_context(),
        )


def test_empty_course_code_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = ContentGeneratorProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("<p>x</p>"))
        ),
    )
    with pytest.raises(ValueError):
        p.generate_page(
            course_code="",
            week_number=1,
            page_id="week_01_content_01_topic",
            page_template="<!--TEMPLATE-->",
            page_context=_sample_page_context(),
        )
