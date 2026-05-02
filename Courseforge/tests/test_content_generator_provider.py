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


# ---------------------------------------------------------------------------
# Pipeline integration (Subtask 14): drives _generate_course_content with
# COURSEFORGE_PROVIDER=local + LOCAL_SYNTHESIS_BASE_URL pointing at a
# MockTransport URL so we observe at least one POST to /v1/chat/completions
# while no Anthropic SDK gets imported.
# ---------------------------------------------------------------------------


_DART_HTML_FIXTURE = """<!DOCTYPE html>
<html lang="en">
<head><title>Photosynthesis Basics</title></head>
<body>
<main id="main-content" role="main">
<section id="objectives-src" aria-labelledby="objectives-src-heading">
  <h2 id="objectives-src-heading">Chapter Objectives</h2>
  <p>After reading this chapter you will be able to:</p>
  <ul>
    <li>Describe the biological process of photosynthesis.</li>
    <li>Explain the two stages of photosynthesis and how they couple.</li>
  </ul>
</section>
<section id="intro" aria-labelledby="intro-heading">
  <h2 id="intro-heading">Introduction to Photosynthesis</h2>
  <p>Photosynthesis is the biological process by which plants, algae, and
  some bacteria convert light energy into chemical energy stored as
  glucose. This fundamental process sustains nearly all life on Earth by
  producing the oxygen we breathe and forming the base of most food webs.</p>
  <p>Photosynthesis occurs primarily in chloroplasts, specialized
  organelles found in the cells of plant leaves. Chloroplasts contain
  chlorophyll, a green pigment that absorbs light energy most effectively
  in the red and blue portions of the visible spectrum.</p>
</section>
<section id="stages" aria-labelledby="stages-heading">
  <h2 id="stages-heading">The Two Stages of Photosynthesis</h2>
  <p>Photosynthesis proceeds in two interconnected stages: the
  light-dependent reactions and the Calvin cycle, also known as the
  light-independent reactions.</p>
  <p>The light-dependent reactions occur in the thylakoid membranes of
  the chloroplast. The Calvin cycle takes place in the stroma, the
  fluid-filled space surrounding the thylakoids.</p>
</section>
</main>
</body>
</html>
"""


def test_pipeline_tools_routes_through_provider_when_env_set(
    monkeypatch, tmp_path
):
    """Subtask 14: COURSEFORGE_PROVIDER=local drives _generate_course_content
    through the in-process provider. Asserts a POST hits the local
    /v1/chat/completions surface and that the Anthropic SDK is never
    imported on the call path.
    """
    pytest.importorskip("httpx")

    import asyncio
    import json as _json

    from MCP.tools import pipeline_tools  # noqa: WPS433
    from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: WPS433

    # Direct env var routes content-generator through the provider.
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://mocktest.local/v1")

    # Verify Anthropic SDK is NOT in sys.modules before the call. (If it
    # was imported by a prior test we can't strictly assert it's absent
    # post-call, but we DO assert the local provider path is what got
    # invoked via the handler being hit.)
    sys.modules.pop("anthropic", None)

    seen_requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        body = (
            "<section><h2>Generated Topic</h2><p>"
            + ("Educational paragraph alpha beta gamma delta. " * 12)
            + "</p></section>"
        )
        return httpx.Response(200, json=_success_body(body))

    # OpenAICompatibleClient lazily builds an httpx.Client when no
    # ``client`` was injected. We patch httpx.Client to a MockTransport-
    # backed factory so the env-var-driven provider construction (no
    # injection seam) still routes through our mock.
    import httpx as _httpx

    _real_client = _httpx.Client

    def _client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return _real_client(transport=_httpx.MockTransport(handler))

    monkeypatch.setattr(_httpx, "Client", _client_factory)

    # Redirect Courseforge inputs + project root to tmp.
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)

    project_id = "PROJ-COURSEFORGE-PROVIDER-TEST"
    project_path = (
        tmp_path / "Courseforge" / "exports" / project_id
    )
    (project_path / "03_content_development").mkdir(parents=True, exist_ok=True)
    config = {
        "project_id": project_id,
        "course_name": "DEMO_PROVIDER_101",
        "duration_weeks": 1,
        "objectives_path": None,
    }
    (project_path / "project_config.json").write_text(
        _json.dumps(config, indent=2), encoding="utf-8"
    )

    # Stage a single DART HTML so build_week_data has at least one
    # renderable topic (the provider seam fires per-topic).
    staging_dir = staging_root / "WF-PROV-01"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "photosynthesis.html").write_text(
        _DART_HTML_FIXTURE, encoding="utf-8"
    )

    registry = _build_tool_registry()
    result = asyncio.run(
        registry["generate_course_content"](
            project_id=project_id,
            staging_dir=str(staging_dir),
        )
    )
    payload = _json.loads(result)
    assert payload.get("success") is True, payload

    # Provider was invoked at least once.
    assert len(seen_requests) >= 1, (
        "Expected at least one POST to the local provider; got none. "
        "COURSEFORGE_PROVIDER routing did not engage the in-process "
        "provider seam."
    )
    assert all(
        str(r.url) == "http://mocktest.local/v1/chat/completions"
        for r in seen_requests
    ), [str(r.url) for r in seen_requests]

    # Anthropic SDK MUST NOT have been imported on the call path. The
    # local provider branch lazy-imports nothing; only the anthropic
    # branch lazy-imports ``anthropic``. If it shows up here, the env
    # var routing leaked into the wrong branch.
    assert "anthropic" not in sys.modules, (
        "Anthropic SDK leaked into sys.modules during a "
        "COURSEFORGE_PROVIDER=local run."
    )
