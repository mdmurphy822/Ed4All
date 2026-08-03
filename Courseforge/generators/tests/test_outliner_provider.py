"""Tests for ``OutlinerProvider`` (Wave W-D14).

Pin the in-process course-outliner LLM provider:

- Construction: default provider is ``anthropic`` when env unset
  (matches ``DEFAULT_PROVIDER``); ``COURSEPLANNER_PROVIDER`` selects an
  alternate at construction time; supported_providers is composed
  dynamically from the W-D12 registry.
- Provider routing — anthropic vs local: ``anthropic`` routes through
  the Anthropic SDK seam; ``local`` routes through the
  ``OpenAICompatibleClient`` HTTP path.
- Structured-output emit: a successful synthesise call returns the
  canonical synthesized_objectives shape (terminal_objectives +
  chapter_objectives groups + mint_method) consumable by
  ``_plan_course_structure``.
- ID re-minting: emitted objectives carry sequential ``TO-NN`` /
  ``CO-NN`` IDs regardless of what the LLM stamped — ID minting is
  authoritative.
- Decision capture: every dispatch emits exactly one
  ``course_outline_call`` event whose rationale interpolates dynamic
  per-call signals (provider, model, course_name, chapter_count_input,
  weeks_input, terminal_count_emit, chapter_count_emit, retry_count).
- Dynamic-rationale regression: rationale records the RUNTIME
  provider name (e.g. ``provider=local``) — NOT a static label.
- Parse exhaustion: the LLM emitting persistent gibberish raises
  ``OutlinerProviderError(code="outliner_exhausted")``.
- Lenient JSON extraction: markdown-fenced response is recovered.

Mirrors ``test_outline_provider.py`` for httpx-fixture patterns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outliner_provider import (  # noqa: E402
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    OutlinerProvider,
    OutlinerProviderError,
    _build_supported_providers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-outliner") -> dict:
    return {
        "id": "cmpl-outliner-test",
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
            "completion_tokens": 80,
            "total_tokens": 280,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _stub_textbook_structure() -> Dict[str, Any]:
    return {
        "chapters": [
            {
                "id": "ch1",
                "title": "Introduction to Photosynthesis",
                "sections": [
                    {"title": "Light reactions"},
                    {"title": "Dark reactions"},
                ],
            },
            {
                "id": "ch2",
                "title": "Cellular Respiration",
                "sections": [
                    {"title": "Glycolysis"},
                    {"title": "Krebs cycle"},
                ],
            },
        ]
    }


def _valid_outliner_payload() -> Dict[str, Any]:
    """Return a JSON object that satisfies the structured emit shape."""
    return {
        "terminal_objectives": [
            {
                "statement": (
                    "Students will analyze biological energy "
                    "transformation."
                ),
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "abcd": {
                    "audience": "Students",
                    "behavior": {
                        "verb": "analyze",
                        "action_object": "biological energy transformation",
                    },
                    "condition": "given a textbook chapter",
                    "degree": "with at least 80% accuracy",
                },
                "source_refs": [{"ref": "ch1", "chunk_ids": []}],
            }
        ],
        "chapter_objectives": [
            {
                "statement": (
                    "Apply the concepts of photosynthesis to a real-world "
                    "scenario."
                ),
                "bloom_level": "apply",
                "bloom_verb": "apply",
                "abcd": {
                    "audience": "Students",
                    "behavior": {
                        "verb": "apply",
                        "action_object": "photosynthesis concepts",
                    },
                    "condition": "given a real-world scenario",
                    "degree": "with reference to two reactants",
                },
                "source_refs": [{"ref": "ch1", "chunk_ids": []}],
            },
            {
                "statement": (
                    "Explain the major stages of cellular respiration."
                ),
                "bloom_level": "understand",
                "bloom_verb": "explain",
                "abcd": {
                    "audience": "Students",
                    "behavior": {
                        "verb": "explain",
                        "action_object": "the major stages of cellular respiration",
                    },
                    "condition": "verbally",
                    "degree": "covering at least three stages",
                },
                "source_refs": [{"ref": "ch2", "chunk_ids": []}],
            },
        ],
    }


class _FakeCapture:
    """Capture stub mirroring the production ``DecisionCapture.events`` shape."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_provider_is_anthropic_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    p = OutlinerProvider(anthropic_client=object())
    assert p._provider == "anthropic"
    assert DEFAULT_PROVIDER == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    """``COURSEPLANNER_PROVIDER=local`` overrides the default."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlinerProvider()
    assert p._provider == "local"


def test_unknown_provider_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError) as excinfo:
        OutlinerProvider(provider="bogus_xyz")
    assert "bogus_xyz" in str(excinfo.value)


def test_supported_providers_dynamic_from_registry():
    """Supported providers include anthropic + every W-D12 registry entry."""
    supported = _build_supported_providers()
    assert "anthropic" in supported
    assert "local" in supported
    assert "together" in supported
    # Registry stubs from W-D12 — should flow through dynamically.
    for stub in ("groq", "fireworks", "deepseek"):
        assert stub in supported, (
            f"Registry provider {stub!r} missing from "
            f"_build_supported_providers — registry-references should be dynamic."
        )


# ---------------------------------------------------------------------------
# Provider routing — anthropic vs local
# ---------------------------------------------------------------------------


def test_local_backend_routes_to_local_base_url(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_outliner_payload())),
        )

    p = OutlinerProvider(
        provider="local",
        client=_make_client(handler),
    )
    out = p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_902",
        chapter_count=2,
        weeks=10,
    )
    # Routed to local base URL.
    assert len(seen) == 1
    assert "localhost:11434/v1/chat/completions" in str(seen[0].url)
    # Output carries terminal + chapter objectives.
    assert isinstance(out, dict)
    assert len(out["terminal_objectives"]) == 1
    assert len(out["chapter_objectives"]) == 2  # group-of-groups shape


def test_anthropic_backend_returns_objectives(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    payload = _valid_outliner_payload()

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    p = OutlinerProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
    )
    out = p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_902",
        chapter_count=2,
        weeks=10,
    )
    assert out["mint_method"].startswith("courseplanner_provider:anthropic")
    assert len(out["terminal_objectives"]) == 1


# ---------------------------------------------------------------------------
# Structured output / canonical shape
# ---------------------------------------------------------------------------


def test_synthesize_objectives_returns_canonical_shape(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_outliner_payload())),
        )

    p = OutlinerProvider(provider="local", client=_make_client(handler))
    out = p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_906",
        chapter_count=2,
        weeks=8,
    )
    # Canonical top-level keys consumed by ``_plan_course_structure``.
    assert set(out.keys()) >= {
        "course_name",
        "terminal_objectives",
        "chapter_objectives",
        "mint_method",
        "duration_weeks",
    }
    assert out["course_name"] == "TST_906"
    # Re-minted IDs follow TO-NN / CO-NN pattern.
    assert out["terminal_objectives"][0]["id"] == "TO-01"
    co_ids = [
        objective["id"]
        for grp in out["chapter_objectives"]
        for objective in grp["objectives"]
    ]
    assert co_ids == ["CO-01", "CO-02"]
    # group-of-groups shape (chapter_objectives is a list of
    # ``{"chapter": "Week N", "objectives": [...]}`` dicts).
    for grp in out["chapter_objectives"]:
        assert "chapter" in grp
        assert "objectives" in grp
        assert isinstance(grp["objectives"], list)


def test_lenient_json_recovers_markdown_fenced_response(monkeypatch):
    """A response wrapped in ``` ```json ... ``` ``` markdown fences is
    recovered by the lenient extractor so the outliner accepts the
    payload on the first attempt."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    payload = _valid_outliner_payload()
    fenced = "```json\n" + json.dumps(payload) + "\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(fenced))

    p = OutlinerProvider(provider="local", client=_make_client(handler))
    out = p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_906",
        chapter_count=2,
        weeks=8,
    )
    assert isinstance(out, dict)
    assert len(out["terminal_objectives"]) == 1


# ---------------------------------------------------------------------------
# Parse exhaustion
# ---------------------------------------------------------------------------


def test_persistent_gibberish_raises_outliner_exhausted(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("not-valid-json"))

    p = OutlinerProvider(provider="local", client=_make_client(handler))
    with pytest.raises(OutlinerProviderError) as excinfo:
        p.synthesize_objectives(
            _stub_textbook_structure(),
            course_name="TST_906",
            chapter_count=2,
            weeks=8,
        )
    assert excinfo.value.code == "outliner_exhausted"


def test_empty_objectives_payload_raises_exhausted(monkeypatch):
    """LLM emits valid JSON but with no usable objectives → exhausted."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps({
                "terminal_objectives": [],
                "chapter_objectives": [],
            })),
        )

    p = OutlinerProvider(provider="local", client=_make_client(handler))
    with pytest.raises(OutlinerProviderError) as excinfo:
        p.synthesize_objectives(
            _stub_textbook_structure(),
            course_name="TST_906",
            chapter_count=2,
            weeks=8,
        )
    assert excinfo.value.code == "outliner_exhausted"


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------


def test_decision_capture_fires_with_dynamic_signals(monkeypatch):
    """Every dispatch emits ONE course_outline_call event whose rationale
    interpolates dynamic per-call signals (provider, model, course_name,
    counts, retry_count). The provider value is the RUNTIME provider name
    — NOT a static label — per the dynamic-rationale contract."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_outliner_payload())),
        )

    p = OutlinerProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_906",
        chapter_count=2,
        weeks=8,
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "course_outline_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20

    # Dynamic rationale: the runtime provider name is interpolated
    # verbatim. NOT a static "course-outliner" or "outliner" label.
    assert "provider=local" in rationale
    assert "model=" in rationale
    assert "course_name=TST_906" in rationale
    assert "chapter_count_input=2" in rationale
    assert "weeks_input=8" in rationale
    assert "terminal_count_emit=1" in rationale
    assert "chapter_count_emit=2" in rationale
    assert "retry_count=" in rationale
    assert "success=True" in rationale

    decision = event["decision"]
    assert "TST_906" in decision
    assert "success" in decision


def test_decision_capture_records_failure_with_last_error(monkeypatch):
    """Failed dispatch still emits one event whose rationale carries
    ``last_error`` and ``success=False`` — provenance for failures
    matters as much as for successes."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("garbage"))

    p = OutlinerProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    with pytest.raises(OutlinerProviderError):
        p.synthesize_objectives(
            _stub_textbook_structure(),
            course_name="TST_906",
            chapter_count=2,
            weeks=8,
        )
    assert len(cap.events) == 1
    rationale = cap.events[0]["rationale"]
    assert "success=False" in rationale
    assert "last_error=" in rationale


def test_decision_capture_dynamic_provider_anthropic(monkeypatch):
    """When the runtime provider is anthropic, rationale stamps
    ``provider=anthropic`` — NOT a static fallback label. Pins the
    dynamic-rationale contract on the second backend path."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    payload = _valid_outliner_payload()

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    cap = _FakeCapture()
    p = OutlinerProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
        capture=cap,
    )
    p.synthesize_objectives(
        _stub_textbook_structure(),
        course_name="TST_906",
        chapter_count=2,
        weeks=8,
    )
    assert len(cap.events) == 1
    rationale = cap.events[0]["rationale"]
    assert "provider=anthropic" in rationale


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_course_name_raises(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlinerProvider(provider="local")
    with pytest.raises(ValueError) as excinfo:
        p.synthesize_objectives(
            _stub_textbook_structure(),
            course_name="",
            chapter_count=2,
            weeks=8,
        )
    assert "course_name" in str(excinfo.value)


def test_non_dict_textbook_structure_raises(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlinerProvider(provider="local")
    with pytest.raises(ValueError) as excinfo:
        p.synthesize_objectives(
            "not-a-dict",  # type: ignore[arg-type]
            course_name="TST_906",
        )
    assert "textbook_structure" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_user_prompt_includes_course_name_and_chapters(monkeypatch):
    """Rendered user prompt carries the course header, every chapter id
    + title, and the strict-JSON closing directive."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlinerProvider(provider="local")
    rendered = p._render_user_prompt(
        textbook_structure=_stub_textbook_structure(),
        course_name="TST_902",
        chapter_count=2,
        weeks=10,
    )
    assert "TST_902" in rendered
    assert "ch1" in rendered
    assert "ch2" in rendered
    assert "Introduction to Photosynthesis" in rendered
    assert "RESPOND ONLY WITH A JSON OBJECT" in rendered
    # One-shot example pinned for smaller models.
    assert "TO-01" in rendered
    assert "abcd" in rendered
