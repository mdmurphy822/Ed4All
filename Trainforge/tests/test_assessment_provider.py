"""Tests for ``AssessmentGeneratorProvider`` (Wave W-D15).

Pin the in-process assessment-generator LLM provider:

- Construction: default provider is ``anthropic`` when env unset
  (matches ``DEFAULT_PROVIDER``); ``TRAINFORGE_ASSESSMENT_PROVIDER``
  selects an alternate at construction time; supported_providers is
  composed dynamically from the W-D12 registry.
- Provider routing — anthropic vs local: ``anthropic`` routes
  through the Anthropic SDK seam; ``local`` routes through the
  ``OpenAICompatibleClient`` HTTP path.
- Structured-output emit: a successful generate_assessments call
  returns the canonical questions[] + skipped_items[] shape with
  ``mint_method=trainforge_assessment_provider:<provider>``.
- evidence_quote: emitted-question entries preserve the LLM's
  ``evidence_quote`` field verbatim; the per-call decision-capture
  rationale carries the dynamic ``evidence_quote_emit_rate``.
- Decision capture: every dispatch emits exactly one
  ``assessment_generator_call`` event whose rationale interpolates
  dynamic per-call signals (provider, model, course_code,
  chunk_count, target_count, emitted_questions,
  distinct_question_types, refusal_count, retry_count,
  evidence_quote_emit_rate).
- Dynamic-rationale regression: rationale records the RUNTIME
  provider name (e.g. ``provider=local``) — NOT a static label.
- Parse exhaustion: the LLM emitting persistent gibberish raises
  ``AssessmentGeneratorProviderError(code="assessment_exhausted")``.
- Lenient JSON extraction: markdown-fenced response is recovered.
- Phantom-objective filter: questions referencing an unknown
  objective_id land in skipped_items rather than questions.

Mirrors W-D14's ``test_outliner_provider.py`` for httpx-fixture +
fake-anthropic-client patterns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.providers._assessment_provider import (  # noqa: E402
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    AssessmentGeneratorProvider,
    AssessmentGeneratorProviderError,
    _build_supported_providers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-assessment") -> dict:
    return {
        "id": "cmpl-assessment-test",
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


def _stub_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "chunk_001",
            "chunk_id": "chunk_001",
            "text": (
                "SHACL is a constraint language used to validate RDF "
                "graphs against a set of conditions. Shapes constrain "
                "the structure of nodes and their property values."
            ),
        },
        {
            "id": "chunk_002",
            "chunk_id": "chunk_002",
            "text": (
                "RDF is a directed, labeled graph data format. Nodes "
                "represent resources and edges represent properties."
            ),
        },
    ]


def _stub_objectives() -> List[Dict[str, Any]]:
    return [
        {"id": "TO-01", "statement": "Understand SHACL constraints"},
        {"id": "CO-01", "statement": "Apply RDF graph principles"},
    ]


def _valid_assessment_payload() -> Dict[str, Any]:
    """Return a JSON object that satisfies the structured emit shape."""
    return {
        "questions": [
            {
                "question_type": "multiple_choice",
                "stem": "Which best describes SHACL?",
                "bloom_level": "understand",
                "objective_id": "TO-01",
                "choices": [
                    {
                        "text": (
                            "A constraint language for validating RDF "
                            "graphs"
                        ),
                        "is_correct": True,
                    },
                    {"text": "A query language", "is_correct": False},
                    {"text": "A serialization format", "is_correct": False},
                    {"text": "A schema language", "is_correct": False},
                ],
                "feedback": "SHACL validates RDF graphs.",
                "source_chunks": ["chunk_001"],
                "evidence_quote": (
                    "SHACL is a constraint language used to validate "
                    "RDF graphs"
                ),
            },
            {
                "question_type": "short_answer",
                "stem": "Apply RDF principles to a graph.",
                "bloom_level": "apply",
                "objective_id": "CO-01",
                "correct_answer": "Use directed labeled edges.",
                "source_chunks": ["chunk_002"],
                "evidence_quote": (
                    "RDF is a directed, labeled graph data format"
                ),
            },
        ],
        "skipped_items": [],
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
    p = AssessmentGeneratorProvider(anthropic_client=object())
    assert p._provider == "anthropic"
    assert DEFAULT_PROVIDER == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    """``TRAINFORGE_ASSESSMENT_PROVIDER=local`` overrides the default."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = AssessmentGeneratorProvider()
    assert p._provider == "local"


def test_unknown_provider_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError) as excinfo:
        AssessmentGeneratorProvider(provider="bogus_xyz")
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


def test_explicit_model_override_via_env(monkeypatch):
    """``TRAINFORGE_ASSESSMENT_MODEL`` pins the model independently of
    the per-backend default. Mirrors COURSEPLANNER_MODEL semantics."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_MODEL", "qwen2.5:32b-pinned")
    p = AssessmentGeneratorProvider()
    assert p._model == "qwen2.5:32b-pinned"


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
            json=_success_body(json.dumps(_valid_assessment_payload())),
        )

    p = AssessmentGeneratorProvider(
        provider="local",
        client=_make_client(handler),
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    # Routed to local base URL.
    assert len(seen) == 1
    assert "localhost:11434/v1/chat/completions" in str(seen[0].url)
    # Output carries questions + skipped_items.
    assert isinstance(out, dict)
    assert len(out["questions"]) == 2
    assert "mint_method" in out
    assert out["mint_method"].endswith(":local")


def test_anthropic_backend_returns_questions(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    payload = _valid_assessment_payload()

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            return {
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    p = AssessmentGeneratorProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert out["mint_method"].startswith(
        "trainforge_assessment_provider:anthropic"
    )
    assert len(out["questions"]) == 2


# ---------------------------------------------------------------------------
# Structured output / canonical shape
# ---------------------------------------------------------------------------


def test_generate_assessments_returns_canonical_shape(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_assessment_payload())),
        )

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert set(out.keys()) >= {"questions", "skipped_items", "mint_method"}
    # Each question carries the canonical fields per W-D15 contract.
    for q in out["questions"]:
        assert "question_id" in q and q["question_id"].startswith("Q-LLM-")
        assert q["question_type"] in {
            "multiple_choice",
            "true_false",
            "short_answer",
            "essay",
            "fill_in_blank",
        }
        assert q["bloom_level"] in {
            "remember",
            "understand",
            "apply",
            "analyze",
            "evaluate",
            "create",
        }
        assert q["objective_id"] in {"TO-01", "CO-01"}
        assert "evidence_quote" in q


def test_evidence_quote_preserved_verbatim(monkeypatch):
    """Per the W-D11 T11.3 contract, the LLM-emitted ``evidence_quote``
    is preserved verbatim on every question entry."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_assessment_payload())),
        )

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    quotes = [q["evidence_quote"] for q in out["questions"]]
    assert all(q for q in quotes)
    assert any(
        "SHACL is a constraint language" in q for q in quotes if q
    )


def test_evidence_quote_directive_in_prompt(monkeypatch):
    """The provider's user-prompt MUST instruct the LLM to emit the
    evidence_quote field — pin the directive text so a refactor can't
    silently drop the W-D11 grounding contract."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    seen_payloads: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_assessment_payload())),
        )

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert len(seen_payloads) == 1
    user_msg = next(
        (
            m["content"]
            for m in seen_payloads[0]["messages"]
            if m["role"] == "user"
        ),
        "",
    )
    assert "evidence_quote" in user_msg
    assert "verbatim substring" in user_msg


def test_lenient_json_recovers_markdown_fenced_response(monkeypatch):
    """A response wrapped in ``` ```json ... ``` ``` markdown fences is
    recovered by the lenient extractor so the provider accepts the
    payload on the first attempt."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    payload = _valid_assessment_payload()
    fenced = "```json\n" + json.dumps(payload) + "\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(fenced))

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert isinstance(out, dict)
    assert len(out["questions"]) == 2


# ---------------------------------------------------------------------------
# Phantom-objective filter
# ---------------------------------------------------------------------------


def test_phantom_objective_id_lands_in_skipped_items(monkeypatch):
    """A question referencing an objective_id NOT in the supplied
    objectives list MUST land in skipped_items rather than questions
    (mirrors AssessmentObjectiveAlignmentValidator's phantom-LO
    rejection on the synthesis surface)."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    payload = _valid_assessment_payload()
    payload["questions"].append({
        "question_type": "multiple_choice",
        "stem": "Phantom-LO question",
        "bloom_level": "remember",
        "objective_id": "TO-99",  # not in stub objectives
        "choices": [
            {"text": "x", "is_correct": True},
            {"text": "y", "is_correct": False},
        ],
        "source_chunks": ["chunk_001"],
        "evidence_quote": "SHACL is a constraint language",
    })

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    out = p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=3,
        course_code="DEMO_101",
    )
    # The phantom-LO question landed in skipped_items, not questions.
    assert len(out["questions"]) == 2
    phantom_skips = [
        s for s in out["skipped_items"]
        if s["reason"] == "phantom_objective_id"
    ]
    assert len(phantom_skips) == 1
    assert phantom_skips[0]["objective_id"] == "TO-99"


# ---------------------------------------------------------------------------
# Parse exhaustion
# ---------------------------------------------------------------------------


def test_persistent_gibberish_raises_assessment_exhausted(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("not-valid-json"))

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    with pytest.raises(AssessmentGeneratorProviderError) as excinfo:
        p.generate_assessments(
            chunks=_stub_chunks(),
            objectives=_stub_objectives(),
            target_count=2,
            course_code="DEMO_101",
        )
    assert excinfo.value.code == "assessment_exhausted"


def test_empty_payload_raises_exhausted(monkeypatch):
    """LLM emits valid JSON but with no usable questions or skips →
    exhausted."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps({
                "questions": [],
                "skipped_items": [],
            })),
        )

    p = AssessmentGeneratorProvider(
        provider="local", client=_make_client(handler)
    )
    with pytest.raises(AssessmentGeneratorProviderError) as excinfo:
        p.generate_assessments(
            chunks=_stub_chunks(),
            objectives=_stub_objectives(),
            target_count=2,
            course_code="DEMO_101",
        )
    assert excinfo.value.code == "assessment_exhausted"


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------


def test_decision_capture_fires_with_dynamic_signals(monkeypatch):
    """Every dispatch emits ONE assessment_generator_call event whose
    rationale interpolates dynamic per-call signals (provider, model,
    course_code, counts, retry_count, evidence_quote_emit_rate). The
    provider value is the RUNTIME provider name — NOT a static label —
    per the dynamic-rationale contract."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    cap = _FakeCapture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(json.dumps(_valid_assessment_payload())),
        )

    p = AssessmentGeneratorProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert len(cap.events) == 1
    event = cap.events[0]
    assert event["decision_type"] == "assessment_generator_call"
    rationale = event["rationale"]
    assert len(rationale) >= 20

    # Dynamic rationale: the runtime provider name is interpolated
    # verbatim. NOT a static "assessment-generator" or "trainforge" label.
    assert "provider=local" in rationale
    assert "model=" in rationale
    assert "course_code=DEMO_101" in rationale
    assert "chunk_count=2" in rationale
    assert "target_count=2" in rationale
    assert "emitted_questions=2" in rationale
    assert "distinct_question_types=" in rationale
    assert "refusal_count=0" in rationale
    assert "evidence_quote_emit_rate=" in rationale
    assert "retry_count=" in rationale
    assert "success=True" in rationale

    decision = event["decision"]
    assert "DEMO_101" in decision
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

    p = AssessmentGeneratorProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    with pytest.raises(AssessmentGeneratorProviderError):
        p.generate_assessments(
            chunks=_stub_chunks(),
            objectives=_stub_objectives(),
            target_count=2,
            course_code="DEMO_101",
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

    payload = _valid_assessment_payload()

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
    p = AssessmentGeneratorProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
        capture=cap,
    )
    p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert len(cap.events) == 1
    rationale = cap.events[0]["rationale"]
    assert "provider=anthropic" in rationale


def test_decision_capture_evidence_quote_emit_rate_dynamic(monkeypatch):
    """The evidence_quote_emit_rate field reports the actual ratio of
    questions that carry a non-empty evidence_quote. When some
    questions emit null quotes, the rate falls below 1.0."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    payload = _valid_assessment_payload()
    # Drop the evidence_quote on the first question.
    payload["questions"][0]["evidence_quote"] = None

    cap = _FakeCapture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = AssessmentGeneratorProvider(
        provider="local",
        capture=cap,
        client=_make_client(handler),
    )
    p.generate_assessments(
        chunks=_stub_chunks(),
        objectives=_stub_objectives(),
        target_count=2,
        course_code="DEMO_101",
    )
    assert len(cap.events) == 1
    rationale = cap.events[0]["rationale"]
    # 1 of 2 questions carries a quote → 0.500 rate.
    assert "evidence_quote_emit_rate=0.500" in rationale


# ---------------------------------------------------------------------------
# Empty-input guards
# ---------------------------------------------------------------------------


def test_empty_chunks_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = AssessmentGeneratorProvider(provider="local")
    with pytest.raises(ValueError):
        p.generate_assessments(
            chunks=[],
            objectives=_stub_objectives(),
            target_count=2,
            course_code="DEMO_101",
        )


def test_empty_objectives_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = AssessmentGeneratorProvider(provider="local")
    with pytest.raises(ValueError):
        p.generate_assessments(
            chunks=_stub_chunks(),
            objectives=[],
            target_count=2,
            course_code="DEMO_101",
        )
