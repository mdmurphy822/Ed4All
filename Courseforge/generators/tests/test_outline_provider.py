"""Tests for ``OutlineProvider`` (Phase 3 Subtask 57).

Exercises the outline-tier LLM-agnostic provider that emits the
structurally-correct JSON skeleton each Phase-3 Block needs (the cheap
first pass of the two-pass router). Coverage:

- Construction: default provider is ``local`` when env unset (matches
  ``DEFAULT_PROVIDER``); ``COURSEFORGE_OUTLINE_PROVIDER`` selects an
  alternate at construction time.
- ``_OUTLINE_KIND_BOUNDS`` shape: per-block-type bounds expose
  ``key_claims`` / ``section_skeleton`` / ``summary_chars`` triples
  with strictly-positive max values where the bounds are non-degenerate.
- User-prompt rendering: includes ``block_id``, objective IDs, the
  per-block-type structural-bounds block, and the canonical strict-JSON
  closing directive (``RESPOND ONLY WITH A JSON OBJECT``).
- Lenient JSON extraction: a ```json ...``` markdown-fenced response
  recovers via :meth:`OpenAICompatibleClient._extract_json_lenient`.
- Retry budget exhaustion: every retry returns invalid JSON (or
  Schema-invalid JSON) → ``OutlineProviderError(code="outline_exhausted")``.
- Per-block-type JSON Schema enforcement: ``_BLOCK_TYPE_JSON_SCHEMAS``
  rejects a missing ``key_claims`` value by triggering the parse-retry
  loop until exhaustion (regression sentinel for the schema-validation
  branch).
- Touch chain: a successful generation appends a single new
  ``Touch(tier="outline", purpose="draft", ...)`` to the input
  ``touched_by`` chain.
- Decision capture: every dispatch (success or failure) emits a
  ``block_outline_call`` event whose rationale interpolates dynamic
  per-call signals (block_id, block_type, page_id, provider, model,
  retry_count, attempts, success).

Mirrors ``Trainforge/tests/test_curriculum_alignment_provider.py`` for
import-path + helper conventions and
``Courseforge/generators/tests/test_rewrite_provider.py`` for the
``httpx.MockTransport`` fixture pattern so the two LLM call-site test
surfaces stay parallel.
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

from Courseforge.generators._outline_provider import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    MAX_PARSE_RETRIES,
    OutlineProvider,
    OutlineProviderError,
    SUPPORTED_PROVIDERS,
    _BLOCK_TYPE_JSON_SCHEMAS,
    _OUTLINE_KIND_BOUNDS,
)
from blocks import BLOCK_TYPES, Block, Touch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-outline") -> dict:
    return {
        "id": "cmpl-outline-test",
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


def _stub_block(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
    page_id: str = "page-1",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=0,
        content="",
    )


def _valid_outline_payload(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
) -> Dict[str, Any]:
    """Return a JSON object that satisfies the per-block-type schema."""
    bounds = _OUTLINE_KIND_BOUNDS.get(block_type, {})
    section_min, _section_max = bounds.get("section_skeleton", (0, 0))
    payload: Dict[str, Any] = {
        "block_id": block_id,
        "block_type": block_type,
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": ["sh:NodeShape"],
        "key_claims": ["The central concept is X."],
        "section_skeleton": [
            {"heading": "Definition"} for _ in range(max(section_min, 1))
        ] if section_min > 0 else [],
        "source_refs": [{"sourceId": "dart:slug#blk1", "role": "primary"}],
        "structural_warnings": [],
    }
    return payload


class _FakeCapture:
    """Capture stub mirroring the production ``DecisionCapture.events`` shape."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_outline_provider_is_local_when_env_unset(monkeypatch):
    """``COURSEFORGE_OUTLINE_PROVIDER`` unset → defaults to ``local``."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider()
    assert p._provider == "local"
    assert DEFAULT_PROVIDER == "local"
    # The default model dial points at a 7B-class instruction model.
    assert DEFAULT_MODEL.startswith("qwen")


def test_env_var_selects_provider(monkeypatch):
    """``COURSEFORGE_OUTLINE_PROVIDER=together`` → routes to Together."""
    monkeypatch.setenv(ENV_PROVIDER, "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = OutlineProvider(
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("{}"))
        )
    )
    assert p._provider == "together"
    assert "together" in SUPPORTED_PROVIDERS


# ---------------------------------------------------------------------------
# Per-block-type bounds
# ---------------------------------------------------------------------------


def test_outline_kind_bounds_per_block_type():
    """Every ``BLOCK_TYPES`` value has a bounds entry exposing the three
    canonical fields (``key_claims`` / ``section_skeleton`` / ``summary_chars``);
    each ``(lo, hi)`` tuple is non-decreasing and admits at least one
    valid value (``hi >= lo``)."""
    canonical_fields = {"key_claims", "section_skeleton", "summary_chars"}
    for block_type in BLOCK_TYPES:
        assert block_type in _OUTLINE_KIND_BOUNDS, (
            f"{block_type!r} missing from _OUTLINE_KIND_BOUNDS"
        )
        bounds = _OUTLINE_KIND_BOUNDS[block_type]
        assert canonical_fields.issubset(bounds.keys()), (
            f"{block_type!r} bounds missing canonical field; got {set(bounds)}"
        )
        for field, (lo, hi) in bounds.items():
            assert lo >= 0
            assert hi >= lo, (
                f"{block_type}.{field}: bounds invalid ({lo} > {hi})"
            )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_outline_user_prompt_includes_block_id_and_objectives(monkeypatch):
    """The rendered user prompt carries the block_id verbatim, the
    page_id, every supplied objective id, and the strict-JSON closing
    directive."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_id="week01_module01#concept_intro_0")
    objectives = [
        {"id": "TO-01", "statement": "Define the central concept."},
        {"id": "CO-02", "statement": "Explain the introductory framing."},
    ]
    chunks = [
        {"id": "dart:slug#blk1", "body": "Source body content."},
    ]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    assert "week01_module01#concept_intro_0" in rendered
    assert block.page_id in rendered
    assert "TO-01" in rendered
    assert "CO-02" in rendered
    assert "Define the central concept." in rendered
    # Closing strict-JSON directive (Wave-113 hardening contract).
    assert "RESPOND ONLY WITH A JSON OBJECT" in rendered


def test_outline_user_prompt_includes_per_block_type_schema_directive(
    monkeypatch,
):
    """The structural-bounds block lists the per-block-type field
    bounds (e.g. concept exposes ``key_claims: (1, 5)`` /
    ``section_skeleton: (1, 3)``); per-type variations (assessment_item,
    prereq_set) inject their dedicated contract paragraph."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")

    concept_block = _stub_block(block_type="concept")
    concept_prompt = p._render_user_prompt(
        block=concept_block, source_chunks=[], objectives=[]
    )
    assert "key_claims" in concept_prompt
    assert "section_skeleton" in concept_prompt
    assert "summary_chars" in concept_prompt
    # The concept bounds report (1, 5) for key_claims per the constants.
    assert "(1, 5)" in concept_prompt

    assessment_block = _stub_block(
        block_type="assessment_item",
        block_id="page-1#assessment_item_q1_0",
    )
    assessment_prompt = p._render_user_prompt(
        block=assessment_block, source_chunks=[], objectives=[]
    )
    assert "Assessment item contract" in assessment_prompt
    assert "objective_refs verbatim" in assessment_prompt

    prereq_block = _stub_block(
        block_type="prereq_set",
        block_id="page-1#prereq_set_setup_0",
    )
    prereq_prompt = p._render_user_prompt(
        block=prereq_block, source_chunks=[], objectives=[]
    )
    assert "Prereq set contract" in prereq_prompt
    assert "prerequisitePages" in prereq_prompt


# ---------------------------------------------------------------------------
# Lenient JSON extraction
# ---------------------------------------------------------------------------


def test_lenient_json_extraction_recovers_from_markdown_fence(monkeypatch):
    """A response wrapped in ``` ```json ... ``` ``` markdown fences is
    recovered by :meth:`OpenAICompatibleClient._extract_json_lenient`
    so the outline tier accepts the payload on the first attempt."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _valid_outline_payload(block_type="concept")
    fenced = "```json\n" + json.dumps(payload) + "\n```"

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(fenced))

    p = OutlineProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _stub_block()
    out = p.generate_outline(block, source_chunks=[], objectives=[])
    assert isinstance(out, Block)
    assert isinstance(out.content, dict)
    assert out.content["block_id"] == payload["block_id"]
    assert len(seen) == 1, "lenient parse should accept on first attempt"


# ---------------------------------------------------------------------------
# Retry exhaustion
# ---------------------------------------------------------------------------


def test_outline_invalid_json_after_max_retries_raises_outline_exhausted(
    monkeypatch,
):
    """Every retry returns invalid JSON → ``MAX_PARSE_RETRIES`` dispatches
    happen, then the provider raises
    ``OutlineProviderError(code="outline_exhausted")``."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # No JSON anywhere in the response — lenient parse returns None.
        return httpx.Response(
            200, json=_success_body("not a json object at all")
        )

    p = OutlineProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _stub_block()
    with pytest.raises(OutlineProviderError) as excinfo:
        p.generate_outline(block, source_chunks=[], objectives=[])
    assert excinfo.value.code == "outline_exhausted"
    assert len(seen) == MAX_PARSE_RETRIES


def test_outline_validates_against_block_type_json_schema(monkeypatch):
    """A JSON object that omits the required ``key_claims`` field fails
    Schema validation; the parse-retry loop then exhausts and the
    provider raises ``outline_exhausted`` with the validation error in
    the message string. Verifies the per-block-type schema map
    (Subtask 19) is wired into the dispatch loop."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    invalid = _valid_outline_payload(block_type="concept")
    invalid.pop("key_claims")
    invalid_text = json.dumps(invalid)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(invalid_text))

    p = OutlineProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _stub_block()
    with pytest.raises(OutlineProviderError) as excinfo:
        p.generate_outline(block, source_chunks=[], objectives=[])
    assert excinfo.value.code == "outline_exhausted"
    # Cross-check: the schema map carries the concept entry the
    # provider validates against.
    schema = _BLOCK_TYPE_JSON_SCHEMAS["concept"]
    assert "key_claims" in schema["required"]


# ---------------------------------------------------------------------------
# Touch chain
# ---------------------------------------------------------------------------


def test_outline_appends_touch_with_tier_outline(monkeypatch):
    """A successful generation returns a Block carrying a single new
    ``Touch(tier="outline", purpose="draft", ...)`` appended to the
    input ``touched_by`` chain. Pre-existing touches are preserved."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _valid_outline_payload(block_type="concept")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(
        provider="local",
        client=_make_client(handler),
    )

    pre_touch = Touch(
        model="prior-tier-model",
        provider="local",
        tier="validation",
        timestamp="2026-05-02T00:00:00Z",
        decision_capture_id="in-memory:0",
        purpose="content_type",
    )
    block = Block(
        block_id="page-1#concept_intro_0",
        block_type="concept",
        page_id="page-1",
        sequence=0,
        content="",
        touched_by=(pre_touch,),
    )
    out = p.generate_outline(block, source_chunks=[], objectives=[])

    assert len(out.touched_by) == 2, "expected pre-existing + outline touches"
    assert out.touched_by[0] == pre_touch
    new_touch = out.touched_by[1]
    assert new_touch.tier == "outline"
    assert new_touch.purpose == "draft"
    assert new_touch.provider == "local"
    assert new_touch.model == p._model
    # Wave 112 invariant — decision_capture_id is ≥1 char.
    assert new_touch.decision_capture_id


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------


def test_outline_failure_emits_decision_event(monkeypatch):
    """A failed dispatch (every retry returns invalid JSON) still emits
    a single ``block_outline_call`` decision-capture event whose
    rationale interpolates per-call signals (block_id, block_type,
    provider, attempts) — required by the LLM call-site instrumentation
    contract in root ``CLAUDE.md``."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    capture = _FakeCapture()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("totally invalid"))

    p = OutlineProvider(
        provider="local",
        capture=capture,
        client=_make_client(handler),
    )
    block = _stub_block(block_id="page-1#concept_failed_0")
    with pytest.raises(OutlineProviderError):
        p.generate_outline(block, source_chunks=[], objectives=[])

    assert len(capture.events) == 1, (
        "expected exactly one block_outline_call event on failure"
    )
    event = capture.events[0]
    assert event["decision_type"] == "block_outline_call"
    rationale = event["rationale"]
    # Wave-12 contract — rationale ≥ 20 chars + dynamic per-call signals.
    assert len(rationale) >= 20
    assert "block_id=page-1#concept_failed_0" in rationale
    assert "block_type=concept" in rationale
    assert "provider=local" in rationale
    assert "success=False" in rationale
    # Captured the attempts count (>= 1; up to MAX_PARSE_RETRIES).
    assert "attempts=" in rationale


def test_outline_success_emits_decision_event(monkeypatch):
    """A successful generation emits a single
    ``block_outline_call`` event tagged ``success=True`` whose
    rationale carries the chosen model + retry_count."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    capture = _FakeCapture()
    payload = _valid_outline_payload(block_type="concept")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(
        provider="local",
        capture=capture,
        client=_make_client(handler),
    )
    block = _stub_block()
    p.generate_outline(block, source_chunks=[], objectives=[])

    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "block_outline_call"
    rationale = event["rationale"]
    assert "success=True" in rationale
    assert f"model={p._model}" in rationale
    assert "retry_count=" in rationale
