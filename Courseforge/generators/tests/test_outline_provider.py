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
import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outline_provider import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENV_MAX_TOKENS,
    ENV_PROVIDER,
    MAX_PARSE_RETRIES,
    OutlineProvider,
    OutlineProviderError,
    SUPPORTED_PROVIDERS,
    _BLOCK_TYPE_JSON_SCHEMAS,
    _DEFAULT_MAX_TOKENS,
    _resolve_outline_max_tokens,
    _OBJECTIVE_ECHO_WARNING,
    _OUTLINE_KIND_BOUNDS,
    _OUTLINE_SYSTEM_PROMPT,
    _block_source_chunk_ids,
    _build_block_outline_schema,
    _drop_objective_echo_claims,
    _match_retry_directive,
    _surface_key_claims_min_items,
    _repair_assessment_item_payload,
    _repair_claim_grounding,
    _repair_outline_bloom_level,
    _repair_outline_curies,
    _repair_outline_key_claims_shape,
    _repair_outline_source_refs,
    _repair_prereq_pages,
    _extract_prereq_phrases_from_source,
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
            # outline-overflow-fix-2026-07: report a realistic served
            # prompt-token count (a properly-served model echoes the true
            # count, ~2k+ for the outline system prompt + user prompt) so
            # the default-ON input-truncation tripwire
            # (COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE) does not false-fire
            # on this stub. A too-low count here is indistinguishable from
            # a real silent head-truncation.
            "prompt_tokens": 8000,
            "completion_tokens": 80,
            "total_tokens": 8080,
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
        "source_refs": [{"sourceId": "semantik:slug#blk1", "role": "primary"}],
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


def test_supported_providers_is_registry_superset():
    """The module-level constant is the registry-superset allow-list:
    ``anthropic`` + every ``kind: openai_compatible`` registry endpoint
    (``local`` / ``together`` / ``nvidia`` + any future cloud row), PLUS the
    legacy ``openai_compatible`` alias. Adding a provider is a registry-entry
    change, never a subclass."""
    from lib.llm.endpoints import load_endpoint_registry

    s = set(SUPPORTED_PROVIDERS)
    assert {"anthropic", "together", "local", "nvidia"}.issubset(s)
    assert "openai_compatible" in s
    registry_seats = {
        name
        for name, row in load_endpoint_registry().items()
        if str(row.get("kind")) == "openai_compatible"
    }
    assert registry_seats.issubset(s)


def test_registry_seat_constructs_and_stamps_valid_touch(monkeypatch):
    """A non-legacy registry endpoint (``groq``) that the router allowlist
    admits now constructs the outline tier without a ValueError, and its
    Touch provenance collapses to the seat's registry ``provenance_provider``
    (``groq`` → ``together``) so Touch validation passes."""
    from Courseforge.generators._outline_provider import _touch_provenance

    monkeypatch.setenv("TOGETHER_API_KEY", "tk")
    p = OutlineProvider(
        provider="groq",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("{}"))
        ),
    )
    assert p._provider == "groq"
    assert _touch_provenance("groq") == "together"


def test_openai_compatible_alias_collapses_to_local(monkeypatch):
    """The legacy ``openai_compatible`` alias collapses to ``local`` at
    constructor entry so a standalone construction matches the router-mediated
    one (and never hits the base's UnknownEndpoint branch)."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(
        provider="openai_compatible",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("{}"))
        ),
    )
    assert p._provider == "local"


def test_unknown_provider_still_fails_fast():
    """An unknown provider name is still rejected at construction (fail-fast),
    now via the base's registry-derived allow-list ValueError."""
    with pytest.raises(ValueError):
        OutlineProvider(
            provider="definitely-not-a-provider",
            client=_make_client(
                lambda r: httpx.Response(200, json=_success_body("{}"))
            ),
        )


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
        {"id": "semantik:slug#blk1", "body": "Source body content."},
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
        p.generate_outline(
            block,
            source_chunks=[
                {"id": "chunk-alpha", "body": "Alpha grounding text."},
            ],
            objectives=[
                {"id": "CO-01", "statement": "Explain alpha."},
            ],
        )

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
    # Capture-quality contract (proficient floor): the event references
    # the REAL inputs the call consumed — the block plus the supplied
    # source-chunk / objective universes — and the genuine retry /
    # escalation alternatives the parse-retry loop weighed.
    inputs_ref = event["inputs_ref"]
    assert {
        "source_type": "block",
        "path_or_id": "page-1#concept_failed_0",
    } in inputs_ref
    assert {
        "source_type": "source_chunk",
        "path_or_id": "chunk-alpha",
    } in inputs_ref
    assert {
        "source_type": "learning_objective",
        "path_or_id": "CO-01",
    } in inputs_ref
    alternatives = event["alternatives_considered"]
    assert alternatives, "failure event must carry genuine alternatives"
    assert any("accept the attempt-" in a for a in alternatives)


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
    # Capture-quality contract (proficient floor): even with an empty
    # chunk/objective universe the event references the real block input
    # and the genuine retry/escalation alternatives the loop weighed —
    # never empty inputs_ref AND empty alternatives (that rates
    # "developing" and excludes the row from the training corpus).
    inputs_ref = event["inputs_ref"]
    assert {
        "source_type": "block",
        "path_or_id": block.block_id,
    } in inputs_ref
    alternatives = event["alternatives_considered"]
    assert alternatives, "success event must carry genuine alternatives"
    assert any("re-dispatch" in a for a in alternatives)
    assert any("outline_exhausted" in a for a in alternatives)


# ---------------------------------------------------------------------------
# Wave 1.5 W1.5.B — per-claim source attribution prompt directives
# ---------------------------------------------------------------------------


def test_outline_system_prompt_carries_structured_key_claims_directive():
    """Wave 1.5 W1.5.B golden-output regression — system prompt sentinel.

    The new authoring directive lands as a literal substring in the
    system prompt so downstream model traffic is steered toward the
    structured ``List[{claim, source_chunk_ids[]}]`` shape. Snapshot-
    style substring assertion (not full string equality) so the
    surrounding prompt text can evolve."""
    assert "key_claims MUST be a list of objects" in _OUTLINE_SYSTEM_PROMPT
    # Cross-check that the structured-shape signature is enumerated
    # verbatim — catches a regression that keeps the sentinel but
    # drops the actual contract paragraph.
    assert "source_chunk_ids" in _OUTLINE_SYSTEM_PROMPT
    assert "source_refs[]" in _OUTLINE_SYSTEM_PROMPT


def test_outline_system_prompt_drops_legacy_string_shape_as_primary():
    """Wave 1.5 W1.5.B negative regression — the system prompt's
    ``key_claims`` paragraph names the structured shape as the
    contract and only references the legacy flat-string shape under
    a PROHIBITED clause. Ensures we don't accidentally reintroduce
    the legacy shape as authoritative."""
    prompt = _OUTLINE_SYSTEM_PROMPT
    # Locate the key_claims paragraph by its sentinel.
    idx = prompt.index("key_claims MUST be a list of objects")
    # Tail of the prompt from the sentinel onward.
    tail = prompt[idx:]
    # The tail MUST flag the flat-string shape as PROHIBITED, not
    # describe it as the primary shape.
    assert "PROHIBITED" in tail
    assert "flat array of strings" in tail


def test_outline_user_prompt_carries_per_claim_attribution_clause(monkeypatch):
    """Wave 1.5 W1.5.B golden-output regression — user-prompt sentinel.

    The closing clause that names the supplied source_chunks as the
    universe ``source_chunk_ids`` may draw from MUST appear in every
    rendered user prompt so the outline-tier model has a concrete
    chunk-id universe bound."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="concept")
    chunks = [
        {"id": "semantik:slug-a#blk1", "body": "First chunk body."},
        {"id": "semantik:slug-b#blk2", "body": "Second chunk body."},
    ]
    objectives = [
        {"id": "TO-01", "statement": "Define the central concept."},
    ]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    assert "source_chunk_ids MUST be a non-empty subset" in rendered
    # The closing clause references the "Source chunks" section
    # already enumerated above so the universe is unambiguous.
    assert "Source chunks" in rendered


# ---------------------------------------------------------------------------
# CB5b — example-block concrete-worked-instance directive
# ---------------------------------------------------------------------------


def test_outline_system_prompt_carries_example_concrete_instance_directive():
    """CB5b system-prompt sentinel: the global ``_OUTLINE_SYSTEM_PROMPT``
    instructs the model that an ``example`` block's ``key_claims`` MUST
    carry a CONCRETE WORKED INSTANCE (problem + steps + answer) drawn
    from a worked Example / TRY IT item, not just the general rule.

    Substring assertion (not full string equality) so the surrounding
    prompt can evolve."""
    prompt = _OUTLINE_SYSTEM_PROMPT
    assert "CONCRETE WORKED INSTANCE" in prompt
    assert "`Example` or `TRY IT`" in prompt
    # The directive must explicitly forbid the abstract-rule-only shape.
    assert "Do NOT state only the general rule or formula" in prompt


def test_outline_user_prompt_example_block_carries_concrete_instance_directive(
    monkeypatch,
):
    """CB5b golden-output regression — for a ``block_type="example"``
    block the rendered user prompt carries the per-block
    concrete-worked-instance contract (the type-specific variation block
    with recency bias). The directive names the SPECIFIC problem /
    steps / answer the rewriter needs."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="example", block_id="page-1#example_div_0")
    chunks = [
        {
            "id": "semantik:frac#blk1",
            "body": (
                "Example: Divide 3/4 by 2/5. Multiply by the reciprocal: "
                "3/4 x 5/2 = 15/8. The answer is 15/8."
            ),
        },
    ]
    objectives = [
        {"id": "TO-01", "statement": "Divide fractions."},
    ]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    assert "Example block contract" in rendered
    assert "CONCRETE WORKED INSTANCE" in rendered
    # Must explicitly forbid the abstract-rule-only failure mode.
    assert "Do NOT emit only the general rule or formula" in rendered


def test_outline_user_prompt_non_example_block_omits_concrete_instance_directive(
    monkeypatch,
):
    """CB5b negative regression — a NON-example block (concept) does NOT
    receive the example-specific per-block concrete-worked-instance
    variation. Guards against the directive leaking onto every block."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="concept")
    chunks = [{"id": "semantik:slug#blk1", "body": "A concept body."}]
    objectives = [{"id": "TO-01", "statement": "Define the concept."}]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    # The per-block variation directive is example-only.
    assert "Example block contract" not in rendered


def test_resolve_outline_max_tokens_default_when_env_unset(monkeypatch):
    """big-model-overflow-fix-2026-07 — env unset resolves to the 4096
    default (bumped from the legacy 7B-calibrated 1200)."""
    monkeypatch.delenv(ENV_MAX_TOKENS, raising=False)
    assert _DEFAULT_MAX_TOKENS == 4096
    assert _resolve_outline_max_tokens(None) == 4096


def test_resolve_outline_max_tokens_env_positive_int(monkeypatch):
    """A positive-int env value is honored."""
    monkeypatch.setenv(ENV_MAX_TOKENS, "8192")
    assert _resolve_outline_max_tokens(None) == 8192


@pytest.mark.parametrize("bad", ["", "  ", "not-an-int", "0", "-5", "3.5"])
def test_resolve_outline_max_tokens_garbage_falls_back(monkeypatch, bad):
    """Garbage / non-positive env → the 4096 default (parse-with-fallback;
    a misconfigured knob never shrinks the cap)."""
    monkeypatch.setenv(ENV_MAX_TOKENS, bad)
    assert _resolve_outline_max_tokens(None) == 4096


def test_resolve_outline_max_tokens_kwarg_wins_over_env(monkeypatch):
    """Per-call kwarg beats the env (kwarg > env > default)."""
    monkeypatch.setenv(ENV_MAX_TOKENS, "8192")
    assert _resolve_outline_max_tokens(1500) == 1500


def test_outline_provider_threads_resolved_max_tokens(monkeypatch):
    """The resolved cap reaches ``OutlineProvider._max_tokens`` (env path)
    and an explicit kwarg still wins."""
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv(ENV_MAX_TOKENS, "6000")
    p = OutlineProvider(provider="local")
    assert p._max_tokens == 6000
    p2 = OutlineProvider(provider="local", max_tokens=1234)
    assert p2._max_tokens == 1234


def test_outline_kind_bounds_lowered_floors_accept_single_claim():
    """big-model-overflow-fix-2026-07 — the ``key_claims`` MINIMUM is lowered
    from 2 to 1 for the six dense-block types a large reasoning seat tends to
    author as ONE dense, well-cited claim (max unchanged). A single-claim
    structured block must now validate for each."""
    lowered = {
        "example": (1, 4),
        "explanation": (1, 6),
        "table": (1, 8),
        "flip_card_grid": (1, 8),
        "acronym": (1, 8),
        "worked_example": (1, 6),
    }
    for block_type, (lo, hi) in lowered.items():
        assert _OUTLINE_KIND_BOUNDS[block_type]["key_claims"] == (lo, hi)
        schema = _build_block_outline_schema(block_type)
        key_claims_schema = schema["properties"]["key_claims"]
        for arm in key_claims_schema["oneOf"]:
            assert arm["minItems"] == 1
            assert arm["maxItems"] == hi
        # A single well-formed structured claim now validates against the
        # key_claims subschema (was rejected under the old floor of 2).
        single_structured = [
            {"claim": "one dense fact", "source_chunk_ids": ["c1"]}
        ]
        jsonschema.Draft202012Validator(key_claims_schema).validate(
            single_structured
        )
        # A single flat-string claim (legacy arm) validates too.
        jsonschema.Draft202012Validator(key_claims_schema).validate(
            ["one dense fact"]
        )


def test_outline_retry_directive_matches_oneof_validation_error():
    """Wave 1.5 W1.5.B retry-directive regression — when the validator
    surfaces ``is not valid under any of the given schemas`` (the
    canonical ``oneOf`` rejection error), ``_match_retry_directive``
    returns the new per-claim attribution recovery directive."""
    err = (
        "[\"flat string claim\", \"another flat string\"] is not valid "
        "under any of the given schemas"
    )
    directive = _match_retry_directive(err)
    assert directive is not None
    # The new directive names the structured shape verbatim so the
    # next parse-retry sees the contract.
    assert "key_claims MUST be a list of objects" in directive
    assert "source_chunk_ids" in directive
    assert "Do NOT emit key_claims as flat strings" in directive
    # The directive cross-references the block-level source_refs[]
    # superset constraint.
    assert "source_refs[]" in directive


# ---------------------------------------------------------------------------
# improvement-map Step 3 (concern #2) — single-claim minItems escalation
# recovery: surface the masked minItems sub-error, match the dedicated
# minItems directive (not W1.5.B), interpolate the per-type min, and keep
# the minItems directive ordered before the W1.5.B oneOf catch-all.
# ---------------------------------------------------------------------------


def test_surface_key_claims_min_items_unmasks_oneof_suberror():
    """A single well-formed structured claim against a ``diagram``
    block (``key_claims`` minItems=2 — one of the types that RETAINS the
    2-floor after the big-model-overflow fix) raises the masking top-level
    ``oneOf`` error; ``_surface_key_claims_min_items`` walks ``exc.context``
    and returns the structured arm's real "too short" sub-error so the
    true cause (NOT the bare oneOf) drives the next retry."""
    schema = _build_block_outline_schema("diagram")
    candidate = {
        "block_id": "b1",
        "block_type": "diagram",
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["CO-01"],
        "curies": [],
        # Exactly ONE structured claim — well-formed, but below the
        # diagram block's key_claims minItems=2 bound.
        "key_claims": [{"claim": "one idea", "source_chunk_ids": ["c1"]}],
        "section_skeleton": [{"h": "x"}],
        "source_refs": [],
        "structural_warnings": [],
    }
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.Draft202012Validator(schema).validate(candidate)
    exc = excinfo.value
    # The bare top-level message is the masking oneOf rejection.
    assert "is not valid under any of the given schemas" in str(exc.message)
    # The surfacing helper unmasks the real minItems cause.
    surfaced = _surface_key_claims_min_items(exc)
    assert surfaced is not None
    assert surfaced.startswith("key_claims:")
    assert "too short" in surfaced.lower()


def test_surface_key_claims_min_items_ignores_non_key_claims_oneof():
    """The surfacing helper is scoped to the ``key_claims`` ``oneOf`` —
    a non-key_claims validation error returns ``None`` (message-building
    only; never alters unrelated validation paths)."""
    schema = _build_block_outline_schema("explanation")
    candidate = {
        # block_id violates minLength — a non-key_claims error.
        "block_id": "",
        "block_type": "explanation",
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["CO-01"],
        "curies": [],
        "key_claims": [
            {"claim": "a", "source_chunk_ids": ["c1"]},
            {"claim": "b", "source_chunk_ids": ["c1"]},
        ],
        "section_skeleton": [{"h": "x"}],
        "source_refs": [],
        "structural_warnings": [],
    }
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.Draft202012Validator(schema).validate(candidate)
    assert _surface_key_claims_min_items(excinfo.value) is None


def test_match_retry_directive_returns_min_items_directive_for_diagram():
    """With the surfaced ``key_claims: ... is too short`` sub-error,
    ``_match_retry_directive`` returns the dedicated minItems decomposition
    directive (NOT the W1.5.B "emit objects not flat strings" directive),
    with the per-type min interpolated to 2 for ``diagram`` (a type that
    RETAINS the 2-floor after the big-model-overflow fix)."""
    err = (
        "key_claims: [{'claim': 'one idea'}] is too short | "
        "[...] is not valid under any of the given schemas"
    )
    directive = _match_retry_directive(err, "diagram")
    assert directive is not None
    # The minItems directive — NOT the W1.5.B oneOf catch-all.
    assert "key_claims has too FEW entries" in directive
    assert "MUST be a list of objects" not in directive  # W1.5.B signature
    # The per-type min (diagram -> 2) is interpolated; the {{...}}
    # escape collapses to a literal {claim, source_chunk_ids} shape hint.
    assert "at least 2 distinct" in directive
    assert "{claim, source_chunk_ids}" in directive
    # Anti-fabrication steer toward decomposition, not padding.
    assert "Do NOT fabricate" in directive
    assert "decomposing" in directive
    # Confirm the resolved min tracks the canonical bound for the type.
    assert _OUTLINE_KIND_BOUNDS["diagram"]["key_claims"][0] == 2


def test_min_items_directive_ordered_before_w15b_oneof_catchall():
    """Directive ordering — the surfaced minItems error must match the
    minItems directive FIRST. A bare oneOf error (no surfaced sub-error)
    still falls through to the W1.5.B catch-all, and the W7 distractors
    "too short" path is NOT stolen by the key_claims-scoped pattern."""
    # Surfaced minItems error -> minItems directive (first match wins).
    surfaced_err = "key_claims: [...] is too short"
    assert "too FEW entries" in _match_retry_directive(surfaced_err, "explanation")
    # Bare oneOf (no surfaced sub-error) -> W1.5.B catch-all unchanged.
    bare_oneof = "[\"a\", \"b\"] is not valid under any of the given schemas"
    w15b = _match_retry_directive(bare_oneof, "explanation")
    assert "MUST be a list of objects" in w15b
    assert "too FEW entries" not in w15b
    # W7 assessment_item distractors "too short" -> assessment directive
    # (NOT the key_claims-scoped minItems directive — the pattern requires
    # a ``key_claims:`` marker which the distractors error lacks).
    distractors_err = "distractors [] is too short"
    w7 = _match_retry_directive(distractors_err, "assessment_item")
    assert "distractors" in w7
    assert "too FEW entries" not in w7


# ---------------------------------------------------------------------------
# Wave 1.7 W1.7.B — Bloom-triple objective rendering, behavioral-outcome
# system-prompt floor, and BLOCK_OBJECTIVE_BLOOM_UNDERMET retry directive
# ---------------------------------------------------------------------------


def test_outline_user_prompt_surfaces_bloom_triple_for_dict_shape(monkeypatch):
    """Wave 1.7 W1.7.B golden-output regression (outline-tier): the
    rendered user prompt for a fixed (block, source_chunks, objectives)
    triple includes the Bloom triple ``[Bloom: <level>, verb: <verb>]``
    verbatim for at least one objective when ``bloom_level`` /
    ``bloom_verb`` are present on the objective dict."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="concept")
    chunks = [
        {"id": "semantik:slug-a#blk1", "body": "First chunk body."},
    ]
    objectives = [
        {
            "id": "CO-08",
            "statement": "Construct subclass and subproperty hierarchies in Turtle.",
            "bloom_level": "create",
            "bloom_verb": "construct",
        },
    ]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    assert "CO-08" in rendered
    assert "[Bloom: create, verb: construct]" in rendered
    assert "Construct subclass and subproperty hierarchies in Turtle." in rendered


def test_outline_user_prompt_falls_back_to_legacy_shape_when_bloom_absent(monkeypatch):
    """Back-compat: legacy fixtures that don't carry ``bloom_level`` /
    ``bloom_verb`` on the objective dict still render unambiguously
    via the legacy ``- {oid}: {stmt}`` shape (no bracketed Bloom
    triple). Pre-Wave-1.7 corpora must not see a regression."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="concept")
    chunks = [
        {"id": "semantik:slug-a#blk1", "body": "First chunk body."},
    ]
    objectives = [
        {"id": "TO-01", "statement": "Define the central concept."},
    ]
    rendered = p._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    # Legacy shape rendered verbatim; no bracketed Bloom triple emitted.
    assert "- TO-01: Define the central concept." in rendered
    assert "[Bloom:" not in rendered


def test_outline_system_prompt_carries_bloom_floor_directive():
    """Wave 1.7 W1.7.B system-prompt sentinel: ``_OUTLINE_SYSTEM_PROMPT``
    must carry the ``MUST be at or above the declared Bloom`` substring
    so the outline-tier model is steered toward emitting a
    ``bloom_level`` at or above the declared Bloom level of the
    objective(s) listed in ``objective_refs``."""
    assert "MUST be at or above the declared Bloom" in _OUTLINE_SYSTEM_PROMPT
    # Cross-checks: the directive paragraph names the ``create``-level
    # explicit prohibition so the model has a concrete enum example.
    assert "create" in _OUTLINE_SYSTEM_PROMPT
    # The distribute-across-blocks clause names the canonical
    # `concept` / `example` / `assessment_item` Bloom-tier ladder.
    assert "distribute the Bloom levels" in _OUTLINE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Wave 2 — validated-id-fallback grounding repair (empty source_chunk_ids)
# ---------------------------------------------------------------------------


def _structured_claim_payload(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
    source_chunk_ids: Any,
) -> Dict[str, Any]:
    """A valid outline payload but with STRUCTURED key_claims carrying the
    supplied (possibly empty / garbage) ``source_chunk_ids`` — the 7B
    failure shape the repair targets."""
    payload = _valid_outline_payload(block_type=block_type, block_id=block_id)
    payload["key_claims"] = [
        {
            "claim": "Apply divisibility tests to classify integers.",
            "source_chunk_ids": source_chunk_ids,
        }
    ]
    return payload


def test_grounding_repair_populates_empty_source_chunk_ids_from_block(
    monkeypatch,
):
    """Wave 2 regression — the production-bug repro.

    A model response whose ``key_claims[].source_chunk_ids`` is EMPTY
    (the live-run failure) used to fail the strict outline schema
    (``minItems: 1``) and exhaust the retry budget. With the validated-id
    fallback, the outline tier now succeeds on the FIRST attempt: the
    claim comes back populated with the block's real chunk ids and the
    block does NOT exhaust retries."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _structured_claim_payload(source_chunk_ids=[])
    block_chunks = [
        {"id": "semantik:slug#blk1", "body": "Divisibility test body."},
        {"id": "semantik:slug#blk2", "body": "Worked-example body."},
    ]
    expected_ids = ["semantik:slug#blk1", "semantik:slug#blk2"]

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(provider="local", client=_make_client(handler))
    block = _stub_block()
    out = p.generate_outline(
        block, source_chunks=block_chunks, objectives=[]
    )

    # Did NOT exhaust retries — one dispatch, success.
    assert len(seen) == 1, "repair should make attempt-1 pass, not retry"
    assert isinstance(out.content, dict)
    claim = out.content["key_claims"][0]
    # Fallback populated the claim with the block's full source-chunk set.
    assert claim["source_chunk_ids"] == expected_ids
    # No-fabrication invariant: every returned id is a real block chunk.
    valid = set(expected_ids)
    assert all(cid in valid for cid in claim["source_chunk_ids"])
    # The transient repair-signal key must NOT leak into Block content.
    assert "_grounding_repair" not in out.content


def test_grounding_repair_filters_garbage_keeps_only_valid_ids(monkeypatch):
    """A claim citing a mix of a real id and a prose-string hallucination
    keeps only the real id (no fallback fires because ≥1 valid id
    remains); the prose string is dropped. No-fabrication invariant
    holds."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _structured_claim_payload(
        source_chunk_ids=[
            "semantik:slug#blk1",
            "Source chunk: a prose hallucination, not an id.",
            "semantik:slug#nonexistent",
        ]
    )
    block_chunks = [
        {"id": "semantik:slug#blk1", "body": "Real body."},
        {"id": "semantik:slug#blk2", "body": "Other body."},
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(provider="local", client=_make_client(handler))
    out = p.generate_outline(
        _stub_block(), source_chunks=block_chunks, objectives=[]
    )
    claim = out.content["key_claims"][0]
    assert claim["source_chunk_ids"] == ["semantik:slug#blk1"]
    valid = {"semantik:slug#blk1", "semantik:slug#blk2"}
    assert all(cid in valid for cid in claim["source_chunk_ids"])


def test_grounding_repair_emits_signals_in_decision_capture(monkeypatch):
    """The repair folds its signals into the existing
    ``block_outline_call`` decision event (the canonical capture at this
    call site): the rationale carries the dynamic repair signals."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    capture = _FakeCapture()
    payload = _structured_claim_payload(source_chunk_ids=[])
    block_chunks = [{"id": "semantik:slug#blk1", "body": "Body."}]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(
        provider="local", capture=capture, client=_make_client(handler)
    )
    p.generate_outline(_stub_block(), source_chunks=block_chunks, objectives=[])

    assert len(capture.events) == 1
    rationale = capture.events[0]["rationale"]
    assert "grounding_repair=" in rationale
    assert "repaired=True" in rationale
    assert "n_fallback_claims=1" in rationale


def test_grounding_repair_no_source_chunks_stays_fail_closed(monkeypatch):
    """Empty-corpus safety: a block with NO source chunks must NOT have
    provenance invented. With empty ``source_chunks`` the repair no-ops,
    so the strict schema (``minItems: 1``) still rejects the empty
    ``source_chunk_ids`` and the block exhausts retries — the existing
    fail-closed behavior is preserved (no fabrication)."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _structured_claim_payload(source_chunk_ids=[])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    p = OutlineProvider(provider="local", client=_make_client(handler))
    with pytest.raises(OutlineProviderError) as excinfo:
        # No source chunks supplied → repair never fabricates an id.
        p.generate_outline(_stub_block(), source_chunks=[], objectives=[])
    assert excinfo.value.code == "outline_exhausted"


def test_repair_helper_unit_no_fabrication_invariant():
    """Unit-level: ``_repair_claim_grounding`` only ever emits ids drawn
    from the supplied ``valid_ids`` set, never invents one, and leaves a
    legacy flat-string ``key_claims`` array untouched."""
    valid = ["c1", "c2"]
    # Empty citations → fallback to the full valid set.
    cand = {"key_claims": [{"claim": "x", "source_chunk_ids": []}]}
    out = _repair_claim_grounding(cand, valid_ids=valid)
    assert out["key_claims"][0]["source_chunk_ids"] == ["c1", "c2"]
    assert out["_grounding_repair"]["n_fallback_claims"] == 1
    # No valid_ids → no-op (never fabricate).
    cand2 = {"key_claims": [{"claim": "x", "source_chunk_ids": []}]}
    out2 = _repair_claim_grounding(cand2, valid_ids=[])
    assert out2["key_claims"][0]["source_chunk_ids"] == []
    assert out2["_grounding_repair"]["repaired"] is False
    # Legacy flat-string arm is left untouched.
    cand3 = {"key_claims": ["a flat string claim"]}
    out3 = _repair_claim_grounding(cand3, valid_ids=valid)
    assert out3["key_claims"] == ["a flat string claim"]


def test_block_source_chunk_ids_honours_id_and_chunk_id():
    """``_block_source_chunk_ids`` mirrors the prompt's id extraction:
    ``id`` (canonical) or ``chunk_id`` (legacy), de-duped, order-preserved."""
    chunks = [
        {"id": "c1", "body": "x"},
        {"chunk_id": "c2", "text": "y"},
        {"id": "c1", "body": "dup"},
        {"body": "no id at all"},
    ]
    assert _block_source_chunk_ids(chunks) == ["c1", "c2"]


def test_outline_retry_directive_matches_block_objective_bloom_undermet():
    """Wave 1.7 W1.7.B retry-directive regression — when the validator
    (lands in W1.7.C) surfaces ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` as
    the GateIssue code, ``_match_retry_directive`` returns the new
    Bloom-floor recovery directive so the next outline-tier prompt
    suffix steers the model up to the declared cognitive demand."""
    err = (
        "BLOCK_OBJECTIVE_BLOOM_UNDERMET: block bloom_level 'remember' "
        "is below objective TO-04 declared level 'create'."
    )
    directive = _match_retry_directive(err)
    assert directive is not None
    # The new directive names the bloom_level surface verbatim so the
    # next parse-retry sees the contract.
    assert "bloom_level is below the declared Bloom level" in directive
    assert "Re-emit with bloom_level at or above" in directive
    # The directive references the prose-side scaffolding obligation.
    assert "scaffold up" in directive


# ---------------------------------------------------------------------------
# Malformed-output normalization repairs (live-7B failure-mode pre-validation
# repairs): mixed key_claims arrays + URL-CURIEs.
# ---------------------------------------------------------------------------


def _valid_explanation_outline() -> Dict[str, Any]:
    """A schema-clean ``explanation`` outline payload.

    explanation bounds: key_claims (2, 6), section_skeleton (1, 4). Used as
    the base into which the two malformed shapes are injected so the tests
    assert the repair turns a REJECTED payload into an ACCEPTED one.
    """
    return {
        "block_id": "page_01#explanation_intro_0",
        "block_type": "explanation",
        "content_type": sorted(__import__("lib.validators.content_type",
                                          fromlist=["get_valid_chunk_types"])
                               .get_valid_chunk_types())[0],
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": [],
        "key_claims": [
            {"claim": "First real claim about the topic.",
             "source_chunk_ids": ["chunk_a"]},
            {"claim": "Second real claim about the topic.",
             "source_chunk_ids": ["chunk_a"]},
        ],
        "section_skeleton": [{"heading": "Overview"}],
        "source_refs": [
            {"sourceId": "semantik:slug#chunk_a", "role": "primary"},
        ],
        "structural_warnings": [],
    }


def _validate_explanation(payload: Dict[str, Any]) -> None:
    """Run the strict explanation schema validator; raises on failure."""
    import jsonschema  # type: ignore[import-untyped]

    schema = _BLOCK_TYPE_JSON_SCHEMAS["explanation"]
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_mixed_key_claims_rejected_before_repair():
    """Baseline: a mixed key_claims array (object claims + a stray bare
    chunk-id string) fails the strict explanation schema — this is the
    exact live-7B ``'<chunk-id>' is not of type 'object'`` failure."""
    import jsonschema  # type: ignore[import-untyped]

    payload = _valid_explanation_outline()
    # Inject the live failure: a bare chunk-id string appended to the
    # otherwise all-object key_claims array.
    payload["key_claims"].append("democourse_chunk_00001")
    with pytest.raises(jsonschema.ValidationError):
        _validate_explanation(payload)


def test_repair_mixed_key_claims_passes_explanation_schema():
    """After ``_repair_outline_key_claims_shape``, a mixed key_claims array
    coerces to the all-object arm and PASSES the strict explanation schema.
    The stray bare chunk-id (which names a real block chunk) is DROPPED;
    no claim is fabricated."""
    payload = _valid_explanation_outline()
    stray_id = "democourse_chunk_00001"
    payload["key_claims"].append(stray_id)
    valid_ids = ["chunk_a", stray_id]

    out = _repair_outline_key_claims_shape(payload, valid_ids=valid_ids)
    meta = out.pop("_key_claims_shape_repair")
    # The grounding repair runs next in the real call path; run it here to
    # mirror the production ordering (object claims keep their ids).
    out = _repair_claim_grounding(out, valid_ids=valid_ids)
    out.pop("_grounding_repair", None)

    # The stray chunk-id string was dropped; only the two real object
    # claims survive — no fabrication, no padding.
    assert meta["repaired"] is True
    assert meta["n_dropped_chunk_id_strings"] == 1
    assert meta["n_wrapped_prose_strings"] == 0
    assert len(out["key_claims"]) == 2
    assert all(isinstance(c, dict) for c in out["key_claims"])
    # The dropped value never reappears anywhere in key_claims.
    assert all(stray_id not in (c.get("claim") or "")
               for c in out["key_claims"])

    _validate_explanation(out)  # must not raise


def test_repair_mixed_key_claims_wraps_genuine_prose_string():
    """A bare string that is NOT a chunk-id is a genuine prose claim under
    the legacy flat arm; it is WRAPPED into an object (not dropped, not
    fabricated). source_chunk_ids is then populated by the grounding
    fallback from the block's real ids."""
    payload = _valid_explanation_outline()
    prose = "A genuinely useful third claim emitted under the flat arm."
    payload["key_claims"].append(prose)
    valid_ids = ["chunk_a"]

    out = _repair_outline_key_claims_shape(payload, valid_ids=valid_ids)
    meta = out.pop("_key_claims_shape_repair")
    out = _repair_claim_grounding(out, valid_ids=valid_ids)
    out.pop("_grounding_repair", None)

    assert meta["repaired"] is True
    assert meta["n_dropped_chunk_id_strings"] == 0
    assert meta["n_wrapped_prose_strings"] == 1
    assert len(out["key_claims"]) == 3
    wrapped = out["key_claims"][-1]
    assert isinstance(wrapped, dict)
    assert wrapped["claim"] == prose
    # Grounding fallback populated source_chunk_ids from the block's
    # real ids — never fabricated outside valid_ids.
    assert wrapped["source_chunk_ids"] == ["chunk_a"]
    _validate_explanation(out)


def test_repair_key_claims_noop_on_homogeneous_object_array():
    """A clean all-object key_claims array is untouched (no repair fires)."""
    payload = _valid_explanation_outline()
    before = json.dumps(payload["key_claims"], sort_keys=True)
    out = _repair_outline_key_claims_shape(payload, valid_ids=["chunk_a"])
    meta = out.pop("_key_claims_shape_repair")
    assert meta["repaired"] is False
    assert json.dumps(out["key_claims"], sort_keys=True) == before


def test_url_curie_rejected_before_repair():
    """Baseline: a URL-CURIE in the curies array fails the strict schema —
    the exact live-7B ``does not match '^[a-z]...'`` failure."""
    import jsonschema  # type: ignore[import-untyped]

    payload = _valid_explanation_outline()
    payload["curies"] = ["foo:bar", "schema:https://example.com/vocab"]
    with pytest.raises(jsonschema.ValidationError):
        _validate_explanation(payload)


def test_repair_curies_drops_url_curie_keeps_valid():
    """After ``_repair_outline_curies``, the valid CURIE survives, the
    URL-CURIE is dropped, and the payload PASSES the strict schema. No
    CURIE is fabricated or rewritten."""
    payload = _valid_explanation_outline()
    payload["curies"] = ["foo:bar", "schema:https://example.com/vocab"]

    out = _repair_outline_curies(payload)
    meta = out.pop("_curie_shape_repair")

    assert meta["repaired"] is True
    assert meta["n_dropped_malformed"] == 1
    assert meta["n_kept"] == 1
    # Valid CURIE survives verbatim; malformed one is gone. Nothing new
    # was invented.
    assert out["curies"] == ["foo:bar"]
    assert "schema:https://example.com/vocab" not in out["curies"]

    _validate_explanation(out)  # must not raise


def test_repair_curies_empty_array_is_schema_valid():
    """Dropping every malformed CURIE leaves an empty array, which is
    schema-valid (curies is not minItems-bound for explanation)."""
    payload = _valid_explanation_outline()
    payload["curies"] = ["schema:https://example.com/vocab"]
    out = _repair_outline_curies(payload)
    meta = out.pop("_curie_shape_repair")
    assert meta["n_kept"] == 0
    assert out["curies"] == []
    _validate_explanation(out)


def test_repair_curies_noop_when_all_valid():
    """All-valid curies array is untouched (no repair fires)."""
    payload = _valid_explanation_outline()
    payload["curies"] = ["foo:bar", "rdf:type"]
    out = _repair_outline_curies(payload)
    meta = out.pop("_curie_shape_repair")
    assert meta["repaired"] is False
    assert out["curies"] == ["foo:bar", "rdf:type"]


def test_combined_repairs_normalize_both_failures_end_to_end():
    """Both live failures present at once (mixed key_claims + URL-CURIE):
    the full pre-validation repair chain coerces both and the payload then
    PASSES the strict explanation schema — the end-to-end contract the
    outline-tier call site relies on."""
    payload = _valid_explanation_outline()
    stray_id = "democourse_chunk_00001"
    payload["key_claims"].append(stray_id)
    payload["curies"] = ["foo:bar", "schema:https://example.com/vocab"]
    valid_ids = ["chunk_a", stray_id]

    # Mirror the production ordering in generate_outline.
    payload = _repair_outline_key_claims_shape(payload, valid_ids=valid_ids)
    payload.pop("_key_claims_shape_repair", None)
    payload = _repair_outline_curies(payload)
    payload.pop("_curie_shape_repair", None)
    payload = _repair_claim_grounding(payload, valid_ids=valid_ids)
    payload.pop("_grounding_repair", None)

    assert payload["curies"] == ["foo:bar"]
    assert len(payload["key_claims"]) == 2
    assert all(isinstance(c, dict) for c in payload["key_claims"])
    _validate_explanation(payload)


# ---------------------------------------------------------------------------
# source_refs bare-string coercion + bloom_level numeric→enum coercion
# (two more live-7B wrong-primitive-shape failure modes).
# ---------------------------------------------------------------------------


def test_bare_source_refs_rejected_before_repair():
    """Baseline: bare-string source_refs items fail the strict schema —
    the exact live-7B ``'semantik:...' is not of type 'object'`` failure."""
    import jsonschema  # type: ignore[import-untyped]

    payload = _valid_explanation_outline()
    payload["source_refs"] = ["semantik:slug#chunk_a", "semantik:slug#chunk_b"]
    with pytest.raises(jsonschema.ValidationError):
        _validate_explanation(payload)


def test_repair_source_refs_wraps_bare_strings_and_backfills_role():
    """After ``_repair_outline_source_refs``, bare cited sourceId strings are
    wrapped with ``role="primary"`` (sourceId preserved verbatim), a
    role-less dict gets ``role="primary"``, and a no-sourceId item is
    dropped. The repaired payload PASSES the strict explanation schema. No
    sourceId is fabricated."""
    payload = _valid_explanation_outline()
    payload["source_refs"] = [
        "semantik:slug#chunk_a",                       # bare string → wrap
        {"sourceId": "semantik:slug#chunk_b"},         # role-less dict → backfill
        {"role": "contributing"},                  # no sourceId → drop
        "   ",                                      # whitespace-only → drop
    ]

    out = _repair_outline_source_refs(payload)
    meta = out.pop("_source_refs_shape_repair")

    assert meta["repaired"] is True
    assert meta["n_wrapped_bare_strings"] == 1
    assert meta["n_role_backfilled"] == 1
    assert meta["n_dropped_malformed"] == 2
    assert meta["n_kept"] == 2

    refs = out["source_refs"]
    assert refs == [
        {"sourceId": "semantik:slug#chunk_a", "role": "primary"},
        {"sourceId": "semantik:slug#chunk_b", "role": "primary"},
    ]
    # Every surviving sourceId is one the model actually cited — nothing
    # invented.
    assert {r["sourceId"] for r in refs} <= {
        "semantik:slug#chunk_a",
        "semantik:slug#chunk_b",
    }
    _validate_explanation(out)  # must not raise


def test_repair_source_refs_empty_array_is_schema_valid():
    """Dropping every unusable source_refs item leaves an empty array, which
    is schema-valid (source_refs is not minItems-bound for explanation)."""
    payload = _valid_explanation_outline()
    payload["source_refs"] = [{"role": "primary"}, 42, None, ""]
    out = _repair_outline_source_refs(payload)
    meta = out.pop("_source_refs_shape_repair")
    assert meta["n_kept"] == 0
    assert out["source_refs"] == []
    _validate_explanation(out)


def test_repair_source_refs_noop_when_all_valid():
    """A clean all-object source_refs array is untouched (no repair fires)."""
    payload = _valid_explanation_outline()
    before = json.dumps(payload["source_refs"], sort_keys=True)
    out = _repair_outline_source_refs(payload)
    meta = out.pop("_source_refs_shape_repair")
    assert meta["repaired"] is False
    assert json.dumps(out["source_refs"], sort_keys=True) == before


def test_repair_source_refs_noop_when_absent_or_not_list():
    """No-op when source_refs is absent or not a list (no crash)."""
    payload = {"foo": "bar"}
    out = _repair_outline_source_refs(payload)
    assert out["_source_refs_shape_repair"]["repaired"] is False
    out.pop("_source_refs_shape_repair")

    payload2 = {"source_refs": "semantik:slug#chunk_a"}
    out2 = _repair_outline_source_refs(payload2)
    assert out2["_source_refs_shape_repair"]["repaired"] is False
    assert out2["source_refs"] == "semantik:slug#chunk_a"


# ---------------------------------------------------------------------------
# prereq_set prerequisitePages repair (the 7B's exclusive empty-array failure)
# ---------------------------------------------------------------------------

# A source chunk that names its prerequisites in an explicit sentence — the
# exact clean-fractions shape the live 7B probe exercises.
_PREREQ_SOURCE = [
    {
        "id": "semantik:source-beta#sec_01",
        "text": (
            "To simplify a fraction, divide both the numerator and the "
            "denominator by their greatest common factor (GCF).\n\n"
            "Prerequisites: before simplifying fractions a learner must know "
            "how to list the factors of a whole number and how to find the "
            "greatest common factor (GCF) of two numbers.\n\n"
            "Try It: Simplify 20/30."
        ),
    }
]


def test_extract_prereq_phrases_from_explicit_sentence():
    """The Prerequisites sentence yields the two topic phrases (leading verb +
    article + 'how to' framing stripped), grounded in the source text."""
    phrases = _extract_prereq_phrases_from_source(_PREREQ_SOURCE)
    assert phrases == [
        "factors of a whole number",
        "greatest common factor (GCF) of two numbers",
    ]


def test_repair_prereq_pages_backfills_empty_from_source_and_drops_garbage():
    """The core fix: an empty `prerequisitePages` carrying page-id-shaped
    garbage is stripped AND backfilled with ≥1 sensible source-grounded
    string (the exact live-7B failure: `[]` + `'p#factors_x_0'` garbage)."""
    candidate = {
        "prerequisitePages": ["p#factors_x_0", "", 42],
    }
    out = _repair_prereq_pages(
        candidate,
        block_type="prereq_set",
        source_chunks=_PREREQ_SOURCE,
        key_terms=("fraction", "numerator"),
        objectives=[
            {"id": "CO-01", "statement": "Simplify a fraction by dividing the "
             "numerator and denominator by their GCF."}
        ],
    )
    meta = out.pop("_prereq_pages_repair")
    pages = out["prerequisitePages"]
    # Garbage stripped (page-id-shaped + empty + non-string).
    assert meta["n_garbage_stripped"] == 3
    # Backfilled with ≥1 sensible string from the source prerequisites.
    assert meta["repaired"] is True
    assert meta["backfill_source"] == "source_prerequisites"
    assert len(pages) >= 1
    assert "factors of a whole number" in pages
    # No page-id-shaped garbage survives.
    assert all(not p.startswith("p#") and "#" not in p for p in pages)
    # NEVER the objective statement.
    assert all("simplify a fraction" not in p.lower() for p in pages)


def test_repair_prereq_pages_missing_array_backfills():
    """A MISSING `prerequisitePages` (the model's other failure mode — emits
    None / omits the key) is backfilled the same way as the empty array."""
    out = _repair_prereq_pages(
        {"key_claims": [{"claim": "x", "source_chunk_ids": ["c"]}]},
        block_type="prereq_set",
        source_chunks=_PREREQ_SOURCE,
        key_terms=(),
        objectives=[],
    )
    assert len(out["prerequisitePages"]) >= 1
    assert out["_prereq_pages_repair"]["backfill_source"] == "source_prerequisites"


def test_repair_prereq_pages_falls_back_to_key_terms():
    """With no Prerequisites sentence in the source, backfill from the block's
    key_terms (a defensible prior-topic proxy) — never fabricated."""
    out = _repair_prereq_pages(
        {"prerequisitePages": []},
        block_type="prereq_set",
        source_chunks=[{"id": "x", "text": "Some prose with no markers."}],
        key_terms=("factoring", "prime numbers"),
        objectives=[],
    )
    assert out["prerequisitePages"] == ["factoring", "prime numbers"]
    assert out["_prereq_pages_repair"]["backfill_source"] == "key_terms"


def test_repair_prereq_pages_never_uses_objective_statement():
    """When the only available signal is the objective text, the array is LEFT
    empty (no source prereqs, no key_terms) — the objective is NEVER used to
    backfill (a known-bad 7B output)."""
    out = _repair_prereq_pages(
        {"prerequisitePages": []},
        block_type="prereq_set",
        source_chunks=[{"id": "x", "text": "No prerequisites here."}],
        key_terms=(),
        objectives=[{"statement": "Simplify a fraction by dividing out the GCF."}],
    )
    # Nothing to ground a backfill → stays empty (strict validator fails closed).
    assert out["prerequisitePages"] == []
    assert out["_prereq_pages_repair"]["backfill_source"] is None


def test_repair_prereq_pages_noop_on_clean_array():
    """A non-empty array of clean (non-garbage) strings is untouched."""
    out = _repair_prereq_pages(
        {"prerequisitePages": ["factoring whole numbers", "finding the GCF"]},
        block_type="prereq_set",
        source_chunks=_PREREQ_SOURCE,
        key_terms=(),
        objectives=[],
    )
    assert out["prerequisitePages"] == ["factoring whole numbers", "finding the GCF"]
    assert out["_prereq_pages_repair"]["repaired"] is False


def test_repair_prereq_pages_noop_on_other_block_types():
    """No-op for any non-prereq_set block (the repair is scoped to prereq_set
    so the other 15 block types are byte-identical)."""
    out = _repair_prereq_pages(
        {"prerequisitePages": []},
        block_type="concept",
        source_chunks=_PREREQ_SOURCE,
        key_terms=("a", "b"),
        objectives=[],
    )
    assert out["prerequisitePages"] == []
    assert out["_prereq_pages_repair"]["repaired"] is False


def test_repair_prereq_pages_output_satisfies_strict_schema():
    """End-to-end: a backfilled candidate validates against the strict
    prereq_set schema (the repair makes the model reliably SATISFY the
    required+minItems:1 constraint, not relax it)."""
    import jsonschema  # type: ignore[import-untyped]

    candidate = {
        "block_id": "p#prereq_set_x_0",
        "block_type": "prereq_set",
        "content_type": "procedure",
        "bloom_level": "remember",
        "objective_refs": ["CO-01"],
        "curies": ["math:gcf"],
        "key_claims": [
            {"claim": "Learners need factoring first.",
             "source_chunk_ids": ["semantik:source-beta#sec_01"]}
        ],
        "section_skeleton": [{"heading": "Prerequisites"}],
        "source_refs": [
            {"sourceId": "semantik:source-beta#sec_01", "role": "primary"}
        ],
        "structural_warnings": [],
        "prerequisitePages": ["p#bad_0"],  # garbage-only → would fail minItems
    }
    out = _repair_prereq_pages(
        candidate,
        block_type="prereq_set",
        source_chunks=_PREREQ_SOURCE,
        key_terms=("fraction",),
        objectives=[],
    )
    out.pop("_prereq_pages_repair")
    schema = _BLOCK_TYPE_JSON_SCHEMAS["prereq_set"]
    # No exception → the backfilled array satisfies required + minItems:1.
    jsonschema.Draft202012Validator(schema).validate(out)
    assert len(out["prerequisitePages"]) >= 1


def test_numeric_bloom_level_rejected_before_repair():
    """Baseline: a numeric bloom_level fails the strict schema — the exact
    live-7B ``2 is not of type 'string'`` failure."""
    import jsonschema  # type: ignore[import-untyped]

    payload = _valid_explanation_outline()
    payload["bloom_level"] = 3
    with pytest.raises(jsonschema.ValidationError):
        _validate_explanation(payload)


def test_repair_bloom_level_maps_numeric_tier_to_enum():
    """A numeric tier 1..6 maps to the canonical enum (1=remember .. 3=apply
    .. 6=create) sourced from lib/ontology/bloom.py::BLOOM_LEVELS. Tested as
    int and numeric-string forms; the repaired payload passes the schema."""
    for raw, expected in [
        (1, "remember"),
        (3, "apply"),
        ("3", "apply"),
        (6, "create"),
    ]:
        payload = _valid_explanation_outline()
        payload["bloom_level"] = raw
        out = _repair_outline_bloom_level(payload)
        meta = out.pop("_bloom_level_repair")
        assert meta["repaired"] is True, raw
        assert out["bloom_level"] == expected, raw
        _validate_explanation(out)  # must not raise


def test_repair_bloom_level_normalizes_case_mismatch_string():
    """A capitalized / whitespace-padded enum value is normalized to the
    canonical lowercase form."""
    payload = _valid_explanation_outline()
    payload["bloom_level"] = "  Apply "
    out = _repair_outline_bloom_level(payload)
    meta = out.pop("_bloom_level_repair")
    assert meta["repaired"] is True
    assert out["bloom_level"] == "apply"
    _validate_explanation(out)


def test_repair_bloom_level_leaves_unmappable_unchanged():
    """A value that can't map to a valid enum member is left untouched —
    the strict validator then fails closed (no fabrication of a level).
    Out-of-range tier 0 / 7 and gibberish strings stay as-is."""
    import jsonschema  # type: ignore[import-untyped]

    for raw in [0, 7, "comprehension", "tier-two", True]:
        payload = _valid_explanation_outline()
        payload["bloom_level"] = raw
        out = _repair_outline_bloom_level(payload)
        meta = out.pop("_bloom_level_repair")
        assert meta["repaired"] is False, raw
        assert out["bloom_level"] == raw, raw
        with pytest.raises(jsonschema.ValidationError):
            _validate_explanation(out)


def test_repair_bloom_level_noop_on_already_valid():
    """An already-canonical bloom_level is untouched (no repair fires)."""
    payload = _valid_explanation_outline()
    assert payload["bloom_level"] == "understand"
    out = _repair_outline_bloom_level(payload)
    meta = out.pop("_bloom_level_repair")
    assert meta["repaired"] is False
    assert out["bloom_level"] == "understand"
    _validate_explanation(out)


def test_all_five_repairs_normalize_every_failure_end_to_end():
    """All four wrong-primitive-shape failures present at once (mixed
    key_claims + URL-CURIE + bare source_refs + numeric bloom_level): the
    full pre-validation repair chain coerces every one and the payload then
    PASSES the strict explanation schema — the end-to-end contract the
    outline-tier call site relies on after the two new repairs land."""
    payload = _valid_explanation_outline()
    stray_id = "democourse_chunk_00001"
    payload["key_claims"].append(stray_id)
    payload["curies"] = ["foo:bar", "schema:https://example.com/vocab"]
    payload["source_refs"] = ["semantik:slug#chunk_a", {"sourceId": "semantik:slug#chunk_b"}]
    payload["bloom_level"] = 2
    valid_ids = ["chunk_a", stray_id]

    # Mirror the production ordering in generate_outline.
    payload = _repair_outline_key_claims_shape(payload, valid_ids=valid_ids)
    payload.pop("_key_claims_shape_repair", None)
    payload = _repair_outline_curies(payload)
    payload.pop("_curie_shape_repair", None)
    payload = _repair_claim_grounding(payload, valid_ids=valid_ids)
    payload.pop("_grounding_repair", None)
    payload = _repair_outline_source_refs(payload)
    payload.pop("_source_refs_shape_repair", None)
    payload = _repair_outline_bloom_level(payload)
    payload.pop("_bloom_level_repair", None)

    assert payload["curies"] == ["foo:bar"]
    assert len(payload["key_claims"]) == 2
    assert all(isinstance(c, dict) for c in payload["key_claims"])
    assert payload["source_refs"] == [
        {"sourceId": "semantik:slug#chunk_a", "role": "primary"},
        {"sourceId": "semantik:slug#chunk_b", "role": "primary"},
    ]
    assert payload["bloom_level"] == "understand"  # tier 2 → understand
    _validate_explanation(payload)


# ---------------------------------------------------------------------------
# assessment-item-descriptor fix (2026-06): the 7B AND the 14B exhaust the
# assessment_item OUTLINE budget by OMITTING the four dedicated fields
# (stem / answer_key / distractors / correct_answer_index) — pouring the
# question into key_claims / section_skeleton — or by emitting a list of
# field-TYPE DESCRIPTOR objects (``[{"type": "stem"}, ...]``) instead of real
# values. The fix names the four fields with real-value emphasis in the
# per-type variation block + the closing field enumeration, and adds a retry
# directive matching the missing-field / descriptor-list failure.
# ---------------------------------------------------------------------------


def test_assessment_item_user_prompt_names_real_value_fields(monkeypatch):
    """The assessment_item user prompt explicitly names the four dedicated
    fields, demands REAL VALUES, and forbids the ``{"type": ...}``
    descriptor-list shape — both in the per-type variation block and the
    closing field enumeration."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(
        block_type="assessment_item",
        block_id="page-1#assessment_item_q1_0",
    )
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    # All four dedicated fields named in the body.
    for field in ("stem", "answer_key", "distractors", "correct_answer_index"):
        assert field in prompt
    # Real-value emphasis + descriptor-list prohibition.
    assert "REAL VALUES" in prompt
    assert '{"type": "stem"}' in prompt
    # Closing enumeration carries the extras too (recency reminder).
    assert "real question text" in prompt
    assert "0-based integer" in prompt


def test_non_assessment_prompt_omits_assessment_extra_fields(monkeypatch):
    """A non-assessment block keeps the byte-stable 10-field closing
    enumeration — the assessment-extra reminder is scoped to assessment_item
    so other block types are unaffected."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = OutlineProvider(provider="local")
    block = _stub_block(block_type="concept")
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    assert "correct_answer_index" not in prompt
    assert "REAL VALUES" not in prompt


def test_retry_directive_matches_missing_assessment_field():
    """A ``'stem' is a required property`` (and ``'answer_key' ...``)
    validator error — the live 7B/14B failure where the dedicated fields are
    omitted — matches the assessment-extras directive, which steers the model
    to emit the real values (not a descriptor list)."""
    for field in ("stem", "answer_key", "distractors", "correct_answer_index"):
        err = f"'{field}' is a required property"
        directive = _match_retry_directive(err, "assessment_item")
        assert directive is not None, field
        # The directive names the four fields + demands real values + forbids
        # the descriptor-list shape.
        assert "REAL VALUES" in directive
        assert '{"type": "stem"}' in directive
        assert "real stem text" in directive


def test_retry_directive_descriptor_list_steered_to_real_values():
    """A descriptor-list candidate (``key_claims``/``section_skeleton`` carrying
    ``{"type": "stem"}`` objects in lieu of the dedicated fields) fails the
    strict schema with a missing-required-field error; the matched directive
    tells the model to REPLACE the descriptor object with real values rather
    than re-rolling the same descriptor list."""
    # The descriptor list lands the model in a missing-field error because the
    # dedicated top-level fields are absent.
    err = "'stem' is a required property"
    directive = _match_retry_directive(err, "assessment_item")
    assert directive is not None
    assert "Replace any {\"type\": ...} descriptor object" in directive


def test_assessment_directive_does_not_steal_key_claims_or_distractors_short():
    """Ordering invariant: the widened assessment-extras pattern must NOT steal
    the key_claims-scoped minItems directive nor break the W7 distractors
    'too short' path."""
    # key_claims-scoped minItems error still routes to the decomposition
    # directive (NOT the assessment directive).
    kc = _match_retry_directive("key_claims: [...] is too short", "assessment_item")
    assert "too FEW entries" in kc
    # distractors 'too short' still routes to the assessment directive.
    short = _match_retry_directive("distractors [] is too short", "assessment_item")
    assert "distractors" in short
    assert "REAL VALUES" in short


# ---------------------------------------------------------------------------
# _repair_assessment_item_payload — distractors/answer_key/index reconciliation
# (regression for the 2026-06-20 inter_tier_validation phase-stop: a 7B
# assessment_item with wrong-only distractors + a correct_answer_index pointing
# past them at the separate answer_key failed the critical
# ``outline_assessment_item_payload`` gate.)
# ---------------------------------------------------------------------------


def _norm_eq(a: str, b: str) -> bool:
    return "".join(a.split()).lower() == "".join(b.split()).lower()


def test_repair_assessment_item_out_of_range_inserts_answer_key():
    # week_01 shape: 2 wrong distractors, answer separate, index past them.
    content = {
        "block_type": "assessment_item",
        "stem": "Simplify 3(2x + 4) + 5(x - 1)",
        "answer_key": "11x + 12",
        "distractors": [{"text": "6x + 7"}, {"text": "8x + 9"}],
        "correct_answer_index": 2,
    }
    out = _repair_assessment_item_payload(content, block_type="assessment_item")
    d = out["distractors"]
    cai = out["correct_answer_index"]
    assert 0 <= cai < len(d)
    assert _norm_eq(d[cai]["text"], "11x + 12")
    assert out["_assessment_item_payload_repair"]["mode"] == "insert_answer_key"


def test_repair_assessment_item_in_range_wrong_option_is_fixed():
    # week_06 shape: in-range index but pointing at a WRONG distractor (the
    # correct answer is in answer_key, absent from distractors).
    content = {
        "block_type": "assessment_item",
        "stem": "Divide: -5 / 7.",
        "answer_key": "-5/7",
        "distractors": [{"text": "-2/7"}, {"text": "3/7"}],
        "correct_answer_index": 0,
    }
    out = _repair_assessment_item_payload(content, block_type="assessment_item")
    d = out["distractors"]
    cai = out["correct_answer_index"]
    assert 0 <= cai < len(d)
    assert _norm_eq(d[cai]["text"], "-5/7")


def test_repair_assessment_item_valid_block_is_idempotent():
    # Correct answer already a distractor at the named index -> no change.
    content = {
        "block_type": "assessment_item",
        "answer_key": "B",
        "distractors": [{"text": "A"}, {"text": "B"}, {"text": "C"}],
        "correct_answer_index": 1,
    }
    out = _repair_assessment_item_payload(content, block_type="assessment_item")
    assert out["correct_answer_index"] == 1
    assert [x["text"] for x in out["distractors"]] == ["A", "B", "C"]
    assert out["_assessment_item_payload_repair"]["repaired"] is False


def test_repair_assessment_item_formatting_variant_reindexes_not_duplicates():
    # answer_key "a + b" vs distractor "a+b" must REINDEX (not insert a dup).
    content = {
        "block_type": "assessment_item",
        "answer_key": "a + b",
        "distractors": [{"text": "a+b"}, {"text": "c"}],
        "correct_answer_index": 1,
    }
    out = _repair_assessment_item_payload(content, block_type="assessment_item")
    assert len(out["distractors"]) == 2  # no duplicate inserted
    assert out["correct_answer_index"] == 0
    assert out["_assessment_item_payload_repair"]["mode"] == "reindex_to_existing"


def test_repair_assessment_item_noop_for_other_block_types():
    content = {"block_type": "concept", "foo": "bar"}
    out = _repair_assessment_item_payload(content, block_type="concept")
    assert out["_assessment_item_payload_repair"]["repaired"] is False
    assert "distractors" not in out


def test_repair_assessment_item_no_answer_key_clamps_out_of_range_index():
    content = {
        "block_type": "assessment_item",
        "distractors": [{"text": "A"}, {"text": "B"}],
        "correct_answer_index": 9,
    }
    out = _repair_assessment_item_payload(content, block_type="assessment_item")
    assert 0 <= out["correct_answer_index"] < len(out["distractors"])
    assert out["_assessment_item_payload_repair"]["mode"] == "clamp_index_no_answer_key"


# ---------------------------------------------------------------------------
# 2026-06-21 content-gap fix — over-demanding minItems FLOORS dropped valid
# activity/summary/checklist/scenario blocks after exhausting the outline
# regen budget. The floors are LOWERED (anti-fabrication: only lowered, never
# below key_claims=1; maxItems unchanged) so a single grounded claim — or a
# section-less activity prompt — SHIPS instead of being dropped. These tests
# assert the now-permitted minimal payloads PASS strict schema validation.
# ---------------------------------------------------------------------------


def _minimal_outline_payload(
    block_type: str,
    *,
    key_claims: List[Dict[str, Any]],
    section_skeleton: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A schema-clean minimal outline payload for ``block_type``.

    Mirrors ``_valid_explanation_outline`` but parametrised on block_type and
    on the two arrays under test, so a single-claim / section-less payload can
    be validated against the per-type schema directly.
    """
    return {
        "block_id": f"page_01#{block_type}_intro_0",
        "block_type": block_type,
        "content_type": sorted(__import__("lib.validators.content_type",
                                          fromlist=["get_valid_chunk_types"])
                               .get_valid_chunk_types())[0],
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": [],
        "key_claims": key_claims,
        "section_skeleton": section_skeleton,
        "source_refs": [
            {"sourceId": "semantik:slug#chunk_a", "role": "primary"},
        ],
        "structural_warnings": [],
    }


def _one_structured_claim() -> List[Dict[str, Any]]:
    return [
        {"claim": "Evaluate the following expressions using order of "
                  "operations.",
         "source_chunk_ids": ["chunk_a"]},
    ]


@pytest.mark.parametrize("block_type", ["checklist", "scenario",
                                        "summary_takeaway"])
def test_single_key_claim_now_passes_schema(block_type):
    """2026-06-21 content-gap fix: checklist/scenario/summary_takeaway now
    accept exactly ONE structured key_claims entry (floor 2 -> 1), so a valid
    block carrying a single grounded claim is no longer dropped after the
    regen budget."""
    lo, _hi = _OUTLINE_KIND_BOUNDS[block_type]["key_claims"]
    assert lo == 1  # the lowered floor
    schema = _build_block_outline_schema(block_type)
    payload = _minimal_outline_payload(
        block_type,
        key_claims=_one_structured_claim(),
        section_skeleton=[{"heading": "Body"}],
    )
    jsonschema.Draft202012Validator(schema).validate(payload)  # must not raise


def test_section_less_activity_now_passes_schema():
    """2026-06-21 content-gap fix: ``activity`` section_skeleton floor 1 -> 0,
    so a section-less activity prompt (empty section_skeleton) ships instead
    of tripping the empty-required-array failure after the regen budget. Its
    key_claims floor stays 1 (never lowered to 0)."""
    sec_lo, _sec_hi = _OUTLINE_KIND_BOUNDS["activity"]["section_skeleton"]
    kc_lo, _kc_hi = _OUTLINE_KIND_BOUNDS["activity"]["key_claims"]
    assert sec_lo == 0  # the lowered section floor
    assert kc_lo == 1   # key_claims floor unchanged (never 0)
    schema = _build_block_outline_schema("activity")
    payload = _minimal_outline_payload(
        "activity",
        key_claims=_one_structured_claim(),
        section_skeleton=[],  # the section-less case
    )
    jsonschema.Draft202012Validator(schema).validate(payload)  # must not raise


def test_content_gap_fix_maxitems_unchanged():
    """Anti-fabrication guard: only the FLOORS moved; every maxItems is
    byte-stable for the four touched block types."""
    assert _OUTLINE_KIND_BOUNDS["checklist"]["key_claims"] == (1, 6)
    assert _OUTLINE_KIND_BOUNDS["scenario"]["key_claims"] == (1, 4)
    assert _OUTLINE_KIND_BOUNDS["summary_takeaway"]["key_claims"] == (1, 5)
    assert _OUTLINE_KIND_BOUNDS["activity"]["section_skeleton"] == (0, 3)
    assert _OUTLINE_KIND_BOUNDS["activity"]["key_claims"] == (1, 4)


# ---------------------------------------------------------------------------
# Objective-echo claim repair (2026-07)
# ---------------------------------------------------------------------------

_ECHO_OBJECTIVE = "Identify the place value of each digit in a given number."


def _echo_objectives() -> List[Dict[str, Any]]:
    return [{"id": "CO-01", "statement": _ECHO_OBJECTIVE, "bloom_level": "apply"}]


def _structured(text: str, ids=("semantik:slug#blk1",)) -> Dict[str, Any]:
    return {"claim": text, "source_chunk_ids": list(ids)}


def test_objective_echo_claim_dropped_for_concept_block():
    """A concept-block claim that verbatim restates the CO statement is
    dropped; the genuine factual claim survives."""
    fact = "The place value of the 4 in 51,493 is hundreds."
    candidate = {
        "key_claims": [_structured(_ECHO_OBJECTIVE), _structured(fact)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=_echo_objectives()
    )
    texts = [c["claim"] for c in out["key_claims"]]
    assert texts == [fact]
    meta = out["_objective_echo_repair"]
    assert meta["repaired"] is True
    assert meta["n_dropped"] == 1
    assert meta["n_objective_echo_warned"] == 0
    assert _OBJECTIVE_ECHO_WARNING not in out["structural_warnings"]


def test_objective_echo_claim_dropped_for_example_block():
    """Same drop behavior on an example block (also non-exempt)."""
    fact = "Dividing 3/4 by 2/5 gives 15/8 after multiplying by the reciprocal."
    candidate = {
        "key_claims": [_structured(fact), _structured(_ECHO_OBJECTIVE)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="example", objectives=_echo_objectives()
    )
    assert [c["claim"] for c in out["key_claims"]] == [fact]
    assert out["_objective_echo_repair"]["n_dropped"] == 1


def test_objective_block_is_exempt_from_echo_drop():
    """An ``objective`` block restating the objective is that block's job —
    the echo claim is preserved untouched."""
    candidate = {
        "key_claims": [_structured(_ECHO_OBJECTIVE)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="objective", objectives=_echo_objectives()
    )
    assert [c["claim"] for c in out["key_claims"]] == [_ECHO_OBJECTIVE]
    meta = out["_objective_echo_repair"]
    assert meta["repaired"] is False
    assert meta["n_dropped"] == 0


def test_all_echo_block_keeps_one_claim_and_warns():
    """When EVERY claim is an echo, never empty the block: keep the first
    claim and append the OBJECTIVE_ECHO_CLAIMS structural warning."""
    shuffle = "The place value of each digit in a given number identify."
    candidate = {
        "key_claims": [_structured(_ECHO_OBJECTIVE), _structured(shuffle)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=_echo_objectives()
    )
    assert len(out["key_claims"]) == 1
    assert out["key_claims"][0]["claim"] == _ECHO_OBJECTIVE
    assert _OBJECTIVE_ECHO_WARNING in out["structural_warnings"]
    meta = out["_objective_echo_repair"]
    assert meta["repaired"] is True
    assert meta["n_objective_echo_warned"] == 1
    assert meta["n_dropped"] == 0


def test_jaccard_word_order_shuffle_is_caught():
    """A pure word-order shuffle of the objective (same token set) exceeds the
    0.85 Jaccard floor and is dropped as an echo."""
    shuffle = "digit place value the of each in a given number identify"
    fact = "A four-digit number has thousands, hundreds, tens, and ones places."
    candidate = {
        "key_claims": [_structured(shuffle), _structured(fact)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=_echo_objectives()
    )
    assert [c["claim"] for c in out["key_claims"]] == [fact]
    assert out["_objective_echo_repair"]["n_dropped"] == 1


def test_legit_fact_sharing_objective_vocab_survives():
    """A concrete factual claim that happens to share objective vocabulary
    ("place value") but asserts a specific fact stays below the Jaccard floor
    and is NOT dropped."""
    fact = "The place value of the 4 in 51,493 is hundreds."
    candidate = {
        "key_claims": [_structured(fact)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=_echo_objectives()
    )
    assert [c["claim"] for c in out["key_claims"]] == [fact]
    meta = out["_objective_echo_repair"]
    assert meta["repaired"] is False
    assert meta["n_dropped"] == 0


def test_echo_drop_handles_legacy_flat_string_claims():
    """The drop pass reads claim text from BOTH the structured object arm and
    the legacy flat-string arm."""
    fact = "The place value of the 4 in 51,493 is hundreds."
    candidate = {
        "key_claims": [_ECHO_OBJECTIVE, fact],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=_echo_objectives()
    )
    assert out["key_claims"] == [fact]


def test_echo_drop_noop_when_no_objectives():
    candidate = {
        "key_claims": [_structured(_ECHO_OBJECTIVE)],
        "structural_warnings": [],
    }
    out = _drop_objective_echo_claims(
        candidate, block_type="concept", objectives=[]
    )
    assert [c["claim"] for c in out["key_claims"]] == [_ECHO_OBJECTIVE]
    assert out["_objective_echo_repair"]["repaired"] is False


def test_outline_system_prompt_carries_objective_echo_directive():
    """PROMPT arm: the system prompt forbids restating the objective as a
    claim and demands factual assertions from the source."""
    prompt = _OUTLINE_SYSTEM_PROMPT.lower()
    assert "restatement or paraphrase of a learning objective" in prompt
    assert "factual, teachable assertion" in prompt
