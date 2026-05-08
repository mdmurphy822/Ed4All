"""W-D11 T11.3 — synthesis-side evidence_quote emit smoke tests.

T11.0 (commit 92d1bf8) landed additive optional ``evidence_quote`` +
``evidence_char_span`` fields on the ``key_claims[]`` (chunk-side) and
``per_claim_support[]`` (pair-side) claim-bearing schemas. T11.3
teaches every synthesis provider to ASK the LLM for the verbatim
quote at synthesis time so the consumer-side validator (T11.2) has
something to enforce.

This module covers the synthesis-surface contract:

a. The canonical evidence-quote prompt directive is present in every
   concrete provider's user prompt (Anthropic, Together, Local). The
   directive itself lives in ``_base_synthesis_provider.py`` so a
   single import is the source of truth.

b. The Claude Session provider threads ``evidence_quote_directive``
   into its dispatch task params (no string-prompt path — the
   subagent driver lives external to the SDK).

c. ``compute_evidence_quote_emit_rate`` correctly buckets the
   today-default (no claims emitted) case as ``(0, 0, 0.0)`` so a
   downstream provider call carries a meaningful "no claims emitted"
   marker in its rationale.

d. When a mock provider returns structured output WITH
   ``evidence_quote`` per claim, the field is threaded onto the
   emitted Pair / Chunk dict.

e. When a mock provider returns output WITHOUT ``evidence_quote``,
   the emit succeeds (graceful-degrade).

f. The shared structured-output schema fragment carries
   ``evidence_quote`` and ``evidence_char_span`` as optional
   ``anyOf [string|null]`` / ``anyOf [array|null]`` arms.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._base_synthesis_provider import (  # noqa: E402
    EVIDENCE_QUOTE_PROMPT_DIRECTIVE,
    EVIDENCE_QUOTE_SCHEMA_FRAGMENT,
    compute_evidence_quote_emit_rate,
    render_evidence_quote_rationale_fragment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_CHUNK_TEXT = (
    "SHACL shapes constrain RDF graphs by attaching property "
    "constraints to typed nodes. The validator walks each shape and "
    "reports violations against the targeted data graph."
)


def _instruction_draft() -> dict:
    return {
        "prompt": "Define the central concept behind topic X.",
        "completion": (
            "Topic X is a central idea in this material. Learners should be "
            "able to recall and restate it."
        ),
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "",
        "template_id": "remember.explanation",
        "provider": "mock",
        "schema_version": "v1",
    }


def _chunk() -> dict:
    return {
        "id": "chunk_001",
        "text": _CHUNK_TEXT,
        "learning_outcome_refs": ["TO-01"],
    }


# ---------------------------------------------------------------------------
# (f) Schema fragment shape
# ---------------------------------------------------------------------------


def test_schema_fragment_carries_optional_evidence_fields() -> None:
    """T11.0 schema arms must be optional — small models that ignore
    the schema gracefully degrade. Both fields ``anyOf [shape|null]``
    so a provider that emits ``null`` (or omits the key entirely)
    passes shape validation."""
    props = EVIDENCE_QUOTE_SCHEMA_FRAGMENT["properties"]
    assert "evidence_quote" in props
    assert "evidence_char_span" in props
    # Required key list is "claim" only — evidence fields are NOT in
    # required so omission is permitted.
    required = EVIDENCE_QUOTE_SCHEMA_FRAGMENT.get("required", [])
    assert "evidence_quote" not in required
    assert "evidence_char_span" not in required
    # 240-char structural cap mirrors the schema (T11.0 commit 92d1bf8).
    quote_arms = props["evidence_quote"]["anyOf"]
    str_arm = next(a for a in quote_arms if a.get("type") == "string")
    assert str_arm["maxLength"] == 240


# ---------------------------------------------------------------------------
# (c) Emit-rate computation
# ---------------------------------------------------------------------------


def test_emit_rate_no_claims_emitted_today_default() -> None:
    """Today's providers paraphrase (prompt, completion) /
    (prompt, chosen, rejected) — no claim arrays. Rate must be 0/0
    with the marker fragment so an operator can distinguish "no
    claims emitted" from "claims emitted without quotes"."""
    parsed = {"prompt": "p", "completion": "c"}
    with_q, total, rate = compute_evidence_quote_emit_rate(parsed)
    assert (with_q, total, rate) == (0, 0, 0.0)
    fragment = render_evidence_quote_rationale_fragment(parsed)
    assert "no claims emitted" in fragment


def test_emit_rate_partial_quote_coverage() -> None:
    """A response with mixed quote / null / missing entries reports
    the actual emit rate (1 of 3 in this case)."""
    parsed = {
        "key_claims": [
            {"claim": "a", "evidence_quote": "verbatim span"},
            {"claim": "b", "evidence_quote": None},
            {"claim": "c"},
        ],
    }
    with_q, total, rate = compute_evidence_quote_emit_rate(parsed)
    assert with_q == 1
    assert total == 3
    assert abs(rate - 1 / 3) < 1e-9
    fragment = render_evidence_quote_rationale_fragment(parsed)
    assert "1/3 claims carry verbatim quotes" in fragment


def test_emit_rate_handles_per_claim_support_arm() -> None:
    """The pair-side ``per_claim_support[]`` arm is recognised the
    same way the chunk-side ``key_claims[]`` arm is."""
    parsed = {
        "per_claim_support": [
            {
                "sentence": "x",
                "entailment": 0.9,
                "contradiction": 0.0,
                "outcome": "entailed",
                "evidence_quote": "lifted from chunk text",
            },
            {
                "sentence": "y",
                "entailment": 0.4,
                "contradiction": 0.5,
                "outcome": "contradicted",
            },
        ],
    }
    with_q, total, _ = compute_evidence_quote_emit_rate(parsed)
    assert (with_q, total) == (1, 2)


def test_emit_rate_legacy_list_str_key_claims_skipped() -> None:
    """Legacy List[str] ``key_claims[]`` shape carries no per-claim
    object — bare strings can't be stamped with ``evidence_quote``,
    so they count as 0 claims for the emit-rate calculation. A
    rebuild on a pre-W1.5 corpus must not crash on the legacy
    shape."""
    parsed = {"key_claims": ["bare-string claim 1", "bare-string claim 2"]}
    with_q, total, rate = compute_evidence_quote_emit_rate(parsed)
    assert (with_q, total, rate) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# (a) Anthropic / Together / Local prompt directive presence
# ---------------------------------------------------------------------------


def test_anthropic_prompt_carries_evidence_quote_directive() -> None:
    """Anthropic provider's ``_render_instruction_user`` /
    ``_render_preference_user`` must append the canonical evidence-
    quote directive sourced from the base provider module."""
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )
    inst = AnthropicSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    assert "evidence_quote" in inst
    assert "verbatim substring" in inst
    pref_draft = {**_instruction_draft(), "chosen": "x", "rejected": "y"}
    pref = AnthropicSynthesisProvider._render_preference_user(
        pref_draft, "chunk_001",
    )
    assert "evidence_quote" in pref


def test_together_prompt_carries_evidence_quote_directive() -> None:
    from Trainforge.generators._together_provider import (
        TogetherSynthesisProvider,
    )
    inst = TogetherSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    assert "evidence_quote" in inst
    pref_draft = {**_instruction_draft(), "chosen": "x", "rejected": "y"}
    pref = TogetherSynthesisProvider._render_preference_user(
        pref_draft, "chunk_001",
    )
    assert "evidence_quote" in pref


def test_local_prompt_carries_evidence_quote_directive() -> None:
    """Local provider's prompt has multiple stacked directives
    (preserve, definition, JSON shape). The evidence-quote directive
    sits between the per-call directives and the strict-JSON shape
    directive — must be present in BOTH instruction + preference
    paths."""
    from Trainforge.generators._local_provider import (
        LocalSynthesisProvider,
    )
    inst = LocalSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    assert "evidence_quote" in inst
    pref_draft = {**_instruction_draft(), "chosen": "x", "rejected": "y"}
    pref = LocalSynthesisProvider._render_preference_user(
        pref_draft, "chunk_001",
    )
    assert "evidence_quote" in pref


def test_canonical_directive_constant_used_uniformly() -> None:
    """Anthropic / Together / Local must all import the SAME directive
    text (single source of truth) — a future edit to
    ``EVIDENCE_QUOTE_PROMPT_DIRECTIVE`` updates every provider in
    lockstep without per-provider drift."""
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )
    from Trainforge.generators._together_provider import (
        TogetherSynthesisProvider,
    )
    from Trainforge.generators._local_provider import (
        LocalSynthesisProvider,
    )
    a = AnthropicSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    t = TogetherSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    # Local doesn't append the directive last (it has stacked
    # directives) — so we don't assert byte-suffix equality, only
    # that the canonical directive substring appears intact.
    assert EVIDENCE_QUOTE_PROMPT_DIRECTIVE in a
    assert EVIDENCE_QUOTE_PROMPT_DIRECTIVE in t
    local_inst = LocalSynthesisProvider._render_instruction_user(
        _instruction_draft(), "chunk_001",
    )
    assert EVIDENCE_QUOTE_PROMPT_DIRECTIVE in local_inst


# ---------------------------------------------------------------------------
# (b) Claude Session dispatch task params carry the directive
# ---------------------------------------------------------------------------


def test_claude_session_dispatch_threads_evidence_quote_directive() -> None:
    """Claude Session has no string-prompt path — the subagent driver
    lives external to the SDK. Thread the directive into the
    dispatch task_params so the subagent receives the same per-claim
    contract as the prompt-string providers."""
    import asyncio

    from Trainforge.generators._claude_session_provider import (
        ClaudeSessionProvider,
    )

    captured_params: dict = {}

    class _FakeDispatcher:
        async def dispatch_task(self, **kwargs: Any) -> dict:
            captured_params.update(kwargs)
            return {
                "success": True,
                "outputs": {
                    "prompt": (
                        "Explain in your own words how the central "
                        "concept of chunk_001 frames downstream "
                        "examples for a learner."
                    ),
                    "completion": (
                        "A clear paraphrased completion explaining the "
                        "concept in fresh language with full sentences."
                    ),
                },
            }

    provider = ClaudeSessionProvider(dispatcher=_FakeDispatcher())
    out = provider.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "claude_session"
    assert "task_params" in captured_params
    task_params = captured_params["task_params"]
    assert "evidence_quote_directive" in task_params
    assert "evidence_quote" in task_params["evidence_quote_directive"]


# ---------------------------------------------------------------------------
# (d) + (e) Mock-provider end-to-end emit (anthropic surface)
# ---------------------------------------------------------------------------


def _mock_anthropic_response(text: str) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 100
    response.usage = usage
    return response


def test_anthropic_emit_threads_key_claims_with_evidence_quote() -> None:
    """When the LLM returns a structured response with key_claims[]
    each carrying ``evidence_quote``, the field is threaded onto
    the emitted instruction-pair dict."""
    import json
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )

    body = {
        "prompt": (
            "Explain how SHACL shapes constrain RDF graphs in "
            "instructional terms suitable for a learner."
        ),
        "completion": (
            "SHACL shapes attach property constraints to typed nodes "
            "and the validator reports violations against the data "
            "graph during the validation walk."
        ),
        "key_claims": [
            {
                "claim": "SHACL shapes constrain RDF graphs.",
                "source_chunk_ids": ["chunk_001"],
                "evidence_quote": "SHACL shapes constrain RDF graphs",
                "evidence_char_span": [0, 33],
            },
        ],
    }
    client = MagicMock()
    client.messages.create.side_effect = [_mock_anthropic_response(json.dumps(body))]
    provider = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = provider.paraphrase_instruction(_instruction_draft(), _chunk())
    assert "key_claims" in out
    assert isinstance(out["key_claims"], list)
    assert out["key_claims"][0]["evidence_quote"] == (
        "SHACL shapes constrain RDF graphs"
    )
    assert out["key_claims"][0]["evidence_char_span"] == [0, 33]


def test_anthropic_emit_graceful_degrade_no_evidence_quote() -> None:
    """When the LLM omits ``evidence_quote`` (older / smaller model
    that ignores the schema directive), the emit still succeeds.
    The downstream T11.2 validator handles the missing-quote arm
    as warning-only."""
    import json
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )

    body = {
        "prompt": (
            "Explain how SHACL shapes constrain RDF graphs in "
            "instructional terms suitable for a learner."
        ),
        "completion": (
            "SHACL shapes attach property constraints to typed nodes "
            "and the validator reports violations against the data "
            "graph during the validation walk."
        ),
        # Note: no key_claims at all — the today-default shape.
    }
    client = MagicMock()
    client.messages.create.side_effect = [_mock_anthropic_response(json.dumps(body))]
    provider = AnthropicSynthesisProvider(api_key="sk-test", client=client)
    out = provider.paraphrase_instruction(_instruction_draft(), _chunk())
    # Emit succeeds — graceful degrade.
    assert "prompt" in out
    assert "completion" in out
    assert out["provider"] == "anthropic"
    # No key_claims threaded because the response didn't carry any.
    assert "key_claims" not in out


def test_anthropic_emit_rate_lands_in_decision_capture() -> None:
    """The ``synthesis_provider_call`` decision rationale interpolates
    ``evidence_quote_emit_rate`` (mirrors W5.D pattern — extend
    existing decision_type rationale, no enum change)."""
    import json
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )

    body = {
        "prompt": (
            "Explain how SHACL shapes constrain RDF graphs in "
            "instructional terms suitable for a learner."
        ),
        "completion": (
            "SHACL shapes attach property constraints to typed nodes "
            "and the validator reports violations against the data "
            "graph during the validation walk."
        ),
        "key_claims": [
            {
                "claim": "SHACL shapes constrain RDF graphs.",
                "evidence_quote": "SHACL shapes constrain RDF graphs",
            },
            {"claim": "Validator reports violations."},
        ],
    }
    client = MagicMock()
    client.messages.create.side_effect = [_mock_anthropic_response(json.dumps(body))]

    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    provider = AnthropicSynthesisProvider(
        api_key="sk-test", client=client, capture=_FakeCapture(),
    )
    provider.paraphrase_instruction(_instruction_draft(), _chunk())
    assert len(captured) == 1
    rationale = captured[0]["rationale"]
    assert "evidence_quote_emit_rate=0.500" in rationale
    assert "1/2 claims carry verbatim quotes" in rationale


def test_anthropic_emit_rate_fragment_when_no_claims_emitted() -> None:
    """The today-default (no ``key_claims`` / ``per_claim_support``
    in the response) emits ``evidence_quote_emit_rate=0.000 (no
    claims emitted)`` so an operator can tell apart "no claims" from
    "claims without quotes" when post-hoc auditing the rationale
    stream."""
    import json
    from Trainforge.generators._anthropic_provider import (
        AnthropicSynthesisProvider,
    )

    body = {
        "prompt": (
            "Explain how SHACL shapes constrain RDF graphs in "
            "instructional terms suitable for a learner."
        ),
        "completion": (
            "SHACL shapes attach property constraints to typed nodes "
            "and the validator reports violations against the data "
            "graph during the validation walk."
        ),
    }
    client = MagicMock()
    client.messages.create.side_effect = [_mock_anthropic_response(json.dumps(body))]

    captured: List[dict] = []

    class _FakeCapture:
        def log_decision(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    provider = AnthropicSynthesisProvider(
        api_key="sk-test", client=client, capture=_FakeCapture(),
    )
    provider.paraphrase_instruction(_instruction_draft(), _chunk())
    assert len(captured) == 1
    rationale = captured[0]["rationale"]
    assert "no claims emitted" in rationale
