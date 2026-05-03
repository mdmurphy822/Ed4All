"""Tests for ``RewriteProvider`` (Phase 3 Subtask 27).

Exercises the rewrite-tier LLM-agnostic provider that consumes an
outline-dict Block and emits a rendered-HTML Block plus a cumulative
``Touch(tier="rewrite", ...)`` audit entry. Coverage:

- Construction: default provider is anthropic, ``COURSEFORGE_REWRITE_PROVIDER``
  selects an alternate, unknown provider raises ``ValueError``.
- Anthropic happy path: the SDK route returns and assembles a Block
  carrying the HTML response and a single ``rewrite``-tier Touch.
- CURIE-preservation gate: when the LLM's first response drops a CURIE,
  the gate appends a remediation directive and retries; when the second
  response includes the CURIE the gate accepts.
- CURIE-preservation exhaustion: when every retry drops the CURIE the
  gate raises ``RewriteProviderError(code="rewrite_curie_drop")`` with
  the missing tokens listed.
- Escalated blocks (``escalation_marker != None``) route through the
  richer prompt template that surfaces the marker context.
- The returned Block carries a single new ``Touch(tier="rewrite",
  purpose="pedagogical_depth")`` appended to the input ``touched_by``
  chain.

Mirrors ``Trainforge/tests/test_curriculum_alignment_provider.py`` for
import-path + helper conventions and the
``Courseforge/tests/test_content_generator_provider.py`` ``httpx.MockTransport``
fixture pattern so the LLM call-site test surfaces stay parallel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    RewriteProvider,
    RewriteProviderError,
    SUPPORTED_PROVIDERS,
)
from blocks import Block, Touch  # noqa: E402  (Phase 2 intermediate format)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-rewrite") -> dict:
    return {
        "id": "cmpl-rewrite-test",
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


def _outline_block(
    *,
    block_type: str = "concept",
    curies: List[str] | None = None,
    escalation_marker: str | None = None,
) -> Block:
    return Block(
        block_id="page#concept_intro_0",
        block_type=block_type,
        page_id="page",
        sequence=0,
        content={
            "key_claims": ["The central concept is X."],
            "curies": list(curies or []),
            "source_refs": ["dart:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_rewrite_provider_is_anthropic_when_env_unset(monkeypatch):
    """``COURSEFORGE_REWRITE_PROVIDER`` unset → defaults to anthropic."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    p = RewriteProvider(anthropic_client=object())
    assert p._provider == "anthropic"
    assert DEFAULT_PROVIDER == "anthropic"


def test_env_var_selects_provider(monkeypatch):
    """``COURSEFORGE_REWRITE_PROVIDER=local`` → routes to the local
    backend regardless of constructor default."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    p = RewriteProvider()
    assert p._provider == "local"


def test_unknown_provider_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    with pytest.raises(ValueError):
        RewriteProvider(provider="bogus")


# ---------------------------------------------------------------------------
# Happy paths per backend
# ---------------------------------------------------------------------------


def test_generate_rewrite_calls_anthropic_path_for_anthropic_provider(
    monkeypatch,
):
    """Anthropic backend dispatches through the SDK; the assistant
    response is unwrapped and assembled into a Block carrying the HTML
    + a rewrite-tier Touch."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")

    create_calls: List[Dict[str, Any]] = []

    class _FakeMessages:
        def create(self, **kwargs: Any) -> dict:
            create_calls.append(kwargs)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<section data-cf-source-ids=\"dart:slug#blk1\">"
                            "<h2 data-cf-content-type=\"explanation\">"
                            "Concept</h2>"
                            "<p>The central concept is X.</p>"
                            "</section>"
                        ),
                    }
                ]
            }

    class _FakeClient:
        messages = _FakeMessages()

    p = RewriteProvider(
        provider="anthropic",
        anthropic_client=_FakeClient(),
    )
    block = _outline_block(curies=[])
    out = p.generate_rewrite(block)

    assert isinstance(out, Block)
    assert isinstance(out.content, str)
    assert "<section" in out.content
    assert "central concept is X" in out.content
    assert len(create_calls) == 1


# ---------------------------------------------------------------------------
# CURIE-preservation gate
# ---------------------------------------------------------------------------


def test_curie_preservation_gate_fires_remediation_on_drop(monkeypatch):
    """First response drops the CURIE → gate appends remediation; the
    second response includes the CURIE → gate accepts. Verifies two
    POSTs land at the local server, the second prompt carries the
    'CURIE' remediation directive, and the returned Block's HTML carries
    the preserved CURIE."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []
    responses_html = [
        # First emit: CURIE stripped to natural language.
        "<section><p>The node shape constrains the focus node.</p></section>",
        # Second emit (post-remediation): CURIE preserved verbatim.
        (
            "<section data-cf-source-ids=\"dart:slug#blk1\">"
            "<p>The <code>sh:NodeShape</code> constrains the focus node.</p>"
            "</section>"
        ),
    ]
    response_idx = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = responses_html[response_idx["i"]]
        response_idx["i"] += 1
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _outline_block(curies=["sh:NodeShape"])
    out = p.generate_rewrite(block)

    assert len(seen) == 2, "expected two POSTs (initial + 1 remediation)"
    # The remediation directive lives in the request body of the SECOND
    # call. Substring-match the canonical remediation phrase.
    second_body = seen[1].read().decode("utf-8")
    assert "did not include the required" in second_body
    assert "sh:NodeShape" in second_body
    # The returned Block carries the second (CURIE-preserving) HTML.
    assert "sh:NodeShape" in out.content


def test_curie_preservation_exhaustion_raises_rewrite_curie_drop(
    monkeypatch,
):
    """Every retry drops the CURIE → gate raises after ``MAX_PARSE_RETRIES + 1``
    dispatches with ``code="rewrite_curie_drop"`` and the missing tokens
    listed in ``missing_curies``."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Always strip the CURIE.
        body = "<section><p>The node shape constrains.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _outline_block(curies=["sh:NodeShape"])
    with pytest.raises(RewriteProviderError) as excinfo:
        p.generate_rewrite(block)
    assert excinfo.value.code == "rewrite_curie_drop"
    assert "sh:NodeShape" in excinfo.value.missing_curies
    # Initial dispatch + MAX_PARSE_RETRIES (=2) more retries = 3 total.
    assert len(seen) == 3


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_escalated_block_uses_richer_prompt(monkeypatch):
    """A block whose ``escalation_marker`` is non-None routes through
    ``_render_escalated_user_prompt``: the prompt body carries
    ``ESCALATED REWRITE`` + the marker name + the outline's CURIE list
    verbatim. Verifies the escalation context paragraph is present.

    The test also asserts the legacy non-escalated prompt header
    (``"Outline (structurally correct, pedagogical-depth missing)"``)
    is NOT present, so a regression where the escalation branch silently
    falls through to the standard prompt is caught."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # CURIE preserved so the gate accepts on first try.
        body = (
            "<section><p>The <code>rdf:type</code> predicate types "
            "the focus node.</p></section>"
        )
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _outline_block(
        curies=["rdf:type"],
        escalation_marker="outline_budget_exhausted",
    )
    out = p.generate_rewrite(block)

    assert isinstance(out, Block)
    assert len(seen) == 1
    request_body = seen[0].read().decode("utf-8")
    assert "ESCALATED REWRITE" in request_body
    assert "outline_budget_exhausted" in request_body
    assert "rdf:type" in request_body
    # Legacy non-escalated header MUST NOT appear when the block is
    # escalated — otherwise the branch silently fell through.
    assert "Outline (structurally correct" not in request_body


# ---------------------------------------------------------------------------
# Required-attribute directive (post-Phase-3.5 prompt tightening)
# ---------------------------------------------------------------------------


def _capture_rewrite_request(
    monkeypatch, *, block: Block,
) -> str:
    """Drive a single rewrite call against a stubbed httpx transport and
    return the concatenated message text from the wire body. The wire
    body is JSON-encoded (so `"` in the prompt becomes `\\"`); decoding
    it back to the message content lets prompt-shape assertions match
    on the literal prompt text the model sees.

    The handler returns CURIE-preserving HTML so the rewrite-tier gate
    accepts on first try — the assertion is on what was SENT, not on
    what came back.
    """
    import json as _json

    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = "<section><p>Stub HTML — prompt content is what the test inspects.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    provider = RewriteProvider(provider="local", client=_make_client(handler))
    provider.generate_rewrite(block)
    assert len(seen) == 1
    payload = _json.loads(seen[0].read().decode("utf-8"))
    return "\n".join(m.get("content", "") for m in payload.get("messages", []))


def test_rewrite_prompt_enumerates_required_attrs_for_concept(monkeypatch):
    """The standard rewrite prompt must enumerate the post-rewrite gate's
    REQUIRED_ATTRS for the block_type AND interpolate the literal
    block_id as the data-cf-block-id value. The two together close the
    Qwen-7B-Q4 regression where the rewrite tier omitted data-cf-* attrs
    because the prompt described them in prose only."""
    block = Block(
        block_id="page1#concept_rdf_basics_0",
        block_type="concept",
        page_id="page1",
        sequence=0,
        content={"key_claims": ["RDF is a graph data model."], "curies": []},
    )
    request_body = _capture_rewrite_request(monkeypatch, block=block)

    assert "Required attributes" in request_body, (
        "rewrite prompt missing the gate-enforced attribute directive"
    )
    # concept REQUIRED_ATTRS = (data-cf-block-id, data-cf-content-type,
    # data-cf-key-terms). Each must appear literally in the prompt.
    assert "data-cf-block-id" in request_body
    assert "data-cf-content-type" in request_body
    assert "data-cf-key-terms" in request_body
    # The block_id literal must be quoted for the model to copy verbatim.
    assert 'data-cf-block-id="page1#concept_rdf_basics_0"' in request_body


def test_rewrite_prompt_enumerates_required_attrs_for_assessment_item(monkeypatch):
    """assessment_item REQUIRED_ATTRS adds data-cf-objective-ref and
    data-cf-bloom-level on top of data-cf-block-id; the prompt must list
    all three. This is the regression class observed on Qwen-7B-Q4 where
    the assessment_item rewrite dropped objective_ref + bloom_level even
    though the outline contained them."""
    block = Block(
        block_id="page1#assessment_item_q1_0",
        block_type="assessment_item",
        page_id="page1",
        sequence=0,
        content={
            "key_claims": ["Question stem"],
            "curies": [],
            "objective_refs": ["TO-01"],
            "bloom_level": "remember",
        },
    )
    request_body = _capture_rewrite_request(monkeypatch, block=block)

    assert "Required attributes" in request_body
    assert "data-cf-block-id" in request_body
    assert "data-cf-objective-ref" in request_body
    assert "data-cf-bloom-level" in request_body
    assert 'data-cf-block-id="page1#assessment_item_q1_0"' in request_body


def test_rewrite_escalated_prompt_also_enumerates_required_attrs(monkeypatch):
    """The escalated prompt branch (escalation_marker != None) carries
    the same required-attribute directive as the standard branch — the
    contract is invariant across the escalation seam."""
    block = Block(
        block_id="page2#concept_x_0",
        block_type="concept",
        page_id="page2",
        sequence=0,
        content={"key_claims": ["X"], "curies": []},
        escalation_marker="outline_budget_exhausted",
    )
    request_body = _capture_rewrite_request(monkeypatch, block=block)

    assert "ESCALATED REWRITE" in request_body
    assert "Required attributes" in request_body
    assert "data-cf-content-type" in request_body
    assert 'data-cf-block-id="page2#concept_x_0"' in request_body


def test_rewrite_system_prompt_carries_html_escape_directive(monkeypatch):
    """Regression: system prompt must instruct the model to escape
    literal angle brackets in placeholder text. Closes the Qwen-7B-Q4
    failure mode where the rewrite tier emitted bare `<subject>` /
    `<predicate>` / `<object>` placeholders that the parser saw as
    unclosed HTML elements (REWRITE_HTML_PARSE_FAIL critical at the
    post-rewrite shape gate)."""
    block = Block(
        block_id="page3#concept_y_0",
        block_type="concept",
        page_id="page3",
        sequence=0,
        content={"key_claims": ["Y"], "curies": []},
    )
    request_body = _capture_rewrite_request(monkeypatch, block=block)

    # The directive must be in the system prompt — assert on the
    # canonical phrases (not on the sample placeholder tokens) so a
    # rewording that preserves intent still passes.
    assert "&lt;" in request_body
    assert "&gt;" in request_body
    assert "<code>" in request_body
    # The directive references the gate by name so the model has the
    # cause-and-effect pinned.
    assert "post-rewrite shape gate" in request_body


# ---------------------------------------------------------------------------
# Touch chain
# ---------------------------------------------------------------------------


def test_rewrite_appends_touch_with_tier_rewrite(monkeypatch):
    """The returned Block carries a single new
    ``Touch(tier="rewrite", purpose="pedagogical_depth")`` appended to
    the input ``touched_by`` chain. Existing touches are preserved."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(_: httpx.Request) -> httpx.Response:
        body = "<section><p>The concept is anchored.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
    )

    # Pre-existing outline-tier Touch on the input block.
    pre_touch = Touch(
        model="qwen2.5:7b",
        provider="local",
        tier="outline",
        timestamp="2026-05-02T00:00:00Z",
        decision_capture_id="in-memory:0",
        purpose="draft",
    )
    block = Block(
        block_id="page#concept_x_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={"key_claims": ["c"], "curies": []},
        touched_by=(pre_touch,),
    )
    out = p.generate_rewrite(block)

    assert len(out.touched_by) == 2, "outline + rewrite touches expected"
    # Pre-existing touch preserved verbatim.
    assert out.touched_by[0] == pre_touch
    # New touch carries the rewrite-tier shape.
    new_touch = out.touched_by[1]
    assert new_touch.tier == "rewrite"
    assert new_touch.purpose == "pedagogical_depth"
    assert new_touch.provider == "local"
    # ``_apply_rewrite_touch`` resolves model from the constructor.
    assert new_touch.model == p._model
    # decision_capture_id is non-empty (Wave 112 invariant).
    assert new_touch.decision_capture_id


# ---------------------------------------------------------------------------
# Misc invariants
# ---------------------------------------------------------------------------


def test_supported_providers_includes_anthropic_together_local():
    """The module-level constant lists at least the three the base
    accepts. ``openai_compatible`` is reserved for a future plumbing
    pass and may or may not be present."""
    s = set(SUPPORTED_PROVIDERS)
    assert {"anthropic", "together", "local"}.issubset(s)
