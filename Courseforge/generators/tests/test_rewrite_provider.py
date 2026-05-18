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
    _REWRITE_SYSTEM_PROMPT,
    _escape_orphan_placeholder_tags,
    _format_objectives,
    _objectives_for_block,
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


def _curie_survives_validator_path(html: str, curie: str) -> bool:
    """Run the EXACT str-path the rewrite-tier gate runs and report
    whether ``curie`` is recovered.

    The gate (``Courseforge/router/inter_tier_gates.py``) calls
    ``_strip_html`` FIRST — deleting every tag *and its attributes* —
    and only THEN ``extract_curies`` over the leftover text. A CURIE
    that lives only in an attribute value is destroyed; only a CURIE
    in TEXT CONTENT survives. This helper asserts the real contract,
    not the weaker "appears in the HTML string" check the pre-R1 test
    used (which an attribute satisfied even though the gate failed).
    """
    from Courseforge.router.inter_tier_gates import _strip_html
    from lib.ontology.curie_extraction import extract_curies

    return curie in extract_curies(_strip_html(html))


def test_curie_preservation_exhaustion_force_injects_curie(
    monkeypatch,
):
    """Every retry drops the CURIE → after ``MAX_PARSE_RETRIES + 1``
    dispatches the rewrite tier FORCE-INJECTS the still-missing CURIE
    as a hidden ``<span>`` whose TEXT CONTENT carries the CURIE token,
    rather than raising (v0.3.0 minted-CURIE propagation contract). The
    CURIE token survives the post-rewrite gate's ``_strip_html`` +
    ``extract_curies`` pipeline."""
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
    out = p.generate_rewrite(block)
    # Initial dispatch + MAX_PARSE_RETRIES (=2) more retries = 3 total.
    assert len(seen) == 3
    # The dropped CURIE was force-injected — and SURVIVES the actual
    # validator path (strip-html-then-extract). This is the assertion
    # the pre-R1 test got wrong: it checked the HTML string only, which
    # an attribute satisfied even though the gate's strip destroyed it.
    assert _curie_survives_validator_path(out.content, "sh:NodeShape")
    # data-cf-curie contract attribute is mirrored onto the span.
    assert 'data-cf-curie="sh:NodeShape"' in out.content


def test_force_injected_curie_passes_block_curie_anchoring_validator(
    monkeypatch,
):
    """Run ``BlockCurieAnchoringValidator``'s str-path over a Block
    carrying force-injected HTML and assert the gate PASSES — the
    end-to-end contract the R1 fix exists to satisfy."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        body = "<section><p>The node shape constrains.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    block = _outline_block(curies=["sh:NodeShape"])
    out = p.generate_rewrite(block)

    from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator

    result = BlockCurieAnchoringValidator().validate({"blocks": [out]})
    assert result.passed, (
        f"force-injected CURIE failed the str-path anchoring gate: "
        f"{[i.code for i in result.issues]}"
    )


def test_minted_curies_from_source_block_survive_into_html(monkeypatch):
    """A source block whose ``curies`` carry minted (prose-corpus)
    CURIEs that the rewrite LLM declines to echo are force-injected so
    every minted CURIE survives the validator's strip+extract path."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        # Natural prose — none of the minted CURIE tokens echoed.
        body = (
            "<section><p>Slope measures the steepness of a line.</p>"
            "</section>"
        )
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
    )
    block = _outline_block(
        curies=["openstaxalg9:slope", "openstaxalg9:y_intercept"],
    )
    out = p.generate_rewrite(block)
    for curie in ("openstaxalg9:slope", "openstaxalg9:y_intercept"):
        assert _curie_survives_validator_path(out.content, curie), (
            f"minted CURIE {curie!r} did not survive the validator path"
        )


def test_force_injected_block_carries_durable_signals(monkeypatch):
    """R6 — force-injection stamps two durable, distinguishable signals
    so a force-injected block is NOT indistinguishable from a clean
    rewrite downstream:

    * the appended span carries ``data-cf-curie-forced="true"`` inside
      ``block.content`` (the report-reachable carrier — content is the
      one Block field that survives every JSONL round trip), detected
      by ``html_has_forced_curie_marker``;
    * the rewrite Touch carries ``purpose="curie_force_injected"``
      (the in-memory / JSON-LD audit-chain carrier) instead of the
      clean-path ``pedagogical_depth``.
    """
    from Courseforge.generators._rewrite_provider import (
        html_has_forced_curie_marker,
        _TOUCH_PURPOSE_CURIE_FORCED,
    )

    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        # Always drop the CURIE -> exhaust the budget -> force-inject.
        body = "<section><p>The node shape constrains.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    block = _outline_block(curies=["sh:NodeShape"])
    out = p.generate_rewrite(block)

    # Content-borne marker (the report-reachable signal).
    assert html_has_forced_curie_marker(out.content) is True
    assert 'data-cf-curie-forced="true"' in out.content

    # Touch-chain audit signal — the rewrite tier's last Touch.
    rewrite_touches = [t for t in out.touched_by if t.tier == "rewrite"]
    assert rewrite_touches, "rewrite tier appended no Touch"
    assert rewrite_touches[-1].purpose == _TOUCH_PURPOSE_CURIE_FORCED


def test_clean_rewrite_carries_no_force_injected_signals(monkeypatch):
    """R6 negative — a clean rewrite (no CURIE drop, so no
    force-injection) carries neither the ``data-cf-curie-forced``
    marker nor the ``curie_force_injected`` Touch purpose; the rewrite
    Touch keeps the clean-path ``pedagogical_depth`` purpose."""
    from Courseforge.generators._rewrite_provider import (
        html_has_forced_curie_marker,
    )

    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        body = "<section><p>The concept is anchored.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    # No CURIEs declared -> the preservation gate never fires, so the
    # rewrite returns via the clean ``_apply_rewrite_touch`` path.
    block = _outline_block(curies=[])
    out = p.generate_rewrite(block)

    assert html_has_forced_curie_marker(out.content) is False
    rewrite_touches = [t for t in out.touched_by if t.tier == "rewrite"]
    assert rewrite_touches[-1].purpose == "pedagogical_depth"


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
# Per-block objectives filtering (2026-05-15 Qwen prompt-flood regression)
# ---------------------------------------------------------------------------


_COURSE_OBJECTIVES = [
    {"id": "TO-01", "statement": "Apply place value to whole numbers."},
    {"id": "TO-02", "statement": "Evaluate algebraic expressions."},
    {"id": "TO-03", "statement": "Apply rules for integer arithmetic."},
]


def test_objectives_for_block_filters_to_declared_ids():
    """A block declaring one objective_id keeps only that objective."""
    block = Block(
        block_id="page#objective_pv_0",
        block_type="objective",
        page_id="page",
        sequence=0,
        content="",
        objective_ids=("TO-01",),
    )
    out = _objectives_for_block(_COURSE_OBJECTIVES, block)
    assert [o["id"] for o in out] == ["TO-01"]


def test_objectives_for_block_falls_back_to_full_list_when_none_declared():
    """A block with no objective_ids (chrome / callout) keeps the full
    list — filtering to an empty set would render an empty prompt."""
    block = Block(
        block_id="page#callout_note_0",
        block_type="callout",
        page_id="page",
        sequence=0,
        content="",
    )
    out = _objectives_for_block(_COURSE_OBJECTIVES, block)
    assert [o["id"] for o in out] == ["TO-01", "TO-02", "TO-03"]


def test_objectives_for_block_falls_back_when_declared_id_unresolvable():
    """When a declared objective_id matches nothing in the supplied list
    (upstream data mismatch), fall back to the full list rather than
    emitting an empty objectives block."""
    block = Block(
        block_id="page#objective_x_0",
        block_type="objective",
        page_id="page",
        sequence=0,
        content="",
        objective_ids=("TO-99",),
    )
    out = _objectives_for_block(_COURSE_OBJECTIVES, block)
    assert [o["id"] for o in out] == ["TO-01", "TO-02", "TO-03"]


def test_escalated_objective_prompt_omits_unrelated_objectives(monkeypatch):
    """Regression: the escalated rewrite prompt for an objective block
    declaring only ``TO-01`` must NOT enumerate every course objective.

    Observed 2026-05-15 on Qwen-14B — handing the model all nine course
    objectives made it emit ``data-cf-objective-id="TO-01,...,TO-09"`` and
    collapse the prose into one run-on sentence covering every objective.
    The fix filters the prompt's ``Objectives:`` block to the block's
    declared ``objective_ids``.
    """
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = (
            '<li data-cf-block-id="page#objective_pv_0" '
            'data-cf-objective-id="TO-01" data-cf-bloom-level="apply">'
            "Apply place value to whole numbers.</li>"
        )
        return httpx.Response(200, json=_success_body(body))

    provider = RewriteProvider(provider="local", client=_make_client(handler))
    block = Block(
        block_id="page#objective_pv_0",
        block_type="objective",
        page_id="page",
        sequence=0,
        content="",
        objective_ids=("TO-01",),
        escalation_marker="outline_budget_exhausted",
    )
    provider.generate_rewrite(block, objectives=_COURSE_OBJECTIVES)

    assert len(seen) == 1
    request_body = seen[0].read().decode("utf-8")
    assert "Apply place value to whole numbers." in request_body
    assert "TO-02" not in request_body
    assert "TO-03" not in request_body


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
# Orphan-tag sanitizer (post-emit)
# ---------------------------------------------------------------------------


def test_sanitizer_escapes_rdf_triple_placeholder():
    """The canonical Qwen-7B-Q4 failure case: bare <subject> <predicate>
    <object> with no closers. All three openers must be escaped."""
    html = (
        "<section><p>An RDF statement is a "
        "<subject> <predicate> <object> triple.</p></section>"
    )
    out = _escape_orphan_placeholder_tags(html)
    assert "&lt;subject&gt;" in out
    assert "&lt;predicate&gt;" in out
    assert "&lt;object&gt;" in out
    assert "<subject>" not in out
    assert "<predicate>" not in out
    assert "<object>" not in out
    # Real elements (with closers) are untouched.
    assert "<section>" in out
    assert "</section>" in out
    assert "<p>" in out
    assert "</p>" in out


def test_sanitizer_leaves_real_elements_with_closers_untouched():
    """``<section>...</section>`` is a balanced real element. No escape."""
    html = "<section><p>Hello</p></section>"
    out = _escape_orphan_placeholder_tags(html)
    assert out == html


def test_sanitizer_leaves_attribute_bearing_elements_untouched():
    """Real attribute-bearing elements never match the bare-opener regex."""
    html = (
        '<section data-cf-block-id="x#concept_0" '
        'data-cf-content-type="definition">'
        "<p>Body</p></section>"
    )
    out = _escape_orphan_placeholder_tags(html)
    assert out == html


def test_sanitizer_leaves_void_elements_untouched():
    """``<br>`` / ``<hr>`` / ``<img>`` are HTML5 void elements; they
    legitimately appear without a closer and must NOT be escaped."""
    html = "<p>Line one<br>Line two<hr><img></p>"
    out = _escape_orphan_placeholder_tags(html)
    assert out == html


def test_sanitizer_escapes_orphan_object_even_though_object_is_real_html_element():
    """``<object>`` IS a real HTML5 element, but in an RDF triple it's
    a placeholder with no closer. The sanitizer escapes orphan-openers
    regardless of whether the tag name appears in the HTML5 element
    set — the closer-presence check is the discriminator."""
    html = "<p>The triple has a subject, predicate, and <object> slot.</p>"
    out = _escape_orphan_placeholder_tags(html)
    assert "&lt;object&gt;" in out


def test_sanitizer_escapes_curie_in_brackets():
    """Placeholder CURIEs like ``<rdf:type>`` need the same escape — the
    colon is allowed in the bare-opener regex."""
    html = "<p>The <rdf:type> predicate types the focus node.</p>"
    out = _escape_orphan_placeholder_tags(html)
    assert "&lt;rdf:type&gt;" in out
    assert "<rdf:type>" not in out


def test_sanitizer_preserves_balanced_object_when_real_element():
    """If the model legitimately uses ``<object>...</object>`` (e.g. for
    embedded content), the closer is found and the sanitizer leaves it."""
    html = '<object data="x.svg">fallback</object>'
    out = _escape_orphan_placeholder_tags(html)
    assert out == html


def test_sanitizer_handles_mixed_orphan_and_real_in_same_string():
    """A single string containing BOTH an orphan placeholder AND a
    real paired element: only the orphan is escaped."""
    html = (
        "<section><p>Triple: <subject> <predicate> <object>.</p>"
        "<p>Closing: <code>rdf:type</code>.</p></section>"
    )
    out = _escape_orphan_placeholder_tags(html)
    assert "&lt;subject&gt;" in out
    assert "&lt;predicate&gt;" in out
    assert "&lt;object&gt;" in out
    # <code>...</code> is balanced; left alone.
    assert "<code>rdf:type</code>" in out


def test_sanitizer_idempotent():
    """Running the sanitizer twice must not double-escape — the first
    pass replaces ``<x>`` with ``&lt;x&gt;``, and the second pass sees
    no bare openers to match."""
    html = "<p>An <orphan> placeholder.</p>"
    once = _escape_orphan_placeholder_tags(html)
    twice = _escape_orphan_placeholder_tags(once)
    assert once == twice
    assert "&lt;orphan&gt;" in twice


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


# ---------------------------------------------------------------------------
# Wave 1.7 W1.7.B — Bloom-triple objective rendering + behavioral-outcome
# rewrite-tier system-prompt directive
# ---------------------------------------------------------------------------


def test_format_objectives_surfaces_bloom_triple_for_dict_shape():
    """Wave 1.7 W1.7.B golden-output regression: dict-shape objective
    carrying ``bloom_level`` + ``bloom_verb`` is rendered with the
    verbatim ``[Bloom: <level>, verb: <verb>]`` triple inline so the
    rewrite-tier model sees the declared cognitive demand pinned next
    to the behavioral outcome it must teach."""
    objs = [
        {
            "id": "TO-04",
            "statement": "Construct an OWL ontology in Turtle.",
            "bloom_level": "create",
            "bloom_verb": "construct",
        }
    ]
    rendered = _format_objectives(objs)
    assert "TO-04" in rendered
    assert "[Bloom: create, verb: construct]" in rendered
    assert "Construct an OWL ontology in Turtle." in rendered
    # The structured shape MUST line up as a single dash-prefixed entry.
    assert rendered.startswith("- TO-04 ")


def test_format_objectives_falls_back_to_legacy_shape_when_bloom_absent():
    """Back-compat: legacy fixtures that don't carry ``bloom_level`` /
    ``bloom_verb`` on the objective dict still render unambiguously
    via the legacy ``- {oid}: {statement}`` shape (no bracketed Bloom
    triple). Pre-Wave-1.7 corpora must not see a regression."""
    objs = [
        {
            "id": "CO-01",
            "statement": "Define the central concept.",
        }
    ]
    rendered = _format_objectives(objs)
    assert rendered == "- CO-01: Define the central concept."
    # No bracketed Bloom triple emitted when both fields are absent.
    assert "[Bloom:" not in rendered


def test_rewrite_system_prompt_carries_behavioral_outcome_directive():
    """Wave 1.7 W1.7.B system-prompt sentinel: ``_REWRITE_SYSTEM_PROMPT``
    must carry the ``MUST teach the BEHAVIORAL OUTCOME`` substring so
    the rewrite-tier model is steered toward delivering pedagogy at or
    above the declared Bloom level of each block's
    ``objective_refs``."""
    assert "MUST teach the BEHAVIORAL OUTCOME" in _REWRITE_SYSTEM_PROMPT
    # Cross-checks: the directive paragraph names the ``bloom_verb``
    # field explicitly so the model knows what surface form to deliver.
    assert "bloom_verb" in _REWRITE_SYSTEM_PROMPT
    # The directive references the per-Bloom-tier obligations
    # (apply / analyze → scaffolded reasoning; evaluate / create →
    # comparison / synthesis / construction prose).
    assert "scaffolded reasoning" in _REWRITE_SYSTEM_PROMPT
