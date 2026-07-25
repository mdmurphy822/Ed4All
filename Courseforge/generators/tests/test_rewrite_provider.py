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
    ENV_MAX_TOKENS,
    ENV_PROVIDER,
    RewriteProvider,
    RewriteProviderError,
    SUPPORTED_PROVIDERS,
    _BLOCK_TYPE_OUTPUT_CONTRACTS,
    _DEFAULT_MAX_TOKENS,
    _REWRITE_SYSTEM_PROMPT,
    _escape_orphan_placeholder_tags,
    _format_objectives,
    _objectives_for_block,
    _resolve_rewrite_max_tokens,
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
            # rewrite-overflow-fix-2026-06: a REALISTIC non-truncated
            # prompt-token tally. The rewrite system prompt alone is
            # ~7,800 tok, so the default-ON input-truncation tripwire
            # (reported < 0.5 * estimate ⇒ head dropped) would (correctly)
            # trip on the old unrealistic 200. A server that actually saw
            # the whole prompt reports a count near the estimate.
            "prompt_tokens": 8800,
            "completion_tokens": 80,
            "total_tokens": 8880,
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
            "source_refs": ["semantik:slug#blk1"],
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
# big-model-overflow-fix-2026-07 — max_tokens env override
# ---------------------------------------------------------------------------


def test_resolve_rewrite_max_tokens_default_when_env_unset(monkeypatch):
    """Env unset resolves to the 4096 default (bumped from the legacy
    7B-calibrated 2400)."""
    monkeypatch.delenv(ENV_MAX_TOKENS, raising=False)
    assert _DEFAULT_MAX_TOKENS == 4096
    assert _resolve_rewrite_max_tokens(None) == 4096


def test_resolve_rewrite_max_tokens_env_positive_int(monkeypatch):
    monkeypatch.setenv(ENV_MAX_TOKENS, "8192")
    assert _resolve_rewrite_max_tokens(None) == 8192


@pytest.mark.parametrize("bad", ["", "  ", "not-an-int", "0", "-5", "3.5"])
def test_resolve_rewrite_max_tokens_garbage_falls_back(monkeypatch, bad):
    """Garbage / non-positive env → the 4096 default (parse-with-fallback)."""
    monkeypatch.setenv(ENV_MAX_TOKENS, bad)
    assert _resolve_rewrite_max_tokens(None) == 4096


def test_resolve_rewrite_max_tokens_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_MAX_TOKENS, "8192")
    assert _resolve_rewrite_max_tokens(1500) == 1500


def test_rewrite_provider_threads_resolved_max_tokens(monkeypatch):
    """The resolved cap reaches ``RewriteProvider._max_tokens`` (env path)
    and an explicit kwarg still wins — on both the OpenAI-compatible (local)
    and the early-return claude_session paths."""
    monkeypatch.setenv(ENV_PROVIDER, "local")
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv(ENV_MAX_TOKENS, "6000")
    assert RewriteProvider()._max_tokens == 6000
    assert RewriteProvider(max_tokens=1234)._max_tokens == 1234
    # claude_session early-return branch also honors the resolved cap.
    p_cs = RewriteProvider(provider="claude_session", dispatcher=object())
    assert p_cs._max_tokens == 6000


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
                            "<section data-cf-source-ids=\"semantik:slug#blk1\">"
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
            "<section data-cf-source-ids=\"semantik:slug#blk1\">"
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
    """M3 tolerant preservation: a minted CURIE whose underlying TERM the
    rewrite tier actually used in prose (``slope``) is enforced and
    force-injected so it survives the validator's strip+extract path; a
    minted CURIE whose term the model did NOT use (``y_intercept``) is
    PRUNED from enforcement rather than forced — that's what stops the
    CURIE-churn truncation. The block still anchors >=1 CURIE."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        # Natural prose — uses "slope" but NOT "y intercept".
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
        curies=["introbio101:slope", "introbio101:y_intercept"],
    )
    out = p.generate_rewrite(block)
    # The on-topic CURIE (its term "slope" is in the prose) survives.
    assert _curie_survives_validator_path(out.content, "introbio101:slope"), (
        "on-topic minted CURIE 'introbio101:slope' did not survive the "
        "validator path"
    )
    # The off-topic CURIE (term "y intercept" absent from the prose) is
    # PRUNED — NOT force-injected. This is the M3 over-forcing fix.
    assert not _curie_survives_validator_path(
        out.content, "introbio101:y_intercept"
    ), (
        "off-topic minted CURIE 'introbio101:y_intercept' was force-injected "
        "even though the rewrite tier never used the term — M3 should prune it"
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
# M3 — tolerant CURIE preservation (CURIE-churn truncation fix)
# ---------------------------------------------------------------------------


def _curie_dense_block(curies: List[str]) -> Block:
    """Outline block whose key_claims mention exactly TWO of the CURIE terms
    (``multiple`` + ``prime factors``) but carry many declared CURIEs —
    the CURIE-dense shape the M3 diagnostic root-caused."""
    return Block(
        block_id="page#concept_lcm_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": [
                "A common multiple is shared; prime factors build it up."
            ],
            "curies": list(curies),
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
    )


_M3_DENSE_CURIES = [
    "democourse:multiple",
    "democourse:prime_factorization",
    "democourse:least_common_multiple",
    "democourse:prime_factors_method",
    "democourse:least_common_multiple_lcm",
    "democourse:order_of_operations",
    "democourse:evaluation",
    "democourse:prime_factors",
]


def test_m3_tolerant_preservation_enforces_only_used_terms(monkeypatch):
    """A CURIE-dense outline block (8 CURIEs) whose rewritten prose mentions
    only TWO of the underlying terms (``multiple`` + ``prime factors``):
    preservation keeps exactly those two (they survive the validator path)
    and PRUNES the other six — it does NOT force-all of them, which is what
    bloated the prose past max_tokens in the failed run."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Prose uses the two terms "multiple" and "prime factors" but echoes
        # none of the synthetic CURIE tokens.
        body = (
            "<section><p>A multiple is a product of an integer; "
            "prime factors are the building blocks.</p></section>"
        )
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_curie_dense_block(_M3_DENSE_CURIES))

    # The two USED terms are enforced -> survive (force-injected because the
    # model didn't echo the token).
    assert _curie_survives_validator_path(out.content, "democourse:multiple")
    assert _curie_survives_validator_path(
        out.content, "democourse:prime_factors"
    )
    # The six UNUSED CURIEs are pruned -> not injected.
    for unused in (
        "democourse:prime_factorization",
        "democourse:least_common_multiple",
        "democourse:prime_factors_method",
        "democourse:least_common_multiple_lcm",
        "democourse:order_of_operations",
        "democourse:evaluation",
    ):
        assert not _curie_survives_validator_path(out.content, unused), (
            f"unused CURIE {unused!r} was forced into the prose — M3 should "
            f"prune it to avoid max_tokens bloat"
        )


def test_m3_used_terms_do_not_trigger_force_all_retry(monkeypatch):
    """When the FIRST response already carries the enforced (on-topic) terms
    in pedagogical voice, the gate accepts on the first dispatch — no
    remediation re-send fires (the long force-all re-send is the truncation
    cause). Only ONE POST lands at the server."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Carry BOTH used CURIE tokens in code-voice so they're already
        # anchored in pedagogical context on the first emit.
        body = (
            "<section><p>The <code>democourse:multiple</code> and "
            "<code>democourse:prime_factors</code> concepts.</p>"
            "</section>"
        )
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_curie_dense_block(_M3_DENSE_CURIES))

    assert len(seen) == 1, (
        "force-all preservation re-send fired even though the enforced "
        "on-topic CURIEs were already anchored — M3 tolerance should accept "
        "on the first dispatch"
    )
    assert html_has_forced_curie_marker_local(out.content) is False


def test_m3_block_with_zero_used_terms_keeps_one_curie(monkeypatch):
    """A block whose prose mentions NONE of the CURIE terms still ends with
    EXACTLY ONE CURIE — the hard >=1 anchoring invariant the
    rewrite_curie_anchoring gate requires. The chosen CURIE is one of the
    outline-declared set (anti-fabrication: never invented)."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        # Prose uses NONE of the CURIE terms.
        body = "<section><p>Numbers can be combined in many ways.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_curie_dense_block(_M3_DENSE_CURIES))

    from Courseforge.router.inter_tier_gates import _strip_html
    from lib.ontology.curie_extraction import extract_curies

    surfaced = extract_curies(_strip_html(out.content))
    declared = set(_M3_DENSE_CURIES)
    kept = [c for c in surfaced if c in declared]
    assert len(kept) == 1, (
        f"expected exactly 1 anchoring CURIE under the >=1 invariant, "
        f"got {kept!r}"
    )
    # Anti-fabrication: the kept CURIE is from the outline-declared set.
    assert kept[0] in declared

    # End-to-end: the str-path anchoring gate PASSES on the kept >=1.
    from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator

    result = BlockCurieAnchoringValidator().validate({"blocks": [out]})
    assert result.passed, [i.code for i in result.issues]


def html_has_forced_curie_marker_local(html: str) -> bool:
    from Courseforge.generators._rewrite_provider import (
        html_has_forced_curie_marker,
    )

    return html_has_forced_curie_marker(html)


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


def test_supported_providers_is_registry_superset():
    """The module-level constant is the registry-superset allow-list:
    ``anthropic`` + every ``kind: openai_compatible`` registry endpoint
    (``local`` / ``together`` / ``nvidia`` + any future cloud row), PLUS
    the two non-registry-endpoint tags the rewrite tier handles specially
    (``claude_session`` subagent dispatch + the legacy ``openai_compatible``
    alias). Adding a provider is a registry-entry change, never a subclass."""
    from lib.llm.endpoints import load_endpoint_registry

    s = set(SUPPORTED_PROVIDERS)
    assert {"anthropic", "together", "local", "nvidia"}.issubset(s)
    assert {"claude_session", "openai_compatible"}.issubset(s)
    # Every openai-compatible registry seat is present (no hardcoded narrowing).
    registry_seats = {
        name
        for name, row in load_endpoint_registry().items()
        if str(row.get("kind")) == "openai_compatible"
    }
    assert registry_seats.issubset(s)


def test_registry_seat_constructs_and_stamps_valid_touch(monkeypatch):
    """A non-legacy registry endpoint (``groq``) that the router allowlist
    admits now constructs the rewrite tier without a ValueError, and its
    self-stamped Touch collapses to the seat's registry ``provenance_provider``
    (``groq`` → ``together``) so Touch validation passes — healing the
    router-allowlist / tier-enforcement split-brain."""
    from Courseforge.generators._rewrite_provider import (
        RewriteProvider,
        _touch_provenance,
    )

    class _FakeOA:
        model = "fake-model"
        base_url = "http://fake"

    p = RewriteProvider(provider="groq", client=_FakeOA())
    assert p._provider == "groq"
    # Registry provenance collapse keeps Touch.provider inside the closed set.
    assert _touch_provenance("groq") == "together"


def test_openai_compatible_alias_collapses_to_local():
    """The legacy ``openai_compatible`` alias collapses to ``local`` at
    constructor entry so a standalone construction matches the
    router-mediated one (and never hits the base's UnknownEndpoint branch)."""
    from Courseforge.generators._rewrite_provider import RewriteProvider

    class _FakeOA:
        model = "fake-model"
        base_url = "http://fake"

    p = RewriteProvider(provider="openai_compatible", client=_FakeOA())
    assert p._provider == "local"


def test_unknown_provider_still_fails_fast():
    """An unknown provider name is still rejected at construction (fail-fast),
    now via the base's registry-derived allow-list ValueError."""
    from Courseforge.generators._rewrite_provider import RewriteProvider

    class _FakeOA:
        model = "fake-model"
        base_url = "http://fake"

    with pytest.raises(ValueError):
        RewriteProvider(provider="definitely-not-a-provider", client=_FakeOA())


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


def test_rewrite_system_prompt_carries_pedagogical_depth_directive():
    """CB1 system-prompt sentinel: ``_REWRITE_SYSTEM_PROMPT`` must carry
    the grounded pedagogical-depth directive so the rewrite tier adds
    instructional depth (rationale / verification / second case / expert
    tip) WITHOUT fabricating material the source does not supply."""
    # The named directive block header.
    assert "PEDAGOGICAL DEPTH (grounded)" in _REWRITE_SYSTEM_PROMPT
    # Per-block-type depth requirements.
    assert "CONCEPTUAL RATIONALE" in _REWRITE_SYSTEM_PROMPT
    assert "VERIFICATION / CHECK" in _REWRITE_SYSTEM_PROMPT
    assert "two or more" in _REWRITE_SYSTEM_PROMPT  # >=2 worked examples
    assert "EXPERT TIP" in _REWRITE_SYSTEM_PROMPT
    # self_check: exactly one correct option, no multiple-true.
    assert "EXACTLY ONE" in _REWRITE_SYSTEM_PROMPT
    assert "simultaneously true" in _REWRITE_SYSTEM_PROMPT
    # Citation hygiene: no raw chunk tokens in visible prose.
    assert "CITATION HYGIENE" in _REWRITE_SYSTEM_PROMPT
    assert "[chunk_12]" in _REWRITE_SYSTEM_PROMPT
    # The load-bearing fabrication guard — depth is grounded, never
    # invented; the NLI gates fail closed on fabrication.
    assert "NEVER FABRICATED" in _REWRITE_SYSTEM_PROMPT
    assert "FAIL CLOSED on fabrication" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_system_prompt_carries_instruction_block_authoring_fixes():
    """Instruction-block authoring sentinel: ``_REWRITE_SYSTEM_PROMPT``
    must carry the four 7B-vs-Sonnet authoring-defect fixes — self-check
    answer-behind-reveal, assessment answer-key + value-distinct
    distractors, structured concept taxonomy, and per-example
    verification — all under the grounded (never-fabricated) constraint."""
    # Defect 1 — self-check answer lives behind a reveal, not inline.
    assert "NEVER REVEAL THE ANSWER INLINE" in _REWRITE_SYSTEM_PROMPT
    assert "<details><summary>Show answer</summary>" in _REWRITE_SYSTEM_PROMPT
    # Defect 2 — assessment items mark the correct answer and use
    # value-distinct (non-equivalent-form) distractors with rationales.
    assert "ALWAYS EMIT AN ANSWER KEY" in _REWRITE_SYSTEM_PROMPT
    assert "CLEARLY MARK THE CORRECT ANSWER" in _REWRITE_SYSTEM_PROMPT
    assert "DISTINCT IN VALUE" in _REWRITE_SYSTEM_PROMPT
    assert "EQUIVALENT FORM" in _REWRITE_SYSTEM_PROMPT
    assert "PER-DISTRACTOR RATIONALE" in _REWRITE_SYSTEM_PROMPT
    # Defect 3 — concept taxonomies use a structured table/list.
    assert "TAXONOMY / CATEGORIES / MULTIPLE TYPES" in _REWRITE_SYSTEM_PROMPT
    assert "STRUCTURED `<table>` or `<ul>`" in _REWRITE_SYSTEM_PROMPT
    # Defect 4 — the verification/check line is mandatory on EVERY
    # worked example, mirroring the explanation block.
    assert "MANDATORY ON EVERY WORKED EXAMPLE" in _REWRITE_SYSTEM_PROMPT
    # The per-block-type output contracts carry the same fixes.
    assert "Show answer" in _REWRITE_SYSTEM_PROMPT
    assert "data-cf-correct" in _REWRITE_SYSTEM_PROMPT
    # The hard fabrication guard still applies to everything added.
    assert "grounded in the source's own" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_assessment_directive_names_canonical_distractor_markup():
    """Canonical-MCQ-markup sentinel: the assessment_item directive must
    name the EXACT markup the W7 payload gate
    (``lib/validators/assessment_item_payload.py``) +
    distractor-plausibility gate parse — ``<li data-cf-distractor-index>``
    siblings — plus the correct-answer mechanism the
    assessment_retrieval_grounding gate reads (``data-cf-correct="true"``
    flag ON that <li>). A prior fix made the 7B emit an answer key in
    NON-canonical markup the validators couldn't see; this pins the
    parseable shape."""
    # The distractor-index attribute the W7 regex scrapes.
    assert "data-cf-distractor-index" in _REWRITE_SYSTEM_PROMPT
    # The correct-answer marker (NOT prose) the grounding gate reads.
    assert 'data-cf-correct="true"' in _REWRITE_SYSTEM_PROMPT
    # The directive names the 0-based contiguous-index contract.
    assert "0-based index" in _REWRITE_SYSTEM_PROMPT
    assert "EXACTLY ONE" in _REWRITE_SYSTEM_PROMPT
    # A concrete parseable example is present in the prompt.
    assert 'data-cf-distractor-index="0"' in _REWRITE_SYSTEM_PROMPT
    # The per-block-type output contract mirrors the canonical markup.
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["assessment_item"]
    assert "data-cf-distractor-index" in contract
    assert 'data-cf-correct="true"' in contract


def test_rewrite_assessment_example_parses_under_validator_regexes():
    """The concrete example embedded in the assessment directive must be
    parseable by the ACTUAL validator regexes — >=2
    ``<li data-cf-distractor-index="N">`` siblings (W7) AND a
    ``data-cf-correct="true"`` flag (retrieval grounding). Guards against
    a prompt example that drifts from the markup the validators parse."""
    from lib.validators.assessment_item_payload import (
        _DATA_CF_DISTRACTOR_INDEX_LI_RE,
    )
    from lib.validators.assessment_retrieval_grounding import _LI_CORRECT_RE

    matches = _DATA_CF_DISTRACTOR_INDEX_LI_RE.findall(_REWRITE_SYSTEM_PROMPT)
    # >=2 distractor-index siblings present, with index 0 contiguous.
    indices = sorted({int(idx) for idx, _ in matches})
    assert len(matches) >= 2
    assert 0 in indices
    # The correct-answer flag is detectable by the grounding regex.
    assert _LI_CORRECT_RE.search(_REWRITE_SYSTEM_PROMPT) is not None


def test_activity_contract_requires_items_consistent_with_instruction():
    """Defect A sentinel: the ``activity`` output contract must require every
    practice item to exercise the SAME operation/skill named in the
    activity's instruction line, drawn only from the source — closing the
    7B failure where a "simplify fractions" instruction listed integer
    expressions that don't exercise the stated skill."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["activity"]
    # Existing structural directives stay intact (additive contract).
    assert 'data-cf-component="activity"' in contract
    assert 'data-cf-purpose="practice"' in contract
    assert "data-cf-source-ids" in contract
    # New directive: items exercise the same operation/skill as the
    # instruction line, drawn only from the source.
    assert "EXERCISE THE SAME" in contract
    assert "instruction line" in contract
    # Names the concrete cross-type failure mode it forbids.
    assert "INTERNALLY CONSISTENT" in contract
    assert "do NOT list" in contract


def test_assessment_contract_requires_distractor_value_follows_misconception():
    """Defect B sentinel: the ``assessment_item`` output contract must
    require each distractor's VALUE to be the actual arithmetic consequence
    of its named misconception — closing the 7B failure where the
    distractor value didn't follow from the error its rationale named. All
    existing assessment directives stay intact (additive)."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["assessment_item"]
    # Existing directives preserved (canonical markup, correct-answer flag,
    # value-distinct distractors, per-distractor rationale).
    assert "data-cf-distractor-index" in contract
    assert 'data-cf-correct="true"' in contract
    assert "DISTINCT IN VALUE" in contract
    assert "rationale naming the misconception" in contract
    # New directive: the distractor value must be the arithmetic
    # consequence of the named misconception, not an arbitrary wrong value.
    assert "ACTUAL RESULT A STUDENT WHO MADE" in contract
    assert "ARITHMETIC CONSEQUENCE" in contract
    assert "not an arbitrary wrong value" in contract


def test_flip_card_grid_contract_constrains_card_front_to_domain_terms():
    """Defect A sentinel (iter-6): the ``flip_card_grid`` output contract must
    constrain each card FRONT to the block's supplied ``key_terms`` (or a
    domain term defined in the source) and explicitly DENY pedagogy /
    structural meta-words — closing the 7B failure where cards emitted
    ``Example`` / ``Exercise`` / ``Problem`` (dictionary definitions of the
    pedagogy meta-words) instead of the chapter's domain vocabulary. All
    iter-2 distinctness directives stay intact (additive)."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["flip_card_grid"]
    # iter-2 distinctness / no-repeat / no-procedure-fill directives preserved.
    assert 'data-cf-component="flip-card"' in contract
    assert "DISTINCT key term" in contract
    assert "NEVER repeat a term" in contract
    assert "generic procedure paragraph" in contract
    # New directive: the card front is one of the supplied key_terms / a
    # source-defined domain term, never a pedagogy meta-word.
    assert "key_terms" in contract
    assert "DOMAIN VOCABULARY" in contract
    assert "META-WORD" in contract
    # The explicit deny-list of pedagogy/structural meta-words.
    for meta_word in (
        "example",
        "exercise",
        "problem",
        "try it",
        "solution",
        "practice",
        "note",
        "activity",
        "summary",
    ):
        assert f"`{meta_word}`" in contract
    # The card back is the term's definition drawn from the source.
    assert "DEFINITION drawn from the source" in contract


def test_activity_contract_forbids_bare_exercise_numbers():
    """Defect B sentinel (iter-6): the ``activity`` output contract must
    require each practice item to STATE THE ACTUAL PROBLEM / TASK in full
    and forbid bare source exercise / reference numbers — closing the 7B
    failure where the activity listed bare textbook exercise indices
    (\"83, 84, 85\") as if they were practice items. The iter-5
    skill-consistency directive stays intact (additive)."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["activity"]
    # iter-5 skill-consistency directive preserved.
    assert "EXERCISE THE SAME" in contract
    assert "INTERNALLY CONSISTENT" in contract
    # New directive: each item states the actual problem/task in full.
    assert "STATE THE ACTUAL PROBLEM" in contract
    # New directive: never emit bare exercise/reference numbers.
    assert "BARE EXERCISE / REFERENCE NUMBERS" in contract
    assert "83, 84, 85" in contract
    assert "SOURCE EXERCISE INDICES" in contract


def test_prereq_set_contract_requires_prior_skills_not_own_objectives():
    """Defect sentinel (real algebra ch5 integers run): the ``prereq_set`` output
    contract must require each ``<ol>`` item to name a PRIOR foundational
    skill/topic the learner needs BEFORE this content, and explicitly forbid
    listing the chapter's OWN learning objectives or emitting a raw
    ``CO-NN`` / ``TO-NN`` objective id as a prerequisite — closing the 7B
    failure where the prereq_set block dumped the chapter's own COs
    (CO-25/CO-26/CO-27) as "prerequisites". The existing ``<ol>`` structural
    directive stays intact (additive contract)."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["prereq_set"]
    # Existing structural directive preserved.
    assert "data-cf-source-ids" in contract
    assert "`<ol>`" in contract
    # New directive: items name PRIOR foundational skills/topics needed
    # BEFORE this content, drawn from the source's stated prerequisites.
    assert "PRIOR FOUNDATIONAL SKILL" in contract
    assert "BEFORE this content" in contract
    assert "source's stated" in contract
    # New prohibition: never the chapter's own objectives, never a raw
    # CO-NN / TO-NN id, never restate this block's objective.
    assert "NEVER list the current chapter's OWN" in contract
    assert "`CO-NN` / `TO-NN`" in contract
    assert "NEVER restate" in contract
    # New fallback directive: when source states no prerequisites, list the
    # foundational concepts this content builds on, not its outcomes.
    assert "no explicit prerequisites" in contract
    assert "FOUNDATIONAL CONCEPTS this content builds on" in contract


def test_callout_contract_constrains_scope_to_single_focused_highlight():
    """Defect sentinel (real algebra ch5 + ch10 run): the ``callout`` output
    contract must constrain a callout to ONE focused, concise highlight —
    a single key tip / warning / caution / note — and explicitly forbid
    turning it into a full lesson with multi-example sequences or
    step-by-step worked solutions. Closes the 7B failure where callouts
    were authored as mini-lessons (ch5: a 942-char like/unlike-sign lesson
    with 4 worked examples; ch10: a multi-example "Rational or Irrational?"
    lesson). The existing structural directive stays intact (additive)."""
    contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["callout"]
    # Existing structural directive preserved (additive contract).
    assert 'class=\\"callout callout-{kind}\\"' in contract or (
        "callout callout-{kind}" in contract
    )
    assert 'data-cf-component=\\"callout\\"' in contract or (
        "data-cf-component" in contract
    )
    assert "data-cf-purpose" in contract
    assert 'data-cf-content-type=\\"callout\\"' in contract or (
        "data-cf-content-type" in contract
    )
    # New SCOPE directive: a callout is a single focused highlight.
    assert "ONE FOCUSED, CONCISE HIGHLIGHT" in contract
    assert "single" in contract
    # New prohibition: it must NOT be a full lesson / multi-example.
    assert "NOT A FULL LESSON" in contract
    assert "MULTI-EXAMPLE" in contract
    assert "step-by-step worked solutions" in contract
    # Names where the instructional content belongs instead.
    assert "concept / explanation / example" in contract
    # Stays grounded — no fabricated facts.
    assert "grounded in the source" in contract


# ---------------------------------------------------------------------------
# Lane "rewrite" authoring fixes (findings 1, 5, 12, 15, 16) — Sonnet-vs-7B
# manual review of a real 7B course export vs its Sonnet baseline.
# Each is a small prompt/parse assertion; the dispatched-prompt cases mock
# the client so no live model call fires.
# ---------------------------------------------------------------------------


def test_rewrite_step_label_colon_inside_label_finding_1():
    """Finding 1: the step-label badge must carry its colon INSIDE the
    ``<span class="step-label">`` (``Step N:</span>``), never after the
    closing tag (``Step N</span>:``) — otherwise the colon floats beside
    the flex badge. Both the styled-components directive and the
    ``example`` output contract must pin the inside-the-span form and name
    the wrong form as prohibited."""
    # The system prompt carries a dedicated STEP LABEL COLON directive.
    assert "STEP LABEL COLON" in _REWRITE_SYSTEM_PROMPT
    # The correct form (colon inside the span) is present...
    assert '<span class="step-label">Step N:</span>' in _REWRITE_SYSTEM_PROMPT
    # ...and the wrong form (colon outside) is named as prohibited.
    assert "NEVER `<span class=\"step-label\">Step N</span>:`" in (
        _REWRITE_SYSTEM_PROMPT
    )
    # The per-block-type ``example`` contract carries the same fix.
    example_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["example"]
    assert '<span class="step-label">Step N:</span>' in example_contract
    assert "the colon goes INSIDE the `step-label` span" in example_contract
    # Regression guard: the legacy colon-outside form must NOT survive in
    # either surface (the bug that floated the colon beside the badge).
    assert '<span class="step-label">Step N</span> …' not in (
        _REWRITE_SYSTEM_PROMPT
    )
    assert '<span class="step-label">Step N</span> ' not in example_contract


def test_rewrite_prefers_tables_and_compact_structure_finding_5():
    """Finding 5: the rewrite tier must emit an HTML ``<table>`` for
    comparison / multi-column content and prefer compact structured blocks
    over long prose (content_01 had 0 tables / verbose prose vs the
    baseline's comparison table)."""
    assert "STRUCTURED OVER PROSE" in _REWRITE_SYSTEM_PROMPT
    # Comparison / contrast / multi-column content → an HTML table.
    assert "COMPARES " in _REWRITE_SYSTEM_PROMPT
    assert "MULTI-COLUMN" in _REWRITE_SYSTEM_PROMPT
    assert "`<thead>`" in _REWRITE_SYSTEM_PROMPT
    # Prefer compact structure over long prose.
    assert "PREFER compact" in _REWRITE_SYSTEM_PROMPT
    # The concept + explanation contracts carry the comparison-table rule.
    concept_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "COMPARES / CONTRASTS" in concept_contract
    assert "`<thead>`" in concept_contract
    explanation_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["explanation"]
    assert "COMPARES / CONTRASTS" in explanation_contract
    assert "`<table>`" in explanation_contract


def test_rewrite_math_rendering_and_numeric_self_check_finding_12():
    """Finding 12: math must render as MathML or Unicode (NOT raw
    ``$...$`` LaTeX, which Studio does not render), and a numeric
    distractor's value must be consistent with its stated rationale
    (3/4 × 5/6 = 15/24, never 9/20)."""
    # Math-rendering directive: no raw LaTeX, render MathML or Unicode.
    assert "MATH RENDERING (Studio has NO LaTeX renderer)" in (
        _REWRITE_SYSTEM_PROMPT
    )
    assert "MathML" in _REWRITE_SYSTEM_PROMPT
    assert "Unicode glyphs" in _REWRITE_SYSTEM_PROMPT
    # The concrete failing literal from the review is named as prohibited.
    assert r"$\frac{9}{20}$" in _REWRITE_SYSTEM_PROMPT
    # Numeric self-check: distractor value must match its rationale.
    assert "NUMERIC SELF-CHECK" in _REWRITE_SYSTEM_PROMPT
    assert "15/24" in _REWRITE_SYSTEM_PROMPT
    assert "9/20" in _REWRITE_SYSTEM_PROMPT
    # The assessment_item contract carries both fixes.
    assessment_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["assessment_item"]
    assert "MATH RENDERING" in assessment_contract
    assert "NEVER raw `$...$`" in assessment_contract
    assert "NUMERIC SELF-CHECK" in assessment_contract
    assert "15/24" in assessment_contract


def test_rewrite_diverse_content_derived_headings_finding_15():
    """Finding 15: generate diverse, content-derived page titles + section
    framing headers — never a repeated generic "Objectives" label or the
    raw page filename."""
    assert "DIVERSE, CONTENT-DERIVED HEADINGS" in _REWRITE_SYSTEM_PROMPT
    # The generic labels + filename the 7B over-used are named as banned.
    assert "generic, repeated label" in _REWRITE_SYSTEM_PROMPT
    assert "week_01_summary" in _REWRITE_SYSTEM_PROMPT
    # Specific, learner-facing framing headers are the steer.
    assert "learner-facing framing header" in _REWRITE_SYSTEM_PROMPT
    # The concept contract carries the content-derived heading rule.
    concept_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "content-derived topic header" in concept_contract


def test_rewrite_key_idea_framing_finding_16():
    """Finding 16: key-rule / callout blocks must carry a recognizable
    "Key Idea"-style framing header (the 42 key-rule blocks lacked one and
    the callout type was authored 0 times)."""
    assert "KEY-IDEA FRAMING" in _REWRITE_SYSTEM_PROMPT
    assert '"Key Idea"' in _REWRITE_SYSTEM_PROMPT
    # The callout contract opens with a Key-Idea framing header.
    callout_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["callout"]
    assert "KEY-IDEA framing header" in callout_contract
    assert '"Key Idea"' in callout_contract
    # The concept + explanation key-rule boxes lead with the framing label.
    concept_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "Key Idea: Rule name" in concept_contract
    explanation_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["explanation"]
    assert '"Key Idea"' in explanation_contract


def test_rewrite_user_prompt_carries_authoring_framing_directives(monkeypatch):
    """The standard (non-escalated) rewrite user prompt dispatched to the
    client must carry the closing authoring-framing directives (findings
    5/12/15): content-derived headings, Unicode/MathML over LaTeX, and
    compact structured elements over long prose. Mocks the client so no
    live model call fires."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = "<section data-cf-source-ids=\"semantik:slug#blk1\"><p>X.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    block = _outline_block()  # concept, no escalation marker
    out = p.generate_rewrite(block)

    assert isinstance(out, Block)
    assert len(seen) == 1
    request_body = seen[0].read().decode("utf-8")
    # Content-derived heading steer (finding 15).
    assert "Title every heading from the SPECIFIC content" in request_body
    # Math rendering steer (finding 12).
    assert "Render math as Unicode or MathML, never raw" in request_body
    # Compact structured elements over long prose (finding 5).
    assert "Prefer compact structured elements" in request_body


def test_rewrite_escalated_prompt_carries_authoring_framing_directives(
    monkeypatch,
):
    """The escalated rewrite user prompt must ALSO carry the closing
    authoring-framing directives (findings 5/12/15) — an escalated block
    must not silently lose the heading / math / compactness steer. Mocks
    the client so no live model call fires."""
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = "<section data-cf-source-ids=\"semantik:slug#blk1\"><p>X.</p></section>"
        return httpx.Response(200, json=_success_body(body))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    block = _outline_block(escalation_marker="outline_budget_exhausted")
    out = p.generate_rewrite(block)

    assert isinstance(out, Block)
    assert len(seen) == 1
    request_body = seen[0].read().decode("utf-8")
    # Escalation branch is taken...
    assert "ESCALATED REWRITE" in request_body
    # ...and still carries the framing directives.
    assert "Title every heading from the SPECIFIC content" in request_body
    assert "Render math as Unicode or MathML, never raw" in request_body
    assert "Prefer compact structured elements" in request_body
