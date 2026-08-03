"""Wave 1.5 W1.5.D regression tests — per-claim source attribution.

Threads the structured ``key_claims[].source_chunk_ids[]`` shape through
the rewrite-tier prompt-builder so the rewrite author has the citation
map salient as a distinct prompt block (instead of buried inside the
serialised outline JSON blob). Symmetric across the regular and
escalated paths.

Plan: ``plans/gpt-feedback-2-wave1.5-per-claim-attribution-2026-05.md``
§2 Fix 1.5.D.

Coverage:

1. Structured-shape outline → rendered prompt body contains the
   per-claim citation map block (substring assertions).
2. Legacy-shape outline → rendered prompt body contains the
   "(legacy shape — no per-claim citation)" annotation.
3. Escalated path → same per-claim block appears in escalated prompt.
4. Prompt-size growth bounded: structured 5-claim block adds
   < 500 chars vs the legacy-shape baseline (calibration for §6.3
   prompt-size risk budget).
5. System-prompt regression: ``_REWRITE_SYSTEM_PROMPT`` carries the
   "Per-claim source attribution" directive sentinel.
6. ``content["key_claims"] = []`` → "(no claims)" placeholder.
7. ``content`` is not a dict (e.g. raw string) →
   "(none — outline content not a dict)" placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    RewriteProvider,
    _REWRITE_SYSTEM_PROMPT,
    _format_per_claim_citations,
)
from blocks import Block  # noqa: E402  (Phase 2 intermediate format)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_block(
    *,
    claims: Optional[List[dict]] = None,
    escalation_marker: Optional[str] = None,
) -> Block:
    """Build an outline-tier Block with structured key_claims."""
    if claims is None:
        claims = [
            {
                "claim": "A SHACL node shape declares constraints on RDF nodes.",
                "source_chunk_ids": ["semantik:source-delta#sec1"],
            },
            {
                "claim": "Targets fire when a node fails a property predicate.",
                "source_chunk_ids": [
                    "semantik:source-delta#sec2",
                    "semantik:source-delta#sec3",
                ],
            },
        ]
    return Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": claims,
            "curies": ["sh:NodeShape"],
            "source_refs": [
                "semantik:source-delta#sec1",
                "semantik:source-delta#sec2",
                "semantik:source-delta#sec3",
            ],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


def _legacy_block(
    *,
    claims: Optional[List[str]] = None,
    escalation_marker: Optional[str] = None,
) -> Block:
    """Build an outline-tier Block with legacy List[str] key_claims."""
    if claims is None:
        claims = [
            "A SHACL node shape declares constraints on RDF nodes.",
            "Targets fire when a node fails a property predicate.",
        ]
    return Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": claims,
            "curies": ["sh:NodeShape"],
            "source_refs": ["semantik:source-delta#sec1"],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


def _provider(monkeypatch) -> RewriteProvider:
    """Cheap RewriteProvider that bypasses any real network plumbing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    return RewriteProvider(anthropic_client=object())


# ---------------------------------------------------------------------------
# Helper formatter unit tests
# ---------------------------------------------------------------------------


def test_format_per_claim_citations_structured_shape():
    """Structured shape renders the verbatim claim text + chunk_id list."""
    content = {
        "key_claims": [
            {"claim": "Foo statement.", "source_chunk_ids": ["chunk_A"]},
            {
                "claim": "Bar statement.",
                "source_chunk_ids": ["chunk_B", "chunk_C"],
            },
        ]
    }
    rendered = _format_per_claim_citations(content)
    assert "claim 1: " in rendered
    assert "claim 2: " in rendered
    assert "Foo statement." in rendered
    assert "Bar statement." in rendered
    assert "chunk_A" in rendered
    assert "chunk_B, chunk_C" in rendered


def test_format_per_claim_citations_legacy_shape():
    """Legacy List[str] shape carries the no-per-claim-citation marker."""
    content = {"key_claims": ["A bare-string claim."]}
    rendered = _format_per_claim_citations(content)
    assert "claim 1: " in rendered
    assert "A bare-string claim." in rendered
    assert "legacy shape — no per-claim citation" in rendered


def test_format_per_claim_citations_no_claims_placeholder():
    """Empty key_claims renders the (no claims) placeholder."""
    assert _format_per_claim_citations({"key_claims": []}) == "(no claims)"
    # Missing key altogether is the same — no claims to render.
    assert _format_per_claim_citations({}) == "(no claims)"


def test_format_per_claim_citations_non_dict_content_placeholder():
    """Non-dict content (e.g. legacy Phase 1 raw string) renders the placeholder."""
    assert (
        _format_per_claim_citations("raw string content")
        == "(none — outline content not a dict)"
    )
    assert (
        _format_per_claim_citations(None)
        == "(none — outline content not a dict)"
    )


def test_format_per_claim_citations_truncates_long_claim_text():
    """Per §6.3: claim text truncated at 80 chars to bound prompt growth."""
    long_claim = "A" * 200
    content = {
        "key_claims": [
            {"claim": long_claim, "source_chunk_ids": ["chunk_X"]},
        ]
    }
    rendered = _format_per_claim_citations(content)
    # Truncated form is the first 77 chars + "..." = 80 chars total.
    assert "A" * 77 + "..." in rendered
    assert "A" * 200 not in rendered


# ---------------------------------------------------------------------------
# Test 1: structured-shape outline surfaces in the regular prompt body
# ---------------------------------------------------------------------------


def test_render_user_prompt_contains_per_claim_block_structured(monkeypatch):
    """Structured outline → prompt body carries the per-claim citation map."""
    p = _provider(monkeypatch)
    block = _structured_block()
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    assert "Per-claim source attribution" in prompt
    assert "claim 1: " in prompt
    assert "claim 2: " in prompt
    # Single chunk_id verbatim.
    assert "[semantik:source-delta#sec1]" in prompt
    # Multiple chunk_ids comma-separated verbatim.
    assert "[semantik:source-delta#sec2, semantik:source-delta#sec3]" in prompt


# ---------------------------------------------------------------------------
# Test 2: legacy-shape outline annotation
# ---------------------------------------------------------------------------


def test_render_user_prompt_contains_legacy_annotation(monkeypatch):
    """Legacy outline → prompt body carries the legacy-shape annotation."""
    p = _provider(monkeypatch)
    block = _legacy_block()
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    assert "Per-claim source attribution" in prompt
    assert "legacy shape — no per-claim citation" in prompt


# ---------------------------------------------------------------------------
# Test 3: escalated path symmetry
# ---------------------------------------------------------------------------


def test_render_escalated_user_prompt_contains_per_claim_block(monkeypatch):
    """Escalated path → same per-claim citation block appears."""
    p = _provider(monkeypatch)
    block = _structured_block(escalation_marker="outline_budget_exhausted")
    prompt = p._render_escalated_user_prompt(
        block=block, source_chunks=[], objectives=[]
    )
    assert "ESCALATED REWRITE" in prompt
    assert "Per-claim source attribution" in prompt
    assert "claim 1: " in prompt
    assert "claim 2: " in prompt
    assert "[semantik:source-delta#sec1]" in prompt
    assert "[semantik:source-delta#sec2, semantik:source-delta#sec3]" in prompt


# ---------------------------------------------------------------------------
# Test 4: prompt-size growth is bounded (< 500 chars on a 5-claim block)
# ---------------------------------------------------------------------------


def test_prompt_size_growth_bounded_on_five_claim_block(monkeypatch):
    """5-claim block: structured shape adds < 500 chars vs legacy shape."""
    p = _provider(monkeypatch)

    structured_claims = [
        {
            "claim": f"Claim number {i} body — short prose statement.",
            "source_chunk_ids": [
                f"semantik:s#c{i}",
                f"semantik:s#c{i + 100}",
            ][: 1 + (i % 3)],
        }
        for i in range(1, 6)
    ]
    legacy_claims = [
        f"Claim number {i} body — short prose statement." for i in range(1, 6)
    ]

    structured_block = _structured_block(claims=structured_claims)
    legacy_block = _legacy_block(claims=legacy_claims)

    structured_prompt = p._render_user_prompt(
        block=structured_block, source_chunks=[], objectives=[]
    )
    legacy_prompt = p._render_user_prompt(
        block=legacy_block, source_chunks=[], objectives=[]
    )

    delta = len(structured_prompt) - len(legacy_prompt)
    # Calibration assertion for plan §6.3 prompt-size budget.
    assert delta < 500, (
        f"per-claim citation map added {delta} chars; budget is < 500. "
        f"Tighten the truncation rule or shrink the block header."
    )


# ---------------------------------------------------------------------------
# Test 5: system-prompt regression — directive sentinel present
# ---------------------------------------------------------------------------


def test_rewrite_system_prompt_carries_per_claim_directive():
    """System prompt grew the per-claim attribution SHOULD-grade directive."""
    # Directive sentinel — verbatim from plan §2 Fix 1.5.D.
    assert "per-claim source attribution" in _REWRITE_SYSTEM_PROMPT
    assert "key_claims[].source_chunk_ids[]" in _REWRITE_SYSTEM_PROMPT
    # SHOULD (not MUST) — per-sentence cite markers in HTML are an
    # explicit Wave 1.5 non-goal (plan §7).
    assert "SHOULD" in _REWRITE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Test 6: edge case — empty key_claims renders the "(no claims)" placeholder
# ---------------------------------------------------------------------------


def test_render_user_prompt_no_claims_edge_case(monkeypatch):
    """content["key_claims"] = [] → "(no claims)" placeholder appears in prompt."""
    p = _provider(monkeypatch)
    block = Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": [],
            "curies": [],
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
    )
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    assert "Per-claim source attribution" in prompt
    assert "(no claims)" in prompt


# ---------------------------------------------------------------------------
# Test 7: edge case — non-dict content renders the "(none — ...)" placeholder
# ---------------------------------------------------------------------------


def test_render_user_prompt_non_dict_content_edge_case(monkeypatch):
    """content as raw string → "(none — outline content not a dict)" placeholder."""
    p = _provider(monkeypatch)
    block = Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content="raw string content",
    )
    prompt = p._render_user_prompt(block=block, source_chunks=[], objectives=[])
    assert "Per-claim source attribution" in prompt
    assert "(none — outline content not a dict)" in prompt
