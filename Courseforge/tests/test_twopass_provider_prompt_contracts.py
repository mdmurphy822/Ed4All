"""Tests for Wave5-W27 propagation to the two-pass router prompt builders.

Wave4-W27 (`ffe517d`) added three MANDATORY directives to the single-pass
`content-generator` subagent prompt (`Courseforge/agents/content-generator.md`):

1. Strict h1 → h2 → h3 → h4 heading hierarchy (cites HEADING_SKIP gate).
2. `data-cf-source-ids` stamping on content wrappers.
3. `data-cf-objective-id` stamping tied to canonical TO-NN / CO-NN LO IDs.

Wave4-I10 (`ccd6374`) promoted `EMPTY_SOURCE_REFS` to a fail-closed critical
gate. When `COURSEFORGE_TWO_PASS=true`, the single-pass `content-generator`
subagent prompt is bypassed; the workflow splits into outline tier
(`OutlineProvider`) + rewrite tier (`RewriteProvider`), both in-process. This
test file enforces that the two providers carry tier-appropriate adaptations
of the three Wave-27 directives so the gates fire correctly on two-pass runs:

- The outline tier emits structured JSON (not HTML), so HEADING_SKIP doesn't
  apply at the outline seam — but `source_refs[]` / `objective_refs[]` MUST
  populate so the rewrite tier has the data to stamp on the rendered HTML.
- The rewrite tier emits HTML, so all three directives apply (heading
  hierarchy + source-id stamping + objective-id stamping).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators.outline._outline_provider import (  # noqa: E402
    ENV_PROVIDER as OUTLINE_ENV_PROVIDER,
    OutlineProvider,
    _OUTLINE_SYSTEM_PROMPT,
)
from Courseforge.generators.rewrite._rewrite_provider import (  # noqa: E402
    ENV_PROVIDER as REWRITE_ENV_PROVIDER,
    RewriteProvider,
    _REWRITE_SYSTEM_PROMPT,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — minimal Block fixtures parallel to the per-provider tests.
# ---------------------------------------------------------------------------


def _outline_block_fixture(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
    page_id: str = "page-1",
) -> Block:
    """Minimal Block for outline-tier prompt rendering."""
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=0,
        content="",
    )


def _rewrite_block_fixture(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
    page_id: str = "page-1",
    source_refs: List[str] | None = None,
    objective_refs: List[str] | None = None,
) -> Block:
    """Outline-dict Block as it lands at the rewrite tier."""
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=0,
        content={
            "key_claims": ["The central concept is X."],
            "curies": [],
            "source_refs": source_refs or ["semantik:slug#blk1"],
            "objective_refs": objective_refs or ["TO-01"],
        },
    )


# ---------------------------------------------------------------------------
# Outline-tier — block-dict metadata field directives.
# ---------------------------------------------------------------------------


def test_outline_system_prompt_carries_wave27_source_refs_directive():
    """`source_refs[]` directive MUST be in the outline-tier system prompt
    so the model populates the block dict with SemantiK source-chunk IDs.
    The rewrite tier reads `source_refs[]` to stamp the HTML
    `data-cf-source-ids` attribute (Wave4-I10 EMPTY_SOURCE_REFS critical
    gate)."""
    assert "source_refs[]" in _OUTLINE_SYSTEM_PROMPT
    # The directive cites the downstream rewrite-tier stamping
    # consumer so the cause-and-effect is explicit.
    assert "data-cf-source-ids" in _OUTLINE_SYSTEM_PROMPT
    # EMPTY_SOURCE_REFS by name pins Wave4-I10's critical gate.
    assert "EMPTY_SOURCE_REFS" in _OUTLINE_SYSTEM_PROMPT


def test_outline_system_prompt_carries_wave27_objective_refs_directive():
    """`objective_refs[]` directive MUST be in the outline-tier system
    prompt so the model populates the block dict with canonical TO-NN /
    CO-NN LO IDs. The rewrite tier reads `objective_refs[]` to stamp the
    HTML `data-cf-objective-id` attribute."""
    assert "objective_refs[]" in _OUTLINE_SYSTEM_PROMPT
    assert "data-cf-objective-id" in _OUTLINE_SYSTEM_PROMPT
    # The canonical TO-NN / CO-NN pattern reference must appear so the
    # model knows the LO ID shape it must preserve verbatim.
    assert "TO-NN" in _OUTLINE_SYSTEM_PROMPT
    assert "CO-NN" in _OUTLINE_SYSTEM_PROMPT


def test_outline_user_prompt_carries_wave27_stamping_contract(monkeypatch):
    """The rendered outline user prompt mentions source_refs[] +
    objective_refs[] + canonical TO-NN/CO-NN pattern so the per-call
    rendering surfaces the Wave-27 contract inline (not just in the
    system prompt). The outline tier emits JSON, not HTML, so HEADING_SKIP
    is intentionally absent at this seam."""
    monkeypatch.delenv(OUTLINE_ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)

    p = OutlineProvider(provider="local")
    block = _outline_block_fixture()
    rendered = p._render_user_prompt(
        block=block,
        source_chunks=[{"id": "semantik:slug#blk1", "body": "Source body."}],
        objectives=[{"id": "TO-01", "statement": "Define X."}],
    )

    assert "source_refs[]" in rendered
    # objective_refs[] is the outline-block-dict field name; the
    # canonical-pattern directive must be present so the model preserves
    # the LO ID shape verbatim.
    assert "objective_refs[]" in rendered or "objective_refs" in rendered
    # Canonical TO-NN / CO-NN pattern — load-bearing for the downstream
    # rewrite tier's data-cf-objective-id stamp.
    assert "TO-NN" in rendered
    assert "CO-NN" in rendered


# ---------------------------------------------------------------------------
# Rewrite-tier — HTML output contract directives (all three Wave-27 dirs).
# ---------------------------------------------------------------------------


def test_rewrite_system_prompt_carries_heading_hierarchy_directive():
    """The rewrite-tier system prompt MUST carry the W-27 heading-hierarchy
    directive. The rewrite tier emits the canonical published HTML, so
    HEADING_SKIP applies here — emitted pages that skip levels fail closed
    at the `ContentStructureValidator` critical gate."""
    assert "HEADING_SKIP" in _REWRITE_SYSTEM_PROMPT
    # Strict h1 → h2 → h3 progression — the canonical W-27 wording.
    assert "h1" in _REWRITE_SYSTEM_PROMPT
    assert "h2" in _REWRITE_SYSTEM_PROMPT
    assert "h3" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_system_prompt_carries_source_id_stamping_directive():
    """The rewrite-tier system prompt MUST carry the W-27 source-id
    stamping directive. EMPTY_SOURCE_REFS is fail-closed critical
    post-Wave4-I10 (`ccd6374`); the rewrite tier MUST stamp
    `data-cf-source-ids` on every `<section>` / content wrapper from the
    outline block's `source_refs[]` payload."""
    # The attribute must be named verbatim so the model emits it on
    # rendered HTML.
    assert "data-cf-source-ids" in _REWRITE_SYSTEM_PROMPT
    # EMPTY_SOURCE_REFS by name pins the Wave4-I10 critical gate.
    assert "EMPTY_SOURCE_REFS" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_system_prompt_carries_objective_id_stamping_directive():
    """The rewrite-tier system prompt MUST carry the W-27 objective-id
    stamping directive — `data-cf-objective-id` derived from the outline
    block's `objective_refs[]` payload, canonical TO-NN / CO-NN pattern."""
    assert "data-cf-objective-id" in _REWRITE_SYSTEM_PROMPT
    # The canonical pattern reference pins the LO ID shape the model
    # must stamp verbatim.
    assert "TO-NN" in _REWRITE_SYSTEM_PROMPT
    assert "CO-NN" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_system_prompt_marks_directives_as_mandatory_output_contract():
    """The three Wave-27 directives are wired as a `MANDATORY OUTPUT
    CONTRACT` block in the rewrite-tier system prompt — verifies the
    integration point is structured (not buried in prose) so a future
    re-wording can't silently demote the directives."""
    assert "MANDATORY OUTPUT CONTRACT" in _REWRITE_SYSTEM_PROMPT


def test_rewrite_user_prompt_surfaces_outline_source_refs_for_stamping(
    monkeypatch,
):
    """The rendered rewrite-tier user prompt MUST surface the outline
    block's `source_refs[]` (so the model has the actual sourceId values
    to stamp on the HTML). The outline-dict JSON dump section carries
    the field verbatim."""
    monkeypatch.delenv(REWRITE_ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    p = RewriteProvider(provider="local")
    block = _rewrite_block_fixture(
        source_refs=["semantik:source-alpha#s3_p1", "semantik:source-alpha#s3_p2"],
        objective_refs=["TO-02", "CO-05"],
    )
    rendered = p._render_user_prompt(block=block)

    # Outline source_refs are dumped via the JSON payload; the sourceId
    # values must appear so the model has the literal data to stamp.
    assert "semantik:source-alpha#s3_p1" in rendered
    assert "semantik:source-alpha#s3_p2" in rendered
    # objective_refs LO IDs likewise.
    assert "TO-02" in rendered
    assert "CO-05" in rendered


# ---------------------------------------------------------------------------
# Cross-provider consistency.
# ---------------------------------------------------------------------------


def test_wave27_directives_present_in_both_tiers_for_their_concerns():
    """Smoke check: both providers carry their tier-specific Wave-27
    directives. Outline tier ensures block-dict metadata fields populate
    (source_refs[] + objective_refs[]); rewrite tier enforces HTML
    attribute stamping (data-cf-source-ids + data-cf-objective-id) plus
    heading hierarchy. The two halves together close the gap that opens
    when `COURSEFORGE_TWO_PASS=true` bypasses the single-pass
    `content-generator` subagent prompt."""
    # Outline-tier concerns (block-dict metadata field directives).
    assert "source_refs[]" in _OUTLINE_SYSTEM_PROMPT
    assert "objective_refs[]" in _OUTLINE_SYSTEM_PROMPT

    # Rewrite-tier concerns (HTML attribute stamping + heading hierarchy).
    assert "HEADING_SKIP" in _REWRITE_SYSTEM_PROMPT
    assert "data-cf-source-ids" in _REWRITE_SYSTEM_PROMPT
    assert "data-cf-objective-id" in _REWRITE_SYSTEM_PROMPT
