"""FR-PLAN-01 — the rewrite prompt threads the planner-selected interaction type.

``_interaction_type_directive`` returns an empty string when no interaction type
is selected (byte-stable prompt) and a concrete authoring directive when one is.
The rendered user prompt surfaces the directive for a block carrying an
``interaction_type``.
"""
from __future__ import annotations

from Courseforge.generators.rewrite._rewrite_provider import (
    RewriteProvider,
    _interaction_type_directive,
)
from Courseforge.scripts.blocks import Block


def test_directive_empty_when_none():
    assert _interaction_type_directive(None) == ""
    assert _interaction_type_directive("") == ""


def test_directive_renders_known_token():
    out = _interaction_type_directive("drag_drop")
    assert "Interaction type (planner-selected)" in out
    assert "drag-and-drop" in out


def test_directive_renders_unknown_token_generically():
    out = _interaction_type_directive("some_new_type")
    assert "some new type" in out


def test_user_prompt_includes_directive_when_set():
    provider = RewriteProvider.__new__(RewriteProvider)  # no network construction
    block = Block(
        block_id="p#assessment_item_x_0", block_type="assessment_item",
        page_id="p", sequence=0, content={"key_claims": ["x"]},
        interaction_type="multiple_choice",
    )
    prompt = RewriteProvider._render_user_prompt(provider, block=block)
    assert "Interaction type (planner-selected)" in prompt
    assert "multiple-choice" in prompt


def test_user_prompt_byte_stable_when_no_interaction_type():
    provider = RewriteProvider.__new__(RewriteProvider)
    block = Block(
        block_id="p#concept_x_0", block_type="concept", page_id="p",
        sequence=0, content={"key_claims": ["x"]},
    )
    prompt = RewriteProvider._render_user_prompt(provider, block=block)
    assert "Interaction type (planner-selected)" not in prompt
