"""P4 regression tests — key-terms definition leakage + Bloom-verb exclusion +
definition-quality classification (the rewrite-tier short-circuit gate) and the
block_planner worked-example / Bloom-spread pedagogical-depth floors.

All assertions are pure-lexical / pure-deterministic — no model, no GPU.
"""

from __future__ import annotations

import json
import os

import pytest

from lib.generation import key_terms as kt
from lib.generation import block_planner as bp


# --------------------------------------------------------------------------- #
# 1a — leakage filter: exercise / heading / answer-key sentences rejected.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "leaky_text",
    [
        "TRY IT :: 1.35 Evaluate 8x minus 3, when x equals 2.",
        "EXAMPLE 1.1 In the number 63,407,218 the place value is shown.",
        "Solution to the place value exercise is shown here below.",
        "Chapter 1 Foundations covers place value across the board.",
        "30 Chapter 1 Foundations Multiply the two place value numbers.",
        "550 place value 47 22,335 place value 48 39,075 answer key run.",
    ],
)
def test_resolve_definition_rejects_leaky_source_sentence(leaky_text):
    """A leaky candidate sentence is skipped → falls through to definition_hint."""
    term = {
        "slug": "place-value",
        "display": "Place value",
        "canonical": "place value",
        "aliases": [],
        "definition_hint": "The value of a digit based on its position.",
    }
    chunks = [{"id": "c1", "text": leaky_text, "item_path": "p.html"}]
    resolved = kt.resolve_definition(term, chunks=chunks)
    assert resolved is not None
    definition, chunk = resolved
    # Leakage rejected → the curated hint wins, defining chunk is None.
    assert definition == "The value of a digit based on its position."
    assert chunk is None


def test_resolve_definition_keeps_clean_definitional_sentence():
    """A genuine definitional source sentence is still preferred (not over-filtered)."""
    term = {
        "slug": "prime-number",
        "display": "Prime number",
        "canonical": "prime number",
        "aliases": ["prime"],
        "definition_hint": "A number divisible only by 1 and itself.",
    }
    chunks = [{
        "id": "c1",
        "text": (
            "A prime number is a whole number greater than one with exactly "
            "two factors."
        ),
        "item_path": "ch1.html",
        "heading": "Primes",
    }]
    resolved = kt.resolve_definition(term, chunks=chunks)
    assert resolved is not None
    definition, chunk = resolved
    assert "whole number greater than one" in definition
    assert chunk is not None and chunk["id"] == "c1"


def test_resolve_definition_strips_stray_glyphs():
    """Return-arrows / circled glyphs are stripped from a definition."""
    term = {
        "slug": "factor",
        "display": "Factor",
        "canonical": "factor",
        "aliases": [],
        "definition_hint": "",
    }
    chunks = [{
        "id": "c1",
        "text": "A factor ① is a number that divides another evenly ↩.",
        "item_path": "p.html",
    }]
    resolved = kt.resolve_definition(term, chunks=chunks)
    assert resolved is not None
    definition, _ = resolved
    assert "①" not in definition and "↩" not in definition


# --------------------------------------------------------------------------- #
# 1b — Bloom verbs are excluded from term candidates.
# --------------------------------------------------------------------------- #


def test_bloom_verb_not_a_term():
    """'evaluate' / 'simplify' are Bloom verbs and must NOT be glossary terms."""
    vocab = {
        "concepts": [
            {"canonical": "evaluate", "definition_hint": "x"},
            {"canonical": "simplify", "definition_hint": "y"},
            {"canonical": "prime number", "definition_hint": "z"},
        ]
    }
    terms = kt.aggregate_terms(
        concept_tags=["evaluate", "simplify", "prime number"],
        chunk_texts=["evaluate simplify a prime number"],
        vocabulary=vocab,
    )
    slugs = {t["slug"] for t in terms}
    assert "evaluate" not in slugs
    assert "simplify" not in slugs
    assert "prime-number" in slugs


def test_nominalized_bloom_form_survives():
    """A nominalized form ('simplification') is a distinct slug and is kept."""
    vocab = {"concepts": [{"canonical": "simplification", "definition_hint": "x"}]}
    terms = kt.aggregate_terms(
        concept_tags=["simplification"],
        chunk_texts=["simplification is the process"],
        vocabulary=vocab,
    )
    assert any(t["slug"] == "simplification" for t in terms)


# --------------------------------------------------------------------------- #
# 1c / 2 — definition-quality classification + block-level reconstruction.
# --------------------------------------------------------------------------- #


def test_definition_quality_hint_is_high():
    assert kt.definition_quality("Any text.", source_chunk=None) == (
        kt.DEFINITION_QUALITY_HIGH
    )


def test_definition_quality_definitional_shape_is_high():
    assert kt.definition_quality(
        "A prime is a number that is divisible by one.",
        source_chunk={"id": "c"},
    ) == kt.DEFINITION_QUALITY_HIGH


def test_definition_quality_bare_mention_is_low():
    assert kt.definition_quality(
        "We multiply two primes together in this passage.",
        source_chunk={"id": "c"},
    ) == kt.DEFINITION_QUALITY_LOW


def test_build_key_terms_cards_stamps_quality():
    term = {
        "slug": "prime-number",
        "display": "Prime number",
        "canonical": "prime number",
        "aliases": [],
        "definition_hint": "A number divisible only by 1 and itself.",
    }
    cards = kt.build_key_terms_cards(terms=[term], chunks=[], course_slug="s")
    assert cards and cards[0]["definition_quality"] == kt.DEFINITION_QUALITY_HIGH


def test_block_definition_quality_reconstructs_from_html():
    """The short-circuit gate's reconstruction matches the card classification."""
    high = kt.render_term_card(
        display="Prime",
        definition="A prime is a number that is divisible by one and itself.",
        source_link=None,
        slug="prime",
    )
    low = kt.render_term_card(
        display="Mult",
        definition="We multiply primes together in this passage today.",
        source_link=None,
        slug="mult",
    )
    # source-grounded definitional shape → high; bare mention → low.
    assert kt.block_definition_quality(content=high, has_source_ids=True) == (
        kt.DEFINITION_QUALITY_HIGH
    )
    assert kt.block_definition_quality(content=low, has_source_ids=True) == (
        kt.DEFINITION_QUALITY_LOW
    )
    # No source ids → definition_hint provenance → always high.
    assert kt.block_definition_quality(content=low, has_source_ids=False) == (
        kt.DEFINITION_QUALITY_HIGH
    )


# --------------------------------------------------------------------------- #
# 2 — rewrite-tier short-circuit fires only for HIGH-quality key_terms blocks.
# --------------------------------------------------------------------------- #


def test_short_circuit_conditional_in_pipeline_tools():
    """The pipeline short-circuit only passes HIGH-quality cards through verbatim."""
    from MCP.tools import pipeline_tools as pt  # noqa: PLC0415
    from Courseforge.scripts.blocks import Block  # noqa: PLC0415

    def _mk(definition, source_ids):
        content = kt.render_term_card(
            display="T", definition=definition, source_link=None, slug="t",
        )
        return Block(
            block_id="p#vocab_card_t_0",
            block_type="vocab_card",
            page_id="week_01_key_terms",
            sequence=0,
            content=content,
            template_type=pt._KEY_TERMS_TEMPLATE_TYPE,
            source_ids=source_ids,
        )

    # HIGH (definition_hint provenance → no source ids) short-circuits.
    high_blk = _mk("We mention the term here in passing only.", ())
    assert (
        kt.block_definition_quality(
            content=high_blk.content,
            has_source_ids=bool(high_blk.source_ids),
        )
        == kt.DEFINITION_QUALITY_HIGH
    )
    # LOW (grounded bare mention) does NOT short-circuit → rewrite authors it.
    low_blk = _mk("We mention the term here in passing only.", ("semantik:s#c1",))
    assert (
        kt.block_definition_quality(
            content=low_blk.content,
            has_source_ids=bool(low_blk.source_ids),
        )
        == kt.DEFINITION_QUALITY_LOW
    )


# --------------------------------------------------------------------------- #
# 3 — block_planner worked-example + Bloom-spread floors (gated; default OFF).
# --------------------------------------------------------------------------- #


class _FakePlanProvider:
    _model = "fake"

    def __init__(self, blocks):
        self._blocks = blocks

    def plan_blocks(self, prompt):  # noqa: ARG002
        return json.dumps({"blocks": self._blocks})


def _plan_with(blocks, cos, **env):
    saved = {}
    for k in ("ED4ALL_WORKED_EXAMPLE_FLOOR", "ED4ALL_BLOOM_SPREAD_FLOOR"):
        saved[k] = os.environ.pop(k, None)
    try:
        for k, v in env.items():
            os.environ[k] = v
        return bp.plan_week_blocks(
            terminal_objective={"id": "TO-01", "statement": "x"},
            chapter_objectives=cos,
            source_chunks=[{"id": "s", "text": "content"}],
            provider=_FakePlanProvider(blocks),
        )
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_worked_example_floor_off_is_byte_stable():
    """Default OFF: no per-CO worked-example block is added by the P4 floor."""
    cos = [{"id": "CO-01", "statement": "apply", "bloom_level": "apply"}]
    plan = _plan_with(
        [{"block_type": "concept", "page_type": "content",
          "target_co_ids": ["CO-01"], "target_bloom": "apply"}],
        cos,
    )
    # The default page-floors may seat generic example/problem fillers, but
    # NONE targets CO-01 (the P4 floor is what attaches the CO id).
    we_for_co = [
        b for b in plan.selected
        if b["block_type"] in ("example", "problem")
        and "CO-01" in (b.get("target_co_ids") or [])
    ]
    assert we_for_co == []


def test_worked_example_floor_on_covers_procedural_co():
    """ON: every procedural CO gains a worked example targeting it; others do not."""
    cos = [
        {"id": "CO-01", "statement": "apply", "bloom_level": "apply"},
        {"id": "CO-02", "statement": "understand", "bloom_level": "understand"},
    ]
    plan = _plan_with(
        [{"block_type": "concept", "page_type": "content",
          "target_co_ids": ["CO-01"], "target_bloom": "apply"}],
        cos,
        ED4ALL_WORKED_EXAMPLE_FLOOR="1",
    )
    we = [
        b for b in plan.selected if b["block_type"] in ("example", "problem")
    ]
    targeted = {cid for b in we for cid in (b.get("target_co_ids") or [])}
    assert "CO-01" in targeted, "procedural CO must get a worked example"
    assert "CO-02" not in targeted, "non-procedural CO must not"


def test_bloom_spread_floor_on_adds_analyze_plus():
    """ON: a week with no analyze+ block gains one; OFF leaves it as planned."""
    cos = [{"id": "CO-01", "statement": "u", "bloom_level": "understand"}]
    # A plan whose only blocks are understand/apply — force no analyze+ by
    # asserting the floor ADDS a misconception/scenario/etc.
    blocks = [{"block_type": "concept", "page_type": "content",
               "target_co_ids": ["CO-01"], "target_bloom": "understand"}]
    plan_on = _plan_with(blocks, cos, ED4ALL_BLOOM_SPREAD_FLOOR="1")
    assert any(
        b["target_bloom"] in ("analyze", "evaluate", "create")
        for b in plan_on.selected
    ), "Bloom-spread floor must seat an analyze-or-higher block"
