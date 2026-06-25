"""IB6.4 helper tests — per-type ceiling + escaped-provenance scrub (FIX 2)."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators._block_rubric_helpers import (
    body_text_of,
    resolve_body_char_ceiling,
)


# --- resolve_body_char_ceiling: per-type budget + precedence ---------------


def test_atomic_default_ceiling_is_200():
    assert resolve_body_char_ceiling() == 200
    # Unknown / untabled block_type falls through to the atomic default.
    assert resolve_body_char_ceiling(block_type="not_a_real_type") == 200


def test_atomic_micro_block_types_keep_200():
    # FIX 3 — genuinely-atomic one-line micro-blocks keep the 200 budget so a
    # multi-idea over-stuffed key_idea / callout still trips the char axis.
    for bt in ("vocab_card", "callout", "formula", "key_idea"):
        assert resolve_body_char_ceiling(block_type=bt) == 200


def test_exposition_types_get_higher_budget():
    # FIX 3 — the FIX 2 six PLUS the single-developed-idea types raised to 1000.
    for bt in (
        "concept",
        "example",
        "worked_example",
        "self_check_question",
        "diagram",
        "scenario",
        "explanation",
        "problem",
        "activity",
        "guided_practice",
        "misconception",
        "hook",
        "multimedia",
    ):
        assert resolve_body_char_ceiling(block_type=bt) == 1000


def test_aggregating_types_get_highest_budget():
    # FIX 3 — structural roll-ups (a prereq set / recap / acronym table is a
    # several-entry aggregate by construction) get the ~1200 budget.
    for bt in (
        "prereq_set",
        "recap",
        "checklist",
        "reflection_prompt",
        "acronym",
        "table",
        "discussion_prompt",
        "resources",
        "flip_card_grid",
    ):
        assert resolve_body_char_ceiling(block_type=bt) == 1200


def test_exempt_types_not_in_ceiling_table():
    # ``objective`` / ``summary_takeaway`` are EXEMPT from the body-overflow
    # check in content.py, so they must NOT carry a (dead) ceiling row here —
    # they fall through to the atomic default, but the validator never reaches
    # this resolver for them.
    from lib.validators._block_rubric_helpers import (
        _BLOCK_BODY_CHAR_CEILING_BY_TYPE,
    )

    assert "objective" not in _BLOCK_BODY_CHAR_CEILING_BY_TYPE
    assert "summary_takeaway" not in _BLOCK_BODY_CHAR_CEILING_BY_TYPE


def test_explicit_override_wins_over_everything():
    # A caller-pinned positive override beats both env and the per-type table.
    assert resolve_body_char_ceiling(777, block_type="concept") == 777


def test_env_override_wins_over_per_type(monkeypatch):
    # PRESERVE historical env semantics: a GLOBAL env override beats the
    # per-type 1000 budget (env=50 must shrink even an exposition concept).
    monkeypatch.setenv("ED4ALL_BLOCK_BODY_CHAR_CEILING", "50")
    assert resolve_body_char_ceiling(block_type="concept") == 50
    assert resolve_body_char_ceiling(block_type="key_idea") == 50


def test_garbage_env_falls_back_to_per_type(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_BODY_CHAR_CEILING", "not-an-int")
    assert resolve_body_char_ceiling(block_type="concept") == 1000
    assert resolve_body_char_ceiling(block_type="key_idea") == 200


# --- body_text_of: escaped-provenance scrub --------------------------------


def _real_body(n_sentences: int = 13) -> str:
    return ("Variables represent unknown quantities in algebra. " * n_sentences).strip()


def test_body_text_of_deinflates_escaped_provenance():
    real = _real_body()
    prov = (
        "&lt;aside class='cf-source-provenance' "
        "data-cf-source-ids='dart:demo-ch1#b1'&gt;"
        + ("Source provenance payload token. " * 60)
        + "&lt;/aside&gt;"
    )
    content = "<p>" + real + "</p>" + prov
    block = Block(
        block_id="b1",
        block_type="key_idea",
        page_id="p",
        sequence=1,
        content=content,
    )
    measured = len(body_text_of(block))
    inflated = len(content)
    # Measured length tracks the real body, NOT the inflated provenance run.
    assert abs(measured - len(real)) <= 5
    assert measured < inflated / 2  # well below the inflated length


def test_body_text_of_preserves_legit_escaped_entities():
    # Legitimately-escaped content (math ``&lt;``, literal ``&amp;``) is real
    # body text and must NOT be scrubbed — only escaped TAG/provenance runs are.
    content = (
        "<p>For all x, if x &lt; 5 and a &amp; b hold, "
        "then the inequality stands.</p>"
    )
    block = Block(
        block_id="b2",
        block_type="concept",
        page_id="p",
        sequence=1,
        content=content,
    )
    text = body_text_of(block)
    assert "&lt;" in text
    assert "&amp;" in text
