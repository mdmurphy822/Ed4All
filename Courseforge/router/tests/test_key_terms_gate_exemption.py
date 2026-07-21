"""Feature I5 BUG 1 — deterministic key_terms blocks are exempt from the
LLM-authoring inter-tier / post-rewrite validation gates.

A ``Block(block_type="vocab_card", template_type="key_terms")`` carries a
pre-rendered, grounded HTML string authored by the deterministic
``lib/generation/key_terms.py`` builder (NOT the LLM outline/rewrite tier). It
has no CURIEs and a non-taxonomy ``content_type`` ("key-terms"), so it CANNOT
pass the LLM-authoring gates. These regressions assert:

* the shared ``_is_deterministic_template_block`` predicate keys on
  ``template_type == "key_terms"``;
* every ``Block*Validator`` SKIPS the key_terms block (passes, not counted as a
  failure) while a normal LLM block in the SAME batch is still validated;
* the key_terms block still SHIPS (is not added to the failed-block set, so the
  validation handler writes it into blocks_validated.jsonl).

Pure-helper unit tests — no LLM client, no model load, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_CF_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_CF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CF_SCRIPTS))

from blocks import Block  # noqa: E402
from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
    _DETERMINISTIC_TEMPLATE_TYPES,
    _is_deterministic_template_block,
)


# --------------------------------------------------------------------------- #
# Fixtures — a deterministic key_terms block + a normal LLM rewrite-tier block.
# --------------------------------------------------------------------------- #


def _key_terms_block() -> Block:
    """The I5 deterministic vocab card: str HTML content, no CURIEs, content_type
    "key-terms" (not in the canonical taxonomy)."""
    html = (
        '<div class="callout vocab-card" data-cf-content-type="key-terms" '
        'data-cf-term="prime-number">\n'
        '      <p><span class="key-term">Prime number</span></p>\n'
        '      <p>A whole number greater than one with exactly two factors.</p>\n'
        '    </div>'
    )
    return Block(
        block_id="week_02_key_terms#vocab_card_prime-number_0",
        block_type="vocab_card",
        page_id="week_02_key_terms",
        sequence=0,
        content=html,
        template_type="key_terms",
        key_terms=("prime-number",),
        objective_ids=("TO-02",),
        target_bloom="remember",
    )


def _normal_block() -> Block:
    """A normal LLM rewrite-tier concept block with a valid content_type, a CURIE,
    and an objective ref — should PASS every gate."""
    html = (
        '<section data-cf-content-type="key_idea" data-cf-objective-id="TO-01">'
        "<p>The mole is sh:path the SI unit. ex:mole grounds this.</p></section>"
    )
    return Block(
        block_id="week_01_content#concept_mole_0",
        block_type="concept",
        page_id="week_01_content",
        sequence=0,
        content=html,
        objective_ids=("TO-01",),
    )


# --------------------------------------------------------------------------- #
# Predicate.
# --------------------------------------------------------------------------- #


def test_predicate_keys_on_template_type():
    assert "key_terms" in _DETERMINISTIC_TEMPLATE_TYPES
    assert _is_deterministic_template_block(_key_terms_block()) is True
    assert _is_deterministic_template_block(_normal_block()) is False


# --------------------------------------------------------------------------- #
# Each Block*Validator skips the key_terms block, audits the normal one.
# --------------------------------------------------------------------------- #


def _run(validator, blocks, **extra):
    return validator.validate({"blocks": blocks, "gate_id": "g", **extra})


def test_curie_anchoring_skips_key_terms_block():
    kt = _key_terms_block()
    normal = _normal_block()
    # In isolation, the key_terms block trips OUTLINE_BLOCK_MISSING_CURIES;
    # with the exemption it must be skipped -> gate passes.
    res_kt_only = _run(BlockCurieAnchoringValidator(), [kt])
    assert res_kt_only.passed is True, "key_terms block not exempted from curie gate"

    # In a mixed batch, the normal block is still validated (it carries a CURIE).
    res_mixed = _run(BlockCurieAnchoringValidator(), [kt, normal])
    assert res_mixed.passed is True
    # No issue mentions the key_terms block.
    locs = {i.location for i in res_mixed.issues}
    assert kt.block_id not in locs


def test_content_type_skips_key_terms_block():
    kt = _key_terms_block()
    normal = _normal_block()
    # "key-terms" is NOT a canonical chunk type -> would trip
    # OUTLINE_BLOCK_INVALID_CONTENT_TYPE without the exemption.
    res_kt_only = _run(BlockContentTypeValidator(), [kt])
    assert res_kt_only.passed is True, "key_terms block not exempted from content_type gate"

    res_mixed = _run(BlockContentTypeValidator(), [kt, normal])
    assert res_mixed.passed is True
    locs = {i.location for i in res_mixed.issues}
    assert kt.block_id not in locs


def test_page_objectives_skips_key_terms_block():
    kt = _key_terms_block()
    # TO-02 is NOT in the seeded valid set; without exemption the gate would
    # flag OUTLINE_BLOCK_UNKNOWN_OBJECTIVE. With exemption it passes.
    res = _run(
        BlockPageObjectivesValidator(), [kt],
        valid_objective_ids=["TO-99"],
    )
    assert res.passed is True
    assert kt.block_id not in {i.location for i in res.issues}


def test_source_refs_skips_key_terms_block():
    kt = _key_terms_block()
    # Give the key_terms block a malformed source id that would trip
    # OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE without the exemption.
    import dataclasses
    kt_bad = dataclasses.replace(kt, source_ids=("not-a-valid-sourceid",))
    res = _run(BlockSourceRefValidator(), [kt_bad], valid_source_ids=["semantik:c#b"])
    assert res.passed is True
    assert kt_bad.block_id not in {i.location for i in res.issues}


def test_normal_block_still_fails_when_broken():
    """Sanity: the exemption is template-keyed, not a blanket pass-through.
    A broken NORMAL block still fails."""
    broken = Block(
        block_id="week_01_content#concept_bad_0",
        block_type="concept",
        page_id="week_01_content",
        sequence=0,
        content='<section data-cf-content-type="concept">no curies here</section>',
        objective_ids=("TO-01",),
    )
    res = _run(BlockCurieAnchoringValidator(), [broken])
    assert res.passed is False
    assert broken.block_id in {i.location for i in res.issues}


def test_key_terms_block_ships_via_validation_handler(tmp_path, monkeypatch):
    """End-to-end through _run_inter_tier_validation: the key_terms block must
    appear in blocks_validated.jsonl (it ships), even though it can't pass the
    LLM gates."""
    import asyncio
    import json

    from MCP.tools import pipeline_tools as _pt

    kt = _key_terms_block()
    normal = _normal_block()
    blocks_path = tmp_path / "blocks_outline.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        for blk in (kt, normal):
            fh.write(json.dumps(
                _pt._block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")

    out = asyncio.run(_pt._run_inter_tier_validation(
        blocks_outline_path=str(blocks_path),
        project_id="",
    ))
    result = json.loads(out)
    assert result.get("success") is True, result
    validated = Path(result["blocks_validated_path"]).read_text(encoding="utf-8")
    assert kt.block_id in validated, "key_terms block did not ship"
