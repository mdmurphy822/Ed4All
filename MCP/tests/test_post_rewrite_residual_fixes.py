"""Regression: residual ``post_rewrite_validation`` gate fixes on a real
7B course export.

Three fixes land here (all DETERMINISTIC — no LLM/GPU):

1. **content_type backstop map** — the four substantive-but-content-type-less
   block types (``reflection_prompt`` / ``prereq_set`` / ``discussion_prompt``
   / ``misconception``) gained canonical-ChunkType entries in
   ``MCP.tools.pipeline_tools._REWRITE_BLOCK_TYPE_CONTENT_TYPE`` so the
   rewrite-tier content-type backstop stamps ``data-cf-content-type`` on them
   and ``BlockContentTypeValidator`` accepts them. The two new ChunkType values
   (``reflection`` / ``discussion``) were added additively to
   ``schemas/taxonomies/content_type.json#/$defs/ChunkType``.

2. **prose-fallback CURIE minting** — ``restamp_blocks_final_jsonl`` (and the
   in-loop rewrite backstop's prose fallback) mint a real domain CURIE from a
   block's OWN HTML-stripped prose when the source-chunk arm finds nothing, so
   a rewrite-authored block whose published prose names a domain concept
   anchors. Anti-fabrication is preserved: a block whose prose names no
   vocabulary concept mints nothing.

No LLM dispatch, no pipeline run.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from lib.validators.content_type import get_valid_chunk_types
import MCP.tools.pipeline_tools as pt
from MCP.tools.pipeline_tools import (
    _REWRITE_BLOCK_TYPE_CONTENT_TYPE,
    _run_post_rewrite_validation,
    restamp_blocks_final_jsonl,
)


def _run(**kwargs) -> dict:
    return json.loads(asyncio.run(_run_post_rewrite_validation(**kwargs)))


# ---------------------------------------------------------------------------
# Fix 1 — content_type backstop map + taxonomy additions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "block_type,expected_ct",
    [
        ("reflection_prompt", "reflection"),
        ("prereq_set", "overview"),
        ("discussion_prompt", "discussion"),
        ("misconception", "common_pitfall"),
    ],
)
def test_content_type_backstop_map_covers_substantive_types(
    block_type, expected_ct,
):
    """The four substantive block types map to a canonical ChunkType."""
    assert _REWRITE_BLOCK_TYPE_CONTENT_TYPE.get(block_type) == expected_ct
    # The mapped value must be a member of the canonical taxonomy or the
    # content-type gate would reject the stamped attribute.
    assert expected_ct in get_valid_chunk_types()


def test_reflection_and_discussion_added_to_chunk_taxonomy():
    """The two new metacognitive/collaborative ChunkType values exist."""
    valid = get_valid_chunk_types()
    assert "reflection" in valid
    assert "discussion" in valid
    # The pre-existing values must still be present (additive change).
    for legacy in (
        "explanation", "example", "common_pitfall", "overview",
        "checklist", "scenario", "problem", "vocabulary", "formula",
    ):
        assert legacy in valid


@pytest.mark.parametrize(
    "block_type,expected_ct",
    [
        ("reflection_prompt", "reflection"),
        ("prereq_set", "overview"),
        ("discussion_prompt", "discussion"),
        ("misconception", "common_pitfall"),
    ],
)
def test_inject_content_type_attr_stamps_canonical_value(
    block_type, expected_ct,
):
    """The injector stamps the backstop label onto a wrapper carrying none."""
    html = (
        f'<div class="{block_type.replace("_", "-")}-card" '
        f'data-cf-block-id="week_01_overview#{block_type}_x_0">'
        f"<p>Prose.</p></div>"
    )
    assert "data-cf-content-type" not in html
    out = pt._inject_content_type_attr(html, expected_ct)
    assert f'data-cf-content-type="{expected_ct}"' in out


def test_content_type_gate_accepts_backstop_stamped_block(tmp_path):
    """A reflection_prompt block whose HTML carries the backstop-stamped
    ``data-cf-content-type="reflection"`` passes ``rewrite_content_type``."""
    entry = {
        "block_id": "week_01_self_check#reflection_prompt_x_1",
        "block_type": "reflection_prompt",
        "page_id": "week_01_self_check",
        "sequence": 0,
        "content": (
            '<div class="reflection-prompt" data-cf-content-type="reflection" '
            'data-cf-source-ids="semantik:src#blk_1">'
            "<p>Reflect on how you would apply the order of operations.</p>"
            "</div>"
        ),
        "source_ids": ["semantik:src#blk_1"],
    }
    blocks_path = tmp_path / "blocks_final.jsonl"
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    out = _run(blocks_final_path=str(blocks_path), project_id="")
    assert out["success"] is True
    ct_gate = next(
        g for g in out["gate_results"]
        if g["gate_id"] == "rewrite_content_type"
    )
    assert ct_gate["passed"] is True, ct_gate


def test_content_type_gate_fails_unstamped_reflection_block(tmp_path):
    """Without the stamp (pre-fix shape) the gate still fails closed — the
    fix is the backstop stamp, NOT a gate skip-set relaxation."""
    entry = {
        "block_id": "week_01_self_check#reflection_prompt_x_1",
        "block_type": "reflection_prompt",
        "page_id": "week_01_self_check",
        "sequence": 0,
        "content": (
            '<div class="reflection-prompt" '
            'data-cf-source-ids="semantik:src#blk_1">'
            "<p>Reflect on the lesson.</p></div>"
        ),
        "source_ids": ["semantik:src#blk_1"],
    }
    blocks_path = tmp_path / "blocks_final.jsonl"
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    out = _run(blocks_final_path=str(blocks_path), project_id="")
    ct_gate = next(
        g for g in out["gate_results"]
        if g["gate_id"] == "rewrite_content_type"
    )
    assert ct_gate["passed"] is False, (
        "an unstamped substantive block must still fail closed"
    )


# ---------------------------------------------------------------------------
# Fix 2 — restamp_blocks_final_jsonl (content_type + prose-fallback CURIE)
# ---------------------------------------------------------------------------

def _write_vocab(libv2_root, slug):
    """Write a minimal domain-concept vocabulary for the re-stamp minter."""
    vdir = libv2_root / "courses" / slug / "concept_graph"
    vdir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "concepts": [
            {
                "canonical": "order-of-operations",
                "aliases": ["order of operations", "pemdas"],
            },
            {"canonical": "fraction", "aliases": ["fraction", "fractions"]},
        ],
    }
    (vdir / "domain_concept_vocabulary.json").write_text(
        json.dumps(vocab), encoding="utf-8",
    )


def test_restamp_stamps_content_type_and_mints_curie(tmp_path):
    """The re-stamp pass stamps content_type AND mints a prose-grounded CURIE
    onto a block that named a vocabulary concept in its prose but carried no
    CURIE token."""
    libv2_root = tmp_path / "libv2"
    slug = "sample-course"
    _write_vocab(libv2_root, slug)

    blocks_path = tmp_path / "blocks_final.jsonl"
    entries = [
        {
            # reflection_prompt — needs content_type AND its prose names a
            # concept ("order of operations") so the CURIE minter fires.
            "block_id": "week_01_self_check#reflection_prompt_x_1",
            "block_type": "reflection_prompt",
            "page_id": "week_01_self_check",
            "sequence": 0,
            "content": (
                '<div class="reflection-prompt" '
                'data-cf-source-ids="semantik:src#blk_1">'
                "<p>Reflect on how the order of operations guides you.</p>"
                "</div>"
            ),
        },
    ]
    blocks_path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
    )

    # Dry-run first (default in_place=False): no write, reports stats.
    dry = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code=slug,
        libv2_root=str(libv2_root),
    )
    assert dry["wrote"] is False
    assert dry["content_type_stamped"] == 1
    assert dry["curie_minted"] == 1
    # The dry-run must not have mutated the file on disk.
    assert "data-cf-content-type" not in blocks_path.read_text(encoding="utf-8")

    # in_place=True writes the stamps back.
    res = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code=slug,
        libv2_root=str(libv2_root),
        in_place=True,
    )
    assert res["wrote"] is True
    # Parse the written JSONL so the assertions run over the block's HTML
    # content (not the JSON-escaped raw line, where quotes are ``\"``).
    written_entry = json.loads(
        blocks_path.read_text(encoding="utf-8").strip()
    )
    content = written_entry["content"]
    assert 'data-cf-content-type="reflection"' in content
    assert "data-cf-curie" in content
    assert written_entry["content_type_label"] == "reflection"


def test_restamp_anti_fabrication_no_concept_no_mint(tmp_path):
    """A block whose prose names NO vocabulary concept mints NOTHING (the
    anti-fabrication contract) — it stays curieless."""
    libv2_root = tmp_path / "libv2"
    slug = "sample-course"
    _write_vocab(libv2_root, slug)

    blocks_path = tmp_path / "blocks_final.jsonl"
    entry = {
        "block_id": "week_12_content_01#example_x_1",
        "block_type": "example",
        "page_id": "week_12_content_01",
        "sequence": 0,
        # Prose about insurance — names no vocabulary concept.
        "content": (
            "<section><h2>Insurance Payment</h2>"
            "<p>An insurer pays a claim minus the deductible.</p></section>"
        ),
    }
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    res = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code=slug,
        libv2_root=str(libv2_root),
        in_place=True,
    )
    assert res["curie_minted"] == 0
    assert res["still_curieless"] == 1
    assert "data-cf-curie" not in blocks_path.read_text(encoding="utf-8")


def test_restamp_no_vocabulary_is_curie_noop(tmp_path):
    """No domain vocabulary (RDF / legacy corpus) → the CURIE pass is a
    complete no-op (content_type stamping still runs)."""
    blocks_path = tmp_path / "blocks_final.jsonl"
    entry = {
        "block_id": "week_01_self_check#reflection_prompt_x_1",
        "block_type": "reflection_prompt",
        "page_id": "week_01_self_check",
        "sequence": 0,
        "content": (
            '<div class="reflection-prompt">'
            "<p>Reflect on the order of operations.</p></div>"
        ),
    }
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    res = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code="course-with-no-vocab",
        libv2_root=str(tmp_path / "empty-libv2"),
        in_place=True,
    )
    assert res["curie_minted"] == 0
    # content_type still stamped (taxonomy-independent).
    assert res["content_type_stamped"] == 1


# ---------------------------------------------------------------------------
# Fix 3 — final deterministic CURIE-anchoring sweep (real calibration run)
# ---------------------------------------------------------------------------
#
# Root cause: a block whose outline DECLARED minted CURIEs and whose published
# prose genuinely discusses the concept could still reach blocks_final.jsonl
# with NO ``<span data-cf-curie>`` (the provider's pedagogical-context
# preservation check considered the bare TERM "present" so its forced-inject
# set was empty, and the in-loop per-block backstop had a sequencing gap). The
# str-path ``BlockCurieAnchoringValidator`` then extracted ZERO CURIE TOKENS
# and failed OUTLINE_BLOCK_MISSING_CURIES — aborting the whole post-rewrite
# gate before the block_quality_rollup. The fix adds a final deterministic
# prose-mint sweep over the assembled rewrite blocks before blocks_final.jsonl
# is written. These tests prove the failure→fix chain THROUGH the actual gate.


def _build_block(block_id, block_type, content, *, page_id="week_01_content_01"):
    """Construct a rewrite-tier Block (str content) for the gate."""
    import dataclasses as _dc
    import sys as _sys
    from pathlib import Path as _Path

    _scripts = _Path(pt.__file__).resolve().parents[1] / "Courseforge" / "scripts"
    if str(_scripts) not in _sys.path:
        _sys.path.insert(0, str(_scripts))
    from blocks import Block  # noqa: PLC0415

    flds = {f.name for f in _dc.fields(Block)}
    kw = {
        "block_id": block_id,
        "block_type": block_type,
        "content": content,
        "page_id": page_id,
        "sequence": 0,
    }
    return Block(**{k: v for k, v in kw.items() if k in flds})


def test_anchoring_sweep_fixes_spanless_block_through_gate(tmp_path):
    """A rewrite block whose prose names a vocab concept but carries NO CURIE
    token reaches the gate FAILING; after the deterministic prose-mint sweep
    (same contract as the in-loop final sweep) it PASSES. End-to-end through
    the real ``BlockCurieAnchoringValidator``."""
    from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator
    from lib.ontology.curie_discovery import resolve_minted_curie_map

    libv2_root = tmp_path / "libv2"
    slug = "sample-course"
    _write_vocab(libv2_root, slug)
    minted_curie_map = resolve_minted_curie_map(
        threaded_path=str(
            libv2_root / "courses" / slug / "concept_graph"
            / "domain_concept_vocabulary.json"
        ),
    )

    # A concept block whose published prose discusses "order of operations"
    # (a vocab concept) but carries NO ``prefix:localname`` CURIE token and NO
    # data-cf-curie span — the exact calib failure shape.
    spanless_html = (
        '<section data-cf-content-type="concept" '
        'data-cf-block-id="week_01_content_01#concept_x_3">'
        "<h2>Evaluating Expressions</h2>"
        "<p>The order of operations tells you which step to perform first "
        "when you simplify an expression.</p></section>"
    )

    # --- BEFORE the sweep: the gate FAILS this block (no extractable CURIE).
    pre_block = _build_block("week_01_content_01#concept_x_3", "concept",
                             spanless_html)
    pre = BlockCurieAnchoringValidator().validate(
        {"blocks": [pre_block], "minted_curie_map": minted_curie_map},
    )
    assert pre.passed is False
    assert any(i.code == "OUTLINE_BLOCK_MISSING_CURIES" for i in pre.issues)

    # --- Apply the deterministic prose-mint sweep (restamp = identical
    # contract to the in-loop final sweep wired into the rewrite handler).
    blocks_path = tmp_path / "blocks_final.jsonl"
    blocks_path.write_text(
        json.dumps({
            "block_id": "week_01_content_01#concept_x_3",
            "block_type": "concept",
            "page_id": "week_01_content_01",
            "sequence": 0,
            "content": spanless_html,
        }) + "\n",
        encoding="utf-8",
    )
    res = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code=slug,
        libv2_root=str(libv2_root),
        in_place=True,
    )
    assert res["curie_minted"] == 1
    assert res["still_curieless"] == 0

    # --- AFTER the sweep: the gate PASSES (a real CURIE was minted from the
    # block's OWN prose and now extracts from the hidden span).
    post_entry = json.loads(blocks_path.read_text(encoding="utf-8").strip())
    assert "data-cf-curie" in post_entry["content"]
    post_block = _build_block("week_01_content_01#concept_x_3", "concept",
                              post_entry["content"])
    post = BlockCurieAnchoringValidator().validate(
        {"blocks": [post_block], "minted_curie_map": minted_curie_map},
    )
    assert post.passed is True
    assert not any(
        i.code == "OUTLINE_BLOCK_MISSING_CURIES" for i in post.issues
    )


def test_anchoring_sweep_preserves_anti_fabrication(tmp_path):
    """A rewrite block whose prose names NO vocabulary concept gains NO span —
    the sweep never invents a CURIE (anti-fabrication), so the block stays
    honestly curieless rather than carrying a fabricated anchor."""
    libv2_root = tmp_path / "libv2"
    slug = "sample-course"
    _write_vocab(libv2_root, slug)

    # Prose about insurance — names no vocabulary concept.
    no_concept_html = (
        '<section data-cf-content-type="example">'
        "<h2>Insurance Payment</h2>"
        "<p>An insurer pays a claim minus the deductible.</p></section>"
    )
    blocks_path = tmp_path / "blocks_final.jsonl"
    blocks_path.write_text(
        json.dumps({
            "block_id": "week_12_content_01#example_x_1",
            "block_type": "example",
            "page_id": "week_12_content_01",
            "sequence": 0,
            "content": no_concept_html,
        }) + "\n",
        encoding="utf-8",
    )
    res = restamp_blocks_final_jsonl(
        blocks_final_path=str(blocks_path),
        course_code=slug,
        libv2_root=str(libv2_root),
        in_place=True,
    )
    assert res["curie_minted"] == 0
    assert res["still_curieless"] == 1
    assert "data-cf-curie" not in blocks_path.read_text(encoding="utf-8")
