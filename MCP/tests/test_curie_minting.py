"""Regression tests for the v0.3.0 dynamic CURIE-minting wiring.

Exercises ``MCP/tools/pipeline_tools.py::_mint_outline_curies`` — the
outline-handler step that stamps per-course minted CURIEs onto
curie-less outline blocks when a Stage-3 ``domain_concept_vocabulary.json``
is present.

Contract assertions:

1. **With a vocabulary present**, an outline block with empty ``curies``
   whose ``key_claims`` text mentions a vocab concept gets the minted
   CURIE stamped into ``content["curies"]``.
2. **Without a vocabulary**, blocks are byte-identical to the input
   (the backward-compat no-op contract — RDF / legacy corpora).
3. **A block that already carries CURIEs** is never overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Block lives at Courseforge/scripts/blocks.py.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from MCP.tools.pipeline_tools import (  # noqa: E402
    _mint_outline_curies,
    _resolve_minted_curie_map_for_validation,
)
from MCP.hardening.gate_input_routing import (  # noqa: E402
    _resolve_minted_curie_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _vocab_file(tmp_path: Path) -> Path:
    """Write a minimal domain_concept_vocabulary.json under a fake LibV2
    course tree and return its path."""
    course_dir = tmp_path / "courses" / "demo-101" / "concept_graph"
    course_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "schema_version": "v1",
        "course_id": "DEMO_101",
        "course_slug": "demo-101",
        "concept_count": 2,
        "concepts": [
            {"canonical": "slope", "aliases": ["gradient"]},
            {"canonical": "y-intercept", "aliases": []},
        ],
    }
    path = course_dir / "domain_concept_vocabulary.json"
    path.write_text(json.dumps(vocab), encoding="utf-8")
    return path


def _outline_block(*, block_id: str, curies, key_claims) -> Block:
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id="week_01_content_01",
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": list(key_claims),
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


# ---------------------------------------------------------------------------
# 1. With a vocabulary present — minting fires.
# ---------------------------------------------------------------------------


def test_minting_stamps_curies_when_vocabulary_present(tmp_path):
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_slope_0",
        curies=[],  # empty — the minting trigger
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # The block was replaced with a minted-CURIE-carrying copy.
    minted = blocks[0].content["curies"]
    assert minted, "expected minted CURIEs to be stamped"
    assert "demo101:slope" in minted


def test_minting_matches_via_alias_surface_form(tmp_path):
    """The block text mentions the alias 'gradient' (not 'slope') — the
    concept-seed match still fires and the minted CURIE lands."""
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=[],
        key_claims=["A line's gradient describes how steep it is."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    assert "demo101:slope" in blocks[0].content["curies"]


# ---------------------------------------------------------------------------
# 2. Without a vocabulary — no-op (backward-compat contract).
# ---------------------------------------------------------------------------


def test_no_vocabulary_leaves_blocks_untouched(tmp_path):
    # tmp_path has NO domain_concept_vocabulary.json.
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=[],
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    before = blocks[0]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # Same object, untouched — byte-identical to the input.
    assert blocks[0] is before
    assert blocks[0].content["curies"] == []


# ---------------------------------------------------------------------------
# 3. Existing CURIEs are never overwritten.
# ---------------------------------------------------------------------------


def test_existing_curies_not_overwritten(tmp_path):
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=["demo101:slope"],  # already a real per-course domain CURIE
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    before = blocks[0]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # Untouched — a real minted domain CURIE wins.
    assert blocks[0] is before
    assert blocks[0].content["curies"] == ["demo101:slope"]


# ---------------------------------------------------------------------------
# 3b. FIX 1: source-chunk minting fallback when key_claims lacks a vocab term.
# ---------------------------------------------------------------------------


def test_minting_via_source_chunk_text_when_claims_lack_term(tmp_path):
    """FIX 1: the block's key_claims do NOT mention any vocab concept, but
    its grounded source-chunk text does — the minted CURIE is stamped from
    the source-chunk match. The source chunks are the block's real
    provenance, so this is rigorous, not fabrication."""
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_b_0",
        curies=[],
        key_claims=["This block talks about nothing in the vocabulary."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
        chunks_lookup={
            "week_01_content_01#objective_b_0": [
                {"id": "c1", "text": "The slope of a line is its rise over run."},
            ],
        },
    )
    assert "demo101:slope" in blocks[0].content["curies"]


def test_no_source_chunk_text_no_mint(tmp_path):
    """FIX 1 fail-closed: key_claims has no vocab term AND the block has no
    grounded source-chunk text (topic-paragraph fallback entry, no 'text'
    key) — nothing is minted; the block fails closed downstream."""
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_c_0",
        curies=[],
        key_claims=["No vocabulary concept here at all."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
        chunks_lookup={
            "week_01_content_01#objective_c_0": [
                {"heading": "Topic", "paragraphs": ["Nothing relevant."]},
            ],
        },
    )
    assert blocks[0].content["curies"] == []


# ---------------------------------------------------------------------------
# 3c. FIX 1: generic / placeholder CURIE is augmented (not overwritten).
# ---------------------------------------------------------------------------


def test_generic_placeholder_curie_augmented_with_domain_curie(tmp_path):
    """FIX 1: a block carrying only a GENERIC / placeholder CURIE (e.g.
    'outline:sample', 'cf:CO-36' — not a key in the per-course minted
    map) is treated as still needing a domain CURIE; the minted domain
    CURIE is APPENDED and the generic token is preserved."""
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_d_0",
        curies=["outline:sample", "cf:CO-36"],  # generic placeholders
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="DEMO_101",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    result = blocks[0].content["curies"]
    # Generic tokens preserved...
    assert "outline:sample" in result
    assert "cf:CO-36" in result
    # ...and the minted domain CURIE appended.
    assert "demo101:slope" in result


# ---------------------------------------------------------------------------
# 4. R5 — resolver parity. Both the gate-input router and the validation
#    handlers must resolve the SAME minted-CURIE map for a vocabulary
#    that exists ONLY on disk (no phase_outputs.concept_extraction
#    thread — the resumed / stage-subcommand scenario). Before R5 the
#    gate-routing copy had no on-disk fallback and returned None while
#    the handler copy found the vocabulary.
# ---------------------------------------------------------------------------


def test_resolver_parity_disk_only_vocabulary(tmp_path):
    """Vocabulary present only on disk, no phase_outputs thread: the
    gate-input router and the validation handler resolve an identical,
    non-empty minted-CURIE map."""
    _vocab_file(tmp_path)  # writes courses/demo-101/concept_graph/...

    # The gate-input router: workflow_params carries the LibV2 root and
    # the course name; phase_outputs is EMPTY (the resumed-run shape).
    gate_map = _resolve_minted_curie_map(
        phase_outputs={},
        workflow_params={
            "course_name": "DEMO_101",
            "libv2_root": str(tmp_path),
        },
    )

    # The validation handler: project_id with no project_config.json on
    # disk falls back to project_id-as-course-code. Using the slug form
    # directly yields the same course slug ("demo-101").
    handler_map = _resolve_minted_curie_map_for_validation(
        project_id="demo-101",
        kwargs={"libv2_root": str(tmp_path), "phase_outputs": {}},
    )

    assert gate_map is not None, "gate-router resolver returned None"
    assert handler_map is not None, "validation-handler resolver returned None"
    # Same verdict — identical maps. (Both mint the same prefix from the
    # vocabulary's own course_id field, so the maps are byte-equal.)
    assert gate_map == handler_map
    assert "demo101:slope" in gate_map


def test_resolver_parity_no_vocabulary_both_none(tmp_path):
    """No vocabulary anywhere: both resolvers return None (legacy
    literal-token anchoring), so the gate verdict still converges."""
    gate_map = _resolve_minted_curie_map(
        phase_outputs={},
        workflow_params={
            "course_name": "DEMO_101",
            "libv2_root": str(tmp_path),
        },
    )
    handler_map = _resolve_minted_curie_map_for_validation(
        project_id="demo-101",
        kwargs={"libv2_root": str(tmp_path), "phase_outputs": {}},
    )
    assert gate_map is None
    assert handler_map is None
