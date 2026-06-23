"""Regression tests for the post_rewrite_validation gate root-cause fixes.

A live full-7B ``textbook_to_course`` run had 35/37 rewrite-tier blocks fail
``post_rewrite_validation`` while the 2 DEGRADED (escalated) blocks vacuously
passed. Root-cause investigation (2026-06-15) found five distinct roots; this
suite pins the fixes so they don't regress:

1. Cache round-trip silently DROPPED ``touched_by`` — a resumed run rehydrates
   blocks from ``blocks_final.jsonl`` with no touch chain, so the per-block
   synthesis manifest emits ``synthesis.tiers:[]`` and fails
   MANIFEST_SCHEMA_INVALID. ``_block_to_snake_case_entry`` now emits it and the
   three ``_entry_to_block`` round-trips + ``_hydrate_blocks_from_path``
   reconstruct it.
2. ``BlockContentTypeValidator`` audited scaffolding ``objective`` blocks (which
   carry no ``data-cf-content-type`` and are not ChunkType members) → an
   objective-only page 100%-failed the gate. The validator now skips
   ``objective`` / ``chrome`` / ``recap``.
3. The str-path objective extractor read a comma-/space-joined
   ``data-cf-objective-id="TO-04, CO-03, CO-07"`` attribute as ONE bogus id; it
   now splits on commas + whitespace so each id resolves.
4. The rewrite backstop now deterministically injects a dropped
   ``data-cf-block-id`` and strips a model-hallucinated standalone
   ``<data-cf-source-ids=...>`` element (both 7B emit defects).
5. ``_emit_block_synthesis_manifest`` skips escalation-marked blocks (no
   successful synthesis → no complete manifest line; the validation report
   already classifies them ``escalated``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block, Touch  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402


def _rewrite_touch() -> Touch:
    return Touch(
        model="qwen2.5:7b-instruct-q4_K_M",
        provider="local",
        tier="rewrite",
        timestamp="2026-06-15T00:00:00Z",
        decision_capture_id="dc-rewrite-1",
        purpose="rewrite",
    )


# --------------------------------------------------------------------------- #
# Root 1 — cache round-trip preserves the full Block field set incl touched_by
# --------------------------------------------------------------------------- #


def test_snake_case_entry_round_trip_preserves_all_fields():
    """``_block_to_snake_case_entry`` ↔ rehydration preserves every field a
    post_rewrite validator reads, INCLUDING the touch chain (the field the
    pre-fix round-trip silently dropped)."""
    src = Block(
        block_id="week_01_content_01#concept_x_0",
        block_type="concept",
        page_id="week_01_content_01",
        sequence=3,
        content="<section>body</section>",
        content_type_label="definition",
        source_ids=("dart:slug#b1", "dart:slug#b2"),
        source_primary="dart:slug#b1",
        source_references=({"sourceId": "dart:slug#b1", "role": "primary"},),
        objective_ids=("CO-01", "TO-02"),
        bloom_level="understand",
        key_terms=("term_a",),
        touched_by=(_rewrite_touch(),),
    )
    entry = _pt._block_to_snake_case_entry(src)

    # touched_by is emitted (the real cache-drop bug).
    assert entry.get("touched_by"), "touched_by must round-trip through the JSONL emit"

    touches = _pt._touches_from_entry(entry)
    assert len(touches) == 1
    assert touches[0].tier == "rewrite"
    assert touches[0].provider == "local"
    assert touches[0].decision_capture_id == "dc-rewrite-1"

    # Every scalar / tuple field survives.
    assert entry["content_type_label"] == "definition"
    assert entry["source_ids"] == ["dart:slug#b1", "dart:slug#b2"]
    assert entry["source_primary"] == "dart:slug#b1"
    assert entry["objective_ids"] == ["CO-01", "TO-02"]
    assert entry["bloom_level"] == "understand"


def test_touches_from_entry_tolerates_missing_and_malformed():
    """No ``touched_by`` key → empty tuple; a malformed Touch dict is dropped,
    not fatal (mirrors the surrounding field-coercion tolerance)."""
    assert _pt._touches_from_entry({}) == ()
    # decision_capture_id is a Wave-112 invariant; an empty one raises in the
    # Touch ctor, so it must be dropped rather than propagate.
    bad = {"touched_by": [{"tier": "rewrite", "provider": "local",
                           "model": "m", "timestamp": "t",
                           "decision_capture_id": "", "purpose": "p"}]}
    assert _pt._touches_from_entry(bad) == ()


def test_hydrate_blocks_from_path_reconstructs_touch_chain(tmp_path):
    """The gate-input router's hydration path reconstructs ``touched_by`` so a
    manifest built over runner-hydrated blocks carries provenance."""
    from MCP.hardening.gate_input_routing import (
        _hydrate_blocks_from_path,
        _touches_from_jsonl_entry,
    )

    blk = Block(
        block_id="week_02_content_01#concept_y_0",
        block_type="concept",
        page_id="week_02_content_01",
        sequence=0,
        content="<section>c</section>",
        touched_by=(_rewrite_touch(),),
    )
    p = tmp_path / "blocks_final.jsonl"
    p.write_text(
        json.dumps(_pt._block_to_snake_case_entry(blk), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hydrated = _hydrate_blocks_from_path(p)
    assert len(hydrated) == 1
    assert len(hydrated[0].touched_by) == 1
    assert hydrated[0].touched_by[0].tier == "rewrite"

    # Direct helper parity with pipeline_tools._touches_from_entry.
    entry = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert len(_touches_from_jsonl_entry(entry)) == 1


# --------------------------------------------------------------------------- #
# Root 4 — rewrite backstop deterministic transforms
# --------------------------------------------------------------------------- #


def test_inject_block_id_attr_stamps_first_tag():
    html = '<li data-cf-objective-id="TO-01">Define a prime.</li>'
    out = _pt._inject_block_id_attr(html, "week_01_content_01#objective_a_0")
    assert 'data-cf-block-id="week_01_content_01#objective_a_0"' in out
    # Only the FIRST tag is stamped (count=1) — no duplicate injection.
    assert out.count("data-cf-block-id") == 1


def test_inject_block_id_attr_noop_on_non_html_and_empty_id():
    assert _pt._inject_block_id_attr("plain text no tags", "b#x_0") == \
        "plain text no tags"
    assert _pt._inject_block_id_attr("<li>x</li>", "") == "<li>x</li>"


def test_bogus_source_id_element_stripped():
    """A model-hallucinated standalone ``<data-cf-source-ids=...>`` element is
    removed (it trips both the source-ref shape gate and the html parser)."""
    html = (
        '<li data-cf-block-id="b#x_0" data-cf-objective-id="TO-04" '
        'data-cf-bloom-level="understand">Understand primes.</li>'
        '<data-cf-source-ids="sample_course_chunk_00009"></data-cf-source-ids>'
    )
    out = _pt._BOGUS_SOURCE_ID_EL_RE.sub("", html)
    assert "data-cf-source-ids" not in out
    assert "Understand primes." in out


# --------------------------------------------------------------------------- #
# Root 5 — manifest emitter skips escalation-marked blocks
# --------------------------------------------------------------------------- #


def test_emit_manifest_skips_escalated_blocks(tmp_path, monkeypatch):
    """A block carrying an ``escalation_marker`` did not undergo successful
    synthesis, so it is EXCLUDED from the synthesis manifest (emitting a
    partial line would fail MANIFEST_SCHEMA_INVALID on empty synthesis.tiers).
    Real blocks still emit + the manifest is well-formed."""
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")

    real = Block(
        block_id="week_01_content_01#objective_a_0",
        block_type="objective",
        page_id="week_01_content_01",
        sequence=0,
        content='<li data-cf-block-id="b" data-cf-objective-id="TO-01">x</li>',
        objective_ids=("TO-01",),
        touched_by=(_rewrite_touch(),),
    )
    escalated = Block(
        block_id="week_07_content_01#objective_b_0",
        block_type="objective",
        page_id="week_07_content_01",
        sequence=0,
        # Degraded dict content (raw outline draft) — NOT empty, so it is not a
        # tombstone, but it carries the consensus-fail marker.
        content={"block_id": "b", "block_type": "objective"},
        escalation_marker="validator_consensus_fail",
    )

    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    mp = _pt._emit_block_synthesis_manifest(
        [real, escalated], content_dir, dart_chunks_path=None, capture=None,
    )
    assert mp is not None and mp.exists()
    lines = [l for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {json.loads(l)["block_id"] for l in lines}
    assert real.block_id in ids
    assert escalated.block_id not in ids, "escalated block must be excluded"


def test_emit_manifest_flag_off_is_noop(tmp_path, monkeypatch):
    """Default-off (flag unset) keeps emit byte-stable — no manifest written."""
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    real = Block(
        block_id="week_01_content_01#objective_a_0",
        block_type="objective",
        page_id="week_01_content_01",
        sequence=0,
        content="<li>x</li>",
        touched_by=(_rewrite_touch(),),
    )
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    mp = _pt._emit_block_synthesis_manifest(
        [real], content_dir, dart_chunks_path=None, capture=None,
    )
    assert mp is None
    assert not (content_dir / "block_synthesis_manifest.jsonl").exists()


# --------------------------------------------------------------------------- #
# Root 6 — 7B prose (explanation/example) blocks stamp CONCEPT CURIEs into the
# data-cf-source-ids attribute instead of canonical dart:{slug}#{chunk} refs.
#
# A live full-7B run produced a NON-HOLLOW course
# whose explanation/example blocks emitted
#   <section data-cf-source-ids="slug:concept_a,slug:concept_b">...
# Those ``slug:concept`` values are domain CURIEs (concept vocab), NOT source
# IDs, so the BLOCKING rewrite_source_refs gate fired
# OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE (action=block) on every prose block.
# The rewrite backstop now resolves each block's REAL source chunks (from the
# W2 outline_chunks.json sidecar -> chunks_lookup) into canonical dart: refs,
# OVERWRITES the bad attribute, sets structural source_ids, and PRESERVES the
# displaced CURIEs as a hidden data-cf-curie span (the curie-anchoring signal).
# --------------------------------------------------------------------------- #


# Slug-free synthetic identity for the pure-unit ref-minting tests below. These
# tests read NO on-disk LibV2 corpus — the slug is an arbitrary argument and the
# chunk ids are synthetic keys into the map — so per the project rule (tracked
# tests never hardcode a course slug) the identity is generic, not a pinned
# real-corpus slug. ``_chunk_id(n)`` derives the chunk-id strings from the slug
# so no literal corpus slug remains.
_SLUG = "sample-course"


def _chunk_id(n: int) -> str:
    """Synthetic ``{slug-as-underscores}_chunk_{n:05d}`` chunk id."""
    return f"{_SLUG.replace('-', '_')}_chunk_{n:05d}"


# Real, content_grounding-resolvable dart:{module_id}#{hash} sourceIds keyed by
# DART chunk id — exactly the chunk.source.source_references[].sourceId shape the
# rewrite backstop harvests from chunks.jsonl. The synthetic dart:{slug}#{chunk}
# mint these tests previously asserted was the root cause of the all-blocks
# UNRESOLVED_SOURCE_ID failure (it never resolves against the DART staging HTML).
_CHUNK_SOURCE_ID_MAP = {
    _chunk_id(1): ["dart:module-1#aaa111"],
    _chunk_id(2): ["dart:module-1#bbb222"],
}


def _apply_source_ref_backstop(
    block: Block, source_chunks, slug: str, chunk_source_id_map=None,
) -> Block:
    """Mirror the str-path source-ref repair the rewrite backstop applies.

    Replicates the canonical-source-id resolution + attribute overwrite +
    CURIE-preservation primitives from
    ``_run_content_generation_rewrite._apply_str_backstops`` using the same
    module-level helpers, so this test exercises the real production helpers
    (not a re-implementation of the ref-minting logic itself).
    """
    import dataclasses as dc
    import re as _re

    content = block.content
    assert isinstance(content, str)
    new_content = content

    if chunk_source_id_map is None:
        chunk_source_id_map = _CHUNK_SOURCE_ID_MAP
    canonical_refs = _pt._canonical_source_ids_for_chunks(
        source_chunks, slug, chunk_source_id_map=chunk_source_id_map,
    )
    displaced_curies = []
    seen = set()
    for m in _pt._SOURCE_IDS_ATTR_RE.finditer(new_content):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if (
                tok
                and not _pt._CANONICAL_DART_SOURCE_ID_RE.match(tok)
                and tok not in seen
            ):
                displaced_curies.append(tok)
                seen.add(tok)

    if canonical_refs:
        val = ",".join(canonical_refs)
        if _pt._SOURCE_IDS_ATTR_RE.search(new_content):
            new_content = _pt._SOURCE_IDS_ATTR_RE.sub(
                f' data-cf-source-ids="{val}"', new_content,
            )
        else:
            new_content = _pt._inject_source_ids_attr(new_content, val)
        block = dc.replace(
            block,
            source_ids=tuple(canonical_refs),
            source_references=tuple(
                {"sourceId": sid, "role": "primary", "extractor": "synthesized"}
                for sid in canonical_refs
            ),
        )
    else:
        new_content = _pt._SOURCE_IDS_ATTR_RE.sub("", new_content)

    if displaced_curies and "data-cf-curie" not in new_content:
        ct = " ".join(displaced_curies)
        new_content = (
            f'{new_content}<span hidden data-cf-curie="{ct}">{ct}</span>'
        )
    _ = _re  # keep import meaningful for readers
    return dc.replace(block, content=new_content)


def _curie_explanation_block() -> Block:
    """An explanation block exactly as the 7B rewrite tier emitted it."""
    return Block(
        block_id="week_01_content_01#explanation_x_1",
        block_type="explanation",
        page_id="week_01_content_01",
        sequence=1,
        content=(
            '<section data-cf-source-ids="samplecourse:prime_factorization,'
            'samplecourse:prime_numbers">'
            '<h2 data-cf-content-type="explanation" '
            'data-cf-block-id="week_01_content_01#explanation_x_1">'
            "Prime Factorization</h2>"
            "<p>A prime number is a natural number greater than 1.</p>"
            "</section>"
        ),
        objective_ids=("CO-01",),
        touched_by=(_rewrite_touch(),),
    )


def _grounded_chunks():
    return [
        {"id": _chunk_id(1), "text": "Prime numbers...",
         "heading": "Primes"},
        {"id": _chunk_id(2), "text": "Factorization...",
         "heading": "Primes"},
    ]


def test_canonical_source_ids_for_chunks_mints_dart_refs():
    """The helper resolves REAL dart:{module_id}#{hash} sourceIds from the
    chunk_source_id_map and never fabricates a ref for an ungrounded
    (no-text / no-id / unmapped) chunk entry."""
    refs = _pt._canonical_source_ids_for_chunks(
        [
            {"id": _chunk_id(1), "text": "real text"},
            {"id": "no_text_chunk", "text": ""},      # no text -> dropped
            {"id": "", "text": "orphan text"},         # no id -> dropped
            {"heading": "topic-fallback"},             # no id/text -> dropped
        ],
        _SLUG,
        chunk_source_id_map=_CHUNK_SOURCE_ID_MAP,
    )
    assert refs == ["dart:module-1#aaa111"]
    src_re = __import__(
        "Courseforge.router.inter_tier_gates", fromlist=["_SOURCE_ID_RE"],
    )._SOURCE_ID_RE
    assert src_re.match(refs[0]), "resolved ref must match the validator shape"


def test_canonical_source_ids_empty_when_no_grounded_chunks():
    """No grounded chunks / no map -> empty list (fail-closed, never
    fabricate)."""
    assert _pt._canonical_source_ids_for_chunks(
        [], _SLUG, chunk_source_id_map=_CHUNK_SOURCE_ID_MAP,
    ) == []
    assert _pt._canonical_source_ids_for_chunks(
        [{"id": _chunk_id(1), "text": ""}], _SLUG,
        chunk_source_id_map=_CHUNK_SOURCE_ID_MAP,
    ) == []
    assert _pt._canonical_source_ids_for_chunks(
        [{"id": _chunk_id(1), "text": "x"}], "",
        chunk_source_id_map=_CHUNK_SOURCE_ID_MAP,
    ) == []
    # No map supplied -> fail closed (never the synthetic dart:{slug}#{chunk}).
    assert _pt._canonical_source_ids_for_chunks(
        [{"id": _chunk_id(1), "text": "x"}], _SLUG,
    ) == []


def test_prose_block_curie_source_ids_rewritten_passes_gate():
    """An explanation block whose data-cf-source-ids carries CURIEs is
    rewritten to canonical dart: refs and then PASSES rewrite_source_refs."""
    from Courseforge.router.inter_tier_gates import BlockSourceRefValidator

    raw = _curie_explanation_block()
    # Sanity: the unfixed block trips the invalid-shape gate.
    pre = BlockSourceRefValidator().validate(
        {"blocks": [raw], "gate_id": "rewrite_source_refs"},
    )
    assert pre.passed is False and pre.action == "block"
    assert any(
        i.code == "OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE" for i in pre.issues
    )

    fixed = _apply_source_ref_backstop(raw, _grounded_chunks(), _SLUG)

    # Canonical refs landed on both the attribute and the structural field —
    # the REAL resolvable sourceIds from the chunk map, not a synthetic mint.
    assert fixed.source_ids == ("dart:module-1#aaa111", "dart:module-1#bbb222")
    assert 'data-cf-source-ids="dart:module-1#' in fixed.content
    assert "samplecourse:prime_factorization" not in (
        fixed.content.split("<span")[0]
    ), "CURIE must be removed from the source-ids attribute"

    post = BlockSourceRefValidator().validate(
        {"blocks": [fixed], "gate_id": "rewrite_source_refs"},
    )
    assert post.passed is True, [i.code for i in post.issues]
    assert post.action is None


def test_prose_block_displaced_curies_preserved_for_anchoring():
    """The CURIEs displaced from the source-ids attribute survive as a hidden
    data-cf-curie span whose TEXT content the curie-anchoring scrape sees."""
    from Courseforge.router.inter_tier_gates import (
        _extract_curies_from_block, _strip_html,
    )

    fixed = _apply_source_ref_backstop(
        _curie_explanation_block(), _grounded_chunks(), _SLUG,
    )
    assert 'data-cf-curie="' in fixed.content
    # The validator strips tags first, so the CURIE must live in TEXT content.
    stripped = _strip_html(fixed.content)
    assert "samplecourse:prime_factorization" in stripped
    scraped = _extract_curies_from_block(fixed)
    assert "samplecourse:prime_factorization" in scraped
    assert "samplecourse:prime_numbers" in scraped


def test_prose_block_no_grounding_strips_curie_attr_defers():
    """A prose block with NO grounded chunks drops the bogus CURIE attribute
    entirely and passes with DEFERRED (empty) attribution — never an invalid
    shape, never a fabricated ref."""
    from Courseforge.router.inter_tier_gates import BlockSourceRefValidator

    fixed = _apply_source_ref_backstop(
        _curie_explanation_block(), [], _SLUG,  # zero grounded chunks
    )
    assert fixed.source_ids in (None, ())
    assert "data-cf-source-ids=" not in fixed.content.split("<span")[0]
    res = BlockSourceRefValidator().validate(
        {"blocks": [fixed], "gate_id": "rewrite_source_refs"},
    )
    assert res.passed is True, [i.code for i in res.issues]
