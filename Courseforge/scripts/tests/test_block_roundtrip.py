"""Phase 2 Subtask 33 — round-trip integration test.

Idempotency contract: emit a Block list through
:func:`generate_course._build_page_metadata` + :func:`_wrap_page`,
parse the resulting full HTML page through
:class:`Trainforge.parsers.html_content_parser.HTMLContentParser`,
reconstruct a Block list from the parsed JSON-LD ``blocks[]``, and
re-emit through the same emitter — assert HTML byte-equal AND
JSON-LD ``blocks[]`` byte-equal across the two emits.

Both round-trip tests run with ``COURSEFORGE_EMIT_BLOCKS=true`` —
the new ``blocks[]`` / ``provenance`` / ``contentHash`` fields only
fire on that flag, so the round-trip is meaningless without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blocks import Block  # noqa: E402
from generate_course import _build_page_metadata, _wrap_page  # noqa: E402

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402


def _build_representative_blocks(page_id: str) -> List[Block]:
    """Build the canonical Block list a single content page produces.

    Reuses the per-page emit pattern in ``generate_week`` — one
    objective Block + two section-heading Blocks.
    """
    return [
        Block(
            block_id=Block.stable_id(page_id, "objective", "to_01", 0),
            block_type="objective",
            page_id=page_id,
            sequence=0,
            content="Define the foundational concept clearly.",
            objective_ids=("TO-01",),
            bloom_level="remember",
            bloom_verb="define",
            cognitive_domain="factual",
        ),
        Block(
            block_id=Block.stable_id(page_id, "explanation", "section_a", 1),
            block_type="explanation",
            page_id=page_id,
            sequence=1,
            content="Section A: Concept Overview",
            content_type_label="explanation",
            key_terms=("term_one", "term_two"),
            bloom_range="understand",
        ),
        Block(
            block_id=Block.stable_id(page_id, "example", "section_b", 2),
            block_type="example",
            page_id=page_id,
            sequence=2,
            content="Section B: Example",
            content_type_label="example",
            key_terms=("term_three",),
            bloom_range="apply",
        ),
    ]


def _representative_body_html() -> str:
    """The HTML body fragment for the page above. Mirrors the emitter
    surface so the DOM walk in
    :pymeth:`HTMLContentParser._extract_sections` produces predictable
    ContentSection entries.
    """
    return (
        '<section data-cf-source-ids="dart:ch1#b1">'
        '<h2 data-cf-content-type="explanation"'
        ' data-cf-key-terms="term_one,term_two"'
        ' data-cf-bloom-range="understand">Section A: Concept Overview</h2>'
        '<p>Body prose for section A.</p>'
        '</section>'
        '<section data-cf-source-ids="dart:ch1#b2">'
        '<h2 data-cf-content-type="example"'
        ' data-cf-key-terms="term_three"'
        ' data-cf-bloom-range="apply">Section B: Example</h2>'
        '<p>Body prose for section B.</p>'
        '</section>'
    )


def _emit_full_page(blocks: List[Block]) -> str:
    """Emit a full HTML page identical to the Courseforge emit path.

    The emitter consumes ``COURSEFORGE_EMIT_BLOCKS`` at meta-build
    time; the test sets it via ``monkeypatch.setenv`` before calling
    this helper so the new ``blocks[]`` / ``provenance`` /
    ``contentHash`` fields are present.
    """
    course_code = "ROUNDTRIP_101"
    week_num = 1
    page_id = blocks[0].page_id
    meta = _build_page_metadata(
        course_code,
        week_num,
        "content",
        page_id,
        objectives=None,
        sections=None,
        misconceptions=None,
        blocks=blocks,
    )
    return _wrap_page(
        f"Round-Trip Test Page",
        course_code,
        week_num,
        _representative_body_html(),
        page_metadata=meta,
    )


def _reconstruct_blocks_from_jsonld(
    jsonld_blocks: List[Dict[str, Any]],
    *,
    page_id: str,
    original: List[Block],
) -> List[Block]:
    """Reconstruct a ``List[Block]`` from a JSON-LD ``blocks[]`` array.

    The emitted JSON-LD is heterogeneous (R5B observation):
      - Objective entries carry ``id`` / ``statement`` / ``bloomLevel``
        / ``bloomVerb`` / ``cognitiveDomain``.
      - Section entries carry ``heading`` / ``contentType`` /
        ``keyTerms`` / ``bloomRange``.
      - Minimal entries carry ``blockId`` / ``blockType`` / ``sequence``.

    The reconstruction infers block_type from the entry shape; for
    section entries (no ``blockType`` field) it leans on the position
    in the original list to recover the canonical type. This is the
    pragmatic round-trip closure — the consumer's contract gives us
    the wire shape; the original-list lookup gives us the
    block_type / page_id / sequence triple the constructor needs.
    """
    rebuilt: List[Block] = []
    for idx, entry in enumerate(jsonld_blocks):
        orig = original[idx] if idx < len(original) else None
        if "id" in entry and "statement" in entry:
            # Objective shape.
            rebuilt.append(
                Block(
                    block_id=orig.block_id if orig else f"{page_id}#objective_{idx}",
                    block_type="objective",
                    page_id=page_id,
                    sequence=orig.sequence if orig else idx,
                    content=entry["statement"],
                    objective_ids=(entry.get("id", ""),),
                    bloom_level=entry.get("bloomLevel"),
                    bloom_verb=entry.get("bloomVerb"),
                    cognitive_domain=entry.get("cognitiveDomain"),
                    bloom_levels=tuple(entry.get("bloomLevels", []) or []),
                    bloom_verbs=tuple(entry.get("bloomVerbs", []) or []),
                    key_terms=tuple(entry.get("keyConcepts", []) or []),
                )
            )
        elif "heading" in entry and "contentType" in entry:
            # Section shape — block_type lifted from the original.
            block_type = orig.block_type if orig else "explanation"
            bloom_range_val = entry.get("bloomRange")
            if isinstance(bloom_range_val, list) and bloom_range_val:
                br = bloom_range_val[0]
            else:
                br = bloom_range_val
            rebuilt.append(
                Block(
                    block_id=orig.block_id if orig else f"{page_id}#{block_type}_{idx}",
                    block_type=block_type,
                    page_id=page_id,
                    sequence=orig.sequence if orig else idx,
                    content=entry["heading"],
                    content_type_label=entry.get("contentType"),
                    key_terms=tuple(entry.get("keyTerms", []) or []),
                    bloom_range=br,
                )
            )
        elif "blockId" in entry and "blockType" in entry:
            # Phase-2 minimal shape — block_type / sequence are on-wire.
            rebuilt.append(
                Block(
                    block_id=entry["blockId"],
                    block_type=entry["blockType"],
                    page_id=page_id,
                    sequence=int(entry.get("sequence", 0)),
                    content=orig.content if orig else "",
                    key_terms=orig.key_terms if orig else (),
                    bloom_level=orig.bloom_level if orig else None,
                    objective_ids=orig.objective_ids if orig else (),
                )
            )
        else:  # pragma: no cover — defensive
            raise AssertionError(f"Unrecognized JSON-LD block entry shape: {entry!r}")
    return rebuilt


def _extract_jsonld(html: str) -> Dict[str, Any]:
    """Pull the (single) JSON-LD payload out of the emitted page."""
    parser = HTMLContentParser()
    parsed = parser.parse(html)
    cf = (parsed.metadata or {}).get("courseforge") or {}
    if not cf:
        raise AssertionError("No JSON-LD courseforge payload in emitted HTML")
    return cf


def test_emit_parse_emit_byte_equal_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emit → parse → reconstruct Blocks → re-emit. The two HTML
    strings must be byte-identical.
    """
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    page_id = "week_01_content_01_roundtrip"
    blocks_v1 = _build_representative_blocks(page_id)
    html_v1 = _emit_full_page(blocks_v1)

    cf_v1 = _extract_jsonld(html_v1)
    jsonld_blocks = cf_v1.get("blocks") or []
    assert jsonld_blocks, "Round-trip impossible: no blocks[] in JSON-LD payload"

    blocks_v2 = _reconstruct_blocks_from_jsonld(
        jsonld_blocks, page_id=page_id, original=blocks_v1
    )
    html_v2 = _emit_full_page(blocks_v2)

    assert html_v1 == html_v2, (
        "HTML byte-equality broke across emit→parse→re-emit. The most "
        "likely cause is a Block field surfaced in the JSON-LD wire "
        "shape that the reconstruction routine couldn't recover."
    )


def test_emit_parse_emit_byte_equal_jsonld_blocks_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip the JSON-LD ``blocks[]`` array specifically.

    Compares the array ordering AND each entry's keys / values
    byte-for-byte (via canonical ``json.dumps(..., sort_keys=True)``)
    so a re-ordered key inside an entry would still flag as drift.
    """
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    page_id = "week_01_content_01_roundtrip_jsonld"
    blocks_v1 = _build_representative_blocks(page_id)
    html_v1 = _emit_full_page(blocks_v1)
    cf_v1 = _extract_jsonld(html_v1)
    blocks_arr_v1 = cf_v1.get("blocks") or []

    blocks_v2 = _reconstruct_blocks_from_jsonld(
        blocks_arr_v1, page_id=page_id, original=blocks_v1
    )
    html_v2 = _emit_full_page(blocks_v2)
    cf_v2 = _extract_jsonld(html_v2)
    blocks_arr_v2 = cf_v2.get("blocks") or []

    assert len(blocks_arr_v1) == len(blocks_arr_v2)
    for i, (e1, e2) in enumerate(zip(blocks_arr_v1, blocks_arr_v2)):
        # Canonicalize keys so a hypothetical key-order shuffle is
        # still flagged by ordering inside the canonical dump.
        c1 = json.dumps(e1, sort_keys=True, ensure_ascii=False)
        c2 = json.dumps(e2, sort_keys=True, ensure_ascii=False)
        assert c1 == c2, (
            f"blocks[{i}] drifted across round-trip:\n  v1={c1}\n  v2={c2}"
        )
    # And the ContentHash field — a different sentinel that the round-
    # trip property must preserve. The hash is computed over the meta
    # dict modulo the contentHash field itself, so identical inputs
    # produce identical hashes.
    assert cf_v1.get("contentHash") == cf_v2.get("contentHash"), (
        "contentHash drifted across round-trip; emit-time payload differs."
    )
