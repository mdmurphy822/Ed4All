"""Phase 2 Subtask 32 — contract test (Trainforge ⇄ Courseforge).

For each applicable ``block_type`` in :data:`blocks.BLOCK_TYPES`, build a
``Block`` instance, project it to a full HTML page via
:func:`generate_course._build_page_metadata` + :func:`_wrap_page`, and
parse the result twice through
:class:`Trainforge.parsers.html_content_parser.HTMLContentParser`:

  1. ``COURSEFORGE_EMIT_BLOCKS=false`` — JSON-LD has no ``blocks[]``, so
     the parser falls through to the legacy regex DOM walk in
     :pymeth:`HTMLContentParser._extract_sections` for every section.
  2. ``COURSEFORGE_EMIT_BLOCKS=true`` — JSON-LD carries the canonical
     ``blocks[]`` projection; the parser merges
     :pymeth:`HTMLContentParser._content_sections_from_blocks` output
     with the DOM walk's output (Round 5B Subtask 30 wiring).

The test asserts per-section semantic equality across the two parse
runs for the key fields the consumer ultimately ingests:
``content_type_label`` (mirrored as ``ContentSection.content_type``),
``key_terms``, ``objective_refs``, ``source_references``, and
``template_type``. ``bloom_level`` lives on
:class:`LearningObjective` (page-level) — for section blocks it
manifests as ``data-cf-bloom-range`` and is captured via the
``bloomRange`` field on JSON-LD; we assert per-page LO consistency
where applicable.

R5B observation: section-shaped blocks emit via the legacy
``_section_jsonld()`` shape (``{heading, contentType, keyTerms,
bloomRange}``) which carries ``heading`` instead of ``blockType``;
:pymeth:`HTMLContentParser._content_sections_from_blocks` filters on
``blockType in _SECTION_BLOCK_TYPES`` so section-shape entries never
contribute extra :class:`ContentSection` rows. The two parse paths
therefore produce IDENTICAL section lists for section-typed blocks
— that identity IS the contract this test enforces.

Skip-if-not-applicable for ``chrome`` / ``prereq_set`` / ``misconception``
per the plan; those don't carry section equivalents in the consumer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blocks import Block  # noqa: E402
from generate_course import _build_page_metadata, _wrap_page  # noqa: E402

from Trainforge.parsers.html_content_parser import (  # noqa: E402
    HTMLContentParser,
    ParsedHTMLModule,
)


# Block types skipped per plan §L: chrome (template chrome — no
# section equivalent on the consumer side), prereq_set (page-level
# JSON-LD prerequisitePages array, no section), misconception
# (top-level JSON-LD misconceptions[] array, not a ContentSection).
_NON_APPLICABLE: frozenset = frozenset({"chrome", "prereq_set", "misconception"})


def _make_block_for_type(block_type: str) -> Tuple[Block, str]:
    """Build a representative Block of the given type plus a matching
    HTML body fragment.

    The HTML body must include whatever DOM elements the legacy
    :pymeth:`HTMLContentParser._extract_sections` would parse to
    produce a :class:`ContentSection` row — typically an
    ``<h2 data-cf-content-type="..." ...>`` heading.

    Returns ``(block, body_html)``.
    """
    page_id = f"page_{block_type}"
    if block_type == "objective":
        block = Block(
            block_id=Block.stable_id(page_id, "objective", "to_01", 0),
            block_type="objective",
            page_id=page_id,
            sequence=0,
            content="Define the foundational concept clearly.",
            objective_ids=("TO-01",),
            bloom_level="remember",
            bloom_verb="define",
            cognitive_domain="factual",
        )
        body = (
            '<section class="objectives" data-cf-source-ids="dart:ch1#b1">'
            '<h2>Learning Objectives</h2>'
            '<ul><li data-cf-objective-id="TO-01" data-cf-bloom-level="remember"'
            ' data-cf-bloom-verb="define" data-cf-cognitive-domain="factual">'
            'Define the foundational concept clearly.</li></ul></section>'
        )
        return block, body
    if block_type in {"explanation", "example", "concept", "summary_takeaway"}:
        # Section-heading blocks: emit `<h2 data-cf-content-type=...>` so
        # the DOM walk produces a ContentSection both ways.
        content_type_label = (
            "summary" if block_type == "summary_takeaway" else block_type
        )
        block = Block(
            block_id=Block.stable_id(page_id, block_type, "section", 0),
            block_type=block_type,
            page_id=page_id,
            sequence=0,
            content="Section Heading",
            content_type_label=content_type_label,
            key_terms=("term_one", "term_two"),
            bloom_range="understand",
        )
        body = (
            f'<section data-cf-source-ids="dart:ch1#b1">'
            f'<h2 data-cf-content-type="{content_type_label}"'
            f' data-cf-key-terms="term_one,term_two"'
            f' data-cf-bloom-range="understand">Section Heading</h2>'
            f'<p>Body prose with content for the section.</p>'
            f'</section>'
        )
        return block, body
    if block_type == "callout":
        block = Block(
            block_id=Block.stable_id(page_id, "callout", "note", 0),
            block_type="callout",
            page_id=page_id,
            sequence=0,
            content={"items": []},
            content_type_label="note",
        )
        body = (
            '<div class="callout" data-cf-content-type="note">'
            '<p>A callout note body.</p></div>'
        )
        return block, body
    if block_type == "flip_card_grid":
        block = Block(
            block_id=Block.stable_id(page_id, "flip_card_grid", "term_one", 0),
            block_type="flip_card_grid",
            page_id=page_id,
            sequence=0,
            content={"terms": [{"term": "Term One", "definition": "Def"}]},
            key_terms=("term_one",),
            teaching_role="reinforce",
        )
        body = (
            '<div class="flip-card" data-cf-component="flip-card"'
            ' data-cf-purpose="term-definition"'
            ' data-cf-teaching-role="reinforce" data-cf-term="term_one">'
            '<div>Term One</div><div>Def</div></div>'
        )
        return block, body
    if block_type == "self_check_question":
        block = Block(
            block_id=Block.stable_id(page_id, "self_check_question", "q1", 0),
            block_type="self_check_question",
            page_id=page_id,
            sequence=0,
            content={"question": "Q1?", "options": []},
            bloom_level="apply",
            teaching_role="assess",
            objective_ids=("TO-01",),
        )
        body = (
            '<div class="self-check" data-cf-component="self-check"'
            ' data-cf-purpose="formative-assessment"'
            ' data-cf-teaching-role="assess"'
            ' data-cf-bloom-level="apply"'
            ' data-cf-objective-ref="TO-01">Q1?</div>'
        )
        return block, body
    if block_type == "activity":
        block = Block(
            block_id=Block.stable_id(page_id, "activity", "practice", 0),
            block_type="activity",
            page_id=page_id,
            sequence=0,
            content={"title": "Practice", "description": "Do it"},
            bloom_level="apply",
            teaching_role="practice",
            objective_ids=("CO-01",),
        )
        body = (
            '<div class="activity-card" data-cf-component="activity"'
            ' data-cf-purpose="practice"'
            ' data-cf-teaching-role="practice"'
            ' data-cf-bloom-level="apply"'
            ' data-cf-objective-ref="CO-01">Practice — Do it</div>'
        )
        return block, body
    if block_type == "recap":
        block = Block(
            block_id=Block.stable_id(page_id, "recap", "wrap", 0),
            block_type="recap",
            page_id=page_id,
            sequence=0,
            content="Recap text.",
        )
        body = (
            '<section class="recap"><h2>Recap</h2>'
            '<p>Recap text.</p></section>'
        )
        return block, body
    if block_type == "reflection_prompt":
        block = Block(
            block_id=Block.stable_id(page_id, "reflection_prompt", "p", 0),
            block_type="reflection_prompt",
            page_id=page_id,
            sequence=0,
            content="Reflect on the topic.",
        )
        body = (
            '<section class="reflection"><h2>Reflection</h2>'
            '<p>Reflect on the topic.</p></section>'
        )
        return block, body
    if block_type == "discussion_prompt":
        block = Block(
            block_id=Block.stable_id(page_id, "discussion_prompt", "p", 0),
            block_type="discussion_prompt",
            page_id=page_id,
            sequence=0,
            content="Discuss with peers.",
        )
        body = (
            '<section class="discussion"><h2>Discussion Forum</h2>'
            '<p>Discuss with peers.</p></section>'
        )
        return block, body
    if block_type == "assessment_item":
        # Assessment items live in QTI XML on the IMSCC side; in HTML
        # this surfaces as a self-check-style probe. Reuse a self-check
        # body shape so the DOM walk has something to parse.
        block = Block(
            block_id=Block.stable_id(page_id, "assessment_item", "q1", 0),
            block_type="assessment_item",
            page_id=page_id,
            sequence=0,
            content={"question": "Q1?", "options": []},
            bloom_level="remember",
            objective_ids=("TO-01",),
        )
        body = (
            '<div class="self-check" data-cf-component="self-check"'
            ' data-cf-purpose="formative-assessment"'
            ' data-cf-bloom-level="remember"'
            ' data-cf-objective-ref="TO-01">Q1?</div>'
        )
        return block, body
    raise pytest.skip(f"No fixture builder for block_type={block_type!r}")


def _section_keys(section: Any) -> Dict[str, Any]:
    """Project a :class:`ContentSection` to the comparable key set.

    Lists are coerced to sorted tuples so ordering noise (set-based
    ``distinct_*`` produces sorted output but some legacy paths emit
    insertion order) doesn't false-flag a contract violation.
    """
    return {
        "heading": (section.heading or "").strip().lower(),
        "level": int(section.level),
        "content_type": section.content_type,
        "key_terms": tuple(sorted(section.key_terms or [])),
        "objective_refs": tuple(sorted(section.objective_refs or [])),
        "source_references": tuple(sorted(section.source_references or [])),
        "template_type": section.template_type,
    }


def _parse_with_emit_blocks(
    block: Block,
    body_html: str,
    *,
    emit_blocks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> ParsedHTMLModule:
    if emit_blocks:
        monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    else:
        monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "false")
    course_code = "TEST_101"
    week_num = 1
    page_id = block.page_id
    # ``objectives`` / ``sections`` / ``misconceptions`` builders read
    # from input dicts (not Block instances) — bypass them and pass the
    # Block list straight through. ``_build_page_metadata`` happily
    # builds a stub-page meta dict when the input collections are
    # empty, and the new ``blocks=`` arm fires off the env-flag check.
    meta = _build_page_metadata(
        course_code,
        week_num,
        "content",
        page_id,
        objectives=None,
        sections=None,
        misconceptions=None,
        blocks=[block],
    )
    html = _wrap_page(
        f"Page for {block.block_type}",
        course_code,
        week_num,
        body_html,
        page_metadata=meta,
    )
    parser = HTMLContentParser()
    return parser.parse(html)


@pytest.mark.parametrize(
    "block_type",
    sorted(t for t in (
        "objective",
        "concept",
        "example",
        "assessment_item",
        "explanation",
        "activity",
        "callout",
        "flip_card_grid",
        "self_check_question",
        "summary_takeaway",
        "reflection_prompt",
        "discussion_prompt",
        "recap",
    )),
)
def test_consumer_paths_produce_equivalent_sections(
    block_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For each applicable block_type, the legacy DOM walk path and
    the new JSON-LD blocks[] path produce semantically equivalent
    :class:`ContentSection` lists.

    The R5B consumer (:pymeth:`HTMLContentParser._content_sections_from_blocks`)
    filters Phase-2 ``blocks[]`` entries by ``blockType in
    _SECTION_BLOCK_TYPES``. Section-shaped Block emits use the legacy
    ``_section_jsonld()`` shape (``{heading, contentType, ...}``) which
    has no ``blockType``, so the new path adds zero extra ContentSection
    rows for them. Phase-2 minimal-shape entries (``flip_card_grid`` /
    ``self_check_question`` / ``activity`` etc.) DO carry ``blockType``,
    but their ``blockType`` value isn't in
    ``_SECTION_BLOCK_TYPES`` so they're also filtered out. Net: the
    consumer's ContentSection output is identical between the two
    paths — that identity is the contract.
    """
    if block_type in _NON_APPLICABLE:
        pytest.skip(f"{block_type} has no consumer-side ContentSection equivalent")

    block, body = _make_block_for_type(block_type)
    parsed_off = _parse_with_emit_blocks(
        block, body, emit_blocks=False, monkeypatch=monkeypatch
    )
    parsed_on = _parse_with_emit_blocks(
        block, body, emit_blocks=True, monkeypatch=monkeypatch
    )

    # Both parses MUST produce the same number of ContentSection rows.
    # If the new consumer path leaks an extra section (or drops one),
    # this fires.
    assert len(parsed_off.sections) == len(parsed_on.sections), (
        f"section count mismatch for block_type={block_type!r}: "
        f"off={len(parsed_off.sections)} on={len(parsed_on.sections)}"
    )

    # Per-section semantic equality on the contract fields. Sort by
    # heading first to absorb any ordering noise between the two
    # paths' section list assembly order.
    off_keys = sorted(
        (_section_keys(s) for s in parsed_off.sections),
        key=lambda k: (k["heading"], k["level"]),
    )
    on_keys = sorted(
        (_section_keys(s) for s in parsed_on.sections),
        key=lambda k: (k["heading"], k["level"]),
    )
    assert off_keys == on_keys, (
        f"per-section field mismatch for block_type={block_type!r}\n"
        f"off:\n{off_keys}\non:\n{on_keys}"
    )

    # Page-level objective_refs must match (both paths read from
    # data-cf-objective-ref attrs on .activity-card / .self-check
    # elements; the new path doesn't touch this field at all).
    assert sorted(parsed_off.objective_refs) == sorted(parsed_on.objective_refs)

    # Page-level source_references must match.
    off_src_ids = sorted(
        ref.get("sourceId", "") for ref in parsed_off.source_references
    )
    on_src_ids = sorted(
        ref.get("sourceId", "") for ref in parsed_on.source_references
    )
    assert off_src_ids == on_src_ids


def test_emit_blocks_flag_actually_changes_jsonld_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check — the env flag MUST in fact toggle whether
    ``blocks[]`` is present in the emitted JSON-LD payload, otherwise
    the contract test above is trivially true (the two paths would
    receive identical input).
    """
    block, body = _make_block_for_type("explanation")
    parsed_off = _parse_with_emit_blocks(
        block, body, emit_blocks=False, monkeypatch=monkeypatch
    )
    parsed_on = _parse_with_emit_blocks(
        block, body, emit_blocks=True, monkeypatch=monkeypatch
    )
    cf_off = parsed_off.metadata.get("courseforge") or {}
    cf_on = parsed_on.metadata.get("courseforge") or {}
    assert "blocks" not in cf_off, (
        "EMIT_BLOCKS=false leaked a blocks[] array into JSON-LD; "
        "the contract test isn't actually exercising the new path."
    )
    assert isinstance(cf_on.get("blocks"), list) and cf_on["blocks"], (
        "EMIT_BLOCKS=true didn't emit a non-empty blocks[]; the "
        "contract test isn't actually exercising the new path."
    )
    # And contentHash + provenance must additionally fire only on the
    # ON path (audit-trail surfaces gated by the same flag).
    assert "contentHash" not in cf_off
    assert "contentHash" in cf_on
    assert "provenance" not in cf_off
    assert "provenance" in cf_on


def test_consumer_extracts_block_metadata_when_emit_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5B's :pymeth:`HTMLContentParser._extract_blocks_from_jsonld`
    surfaces the ``blocks[]`` array verbatim. Confirm the blocks the
    Courseforge emitter wrote are recoverable verbatim through the
    consumer's parse path — no field drops, no shape mutation.
    """
    block, body = _make_block_for_type("explanation")
    parsed_on = _parse_with_emit_blocks(
        block, body, emit_blocks=True, monkeypatch=monkeypatch
    )
    cf = parsed_on.metadata.get("courseforge") or {}
    blocks_arr = cf.get("blocks") or []
    assert len(blocks_arr) == 1, blocks_arr
    entry = blocks_arr[0]
    # Section-shape entry: heading + contentType emitted; blockType
    # intentionally absent (R5B heterogeneous-shape contract).
    assert entry.get("heading") == "Section Heading"
    assert entry.get("contentType") == "explanation"
    assert entry.get("keyTerms") == ["term_one", "term_two"]
    assert entry.get("bloomRange") == ["understand"]
