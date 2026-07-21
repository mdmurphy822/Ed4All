"""Regression tests for the post_rewrite_validation blocker fix.

Covers the four Option-B repairs landed in
``MCP/tools/pipeline_tools.py``:

1. ``_canonical_source_ids_for_chunks`` resolves REAL resolvable
   ``semantik:{module_id}#{hash}`` sourceIds from a ``chunk_id -> [sourceId]``
   map (instead of minting a synthetic ``semantik:{course-slug}#{chunk_id}``),
   and fails closed (mints nothing) when a chunk is absent / has no text /
   resolves to zero sourceIds / no map is supplied.
2. ``_build_chunk_source_id_map`` harvests the real sourceIds from a
   chunks.jsonl ``source.source_references[].sourceId``.
3. ``_repair_rewrite_html`` deterministically repairs the five concrete 7B
   PARSE-FAIL malformations and is idempotent.

These are CPU-only unit tests — no ``ed4all run`` / model dispatch.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402


# --------------------------------------------------------------------- #
# 1. _canonical_source_ids_for_chunks: real sourceIds + anti-fabrication
# --------------------------------------------------------------------- #


def test_canonical_source_ids_emits_real_source_ids_from_map():
    """A grounded chunk resolves to its REAL semantik: sourceIds via the map."""
    chunks = [{"id": "chunk_00001", "text": "Real grounded prose."}]
    cmap = {"chunk_00001": ["semantik:module-1#abc123", "semantik:module-1#def456"]}
    out = _pt._canonical_source_ids_for_chunks(
        chunks, "course-slug", chunk_source_id_map=cmap,
    )
    assert out == ["semantik:module-1#abc123", "semantik:module-1#def456"]


def test_canonical_source_ids_dedups_in_discovery_order():
    chunks = [
        {"id": "c1", "text": "x"},
        {"id": "c2", "text": "y"},
    ]
    cmap = {
        "c1": ["semantik:m#a", "semantik:m#b"],
        "c2": ["semantik:m#b", "semantik:m#c"],  # b duplicate
    }
    out = _pt._canonical_source_ids_for_chunks(chunks, "s", chunk_source_id_map=cmap)
    assert out == ["semantik:m#a", "semantik:m#b", "semantik:m#c"]


def test_canonical_source_ids_mints_nothing_without_map():
    """Legacy / no-map callers fail closed to empty (never a synthetic ref)."""
    chunks = [{"id": "chunk_00001", "text": "prose"}]
    assert _pt._canonical_source_ids_for_chunks(chunks, "slug") == []
    assert _pt._canonical_source_ids_for_chunks(
        chunks, "slug", chunk_source_id_map={},
    ) == []


def test_canonical_source_ids_mints_nothing_when_chunk_absent():
    chunks = [{"id": "missing_chunk", "text": "prose"}]
    cmap = {"other_chunk": ["semantik:m#a"]}
    assert _pt._canonical_source_ids_for_chunks(
        chunks, "s", chunk_source_id_map=cmap,
    ) == []


def test_canonical_source_ids_mints_nothing_without_text():
    """A topic-paragraph fallback entry (no text) is not grounded provenance."""
    chunks = [{"id": "chunk_00001"}]  # no text key
    cmap = {"chunk_00001": ["semantik:m#a"]}
    assert _pt._canonical_source_ids_for_chunks(
        chunks, "s", chunk_source_id_map=cmap,
    ) == []


def test_canonical_source_ids_mints_nothing_when_map_value_empty():
    chunks = [{"id": "c1", "text": "prose"}]
    cmap = {"c1": []}
    assert _pt._canonical_source_ids_for_chunks(
        chunks, "s", chunk_source_id_map=cmap,
    ) == []


def test_canonical_source_ids_rejects_malformed_source_id():
    chunks = [{"id": "c1", "text": "prose"}]
    cmap = {"c1": ["NOT-A-SOURCE-ID", "semantik:m#good"]}
    out = _pt._canonical_source_ids_for_chunks(
        chunks, "s", chunk_source_id_map=cmap,
    )
    assert out == ["semantik:m#good"]


# --------------------------------------------------------------------- #
# 2. _build_chunk_source_id_map
# --------------------------------------------------------------------- #


def test_build_chunk_source_id_map_harvests_real_ids(tmp_path):
    import json

    chunks_path = tmp_path / "chunks.jsonl"
    lines = [
        {
            "id": "chunk_00001",
            "text": "prose",
            "source": {
                "source_references": [
                    {"sourceId": "semantik:module-1#aaa"},
                    {"sourceId": "semantik:module-1#bbb"},
                    {"sourceId": "not-canonical"},  # dropped
                ]
            },
        },
        {
            "id": "chunk_00002",
            "source": {"source_references": []},  # no valid ids -> no entry
        },
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )
    cmap = _pt._build_chunk_source_id_map(chunks_path)
    assert cmap == {"chunk_00001": ["semantik:module-1#aaa", "semantik:module-1#bbb"]}


def test_build_chunk_source_id_map_missing_file():
    assert _pt._build_chunk_source_id_map(Path("/nonexistent/chunks.jsonl")) == {}
    assert _pt._build_chunk_source_id_map(None) == {}


# --------------------------------------------------------------------- #
# 3. _repair_rewrite_html: the five concrete malformations + idempotence
# --------------------------------------------------------------------- #


def _is_balanced(html_text: str) -> bool:
    """True when span/cite/code are balanced and there are no bare ``<``."""
    counts = {"span": 0, "cite": 0, "code": 0}
    ok = {"balanced": True}

    class _S(HTMLParser):
        def handle_starttag(self, tag, attrs):  # noqa: ANN001
            if tag in counts:
                counts[tag] += 1

        def handle_endtag(self, tag):  # noqa: ANN001
            if tag in counts:
                counts[tag] -= 1
                if counts[tag] < 0:
                    ok["balanced"] = False

    p = _S(convert_charrefs=False)
    p.feed(html_text)
    p.close()
    if any(v != 0 for v in counts.values()):
        return False
    return ok["balanced"]


def _assert_repaired_and_idempotent(bad: str):
    fixed = _pt._repair_rewrite_html(bad)
    assert _is_balanced(fixed), f"not balanced: {fixed!r}"
    # Idempotent: a second pass changes nothing.
    assert _pt._repair_rewrite_html(fixed) == fixed
    return fixed


def test_repair_unclosed_span():
    bad = '<p>An example <span class="x">fraction is one half.</p>'
    _assert_repaired_and_idempotent(bad)


def test_repair_attribute_name_as_tag():
    bad = '<data-cf-block-id="week_04#objective_x_0">Objective text.</data-cf-block-id>'
    fixed = _assert_repaired_and_idempotent(bad)
    assert "data-cf-block-id=" not in fixed or "&lt;" not in fixed
    # The bogus pseudo-element is stripped entirely.
    assert "<data-cf-block-id" not in fixed
    assert "</data-cf-block-id>" not in fixed
    assert "Objective text." in fixed


def test_repair_unclosed_cite_multiple():
    bad = (
        '<p>See <cite>Source A and <cite>Source B and '
        '<cite>Source C and <cite>Source D.</p>'
    )
    _assert_repaired_and_idempotent(bad)


def test_repair_stray_code_closers():
    bad = "<p>Compute the value" + "</code>" * 18 + " here.</p>"
    fixed = _assert_repaired_and_idempotent(bad)
    assert "</code>" not in fixed


def test_repair_bare_pseudo_tag_curie():
    bad = "<p>The fraction <democourse:fraction>1/2 is shown.</p>"
    fixed = _assert_repaired_and_idempotent(bad)
    # The CURIE-as-tag is escaped to literal text, not left as a tag.
    assert "<democourse:fraction>" not in fixed
    assert "&lt;democourse:fraction&gt;" in fixed


def test_repair_idempotent_on_valid_html():
    good = '<section data-cf-block-id="x"><p>Clean prose with <span>emphasis</span>.</p></section>'
    assert _pt._repair_rewrite_html(good) == good


def test_repair_preserves_valid_tags_and_attrs():
    good = '<section data-cf-source-ids="semantik:m#a" class="c"><p>Text.</p></section>'
    fixed = _pt._repair_rewrite_html(good)
    assert 'data-cf-source-ids="semantik:m#a"' in fixed
    assert "<p>Text.</p>" in fixed


# --------------------------------------------------------------------- #
# 3b. CB2: visible leaked bracketed chunk-id tokens stripped from prose
# --------------------------------------------------------------------- #


def test_cb2_strips_leaked_chunk_token_and_leading_space():
    """A visible ``[..._chunk_NNNNN]`` token (and its leading space) is removed
    from learner prose; the surrounding prose stays intact; second run no-op."""
    bad = "<p>multiply by the reciprocal. [democourse_chunk_00043]</p>"
    fixed = _pt._repair_rewrite_html(bad)
    # Token gone; the leading space consumed (no trailing double space).
    assert "chunk_00043" not in fixed
    assert "[" not in fixed and "]" not in fixed
    assert fixed == "<p>multiply by the reciprocal.</p>"
    # Idempotent.
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_does_not_touch_source_id_attribute():
    """The bare (non-bracketed) chunk id inside a ``data-cf-source-ids``
    attribute value is NEVER stripped — only the bracketed form is removed."""
    good = (
        '<section data-cf-source-ids="democourse_chunk_00043">'
        "<p>Clean prose.</p></section>"
    )
    fixed = _pt._repair_rewrite_html(good)
    assert (
        'data-cf-source-ids="democourse_chunk_00043"' in fixed
    )
    assert fixed == good  # untouched + idempotent
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_strips_short_chunk_token_form():
    """The minimal ``[chunk_NN]`` form is also stripped."""
    bad = "<p>See the figure [chunk_12] for detail.</p>"
    fixed = _pt._repair_rewrite_html(bad)
    assert "chunk_12" not in fixed
    assert fixed == "<p>See the figure for detail.</p>"
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 3b-bis. CB2 (broadened): other bracketed source-id-like tokens stripped
#         (a source id ending in ``_NN`` with no "chunk" segment), while
#         numeric ref markers / ordinary bracketed words are KEPT.
# --------------------------------------------------------------------- #


def test_cb2_strips_non_chunk_source_id_token_and_leading_space():
    """A leaked ``[clean_simplify_fractions_01]`` (source id, no "chunk") token
    and its single leading space are removed from learner prose; prose intact;
    second run is a no-op."""
    bad = "<p>Divide the numerator by the denominator. [clean_simplify_fractions_01]</p>"
    fixed = _pt._repair_rewrite_html(bad)
    assert "clean_simplify_fractions_01" not in fixed
    assert "[" not in fixed and "]" not in fixed
    assert fixed == "<p>Divide the numerator by the denominator.</p>"
    # Idempotent.
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_broadened_still_strips_chunk_token():
    """The original ``[..._chunk_NNNNN]`` form is still stripped by the
    broadened pass (the chunk-only regex is subsumed, behavior preserved)."""
    bad = "<p>multiply by the reciprocal. [democourse_chunk_00043]</p>"
    fixed = _pt._repair_rewrite_html(bad)
    assert "chunk_00043" not in fixed
    assert fixed == "<p>multiply by the reciprocal.</p>"
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_keeps_numeric_reference_markers():
    """Short numeric citation markers ``[1]`` / ``[12]`` (no letter, no
    underscore) are legitimate reference markers and must be KEPT."""
    good = "<p>As shown earlier [1], the rule [12] holds.</p>"
    fixed = _pt._repair_rewrite_html(good)
    assert "[1]" in fixed
    assert "[12]" in fixed
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_keeps_bracketed_words():
    """Ordinary bracketed words ``[note]`` / ``[important]`` (no ``_<digits>``)
    must be KEPT — they are legitimate bracketed prose, not source ids."""
    good = "<p>[note] read this [important] section.</p>"
    fixed = _pt._repair_rewrite_html(good)
    assert "[note]" in fixed
    assert "[important]" in fixed
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_strips_legacy_dart_shaped_source_id_token():
    """LEGACY DUAL-READ COMPAT: a leaked legacy-prefixed source id
    ``[dart:photosynthesis#sec_01]`` (colon + '#', no ``_<digits>`` suffix) is
    stripped from learner prose; prose intact; second run is a no-op. The
    bracketed-token strip regex only matches the legacy ``dart:`` prefix, so
    this fixture deliberately keeps it to pin that legacy read path."""
    bad = (
        "<p>plants make their own food by photosynthesis. "
        "[dart:photosynthesis#sec_01]</p>"
    )
    fixed = _pt._repair_rewrite_html(bad)
    assert "dart:photosynthesis" not in fixed
    assert "[" not in fixed and "]" not in fixed
    assert fixed == "<p>plants make their own food by photosynthesis.</p>"
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_legacy_dart_strip_does_not_touch_source_id_attribute():
    """LEGACY DUAL-READ COMPAT: the bare legacy-prefixed id inside a
    ``data-cf-source-ids`` attribute value is NEVER stripped (bracketed form
    only)."""
    good = (
        '<section data-cf-source-ids="dart:photosynthesis#sec_01">'
        "<p>Clean prose.</p></section>"
    )
    fixed = _pt._repair_rewrite_html(good)
    assert 'data-cf-source-ids="dart:photosynthesis#sec_01"' in fixed
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb2_broadened_does_not_touch_source_id_attribute():
    """The bare (non-bracketed) source id inside a ``data-cf-source-ids``
    attribute value is NEVER stripped by the broadened pass."""
    good = (
        '<section data-cf-source-ids="clean_simplify_fractions_01">'
        "<p>Clean prose.</p></section>"
    )
    fixed = _pt._repair_rewrite_html(good)
    assert 'data-cf-source-ids="clean_simplify_fractions_01"' in fixed
    assert fixed == good  # untouched + idempotent
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 3b-ter. CB-FENCE: a markdown code fence WRAPPING the whole block (a leading
#         ``` ```html ``` opener + trailing ``` ``` ``` closer the 7B leaks)
#         is stripped; inner HTML intact; a fence-free block (incl. mid-content
#         backticks) is byte-identical; idempotent.
# --------------------------------------------------------------------- #


def test_cbfence_strips_wrapping_html_fence():
    """A whole block wrapped in a ``` ```html ... ``` ``` fence (the real
    captured ``assessment_item`` leak) has the fence stripped; the inner HTML is
    intact and starts with ``<div class="assessment-item"``; second run no-op."""
    bad = (
        '```html\n'
        '<div class="assessment-item" data-cf-block-id="ai-1">\n'
        "    <p>Divide: 3/4 by 1/2.</p>\n"
        "</div>\n"
        '```'
    )
    fixed = _pt._repair_rewrite_html(bad)
    assert "```" not in fixed
    assert fixed.startswith('<div class="assessment-item"')
    assert "<p>Divide: 3/4 by 1/2.</p>" in fixed
    assert fixed == (
        '<div class="assessment-item" data-cf-block-id="ai-1">\n'
        "    <p>Divide: 3/4 by 1/2.</p>\n"
        "</div>"
    )
    # Idempotent.
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbfence_strips_bare_fence_no_language():
    """A bare ``` ``` ``` fence (no language token) wrapping the block is
    stripped; inner HTML intact; second run no-op."""
    bad = "```\n<p>x</p>\n```"
    fixed = _pt._repair_rewrite_html(bad)
    assert "```" not in fixed
    assert fixed == "<p>x</p>"
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbfence_keeps_mid_content_backtick():
    """A block with NO wrapping fence and an inline mid-prose backtick is
    byte-identical out (no over-stripping); idempotent."""
    good = "<p>Use the <code>`x`</code> variable in `inline` prose.</p>"
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbfence_idempotent_on_clean_block():
    """An already-clean fence-free block is unchanged + idempotent."""
    good = '<div class="assessment-item"><p>Clean.</p></div>'
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 3b-quater. CB-LEAK: a leaked REASONING PREAMBLE (CJK chain-of-thought or a
#     short English preface) preceding the real HTML is stripped; the inner HTML
#     is intact; a clean block / legitimate intro prose is byte-identical;
#     idempotent. Mirrors the real ``flip_card_grid`` capture (a real
#     algebra-course capture): a TRUNCATED
#     first attempt, then the literal Chinese reasoning line
#     ``相反地，我将根据要求重新构造HTML内容：``, then a ``` ```html ``` fence wrapping a
#     COMPLETE second copy of the block (the leak doubled as a restart).
# --------------------------------------------------------------------- #


# The exact captured leak shape: a truncated head, the CJK reasoning line, then a
# ``` ```html ``` fence wrapping the complete restated block (closing ``` ``` ```).
_CAPTURED_CJK_PREAMBLE_LEAK = (
    '<div class="flip-card-grid" data-cf-block-id="fcg-1">\n'
    '    <div class="flip-card">\n'
    '        <div class="card-front"><span>Example</span></div>\n'
    '        <div class'  # <-- truncated first attempt cuts off mid-tag
    '相反地，我将根据要求重新构造HTML内容：\n\n'
    '```html\n'
    '<div class="flip-card-grid" data-cf-block-id="fcg-1">\n'
    '    <div class="flip-card">\n'
    '        <div class="card-front"><span>Example</span></div>\n'
    '        <div class="card-back">An example illustrates a concept.</div>\n'
    '    </div>\n'
    '</div>\n'
    '```'
)


def test_cbleak_strips_captured_cjk_reasoning_preamble():
    """The real captured CJK-preamble leak: the truncated head + Chinese
    reasoning line + ``` ```html ``` fence are stripped, leaving the COMPLETE
    restated block; no CJK / backticks survive; idempotent."""
    fixed = _pt._repair_rewrite_html(_CAPTURED_CJK_PREAMBLE_LEAK)
    # The Chinese chain-of-thought line is gone.
    assert "相反地" not in fixed
    assert not any("一" <= c <= "鿿" for c in fixed)
    # The wrapping fence is gone.
    assert "```" not in fixed
    # The complete restated block is kept (starts with the grid opener, carries
    # the real card-back prose that only the second copy contained).
    assert fixed.startswith('<div class="flip-card-grid"')
    assert "An example illustrates a concept." in fixed
    assert fixed.endswith("</div>")
    # Exactly ONE copy of the grid wrapper survives (the truncated head's partial
    # copy is gone — the leak is de-duped).
    assert fixed.count("flip-card-grid") == 1
    # Idempotent.
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbleak_strips_leading_english_preamble():
    """A LEADING English model preface ("Conversely, I will…") + ``` ```html ```
    fence before the block is stripped; the HTML is intact; idempotent."""
    bad = (
        'Conversely, I will rewrite the HTML content:\n\n'
        '```html\n'
        '<div class="concept"><p>A fraction is part of a whole.</p></div>\n'
        '```'
    )
    fixed = _pt._repair_rewrite_html(bad)
    assert "Conversely" not in fixed
    assert "```" not in fixed
    assert fixed == '<div class="concept"><p>A fraction is part of a whole.</p></div>'
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbleak_idempotent_on_clean_block_starting_with_div():
    """A clean block that already STARTS with ``<div>`` (no leak) is
    byte-identical out (no over-stripping); idempotent."""
    good = (
        '<div class="flip-card-grid" data-cf-block-id="fcg-1">\n'
        '    <div class="flip-card"><div class="card-back">Real.</div></div>\n'
        '</div>'
    )
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbleak_keeps_legitimate_leading_english_prose():
    """Legitimate leading English intro prose that is NOT an allowlisted preface
    and has NO CJK is left untouched (false-positive guard) — even though a
    block-level ``<div>`` follows it."""
    good = (
        'In this section we explore algebraic expressions in depth.\n'
        '<div class="concept"><p>An expression combines numbers and '
        'variables.</p></div>'
    )
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbleak_does_not_eat_p_with_here_is_before_div():
    """An ``<p>Here is …</p>`` paragraph (HTML already started) followed by a
    ``<div>`` is NOT stripped — the English arm is leading-only and refuses to
    cross the opening ``<p`` (guards against eating valid first paragraphs)."""
    good = '<p>Here is an introduction to fractions.</p>\n<div>real content</div>'
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 3b-bis. CB-SCRIPT: a stray ``<script>...</script>`` JSON-LD fragment the 7B
#     leaks into the rendered body (after the legitimate ``</section>``) is
#     stripped; the surrounding prose is intact; clean blocks are byte-identical.
#
# The existing ``data-cf-block-id`` pseudo-element strip ALREADY removes the
# trailing ``<data-cf-block-id="..." ...></data-cf-block-id>`` (its ``[^>]*>``
# tail covers the attribute-style form), but it does NOT touch ``<script>`` —
# the escape step treats ``<script ...>`` as a valid tag opener and copies the
# whole JSON-LD blob (incl. its visible JSON body) into the learner output. The
# CB-SCRIPT step closes that gap.
# --------------------------------------------------------------------- #


# The exact captured trailing leak region of the ``explanation`` block (block
# index 1 of ``inputs/contentgen/alg_blocks.json``) from the Sonnet-vs-7B
# real algebra-course fractions capture: a JSON-LD ``<script>`` blob AND a
# malformed ``<data-cf-block-id ...>`` pseudo-element appended after the
# legitimate ``</section>``.
_CAPTURED_EXPLANATION_LEAK = (
    '<section data-cf-source-ids="c1 c2 c3">\n'
    '    <h2 data-cf-content-type="explanation">Simplifying Fractions and '
    "Dividing Fractions</h2>\n"
    "    <p>To simplify fractions, we identify and divide out common "
    "factors.</p>\n"
    "</section> \n\n"
    '<script type="application/ld+json">\n'
    "    {\n"
    '        "@context": "https://schema.org",\n'
    '        "@type": "EducationalMaterial",\n'
    '        "name": "Simplifying Fractions and Dividing Fractions",\n'
    '        "description": "Explanation of simplifying fractions."\n'
    "    }\n"
    "</script> \n\n"
    "<!-- Required attributes -->\n"
    '<data-cf-block-id="week_07_alg_blocks#explanation_fractions_1" '
    'data-cf-content-type="explanation"></data-cf-block-id>'
)


def test_cbscript_strips_captured_explanation_jsonld_and_blockid():
    """The real captured ``explanation`` leak: the stray JSON-LD ``<script>``
    AND the malformed ``<data-cf-block-id ...>`` pseudo-element are stripped, the
    legitimate ``<section>`` prose is intact, and a second run is a no-op."""
    fixed = _pt._repair_rewrite_html(_CAPTURED_EXPLANATION_LEAK)
    # The stray script element + its JSON-LD body are gone (real + escaped form).
    assert "<script" not in fixed
    assert "</script>" not in fixed
    assert "&lt;script" not in fixed
    assert "ld+json" not in fixed
    assert "EducationalMaterial" not in fixed
    assert "@context" not in fixed
    # The malformed block-id pseudo-element is gone (existing strip still fires).
    assert "<data-cf-block-id" not in fixed
    assert "</data-cf-block-id>" not in fixed
    # The legitimate section prose is intact.
    assert (
        '<section data-cf-source-ids="c1 c2 c3">' in fixed
    )
    assert "Simplifying Fractions and Dividing Fractions" in fixed
    assert "To simplify fractions, we identify and divide out common factors." in fixed
    assert "</section>" in fixed
    # Idempotent.
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbscript_strips_escaped_script_fragment():
    """The entity-escaped ``&lt;script&gt;...&lt;/script&gt;`` form (the 7B
    double-escapes its own HTML) is also stripped; prose intact; idempotent."""
    bad = (
        "<p>Real prose stays.</p> "
        "&lt;script type=\"application/ld+json\"&gt;{\"@type\": \"X\"}"
        "&lt;/script&gt;"
    )
    fixed = _pt._repair_rewrite_html(bad)
    assert "script" not in fixed
    assert "@type" not in fixed
    assert "<p>Real prose stays.</p>" in fixed
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cbscript_clean_block_byte_identical():
    """A clean block carrying NO stray script is byte-identical out (no
    over-stripping); idempotent."""
    good = (
        '<section data-cf-source-ids="semantik:m#a">'
        "<p>Clean prose about fractions with <span>emphasis</span>.</p>"
        "</section>"
    )
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 3c. CB3: malformed list-item OPENER (missing ``>``) degrades to graceful,
#     STABLE, idempotent escaping — NOT auto-repaired (documented limitation).
#
# Decision: the ``<li ...text`` (missing ``>``) class is NOT auto-repaired
# because inserting a synthetic ``>`` at the next-``<`` boundary would absorb
# visible prose into the start tag, corrupting valid content. There is no
# deterministic attribute/text boundary once the ``>`` is gone. So the class
# degrades to graceful escaping (or tolerant pass-through), which these tests
# pin as STABLE + IDEMPOTENT, and is mitigated upstream by CB1's prompt.
# --------------------------------------------------------------------- #


def test_cb3_li_opener_missing_close_with_following_gt_is_stable():
    """A ``<li ...`` opener missing its ``>`` but with a later ``>`` (the
    ``</li>`` closer) is NOT corrupted: the malformed span is passed through
    verbatim, the result is byte-stable, and a second run is a no-op.

    Crucially, the repair pass must NEVER mangle this into something like
    ``<li class="y"Item two...>`` that swallows the prose — it stays as-is.
    """
    bad = (
        "<ul><li class=\"x\">Item one.</li>"
        "<li class=\"y\"Item two without close bracket.</li></ul>"
    )
    fixed = _pt._repair_rewrite_html(bad)
    # Graceful: no corruption, no prose lost; byte-stable + idempotent.
    assert fixed == bad
    assert "Item two without close bracket." in fixed
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb3_unterminated_opener_at_eof_escapes_gracefully():
    """A ``<li`` with no ``>`` anywhere after it (end of string) degrades to the
    safe literal-``&lt;`` escape — stable and idempotent (documented limitation,
    not a repair)."""
    bad = "<p>trailing <li no close at all"
    fixed = _pt._repair_rewrite_html(bad)
    assert "&lt;li" in fixed  # gracefully escaped, never auto-repaired
    assert _pt._repair_rewrite_html(fixed) == fixed


def test_cb3_valid_list_html_unchanged():
    """Well-formed list markup is a byte-identical no-op (no corruption)."""
    good = "<ul><li>One</li><li>Two</li></ul>"
    fixed = _pt._repair_rewrite_html(good)
    assert fixed == good
    assert _pt._repair_rewrite_html(fixed) == fixed


# --------------------------------------------------------------------- #
# 4. End-to-end backstop: source_ids + source_references + strip-unknown
# --------------------------------------------------------------------- #

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402

from Courseforge.scripts.blocks import Block  # noqa: E402


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives").mkdir(exist_ok=True)
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    return project_path


def _seed_outline_blocks(project_path: Path, blocks):
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = out_dir / "blocks_outline.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            fh.write(json.dumps(
                _pt._block_to_snake_case_entry(blk), ensure_ascii=False,
            ) + "\n")
    return blocks_path


def _seed_sidecars(project_path: Path, chunks_lookup, objectives):
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "outline_chunks.json"
    objectives_path = out_dir / "outline_objectives.json"
    chunks_path.write_text(json.dumps(chunks_lookup), encoding="utf-8")
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")
    return chunks_path, objectives_path


def _seed_libv2_chunks(libv2_root: Path, slug: str, chunks):
    chunks_dir = libv2_root / "courses" / slug / "semantik_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    return path


def _run_backstop_phase(tmp_path, monkeypatch, project_id, outline_block,
                        chunks_lookup, objectives, libv2_chunks, rewrite_html):
    project_path = _seed_project(tmp_path, project_id)
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)
    blocks_path = _seed_outline_blocks(project_path, [outline_block])
    chunks_path, objectives_path = _seed_sidecars(
        project_path, chunks_lookup, objectives,
    )
    libv2_root = tmp_path / "LibV2"
    slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_libv2_chunks(libv2_root, slug, libv2_chunks)

    from Courseforge.router import router as _router_mod

    def fake_remediation(self, block, **kw):
        return dataclasses.replace(block, content=rewrite_html)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        fake_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        outline_chunks_path=str(chunks_path),
        outline_objectives_path=str(objectives_path),
        libv2_root=str(libv2_root),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload
    final_path = project_path / "04_rewrite" / "blocks_final.jsonl"
    entries = [
        json.loads(l) for l in final_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return entries


def test_backstop_populates_source_ids_and_source_references(
    tmp_path, monkeypatch,
):
    """Backstop sets BOTH source_ids AND source_references from real chunk
    sourceIds (the latter closes the manifest_completeness cascade)."""
    block_id = "week_01_content_01#objective_to-01_0"
    outline = Block(
        block_id=block_id, block_type="objective",
        page_id="week_01_content_01", sequence=0,
        content={"objective_refs": ["TO-01"]},
        objective_ids=("TO-01",), bloom_level="remember",
    )
    chunks_lookup = {block_id: [{"id": "chunk_00001", "text": "Grounded prose."}]}
    objectives = [{"id": "TO-01", "statement": "Describe concept A clearly."}]
    libv2_chunks = [{
        "id": "chunk_00001",
        "text": "Grounded prose.",
        "source": {"source_references": [{"sourceId": "semantik:module-1#hashabc"}]},
    }]
    rewrite_html = '<section data-cf-objective-id="TO-01"><p>Body.</p></section>'

    entries = _run_backstop_phase(
        tmp_path, monkeypatch, "TEST_SRC_REFS", outline,
        chunks_lookup, objectives, libv2_chunks, rewrite_html,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["source_ids"] == ["semantik:module-1#hashabc"]
    assert e["source_references"] == [
        {"sourceId": "semantik:module-1#hashabc", "role": "primary",
         "extractor": "synthesized"}
    ]
    assert 'data-cf-source-ids="semantik:module-1#hashabc"' in e["content"]


def test_backstop_strips_unknown_objective_id(tmp_path, monkeypatch):
    """A hallucinated objective id (week_27 TO-27) not in the canonical set is
    stripped when the block resolves to no canonical objective."""
    block_id = "week_27_content_01#objective_x_0"
    # Outline block has NO objective_ids and NO page sibling with one, so it
    # resolves to no canonical objective -> strip-unknown fires.
    outline = Block(
        block_id=block_id, block_type="explanation",
        page_id="week_27_content_01", sequence=0,
        content="<p>stub</p>", bloom_level="understand",
    )
    chunks_lookup = {block_id: [{"id": "c1", "text": "Grounded prose."}]}
    objectives = [{"id": "TO-01", "statement": "Real canonical objective text."}]
    libv2_chunks = [{
        "id": "c1", "text": "Grounded prose.",
        "source": {"source_references": [{"sourceId": "semantik:m#h1"}]},
    }]
    rewrite_html = (
        '<section data-cf-objective-id="TO-27"><p>Hallucinated objective.</p>'
        '</section>'
    )
    entries = _run_backstop_phase(
        tmp_path, monkeypatch, "TEST_STRIP_UNK", outline,
        chunks_lookup, objectives, libv2_chunks, rewrite_html,
    )
    assert len(entries) == 1
    content = entries[0]["content"]
    assert "TO-27" not in content
    assert "Hallucinated objective." in content


def test_backstop_repairs_malformed_html(tmp_path, monkeypatch):
    """A block whose rewrite HTML is PARSE-FAIL (unclosed span) is repaired so
    the persisted content parses."""
    block_id = "week_01_content_01#objective_to-01_0"
    outline = Block(
        block_id=block_id, block_type="objective",
        page_id="week_01_content_01", sequence=0,
        content="<p>stub</p>", objective_ids=("TO-01",),
        bloom_level="remember",
    )
    chunks_lookup = {block_id: [{"id": "c1", "text": "Grounded prose."}]}
    objectives = [{"id": "TO-01", "statement": "Real canonical objective text."}]
    libv2_chunks = [{
        "id": "c1", "text": "Grounded prose.",
        "source": {"source_references": [{"sourceId": "semantik:m#h1"}]},
    }]
    rewrite_html = (
        '<section data-cf-objective-id="TO-01"><p>A <span>fraction is one '
        'half.</p></section>'
    )
    entries = _run_backstop_phase(
        tmp_path, monkeypatch, "TEST_HTML_FIX", outline,
        chunks_lookup, objectives, libv2_chunks, rewrite_html,
    )
    assert _is_balanced(entries[0]["content"]), entries[0]["content"]
