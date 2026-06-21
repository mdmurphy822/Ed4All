"""Feature I5 — deterministic per-TO "Key Terms" page regressions.

Covers the CODE-only landing of the opt-in key-terms page (no GPU, no LLM):

* builder unit — deduped term union, source-preferred definition with
  ``definition_hint`` fallback, anti-fabrication OMISSION when neither resolves,
  well-formed source link;
* plumbing — page-type regex/enum/label/id-helper recognize ``key_terms``;
* deep-link fragment pinned to ``gui/services/source_page.heading_slug``;
* packaging — a ``week_NN_key_terms.html`` groups under its week via
  ``_iter_week_groups`` and sorts before ``summary``;
* determinism — byte-identical re-run of the builder;
* flag-off byte-stability — the page is NOT emitted and ``_WEEK_PAGE_TYPES``
  (the five-type LLM loop) is unchanged when the flag is off.

Pure-helper unit tests — no LLM client, no router, no model load.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_CF_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_CF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CF_SCRIPTS))

from MCP.tools import pipeline_tools as _pt  # noqa: E402
from lib.generation import key_terms as _kt  # noqa: E402
from gui.services.source_page import heading_slug as _gui_heading_slug  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures (synthesized — NO course slug / device path literals).
# --------------------------------------------------------------------------- #


def _vocab():
    return {
        "concepts": [
            {
                "canonical": "prime number",
                "aliases": ["prime"],
                "definition_hint": "A number divisible only by 1 and itself.",
            },
            {
                "canonical": "composite number",
                "aliases": [],
                "definition_hint": "A number with more than two factors.",
            },
            # No source sentence + no hint -> must be OMITTED.
            {"canonical": "orphan term", "aliases": [], "definition_hint": ""},
        ]
    }


def _chunks():
    return [
        {
            "id": "c1",
            "text": (
                "A prime number is a whole number greater than one with "
                "exactly two factors."
            ),
            "item_path": "ch1.html",
            "heading": "Section 1: Primes",
            "concept_tags": ["prime number"],
        },
        {
            # orphan term is a CANDIDATE (concept_tag) but its body carries no
            # sentence defining it AND the vocab concept has no hint -> the
            # block builder must OMIT it (anti-fabrication).
            "id": "c2",
            "text": "This passage is about unrelated arithmetic with apples.",
            "item_path": "ch1.html",
            "heading": "Section 2",
            "concept_tags": ["orphan term"],
        },
    ]


# --------------------------------------------------------------------------- #
# Builder unit — term union, definition resolution, OMISSION, source link.
# --------------------------------------------------------------------------- #


def test_aggregate_terms_dedupes_and_sorts():
    """Union of concept_tags + vocab surface-form + keyConcepts, dedup, sorted."""
    terms = _kt.aggregate_terms(
        concept_tags=["prime number", "Prime"],  # dup surface forms
        chunk_texts=[c["text"] for c in _chunks()],
        vocabulary=_vocab(),
        key_concepts=["composite number"],
    )
    slugs = [t["slug"] for t in terms]
    # prime-number reached via concept_tag + chunk surface-form collapses once.
    assert slugs == sorted(slugs), "terms not sorted by slug"
    assert slugs.count("prime-number") == 1, "duplicate term not deduped"
    assert "composite-number" in slugs  # via keyConcepts.


def test_definition_prefers_source_sentence_over_hint():
    """A grounding source sentence wins over the definition_hint."""
    term = {
        "slug": "prime-number",
        "display": "Prime number",
        "canonical": "prime number",
        "aliases": ["prime"],
        "definition_hint": "A number divisible only by 1 and itself.",
    }
    resolved = _kt.resolve_definition(term, chunks=_chunks())
    assert resolved is not None
    definition, chunk = resolved
    assert "whole number greater than one" in definition  # the source sentence.
    assert chunk is not None and chunk["id"] == "c1"


def test_definition_falls_back_to_hint_when_no_source_sentence():
    """No grounding sentence -> the definition_hint is used (chunk is None)."""
    term = {
        "slug": "composite-number",
        "display": "Composite number",
        "canonical": "composite number",
        "aliases": [],
        "definition_hint": "A number with more than two factors.",
    }
    resolved = _kt.resolve_definition(
        term, chunks=[{"id": "x", "text": "unrelated apples text"}]
    )
    assert resolved is not None
    definition, chunk = resolved
    assert definition == "A number with more than two factors."
    assert chunk is None  # hint fallback -> no defining chunk -> no link.


def test_anti_fabrication_omits_term_with_no_definition():
    """Neither a source sentence NOR a hint -> the term is OMITTED."""
    term = {
        "slug": "orphan-term",
        "display": "Orphan term",
        "canonical": "orphan term",
        "aliases": [],
        "definition_hint": "",
    }
    # The chunk mentions the term but in no full definition sentence form
    # AND there is no hint -> resolve_definition returns None.
    assert _kt.resolve_definition(term, chunks=[]) is None
    cards = _kt.build_key_terms_cards(
        terms=[term], chunks=[], course_slug="s"
    )
    assert cards == [], "undefinable term must be omitted (anti-fabrication)"


def test_source_link_is_well_formed_and_uses_heading_fragment():
    chunk = {"item_path": "ch1.html", "heading": "Section 1: Primes"}
    link = _kt.resolve_source_link(chunk, course_slug="my-course")
    assert link == (
        "/api/learn/source/my-course?item_path=ch1.html&fragment=section-1-primes"
    )


def test_source_link_none_without_item_path():
    assert _kt.resolve_source_link({"heading": "x"}, course_slug="s") is None
    assert _kt.resolve_source_link(None, course_slug="s") is None


# --------------------------------------------------------------------------- #
# Deep-link fragment pinned to source_page.heading_slug (FROZEN algorithm).
# --------------------------------------------------------------------------- #


def test_heading_slug_matches_gui_source_page():
    for h in [
        "Section 1: Primes",
        "Chapter 3 — Factoring!",
        "Mixed  CASE / and-stuff",
        "",
    ]:
        assert _kt.heading_slug(h) == _gui_heading_slug(h), h


# --------------------------------------------------------------------------- #
# Plumbing — page-type regex / enum / label / id-helper recognize key_terms.
# --------------------------------------------------------------------------- #


def test_page_id_type_re_recognizes_key_terms():
    m = _pt._PAGE_ID_TYPE_RE.match("week_04_key_terms")
    assert m is not None and m.group("ptype") == "key_terms"


def test_page_id_for_emits_key_terms_filename():
    assert _pt._page_id_for(4, "key_terms") == "week_04_key_terms"


def test_page_type_label_and_block_plan_present():
    assert _pt._PAGE_TYPE_LABELS["key_terms"] == "Key Terms"
    plan = _pt._page_block_plan_for("key_terms")
    assert plan and plan[0][0] == "vocab_card"


def test_key_terms_not_in_llm_page_loop():
    """key_terms is a deterministic post-pass, NOT in the five-type LLM loop."""
    assert _pt._WEEK_PAGE_TYPES == (
        "overview", "content", "application", "self_check", "summary",
    ), "the five-type LLM loop must stay byte-identical"
    assert "key_terms" not in _pt._WEEK_PAGE_TYPES


# --------------------------------------------------------------------------- #
# Builder -> Block integration (deterministic vocab_card blocks).
# --------------------------------------------------------------------------- #


def test_build_key_terms_blocks_emits_marked_vocab_cards():
    blocks = _pt._build_key_terms_blocks(
        week_num=2,
        objective_ids=("TO-02",),
        week_cos=[],
        week_chunk_details=_chunks(),
        vocabulary=_vocab(),
        course_slug="my-course",
        capture=None,
    )
    # orphan-term omitted; prime-number + composite-number survive.
    assert blocks, "no key-terms blocks emitted"
    for b in blocks:
        assert b.block_type == "vocab_card"
        assert b.page_id == "week_02_key_terms"
        # The deterministic-marker the rewrite tier reads to skip the LLM.
        assert b.template_type == _kt.KEY_TERMS_TEMPLATE_TYPE
        assert isinstance(b.content, str)
        assert 'data-cf-content-type="key-terms"' in b.content
        assert b.objective_ids == ("TO-02",)
    slugs = {b.key_terms[0] for b in blocks}
    assert "orphan-term" not in slugs  # anti-fabrication OMISSION.


def test_build_key_terms_blocks_empty_when_no_terms():
    blocks = _pt._build_key_terms_blocks(
        week_num=1,
        objective_ids=("TO-01",),
        week_cos=[],
        week_chunk_details=[],
        vocabulary=None,
        course_slug="s",
        capture=None,
    )
    assert blocks == []  # no terms -> no page emitted (no empty page).


# --------------------------------------------------------------------------- #
# Determinism — byte-identical re-run.
# --------------------------------------------------------------------------- #


def test_builder_is_deterministic():
    def _run():
        return [
            b.content
            for b in _pt._build_key_terms_blocks(
                week_num=3,
                objective_ids=("TO-03",),
                week_cos=[],
                week_chunk_details=_chunks(),
                vocabulary=_vocab(),
                course_slug="my-course",
                capture=None,
            )
        ]

    assert _run() == _run(), "builder output not byte-stable across re-runs"


# --------------------------------------------------------------------------- #
# Flag — default OFF, parse-with-fallback truthy.
# --------------------------------------------------------------------------- #


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_KEY_TERMS_PAGE", raising=False)
    assert _pt._key_terms_page_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
def test_flag_truthy_values(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_KEY_TERMS_PAGE", val)
    assert _pt._key_terms_page_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage", ""])
def test_flag_falsey_values(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_KEY_TERMS_PAGE", val)
    assert _pt._key_terms_page_enabled() is False


# --------------------------------------------------------------------------- #
# Packaging — a flat week_NN_key_terms.html groups under its week.
# --------------------------------------------------------------------------- #


def test_packager_groups_key_terms_page():
    from package_multifile_imscc import _iter_week_groups  # noqa: PLC0415

    d = Path(tempfile.mkdtemp())
    for name in (
        "week_02_overview.html",
        "week_02_self_check.html",
        "week_02_key_terms.html",
        "week_02_summary.html",
    ):
        (d / name).write_text("<html></html>", encoding="utf-8")
    groups = _iter_week_groups(d)
    assert len(groups) == 1
    names = [f.name for f in groups[0][3]]
    assert "week_02_key_terms.html" in names
    # Ordered after self_check, before summary.
    assert (
        names.index("week_02_self_check.html")
        < names.index("week_02_key_terms.html")
        < names.index("week_02_summary.html")
    )


# --------------------------------------------------------------------------- #
# Studio renderer — generate_course._render_key_terms_section.
# --------------------------------------------------------------------------- #


def test_render_key_terms_section_emits_cards_and_links():
    import generate_course as gc  # noqa: PLC0415

    terms = [
        {
            "term": "Prime Number",
            "definition": "Divisible only by 1 and itself.",
            "sourceLink": "/api/learn/source/c?item_path=ch1.html&fragment=primes",
        },
        {"term": "Composite", "definition": "More than two factors."},
    ]
    html = gc._render_key_terms_section(terms, page_id="week_02_key_terms")
    assert 'data-cf-content-type="key-terms"' in html
    assert "Prime Number" in html and "Composite" in html
    assert "View source" in html
    # The link is HTML-escaped (& -> &amp;) but present.
    assert "fragment=primes" in html


# --------------------------------------------------------------------------- #
# BUG 2 — per-page term cap (ED4ALL_KEY_TERMS_MAX_PER_PAGE, default 15).
# --------------------------------------------------------------------------- #


def test_max_terms_per_page_default_and_env(monkeypatch):
    monkeypatch.delenv("ED4ALL_KEY_TERMS_MAX_PER_PAGE", raising=False)
    assert _kt.resolve_max_terms_per_page() == 15
    monkeypatch.setenv("ED4ALL_KEY_TERMS_MAX_PER_PAGE", "8")
    assert _kt.resolve_max_terms_per_page() == 8
    # Explicit arg wins.
    assert _kt.resolve_max_terms_per_page(override=3) == 3
    # Garbage / non-positive -> default.
    for bad in ("0", "-2", "garbage", ""):
        monkeypatch.setenv("ED4ALL_KEY_TERMS_MAX_PER_PAGE", bad)
        assert _kt.resolve_max_terms_per_page() == 15


def _capacity_cards(n):
    """n cards: even-index source-defined, odd-index hint-only (source_chunk None)."""
    cards = []
    for i in range(n):
        slug = f"term-{i:02d}"
        cards.append({
            "slug": slug,
            "display": f"Term {i:02d}",
            "definition": f"def {i}",
            "source_link": None,
            "source_chunk": {"id": f"c{i}"} if i % 2 == 0 else None,
        })
    return sorted(cards, key=lambda c: c["slug"])


def test_select_capped_cards_caps_and_prefers_source_defined():
    cards = _capacity_cards(40)  # > cap
    capped = _kt.select_capped_cards(cards, max_terms=15)
    assert len(capped) == 15
    # All survivors must be source-defined (20 source-defined cards exist >= 15).
    assert all(isinstance(c.get("source_chunk"), dict) for c in capped)
    # Page order stays slug-sorted (the cap only drops the tail).
    slugs = [c["slug"] for c in capped]
    assert slugs == sorted(slugs)


def test_select_capped_cards_relevance_ranks_by_frequency():
    # Two source-defined cards; "alpha" mentioned more in chunk text + COs.
    cards = [
        {"slug": "alpha", "display": "Alpha", "definition": "d",
         "source_chunk": {"id": "c1"}, "source_link": None},
        {"slug": "beta", "display": "Beta", "definition": "d",
         "source_chunk": {"id": "c2"}, "source_link": None},
        {"slug": "gamma", "display": "Gamma", "definition": "d",
         "source_chunk": None, "source_link": None},
    ]
    capped = _kt.select_capped_cards(
        cards,
        max_terms=1,
        chunk_texts=["alpha alpha alpha beta"],
        co_statements=["alpha is important"],
    )
    assert len(capped) == 1
    assert capped[0]["slug"] == "alpha"  # highest frequency among source-defined.


def test_select_capped_cards_noop_under_cap():
    cards = _capacity_cards(5)
    assert _kt.select_capped_cards(cards, max_terms=15) == cards


def test_select_capped_cards_deterministic():
    cards = _capacity_cards(40)
    a = _kt.select_capped_cards(cards, max_terms=12, chunk_texts=["term-04 term-04"])
    b = _kt.select_capped_cards(cards, max_terms=12, chunk_texts=["term-04 term-04"])
    assert [c["slug"] for c in a] == [c["slug"] for c in b]


def test_build_key_terms_blocks_honors_cap(monkeypatch):
    # Aggregation yields many candidate terms; cap to a small number.
    concepts = [
        {"canonical": f"concept number {i}", "aliases": [],
         "definition_hint": f"Definition hint {i} with enough length."}
        for i in range(30)
    ]
    vocab = {"concepts": concepts}
    chunks = [{
        "id": f"c{i}",
        "text": f"This passage discusses concept number {i} in detail.",
        "item_path": "ch.html",
        "heading": "Sec",
        "concept_tags": [f"concept number {i}"],
    } for i in range(30)]
    monkeypatch.setenv("ED4ALL_KEY_TERMS_MAX_PER_PAGE", "10")
    blocks = _pt._build_key_terms_blocks(
        week_num=5,
        objective_ids=("TO-05",),
        week_cos=[],
        week_chunk_details=chunks,
        vocabulary=vocab,
        course_slug="my-course",
        capture=None,
    )
    assert len(blocks) == 10, "term cap not honored"
