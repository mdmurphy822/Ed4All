"""P3 (7b-sonnet-parity) — five page TYPES per week in the two-pass path.

The two-pass outline→rewrite tier historically emitted only one page type
per week (``content_NN``), giving ~2 pages/week. The Sonnet baseline emits
FIVE page types — ``overview`` / ``content`` / ``application`` /
``self_check`` / ``summary`` — which is the keystone for structural variety
AND chunking granularity (more page types × more heading boundaries →
more source sections → more chunks).

These pure-unit tests (no LLM, no pipeline) pin:

* the per-page-type plan yields five page types, each with the expected
  block-type composition;
* every (block_type, target_bloom) in every plan is wireable through the
  outline + rewrite tiers (member of BLOCK_TYPES, has outline bounds, has a
  rewrite output contract, valid Bloom level);
* the page_id helper encodes the page TYPE so the rewrite-tier group-by-
  ``page_id`` assembly produces five files per week;
* a week with N topics distributes blocks across the five page types (not
  all on ``content``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outline_provider import _OUTLINE_KIND_BOUNDS
from Courseforge.generators._rewrite_provider import (
    _BLOCK_TYPE_OUTPUT_CONTRACTS,
)
from Courseforge.scripts.blocks import BLOCK_TYPES
from lib.ontology.bloom import BLOOM_LEVELS
from MCP.tools.pipeline_tools import (
    _PAGE_BLOCK_PLAN,
    _PAGE_TYPE_BLOCK_PLAN,
    _WEEK_PAGE_TYPES,
    _page_block_plan_for,
    _page_id_for,
)


# --------------------------------------------------------------------------- #
# Page-type plan shape
# --------------------------------------------------------------------------- #


def test_five_canonical_page_types_present():
    """The plan covers exactly the five Sonnet-baseline page types."""
    assert set(_PAGE_TYPE_BLOCK_PLAN.keys()) == {
        "overview",
        "content",
        "application",
        "self_check",
        "summary",
    }
    # Emission order matters (overview first, summary last).
    assert _WEEK_PAGE_TYPES == (
        "overview",
        "content",
        "application",
        "self_check",
        "summary",
    )


def test_each_page_type_has_expected_block_composition():
    """Each page type carries the block mix mirroring the Sonnet pages."""
    plan = _PAGE_TYPE_BLOCK_PLAN
    # overview = intro + objectives + roadmap (objective + explanation).
    assert [bt for bt, _ in plan["overview"]] == ["objective", "explanation"]
    # content = concept/definition + explanation + worked example.
    assert [bt for bt, _ in plan["content"]] == [
        "concept",
        "explanation",
        "example",
    ]
    # application = scenario/problem cards + a worked example.
    assert [bt for bt, _ in plan["application"]] == ["activity", "example"]
    # self_check = two question/answer/reveal cards.
    assert [bt for bt, _ in plan["self_check"]] == [
        "self_check_question",
        "self_check_question",
    ]
    # summary = recap + checklist.
    assert [bt for bt, _ in plan["summary"]] == [
        "summary_takeaway",
        "recap",
    ]


def test_no_page_type_collapses_to_only_content():
    """The five page types are genuinely distinct compositions.

    Regression guard against silently reverting to a single page type:
    at least four DISTINCT block-type compositions across the five pages.
    """
    compositions = {
        tuple(bt for bt, _ in plan)
        for plan in _PAGE_TYPE_BLOCK_PLAN.values()
    }
    assert len(compositions) >= 4


def test_every_plan_entry_is_wireable():
    """Every (block_type, bloom) must dispatch through both tiers."""
    for page_type, plan in _PAGE_TYPE_BLOCK_PLAN.items():
        assert plan, f"page type {page_type} has an empty plan"
        for block_type, target_bloom in plan:
            assert block_type in BLOCK_TYPES, (
                f"{page_type}: {block_type} not in BLOCK_TYPES"
            )
            assert block_type in _OUTLINE_KIND_BOUNDS, (
                f"{page_type}: {block_type} missing outline bounds"
            )
            assert block_type in _BLOCK_TYPE_OUTPUT_CONTRACTS, (
                f"{page_type}: {block_type} missing rewrite contract"
            )
            assert target_bloom in BLOOM_LEVELS, (
                f"{page_type}: {target_bloom} not a Bloom level"
            )


def test_back_compat_alias_equals_content_plan():
    """The legacy flat ``_PAGE_BLOCK_PLAN`` aliases the content page type."""
    assert _PAGE_BLOCK_PLAN == _PAGE_TYPE_BLOCK_PLAN["content"]


# --------------------------------------------------------------------------- #
# page_id helper — encodes the page TYPE (the assembly group key)
# --------------------------------------------------------------------------- #


def test_page_id_encodes_page_type():
    """The page_id encodes the page TYPE so group-by-page yields 5 files."""
    assert _page_id_for(1, "overview") == "week_01_overview"
    assert _page_id_for(3, "application") == "week_03_application"
    assert _page_id_for(12, "self_check") == "week_12_self_check"
    assert _page_id_for(7, "summary") == "week_07_summary"
    # content pages carry the 1-based topic index.
    assert _page_id_for(1, "content", 1) == "week_01_content_01"
    assert _page_id_for(1, "content", 2) == "week_01_content_02"


def test_page_id_parses_as_week_plus_rest():
    """Every page_id splits into >=3 ``_``-parts (the _page_source_ids
    contract: ``week_NN_<rest>``)."""
    for page_type in _WEEK_PAGE_TYPES:
        pid = _page_id_for(4, page_type, 1)
        parts = pid.split("_")
        assert len(parts) >= 3
        assert re.match(r"^week_\d{2}_", pid)


def test_page_block_plan_for_unknown_falls_back_to_content():
    """An unknown page type defensively resolves to the content plan."""
    assert _page_block_plan_for("nonsense") == _PAGE_TYPE_BLOCK_PLAN["content"]
    for page_type in _WEEK_PAGE_TYPES:
        assert _page_block_plan_for(page_type) == _PAGE_TYPE_BLOCK_PLAN[
            page_type
        ]


# --------------------------------------------------------------------------- #
# Assembly distribution — blocks spread across five pages, not one
# --------------------------------------------------------------------------- #


def _simulate_week_page_descriptors(n_topics: int):
    """Mirror the outline-tier descriptor build: one content page per topic
    plus the four singleton page types, in canonical order.

    Returns the list of (page_type, content_idx) the loop would iterate.
    """
    descriptors = []
    for ptype in _WEEK_PAGE_TYPES:
        if ptype == "content":
            n_content = max(n_topics, 1)
            for ci in range(n_content):
                descriptors.append(("content", ci + 1))
        else:
            descriptors.append((ptype, 1))
    return descriptors


def test_week_distributes_blocks_across_five_page_types():
    """A 3-topic week yields five DISTINCT page types and emits each
    type's block plan onto its own page_id (not all on content)."""
    descriptors = _simulate_week_page_descriptors(n_topics=3)

    # Group emitted blocks by page_id the way the rewrite assembly does.
    pages: dict = {}
    for page_type, content_idx in descriptors:
        page_id = _page_id_for(2, page_type, content_idx)
        for block_type, _bloom in _page_block_plan_for(page_type):
            pages.setdefault(page_id, []).append(block_type)

    # Five page TYPES are present (overview/application/self_check/summary
    # singletons + 3 content pages = 7 page FILES, 5 distinct TYPES).
    page_ids = sorted(pages)
    assert "week_02_overview" in page_ids
    assert "week_02_application" in page_ids
    assert "week_02_self_check" in page_ids
    assert "week_02_summary" in page_ids
    content_pages = [p for p in page_ids if "_content_" in p]
    assert len(content_pages) == 3  # one per topic
    assert len(page_ids) == 7  # 4 singletons + 3 content

    # Blocks are NOT all on the content page(s) — each non-content type
    # carries its own block mix.
    assert pages["week_02_overview"] == ["objective", "explanation"]
    assert pages["week_02_application"] == ["activity", "example"]
    assert pages["week_02_self_check"] == [
        "self_check_question",
        "self_check_question",
    ]
    assert pages["week_02_summary"] == ["summary_takeaway", "recap"]


def test_zero_topic_week_still_emits_five_page_types():
    """A topic-less week (all headings filtered) still emits one content
    page plus the four singleton page types — never collapses to one."""
    descriptors = _simulate_week_page_descriptors(n_topics=0)
    page_types_emitted = {pt for pt, _ in descriptors}
    assert page_types_emitted == {
        "overview",
        "content",
        "application",
        "self_check",
        "summary",
    }
    # Exactly one content page in the topic-less case.
    assert sum(1 for pt, _ in descriptors if pt == "content") == 1
