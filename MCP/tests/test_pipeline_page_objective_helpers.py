"""Lane C — unit tests for the pipeline_tools page-objective emit helpers.

Covers the four findings wired into ``MCP/tools/pipeline_tools.py``:

* Finding 9 (``_humanize_page_title``): the page ``<h1>`` is a human-readable
  ``"Week N: <label>"`` — never the bare ``week_NN_summary`` slug.
* Finding 11 + 18 (``_build_objectives_section_html``): a page with no
  objectives OMITS the objectives block (no blank heading); an overview page
  with a TO enumerates the TO + its child COs with their statements.
* Finding 17 (``lib.objectives.sub_objectives.populate_sub_objectives``): the
  module the planner wires in derives a grounded concept sub-layer per CO and
  stays a strict no-op when the CO carries no grounded chunks.

The helpers are factored to be pure so these tests run with no pipeline /
model dispatch. CPU-pinned in the run command; no course slugs / device paths
are hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_objectives_section_html,
    _humanize_page_title,
)


# --------------------------------------------------------------------------- #
# Finding 9 — humanized page <h1>
# --------------------------------------------------------------------------- #


def test_humanize_page_title_never_raw_slug() -> None:
    title = _humanize_page_title("week_01_summary", 1, "summary", None)
    assert title.startswith("Week 1:")
    # The bare snake-case slug must never survive into the rendered title.
    assert "week_01_summary" not in title
    # Falls back to a friendly page-type label when no TO statement is present.
    assert "Summary" in title


def test_humanize_page_title_uses_terminal_statement() -> None:
    week_objectives = [
        {"id": "TO-01", "statement": "Analyze linear equations in one variable"}
    ]
    title = _humanize_page_title(
        "week_03_overview", 3, "overview", week_objectives
    )
    assert title.startswith("Week 3:")
    assert "Analyze linear equations" in title


def test_humanize_page_title_appends_label_for_content_pages() -> None:
    week_objectives = [{"id": "TO-02", "statement": "Solve quadratic equations"}]
    title = _humanize_page_title(
        "week_02_content", 2, "content", week_objectives
    )
    # Content pages keep a page-type label so siblings stay distinct.
    assert title.startswith("Week 2:")
    assert "Solve quadratic equations" in title
    assert "Core Concepts" in title


# --------------------------------------------------------------------------- #
# Finding 11 — empty objectives block is OMITTED
# --------------------------------------------------------------------------- #


def test_objectives_section_omitted_when_empty() -> None:
    # No <li>s and no TO records → OMIT the block entirely (no blank heading).
    html = _build_objectives_section_html(
        objective_lis="",
        page_type="summary",
        page_tos=[],
        page_cos=[],
    )
    assert html == ""


def test_objectives_section_omitted_when_whitespace_only() -> None:
    html = _build_objectives_section_html(
        objective_lis="   \n  ",
        page_type="content",
        page_tos=None,
        page_cos=None,
    )
    assert html == ""


def test_objectives_section_compact_ul_for_non_overview() -> None:
    lis = '<li data-cf-objective-id="CO-01"><strong>CO-01:</strong> X</li>'
    html = _build_objectives_section_html(
        objective_lis=lis,
        page_type="content",
        page_tos=[{"id": "TO-01", "statement": "T"}],
        page_cos=[],
    )
    assert '<section class="objectives">' in html
    assert "<h2>Objectives</h2>" in html
    assert "CO-01" in html


# --------------------------------------------------------------------------- #
# Finding 18 — overview enumerates TO + its child COs with statements
# --------------------------------------------------------------------------- #


def test_overview_enumerates_terminal_and_child_cos() -> None:
    page_tos = [{"id": "TO-01", "statement": "Master arithmetic operations"}]
    page_cos = [
        {
            "id": "CO-01",
            "statement": "Add and subtract integers",
            "terminal_id": "TO-01",
        },
        {
            "id": "CO-02",
            "statement": "Multiply and divide fractions",
            "terminal_id": "TO-01",
        },
    ]
    objective_lis = '<li data-cf-objective-id="TO-01">TO-01</li>'
    html = _build_objectives_section_html(
        objective_lis=objective_lis,
        page_type="overview",
        page_tos=page_tos,
        page_cos=page_cos,
    )
    assert html, "overview with a TO must render an objectives section"
    # The TO statement and BOTH child-CO statements appear (not just TO id).
    assert "Master arithmetic operations" in html
    assert "Add and subtract integers" in html
    assert "Multiply and divide fractions" in html
    assert "CO-01" in html and "CO-02" in html


def test_overview_without_tos_falls_back_to_compact_ul() -> None:
    objective_lis = '<li data-cf-objective-id="CO-09">CO-09</li>'
    html = _build_objectives_section_html(
        objective_lis=objective_lis,
        page_type="overview",
        page_tos=[],  # no TO records — cannot enumerate the TO->CO tree
        page_cos=[],
    )
    # Still renders (has <li>s) but as the compact <ul>, not omitted.
    assert "<ul>" in html
    assert "CO-09" in html


# --------------------------------------------------------------------------- #
# Finding 17 — sub-objective derivation wiring contract (module call)
# --------------------------------------------------------------------------- #


def test_populate_sub_objectives_no_chunks_is_noop() -> None:
    """A CO with no grounded chunks keeps empty sub_objectives (byte-stable)."""
    from lib.objectives.sub_objectives import populate_sub_objectives

    chapter = [{"id": "CO-01", "statement": "Concept", "sub_objectives": []}]
    counters = populate_sub_objectives(
        chapter_objectives=chapter,
        chunks_by_id={},
        embedder=None,
    )
    assert counters["cos_seen"] == 1
    assert counters["cos_populated"] == 0
    assert chapter[0]["sub_objectives"] == []


def test_populate_sub_objectives_grounds_in_cited_chunks() -> None:
    """A CO with grounded chunks gets a sub_objective whose ids ⊆ cited set."""
    from lib.objectives.sub_objectives import populate_sub_objectives

    chapter = [
        {
            "id": "CO-01",
            "statement": "Understand prime factorization",
            "source_chunk_ids": ["c1"],
            "sub_objectives": [],
        }
    ]
    chunks_by_id = {
        "c1": {"id": "c1", "text": "Prime factorization breaks a number into "
                                   "the product of its prime factors."}
    }
    counters = populate_sub_objectives(
        chapter_objectives=chapter,
        chunks_by_id=chunks_by_id,
        embedder=None,  # degrades to a single un-clustered sub-objective
    )
    assert counters["cos_seen"] == 1
    derived = chapter[0]["sub_objectives"]
    assert derived, "a grounded CO must get >=1 sub-objective"
    # ANTI-FABRICATION: every sub-objective's chunk ids ⊆ the CO's cited set.
    for so in derived:
        assert set(so["source_chunk_ids"]) <= {"c1"}
