"""Pipeline-lane wiring regressions — findings 9, 11, 18.

Covers the three pipeline-WIRING fixes landed in
``MCP/tools/pipeline_tools.py`` for the Sonnet-vs-7B review:

* Finding 9 — the page ``<h1>`` is a HUMAN title derived from the week's
  terminal-objective / page-type, never the raw ``week_NN_summary``
  page_id (``_humanize_page_title``).
* Finding 11 — when the terminal-objective count is below the (pinned)
  week count, EVERY week still maps to a valid TO; no week ships an empty
  objectives block (``_assign_terminal_objectives_to_weeks``).
* Finding 18 — the overview page enumerates the week's TO + ALL its child
  COs (each with its statement) via the shared
  ``lib.objectives.lo_map_builder.render_objective_list`` builder, indexed
  off ``_index_objective_records``.

Pure-helper unit tests — no LLM client, no router, no filesystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402
from lib.objectives.lo_map_builder import render_objective_list  # noqa: E402


def _make_tos(n: int):
    return [
        {"id": f"TO-{i:02d}", "statement": f"Terminal objective {i} statement."}
        for i in range(1, n + 1)
    ]


def _make_cos_for_to(to_id: str, ids):
    return [
        {"id": cid, "statement": f"Chapter objective {cid} statement.",
         "terminal_id": to_id}
        for cid in ids
    ]


# --------------------------------------------------------------------------- #
# Finding 11 — no week left objective-less under a pinned --weeks.
# --------------------------------------------------------------------------- #


def test_assign_to_weeks_no_empty_week_when_tos_fewer_than_weeks():
    """9 TOs pinned to --weeks 12 -> every week gets a TO (the live bug)."""
    tos = _make_tos(9)
    assignment = _pt._assign_terminal_objectives_to_weeks(tos, 12)
    assert set(assignment) == set(range(1, 13))
    empty = [w for w, objs in assignment.items() if not objs]
    assert empty == [], f"weeks left objective-less: {empty}"
    # Trailing weeks (10/11/12) wrap round-robin back onto real TOs.
    assert assignment[10][0]["id"] == "TO-01"
    assert assignment[11][0]["id"] == "TO-02"
    assert assignment[12][0]["id"] == "TO-03"
    # Every TO still appears at least once across the assignment.
    seen = {o["id"] for objs in assignment.values() for o in objs}
    assert {t["id"] for t in tos} <= seen


def test_assign_to_weeks_ceil_stride_when_weeks_fewer_than_tos():
    """Weeks <= TOs -> contiguous ceil-stride slice, no TO dropped."""
    tos = _make_tos(10)
    assignment = _pt._assign_terminal_objectives_to_weeks(tos, 4)
    assert set(assignment) == {1, 2, 3, 4}
    assert all(assignment[w] for w in assignment), "a week was left empty"
    placed = {o["id"] for objs in assignment.values() for o in objs}
    assert placed == {t["id"] for t in tos}, "ceil-stride dropped a TO"


def test_assign_to_weeks_empty_when_no_tos():
    assert _pt._assign_terminal_objectives_to_weeks([], 8) == {}
    assert _pt._assign_terminal_objectives_to_weeks(None, 8) == {}


# --------------------------------------------------------------------------- #
# Finding 9 — human page <h1>, never the raw filename/page_id.
# --------------------------------------------------------------------------- #


def test_humanize_title_uses_terminal_objective_statement():
    title = _pt._humanize_page_title(
        "week_01_summary", 1, "summary",
        [{"id": "TO-01", "statement": "Explain prime numbers and factors."}],
    )
    assert title.startswith("Week 1:")
    assert "week_01_summary" not in title
    assert "Explain prime numbers" in title


def test_humanize_title_appends_page_type_label_for_content():
    title = _pt._humanize_page_title(
        "week_03_content_01", 3, "content",
        [{"id": "TO-03", "statement": "Convert between number representations."}],
    )
    assert title.startswith("Week 3:")
    assert "Convert between number representations" in title
    assert "Core Concepts" in title  # page-type framing keeps siblings distinct


def test_humanize_title_falls_back_to_page_type_label_without_tos():
    """No TO available -> friendly page-type label, still not the raw id."""
    title = _pt._humanize_page_title("week_05_self_check", 5, "self_check", [])
    assert title == "Week 5: Check Your Understanding"
    assert "week_05_self_check" not in title


# --------------------------------------------------------------------------- #
# Finding 18 — overview enumerates TO + ALL its COs (each statement).
# --------------------------------------------------------------------------- #


def test_index_objective_records_splits_tos_and_cos():
    doc = {
        "terminal_objectives": _make_tos(2),
        "chapter_objectives": [
            {"chapter": "Ch1",
             "objectives": _make_cos_for_to("TO-01", ["CO-01", "CO-02"])},
            {"chapter": "Ch2",
             "objectives": _make_cos_for_to("TO-02", ["CO-03"])},
        ],
    }
    to_map, co_map = _pt._index_objective_records(doc)
    assert set(to_map) == {"TO-01", "TO-02"}
    assert set(co_map) == {"CO-01", "CO-02", "CO-03"}
    assert co_map["CO-01"]["terminal_id"] == "TO-01"


def test_overview_render_lists_to_and_all_child_cos():
    """The overview fragment enumerates the TO and each of its COs + stmt."""
    to1 = {"id": "TO-01", "statement": "Reason about divisibility."}
    cos = _make_cos_for_to("TO-01", ["CO-01", "CO-02", "CO-03"])
    html = render_objective_list([to1], cos, heading_level=3)
    # TO present.
    assert 'data-cf-objective-id="TO-01"' in html
    assert "Reason about divisibility." in html
    # EVERY child CO id + its statement present (finding 18: 0 COs -> all COs).
    for co in cos:
        assert f'data-cf-objective-id="{co["id"]}"' in html
        assert co["statement"] in html


def test_overview_co_backfill_from_terminal_id():
    """An overview that declared only the TO still resolves its child COs.

    Mirrors the rewrite-phase backfill: pull every CO whose ``terminal_id``
    points at the page's TO even when the overview blocks declared only the
    TO id (the live 7B overview listed the TO only).
    """
    doc = {
        "terminal_objectives": _make_tos(1),
        "chapter_objectives": [
            {"chapter": "Ch1",
             "objectives": _make_cos_for_to("TO-01", ["CO-01", "CO-02"])},
        ],
    }
    to_map, co_map = _pt._index_objective_records(doc)
    # Page declared only TO-01 (objective_ids from the overview blocks).
    page_objective_ids = ["TO-01"]
    page_tos = [to_map[o] for o in page_objective_ids if o in to_map]
    to_ids = {t["id"] for t in page_tos}
    page_cos = [c for c in co_map.values() if c.get("terminal_id") in to_ids]
    assert {c["id"] for c in page_cos} == {"CO-01", "CO-02"}
    html = render_objective_list(page_tos, page_cos)
    assert 'data-cf-objective-id="CO-01"' in html
    assert 'data-cf-objective-id="CO-02"' in html
