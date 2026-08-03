"""Page-per-CO coverage-gap regression (the 193/604 stranded-CO defect).

Reproduces the live failure mode observed on a ``--reuse-objectives`` +
``ED4ALL_WEEK_TO_GROUPS=1`` + ``ED4ALL_CONTENT_PAGE_PER_CO=1`` run:

* The reused objectives doc's persisted ``chapter_objectives`` carried the
  week slicing of the RUN THAT MINTED IT (old ceil-stride "Week N" groups),
  which disagrees with the TO-membership grouping the CURRENT env flags ask
  for — and the reuse path persisted it verbatim, so the stale grouping flowed
  into the outline tier's ``week_co_objectives_index``.
* Independently, the page-per-CO NEVER-INCREASE guard
  (``_resolve_content_page_count``) resolves FEWER content pages than a
  CO-rich week has COs, and the historical 1:1 positional page↔CO binding
  stranded every CO past index ``n_pages - 1`` (193 stamped / 604 missing on
  the live 797-CO doc).

The fix (both halves locked here):

1. ``_page_co_slice_bounds`` — content page ``i`` binds a contiguous
   book-order SLICE of the week's COs; the slices exactly tile the week's CO
   list, so every CO lands on exactly one page even when pages < COs.
2. Week-group re-derivation — with ``ED4ALL_WEEK_TO_GROUPS`` on, both the
   runner's ``--reuse-objectives`` persist path and the outline tier rebuild
   the "Week N" groups live via the single-source ``_week_co_groups`` (doc's
   flat COs + terminal objectives) instead of trusting the persisted grouping.
   Flag off → persisted grouping consumed/persisted verbatim (byte-identical).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture doc: chapter_objectives grouping DISAGREES with TO membership.
# --------------------------------------------------------------------------- #
# TO membership (terminal_id backlinks): TO-01 → {CO-01}, TO-02 → {CO-02,
# CO-03, CO-04}. Persisted grouping: old ceil-stride slices — Week 1 holds
# {CO-01, CO-02}, Week 2 holds {CO-03, CO-04}. duration_weeks == num_tos == 2.
def _mismatched_reuse_doc() -> dict:
    cos = [
        {"id": "CO-01", "statement": "Identify the base terms.",
         "bloom_level": "remember", "parent_terminal": "TO-01",
         "source_refs": [{"ref": "TO-01", "chunk_ids": ["c1", "c2"]}]},
        {"id": "CO-02", "statement": "Apply the base terms.",
         "bloom_level": "apply", "parent_terminal": "TO-02",
         "source_refs": [{"ref": "TO-02", "chunk_ids": ["c3", "c4"]}]},
        {"id": "CO-03", "statement": "Analyze combined cases.",
         "bloom_level": "analyze", "parent_terminal": "TO-02",
         "source_refs": [{"ref": "TO-02", "chunk_ids": ["c5", "c6"]}]},
        {"id": "CO-04", "statement": "Evaluate edge cases.",
         "bloom_level": "evaluate", "parent_terminal": "TO-02",
         "source_refs": [{"ref": "TO-02", "chunk_ids": ["c7", "c8"]}]},
    ]
    return {
        "course_name": "FXCOVERAGE_101",
        "duration_weeks": 2,
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Foundations.",
             "bloom_level": "understand", "child_co_ids": ["CO-01"]},
            {"id": "TO-02", "statement": "Applications.",
             "bloom_level": "apply",
             "child_co_ids": ["CO-02", "CO-03", "CO-04"]},
        ],
        # STALE ceil-stride grouping (disagrees with TO membership above).
        "chapter_objectives": [
            {"chapter": "Week 1", "objectives": [cos[0], cos[1]]},
            {"chapter": "Week 2", "objectives": [cos[2], cos[3]]},
        ],
    }


def _group_ids(groups: list) -> list:
    return [[str(o["id"]) for o in g.get("objectives") or []] for g in groups]


# --------------------------------------------------------------------------- #
# 1. _page_co_slice_bounds — zero-stranded slice partition
# --------------------------------------------------------------------------- #
def test_slice_bounds_one_to_one_identity_when_pages_equal_cos():
    """n_pages == co_count → slice (i, i+1): byte-identical to 1:1 binding."""
    for i in range(5):
        assert _pt._page_co_slice_bounds(
            page_idx=i, n_pages=5, co_count=5
        ) == (i, i + 1)


def test_slice_bounds_tile_exactly_when_pages_fewer_than_cos():
    """The regression shape: pages < COs must still cover EVERY CO once."""
    for n_pages, co_count in [(2, 4), (19, 156), (20, 46), (15, 49), (1, 7)]:
        covered = []
        for i in range(n_pages):
            lo, hi = _pt._page_co_slice_bounds(
                page_idx=i, n_pages=n_pages, co_count=co_count
            )
            assert hi > lo, "a content page must never bind an empty slice"
            covered.extend(range(lo, hi))
        assert covered == list(range(co_count)), (
            f"slices for n_pages={n_pages}, co_count={co_count} do not tile"
        )


def test_slice_bounds_out_of_range_returns_empty():
    assert _pt._page_co_slice_bounds(page_idx=3, n_pages=3, co_count=5) == (0, 0)
    assert _pt._page_co_slice_bounds(page_idx=-1, n_pages=3, co_count=5) == (0, 0)
    assert _pt._page_co_slice_bounds(page_idx=0, n_pages=0, co_count=5) == (0, 0)
    assert _pt._page_co_slice_bounds(page_idx=0, n_pages=3, co_count=0) == (0, 0)


# --------------------------------------------------------------------------- #
# 2. Coverage: index + page-count + slice binding compose to zero stranded
# --------------------------------------------------------------------------- #
def test_coverage_clean_with_to_groups_and_page_cap(monkeypatch):
    """End-to-end helper composition: the exact live mechanism, fixed.

    Re-derive the week groups from TO membership (the outline-tier recipe),
    build the per-CO indexes, resolve the capped page count (topic-thin week:
    1 topic vs 3 COs — the NEVER-INCREASE guard bites), stamp each page's
    bound CO slice, and assert the coverage audit's expected-vs-stamped diff
    is EMPTY (pre-fix, CO-03/CO-04 were stranded).
    """
    monkeypatch.setenv("ED4ALL_WEEK_TO_GROUPS", "1")
    doc = _mismatched_reuse_doc()
    groups = _pt._normalize_chapter_objectives_to_groups(
        doc["chapter_objectives"]
    )
    flat = [o for g in groups for o in g.get("objectives") or []]
    duration_weeks = 2

    # Outline-tier re-derivation recipe (WEEK_TO_GROUPS on → live groups).
    live = _pt._week_co_groups(flat, doc["terminal_objectives"], duration_weeks)
    rederived = [
        {"chapter": f"Week {w}", "objectives": list(live.get(w, []))}
        for w in range(1, duration_weeks + 1)
    ]
    # Group N == TO-N's child_co_ids, book order.
    assert _group_ids(rederived) == [["CO-01"], ["CO-02", "CO-03", "CO-04"]]

    oi = _pt._build_week_co_objectives_index(
        rederived, duration_weeks=duration_weeks
    )
    ci = _pt._build_week_co_chunk_index(
        rederived, duration_weeks=duration_weeks, require_id=True
    )
    valid = {str(o["id"]) for o in flat}

    topic_counts = {1: 1, 2: 1}  # topic-thin: the never-increase guard bites
    stamped: dict = {}
    for w in range(1, duration_weeks + 1):
        m = len(ci.get(w, []))
        n = _pt._resolve_content_page_count(
            topic_count=topic_counts[w], co_count=m, page_per_co=True
        )
        assert n <= m  # the guard capped pages below the CO count in week 2
        objs = oi.get(w, [])
        for i in range(n):
            lo, hi = _pt._page_co_slice_bounds(
                page_idx=i, n_pages=n, co_count=m
            )
            for o in objs[lo:hi]:
                oid = str(o.get("id") or "")
                if oid in valid:
                    stamped[oid] = stamped.get(oid, 0) + 1

    expected = {
        str(o.get("id") or "")
        for w in range(1, duration_weeks + 1)
        for o in oi.get(w, [])
        if str(o.get("id") or "") in valid
    }
    # The coverage audit's exact comparison: zero missing, and exactly once.
    assert expected - set(stamped) == set()
    assert all(count == 1 for count in stamped.values())


# --------------------------------------------------------------------------- #
# 2b. ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED — true one-page-per-CO opt-in
# --------------------------------------------------------------------------- #
def test_uncapped_flag_lifts_topic_cap_to_one_page_per_co(monkeypatch):
    """(a) Flag on → n_pages == co_count and every page binds exactly 1 CO."""
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", "1")
    n = _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=True
    )
    assert n == 156
    for i in range(n):
        assert _pt._page_co_slice_bounds(
            page_idx=i, n_pages=n, co_count=156
        ) == (i, i + 1)


def test_uncapped_co_rich_topic_thin_week_headings_from_co(monkeypatch):
    """(b) 156 COs / 19 topics → 156 pages; pages beyond the topic list get a
    ``None`` topic and must derive heading/slug from the bound CO statement
    without crashing (the descriptor builder emits ``topic=None`` for
    ``page_idx >= topic_count`` and every downstream use is
    ``(topic or {})``-guarded)."""
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", "true")
    from lib.ontology.slugs import canonical_slug

    topic_count, co_count = 19, 156
    n = _pt._resolve_content_page_count(
        topic_count=topic_count, co_count=co_count, page_per_co=True
    )
    assert n == co_count
    # A page beyond the topic list: topic=None, bound to its own CO.
    stmt = "Evaluate rational expressions for excluded values."
    heading, slug = _pt._content_page_heading_slug(
        page_bound_co_id="CO-42",
        page_bound_co_statement=stmt,
        topic=None,  # pages past topic_count carry no topic dict
        page_type="content",
        week_num=1,
        slug_fn=canonical_slug,
    )
    assert heading == stmt
    assert slug == canonical_slug(" ".join(stmt.split()[:8]))
    # And with no bound CO either (edge), the None topic must not crash:
    heading2, slug2 = _pt._content_page_heading_slug(
        page_bound_co_id=None,
        page_bound_co_statement="",
        topic=None,
        page_type="content",
        week_num=1,
        slug_fn=canonical_slug,
    )
    assert heading2 == "week_01"
    assert slug2


def test_uncapped_unset_keeps_never_increase_guard(monkeypatch):
    """(c) Flag unset (and garbage) → capped behavior byte-identical."""
    monkeypatch.delenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", raising=False)
    assert _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=True
    ) == 19
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", "garbage")
    assert _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=True
    ) == 19
    # Inert when page_per_co is off — the legacy topic driver is untouched.
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", "1")
    assert _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=False
    ) == 19
    # Explicit arg wins over env (both directions).
    monkeypatch.delenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", raising=False)
    assert _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=True, uncapped=True
    ) == 156
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED", "1")
    assert _pt._resolve_content_page_count(
        topic_count=19, co_count=156, page_per_co=True, uncapped=False
    ) == 19


# --------------------------------------------------------------------------- #
# 3. Runner --reuse-objectives persist path re-derives under the flag
# --------------------------------------------------------------------------- #
@pytest.fixture
def runner_stub():
    from MCP.core.workflow_runner import WorkflowRunner

    return WorkflowRunner(executor=object(), config=object())


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    project = tmp_path / "PROJ-FXCOVERAGE_101-20260715"
    (project / "01_learning_objectives").mkdir(parents=True)
    return project


def _phase_outputs(project_dir: Path) -> dict:
    return {
        "objective_extraction": {
            "project_id": project_dir.name,
            "project_path": str(project_dir),
            "_completed": True,
        },
    }


def _run_reuse(runner_stub, tmp_path: Path, project_dir: Path) -> dict:
    reuse_path = tmp_path / "reuse_objectives.json"
    reuse_path.write_text(
        json.dumps(_mismatched_reuse_doc()), encoding="utf-8"
    )
    out = runner_stub._synthesize_course_planning_reuse_output(
        {"reuse_objectives_path": str(reuse_path),
         "course_name": "FXCOVERAGE_101"},
        _phase_outputs(project_dir),
    )
    assert out is not None
    persisted = (
        project_dir / "01_learning_objectives" / "synthesized_objectives.json"
    )
    return json.loads(persisted.read_text(encoding="utf-8"))


def test_reuse_rederives_week_groups_when_flag_on(
    monkeypatch, runner_stub, tmp_path, project_dir
):
    """Flag on → the doc's stale ceil-stride grouping must NOT silently win:
    the persisted chapter_objectives are re-derived to TO membership."""
    monkeypatch.setenv("ED4ALL_WEEK_TO_GROUPS", "1")
    doc = _run_reuse(runner_stub, tmp_path, project_dir)
    assert _group_ids(doc["chapter_objectives"]) == [
        ["CO-01"], ["CO-02", "CO-03", "CO-04"],
    ]
    assert [g["chapter"] for g in doc["chapter_objectives"]] == [
        "Week 1", "Week 2",
    ]
    # Flat LO surface unchanged — regrouping never adds/drops/edits a CO.
    co_rows = [
        e for e in doc["learning_outcomes"]
        if e.get("hierarchy_level") == "chapter"
    ]
    assert sorted(e["id"] for e in co_rows) == [
        "CO-01", "CO-02", "CO-03", "CO-04",
    ]


def test_reuse_persists_verbatim_when_flag_off(
    monkeypatch, runner_stub, tmp_path, project_dir
):
    """Default off → byte-identical: the doc's grouping persists verbatim."""
    monkeypatch.delenv("ED4ALL_WEEK_TO_GROUPS", raising=False)
    doc = _run_reuse(runner_stub, tmp_path, project_dir)
    assert _group_ids(doc["chapter_objectives"]) == [
        ["CO-01", "CO-02"], ["CO-03", "CO-04"],
    ]
