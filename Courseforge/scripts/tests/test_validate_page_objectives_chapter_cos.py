"""
Tests for per-week chapter-objective (CO-NN) conformance in the Courseforge
packager's page-objectives validator.

What this defends against
-------------------------
The packager rejected legitimately-declared chapter objectives. A faithful
run produced ``synthesized_objectives.json`` whose ``chapter_objectives`` are
a *dict-of-lists keyed on chapter number* (``{"1": [...], "2": [...]}``), with
each CO carrying a ``terminal_id``. The content generator
(``MCP/tools/pipeline_tools.py::_generate_course_content``) assigns those COs
to weeks by positional round-robin slicing across ``duration_weeks`` and
embeds them in each page's JSON-LD ``learningObjectives`` block.

The validator's ``load_canonical_objectives`` / ``resolve_week_objectives``,
however, computed the per-week allowed set with a ``[Ww]eek N(-M)?`` regex over
the chapter *label*. Number-keyed chapters carry no ``"Week N"`` substring, so
the regex matched nothing and every week's allowed set degraded to terminal
objectives only — packaging then failed with
``week N LO JSON-LD references ['CO-41', ...] which are NOT declared for this
week. Allowed for week N: ['TO-01'..'TO-11']``.

The fix teaches ``load_canonical_objectives`` to fall back to the emitter's
positional slicing for number-keyed / unlabeled chapters so the validator's
allowed set is byte-identical to what content generation emits.

Design side chosen: pages SHOULD be able to reference the chapter objectives
declared/assigned to their week (option (a) in the task brief). COs roll up to
terminals via ``terminal_id`` and are deliberately emitted by the generator;
the gate's job is to catch the LO-fanout / collapse defect (invented or
cross-course IDs), NOT to reject legitimately-assigned COs. Terminals stay
allowed course-wide.
"""

import json
import sys
from pathlib import Path

import pytest

# The scripts directory sits one level up from this tests/ dir.
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import (  # noqa: E402
    load_canonical_objectives,
    resolve_week_objectives,
)
from validate_page_objectives import validate_page  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: dict-of-lists chapter_objectives keyed by chapter NUMBER, the shape
# the plan_course_structure phase emits. 4 weeks, 8 COs over 4 chapters, plus
# 2 terminals. With 8 COs over 4 weeks the emitter's slicing assigns 2 COs per
# week in source order: week1->CO-01/CO-02, week2->CO-03/CO-04, etc.
# ---------------------------------------------------------------------------
NUMBER_KEYED_OBJECTIVES = {
    "course_name": "TST_908",
    "course_title": "Number-Keyed Chapter Objectives Fixture",
    "duration_weeks": 4,
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Evaluate accessibility", "bloomLevel": "evaluate"},
        {"id": "TO-02", "statement": "Design accessible UIs", "bloomLevel": "create"},
    ],
    "chapter_objectives": {
        "1": [
            {"id": "CO-01", "terminal_id": "TO-01", "statement": "Explain POUR", "bloomLevel": "understand"},
            {"id": "CO-02", "terminal_id": "TO-01", "statement": "Disability models", "bloomLevel": "understand"},
        ],
        "2": [
            {"id": "CO-03", "terminal_id": "TO-01", "statement": "Conformance levels", "bloomLevel": "understand"},
            {"id": "CO-04", "terminal_id": "TO-02", "statement": "Accessible names", "bloomLevel": "apply"},
        ],
        "3": [
            {"id": "CO-05", "terminal_id": "TO-02", "statement": "ARIA roles", "bloomLevel": "apply"},
            {"id": "CO-06", "terminal_id": "TO-02", "statement": "Keyboard nav", "bloomLevel": "apply"},
        ],
        "4": [
            {"id": "CO-07", "terminal_id": "TO-01", "statement": "Contrast ratios", "bloomLevel": "apply"},
            {"id": "CO-08", "terminal_id": "TO-02", "statement": "Color palettes", "bloomLevel": "create"},
        ],
    },
}


def _page_html(lo_ids):
    """Build a minimal HTML page whose JSON-LD references ``lo_ids``."""
    los = ",".join(f'{{"id": "{i}"}}' for i in lo_ids)
    return (
        "<!DOCTYPE html><html><head>"
        '<script type="application/ld+json">'
        f'{{"learningObjectives": [{los}]}}'
        "</script></head><body><h1>Page</h1></body></html>"
    )


@pytest.fixture
def canonical_path(tmp_path):
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(json.dumps(NUMBER_KEYED_OBJECTIVES))
    return p


@pytest.fixture
def canonical(canonical_path):
    return load_canonical_objectives(canonical_path)


# ---------------------------------------------------------------------------
# Unit: number-keyed chapters now resolve a non-terminal-only allowed set.
# ---------------------------------------------------------------------------

class TestNumberKeyedResolution:
    def test_every_week_gets_chapter_objectives(self, canonical):
        """Before the fix, number-keyed chapters mapped to NO weeks; allowed
        sets degraded to terminals only. Now every week carries COs."""
        weeks = canonical["week_to_chapter_objectives"]
        assert sorted(weeks.keys()) == [1, 2, 3, 4]
        for w in (1, 2, 3, 4):
            cos = [o["id"] for o in weeks[w]]
            assert cos, f"week {w} must carry chapter objectives, got {cos}"

    def test_week_allowed_set_matches_emitter_slicing(self, canonical):
        """The allowed CO set per week equals the emitter's positional slice."""
        week_cos = {
            w: sorted(o["id"] for o in resolve_week_objectives(w, canonical)
                      if o["id"].startswith("CO"))
            for w in (1, 2, 3, 4)
        }
        assert week_cos[1] == ["CO-01", "CO-02"]
        assert week_cos[2] == ["CO-03", "CO-04"]
        assert week_cos[3] == ["CO-05", "CO-06"]
        assert week_cos[4] == ["CO-07", "CO-08"]

    def test_terminals_allowed_every_week(self, canonical):
        for w in (1, 2, 3, 4):
            ids = {o["id"] for o in resolve_week_objectives(w, canonical)}
            assert {"TO-01", "TO-02"} <= ids


# ---------------------------------------------------------------------------
# Integration: validate_page accepts declared COs, rejects undeclared ones.
# ---------------------------------------------------------------------------

class TestValidatePageWithChapterObjectives:
    def test_page_referencing_declared_chapter_co_passes(self, tmp_path, canonical):
        """A week-2 page referencing CO-03 (declared/assigned to week 2) and a
        terminal must PASS — this is the case the packager wrongly rejected."""
        page = tmp_path / "week_02" / "content_01.html"
        page.parent.mkdir(parents=True)
        page.write_text(_page_html(["TO-01", "CO-03", "CO-04"]))
        ok, msg = validate_page(page, canonical)
        assert ok, f"declared CO references must pass; got: {msg}"

    def test_page_referencing_co_not_declared_for_course_fails(self, tmp_path, canonical):
        """A CO that exists in NO chapter of the course must still FAIL — the
        gate is not loosened to accept any id."""
        page = tmp_path / "week_02" / "content_01.html"
        page.parent.mkdir(parents=True)
        page.write_text(_page_html(["TO-01", "CO-99"]))
        ok, msg = validate_page(page, canonical)
        assert not ok, "undeclared CO-99 must be rejected"
        assert "CO-99" in msg

    def test_page_referencing_co_from_other_week_fails(self, tmp_path, canonical):
        """A CO declared for the course but assigned to a DIFFERENT week must
        fail on this week — the per-week scoping is preserved."""
        # CO-07 is assigned to week 4; referencing it from week 1 must fail.
        page = tmp_path / "week_01" / "content_01.html"
        page.parent.mkdir(parents=True)
        page.write_text(_page_html(["TO-01", "CO-07"]))
        ok, msg = validate_page(page, canonical)
        assert not ok, "CO-07 belongs to week 4, must not pass on week 1"
        assert "CO-07" in msg


# ---------------------------------------------------------------------------
# Regression: the legacy "Week N-M" labeled-chapter scheme still works.
# ---------------------------------------------------------------------------

WEEK_RANGE_OBJECTIVES = {
    "course_title": "Week-Range Labeled Fixture",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Evaluate", "bloomLevel": "evaluate"},
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1-2: Foundations",
            "objectives": [
                {"id": "CO-01", "statement": "Explain POUR", "bloomLevel": "understand"},
            ],
        },
        {
            "chapter": "Week 3-4: Visual Design",
            "objectives": [
                {"id": "CO-02", "statement": "Contrast", "bloomLevel": "apply"},
            ],
        },
    ],
}


class TestWeekRangeLabelStillWorks:
    @pytest.fixture
    def canonical(self, tmp_path):
        p = tmp_path / "objectives.json"
        p.write_text(json.dumps(WEEK_RANGE_OBJECTIVES))
        return load_canonical_objectives(p)

    def test_week_range_label_maps_to_both_weeks(self, canonical):
        for w in (1, 2):
            ids = {o["id"] for o in resolve_week_objectives(w, canonical)}
            assert "CO-01" in ids and "CO-02" not in ids
        for w in (3, 4):
            ids = {o["id"] for o in resolve_week_objectives(w, canonical)}
            assert "CO-02" in ids and "CO-01" not in ids
