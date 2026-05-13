"""Tests for _plan_course_structure (Wave 24).

Covers the new course-outliner dispatch target that synthesizes real
TO-NN / CO-NN objectives from the textbook structure (or supplied
objectives JSON) and persists synthesized_objectives.json.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _normalize_chapter_objectives_to_groups,
)

_LO_ID_RE = re.compile(r"^[A-Z]{2,}-\d{2,}$")


@pytest.fixture
def planner_fixture(tmp_path, monkeypatch):
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    exports = fake_root / "Courseforge" / "exports"
    exports.mkdir(parents=True)
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )

    # Pre-create a project. Project dir name embeds the course name so
    # the planner's course_name → project directory lookup works.
    project_id = "PROJ-TESTCOURSE_101-20260420000000"
    project_dir = exports / project_id
    project_dir.mkdir()
    for subdir in ("00_template_analysis", "01_learning_objectives",
                   "02_course_planning", "03_content_development",
                   "04_quality_validation", "05_final_package"):
        (project_dir / subdir).mkdir()
    (project_dir / "project_config.json").write_text(
        json.dumps({
            "project_id": project_id,
            "course_name": "TESTCOURSE_101",
            "duration_weeks": 4,
            "credit_hours": 3,
        }, indent=2),
        encoding="utf-8",
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "staging_dir": staging,
    }


def _write_dart_html(path: Path, headings: list, learning_objectives: list = None):
    """Write minimal DART HTML with headings + paragraphs per section."""
    parts = ['<a class="skip-link" href="#main">Skip</a><main role="main">']
    if learning_objectives:
        parts.append('<section aria-labelledby="lo-head"><h2 id="lo-head">Learning Objectives</h2><ul>')
        for lo in learning_objectives:
            parts.append(f"<li>{lo}</li>")
        parts.append("</ul></section>")
    for idx, h in enumerate(headings, start=1):
        parts.append(
            f'<section aria-labelledby="s{idx}"><h2 id="s{idx}">{h}</h2>'
        )
        # Each paragraph must be >=40 chars and the whole section >=30 words.
        parts.append(
            f"<p>{h} is a foundational concept covered in this chapter of "
            f"the course. Understanding {h} requires students to carefully "
            f"examine its component parts and the relationships between "
            f"these parts in real-world educational contexts and applications.</p>"
            f"<p>Advanced study of {h} builds on prior knowledge of related "
            f"topics and emphasizes deep comprehension over superficial "
            f"memorization across multiple learning dimensions.</p>"
        )
        parts.append("</section>")
    parts.append("</main>")
    path.write_text("<html><body>" + "".join(parts) + "</body></html>",
                    encoding="utf-8")


async def _call(**kwargs):
    registry = _build_tool_registry()
    assert "plan_course_structure" in registry
    fn = registry["plan_course_structure"]
    raw = await fn(**kwargs)
    return json.loads(raw)


def test_missing_project_returns_error(planner_fixture):
    """Unknown project_id + no course_name → error."""
    result = asyncio.run(_call(project_id="NONEXISTENT-999"))
    assert "error" in result


def test_synthesizes_from_headings(planner_fixture):
    """No objectives file → synthesize from staged HTML headings."""
    fx = planner_fixture
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Photosynthesis Basics", "Light-Dependent Reactions",
         "The Calvin Cycle", "Factors Affecting Photosynthesis"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    assert result["mint_method"] == "synthesize_objectives_from_topics"
    # At least one TO and one CO minted.
    assert result["terminal_count"] >= 1

    objectives_path = Path(result["synthesized_objectives_path"])
    assert objectives_path.exists()
    doc = json.loads(objectives_path.read_text(encoding="utf-8"))
    assert "learning_outcomes" in doc
    for lo in doc["learning_outcomes"]:
        assert _LO_ID_RE.match(lo["id"]), (
            f"LO id {lo['id']!r} doesn't match canonical pattern"
        )
        assert lo["hierarchy_level"] in ("terminal", "chapter")


def test_populates_project_config_objectives_path(planner_fixture):
    """After planning, project_config.json carries synthesized_objectives_path."""
    fx = planner_fixture
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Intro Topic One", "Intro Topic Two"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    cfg_path = fx["project_dir"] / "project_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("synthesized_objectives_path") == result["synthesized_objectives_path"]
    # objectives_path also populated so downstream phases use it.
    assert cfg.get("objectives_path")


def test_objective_ids_are_list_of_canonical_ids(planner_fixture):
    """Wave2-I7: ``objective_ids`` is a ``List[str]`` of real LO IDs.

    Replaces the pre-Wave2-I7 comma-joined-string shape. The runner's
    ``_route_params`` (workflow_runner.py:1687-1688) collapses list
    values to comma-string on the wire, so downstream
    ``_generate_assessments`` (which already handles both shapes) is
    unaffected — but the in-memory phase_output stays a list so the
    ``trainforge_assessment.inputs_from`` route resolves to a non-None
    truthy value when the workflow resumes from this phase's
    checkpoint (dispatch-7 Finding 8).
    """
    fx = planner_fixture
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Topic A", "Topic B", "Topic C"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    ids = result["objective_ids"]
    assert isinstance(ids, list), (
        f"objective_ids must be a List[str]; got {type(ids).__name__}"
    )
    assert ids, "objective_ids must not be empty when topics are present"
    assert all(isinstance(x, str) for x in ids), (
        "every objective_ids entry must be a str"
    )
    # None of the IDs should be the pre-Wave-24 {COURSE}_OBJ_N phantom shape.
    for _id in ids:
        assert "_OBJ_" not in _id, (
            f"phantom id {_id!r} resurfaced — scope 2 regression"
        )
        assert _LO_ID_RE.match(_id), (
            f"id {_id!r} is not canonical"
        )


def test_objective_ids_canonical_order_terminals_first(planner_fixture):
    """Wave2-I7: ``objective_ids`` returns terminals before chapter LOs.

    Regression for dispatch-7 Finding 8: the
    ``trainforge_assessment.inputs_from`` route consumes this list and
    feeds the assessment generator; the canonical order keeps TO-NN
    grouping intact (terminal outcomes drive top-level assessment
    families, chapter outcomes nest under them).
    """
    fx = planner_fixture
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Topic Alpha", "Topic Beta", "Topic Gamma", "Topic Delta"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    ids = result["objective_ids"]
    assert isinstance(ids, list)

    # Cross-check against the persisted file so we test the canonical
    # shape rather than the in-memory return shape alone.
    doc = json.loads(
        Path(result["synthesized_objectives_path"]).read_text(encoding="utf-8")
    )
    expected_terminal = [
        str(t["id"]) for t in (doc.get("terminal_objectives") or [])
        if isinstance(t, dict) and t.get("id")
    ]
    expected_chapter: list = []
    for group in doc.get("chapter_objectives") or []:
        if not isinstance(group, dict):
            continue
        for entry in group.get("objectives") or []:
            if isinstance(entry, dict) and entry.get("id"):
                expected_chapter.append(str(entry["id"]))

    # Wave 24 mints TO-* before CO-* — assert ordering is preserved.
    assert ids == expected_terminal + expected_chapter, (
        "objective_ids must be terminals-first then chapter LOs; "
        f"got {ids!r}, expected {expected_terminal + expected_chapter!r}"
    )

    # Sanity: terminals come strictly before any chapter LO in the list.
    if expected_terminal and expected_chapter:
        last_terminal_idx = max(ids.index(t) for t in expected_terminal)
        first_chapter_idx = min(ids.index(c) for c in expected_chapter)
        assert last_terminal_idx < first_chapter_idx, (
            "terminals must precede chapter LOs in objective_ids"
        )


def test_objective_ids_empty_when_no_topics(planner_fixture):
    """Wave2-I7: empty corpus → empty list (NOT None, NOT missing key)."""
    fx = planner_fixture
    # No HTML in staging_dir → no topics → no LOs minted.
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    assert "objective_ids" in result, (
        "objective_ids key must be present even on empty corpus"
    )
    assert result["objective_ids"] == [], (
        f"expected empty list on empty corpus; got {result['objective_ids']!r}"
    )


def test_honors_supplied_objectives_json(planner_fixture):
    """When objectives_path is supplied, planner surfaces + persists them without re-synthesis."""
    fx = planner_fixture
    supplied = fx["project_dir"] / "supplied_objectives.json"
    supplied.write_text(json.dumps({
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Manually supplied terminal outcome.",
             "bloom_level": "analyze"},
        ],
        "chapter_objectives": [{
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "statement": "A manually supplied chapter outcome.",
                 "bloom_level": "remember"}
            ],
        }],
    }), encoding="utf-8")

    result = asyncio.run(_call(
        project_id=fx["project_id"],
        objectives_path=str(supplied),
    ))
    assert result["success"]
    assert result["mint_method"] == "user_supplied_objectives_json"
    assert result["terminal_count"] >= 1


def test_empty_corpus_falls_back_gracefully(planner_fixture):
    """No HTML and no supplied objectives → empty LO list; does not crash."""
    fx = planner_fixture
    # Empty staging dir.
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
    ))
    # Should still succeed; the persisted JSON just carries an empty list.
    assert result["success"]
    objectives_path = Path(result["synthesized_objectives_path"])
    assert objectives_path.exists()


def test_project_location_by_course_name(planner_fixture):
    """Lookup by course_name when project_id not passed."""
    fx = planner_fixture
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Alpha", "Beta"],
    )
    result = asyncio.run(_call(
        course_name="TESTCOURSE_101",
        staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    assert result["project_id"] == fx["project_id"]


# ---------------------------------------------------------------------- #
# Wave 40: duration_weeks precedence — auto-scaled config must beat stale
# kwargs when duration_weeks_explicit=False. Regression for the Wave 39
# smoke-test bug where the planner rewrote the auto-scaled 8 back to the
# kwargs default of 12, fanning out into 12 emitted week dirs instead of 8.
# ---------------------------------------------------------------------- #


def _seed_autoscaled_config(project_dir: Path, auto_weeks: int = 8) -> None:
    """Overwrite project_config.json to mimic a post-extractor auto-scaled state."""
    (project_dir / "project_config.json").write_text(
        json.dumps({
            "project_id": project_dir.name,
            "course_name": "TESTCOURSE_101",
            "duration_weeks": auto_weeks,
            "duration_weeks_autoscaled": True,
            "credit_hours": 3,
        }, indent=2),
        encoding="utf-8",
    )


def test_autoscaled_config_wins_when_duration_not_explicit(planner_fixture):
    """Wave 40: duration_weeks_explicit=False → config's 8 wins over stale kwargs 12."""
    fx = planner_fixture
    _seed_autoscaled_config(fx["project_dir"], auto_weeks=8)
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Topic One", "Topic Two"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        duration_weeks=12,
        duration_weeks_explicit=False,
    ))
    assert result["success"]

    cfg = json.loads(
        (fx["project_dir"] / "project_config.json").read_text(encoding="utf-8")
    )
    assert cfg["duration_weeks"] == 8, (
        f"stale kwargs (12) clobbered auto-scaled config (8): {cfg['duration_weeks']}"
    )

    objectives = json.loads(
        Path(result["synthesized_objectives_path"]).read_text(encoding="utf-8")
    )
    assert objectives["duration_weeks"] == 8


def test_explicit_duration_kwargs_wins(planner_fixture):
    """Wave 40: duration_weeks_explicit=True → kwargs value still wins."""
    fx = planner_fixture
    _seed_autoscaled_config(fx["project_dir"], auto_weeks=8)
    _write_dart_html(
        fx["staging_dir"] / "book.html",
        ["Topic One", "Topic Two"],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        duration_weeks=12,
        duration_weeks_explicit=True,
    ))
    assert result["success"]

    cfg = json.loads(
        (fx["project_dir"] / "project_config.json").read_text(encoding="utf-8")
    )
    assert cfg["duration_weeks"] == 12, (
        f"explicit kwargs should win over auto-scaled config, got {cfg['duration_weeks']}"
    )

    objectives = json.loads(
        Path(result["synthesized_objectives_path"]).read_text(encoding="utf-8")
    )
    assert objectives["duration_weeks"] == 12


# Wave 1.8: objective-driven dynamic week count. After the synthesizer
# emits real CO-NN objectives, re-scale duration_weeks to
# max(8, ceil(len(chapter_objectives) / _COS_PER_WEEK)) when
# duration_weeks_explicit=False. Replaces the chapter-driven pre-scale
# from the extractor with the authoritative objective count.
# ---------------------------------------------------------------------- #


def test_wave18_rescales_duration_weeks_from_objective_count(planner_fixture):
    """duration_weeks_explicit=False + many synthesized COs → re-scale.

    With 30 source headings the synthesizer emits enough chapter
    objectives that ``ceil(N / _COS_PER_WEEK)`` exceeds the 8-week
    floor and the planner promotes ``duration_weeks`` accordingly.
    """
    fx = planner_fixture
    _seed_autoscaled_config(fx["project_dir"], auto_weeks=8)
    # 30 distinct headings — enough to push CO count past 16 so
    # ceil(N/2) > 8 and the re-scale fires.
    headings = [f"Topic {i:02d}" for i in range(1, 31)]
    _write_dart_html(fx["staging_dir"] / "book.html", headings)

    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        duration_weeks=8,
        duration_weeks_explicit=False,
    ))
    assert result["success"]
    co_count = int(result["chapter_count"])
    expected_weeks = max(8, (co_count + 1) // 2)
    # Re-scale must fire: produced more weeks than the extractor's 8.
    assert expected_weeks > 8, (
        f"test fixture didn't trigger re-scale; got {co_count} COs / "
        f"expected_weeks={expected_weeks}"
    )
    assert result["duration_weeks"] == expected_weeks, (
        f"expected re-scale to {expected_weeks} weeks for {co_count} COs "
        f"(_COS_PER_WEEK=2); got {result['duration_weeks']}"
    )

    cfg = json.loads(
        (fx["project_dir"] / "project_config.json").read_text(encoding="utf-8")
    )
    assert cfg["duration_weeks"] == expected_weeks


def test_wave18_preserves_8week_floor_for_short_courses(planner_fixture):
    """Short courses (≤16 COs) stay at the 8-week floor."""
    fx = planner_fixture
    _seed_autoscaled_config(fx["project_dir"], auto_weeks=8)
    # 6 headings → 6 chapter objectives → ceil(6/2) = 3, max(8, 3) = 8.
    headings = [f"Topic {i:02d}" for i in range(1, 7)]
    _write_dart_html(fx["staging_dir"] / "book.html", headings)

    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        duration_weeks=8,
        duration_weeks_explicit=False,
    ))
    assert result["success"]
    assert result["duration_weeks"] == 8


def test_wave18_explicit_kwargs_skips_rescale(planner_fixture):
    """duration_weeks_explicit=True → re-scale is skipped, kwargs win."""
    fx = planner_fixture
    _seed_autoscaled_config(fx["project_dir"], auto_weeks=8)
    headings = [f"Topic {i:02d}" for i in range(1, 19)]  # would re-scale to 9
    _write_dart_html(fx["staging_dir"] / "book.html", headings)

    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        duration_weeks=12,
        duration_weeks_explicit=True,
    ))
    assert result["success"]
    # Operator's explicit 12 wins; no re-scale.
    assert result["duration_weeks"] == 12


# ---------------------------------------------------------------------- #
# Wave2b: _collect_lo_ids (inside _plan_course_structure) and
# _project_synthesized_objectives_to_course_json share a module-level
# normalizer that accepts three chapter_objectives shapes:
# list-of-groups (Courseforge synthesized), flat-list (LibV2
# component_objectives), and dict-of-lists (OpenStax shape). Before
# Wave2b the dict-of-lists shape silently dropped every CO-NN id from
# both helpers. Direct unit-coverage of the normalizer guards both
# call sites against regression.
#
# Cross-link: plans/wave2-smoke-verification-2026-05.md "Surprises".
# ---------------------------------------------------------------------- #


class TestNormalizeChapterObjectivesToGroups:
    """Wave2b: shared shape normalizer for chapter_objectives."""

    def test_dict_of_lists_shape_yields_all_co_ids(self):
        """OpenStax dict-of-lists shape: every CO-NN id surfaces in
        canonical sorted-key order, wrapped in synthetic groups whose
        ``chapter`` is the dict key.
        """
        raw = {
            "2": [
                {"id": "CO-03", "statement": "C3."},
                {"id": "CO-04", "statement": "C4."},
            ],
            "1": [
                {"id": "CO-01", "statement": "C1."},
                {"id": "CO-02", "statement": "C2."},
            ],
        }
        groups = _normalize_chapter_objectives_to_groups(raw)
        # Deterministic sorted-by-key iteration: "1" group before "2".
        assert [g["chapter"] for g in groups] == ["1", "2"]
        all_ids = [
            co["id"]
            for grp in groups
            for co in grp["objectives"]
        ]
        assert all_ids == ["CO-01", "CO-02", "CO-03", "CO-04"], (
            f"expected CO-01..CO-04 in sorted-chapter order; got {all_ids!r}"
        )

    def test_list_of_groups_shape_is_passed_through(self):
        """Courseforge synthesized list-of-groups shape: every CO-NN id
        is preserved with the original chapter labels.
        """
        raw = [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01"},
                    {"id": "CO-02"},
                ],
            },
            {
                "chapter": "Week 2",
                "objectives": [
                    {"id": "CO-03"},
                ],
            },
        ]
        groups = _normalize_chapter_objectives_to_groups(raw)
        assert [g["chapter"] for g in groups] == ["Week 1", "Week 2"]
        all_ids = [
            co["id"]
            for grp in groups
            for co in grp["objectives"]
        ]
        assert all_ids == ["CO-01", "CO-02", "CO-03"]

    def test_flat_list_shape_wraps_in_synthetic_group(self):
        """LibV2 archive flat-list shape (component_objectives without
        chapter grouping): every CO-NN id is preserved under a single
        synthetic ``chapter="(ungrouped)"`` group so the downstream
        walk still yields the full id set.
        """
        raw = [
            {"id": "CO-01"},
            {"id": "CO-02"},
            {"id": "CO-03"},
        ]
        groups = _normalize_chapter_objectives_to_groups(raw)
        # Single synthetic group; not silently dropped.
        assert len(groups) == 1
        assert groups[0]["chapter"] == "(ungrouped)"
        all_ids = [co["id"] for co in groups[0]["objectives"]]
        assert all_ids == ["CO-01", "CO-02", "CO-03"]


def _seed_supplied_objectives(
    project_dir: Path, *, chapter_objectives: object,
) -> Path:
    """Write a supplied-objectives JSON with the given chapter shape
    + return its path. Used by the ``_plan_course_structure``
    end-to-end shape-regression tests below.
    """
    supplied = project_dir / "supplied_objectives.json"
    supplied.write_text(json.dumps({
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "T1 statement.",
                "bloom_level": "apply",
            },
            {
                "id": "TO-02",
                "statement": "T2 statement.",
                "bloom_level": "analyze",
            },
        ],
        "chapter_objectives": chapter_objectives,
    }), encoding="utf-8")
    return supplied


def test_plan_course_structure_objective_ids_list_of_groups(planner_fixture):
    """Wave2b regression: supplied list-of-groups objectives → all
    TO-NN + CO-NN ids surface in objective_ids in terminals-first
    canonical order.
    """
    fx = planner_fixture
    supplied = _seed_supplied_objectives(
        fx["project_dir"],
        chapter_objectives=[
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "C1.", "bloom_level": "remember"},
                    {"id": "CO-02", "statement": "C2.", "bloom_level": "understand"},
                ],
            },
            {
                "chapter": "Week 2",
                "objectives": [
                    {"id": "CO-03", "statement": "C3.", "bloom_level": "apply"},
                ],
            },
        ],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        objectives_path=str(supplied),
    ))
    assert result["success"]
    ids = result["objective_ids"]
    assert ids[:2] == ["TO-01", "TO-02"]
    assert set(ids) == {"TO-01", "TO-02", "CO-01", "CO-02", "CO-03"}


def test_plan_course_structure_objective_ids_flat_list(planner_fixture):
    """Wave2b regression: supplied flat-list chapter_objectives →
    every CO-NN id surfaces in objective_ids.
    """
    fx = planner_fixture
    supplied = _seed_supplied_objectives(
        fx["project_dir"],
        chapter_objectives=[
            {"id": "CO-01", "statement": "C1.", "bloom_level": "remember"},
            {"id": "CO-02", "statement": "C2.", "bloom_level": "understand"},
            {"id": "CO-03", "statement": "C3.", "bloom_level": "apply"},
        ],
    )
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        objectives_path=str(supplied),
    ))
    assert result["success"]
    ids = result["objective_ids"]
    # Terminals first, then chapters.
    assert ids[:2] == ["TO-01", "TO-02"]
    assert set(ids) == {"TO-01", "TO-02", "CO-01", "CO-02", "CO-03"}


def test_collect_lo_ids_dict_of_lists_shape():
    """Wave2b: ``_collect_lo_ids`` (closure inside
    ``_plan_course_structure``) must recover every CO-NN id from a
    dict-of-lists ``chapter_objectives`` payload.

    The closure isn't importable directly, so we replicate its body
    here using the SAME module-level normalizer it calls. Pre-Wave2b
    the closure walked ``chapter_objectives`` assuming list-only
    iteration; a dict payload was silently iterated as keys (strings),
    every ``isinstance(entry, dict)`` check failed, and zero CO-NN ids
    surfaced. The normalizer routes the dict-of-lists shape through
    the canonical list-of-groups walk, recovering every id.

    Caveat: in current production, ``_plan_course_structure``'s
    ``synthesized`` dict is always built in list-of-groups shape
    (`pipeline_tools.py:5519-5522`), and the supplied-objectives
    loader (`_content_gen_helpers.load_objectives_json`) flattens the
    list-of-groups shape upstream. The dict-of-lists code path in
    ``_collect_lo_ids`` is a defensive branch — exercised when a
    future caller injects a dict-shaped payload directly (e.g. an
    in-memory smoke-test, or a future loader update that preserves
    the dict shape end-to-end). This test pins the closure's
    dict-handling contract so the defensive branch can't silently
    regress.
    """
    payload = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "T1."},
            {"id": "TO-02", "statement": "T2."},
        ],
        "chapter_objectives": {
            "1": [
                {"id": "CO-01", "statement": "C1."},
                {"id": "CO-02", "statement": "C2."},
            ],
            "2": [
                {"id": "CO-03", "statement": "C3."},
                {"id": "CO-04", "statement": "C4."},
            ],
        },
    }

    # Replica of ``_collect_lo_ids`` inside
    # ``MCP/tools/pipeline_tools.py::_plan_course_structure`` —
    # exercises the same normalizer-based walk.
    def _collect_lo_ids_replica(p):
        ids = []
        terminals = (
            p.get("terminal_objectives")
            or p.get("terminal_outcomes")
            or []
        )
        for entry in terminals:
            if isinstance(entry, dict) and entry.get("id"):
                ids.append(str(entry["id"]))
        chapters_raw = (
            p.get("chapter_objectives")
            or p.get("component_objectives")
            or []
        )
        for group in _normalize_chapter_objectives_to_groups(chapters_raw):
            for entry in group.get("objectives") or []:
                if isinstance(entry, dict) and entry.get("id"):
                    ids.append(str(entry["id"]))
        return ids

    ids = _collect_lo_ids_replica(payload)
    # Terminals first, then every CO-NN id (sorted-by-chapter-key
    # order: "1" group before "2" group).
    assert ids == [
        "TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04",
    ], f"dict-of-lists chapter_objectives dropped CO-NN ids: {ids!r}"
