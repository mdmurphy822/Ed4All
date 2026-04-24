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
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

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


def test_objective_ids_are_comma_separated(planner_fixture):
    """Returned objective_ids is a comma-joined string of real LO IDs."""
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
    ids = [x.strip() for x in result["objective_ids"].split(",") if x.strip()]
    assert ids, "objective_ids must not be empty when topics are present"
    # None of the IDs should be the pre-Wave-24 {COURSE}_OBJ_N phantom shape.
    for _id in ids:
        assert "_OBJ_" not in _id, (
            f"phantom id {_id!r} resurfaced — scope 2 regression"
        )
        assert _LO_ID_RE.match(_id), (
            f"id {_id!r} is not canonical"
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
