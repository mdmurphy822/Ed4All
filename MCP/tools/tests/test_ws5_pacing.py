"""WS5 §6 — TO-based pacing rescale (§3.2) + override guards (§4).

The pacing formula lives in the single-sourced helper
``MCP.tools.pipeline_tools._ws5_resolve_objective_weeks`` (consumed inline by
``_plan_course_structure``). These tests exercise the helper directly for the
arithmetic (weeks track TO count; floor 8) + the two override guards, and run
one end-to-end ``plan_course_structure`` slice through the tool registry to
confirm the ``user_supplied_objectives_json`` guard preserves operator pacing
on disk (no rescale) AND that the §2.2 cap-lift placed all COs.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _ws5_resolve_objective_weeks,
)

# Shared single-sourced slicer (the same one the emitter + validator consume).
_CF_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_CF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CF_SCRIPTS))
from generate_course import _slice_cos_for_week  # noqa: E402


def _all_placed(n_cos, weeks):
    cos = [{"id": f"CO-{i:02d}"} for i in range(1, n_cos + 1)]
    placed = set()
    for w in range(1, weeks + 1):
        placed |= {o["id"] for o in _slice_cos_for_week(cos, weeks, w)}
    return placed, {o["id"] for o in cos}


# --------------------------------------------------------------------------- #
# §3.2 — weeks track the WS1 TO count (NOT the 31-week CO-count balloon).
# --------------------------------------------------------------------------- #

class TestWeeksTrackToCount:
    def test_weeks_track_to_count(self, monkeypatch):
        """terminal=10, chapter=62 → weeks 10 (one week per TO), not 31."""
        monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
        weeks = _ws5_resolve_objective_weeks(
            num_tos=10, num_cos=62, duration_weeks=8,
            duration_explicit=False, mint_method="synthesize_objectives_from_topics",
        )
        assert weeks == 10, "weeks must equal the TO count, not the CO-count balloon"
        # And all 62 COs still place across 10 weeks (§2.2 cap-lift).
        placed, all_ids = _all_placed(62, weeks)
        assert all_ids <= placed

    def test_floor_at_eight(self, monkeypatch):
        """terminal=3 → max(8,3)=8; all 62 COs still placed."""
        monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
        weeks = _ws5_resolve_objective_weeks(
            num_tos=3, num_cos=62, duration_weeks=8,
            duration_explicit=False, mint_method="synthesize_objectives_from_topics",
        )
        assert weeks == 8
        placed, all_ids = _all_placed(62, weeks)
        assert all_ids <= placed, "floor-8 weeks must still place every CO"

    def test_weeks_above_floor_use_to_count(self, monkeypatch):
        monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
        assert _ws5_resolve_objective_weeks(
            num_tos=12, num_cos=141, duration_weeks=8,
            duration_explicit=False, mint_method="synthesize_objectives_from_topics",
        ) == 12

    def test_no_tos_falls_back_to_legacy_co_formula(self, monkeypatch):
        """No TOs → legacy max(8, ceil(num_cos / WAVE18_COS_PER_WEEK))."""
        monkeypatch.setenv("WAVE18_COS_PER_WEEK", "2")
        assert _ws5_resolve_objective_weeks(
            num_tos=0, num_cos=62, duration_weeks=8,
            duration_explicit=False, mint_method="synthesize_objectives_from_topics",
        ) == 31  # max(8, ceil(62/2))


# --------------------------------------------------------------------------- #
# §4 — override guards (verbatim).
# --------------------------------------------------------------------------- #

class TestOverrideGuards:
    def test_weeks_explicit_override_preserved(self, monkeypatch):
        """--weeks 12 (duration_explicit) → no rescale even with 10 TOs."""
        monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
        weeks = _ws5_resolve_objective_weeks(
            num_tos=10, num_cos=62, duration_weeks=12,
            duration_explicit=True, mint_method="synthesize_objectives_from_topics",
        )
        assert weeks == 12, "explicit operator weeks must win"
        # The §2.2 cap-lift is UNCONDITIONAL — all COs place at the explicit 12.
        placed, all_ids = _all_placed(62, 12)
        assert all_ids <= placed

    def test_user_supplied_objectives_no_rescale(self, monkeypatch):
        """mint_method == user_supplied_objectives_json → no rescale."""
        monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
        weeks = _ws5_resolve_objective_weeks(
            num_tos=10, num_cos=62, duration_weeks=6,
            duration_explicit=False, mint_method="user_supplied_objectives_json",
        )
        assert weeks == 6, "hand-curated pacing must be preserved verbatim"


# --------------------------------------------------------------------------- #
# Unconditional cap-lift: even a short explicit --weeks 8 places all 62 COs.
# --------------------------------------------------------------------------- #

class TestUnconditionalCapLift:
    def test_explicit_short_weeks_still_places_all_cos(self):
        placed, all_ids = _all_placed(62, 8)
        assert all_ids <= placed, "explicit --weeks 8 must still place all 62 COs"


# --------------------------------------------------------------------------- #
# End-to-end: user-supplied objectives persist with NO rescale + all COs in the
# per-week groups (§2.4(A) parity persistence).
# --------------------------------------------------------------------------- #

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
    project_id = "PROJ-WS5_101-20260616000000"
    project_dir = exports / project_id
    project_dir.mkdir()
    for subdir in ("00_template_analysis", "01_learning_objectives",
                   "02_course_planning", "03_content_development",
                   "04_quality_validation", "05_final_package"):
        (project_dir / subdir).mkdir()
    (project_dir / "project_config.json").write_text(
        json.dumps({
            "project_id": project_id,
            "course_name": "WS5_101",
            "duration_weeks": 6,
            "credit_hours": 3,
        }, indent=2),
        encoding="utf-8",
    )
    return {"project_id": project_id, "project_dir": project_dir}


async def _call(**kwargs):
    registry = _build_tool_registry()
    fn = registry["plan_course_structure"]
    return json.loads(await fn(**kwargs))


def test_user_supplied_objectives_persist_no_rescale(planner_fixture, monkeypatch):
    """End-to-end: a supplied objectives JSON with 6 weeks declared persists
    duration_weeks=6 unchanged (no TO-rescale) and the per-week groups hold
    every supplied CO (§2.4(A) parity persistence)."""
    monkeypatch.delenv("WAVE18_COS_PER_WEEK", raising=False)
    fx = planner_fixture
    supplied = fx["project_dir"] / "supplied_objectives.json"
    supplied.write_text(json.dumps({
        "duration_weeks": 6,
        "terminal_objectives": [
            {"id": f"TO-{i:02d}", "statement": f"Terminal outcome {i}.",
             "bloom_level": "analyze"}
            for i in range(1, 4)
        ],
        "chapter_objectives": [{
            "chapter": "Week 1",
            "objectives": [
                {"id": f"CO-{i:02d}", "statement": f"Chapter outcome {i}.",
                 "bloom_level": "remember"}
                for i in range(1, 13)
            ],
        }],
    }), encoding="utf-8")

    result = asyncio.run(_call(
        project_id=fx["project_id"],
        objectives_path=str(supplied),
        duration_weeks=6,
        duration_weeks_explicit=False,
    ))
    assert result["success"]
    assert result["mint_method"] == "user_supplied_objectives_json"

    doc = json.loads(
        Path(result["synthesized_objectives_path"]).read_text(encoding="utf-8")
    )
    # No rescale: hand-curated pacing preserved.
    assert doc["duration_weeks"] == 6
    # §2.4(A): every supplied CO appears across the per-week groups (no drop).
    placed = set()
    for grp in doc["chapter_objectives"]:
        placed |= {o["id"] for o in grp["objectives"]}
    supplied_co_ids = {f"CO-{i:02d}" for i in range(1, 13)}
    assert supplied_co_ids <= placed, (
        f"cap-lift dropped supplied COs: {sorted(supplied_co_ids - placed)}"
    )
