"""Trap 3 — provenance-aware objectives reuse in _plan_course_structure.

Self-poisoning guard: after synthesis, the pipeline stamps
``project_config.json::objectives_path`` with THIS phase's own output.
On a re-run, that pipeline-stamped pointer must NOT be treated as a
``--reuse-objectives`` operator pin (which would silently re-normalize a
possibly-degenerate prior synthesis instead of re-synthesizing).

  * An OPERATOR pin (``objectives_path`` kwarg, or a config pin whose
    provenance is operator-supplied) IS honored (short-circuits synthesis).
  * A PIPELINE-stamped config pin (``objectives_source == "pipeline"``)
    re-synthesizes on a re-run.
  * A LEGACY config (no ``objectives_source`` field) whose pinned file is
    synthesis-minted re-synthesizes; a user-supplied-minted pinned file is
    reused.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


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

    project_id = "PROJ-DEMO_101-20260701000000"
    project_dir = exports / project_id
    project_dir.mkdir()
    for subdir in ("01_learning_objectives", "03_content_development"):
        (project_dir / subdir).mkdir()
    (project_dir / "project_config.json").write_text(
        json.dumps(
            {
                "project_id": project_id,
                "course_name": "DEMO_101",
                "duration_weeks": 4,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_dart_html(
        staging / "book.html",
        ["Photosynthesis Basics", "Light Reactions", "The Calvin Cycle",
         "Factors Affecting Photosynthesis"],
    )
    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "staging_dir": staging,
    }


def _write_dart_html(path: Path, headings: list) -> None:
    parts = ['<a class="skip-link" href="#main">Skip</a><main role="main">']
    for idx, h in enumerate(headings, start=1):
        parts.append(f'<section aria-labelledby="s{idx}"><h2 id="s{idx}">{h}</h2>')
        parts.append(
            f"<p>{h} is a foundational concept covered in this chapter of "
            f"the course. Understanding {h} requires students to carefully "
            f"examine its component parts and the relationships between "
            f"these parts in real-world educational contexts.</p>"
            f"<p>Advanced study of {h} builds on prior knowledge of related "
            f"topics and emphasizes deep comprehension over superficial "
            f"memorization across multiple dimensions.</p>"
        )
        parts.append("</section>")
    parts.append("</main>")
    path.write_text("<html><body>" + "".join(parts) + "</body></html>",
                    encoding="utf-8")


def _objectives_doc(mint_method: str) -> dict:
    return {
        "mint_method": mint_method,
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Explain the core supplied concepts."}
        ],
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Describe a supplied subtopic clearly.",
             "parent_terminal": "TO-01"}
        ],
    }


async def _call(**kwargs):
    registry = _build_tool_registry()
    fn = registry["plan_course_structure"]
    return json.loads(await fn(**kwargs))


def _config(project_dir: Path) -> dict:
    return json.loads((project_dir / "project_config.json").read_text())


def _set_config(project_dir: Path, **updates) -> None:
    cfg = _config(project_dir)
    cfg.update(updates)
    (project_dir / "project_config.json").write_text(json.dumps(cfg, indent=2))


# --------------------------------------------------------------------------
# 1. Operator pin honored (objectives_path kwarg).
# --------------------------------------------------------------------------
def test_operator_pin_kwarg_is_honored(planner_fixture, tmp_path):
    fx = planner_fixture
    op_file = tmp_path / "operator_objectives.json"
    op_file.write_text(json.dumps(_objectives_doc("hand_authored")))
    result = asyncio.run(_call(
        project_id=fx["project_id"],
        staging_dir=str(fx["staging_dir"]),
        objectives_path=str(op_file),
    ))
    assert result["success"]
    assert result["mint_method"] == "user_supplied_objectives_json"


# --------------------------------------------------------------------------
# 2. Pipeline-stamped path re-synthesizes on re-run.
# --------------------------------------------------------------------------
def test_pipeline_stamped_pin_resynthesizes(planner_fixture):
    fx = planner_fixture
    # Run 1: fresh synthesis stamps objectives_source="pipeline".
    r1 = asyncio.run(_call(
        project_id=fx["project_id"], staging_dir=str(fx["staging_dir"]),
    ))
    assert r1["success"]
    assert r1["mint_method"] == "synthesize_objectives_from_topics"
    cfg = _config(fx["project_dir"])
    assert cfg.get("objectives_source") == "pipeline"
    assert cfg.get("objectives_path")

    # Run 2: the config now pins the phase's own prior output — must NOT
    # be reused; re-synthesize instead.
    r2 = asyncio.run(_call(
        project_id=fx["project_id"], staging_dir=str(fx["staging_dir"]),
    ))
    assert r2["success"]
    assert r2["mint_method"] == "synthesize_objectives_from_topics", (
        "pipeline-stamped self-output must re-synthesize, not reuse"
    )


# --------------------------------------------------------------------------
# 3. Legacy config (no provenance) + synthesis-minted file re-synthesizes.
# --------------------------------------------------------------------------
def test_legacy_config_synthesis_minted_pin_resynthesizes(planner_fixture):
    fx = planner_fixture
    pin = fx["project_dir"] / "01_learning_objectives" / "synthesized_objectives.json"
    pin.write_text(json.dumps(_objectives_doc("synthesize_objectives_from_topics")))
    # Legacy config: objectives_path set, NO objectives_source field.
    _set_config(fx["project_dir"], objectives_path=str(pin))
    assert "objectives_source" not in _config(fx["project_dir"])

    result = asyncio.run(_call(
        project_id=fx["project_id"], staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    assert result["mint_method"] == "synthesize_objectives_from_topics"
    # And provenance is now stamped as pipeline.
    assert _config(fx["project_dir"]).get("objectives_source") == "pipeline"


# --------------------------------------------------------------------------
# 4. Legacy config (no provenance) + user-supplied-minted file IS reused.
# --------------------------------------------------------------------------
def test_legacy_config_user_supplied_minted_pin_is_reused(planner_fixture):
    fx = planner_fixture
    pin = fx["project_dir"] / "01_learning_objectives" / "prior.json"
    pin.write_text(json.dumps(_objectives_doc("user_supplied_objectives_json")))
    _set_config(fx["project_dir"], objectives_path=str(pin))

    result = asyncio.run(_call(
        project_id=fx["project_id"], staging_dir=str(fx["staging_dir"]),
    ))
    assert result["success"]
    assert result["mint_method"] == "user_supplied_objectives_json", (
        "a legacy operator-minted pin must still be honored"
    )
