"""Assessment-tail fixes (alg-glm-02 production run, 2026-07-19) — regression net.

Three bug classes from the overnight run, one test section each:

1. **Gate-input mis-routing** — ``_locate`` iterated phase_outputs
   PHASE-major, so ``semantik_conversion`` (first phase, emits
   ``output_path`` = the accessible HTML) shadowed
   ``trainforge_assessment``'s real ``assessments_path``; the
   ``assessment_quality`` / ``assessment_objective_alignment`` gates then
   json-parsed an HTML file and crashed with
   "Expecting value: line 1 column 1". Now KEY-major.

2. **run_assessment_synthesis kwargs fallback** — the phase received
   ``project_id=None`` (lost across a course_planning fail-open + resume)
   and error-returned in <100ms WITHOUT creating ``06_assessments/``; the
   ``{"error": ...}`` envelope carried no ``success`` key so the executor
   marked the task COMPLETE (silent). Now: newest ``PROJ-<course>-*``
   export fallback (the courseforge-rewrite subcommand precedent) + loud
   logging + ``success: False`` fail-loud envelopes.

3. **generate_assessments (trainforge_assessment) silence** — error
   returns now carry ``success: False`` + ``error_code`` so the executor
   FAILS the task instead of treating the envelope as success.

No course slugs / device paths hardcoded; all fixtures under tmp_path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.hardening.gate_input_routing import (  # noqa: E402
    _build_assessment_objective_alignment,
    _build_assessment_quality,
    _locate,
)
from MCP.tools import pipeline_tools as pt  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# 1. _locate is KEY-major (specific keys beat earlier phases' generic keys).
# ---------------------------------------------------------------------------


def test_locate_is_key_major():
    phase_outputs = {
        # First phase in insertion order carries only the GENERIC fallback
        # key — must NOT shadow the specific key in a later phase.
        "semantik_conversion": {"output_path": "/x/semantik.html"},
        "trainforge_assessment": {
            "assessments_path": "/y/assessments.json",
            "output_path": "/y/assessments.json",
        },
    }
    assert (
        _locate(phase_outputs, "assessments_path", "output_path")
        == "/y/assessments.json"
    )


def test_locate_falls_back_to_lower_priority_key():
    phase_outputs = {"only": {"output_path": "/z/thing.json"}}
    assert (
        _locate(phase_outputs, "assessments_path", "output_path")
        == "/z/thing.json"
    )
    assert _locate(phase_outputs, "missing_key") is None


def _production_shaped_phase_outputs(tmp_path: Path):
    """Mirror the alg-glm-02 run's phase_outputs shape: semantik first
    (output_path = HTML), trainforge_assessment later (real JSON)."""
    html = tmp_path / "book_accessible.html"
    html.write_text("<html><body>not json</body></html>", encoding="utf-8")
    assessments = tmp_path / "trainforge" / "assessments.json"
    assessments.parent.mkdir(parents=True)
    assessments.write_text(
        json.dumps({"assessment_id": "ASM-T", "questions": []}),
        encoding="utf-8",
    )
    chunks = assessments.parent / "imscc_chunks" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    chunks.write_text(
        json.dumps({"id": "chunk_0001", "text": "t"}) + "\n", encoding="utf-8"
    )
    return {
        "semantik_conversion": {"output_path": str(html)},
        "packaging": {"package_path": str(tmp_path / "pkg.imscc")},
        "trainforge_assessment": {
            "output_path": str(assessments),
            "assessments_path": str(assessments),
        },
    }, assessments


def test_assessment_quality_builder_prefers_real_assessments(tmp_path):
    phase_outputs, assessments = _production_shaped_phase_outputs(tmp_path)
    inputs, missing = _build_assessment_quality(phase_outputs, {})
    assert not missing
    assert inputs["assessment_path"] == str(assessments)


def test_assessment_objective_alignment_builder_prefers_real_assessments(
    tmp_path,
):
    phase_outputs, assessments = _production_shaped_phase_outputs(tmp_path)
    inputs, missing = _build_assessment_objective_alignment(phase_outputs, {})
    assert not missing
    assert inputs["assessments_path"] == str(assessments)


# ---------------------------------------------------------------------------
# 2. run_assessment_synthesis — defensive project_id + fail-loud envelope.
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry():
    return _build_tool_registry()


async def test_assessment_synthesis_resolves_newest_export(
    registry, tmp_path, monkeypatch
):
    course = "TAILFIXT"
    exports_root = tmp_path / "exports"
    old_proj = exports_root / f"PROJ-{course}-20260101000000"
    new_proj = exports_root / f"PROJ-{course}-20260201000000"
    for p in (old_proj, new_proj):
        p.mkdir(parents=True)
    # Pin mtimes so "newest" is deterministic.
    now = time.time()
    os.utime(old_proj, (now - 1000, now - 1000))
    os.utime(new_proj, (now, now))
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    tool = registry["run_assessment_synthesis"]
    # NO project_id kwarg — the production failure shape.
    result = json.loads(await tool(course_name=course))

    # No objectives in the fixture project → clean empty emit, but the
    # point is: the NEWEST export was resolved, not an error return.
    assert result.get("success") is True
    assert result.get("reason") == "no_objectives"
    assert str(new_proj) in result["assessments_dir"]
    assert (new_proj / "06_assessments" / "manifest.json").exists()


async def test_assessment_synthesis_fails_loud_without_any_project(
    registry, tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    tool = registry["run_assessment_synthesis"]
    result = json.loads(await tool(course_name="NOSUCHCOURSE"))

    # The executor treats success=False as a permanent LOUD task failure;
    # a bare {"error": ...} envelope would be marked COMPLETE (the silent
    # production no-op).
    assert result.get("success") is False
    assert result.get("error_code") == "ASSESSMENT_SYNTHESIS_NO_PROJECT_ID"


async def test_assessment_synthesis_missing_project_dir_fails_loud(
    registry, tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    tool = registry["run_assessment_synthesis"]
    result = json.loads(await tool(project_id="PROJ-GHOST-1"))

    assert result.get("success") is False
    assert (
        result.get("error_code") == "ASSESSMENT_SYNTHESIS_PROJECT_DIR_MISSING"
    )


# ---------------------------------------------------------------------------
# 3. generate_assessments (trainforge_assessment) — fail-loud envelopes.
# ---------------------------------------------------------------------------


async def test_generate_assessments_missing_imscc_fails_loud(
    registry, tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    proj = exports_root / "PROJ-TAILFIXT-1"
    proj.mkdir(parents=True)
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    tool = registry["generate_assessments"]
    result = json.loads(
        await tool(course_id="TAILFIXT", project_id="PROJ-TAILFIXT-1")
    )

    assert result.get("success") is False
    assert result.get("error_code") == "TRAINFORGE_IMSCC_MISSING"


async def test_generate_assessments_no_workspace_fails_loud(
    registry, tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)

    tool = registry["generate_assessments"]
    result = json.loads(await tool(course_id="NOSUCHCOURSE"))

    assert result.get("success") is False
    assert result.get("error_code") == "TRAINFORGE_NO_PROJECT_WORKSPACE"
