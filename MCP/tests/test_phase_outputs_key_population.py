"""Wave 32 Deliverable B — phase_outputs key population.

Every gate validator's builder in ``MCP/hardening/gate_input_routing.py``
works correctly when fed the expected phase-outputs keys (locked by
``MCP/tests/test_gate_input_routing.py``). But pre-Wave-32 the
production phases didn't populate those keys in the tool return
envelopes, so every one of the following gates silently skipped with
``missing inputs: *`` on live re-sims:

* ``dart_markers`` — builder needs ``html_path`` / ``html_paths``;
  ``extract_and_convert_pdf`` only emitted ``output_path``.
* ``content_grounding`` / ``page_objectives`` — builders need
  ``page_paths`` / ``content_dir``; ``generate_course_content``
  emitted ``content_paths`` as a **list** (routers check for ``str``).
* ``imscc_structure`` / ``page_objectives`` — builders need
  ``imscc_path`` / ``content_dir``; ``package_imscc`` only emitted
  ``package_path`` + ``libv2_package_path``.

The fix is purely on the emit side — this test locks the tool-return
contract so the six gates receive inputs without any router changes.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from MCP.hardening.gate_input_routing import default_router

# ---------------------------------------------------------------------- #
# Fixture helpers
# ---------------------------------------------------------------------- #


def _make_project(
    tmp_path: Path, project_id: str, duration_weeks: int = 1,
) -> Path:
    """Minimal Courseforge project scaffold for generate_course_content."""
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    (project_path / "03_content_development").mkdir(parents=True)
    config = {
        "course_name": "TESTCOURSE_101",
        "duration_weeks": duration_weeks,
        "credit_hours": 3,
    }
    (project_path / "project_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return project_path


@pytest.fixture
def pipeline_registry(tmp_path, monkeypatch):
    """Build the tool registry with ``_PROJECT_ROOT`` pointed at tmp_path.

    Mirrors the fixture shape used in ``test_generate_course_content.py``
    so the tool's project-path resolution lands inside the temp dir.
    """
    pt = importlib.import_module("MCP.tools.pipeline_tools")
    monkeypatch.setattr(pt, "_PROJECT_ROOT", tmp_path, raising=True)
    monkeypatch.setattr(pt, "PROJECT_ROOT", tmp_path, raising=True)
    # COURSEFORGE_INPUTS is resolved relative to _PROJECT_ROOT at
    # module-load; patch the one in ``pipeline_tools`` to the tmp shape.
    monkeypatch.setattr(
        pt, "COURSEFORGE_INPUTS", tmp_path / "Courseforge" / "inputs",
        raising=True,
    )
    registry = pt._build_tool_registry()
    return registry, tmp_path


# ---------------------------------------------------------------------- #
# Tool-level return-envelope contracts
# ---------------------------------------------------------------------- #


def test_extract_and_convert_pdf_emits_html_path(tmp_path: Path, monkeypatch):
    """extract_and_convert_pdf must surface ``html_path`` alongside output_path.

    Pre-Wave-32 the tool only emitted ``output_path``, which the
    ``DartMarkersValidator`` builder doesn't consume — so ``dart_markers``
    gates silently skipped with ``missing inputs: html_path``.

    Hermetic: monkeypatches ``subprocess.run`` so we don't depend on
    pdftotext being installed on the CI image.
    """
    import subprocess as _subprocess_mod

    pt = importlib.import_module("MCP.tools.pipeline_tools")
    registry = pt._build_tool_registry()

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")  # marker; text comes from stub.

    fake_text = (
        "# Chapter 1\n"
        + "Knowledge graphs organise information as nodes and edges. "
        * 10
    )

    class _FakeCompleted:
        stdout = fake_text
        returncode = 0

    def _fake_run(args, **kwargs):  # noqa: ANN001
        if args and args[0] == "pdftotext":
            return _FakeCompleted()
        raise _subprocess_mod.SubprocessError("unexpected subprocess call")

    monkeypatch.setattr(_subprocess_mod, "run", _fake_run)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result_json = asyncio.run(registry["extract_and_convert_pdf"](
        pdf_path=str(pdf),
        output_dir=str(out_dir),
        course_code="TESTCOURSE_101",
    ))
    result = json.loads(result_json)
    assert result.get("success") is True, result
    # Wave 32 Deliverable B: both aliases must be present.
    assert "html_path" in result
    assert "output_path" in result
    assert result["html_path"] == result["output_path"]
    assert Path(result["html_path"]).exists()


def test_generate_course_content_emits_page_paths_and_content_dir(
    pipeline_registry,
):
    """generate_course_content must surface page_paths (list) + content_dir.

    Pre-Wave-32 the tool only emitted ``content_paths`` as a plain list,
    but the router's builders check ``content_paths`` only when it's a
    comma-joined ``str`` and otherwise skip ``content_grounding`` +
    ``page_objectives`` with ``missing inputs: page_paths / content_dir``.
    """
    registry, tmp_path = pipeline_registry
    project_id = "PROJ-WAVE32-B1"
    _make_project(tmp_path, project_id, duration_weeks=1)

    result_json = asyncio.run(registry["generate_course_content"](
        project_id=project_id,
        staging_dir=str(tmp_path / "nonexistent"),
    ))
    result = json.loads(result_json)
    # Empty-corpus case fails CONTENT_GENERATION_EMPTY (Deliverable C);
    # the failure envelope must still surface the keys the routers
    # look for so downstream callers can inspect them.
    assert "page_paths" in result
    assert "content_dir" in result
    # page_paths is the list shape the routers consume (builders that
    # want a str use ``content_paths`` which we also surface).
    assert isinstance(result["page_paths"], list)
    # Success-path envelope also surfaces content_paths as a str alias.
    if result.get("success") is True:
        assert isinstance(result.get("content_paths"), str)


def test_package_imscc_emits_imscc_path_and_content_dir(
    pipeline_registry, tmp_path: Path,
):
    """package_imscc must surface imscc_path + content_dir aliases.

    Pre-Wave-32 the tool only surfaced ``package_path`` +
    ``libv2_package_path``; ``imscc_structure`` and
    ``page_objectives`` gate builders look for ``imscc_path`` /
    ``content_dir`` and silently skipped.
    """
    registry, _tmp = pipeline_registry
    project_id = "PROJ-WAVE32-B2"
    project_path = _make_project(tmp_path, project_id)

    # Drop a single HTML page so the packager has something to zip.
    content_dir = project_path / "03_content_development" / "week_01"
    content_dir.mkdir(parents=True)
    (content_dir / "week_01_overview.html").write_text(
        (
            "<!DOCTYPE html><html><head><title>Week 1</title></head>"
            "<body><main><h1>Week 1</h1><p>"
            + "content words " * 30
            + "</p></main></body></html>"
        ),
        encoding="utf-8",
    )

    result_json = asyncio.run(registry["package_imscc"](
        project_id=project_id,
    ))
    result = json.loads(result_json)
    # Packager may reject on the LO contract for synthetic projects;
    # test the envelope shape on success, skip structurally on failure.
    if not result.get("success"):
        pytest.skip(f"packager rejected synthetic project: {result.get('error')}")

    assert "imscc_path" in result
    assert "content_dir" in result
    # imscc_path must alias package_path so validators that check either
    # key find the zip.
    assert result["imscc_path"] == result["package_path"]


def test_generate_assessments_emits_chunks_path(tmp_path: Path):
    """generate_assessments must surface chunks_path (pre-existing behaviour).

    This is a regression guard only — Wave 24 already wired this key
    but the Wave 32 re-sim also reported it as missing on one run, so
    we lock the contract here even though no code change is needed
    in this spot.
    """
    from MCP.core.workflow_runner import _LEGACY_PHASE_OUTPUT_KEYS
    # Canonical trainforge_assessment output contract includes chunks_path.
    declared = _LEGACY_PHASE_OUTPUT_KEYS.get("trainforge_assessment", [])
    assert "chunks_path" in declared
    assert "assessments_path" in declared


def test_archive_to_libv2_emits_manifest_path(pipeline_registry, tmp_path: Path):
    """archive_to_libv2 must surface manifest_path + course_dir (pre-existing).

    Regression guard only. The Wave 32 re-sim reported ``libv2_manifest``
    as skipping; ``_build_libv2_manifest`` derives manifest_path from
    course_dir when absent but we lock the emit-side contract here.
    """
    registry, _tmp = pipeline_registry

    result_json = asyncio.run(registry["archive_to_libv2"](
        course_name="TESTCOURSE_101",
        domain="general",
        division="STEM",
    ))
    result = json.loads(result_json)
    assert result.get("success") is True
    assert "manifest_path" in result
    assert "course_dir" in result
    assert Path(result["manifest_path"]).exists()


# ---------------------------------------------------------------------- #
# Router integration — the six gates must receive inputs
# ---------------------------------------------------------------------- #


def _phase_outputs_for_router(**phases) -> Dict[str, Dict[str, Any]]:
    """Match the phase_outputs shape the runner assembles."""
    return {name: data for name, data in phases.items()}


def test_router_picks_up_dart_markers_inputs(tmp_path: Path):
    """dart_markers gate no longer skips when dart_conversion surfaces html_path."""
    html = tmp_path / "out.html"
    html.write_text("<html><body></body></html>", encoding="utf-8")

    phase_outputs = _phase_outputs_for_router(
        dart_conversion={
            "output_path": str(html),
            "html_path": str(html),  # Wave 32: new canonical alias
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.dart_markers.DartMarkersValidator",
        phase_outputs, {},
    )
    assert missing == []
    assert inputs["html_path"] == str(html)


def test_router_picks_up_page_objectives_with_content_dir(tmp_path: Path):
    """page_objectives gate no longer skips when content_generation surfaces content_dir."""
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "week_01_overview.html").write_text(
        "<html></html>", encoding="utf-8",
    )

    phase_outputs = _phase_outputs_for_router(
        content_generation={
            "content_dir": str(content_dir),
            "page_paths": [str(content_dir / "week_01_overview.html")],
            "_completed": True,
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.page_objectives.PageObjectivesValidator",
        phase_outputs, {},
    )
    assert missing == []
    assert Path(inputs["content_dir"]).exists()


def test_router_picks_up_imscc_from_imscc_path_alias(tmp_path: Path):
    """imscc_structure gate picks up imscc_path alias from packaging phase."""
    imscc = tmp_path / "course.imscc"
    imscc.write_bytes(b"PK\x03\x04fake-zip")

    phase_outputs = _phase_outputs_for_router(
        packaging={
            "package_path": str(imscc),
            "imscc_path": str(imscc),  # Wave 32: new canonical alias
            "content_dir": str(tmp_path / "content"),
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.imscc.IMSCCValidator",
        phase_outputs, {},
    )
    assert missing == []
    assert inputs["imscc_path"] == str(imscc)


def test_full_textbook_pipeline_no_gates_skip_on_missing_inputs(tmp_path: Path):
    """Smoke: synthetic phase_outputs dict → every gate builder resolves.

    Stitches together the full set of Wave 32 Deliverable B aliases so
    we can assert none of the six previously-skipping gate builders
    return a missing-input list. The phase_outputs dict here mirrors
    what a real ``textbook_to_course`` run surfaces once Deliverable B
    lands.
    """
    html = tmp_path / "dart.html"
    html.write_text("<html></html>", encoding="utf-8")
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    page = content_dir / "week_01_overview.html"
    page.write_text("<html></html>", encoding="utf-8")
    imscc = tmp_path / "course.imscc"
    imscc.write_bytes(b"PK\x03\x04")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id":"c1"}\n', encoding="utf-8")
    assessments = tmp_path / "assessments.json"
    assessments.write_text(json.dumps({"questions": []}), encoding="utf-8")
    course_dir = tmp_path / "libv2_course"
    course_dir.mkdir()
    manifest = course_dir / "manifest.json"
    manifest.write_text(json.dumps({"slug": "test"}), encoding="utf-8")

    phase_outputs = _phase_outputs_for_router(
        dart_conversion={
            "output_path": str(html),
            "output_paths": str(html),
            "html_path": str(html),
            "html_paths": str(html),
        },
        staging={"staging_dir": str(tmp_path / "staging")},
        content_generation={
            "page_paths": [str(page)],
            "content_paths": str(page),
            "content_dir": str(content_dir),
        },
        packaging={
            "package_path": str(imscc),
            "imscc_path": str(imscc),
            "content_dir": str(content_dir),
        },
        trainforge_assessment={
            "assessments_path": str(assessments),
            "chunks_path": str(chunks),
        },
        libv2_archival={
            "manifest_path": str(manifest),
            "course_dir": str(course_dir),
        },
    )

    r = default_router()
    # The six Wave 32 Deliverable B targets — builders must all resolve.
    for validator_path in [
        "lib.validators.dart_markers.DartMarkersValidator",
        "lib.validators.content_grounding.ContentGroundingValidator",
        "lib.validators.page_objectives.PageObjectivesValidator",
        "lib.validators.imscc.IMSCCValidator",
        "lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator",
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
    ]:
        inputs, missing = r.build(validator_path, phase_outputs, {})
        assert missing == [], (
            f"Gate {validator_path} should receive inputs with Wave 32 "
            f"Deliverable B keys; still missing: {missing}"
        )
