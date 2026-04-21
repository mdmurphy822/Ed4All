"""Wave 30 Gap 3 — synthesize_training phase wiring.

Pre-Wave-30, ``Trainforge/synthesize_training.py`` had zero callers
inside any end-to-end pipeline run. Every LibV2 course was missing
``training_specs/instruction_pairs.jsonl`` +
``training_specs/preference_pairs.jsonl``, so ``ed4all export-training
... --format dpo`` surfaced decision-capture records instead of real
Q&A pairs. Wave 30 Gap 3 wires the synthesizer in:

* ``synthesize_training`` tool (both ``@mcp.tool()`` surface + registry
  variant for pipeline dispatch).
* New ``training_synthesis`` phase in ``textbook_to_course`` that runs
  after ``trainforge_assessment`` and feeds ``libv2_archival``.
* ``libv2_archival`` now copies the two new JSONL files alongside
  ``assessments.json``.
* New ``training-synthesizer`` agent in ``AGENT_TOOL_MAPPING``.

These tests exercise the wiring contract against synthetic fixtures —
no real IMSCC processing, no LLM traffic.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "Trainforge"
    / "tests"
    / "fixtures"
    / "mini_course_training"
)


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the read-only Trainforge training fixture so the registry
    tool can write into it."""
    dst = tmp_path / "mini_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
    ):
        if stale.exists():
            stale.unlink()
    return dst


@pytest.mark.asyncio
async def test_registry_has_training_synthesizer_wired():
    """Wave 30 Gap 3: the ``training-synthesizer`` agent must route to
    the ``synthesize_training`` tool, which must exist in the registry.
    Regression guard for the executor-side wiring."""
    from MCP.core.executor import AGENT_TOOL_MAPPING

    assert "training-synthesizer" in AGENT_TOOL_MAPPING, (
        "Wave 30 Gap 3 agent mapping regressed"
    )
    assert AGENT_TOOL_MAPPING["training-synthesizer"] == "synthesize_training"

    registry = _build_tool_registry()
    assert "synthesize_training" in registry, (
        "synthesize_training must be registered for pipeline dispatch"
    )


@pytest.mark.asyncio
async def test_synthesize_training_produces_jsonl_pairs(tmp_path):
    """Call the registry variant with real synthetic chunks and assert
    both JSONL artifacts land on disk with non-zero pair counts."""
    corpus_dir = _copy_fixture(tmp_path)
    registry = _build_tool_registry()
    tool = registry["synthesize_training"]

    result_raw = await tool(
        corpus_dir=str(corpus_dir),
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    result = json.loads(result_raw)
    assert result.get("success") is True, result
    assert result.get("skipped", False) is False

    instr_path = Path(result["instruction_pairs_path"])
    pref_path = Path(result["preference_pairs_path"])
    assert instr_path.exists()
    assert pref_path.exists()

    instr_lines = [
        json.loads(l) for l in instr_path.read_text().splitlines() if l.strip()
    ]
    pref_lines = [
        json.loads(l) for l in pref_path.read_text().splitlines() if l.strip()
    ]
    # Fixture has 3 eligible chunks — exact count enforced by
    # ``test_training_synthesis.py``. Here we just need non-zero.
    assert len(instr_lines) > 0
    assert len(pref_lines) > 0
    assert result["instruction_pairs_count"] == len(instr_lines)
    assert result["preference_pairs_count"] == len(pref_lines)


@pytest.mark.asyncio
async def test_synthesize_training_missing_chunks_skips_gracefully(tmp_path):
    """No ``corpus/chunks.jsonl`` → skipped=true, no crash. This is
    the no-LLM-available / no-corpus safe path the audit flagged."""
    registry = _build_tool_registry()
    tool = registry["synthesize_training"]

    empty_dir = tmp_path / "empty_corpus"
    empty_dir.mkdir()

    result_raw = await tool(
        corpus_dir=str(empty_dir),
        course_code="EMPTY_001",
    )
    result = json.loads(result_raw)
    assert result.get("success") is True
    assert result.get("skipped") is True
    assert result.get("reason") == "chunks_missing"


@pytest.mark.asyncio
async def test_synthesize_training_resolves_corpus_from_assessments_path(
    tmp_path,
):
    """The registry variant must accept ``assessments_path`` and derive
    ``corpus_dir`` from it so the workflow's phase_outputs routing
    (``trainforge_assessment.assessments_path`` → this phase) works
    without an explicit corpus_dir kwarg."""
    corpus_dir = _copy_fixture(tmp_path)
    # Simulate assessments.json living at the corpus root.
    fake_assessments = corpus_dir / "assessments.json"
    fake_assessments.write_text(json.dumps({"questions": []}))

    registry = _build_tool_registry()
    tool = registry["synthesize_training"]

    result_raw = await tool(
        assessments_path=str(fake_assessments),
        course_name="MINI_TRAINING_101",
        provider="mock",
    )
    result = json.loads(result_raw)
    assert result.get("success") is True, result
    assert result.get("skipped", False) is False
    assert result["instruction_pairs_count"] > 0


@pytest.mark.asyncio
async def test_libv2_archival_copies_training_specs(tmp_path):
    """LibV2 archival must copy ``instruction_pairs.jsonl`` +
    ``preference_pairs.jsonl`` alongside ``assessments.json`` when the
    training_synthesis phase has populated them."""
    # Set up a fake trainforge dir with all artifacts.
    trainforge_dir = tmp_path / "trainforge"
    (trainforge_dir / "corpus").mkdir(parents=True)
    (trainforge_dir / "graph").mkdir(parents=True)
    (trainforge_dir / "training_specs").mkdir(parents=True)
    (trainforge_dir / "quality").mkdir(parents=True)

    (trainforge_dir / "corpus" / "chunks.jsonl").write_text(
        '{"id":"c1","text":"ex"}\n'
    )
    (trainforge_dir / "training_specs" / "assessments.json").write_text(
        json.dumps({"questions": []})
    )
    # Wave 30 Gap 3 new artifacts.
    (trainforge_dir / "training_specs" / "instruction_pairs.jsonl").write_text(
        '{"chunk_id":"c1","prompt":"q","completion":"a"}\n'
    )
    (trainforge_dir / "training_specs" / "preference_pairs.jsonl").write_text(
        '{"chunk_id":"c1","chosen":"a","rejected":"b"}\n'
    )
    (trainforge_dir / "quality" / "quality_report.json").write_text("{}")

    # Point archive_to_libv2 at the trainforge dir explicitly via
    # project_workspace. Use the registry variant so we can isolate from
    # the global LibV2 root (the MCP variant writes to LibV2/courses/).
    registry = _build_tool_registry()
    tool = registry["archive_to_libv2"]

    # Redirect LIBV2 root through the registry path — we need to check
    # the actual final course_dir, which includes "training_specs/".
    course_name = "WAVE30_GAP3_TEST"
    result_raw = await tool(
        course_name=course_name,
        project_workspace=str(trainforge_dir.parent),
        # No PDFs / HTML — just testing the trainforge copy pipeline.
        pdf_paths="",
        html_paths="",
    )
    result = json.loads(result_raw)
    assert "error" not in result, result
    # The slug is derived from the course name.
    slug = course_name.lower().replace("_", "-")

    # Walk LibV2 to find the course dir — registry variant writes under
    # the repo's LibV2/courses/.
    libv2_root = Path(__file__).resolve().parents[2] / "LibV2" / "courses"
    course_dir = libv2_root / slug

    try:
        instr = course_dir / "training_specs" / "instruction_pairs.jsonl"
        pref = course_dir / "training_specs" / "preference_pairs.jsonl"
        assert instr.exists(), (
            f"Wave 30 Gap 3: instruction_pairs.jsonl not archived to {instr}"
        )
        assert pref.exists(), (
            f"Wave 30 Gap 3: preference_pairs.jsonl not archived to {pref}"
        )
    finally:
        # Cleanup — test isolation.
        if course_dir.exists():
            shutil.rmtree(course_dir, ignore_errors=True)


def test_training_synthesis_phase_present_in_workflow_config():
    """Regression guard: the ``training_synthesis`` phase must be
    wired into ``textbook_to_course`` between ``trainforge_assessment``
    and ``libv2_archival``. Pre-Wave-30 the phase didn't exist at
    all — every textbook-to-course run skipped pair synthesis silently."""
    import yaml

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "workflows.yaml"
    )
    with config_path.open() as fh:
        workflows = yaml.safe_load(fh)

    t2c = workflows["workflows"]["textbook_to_course"]
    phase_names = [p["name"] for p in t2c["phases"]]
    assert "training_synthesis" in phase_names, (
        "Wave 30 Gap 3 regressed: training_synthesis phase missing"
    )
    assert (
        phase_names.index("trainforge_assessment")
        < phase_names.index("training_synthesis")
        < phase_names.index("libv2_archival")
    ), "Phase ordering: trainforge_assessment → training_synthesis → libv2_archival"

    # The phase routes to the training-synthesizer agent.
    synth_phase = next(
        p for p in t2c["phases"] if p["name"] == "training_synthesis"
    )
    assert synth_phase["agents"] == ["training-synthesizer"]
    assert synth_phase.get("optional") is True
