"""Wave 26 — ``generate_assessments`` (trainforge_tools.py) unification tests.

Pre-Wave-26 bug: ``MCP/tools/trainforge_tools.py:365-388`` hand-rolled
question dicts with literal ``"Correct answer based on content"``
placeholders and always returned success. MCP clients invoking the
externally-registered ``generate_assessments`` tool got placeholder
output.

Wave 26 fix: the tool dispatches through
:class:`Trainforge.generators.assessment_generator.AssessmentGenerator`,
the same generator the internal pipeline uses. On generator error the
tool returns a structured error — never placeholder success.
"""
from __future__ import annotations

import asyncio
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _CapturingMCP:
    """Minimal stand-in for a FastMCP server that records decorated tools."""

    def __init__(self):
        self.tools = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator


@pytest.fixture
def generate_assessments_tool(tmp_path, monkeypatch):
    """Register the Trainforge MCP tools and return the
    ``generate_assessments`` callable bound to a tmp training dir."""
    # Redirect TRAINING_OUTPUT + _PROJECT_ROOT to tmp_path so tests
    # don't pollute the real exports/ tree and so the secure_paths
    # validator doesn't reject tmp IMSCC paths.
    from MCP.tools import trainforge_tools

    trainforge_tools.TRAINING_OUTPUT = tmp_path / "trainforge_out"
    trainforge_tools.TRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(trainforge_tools, "_PROJECT_ROOT", tmp_path)

    mcp = _CapturingMCP()
    trainforge_tools.register_trainforge_tools(mcp)
    assert "generate_assessments" in mcp.tools
    return mcp.tools["generate_assessments"]


def _build_imscc(tmp_path: Path) -> Path:
    """Build a small IMSCC with readable HTML so the canonical generator
    can actually extract content (and thus avoid template fallbacks)."""
    imscc_path = tmp_path / "test_course.imscc"
    manifest = """<?xml version="1.0"?>
<manifest identifier="M1" xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">
  <resources>
    <resource identifier="R1" type="webcontent" href="mitosis.html">
      <file href="mitosis.html"/>
    </resource>
  </resources>
</manifest>"""
    html = """<!DOCTYPE html><html><body>
<h1>Mitosis Overview</h1>
<p><strong>Mitosis</strong> is the process of cell division in eukaryotes
that produces two genetically identical daughter cells. It consists of
four phases: prophase, metaphase, anaphase, and telophase. Mitosis
ensures accurate distribution of chromosomes to daughter cells.</p>
<p><strong>Prophase</strong> is the first phase of mitosis during which
chromatin condenses into visible chromosomes. Prophase is the phase when
chromosomes first become visible under a light microscope.</p>
<p><strong>Metaphase</strong> is the phase in which chromosomes align at
the metaphase plate. Metaphase provides the checkpoint before
segregation.</p>
</body></html>"""
    with zipfile.ZipFile(imscc_path, "w") as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("mitosis.html", html)
    return imscc_path


def test_dispatches_through_assessment_generator(generate_assessments_tool, tmp_path):
    """The MCP tool must call through to :class:`AssessmentGenerator`,
    not the legacy hand-rolled loop."""
    imscc_path = _build_imscc(tmp_path)

    with patch(
        "Trainforge.generators.assessment_generator.AssessmentGenerator.generate",
        wraps=None,
    ) as mock_gen:
        # Return a minimal AssessmentData-shaped object.
        from Trainforge.generators.assessment_generator import (
            AssessmentData,
            QuestionData,
        )
        mock_gen.return_value = AssessmentData(
            assessment_id="ASM-TEST",
            title="Test",
            course_code="TEST",
            questions=[
                QuestionData(
                    question_id="q-001",
                    question_type="multiple_choice",
                    stem="<p>Explain mitosis</p>",
                    bloom_level="understand",
                    objective_id="LO-01",
                    choices=[
                        {"text": "<p>cell division</p>", "is_correct": True},
                        {"text": "<p>wrong</p>", "is_correct": False},
                        {"text": "<p>also wrong</p>", "is_correct": False},
                        {"text": "<p>still wrong</p>", "is_correct": False},
                    ],
                ),
            ],
            objectives_targeted=["LO-01"],
            bloom_levels=["understand"],
        )

        result = asyncio.run(generate_assessments_tool(
            course_id="TEST",
            objective_ids="LO-01",
            bloom_levels="understand",
            question_count=1,
            imscc_path=str(imscc_path),
        ))

    payload = json.loads(result)
    assert payload.get("success") is True, payload
    # The wrapper should have been called exactly once.
    assert mock_gen.called, "AssessmentGenerator.generate was not called"
    # Returned payload carries the generator_path marker.
    assert payload.get("generator_path") == "AssessmentGenerator"


def test_no_placeholder_strings_in_output(generate_assessments_tool, tmp_path):
    """The returned assessment JSON (on disk + in payload) must NOT
    contain the legacy placeholder strings like 'Correct answer based
    on content'."""
    imscc_path = _build_imscc(tmp_path)

    result = asyncio.run(generate_assessments_tool(
        course_id="TEST",
        objective_ids="LO-01",
        bloom_levels="understand",
        question_count=2,
        imscc_path=str(imscc_path),
    ))

    payload = json.loads(result)
    assert payload.get("success") is True, payload

    output_path = Path(payload["output_path"])
    text = output_path.read_text()

    # Placeholders from the pre-Wave-26 hand-rolled loop.
    forbidden = [
        "Correct answer based on content",
        "Plausible distractor A",
        "Plausible distractor B",
        "Plausible distractor C",
    ]
    for phrase in forbidden:
        assert phrase not in text, (
            f"Placeholder string {phrase!r} leaked into generated "
            f"assessment. Wave 26 requires the canonical generator path."
        )


def test_error_when_no_chunks_and_no_rag(generate_assessments_tool, tmp_path):
    """With no valid RAG and no imscc_path, the tool must return a
    structured error — NOT a placeholder success response."""
    result = asyncio.run(generate_assessments_tool(
        course_id="TEST",
        objective_ids="LO-01",
        bloom_levels="understand",
        question_count=1,
        # No imscc_path, no course_slug → no chunks available.
    ))

    payload = json.loads(result)
    assert "error" in payload, payload
    assert "success" not in payload or payload.get("success") is not True
    # Cause field surfaces the specific failure mode.
    assert payload.get("cause") in ("no_chunks", "empty_bloom_levels",
                                     "empty_objective_ids", "import_failed")


def test_output_shape_matches_pipeline_path(generate_assessments_tool, tmp_path):
    """The MCP surface must return a shape the pipeline runner can
    consume — required keys present, types correct."""
    imscc_path = _build_imscc(tmp_path)

    result = asyncio.run(generate_assessments_tool(
        course_id="TESTBETA",
        objective_ids="LO-01,LO-02",
        bloom_levels="remember,understand",
        question_count=4,
        imscc_path=str(imscc_path),
    ))

    payload = json.loads(result)
    assert payload.get("success") is True, payload
    for key in (
        "success", "assessment_id", "question_count", "output_path",
        "generator_path",
    ):
        assert key in payload, f"missing key {key!r}: {payload}"
    assert isinstance(payload["question_count"], int)

    # Written file is a single well-formed JSON document with the
    # AssessmentData.to_dict() shape.
    doc = json.loads(Path(payload["output_path"]).read_text())
    assert doc["assessment_id"] == payload["assessment_id"]
    assert isinstance(doc["questions"], list)
    # Each question carries the generator's canonical shape keys.
    for q in doc["questions"]:
        assert "question_id" in q
        assert "bloom_level" in q
        assert "stem" in q
