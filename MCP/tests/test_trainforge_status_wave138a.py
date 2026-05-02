"""Wave 138a — MCP-surface tests for the new
``analyze_teaching_role_alignment`` tool plus the
``get_trainforge_status`` extensions (in-flight checkpoint sidecars,
latest eval ``alignment_rate``).

The fixture pattern mirrors
``MCP/tests/test_generate_assessments_single_path.py``: a minimal
capturing MCP that records the decorated tools, plus tmp-path
redirection so the tests don't touch the real ``LibV2/courses/`` or
``training-captures/`` trees.
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
def trainforge_tools_module(tmp_path, monkeypatch):
    """Register Trainforge MCP tools against a tmp LibV2 + training tree.

    Returns the registered tool dict so individual tests can pull
    ``analyze_teaching_role_alignment`` / ``get_trainforge_status``.
    """
    from MCP.tools import trainforge_tools

    libv2_courses = tmp_path / "LibV2_courses"
    libv2_courses.mkdir()
    training_output = tmp_path / "trainforge_out"
    training_output.mkdir()

    monkeypatch.setattr(trainforge_tools, "TRAINING_OUTPUT", training_output)
    monkeypatch.setattr(trainforge_tools, "LIBV2_COURSES", libv2_courses)
    monkeypatch.setattr(trainforge_tools, "_PROJECT_ROOT", tmp_path)

    mcp = _CapturingMCP()
    trainforge_tools.register_trainforge_tools(mcp)
    return {
        "tools": mcp.tools,
        "libv2_courses": libv2_courses,
        "training_output": training_output,
        "tmp_path": tmp_path,
    }


def _write_chunks(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# analyze_teaching_role_alignment
# ---------------------------------------------------------------------------


def test_analyze_teaching_role_alignment_emits_summary(trainforge_tools_module):
    """The tool wraps the evaluator; the ``summary`` shape is preserved."""
    tool = trainforge_tools_module["tools"]["analyze_teaching_role_alignment"]
    libv2 = trainforge_tools_module["libv2_courses"]

    chunks = libv2 / "demo-101" / "corpus" / "chunks.jsonl"
    rows = []
    # 6 definition chunks, all dominantly "introduce" → aligned
    for i in range(6):
        rows.append({
            "chunk_id": f"d{i}",
            "content_type_label": "definition",
            "teaching_role": "introduce",
        })
    # 6 real_world_scenario chunks, all "elaborate" (NOT "transfer")
    # → mismatch (expected_role=transfer, min_share=0.70)
    for i in range(6):
        rows.append({
            "chunk_id": f"r{i}",
            "content_type_label": "real_world_scenario",
            "teaching_role": "elaborate",
        })
    _write_chunks(chunks, rows)

    raw = asyncio.run(tool(str(chunks)))
    payload = json.loads(raw)

    assert "summary" in payload
    summary = payload["summary"]
    assert summary["content_types_with_expected_mode"] == 2
    assert "real_world_scenario" in summary["mismatched_content_types"]
    # 1 of 2 rule-bearing types aligned → 0.5
    assert summary["alignment_rate"] == pytest.approx(0.5)
    # The corpus path is annotated for the consumer.
    assert payload["chunks_path"] == str(chunks)


def test_analyze_teaching_role_alignment_missing_path(trainforge_tools_module):
    tool = trainforge_tools_module["tools"]["analyze_teaching_role_alignment"]
    raw = asyncio.run(tool("/no/such/chunks.jsonl"))
    payload = json.loads(raw)
    assert payload.get("cause") == "missing_chunks"


def test_analyze_teaching_role_alignment_invalid_min_chunks(
    trainforge_tools_module, tmp_path
):
    tool = trainforge_tools_module["tools"]["analyze_teaching_role_alignment"]
    chunks = tmp_path / "stub.jsonl"
    chunks.write_text("", encoding="utf-8")
    raw = asyncio.run(tool(str(chunks), min_chunks_for_flag=0))
    payload = json.loads(raw)
    assert payload.get("cause") == "invalid_argument"


# ---------------------------------------------------------------------------
# get_trainforge_status — Wave 138a extensions
# ---------------------------------------------------------------------------


def test_get_trainforge_status_surfaces_in_flight_checkpoints(
    trainforge_tools_module,
):
    tool = trainforge_tools_module["tools"]["get_trainforge_status"]
    libv2 = trainforge_tools_module["libv2_courses"]

    course = libv2 / "demo-101"
    (course / "training_specs").mkdir(parents=True)
    (course / "corpus").mkdir(parents=True)
    (course / "eval").mkdir(parents=True)
    (course / "models" / "demo-101-qwen2-5-1-5b-abcdef12" / "eval").mkdir(parents=True)

    # All four sidecar shapes the extension surveys.
    (course / "training_specs" / ".synthesis_pairs_checkpoint.jsonl").write_text(
        '{"chunk_id":"c1","kind":"instruction"}\n', encoding="utf-8",
    )
    (course / "corpus" / ".teaching_role_checkpoint.jsonl").write_text(
        '{"chunk_id":"c1","teaching_role":"introduce"}\n', encoding="utf-8",
    )
    (course / "eval" / ".eval_results_checkpoint.jsonl").write_text(
        '{"stage":"faithfulness","accuracy":0.5}\n', encoding="utf-8",
    )
    (
        course / "models" / "demo-101-qwen2-5-1-5b-abcdef12"
        / "eval" / ".eval_results_checkpoint.jsonl"
    ).write_text(
        '{"stage":"source_match","accuracy":0.4}\n', encoding="utf-8",
    )

    raw = asyncio.run(tool())
    payload = json.loads(raw)

    sidecars = payload["in_flight_checkpoints"]
    kinds = {entry["kind"] for entry in sidecars}
    assert kinds == {
        "synthesis_pairs",
        "teaching_role",
        "eval_stage_course",
        "eval_stage_adapter",
    }
    # Adapter sidecar carries a model_id so an operator knows which
    # adapter has resumable eval state.
    adapter_entry = next(
        e for e in sidecars if e["kind"] == "eval_stage_adapter"
    )
    assert adapter_entry["model_id"] == "demo-101-qwen2-5-1-5b-abcdef12"
    assert adapter_entry["course_slug"] == "demo-101"
    assert adapter_entry["size_bytes"] > 0


def test_get_trainforge_status_surfaces_alignment_rate(
    trainforge_tools_module,
):
    tool = trainforge_tools_module["tools"]["get_trainforge_status"]
    libv2 = trainforge_tools_module["libv2_courses"]

    course = libv2 / "demo-101"
    adapter = course / "models" / "demo-101-qwen2-5-1-5b-deadbeef"
    adapter.mkdir(parents=True)

    eval_report = {
        "model_id": "demo-101-qwen2-5-1-5b-deadbeef",
        "content_type_role_alignment_summary": {
            "alignment_rate": 0.83,
            "mismatched_content_types": ["real_world_scenario"],
            "content_types_with_expected_mode": 6,
        },
    }
    (adapter / "eval_report.json").write_text(
        json.dumps(eval_report), encoding="utf-8",
    )

    raw = asyncio.run(tool())
    payload = json.loads(raw)

    assert payload["role_alignment"], "expected role_alignment entries"
    entry = payload["role_alignment"][0]
    assert entry["course_slug"] == "demo-101"
    assert entry["model_id"] == "demo-101-qwen2-5-1-5b-deadbeef"
    assert entry["alignment_rate"] == pytest.approx(0.83)
    assert "real_world_scenario" in entry["mismatched_content_types"]


def test_get_trainforge_status_skips_smoke_eval_reports(
    trainforge_tools_module,
):
    """Per the harness contract, smoke-mode eval reports are advisory
    and must not be surfaced as a real alignment_rate."""
    tool = trainforge_tools_module["tools"]["get_trainforge_status"]
    libv2 = trainforge_tools_module["libv2_courses"]

    adapter = libv2 / "demo-101" / "models" / "demo-101-smoke"
    adapter.mkdir(parents=True)
    smoke_report = {
        "smoke_mode": True,
        "content_type_role_alignment_summary": {
            "alignment_rate": 0.99,
            "mismatched_content_types": [],
            "content_types_with_expected_mode": 1,
        },
    }
    (adapter / "eval_report.json").write_text(
        json.dumps(smoke_report), encoding="utf-8",
    )

    raw = asyncio.run(tool())
    payload = json.loads(raw)

    entry = payload["role_alignment"][0]
    assert entry.get("smoke_mode") is True
    # alignment_rate from a smoke run is intentionally None — operators
    # should not gate retrain decisions on a 3-prompt eval.
    assert entry["alignment_rate"] is None


def test_get_trainforge_status_handles_empty_libv2(trainforge_tools_module):
    """A clean checkout with no LibV2 courses returns empty arrays, not
    an error."""
    tool = trainforge_tools_module["tools"]["get_trainforge_status"]
    raw = asyncio.run(tool())
    payload = json.loads(raw)

    assert payload["in_flight_checkpoints"] == []
    assert payload["role_alignment"] == []
    assert "error" not in payload


def test_get_trainforge_status_handles_pre_wave138a_eval_reports(
    trainforge_tools_module,
):
    """A legacy eval_report.json without the Wave 138a alignment field
    should surface a ``note`` rather than an error."""
    tool = trainforge_tools_module["tools"]["get_trainforge_status"]
    libv2 = trainforge_tools_module["libv2_courses"]

    adapter = libv2 / "legacy-101" / "models" / "legacy-101-qwen-old"
    adapter.mkdir(parents=True)
    (adapter / "eval_report.json").write_text(
        json.dumps({"model_id": "legacy-101-qwen-old", "faithfulness": 0.6}),
        encoding="utf-8",
    )

    raw = asyncio.run(tool())
    payload = json.loads(raw)
    entry = payload["role_alignment"][0]
    assert entry["alignment_rate"] is None
    assert "predates Wave 138a" in entry.get("note", "")
