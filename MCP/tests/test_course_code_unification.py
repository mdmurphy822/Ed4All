"""Wave 29 Defect 5 — course-code unification tests.

The OLSR_SIM_01 reproduction showed FOUR course codes in one run:

* DART derived ``BATES_540`` from the PDF filename,
* orchestrator derived ``OLSR_455`` from the workflow_id hash,
* CF/TF used ``OLSR_SIM_01`` from the CLI,
* a phantom ``OLSR_201`` surfaced in an intermediate capture.

Wave 29 pins ``params.canonical_course_code`` at workflow-creation
time (normalised ONCE from ``params.course_name``) and every
``DecisionCapture`` reads from that single source of truth. Legacy
call sites (no workflow state threaded through) fall back to the
Wave 22 DC4 PDF-stem derivation with a DEBUG log line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.decision_capture import normalize_course_code

# --------------------------------------------------------------------- #
# Normalisation helpers (preserved from Wave 22 DC4)
# --------------------------------------------------------------------- #


def test_normalize_course_code_idempotent():
    """Normalising a code twice yields the same value — a cached
    canonical_course_code must stay stable across phase boundaries."""
    first = normalize_course_code("my-awesome-course-2026")
    second = normalize_course_code(first)
    assert first == second


def test_normalize_course_code_deterministic():
    """Same input → same output. Tests that two captures built from
    the same source string agree on their course_id."""
    a = normalize_course_code("PHYS_101")
    b = normalize_course_code("PHYS_101")
    assert a == b == "PHYS_101"


# --------------------------------------------------------------------- #
# Workflow params carry canonical_course_code
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_workflow_impl_injects_canonical_course_code(tmp_path, monkeypatch):
    """The generic ``create_workflow_impl`` path should auto-populate
    ``canonical_course_code`` from ``course_name``."""
    from MCP.tools import orchestrator_tools as ot

    # Redirect workflow state into a temp dir.
    monkeypatch.setattr(ot, "STATE_PATH", tmp_path)

    raw_params = {"course_name": "TestCourse-2026", "corpus": "x.pdf"}
    result = await ot.create_workflow_impl(
        workflow_type="textbook_to_course",
        params=json.dumps(raw_params),
    )
    data = json.loads(result)
    assert data.get("success") is True

    # Load the persisted workflow state — its params should carry the
    # canonical course code pinned at creation.
    wf_path = Path(data["workflow_path"])
    state = json.loads(wf_path.read_text())
    assert "canonical_course_code" in state["params"]
    assert state["params"]["canonical_course_code"] == normalize_course_code(
        "TestCourse-2026"
    )


@pytest.mark.asyncio
async def test_create_workflow_impl_preserves_supplied_canonical_cc(tmp_path, monkeypatch):
    """When the caller already supplied ``canonical_course_code`` the
    helper must not overwrite it (trust the caller's pre-normalisation
    — idempotency guarantees they agree anyway)."""
    from MCP.tools import orchestrator_tools as ot

    monkeypatch.setattr(ot, "STATE_PATH", tmp_path)

    raw_params = {
        "course_name": "ignored_here",
        "canonical_course_code": "PINNED_042",
    }
    result = await ot.create_workflow_impl(
        workflow_type="textbook_to_course",
        params=json.dumps(raw_params),
    )
    data = json.loads(result)
    wf_path = Path(data["workflow_path"])
    state = json.loads(wf_path.read_text())
    assert state["params"]["canonical_course_code"] == "PINNED_042"


# --------------------------------------------------------------------- #
# Orchestrator reads canonical_course_code when building executor capture
# --------------------------------------------------------------------- #


def test_pipeline_orchestrator_reads_canonical_course_code(tmp_path, monkeypatch):
    """``_get_executor`` should use ``params.canonical_course_code``
    when building its DecisionCapture, NOT re-derive from the
    workflow_id / workflow_type."""
    from lib.decision_capture import DecisionCapture
    from MCP.orchestrator.pipeline_orchestrator import PipelineOrchestrator

    captured = {}

    def spy_init(
        self,
        course_code,
        phase,
        tool="courseforge",
        streaming=True,
        task_id=None,
    ):
        captured["course_code"] = course_code
        captured["phase"] = phase
        captured["tool"] = tool
        # Simulate minimal DecisionCapture shape without touching disk.
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        self.session_id = "test"
        self.streaming_mode = False
        self.decisions = []
        self.course_id = course_code
        self.run_id = "test_run"

    monkeypatch.setattr(DecisionCapture, "__init__", spy_init)
    monkeypatch.setattr(DecisionCapture, "close", lambda self: None)

    orch = PipelineOrchestrator()
    state = {
        "workflow_id": "WF-20260420-abc12345",
        "type": "textbook_to_course",
        "params": {
            "course_name": "SomeCourseName-2026",
            "canonical_course_code": "CUSTOM_PINNED_042",
        },
    }
    orch._get_executor(workflow_state=state)

    # The capture should have been constructed with the pinned code,
    # not a re-normalised value derived from course_name.
    assert captured.get("course_code") == "CUSTOM_PINNED_042"


def test_pipeline_orchestrator_falls_back_without_canonical_code(tmp_path, monkeypatch):
    """When workflow_state lacks ``canonical_course_code`` (legacy
    states created pre-Wave-29), the orchestrator falls back to
    normalising ``course_name``."""
    from lib.decision_capture import DecisionCapture
    from MCP.orchestrator.pipeline_orchestrator import PipelineOrchestrator

    captured = {}

    def spy_init(
        self,
        course_code,
        phase,
        tool="courseforge",
        streaming=True,
        task_id=None,
    ):
        captured["course_code"] = course_code
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        self.session_id = "test"
        self.streaming_mode = False
        self.decisions = []
        self.course_id = course_code
        self.run_id = "test_run"

    monkeypatch.setattr(DecisionCapture, "__init__", spy_init)
    monkeypatch.setattr(DecisionCapture, "close", lambda self: None)

    orch = PipelineOrchestrator()
    state = {
        "workflow_id": "WF-LEGACY",
        "type": "textbook_to_course",
        "params": {
            "course_name": "LegacyWorkflow",
            # canonical_course_code INTENTIONALLY absent.
        },
    }
    # Force a fresh executor build.
    orch._executor = None
    orch._get_executor(workflow_state=state)

    expected = normalize_course_code("LegacyWorkflow")
    assert captured["course_code"] == expected


# --------------------------------------------------------------------- #
# Pipeline tools convert helper honours canonical_course_code
# --------------------------------------------------------------------- #


def test_raw_text_to_accessible_html_accepts_canonical_course_code(tmp_path):
    """The DART converter entry point now accepts a
    ``canonical_course_code`` kwarg that overrides the PDF-stem-derived
    code used by the short-lived owned capture."""
    import inspect

    from MCP.tools.pipeline_tools import _raw_text_to_accessible_html

    sig = inspect.signature(_raw_text_to_accessible_html)
    assert "canonical_course_code" in sig.parameters, (
        "Expected canonical_course_code kwarg on _raw_text_to_accessible_html"
    )


def test_extract_and_convert_pdf_threads_canonical_course_code():
    """The ``extract_and_convert_pdf`` registry entry should read
    ``canonical_course_code`` from kwargs so the orchestrator's
    threaded value overrides PDF-stem derivation."""
    import inspect

    from MCP.tools import pipeline_tools

    src = inspect.getsource(pipeline_tools._build_tool_registry)
    # A literal presence check is sufficient — the handler builds a
    # closure inside ``_build_tool_registry`` and we need to confirm it
    # reads the kwarg name in that scope.
    assert "canonical_course_code" in src, (
        "Expected canonical_course_code threading in _build_tool_registry"
    )
