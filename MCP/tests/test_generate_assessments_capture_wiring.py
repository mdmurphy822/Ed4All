"""Wave 38 — ``_generate_assessments`` capture-wiring regression.

Pre-Wave-38 the Trainforge phase inside
``MCP/tools/pipeline_tools.py::_generate_assessments`` threaded a
``capture`` into ``AssessmentGenerator`` and logged one
``content_selection`` decision at the end, but no regression test
asserted the wiring. A silent regression on the capture side would
have gone unnoticed until post-hoc training-data review (by which
point the run's captures are already missing). This module pins the
contract: on a successful run, at least one decision is emitted via
the ``capture`` returned from ``create_trainforge_capture``.

Follows the precedent set by
``DART/tests/test_llm_classifier_capture_wiring.py`` and
``DART/tests/test_alt_text_generator_capture_wiring.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-use the existing IMSCC fixture from the contract tests so this
# wiring test doesn't duplicate the minimal HTML / manifest payload.
from MCP.tests.test_generate_assessments import (  # noqa: E402
    COURSE_CODE,
    _build_imscc,
)
from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def pipeline_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    tools: Dict[str, Callable] = _build_tool_registry()
    return tools, tmp_path


@pytest.mark.asyncio
async def test_generate_assessments_fires_content_selection_capture(
    pipeline_registry, monkeypatch,
):
    """On a successful Trainforge phase, the capture must emit at
    least one ``content_selection`` decision with dynamic rationale
    (chunks, misconceptions, questions, bloom_levels, question_count).

    Precedent: the per-figure alt-text generator, the per-batch LLM
    classifier, and the pipeline-run-attribution entry point all have
    capture-wiring tests — this closes the matching gap on the
    assessment-generation call site (root CLAUDE.md's enumerated
    precedents list).
    """
    tools, tmp_path = pipeline_registry
    project_id = f"PROJ-{COURSE_CODE}-CAPTURE"
    imscc_path = _build_imscc(tmp_path, project_id)

    # Capture wiring target: the module-level
    # ``create_trainforge_capture`` import resolves ``lib.trainforge_capture``
    # at call time (inside the tool body). Patch the binding on the
    # imported module so the tool receives our MagicMock.
    capture_mock = MagicMock()
    # Support context-manager protocol in case future wiring uses
    # ``with create_trainforge_capture(...) as capture:``; today it
    # uses the return value directly.
    capture_mock.__enter__ = MagicMock(return_value=capture_mock)
    capture_mock.__exit__ = MagicMock(return_value=False)

    def _fake_create_capture(*args: Any, **kwargs: Any):
        return capture_mock

    import lib.trainforge_capture as trainforge_capture_mod

    monkeypatch.setattr(
        trainforge_capture_mod,
        "create_trainforge_capture",
        _fake_create_capture,
    )

    result = await tools["generate_assessments"](
        course_id=COURSE_CODE,
        imscc_path=str(imscc_path),
        question_count=6,
        bloom_levels="remember,understand,apply",
        objective_ids=f"{COURSE_CODE}_OBJ_1,{COURSE_CODE}_OBJ_2",
        project_id=project_id,
        domain="general",
        division="STEM",
    )
    payload = json.loads(result)
    assert payload.get("success") is True, (
        f"Trainforge phase did not complete (cannot assert capture "
        f"wiring on a failed run). payload={payload}"
    )

    # One-or-more decisions must fire. The call-site-level
    # ``content_selection`` decision is the minimum contract; inner
    # ``AssessmentGenerator`` emits may add more.
    assert capture_mock.log_decision.called, (
        "create_trainforge_capture returned a capture but no "
        "log_decision call fired — capture wiring regressed."
    )

    # Specifically, one call must carry ``content_selection``.
    decision_types = {
        call.kwargs.get("decision_type") or (call.args[0] if call.args else None)
        for call in capture_mock.log_decision.call_args_list
    }
    assert "content_selection" in decision_types, (
        f"Expected a content_selection decision from the "
        f"_generate_assessments tail. Emitted types: {sorted(decision_types)}"
    )

    # Find the content_selection call and verify rationale carries
    # the dynamic signals documented in root CLAUDE.md ("rationale
    # must interpolate dynamic signals specific to the call").
    content_selection_calls = [
        call for call in capture_mock.log_decision.call_args_list
        if (call.kwargs.get("decision_type")
            or (call.args[0] if call.args else None)) == "content_selection"
    ]
    assert content_selection_calls, "no content_selection call captured"
    cs_call = content_selection_calls[0]
    rationale = (
        cs_call.kwargs.get("rationale")
        or (cs_call.args[2] if len(cs_call.args) >= 3 else "")
    )
    assert isinstance(rationale, str) and len(rationale) >= 20, (
        f"rationale must be >= 20 chars per project decision-capture "
        f"standard; got {rationale!r}"
    )
    # Dynamic signals required by the capture contract.
    assert "remember" in rationale or "apply" in rationale, (
        f"rationale should interpolate the chosen bloom_levels; "
        f"got {rationale!r}"
    )
    assert "6" in rationale, (
        f"rationale should mention the chosen question_count; "
        f"got {rationale!r}"
    )
