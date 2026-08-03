"""Wave W-D15 — TRAINFORGE_ASSESSMENT_PROVIDER short-circuit coverage.

The executor's ``_invoke_tool`` gained a fifth fork mirroring the
W-D11.A ``COURSEFORGE_PROVIDER`` and W-D14 ``COURSEPLANNER_PROVIDER``
short-circuits:

- Setting ``TRAINFORGE_ASSESSMENT_PROVIDER`` to a non-empty value
  AND ``agent_type == "assessment-generator"`` bypasses the Wave-74
  subagent dispatch and falls through to the legacy in-process
  registry path (which then constructs an
  ``AssessmentGeneratorProvider`` to author the questions via the
  operator-selected license-clean LLM).
- All other agents (course-outliner under
  TRAINFORGE_ASSESSMENT_PROVIDER, content-generator,
  oscqr-course-evaluator, etc.) keep dispatching unchanged.
- TRAINFORGE_ASSESSMENT_PROVIDER unset / empty preserves the
  pre-W-D15 behaviour byte-for-byte: the dispatcher is invoked
  normally for assessment-generator when
  ``ED4ALL_AGENT_DISPATCH=true``.

Mirrors ``test_executor_courseplanner_provider.py``'s
``DummyDispatcher`` fixture pattern so the routing-fork invariants
stay parallel.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from MCP.core.executor import (
    AGENT_SUBAGENT_SET,
    TaskExecutor,
)


class DummyDispatcher:
    """Records every ``dispatch_task`` call and returns a fixed envelope."""

    def __init__(
        self,
        response: Optional[Dict[str, Any]] = None,
        raise_exc: Optional[Exception] = None,
    ):
        self.response = response or {
            "success": True,
            "dispatch_mode": "dummy",
            "outputs": {"ok": True},
            "artifacts": [],
        }
        self.raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []

    async def dispatch_task(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append({
            "task_name": task_name,
            "agent_type": agent_type,
            "task_params": dict(task_params),
            "run_id": run_id,
            "phase_context": phase_context,
        })
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


async def _dummy_tool(**kwargs) -> str:
    """Stand-in for ``generate_assessments`` — returns a JSON envelope
    the executor parses back into a dict, mirroring real MCP tools."""
    return json.dumps({
        "success": True,
        "dispatch_mode": "in_process",
        "received_params": sorted(kwargs.keys()),
    })


def _make_executor(
    dispatcher: Optional[DummyDispatcher] = None,
) -> TaskExecutor:
    """Minimal executor wired to ``_dummy_tool`` for the relevant tool names."""
    return TaskExecutor(
        tool_registry={
            "generate_assessments": _dummy_tool,
            "plan_course_structure": _dummy_tool,
            "generate_course_content": _dummy_tool,
            "validate_assessment": _dummy_tool,
        },
        dispatcher=dispatcher,
        run_id="TEST_RUN_W_D15",
    )


# ---------------------------------------------------------------------------
# AGENT_SUBAGENT_SET pre-condition pin
# ---------------------------------------------------------------------------


def test_assessment_generator_is_classified_as_subagent():
    """W-D15 short-circuit only matters if ``assessment-generator`` is
    in ``AGENT_SUBAGENT_SET`` to begin with — otherwise the dispatcher
    branch never fires and the env-var has nothing to short-circuit.
    Pin the membership so a future refactor can't silently break the
    short-circuit by removing the agent from the set."""
    assert "assessment-generator" in AGENT_SUBAGENT_SET


# ---------------------------------------------------------------------------
# Short-circuit fires when TRAINFORGE_ASSESSMENT_PROVIDER is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_set_bypasses_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Happy path — flag + dispatcher + assessment-generator agent +
    TRAINFORGE_ASSESSMENT_PROVIDER set → fall through to in-process
    path."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "local")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "generate_assessments",
        {
            "agent_type": "assessment-generator",
            "params": {
                "course_id": "FXALPHA_101",
                "objective_ids": "TO-01",
                "bloom_levels": "understand",
            },
        },
    )
    # Fell through to the in-process tool — dispatcher untouched.
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_anthropic_value_bypasses_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Any non-empty TRAINFORGE_ASSESSMENT_PROVIDER value triggers
    bypass — not only ``local``. The in-process registry path then
    resolves the actual provider via the AssessmentGeneratorProvider's
    env-var chain."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "anthropic")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "generate_assessments",
        {
            "agent_type": "assessment-generator",
            "params": {
                "course_id": "FXALPHA_101",
                "objective_ids": "TO-01",
                "bloom_levels": "understand",
            },
        },
    )
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


# ---------------------------------------------------------------------------
# Backward-compat regression: env unset preserves dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_unset_preserves_dispatch(
    monkeypatch, state_runs_isolated,
):
    """TRAINFORGE_ASSESSMENT_PROVIDER unset/empty → dispatcher still
    fires for assessment-generator. Pre-W-D15 behaviour byte-stable."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.delenv("TRAINFORGE_ASSESSMENT_PROVIDER", raising=False)
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy",
            "outputs": {"question_ids": "Q-001,Q-002"},
            "artifacts": [],
        },
    )
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "generate_assessments",
        {
            "agent_type": "assessment-generator",
            "params": {
                "course_id": "FXALPHA_101",
                "objective_ids": "TO-01",
                "bloom_levels": "understand",
            },
        },
    )
    # Dispatched as a subagent (legacy Wave-74 path).
    assert result["dispatch_mode"] == "dummy"
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "assessment-generator"


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_empty_string_preserves_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Empty string is treated as "unset" for the short-circuit
    predicate — same semantics as the COURSEFORGE_PROVIDER /
    COURSEPLANNER_PROVIDER short-circuits at ``executor.py``."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "   ")  # whitespace
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    await ex._invoke_tool(
        "generate_assessments",
        {
            "agent_type": "assessment-generator",
            "params": {
                "course_id": "FXALPHA_101",
                "objective_ids": "TO-01",
                "bloom_levels": "understand",
            },
        },
    )
    assert len(dispatcher.calls) == 1


# ---------------------------------------------------------------------------
# Surgical scope: only assessment-generator is short-circuited
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_does_not_affect_course_outliner(
    monkeypatch, state_runs_isolated,
):
    """The TRAINFORGE_ASSESSMENT_PROVIDER short-circuit only fires for
    the ``assessment-generator`` agent. Other Wave-74 agents
    (course-outliner, content-generator, oscqr-course-evaluator, etc.)
    keep dispatching unchanged. Mirrors the surgical scope of the
    COURSEFORGE_PROVIDER / COURSEPLANNER_PROVIDER short-circuits."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "local")
    # Important: clear OTHER provider envs so this test's
    # course-outliner dispatch path isn't short-circuited by a
    # stale env.
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy_outliner",
            "outputs": {},
            "artifacts": [],
        },
    )
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "plan_course_structure",
        {
            "agent_type": "course-outliner",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    # Dispatcher fired — course-outliner is NOT short-circuited.
    assert result["dispatch_mode"] == "dummy_outliner"
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "course-outliner"


@pytest.mark.asyncio
async def test_trainforge_assessment_provider_does_not_affect_content_generator(
    monkeypatch, state_runs_isolated,
):
    """Content-generator agent keeps dispatching even when
    TRAINFORGE_ASSESSMENT_PROVIDER is set. Pin the surgical scope."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_ASSESSMENT_PROVIDER", "local")
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy_content_gen",
            "outputs": {},
            "artifacts": [],
        },
    )
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "week_number": 1},
        },
    )
    assert result["dispatch_mode"] == "dummy_content_gen"
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "content-generator"
