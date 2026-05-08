"""Wave W-D14 — COURSEPLANNER_PROVIDER short-circuit coverage.

The executor's ``_invoke_tool`` gained a fourth fork mirroring the
W-D11.A ``COURSEFORGE_PROVIDER`` short-circuit:

- Setting ``COURSEPLANNER_PROVIDER`` to a non-empty value AND
  ``agent_type == "course-outliner"`` bypasses the Wave-74 subagent
  dispatch and falls through to the legacy in-process registry path
  (which then constructs an ``OutlinerProvider`` to author the
  objectives via the operator-selected license-clean LLM).
- All other agents (content-generator under COURSEPLANNER_PROVIDER,
  oscqr-course-evaluator, etc.) keep dispatching unchanged.
- COURSEPLANNER_PROVIDER unset / empty preserves the pre-W-D14
  behaviour byte-for-byte: the dispatcher is invoked normally for
  course-outliner when ``ED4ALL_AGENT_DISPATCH=true``.

Mirrors ``test_executor_dispatch_routing.py``'s ``DummyDispatcher``
fixture pattern so the routing-fork invariants stay parallel.
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
    """Stand-in for ``plan_course_structure`` — returns a JSON envelope
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
            "plan_course_structure": _dummy_tool,
            "generate_course_content": _dummy_tool,
            "get_courseforge_status": _dummy_tool,
        },
        dispatcher=dispatcher,
        run_id="TEST_RUN_W_D14",
    )


# ---------------------------------------------------------------------------
# AGENT_SUBAGENT_SET pre-condition pin
# ---------------------------------------------------------------------------


def test_course_outliner_is_classified_as_subagent():
    """W-D14 short-circuit only matters if ``course-outliner`` is in
    ``AGENT_SUBAGENT_SET`` to begin with — otherwise the dispatcher
    branch never fires and the env-var has nothing to short-circuit.
    Pin the membership so a future refactor can't silently break the
    short-circuit by removing the agent from the set."""
    assert "course-outliner" in AGENT_SUBAGENT_SET


# ---------------------------------------------------------------------------
# Short-circuit fires when COURSEPLANNER_PROVIDER is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_courseplanner_provider_set_bypasses_dispatch_for_course_outliner(
    monkeypatch, state_runs_isolated,
):
    """Happy path — flag + dispatcher + course-outliner agent +
    COURSEPLANNER_PROVIDER set → fall through to in-process path."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "local")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "plan_course_structure",
        {
            "agent_type": "course-outliner",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    # Fell through to the in-process tool — dispatcher untouched.
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_courseplanner_provider_set_with_anthropic_value_bypasses_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Any non-empty COURSEPLANNER_PROVIDER value triggers bypass —
    not only ``local``. The in-process registry path then resolves
    the actual provider via the OutlinerProvider's env-var chain."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "anthropic")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "plan_course_structure",
        {
            "agent_type": "course-outliner",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


# ---------------------------------------------------------------------------
# Backward-compat regression: env unset preserves dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_courseplanner_provider_unset_preserves_dispatch(
    monkeypatch, state_runs_isolated,
):
    """COURSEPLANNER_PROVIDER unset/empty → dispatcher still fires for
    course-outliner. Pre-W-D14 behaviour byte-stable."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy",
            "outputs": {"objective_ids": "TO-01,CO-01"},
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
    # Dispatched as a subagent (legacy Wave-74 path).
    assert result["dispatch_mode"] == "dummy"
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "course-outliner"


@pytest.mark.asyncio
async def test_courseplanner_provider_empty_string_preserves_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Empty string is treated as "unset" for the short-circuit predicate —
    same semantics as the COURSEFORGE_PROVIDER short-circuit at
    ``executor.py:903``."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "   ")  # whitespace
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    await ex._invoke_tool(
        "plan_course_structure",
        {
            "agent_type": "course-outliner",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert len(dispatcher.calls) == 1


# ---------------------------------------------------------------------------
# Surgical scope: only course-outliner is short-circuited
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_courseplanner_provider_does_not_affect_content_generator(
    monkeypatch, state_runs_isolated,
):
    """The COURSEPLANNER_PROVIDER short-circuit only fires for the
    ``course-outliner`` agent. Other Wave-74 agents (content-generator,
    oscqr-course-evaluator, etc.) keep dispatching unchanged. Mirrors
    the surgical scope of the COURSEFORGE_PROVIDER short-circuit."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "local")
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    # content-generator with COURSEPLANNER_PROVIDER set should STILL
    # dispatch — the short-circuit is course-outliner-only.
    await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "content-generator"


@pytest.mark.asyncio
async def test_courseplanner_provider_independent_of_courseforge_provider(
    monkeypatch, state_runs_isolated,
):
    """The two short-circuits compose: setting COURSEPLANNER_PROVIDER
    bypasses dispatch for course-outliner, COURSEFORGE_PROVIDER for
    content-generator. They're independent — one being set doesn't
    affect the other agent."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEPLANNER_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "local")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    # course-outliner — bypassed by COURSEPLANNER_PROVIDER.
    result_outliner = await ex._invoke_tool(
        "plan_course_structure",
        {
            "agent_type": "course-outliner",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert result_outliner["dispatch_mode"] == "in_process"

    # content-generator — bypassed by COURSEFORGE_PROVIDER (which is
    # also set in this test).
    result_content = await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert result_content["dispatch_mode"] == "in_process"

    # Neither hit the dispatcher.
    assert dispatcher.calls == []
