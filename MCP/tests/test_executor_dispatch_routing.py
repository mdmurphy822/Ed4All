"""Wave 74 Session 1 — TaskExecutor routing fork coverage.

The executor's ``_invoke_tool`` gained a three-way fork:

1. ``ED4ALL_AGENT_DISPATCH=true`` + dispatcher wired + ``agent_type`` in
   ``AGENT_SUBAGENT_SET`` → route through ``dispatcher.dispatch_task``.
2. Any one of those conditions missing → legacy in-process
   ``tool_registry[tool_name](**mapped_params)`` path.
3. Subagent-classified agent_type but feature flag off → still legacy
   in-process path (so Session 1 lands inert).

These tests pin the routing invariants end-to-end without exercising
the mailbox bridge (that's covered in
``test_local_dispatcher_dispatch_task.py``). The dispatcher is stubbed
with a ``DummyDispatcher`` that records invocations.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from MCP.core.executor import (
    AGENT_PROVIDER_ENV_MAP,
    AGENT_SUBAGENT_SET,
    TaskExecutor,
    _agent_dispatch_enabled,
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
    """Stand-in for any entry in ``tool_registry``. Returns a JSON string
    the executor's ``_invoke_tool`` parses back into a dict, mirroring
    real MCP tool envelopes.
    """
    return json.dumps({
        "success": True,
        "dispatch_mode": "in_process",
        "received_params": sorted(kwargs.keys()),
    })


def _make_executor(
    dispatcher: Optional[DummyDispatcher] = None,
) -> TaskExecutor:
    """Minimal executor wired to ``_dummy_tool`` for every mapped name."""
    # Pick a tool_name the param_mapper already knows about so
    # mapping doesn't blow up. ``generate_course_content`` is in the
    # real registry; ``get_courseforge_status`` is a simple status tool
    # that accepts no required params. Both exist in the parameter
    # mapper's schema registry.
    return TaskExecutor(
        tool_registry={
            "generate_course_content": _dummy_tool,
            "get_courseforge_status": _dummy_tool,
            "stage_dart_outputs": _dummy_tool,
            "synthesize_training": _dummy_tool,
        },
        dispatcher=dispatcher,
        run_id="TEST_RUN",
    )


@pytest.mark.asyncio
async def test_feature_flag_off_routes_to_in_process_tool(monkeypatch, state_runs_isolated):
    """Default (flag unset) — every agent, subagent-classified or not,
    runs through the legacy ``tool_registry`` path. Session 1's
    promise: landing the code doesn't change any existing run."""
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    # content-generator IS in AGENT_SUBAGENT_SET — yet flag is off, so
    # we must still route to the in-process tool. Tool required
    # params are what ``generate_course_content`` expects.
    result = await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )

    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []  # dispatcher never invoked


@pytest.mark.asyncio
async def test_feature_flag_on_subagent_agent_routes_to_dispatcher(
    monkeypatch, state_runs_isolated,
):
    """Happy path — flag + dispatcher + classified agent → dispatch_task."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    assert _agent_dispatch_enabled()
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy",
            "outputs": {"weeks_generated": 12},
            "artifacts": ["week_01_overview.html"],
        },
    )
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )

    assert result["dispatch_mode"] == "dummy"
    assert result["outputs"]["weeks_generated"] == 12
    assert len(dispatcher.calls) == 1
    call = dispatcher.calls[0]
    assert call["task_name"] == "generate_course_content"
    assert call["agent_type"] == "content-generator"
    assert call["run_id"] == "TEST_RUN"
    # Param-mapping still runs before dispatch so the subagent sees the
    # same mapped kwargs the Python tool would have.
    assert "project_id" in call["task_params"]


@pytest.mark.asyncio
async def test_feature_flag_on_python_tool_agent_stays_in_process(
    monkeypatch, state_runs_isolated,
):
    """Flag on, but agent is Python-tool-classified (not in
    AGENT_SUBAGENT_SET) → legacy path. SemantiK conversion, packaging,
    TF-IDF routing etc. must stay in-process. Uses
    ``get_courseforge_status`` which has zero required params so the
    test stays focused on routing behaviour, not parameter mapping."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "get_courseforge_status",
        {
            "agent_type": "requirements-collector",  # NOT in AGENT_SUBAGENT_SET
            "project_id": "PROJ-X",
        },
    )

    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_feature_flag_on_no_dispatcher_falls_through(monkeypatch, state_runs_isolated):
    """Flag on but dispatcher unset → legacy path. Guards against a
    PipelineOrchestrator misconfiguration where ``_get_dispatcher``
    returned None but the flag was still set."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    ex = _make_executor(dispatcher=None)  # no dispatcher

    result = await ex._invoke_tool(
        "generate_course_content",
        {
            "agent_type": "content-generator",
            "params": {"project_id": "PROJ-X", "course_name": "X"},
        },
    )
    assert result["dispatch_mode"] == "in_process"


@pytest.mark.asyncio
async def test_feature_flag_on_unknown_agent_type_falls_through(monkeypatch, state_runs_isolated):
    """Agent type not in AGENT_SUBAGENT_SET (and maybe not in any set
    at all — e.g. a typo or a new agent not yet classified) falls
    through to the legacy path. We don't second-guess — if the
    workflow routed the task to a Python tool, we call the Python
    tool."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "get_courseforge_status",
        {
            "agent_type": "unknown-agent-type-xyz",
            "project_id": "PROJ-X",
        },
    )
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


@pytest.mark.asyncio
async def test_feature_flag_truthy_variants(monkeypatch, state_runs_isolated):
    """`1`, `true`, `yes`, `on` (any case) all enable the flag.
    Anything else — including `0`, `false`, empty — disables it.
    Uses ``generate_course_content`` with its minimal required
    ``project_id`` so parameter mapping passes; the assertion focuses
    on whether the dispatcher was invoked."""
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    task_shape = {
        "agent_type": "content-generator",
        "params": {"project_id": "P", "course_name": "C"},
    }

    for truthy in ("1", "true", "TRUE", "True", "yes", "Yes", "on", "ON"):
        monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", truthy)
        dispatcher.calls.clear()
        await ex._invoke_tool("generate_course_content", task_shape)
        assert len(dispatcher.calls) == 1, f"flag={truthy!r} should enable dispatch"

    for falsy in ("", "0", "false", "no", "off", "sometimes"):
        monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", falsy)
        dispatcher.calls.clear()
        await ex._invoke_tool("generate_course_content", task_shape)
        assert dispatcher.calls == [], f"flag={falsy!r} should disable dispatch"


@pytest.mark.asyncio
async def test_subagent_set_covers_core_reasoning_agents():
    """Pin the AGENT_SUBAGENT_SET membership so a future refactor can't
    silently drop content-generator or assessment-generator — the two
    agents whose in-process templates caused the empty-KG regression
    that motivated Wave 74."""
    must_be_subagent = {
        "content-generator",
        "assessment-generator",
        "content-analyzer",
        "accessibility-remediation",
        "content-quality-remediation",
        "intelligent-design-mapper",
    }
    missing = must_be_subagent - AGENT_SUBAGENT_SET
    assert missing == set(), (
        f"AGENT_SUBAGENT_SET regression — missing: {sorted(missing)}"
    )

    must_NOT_be_subagent = {
        "dart-converter",
        "dart-automation-coordinator",
        "imscc-intake-parser",
        "brightspace-packager",
        "source-router",
        "libv2-archivist",
        "textbook-stager",
    }
    leaked = must_NOT_be_subagent & AGENT_SUBAGENT_SET
    assert leaked == set(), (
        f"Python-tool agents leaked into AGENT_SUBAGENT_SET: {sorted(leaked)}"
    )


@pytest.mark.asyncio
async def test_trainforge_synthesis_provider_set_bypasses_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Wave1-I1 regression: with ``TRAINFORGE_SYNTHESIS_PROVIDER=local``
    set (and ``ED4ALL_AGENT_DISPATCH=true`` + the dispatcher wired +
    ``agent_type == "training-synthesizer"`` — i.e. every condition
    that would normally route to the Claude Code subagent), the
    executor MUST fall through to the in-process registry tool instead
    of calling ``dispatcher.dispatch_task``.

    The training-synthesizer surface authors the canonical SLM
    training-pair corpus (instruction + preference pairs); per
    ``docs/LICENSING.md`` § "Synthesis providers" Claude must NEVER
    author this content — only operator-selected license-clean
    providers may. Pre-Wave1-I1 the only safety was operator-discipline
    via ``--skip-training``; this short-circuit makes the guard
    fail-loud at the dispatcher fork.

    Pin the routing invariant so a future refactor can't silently
    re-enable Claude authorship of training-pair corpus content.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_PROVIDER", "local")
    # Clear sibling provider envs so this test's training-synthesizer
    # short-circuit is the one being exercised, not a stale env.
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("TRAINFORGE_ASSESSMENT_PROVIDER", raising=False)

    # Pre-condition pin: training-synthesizer must be in the subagent
    # set or the short-circuit would never matter (the dispatcher
    # branch wouldn't fire to begin with).
    assert "training-synthesizer" in AGENT_SUBAGENT_SET

    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "synthesize_training",
        {
            "agent_type": "training-synthesizer",
            "params": {
                "course_code": "FXALPHA_101",
                "corpus_dir": "/tmp/fake",
            },
        },
    )

    # Fell through to the in-process tool — dispatcher untouched.
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == [], (
        "TRAINFORGE_SYNTHESIS_PROVIDER short-circuit regression — "
        "training-synthesizer must NOT route through "
        "LocalDispatcher.dispatch_task (Claude subagent path) when "
        "the env var is set. Wave1-I1."
    )


@pytest.mark.asyncio
async def test_trainforge_synthesis_provider_unset_preserves_dispatch(
    monkeypatch, state_runs_isolated,
):
    """Wave1-I1 backward-compat regression: with
    ``TRAINFORGE_SYNTHESIS_PROVIDER`` unset, the dispatcher still
    fires for training-synthesizer when ``ED4ALL_AGENT_DISPATCH=true``.
    Pre-Wave1-I1 behavior byte-stable so existing runs aren't
    perturbed by the short-circuit landing.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_PROVIDER", raising=False)
    dispatcher = DummyDispatcher(
        response={
            "success": True,
            "dispatch_mode": "dummy",
            "outputs": {"pairs_generated": 42},
            "artifacts": [],
        },
    )
    ex = _make_executor(dispatcher=dispatcher)

    result = await ex._invoke_tool(
        "synthesize_training",
        {
            "agent_type": "training-synthesizer",
            "params": {
                "course_code": "FXALPHA_101",
                "corpus_dir": "/tmp/fake",
            },
        },
    )
    # Dispatched as a subagent (legacy Wave-74 path).
    assert result["dispatch_mode"] == "dummy"
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["agent_type"] == "training-synthesizer"


@pytest.mark.asyncio
async def test_missing_agent_type_in_task_params_falls_through(monkeypatch, state_runs_isolated):
    """Legacy task shapes (pre-Wave-74 workflows, MCP-tool callers)
    may lack an ``agent_type`` field. The fork must not crash on that
    and must fall through to the in-process path."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    dispatcher = DummyDispatcher()
    ex = _make_executor(dispatcher=dispatcher)

    # No agent_type key at all.
    result = await ex._invoke_tool(
        "get_courseforge_status",
        {"project_id": "P"},
    )
    assert result["dispatch_mode"] == "in_process"
    assert dispatcher.calls == []


def test_agent_provider_env_map_pinned_shape():
    """Wave1-I8 constant-lift: pin the agent → env-var mapping shape.

    The map is the single source of truth consumed by both the
    executor's short-circuit checks (``_dispatch_task_via_tool``) and
    the workflow runner's provider banner. Drift between this map
    and either consumer is a silent regression — drift between this
    map and the executor's inline checks reverts the lift; drift
    between this map and the banner produces misleading operator
    output.
    """
    assert AGENT_PROVIDER_ENV_MAP["content-generator"] == "COURSEFORGE_PROVIDER"
    assert AGENT_PROVIDER_ENV_MAP["course-outliner"] == "COURSEPLANNER_PROVIDER"
    assert (
        AGENT_PROVIDER_ENV_MAP["assessment-generator"]
        == "TRAINFORGE_ASSESSMENT_PROVIDER"
    )
    # Every agent with a provider env var MUST also be in
    # AGENT_SUBAGENT_SET — otherwise the executor's short-circuit
    # would be unreachable (the inline check sits *before* the
    # subagent dispatch but only matters when subagent dispatch
    # would otherwise fire).
    for agent in AGENT_PROVIDER_ENV_MAP:
        assert agent in AGENT_SUBAGENT_SET, (
            f"{agent} is in AGENT_PROVIDER_ENV_MAP but not in "
            f"AGENT_SUBAGENT_SET — the short-circuit is unreachable."
        )
