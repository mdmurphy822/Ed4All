"""Wave1-I8 — provider-banner regression tests.

The workflow runner emits one log line per entry in
``AGENT_PROVIDER_ENV_MAP`` at workflow start so operators can see at a
glance how each subagent-classified agent will dispatch:

1. ``local-provider`` — the agent's provider env var (e.g.
   ``COURSEFORGE_PROVIDER``) is set; the executor short-circuits the
   Wave-74 subagent dispatch and routes through the in-process
   Wave-D provider.
2. ``subagent (claude)`` — ``ED4ALL_AGENT_DISPATCH=true`` is set but
   no provider env var is; the executor dispatches to the Claude
   Code subagent.
3. ``in-process-stub`` — neither is set; the executor falls through
   to the in-process tool registry (test/dry-run path).

Pure observability — these tests pin the log format + precedence
order without exercising any phase. They go through
``WorkflowRunner._emit_provider_banner`` directly because the banner
is a one-shot helper invoked once per ``run_workflow`` call.

Finding 7 of plans/dispatch-7-execution-inspection-2026-05.md.
"""
from __future__ import annotations

import logging
from typing import Mapping
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import AGENT_PROVIDER_ENV_MAP
from MCP.core.workflow_runner import WorkflowRunner


def _make_runner() -> WorkflowRunner:
    """Build a minimal ``WorkflowRunner`` for banner-only tests.

    The banner helper depends on neither ``executor`` nor ``config``,
    so stub references are sufficient.
    """
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _banner_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return ``[provider-banner]`` log records' messages, ordered."""
    return [
        record.getMessage()
        for record in caplog.records
        if "[provider-banner]" in record.getMessage()
    ]


def _assert_one_line_per_agent(
    lines: list[str], expected_map: Mapping[str, str]
) -> None:
    """Pin one line per ``AGENT_PROVIDER_ENV_MAP`` entry, no duplicates."""
    assert len(lines) == len(expected_map), (
        f"Expected {len(expected_map)} banner lines (one per agent in "
        f"AGENT_PROVIDER_ENV_MAP), got {len(lines)}: {lines!r}"
    )
    for agent in expected_map:
        agent_lines = [line for line in lines if f" {agent}:" in line]
        assert len(agent_lines) == 1, (
            f"Expected exactly one banner line for agent {agent!r}, got "
            f"{agent_lines!r}"
        )


def test_banner_all_in_process_stub_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No env vars set → every agent banner reads ``in-process-stub``."""
    # Clear all provider env vars + the dispatch flag.
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "in-process-stub" in line, line
        assert "ED4ALL_AGENT_DISPATCH=false" in line, line
        # Defence-in-depth: no spurious local-provider / subagent strings.
        assert "local-provider" not in line, line
        assert "subagent" not in line, line


def test_banner_all_subagent_when_only_agent_dispatch_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ED4ALL_AGENT_DISPATCH=true`` only → all show ``subagent (claude)``."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line
        assert "in-process-stub" not in line, line
        assert "local-provider" not in line, line


def test_banner_local_provider_takes_precedence_over_subagent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider env var wins over ``ED4ALL_AGENT_DISPATCH=true``.

    Mirrors the executor's short-circuit precedence: when both are
    set, the in-process provider runs (Wave-D ToS unblock) and the
    subagent dispatch is bypassed for that agent only — other agents
    in the map still log ``subagent (claude)``.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "together")
    # Clear the other provider env vars so they fall through to subagent.
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)

    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    assert "local-provider" in content_gen_lines[0]
    assert "COURSEFORGE_PROVIDER=together" in content_gen_lines[0]

    other_lines = [
        line for line in lines if " content-generator:" not in line
    ]
    for line in other_lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line


def test_banner_empty_or_whitespace_env_var_does_not_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whitespace-only provider env var must NOT flip the banner.

    Mirrors the executor's ``.strip()`` guard inside
    ``_dispatch_task_via_tool`` — an empty / whitespace env var falls
    through to the subagent / stub path. The banner must report the
    same.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    # Whitespace-only — should be treated as unset.
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "   ")
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    # Should NOT have flipped to local-provider on whitespace.
    assert "local-provider" not in content_gen_lines[0]
    assert "subagent (claude)" in content_gen_lines[0]
