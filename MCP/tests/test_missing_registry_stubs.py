"""Verify the 7 previously-missing agent-tool mappings now resolve.

MCP audit (`plans/pipeline-remediation/mcp-audit.md` § Q1) flagged 6
distinct tool names + `convert_pdf_multi_source` (Q6) as being mapped in
``MCP.core.executor.AGENT_TOOL_MAPPING`` while absent from
``MCP.tools.pipeline_tools._build_tool_registry()``. Without a registry
stub, every agent routed to one of these tools would fail with
``Tool not registered: X`` (the same failure mode PR #45 fixed for
``extract_and_convert_pdf``).

This test validates the post-remediation state:

1. Every expected tool name exists in ``_build_tool_registry()``.
2. Each registered tool is a callable async coroutine function.
3. Each tool's arity accepts ``**kwargs`` so the TaskExecutor's
   parameter mapper can invoke it uniformly.

The test does NOT execute the tools against live filesystem state —
that's covered by the tool-specific tests in the same directory. This is
a wiring / shape assertion only.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

# These are the 6 tools listed explicitly in the audit report as
# "mapped but missing from runtime registry", plus convert_pdf_multi_source
# which Q6 flagged as @mcp.tool()-only (and is a natural DART entry point).
EXPECTED_NEW_TOOLS = [
    "get_courseforge_status",
    "validate_wcag_compliance",
    "batch_convert_multi_source",
    "intake_imscc_package",
    "remediate_course_content",
    "validate_assessment",
    "convert_pdf_multi_source",
]


@pytest.fixture(scope="module")
def registry():
    return _build_tool_registry()


class TestMissingRegistryStubsPresent:
    """Each expected tool must be keyed into the registry."""

    @pytest.mark.parametrize("tool_name", EXPECTED_NEW_TOOLS)
    def test_tool_present_in_registry(self, registry, tool_name):
        assert tool_name in registry, (
            f"Runtime registry is missing '{tool_name}'. Agent dispatch "
            f"through MCP.core.executor.AGENT_TOOL_MAPPING will fail."
        )

    def test_registry_has_all_seven(self, registry):
        missing = [t for t in EXPECTED_NEW_TOOLS if t not in registry]
        assert not missing, (
            f"Missing registry stubs: {missing}. "
            f"AGENT_TOOL_MAPPING routes agents to these names."
        )


class TestRegistryStubsCallableAsync:
    """Every stub must be an async callable accepting **kwargs."""

    @pytest.mark.parametrize("tool_name", EXPECTED_NEW_TOOLS)
    def test_tool_is_async_callable(self, registry, tool_name):
        tool = registry.get(tool_name)
        assert tool is not None
        assert callable(tool), f"{tool_name} is not callable"
        assert inspect.iscoroutinefunction(tool), (
            f"{tool_name} must be an async def so TaskExecutor can await it"
        )

    @pytest.mark.parametrize("tool_name", EXPECTED_NEW_TOOLS)
    def test_tool_accepts_kwargs(self, registry, tool_name):
        tool = registry[tool_name]
        sig = inspect.signature(tool)
        # Must accept **kwargs (the TaskExecutor's invocation convention)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_var_kw, (
            f"{tool_name} must accept **kwargs — the TaskExecutor passes "
            f"mapped params via keyword expansion."
        )


class TestExecutorMappingResolves:
    """Every agent in AGENT_TOOL_MAPPING must resolve to a registered tool."""

    def test_all_agent_mappings_have_registry_entries(self, registry):
        from MCP.core.executor import AGENT_TOOL_MAPPING
        unresolved = []
        for agent, tool_name in AGENT_TOOL_MAPPING.items():
            if tool_name not in registry:
                unresolved.append((agent, tool_name))
        assert not unresolved, (
            f"Agent→tool mappings referencing unregistered tools: "
            f"{unresolved}"
        )


class TestRegistryStubInvocationDoesNotRaise:
    """Smoke test: invoking each stub with empty kwargs must not raise.

    The stubs delegate to the @mcp.tool() implementations. We pass no
    required params so most will return a structured error (JSON string
    with "error" key) — the point of this test is that the WRAPPER
    (closure layer) is intact and doesn't raise at the registry boundary.
    """

    @pytest.mark.parametrize("tool_name", EXPECTED_NEW_TOOLS)
    def test_stub_returns_without_raising(self, registry, tool_name):
        tool = registry[tool_name]
        # Call with empty kwargs — delegate should handle missing params
        # by returning an error JSON, not by raising.
        try:
            result = asyncio.run(tool())
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"{tool_name} raised at the registry wrapper boundary: "
                f"{type(exc).__name__}: {exc}. Stubs must return "
                f"structured error JSON rather than propagating."
            )
        # Any string return is acceptable — most will be JSON error blobs.
        assert isinstance(result, str)
