"""Regression test: every tool in _build_tool_registry() must have a
TOOL_SCHEMAS entry.

Without a schema entry, ``TaskParameterMapper.map_task_to_tool_params``
raises ``ParameterMappingError("Unknown tool: <name>")`` before the
executor can reach the registry, causing the phase to exhaust its
3-retry cap and halt the workflow.

This test would have caught the Phase 3 (chunking) regression where
``run_dart_chunking``, ``run_imscc_chunking``, and
``run_concept_extraction`` were registered in ``_build_tool_registry()``
but absent from ``TOOL_SCHEMAS``.

Invariant: registry_keys ⊆ schema_keys
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.tool_schemas import TOOL_SCHEMAS  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return _build_tool_registry()


@pytest.fixture(scope="module")
def schema_keys():
    return set(TOOL_SCHEMAS.keys())


class TestToolRegistrySchemaParity:
    """Every runtime tool registered by _build_tool_registry() must have a
    corresponding TOOL_SCHEMAS entry so the param-mapper can resolve it."""

    def test_registry_is_subset_of_schemas(self, registry, schema_keys):
        registry_keys = set(registry.keys())
        missing = registry_keys - schema_keys
        assert not missing, (
            "The following tools are registered in _build_tool_registry() "
            "but have NO entry in TOOL_SCHEMAS. "
            "TaskParameterMapper will raise ParameterMappingError('Unknown "
            "tool: <name>') on every dispatch, exhausting the 3-retry cap "
            "and halting any phase that invokes them.\n\n"
            f"Missing schema entries: {sorted(missing)}\n\n"
            "Fix: add a TOOL_SCHEMAS entry to MCP/core/tool_schemas.py "
            "for each tool above (required/optional/param_mapping/description)."
        )

    @pytest.mark.parametrize(
        "tool_name",
        [
            "run_dart_chunking",
            "run_imscc_chunking",
            "run_concept_extraction",
        ],
    )
    def test_known_regression_tools_have_schemas(self, schema_keys, tool_name):
        """Pinned regression guard for the three tools that caused Phase 3
        failures before this fix. Separate from the set-subset test so a
        future registry refactor that accidentally removes these names
        surfaces a named failure rather than a silent pass."""
        assert tool_name in schema_keys, (
            f"'{tool_name}' is missing from TOOL_SCHEMAS — "
            "this is the exact regression that blocked Phase 3 (chunking) "
            "in the textbook-to-course smoke test."
        )

    def test_known_regression_tools_in_registry(self, registry):
        """Confirm all three Phase 3 regression tools remain in the registry."""
        missing = [
            t for t in ("run_dart_chunking", "run_imscc_chunking", "run_concept_extraction")
            if t not in registry
        ]
        assert not missing, (
            f"Registry missing expected Phase 3 tools: {missing}"
        )
