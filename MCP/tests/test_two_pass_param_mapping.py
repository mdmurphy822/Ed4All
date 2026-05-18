"""Regression test: the four two-pass router tools must have
``"id" -> "project_id"`` and ``"project" -> "project_id"`` aliases in their
``param_mapping`` so that ``WorkflowRunner`` dispatching with ``{"id": ...}``
does not raise ``ParameterMappingError: Missing required parameters: ['project_id']``.

This would have caught the Wave-fix regression where the four schemas were
registered with empty ``param_mapping: {}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.tool_schemas import TOOL_SCHEMAS  # noqa: E402
from MCP.core.param_mapper import TaskParameterMapper  # noqa: E402

TWO_PASS_TOOLS = [
    "run_content_generation_outline",
    "run_inter_tier_validation",
    "run_content_generation_rewrite",
    "run_post_rewrite_validation",
]


@pytest.mark.parametrize("tool_name", TWO_PASS_TOOLS)
def test_two_pass_tool_has_id_alias(tool_name):
    """Each two-pass router schema must alias ``id`` -> ``project_id``."""
    schema = TOOL_SCHEMAS.get(tool_name)
    assert schema is not None, f"{tool_name} is missing from TOOL_SCHEMAS"
    mapping = schema.get("param_mapping", {})
    assert mapping.get("id") == "project_id", (
        f"{tool_name}: param_mapping missing 'id' -> 'project_id'. "
        f"WorkflowRunner dispatches with key 'id'; without this alias "
        f"TaskParameterMapper raises ParameterMappingError."
    )


@pytest.mark.parametrize("tool_name", TWO_PASS_TOOLS)
def test_two_pass_tool_has_project_alias(tool_name):
    """Each two-pass router schema must alias ``project`` -> ``project_id``."""
    schema = TOOL_SCHEMAS.get(tool_name)
    assert schema is not None, f"{tool_name} is missing from TOOL_SCHEMAS"
    mapping = schema.get("param_mapping", {})
    assert mapping.get("project") == "project_id", (
        f"{tool_name}: param_mapping missing 'project' -> 'project_id'."
    )


@pytest.mark.parametrize("tool_name", TWO_PASS_TOOLS)
def test_two_pass_param_mapper_resolves_id(tool_name):
    """``TaskParameterMapper`` must map ``id`` to ``project_id`` for each tool.

    Simulates what WorkflowRunner does when dispatching a phase task:
    passes ``{"id": "<project_id>", ...}`` and expects ``project_id`` in
    the resolved output.
    """
    mapper = TaskParameterMapper(strict=False)
    # Build a task dict that mirrors what WorkflowRunner passes: top-level
    # ``id`` key + a ``params`` sub-dict.  Provide extra keys so every tool's
    # required params are satisfiable without unrelated mapping errors.
    task_dict = {
        "params": {
            "id": "PROJ-TEST-001",
            "blocks_outline_path": "/tmp/outline.jsonl",
            "blocks_validated_path": "/tmp/validated.jsonl",
            "blocks_final_path": "/tmp/final.jsonl",
        }
    }
    result = mapper.map_task_to_tool_params(task_dict, tool_name)
    assert result.get("project_id") == "PROJ-TEST-001", (
        f"{tool_name}: expected project_id='PROJ-TEST-001' after mapping, "
        f"got result={result}"
    )
