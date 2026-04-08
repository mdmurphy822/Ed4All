"""
Task Parameter Mapper

Maps generic task parameters to tool-specific parameter signatures.
This bridges the gap between the executor's task format and
the actual MCP tool function signatures.

Task Format:
    {
        "prompt": "...",           # Optional prompt/instructions
        "params": {...},           # Task-specific parameters
        "context": {...},          # Additional context
        "input": "...",            # Common shorthand for input path
        "output": "...",           # Common shorthand for output path
    }

Tool Format:
    tool_func(pdf_path="...", output_dir="...", options=None)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .tool_schemas import (
    TOOL_SCHEMAS,
    get_tool_schema,
    get_required_params,
    get_optional_params,
    get_param_mapping,
    get_defaults,
    validate_tool_params,
)

logger = logging.getLogger(__name__)


class ParameterMappingError(Exception):
    """Raised when parameter mapping fails."""
    pass


class TaskParameterMapper:
    """
    Maps task parameters to tool-specific function signatures.

    This class handles the translation between the generic task format
    used by the orchestrator and the specific parameter names expected
    by each MCP tool.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the parameter mapper.

        Args:
            strict: If True, raise errors for unmapped params.
                   If False, pass unmapped params through as-is.
        """
        self.strict = strict

    def map_task_to_tool_params(
        self,
        task: Dict[str, Any],
        tool_name: str,
    ) -> Dict[str, Any]:
        """
        Convert task dict to tool kwargs.

        This method:
        1. Extracts parameters from the task dict
        2. Applies parameter name mappings
        3. Fills in defaults for optional parameters
        4. Validates required parameters are present

        Args:
            task: Task dict with prompt, params, context, etc.
            tool_name: Name of the target tool

        Returns:
            Dict of kwargs suitable for calling the tool

        Raises:
            ParameterMappingError: If required params are missing or tool unknown
        """
        schema = get_tool_schema(tool_name)
        if not schema:
            raise ParameterMappingError(f"Unknown tool: {tool_name}")

        param_mapping = get_param_mapping(tool_name)
        defaults = get_defaults(tool_name)
        required = get_required_params(tool_name)
        optional = get_optional_params(tool_name)

        # Collect all possible parameters from task
        # Priority: task.params > task top-level > defaults
        all_task_params = {}

        # Start with top-level task fields (like "input", "output")
        for key in ["input", "output", "source", "target", "path", "id"]:
            if key in task:
                all_task_params[key] = task[key]

        # Add context fields
        context = task.get("context", {})
        if isinstance(context, dict):
            all_task_params.update(context)

        # Override with explicit params
        params = task.get("params", {})
        if isinstance(params, dict):
            all_task_params.update(params)

        # Now map to tool-specific parameter names
        tool_params = {}

        # Apply mapping transformations
        for task_key, task_value in all_task_params.items():
            if task_key in param_mapping:
                # This task key maps to a different tool key
                tool_key = param_mapping[task_key]
                tool_params[tool_key] = task_value
            elif task_key in required or task_key in optional:
                # This key is already a valid tool param name
                tool_params[task_key] = task_value
            elif not self.strict:
                # Pass through unmapped params in non-strict mode
                tool_params[task_key] = task_value
            # In strict mode, unmapped params are dropped

        # Apply defaults for missing optional parameters
        for opt_param in optional:
            if opt_param not in tool_params and opt_param in defaults:
                tool_params[opt_param] = defaults[opt_param]

        # Validate required parameters
        missing = []
        for req_param in required:
            if req_param not in tool_params or tool_params[req_param] is None:
                missing.append(req_param)

        if missing:
            raise ParameterMappingError(
                f"Missing required parameters for {tool_name}: {missing}. "
                f"Received params: {list(tool_params.keys())}"
            )

        # Filter to only expected parameters
        all_expected = set(required) | set(optional)
        if self.strict:
            # In strict mode, identify and warn about dropped params
            dropped_params = {
                k: v for k, v in tool_params.items()
                if k not in all_expected
            }
            if dropped_params:
                logger.warning(
                    f"Dropped {len(dropped_params)} unknown params for {tool_name}: "
                    f"{list(dropped_params.keys())}. "
                    f"Expected: {sorted(all_expected)}"
                )
            tool_params = {
                k: v for k, v in tool_params.items()
                if k in all_expected
            }
        else:
            # In non-strict mode, warn about unmapped params being passed through
            unmapped_params = {
                k for k in tool_params.keys()
                if k not in all_expected
            }
            if unmapped_params:
                logger.info(
                    f"Passing through {len(unmapped_params)} unmapped params for {tool_name}: "
                    f"{sorted(unmapped_params)}. Consider adding to tool schema."
                )

        logger.debug(
            f"Mapped task to {tool_name}: {list(tool_params.keys())}"
        )

        return tool_params

    def extract_prompt(self, task: Dict[str, Any]) -> str:
        """
        Extract the prompt/instructions from a task.

        Args:
            task: Task dict

        Returns:
            Prompt string (empty if not found)
        """
        return task.get("prompt", "")

    def create_task_from_params(
        self,
        tool_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a task dict from tool-specific parameters.

        This is the inverse of map_task_to_tool_params - useful for
        creating tasks programmatically.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool-specific parameters

        Returns:
            Task dict suitable for the orchestrator
        """
        schema = get_tool_schema(tool_name)
        if not schema:
            raise ParameterMappingError(f"Unknown tool: {tool_name}")

        required = get_required_params(tool_name)
        optional = get_optional_params(tool_name)

        # Validate required params
        for req in required:
            if req not in kwargs:
                raise ParameterMappingError(
                    f"Missing required parameter for {tool_name}: {req}"
                )

        # Build task
        return {
            "params": kwargs,
            "tool_name": tool_name,
        }

    def validate_task_params(
        self,
        task: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate task parameters without mapping.

        Args:
            task: Task dict
            tool_name: Name of the tool

        Returns:
            Tuple of (is_valid, list of issues)
        """
        try:
            mapped = self.map_task_to_tool_params(task, tool_name)
            return True, []
        except ParameterMappingError as e:
            return False, [str(e)]

    def get_expected_params(self, tool_name: str) -> Dict[str, Any]:
        """
        Get information about expected parameters for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict with required, optional, and defaults
        """
        schema = get_tool_schema(tool_name)
        if not schema:
            return {}

        return {
            "required": get_required_params(tool_name),
            "optional": get_optional_params(tool_name),
            "defaults": get_defaults(tool_name),
            "mapping": get_param_mapping(tool_name),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_mapper = TaskParameterMapper(strict=False)


def map_params(task: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Convenience function to map task params to tool params.

    Args:
        task: Task dict
        tool_name: Target tool name

    Returns:
        Mapped parameters
    """
    return _default_mapper.map_task_to_tool_params(task, tool_name)


def validate_params(task: Dict[str, Any], tool_name: str) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate task params.

    Args:
        task: Task dict
        tool_name: Target tool name

    Returns:
        Tuple of (is_valid, issues)
    """
    return _default_mapper.validate_task_params(task, tool_name)


# =============================================================================
# LEGACY SUPPORT
# =============================================================================

def extract_legacy_params(
    prompt: str,
    params: Dict[str, Any],
    tool_name: str,
) -> Dict[str, Any]:
    """
    Support the old calling convention by wrapping in task format.

    Old convention:
        tool(prompt=prompt, **params)

    New convention:
        tool(**mapped_params)

    Args:
        prompt: The prompt string from old convention
        params: The params dict from old convention
        tool_name: Target tool name

    Returns:
        Mapped parameters suitable for new calling convention
    """
    task = {
        "prompt": prompt,
        "params": params,
    }
    return map_params(task, tool_name)
