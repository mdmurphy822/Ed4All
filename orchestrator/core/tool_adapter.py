"""
Tool Adapter Layer

Provides a thin abstraction for tool invocation with support for:
- Parameter transformation
- Pre/post processing hooks
- Consistent error handling
- Logging and metrics

Usage:
    from orchestrator.core.tool_adapter import ToolAdapterRegistry

    registry = ToolAdapterRegistry()
    result = await registry.invoke("convert_pdf_multi_source", params)

Architecture:
    Task → ToolAdapter → MCP Tool → Result
                ↓
         Transform params
         Add logging
         Handle errors
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type

from .param_mapper import TaskParameterMapper, ParameterMappingError

logger = logging.getLogger(__name__)


# =============================================================================
# BASE ADAPTER
# =============================================================================

class ToolAdapter(ABC):
    """
    Base class for tool adapters.

    Adapters handle the translation between orchestrator task format
    and the specific parameter requirements of each MCP tool.

    Subclass this for tools that need custom parameter transformation
    or pre/post processing.
    """

    def __init__(self, tool_name: str):
        """
        Initialize adapter.

        Args:
            tool_name: Name of the MCP tool this adapter handles
        """
        self.tool_name = tool_name
        self.param_mapper = TaskParameterMapper(strict=False)

    @abstractmethod
    async def adapt(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transform task parameters to tool parameters.

        Args:
            params: Task parameters from orchestrator
            context: Additional context (workflow_id, run_id, etc.)

        Returns:
            Parameters suitable for MCP tool invocation
        """
        pass

    def pre_invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before tool invocation.

        Override to add validation, logging, or transformation.

        Args:
            params: Transformed parameters

        Returns:
            Modified parameters
        """
        return params

    def post_invoke(
        self,
        result: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Hook called after successful tool invocation.

        Override to add result transformation or logging.

        Args:
            result: Tool execution result
            params: Parameters that were used

        Returns:
            Modified result
        """
        return result

    def on_error(
        self,
        error: Exception,
        params: Dict[str, Any],
    ) -> None:
        """
        Hook called when tool invocation fails.

        Override to add custom error handling or logging.

        Args:
            error: The exception that occurred
            params: Parameters that were used
        """
        logger.error(f"Tool {self.tool_name} failed: {error}")


# =============================================================================
# DEFAULT ADAPTER
# =============================================================================

class DefaultToolAdapter(ToolAdapter):
    """
    Default adapter that uses TaskParameterMapper for transformation.

    This adapter handles most tools that follow standard parameter
    conventions. Use this unless you need custom transformation.
    """

    async def adapt(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transform using standard parameter mapping.

        Args:
            params: Task parameters
            context: Additional context

        Returns:
            Mapped parameters for MCP tool
        """
        task = {
            "params": params,
            "context": context or {},
        }

        try:
            mapped = self.param_mapper.map_task_to_tool_params(task, self.tool_name)
            logger.debug(f"Mapped params for {self.tool_name}: {list(mapped.keys())}")
            return mapped
        except ParameterMappingError as e:
            logger.warning(f"Parameter mapping failed for {self.tool_name}: {e}")
            # Fall back to passing params through
            return params


# =============================================================================
# SPECIALIZED ADAPTERS
# =============================================================================

class DARTConversionAdapter(ToolAdapter):
    """
    Adapter for DART PDF conversion tools.

    Handles the specific parameter requirements for multi-source
    PDF conversion including path validation and option defaults.
    """

    async def adapt(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transform parameters for DART conversion."""
        context = context or {}

        # Map common parameter names
        adapted = {}

        # Input handling - support multiple input param names
        for input_key in ['pdf_path', 'input', 'source', 'combined_json_path']:
            if input_key in params:
                adapted['combined_json_path'] = params[input_key]
                break

        # Output handling
        for output_key in ['output_path', 'output', 'target', 'output_dir']:
            if output_key in params:
                adapted['output_path'] = params[output_key]
                break

        # Options with defaults
        adapted['options'] = params.get('options', {})
        if 'accessibility_level' not in adapted['options']:
            adapted['options']['accessibility_level'] = 'AA'

        # Pass through run_id if available
        if 'run_id' in context:
            adapted['run_id'] = context['run_id']

        return adapted

    def pre_invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate paths before invocation."""
        from lib.secure_paths import validate_path_within_root
        from lib.paths import PROJECT_ROOT

        if 'combined_json_path' in params:
            validate_path_within_root(params['combined_json_path'], PROJECT_ROOT)
        if 'output_path' in params:
            validate_path_within_root(params['output_path'], PROJECT_ROOT)

        return params


class CourseforgeAdapter(ToolAdapter):
    """
    Adapter for Courseforge content generation tools.

    Handles course code validation and output path construction.
    """

    async def adapt(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transform parameters for Courseforge tools."""
        context = context or {}
        adapted = dict(params)

        # Ensure course_code is present
        if 'course_code' not in adapted and 'course_id' in adapted:
            adapted['course_code'] = adapted.pop('course_id')

        # Add workflow context
        if 'workflow_id' in context:
            adapted['workflow_id'] = context['workflow_id']

        return adapted


class TrainforgeAdapter(ToolAdapter):
    """
    Adapter for Trainforge assessment generation tools.

    Handles RAG context and quality settings.
    """

    async def adapt(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transform parameters for Trainforge tools."""
        adapted = dict(params)

        # Default quality settings
        if 'min_quality' not in adapted:
            adapted['min_quality'] = 'proficient'

        # Default bloom's levels if generating assessments
        if 'bloom_levels' not in adapted:
            adapted['bloom_levels'] = ['understand', 'apply', 'analyze']

        return adapted


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

class ToolAdapterRegistry:
    """
    Registry for tool adapters.

    Maintains mapping from tool names to their adapters and handles
    adapter creation and caching.

    Usage:
        registry = ToolAdapterRegistry()
        registry.register("my_tool", MyCustomAdapter)
        result = await registry.invoke("my_tool", params, invoke_func)
    """

    # Default adapter mappings
    DEFAULT_ADAPTERS: Dict[str, Type[ToolAdapter]] = {
        "convert_pdf_multi_source": DARTConversionAdapter,
        "batch_convert_multi_source": DARTConversionAdapter,
        "create_course_project": CourseforgeAdapter,
        "generate_course_content": CourseforgeAdapter,
        "package_imscc": CourseforgeAdapter,
        "generate_assessments": TrainforgeAdapter,
        "validate_assessment": TrainforgeAdapter,
    }

    def __init__(self):
        """Initialize the registry."""
        self._adapters: Dict[str, ToolAdapter] = {}
        self._custom_mappings: Dict[str, Type[ToolAdapter]] = {}

    def register(
        self,
        tool_name: str,
        adapter_class: Type[ToolAdapter],
    ) -> None:
        """
        Register a custom adapter for a tool.

        Args:
            tool_name: Name of the MCP tool
            adapter_class: Adapter class to use
        """
        self._custom_mappings[tool_name] = adapter_class
        # Clear cached instance if exists
        if tool_name in self._adapters:
            del self._adapters[tool_name]

    def get_adapter(self, tool_name: str) -> ToolAdapter:
        """
        Get adapter for a tool (creates if needed).

        Args:
            tool_name: Name of the MCP tool

        Returns:
            ToolAdapter instance for the tool
        """
        if tool_name not in self._adapters:
            # Check custom mappings first, then defaults
            if tool_name in self._custom_mappings:
                adapter_class = self._custom_mappings[tool_name]
            elif tool_name in self.DEFAULT_ADAPTERS:
                adapter_class = self.DEFAULT_ADAPTERS[tool_name]
            else:
                adapter_class = DefaultToolAdapter

            self._adapters[tool_name] = adapter_class(tool_name)

        return self._adapters[tool_name]

    async def invoke(
        self,
        tool_name: str,
        params: Dict[str, Any],
        invoke_func: Callable[..., Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Invoke a tool through its adapter.

        Args:
            tool_name: Name of the MCP tool
            params: Task parameters
            invoke_func: Function to call the actual MCP tool
            context: Additional context

        Returns:
            Tool execution result

        Raises:
            Exception: If tool invocation fails
        """
        adapter = self.get_adapter(tool_name)

        # Transform parameters
        adapted_params = await adapter.adapt(params, context)

        # Pre-invoke hook
        adapted_params = adapter.pre_invoke(adapted_params)

        # Log invocation
        start_time = datetime.now()
        logger.info(f"Invoking {tool_name} with {len(adapted_params)} params")

        try:
            # Call the actual tool
            result = await invoke_func(**adapted_params)

            # Post-invoke hook
            result = adapter.post_invoke(result, adapted_params)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Tool {tool_name} completed in {duration:.2f}s")

            return result

        except Exception as e:
            adapter.on_error(e, adapted_params)
            raise

    def list_adapters(self) -> Dict[str, str]:
        """
        List all registered adapters.

        Returns:
            Dict mapping tool names to adapter class names
        """
        all_adapters = {}

        # Add defaults
        for tool, adapter_class in self.DEFAULT_ADAPTERS.items():
            all_adapters[tool] = adapter_class.__name__

        # Override with custom
        for tool, adapter_class in self._custom_mappings.items():
            all_adapters[tool] = adapter_class.__name__

        return all_adapters


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance
_default_registry: Optional[ToolAdapterRegistry] = None


def get_registry() -> ToolAdapterRegistry:
    """Get the default adapter registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolAdapterRegistry()
    return _default_registry


def get_adapter(tool_name: str) -> ToolAdapter:
    """Get adapter for a tool from the default registry."""
    return get_registry().get_adapter(tool_name)


async def invoke_tool(
    tool_name: str,
    params: Dict[str, Any],
    invoke_func: Callable[..., Any],
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Invoke a tool through the default registry.

    Args:
        tool_name: Name of the MCP tool
        params: Task parameters
        invoke_func: Function to call the actual tool
        context: Additional context

    Returns:
        Tool execution result
    """
    return await get_registry().invoke(tool_name, params, invoke_func, context)
