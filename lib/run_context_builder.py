"""
Run Context Builder - Builder Pattern for RunContext Initialization

Provides a fluent builder interface for creating fully-wired RunContext
instances with all optional components initialized.

Phase 0.5 Enhancement: Builder Pattern for RunContext (C2)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .run_manager import (
    HARDENED_MODE,
    RUNS_PATH,
    RunContext,
    RunManager,
)

if TYPE_CHECKING:
    from .audit_logger import AuditLogger
    from .sequence_manager import SequenceManager


# ============================================================================
# RUN CONTEXT BUILDER
# ============================================================================

class RunContextBuilder:
    """
    Builder pattern for complete RunContext initialization.

    Provides fluent interface for configuring all RunContext components
    before building the final context.

    Usage:
        context = (
            RunContextBuilder()
            .with_workflow_type("course_generation")
            .with_sequence_manager()
            .with_audit_logger()
            .with_registry_snapshot()
            .build()
        )
    """

    def __init__(
        self,
        runs_path: Path = RUNS_PATH,
        hardened_mode: Optional[bool] = None,
    ):
        """
        Initialize builder.

        Args:
            runs_path: Base path for runs
            hardened_mode: Override hardened mode setting
        """
        self.runs_path = runs_path
        self.hardened_mode = hardened_mode if hardened_mode is not None else HARDENED_MODE

        # Required fields
        self._workflow_type: Optional[str] = None
        self._workflow_params: Dict[str, Any] = {}
        self._operator: str = ""
        self._goals: List[str] = []

        # Optional components
        self._with_sequence_manager: bool = False
        self._with_audit_logger: bool = False
        self._with_registry_snapshot: bool = False

        # Pre-built components (if provided externally)
        self._sequence_manager: Optional[SequenceManager] = None
        self._audit_logger: Optional[AuditLogger] = None
        self._registry_snapshot: Optional[Dict[str, Any]] = None

        # Run manager instance
        self._run_manager = RunManager(runs_path)

    # ========================================================================
    # REQUIRED CONFIGURATION
    # ========================================================================

    def with_workflow_type(self, workflow_type: str) -> RunContextBuilder:
        """Set the workflow type (required)."""
        self._workflow_type = workflow_type
        return self

    def with_workflow_params(self, params: Dict[str, Any]) -> RunContextBuilder:
        """Set workflow parameters."""
        self._workflow_params = params
        return self

    def with_operator(self, operator: str) -> RunContextBuilder:
        """Set the operator (user/system initiating the run)."""
        self._operator = operator
        return self

    def with_goals(self, goals: List[str]) -> RunContextBuilder:
        """Set the run goals."""
        self._goals = goals
        return self

    # ========================================================================
    # OPTIONAL COMPONENT CONFIGURATION
    # ========================================================================

    def with_sequence_manager(
        self,
        manager: Optional[SequenceManager] = None,
    ) -> RunContextBuilder:
        """
        Enable sequence manager for this context.

        Args:
            manager: Pre-built manager, or None to create one
        """
        self._with_sequence_manager = True
        self._sequence_manager = manager
        return self

    def with_audit_logger(
        self,
        logger: Optional[AuditLogger] = None,
    ) -> RunContextBuilder:
        """
        Enable audit logger for this context.

        Args:
            logger: Pre-built logger, or None to create one
        """
        self._with_audit_logger = True
        self._audit_logger = logger
        return self

    def with_registry_snapshot(
        self,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> RunContextBuilder:
        """
        Enable tool registry snapshot.

        Args:
            snapshot: Pre-built snapshot, or None to create one
        """
        self._with_registry_snapshot = True
        self._registry_snapshot = snapshot
        return self

    def with_all_components(self) -> RunContextBuilder:
        """Enable all optional components."""
        self._with_sequence_manager = True
        self._with_audit_logger = True
        self._with_registry_snapshot = True
        return self

    # ========================================================================
    # BUILD
    # ========================================================================

    def build(self) -> RunContext:
        """
        Build the fully-wired RunContext.

        Returns:
            Configured RunContext

        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        if not self._workflow_type:
            raise ValueError("workflow_type is required")

        # Initialize run via RunManager
        context = self._run_manager.initialize_run(
            workflow_type=self._workflow_type,
            workflow_params=self._workflow_params,
            operator=self._operator,
            goals=self._goals,
        )

        # Set hardened mode
        context.hardened_mode = self.hardened_mode

        # Set timing
        context.started_at = datetime.now()

        # Initialize sequence manager if requested
        if self._with_sequence_manager:
            if self._sequence_manager is not None:
                context.sequence_manager = self._sequence_manager
            else:
                context.sequence_manager = self._create_sequence_manager(context.run_id)

        # Initialize audit logger if requested
        if self._with_audit_logger:
            if self._audit_logger is not None:
                context.audit_logger = self._audit_logger
            else:
                context.audit_logger = self._create_audit_logger(context)

        # Initialize registry snapshot if requested
        if self._with_registry_snapshot:
            if self._registry_snapshot is not None:
                context.registry_snapshot = self._registry_snapshot
            else:
                context.registry_snapshot = self._create_registry_snapshot()

        return context

    def build_from_existing(self, run_id: str) -> RunContext:
        """
        Build a RunContext from an existing run.

        Args:
            run_id: Existing run ID to load

        Returns:
            Configured RunContext
        """
        # Load existing run
        context = self._run_manager.load_run(run_id)

        # Set hardened mode
        context.hardened_mode = self.hardened_mode

        # Initialize optional components
        if self._with_sequence_manager:
            if self._sequence_manager is not None:
                context.sequence_manager = self._sequence_manager
            else:
                context.sequence_manager = self._create_sequence_manager(context.run_id)

        if self._with_audit_logger:
            if self._audit_logger is not None:
                context.audit_logger = self._audit_logger
            else:
                context.audit_logger = self._create_audit_logger(context)

        if self._with_registry_snapshot:
            if self._registry_snapshot is not None:
                context.registry_snapshot = self._registry_snapshot
            else:
                context.registry_snapshot = self._create_registry_snapshot()

        return context

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    def _create_sequence_manager(self, run_id: str) -> SequenceManager:
        """Create a sequence manager for the run."""
        from .path_constants import (
            get_lock_retry_backoff,
            get_lock_retry_count,
            get_lock_timeout,
        )
        from .sequence_manager import SequenceManager

        return SequenceManager(
            run_id=run_id,
            runs_path=self.runs_path,
            lock_timeout_seconds=get_lock_timeout(),
            lock_retry_count=get_lock_retry_count(),
            lock_retry_backoff=get_lock_retry_backoff(),
        )

    def _create_audit_logger(self, context: RunContext) -> AuditLogger:
        """Create an audit logger for the run."""
        from .audit_logger import AuditLogger

        return AuditLogger(
            run_id=context.run_id,
            audit_path=context.audit_path,
        )

    def _create_registry_snapshot(self) -> Dict[str, Any]:
        """Create a tool registry snapshot."""
        try:
            from .tool_registry import ToolRegistry

            registry = ToolRegistry()
            return registry.snapshot()
        except ImportError:
            return {"error": "tool_registry not available"}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_run_context(
    workflow_type: str,
    workflow_params: Optional[Dict[str, Any]] = None,
    operator: str = "",
    goals: Optional[List[str]] = None,
    with_all_components: bool = True,
) -> RunContext:
    """
    Convenience function to build a fully-wired RunContext.

    Args:
        workflow_type: Type of workflow
        workflow_params: Workflow parameters
        operator: Operator name
        goals: Run goals
        with_all_components: Whether to enable all optional components

    Returns:
        Configured RunContext
    """
    builder = RunContextBuilder()
    builder.with_workflow_type(workflow_type)

    if workflow_params:
        builder.with_workflow_params(workflow_params)

    if operator:
        builder.with_operator(operator)

    if goals:
        builder.with_goals(goals)

    if with_all_components:
        builder.with_all_components()

    return builder.build()


def load_run_context(
    run_id: str,
    with_all_components: bool = True,
) -> RunContext:
    """
    Load an existing run with optional component initialization.

    Args:
        run_id: Run ID to load
        with_all_components: Whether to enable all optional components

    Returns:
        Configured RunContext
    """
    builder = RunContextBuilder()

    if with_all_components:
        builder.with_all_components()

    return builder.build_from_existing(run_id)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "RunContextBuilder",
    "build_run_context",
    "load_run_context",
]
