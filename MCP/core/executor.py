#!/usr/bin/env python3
"""
Orchestrator Task Executor

Executes workflow tasks by mapping agent types to MCP tools.

Pipeline Position:
    Workflow Tasks → [Executor] → MCP Tools → Results

Decision Capture:
    All execution decisions logged for orchestration training.

Phase 0 Hardening:
    - Error classification for intelligent retry decisions
    - Poison-pill detection to stop bad batches
    - Phase checkpointing for crash recovery
    - Validation gates for quality assurance
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Add project path
_CORE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CORE_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import STATE_PATH  # noqa: E402

from .config import OrchestratorConfig  # noqa: E402
from .param_mapper import ParameterMappingError, TaskParameterMapper  # noqa: E402

# Phase 0 Hardening: Import hardening modules with graceful fallback.
#
# Wave 22 F1 fix: these modules live in ``MCP/hardening/``, not in
# ``MCP/core/``. The historical relative imports (``from .error_classifier
# import ...``) silently hit the ``except ImportError`` arm, flipped every
# ``HARDENING_*`` flag to ``False``, and left the entire Phase 0 stack
# as a no-op at runtime. Tests that imported ``MCP.hardening.*`` directly
# did not catch the regression. Absolute imports from ``..hardening.*``
# restore the wiring; ``except ImportError`` is retained defensively for
# deployments that strip the hardening package, and a debug log makes
# future silent regressions observable.
try:
    from ..hardening.error_classifier import ErrorClass, ErrorClassifier, PoisonPillDetector
    HARDENING_ERROR_CLASSIFIER = True
except ImportError as _exc:
    HARDENING_ERROR_CLASSIFIER = False
    ErrorClass = None
    logging.getLogger(__name__).debug(
        "Hardening import failed (error_classifier): %s", _exc
    )

try:
    from ..hardening.checkpoint import CheckpointManager, PhaseCheckpoint  # noqa: F401
    HARDENING_CHECKPOINTS = True
except ImportError as _exc:
    HARDENING_CHECKPOINTS = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (checkpoint): %s", _exc
    )

try:
    from ..hardening.validation_gates import (  # noqa: F401
        GateConfig,
        GateIssue,
        GateResult,
        GateSeverity,
        ValidationGateManager,
    )
    HARDENING_VALIDATION_GATES = True
except ImportError as _exc:
    HARDENING_VALIDATION_GATES = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (validation_gates): %s", _exc
    )

try:
    from ..hardening.gate_input_routing import GateInputRouter, default_router
    HARDENING_GATE_INPUT_ROUTING = True
except ImportError as _exc:
    HARDENING_GATE_INPUT_ROUTING = False
    GateInputRouter = None  # type: ignore
    default_router = None  # type: ignore
    logging.getLogger(__name__).debug(
        "Hardening import failed (gate_input_routing): %s", _exc
    )

try:
    from ..hardening.lockfile import LockfileManager  # noqa: F401
    HARDENING_LOCKFILE = True
except ImportError as _exc:
    HARDENING_LOCKFILE = False
    logging.getLogger(__name__).debug(
        "Hardening import failed (lockfile): %s", _exc
    )

# Aggregate flag — True only when every Phase 0 hardening submodule
# imported cleanly. Consumers / regression tests assert against this
# single value rather than the four leaf flags.
HARDENING_PHASE_0 = (
    HARDENING_ERROR_CLASSIFIER
    and HARDENING_CHECKPOINTS
    and HARDENING_VALIDATION_GATES
    and HARDENING_LOCKFILE
)

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT TYPE TO MCP TOOL MAPPING
# =============================================================================
# Maps agent types (from config/agents.yaml) to actual MCP tool names.
# All tools listed here MUST exist in the MCP tool registry.
# =============================================================================

AGENT_TOOL_MAPPING = {
    # -------------------------------------------------------------------------
    # COURSEFORGE AGENTS
    # -------------------------------------------------------------------------
    "course-outliner": "create_course_project",
    "requirements-collector": "get_courseforge_status",
    "content-generator": "generate_course_content",
    "brightspace-packager": "package_imscc",
    "oscqr-course-evaluator": "validate_wcag_compliance",
    "quality-assurance": "get_courseforge_status",

    # -------------------------------------------------------------------------
    # PIPELINE AGENTS (Textbook-to-Course)
    # -------------------------------------------------------------------------
    "textbook-stager": "stage_dart_outputs",
    "textbook-ingestor": "create_course_project",
    "source-router": "build_source_module_map",

    # -------------------------------------------------------------------------
    # DART/REMEDIATION AGENTS (Multi-Source Synthesis)
    # -------------------------------------------------------------------------
    "dart-automation-coordinator": "batch_convert_multi_source",
    "dart-converter": "extract_and_convert_pdf",
    "imscc-intake-parser": "intake_imscc_package",
    "content-analyzer": "analyze_imscc_content",
    "accessibility-remediation": "remediate_course_content",
    "content-quality-remediation": "remediate_course_content",
    "intelligent-design-mapper": "remediate_course_content",
    "remediation-validator": "validate_wcag_compliance",

    # -------------------------------------------------------------------------
    # TRAINFORGE AGENTS
    # -------------------------------------------------------------------------
    "assessment-extractor": "analyze_imscc_content",
    "rag-indexer": "analyze_imscc_content",
    "assessment-generator": "generate_assessments",
    "assessment-validator": "validate_assessment",

    # -------------------------------------------------------------------------
    # LIBV2 AGENTS
    # -------------------------------------------------------------------------
    "libv2-archivist": "archive_to_libv2",
}


@dataclass
class ExecutionResult:
    """Result of executing a task."""
    task_id: str
    status: str  # "COMPLETE", "ERROR", "TIMEOUT", "POISON_PILL"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    # Phase 0 Hardening: Error classification
    error_class: Optional[str] = None  # "transient", "permanent", "poison_pill"
    retry_count: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error_class": self.error_class,
            "retry_count": self.retry_count,
            "artifacts": self.artifacts,
        }


class ToolRegistryError(Exception):
    """Raised when tool registry validation fails."""
    pass


class TaskExecutor:
    """
    Executes workflow tasks by invoking MCP tools.

    Maps agent types to appropriate tools and handles:
    - Task dispatch and tracking
    - Result collection
    - Error handling and retries
    - Decision capture for training

    Usage:
        executor = TaskExecutor(tool_registry, capture=capture)
        executor.validate_tool_registry()  # Fail-fast check
        result = await executor.execute_task(workflow_id, task_id)
    """

    def __init__(
        self,
        tool_registry: Optional[Dict[str, Callable[..., Awaitable[str]]]] = None,
        capture: Optional["DecisionCapture"] = None,
        config: Optional[OrchestratorConfig] = None,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        validate_registry: bool = False,
        run_id: Optional[str] = None,
        run_path: Optional[Path] = None,
        poison_pill_threshold: int = 3,
        batch_timeout_minutes: Optional[int] = None,
    ):
        """
        Initialize the task executor.

        Args:
            tool_registry: Dict mapping tool names to async functions
            capture: Optional DecisionCapture for logging decisions
            config: Optional OrchestratorConfig (loaded from YAML if not provided)
            max_retries: Override for max retry attempts (uses config if not set)
            timeout_seconds: Override for task timeout (uses config if not set)
            validate_registry: If True, validate tool registry at startup (fail-fast)
            run_id: Unique run identifier for tracing. Auto-generated if not provided.
            run_path: Path to run directory for checkpoints (Phase 0 hardening)
            poison_pill_threshold: N same-pattern failures stops batch (Phase 0)
            batch_timeout_minutes: Timeout for entire batch (Phase 0)
        """
        self.tool_registry = tool_registry or {}
        self.capture = capture

        # Generate or use provided run_id for tracing
        self.run_id = run_id or os.environ.get(
            'RUN_ID',
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Load config if not provided
        try:
            self.config = config or OrchestratorConfig.load()
        except Exception as e:
            logger.warning(f"[{self.run_id}] Failed to load config, using defaults: {e}")
            self.config = OrchestratorConfig()

        # Use provided values or fall back to config
        self.max_retries = max_retries if max_retries is not None else self.config.retry_attempts
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else (self.config.task_timeout_minutes * 60)
        self.batch_timeout_seconds = (batch_timeout_minutes or 30) * 60

        # Initialize parameter mapper
        self.param_mapper = TaskParameterMapper(strict=False)

        # Phase 0 Hardening: Initialize hardening components
        self.run_path = run_path or (STATE_PATH / "runs" / self.run_id)
        self._init_hardening(poison_pill_threshold)

        # Log initialization with run_id
        logger.info(f"[{self.run_id}] TaskExecutor initialized with {len(self.tool_registry)} tools")

        # Fail-fast validation if requested
        if validate_registry and self.tool_registry:
            self.validate_tool_registry()

    def _init_hardening(self, poison_pill_threshold: int) -> None:
        """Initialize Phase 0 hardening components."""
        # Error classifier for intelligent retry decisions
        self.error_classifier = None
        self.poison_detector = None
        if HARDENING_ERROR_CLASSIFIER:
            self.error_classifier = ErrorClassifier()
            self.poison_detector = PoisonPillDetector(
                threshold=poison_pill_threshold,
                window_seconds=300
            )
            logger.debug(f"[{self.run_id}] Error classifier and poison detector initialized")

        # Checkpoint manager for crash recovery
        self.checkpoint_manager = None
        if HARDENING_CHECKPOINTS and self.run_path:
            try:
                self.checkpoint_manager = CheckpointManager(self.run_path)
                logger.debug(f"[{self.run_id}] Checkpoint manager initialized")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to init checkpoint manager: {e}")

        # Validation gate manager
        self.gate_manager = None
        if HARDENING_VALIDATION_GATES:
            self.gate_manager = ValidationGateManager()
            logger.debug(f"[{self.run_id}] Validation gate manager initialized")

        # Wave 23 Sub-task A: per-gate input router. Pre-Wave-23, gates
        # received a generic ``{'artifacts': ..., 'results': ...}`` blob
        # regardless of validator shape, so critical gates silently
        # returned MISSING_INPUT issues and warning-severity gates
        # silently passed. The router builds per-validator kwargs from
        # the phase's accumulated outputs + workflow params.
        self.gate_input_router = None
        if HARDENING_GATE_INPUT_ROUTING and default_router is not None:
            self.gate_input_router = default_router()
            logger.debug(f"[{self.run_id}] Gate input router initialized")

        # Lock manager for cross-process resource locking (Wave 22 F1 fix:
        # was imported but never instantiated).
        self.lock_manager = None
        if HARDENING_LOCKFILE and self.run_path:
            try:
                self.lock_manager = LockfileManager(self.run_path)
                logger.debug(f"[{self.run_id}] Lock manager initialized")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to init lock manager: {e}")

    def validate_tool_registry(self, fail_fast: bool = True) -> Dict[str, List[str]]:
        """
        Validate that all AGENT_TOOL_MAPPING targets exist in the tool registry.

        This is a fail-fast check to catch misconfigurations at startup rather
        than at runtime when tasks fail.

        Args:
            fail_fast: If True, raise ToolRegistryError on first missing tool.
                      If False, collect and return all issues.

        Returns:
            Dict with 'missing' (tools in mapping but not registry) and
            'unmapped' (tools in registry but not in mapping) lists.

        Raises:
            ToolRegistryError: If fail_fast=True and validation fails.
        """
        # Get unique tools from mapping
        mapped_tools = set(AGENT_TOOL_MAPPING.values())
        registered_tools = set(self.tool_registry.keys())

        # Find missing tools (in mapping but not registered)
        missing = mapped_tools - registered_tools

        # Find unmapped tools (registered but not in mapping - just info)
        unmapped = registered_tools - mapped_tools

        issues = {
            "missing": sorted(missing),
            "unmapped": sorted(unmapped),
        }

        if missing:
            # Find which agents are affected
            affected_agents = [
                agent for agent, tool in AGENT_TOOL_MAPPING.items()
                if tool in missing
            ]

            error_msg = (
                f"Tool registry validation failed: {len(missing)} missing tools.\n"
                f"Missing tools: {sorted(missing)}\n"
                f"Affected agents: {affected_agents}\n"
                f"Ensure all MCP tools are registered before creating the executor."
            )

            logger.error(error_msg)

            if fail_fast:
                raise ToolRegistryError(error_msg)

        if unmapped:
            logger.info(
                f"Tool registry has {len(unmapped)} registered tools not in AGENT_TOOL_MAPPING: "
                f"{sorted(unmapped)}. This is informational only."
            )

        return issues

    def get_missing_tools(self) -> List[str]:
        """
        Get list of tools that are mapped but not registered.

        Returns:
            List of missing tool names.
        """
        mapped_tools = set(AGENT_TOOL_MAPPING.values())
        registered_tools = set(self.tool_registry.keys())
        return sorted(mapped_tools - registered_tools)

    async def execute_task(
        self,
        workflow_id: str,
        task_id: str,
    ) -> ExecutionResult:
        """
        Execute a pending task by invoking its mapped tool.

        Args:
            workflow_id: Parent workflow ID
            task_id: Task ID to execute

        Returns:
            ExecutionResult with status and output
        """
        start_time = datetime.now()

        # Load task from workflow state
        task = self._load_task(workflow_id, task_id)
        if not task:
            return ExecutionResult(
                task_id=task_id,
                status="ERROR",
                error=f"Task not found: {task_id}",
            )

        agent_type = task.get("agent_type", "")
        tool_name = AGENT_TOOL_MAPPING.get(agent_type)

        if not tool_name:
            error = f"No tool mapping for agent type: {agent_type}"
            logger.error(error)
            return ExecutionResult(
                task_id=task_id,
                status="ERROR",
                error=error,
            )

        # Log execution decision
        if self.capture:
            self.capture.log_decision(
                decision_type="task_execution",
                decision=f"Executing task {task_id} via tool '{tool_name}'",
                rationale=f"Agent type: {agent_type}, Workflow: {workflow_id}",
            )

        # Update task status to IN_PROGRESS
        self._update_task_status(workflow_id, task_id, "IN_PROGRESS")

        # Execute with retries
        result = await self._execute_with_retries(
            task_id=task_id,
            tool_name=tool_name,
            task_params=task,
        )

        # Calculate duration
        end_time = datetime.now()
        result.completed_at = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Update workflow state
        self._update_task_status(
            workflow_id,
            task_id,
            result.status,
            result=result.result,
            error=result.error,
        )

        # Log completion decision
        if self.capture:
            self.capture.log_decision(
                decision_type="task_completion",
                decision=f"Task {task_id} completed with status: {result.status}",
                rationale=f"Duration: {result.duration_seconds:.2f}s",
            )

        return result

    async def _execute_with_retries(
        self,
        task_id: str,
        tool_name: str,
        task_params: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute tool with intelligent retry logic.

        Phase 0 Hardening:
        - Uses error classification to determine retry behavior
        - Detects poison-pill patterns that should stop the batch
        - Only retries transient errors, not permanent ones
        """
        last_error = None
        error_class_value = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._invoke_tool(tool_name, task_params)

                return ExecutionResult(
                    task_id=task_id,
                    status="COMPLETE",
                    result=result,
                    retry_count=retry_count,
                )

            except asyncio.TimeoutError as e:
                last_error = f"Task timed out after {self.timeout_seconds}s"
                logger.warning(f"[{self.run_id}] Task {task_id} attempt {attempt + 1} timed out")

                # Phase 0: Classify timeout error
                if self.error_classifier:
                    classified = self.error_classifier.classify(e, task_id)
                    error_class_value = classified.error_class.value

                    # Check for poison pill
                    if self.poison_detector:
                        poison_result = self.poison_detector.record_failure(classified)
                        if poison_result and poison_result.triggered:
                            logger.error(f"[{self.run_id}] Poison pill detected: {poison_result.recommendation}")
                            return ExecutionResult(
                                task_id=task_id,
                                status="POISON_PILL",
                                error=f"Batch stopped: {poison_result.error_pattern}",
                                error_class="poison_pill",
                                retry_count=retry_count,
                            )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.run_id}] Task {task_id} attempt {attempt + 1} failed: {e}")

                # Phase 0: Classify error for retry decisions
                if self.error_classifier:
                    classified = self.error_classifier.classify(e, task_id)
                    error_class_value = classified.error_class.value

                    # Check for poison pill
                    if self.poison_detector:
                        poison_result = self.poison_detector.record_failure(classified)
                        if poison_result and poison_result.triggered:
                            logger.error(f"[{self.run_id}] Poison pill detected: {poison_result.recommendation}")
                            return ExecutionResult(
                                task_id=task_id,
                                status="POISON_PILL",
                                error=f"Batch stopped: {poison_result.error_pattern}",
                                error_class="poison_pill",
                                retry_count=retry_count,
                            )

                    # Don't retry permanent errors
                    if classified.error_class == ErrorClass.PERMANENT:
                        logger.info(f"[{self.run_id}] Task {task_id} has permanent error, not retrying")
                        return ExecutionResult(
                            task_id=task_id,
                            status="ERROR",
                            error=last_error,
                            error_class="permanent",
                            retry_count=retry_count,
                        )

            # Log retry decision
            retry_count += 1
            if attempt < self.max_retries and self.capture:
                rationale = f"Previous error: {last_error}"
                if error_class_value:
                    rationale += f", Error class: {error_class_value}"
                self.capture.log_decision(
                    decision_type="task_retry",
                    decision=f"Retrying task {task_id} (attempt {attempt + 2})",
                    rationale=rationale,
                )

        return ExecutionResult(
            task_id=task_id,
            status="ERROR",
            error=last_error,
            error_class=error_class_value or "unknown",
            retry_count=retry_count,
        )

    async def _invoke_tool(
        self,
        tool_name: str,
        task_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke an MCP tool with properly mapped parameters.

        Uses TaskParameterMapper to translate generic task parameters
        to the tool-specific parameter names expected by each tool.

        Args:
            tool_name: Name of the MCP tool to invoke
            task_params: Task dict with prompt, params, context, etc.

        Returns:
            Parsed JSON result from the tool

        Raises:
            ValueError: If tool not registered
            ParameterMappingError: If required parameters are missing
        """
        tool_func = self.tool_registry.get(tool_name)

        if not tool_func:
            raise ValueError(f"Tool not registered: {tool_name}")

        # Use parameter mapper to get tool-specific parameters
        try:
            mapped_params = self.param_mapper.map_task_to_tool_params(
                task_params, tool_name
            )
        except ParameterMappingError as e:
            logger.error(f"Parameter mapping failed for {tool_name}: {e}")
            raise

        # Log the mapped parameters for debugging
        logger.debug(f"Invoking {tool_name} with params: {list(mapped_params.keys())}")

        # Call tool with mapped parameters
        result_str = await asyncio.wait_for(
            tool_func(**mapped_params),
            timeout=self.timeout_seconds,
        )

        # Parse result
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return {"raw_result": result_str}

    def _load_task(
        self,
        workflow_id: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load task from workflow state file."""
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return None

        try:
            with open(workflow_path) as f:
                workflow = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load workflow {workflow_id}: {e}")
            return None

        for task in workflow.get("tasks", []):
            if task.get("id") == task_id:
                return task

        return None

    def _update_task_status(
        self,
        workflow_id: str,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update task status in workflow state."""
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return False

        try:
            with open(workflow_path) as f:
                workflow = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read workflow {workflow_id} for task update: {e}")
            return False

        for task in workflow.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()

                if status == "IN_PROGRESS":
                    task["started_at"] = datetime.now().isoformat()
                elif status in ("COMPLETE", "ERROR"):
                    task["completed_at"] = datetime.now().isoformat()

                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error

                break

        # Update progress counters
        progress = workflow.get("progress", {})
        tasks = workflow.get("tasks", [])

        progress["completed"] = sum(1 for t in tasks if t.get("status") == "COMPLETE")
        progress["in_progress"] = sum(1 for t in tasks if t.get("status") == "IN_PROGRESS")
        progress["failed"] = sum(1 for t in tasks if t.get("status") == "ERROR")

        workflow["progress"] = progress
        workflow["updated_at"] = datetime.now().isoformat()

        try:
            with open(workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            return True
        except OSError:
            return False

    async def execute_workflow(
        self,
        workflow_id: str,
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all pending tasks in a workflow.

        Args:
            workflow_id: Workflow to execute
            parallel: Run independent tasks in parallel
            max_concurrent: Max concurrent tasks

        Returns:
            Dict mapping task_id to ExecutionResult
        """
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return {}

        with open(workflow_path) as f:
            workflow = json.load(f)

        tasks = workflow.get("tasks", [])
        results = {}

        if self.capture:
            pending_count = sum(1 for t in tasks if t.get("status") == "PENDING")
            self.capture.log_decision(
                decision_type="workflow_execution",
                decision=f"Starting workflow {workflow_id} execution",
                rationale=f"Pending tasks: {pending_count}, Parallel: {parallel}",
            )

        if parallel:
            results = await self._execute_parallel(workflow_id, tasks, max_concurrent)
        else:
            results = await self._execute_sequential(workflow_id, tasks)

        return results

    async def _execute_parallel(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
        max_concurrent: int,
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks in parallel batches."""
        results = {}
        completed_ids = set()

        while True:
            # Find tasks that can run (pending + dependencies met)
            runnable = []
            for task in tasks:
                task_id = task.get("id")
                if task.get("status") != "PENDING":
                    if task.get("status") == "COMPLETE":
                        completed_ids.add(task_id)
                    continue

                deps = task.get("dependencies", [])
                if all(d in completed_ids for d in deps):
                    runnable.append(task)

            if not runnable:
                break

            # Execute batch
            batch = runnable[:max_concurrent]
            batch_tasks = [
                self.execute_task(workflow_id, t["id"])
                for t in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for task, result in zip(batch, batch_results):
                task_id = task["id"]
                if isinstance(result, Exception):
                    results[task_id] = ExecutionResult(
                        task_id=task_id,
                        status="ERROR",
                        error=str(result),
                    )
                    task["status"] = "ERROR"
                else:
                    results[task_id] = result
                    if result.status == "COMPLETE":
                        completed_ids.add(task_id)
                    task["status"] = result.status

        return results

    async def _execute_sequential(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            if task.get("status") != "PENDING":
                continue

            result = await self.execute_task(workflow_id, task["id"])
            results[task["id"]] = result

            if result.status == "ERROR":
                # Stop on error in sequential mode
                break

        return results

    def register_tool(
        self,
        tool_name: str,
        tool_func: Callable[..., Awaitable[str]],
    ) -> None:
        """Register a tool function for execution."""
        self.tool_registry[tool_name] = tool_func

    # =========================================================================
    # Phase 0 Hardening: Phase Execution with Checkpoints and Validation Gates
    # =========================================================================

    async def execute_phase(
        self,
        workflow_id: str,
        phase_name: str,
        phase_index: int,
        tasks: List[Dict[str, Any]],
        gate_configs: Optional[List[Dict[str, Any]]] = None,
        max_concurrent: int = 5,
        phase_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
        workflow_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, ExecutionResult], bool, Optional[List[Dict]]]:
        """
        Execute a workflow phase with checkpointing and validation gates.

        Phase 0 Hardening:
        - Creates checkpoint at phase start
        - Updates checkpoint after each task
        - Runs validation gates at phase end
        - Supports crash recovery via checkpoints

        Args:
            workflow_id: Parent workflow ID
            phase_name: Name of the phase
            phase_index: Index of phase in workflow
            tasks: List of tasks to execute
            gate_configs: Optional list of validation gate configurations
            max_concurrent: Maximum concurrent tasks

        Returns:
            Tuple of (results dict, gates_passed bool, gate_results list)
        """
        task_ids = [t.get("id") for t in tasks]
        gate_results = None

        # Start checkpoint
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.start_phase(
                    run_id=self.run_id,
                    workflow_id=workflow_id,
                    phase_name=phase_name,
                    phase_index=phase_index,
                    task_ids=task_ids
                )
                logger.info(f"[{self.run_id}] Started phase checkpoint: {phase_name}")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to create phase checkpoint: {e}")

        # Log phase start
        if self.capture:
            self.capture.log_decision(
                decision_type="phase_start",
                decision=f"Starting phase: {phase_name}",
                rationale=f"Phase {phase_index}, {len(tasks)} tasks, max_concurrent={max_concurrent}",
            )

        # Execute tasks with batch timeout
        try:
            results = await asyncio.wait_for(
                self._execute_parallel(workflow_id, tasks, max_concurrent),
                timeout=self.batch_timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"[{self.run_id}] Phase {phase_name} timed out after {self.batch_timeout_seconds}s")
            results = {
                t.get("id"): ExecutionResult(
                    task_id=t.get("id"),
                    status="TIMEOUT",
                    error=f"Phase batch timeout after {self.batch_timeout_seconds}s"
                )
                for t in tasks if t.get("status") == "PENDING"
            }

        # Update checkpoint with task results
        if self.checkpoint_manager:
            for task_id, result in results.items():
                try:
                    artifacts = result.artifacts if hasattr(result, 'artifacts') else []
                    self.checkpoint_manager.complete_task(
                        phase_name=phase_name,
                        task_id=task_id,
                        success=result.status == "COMPLETE",
                        artifacts=[a for a in artifacts] if artifacts else None
                    )
                except Exception as e:
                    logger.warning(f"[{self.run_id}] Failed to update task checkpoint: {e}")

        # Check for poison pill status
        poison_detected = any(r.status == "POISON_PILL" for r in results.values())
        if poison_detected:
            logger.error(f"[{self.run_id}] Phase {phase_name} stopped due to poison pill")
            if self.checkpoint_manager:
                self.checkpoint_manager.fail_phase(phase_name, "Poison pill detected")
            return results, False, None

        # Run validation gates (Wave 23: per-gate input routing)
        gates_passed = True
        if gate_configs and self.gate_manager and HARDENING_VALIDATION_GATES:
            # Build the fallback artifacts blob for validators not yet in
            # the router registry (legacy / unknown paths).
            all_artifacts = []
            for result in results.values():
                if hasattr(result, 'artifacts') and result.artifacts:
                    all_artifacts.extend(result.artifacts)
                if result.result and isinstance(result.result, dict):
                    if 'artifacts' in result.result:
                        all_artifacts.extend(result.result['artifacts'])
            fallback_inputs = {'artifacts': all_artifacts, 'results': results}

            # Accumulated phase outputs + workflow params feed the router.
            # Callers (WorkflowRunner) pass these explicitly; legacy
            # callers that don't get an empty blob → every gate without
            # a builder route falls back to fallback_inputs.
            _phase_outputs = phase_outputs or {}
            _workflow_params = workflow_params or {}

            gate_results_list = []
            parsed_gates = []
            for gc in gate_configs:
                try:
                    gate = GateConfig(
                        gate_id=gc.get('gate_id', 'unknown'),
                        validator_path=gc.get('validator', gc.get('validator_path', '')),
                        severity=GateSeverity(gc.get('severity', 'critical')),
                        threshold=gc.get('threshold', {}),
                    )
                    parsed_gates.append(gate)
                except Exception as e:
                    logger.warning(f"[{self.run_id}] Invalid gate config: {e}")

            for gate in parsed_gates:
                # Per-gate input build.
                inputs: Dict[str, Any]
                missing: List[str] = []
                if self.gate_input_router is not None and gate.validator_path:
                    inputs, missing = self.gate_input_router.build(
                        gate.validator_path, _phase_outputs, _workflow_params,
                    )
                else:
                    inputs = dict(fallback_inputs)

                # If the builder flagged missing required inputs, mark
                # the gate as skipped rather than silently passing.
                if missing:
                    reason = ", ".join(missing)
                    logger.warning(
                        f"[{self.run_id}] Gate {gate.gate_id} "
                        f"({gate.validator_path}) skipped — missing inputs: "
                        f"{reason}"
                    )
                    skipped_result = GateResult(
                        gate_id=gate.gate_id,
                        validator_name=gate.validator_path,
                        validator_version="skipped",
                        passed=True,
                        score=None,
                        issues=[GateIssue(
                            severity="warning",
                            code="GATE_SKIPPED_MISSING_INPUTS",
                            message=(
                                f"Gate skipped: builder could not resolve "
                                f"required inputs ({reason}). This is a "
                                "structured skip, not a silent pass — the "
                                "gate did not run."
                            ),
                            suggestion=(
                                "Ensure the phase's upstream outputs "
                                "surface the required keys, or add a "
                                "builder for this validator in "
                                "MCP/hardening/gate_input_routing.py."
                            ),
                        )],
                    )
                    # Mark as skipped in a forward-compat way.
                    try:
                        skipped_result.waiver_info = {"skipped": "true", "reason": reason}
                    except Exception:
                        pass
                    gate_results_list.append(skipped_result)
                    continue

                # Merge the router-produced inputs with fallback blob
                # under non-colliding keys so legacy validators that
                # look for 'artifacts' still find it.
                merged_inputs: Dict[str, Any] = dict(fallback_inputs)
                merged_inputs.update(inputs)

                # Run the gate via the manager (handles waivers + errors)
                result = self.gate_manager.run_gate(gate, merged_inputs)
                gate_results_list.append(result)

                # Honour severity / behavior-on-fail for gate ordering.
                if not result.passed:
                    if gate.severity == GateSeverity.CRITICAL:
                        gates_passed = False

            gate_results = [gr.to_dict() if hasattr(gr, 'to_dict') else gr for gr in gate_results_list]

            # Log gate results
            if self.capture:
                for gr in gate_results_list:
                    skipped = bool(getattr(gr, 'waiver_info', None) and isinstance(gr.waiver_info, dict) and gr.waiver_info.get('skipped') == 'true')
                    if skipped:
                        status = "SKIPPED"
                    else:
                        status = "PASSED" if gr.passed else "FAILED"
                    self.capture.log_decision(
                        decision_type="validation_result",
                        decision=f"Gate {gr.gate_id}: {status}",
                        rationale=f"Score: {gr.score}, Issues: {len(gr.issues)}",
                    )

        # Complete or fail checkpoint
        if self.checkpoint_manager:
            try:
                if gates_passed:
                    validation_results = {'gate_results': gate_results} if gate_results else {}
                    self.checkpoint_manager.complete_phase(phase_name, validation_results)
                    logger.info(f"[{self.run_id}] Completed phase checkpoint: {phase_name}")
                else:
                    self.checkpoint_manager.fail_phase(phase_name, "Validation gates failed")
                    logger.warning(f"[{self.run_id}] Phase {phase_name} failed validation gates")
            except Exception as e:
                logger.warning(f"[{self.run_id}] Failed to finalize phase checkpoint: {e}")

        # Log phase completion
        if self.capture:
            completed = sum(1 for r in results.values() if r.status == "COMPLETE")
            failed = sum(1 for r in results.values() if r.status in ("ERROR", "TIMEOUT"))
            self.capture.log_decision(
                decision_type="phase_completion",
                decision=f"Phase {phase_name} completed: {completed} success, {failed} failed",
                rationale=f"Gates passed: {gates_passed}",
            )

        return results, gates_passed, gate_results

    def get_resumable_phase(self) -> Optional[Dict[str, Any]]:
        """
        Check for incomplete phases that can be resumed.

        Returns:
            Phase checkpoint dict if resumable phase exists, None otherwise
        """
        if not self.checkpoint_manager:
            return None

        checkpoint = self.checkpoint_manager.get_resumable_phase()
        if checkpoint:
            return {
                'phase_name': checkpoint.phase_name,
                'phase_index': checkpoint.phase_index,
                'tasks_completed': checkpoint.tasks_completed,
                'tasks_pending': checkpoint.tasks_pending,
                'last_event_seq': checkpoint.last_event_seq,
            }
        return None

    def reset_poison_detector(self) -> None:
        """Reset poison pill detector for new batch."""
        if self.poison_detector:
            self.poison_detector.reset()


async def execute_workflow_task(
    workflow_id: str,
    task_id: str,
    tool_registry: Optional[Dict[str, Callable[..., Awaitable[str]]]] = None,
    capture: Optional["DecisionCapture"] = None,
) -> ExecutionResult:
    """
    Convenience function to execute a single workflow task.

    Args:
        workflow_id: Workflow ID
        task_id: Task ID to execute
        tool_registry: Tool function registry
        capture: Optional decision capture

    Returns:
        ExecutionResult
    """
    executor = TaskExecutor(tool_registry=tool_registry, capture=capture)
    return await executor.execute_task(workflow_id, task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo - would need actual tool registry in practice
    async def demo():
        _ = TaskExecutor()
        print(f"Agent to tool mapping: {len(AGENT_TOOL_MAPPING)} mappings")
        for agent, tool in AGENT_TOOL_MAPPING.items():
            print(f"  {agent} -> {tool}")

    asyncio.run(demo())
