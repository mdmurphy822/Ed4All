"""
Pydantic Task Schemas for Orchestrator

Provides strongly-typed models for task definitions and results
with validation at construction time.

Usage:
    from orchestrator.core.schemas import Task, TaskResult, TaskStatus

    task = Task(
        id="task_001",
        workflow_id="wf_001",
        agent_type="content-generator",
        params={"course_code": "INT_101"}
    )

    result = TaskResult(
        task_id="task_001",
        status=TaskStatus.COMPLETE,
        result={"files_created": 3}
    )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# ENUMS
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a task in the workflow."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Status of an entire workflow."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# =============================================================================
# TASK MODELS
# =============================================================================

class Task(BaseModel):
    """
    A single task to be executed by an agent.

    Tasks are the atomic units of work in a workflow. Each task
    is assigned to a specific agent type and may depend on other tasks.

    Attributes:
        id: Unique task identifier
        workflow_id: Parent workflow identifier
        agent_type: Type of agent to execute this task
        status: Current task status
        prompt: Optional instructions for the agent
        params: Task-specific parameters
        context: Additional context passed to the agent
        dependencies: List of task IDs that must complete first
        priority: Task priority (higher = more important)
        retries: Number of retry attempts remaining
        created_at: When the task was created
    """
    id: str = Field(..., min_length=1, description="Unique task identifier")
    workflow_id: str = Field(..., min_length=1, description="Parent workflow ID")
    agent_type: str = Field(..., min_length=1, description="Agent type to execute task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    prompt: Optional[str] = Field(default=None, description="Instructions for agent")
    params: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    dependencies: List[str] = Field(default_factory=list, description="Task IDs to wait for")
    priority: int = Field(default=0, ge=0, le=100, description="Priority (0-100)")
    retries: int = Field(default=3, ge=0, description="Remaining retry attempts")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """
        Validate that agent_type is a known agent.

        Note: This imports AGENT_TOOL_MAPPING lazily to avoid circular imports.
        """
        # Import lazily to avoid circular dependency
        from .executor import AGENT_TOOL_MAPPING
        if v not in AGENT_TOOL_MAPPING:
            raise ValueError(
                f"Unknown agent type: '{v}'. "
                f"Valid types: {sorted(AGENT_TOOL_MAPPING.keys())}"
            )
        return v

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies_not_self(cls, v: List[str], info) -> List[str]:
        """Ensure task doesn't depend on itself."""
        task_id = info.data.get('id')
        if task_id and task_id in v:
            raise ValueError(f"Task cannot depend on itself: {task_id}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "task_001",
                "workflow_id": "wf_course_gen_001",
                "agent_type": "content-generator",
                "status": "PENDING",
                "params": {"course_code": "INT_101", "week": 1},
                "dependencies": [],
            }
        }
    }


class TaskResult(BaseModel):
    """
    Result of executing a task.

    Captures the outcome of task execution including timing,
    any produced outputs, and error information if applicable.

    Attributes:
        task_id: ID of the executed task
        status: Final status of execution
        result: Output data from successful execution
        error: Error message if execution failed
        started_at: When execution started
        completed_at: When execution finished
        duration_seconds: Total execution time
        retry_count: How many retries were attempted
    """
    task_id: str = Field(..., min_length=1, description="Executed task ID")
    status: TaskStatus = Field(..., description="Final execution status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Execution output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(default=None, description="Execution end time")
    duration_seconds: float = Field(default=0.0, ge=0, description="Total execution time")
    retry_count: int = Field(default=0, ge=0, description="Retries attempted")

    @model_validator(mode='after')
    def validate_error_on_failure(self) -> 'TaskResult':
        """Ensure error is set when status is ERROR."""
        if self.status == TaskStatus.ERROR and not self.error:
            raise ValueError("Error status requires an error message")
        return self

    @model_validator(mode='after')
    def calculate_duration(self) -> 'TaskResult':
        """Calculate duration if completed_at is set."""
        if self.completed_at and self.duration_seconds == 0.0:
            delta = self.completed_at - self.started_at
            object.__setattr__(self, 'duration_seconds', delta.total_seconds())
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_001",
                "status": "COMPLETE",
                "result": {"files_created": ["module_1.html", "module_2.html"]},
                "started_at": "2026-01-07T10:00:00",
                "completed_at": "2026-01-07T10:05:30",
                "duration_seconds": 330.0,
            }
        }
    }


# =============================================================================
# WORKFLOW MODELS
# =============================================================================

class WorkflowPhase(BaseModel):
    """
    A phase within a workflow containing multiple tasks.

    Attributes:
        name: Phase identifier
        description: Human-readable description
        tasks: List of task IDs in this phase
        depends_on: Phases that must complete first
        status: Current phase status
    """
    name: str = Field(..., min_length=1, description="Phase identifier")
    description: Optional[str] = Field(default=None, description="Phase description")
    tasks: List[str] = Field(default_factory=list, description="Task IDs in phase")
    depends_on: List[str] = Field(default_factory=list, description="Dependent phases")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Phase status")


class Workflow(BaseModel):
    """
    A complete workflow definition.

    Workflows orchestrate multiple tasks across phases with
    dependency management and parallel execution support.

    Attributes:
        id: Unique workflow identifier
        name: Workflow name
        status: Current workflow status
        phases: Ordered list of workflow phases
        tasks: All tasks in the workflow by ID
        max_concurrent: Maximum parallel tasks
        created_at: When workflow was created
        started_at: When execution started
        completed_at: When execution finished
    """
    id: str = Field(..., min_length=1, description="Workflow identifier")
    name: str = Field(..., min_length=1, description="Workflow name")
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED, description="Current status")
    phases: List[WorkflowPhase] = Field(default_factory=list, description="Workflow phases")
    tasks: Dict[str, Task] = Field(default_factory=dict, description="All tasks by ID")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Max parallel tasks")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")

    def get_pending_tasks(self) -> List[Task]:
        """Get all tasks with PENDING status."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks whose dependencies are all complete."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            # Check all dependencies are complete
            deps_complete = all(
                self.tasks.get(dep_id, Task(id=dep_id, workflow_id=self.id, agent_type="")).status == TaskStatus.COMPLETE
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            if deps_complete:
                ready.append(task)
        return ready

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update a task's status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status


# =============================================================================
# BATCH MODELS
# =============================================================================

class BatchRequest(BaseModel):
    """
    Request to execute a batch of tasks.

    Attributes:
        workflow_id: Parent workflow
        task_ids: Tasks to execute in this batch
        timeout_seconds: Batch timeout
    """
    workflow_id: str = Field(..., description="Parent workflow ID")
    task_ids: List[str] = Field(..., min_length=1, max_length=10, description="Task IDs")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Batch timeout")

    @field_validator('task_ids')
    @classmethod
    def validate_batch_size(cls, v: List[str]) -> List[str]:
        """Enforce maximum batch size of 10."""
        if len(v) > 10:
            raise ValueError(f"Batch size {len(v)} exceeds maximum of 10")
        return v


class BatchResult(BaseModel):
    """
    Result of executing a batch of tasks.

    Attributes:
        batch_id: Unique batch identifier
        workflow_id: Parent workflow
        results: Individual task results
        total_duration_seconds: Total batch execution time
        success_count: Number of successful tasks
        error_count: Number of failed tasks
    """
    batch_id: str = Field(..., description="Batch identifier")
    workflow_id: str = Field(..., description="Parent workflow ID")
    results: List[TaskResult] = Field(default_factory=list, description="Task results")
    total_duration_seconds: float = Field(default=0.0, description="Total batch time")
    success_count: int = Field(default=0, ge=0, description="Successful tasks")
    error_count: int = Field(default=0, ge=0, description="Failed tasks")

    @model_validator(mode='after')
    def calculate_counts(self) -> 'BatchResult':
        """Calculate success and error counts from results."""
        success = sum(1 for r in self.results if r.status == TaskStatus.COMPLETE)
        errors = sum(1 for r in self.results if r.status in (TaskStatus.ERROR, TaskStatus.TIMEOUT))
        object.__setattr__(self, 'success_count', success)
        object.__setattr__(self, 'error_count', errors)
        return self


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_task(
    workflow_id: str,
    agent_type: str,
    task_id: Optional[str] = None,
    **kwargs
) -> Task:
    """
    Factory function to create a Task with auto-generated ID.

    Args:
        workflow_id: Parent workflow ID
        agent_type: Type of agent to execute task
        task_id: Optional task ID (auto-generated if not provided)
        **kwargs: Additional task parameters

    Returns:
        Validated Task instance
    """
    if task_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_id = f"task_{agent_type}_{timestamp}"

    return Task(
        id=task_id,
        workflow_id=workflow_id,
        agent_type=agent_type,
        **kwargs
    )


def create_task_result(
    task_id: str,
    status: TaskStatus,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> TaskResult:
    """
    Factory function to create a TaskResult.

    Args:
        task_id: ID of the executed task
        status: Execution status
        result: Output data (for successful execution)
        error: Error message (for failed execution)

    Returns:
        Validated TaskResult instance
    """
    return TaskResult(
        task_id=task_id,
        status=status,
        result=result,
        error=error,
        started_at=datetime.now(),
        completed_at=datetime.now() if status != TaskStatus.IN_PROGRESS else None,
    )
