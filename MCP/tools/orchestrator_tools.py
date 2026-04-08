"""
Orchestrator MCP Tools

Tools for workflow orchestration, state management, and agent coordination.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import PROJECT_ROOT, STATE_PATH, CONFIG_PATH, STATE_LOCKS
from lib.state_manager import atomic_write_json
from orchestrator.ipc import StatusTracker

# Shared StatusTracker instance
_status_tracker = None

def _get_tracker() -> StatusTracker:
    """Get shared StatusTracker instance."""
    global _status_tracker
    if _status_tracker is None:
        _status_tracker = StatusTracker()
    return _status_tracker

logger = logging.getLogger(__name__)

# Derived paths
STATUS_PATH = STATE_PATH / "status"
LOCKS_PATH = STATE_LOCKS


def _validate_orchestrator_paths():
    """Validate orchestrator paths at module load."""
    required = [STATE_PATH, STATUS_PATH, LOCKS_PATH]
    for path in required:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            logger.debug(f"Directory exists: {path}")


_validate_orchestrator_paths()


def register_orchestrator_tools(mcp):
    """Register orchestrator tools with the MCP server."""

    @mcp.tool()
    async def create_workflow(
        workflow_type: str,
        params: str,
        priority: str = "normal"
    ) -> str:
        """
        Create a new workflow execution.

        Args:
            workflow_type: Type of workflow
                Options: course_generation, intake_remediation, batch_dart, rag_training
            params: JSON string with workflow-specific parameters
            priority: Execution priority ("low", "normal", "high")

        Returns:
            Workflow ID and initial status
        """
        try:
            workflow_id = f"WF-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"

            workflow_params = json.loads(params) if params else {}

            workflow = {
                "id": workflow_id,
                "type": workflow_type,
                "params": workflow_params,
                "priority": priority,
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "tasks": [],
                "progress": {
                    "total": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "failed": 0
                }
            }

            # Save workflow state
            workflows_dir = STATE_PATH / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)

            workflow_path = workflows_dir / f"{workflow_id}.json"
            atomic_write_json(workflow_path, workflow)

            # Update GENERATION_PROGRESS.md
            _get_tracker().update_progress_md()

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "status": "PENDING",
                "workflow_path": str(workflow_path)
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_workflow_status(workflow_id: str) -> str:
        """
        Get current status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status with task breakdown and progress
        """
        try:
            workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})

            with open(workflow_path) as f:
                workflow = json.load(f)

            return json.dumps(workflow)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def dispatch_agent_task(
        workflow_id: str,
        agent_type: str,
        task_prompt: str,
        dependencies: Optional[str] = None
    ) -> str:
        """
        Dispatch a task to a specific agent.

        Args:
            workflow_id: Parent workflow ID
            agent_type: Agent type to dispatch
                Options: content-generator, brightspace-packager, dart-automation-coordinator, etc.
            task_prompt: Task instructions for the agent
            dependencies: Optional comma-separated list of task IDs that must complete first

        Returns:
            Task ID and dispatch status
        """
        try:
            # Import here to avoid circular imports
            from orchestrator.core.executor import AGENT_TOOL_MAPPING
            from orchestrator.core.config import OrchestratorConfig

            # Validate agent type exists in config
            try:
                config = OrchestratorConfig.load()
                agent_config = config.get_agent(agent_type)

                if not agent_config:
                    available_agents = list(config.agents.keys())
                    return json.dumps({
                        "error": f"Unknown agent type: {agent_type}",
                        "available_agents": available_agents[:10],  # Limit for readability
                        "total_agents": len(available_agents)
                    })
            except Exception as config_err:
                logger.warning(f"Config validation skipped: {config_err}")
                # Continue without config validation

            # Validate tool mapping exists
            tool_name = AGENT_TOOL_MAPPING.get(agent_type)
            if not tool_name:
                return json.dumps({
                    "error": f"No tool mapping for agent type: {agent_type}",
                    "available_mappings": list(AGENT_TOOL_MAPPING.keys())
                })

            # Validate workflow exists
            workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})

            with open(workflow_path) as f:
                workflow = json.load(f)

            task_id = f"T-{str(uuid.uuid4())[:8]}"
            dep_list = [d.strip() for d in dependencies.split(",")] if dependencies else []

            task = {
                "id": task_id,
                "agent_type": agent_type,
                "tool_name": tool_name,  # Include mapped tool name for transparency
                "prompt": task_prompt,
                "dependencies": dep_list,
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "result": None
            }

            workflow["tasks"].append(task)
            workflow["progress"]["total"] += 1
            workflow["updated_at"] = datetime.now().isoformat()

            atomic_write_json(workflow_path, workflow)

            return json.dumps({
                "success": True,
                "task_id": task_id,
                "workflow_id": workflow_id,
                "agent_type": agent_type,
                "tool_name": tool_name,
                "status": "PENDING"
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def poll_task_completions(workflow_id: Optional[str] = None) -> str:
        """
        Poll for completed or errored tasks.

        Args:
            workflow_id: Filter by workflow (None for all)

        Returns:
            List of completed/errored tasks since last poll
        """
        try:
            workflows_dir = STATE_PATH / "workflows"
            if not workflows_dir.exists():
                return json.dumps({"completed": [], "errored": []})

            completed = []
            errored = []

            workflow_files = (
                [workflows_dir / f"{workflow_id}.json"]
                if workflow_id
                else list(workflows_dir.glob("*.json"))
            )

            for wf_path in workflow_files:
                if not wf_path.exists():
                    continue

                with open(wf_path) as f:
                    workflow = json.load(f)

                for task in workflow.get("tasks", []):
                    if task["status"] == "COMPLETE":
                        completed.append({
                            "workflow_id": workflow["id"],
                            "task_id": task["id"],
                            "agent_type": task["agent_type"],
                            "completed_at": task.get("completed_at")
                        })
                    elif task["status"] == "ERROR":
                        errored.append({
                            "workflow_id": workflow["id"],
                            "task_id": task["id"],
                            "agent_type": task["agent_type"],
                            "error": task.get("error")
                        })

            return json.dumps({
                "completed": completed,
                "errored": errored,
                "poll_time": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def update_generation_progress(
        component: str,
        status: str,
        details: Optional[str] = None
    ) -> str:
        """
        Update GENERATION_PROGRESS.md shared state.

        Args:
            component: Component identifier (e.g., "DART_BATCH_1", "COURSE_MTH_301")
            status: Status value ("PENDING", "IN_PROGRESS", "COMPLETE", "ERROR")
            details: Optional JSON string with additional status details

        Returns:
            Confirmation of progress update
        """
        try:
            status_dir = STATE_PATH / "status"
            status_dir.mkdir(parents=True, exist_ok=True)

            status_file = status_dir / f"{component}.json"
            detail_data = json.loads(details) if details else {}

            status_data = {
                "component": component,
                "status": status,
                "updated_at": datetime.now().isoformat(),
                "details": detail_data
            }

            atomic_write_json(status_file, status_data)

            # Update progress markdown
            _get_tracker().update_progress_md()

            return json.dumps({
                "success": True,
                "component": component,
                "status": status,
                "status_file": str(status_file)
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def acquire_batch_lock(
        resource: str,
        owner: str,
        ttl_seconds: int = 3600
    ) -> str:
        """
        Acquire exclusive lock on a resource for batch processing.

        Args:
            resource: Resource identifier (e.g., "courseforge/exports", "dart/output")
            owner: Lock owner identifier (e.g., workflow ID)
            ttl_seconds: Lock time-to-live in seconds (default: 3600)

        Returns:
            Lock status and expiration time
        """
        try:
            locks_dir = STATE_PATH / "locks"
            locks_dir.mkdir(parents=True, exist_ok=True)

            lock_file = locks_dir / f"{resource.replace('/', '_')}.lock"

            # Check existing lock
            if lock_file.exists():
                with open(lock_file) as f:
                    existing = json.load(f)

                expires = datetime.fromisoformat(existing["expires"])
                if datetime.now() < expires:
                    return json.dumps({
                        "success": False,
                        "error": "Resource locked",
                        "current_owner": existing["owner"],
                        "expires": existing["expires"]
                    })

            # Acquire lock
            lock_data = {
                "resource": resource,
                "owner": owner,
                "acquired": datetime.now().isoformat(),
                "expires": (datetime.now().replace(microsecond=0) +
                           timedelta(seconds=ttl_seconds)).isoformat()
            }

            atomic_write_json(lock_file, lock_data)

            return json.dumps({
                "success": True,
                "lock": lock_data
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def release_batch_lock(resource: str, owner: str) -> str:
        """
        Release a batch lock.

        Args:
            resource: Resource identifier
            owner: Lock owner (must match to release)

        Returns:
            Release confirmation
        """
        try:
            locks_dir = STATE_PATH / "locks"
            lock_file = locks_dir / f"{resource.replace('/', '_')}.lock"

            if not lock_file.exists():
                return json.dumps({"success": True, "message": "No lock exists"})

            with open(lock_file) as f:
                existing = json.load(f)

            if existing["owner"] != owner:
                return json.dumps({
                    "success": False,
                    "error": "Not lock owner",
                    "current_owner": existing["owner"]
                })

            lock_file.unlink()

            return json.dumps({
                "success": True,
                "message": f"Lock released for {resource}"
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def execute_workflow_task(
        workflow_id: str,
        task_id: str
    ) -> str:
        """
        Execute a pending task by invoking its mapped agent tool.

        This tool maps agent types to MCP tools and executes them.
        Tasks are tracked in the workflow state file.

        Args:
            workflow_id: Parent workflow ID
            task_id: Task ID to execute

        Returns:
            Execution result with status and output
        """
        try:
            from orchestrator.core.executor import TaskExecutor, AGENT_TOOL_MAPPING

            # Load task from workflow state
            workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})

            with open(workflow_path) as f:
                workflow = json.load(f)

            # Find task
            task = None
            for t in workflow.get("tasks", []):
                if t.get("id") == task_id:
                    task = t
                    break

            if not task:
                return json.dumps({"error": f"Task not found: {task_id}"})

            agent_type = task.get("agent_type", "")
            tool_name = AGENT_TOOL_MAPPING.get(agent_type)

            if not tool_name:
                return json.dumps({
                    "error": f"No tool mapping for agent type: {agent_type}",
                    "available_mappings": list(AGENT_TOOL_MAPPING.keys())
                })

            # Update task status to IN_PROGRESS
            task["status"] = "IN_PROGRESS"
            task["started_at"] = datetime.now().isoformat()
            workflow["progress"]["in_progress"] = sum(
                1 for t in workflow["tasks"] if t.get("status") == "IN_PROGRESS"
            )
            workflow["updated_at"] = datetime.now().isoformat()

            atomic_write_json(workflow_path, workflow)

            _get_tracker().update_progress_md()

            return json.dumps({
                "success": True,
                "task_id": task_id,
                "workflow_id": workflow_id,
                "agent_type": agent_type,
                "tool_name": tool_name,
                "status": "IN_PROGRESS",
                "message": f"Task execution started. Tool '{tool_name}' will handle agent '{agent_type}'.",
                "task_prompt": task.get("prompt", "")[:200] + "..." if len(task.get("prompt", "")) > 200 else task.get("prompt", "")
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def complete_workflow_task(
        workflow_id: str,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Mark a workflow task as complete or failed.

        Args:
            workflow_id: Parent workflow ID
            task_id: Task ID to update
            status: New status ("COMPLETE" or "ERROR")
            result: JSON string with task result (for COMPLETE)
            error: Error message (for ERROR)

        Returns:
            Updated task status
        """
        try:
            workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})

            with open(workflow_path) as f:
                workflow = json.load(f)

            # Find and update task
            task_found = False
            for task in workflow.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = status
                    task["completed_at"] = datetime.now().isoformat()

                    if result:
                        try:
                            task["result"] = json.loads(result)
                        except json.JSONDecodeError:
                            task["result"] = {"raw": result}

                    if error:
                        task["error"] = error

                    task_found = True
                    break

            if not task_found:
                return json.dumps({"error": f"Task not found: {task_id}"})

            # Update progress counters
            tasks = workflow.get("tasks", [])
            workflow["progress"]["completed"] = sum(
                1 for t in tasks if t.get("status") == "COMPLETE"
            )
            workflow["progress"]["in_progress"] = sum(
                1 for t in tasks if t.get("status") == "IN_PROGRESS"
            )
            workflow["progress"]["failed"] = sum(
                1 for t in tasks if t.get("status") == "ERROR"
            )
            workflow["updated_at"] = datetime.now().isoformat()

            # Check if workflow is complete
            total = workflow["progress"].get("total", 0)
            completed = workflow["progress"].get("completed", 0)
            failed = workflow["progress"].get("failed", 0)

            if completed + failed >= total:
                workflow["status"] = "COMPLETE" if failed == 0 else "PARTIAL"

            atomic_write_json(workflow_path, workflow)

            _get_tracker().update_progress_md()

            return json.dumps({
                "success": True,
                "task_id": task_id,
                "status": status,
                "workflow_status": workflow["status"],
                "progress": workflow["progress"]
            })

        except Exception as e:
            return json.dumps({"error": str(e)})
