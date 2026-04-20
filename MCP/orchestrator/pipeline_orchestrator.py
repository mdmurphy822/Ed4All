"""
PipelineOrchestrator — front controller for mode-agnostic workflow runs.

Sits on top of :class:`MCP.core.workflow_runner.WorkflowRunner`, reusing its
phase-ordering, state-persistence, and validation-gate logic verbatim. The
orchestrator's only responsibility is to choose *how* each phase runs:

- In ``local`` mode, phases are dispatched via ``LocalDispatcher``
  (Claude Code subagent calls).
- In ``api`` mode, phases run as Python coroutines via ``APIDispatcher``
  using an injected ``LLMBackend``.

Wave 7 wires the dispatcher selection end-to-end; for the default run path
we fall through to the existing ``WorkflowRunner.run_workflow`` engine,
which already handles in-process phase execution. This keeps the refactor
additive and preserves behavior for existing callers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from MCP.core.config import OrchestratorConfig
from MCP.core.executor import TaskExecutor
from MCP.core.workflow_runner import STATE_PATH, WorkflowRunner

from .llm_backend import BackendSpec, LLMBackend, build_backend
from .worker_contracts import PhaseInput, PhaseOutput

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Final result returned by ``PipelineOrchestrator.run()``."""

    workflow_id: str
    status: Literal["ok", "failed", "dry_run"]
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dispatched_phases: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "phase_results": self.phase_results,
            "phase_outputs": self.phase_outputs,
            "dispatched_phases": self.dispatched_phases,
            "error": self.error,
        }


class PipelineOrchestrator:
    """Mode-aware workflow runner.

    Construct with either an explicit ``llm_factory`` or a ``BackendSpec``
    describing how to build one. The orchestrator caches a single factory and
    hands it to dispatchers so they can mint fresh backend instances per
    phase (useful for isolating per-phase rate limits / model overrides).

    Design invariant: the orchestrator never calls LLMs directly. It only
    chooses a dispatcher and feeds it a factory.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        *,
        mode: Literal["local", "api"] = "local",
        llm_factory: Optional[Callable[[], LLMBackend]] = None,
        backend_spec: Optional[BackendSpec] = None,
        executor: Optional[TaskExecutor] = None,
        project_root: Optional[Path] = None,
    ):
        self.config = config or OrchestratorConfig.load()
        self.mode = mode
        self._executor = executor
        self._project_root = project_root

        if llm_factory is not None:
            self.llm_factory = llm_factory
        else:
            spec = backend_spec or BackendSpec(mode=mode)
            self.llm_factory = lambda spec=spec: build_backend(spec)

        self._dispatcher = None  # built lazily

    # ------------------------------------------------------------------ utils

    @property
    def project_root(self) -> Path:
        if self._project_root is None:
            self._project_root = STATE_PATH.parent
        return self._project_root

    @property
    def state_dir(self) -> Path:
        return STATE_PATH

    def _captures_dir(self, tool: str, course_code: str, phase: str) -> Path:
        captures = self.project_root / "training-captures" / tool / course_code
        return captures / f"phase_{phase}"

    # ---------------------------------------------------------- dispatcher

    def _get_dispatcher(self):
        """Lazily build the mode-appropriate dispatcher."""
        if self._dispatcher is not None:
            return self._dispatcher

        if self.mode == "local":
            # Import here to avoid circular imports at module load
            from .local_dispatcher import LocalDispatcher

            self._dispatcher = LocalDispatcher(llm_factory=self.llm_factory)
        elif self.mode == "api":
            from .api_dispatcher import APIDispatcher

            self._dispatcher = APIDispatcher(
                llm_factory=self.llm_factory,
                executor=self._get_executor(),
                config=self.config,
            )
        else:
            raise ValueError(f"Unknown orchestrator mode: {self.mode}")
        return self._dispatcher

    def _get_executor(self) -> TaskExecutor:
        if self._executor is None:
            # Fallback: build an executor wired with the full pipeline tool
            # registry so phase tasks can resolve their tool names. Without this,
            # `ed4all run` fails with "Tool not registered" at first phase.
            from MCP.tools.pipeline_tools import _build_tool_registry
            self._executor = TaskExecutor(tool_registry=_build_tool_registry())
        return self._executor

    # ---------------------------------------------------------------- plan

    def plan(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Return the planned phase sequence for ``workflow_id`` without running.

        Used by ``ed4all run --dry-run`` to preview the pipeline.
        """
        state = self._load_workflow_state(workflow_id)
        if state is None:
            return []

        workflow_type = state.get("type", "")
        wf_config = self.config.get_workflow(workflow_type)
        if not wf_config:
            return []

        runner = self._build_runner()
        phases = runner._topological_sort(wf_config.phases)
        plan = []
        for idx, phase in enumerate(phases):
            plan.append(
                {
                    "order": idx + 1,
                    "name": phase.name,
                    "agents": list(phase.agents),
                    "parallel": getattr(phase, "parallel", True),
                    "max_concurrent": getattr(phase, "max_concurrent", 5),
                    "depends_on": list(phase.depends_on or []),
                    "optional": bool(getattr(phase, "optional", False)),
                    "description": getattr(phase, "description", "") or "",
                }
            )
        return plan

    # ------------------------------------------------------------------ run

    async def run(self, workflow_id: str) -> OrchestratorResult:
        """Execute all phases of ``workflow_id`` via the mode's dispatcher.

        For Wave 7 we reuse the existing WorkflowRunner engine as the default
        execution path. The dispatcher is consulted to emit orchestrator-level
        decision captures and (in later waves) to replace in-process phase
        execution entirely with subagent / coroutine dispatch.
        """
        state = self._load_workflow_state(workflow_id)
        if state is None:
            return OrchestratorResult(
                workflow_id=workflow_id,
                status="failed",
                error=f"Workflow not found: {workflow_id}",
            )

        dispatcher = self._get_dispatcher()
        logger.info(
            "PipelineOrchestrator dispatching workflow %s in %s mode via %s",
            workflow_id,
            self.mode,
            type(dispatcher).__name__,
        )

        # Optional: pre-dispatch hook (used for decision capture + metrics)
        await dispatcher.before_run(workflow_id=workflow_id, state=state)

        runner = self._build_runner()
        try:
            raw = await runner.run_workflow(workflow_id)
        except Exception as exc:  # noqa: BLE001 — surface exact error to caller
            logger.exception("Workflow run crashed: %s", exc)
            await dispatcher.on_error(workflow_id=workflow_id, error=str(exc))
            return OrchestratorResult(
                workflow_id=workflow_id,
                status="failed",
                error=str(exc),
            )

        # Post-dispatch hook
        dispatched = await dispatcher.after_run(workflow_id=workflow_id, result=raw)

        status = "ok" if raw.get("status") == "COMPLETE" else "failed"
        return OrchestratorResult(
            workflow_id=workflow_id,
            status=status,
            phase_results=raw.get("phase_results", {}),
            phase_outputs=raw.get("phase_outputs", {}),
            dispatched_phases=list(dispatched or []),
            error=raw.get("error"),
        )

    # --------------------------------------------------------------- helpers

    def _build_runner(self) -> WorkflowRunner:
        return WorkflowRunner(self._get_executor(), self.config)

    def _load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # -------------------------------------------------------- phase input

    def build_phase_input(
        self,
        *,
        run_id: str,
        workflow_type: str,
        phase_name: str,
        phase_config: Dict[str, Any],
        params: Dict[str, Any],
        course_code: str = "UNKNOWN",
        tool: str = "orchestrator",
    ) -> PhaseInput:
        """Build a ``PhaseInput`` wired up to this orchestrator's environment.

        Dispatchers use this when they need to construct a fresh PhaseInput
        for a phase they're about to run.
        """
        return PhaseInput(
            run_id=run_id,
            workflow_type=workflow_type,
            phase_name=phase_name,
            phase_config=phase_config,
            params=params,
            mode=self.mode,
            llm_factory=self.llm_factory,
            project_root=self.project_root,
            state_dir=self.state_dir,
            captures_dir=self._captures_dir(tool, course_code, phase_name),
        )

    # ---------------------------------------------------------- diagnostics

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable snapshot of the orchestrator's config."""
        return {
            "mode": self.mode,
            "dispatcher": type(self._get_dispatcher()).__name__,
            "project_root": str(self.project_root),
            "state_dir": str(self.state_dir),
            "backend_factory": repr(self.llm_factory),
            "timestamp": datetime.now().isoformat(),
        }
