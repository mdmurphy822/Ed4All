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
    """Final result returned by ``PipelineOrchestrator.run()``.

    Wave 29 Defect 3: carries an aggregated ``gates_passed`` flag so
    downstream callers (CLI, programmatic consumers) don't have to
    re-scan every phase to know whether gates ran cleanly. Aggregation
    rule: ``gates_passed`` is ``True`` iff every phase that reports the
    flag reports ``True`` — phases without a ``gates_passed`` entry
    (no gates configured) are treated as passing.
    """

    workflow_id: str
    status: Literal["ok", "failed", "dry_run"]
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dispatched_phases: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def gates_passed(self) -> bool:
        """Return True iff no phase reported a gate failure."""
        for info in (self.phase_results or {}).values():
            if not isinstance(info, dict):
                continue
            if info.get("gates_passed") is False:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "phase_results": self.phase_results,
            "phase_outputs": self.phase_outputs,
            "dispatched_phases": self.dispatched_phases,
            "error": self.error,
            "gates_passed": self.gates_passed,
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

    def _get_dispatcher(
        self,
        workflow_state: Optional[Dict[str, Any]] = None,
    ):
        """Lazily build the mode-appropriate dispatcher.

        Wave 23 Sub-task B: forwards ``workflow_state`` into
        ``_get_executor`` so the API dispatcher gets a properly-wired
        executor (run_id / run_path / capture) rather than the
        timestamp-orphan fallback.
        """
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
                executor=self._get_executor(workflow_state=workflow_state),
                config=self.config,
            )
        else:
            raise ValueError(f"Unknown orchestrator mode: {self.mode}")
        return self._dispatcher

    def _get_executor(
        self,
        workflow_state: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutor:
        """Return a ``TaskExecutor`` wired for a specific workflow.

        Wave 23 Sub-task B: pre-Wave-23 this method built an executor
        with no ``run_id``, no ``run_path``, and no ``capture=``. That
        left ``TaskExecutor.run_id`` auto-generating from a timestamp
        (so checkpoints landed in an orphan ``state/runs/run_{ts}/``
        dir nobody ever reads), and ``self.capture is None`` meant the
        ``phase_start`` / ``phase_completion`` / ``task_retry`` /
        ``workflow_execution`` emit sites at
        ``MCP/core/executor.py:728, 875, 981`` never fired.

        When a workflow state is known, this method now resolves the
        workflow's actual ``params.run_id`` (e.g. the
        ``TTC_{course_name}_{timestamp}`` run IDs minted by
        ``create_textbook_pipeline``) and builds:

        - ``run_path`` at ``state/runs/{run_id}/`` — matches what
          ``CheckpointManager`` + ``LockfileManager`` use.
        - ``capture = DecisionCapture(course_code=normalize_course_code(...),
          phase="orchestrator", tool="pipeline", ...)`` — the course
          code is normalised via the Wave-22 DC4 pattern so the
          orchestrator capture doesn't re-introduce the
          ``course_id`` validation-issue noise the previous wave fixed.

        Legacy callers that don't supply ``workflow_state`` (mostly
        tests) still get a bare executor. A cached executor is
        returned on repeat calls so the executor identity survives
        across dispatcher callbacks.
        """
        if self._executor is not None:
            return self._executor

        # Fallback: build an executor wired with the full pipeline tool
        # registry so phase tasks can resolve their tool names. Without this,
        # `ed4all run` fails with "Tool not registered" at first phase.
        from MCP.tools.pipeline_tools import _build_tool_registry

        tool_registry = _build_tool_registry()

        run_id: Optional[str] = None
        run_path = None
        capture = None
        if workflow_state is not None:
            params = workflow_state.get("params") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (ValueError, TypeError):
                    params = {}
            # Prefer an explicit run_id emitted by the workflow
            # creator (e.g. ``TTC_{course}_{ts}`` from
            # ``pipeline_tools.create_textbook_pipeline``); fall back
            # to the workflow_id which is always present.
            run_id = params.get("run_id") if isinstance(params, dict) else None
            if not run_id:
                run_id = workflow_state.get("workflow_id") or workflow_state.get("id")

            run_path = (self.state_dir / "runs" / run_id) if run_id else None

            # Build orchestrator-level decision capture. Best-effort —
            # a DecisionCapture construction failure must not block
            # the executor build.
            #
            # Wave 29 Defect 5: prefer the canonical course code pinned
            # onto ``params.canonical_course_code`` at workflow-creation
            # time (see ``MCP/tools/pipeline_tools.py::create_textbook_pipeline``).
            # That single source of truth is normalised ONCE at creation
            # from ``params.course_name`` so the orchestrator capture
            # doesn't re-normalise and drift out of alignment with the
            # captures downstream CF/TF phases create from the same
            # ``course_name``. Falls back to on-the-fly normalisation
            # when the canonical code isn't available (legacy workflow
            # states created before this change).
            canonical_cc = None
            if isinstance(params, dict):
                canonical_cc = params.get("canonical_course_code")
            course_code_raw = (
                params.get("course_name") if isinstance(params, dict) else None
            ) or (workflow_state.get("type") or "PIPELINE")
            try:
                from lib.decision_capture import (
                    DecisionCapture,
                    normalize_course_code,
                )

                if canonical_cc:
                    cc = canonical_cc
                else:
                    cc = normalize_course_code(str(course_code_raw))
                    logger.debug(
                        "DC5 fallback: workflow_state missing "
                        "canonical_course_code; derived %s from %r",
                        cc,
                        course_code_raw,
                    )

                capture = DecisionCapture(
                    course_code=cc,
                    phase="orchestrator",
                    tool="pipeline",
                    streaming=True,
                )
            except Exception as exc:  # noqa: BLE001 — capture is best-effort
                logger.debug(
                    "DecisionCapture construction failed in _get_executor: %s",
                    exc,
                )
                capture = None

        self._executor = TaskExecutor(
            tool_registry=tool_registry,
            run_id=run_id,
            run_path=run_path,
            capture=capture,
        )
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

        runner = self._build_runner(workflow_state=state)
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

        ⚠  **Known architectural gap (Wave 38 code-review finding #2):**
        ``WorkflowRunner.run_workflow`` → ``TaskExecutor.execute_phase`` →
        ``_invoke_tool`` invokes the Python tool registry directly. It
        does NOT call ``dispatcher.dispatch_phase``. That means:

        * ``--mode local`` and ``--mode api`` execute the *same* code
          path today — the mode choice only affects which dispatcher
          receives ``before_run`` / ``after_run`` / ``on_error`` hooks.
        * Wave 34's ``TaskMailbox`` bridge + ``ed4all mailbox watch``
          CLI are reachable only via the dispatcher's ``dispatch_phase``
          method, which is currently only exercised in unit tests
          (``MCP/tests/test_local_dispatcher_*``, ``test_mailbox_bridge_smoke``).
        * Subagent prompt construction
          (``LocalDispatcher._build_subagent_prompt``) and the stub /
          mailbox / injected ``agent_tool`` three-path selection are
          dead code on the production run path.

        Closing the gap means routing ``execute_phase`` — or the per-task
        invocation inside it — through ``dispatcher.dispatch_phase``
        when the dispatcher opts into task-level execution. That's a
        non-trivial refactor because ``dispatch_phase`` is per-phase
        and our phases routinely fan out to 10+ concurrent tasks
        (content_generation, etc.). The clean design is either (a)
        per-task dispatch via a new ``dispatcher.dispatch_task`` hook
        or (b) folding the task-fan-out into the dispatcher. Either
        way affects every existing ``TaskExecutor`` callsite + test.

        Scope for a later wave: not blocking current pipeline runs
        (the templated path produces real grounded content via the
        in-process tool registry — see Wave 35 / sim-05 diagnostics).
        """
        state = self._load_workflow_state(workflow_id)
        if state is None:
            return OrchestratorResult(
                workflow_id=workflow_id,
                status="failed",
                error=f"Workflow not found: {workflow_id}",
            )

        dispatcher = self._get_dispatcher(workflow_state=state)
        logger.info(
            "PipelineOrchestrator dispatching workflow %s in %s mode via %s",
            workflow_id,
            self.mode,
            type(dispatcher).__name__,
        )

        # Optional: pre-dispatch hook (used for decision capture + metrics)
        await dispatcher.before_run(workflow_id=workflow_id, state=state)

        runner = self._build_runner(workflow_state=state)
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

    def _build_runner(
        self,
        workflow_state: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRunner:
        return WorkflowRunner(
            self._get_executor(workflow_state=workflow_state),
            self.config,
        )

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
