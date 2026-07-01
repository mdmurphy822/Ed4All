"""
APIDispatcher — runs phase workers as Python coroutines (api mode).

When ``--mode api``, the orchestrator is a long-running Python process and
each phase is executed as a coroutine in-process. Workers that need LLM
access pull a backend from the injected factory (typically
:class:`AnthropicBackend`).

This dispatcher intentionally stays thin in Wave 7: the actual phase
execution still goes through the existing ``WorkflowRunner`` engine, which
has all the state-persistence, gate-running, and retry logic we want. The
dispatcher's contribution is the hook surface — ``before_run``, ``after_run``,
``on_error`` — plus ``dispatch_phase`` for tests and future waves that
bypass ``WorkflowRunner`` for certain phases (e.g., a content-generation
phase that wants raw coroutine parallelism across weeks).

Concurrency is bounded by ``phase_config.max_concurrent`` when the
dispatcher runs a phase's tasks directly; falls back to the config default
when absent.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from MCP.core.config import OrchestratorConfig
from MCP.core.executor import TaskExecutor

from .worker_contracts import PhaseInput, PhaseOutput

logger = logging.getLogger(__name__)

# Env flag: set truthy to allow the stub ``APIDispatcher.dispatch_task`` path
# to return a fake ``success=True`` envelope without doing real work. Default
# off so production ``--mode api`` runs (Wave 74 per-task subagent dispatch,
# ``ED4ALL_AGENT_DISPATCH=true``) fail loudly instead of silently succeeding
# with empty outputs. Reuses ``LocalDispatcher``'s stub-allow env so the two
# dispatchers share one opt-in. Tests set this to exercise the stub path.
_ALLOW_STUB_ENV = "LOCAL_DISPATCHER_ALLOW_STUB"


def _stub_allowed() -> bool:
    """Return True when the ungated-stub opt-in env is truthy.

    Parse-with-fallback: any non-truthy / garbage token → False (off).
    """
    return os.environ.get(_ALLOW_STUB_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class APIDispatcherStubNotAllowed(RuntimeError):
    """Raised when ``dispatch_task`` would emit an ungated fake-success stub.

    The Session-1 ``dispatch_task`` has no real per-agent LLM round-trip yet
    (Session 2 wires the prompt templates). Returning ``success=True`` with
    empty outputs on a real run silently fakes success. This guard makes that
    path opt-in via ``LOCAL_DISPATCHER_ALLOW_STUB``; without it, dispatch
    fails closed.
    """


class APIDispatcher:
    """Dispatches phase workers as coroutines (api mode)."""

    def __init__(
        self,
        *,
        llm_factory: Optional[Callable[[], Any]] = None,
        executor: Optional[TaskExecutor] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.llm_factory = llm_factory
        self.executor = executor
        self.config = config
        self._dispatched: List[str] = []

    # ------------------------------------------------- orchestrator hooks

    async def before_run(
        self, *, workflow_id: str, state: Dict[str, Any]
    ) -> None:
        logger.info("APIDispatcher starting workflow %s (api mode)", workflow_id)

    async def after_run(
        self, *, workflow_id: str, result: Dict[str, Any]
    ) -> List[str]:
        logger.info(
            "APIDispatcher completed workflow %s (status=%s)",
            workflow_id,
            result.get("status"),
        )
        return list(self._dispatched)

    async def on_error(self, *, workflow_id: str, error: str) -> None:
        logger.error("APIDispatcher workflow %s errored: %s", workflow_id, error)

    # ------------------------------------------------------------ dispatch

    async def dispatch_phase(
        self,
        phase_input: PhaseInput,
        *,
        worker: Optional[Callable[[PhaseInput], Awaitable[PhaseOutput]]] = None,
    ) -> PhaseOutput:
        """Run a phase in-process as a coroutine.

        ``worker`` is the async callable that actually performs the phase
        work. If omitted, the dispatcher emits a stub PhaseOutput (useful for
        tests that want to verify plumbing without real work happening).

        Concurrency: the dispatcher honors ``phase_config.max_concurrent``
        only when the worker handles its own per-task parallelism. For
        single-task workers, the coroutine is awaited directly.
        """
        self._dispatched.append(phase_input.phase_name)

        if worker is None:
            logger.info(
                "APIDispatcher: no worker passed for phase=%s; returning stub",
                phase_input.phase_name,
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                outputs={"dispatch_mode": "stub"},
                status="ok",
            )

        try:
            return await worker(phase_input)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "APIDispatcher: worker raised for %s", phase_input.phase_name
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=str(exc),
            )

    # Wave 74: per-task subagent dispatch parity with LocalDispatcher.
    # In api mode the expected path is to call the injected LLMBackend
    # directly inside the agent-style tool rather than round-tripping
    # through an external operator. Session 1 lands a minimal contract-
    # conformant implementation that degrades to the stub shape — api
    # mode callers can switch to ``ED4ALL_AGENT_DISPATCH=true`` without
    # changing the orchestrator contract, and Session 2 wires real
    # per-agent prompt templates (same templates local-mode operators
    # use) so the same subagent logic executes against the SDK instead
    # of an outer Claude Code session.
    async def dispatch_task(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch one phase task to an in-process agent coroutine.

        Wave 74 Session 1: the agent-prompt templates + LLMBackend
        round-trip land in Session 2, so there is no real per-agent work
        to do here yet. Rather than silently return a fake ``success``
        envelope (which makes a real ``--mode api`` run appear to succeed
        with empty outputs), this path is gated behind
        ``LOCAL_DISPATCHER_ALLOW_STUB`` (mirroring ``LocalDispatcher``):

        * flag truthy → emit the stub envelope (shape matches what
          ``LocalDispatcher.dispatch_task`` emits under the same flag),
          letting the executor routing fork round-trip in dry-run / CI.
        * flag off (default) → raise ``APIDispatcherStubNotAllowed`` so
          the run fails loudly instead of faking success.
        """
        self._dispatched.append(f"{agent_type}:{task_name}")

        if not _stub_allowed():
            msg = (
                "APIDispatcher.dispatch_task has no real agent round-trip "
                f"(Session-1 stub) for agent={agent_type} tool={task_name} "
                f"run_id={run_id}; refusing to fake success with empty "
                f"outputs. Set {_ALLOW_STUB_ENV}=1 to accept the stub "
                "envelope (dry-run / CI only)."
            )
            logger.error(msg)
            raise APIDispatcherStubNotAllowed(msg)

        logger.info(
            "APIDispatcher.dispatch_task (Session-1 stub) — "
            "%s set — agent=%s tool=%s run_id=%s",
            _ALLOW_STUB_ENV, agent_type, task_name, run_id,
        )
        return {
            "success": True,
            "dispatch_mode": "api_stub",
            "agent_type": agent_type,
            "tool_name": task_name,
            "outputs": {},
            "artifacts": [],
        }

    # ------------------------------------------------------------ parallel

    async def dispatch_batch(
        self,
        phase_inputs: List[PhaseInput],
        worker: Callable[[PhaseInput], Awaitable[PhaseOutput]],
        *,
        max_concurrent: int = 5,
    ) -> List[PhaseOutput]:
        """Run multiple phases concurrently with a semaphore.

        Useful when a single logical phase (e.g., content generation) is
        decomposed into many independent tasks.
        """
        sem = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def _guarded(pi: PhaseInput) -> PhaseOutput:
            async with sem:
                return await self.dispatch_phase(pi, worker=worker)

        results = await asyncio.gather(
            *[_guarded(pi) for pi in phase_inputs], return_exceptions=False
        )
        return list(results)
