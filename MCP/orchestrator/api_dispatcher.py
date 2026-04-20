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
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from MCP.core.config import OrchestratorConfig
from MCP.core.executor import TaskExecutor

from .worker_contracts import PhaseInput, PhaseOutput

logger = logging.getLogger(__name__)


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
