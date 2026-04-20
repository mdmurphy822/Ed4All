"""
LocalDispatcher — dispatches phase workers as Claude Code subagents.

In ``--mode local`` runs, the enclosing Claude Code session is the
orchestrator. Each phase is represented by an agent spec (``*.md`` file
under ``Courseforge/agents/``, ``Trainforge/agents/``, etc.) and the
dispatcher's job is to (a) build the subagent prompt, (b) invoke the MCP
``Agent`` tool, and (c) parse the returned JSON into a ``PhaseOutput``.

Wave 7 behavior: ``dispatch_phase`` is scaffolded with the correct
contract (prompt construction, JSON parse, PhaseOutput), but falls back to
recording dispatch intent when no ``Agent`` callable is wired in. This
keeps the abstraction testable without requiring a live MCP server during
unit tests — and leaves a clear extension point for later waves that
actually bridge orchestrator code to the running session.

Hooks (``before_run``, ``after_run``, ``on_error``) exist so the
orchestrator can emit decision-capture events for dispatch decisions
without coupling the capture machinery to this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .worker_contracts import PhaseInput, PhaseOutput

logger = logging.getLogger(__name__)


# Registry of known agent-spec directories. The dispatcher searches these
# in order when it needs to resolve an agent name to a markdown spec.
AGENT_SPEC_DIRS = (
    Path("Courseforge") / "agents",
    Path("Trainforge") / "agents",
    Path("DART") / "agents",  # may not exist; dispatcher degrades gracefully
)


class LocalDispatcher:
    """Dispatches phase workers as Claude Code subagents (local mode)."""

    def __init__(
        self,
        llm_factory: Optional[Callable[[], Any]] = None,
        *,
        agent_tool: Optional[Callable[[Dict[str, Any]], Awaitable[str]]] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Args:
            llm_factory: Factory producing ``LLMBackend`` instances. Local
                subagents have their own session context for LLM work, so the
                factory is typically unused here — but dispatcher tests and
                API-style fallback paths need it.
            agent_tool: Async callable that invokes the MCP ``Agent`` tool
                with a dict of ``{subagent_type, prompt, ...}`` and returns
                the subagent's JSON response as a string. Injected so unit
                tests can stub dispatch without a live server.
            project_root: Project root for resolving agent spec paths.
        """
        self.llm_factory = llm_factory
        self.agent_tool = agent_tool
        self.project_root = project_root or Path.cwd()
        self._dispatched: List[str] = []

    # ------------------------------------------------- orchestrator hooks

    async def before_run(
        self, *, workflow_id: str, state: Dict[str, Any]
    ) -> None:
        logger.info("LocalDispatcher starting workflow %s", workflow_id)

    async def after_run(
        self, *, workflow_id: str, result: Dict[str, Any]
    ) -> List[str]:
        logger.info(
            "LocalDispatcher completed workflow %s (status=%s, phases=%d)",
            workflow_id,
            result.get("status"),
            len(result.get("phase_outputs", {})),
        )
        return list(self._dispatched)

    async def on_error(self, *, workflow_id: str, error: str) -> None:
        logger.error("LocalDispatcher workflow %s errored: %s", workflow_id, error)

    # ------------------------------------------------------------ dispatch

    async def dispatch_phase(self, phase_input: PhaseInput) -> PhaseOutput:
        """Dispatch a single phase to a subagent and collect its PhaseOutput.

        If ``agent_tool`` was not provided, returns a stub PhaseOutput that
        records dispatch intent (status="ok", empty outputs). This keeps the
        dispatcher useful in environments where the Agent tool isn't directly
        callable (e.g., unit tests) without silently dropping work.
        """
        self._dispatched.append(phase_input.phase_name)

        prompt = self._build_subagent_prompt(phase_input)

        if self.agent_tool is None:
            logger.info(
                "LocalDispatcher: no agent_tool wired; emitting stub PhaseOutput "
                "for phase=%s (intended subagent_type=%s)",
                phase_input.phase_name,
                self._resolve_agent_type(phase_input),
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                outputs={"dispatch_mode": "stub", "prompt_preview": prompt[:160]},
                status="ok",
            )

        subagent_request = {
            "subagent_type": self._resolve_agent_type(phase_input),
            "prompt": prompt,
            "phase_input": phase_input.to_dict(),
        }
        try:
            raw = await self.agent_tool(subagent_request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LocalDispatcher: agent_tool raised for %s", phase_input.phase_name)
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=str(exc),
            )

        return self._parse_subagent_response(raw, phase_input)

    # --------------------------------------------------------------- utils

    def _resolve_agent_type(self, phase_input: PhaseInput) -> str:
        agents = phase_input.phase_config.get("agents") or []
        if not agents:
            return "general-purpose"
        return agents[0]

    def _build_subagent_prompt(self, phase_input: PhaseInput) -> str:
        """Assemble a prompt that includes (a) the agent spec, (b) the phase
        config, and (c) the routed params.

        The spec is loaded from ``<project>/<AGENT_SPEC_DIRS>/<name>.md`` if
        available; otherwise we fall back to a minimal prompt.
        """
        agent_type = self._resolve_agent_type(phase_input)
        spec_text = self._load_agent_spec(agent_type)

        header = (
            f"# Phase: {phase_input.phase_name}\n"
            f"Workflow: {phase_input.workflow_type}  Run: {phase_input.run_id}\n"
        )
        params_block = (
            "## Routed params\n"
            "```json\n"
            f"{json.dumps(phase_input.params, indent=2, default=str)}\n"
            "```\n"
        )
        spec_block = f"## Agent spec: {agent_type}\n{spec_text}\n"

        return f"{header}\n{spec_block}\n{params_block}\n"

    def _load_agent_spec(self, agent_type: str) -> str:
        """Locate and read the agent spec markdown file."""
        for base in AGENT_SPEC_DIRS:
            candidate = self.project_root / base / f"{agent_type}.md"
            if candidate.exists():
                try:
                    return candidate.read_text(encoding="utf-8")
                except OSError as exc:  # pragma: no cover
                    logger.warning("Could not read agent spec %s: %s", candidate, exc)
                    continue
        return f"(no spec file found for agent '{agent_type}')"

    def _parse_subagent_response(
        self, raw: str, phase_input: PhaseInput
    ) -> PhaseOutput:
        """Parse the subagent JSON response into a ``PhaseOutput``.

        Subagents are expected to return a JSON object matching
        :meth:`PhaseOutput.to_dict`. If parsing fails, we still return a
        PhaseOutput so the workflow can continue (status=fail), rather than
        raising and tearing down the whole run.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "LocalDispatcher: could not parse subagent response for %s: %s",
                phase_input.phase_name,
                exc,
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=f"invalid subagent JSON: {exc}",
            )

        if not isinstance(data, dict):
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error="subagent response was not a JSON object",
            )

        # Ensure run_id / phase_name agree with what we dispatched
        data.setdefault("run_id", phase_input.run_id)
        data.setdefault("phase_name", phase_input.phase_name)
        return PhaseOutput.from_dict(data)
