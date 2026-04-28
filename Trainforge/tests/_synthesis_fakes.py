"""Test fakes for ClaudeSessionProvider unit tests.

The production provider holds a ``LocalDispatcher`` reference and awaits
``dispatcher.dispatch_task(...)`` per paraphrase call. Tests inject
``FakeLocalDispatcher`` instead so the dispatch path is exercised without
a real subagent / mailbox.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional


class FakeLocalDispatcher:
    """Minimal stand-in for ``MCP.orchestrator.local_dispatcher.LocalDispatcher``.

    Holds a user-supplied async callable that returns the dispatch result.
    Records every dispatch call as ``(agent_type, task_params)`` tuples
    on ``self.calls`` so tests can assert payload shape.
    """

    def __init__(
        self,
        agent_tool: Callable[..., Awaitable[Dict[str, Any]]],
    ) -> None:
        self._agent_tool = agent_tool
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    async def dispatch_task(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append((agent_type, dict(task_params)))
        return await self._agent_tool(
            task_name=task_name,
            agent_type=agent_type,
            task_params=task_params,
            run_id=run_id,
            phase_context=phase_context,
        )


def make_instruction_response(prompt: str, completion: str) -> Dict[str, Any]:
    """Shape the dispatcher returns for a successful instruction paraphrase."""
    return {
        "success": True,
        "outputs": {"prompt": prompt, "completion": completion},
        "artifacts": [],
    }


def make_preference_response(
    prompt: str, chosen: str, rejected: str
) -> Dict[str, Any]:
    """Shape the dispatcher returns for a successful preference paraphrase."""
    return {
        "success": True,
        "outputs": {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        "artifacts": [],
    }


def make_failure_response(error: str, error_code: str = "AGENT_ERROR") -> Dict[str, Any]:
    """Shape the dispatcher returns when the subagent reports failure."""
    return {
        "success": False,
        "error_code": error_code,
        "error": error,
    }
