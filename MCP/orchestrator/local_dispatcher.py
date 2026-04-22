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

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .task_mailbox import TaskMailbox
from .worker_contracts import PhaseInput, PhaseOutput

logger = logging.getLogger(__name__)

# Env flag: set to "1" to allow the stub ``LocalDispatcher`` path to return
# ``status="ok"`` without firing a real subagent. Default off so production
# ``--mode local`` runs fail loudly instead of emitting empty phase outputs
# while appearing to succeed. Tests set this to exercise the stub path.
_ALLOW_STUB_ENV = "LOCAL_DISPATCHER_ALLOW_STUB"

# Default timeout (seconds) for mailbox-brokered subagent tasks. The outer
# Claude Code watcher script reads a task spec, dispatches a subagent via
# the ``Agent`` tool, and writes the result back. 600s gives room for
# content-generation-size tasks without pinning the dispatcher forever.
_DEFAULT_MAILBOX_TIMEOUT = 600.0


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
        mailbox_base_dir: Optional[Path] = None,
        mailbox_timeout_seconds: float = _DEFAULT_MAILBOX_TIMEOUT,
        mailbox_poll_interval: float = 0.25,
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
                tests can stub dispatch without a live server. When present,
                the dispatcher bypasses the mailbox bridge entirely.
            project_root: Project root for resolving agent spec paths.
            mailbox_base_dir: Parent dir for ``TaskMailbox``. Defaults to
                ``<project_root>/state/runs``. Tests pass a tmp_path.
            mailbox_timeout_seconds: Max seconds to wait for an outer
                watcher to complete a mailbox task. Default 600s.
            mailbox_poll_interval: Poll cadence for
                ``wait_for_completion`` (kept short in tests).
        """
        self.llm_factory = llm_factory
        self.agent_tool = agent_tool
        self.project_root = project_root or Path.cwd()
        self.mailbox_base_dir = (
            Path(mailbox_base_dir)
            if mailbox_base_dir is not None
            else self.project_root / "state" / "runs"
        )
        self.mailbox_timeout_seconds = float(mailbox_timeout_seconds)
        self.mailbox_poll_interval = float(mailbox_poll_interval)
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

        Three-path dispatch (Wave 34):

        1. ``agent_tool`` callable injected → call it directly. This is the
           test-harness path and preserves existing Wave 28 wiring.
        2. ``agent_tool`` missing AND ``LOCAL_DISPATCHER_ALLOW_STUB=1`` →
           return a stub ``PhaseOutput`` so dry-run / unit tests exercise
           dispatch without a real subagent pathway.
        3. ``agent_tool`` missing AND stub flag off → write the task spec
           to ``state/runs/{run_id}/mailbox/pending/{task_id}.json`` and
           block on ``TaskMailbox.wait_for_completion``. An outer Claude
           Code session (see ``ed4all mailbox watch``) claims the pending
           task, dispatches a real subagent via the ``Agent`` tool, and
           writes the result to ``completed/``. Timeout → ``status="fail"``
           with ``error_code=MAILBOX_TIMEOUT``.
        """
        self._dispatched.append(phase_input.phase_name)

        prompt = self._build_subagent_prompt(phase_input)
        agent_type = self._resolve_agent_type(phase_input)

        if self.agent_tool is None:
            allow_stub = os.environ.get(_ALLOW_STUB_ENV, "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            if allow_stub:
                logger.info(
                    "LocalDispatcher: no agent_tool wired; "
                    "LOCAL_DISPATCHER_ALLOW_STUB set — emitting stub "
                    "PhaseOutput for phase=%s (subagent_type=%s)",
                    phase_input.phase_name,
                    agent_type,
                )
                return PhaseOutput(
                    run_id=phase_input.run_id,
                    phase_name=phase_input.phase_name,
                    outputs={
                        "dispatch_mode": "stub",
                        "prompt_preview": prompt[:160],
                    },
                    status="ok",
                )

            # Mailbox bridge path: hand the task off to an outer watcher.
            return await self._dispatch_via_mailbox(
                phase_input=phase_input,
                prompt=prompt,
                agent_type=agent_type,
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

    async def _dispatch_via_mailbox(
        self,
        *,
        phase_input: PhaseInput,
        prompt: str,
        agent_type: str,
    ) -> PhaseOutput:
        """Write a pending task to the mailbox and wait for an outer watcher
        to complete it.

        The completion envelope shape the watcher writes is:

            {
              "success": bool,
              "result": <subagent_json_response_parsed_as_dict> | None,
              "raw": <raw_string_if_available>,
              "error": <string_if_success_false>,
              "error_code": <classifier_tag_optional>
            }

        ``result`` is preferred: when present it is passed through
        ``PhaseOutput.from_dict`` after run_id / phase_name defaults are
        set. If only ``raw`` is present we parse it through
        ``_parse_subagent_response`` so the watcher has a lightweight
        fallback path.
        """
        mailbox = TaskMailbox(
            run_id=phase_input.run_id, base_dir=self.mailbox_base_dir,
        )

        # Task id is phase-scoped so the same run can have multiple
        # pending tasks for different phases. The uuid suffix prevents
        # accidental collisions if the same phase runs twice (e.g. retry).
        task_id = f"{phase_input.phase_name}-{uuid.uuid4().hex[:8]}"

        task_spec = {
            "subagent_type": agent_type,
            "prompt": prompt,
            "phase_input": phase_input.to_dict(),
        }
        try:
            mailbox.put_pending(task_id, task_spec)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LocalDispatcher: failed to write pending mailbox task for %s",
                phase_input.phase_name,
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=f"mailbox put_pending failed: {exc}",
            )

        logger.info(
            "LocalDispatcher: mailbox task %s queued for phase=%s (timeout=%.0fs)",
            task_id,
            phase_input.phase_name,
            self.mailbox_timeout_seconds,
        )

        # Run the blocking wait in a thread so we don't stall the event
        # loop. wait_for_completion uses short sleeps internally.
        loop = asyncio.get_event_loop()
        try:
            envelope = await loop.run_in_executor(
                None,
                lambda: mailbox.wait_for_completion(
                    task_id,
                    timeout_seconds=self.mailbox_timeout_seconds,
                    poll_interval=self.mailbox_poll_interval,
                ),
            )
        except TimeoutError:
            logger.error(
                "LocalDispatcher: mailbox timeout for task=%s phase=%s",
                task_id,
                phase_input.phase_name,
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=(
                    f"MAILBOX_TIMEOUT: no completion from outer watcher within "
                    f"{self.mailbox_timeout_seconds:.0f}s for task {task_id!r}. "
                    f"Recovery: run `ed4all mailbox watch --run-id "
                    f"{phase_input.run_id}` in a Claude Code session, or inject "
                    f"an agent_tool callable, or rerun with --mode api."
                ),
                metrics={"error_code": "MAILBOX_TIMEOUT", "mailbox_task_id": task_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LocalDispatcher: mailbox read failed for %s", task_id
            )
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=f"mailbox wait_for_completion failed: {exc}",
                metrics={"mailbox_task_id": task_id},
            )
        finally:
            # Completion envelope is read; prune the per-task files so the
            # mailbox stays bounded across long runs.
            try:
                mailbox.cleanup(task_id)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "LocalDispatcher: cleanup failed for %s (non-fatal)",
                    task_id,
                )

        return self._phase_output_from_envelope(envelope, phase_input, task_id)

    def _phase_output_from_envelope(
        self,
        envelope: Dict[str, Any],
        phase_input: PhaseInput,
        task_id: str,
    ) -> PhaseOutput:
        """Convert a mailbox completion envelope into a ``PhaseOutput``."""
        if not isinstance(envelope, dict):
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error="mailbox envelope was not a JSON object",
                metrics={"mailbox_task_id": task_id},
            )

        if not envelope.get("success", False):
            err = envelope.get("error") or "outer watcher reported failure"
            code = envelope.get("error_code")
            metrics = {"mailbox_task_id": task_id}
            if code:
                metrics["error_code"] = code
            return PhaseOutput(
                run_id=phase_input.run_id,
                phase_name=phase_input.phase_name,
                status="fail",
                error=str(err),
                metrics=metrics,
            )

        # Prefer structured 'result' dict, fall back to raw JSON string.
        result = envelope.get("result")
        if isinstance(result, dict):
            result.setdefault("run_id", phase_input.run_id)
            result.setdefault("phase_name", phase_input.phase_name)
            output = PhaseOutput.from_dict(result)
            metrics = dict(output.metrics)
            metrics.setdefault("mailbox_task_id", task_id)
            output.metrics = metrics
            return output

        raw = envelope.get("raw")
        if isinstance(raw, str):
            return self._parse_subagent_response(raw, phase_input)

        return PhaseOutput(
            run_id=phase_input.run_id,
            phase_name=phase_input.phase_name,
            status="fail",
            error=(
                "mailbox envelope reported success but carried neither "
                "'result' dict nor 'raw' JSON string"
            ),
            metrics={"mailbox_task_id": task_id},
        )

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
