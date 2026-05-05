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

# Wave 74: per-task subagent dispatch carries different latency
# expectations than phase-level dispatch. A content-generator working
# through a full week's modules can legitimately take 10+ minutes.
# Overridable via ``ED4ALL_AGENT_TIMEOUT_SECONDS`` env var (Session 2
# exposes this on the CLI too).
_DEFAULT_AGENT_TASK_TIMEOUT = 1800.0
_AGENT_TASK_TIMEOUT_ENV = "ED4ALL_AGENT_TIMEOUT_SECONDS"


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
        if mailbox_base_dir is not None:
            self.mailbox_base_dir = Path(mailbox_base_dir)
        else:
            # Honor ED4ALL_STATE_RUNS_DIR override so unit tests can
            # redirect mailbox writes into tmp_path.
            env_override = os.environ.get("ED4ALL_STATE_RUNS_DIR")
            self.mailbox_base_dir = (
                Path(env_override) if env_override
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

    # ---------------------------------------------------------- per-task
    #
    # Wave 74: per-task dispatch hook. Closes the Wave 38 gap where
    # ``TaskExecutor._invoke_tool`` called the Python tool registry
    # directly on every phase task, bypassing the dispatcher entirely.
    # When a phase task's agent is classified as subagent-dispatched
    # (see ``MCP/core/executor.AGENT_SUBAGENT_SET``) AND
    # ``ED4ALL_AGENT_DISPATCH=true``, the executor routes through this
    # method instead of invoking the in-process templated emitter.
    #
    # Contract mirrors the Python tool's envelope: caller hands a dict
    # of mapped params, receives back a JSON-serialisable dict matching
    # the tool's historical return shape (e.g. ``{"success": true,
    # "artifacts": [...], "outputs": {...}}``). Unsuccessful work
    # returns the Wave 33 ``{"success": false, "error_code": ...,
    # "error": ...}`` envelope so the executor retry path and gate
    # aggregation stay unchanged.

    async def dispatch_task(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch one phase task to a subagent, return tool-shape dict.

        Three-path selector (mirrors ``dispatch_phase``'s selector):

        1. ``agent_tool`` callable injected → call it directly.
           Test-harness path + future in-process integrations.
        2. ``agent_tool`` missing AND ``LOCAL_DISPATCHER_ALLOW_STUB=1`` →
           return a stub envelope so dry-run / CI exercise the routing
           fork without a real subagent pathway.
        3. ``agent_tool`` missing AND stub flag off → write a pending
           ``kind="agent_task"`` spec to the run's mailbox, block on
           ``TaskMailbox.wait_for_completion``. An outer Claude Code
           operator dispatches the actual subagent via the MCP
           ``Agent`` tool (using the agent spec markdown at
           ``{project}/{Courseforge|Trainforge|DART}/agents/{agent_type}.md``
           to drive the prompt) and writes the completion envelope.
           Timeout → ``{"success": false, "error_code":
           "MAILBOX_TIMEOUT"}``.

        Args:
            task_name: Python tool name the agent would have called
                in the legacy path. Carried through so operators can
                disambiguate when multiple agents map to the same tool
                (three Courseforge agents all map to
                ``remediate_course_content``).
            agent_type: The agent type the executor picked up from the
                task's ``agent_type`` field. Drives agent-spec resolution
                + the pending-task prefix so operators filter by kind.
            task_params: Already param-mapped via ``TaskParameterMapper``.
                Serialised into the mailbox spec verbatim.
            run_id: Workflow run id — pins the mailbox directory.
            phase_context: Optional phase-level context (phase outputs
                so far, workflow params). Emitted into the pending spec
                so subagents can resolve cross-phase inputs without
                re-reading state JSON.
        """
        self._dispatched.append(f"{agent_type}:{task_name}")

        if self.agent_tool is not None:
            return await self._dispatch_task_via_callable(
                task_name=task_name,
                agent_type=agent_type,
                task_params=task_params,
                run_id=run_id,
                phase_context=phase_context,
            )

        allow_stub = os.environ.get(_ALLOW_STUB_ENV, "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if allow_stub:
            logger.info(
                "LocalDispatcher.dispatch_task: no agent_tool wired; "
                "LOCAL_DISPATCHER_ALLOW_STUB set — emitting stub envelope "
                "for agent=%s tool=%s", agent_type, task_name,
            )
            return {
                "success": True,
                "dispatch_mode": "stub",
                "agent_type": agent_type,
                "tool_name": task_name,
                "outputs": {},
                "artifacts": [],
            }

        return await self._dispatch_task_via_mailbox(
            task_name=task_name,
            agent_type=agent_type,
            task_params=task_params,
            run_id=run_id,
            phase_context=phase_context,
        )

    async def _dispatch_task_via_callable(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Happy-path: caller injected an ``agent_tool``. Call it and
        parse the returned JSON into the tool-shape dict.
        """
        request = {
            "subagent_type": agent_type,
            "agent_type": agent_type,
            "tool_name": task_name,
            "task_params": task_params,
            "run_id": run_id,
            "phase_context": phase_context or {},
        }
        try:
            raw = await self.agent_tool(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LocalDispatcher.dispatch_task: agent_tool raised for "
                "agent=%s tool=%s", agent_type, task_name,
            )
            return {
                "success": False,
                "error": str(exc),
                "error_code": "AGENT_TOOL_RAISED",
            }

        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "error": f"agent_tool returned non-JSON string: {exc}",
                    "error_code": "INVALID_AGENT_RESPONSE",
                    "raw": raw[:2000],
                }
            if isinstance(parsed, dict):
                return parsed
            return {
                "success": False,
                "error": "agent_tool returned JSON that wasn't an object",
                "error_code": "INVALID_AGENT_RESPONSE",
            }
        return {
            "success": False,
            "error": f"agent_tool returned unexpected type {type(raw).__name__}",
            "error_code": "INVALID_AGENT_RESPONSE",
        }

    async def _dispatch_task_via_mailbox(
        self,
        *,
        task_name: str,
        agent_type: str,
        task_params: Dict[str, Any],
        run_id: str,
        phase_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Mailbox-bridge path — writes a pending agent task spec and
        blocks on the outer Claude Code operator's completion envelope.
        """
        mailbox = TaskMailbox(run_id=run_id, base_dir=self.mailbox_base_dir)

        task_id = f"{agent_type}-{uuid.uuid4().hex[:12]}"
        spec: Dict[str, Any] = {
            "kind": "agent_task",
            "agent_type": agent_type,
            "tool_name": task_name,
            "task_params": task_params,
            "phase_context": phase_context or {},
            "agent_spec_path": self._resolve_agent_spec_path(agent_type),
        }

        try:
            mailbox.put_pending(task_id, spec)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LocalDispatcher.dispatch_task: failed to queue task "
                "for agent=%s tool=%s", agent_type, task_name,
            )
            return {
                "success": False,
                "error": f"mailbox put_pending failed: {exc}",
                "error_code": "MAILBOX_PUT_FAILED",
            }

        timeout = self._resolve_agent_task_timeout()
        logger.info(
            "LocalDispatcher.dispatch_task: agent task %s queued for "
            "agent=%s tool=%s (timeout=%.0fs)",
            task_id, agent_type, task_name, timeout,
        )

        # Wave W5: call the async-native mailbox waiter directly so
        # 10-way fanout doesn't saturate the asyncio default thread
        # pool via ``run_in_executor`` wrappers.
        try:
            envelope = await mailbox.await_completion_async(
                task_id,
                timeout_seconds=timeout,
                poll_interval=self.mailbox_poll_interval,
            )
        except TimeoutError:
            logger.error(
                "LocalDispatcher.dispatch_task: mailbox timeout for "
                "task=%s agent=%s", task_id, agent_type,
            )
            return {
                "success": False,
                "error": (
                    f"MAILBOX_TIMEOUT: no completion from outer operator "
                    f"within {timeout:.0f}s for task {task_id!r}. "
                    f"Recovery: run the mailbox operator loop (Session 2 "
                    f"ships ``ed4all mailbox-bridge peek-agent``) in a "
                    f"Claude Code session, or set ED4ALL_AGENT_TIMEOUT_SECONDS "
                    f"higher, or disable ED4ALL_AGENT_DISPATCH to fall back "
                    f"to the in-process templated path."
                ),
                "error_code": "MAILBOX_TIMEOUT",
                "mailbox_task_id": task_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "LocalDispatcher.dispatch_task: mailbox wait failed for %s",
                task_id,
            )
            return {
                "success": False,
                "error": f"mailbox wait_for_completion failed: {exc}",
                "error_code": "MAILBOX_WAIT_FAILED",
                "mailbox_task_id": task_id,
            }
        finally:
            try:
                mailbox.cleanup(task_id)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "LocalDispatcher.dispatch_task: cleanup failed for %s "
                    "(non-fatal)", task_id,
                )

        return self._tool_dict_from_envelope(envelope, task_id)

    @staticmethod
    def _tool_dict_from_envelope(
        envelope: Dict[str, Any], task_id: str,
    ) -> Dict[str, Any]:
        """Unwrap a mailbox completion envelope into a tool-shape dict.

        Accepted shapes:

        * ``{"success": true, "result": {...tool envelope...}}``
          — canonical.
        * ``{"success": true, ...}`` with no ``result`` key → the
          envelope IS the tool dict (convenience for operators that
          return a flat structure).
        * ``{"success": false, ...}`` → pass through (tool-shape
          failure envelope).
        """
        if not isinstance(envelope, dict):
            return {
                "success": False,
                "error": "agent task completion envelope was not a JSON object",
                "error_code": "INVALID_ENVELOPE",
                "mailbox_task_id": task_id,
            }

        if envelope.get("success") is False:
            out = dict(envelope)
            out.setdefault("mailbox_task_id", task_id)
            return out

        result = envelope.get("result")
        if isinstance(result, dict):
            result = dict(result)
            result.setdefault("mailbox_task_id", task_id)
            return result

        # Flat-envelope operator convenience.
        if envelope.get("success") is True:
            out = {k: v for k, v in envelope.items() if k not in ("kind", "task_id")}
            out.setdefault("mailbox_task_id", task_id)
            return out

        return {
            "success": False,
            "error": (
                "agent task envelope reported success but carried no "
                "recognisable result payload"
            ),
            "error_code": "INVALID_ENVELOPE",
            "mailbox_task_id": task_id,
        }

    def _resolve_agent_spec_path(self, agent_type: str) -> Optional[str]:
        """Locate the agent spec markdown so operators can inject it into
        subagent prompts. Returns the relative path (from project root)
        or ``None`` if no spec is found.
        """
        for base in AGENT_SPEC_DIRS:
            candidate = self.project_root / base / f"{agent_type}.md"
            if candidate.exists():
                try:
                    return str(candidate.relative_to(self.project_root))
                except ValueError:
                    return str(candidate)
        return None

    def _resolve_agent_task_timeout(self) -> float:
        """Read ``ED4ALL_AGENT_TIMEOUT_SECONDS`` at call time (not import
        time) so tests / ops can override per-run via environment.
        """
        raw = os.environ.get(_AGENT_TASK_TIMEOUT_ENV, "").strip()
        if raw:
            try:
                parsed = float(raw)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning(
                    "Invalid %s=%r; falling back to default %.0fs",
                    _AGENT_TASK_TIMEOUT_ENV, raw, _DEFAULT_AGENT_TASK_TIMEOUT,
                )
        return _DEFAULT_AGENT_TASK_TIMEOUT

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

        # Wave W5: call the async-native mailbox waiter directly. The
        # legacy ``run_in_executor`` wrapper held a thread-pool slot
        # for the entire wait, which under 10-way phase fanout
        # saturated the asyncio default executor (default 16 slots).
        try:
            envelope = await mailbox.await_completion_async(
                task_id,
                timeout_seconds=self.mailbox_timeout_seconds,
                poll_interval=self.mailbox_poll_interval,
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
