"""
LLM backend abstraction for the pipeline orchestrator.

Centralizes every LLM call in the codebase behind a single ``LLMBackend``
Protocol. Domain code (DART, Courseforge, Trainforge) receives an injected
backend instead of reaching for ``anthropic.Anthropic()`` directly — which
means the same code path works under ``local`` mode (Claude Code session),
``api`` mode (Anthropic SDK), or tests (MockBackend).

Wave 7 ships:

- ``LocalBackend`` — placeholder that raises NotImplementedError with a clear
  message pointing at the dispatcher; the local dispatcher handles LLM needs
  via the enclosing Claude Code session rather than a callable backend.
- ``AnthropicBackend`` — production path; direct SDK call, non-streaming.
- ``OpenAIBackend`` — stub; reserved for a later wave per decision O2.
- ``MockBackend`` — records calls, returns deterministic responses for tests.

Streaming (``stream=True``) is intentionally deferred per decision O3 and
currently raises ``NotImplementedError`` with an explicit message.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# Default model identifiers per provider. These map to the models the
# codebase has standardized on — override via env (LLM_MODEL) or per-call.
#
# Phase 6 Subtask 23 (Phase 3c env-vars): env-var-first resolution chain
# for the Anthropic default model. ``LLM_MODEL`` (read inside
# ``build_backend()``) is the canonical workflow-wide override; the new
# ``MCP_ORCHESTRATOR_LLM_MODEL`` env var pins specifically the Anthropic
# default at module-import time so callers that bypass ``build_backend``
# (e.g. ``AnthropicBackend(...)`` constructed directly) still honour an
# operator pin without code edits. Resolution at module import:
#   1. ``MCP_ORCHESTRATOR_LLM_MODEL`` env var when set (and non-empty).
#   2. ``DEFAULT_ANTHROPIC_MODEL_DEFAULT`` (preserves legacy
#      ``claude-opus-4-7`` behavior).
# ``build_backend()`` keeps the ``LLM_MODEL`` env var as a higher-priority
# per-run override (precedence chain: explicit overrides > spec.model >
# ``LLM_MODEL`` env > ``DEFAULT_ANTHROPIC_MODEL``).
MCP_ORCHESTRATOR_LLM_MODEL_ENV = "MCP_ORCHESTRATOR_LLM_MODEL"
DEFAULT_ANTHROPIC_MODEL_DEFAULT = "claude-opus-4-7"
DEFAULT_ANTHROPIC_MODEL = (
    os.environ.get(MCP_ORCHESTRATOR_LLM_MODEL_ENV)
    or DEFAULT_ANTHROPIC_MODEL_DEFAULT
)
DEFAULT_OPENAI_MODEL = "gpt-4o"


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol every backend must satisfy.

    Implementations MUST be callable both as ``await backend.complete(...)``
    and via the sync helper ``backend.complete_sync(...)`` for call sites
    that are still synchronous (the three Wave 7 refactor sites are sync).

    Returning ``str`` for ``stream=False`` and ``AsyncIterator[str]`` for
    ``stream=True`` follows the same contract shape the Anthropic SDK uses.
    """

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """Run a completion. See implementations for details."""
        ...

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Synchronous convenience wrapper. Never streams."""
        ...


# =============================================================================
# AnthropicBackend — production API path
# =============================================================================


class AnthropicBackend:
    """Direct Anthropic SDK backend.

    The ``anthropic`` package is imported lazily so that ``LLMBackend``
    consumers who never reach API mode don't pay the import cost and don't
    require the package to be installed.

    Per decision O3, token streaming is not supported in Wave 7 — passing
    ``stream=True`` raises ``NotImplementedError``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_ANTHROPIC_MODEL,
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "AnthropicBackend requires an API key. Pass api_key=... or "
                "set ANTHROPIC_API_KEY in the environment."
            )
        self.api_key = resolved_key
        self.default_model = default_model
        self._client = None  # lazy init

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 — lazy import by design
            except ImportError as exc:  # pragma: no cover — exercised via mocks
                raise ImportError(
                    "anthropic package is required for AnthropicBackend. "
                    "Install with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _build_messages(
        self,
        user: str,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the messages array, attaching image blocks if provided."""
        if not images:
            return [{"role": "user", "content": user}]

        content: List[Dict[str, Any]] = []
        for img in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                }
            )
        content.append({"type": "text", "text": user})
        return [{"role": "user", "content": content}]

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Synchronous completion. The SDK is itself sync, so no await needed."""
        messages = self._build_messages(user, images=images)
        kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        if stream:
            raise NotImplementedError(
                "Streaming is not supported in Wave 7 (decision O3). "
                "Call with stream=False; --watch streaming will land in a later wave."
            )
        return self.complete_sync(
            system,
            user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
        )


# =============================================================================
# LocalBackend — Claude Code session placeholder
# =============================================================================


class LocalBackend:
    """Backend for ``--mode local`` runs.

    When the pipeline runs inside a Claude Code session, the *session itself*
    is the LLM: phase workers are dispatched as subagents via
    ``LocalDispatcher``, and any LLM completions those subagents need happen
    through their own subagent context. There is no Python-side callable that
    produces tokens.

    Calling ``.complete()`` on this backend directly is therefore a
    configuration error — the caller should either be running through the
    LocalDispatcher (which never invokes ``.complete()`` on the backend)
    or should be in ``api`` mode.
    """

    def __init__(self, *, description: str = "local Claude Code session"):
        self.description = description

    def _err(self) -> NotImplementedError:
        return NotImplementedError(
            "LocalBackend.complete() is not directly callable. In local mode, "
            "LLM work happens inside the Claude Code subagent dispatched by "
            "LocalDispatcher. If a domain module needs a callable backend, "
            "run in api mode (LLM_MODE=api) or inject a MockBackend for tests."
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        raise self._err()

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        raise self._err()


# =============================================================================
# MailboxBrokeredBackend — bridges ``complete()`` to a Claude Code session
# =============================================================================


class MailboxBrokeredBackend:
    """``LLMBackend`` that routes completions through a ``TaskMailbox``.

    Wave 73: in ``--mode local`` runs the orchestrator is a Python subprocess
    that has no direct access to an LLM API. Historically this meant every
    in-process LLM call site (``DART.converter.llm_classifier``,
    ``DART.pdf_converter.alt_text_generator``, ``Trainforge.align_chunks``)
    either refused to start (``LocalBackend`` throws) or silently fell back
    to a heuristic / no-op path — so "local mode" shipped real grounded
    templated content but no real Claude-generated enrichment anywhere.

    This backend closes that gap by brokering every ``complete()`` call
    through the same ``TaskMailbox`` infrastructure that ``LocalDispatcher``
    uses for phase-level dispatch:

    1. ``complete_sync()`` writes a pending task spec carrying
       ``kind="llm_call"`` plus the ``system`` / ``user`` / ``model`` /
       ``max_tokens`` / ``temperature`` / ``images`` payload.
    2. It blocks on ``TaskMailbox.wait_for_completion`` up to
       ``timeout_seconds``.
    3. An outer Claude Code operator (polling ``mailbox/pending/``)
       dispatches an ``Agent`` subagent to produce the completion, then
       writes a completion envelope ``{"success": true, "result":
       {"response_text": "<str>"}}`` to ``mailbox/completed/{task_id}.json``.
    4. The backend reads ``response_text`` and returns it to the caller —
       indistinguishable from a direct SDK completion from the call site's
       perspective.

    The envelope shape mirrors the phase-dispatch completion shape (see
    ``LocalDispatcher._dispatch_via_mailbox``) except ``result`` carries
    ``response_text`` rather than a full ``PhaseOutput`` payload — this
    lets operators disambiguate "LLM call" tasks from phase tasks by the
    ``kind`` field and the ``result`` schema.

    Streaming is explicitly unsupported: the mailbox protocol is
    request/response and the upstream codebase defers streaming per
    decision O3 anyway.
    """

    def __init__(
        self,
        mailbox,
        *,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.25,
        default_model: Optional[str] = None,
        task_id_prefix: str = "llm",
    ):
        """
        Args:
            mailbox: A ``MCP.orchestrator.task_mailbox.TaskMailbox`` bound
                to the active run's state directory.
            timeout_seconds: Maximum seconds to block waiting for the
                outer operator to write the completion envelope. Default
                120s — classifier batches and alt-text generations are
                typically tens of seconds, so 2 minutes gives headroom
                for operator turnaround without pinning forever.
            poll_interval: Seconds between mailbox polls. Kept short
                (0.25s) so the call latency is dominated by operator
                dispatch, not poll granularity.
            default_model: Informational only — passed through to the
                operator so decision captures can pin the model. The
                operator chooses the actual serving model.
            task_id_prefix: Prefix for generated task_ids. ``llm``
                distinguishes LLM-completion tasks from phase-dispatch
                tasks when they share a mailbox.
        """
        # Lazy import to avoid a hard dependency for consumers who never
        # build this backend (it lives in the same package so this is
        # cheap; kept lazy for symmetry with other backends).
        from .task_mailbox import TaskMailbox  # noqa: PLC0415

        if not isinstance(mailbox, TaskMailbox):
            raise TypeError(
                "MailboxBrokeredBackend requires a TaskMailbox instance. "
                f"Got {type(mailbox).__name__}."
            )
        self.mailbox = mailbox
        self.timeout_seconds = float(timeout_seconds)
        self.poll_interval = float(poll_interval)
        self.default_model = default_model or DEFAULT_ANTHROPIC_MODEL
        self.task_id_prefix = str(task_id_prefix)
        self._call_counter = 0

    def _next_task_id(self) -> str:
        """Return a mailbox task id globally unique across concurrent backends.

        Wave 73 code-review P1: the original implementation returned
        ``f"{prefix}-{counter:04d}"`` with a per-instance counter. Two
        parallel phase tasks (``TaskExecutor._execute_parallel`` dispatches
        via ``asyncio.gather``; ``textbook_to_course.dart_conversion``
        runs with ``max_concurrent: 4``) each auto-resolve their own
        ``MailboxBrokeredBackend`` at the ``pipeline_tools.py`` injection
        site, so both started from ``llm-0001`` and collided on
        ``TaskMailbox.put_pending`` — at best overwriting each other's
        spec files via ``os.replace``, at worst (and more commonly)
        two callers waited on the same ``completed/llm-0001.json`` and
        consumed the same response for different figures. No
        exception was raised; the bug surfaced only as mislabeled
        alt-text / misclassified blocks downstream.

        Switching to a UUID-suffixed id mirrors the phase-dispatch
        shape at ``LocalDispatcher._dispatch_via_mailbox`` (``{phase}-{uuid8}``).
        We also keep a monotonic counter as a debugging aid (visible
        via ``backend._call_counter``); it no longer participates in
        the task_id, so its per-instance scope is harmless.
        """
        import uuid as _uuid  # noqa: PLC0415 — lazy so the module stays light

        self._call_counter += 1
        return f"{self.task_id_prefix}-{_uuid.uuid4().hex[:12]}"

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        task_id = self._next_task_id()
        spec: Dict[str, Any] = {
            "kind": "llm_call",
            "system": system or "",
            "user": user,
            "model": model or self.default_model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if images:
            spec["images"] = images

        self.mailbox.put_pending(task_id, spec)
        logger.debug(
            "MailboxBrokeredBackend: queued %s (len(user)=%d, max_tokens=%d)",
            task_id,
            len(user or ""),
            max_tokens,
        )

        try:
            envelope = self.mailbox.wait_for_completion(
                task_id,
                timeout_seconds=self.timeout_seconds,
                poll_interval=self.poll_interval,
            )
        finally:
            # Prune per-task files regardless of success so the mailbox
            # stays bounded across long runs. Mirrors LocalDispatcher's
            # cleanup pattern.
            try:
                self.mailbox.cleanup(task_id)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "MailboxBrokeredBackend: cleanup failed for %s (non-fatal)",
                    task_id,
                )

        return self._text_from_envelope(envelope, task_id)

    @staticmethod
    def _text_from_envelope(envelope: Dict[str, Any], task_id: str) -> str:
        """Extract the completion text from a mailbox envelope.

        Accepted shapes (in precedence order):

        * ``{"success": true, "result": {"response_text": "..."}}``
          — canonical Wave 73 shape.
        * ``{"success": true, "result": "..."}``
          — convenience for operators that return a bare string.
        * ``{"success": true, "raw": "..."}``
          — fallback; ``raw`` is returned verbatim.

        Raises ``RuntimeError`` on ``success: false`` or a missing text
        payload so the call site can surface the mailbox failure instead
        of silently returning empty string (which would masquerade as a
        zero-length completion and suppress downstream heuristic
        fallbacks that key on the exception path).
        """
        if not isinstance(envelope, dict):
            raise RuntimeError(
                f"MailboxBrokeredBackend: task {task_id!r} completion "
                f"envelope was not a JSON object"
            )
        if not envelope.get("success", False):
            err = envelope.get("error") or "outer operator reported failure"
            code = envelope.get("error_code")
            suffix = f" (error_code={code})" if code else ""
            raise RuntimeError(
                f"MailboxBrokeredBackend: task {task_id!r} failed: {err}{suffix}"
            )

        result = envelope.get("result")
        if isinstance(result, dict):
            text = result.get("response_text")
            if isinstance(text, str):
                return text
        if isinstance(result, str):
            return result
        raw = envelope.get("raw")
        if isinstance(raw, str):
            return raw
        raise RuntimeError(
            f"MailboxBrokeredBackend: task {task_id!r} completion envelope "
            f"reported success but carried no response_text"
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        if stream:
            raise NotImplementedError(
                "MailboxBrokeredBackend does not support streaming "
                "(deferred per decision O3). Call with stream=False."
            )
        # Off-thread the blocking mailbox wait so the event loop isn't pinned.
        import asyncio as _asyncio  # noqa: PLC0415

        loop = _asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.complete_sync(
                system,
                user,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                images=images,
            ),
        )


# =============================================================================
# OpenAIBackend — stub reserved for later wave
# =============================================================================


class OpenAIBackend:
    """Reserved for a future wave (decision O2).

    Construction is allowed so the provider registry surface works, but any
    completion call raises ``NotImplementedError`` to keep the contract
    honest. Swap to ``AnthropicBackend`` or wait for the follow-up wave that
    lands the OpenAI SDK integration.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_OPENAI_MODEL,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.default_model = default_model

    def _err(self) -> NotImplementedError:
        return NotImplementedError(
            "OpenAIBackend is a stub reserved for a later wave (decision O2). "
            "Use AnthropicBackend for Wave 7 api-mode runs, or pin "
            "LLM_PROVIDER=anthropic."
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        raise self._err()

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        raise self._err()


# =============================================================================
# MockBackend — deterministic fixture-driven backend for tests
# =============================================================================


@dataclass
class _MockCall:
    """Record of a single backend invocation (test introspection)."""

    system: str
    user: str
    model: Optional[str]
    max_tokens: int
    temperature: float
    stream: bool
    images: Optional[List[Dict[str, Any]]]


class MockBackend:
    """Test backend that records calls and returns fixture-driven responses.

    Two ways to configure responses:

    1. ``responses``: list of strings consumed in FIFO order per call.
    2. ``response_fn``: callable ``(system, user) -> str`` for dynamic responses.
    3. ``fixture_dir``: directory of JSON files; filename is the sha256 of
       ``system + "\\n" + user`` (first 16 chars) + ``.json``. File contains
       ``{"text": "..."}``. Used by the refactored call-site tests so the
       same fixture can be shared across test files.

    Exactly one of ``responses``, ``response_fn``, or ``fixture_dir`` is typical.
    When multiple are set, priority is: response_fn > fixture_dir > responses.
    If nothing is configured, returns ``default_response`` (empty by default).
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        response_fn: Optional[Callable[[str, str], str]] = None,
        fixture_dir: Optional[Path] = None,
        default_response: str = "",
    ):
        self._responses: List[str] = list(responses) if responses else []
        self._response_fn = response_fn
        self._fixture_dir = Path(fixture_dir) if fixture_dir else None
        self._default_response = default_response
        self.calls: List[_MockCall] = []

    @staticmethod
    def _fixture_key(system: str, user: str) -> str:
        import hashlib

        payload = f"{system}\n{user}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _resolve_response(self, system: str, user: str) -> str:
        if self._response_fn is not None:
            return self._response_fn(system, user)

        if self._fixture_dir is not None:
            key = self._fixture_key(system, user)
            fixture_path = self._fixture_dir / f"{key}.json"
            if fixture_path.exists():
                with open(fixture_path) as f:
                    data = json.load(f)
                return data.get("text", self._default_response)

        if self._responses:
            return self._responses.pop(0)

        return self._default_response

    def _record(
        self,
        system: str,
        user: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        stream: bool,
        images: Optional[List[Dict[str, Any]]],
    ) -> None:
        self.calls.append(
            _MockCall(
                system=system,
                user=user,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                images=images,
            )
        )

    def complete_sync(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self._record(system, user, model, max_tokens, temperature, False, images)
        return self._resolve_response(system, user)

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, AsyncIterator[str]]:
        self._record(system, user, model, max_tokens, temperature, stream, images)
        if stream:
            raise NotImplementedError(
                "MockBackend does not emulate streaming (deferred per O3). "
                "Call with stream=False in tests."
            )
        return self._resolve_response(system, user)


# =============================================================================
# Factory helpers
# =============================================================================


@dataclass
class BackendSpec:
    """Serializable description of which backend to build.

    Used by the orchestrator and dispatchers when they need to hand a factory
    to phase workers that don't share a process.
    """

    mode: Literal["local", "api"] = "local"
    provider: Literal["anthropic", "openai", "mock"] = "anthropic"
    model: Optional[str] = None
    api_key: Optional[str] = None
    mock_responses: List[str] = field(default_factory=list)
    # Wave 73: when mode=local, ``run_id`` + optional ``mailbox_base_dir``
    # select a ``MailboxBrokeredBackend`` (Claude Code operator loop) over
    # the default ``LocalBackend`` stub. Empty run_id keeps the pre-Wave-73
    # throwing behavior so tests / callers that haven't opted in stay
    # loud if they accidentally call ``.complete()``.
    run_id: Optional[str] = None
    mailbox_base_dir: Optional[str] = None


def build_backend(spec: Optional[BackendSpec] = None, **overrides: Any) -> LLMBackend:
    """Build an ``LLMBackend`` from a spec + env fallbacks.

    Precedence: explicit ``overrides`` > ``spec`` fields > env vars > defaults.

    Recognized env vars: ``LLM_MODE``, ``LLM_PROVIDER``, ``LLM_MODEL``,
    ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``ED4ALL_RUN_ID``,
    ``ED4ALL_MAILBOX_BASE_DIR``.

    Wave 73 local-mode path: when ``mode=local`` and a ``run_id`` is
    resolvable (via overrides, spec, or ``ED4ALL_RUN_ID`` env), build a
    ``MailboxBrokeredBackend`` bound to ``{mailbox_base_dir}/{run_id}/
    mailbox/``. This lets in-process LLM call sites (classifier, alt-text,
    align_chunks) route through the TaskMailbox to a Claude Code operator
    loop — the "local LLM" pathway that was scaffolded but not wired in
    Waves 7 / 34.

    When ``mode=local`` but no ``run_id`` is available, fall through to
    the throwing ``LocalBackend`` to preserve the pre-Wave-73 contract:
    callers that accidentally try to call ``.complete()`` without
    opting into the mailbox path fail loudly.
    """
    spec = spec or BackendSpec()

    mode = overrides.get("mode") or spec.mode or os.environ.get("LLM_MODE", "local")
    provider = (
        overrides.get("provider")
        or spec.provider
        or os.environ.get("LLM_PROVIDER", "anthropic")
    )
    model = overrides.get("model") or spec.model or os.environ.get("LLM_MODEL")

    if mode == "local":
        run_id = (
            overrides.get("run_id")
            or spec.run_id
            or os.environ.get("ED4ALL_RUN_ID")
        )
        if run_id:
            mailbox_base_dir = (
                overrides.get("mailbox_base_dir")
                or spec.mailbox_base_dir
                or os.environ.get("ED4ALL_MAILBOX_BASE_DIR")
            )
            from .task_mailbox import TaskMailbox  # noqa: PLC0415

            base_path = Path(mailbox_base_dir) if mailbox_base_dir else None
            mailbox = TaskMailbox(run_id=run_id, base_dir=base_path)
            timeout = overrides.get("mailbox_timeout_seconds")
            kwargs: Dict[str, Any] = {"default_model": model}
            if timeout is not None:
                kwargs["timeout_seconds"] = float(timeout)
            return MailboxBrokeredBackend(mailbox, **kwargs)
        return LocalBackend()

    # api mode
    if provider == "mock":
        return MockBackend(responses=list(spec.mock_responses))
    if provider == "anthropic":
        api_key = overrides.get("api_key") or spec.api_key
        return AnthropicBackend(
            api_key=api_key,
            default_model=model or DEFAULT_ANTHROPIC_MODEL,
        )
    if provider == "openai":
        api_key = overrides.get("api_key") or spec.api_key
        return OpenAIBackend(
            api_key=api_key,
            default_model=model or DEFAULT_OPENAI_MODEL,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
