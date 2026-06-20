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
import time
import warnings
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
from urllib.parse import urlparse

# Single source of truth for the OpenAI-compatible endpoint registry. The
# module-level ``_OPENAI_COMPATIBLE_PROVIDERS`` dict below is a PROJECTION of
# ``config/endpoints.yaml`` built via this helper — never a hand-maintained
# literal. Import is anti-cycle by design (``lib.llm.endpoints`` imports only
# stdlib + yaml + jsonschema + lib.paths, never back into this module).
from lib.llm.endpoints import (
    openai_compatible_legacy_registry as _openai_compatible_legacy_registry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base class — provider-agnostic decision-capture wiring
# =============================================================================


class _CaptureMixin:
    """Provider-agnostic ``DecisionCapture`` wiring for every backend.

    Every concrete ``LLMBackend`` subclass that subclasses this mixin gets
    one canonical ``decision_type="llm_chat_call"`` event per ``invoke`` /
    ``complete_sync`` dispatch when ``capture`` is wired. The helper is a
    no-op when ``capture is None`` so callers can opt out without ceremony.

    Mirrors the LLM-agnostic pattern at
    ``Trainforge/generators/_openai_compatible_client.py``: rationale
    interpolates dynamic signals (model, max_tokens, latency_ms,
    messages_count, response_text_len) so post-hoc replay sees per-call
    structure. The ``provider`` field is set per-backend (``"anthropic"``,
    ``"local"``, ``"mailbox"``, ``"openai"``, ``"mock"``) so the audit
    trail records which backend fired without leaking provider names
    into field names.

    Subclasses MUST set ``self.provider_label`` and call
    ``self._set_capture(capture)`` from ``__init__``. Each ``invoke`` /
    ``complete_sync`` body wraps the underlying provider call in a
    try/finally that records ``time.monotonic()`` before/after and calls
    ``self._emit_llm_chat_capture(...)``. Capture failures NEVER crash
    the LLM dispatch (matches the project's existing capture-wrapping
    convention).
    """

    # Default — subclasses override.
    provider_label: str = "unknown"

    def _set_capture(self, capture: Optional[Any]) -> None:
        """Stash the capture handle. ``None`` is a clean no-op."""
        self._capture = capture

    def _emit_llm_chat_capture(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        messages_count: int,
        response_text_len: int,
        latency_ms: float,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit one ``decision_type="llm_chat_call"`` event.

        Generic field names only — no ``claude_*`` / ``anthropic_*`` /
        ``gpt_*`` keys. The ``provider`` value distinguishes backends
        in the audit trail. Wrapped in try/except so a logging failure
        never crashes the LLM dispatch.
        """
        capture = getattr(self, "_capture", None)
        if capture is None:
            return
        try:
            extra_str = ""
            if extra:
                pairs = [
                    f"{k}={extra[k]}"
                    for k in sorted(extra.keys())
                    if extra[k] is not None
                ]
                if pairs:
                    extra_str = "; " + ", ".join(pairs)

            decision = (
                f"{self.provider_label} chat call to model {model}; "
                f"messages_count={messages_count}, "
                f"response_text_len={response_text_len}, "
                f"latency_ms={latency_ms:.1f}{extra_str}."
            )
            rationale = (
                f"LLM chat dispatch via provider={self.provider_label}, "
                f"model={model}, max_tokens={max_tokens}, "
                f"temperature={temperature}, "
                f"messages_count={messages_count}, "
                f"response_text_len={response_text_len}, "
                f"latency_ms={latency_ms:.1f}, stream={stream}."
            )
            capture.log_decision(
                decision_type="llm_chat_call",
                decision=decision,
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "llm_chat_call capture failed for provider=%s model=%s: %s",
                self.provider_label, model, exc,
            )


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


class AnthropicBackend(_CaptureMixin):
    """Direct Anthropic SDK backend.

    The ``anthropic`` package is imported lazily so that ``LLMBackend``
    consumers who never reach API mode don't pay the import cost and don't
    require the package to be installed.

    Per decision O3, token streaming is not supported in Wave 7 — passing
    ``stream=True`` raises ``NotImplementedError``.
    """

    provider_label = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        capture: Optional[Any] = None,
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
        self._set_capture(capture)

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
        resolved_model = model or self.default_model
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        start = time.monotonic()
        text: str = ""
        try:
            response = self.client.messages.create(**kwargs)
            text = response.content[0].text
            return text
        finally:
            self._emit_llm_chat_capture(
                model=resolved_model,
                max_tokens=int(max_tokens),
                temperature=float(temperature) if temperature is not None else 0.0,
                messages_count=len(messages),
                response_text_len=len(text or ""),
                latency_ms=(time.monotonic() - start) * 1000.0,
                extra={"images_count": len(images) if images else 0},
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


class LocalBackend(_CaptureMixin):
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

    provider_label = "local"

    def __init__(
        self,
        *,
        description: str = "local Claude Code session",
        capture: Optional[Any] = None,
    ):
        self.description = description
        self._set_capture(capture)

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


class MailboxBrokeredBackend(_CaptureMixin):
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

    provider_label = "mailbox"

    def __init__(
        self,
        mailbox,
        *,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.25,
        default_model: Optional[str] = None,
        task_id_prefix: str = "llm",
        capture: Optional[Any] = None,
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
        self._set_capture(capture)

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
        resolved_model = model or self.default_model
        spec: Dict[str, Any] = {
            "kind": "llm_call",
            "system": system or "",
            "user": user,
            "model": resolved_model,
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

        start = time.monotonic()
        text: str = ""
        # Mailbox dispatch flows through one logical "messages" pair
        # (system + user). Match the count to the AnthropicBackend shape
        # so the audit trail compares cleanly across providers.
        messages_count = 2 if system else 1
        try:
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

            text = self._text_from_envelope(envelope, task_id)
            return text
        finally:
            self._emit_llm_chat_capture(
                model=resolved_model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages_count=messages_count,
                response_text_len=len(text or ""),
                latency_ms=(time.monotonic() - start) * 1000.0,
                extra={
                    "task_id": task_id,
                    "images_count": len(images) if images else 0,
                },
            )

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


class OpenAIBackend(_CaptureMixin):
    """Reserved for a future wave (decision O2).

    Construction is allowed so the provider registry surface works, but any
    completion call raises ``NotImplementedError`` to keep the contract
    honest. Swap to ``AnthropicBackend`` or wait for the follow-up wave that
    lands the OpenAI SDK integration.
    """

    provider_label = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_OPENAI_MODEL,
        *,
        capture: Optional[Any] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.default_model = default_model
        self._set_capture(capture)

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
# OpenAICompatibleBackend — generic backend for any OpenAI-compatible provider
# =============================================================================
#
# DESIGN INVARIANT (Wave W-D12): there is exactly ONE class for every
# OpenAI-compatible provider (local Ollama / vLLM / llama.cpp / LM Studio,
# Together AI, Groq, Fireworks, DeepSeek, Mistral, Gemini-via-OpenAI-shim,
# etc.) — driven entirely by configuration in
# ``_OPENAI_COMPATIBLE_PROVIDERS``. There are NO per-provider subclasses.
# Adding a new provider is a registry-entry change, not a new class file.
# The dynamic ``provider_name`` argument is opaque metadata — it surfaces
# in decision-capture rationales for the audit trail but never branches
# behavior inside the class.
#
# The backend wraps ``Trainforge.generators._openai_compatible_client.
# OpenAICompatibleClient`` (the same HTTP client the synthesis pipeline
# uses) — composition-over-inheritance, lazy-imported so the orchestrator
# stays light when only Anthropic / Mock paths are exercised.
#
# Vision (image inputs) lands in W-D13 — the backend now passes
# ``images=`` through to the underlying ``OpenAICompatibleClient``
# (which translates to OpenAI's content-block shape on the wire).
# Per-provider vision capability is declared in
# ``_OPENAI_COMPATIBLE_PROVIDERS[name]["vision_capable"]``; non-vision
# providers raise a clear ``RuntimeError`` BEFORE the HTTP round-trip.

OPENAI_COMPATIBLE_VISION_MESSAGE = (
    "Provider {provider!r} is not vision-capable; set vision_capable=True "
    "in the registry entry or pick a vision-capable provider "
    "(e.g. anthropic, together-vision, or local with LOCAL_VISION_CAPABLE=true)."
)


# Provider registry — module-level dict, NOT a separate file. Each entry
# declares the env-var resolution chain for ``base_url`` / ``api_key`` /
# ``default_model`` plus whether the provider rejects requests without a
# valid api_key (cloud OSS providers like Together require one; local
# servers like Ollama do not). To add a new provider (Groq, Fireworks,
# DeepSeek, Mistral, hosted Gemini-via-OpenAI-shim, ...), append a new
# entry — DO NOT add a subclass.
#
# Schema per entry:
#   - ``base_url_env``: env var that overrides the base URL, or ``None``
#     if the URL is fixed (cloud providers usually have a single
#     production endpoint).
#   - ``base_url_default``: default base URL the resolver returns when
#     ``base_url_env`` is unset.
#   - ``api_key_env``: env var the resolver reads for the bearer token.
#   - ``api_key_default``: optional fallback when the env var is unset.
#     ``None`` means "no fallback" — combined with
#     ``api_key_required=True`` this triggers a ``RuntimeError`` at
#     resolve time. Local-style providers use a placeholder string
#     (``"local"``) so reverse-proxy servers that DO check auth see a
#     stable value.
#   - ``model_env``: env var that overrides the default model.
#   - ``model_default``: default model identifier.
#   - ``api_key_required``: whether a missing api_key fails closed at
#     resolve time. Cloud providers: ``True``. Local servers: ``False``.
#   - ``unverified`` (optional): truthy when the registry entry is a
#     stub that hasn't been verified against the provider's docs;
#     callers see a warning at resolve time so an operator knows to
#     double-check the base_url / model name before relying on it.
#   - ``vision_capable`` (optional, W-D13): truthy when the
#     resolved-by-default model under this entry can accept image
#     inputs. The backend reads this flag to short-circuit a vision
#     request against a non-vision-capable provider with a clear
#     error BEFORE the wire round-trip. ``local`` defaults to
#     ``False`` (default Qwen 14B is text-only); operators with a
#     vision model loaded into Ollama / vLLM flip this on via the
#     ``LOCAL_VISION_CAPABLE=true`` env var (or via a substring check
#     on ``LOCAL_SYNTHESIS_MODEL`` containing ``vision`` /
#     ``llava`` / ``-vl-``). ``together-vision`` is a sibling entry
#     pinned to Llama-3.2-90B-Vision-Instruct-Turbo for cloud OSS
#     vision; ``together`` (text) stays vision-incapable to avoid
#     silently flipping the default text model into vision mode.
#   - ``vision_capable_env`` (optional, W-D13): env var the resolver
#     consults to override ``vision_capable`` per deployment. When
#     set to a truthy value (``true`` / ``1`` / ``yes`` / ``on``,
#     case-insensitive), the resolved entry's ``vision_capable``
#     flips to ``True`` regardless of the registry default.
# Built ONCE at import as a PROJECTION of the ``openai_compatible`` rows in
# ``config/endpoints.yaml`` (loader ``lib/llm/endpoints.py``) — NOT a
# hand-maintained literal. The YAML is the single source of truth; adding a
# provider is a one-row change there (plus the provenance codegen). Every
# consumer of this dict (the answer path, the Courseforge / Trainforge
# synthesis + outliner + assessment providers, ``gui/env_catalog.py``) thus
# resolves its endpoint BY NAME from the unified registry. The legacy field
# shape (``base_url_env`` / ``base_url_default`` / ``api_key_env`` /
# ``api_key_default`` / ``model_env`` / ``model_default`` / ``api_key_required``
# + optional ``vision_capable`` / ``vision_capable_env`` / ``unverified``) is
# preserved exactly by the projection so consumers are byte-unchanged. The
# W-D12 dynamic-extension tests still ``monkeypatch.setitem`` this module-level
# dict at runtime (the projection returns a fresh mutable dict, so a patch
# can't poison the cached YAML registry).
_OPENAI_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, Any]] = (
    _openai_compatible_legacy_registry()
)


def _redact_base_url_for_capture(base_url: str) -> str:
    """Return host-only form of ``base_url`` for decision-capture rationale.

    Stripping the path keeps the rationale string tight and avoids
    leaking deployment-specific suffixes (e.g. ``/v1/openai`` proxy
    routes) into the audit trail. Returns the input verbatim when the
    URL is unparseable so we never crash the capture path.
    """
    try:
        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:  # noqa: BLE001 — defensive
        pass
    return base_url


class OpenAICompatibleBackend(_CaptureMixin):
    """Generic ``LLMBackend`` for any OpenAI-compatible chat provider.

    ONE class. NO subclasses. Provider semantics (base_url, default
    model, api_key resolution, "must I have an api_key?") flow through
    the registry, not via inheritance. The ``provider_name`` constructor
    arg is opaque metadata — it surfaces in decision-capture rationales
    so the audit trail records which OSS provider produced each call,
    but no branch inside the class body keys on the value.

    Wraps ``Trainforge.generators._openai_compatible_client.
    OpenAICompatibleClient``: lazy import keeps the orchestrator slim
    when only Anthropic / Mock paths are wired, and reuses the existing
    HTTP retry / JSON parse / error-mapping code instead of duplicating
    it here.

    Vision support (W-D13): when the provider entry sets
    ``vision_capable=True`` (or the resolver flips it via
    ``vision_capable_env``), the backend forwards ``images=`` through
    to the underlying ``OpenAICompatibleClient.chat_completion`` which
    translates to OpenAI's content-block shape on the wire. Calling
    with ``images=`` against a non-vision-capable provider raises
    ``RuntimeError`` BEFORE the HTTP round-trip. Streaming stays
    deferred per project decision O3.
    """

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        default_model: str,
        api_key: Optional[str] = None,
        capture: Optional[Any] = None,
        request_timeout: float = 120.0,
        vision_capable: bool = False,
    ):
        if not provider_name:
            raise ValueError("OpenAICompatibleBackend requires a provider_name")
        if not base_url:
            raise ValueError("OpenAICompatibleBackend requires a base_url")
        if not default_model:
            raise ValueError(
                "OpenAICompatibleBackend requires a non-empty default_model"
            )
        # provider_label is the canonical _CaptureMixin field that flows
        # into the rationale. Setting it from the constructor arg means a
        # new provider is one registry-entry away — no per-provider class.
        self.provider_label = str(provider_name)
        self.provider_name = str(provider_name)
        self.base_url = str(base_url).rstrip("/")
        self.default_model = str(default_model)
        self.api_key = api_key
        self.request_timeout = float(request_timeout)
        # W-D13: declarative vision capability. The flag is set at
        # construction time from the registry entry (or operator
        # override); calling ``complete_sync(..., images=...)`` against
        # a backend with ``vision_capable=False`` raises before the
        # wire round-trip.
        self.vision_capable = bool(vision_capable)
        self._client = None  # lazy
        self._set_capture(capture)

    @property
    def client(self):
        """Lazy ``OpenAICompatibleClient``. Built on first use."""
        if self._client is None:
            try:
                # Lazy import — keeps the orchestrator import-light in
                # paths that never reach an OpenAI-compatible provider.
                from Trainforge.generators._openai_compatible_client import (  # noqa: PLC0415
                    OpenAICompatibleClient,
                )
            except ImportError as exc:  # pragma: no cover — defensive
                raise ImportError(
                    "OpenAICompatibleBackend requires "
                    "Trainforge.generators._openai_compatible_client. "
                    "This module is shipped with the project; if you see "
                    "this error your install is incomplete."
                ) from exc
            self._client = OpenAICompatibleClient(
                base_url=self.base_url,
                model=self.default_model,
                api_key=self.api_key,
                # Capture is wired at the backend layer (this class), not
                # at the inner client layer — we want a single
                # ``llm_chat_call`` event per backend call, not two.
                capture=None,
                timeout=self.request_timeout,
                provider_label=self.provider_label,
                vision_capable=self.vision_capable,
            )
        return self._client

    @staticmethod
    def _build_messages(system: str, user: str) -> List[Dict[str, str]]:
        """Translate ``LLMBackend``-style (system, user) → OpenAI messages."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

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
        # W-D13: vision-capability gate. Fail loudly BEFORE the wire
        # round-trip if the operator routed an image-bearing call
        # against a non-vision-capable provider entry. The error
        # message points at every escape hatch (registry flag,
        # vision_capable_env, sibling vision provider entries) so the
        # operator sees the fix inline.
        if images and not self.vision_capable:
            raise RuntimeError(
                OPENAI_COMPATIBLE_VISION_MESSAGE.format(
                    provider=self.provider_name
                )
            )

        resolved_model = model or self.default_model
        messages = self._build_messages(system, user)
        client = self.client
        # If the caller overrides the default model, swap the inner
        # client's model field for this call only.
        previous_model: Optional[str] = None
        if model and model != client.model:
            previous_model = client.model
            client._model = str(model)  # noqa: SLF001 — client owns its own state

        start = time.monotonic()
        text: str = ""
        try:
            text = client.chat_completion(
                messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                images=images,
            )
            return text
        finally:
            if previous_model is not None:
                client._model = previous_model  # noqa: SLF001
            self._emit_llm_chat_capture(
                model=resolved_model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages_count=len(messages),
                response_text_len=len(text or ""),
                latency_ms=(time.monotonic() - start) * 1000.0,
                # ``provider_name`` is the dynamic field — it records
                # WHICH OSS provider produced each call. Adding a new
                # provider via the registry surfaces here unchanged.
                # ``images_count`` lets the audit trail distinguish
                # vision calls from text-only calls without the rationale
                # branching on the value.
                extra={
                    "provider_name": self.provider_name,
                    "base_url": _redact_base_url_for_capture(self.base_url),
                    "images_count": len(images) if images else 0,
                },
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
                "OpenAICompatibleBackend does not support streaming "
                "(deferred per decision O3). Call with stream=False."
            )
        # Off-thread the blocking HTTP call so the event loop isn't pinned.
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


def resolve_openai_compatible_backend(
    provider_name: str,
    *,
    model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
    capture: Optional[Any] = None,
    request_timeout: float = 120.0,
) -> OpenAICompatibleBackend:
    """Resolve a registry entry into a built ``OpenAICompatibleBackend``.

    The resolution chain per registry field:

    - ``base_url``: env var (when ``base_url_env`` is set and populated)
      → ``base_url_default``.
    - ``api_key``: explicit ``api_key_override`` → env var → registry
      ``api_key_default`` (which may be ``None``). When the registry
      entry sets ``api_key_required=True`` and the resolved key is
      empty, a ``RuntimeError`` fires before the backend is built.
    - ``default_model``: explicit ``model_override`` → env var →
      ``model_default``.

    Unknown ``provider_name`` raises ``ValueError`` listing every
    registered provider so the operator sees the registry surface
    inline. Add a new provider by appending an entry to
    ``_OPENAI_COMPATIBLE_PROVIDERS`` — there is no subclassing path.
    """
    entry = _OPENAI_COMPATIBLE_PROVIDERS.get(provider_name)
    if entry is None:
        registered = sorted(_OPENAI_COMPATIBLE_PROVIDERS.keys())
        raise ValueError(
            f"Unknown OpenAI-compatible provider: {provider_name!r}. "
            f"Registered providers: {registered}. Add a registry entry "
            f"to _OPENAI_COMPATIBLE_PROVIDERS in "
            f"MCP/orchestrator/llm_backend.py — no subclassing required."
        )

    base_url_env = entry.get("base_url_env")
    base_url = (
        os.environ.get(base_url_env) if base_url_env else None
    ) or entry["base_url_default"]

    api_key_env = entry.get("api_key_env")
    resolved_api_key = api_key_override
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.environ.get(api_key_env)
    if not resolved_api_key:
        resolved_api_key = entry.get("api_key_default")

    if entry.get("api_key_required", False) and not resolved_api_key:
        raise RuntimeError(
            f"Provider {provider_name!r} requires {api_key_env}; set the "
            f"environment variable or pass api_key_override="
        )

    model_env = entry.get("model_env")
    resolved_model = (
        model_override
        or (os.environ.get(model_env) if model_env else None)
        or entry["model_default"]
    )

    if entry.get("unverified", False):
        logger.warning(
            "resolve_openai_compatible_backend: provider %r is marked "
            "unverified in the registry; double-check base_url=%s and "
            "model=%s before relying on it.",
            provider_name, base_url, resolved_model,
        )

    # W-D13: per-entry vision capability with an env-var override
    # path. ``vision_capable_env`` (when set on the entry) lets an
    # operator flip the entry's default at deployment time — e.g.
    # ``LOCAL_VISION_CAPABLE=true`` for an Ollama install carrying a
    # vision-capable model (llava, llama3.2-vision, qwen2.5-vl).
    # Heuristic: the model identifier carrying ``vision`` / ``llava``
    # / ``-vl`` substrings ALSO flips the flag (covers the common
    # case where an operator just sets ``LOCAL_SYNTHESIS_MODEL`` to a
    # vision model without remembering the env-var hatch).
    vision_capable = bool(entry.get("vision_capable", False))
    vision_env = entry.get("vision_capable_env")
    if vision_env:
        env_val = (os.environ.get(vision_env) or "").strip().lower()
        if env_val in {"true", "1", "yes", "on"}:
            vision_capable = True
    if not vision_capable:
        # Substring heuristic on the resolved model identifier — common
        # surface forms across Ollama / vLLM / Together's vision models.
        model_lower = (resolved_model or "").lower()
        if any(token in model_lower for token in ("vision", "llava", "-vl")):
            vision_capable = True

    return OpenAICompatibleBackend(
        provider_name=provider_name,
        base_url=base_url,
        default_model=resolved_model,
        api_key=resolved_api_key,
        capture=capture,
        request_timeout=request_timeout,
        vision_capable=vision_capable,
    )


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


class MockBackend(_CaptureMixin):
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

    provider_label = "mock"

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        response_fn: Optional[Callable[[str, str], str]] = None,
        fixture_dir: Optional[Path] = None,
        default_response: str = "",
        *,
        capture: Optional[Any] = None,
    ):
        self._responses: List[str] = list(responses) if responses else []
        self._response_fn = response_fn
        self._fixture_dir = Path(fixture_dir) if fixture_dir else None
        self._default_response = default_response
        self.calls: List[_MockCall] = []
        self._set_capture(capture)

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
        start = time.monotonic()
        text: str = ""
        try:
            text = self._resolve_response(system, user)
            return text
        finally:
            self._emit_llm_chat_capture(
                model=model or "mock",
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages_count=2 if system else 1,
                response_text_len=len(text or ""),
                latency_ms=(time.monotonic() - start) * 1000.0,
                extra={"images_count": len(images) if images else 0},
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
        self._record(system, user, model, max_tokens, temperature, stream, images)
        if stream:
            raise NotImplementedError(
                "MockBackend does not emulate streaming (deferred per O3). "
                "Call with stream=False in tests."
            )
        start = time.monotonic()
        text: str = ""
        try:
            text = self._resolve_response(system, user)
            return text
        finally:
            self._emit_llm_chat_capture(
                model=model or "mock",
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages_count=2 if system else 1,
                response_text_len=len(text or ""),
                latency_ms=(time.monotonic() - start) * 1000.0,
                stream=stream,
                extra={"images_count": len(images) if images else 0},
            )


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


def build_backend(
    spec: Optional[BackendSpec] = None,
    *,
    capture: Optional[Any] = None,
    **overrides: Any,
) -> LLMBackend:
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
            kwargs: Dict[str, Any] = {"default_model": model, "capture": capture}
            if timeout is not None:
                kwargs["timeout_seconds"] = float(timeout)
            return MailboxBrokeredBackend(mailbox, **kwargs)
        return LocalBackend(capture=capture)

    # api mode
    if provider == "mock":
        return MockBackend(responses=list(spec.mock_responses), capture=capture)
    if provider == "anthropic":
        api_key = overrides.get("api_key") or spec.api_key
        return AnthropicBackend(
            api_key=api_key,
            default_model=model or DEFAULT_ANTHROPIC_MODEL,
            capture=capture,
        )
    if provider == "openai":
        # W-D12: ``provider="openai"`` is now a deprecated alias for the
        # ``local`` OpenAI-compatible registry entry. The pre-W-D12
        # ``OpenAIBackend`` was a stub that raised on every call —
        # routing legacy callers through ``local`` lets a deployment
        # that pinned the legacy spelling keep running against a local
        # OpenAI-compatible server (Ollama / vLLM / llama.cpp / LM
        # Studio) with no code edit beyond an env tweak. Operators
        # should migrate to ``provider="local"`` (or another registry
        # entry) explicitly.
        warnings.warn(
            "LLM_PROVIDER=openai is deprecated; use 'local' or another "
            "OpenAI-compatible provider name. Available providers: "
            f"{sorted(_OPENAI_COMPATIBLE_PROVIDERS.keys())}. See the "
            "_OPENAI_COMPATIBLE_PROVIDERS registry in "
            "MCP/orchestrator/llm_backend.py.",
            DeprecationWarning,
            stacklevel=2,
        )
        api_key_override = overrides.get("api_key") or spec.api_key
        return resolve_openai_compatible_backend(
            "local",
            model_override=model,
            api_key_override=api_key_override,
            capture=capture,
        )
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        # W-D12 dynamic dispatch: any registered OpenAI-compatible
        # provider resolves through the registry, NOT via a
        # per-provider subclass. Adding a new provider is a registry
        # entry change.
        api_key_override = overrides.get("api_key") or spec.api_key
        return resolve_openai_compatible_backend(
            provider,
            model_override=model,
            api_key_override=api_key_override,
            capture=capture,
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Built-in providers: "
        f"['anthropic', 'mock']. OpenAI-compatible registry: "
        f"{sorted(_OPENAI_COMPATIBLE_PROVIDERS.keys())}. To add a new "
        f"provider, append a registry entry to "
        f"_OPENAI_COMPATIBLE_PROVIDERS in MCP/orchestrator/llm_backend.py."
    )
