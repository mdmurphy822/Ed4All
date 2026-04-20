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
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
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


def build_backend(spec: Optional[BackendSpec] = None, **overrides: Any) -> LLMBackend:
    """Build an ``LLMBackend`` from a spec + env fallbacks.

    Precedence: explicit ``overrides`` > ``spec`` fields > env vars > defaults.

    Recognized env vars: ``LLM_MODE``, ``LLM_PROVIDER``, ``LLM_MODEL``,
    ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``.
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
