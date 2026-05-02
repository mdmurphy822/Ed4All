#!/usr/bin/env python3
"""Courseforge generators — shared LLM-agnostic base class.

Phase 3 Subtask 9: extract the HTTP / dispatch / decision-capture
skeleton out of :class:`Courseforge.generators._provider.ContentGeneratorProvider`
into a reusable abstract base. Phase 1's ``ContentGeneratorProvider``
becomes a thin subclass that overrides only ``_render_user_prompt``
(page-authoring) and the public ``generate_page`` entry point.
Phase 3's :class:`OutlineProvider` and :class:`RewriteProvider`
sibling subclasses share this base so the per-tier env-var contract
(``COURSEFORGE_PROVIDER`` / ``COURSEFORGE_OUTLINE_*`` /
``COURSEFORGE_REWRITE_*``) plugs in via constructor kwargs without
duplicating the dispatch plumbing.

Constructor surface:

- ``provider`` / ``model`` / ``api_key`` / ``base_url`` — operator
  knobs that fall back to env vars.
- ``capture`` — :class:`DecisionCapture` (optional).
- ``max_tokens`` / ``temperature`` — sampling.
- ``client`` / ``anthropic_client`` — test injection seams.
- ``env_provider_var`` — name of the env var the subclass reads to
  resolve the provider (e.g. ``COURSEFORGE_PROVIDER`` for Phase 1,
  ``COURSEFORGE_OUTLINE_PROVIDER`` for the outline tier).
- ``default_provider`` — the default when the env var is unset.
- ``default_model_anthropic`` / ``default_model_together`` /
  ``default_model_local`` — per-backend default model IDs the
  subclass passes through. Subclasses may resolve their own model
  via tier-specific env vars (e.g. ``COURSEFORGE_OUTLINE_MODEL``);
  the base only wires the per-backend baseline.
- ``default_base_url_local`` — default base URL for the ``local``
  backend (Ollama on :11434 by default).
- ``supported_providers`` — tuple of allowed provider strings the
  subclass enforces in its ``__init__``.
- ``system_prompt`` — the always-on authoring contract the subclass
  injects into every call.

Subclasses MUST override:

- ``_render_user_prompt(...) -> str`` — task-specific user prompt.
- ``_emit_per_call_decision(*, raw_text: str, retry_count: int,
  **call_context) -> None`` — task-specific decision-capture event
  whose ``decision_type`` matches the canonical enum at
  ``schemas/events/decision_event.schema.json``.

The base owns:

- ``_dispatch_call(user_prompt) -> Tuple[str, int]`` — routes to the
  Anthropic SDK or the OpenAI-compatible client.
- ``_call_anthropic(user_prompt) -> Tuple[str, int]`` — Anthropic
  SDK lazy-import + text extraction.
- ``_last_capture_id() -> str`` — Wave 112 audit-trail
  ``{file_basename}:{event_index}`` resolution.
- ``_emit_decision(*, decision_type, decision, rationale)`` —
  generic capture-emit helper subclasses call from their
  ``_emit_per_call_decision`` overrides.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Trainforge.generators._anthropic_provider import (
    DEFAULT_SYNTHESIS_MODEL as ANTHROPIC_DEFAULT_MODEL,
)
from Trainforge.generators._anthropic_provider import (
    ENV_API_KEY as ANTHROPIC_ENV_API_KEY,
)
from Trainforge.generators._local_provider import (
    DEFAULT_BASE_URL as LOCAL_DEFAULT_BASE_URL,
)
from Trainforge.generators._local_provider import (
    DEFAULT_SYNTHESIS_MODEL as LOCAL_DEFAULT_MODEL,
)
from Trainforge.generators._local_provider import (
    ENV_API_KEY as LOCAL_ENV_API_KEY,
)
from Trainforge.generators._local_provider import (
    ENV_BASE_URL as LOCAL_ENV_BASE_URL,
)
from Trainforge.generators._local_provider import (
    ENV_MODEL as LOCAL_ENV_MODEL,
)
from Trainforge.generators._openai_compatible_client import (
    OpenAICompatibleClient,
)
from Trainforge.generators._together_provider import (
    DEFAULT_BASE_URL as TOGETHER_DEFAULT_BASE_URL,
)
from Trainforge.generators._together_provider import (
    DEFAULT_SYNTHESIS_MODEL as TOGETHER_DEFAULT_MODEL,
)
from Trainforge.generators._together_provider import (
    ENV_API_KEY as TOGETHER_ENV_API_KEY,
)
from Trainforge.generators._together_provider import (
    ENV_MODEL as TOGETHER_ENV_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults shared across Courseforge LLM tiers.
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORTED_PROVIDERS: Tuple[str, ...] = ("anthropic", "together", "local")


class _BaseLLMProvider(ABC):
    """Shared LLM dispatch skeleton for Courseforge generator tiers.

    Subclasses (``ContentGeneratorProvider``, ``OutlineProvider``,
    ``RewriteProvider``) compose this base via ``super().__init__(...)``
    and override only the task-specific surface
    (``_render_user_prompt`` + ``_emit_per_call_decision`` + the
    public entry point such as ``generate_page`` /
    ``generate_outline`` / ``generate_rewrite``).

    The base itself is provider-agnostic: it resolves the backend
    (``anthropic`` / ``together`` / ``local``) from constructor kwargs
    or the subclass-supplied ``env_provider_var``, then wires either
    the lazy-imported Anthropic SDK or the shared
    :class:`OpenAICompatibleClient` for OpenAI-compatible backends.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capture: Optional[Any] = None,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        # Per-tier knobs supplied by subclasses.
        env_provider_var: str = "COURSEFORGE_PROVIDER",
        default_provider: str = "anthropic",
        default_model_anthropic: Optional[str] = None,
        default_model_together: Optional[str] = None,
        default_model_local: Optional[str] = None,
        default_base_url_local: Optional[str] = None,
        supported_providers: Tuple[str, ...] = _DEFAULT_SUPPORTED_PROVIDERS,
        system_prompt: str = "",
    ) -> None:
        resolved_provider = (
            provider
            or os.environ.get(env_provider_var)
            or default_provider
        ).lower()
        if resolved_provider not in supported_providers:
            raise ValueError(
                f"{type(self).__name__}: unknown provider "
                f"{resolved_provider!r}; expected one of "
                f"{list(supported_providers)}"
            )
        self._provider = resolved_provider
        self._capture = capture
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._system_prompt = system_prompt
        self._supported_providers = tuple(supported_providers)
        self._env_provider_var = env_provider_var

        # Per-backend default-model fallbacks. When a subclass passes
        # ``None`` for any of the per-backend defaults, fall back to
        # the project-wide synthesis defaults so a vanilla
        # ``_BaseLLMProvider`` works out of the box for tests.
        anthropic_baseline = (
            default_model_anthropic or ANTHROPIC_DEFAULT_MODEL
        )
        together_baseline = (
            default_model_together or TOGETHER_DEFAULT_MODEL
        )
        local_baseline = default_model_local or LOCAL_DEFAULT_MODEL
        local_base_url_baseline = (
            default_base_url_local or LOCAL_DEFAULT_BASE_URL
        )

        # Each branch resolves model / base_url / api_key off the
        # synthesis-pipeline env vars so an operator running a single
        # local server (Ollama on :11434, say) doesn't have to set a
        # separate COURSEFORGE_*_BASE_URL for the same endpoint.
        if resolved_provider == "anthropic":
            self._model = (
                model
                or os.environ.get("ANTHROPIC_SYNTHESIS_MODEL")
                or anthropic_baseline
            )
            resolved_key = api_key or os.environ.get(ANTHROPIC_ENV_API_KEY)
            if anthropic_client is None and not resolved_key:
                raise RuntimeError(
                    f"{ANTHROPIC_ENV_API_KEY} required for "
                    f"{type(self).__name__}(provider='anthropic'); "
                    "set the env var or inject an anthropic_client "
                    "(tests)."
                )
            self._api_key = resolved_key
            self._anthropic_client = anthropic_client
            self._oa_client: Optional[OpenAICompatibleClient] = None
            self._base_url: Optional[str] = None

        elif resolved_provider == "together":
            self._model = (
                model
                or os.environ.get(TOGETHER_ENV_MODEL)
                or together_baseline
            )
            resolved_key = api_key or os.environ.get(TOGETHER_ENV_API_KEY)
            if client is None and not resolved_key:
                raise RuntimeError(
                    f"{TOGETHER_ENV_API_KEY} required for "
                    f"{type(self).__name__}(provider='together'); "
                    "set the env var or inject a client (tests)."
                )
            self._api_key = resolved_key
            self._base_url = (
                base_url or TOGETHER_DEFAULT_BASE_URL
            ).rstrip("/")
            self._oa_client = OpenAICompatibleClient(
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
                capture=None,
                provider_label="together",
                client=client,
            )
            self._anthropic_client = None

        else:  # local
            self._model = (
                model
                or os.environ.get(LOCAL_ENV_MODEL)
                or local_baseline
            )
            resolved_key = (
                api_key
                or os.environ.get(LOCAL_ENV_API_KEY)
                or "local"
            )
            self._api_key = resolved_key
            env_base_url = os.environ.get(LOCAL_ENV_BASE_URL)
            self._base_url = (
                base_url or env_base_url or local_base_url_baseline
            ).rstrip("/")
            self._oa_client = OpenAICompatibleClient(
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
                capture=None,
                provider_label="local",
                client=client,
            )
            self._anthropic_client = None

    # ------------------------------------------------------------------
    # Abstract surface (subclass MUST override)
    # ------------------------------------------------------------------

    @abstractmethod
    def _render_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Render the task-specific user prompt for this tier."""
        raise NotImplementedError

    @abstractmethod
    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Emit one decision-capture event per LLM call.

        Subclasses pick the canonical ``decision_type`` (e.g.
        ``content_generator_call`` for Phase 1, ``block_outline_call``
        for the outline tier) and interpolate the per-call rationale
        per the project's LLM call-site instrumentation contract
        (≥20 chars, dynamic signals interpolated).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dispatch plumbing (shared)
    # ------------------------------------------------------------------

    def _dispatch_call(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Route through the selected backend; return ``(text, retries)``.

        Together / Local both go through
        :class:`OpenAICompatibleClient` via the embedded
        ``self._oa_client``. We drop down to ``_post_with_retry`` so
        the retry count surfaces on the decision-capture rationale.
        Anthropic routes through the SDK via :meth:`_call_anthropic`.

        Phase 3 Subtask 21: ``extra_payload`` is an optional dict whose
        keys are merged into the OpenAI-compatible request body before
        the POST. The Phase 3 router uses this to plumb per-block-type
        grammar / JSON-Schema payloads (``grammar``, ``guided_json``,
        ``guided_grammar``, ``guided_regex``, ``format`` as a JSON-Schema
        dict for Ollama 0.5+, ``response_format`` for json_schema mode)
        through to the wire without mutating the client. Caller-supplied
        keys take precedence over the base payload (a `model` override
        in ``extra_payload`` would replace the constructor-resolved
        model). When ``provider == "anthropic"``, ``extra_payload`` is
        ignored — the Anthropic SDK does not accept arbitrary
        OpenAI-compatible fields.
        """
        if self._provider == "anthropic":
            return self._call_anthropic(user_prompt)
        assert self._oa_client is not None
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if extra_payload:
            # Caller-supplied values win — mirrors
            # ``OpenAICompatibleClient.chat_completion`` (`:252-256`)
            # so the merge semantics stay consistent across the two
            # OpenAI-compatible call sites.
            for key, value in extra_payload.items():
                payload[key] = value
        body, retry_count = self._oa_client._post_with_retry(payload)
        text = self._oa_client._extract_text(body)
        return text, retry_count

    def _call_anthropic(self, user_prompt: str) -> Tuple[str, int]:
        """Run the call against the Anthropic SDK.

        Lazy-imports ``anthropic`` so callers using only Together /
        Local don't pay the import cost. Mirrors the
        :class:`Trainforge.generators._anthropic_provider.AnthropicSynthesisProvider`
        pattern for consistency. Returns
        ``(assistant_text, retry_count=0)`` — the SDK has its own
        retry policy so we don't double-count here.
        """
        client = self._anthropic_client
        if client is None:
            try:
                import anthropic  # noqa: PLC0415 — lazy by design
            except ImportError as exc:  # pragma: no cover — covered via mocks
                raise RuntimeError(
                    "anthropic package required for "
                    f"{type(self).__name__}(provider='anthropic'). "
                    "Install with: pip install anthropic"
                ) from exc
            client = anthropic.Anthropic(api_key=self._api_key)
            self._anthropic_client = client
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Pull text from a content list (mirrors the AnthropicSynthesisProvider
        # ``_extract_text``). Mocks may pass a dict; SDK passes typed objects.
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if not content:
            return "", 0
        parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "text":
                continue
            t = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if t:
                parts.append(str(t))
        return "".join(parts), 0

    # ------------------------------------------------------------------
    # Decision capture (shared)
    # ------------------------------------------------------------------

    def _last_capture_id(self) -> str:
        """Return ``{file_basename}:{event_index}`` for the most recent
        decision-capture event emitted via :meth:`_emit_decision`.

        Format mirrors the Wave 112 audit-trail convention so a
        ``Touch.decision_capture_id`` always resolves to the exact
        JSONL line that explained the LLM call. When the capture handle
        isn't a real :class:`DecisionCapture` (test injection of a
        ``_FakeCapture`` shape, ``capture=None``, or a streaming-disabled
        capture missing a stream path), falls back to
        ``in-memory:{id(self)}`` so the Wave 112 invariant
        (``decision_capture_id`` must be ≥1 char) is preserved without
        forcing tests to wire up a full capture surface.
        """
        capture = self._capture
        if capture is None:
            return f"in-memory:{id(self)}"

        # Resolve event index. ``DecisionCapture`` exposes ``decisions``;
        # the test fake exposes ``events``. Fall back to 0 when neither
        # surface is present.
        index: Optional[int] = None
        for attr in ("decisions", "events"):
            seq = getattr(capture, attr, None)
            if isinstance(seq, list):
                # ``log_decision`` was already called for the current
                # event, so the most recent entry sits at
                # ``len(seq) - 1``. Negative falls back to 0.
                index = max(len(seq) - 1, 0)
                break

        # Resolve file basename from the streaming-mode stream path
        # when present; otherwise tag the capture as in-memory.
        stream_path = getattr(capture, "_stream_path", None)
        if stream_path is not None:
            try:
                basename = Path(str(stream_path)).name
            except (TypeError, ValueError):  # pragma: no cover — defensive
                basename = None
            if basename:
                return f"{basename}:{index if index is not None else 0}"

        return f"in-memory:{id(self)}"

    def _emit_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
    ) -> None:
        """Generic capture-emit helper.

        Subclasses build the ``decision_type`` / ``decision`` /
        ``rationale`` strings (per the canonical decision_event enum
        at ``schemas/events/decision_event.schema.json``) and call
        through here so the swallow-on-error semantics live in one
        place.
        """
        if self._capture is None:
            return
        try:
            self._capture.log_decision(
                decision_type=decision_type,
                decision=decision,
                rationale=rationale,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "%s capture failed: %s", decision_type, exc
            )


__all__ = [
    "_BaseLLMProvider",
]
