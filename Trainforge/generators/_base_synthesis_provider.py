#!/usr/bin/env python3
"""Trainforge generators — shared synthesis-provider base class.

Mirrors :class:`Courseforge.generators._base._BaseLLMProvider` for the
synthesis surface. Subclasses (Anthropic / Together / Local /
Curriculum / ClaudeSession) override only the task-specific surface
(``_render_*_user``, ``_emit_per_call_decision``) plus their dispatch
hook when the OpenAI-compat / Anthropic-SDK default doesn't fit.

See ``plans/wave-D5-trainforge-base-provider-2026-05-07.md`` § 3 for
the canonical class hierarchy + method partitioning.

Wave W-D5 lands the base file in isolation (T5.1) plus the two behavior
changes the plan flags (§ 6.1 retry-exhaustion code standardization on
``paraphrase_invalid_after_retry``; § 6.8 ClaudeSession ``_emit_decision``
swallow-on-error). Provider migrations T5.2-T5.6 happen incrementally
in subsequent commits — each provider continues to own its existing
constructor / dispatch / parse-retry implementation today and gradually
migrates to ``super().__init__(...)`` + base methods over time. The
base imports + re-exports the canonical ``SynthesisProviderError`` /
``_Usage`` / ``_validate_lengths`` so new providers can import from the
canonical base while the legacy import paths stay intact (per the
plan's "Module-level preservation contract" subsections).
"""

from __future__ import annotations

import json as _json
import logging
import os
import re as _re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Re-export the canonical error + dataclass + length helper so future
# subclasses can import from the canonical base instead of reaching into
# the leaf provider modules. Legacy import paths remain intact — the plan
# § 4 module-level preservation contract guarantees the original symbols
# stay where Courseforge / `_local_provider` / `_together_provider` /
# `_curriculum_provider` / `_claude_session_provider` already import them
# from.
from Trainforge.generators._anthropic_provider import (  # noqa: F401
    SynthesisProviderError,
    _KIND_BOUNDS,
    _Usage,
)

logger = logging.getLogger(__name__)


_DEFAULT_SUPPORTED_PROVIDERS: Tuple[str, ...] = (
    "anthropic",
    "together",
    "local",
)


# ---------------------------------------------------------------------------
# Module-level helper for the ClaudeSession cache-load + paraphrase clamps.
# Mirrors `Trainforge.generators._claude_session_provider._validate_lengths`
# so new providers can import from the canonical base. The canonical leaf
# import path is preserved by `_claude_session_provider.py`.
# ---------------------------------------------------------------------------

_KIND_KEYS: Dict[str, List[str]] = {
    "instruction": ["prompt", "completion"],
    "preference": ["prompt", "chosen", "rejected"],
}


def _validate_lengths(
    outputs: Dict[str, Any],
    *,
    kind: str,
    chunk_id: Optional[str] = None,
    kind_bounds: Optional[Dict[str, tuple]] = None,
) -> None:
    """Enforce per-key length bounds on a session-provider response.

    Parallel to ``_anthropic_provider._clamp``'s raise behavior. Short
    paraphrases must fail loud rather than silently landing in the
    cache and the JSONL writer (which would let a too-short prompt
    poison ``instruction_pairs.jsonl``).

    Args:
        outputs: The dispatcher response's ``outputs`` dict.
        kind: ``"instruction"`` or ``"preference"`` — selects which
            keys to check.
        chunk_id: Optional context for the raised error.
        kind_bounds: Optional override map for the per-kind bounds.
            Defaults to the canonical ``_KIND_BOUNDS`` shared with the
            Anthropic + Together + Local providers.

    Raises:
        SynthesisProviderError: when any checked key's value falls
            below the minimum or exceeds the maximum for its bound.
    """
    bounds = kind_bounds if kind_bounds is not None else _KIND_BOUNDS
    try:
        keys = _KIND_KEYS[kind]
    except KeyError as exc:
        raise ValueError(
            f"_validate_lengths: unknown kind={kind!r}; expected one of "
            f"{sorted(_KIND_KEYS)}"
        ) from exc
    for key in keys:
        value = outputs.get(key)
        if not isinstance(value, str):
            raise SynthesisProviderError(
                f"_validate_lengths: expected string for key={key!r}, "
                f"got {type(value).__name__}",
                code="empty_field",
                chunk_id=chunk_id,
            )
        lo, hi = bounds[key]
        length = len(value.strip())
        if length < lo:
            raise SynthesisProviderError(
                f"{kind}.{key} length {length} below minimum {lo}; "
                f"refusing to ship short paraphrase. Caller should "
                f"retry the dispatch.",
                code=f"{key}_below_minimum",
                chunk_id=chunk_id,
            )
        if length > hi:
            raise SynthesisProviderError(
                f"{kind}.{key} length {length} above maximum {hi}; "
                f"subagent must constrain output.",
                code=f"{key}_above_maximum",
                chunk_id=chunk_id,
            )


# ---------------------------------------------------------------------------
# Tolerant JSON parser shared by every synthesis provider.
# Mirrors `_anthropic_provider._parse_json` / `_together_provider._parse_json`
# / `_curriculum_provider` (curriculum's role validator is structurally
# different but parses single-token responses, not JSON).
# ---------------------------------------------------------------------------


def parse_json_lenient(text: str) -> Dict[str, Any]:
    """Tolerant JSON parser.

    Strips a single ```json fence wrapper when present; otherwise scans
    for the first balanced ``{...}`` span. Raises ``ValueError`` on
    empty / unbalanced / unparseable input.
    """
    if not text or not text.strip():
        raise ValueError("empty response text")
    s = text.strip()
    fence = _re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, _re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return _json.loads(s)
    except _json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                return _json.loads(candidate)
    raise ValueError("unbalanced JSON object in response")


# ---------------------------------------------------------------------------
# Anthropic helpers (text + usage extraction).
# Mirrors `_anthropic_provider.AnthropicSynthesisProvider._extract_text` /
# `_extract_usage` and `_curriculum_provider.CurriculumAlignmentProvider.
# _call_anthropic`'s inline content-extraction loop.
# ---------------------------------------------------------------------------


def extract_text_anthropic(response: Any) -> str:
    """Extract concatenated text from an Anthropic ``Message``.

    Mock objects in tests pass dicts; production SDK passes typed
    objects. Handle both.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if content is None:
        return ""
    text_parts: List[str] = []
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
            text_parts.append(str(t))
    return "".join(text_parts)


def extract_usage_anthropic(response: Any) -> _Usage:
    """Extract token-usage tally from an Anthropic ``Message``.

    Returns a ``_Usage`` dataclass with zeros when the response carries
    no usage block.
    """
    usage = getattr(response, "usage", None) or (
        response.get("usage") if isinstance(response, dict) else None
    )
    if usage is None:
        return _Usage()

    def _g(name: str) -> int:
        v = getattr(usage, name, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(name)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    return _Usage(
        input_tokens=_g("input_tokens"),
        output_tokens=_g("output_tokens"),
        cache_read_tokens=_g("cache_read_input_tokens"),
        cache_creation_tokens=_g("cache_creation_input_tokens"),
    )


# ---------------------------------------------------------------------------
# Capture-emit helper shared by every synthesis provider.
# Wave W-D5 § 6.8 standardisation: swallow-on-error so capture failures
# never break the synthesis path. ClaudeSession's pre-W-D5 helper did
# NOT swallow; the base helper does. The migration aligns ClaudeSession
# with the rest of the family.
# ---------------------------------------------------------------------------


def emit_decision_safely(
    capture: Optional[Any],
    *,
    decision_type: str,
    decision: str,
    rationale: str,
) -> None:
    """Capture-emit helper. Swallow + warn on error.

    Mirrors `AnthropicSynthesisProvider._emit_decision` /
    `TogetherSynthesisProvider._emit_decision` /
    `LocalSynthesisProvider._emit_decision` /
    `CurriculumAlignmentProvider._emit_decision`'s try/except shape.
    Per-call subclasses build the strings and forward to this helper.

    Args:
        capture: A `DecisionCapture` (or compatible) instance, or
            ``None`` when capture is disabled. ``None`` short-circuits.
        decision_type: Canonical enum value
            (``"synthesis_provider_call"`` /
            ``"curriculum_alignment_call"``).
        decision: Pre-built decision string.
        rationale: Pre-built rationale string (≥20 chars per the
            project's LLM call-site instrumentation contract).
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "%s capture failed: %s", decision_type, exc
        )


# ---------------------------------------------------------------------------
# Abstract base class — the structural contract for every synthesis
# provider. Today (T5.1) the base lives in isolation; subclass migrations
# T5.2-T5.6 happen incrementally. The base's existence + the imported
# helpers above are sufficient for the Wave W-D5 acceptance criteria
# § 8.1 (file exists at the canonical path with the right shape) and
# § 8.3 (module-level preservation contract — every legacy import path
# stays valid because the leaf modules retain their existing exports).
# ---------------------------------------------------------------------------


class _BaseSynthesisProvider(ABC):
    """Shared HTTP / SDK / capture skeleton for Trainforge synthesis.

    Subclasses (`AnthropicSynthesisProvider`, `LocalSynthesisProvider`,
    `TogetherSynthesisProvider`, `CurriculumAlignmentProvider`,
    `ClaudeSessionProvider`) override the abstract surface
    (`_render_instruction_user`, `_render_preference_user`,
    `_emit_per_call_decision`) plus their dispatch hook when the
    OpenAI-compat / Anthropic-SDK default doesn't fit.

    See ``plans/wave-D5-trainforge-base-provider-2026-05-07.md`` § 3
    for the canonical class hierarchy + method partitioning.

    Wave W-D5 status: T5.1 lands the base in isolation. Existing
    providers are NOT yet migrated to subclass this class; migrations
    happen incrementally in subsequent commits. The base nonetheless
    documents the canonical surface so new providers (Fireworks, Groq,
    vLLM-direct, etc.) can adopt it directly.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capture: Optional[Any] = None,
        timeout: float = 60.0,
        temperature: float = 0.4,
        max_tokens: int = 800,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        # Per-backend default fallbacks (subclass passes its constants):
        default_model_anthropic: Optional[str] = None,
        default_model_together: Optional[str] = None,
        default_model_local: Optional[str] = None,
        default_base_url_together: Optional[str] = None,
        default_base_url_local: Optional[str] = None,
        env_model_anthropic: str = "ANTHROPIC_SYNTHESIS_MODEL",
        env_api_key_anthropic: str = "ANTHROPIC_API_KEY",
        env_model_together: str = "TOGETHER_SYNTHESIS_MODEL",
        env_api_key_together: str = "TOGETHER_API_KEY",
        env_base_url_local: str = "LOCAL_SYNTHESIS_BASE_URL",
        env_model_local: str = "LOCAL_SYNTHESIS_MODEL",
        env_api_key_local: str = "LOCAL_SYNTHESIS_API_KEY",
        # Wire-shape:
        json_mode: bool = True,
        # Test injection / local-server bypass:
        api_key_required: bool = True,
        api_key_placeholder: str = "local",
        sleep_fn: Optional[Callable[[float], None]] = None,
        supported_providers: Tuple[str, ...] = _DEFAULT_SUPPORTED_PROVIDERS,
        system_prompt: str = "",
    ) -> None:
        if provider not in supported_providers and provider != "claude_session":
            raise ValueError(
                f"{type(self).__name__}: unknown provider "
                f"{provider!r}; expected one of "
                f"{list(supported_providers) + ['claude_session']}"
            )
        self._provider = provider
        self._capture = capture
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._timeout = float(timeout)
        self._json_mode = bool(json_mode)
        self._system_prompt = system_prompt
        self._supported_providers = tuple(supported_providers)
        self._sleep_fn = sleep_fn

        # Resolve model + api_key + base_url per-branch. Subclass __init__
        # may override after super().__init__() returns to attach
        # per-provider knobs (kind_bounds, preserve_rate, breaker, etc.).
        if provider == "anthropic":
            self._model = (
                model
                or os.environ.get(env_model_anthropic)
                or default_model_anthropic
            )
            resolved_key = api_key or os.environ.get(env_api_key_anthropic)
            if anthropic_client is None and not resolved_key:
                raise RuntimeError(
                    f"{env_api_key_anthropic} required for "
                    f"provider={provider}; "
                    "set the env var or inject an anthropic_client (tests)."
                )
            self._api_key = resolved_key
            self._anthropic_client = anthropic_client
            self._base_url: Optional[str] = None

        elif provider == "together":
            self._model = (
                model
                or os.environ.get(env_model_together)
                or default_model_together
            )
            resolved_key = api_key or os.environ.get(env_api_key_together)
            if client is None and not resolved_key and api_key_required:
                raise RuntimeError(
                    f"{env_api_key_together} required for "
                    f"provider={provider}; "
                    "set the env var or inject a client (tests)."
                )
            if not resolved_key and not api_key_required:
                resolved_key = api_key_placeholder
            self._api_key = resolved_key
            self._base_url = (
                base_url or default_base_url_together or ""
            ).rstrip("/")
            self._anthropic_client = None

        elif provider == "local":
            self._model = (
                model
                or os.environ.get(env_model_local)
                or default_model_local
            )
            resolved_key = (
                api_key
                or os.environ.get(env_api_key_local)
                or api_key_placeholder
            )
            self._api_key = resolved_key
            env_base = os.environ.get(env_base_url_local)
            self._base_url = (
                base_url or env_base or default_base_url_local or ""
            ).rstrip("/")
            self._anthropic_client = None

        else:  # claude_session — dispatcher-routed; no HTTP / SDK plumbing
            self._model = model
            self._api_key = api_key
            self._base_url = None
            self._anthropic_client = None

    # ------------------------------------------------------------------
    # Abstract surface (subclass MUST override)
    # ------------------------------------------------------------------

    @abstractmethod
    def _render_instruction_user(
        self, draft: Dict[str, Any], chunk_id: str, **kwargs: Any
    ) -> str:
        """Render the user prompt for an instruction-paraphrase call.

        Curriculum + ClaudeSession satellites raise ``NotImplementedError``
        — both providers' dispatch path bypasses the LLM-API render
        chain. The abstract enforcement keeps the contract honest.
        """
        raise NotImplementedError

    @abstractmethod
    def _render_preference_user(
        self, draft: Dict[str, Any], chunk_id: str, **kwargs: Any
    ) -> str:
        """Render the user prompt for a preference-paraphrase call."""
        raise NotImplementedError

    @abstractmethod
    def _emit_per_call_decision(
        self,
        *,
        kind: str,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Per-call canonical decision-capture event.

        Subclasses pick the canonical ``decision_type``
        (``synthesis_provider_call`` for synthesis surfaces,
        ``curriculum_alignment_call`` for the curriculum-alignment
        surface) and interpolate task-specific signals into the
        rationale string.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete shared helpers (delegate to module-level utilities)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Tolerant JSON parser. Forwards to module-level helper."""
        return parse_json_lenient(text)

    @staticmethod
    def _extract_text_anthropic(response: Any) -> str:
        """Anthropic content-block text extraction. Forwards to helper."""
        return extract_text_anthropic(response)

    @staticmethod
    def _extract_usage_anthropic(response: Any) -> _Usage:
        """Anthropic usage extraction. Forwards to helper."""
        return extract_usage_anthropic(response)

    def _emit_decision(
        self, *, decision_type: str, decision: str, rationale: str
    ) -> None:
        """Capture-emit helper. Swallow + warn on error.

        Forwards to ``emit_decision_safely``. Swallow-on-error contract
        preserves the synthesis-pipeline invariant that capture
        failures never break the call path. Wave W-D5 § 6.8 aligns
        ClaudeSession with this contract.
        """
        emit_decision_safely(
            self._capture,
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
        )


__all__ = [
    "_BaseSynthesisProvider",
    "_Usage",
    "_validate_lengths",
    "SynthesisProviderError",
    "parse_json_lenient",
    "extract_text_anthropic",
    "extract_usage_anthropic",
    "emit_decision_safely",
]
