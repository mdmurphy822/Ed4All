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
from Trainforge.generators._synthesis_common import (  # noqa: F401
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


# ---------------------------------------------------------------------------
# W-D11 T11.3 — per-claim evidence-quote emit (synthesis side).
#
# T11.0 (commit 92d1bf8) landed two additive optional fields on the
# claim-bearing schemas:
#
#   * ``schemas/knowledge/chunk_v4.schema.json::key_claims`` structured
#     arm: ``evidence_quote`` (string, ≤240 chars, nullable) +
#     ``evidence_char_span`` ([start, end), nullable).
#   * ``schemas/knowledge/pair_audit_fields.schema.json::$defs.PerClaimSupport``:
#     same two fields, same null arms.
#
# T11.3 teaches every synthesis provider to ASK the LLM for the
# verbatim quote at synthesis time — the consumer-side validator
# (T11.2) handles the missing-quote case as warning-only so a small
# local model that ignores the schema gracefully degrades. The
# directive + schema fragment + emit-rate helper live ONCE here so
# every concrete provider (Anthropic / Together / Local /
# ClaudeSession) imports a single source of truth.
#
# Cap is structural — schema enforces ``maxLength: 240`` so a provider
# that emits a longer quote fails at downstream schema validation
# rather than the synthesis surface itself.
# ---------------------------------------------------------------------------


# Per-claim object shape, referenced by every provider's structured-
# output schema in JSON-mode. Optional fields → providers that fail to
# emit them gracefully degrade.
EVIDENCE_QUOTE_SCHEMA_FRAGMENT: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "claim": {"type": "string", "minLength": 1},
        "source_chunk_ids": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ],
        },
        # T11.3 optional emit — LLM is asked to fill these per-claim.
        "evidence_quote": {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 240},
                {"type": "null"},
            ],
        },
        "evidence_char_span": {
            "anyOf": [
                {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0},
                },
                {"type": "null"},
            ],
        },
    },
}


# Canonical per-claim prompt directive. Appended to every claim-bearing
# user prompt (block-emit instruction-paraphrase, block-emit
# preference-paraphrase, pair-side per_claim_support emit). The
# directive text mirrors the schema description so a small local model
# has the same contract whether it reads the prompt or the schema.
EVIDENCE_QUOTE_PROMPT_DIRECTIVE: str = (
    "\n\nIf your response includes per-claim attribution (a `key_claims[]` "
    "array on a chunk, or a `per_claim_support[]` array on a pair), "
    "include for each claim entry an `evidence_quote` field whose value "
    "is a contiguous verbatim substring (≤240 chars) copied character-"
    "for-character from the cited chunk's text that supports the claim. "
    "The quote MUST appear in the chunk text exactly as written — do "
    "NOT paraphrase. If you also emit `evidence_char_span`, it must be "
    "the [start, end) char offsets of the quote within the chunk's "
    "text field. If you cannot locate a contiguous span that supports "
    "the claim, emit `evidence_quote: null` and `evidence_char_span: "
    "null` rather than fabricating one. Example claim entry: "
    '{"claim": "SHACL shapes constrain RDF graphs.", '
    '"source_chunk_ids": ["chunk_001"], '
    '"evidence_quote": "SHACL shapes constrain RDF graphs.", '
    '"evidence_char_span": [12, 46]}.'
)


def _iter_claim_entries(parsed: Any) -> List[Dict[str, Any]]:
    """Walk a parsed provider response and return every claim-bearing
    object entry under ``key_claims[]`` or ``per_claim_support[]``.

    Tolerant of:
    - Missing keys (returns empty list).
    - Legacy List[str] shape on ``key_claims[]`` (skipped — bare-string
      claims have no per-claim object to stamp ``evidence_quote`` onto;
      they count as 0 claims for the emit-rate calculation since the
      provider can't be asked to attach a quote to a string).
    - Non-dict entries (skipped).
    - Nested envelope shapes (e.g. ``{"chunk": {...}}``). Walks dict
      values recursively but caps at depth 4 to bound cost.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return out

    def _walk(node: Any, depth: int) -> None:
        if depth > 4 or not isinstance(node, dict):
            return
        for key in ("key_claims", "per_claim_support"):
            arr = node.get(key)
            if isinstance(arr, list):
                for entry in arr:
                    if isinstance(entry, dict):
                        out.append(entry)
        for v in node.values():
            if isinstance(v, dict):
                _walk(v, depth + 1)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item, depth + 1)

    _walk(parsed, 0)
    return out


def compute_evidence_quote_emit_rate(parsed: Any) -> Tuple[int, int, float]:
    """Return ``(claims_with_quote, total_claims, emit_rate)``.

    Used by every concrete provider's ``_emit_per_call_decision`` to
    interpolate a dynamic ``evidence_quote_emit_rate`` signal into the
    ``synthesis_provider_call`` rationale. Mirrors the W5.D pattern of
    piggybacking on the existing decision_type rather than minting a
    new enum value.

    A claim entry "has a quote" when ``entry["evidence_quote"]`` is a
    non-empty string after strip. Null / missing / empty-string all
    count as "no quote", which aligns with the day-1 graceful-degrade
    contract — the consumer-side validator (T11.2) handles those arms
    as warning-only.

    When no claim-bearing arrays are present (the common case for
    today's providers, which paraphrase ``(prompt, completion)`` /
    ``(prompt, chosen, rejected)`` shapes only), returns ``(0, 0, 0.0)``
    — the rationale interpolates ``emit_rate=0.000 (no claims)`` so an
    operator can tell apart "no claims emitted" from "claims emitted
    without quotes".
    """
    entries = _iter_claim_entries(parsed)
    total = len(entries)
    if total == 0:
        return (0, 0, 0.0)
    with_quote = 0
    for entry in entries:
        q = entry.get("evidence_quote")
        if isinstance(q, str) and q.strip():
            with_quote += 1
    return (with_quote, total, with_quote / total)


def render_evidence_quote_rationale_fragment(parsed: Any) -> str:
    """Build the rationale-fragment string every provider appends.

    Returns a single-line interpolation suitable for concatenation
    into the broader ``synthesis_provider_call`` rationale. Always
    safe to append regardless of provider response shape.
    """
    with_quote, total, rate = compute_evidence_quote_emit_rate(parsed)
    if total == 0:
        return "evidence_quote_emit_rate=0.000 (no claims emitted)"
    return (
        f"evidence_quote_emit_rate={rate:.3f} "
        f"({with_quote}/{total} claims carry verbatim quotes)"
    )


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
    # W-D11 T11.3 — per-claim evidence-quote synthesis surface.
    "EVIDENCE_QUOTE_SCHEMA_FRAGMENT",
    "EVIDENCE_QUOTE_PROMPT_DIRECTIVE",
    "compute_evidence_quote_emit_rate",
    "render_evidence_quote_rationale_fragment",
]
