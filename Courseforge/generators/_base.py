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
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Trainforge.generators._openai_compatible_client import (
    OpenAICompatibleClient,
    apply_reasoning_thinking_off_payload,
    maybe_append_usage_row,
)

# Per-backend IDENTITY (base_url / default_model) is sourced from the unified
# endpoint registry (``config/endpoints.yaml`` via ``lib/llm/endpoints.py``) so
# every Courseforge LLM tier — the local 7B included — attaches to a NAMED
# registry endpoint rather than a per-vendor constant. The env-var NAMES below
# (``*_ENV_*``) match each registry row's ``*_env`` field exactly; resolving
# them HERE (not in the registry) preserves the historical
# ``explicit kwarg > env > registry-default`` precedence byte-for-byte while
# keeping the YAML the single source of truth for the default base_url/model.

# Env-var name constants (the registry rows carry these same names in their
# ``*_env`` fields; kept here as the precedence-resolution keys).
ANTHROPIC_ENV_API_KEY = "ANTHROPIC_API_KEY"
LOCAL_ENV_API_KEY = "LOCAL_SYNTHESIS_API_KEY"
LOCAL_ENV_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"
LOCAL_ENV_MODEL = "LOCAL_SYNTHESIS_MODEL"
TOGETHER_ENV_API_KEY = "TOGETHER_API_KEY"
TOGETHER_ENV_MODEL = "TOGETHER_SYNTHESIS_MODEL"
NVIDIA_ENV_API_KEY = "NVIDIA_API_KEY"
NVIDIA_ENV_BASE_URL = "NVIDIA_BASE_URL"
NVIDIA_ENV_MODEL = "NVIDIA_LARGE_MODEL"


def _registry_default_model(endpoint_name: str, fallback: str) -> str:
    """Return ``endpoint_name``'s registry default_model (no env/key needed).

    Resolves against ``config/endpoints.yaml`` ignoring env overrides (the
    branch below applies the env chain itself), so the baseline is the YAML
    ``default_model``. Falls back to ``fallback`` if the registry can't be
    read (anti-cycle / missing-file hardening) so a vanilla provider never
    crashes at import.
    """
    try:
        from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

        row = load_endpoint_registry().get(endpoint_name)
        if row and row.get("default_model"):
            return str(row["default_model"])
    except Exception:  # noqa: BLE001 — defensive; never crash on registry I/O
        pass
    return fallback


def _registry_default_base_url(endpoint_name: str, fallback: str) -> str:
    """Return ``endpoint_name``'s registry base_url (ignoring env override)."""
    try:
        from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

        row = load_endpoint_registry().get(endpoint_name)
        if row and row.get("base_url"):
            return str(row["base_url"])
    except Exception:  # noqa: BLE001 — defensive
        pass
    return fallback


# Per-backend baselines projected from the registry rows. These are the
# LAST link in the ``explicit > env > default`` chain each branch applies.
ANTHROPIC_DEFAULT_MODEL = _registry_default_model("anthropic", "claude-sonnet-4-6")
LOCAL_DEFAULT_MODEL = _registry_default_model("local", "qwen2.5:7b-instruct-q4_K_M")
LOCAL_DEFAULT_BASE_URL = _registry_default_base_url("local", "http://localhost:11434/v1")
TOGETHER_DEFAULT_MODEL = _registry_default_model(
    "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)
TOGETHER_DEFAULT_BASE_URL = _registry_default_base_url(
    "together", "https://api.together.xyz/v1"
)
NVIDIA_DEFAULT_MODEL = _registry_default_model(
    "nvidia", "nvidia/nemotron-3-nano-30b-a3b"
)
NVIDIA_DEFAULT_BASE_URL = _registry_default_base_url(
    "nvidia", "https://integrate.api.nvidia.com/v1"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults shared across Courseforge LLM tiers.
# ---------------------------------------------------------------------------

def _default_supported_providers() -> Tuple[str, ...]:
    """Compose the base's default provider allow-list from the registry.

    ``anthropic`` (SDK transport) plus EVERY ``openai_compatible`` endpoint
    row in ``config/endpoints.yaml`` — so a vanilla ``_BaseLLMProvider``
    can reach ALL registry endpoints (``nvidia-deepseek``, ``together-
    vision``, ``groq``, ``fireworks``, ``deepseek``, …), not just the four
    per-vendor branches the legacy hardcoded tuple admitted. The
    ``claude_session`` dispatcher row is intentionally EXCLUDED — it is not
    an HTTP endpoint and the base has no dispatcher plumbing (the rewrite
    tier intercepts it before ``super().__init__``). Subclasses that pin a
    narrower ``supported_providers`` (e.g. the rewrite / outline tiers'
    ``("anthropic", "together", "local", "nvidia")``) keep restricting;
    this only widens the DEFAULT used when a subclass passes none.

    Anti-cycle / missing-file hardening: a registry read failure falls back
    to the legacy trio so a vanilla provider never crashes at import.
    """
    providers = ["anthropic"]
    try:
        from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

        for name, row in load_endpoint_registry().items():
            if str(row.get("kind")) == "openai_compatible":
                providers.append(name)
    except Exception:  # noqa: BLE001 — defensive; never crash on registry I/O
        providers.extend(["together", "local", "nvidia"])
    # De-dupe preserving registry order (``anthropic`` first).
    seen: set = set()
    out: List[str] = []
    for p in providers:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


_DEFAULT_SUPPORTED_PROVIDERS: Tuple[str, ...] = _default_supported_providers()


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
        # Per-call HTTP timeout (seconds) for the OpenAI-compatible
        # backends (together / local). ``None`` lets the
        # OpenAICompatibleClient apply its own DEFAULT_TIMEOUT_SECONDS.
        # Long-context tiers (e.g. textbook synthesis) pass a larger
        # value through their subclass __init__.
        timeout: Optional[float] = None,
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
        # 2026-05-14: tier-specific JSON-mode opt-out. Outline tier
        # needs json_mode=True (canonical JSON output); rewrite tier
        # needs json_mode=False (raw HTML output, no JSON wrapper).
        # Default True preserves backward compatibility for callers
        # that don't pass it.
        json_mode: bool = True,
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
        self._timeout = (
            float(timeout) if timeout is not None else None
        )
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
            # Anthropic SDK path forwards no OpenAI-compatible request-body
            # extras (the SDK does not accept arbitrary top-level fields).
            self._extra_body: Optional[Dict[str, Any]] = None

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
            # W9.2: build the client via the unified endpoint registry
            # (``build_openai_compatible_client``) instead of a direct
            # ``OpenAICompatibleClient(...)`` construction. The identity
            # (base_url / model / api_key) is PRE-RESOLVED above on the same
            # env vars + registry defaults and passed as explicit overrides,
            # so dispatch stays byte-identical for this already-reachable
            # seat; the registry ALSO surfaces the row's optional
            # ``extra_body`` (None for ``together``) which is threaded into
            # every request payload. Plan §3.2 ``json_mode`` opt-in is
            # forwarded verbatim.
            self._oa_client, self._extra_body = self._build_registry_oa_client(
                "together",
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                json_mode=json_mode,
                client=client,
            )
            self._anthropic_client = None

        elif resolved_provider == "nvidia":
            # NVIDIA hosted OpenAI-compatible inference API. Same wire
            # shape as ``together`` (a hosted ``/chat/completions``
            # endpoint behind a required bearer key). Model resolution
            # honors the per-tier env override (``COURSEFORGE_REWRITE_MODEL``
            # is applied upstream in the router projector for LOCAL tiers
            # only, so the NVIDIA tier keeps its YAML-pinned model; here we
            # resolve the NVIDIA-specific ``NVIDIA_LARGE_MODEL`` env override
            # and the default nemotron model). The API key is REQUIRED — the
            # hosted endpoint authenticates every request — and is read from
            # ``NVIDIA_API_KEY`` (never hardcoded).
            self._model = (
                model
                or os.environ.get(NVIDIA_ENV_MODEL)
                or NVIDIA_DEFAULT_MODEL
            )
            resolved_key = api_key or os.environ.get(NVIDIA_ENV_API_KEY)
            if client is None and not resolved_key:
                raise RuntimeError(
                    f"{NVIDIA_ENV_API_KEY} required for "
                    f"{type(self).__name__}(provider='nvidia'); "
                    "set the env var or inject a client (tests)."
                )
            self._api_key = resolved_key
            env_base_url = os.environ.get(NVIDIA_ENV_BASE_URL)
            self._base_url = (
                base_url or env_base_url or NVIDIA_DEFAULT_BASE_URL
            ).rstrip("/")
            # W9.2: registry-built client (pre-resolved identity passed as
            # explicit overrides → byte-identical) + row ``extra_body``.
            self._oa_client, self._extra_body = self._build_registry_oa_client(
                "nvidia",
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                json_mode=json_mode,
                client=client,
            )
            self._anthropic_client = None

        elif resolved_provider == "local":
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
            # W9.2: registry-built client. Identity is pre-resolved on the
            # SAME ``LOCAL_SYNTHESIS_*`` env vars + registry defaults (incl.
            # the ``default_base_url_local`` / ``default_model_local``
            # subclass baselines) and passed as explicit overrides, so the
            # local/default seat's dispatch (base_url / model / headers)
            # stays BYTE-IDENTICAL; ``extra_body`` is None for ``local``.
            self._oa_client, self._extra_body = self._build_registry_oa_client(
                "local",
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                json_mode=json_mode,
                client=client,
            )
            self._anthropic_client = None

        else:
            # W9.2 generic openai-compatible seat: any OTHER registry
            # endpoint the legacy hardcoded branches never reached
            # (``nvidia-deepseek``, ``together-vision``, ``groq``,
            # ``fireworks``, ``deepseek``, …). Identity (base_url / model /
            # api_key) resolves ENTIRELY from the endpoint row — the base
            # holds no per-vendor env constants for these — via
            # ``build_openai_compatible_client`` (kwarg overrides win when
            # supplied). The row's optional ``extra_body`` (e.g.
            # ``nvidia-deepseek``'s ``chat_template_kwargs:{thinking:false}``
            # reasoning-suppression) is captured and threaded into every
            # request payload.
            self._oa_client, self._extra_body = self._build_registry_oa_client(
                resolved_provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                json_mode=json_mode,
                client=client,
            )
            self._model = self._oa_client.model
            self._base_url = self._oa_client.base_url
            self._api_key = api_key
            self._anthropic_client = None

    # ------------------------------------------------------------------
    # Registry-driven client construction (W9.2)
    # ------------------------------------------------------------------

    def _build_registry_oa_client(
        self,
        endpoint_name: str,
        *,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        json_mode: bool,
        client: Optional[Any],
    ) -> Tuple[OpenAICompatibleClient, Optional[Dict[str, Any]]]:
        """Build an OpenAI-compatible client from the unified registry.

        The ONE constructor for every openai-compatible Courseforge seat:
        resolves ``endpoint_name`` through
        :func:`lib.llm.endpoints.build_openai_compatible_client` (kwarg
        overrides win with the frozen ``explicit > *_env > registry
        default`` precedence) and ALSO returns the endpoint row's optional
        ``extra_body`` request-body extras (``None`` when the row declares
        none — byte-stable). Consumers thread the returned ``extra_body``
        into every request payload (see :meth:`_dispatch_call_with_usage`)
        so vendor-specific fields (e.g. a reasoning model's
        ``chat_template_kwargs:{thinking:false}``) are no longer dropped.

        ``provider_label`` is pinned to the endpoint name so decision-
        capture rationales keep naming the seat exactly as the legacy
        per-vendor branches did. ``timeout`` is forwarded from
        ``self._timeout`` (``None`` → the client resolves its own default).
        """
        from lib.llm.endpoints import (  # noqa: PLC0415 — lazy, transport-free
            build_openai_compatible_client,
            load_endpoint_registry,
        )

        oa_client = build_openai_compatible_client(
            endpoint_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            capture=None,
            provider_label=endpoint_name,
            client=client,
            json_mode=json_mode,
            timeout=self._timeout,
        )
        # Read the row's optional ``extra_body`` directly (a copy so a caller
        # mutating the resolved view can't poison the cached registry).
        # Reading the row avoids re-running ``resolve_endpoint``'s
        # api_key_required check on an injected-client seat.
        row = load_endpoint_registry().get(endpoint_name) or {}
        raw_extra_body = row.get("extra_body")
        extra_body = (
            dict(raw_extra_body) if isinstance(raw_extra_body, dict) else None
        )
        return oa_client, extra_body

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
        text, retry_count, _usage = self._dispatch_call_with_usage(
            user_prompt, extra_payload=extra_payload
        )
        return text, retry_count

    def _dispatch_call_with_usage(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Like :meth:`_dispatch_call` but ALSO returns the usage dict.

        Returns ``(text, retry_count, usage)`` where ``usage`` is the
        server-reported token tally (``{"prompt_tokens", ...}``) extracted
        from the OpenAI-compatible response body. The Anthropic branch
        returns an EMPTY usage dict (the SDK response is parsed for text
        only, with no per-request prompt-token tally threaded here) — a
        genuine no-op signal so a downstream input-truncation tripwire
        fail-OPENs on the Anthropic path.

        Introduced for the rewrite-overflow-fix-2026-06 tripwire: usage
        travels by RETURN VALUE (not a shared-mutable ``last_usage`` on the
        client), so a cloud block can never read a stale OAI count from a
        prior local call. The legacy 2-tuple :meth:`_dispatch_call` stays
        the surface every other tier consumes.
        """
        if self._provider == "anthropic":
            text, retry_count = self._call_anthropic(user_prompt)
            return text, retry_count, {}
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
        # W9.2: thread the resolved endpoint row's ``extra_body`` request-
        # body extras (e.g. a reasoning model's
        # ``chat_template_kwargs:{thinking:false}``) at the top level of the
        # body. ``None`` for every legacy seat → byte-stable no-op. Applied
        # BEFORE ``extra_payload`` so a caller-supplied per-call override
        # still wins.
        if self._extra_body:
            for key, value in self._extra_body.items():
                payload[key] = value
        if extra_payload:
            # Caller-supplied values win — mirrors
            # ``OpenAICompatibleClient.chat_completion`` (`:252-256`)
            # so the merge semantics stay consistent across the two
            # OpenAI-compatible call sites.
            for key, value in extra_payload.items():
                payload[key] = value
        # Nemotron "detailed thinking off": this call site builds the payload
        # dict directly and POSTs via ``_post_with_retry``, BYPASSING
        # ``OpenAICompatibleClient.chat_completion`` where the
        # ``ED4ALL_REASONING_THINKING_OFF`` dual-injection normally lives. Apply
        # it here so a reasoning model (nemotron_v3) served on the local /
        # together / nvidia / registry OpenAI-compatible seat doesn't spend its
        # ``max_tokens`` budget on ``<think>`` tokens and trip the
        # ``finish_reason='length'`` truncation guard (the objective_extraction
        # / textbook-synthesis failure mode). No-op when the env is off
        # (byte-identical). Applied AFTER the ``extra_body`` + ``extra_payload``
        # merges so an explicit caller / endpoint-row override still wins.
        apply_reasoning_thinking_off_payload(payload)
        _call_start = time.monotonic()
        body, retry_count = self._oa_client._post_with_retry(payload)
        _duration_ms = (time.monotonic() - _call_start) * 1000.0
        usage = body.get("usage") if isinstance(body, dict) else None
        usage_dict: Dict[str, Any] = usage if isinstance(usage, dict) else {}
        # OP2 usage tap — this call site POSTs via ``_post_with_retry``
        # DIRECTLY (bypassing ``chat_completion``, see the thinking-off note
        # above), so ``_maybe_append_usage_row`` never fires for it. Mirror
        # the tap here through the SHARED module-level helper so every
        # Courseforge tier dispatch (outline / rewrite / content-generator /
        # the together-local outliner + textbook-synthesis branches) meters
        # into the same ``state/runs/<run_id>/llm_usage.jsonl`` ledger.
        # Contract-identical: best-effort (the helper swallows everything),
        # gated on ``ED4ALL_RUN_ID``, real server-reported token counts only
        # (an absent usage block defaults to 0 per the existing row shape),
        # and appended BEFORE ``_extract_text`` — which RAISES
        # ``output_truncated`` on ``finish_reason == "length"`` — so a
        # truncated call still records the row that names the truncation.
        maybe_append_usage_row(
            provider_label=getattr(
                self._oa_client, "provider_label", self._provider
            ),
            model=self._model,
            usage=usage_dict,
            duration_ms=_duration_ms,
            finish_reason=OpenAICompatibleClient._extract_finish_reason(body)
            if isinstance(body, dict)
            else None,
        )
        text = self._oa_client._extract_text(body)
        return text, retry_count, usage_dict

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
        alternatives_considered: Optional[List[str]] = None,
        inputs_ref: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Generic capture-emit helper.

        Subclasses build the ``decision_type`` / ``decision`` /
        ``rationale`` strings (per the canonical decision_event enum
        at ``schemas/events/decision_event.schema.json``) and call
        through here so the swallow-on-error semantics live in one
        place.

        ``alternatives_considered`` / ``inputs_ref`` are optional
        pass-throughs to ``DecisionCapture.log_decision``. Callers pass
        them ONLY when genuine values exist in scope (real code-path
        alternatives that were weighed / real input references the call
        consumed) — never fabricated or padded. The capture quality
        gate (``lib/quality.py::assess_decision_quality``) requires at
        least one of them non-empty for a "proficient" rating, so an
        emit that omits both is flagged for exclusion from the
        decision-capture training corpus.
        """
        if self._capture is None:
            return
        # Pass the optional fields ONLY when supplied so legacy call
        # sites keep their exact log_decision call shape (test fakes may
        # pin a strict keyword signature).
        extra: Dict[str, Any] = {}
        if alternatives_considered:
            extra["alternatives_considered"] = alternatives_considered
        if inputs_ref:
            extra["inputs_ref"] = inputs_ref
        try:
            self._capture.log_decision(
                decision_type=decision_type,
                decision=decision,
                rationale=rationale,
                **extra,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "%s capture failed: %s", decision_type, exc
            )


__all__ = [
    "_BaseLLMProvider",
]
