"""Answer-backend resolution + client construction (D1).

The grounded-answer path uses a local OpenAI-compatible model server
(Ollama / vLLM / llama.cpp / LM Studio) exactly like the synthesis
providers. Resolution reads the W-D12 registry
(``MCP.orchestrator.llm_backend._OPENAI_COMPATIBLE_PROVIDERS``) — NO new
registry, NO subclass — but constructs the layer-2
``OpenAICompatibleClient`` directly, because the backend layer doesn't
expose ``json_mode`` and the structured answer envelope needs the
Wave-113 lenient-JSON contract a 14B-Q4 model requires.

Single source of truth: ``_OPENAI_COMPATIBLE_PROVIDERS`` is itself a
projection of the ``openai_compatible`` rows in ``config/endpoints.yaml``
(loader ``lib/llm/endpoints.py``), so this path consumes the unified
endpoint registry transitively. The loopback gate below is a
CONSUMER-side policy overlay enforced here (NOT in the registry) — the
endpoint rows carry a ``loopback_only`` flag but the answer path imposes
loopback unconditionally on the RESOLVED url regardless.

Phase-IA posture (enforced here, no escape hatch):

* The resolved ``base_url`` host MUST be loopback. Anything else raises
  :class:`AnswerProviderNotLocal` naming the posture. The check is on the
  RESOLVED url, not the provider name, so a reverse-proxied cloud URL in
  ``LOCAL_SYNTHESIS_BASE_URL`` is also refused.
* A connection failure at call time surfaces as
  :class:`AnswerBackendUnavailable` — NEVER a canned-answer fallback.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lib.decision_capture import DecisionCapture
    from Trainforge.generators._openai_compatible_client import (
        OpenAICompatibleClient,
    )


# Env knobs (defaults documented in root CLAUDE.md + docs/LICENSING.md).
ENV_ANSWER_PROVIDER = "ED4ALL_ANSWER_PROVIDER"        # default "local"
ENV_ANSWER_MODEL = "ED4ALL_ANSWER_MODEL"              # default: registry model chain
ENV_ANSWER_TIMEOUT = "ED4ALL_ANSWER_TIMEOUT_SECONDS"  # default 120.0

_DEFAULT_PROVIDER = "local"
_DEFAULT_TIMEOUT = 120.0


class AnswerBackendUnavailable(RuntimeError):
    """Local model server absent/refusing. NEVER triggers a canned answer."""


class PromptTruncatedError(RuntimeError):
    """The model server silently truncated the prompt HEAD (fail-closed).

    Ollama (and other local servers) silently drop the leading tokens of a
    prompt that exceeds the served ``num_ctx`` window — the system prompt +
    leading passage ids vanish and the model fabricates citations. We detect
    this by comparing the server-reported ``usage.prompt_tokens`` against the
    local estimate: a reported count far BELOW the estimate means the head was
    dropped. Raised (fail-closed) instead of letting the silent truncation
    produce a fabricated answer.

    The message names the two operator fixes: raise the server window
    (``OLLAMA_CONTEXT_LENGTH`` / a Modelfile ``num_ctx`` PARAMETER) and tell
    the budget about it via ``ED4ALL_ANSWER_NUM_CTX``.
    """


class AnswerProviderNotLocal(ValueError):
    """Resolved base_url is non-loopback. Phase IA: no cloud answer path, ever."""


@dataclass(frozen=True)
class ResolvedAnswerBackend:
    provider_name: str
    model_id: str
    base_url: str
    api_key: Optional[str]
    timeout: float


def _is_loopback_host(base_url: str) -> bool:
    """True iff the URL's host is a loopback address / localhost."""
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A bare hostname that isn't "localhost" — treat as non-loopback.
        return False


def _resolve_timeout() -> float:
    raw = os.environ.get(ENV_ANSWER_TIMEOUT)
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


def resolve_answer_backend(
    provider_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> ResolvedAnswerBackend:
    """Resolve the answer backend via the W-D12 registry.

    Resolution chain (per field): explicit kwarg > env var > registry
    default. The provider name comes from ``provider_name`` kwarg, else
    ``ED4ALL_ANSWER_PROVIDER``, else ``"local"``.

    Enforcement: the resolved ``base_url`` host MUST be loopback —
    anything else raises :class:`AnswerProviderNotLocal`. There is
    deliberately NO escape-hatch env for cloud answer routing.
    """
    from MCP.orchestrator.llm_backend import _OPENAI_COMPATIBLE_PROVIDERS

    name = (
        provider_name
        or os.environ.get(ENV_ANSWER_PROVIDER)
        or _DEFAULT_PROVIDER
    )

    entry = _OPENAI_COMPATIBLE_PROVIDERS.get(name)
    if entry is None:
        registered = sorted(_OPENAI_COMPATIBLE_PROVIDERS.keys())
        raise ValueError(
            f"Unknown answer provider: {name!r}. Registered providers: "
            f"{registered}. Add a registry entry to "
            f"_OPENAI_COMPATIBLE_PROVIDERS in "
            f"MCP/orchestrator/llm_backend.py — no subclassing required."
        )

    base_url_env = entry.get("base_url_env")
    base_url = (
        (os.environ.get(base_url_env) if base_url_env else None)
        or entry["base_url_default"]
    )

    if not _is_loopback_host(base_url):
        raise AnswerProviderNotLocal(
            f"Answer provider {name!r} resolves to a non-loopback "
            f"base_url ({base_url!r}). Phase IA permits no cloud arm on "
            f"the answer path — the resolved URL host must be loopback "
            f"(localhost / 127.0.0.0/8 / [::1]). There is no escape-hatch "
            f"env for cloud answer routing."
        )

    # Resolve the answer model via the answer-specific env override
    # first, then the registry's own model_env, then the registry default.
    model_env = entry.get("model_env")
    resolved_model = (
        model_id
        or os.environ.get(ENV_ANSWER_MODEL)
        or (os.environ.get(model_env) if model_env else None)
        or entry["model_default"]
    )

    api_key_env = entry.get("api_key_env")
    resolved_api_key = (
        (os.environ.get(api_key_env) if api_key_env else None)
        or entry.get("api_key_default")
    )

    return ResolvedAnswerBackend(
        provider_name=name,
        model_id=resolved_model,
        base_url=base_url,
        api_key=resolved_api_key,
        timeout=_resolve_timeout(),
    )


def build_answer_client(
    resolved: Optional[ResolvedAnswerBackend] = None,
    *,
    capture: Optional["DecisionCapture"] = None,
    json_mode: bool = True,
) -> "OpenAICompatibleClient":
    """Build a layer-2 ``OpenAICompatibleClient`` for the answer path.

    The same ``capture`` handle threads the client's ``llm_chat_call``
    wire-level event into the composer's decision stream. ``None``
    resolved → :func:`resolve_answer_backend` with env/defaults.

    ``json_mode`` (default ``True``, Wave-113 lenient-JSON envelope):
    when ``True`` the request payload carries the JSON-mode fields
    (``format: "json"`` + ``response_format: {"type": "json_object"}``)
    so the grounded pipeline's structured answer-envelope parsing works.
    The default preserves the grounded pipeline's behavior byte-for-byte.
    Callers that need the model's RAW free-text output — e.g. the eval
    BASE arm, which measures what a raw LLM actually emits to a user —
    pass ``json_mode=False``; forcing a JSON envelope there would make
    scorer-v2's artifact filter discard the answer as a JSON literal,
    hiding the very fabrication that arm exists to measure.

    Connection-failure mapping (httpx errors → AnswerBackendUnavailable)
    is the composer's responsibility — this builder only constructs.
    """
    if resolved is None:
        resolved = resolve_answer_backend()

    # Lazy import keeps the heavy Trainforge module off the import path
    # for callers that only need resolution.
    from Trainforge.generators._openai_compatible_client import (
        OpenAICompatibleClient,
    )

    return OpenAICompatibleClient(
        base_url=resolved.base_url,
        model=resolved.model_id,
        api_key=resolved.api_key,
        capture=capture,
        timeout=resolved.timeout,
        provider_label=f"answer:{resolved.provider_name}",
        json_mode=json_mode,
    )


__all__ = [
    "ENV_ANSWER_PROVIDER",
    "ENV_ANSWER_MODEL",
    "ENV_ANSWER_TIMEOUT",
    "AnswerBackendUnavailable",
    "PromptTruncatedError",
    "AnswerProviderNotLocal",
    "ResolvedAnswerBackend",
    "resolve_answer_backend",
    "build_answer_client",
]
