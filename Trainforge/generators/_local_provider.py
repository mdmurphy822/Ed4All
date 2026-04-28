#!/usr/bin/env python3
"""Local OpenAI-compatible synthesis provider — Wave 113 Task X.

Third synthesis path alongside ``anthropic`` / ``claude_session`` /
``together``. Speaks the same OpenAI-compatible chat-completions wire
shape Together AI uses, so almost the entire HTTP loop +
length-clamp + decision-capture wiring is inherited from
:class:`TogetherSynthesisProvider`. The differences are:

- Default base URL is the Ollama default
  (``http://localhost:11434/v1``); other servers (vLLM
  ``:8000/v1``, llama.cpp server ``:8080/v1``, LM Studio
  ``:1234/v1``) work after one env-var flip
  (``LOCAL_SYNTHESIS_BASE_URL``).
- API key is **optional**: most local servers ignore the auth
  header. The provider sends the placeholder string ``"local"`` when
  no key is resolved so reverse-proxies that DO check auth see a
  stable value rather than an unset header. ``LOCAL_SYNTHESIS_API_KEY``
  is still honoured when set.
- Default model is ``qwen2.5:14b-instruct-q4_K_M`` — a sensible
  out-of-box pick that fits an 8 GB GPU on Ollama.
  ``LOCAL_SYNTHESIS_MODEL`` overrides per server.
- ``out["provider"]`` is set to ``"local"`` so downstream consumers
  (pair schemas, the Wave-107 mock-corpus gate, the eval harness)
  can distinguish local-server output from the Together-hosted
  output.

Tradeoff vs Together: latency is 5-30s per call (depends on local
hardware + model size), but cost-per-call is zero after the hardware
investment and there's no ToS exposure — fully offline / air-gapped
synthesis is supported. Retry policy is identical (5xx / 429 with
exponential backoff up to ``MAX_HTTP_RETRIES`` attempts); 5xx is
slightly more likely on a local server, 429 is essentially never.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from Trainforge.generators._anthropic_provider import (  # noqa: F401
    COMPLETION_MAX,
    COMPLETION_MIN,
    PROMPT_MAX,
    PROMPT_MIN,
    SynthesisProviderError,
    _KIND_BOUNDS,
)
from Trainforge.generators._together_provider import (
    TogetherSynthesisProvider,
)

logger = logging.getLogger(__name__)


# Defaults — kept as module-level constants so callers (and tests) can
# import them without instantiating the provider.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_SYNTHESIS_MODEL = "qwen2.5:14b-instruct-q4_K_M"
ENV_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"
ENV_MODEL = "LOCAL_SYNTHESIS_MODEL"
ENV_API_KEY = "LOCAL_SYNTHESIS_API_KEY"


class LocalSynthesisProvider(TogetherSynthesisProvider):
    """Paraphrases mock-provider drafts via a local OpenAI-compatible server.

    Inherits the Together AI HTTP retry loop, JSON parsing, length
    clamping, and decision-capture wiring; overrides only the
    subclass-hook attributes that select the endpoint, model, env
    variables, auth requirement, and ``provider`` tag.

    Constructor accepts ``base_url`` / ``model`` / ``api_key`` as
    explicit kwargs; each falls back to its env var, then the class
    default. ``api_key`` is optional — when neither kwarg nor env var
    is set, the provider sends a placeholder so the Authorization
    header is always present (some reverse proxies require it).
    """

    # ------------------------------------------------------------------
    # Subclass hooks (overrides on TogetherSynthesisProvider).
    # ------------------------------------------------------------------
    _default_base_url: str = DEFAULT_BASE_URL
    _default_model: str = DEFAULT_SYNTHESIS_MODEL
    _default_api_key_env: str = ENV_API_KEY
    _default_model_env: str = ENV_MODEL
    _default_base_url_env: Optional[str] = ENV_BASE_URL
    _api_key_required: bool = False
    _provider_name: str = "local"

    # ------------------------------------------------------------------
    # Decision-capture string builders. Crucial difference from the
    # Together base class: the rationale interpolates ``base_url`` so
    # post-hoc audit can tell which local server (Ollama on workstation
    # X vs vLLM on workstation Y vs an air-gapped llama.cpp on the
    # offline node) produced each pair.
    # ------------------------------------------------------------------

    def _build_decision_string(
        self,
        *,
        kind: str,
        chunk_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        retry_count: int,
    ) -> str:
        return (
            f"Local-server paraphrase ({kind}) for chunk {chunk_id} "
            f"using model {self._model} at {self._base_url}; "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"http_retries={retry_count}."
        )

    def _build_decision_rationale(
        self,
        *,
        kind: str,
        draft: Dict[str, Any],
        chunk_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        retry_count: int,
    ) -> str:
        return (
            f"Routing template-generated {kind} draft "
            f"(template_id={draft.get('template_id','n/a')}, "
            f"draft_prompt_len={len(str(draft.get('prompt','')))}, "
            f"chunk_id={chunk_id}) through a local OpenAI-compatible "
            f"model server at base_url={self._base_url} using model "
            f"{self._model} for paraphrase. Local synthesis has zero "
            f"per-call cost after hardware setup and zero ToS exposure "
            f"(fully offline / air-gapped); the tradeoff is local "
            f"hardware capability and 5-30s per-call latency. "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"http_retries={retry_count}."
        )


__all__ = [
    "LocalSynthesisProvider",
    "DEFAULT_BASE_URL",
    "DEFAULT_SYNTHESIS_MODEL",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_API_KEY",
    "SynthesisProviderError",
]
