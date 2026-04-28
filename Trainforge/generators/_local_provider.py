#!/usr/bin/env python3
"""Local OpenAI-compatible synthesis provider.

Third synthesis path alongside ``anthropic`` / ``claude_session`` /
``together``. Speaks the same OpenAI-compatible chat-completions wire
shape Together AI uses, so the entire HTTP loop + JSON parse +
decision-capture-rationale machinery is provided by composing one
:class:`Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`
instance — exactly as :class:`TogetherSynthesisProvider` does.
Composition over inheritance: this class is no longer a subclass of
``TogetherSynthesisProvider``; both providers compose the same
LLM-agnostic client and pin their own task semantics. Adding a future
provider (Fireworks, Groq, hosted Mistral, etc.) is the same shape:
one new file with a small constructor + paraphrase methods that
delegate to the embedded client.

Differences from the Together provider:

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

import json as _json
import logging
import os
import re as _re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from Trainforge.generators._anthropic_provider import (  # noqa: F401
    COMPLETION_MAX,
    COMPLETION_MIN,
    PROMPT_MAX,
    PROMPT_MIN,
    SynthesisProviderError,
    _KIND_BOUNDS,
)
from Trainforge.generators._openai_compatible_client import (
    OpenAICompatibleClient,
)
from Trainforge.generators._together_provider import (
    INITIAL_BACKOFF_SECONDS,
    MAX_HTTP_RETRIES,
    MAX_PARSE_RETRIES,
    _PREFERENCE_SYSTEM_PROMPT,
    _INSTRUCTION_SYSTEM_PROMPT,
    _RETRYABLE_STATUS,
)

logger = logging.getLogger(__name__)


# Defaults — kept as module-level constants so callers (and tests) can
# import them without instantiating the provider.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_SYNTHESIS_MODEL = "qwen2.5:14b-instruct-q4_K_M"
ENV_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"
ENV_MODEL = "LOCAL_SYNTHESIS_MODEL"
ENV_API_KEY = "LOCAL_SYNTHESIS_API_KEY"
DEFAULT_TIMEOUT = 60.0


class LocalSynthesisProvider:
    """Paraphrases mock-provider drafts via a local OpenAI-compatible server.

    Composes one :class:`OpenAICompatibleClient` configured for the
    local-server case (no auth required by default, base-URL env-var
    overridable). The HTTP retry loop, JSON parse of the response
    envelope, and ``llm_chat_call`` capture surface live in the embedded
    client; this provider only owns the paraphrase prompts, length
    clamping, parse-retry, and ``synthesis_provider_call`` audit emit.

    Constructor accepts ``base_url`` / ``model`` / ``api_key`` as
    explicit kwargs; each falls back to its env var, then the class
    default. ``api_key`` is optional — when neither kwarg nor env var
    is set, the provider sends a placeholder so the Authorization
    header is always present (some reverse proxies require it).
    """

    # Provider tag written to ``out["provider"]`` and surfaced in the
    # decision-capture event for audit.
    _provider_name: str = "local"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        capture: Optional[Any] = None,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = 0.4,
        max_tokens: int = 800,
    ) -> None:
        # API-key resolution. Local servers usually ignore auth; we
        # accept absence and substitute a stable placeholder so reverse
        # proxies that DO check auth see something rather than an unset
        # header.
        resolved_key = api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            resolved_key = "local"
        self._api_key = resolved_key

        self._model = (
            model
            or os.environ.get(ENV_MODEL)
            or DEFAULT_SYNTHESIS_MODEL
        )
        env_base_url = os.environ.get(ENV_BASE_URL)
        self._base_url = (
            base_url or env_base_url or DEFAULT_BASE_URL
        ).rstrip("/")
        self._capture = capture
        self._timeout = float(timeout)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

        # Composition: build the LLM-agnostic client. Same client class
        # the Together provider composes — only the configuration
        # differs.
        self._oa_client = OpenAICompatibleClient(
            base_url=self._base_url,
            model=self._model,
            api_key=self._api_key,
            capture=None,
            timeout=self._timeout,
            max_retries=MAX_HTTP_RETRIES,
            retry_status_codes=tuple(sorted(_RETRYABLE_STATUS)),
            initial_backoff_seconds=INITIAL_BACKOFF_SECONDS,
            provider_label=self._provider_name,
            client=client,
            # Route retry-backoff sleep through the together-provider
            # module's ``time.sleep`` so existing local-provider tests
            # that patch ``Trainforge.generators._together_provider.time.sleep``
            # (per the local test docstring's stated contract) keep
            # working post-refactor.
            sleep_fn=_local_sleep,
        )

    @property
    def api_url(self) -> str:
        """Full chat-completions endpoint URL for this provider."""
        return f"{self._base_url}/chat/completions"

    @property
    def client(self) -> httpx.Client:
        """Backwards-compat: return the underlying httpx client."""
        return self._oa_client.client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paraphrase_instruction(
        self, draft: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Paraphrase a draft instruction pair against its chunk."""
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        user_prompt = self._render_instruction_user(draft, chunk_id)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=_INSTRUCTION_SYSTEM_PROMPT,
            chunk_text=chunk_text,
            user_prompt=user_prompt,
            required_keys=("prompt", "completion"),
        )

        out = dict(draft)
        out["prompt"] = self._clamp(
            parsed["prompt"], kind="prompt", chunk_id=chunk_id
        )
        out["completion"] = self._clamp(
            parsed["completion"], kind="completion", chunk_id=chunk_id
        )
        out["provider"] = self._provider_name

        self._emit_decision(
            kind="instruction",
            draft=draft,
            chunk_id=chunk_id,
            usage=usage,
            retry_count=retry_count,
        )
        return out

    def paraphrase_preference(
        self, draft: Dict[str, Any], chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Paraphrase a draft preference triple against its chunk."""
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        user_prompt = self._render_preference_user(draft, chunk_id)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=_PREFERENCE_SYSTEM_PROMPT,
            chunk_text=chunk_text,
            user_prompt=user_prompt,
            required_keys=("prompt", "chosen", "rejected"),
        )

        out = dict(draft)
        out["prompt"] = self._clamp(
            parsed["prompt"], kind="prompt", chunk_id=chunk_id
        )
        out["chosen"] = self._clamp(
            parsed["chosen"], kind="chosen", chunk_id=chunk_id
        )
        out["rejected"] = self._clamp(
            parsed["rejected"], kind="rejected", chunk_id=chunk_id
        )
        out["provider"] = self._provider_name

        self._emit_decision(
            kind="preference",
            draft=draft,
            chunk_id=chunk_id,
            usage=usage,
            retry_count=retry_count,
        )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self, system_prompt: str, chunk_text: str, user_prompt: str
    ) -> List[Dict[str, str]]:
        full_system = (
            f"{system_prompt}\n\nSource chunk text:\n\n{chunk_text}"
        )
        return [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]

    def _call_with_parse(
        self,
        *,
        system_prompt: str,
        chunk_text: str,
        user_prompt: str,
        required_keys: tuple,
    ) -> Tuple[Dict[str, Any], Dict[str, int], int]:
        messages = self._build_messages(system_prompt, chunk_text, user_prompt)
        attempts = 0
        last_err: Optional[Exception] = None
        last_text: str = ""
        total_http_retries = 0
        last_usage: Dict[str, int] = {}
        while attempts < MAX_PARSE_RETRIES:
            attempts += 1
            text, usage, http_retries = self._chat_completion_raw(messages)
            total_http_retries += http_retries
            last_usage = usage
            last_text = text
            try:
                parsed = self._parse_json(text)
            except ValueError as exc:
                last_err = exc
                logger.warning(
                    "%s synthesis: parse retry %d/%d: %s",
                    self._provider_name, attempts, MAX_PARSE_RETRIES, exc,
                )
                continue
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                last_err = ValueError(
                    f"response missing required keys: {missing}"
                )
                continue
            return parsed, last_usage, total_http_retries
        raise SynthesisProviderError(
            f"{type(self).__name__}: failed to parse a valid JSON "
            f"response after {MAX_PARSE_RETRIES} attempts. "
            f"Last error: {last_err}; tail of last response: "
            f"{last_text[-200:]!r}",
            code="parse_retries_exhausted",
        )

    def _chat_completion_raw(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, int], int]:
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        body, retry_count = self._oa_client._post_with_retry(payload)
        text = self._oa_client._extract_text(body)
        usage = self._oa_client._extract_usage(body)
        return text, usage, retry_count

    @staticmethod
    def _render_instruction_user(
        draft: Dict[str, Any], chunk_id: str
    ) -> str:
        return (
            f"Chunk ID: {chunk_id}\n"
            f"Bloom level: {draft.get('bloom_level','unknown')}\n"
            f"Content type: {draft.get('content_type','unknown')}\n"
            f"Template ID: {draft.get('template_id','unknown')}\n"
            f"\n"
            f"Draft prompt:\n{draft.get('prompt','')}\n"
            f"\n"
            f"Draft completion:\n{draft.get('completion','')}\n"
            f"\n"
            f"Rewrite the prompt and completion. Return JSON with keys "
            f"'prompt' and 'completion'."
        )

    @staticmethod
    def _render_preference_user(
        draft: Dict[str, Any], chunk_id: str
    ) -> str:
        return (
            f"Chunk ID: {chunk_id}\n"
            f"Source: {draft.get('rejected_source','unknown')}\n"
            f"\n"
            f"Draft prompt:\n{draft.get('prompt','')}\n"
            f"\n"
            f"Draft chosen:\n{draft.get('chosen','')}\n"
            f"\n"
            f"Draft rejected:\n{draft.get('rejected','')}\n"
            f"\n"
            f"Rewrite all three. Return JSON with keys 'prompt', "
            f"'chosen', and 'rejected'."
        )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
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

    @staticmethod
    def _clamp(text: str, kind: str, *, chunk_id: Optional[str] = None) -> str:
        try:
            lo, hi = _KIND_BOUNDS[kind]
        except KeyError as exc:
            raise ValueError(
                f"_clamp: unknown kind={kind!r}; expected one of "
                f"{sorted(_KIND_BOUNDS)}"
            ) from exc
        s = (text or "").strip()
        if len(s) < lo:
            raise SynthesisProviderError(
                f"{kind} length {len(s)} below minimum {lo}; refusing to "
                f"inject sentinel filler. Caller should retry the paraphrase.",
                code=f"{kind}_below_minimum",
                chunk_id=chunk_id,
            )
        if len(s) > hi:
            hard = s[:hi]
            period = hard.rfind(". ")
            if period > lo:
                s = hard[: period + 1]
            else:
                s = hard.rstrip() + "..."
        return s

    def _emit_decision(
        self,
        *,
        kind: str,
        draft: Dict[str, Any],
        chunk_id: str,
        usage: Dict[str, int],
        retry_count: int,
    ) -> None:
        if self._capture is None:
            return
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        try:
            self._capture.log_decision(
                decision_type="synthesis_provider_call",
                decision=self._build_decision_string(
                    kind=kind,
                    chunk_id=chunk_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                ),
                rationale=self._build_decision_rationale(
                    kind=kind,
                    draft=draft,
                    chunk_id=chunk_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                ),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("synthesis_provider_call capture failed: %s", exc)

    # ------------------------------------------------------------------
    # Decision-capture string builders. Crucial difference from the
    # Together provider: the rationale interpolates ``base_url`` so
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


def _local_sleep(seconds: float) -> None:
    """Forward retry-backoff sleeps through the together-provider module.

    The local-provider tests patch
    ``Trainforge.generators._together_provider.time.sleep`` to keep
    test runs fast (per the test docstring). We honor that contract by
    routing the embedded client's backoff sleeps through that module's
    ``time.sleep`` reference rather than the local module's, so a
    single patch covers both providers' retry paths.
    """
    from Trainforge.generators import _together_provider as _tg

    _tg.time.sleep(seconds)


__all__ = [
    "LocalSynthesisProvider",
    "DEFAULT_BASE_URL",
    "DEFAULT_SYNTHESIS_MODEL",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_API_KEY",
    "SynthesisProviderError",
]
