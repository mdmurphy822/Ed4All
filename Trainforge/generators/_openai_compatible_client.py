#!/usr/bin/env python3
"""LLM-agnostic OpenAI-compatible chat-completions client.

Owns the OpenAI ``/v1/chat/completions`` wire format machinery — HTTP
client lifecycle, bounded retries, JSON parse, error mapping to
``SynthesisProviderError``, and decision-capture rationale construction.
Any backend that speaks the OpenAI shape (Together AI, Ollama, vLLM,
llama.cpp server, LM Studio, Fireworks, Groq, DeepInfra, hosted Mistral,
etc.) plugs in by passing ``base_url`` + optional ``api_key`` + model
identifier.

What this module deliberately does NOT own:

- Task-specific prompt construction (paraphrase vs classification have
  different system prompts).
- Length clamping (``PROMPT_MIN/MAX``, ``COMPLETION_MIN/MAX`` differ per
  task — synthesis caps differ from curriculum-alignment caps).
- Response post-processing / interpretation. The caller decides how to
  parse the assistant string.

Composition over inheritance is the contract: providers (Together,
Local, Curriculum) hold a client instance and delegate to
``chat_completion``. They do not inherit from this class. Adding a
new OpenAI-compatible provider is a single file with a small number
of task-specific signals — no HTTP wire duplication.

Decision-capture surface: every ``chat_completion`` call emits exactly
one ``decision_type="llm_chat_call"`` event when ``capture`` is wired.
Rationale interpolates ``provider_label``, ``model``, ``prompt_tokens``,
``completion_tokens``, ``http_retries`` plus everything the caller put
in ``decision_metadata`` — so a Together / Local / Curriculum / future
Fireworks call all share one audit shape but interpolate their own
task-specific signals (chunk_id, template_id, classification_target,
etc.). Rationale ≥20 chars per the project contract.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from Trainforge.generators._anthropic_provider import (  # noqa: F401
    SynthesisProviderError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants — defaults the client uses when the caller doesn't pass
# explicit values. None of these encode a specific backend's preferences;
# they're generic transport/retry tuning that any OpenAI-compatible server
# is expected to handle. Per-provider preferences live in the provider
# class that composes this client (Together / Local / Curriculum).
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_INITIAL_BACKOFF_SECONDS: float = 1.0
DEFAULT_PROVIDER_LABEL: str = "openai_compatible"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """LLM-agnostic chat completions client.

    Speaks the OpenAI ``/v1/chat/completions`` wire format. Any compatible
    backend (Together AI, Ollama, vLLM, llama.cpp server, LM Studio,
    Fireworks, DeepInfra, Groq, ...) plugs in by passing a base_url +
    optional api_key + model identifier.

    Owns: HTTP client lifecycle, retry policy, JSON parse, decision-
    capture rationale construction, error mapping to
    ``SynthesisProviderError``.

    Does NOT own: task-specific prompt construction, length clamping
    (mins/maxs differ per task — paraphrase vs classification),
    response post-processing (caller decides how to interpret content).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        capture: Optional[Any] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_status_codes: Sequence[int] = DEFAULT_RETRY_STATUS_CODES,
        initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
        provider_label: str = DEFAULT_PROVIDER_LABEL,
        client: Optional[httpx.Client] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Build an LLM-agnostic chat-completions client.

        Args:
            base_url: OpenAI-compatible API base. The trailing slash is
                stripped and ``/chat/completions`` is appended at request
                time. Example: ``"https://api.together.xyz/v1"``.
            model: Model identifier the backend understands. Sent in the
                request payload's ``model`` field.
            api_key: Bearer token. Optional — when ``None`` the
                Authorization header is omitted (some local servers
                ignore auth entirely).
            capture: Optional ``DecisionCapture``-shaped object. When
                set, every call emits one ``llm_chat_call`` event.
            timeout: Per-call HTTP timeout in seconds.
            max_retries: Total attempts the client makes per call,
                including the initial one. Status codes in
                ``retry_status_codes`` and transport errors trigger
                exponential backoff up to this cap.
            retry_status_codes: HTTP status codes the client treats as
                transient. Default covers 429 (rate limit) + the 5xx
                family.
            initial_backoff_seconds: Backoff for the first retry.
                Doubled per attempt.
            provider_label: Audit string. Surfaces in
                ``SynthesisProviderError`` messages and decision-capture
                rationales so post-hoc analysis can tell which backend
                produced each call. The Together / Local / Curriculum
                providers each pin a specific label.
            client: Optional pre-built ``httpx.Client``. Tests inject
                one with ``httpx.MockTransport``. Production callers
                let the property build one lazily.
        """
        if not base_url:
            raise ValueError("OpenAICompatibleClient requires a non-empty base_url")
        if not model:
            raise ValueError("OpenAICompatibleClient requires a non-empty model")
        self._base_url = base_url.rstrip("/")
        self._model = str(model)
        self._api_key = api_key
        self._capture = capture
        self._timeout = float(timeout)
        self._max_retries = max(1, int(max_retries))
        self._retry_status_codes = frozenset(int(s) for s in retry_status_codes)
        self._initial_backoff_seconds = float(initial_backoff_seconds)
        self._provider_label = str(provider_label)
        self._client = client
        # ``sleep_fn`` lets composing providers route the retry-backoff
        # sleep through their own module so fixtures patching e.g.
        # ``Trainforge.generators._together_provider.time.sleep`` keep
        # working post-refactor. Default is the stdlib ``time.sleep``.
        self._sleep_fn = sleep_fn or time.sleep

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_label(self) -> str:
        return self._provider_label

    @property
    def api_url(self) -> str:
        """Full chat-completions endpoint URL."""
        return f"{self._base_url}/chat/completions"

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.4,
        decision_metadata: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Issue one chat-completion call and return the assistant text.

        Args:
            messages: OpenAI-shaped messages list. Each entry is
                ``{"role": "system"|"user"|"assistant", "content": str}``.
                The caller fully owns prompt construction.
            max_tokens: Generation cap.
            temperature: Sampling temperature. The caller picks (0.0 for
                classification, 0.4 for paraphrase, etc.).
            decision_metadata: Optional dict of task-specific signals to
                interpolate into the decision-capture rationale (e.g.
                ``{"chunk_id": ..., "template_id": ...}``). The client
                does not inspect these — it just stringifies them in
                deterministic key order. Lets callers inject task
                semantics without the client needing to know what they
                are.
            extra_payload: Optional dict merged into the request body
                after the canonical fields are set. For backends that
                accept extra OpenAI-compatible knobs (``top_p``,
                ``stop``, ``response_format``, etc.) without the client
                needing to grow surface.

        Returns:
            The assistant message ``content`` string from the first
            choice.

        Raises:
            SynthesisProviderError: HTTP 4xx other than retried codes,
                retries exhausted on transient codes, malformed JSON
                response, missing ``choices`` array, or transport
                failure after retries.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra_payload:
            for key, value in extra_payload.items():
                if key in {"model", "messages"}:
                    continue
                payload[key] = value

        body, retry_count = self._post_with_retry(payload)
        text = self._extract_text(body)
        usage = self._extract_usage(body)

        self._emit_decision(
            text=text,
            usage=usage,
            retry_count=retry_count,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            decision_metadata=decision_metadata or {},
        )
        return text

    # ------------------------------------------------------------------
    # Internals — HTTP
    # ------------------------------------------------------------------

    def _post_with_retry(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """POST with bounded retries.

        Returns ``(parsed_response_dict, retry_count)``. ``retry_count``
        is the number of retries that fired (initial attempt is not
        counted). Surfaces 4xx outside ``retry_status_codes`` immediately
        as ``SynthesisProviderError(code=str(status))``. Persistent
        transient failures after ``max_retries`` attempts also raise.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = self.api_url
        provider_label = self._provider_label
        last_status: Optional[int] = None
        last_body: str = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_body = str(exc)
                last_status = None
                if attempt >= self._max_retries:
                    # Retry budget exhausted purely on transport-level
                    # failures (no HTTP status to surface). Distinct
                    # from the retries-exhausted-on-transient-status
                    # case below — that one surfaces ``code=str(status)``
                    # so the legacy provider behavior is preserved.
                    raise SynthesisProviderError(
                        f"{provider_label} request to {url} failed after "
                        f"{self._max_retries} transport attempts: {exc}",
                        code="max_retries_exceeded",
                    ) from exc
                self._sleep_for_attempt(attempt)
                continue

            status = response.status_code
            if status == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise SynthesisProviderError(
                        f"{provider_label} returned non-JSON 200 body: {exc}",
                        code="malformed_response",
                    ) from exc
                if not isinstance(body, dict):
                    raise SynthesisProviderError(
                        f"{provider_label} returned 200 body that wasn't a "
                        f"JSON object",
                        code="malformed_response",
                    )
                return body, attempt - 1

            last_status = status
            try:
                last_body = response.text[:500]
            except Exception:
                last_body = "<unreadable>"

            if (
                status in self._retry_status_codes
                and attempt < self._max_retries
            ):
                logger.warning(
                    "%s: transient HTTP %d on attempt %d/%d; retrying",
                    provider_label, status, attempt, self._max_retries,
                )
                self._sleep_for_attempt(attempt)
                continue

            # Either retries exhausted on a transient status or a non-
            # retryable 4xx (e.g. 400, 401, 403, 404). Both surface with
            # ``code=str(status)`` — the legacy provider behavior the
            # composing tests assert against.
            raise SynthesisProviderError(
                f"{provider_label} returned HTTP {status}: {last_body!r}",
                code=str(status),
            )

        # Defensive: loop exits via return / raise above. Reaching here
        # means the retry budget was exhausted purely on transport
        # errors (no HTTP status to surface), since any successful HTTP
        # status above would have either returned or raised already.
        raise SynthesisProviderError(
            f"{provider_label}: max retries exhausted on transport "
            f"errors (last_status={last_status}, last_body={last_body!r})",
            code="max_retries_exceeded",
        )

    def _sleep_for_attempt(self, attempt: int) -> None:
        """Exponential backoff, doubling per retry.

        Routes through ``self._sleep_fn`` so composing providers can
        forward their own ``time.sleep`` reference and keep test
        fixtures patching that module path effective.
        """
        backoff = self._initial_backoff_seconds * (2 ** (attempt - 1))
        self._sleep_fn(backoff)

    # ------------------------------------------------------------------
    # Internals — response shape
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> str:
        """Pull assistant content from an OpenAI-shaped response.

        Raises ``SynthesisProviderError(code="malformed_response")``
        when the response is missing the ``choices[0].message.content``
        path entirely. Empty-string content is returned as ``""`` for
        the caller to handle (some classification calls legitimately
        expect short or empty output).
        """
        if "choices" not in body:
            raise SynthesisProviderError(
                f"response missing required 'choices' field; keys present: "
                f"{sorted(body.keys())}",
                code="malformed_response",
            )
        choices = body.get("choices") or []
        if not choices:
            raise SynthesisProviderError(
                "response 'choices' array was empty",
                code="malformed_response",
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise SynthesisProviderError(
                "response 'choices[0]' was not a JSON object",
                code="malformed_response",
            )
        message = first.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _extract_usage(body: Dict[str, Any]) -> Dict[str, int]:
        """Pull token-usage tally. Missing fields default to 0."""
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            return {}

        def _g(name: str) -> int:
            v = usage.get(name)
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        return {
            "prompt_tokens": _g("prompt_tokens"),
            "completion_tokens": _g("completion_tokens"),
            "total_tokens": _g("total_tokens"),
        }

    # ------------------------------------------------------------------
    # Internals — decision capture
    # ------------------------------------------------------------------

    def _emit_decision(
        self,
        *,
        text: str,
        usage: Dict[str, int],
        retry_count: int,
        max_tokens: int,
        temperature: float,
        decision_metadata: Dict[str, Any],
    ) -> None:
        if self._capture is None:
            return
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        try:
            self._capture.log_decision(
                decision_type="llm_chat_call",
                decision=self._build_decision_string(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                    decision_metadata=decision_metadata,
                ),
                rationale=self._build_decision_rationale(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    decision_metadata=decision_metadata,
                    response_len=len(text or ""),
                ),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("llm_chat_call capture failed: %s", exc)

    def _build_decision_string(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        retry_count: int,
        decision_metadata: Dict[str, Any],
    ) -> str:
        meta_str = self._format_metadata(decision_metadata)
        suffix = f"; {meta_str}" if meta_str else ""
        return (
            f"{self._provider_label} chat call to model {self._model}; "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"http_retries={retry_count}{suffix}."
        )

    def _build_decision_rationale(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        retry_count: int,
        max_tokens: int,
        temperature: float,
        decision_metadata: Dict[str, Any],
        response_len: int,
    ) -> str:
        meta_str = self._format_metadata(decision_metadata)
        meta_clause = f" Task signals: {meta_str}." if meta_str else ""
        return (
            f"OpenAI-compatible chat call via provider_label="
            f"{self._provider_label}, base_url={self._base_url}, "
            f"model={self._model}, max_tokens={max_tokens}, "
            f"temperature={temperature}, prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"response_chars={response_len}, "
            f"http_retries={retry_count}.{meta_clause}"
        )

    @staticmethod
    def _format_metadata(decision_metadata: Dict[str, Any]) -> str:
        """Stringify task-specific signals in deterministic key order.

        Keeps decision-capture rationales reproducible across runs so
        replay tooling sees a stable string. Skips ``None`` values to
        keep the rationale tight.
        """
        if not decision_metadata:
            return ""
        parts: List[str] = []
        for key in sorted(decision_metadata.keys()):
            value = decision_metadata[key]
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return ", ".join(parts)


__all__ = [
    "OpenAICompatibleClient",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_STATUS_CODES",
    "DEFAULT_INITIAL_BACKOFF_SECONDS",
    "DEFAULT_PROVIDER_LABEL",
    "SynthesisProviderError",
]
