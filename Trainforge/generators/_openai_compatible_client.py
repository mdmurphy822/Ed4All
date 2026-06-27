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

import json as _json
import logging
import math
import os
import random as _random
import re as _re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from Trainforge.generators._synthesis_common import (  # noqa: F401
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

# Stage-0 429 fix: when a retryable response carries a ``Retry-After``
# (or, secondarily, ``X-RateLimit-Reset``) header, the server is telling
# us EXACTLY how long to wait — honor it in preference to the local
# exponential backoff. This benefits every NVIDIA / Together / hosted-OSS
# consumer (Courseforge rewrite, objective-review, block-planner,
# SemantiK-via-client) uniformly, since they all compose this one client.
# The header value is clamped to this ceiling so a pathological /
# misconfigured header (e.g. ``Retry-After: 86400``) can't hang a run.
_RETRY_AFTER_MAX_SECONDS: float = 60.0

# Cross-cutting env override for the per-request HTTP timeout. When the
# caller does NOT pass an explicit ``timeout`` to the constructor, the
# client resolves its default from this env var (falling back to
# ``DEFAULT_TIMEOUT_SECONDS`` when unset / garbage / non-positive). The
# motivation is local 7B content-generation (Courseforge outline /
# rewrite tiers) authoring multi-paragraph prose, which routinely
# exceeds the 60s default. Resolution / precedence (high → low):
# explicit per-call ``timeout`` arg > ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS``
# > ``DEFAULT_TIMEOUT_SECONDS`` (60.0). Mirrors the parse-with-fallback
# robustness pattern of the other ``ED4ALL_*`` timeout knobs (e.g.
# ``ED4ALL_ANSWER_TIMEOUT_SECONDS`` in ``lib/retrieval/answer_backend.py``).
ENV_REQUEST_TIMEOUT: str = "ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS"


def resolve_default_timeout() -> float:
    """Return the default per-request timeout when no explicit value.

    Reads ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` and parses it as a
    positive float. A missing, unparseable, or non-positive value falls
    back to :data:`DEFAULT_TIMEOUT_SECONDS` (60.0) — garbage never
    silently shrinks the timeout to zero / negative.
    """
    raw = os.environ.get(ENV_REQUEST_TIMEOUT)
    if not raw or not str(raw).strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    # Reject NaN / inf / non-positive — garbage never shrinks the timeout.
    if not math.isfinite(parsed) or parsed <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return parsed


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
        timeout: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_status_codes: Sequence[int] = DEFAULT_RETRY_STATUS_CODES,
        initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
        provider_label: str = DEFAULT_PROVIDER_LABEL,
        client: Optional[httpx.Client] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        json_mode: bool = False,
        vision_capable: bool = False,
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
            timeout: Per-call HTTP timeout in seconds. When ``None``
                (the default), the client resolves its timeout from
                ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (via
                :func:`resolve_default_timeout`), falling back to
                ``DEFAULT_TIMEOUT_SECONDS`` (60.0) when that env var is
                unset / garbage. An explicit value passed here always
                wins over the env var.
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
            json_mode: Wave 113 — when ``True``, every request payload
                carries BOTH ``"format": "json"`` (the Ollama-style
                top-level field that triggers JSON-grammar-constrained
                decoding on Ollama 0.4+) AND
                ``"response_format": {"type": "json_object"}`` (the
                OpenAI-spec field). Servers that don't recognize one
                or the other ignore it silently — being permissive in
                what we send is cheap and gives defense in depth across
                hosted-OSS providers (Together, Fireworks, Groq) and
                local servers (Ollama, vLLM, llama.cpp, LM Studio) at
                once. Default ``False`` for backward compat — providers
                explicitly opt in.
            vision_capable: Wave W-D13 — declarative flag the BACKEND
                layer reads to decide whether to permit ``images=...``
                on a ``chat_completion`` call. The client itself does
                NOT enforce this (vision content-block translation is
                always wired); the flag is metadata so a calling
                ``OpenAICompatibleBackend`` can short-circuit a vision
                request against a non-vision-capable model server with
                a clear error before paying the HTTP round-trip.
                Default ``False``.
        """
        if not base_url:
            raise ValueError("OpenAICompatibleClient requires a non-empty base_url")
        if not model:
            raise ValueError("OpenAICompatibleClient requires a non-empty model")
        self._base_url = base_url.rstrip("/")
        self._model = str(model)
        self._api_key = api_key
        self._capture = capture
        # Explicit per-call timeout wins; otherwise resolve from
        # ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (fallback 60.0).
        self._timeout = (
            float(timeout)
            if timeout is not None
            else resolve_default_timeout()
        )
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
        self._json_mode = bool(json_mode)
        self._vision_capable = bool(vision_capable)
        # Server-reported token usage of the most recent chat_completion
        # call (``{}`` until the first call). Read by the grounded-answer
        # truncation tripwire; harmless for every other caller.
        self.last_usage: Dict[str, int] = {}

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
    def vision_capable(self) -> bool:
        return self._vision_capable

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
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.4,
        decision_metadata: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
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

                Phase 3 router contract: Courseforge's two-pass router
                routes per-block-type constrained-decoding payloads
                through this kwarg without the client growing per-
                provider branches. Recognised grammar-related fields
                that pass through unchanged:

                - ``grammar`` (str): GBNF grammar string for
                  llama.cpp-compatible servers.
                - ``format`` (dict): JSON-Schema dict for Ollama 0.5+
                  constrained-decoding mode (distinct from the
                  ``"json"`` literal that ``json_mode=True`` injects).
                - ``guided_grammar`` / ``guided_json`` /
                  ``guided_regex`` (vLLM): typically passed nested
                  inside ``extra_body`` per the vLLM API surface.
                - ``extra_body`` (dict): vLLM passthrough container
                  for the ``guided_*`` fields.
                - ``response_format`` (dict): OpenAI-style
                  json_schema mode (``{"type": "json_schema",
                  "json_schema": {...}}``); used by Together AI's
                  strict structured-output path.

                The client does not validate or interpret these — it
                just passes them through to the wire. Servers that
                don't recognise a given field silently ignore it
                (Wave-113 ``json_mode`` uses the same pattern).
            images: Wave W-D13 — optional list of base64-encoded image
                blocks. Each entry is ``{"media_type": "image/jpeg" |
                "image/png" | ..., "data": "<base64>"}``. When
                non-empty, the LAST user message in ``messages`` is
                rewritten from ``{"role": "user", "content": <str>}``
                to OpenAI's vision content-block list shape:

                ::

                    {"role": "user", "content": [
                        {"type": "text", "text": "<original prompt>"},
                        {"type": "image_url", "image_url": {
                            "url": "data:<media_type>;base64,<data>"
                        }},
                        ...
                    ]}

                This shape matches the public OpenAI vision API,
                Together AI's Llama-3.2-Vision endpoints, and Ollama's
                vision-capable models (llava, llama3.2-vision,
                qwen2.5-vl). Servers that don't speak vision will
                surface a 400 from this client — the calling backend
                is expected to short-circuit on
                ``vision_capable=False`` BEFORE reaching the wire.
                Non-user messages are left unchanged. ``None`` /
                empty list keeps the legacy ``content: <str>`` shape
                so existing callers don't regress.

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
        # Wave W-D13: vision content-block translation. When images
        # are supplied, rewrite the LAST user message from a string
        # ``content`` to OpenAI's content-block list shape so the
        # server sees the canonical multimodal payload. Non-user
        # messages and intermediate user messages stay untouched —
        # the caller always attaches images to the most recent
        # prompt. ``None`` / empty list preserves the legacy
        # string-content shape verbatim so non-vision callers don't
        # regress.
        if images:
            messages = self._attach_vision_blocks(messages, images)
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
        # Expose the server-reported token usage of the LAST call so a
        # caller can run a post-call check (e.g. the grounded-answer
        # truncation tripwire compares reported prompt_tokens against a
        # local estimate to detect silent head-truncation). Set
        # unconditionally — an empty dict means the server omitted usage.
        self.last_usage = dict(usage)

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
    # Internals — vision content-block translation (Wave W-D13)
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_vision_blocks(
        messages: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rewrite the LAST user message to carry vision content blocks.

        Builds a fresh messages list (does NOT mutate the caller's list
        in place) so a caller that retains a reference to ``messages``
        after the call still sees the original string-content shape.
        Picks the last user message because that's where the prompt
        attached to the image lives by convention; non-user messages
        and intermediate user messages pass through unchanged.

        Each image dict must carry ``media_type`` and ``data`` (base64
        string). The resulting content block uses OpenAI's
        ``image_url`` shape with a ``data:`` URI — the same format
        Together AI's Llama-3.2-Vision endpoints + Ollama's vision
        models accept.
        """
        # Find the index of the last user message. Iterate in reverse
        # so we land on it without scanning past the front of the list.
        last_user_idx: Optional[int] = None
        for idx in range(len(messages) - 1, -1, -1):
            entry = messages[idx]
            if isinstance(entry, dict) and entry.get("role") == "user":
                last_user_idx = idx
                break
        if last_user_idx is None:
            # No user message to attach to. Vision without a user-side
            # prompt is meaningless — bail loudly so the caller learns
            # they're holding the API wrong rather than silently
            # sending an image with no prompt.
            raise ValueError(
                "vision images supplied but no user message present in "
                "messages; vision attaches to the last user message"
            )

        original = messages[last_user_idx]
        original_text = original.get("content", "")
        if not isinstance(original_text, str):
            # The caller already pre-built a content-block list — leave
            # it alone (caller's contract). Just append our image blocks.
            new_content: List[Dict[str, Any]] = list(original_text)
        else:
            new_content = [{"type": "text", "text": original_text}]
        for img in images:
            media_type = img.get("media_type")
            data = img.get("data")
            if not media_type or not data:
                raise ValueError(
                    "vision image entry missing required "
                    "'media_type' or 'data' field"
                )
            new_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }
            )

        rewritten = list(messages)
        rewritten[last_user_idx] = {
            **{k: v for k, v in original.items() if k != "content"},
            "role": "user",
            "content": new_content,
        }
        return rewritten

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
        # NVIDIA-70b-everywhere SETUP scaffold: shared cloud admission gate
        # (RPM/TPM token buckets + concurrency semaphore). Default-OFF — when
        # ``ED4ALL_CLOUD_RATE_LIMIT`` is unset/falsey OR no ceiling env is
        # configured, ``get_admission_gate`` returns a no-op gate so this is
        # byte-identical to the legacy path (no sleeps, no state). This is the
        # chokepoint for every COMPOSED authoring seat (Together / Local /
        # Curriculum / Courseforge rewrite + outline / textbook synthesis /
        # the NVIDIA large tier), so a pure-nvidia build is fully covered.
        #
        # TODO(nvidia-70b-everywhere RUN phase): the api-mode orchestrator
        # ``MCP/orchestrator/llm_backend.py::AnthropicBackend`` uses a raw
        # ``anthropic.Anthropic`` client that BYPASSES this method, so a hook
        # here does NOT cover api-mode orchestration. A pure-nvidia build
        # routes its authoring through the composed seats above (all covered);
        # harden the AnthropicBackend bypass only if/when api-mode runs cloud.
        from lib.llm.rate_limiter import get_admission_gate

        _admission = get_admission_gate()
        _est_tokens = self._estimate_payload_tokens(payload)
        _reservation = _admission.acquire(_est_tokens)
        try:
            return self._post_with_retry_inner(payload, _reservation)
        finally:
            _reservation.release()

    def _estimate_payload_tokens(self, payload: Dict[str, Any]) -> float:
        """Cheap up-front token estimate for the TPM bucket debit.

        Sums the character length of every message ``content`` (string or
        content-block list) plus the requested ``max_tokens``, divided by a
        ~4-chars/token heuristic. Reconciled later against the server-reported
        ``usage`` so the estimate only needs to be in the right ballpark.
        Returns 0.0 when rate-limiting is off (the gate ignores it anyway).
        """
        chars = 0
        for message in payload.get("messages", []) or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        chars += len(str(block.get("text", "")))
        prompt_tokens = chars / 4.0
        try:
            completion_tokens = float(payload.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            completion_tokens = 0.0
        return prompt_tokens + completion_tokens

    def _post_with_retry_inner(
        self, payload: Dict[str, Any], reservation: Any
    ) -> Tuple[Dict[str, Any], int]:
        """The legacy POST-with-retry body, wrapped by the admission gate."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Wave 113: when json_mode is on, inject BOTH the Ollama-style
        # top-level ``format`` field AND the OpenAI-spec
        # ``response_format`` object. Servers ignore whichever they
        # don't recognize. Caller-supplied values (extra_payload) take
        # precedence so a future provider can opt out per-call.
        if self._json_mode:
            payload.setdefault("format", "json")
            payload.setdefault(
                "response_format", {"type": "json_object"}
            )
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
                # Reconcile the TPM bucket against server-reported usage so the
                # bucket tracks REAL consumption rather than our estimate
                # (no-op when rate-limiting is off). total_tokens preferred;
                # fall back to prompt+completion.
                usage = body.get("usage") if isinstance(body, dict) else None
                if isinstance(usage, dict):
                    actual = usage.get("total_tokens")
                    if actual is None:
                        actual = (usage.get("prompt_tokens", 0) or 0) + (
                            usage.get("completion_tokens", 0) or 0
                        )
                    try:
                        reservation.reconcile(float(actual))
                    except (TypeError, ValueError):
                        pass
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
                self._sleep_for_attempt(attempt, response=response)
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

    def _sleep_for_attempt(
        self, attempt: int, response: Optional["httpx.Response"] = None
    ) -> None:
        """Sleep before the next retry.

        Header-honoring (Stage-0 429 fix): when ``response`` carries a
        ``Retry-After`` header (integer seconds OR an HTTP-date per
        RFC 7231), that value — clamped to
        :data:`_RETRY_AFTER_MAX_SECONDS` — is used in PREFERENCE to the
        computed exponential backoff. ``X-RateLimit-Reset`` is accepted
        as a secondary source when ``Retry-After`` is absent. When the
        server sends no usable header (or no response — the transport-
        error retry path), the client falls back to the legacy
        exponential backoff (doubling per retry) — byte-stable behavior
        for servers that send no ``Retry-After``.

        Routes through ``self._sleep_fn`` so composing providers can
        forward their own ``time.sleep`` reference and keep test
        fixtures patching that module path effective.
        """
        server_wait = self._parse_retry_after(response)
        if server_wait is not None:
            self._sleep_fn(server_wait)
            return
        self._sleep_fn(self._compute_backoff(attempt))

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter, doubling per retry.

        ``full jitter`` (random in ``[0, base]``) de-synchronizes
        concurrent retriers so a fleet of NVIDIA consumers that all 429
        at the same instant don't re-stampede the endpoint in lockstep.
        The mean wait is half the base — still monotonically growing per
        attempt — and the legacy ``initial_backoff * 2**(attempt-1)`` is
        the upper bound, so this never sleeps LONGER than the old fixed
        schedule.
        """
        base = self._initial_backoff_seconds * (2 ** (attempt - 1))
        if base <= 0:
            return 0.0
        return _random.uniform(0.0, base)

    def _parse_retry_after(
        self, response: Optional["httpx.Response"]
    ) -> Optional[float]:
        """Return seconds to wait per the server's rate-limit headers.

        Preference order: ``Retry-After`` (integer seconds, or an
        HTTP-date → seconds-from-now) then ``X-RateLimit-Reset`` (epoch
        seconds → seconds-from-now, or a bare delta in seconds). The
        result is clamped to ``[0, _RETRY_AFTER_MAX_SECONDS]`` so a
        pathological header can't hang the run. Returns ``None`` when no
        usable header is present (the fallback-to-exponential signal),
        which keeps behavior byte-stable for servers that send neither.
        """
        if response is None:
            return None
        try:
            headers = response.headers
        except Exception:
            return None

        raw = headers.get("Retry-After")
        if raw is not None and str(raw).strip():
            seconds = self._retry_after_value_to_seconds(str(raw).strip())
            if seconds is not None:
                return max(0.0, min(seconds, _RETRY_AFTER_MAX_SECONDS))

        reset = headers.get("X-RateLimit-Reset")
        if reset is not None and str(reset).strip():
            seconds = self._rate_limit_reset_to_seconds(str(reset).strip())
            if seconds is not None:
                return max(0.0, min(seconds, _RETRY_AFTER_MAX_SECONDS))

        return None

    @staticmethod
    def _retry_after_value_to_seconds(value: str) -> Optional[float]:
        """Parse a ``Retry-After`` header value into seconds-from-now.

        Per RFC 7231 the value is EITHER a non-negative integer number
        of seconds OR an HTTP-date. Returns ``None`` on an unparseable
        value (the caller then falls back to exponential backoff).
        """
        # Integer / numeric seconds form.
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        # HTTP-date form → seconds from now.
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        try:
            now = time.time()
            delta = when.timestamp() - now
        except (OverflowError, OSError, ValueError):
            return None
        # A date in the past means "retry now".
        return max(0.0, delta)

    @staticmethod
    def _rate_limit_reset_to_seconds(value: str) -> Optional[float]:
        """Parse ``X-RateLimit-Reset`` into seconds-from-now.

        Trivially supported as a fallback: the value is either an
        absolute epoch timestamp (heuristically, anything close to or
        beyond ``now``) or a bare delta in seconds. Returns ``None`` on
        an unparseable value.
        """
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed) or parsed < 0:
            return None
        now = time.time()
        # Treat a value within the current epoch range as an absolute
        # reset time; otherwise it's a bare delta-seconds value.
        if parsed > now - _RETRY_AFTER_MAX_SECONDS:
            return max(0.0, parsed - now)
        return parsed

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

        H1: when the first choice's ``finish_reason`` is ``"length"``
        the model hit the ``max_tokens`` cap mid-output — for a
        JSON-mode caller the returned ``content`` is then truncated and
        any downstream ``json.loads`` fails with a generic, misleading
        parse error. We surface the truncation distinctly as
        ``SynthesisProviderError(code="output_truncated")`` so the
        caller can branch on it (e.g. raise the cap) rather than blame
        the model's JSON discipline.
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
        finish_reason = first.get("finish_reason")
        if finish_reason == "length":
            raise SynthesisProviderError(
                "response finish_reason='length' — model hit the "
                "max_tokens cap and the output is truncated; raise "
                "max_tokens for this call",
                code="output_truncated",
            )
        message = first.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _extract_json_lenient(text: str) -> Optional[Dict[str, Any]]:
        """Lenient JSON extraction for 7B-class instruction-model drift.

        7B-class instruction models in 4-bit quantization (e.g.
        ``qwen2.5:7b-instruct-q4_K_M`` on Ollama) frequently wrap
        their JSON output in markdown code fences or surround it
        with leading / trailing prose despite explicit "JSON only"
        directives. This helper recovers the JSON object across the
        three common drift patterns; returns ``None`` only when the
        response is genuinely unrecoverable (e.g. "I cannot help with
        that.").

        Strategy, in order:
        1. Direct ``json.loads(text)`` — the happy path.
        2. Strip enclosing markdown fences (``` ```json ... ``` ``` or
           ``` ``` ... ``` ```), then retry ``json.loads`` on the
           stripped body.
        3. Find the first balanced ``{...}`` substring and ``json.loads``
           it. Handles ``"Sure! {...} hope this helps"``-style drift.

        Caller decides retry strategy on ``None``. Does not enforce
        required keys — that's the caller's contract.
        """
        if not text or not text.strip():
            return None
        s = text.strip()
        # Strategy 1: direct parse.
        try:
            parsed = _json.loads(s)
        except _json.JSONDecodeError:
            pass
        else:
            return parsed if isinstance(parsed, dict) else None

        # Strategy 2: strip markdown code fences. The fence regex tolerates
        # the language hint (```json) and arbitrary leading / trailing
        # whitespace inside the fence.
        fence = _re.match(
            r"^```(?:[a-zA-Z0-9_+-]*\n)?\s*(.*?)\s*```\s*$", s, _re.DOTALL
        )
        if fence:
            inner = fence.group(1).strip()
            try:
                parsed = _json.loads(inner)
            except _json.JSONDecodeError:
                pass
            else:
                return parsed if isinstance(parsed, dict) else None

        # Strategy 3: scan for the first balanced JSON object span.
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        parsed = _json.loads(candidate)
                    except _json.JSONDecodeError:
                        return None
                    return parsed if isinstance(parsed, dict) else None
        return None

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
    "ENV_REQUEST_TIMEOUT",
    "resolve_default_timeout",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_STATUS_CODES",
    "DEFAULT_INITIAL_BACKOFF_SECONDS",
    "DEFAULT_PROVIDER_LABEL",
    "SynthesisProviderError",
]
