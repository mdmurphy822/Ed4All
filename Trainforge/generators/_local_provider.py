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


# Wave 114: local-class kind bounds. The 40-char prompt floor inherited
# from ``_anthropic_provider.py::_KIND_BOUNDS`` is too strict for 7B-Q4
# paraphrases that legitimately compress (e.g. "Explain in your own
# words what owl:sameAs means and why it matters." -> "What does
# owl:sameAs mean and why?"). 25 chars still rejects degenerate
# 3-word stub outputs without rejecting valid compressed paraphrases.
# Completion floor stays at 50 — a 30-char training-target completion
# has legitimate quality concerns the prompt floor does not.
# Wave 114: terse system prompts for the local path. 7B-Q4 instruction
# models attend less reliably to long behavioral preambles; the
# trailing JSON directive in the user message is the most-respected
# part of the prompt. Keeping these <50 words frees attention for the
# task itself. Anthropic / Together providers retain their original
# verbose prompts (Sonnet-class models use the extra context fine).
_LOCAL_INSTRUCTION_SYSTEM_PROMPT = (
    "You paraphrase training pairs from a deterministic template. "
    "Rewrite the prompt and completion using different wording but "
    "the same meaning. Do not add facts not in the chunk text. "
    "Preserve the Bloom cognitive level. Output JSON only: "
    '{"prompt": "...", "completion": "..."}.'
)

_LOCAL_PREFERENCE_SYSTEM_PROMPT = (
    "You paraphrase preference triples for DPO training. Rewrite "
    "prompt, chosen, and rejected using different wording but the "
    "same meaning. The chosen completion stays factually correct; "
    "the rejected stays plausibly wrong. Do not add facts not in "
    "the chunk text. Output JSON only: "
    '{"prompt": "...", "chosen": "...", "rejected": "..."}.'
)


DEFAULT_LOCAL_KIND_BOUNDS: Dict[str, tuple] = {
    "prompt": (25, PROMPT_MAX),
    "completion": (COMPLETION_MIN, COMPLETION_MAX),
    "chosen": (COMPLETION_MIN, COMPLETION_MAX),
    "rejected": (COMPLETION_MIN, COMPLETION_MAX),
}


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
        kind_bounds: Optional[Dict[str, tuple]] = None,
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
        self._kind_bounds: Dict[str, tuple] = (
            dict(kind_bounds) if kind_bounds else dict(DEFAULT_LOCAL_KIND_BOUNDS)
        )

        # Composition: build the LLM-agnostic client. Same client class
        # the Together provider composes — only the configuration
        # differs.
        #
        # Wave 113 hardening: ``json_mode=True`` makes every request
        # carry both the Ollama-style ``format: "json"`` field AND the
        # OpenAI-spec ``response_format: {"type": "json_object"}`` field.
        # 7B-class instruction models in 4-bit quantization (e.g.
        # ``qwen2.5:7b-instruct-q4_K_M`` on Ollama) are unreliable at
        # strict-JSON output without grammar-constrained decoding;
        # ``format: "json"`` triggers Ollama's JSON-grammar mode and
        # eliminates the natural-language drift that crashed the Wave
        # 113 Task 10 pilot run after 3 parse retries.
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
            json_mode=True,
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
            system_prompt=_LOCAL_INSTRUCTION_SYSTEM_PROMPT,
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
            system_prompt=_LOCAL_PREFERENCE_SYSTEM_PROMPT,
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
        """Call the local server and parse JSON via the lenient extractor.

        Wave 113 hardening: 7B-class instruction models in 4-bit
        quantization wrap their JSON in markdown code fences or
        surround it with prose despite explicit "JSON only" prompt
        directives. The embedded client's
        :meth:`OpenAICompatibleClient._extract_json_lenient` recovers
        the JSON across the three common drift patterns before parse
        failure escalates.

        Wave 114 hardening: a parsed response whose required-key
        values fall below the per-kind length floor now triggers a
        remediation retry (length-retry), parallel to JSON-parse
        retry. The model receives a corrective user message stating
        the observed-vs-required length and rewrites its own prior
        output. Preserves Wave 112's no-sentinel-injection invariant
        — the retry asks the model to expand, never injects filler.

        After ``MAX_PARSE_RETRIES`` exhaustion (across either failure
        class), raises ``SynthesisProviderError`` with code
        ``paraphrase_invalid_after_retry`` and a truncated 500-char
        tail of the last response — postmortem visibility on what the
        model actually emitted.
        """
        messages = self._build_messages(system_prompt, chunk_text, user_prompt)
        attempts = 0
        last_err: Optional[str] = None
        last_text: str = ""
        total_http_retries = 0
        last_usage: Dict[str, int] = {}
        while attempts < MAX_PARSE_RETRIES:
            attempts += 1
            text, usage, http_retries = self._chat_completion_raw(messages)
            total_http_retries += http_retries
            last_usage = usage
            last_text = text
            parsed = self._oa_client._extract_json_lenient(text)
            if parsed is None:
                last_err = "lenient JSON extraction returned None"
                logger.warning(
                    "%s synthesis: lenient parse retry %d/%d: "
                    "no JSON object recoverable from response tail %r",
                    self._provider_name, attempts, MAX_PARSE_RETRIES,
                    text[-120:],
                )
                continue
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                last_err = f"response missing required keys: {missing}"
                logger.warning(
                    "%s synthesis: lenient parse retry %d/%d: %s",
                    self._provider_name, attempts, MAX_PARSE_RETRIES,
                    last_err,
                )
                continue
            short = self._first_short_field(parsed, required_keys)
            if short is not None:
                field, length, floor = short
                last_err = f"{field} length {length} below minimum {floor}"
                logger.warning(
                    "%s synthesis: length-retry %d/%d: %s",
                    self._provider_name, attempts, MAX_PARSE_RETRIES,
                    last_err,
                )
                messages = self._append_length_remediation(
                    messages, field, length, floor,
                )
                continue
            return parsed, last_usage, total_http_retries
        raise SynthesisProviderError(
            f"{type(self).__name__}: failed to obtain a valid paraphrase "
            f"after {MAX_PARSE_RETRIES} attempts. Last error: {last_err}; "
            f"tail of last response: {last_text[-500:]!r}",
            code="paraphrase_invalid_after_retry",
        )

    def _first_short_field(
        self, parsed: Dict[str, Any], required_keys: tuple
    ) -> Optional[Tuple[str, int, int]]:
        """Return ``(field_name, length, floor)`` for the first required
        key whose stripped value is shorter than its kind floor. None
        when all required fields meet the floor.

        Mirrors the kind-mapping ``_clamp`` enforces, so the retry loop
        pre-checks lengths before commit. Keys without a registered
        floor in ``self._kind_bounds`` are skipped (the call site is
        responsible for using kind names that are bound; mismatches
        surface as a dropped check rather than a hidden failure).
        """
        for key in required_keys:
            value = str(parsed.get(key, "") or "").strip()
            try:
                lo, _ = self._kind_bounds[key]
            except KeyError:
                continue
            if len(value) < lo:
                return (key, len(value), lo)
        return None

    @staticmethod
    def _append_length_remediation(
        messages: List[Dict[str, str]],
        field: str,
        length: int,
        floor: int,
    ) -> List[Dict[str, str]]:
        """Return a new message list with a corrective user turn appended.

        The remediation message is short and concrete: states the
        observed-vs-required length and asks for a specific edit.
        Avoids any sentinel filler — the model rewrites its own prior
        output, preserving Wave 112's no-injection invariant.
        """
        remediation = (
            f"The prior response had {field}={length} chars but the "
            f"minimum is {floor}. Rewrite that field to be at least "
            f"{floor} chars while preserving the same meaning. "
            f"Output the same JSON object shape, JSON only."
        )
        return list(messages) + [{"role": "user", "content": remediation}]

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

    # ------------------------------------------------------------------
    # Strict-JSON directives. Appended verbatim at the END of the user
    # message (NOT the system message — keeping the system message
    # unchanged preserves provider parity with Together / Anthropic).
    # 7B-class models in 4-bit quantization respect end-of-prompt
    # directives more reliably than buried-in-system-prompt directives.
    # ------------------------------------------------------------------
    _INSTRUCTION_JSON_DIRECTIVE = (
        "\n\nRESPOND ONLY WITH A JSON OBJECT. Use EXACTLY this shape, "
        "nothing else:\n"
        "{\"prompt\": \"<paraphrased prompt>\", "
        "\"completion\": \"<paraphrased completion>\"}\n"
        "Do not wrap in markdown. Do not add commentary. Output the "
        "JSON object only."
    )

    _PREFERENCE_JSON_DIRECTIVE = (
        "\n\nRESPOND ONLY WITH A JSON OBJECT. Use EXACTLY this shape, "
        "nothing else:\n"
        "{\"prompt\": \"<paraphrased prompt>\", "
        "\"chosen\": \"<paraphrased chosen>\", "
        "\"rejected\": \"<paraphrased rejected>\"}\n"
        "Do not wrap in markdown. Do not add commentary. Output the "
        "JSON object only."
    )

    @classmethod
    def _render_instruction_user(
        cls, draft: Dict[str, Any], chunk_id: str
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
            f"{cls._INSTRUCTION_JSON_DIRECTIVE}"
        )

    @classmethod
    def _render_preference_user(
        cls, draft: Dict[str, Any], chunk_id: str
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
            f"{cls._PREFERENCE_JSON_DIRECTIVE}"
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

    def _clamp(self, text: str, kind: str, *, chunk_id: Optional[str] = None) -> str:
        try:
            lo, hi = self._kind_bounds[kind]
        except KeyError as exc:
            raise ValueError(
                f"_clamp: unknown kind={kind!r}; expected one of "
                f"{sorted(self._kind_bounds)}"
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
