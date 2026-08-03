#!/usr/bin/env python3
"""Local OpenAI-compatible synthesis provider.

Third synthesis path alongside ``anthropic`` / ``claude_session`` /
``together``. Speaks the same OpenAI-compatible chat-completions wire
shape Together AI uses, so the entire HTTP loop + JSON parse +
decision-capture-rationale machinery is provided by composing one
:class:`Trainforge.generators.providers._openai_compatible_client.OpenAICompatibleClient`
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
- Default model is ``qwen2.5:7b-instruct-q4_K_M`` — a sensible
  out-of-box pick that fits an 8 GB GPU on Ollama.
  ``LOCAL_SYNTHESIS_MODEL`` overrides per server.
- ``out["provider"]`` is set to ``"local"`` so downstream consumers
  (pair schemas, the mock-corpus gate, the eval harness) can
  distinguish local-server output from Together-hosted output.

Tradeoff vs Together: latency is 5-30s per call (depends on local
hardware + model size), but cost-per-call is zero after the hardware
investment and there's no ToS exposure — fully offline / air-gapped
synthesis is supported. Retry policy is identical (5xx / 429 with
exponential backoff up to ``MAX_HTTP_RETRIES`` attempts); 5xx is
slightly more likely on a local server, 429 is essentially never.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from Trainforge.generators.providers._synthesis_common import (  # noqa: F401
    COMPLETION_MAX,
    COMPLETION_MIN,
    PROMPT_MAX,
    PROMPT_MIN,
    SynthesisProviderError,
    _KIND_BOUNDS,
)
from Trainforge.generators.providers._openai_compatible_client import (
    OpenAICompatibleClient,
)
from Trainforge.generators.providers._together_provider import (
    INITIAL_BACKOFF_SECONDS,
    MAX_HTTP_RETRIES,
    MAX_PARSE_RETRIES,
    _RETRYABLE_STATUS,
)

logger = logging.getLogger(__name__)


# Defaults — kept as module-level constants so callers (and tests) can
# import them without instantiating the provider.
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_SYNTHESIS_MODEL = "nemotron-3-nano-30b-a3b"
ENV_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"
ENV_MODEL = "LOCAL_SYNTHESIS_MODEL"
ENV_API_KEY = "LOCAL_SYNTHESIS_API_KEY"
DEFAULT_TIMEOUT = 60.0


# Soft-floor at 0.0: the force-injection path in ``instruction_factory``
# / ``preference_factory`` is the canonical-anchor authority, so the LLM
# only owes natural-language variety and the deterministic injector
# supplies CURIE anchoring downstream. A strict per-call floor here is
# the wrong lever — a chunk carrying many preserve_tokens fails it most
# of the time on quantized 7B/14B models, discarding good paraphrases
# for anchoring the injector would have restored anyway. Operators who
# do want the strict gate pass ``min_preserve_rate=1.0``.
DEFAULT_MIN_PRESERVE_RATE = 0.0


# The local prompt floor MUST stay at 40 to match
# ``schemas/knowledge/instruction_pair.schema.json``'s
# ``prompt.minLength: 40``. A lower provider-side floor (tempting, since
# quantized 7B models compress paraphrases) admits pairs the schema then
# rejects at audit time — the misalignment surfaces late, after the run.
# Callers who genuinely need a lower floor pass an explicit
# ``kind_bounds={"prompt": (25, PROMPT_MAX), ...}``. Completion floor
# stays at 50: a 30-char training target has quality problems a short
# prompt does not.
#
# The local system prompts are deliberately terse (<50 words). Quantized
# instruction models attend less reliably to long behavioral preambles,
# and the trailing JSON directive in the user message is the
# most-respected part of the prompt. Anthropic / Together keep their
# verbose prompts — larger models use the extra context fine.
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
    "prompt": (PROMPT_MIN, PROMPT_MAX),
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
        max_parse_retries: Optional[int] = None,
        min_preserve_rate: Optional[float] = None,
        response_dialect: Optional[str] = None,
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
        # Smoke-mode callers pass a low cap (typically 1) so the
        # parse-retry budget bounds smoke wall time. Default None ->
        # the module constant, for production runs that prioritise
        # paraphrase quality over speed.
        self._max_parse_retries = (
            int(max_parse_retries)
            if max_parse_retries is not None
            else MAX_PARSE_RETRIES
        )
        # Soft preservation contract — fraction of ``preserve_tokens``
        # that must appear verbatim per call. Default 0.0 (see
        # DEFAULT_MIN_PRESERVE_RATE); 1.0 is the strict per-call gate.
        self._min_preserve_rate = (
            float(min_preserve_rate)
            if min_preserve_rate is not None
            else DEFAULT_MIN_PRESERVE_RATE
        )
        if not 0.0 <= self._min_preserve_rate <= 1.0:
            raise ValueError(
                f"min_preserve_rate must be in [0.0, 1.0], "
                f"got {self._min_preserve_rate}"
            )

        # Composition: build the LLM-agnostic client. Same client class
        # the Together provider composes — only the configuration
        # differs.
        #
        # ``json_mode=True`` makes every request carry BOTH the
        # Ollama-style ``format: "json"`` field and the OpenAI-spec
        # ``response_format: {"type": "json_object"}`` field — different
        # local servers honour different ones. Quantized 7B-class
        # instruction models are unreliable at strict-JSON output
        # without grammar-constrained decoding; ``format: "json"``
        # triggers Ollama's JSON-grammar mode, which is what keeps a run
        # from exhausting its parse retries on natural-language drift.
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
            # that patch ``Trainforge.generators.providers._together_provider.time.sleep``
            # (per the local test docstring's stated contract) keep
            # working post-refactor.
            sleep_fn=_local_sleep,
            json_mode=True,
            response_dialect=response_dialect,
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
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Paraphrase a draft instruction pair against its chunk.

        ``preserve_tokens`` lists technical CURIEs / surface forms that
        MUST appear verbatim in the model's output (e.g.
        ``["sh:NodeShape", "rdfs:subClassOf"]``). The directive is
        injected into the user prompt and verified after parse — a
        response falling below ``min_preserve_rate`` triggers a
        remediation retry. Exhaustion raises
        ``surface_form_preservation_failed`` so the caller can fall back
        to the deterministic draft.
        """
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        preserve = list(preserve_tokens or [])
        user_prompt = self._render_instruction_user(draft, chunk_id, preserve)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=_LOCAL_INSTRUCTION_SYSTEM_PROMPT,
            chunk_text=chunk_text,
            user_prompt=user_prompt,
            required_keys=("prompt", "completion"),
            preserve_tokens=preserve,
            preserve_in_keys=("prompt", "completion"),
        )

        out = dict(draft)
        out["prompt"] = self._clamp(
            parsed["prompt"], kind="prompt", chunk_id=chunk_id
        )
        out["completion"] = self._clamp(
            parsed["completion"], kind="completion", chunk_id=chunk_id
        )
        out["provider"] = self._provider_name

        # Thread structured-claim arrays through emit when the LLM
        # produced them. Optional — absent on the default
        # (prompt, completion) shape.
        for k in ("key_claims", "per_claim_support"):
            if k in parsed:
                out[k] = parsed[k]

        self._emit_decision(
            kind="instruction",
            draft=draft,
            chunk_id=chunk_id,
            bloom_level=str(chunk.get("bloom_level") or "unknown"),
            concept_tags=list((chunk.get("concept_tags") or [])[:3]),
            usage=usage,
            retry_count=retry_count,
            parsed=parsed,
        )
        return out

    def paraphrase_preference(
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Paraphrase a draft preference triple against its chunk.

        ``preserve_tokens`` CURIEs are checked against the ``chosen``
        field (the factually-correct completion) only. ``rejected`` is
        deliberately NOT checked — the rule-synthesized rejection may
        legitimately omit the technical token.
        """
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        preserve = list(preserve_tokens or [])
        user_prompt = self._render_preference_user(draft, chunk_id, preserve)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=_LOCAL_PREFERENCE_SYSTEM_PROMPT,
            chunk_text=chunk_text,
            user_prompt=user_prompt,
            required_keys=("prompt", "chosen", "rejected"),
            preserve_tokens=preserve,
            preserve_in_keys=("chosen",),
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

        # Thread structured-claim arrays through emit when the LLM
        # produced them.
        for k in ("key_claims", "per_claim_support"):
            if k in parsed:
                out[k] = parsed[k]

        self._emit_decision(
            kind="preference",
            draft=draft,
            chunk_id=chunk_id,
            bloom_level=str(chunk.get("bloom_level") or "unknown"),
            concept_tags=list((chunk.get("concept_tags") or [])[:3]),
            usage=usage,
            retry_count=retry_count,
            parsed=parsed,
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
        preserve_tokens: Optional[List[str]] = None,
        preserve_in_keys: tuple = (),
    ) -> Tuple[Dict[str, Any], Dict[str, int], int]:
        """Call the local server and parse JSON via the lenient extractor.

        Quantized instruction models wrap their JSON in markdown code
        fences or surround it with prose despite explicit "JSON only"
        directives, so parsing goes through the embedded client's
        :meth:`OpenAICompatibleClient._extract_json_lenient` rather than
        a bare ``json.loads``.

        A parsed response whose required-key values fall below the
        per-kind length floor triggers a remediation retry, parallel to
        the JSON-parse retry: the model receives a corrective user
        message stating observed-vs-required length and rewrites its own
        prior output. This preserves the no-sentinel-injection invariant
        — the retry asks the model to expand, and NEVER pads the field
        with filler, which would poison the pair.

        After ``MAX_PARSE_RETRIES`` exhaustion (across either failure
        class), raises ``SynthesisProviderError`` with code
        ``paraphrase_invalid_after_retry`` and a truncated 500-char
        tail of the last response, for postmortem visibility into what
        the model actually emitted.
        """
        messages = self._build_messages(system_prompt, chunk_text, user_prompt)
        attempts = 0
        last_err: Optional[str] = None
        last_text: str = ""
        total_http_retries = 0
        last_usage: Dict[str, int] = {}
        retry_budget = self._max_parse_retries
        while attempts < retry_budget:
            attempts += 1
            try:
                text, usage, http_retries = self._chat_completion_raw(messages)
            except SynthesisProviderError as exc:
                if not getattr(exc, "_direct_capture_emitted", False):
                    self._capture_failed_raw_call(
                        required_keys, attempts,
                        str(getattr(exc, "code", "") or type(exc).__name__),
                        {},
                    )
                raise
            total_http_retries += http_retries
            last_usage = usage
            last_text = text
            parsed = self._oa_client._extract_json_lenient(text)
            if parsed is None:
                last_err = "lenient JSON extraction returned None"
                self._capture_failed_raw_call(
                    required_keys, attempts, last_err, usage,
                )
                logger.warning(
                    "%s synthesis: lenient parse retry %d/%d: "
                    "no JSON object recoverable from response tail %r",
                    self._provider_name, attempts, retry_budget,
                    text[-120:],
                )
                continue
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                last_err = f"response missing required keys: {missing}"
                self._capture_failed_raw_call(
                    required_keys, attempts, last_err, usage,
                )
                logger.warning(
                    "%s synthesis: lenient parse retry %d/%d: %s",
                    self._provider_name, attempts, retry_budget,
                    last_err,
                )
                continue
            short = self._first_short_field(parsed, required_keys)
            if short is not None:
                field, length, floor = short
                last_err = f"{field} length {length} below minimum {floor}"
                self._capture_failed_raw_call(
                    required_keys, attempts, last_err, usage,
                )
                logger.warning(
                    "%s synthesis: length-retry %d/%d: %s",
                    self._provider_name, attempts, retry_budget,
                    last_err,
                )
                messages = self._append_length_remediation(
                    messages, field, length, floor,
                )
                continue
            # Surface-form preservation gate. After length passes,
            # verify each preserve_token appears verbatim in the
            # nominated fields (instruction: prompt + completion;
            # preference: chosen only). Local models silently rewrite
            # ``sh:NodeShape`` -> "node shape", which reads fine but
            # destroys the CURIE anchoring the pair exists to teach.
            missing_tokens = self._missing_preserve_tokens(
                parsed, preserve_tokens or [], preserve_in_keys,
            )
            # Soft preservation contract: compute the actual
            # preservation rate against the configured floor and retry
            # only when the rate falls BELOW it. The remediation hint
            # still names the specific missing tokens so a retry has the
            # best chance of recovering them, but a result at or above
            # the floor is accepted without burning another call.
            preserve_token_count = len(preserve_tokens or [])
            if preserve_token_count > 0 and missing_tokens:
                preserved_rate = (
                    preserve_token_count - len(missing_tokens)
                ) / preserve_token_count
                # When _min_preserve_rate=0.0 (default), this branch is
                # unreachable — force-injection handles canonical anchoring
                # downstream.
                if preserved_rate < self._min_preserve_rate:
                    last_err = (
                        f"surface forms missing from {list(preserve_in_keys)}: "
                        f"{missing_tokens} "
                        f"(preserved {preserved_rate:.2f} < floor "
                        f"{self._min_preserve_rate:.2f})"
                    )
                    self._capture_failed_raw_call(
                        required_keys, attempts, last_err, usage,
                    )
                    logger.warning(
                        "%s synthesis: preserve-retry %d/%d: %s",
                        self._provider_name, attempts, retry_budget,
                        last_err,
                    )
                    messages = self._append_preserve_remediation(
                        messages, missing_tokens, preserve_in_keys,
                    )
                    continue
                # rate >= floor: accept with a debug log so postmortem
                # audits can correlate per-pair partial-preservation
                # against retention metrics.
                logger.debug(
                    "%s synthesis: soft-accept (rate %.2f >= floor %.2f); "
                    "tokens dropped: %s",
                    self._provider_name, preserved_rate,
                    self._min_preserve_rate, missing_tokens,
                )
            return parsed, last_usage, total_http_retries
        # Distinguish preservation failure with its own error code so
        # the caller can fall back to the deterministic draft instead of
        # dropping the pair entirely.
        if preserve_tokens and last_err and "surface forms missing" in last_err:
            raise SynthesisProviderError(
                f"{type(self).__name__}: paraphrase dropped required surface "
                f"forms after {retry_budget} attempts. {last_err}; "
                f"tail of last response: {last_text[-500:]!r}",
                code="surface_form_preservation_failed",
            )
        raise SynthesisProviderError(
            f"{type(self).__name__}: failed to obtain a valid paraphrase "
            f"after {retry_budget} attempts. Last error: {last_err}; "
            f"tail of last response: {last_text[-500:]!r}",
            code="paraphrase_invalid_after_retry",
        )

    def _capture_failed_raw_call(
        self, required_keys: tuple, attempt: int, error: str,
        usage: Dict[str, int],
    ) -> None:
        if self._capture is None:
            return
        kind = "preference" if "chosen" in required_keys else "instruction"
        self._capture.log_decision(
            decision_type="synthesis_provider_call",
            decision=f"Rejected {kind} raw synthesis attempt {attempt}.",
            rationale=(
                f"model={self._model}, stage={kind}_raw, attempt={attempt}, "
                f"max_tokens={self._max_tokens}, error={error}, "
                f"prompt_tokens={int(usage.get('prompt_tokens', 0))}, "
                f"completion_tokens={int(usage.get('completion_tokens', 0))}; "
                "the response was not accepted into a training pair."
            ),
        )

    @staticmethod
    def _missing_preserve_tokens(
        parsed: Dict[str, Any], tokens: List[str], in_keys: tuple
    ) -> List[str]:
        """Return tokens that don't appear verbatim in any of ``in_keys``.

        A token is considered preserved if it appears in at least one of
        the listed fields (e.g. instruction pairs check both prompt and
        completion; preference pairs check only chosen)."""
        if not tokens or not in_keys:
            return []
        haystacks = [str(parsed.get(k, "") or "") for k in in_keys]
        missing = []
        for tok in tokens:
            if not any(tok in h for h in haystacks):
                missing.append(tok)
        return missing

    @staticmethod
    def _append_preserve_remediation(
        messages: List[Dict[str, str]],
        missing: List[str],
        in_keys: tuple,
    ) -> List[Dict[str, str]]:
        """Corrective user turn naming the dropped tokens. The model
        rewrites its prior output preserving these literal CURIEs."""
        token_list = ", ".join(repr(t) for t in missing)
        field_list = " and ".join(in_keys) if in_keys else "the response"
        remediation = (
            f"The prior response did not include the required tokens "
            f"{token_list} in {field_list}. Rewrite the response so each "
            f"of those tokens appears VERBATIM (exactly as written, with "
            f"the colon and case intact) in {field_list}. Output the "
            f"same JSON object shape, JSON only."
        )
        return list(messages) + [{"role": "user", "content": remediation}]

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
        output, preserving the no-filler-injection invariant.
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
        # Local training-pair generation is a constrained structured rewrite,
        # not a reasoning/judgment pass. Keep hidden reasoning from consuming
        # the fixed response budget even on the legacy rollback provider.
        from Trainforge.generators.providers._openai_compatible_client import (
            apply_reasoning_thinking_off_payload,
        )

        apply_reasoning_thinking_off_payload(payload, force_thinking_off=True)
        body, retry_count = self._oa_client.post_with_usage(
            payload, task="local_synthesis_raw",
        )
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

    @staticmethod
    def _preserve_directive(tokens: List[str], where: str) -> str:
        """Render a 'PRESERVE THESE TOKENS VERBATIM' directive. Empty when
        no tokens. ``where`` describes the target fields (e.g. 'the
        prompt and completion')."""
        if not tokens:
            return ""
        token_list = ", ".join(repr(t) for t in tokens)
        return (
            f"\n\nPRESERVE THESE TOKENS VERBATIM in {where} (exact "
            f"spelling, colon, and case): {token_list}. These are "
            f"technical CURIEs the learner must see literally — do not "
            f"rewrite them as natural language."
        )

    @classmethod
    def _render_instruction_user(
        cls, draft: Dict[str, Any], chunk_id: str,
        preserve_tokens: Optional[List[str]] = None,
    ) -> str:
        from Trainforge.generators.providers._base_synthesis_provider import (
            EVIDENCE_QUOTE_PROMPT_DIRECTIVE,
        )
        preserve = cls._preserve_directive(
            preserve_tokens or [], "the prompt or completion",
        )
        # Definition-style chunks (content_type=definition or
        # Bloom='remember') elicit short "Define X." prompts that fall
        # under the schema's 40-char floor. Inject an explicit
        # explanation-asking directive so the FIRST attempt produces a
        # long-form question rather than relying on the retry path.
        content_type = str(draft.get("content_type", "")).lower()
        bloom_level = str(draft.get("bloom_level", "")).lower()
        is_definition_kind = (
            "definition" in content_type
            or "glossary" in content_type
            or "key_term" in content_type
            or bloom_level == "remember"
        )
        definition_directive = (
            "\n\nThis is a definition / recall chunk. The prompt MUST be an "
            "EXPLANATION-asking question of at least 40 characters — for "
            "example, 'Explain what an IRI is in RDF and describe its role "
            "in identifying resources globally.' Do NOT emit a bare 'Define "
            "X.' or 'What is X?' prompt — those are too short."
            if is_definition_kind
            else ""
        )
        grounding_directive = (
            "\n\nSOURCE GROUNDING: Treat the draft as a format hint, not a "
            "source of facts. Replace generic draft assertions with one "
            "concrete relationship, rule, or procedure supported by "
            "the supplied source chunk. Reuse necessary technical "
            "nouns, but change syntax and word order. No prompt or "
            "completion may copy 50 consecutive source characters. "
            "The completion must contain only one "
            "to three short source-supported sentences. Do not add transfer "
            "claims, learning advice, examples, quantities, or background "
            "facts absent from the source."
        )
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
            f"{preserve}"
            f"{definition_directive}"
            f"{grounding_directive}"
            f"{EVIDENCE_QUOTE_PROMPT_DIRECTIVE}"
            f"{cls._INSTRUCTION_JSON_DIRECTIVE}"
        )

    @classmethod
    def _render_preference_user(
        cls, draft: Dict[str, Any], chunk_id: str,
        preserve_tokens: Optional[List[str]] = None,
    ) -> str:
        from Trainforge.generators.providers._base_synthesis_provider import (
            EVIDENCE_QUOTE_PROMPT_DIRECTIVE,
        )
        preserve = cls._preserve_directive(
            preserve_tokens or [], "the chosen completion",
        )
        grounding_directive = (
            "\n\nSOURCE GROUNDING: Treat the drafts as format hints, not "
            "sources of facts. Rewrite the chosen answer around one concrete "
            "relationship, rule, or procedure supported by the supplied "
            "source chunk, reusing necessary technical nouns while changing "
            "syntax and word order. No prompt or chosen answer may copy 50 "
            "consecutive source characters. The chosen answer must be exactly ONE sentence of no more "
            "than 25 words, closely restating one source fact without an "
            "example, semicolon, learning advice, or transfer claim. Rewrite "
            "the rejected answer as exactly one plausible, source-relevant "
            "misconception in substantially different wording; fewer than "
            "half its content words should overlap the chosen answer. Do not "
            "form it by merely inserting 'not' or swapping one word."
        )
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
            f"{preserve}"
            f"{grounding_directive}"
            f"{EVIDENCE_QUOTE_PROMPT_DIRECTIVE}"
            f"{cls._PREFERENCE_JSON_DIRECTIVE}"
        )

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
        bloom_level: str,
        concept_tags: List[str],
        usage: Dict[str, int],
        retry_count: int,
        parsed: Optional[Dict[str, Any]] = None,
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
                    bloom_level=bloom_level,
                    concept_tags=concept_tags,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                    parsed=parsed,
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
        bloom_level: str,
        concept_tags: List[str],
        prompt_tokens: int,
        completion_tokens: int,
        retry_count: int,
        parsed: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Interpolate per-chunk pedagogical signals (bloom_level,
        # concept_tags) so the rationale varies materially across calls:
        # the decision-capture validator scores formulaic rationales as
        # 'developing'. None of this changes what is sent to the model.
        # Also appends the per-claim evidence_quote emit rate.
        from Trainforge.generators.providers._base_synthesis_provider import (
            render_evidence_quote_rationale_fragment,
        )
        tags_repr = ",".join(concept_tags) if concept_tags else "<none>"
        evidence_fragment = render_evidence_quote_rationale_fragment(parsed)
        return (
            f"Routing template-generated {kind} draft "
            f"(template_id={draft.get('template_id','n/a')}, "
            f"bloom_level={bloom_level}, "
            f"concept_tags=[{tags_repr}], "
            f"draft_prompt_len={len(str(draft.get('prompt','')))}, "
            f"chunk_id={chunk_id}) through a local OpenAI-compatible "
            f"model server at base_url={self._base_url} using model "
            f"{self._model} for paraphrase. Local synthesis has zero "
            f"per-call cost after hardware setup and zero ToS exposure "
            f"(fully offline / air-gapped); the tradeoff is local "
            f"hardware capability and 5-30s per-call latency. "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"http_retries={retry_count}. "
            f"{evidence_fragment}."
        )


def _local_sleep(seconds: float) -> None:
    """Forward retry-backoff sleeps through the together-provider module.

    The local-provider tests patch
    ``Trainforge.generators.providers._together_provider.time.sleep`` to keep
    test runs fast (per the test docstring). We honor that contract by
    routing the embedded client's backoff sleeps through that module's
    ``time.sleep`` reference rather than the local module's, so a
    single patch covers both providers' retry paths.
    """
    from Trainforge.generators.providers import _together_provider as _tg

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
