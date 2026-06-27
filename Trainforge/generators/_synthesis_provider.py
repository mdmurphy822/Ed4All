#!/usr/bin/env python3
"""LLM-agnostic synthesis provider (Phase 2).

ONE registry-driven synthesis provider that supersedes the per-vendor
leaves (``LocalSynthesisProvider`` / ``TogetherSynthesisProvider`` /
the OpenAI-wire branch of the NVIDIA / Groq / Fireworks family). Any
OpenAI-wire backend is a ``config/endpoints.yaml`` ROW resolved BY NAME
via :func:`lib.llm.endpoints.build_openai_compatible_client` — NOT a
per-vendor subclass. This matches the project's standing "provider
references stay dynamic — new providers are registry entries, not
subclasses" principle and mirrors the existing registry-driven
templates :class:`Trainforge.generators._assessment_provider.AssessmentGeneratorProvider`
and :func:`lib.objectives.objective_review._build_review_client`.

Adding a future provider (a hosted Mistral row, a DGX Spark vLLM box,
…) is a one-row change to ``config/endpoints.yaml`` — zero new code,
zero new provider classes.

Design:

- The constructor resolves its HTTP client from the unified endpoint
  registry: ``build_openai_compatible_client(provider, ...)`` flows the
  base_url / api-key env / model from the named row. ``provider`` is the
  endpoint NAME (``"local"`` / ``"together"`` / ``"nvidia"`` / …) and is
  written verbatim to ``out["provider"]`` + surfaced in the
  ``synthesis_provider_call`` decision-capture event for audit.
- The task logic (paraphrase prompts, length clamping, parse-retry +
  length-remediation + surface-form preservation gate, decision capture)
  is lifted VERBATIM from the ``LocalSynthesisProvider`` superset. The
  preserve / length gate stays UNCONDITIONAL — it is a natural no-op when
  ``preserve_tokens`` is empty and ``min_preserve_rate`` is 0.0 (the
  default), so the hosted case is inert without a per-vendor branch.
- ``terse_prompts`` is the ONLY prompt-behavior switch: ``True`` (default,
  the local seat) uses the terse local system prompts; ``False`` uses the
  verbose hosted system prompts. Everything else is identical regardless
  of provider.
- The LENIENT JSON parse (``OpenAICompatibleClient._extract_json_lenient``)
  is used for ALL providers — strictly more permissive than the strict
  ``together`` parser (it recovers fenced / prose-wrapped JSON), so the
  agnostic provider can RECOVER a malformed response where the legacy
  strict-together path would have retried / exhausted.

Phase 2 is ADDITIVE: this class is NOT wired into ``run_synthesis`` or
the provider factories yet. It ships alongside the per-vendor leaves
unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from Trainforge.generators._synthesis_common import (  # noqa: F401
    COMPLETION_MAX,
    COMPLETION_MIN,
    PROMPT_MAX,
    PROMPT_MIN,
    SynthesisProviderError,
    _KIND_BOUNDS,
)
from Trainforge.generators._local_provider import (
    DEFAULT_LOCAL_KIND_BOUNDS,
    DEFAULT_MIN_PRESERVE_RATE,
    _LOCAL_INSTRUCTION_SYSTEM_PROMPT,
    _LOCAL_PREFERENCE_SYSTEM_PROMPT,
)
from Trainforge.generators._together_provider import (
    MAX_PARSE_RETRIES,
    _INSTRUCTION_SYSTEM_PROMPT,
    _PREFERENCE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# Env var the registry-driven provider reads for an explicit model
# override before falling back to the endpoint row's registry default.
ENV_MODEL = "TRAINFORGE_SYNTHESIS_MODEL"

# Default provider (endpoint name) — the local seat.
DEFAULT_PROVIDER = "local"


class SynthesisProvider:
    """Paraphrases mock-provider drafts via ANY OpenAI-wire backend.

    Resolves its embedded :class:`OpenAICompatibleClient` from the unified
    endpoint registry by NAME (``provider``), so the transport + identity
    (base_url / api-key env / model) flow from ``config/endpoints.yaml``.
    The paraphrase prompts, length clamping, parse-retry +
    length-remediation + surface-form preservation gate, and
    ``synthesis_provider_call`` audit emit are lifted from the
    ``LocalSynthesisProvider`` superset and parameterized — no per-vendor
    branch.

    Constructor params:

    - ``provider`` — endpoint NAME in the registry (``"local"`` default /
      ``"together"`` / ``"nvidia"`` / any ``openai_compatible`` row).
      Written to ``out["provider"]`` + the decision-capture event.
    - ``model`` — explicit model override. ``None`` → ``TRAINFORGE_SYNTHESIS_MODEL``
      → the registry row's default.
    - ``capture`` — optional ``DecisionCapture`` for the per-call
      ``synthesis_provider_call`` event.
    - ``client`` — pre-built ``httpx.Client`` (tests inject one with
      ``httpx.MockTransport``); threaded into the registry constructor so
      the ``api_key_required`` fail-loud is suppressed for test clients.
    - ``timeout`` — per-call HTTP timeout (``None`` → the client resolves
      from ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` → 60s).
    - ``temperature`` / ``max_tokens`` — sampling knobs.
    - ``kind_bounds`` — per-``kind`` length bounds (``None`` →
      ``DEFAULT_LOCAL_KIND_BOUNDS``, identical to the canonical
      ``_KIND_BOUNDS``).
    - ``max_parse_retries`` — parse / length-retry budget (``None`` →
      ``MAX_PARSE_RETRIES``).
    - ``min_preserve_rate`` — fraction of ``preserve_tokens`` that must
      survive a paraphrase (``None`` → ``DEFAULT_MIN_PRESERVE_RATE`` =
      0.0, a NO-OP when no tokens are passed).
    - ``terse_prompts`` — ``True`` (default) uses the terse local system
      prompts; ``False`` uses the verbose hosted prompts. The ONLY
      prompt-behavior switch.
    """

    def __init__(
        self,
        *,
        provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        capture: Optional[Any] = None,
        client: Optional[httpx.Client] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.4,
        max_tokens: int = 800,
        kind_bounds: Optional[Dict[str, tuple]] = None,
        max_parse_retries: Optional[int] = None,
        min_preserve_rate: Optional[float] = None,
        terse_prompts: bool = True,
    ) -> None:
        # Provider tag written to ``out["provider"]`` and surfaced in the
        # decision-capture event for audit.
        self._provider_name = provider

        # Model resolution: explicit arg > TRAINFORGE_SYNTHESIS_MODEL >
        # (let the registry row's default fill when None).
        resolved_model = model or os.environ.get(ENV_MODEL)

        self._capture = capture
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._kind_bounds: Dict[str, tuple] = (
            dict(kind_bounds) if kind_bounds else dict(DEFAULT_LOCAL_KIND_BOUNDS)
        )
        self._max_parse_retries = (
            int(max_parse_retries)
            if max_parse_retries is not None
            else MAX_PARSE_RETRIES
        )
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

        # Prompt-behavior switch — the ONLY per-mode difference.
        self._terse_prompts = bool(terse_prompts)
        if self._terse_prompts:
            self._instruction_system_prompt = _LOCAL_INSTRUCTION_SYSTEM_PROMPT
            self._preference_system_prompt = _LOCAL_PREFERENCE_SYSTEM_PROMPT
        else:
            self._instruction_system_prompt = _INSTRUCTION_SYSTEM_PROMPT
            self._preference_system_prompt = _PREFERENCE_SYSTEM_PROMPT

        # Composition: resolve the LLM-agnostic client from the endpoint
        # registry BY NAME. The transport (base_url) + identity (api-key
        # env, default model) flow from the ``provider`` row in
        # ``config/endpoints.yaml`` — the registry default fills the model
        # when ``resolved_model`` is None. json_mode=True mirrors the
        # per-vendor leaves (Ollama JSON-grammar + OpenAI response_format).
        from lib.llm.endpoints import (  # noqa: PLC0415 — lazy, transport-free import
            build_openai_compatible_client,
        )

        self._oa_client = build_openai_compatible_client(
            provider,
            model=resolved_model,
            capture=None,
            provider_label=provider,
            client=client,
            json_mode=True,
            timeout=timeout,
        )
        # Read back the resolved model id (registry default fills it when
        # ``resolved_model`` is None) so the payload + rationale use the
        # actual model the registry chose.
        self._model = self._oa_client.model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api_url(self) -> str:
        """Full chat-completions endpoint URL for this provider."""
        return self._oa_client.api_url

    @property
    def base_url(self) -> str:
        """Resolved base URL of the registry endpoint."""
        return self._oa_client.base_url

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
        MUST appear verbatim in the model's output. The directive is
        injected into the user prompt and verified after parse under the
        ``min_preserve_rate`` floor (default 0.0 → the gate is a no-op).
        """
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        preserve = list(preserve_tokens or [])
        user_prompt = self._render_instruction_user(draft, chunk_id, preserve)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=self._instruction_system_prompt,
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
        # produced them. Optional emit; absent on the today-default shape.
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

        ``preserve_tokens`` listed CURIEs are checked against the
        ``chosen`` field (the factually-correct completion) only —
        ``rejected`` is not checked (the rule-synthesized rejection may
        legitimately omit the technical token).
        """
        if not isinstance(draft, dict):
            raise TypeError("draft must be a dict")
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or "")
        preserve = list(preserve_tokens or [])
        user_prompt = self._render_preference_user(draft, chunk_id, preserve)

        parsed, usage, retry_count = self._call_with_parse(
            system_prompt=self._preference_system_prompt,
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
    # Internals — lifted verbatim from the LocalSynthesisProvider superset
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
        """Call the backend and parse JSON via the lenient extractor.

        Superset of the strict-together loop: recovers fenced / prose-
        wrapped JSON (lenient extractor), retries on missing required
        keys, retries on below-floor field lengths (length-remediation),
        and retries on a below-``min_preserve_rate`` surface-form drop.
        After ``max_parse_retries`` exhaustion raises
        ``SynthesisProviderError`` (``paraphrase_invalid_after_retry`` or,
        on a preservation miss, ``surface_form_preservation_failed``).
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
                    self._provider_name, attempts, retry_budget,
                    text[-120:],
                )
                continue
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                last_err = f"response missing required keys: {missing}"
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
                logger.warning(
                    "%s synthesis: length-retry %d/%d: %s",
                    self._provider_name, attempts, retry_budget,
                    last_err,
                )
                messages = self._append_length_remediation(
                    messages, field, length, floor,
                )
                continue
            # Surface-form preservation gate. After length passes, verify
            # each preserve_token appears verbatim in the nominated fields.
            missing_tokens = self._missing_preserve_tokens(
                parsed, preserve_tokens or [], preserve_in_keys,
            )
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
                    logger.warning(
                        "%s synthesis: preserve-retry %d/%d: %s",
                        self._provider_name, attempts, retry_budget,
                        last_err,
                    )
                    messages = self._append_preserve_remediation(
                        messages, missing_tokens, preserve_in_keys,
                    )
                    continue
                logger.debug(
                    "%s synthesis: soft-accept (rate %.2f >= floor %.2f); "
                    "tokens dropped: %s",
                    self._provider_name, preserved_rate,
                    self._min_preserve_rate, missing_tokens,
                )
            return parsed, last_usage, total_http_retries
        # Distinguish preservation failure so the caller can fall back to
        # the deterministic draft instead of dropping the pair entirely.
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

    @staticmethod
    def _missing_preserve_tokens(
        parsed: Dict[str, Any], tokens: List[str], in_keys: tuple
    ) -> List[str]:
        """Return tokens that don't appear verbatim in any of ``in_keys``."""
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
        """Corrective user turn naming the dropped tokens."""
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
        """Return ``(field_name, length, floor)`` for the first below-floor
        required key, or None when all required fields meet the floor."""
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
        """Return a new message list with a corrective user turn appended."""
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
    # Strict-JSON directives — appended at the END of the user message.
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
        no tokens."""
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
        from Trainforge.generators._base_synthesis_provider import (
            EVIDENCE_QUOTE_PROMPT_DIRECTIVE,
        )
        preserve = cls._preserve_directive(
            preserve_tokens or [], "the prompt or completion",
        )
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
            f"{EVIDENCE_QUOTE_PROMPT_DIRECTIVE}"
            f"{cls._INSTRUCTION_JSON_DIRECTIVE}"
        )

    @classmethod
    def _render_preference_user(
        cls, draft: Dict[str, Any], chunk_id: str,
        preserve_tokens: Optional[List[str]] = None,
    ) -> str:
        from Trainforge.generators._base_synthesis_provider import (
            EVIDENCE_QUOTE_PROMPT_DIRECTIVE,
        )
        preserve = cls._preserve_directive(
            preserve_tokens or [], "the chosen completion",
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
            f"{self._provider_name} paraphrase ({kind}) for chunk {chunk_id} "
            f"using model {self._model} at {self.base_url}; "
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
        from Trainforge.generators._base_synthesis_provider import (
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
            f"chunk_id={chunk_id}) through the registry endpoint "
            f"'{self._provider_name}' at base_url={self.base_url} using "
            f"model {self._model} for paraphrase. The OpenAI-wire backend "
            f"is resolved by NAME from config/endpoints.yaml — adding a "
            f"provider is a registry row, not a subclass. "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"http_retries={retry_count}. "
            f"{evidence_fragment}."
        )


__all__ = [
    "SynthesisProvider",
    "ENV_MODEL",
    "DEFAULT_PROVIDER",
    "SynthesisProviderError",
]
