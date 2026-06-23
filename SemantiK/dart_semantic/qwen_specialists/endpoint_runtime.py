"""Hosted-endpoint Qwen-specialist runtime (OpenAI-compatible /chat/completions).

This is the HEAVY-generation arm of the Stage 6 specialist runtime. The
8 GB box keeps the SMALL components local (council/structure BERTs, region
detection, theta), but the math/table/prose/gap_fill specialist generation
can route to a hosted 70B model (the NVIDIA seat) instead of a local GGUF.

``OpenAICompatibleRuntime`` implements the same :class:`QwenRuntime`
Protocol as :class:`~.runtime.LlamaCppRuntime`, so the Stage 6 runner +
:class:`~.base.AdapterSwap` dispatch against it identically:

    * ``load(path)`` / ``free()`` are no-ops — there is NO local model to
      bind or release. (AdapterSwap already skips ``load()`` for the
      null-adapter v1 config; this runtime additionally never needs a
      GGUF path, so even a configured path is ignored.)
    * ``generate(prompt, *, n, ...) -> list[str]`` POSTs to
      ``{base_url}/chat/completions`` and returns ``n`` assistant message
      strings (one HTTP call per candidate so the K samples stay distinct
      under temperature, mirroring ``LlamaCppRuntime``'s K-loop).

Self-contained on purpose
-------------------------

We do NOT import ``Trainforge.generators._openai_compatible_client`` — the
vendored SemantiK tree must run inside Semantic's venv (which ships
``requests`` but not the Trainforge package). We mirror that client's
shape with a small ``requests``-based POST instead.

No-silent-fallback invariant
----------------------------

Every error path FAILS LOUD (matches ``feedback_no_silent_fallbacks.md``):

    * Missing API key                → ``EndpointRuntimeError``.
    * Non-200 HTTP response          → ``EndpointRuntimeError`` (status + body head).
    * Timeout / connection error     → ``EndpointRuntimeError``.
    * Malformed / empty response JSON→ ``EndpointRuntimeError``.

There is NO degrade-to-mock arm: an operator who opted into the endpoint
provider gets an endpoint or an exception, never fabricated content.

Config — ENV only (no hardcoded key, no machine paths)
------------------------------------------------------

    base_url  : SEMANTIK_SPECIALIST_BASE_URL > NVIDIA_BASE_URL
    api_key   : SEMANTIK_SPECIALIST_API_KEY  > NVIDIA_API_KEY
    model     : SEMANTIK_SPECIALIST_MODEL    > NVIDIA_LARGE_MODEL
                > "meta/llama-3.3-70b-instruct" (sane literal default)
    timeout   : SEMANTIK_SPECIALIST_TIMEOUT_SECONDS (default 120.0)

The system prompt instructs the hosted model to act as the relevant
specialist and to emit the SAME OUTPUT ENVELOPE the local adapter
produces (one HTML5 / MathML fragment, no commentary, no code fences) so
the Stage 9 assembler parses it identically.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# Sane literal default model — NOT machine-specific. The NVIDIA seat's
# 70B rewrite model id, matching Ed4All's ED4ALL_DYNAMIC_BLOCK_PLAN /
# objective-review default.
_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
_DEFAULT_TIMEOUT_SECONDS = 120.0
# Default number of concurrent in-flight POSTs for generate_batch. The
# hosted seat tolerates a small fan-out; 8 keeps wall-clock ~one call for a
# whole region set without hammering the endpoint. Overridable via
# SEMANTIK_SPECIALIST_CONCURRENCY (parse-with-fallback).
_DEFAULT_CONCURRENCY = 8

# --- Batched multi-region primitive (the HTTP-429 rate-limit defeat) -------
#
# The hosted seat caps REQUESTS per minute (the NVIDIA seat ~40/min), not
# tokens. The per-region POST path fires ~197×K POSTs for a real slice and
# 429s on ~every call. generate_multi packs many regions into ONE POST: the
# limit is requests, so a handful of large calls survive where hundreds of
# small ones do not.
#
# Default regions packed per batched POST. Overridable via
# SEMANTIK_SPECIALIST_BATCH_REGIONS (parse-with-fallback, mirrors the
# concurrency knob). 12 keeps each batched prompt comfortably inside a 70B
# context window while collapsing ~197 prose regions to ~17 POSTs.
_DEFAULT_BATCH_REGIONS = 12

# Hard ceiling on the summed per-region max_tokens for ONE batched call —
# the truncation guard. A batch whose summed output budget would exceed this
# (minus the safety margin) is capped here; the runner's _pack_batches also
# splits big-output batches so this cap is rarely the binding constraint.
# 16384 is a conservative completion ceiling shared by the 70B seats.
_DEFAULT_OUTPUT_TOKEN_CEILING = 16384
# Safety margin subtracted from the ceiling so the batched response always
# has headroom (the model never emits exactly max_tokens of useful content).
_OUTPUT_TOKEN_SAFETY_MARGIN = 512

# Per-region delimiters the batched prompt instructs the model to wrap each
# fragment in. DOTALL-parsed back into {region_id: fragment}. The id is the
# region's stable r{idx} id so the runner maps fragments back to jobs.
_BATCH_REGION_OPEN = '<<<DART_REGION id="{rid}">>>'
_BATCH_REGION_CLOSE = '<<<DART_REGION_END id="{rid}">>>'
# DOTALL regex: capture the body between a matching open/close pair. The
# backreference \1 forces the END tag's id to equal the OPEN tag's id, so a
# region whose END tag is missing/mismatched simply does not match -> None.
_BATCH_REGION_RE = re.compile(
    r'<<<DART_REGION id="(r\d+)">>>(.*?)<<<DART_REGION_END id="\1">>>',
    re.DOTALL,
)


class EndpointRuntimeError(RuntimeError):
    """Raised on any endpoint failure (missing key, non-200, timeout,
    malformed response). Fail-loud — there is no silent mock fallback.

    Carries a ``transient`` flag so the retry/fail-soft policy can tell a
    retryable error (ReadTimeout / ConnectionError / 5xx / 429) from a
    clearly-permanent one (missing key, 400/401/403, malformed request).
    Permanent errors are NOT retried — re-asking a bad key or a malformed
    request just burns the timeout budget."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        self.transient = transient
        super().__init__(message)


class EndpointBatchItemError(EndpointRuntimeError):
    """Raised when a SINGLE item in a ``generate_batch`` call fails.

    Carries the failing prompt's index so the caller can pin the error to
    the originating region rather than blaming the whole batch. The batch
    driver re-raises this (fail-loud) — one bad slot does NOT silently
    drop or fabricate that region's completion."""

    def __init__(self, index: int, message: str, *, transient: bool = False) -> None:
        self.index = index
        super().__init__(f"batch item {index}: {message}", transient=transient)


# Default bounded-retry count for a region's endpoint call on TRANSIENT
# errors only. Mirrors the existing SEMANTIK_SPECIALIST_* parse-with-
# fallback knob style. 2 retries = up to 3 total attempts per region.
_DEFAULT_MAX_RETRIES = 2
# Base back-off (seconds) between retry attempts; multiplied by the attempt
# number for a short linear back-off (attempt 1 -> 0.5s, attempt 2 -> 1.0s).
_RETRY_BACKOFF_BASE_SECONDS = 0.5


def resolve_specialist_max_retries() -> int:
    """Parse-with-fallback ``SEMANTIK_SPECIALIST_MAX_RETRIES`` (default 2).

    The number of EXTRA attempts after the first on a TRANSIENT endpoint
    error (ReadTimeout / ConnectionError / 5xx / 429). Garbage / negative
    values fall back to the default; ``0`` is honoured (no retry — a single
    attempt), mirroring the timeout/concurrency knobs."""
    raw = os.environ.get("SEMANTIK_SPECIALIST_MAX_RETRIES")
    if not raw:
        return _DEFAULT_MAX_RETRIES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RETRIES
    if val < 0:
        return _DEFAULT_MAX_RETRIES
    return val


def resolve_disable_thinking() -> bool:
    """Parse-with-fallback ``SEMANTIK_SPECIALIST_DISABLE_THINKING`` (default off).

    When truthy (``1``/``true``/``yes``/``on``), the endpoint request body
    carries ``chat_template_kwargs={"thinking": False}`` so a REASONING model
    (e.g. ``deepseek-ai/deepseek-v4-pro``) suppresses chain-of-thought and
    returns the structured answer directly. Without it a reasoning model
    emits a large ``<think>`` budget that overruns ``max_tokens`` on the
    structured authoring/editing tasks (the same failure mode the
    non-reasoning nemotron sibling was chosen to avoid). Default OFF →
    byte-identical for the non-reasoning seats (no extra body field); the
    field is harmlessly ignored by models that do not read it."""
    raw = (os.environ.get("SEMANTIK_SPECIALIST_DISABLE_THINKING") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_specialist_concurrency() -> int:
    """Parse-with-fallback ``SEMANTIK_SPECIALIST_CONCURRENCY`` (default 8).

    Garbage / non-positive values fall back to the default, mirroring the
    timeout knob. Bounds the ThreadPoolExecutor in :meth:`generate_batch`."""
    raw = os.environ.get("SEMANTIK_SPECIALIST_CONCURRENCY")
    if not raw:
        return _DEFAULT_CONCURRENCY
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CONCURRENCY
    if val <= 0:
        return _DEFAULT_CONCURRENCY
    return val


def resolve_specialist_batch_regions() -> int:
    """Parse-with-fallback ``SEMANTIK_SPECIALIST_BATCH_REGIONS`` (default 12).

    The maximum number of regions packed into ONE batched endpoint POST by
    :meth:`OpenAICompatibleRuntime.generate_multi` (via the runner's
    ``_pack_batches``). Garbage / non-positive values fall back to the
    default, mirroring :func:`resolve_specialist_concurrency`. Larger values
    cut total POSTs (the rate-limit win) at the cost of a bigger per-call
    prompt; the output-token cap independently shrinks batches whose summed
    output budget is large (e.g. tables)."""
    raw = os.environ.get("SEMANTIK_SPECIALIST_BATCH_REGIONS")
    if not raw:
        return _DEFAULT_BATCH_REGIONS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BATCH_REGIONS
    if val <= 0:
        return _DEFAULT_BATCH_REGIONS
    return val


def _strip_code_fences(text: str) -> str:
    """Strip a single Markdown code fence (```lang … ``` / bare ```).

    Mirrors ``reviewer._strip_code_fences`` (kept local to avoid a
    cross-module import) so a model that wraps the whole batched response in
    a fence does not defeat the delimiter parse. Returns the inner body when
    fenced; otherwise the input unchanged."""
    if not text:
        return ""
    s = text.strip()
    if "```" not in s:
        return s
    first = s.find("```")
    rest = s[first + 3:]
    nl = rest.find("\n")
    if nl != -1:
        head = rest[:nl].strip().lower()
        if head and head.isalpha():
            rest = rest[nl + 1:]
    close = rest.find("```")
    if close != -1:
        return rest[:close].strip()
    return rest.strip()


def parse_batch_envelope(
    response: str, region_ids: list[str]
) -> dict[str, str | None]:
    """Parse a batched response into ``{region_id: fragment | None}``.

    Tolerant DOTALL extraction of every
    ``<<<DART_REGION id="rN">>> … <<<DART_REGION_END id="rN">>>`` pair (the
    END id is backreference-pinned to the OPEN id, so a missing/mismatched
    END tag yields no match for that region). Code fences are stripped first.
    Every requested ``region_id`` is present in the result; any region whose
    pair is absent maps to ``None`` (the fail-soft sentinel, same contract as
    :meth:`generate_batch`)."""
    body = _strip_code_fences(response or "")
    found: dict[str, str] = {}
    for match in _BATCH_REGION_RE.finditer(body):
        rid = match.group(1)
        # First occurrence wins (defensive against a duplicated block).
        found.setdefault(rid, match.group(2).strip())
    # Single-region tolerance: when exactly ONE region was requested and the
    # model emitted its fragment WITHOUT the envelope wrapper (common — given a
    # single item the model just answers) OR truncated before the END tag (so
    # the backreference-pinned pair never matched), treat the whole stripped
    # body as that region's fragment instead of dropping it to the None
    # sentinel. Only fires for a 1-region batch with zero parsed pairs, so
    # there is no region-to-fragment ambiguity; multi-region batches are
    # unchanged (the envelope is required to disambiguate them).
    if len(region_ids) == 1 and not found and body.strip():
        return {region_ids[0]: body.strip()}
    return {rid: found.get(rid) for rid in region_ids}


def _resolve_base_url() -> str | None:
    return (
        os.environ.get("SEMANTIK_SPECIALIST_BASE_URL")
        or os.environ.get("NVIDIA_BASE_URL")
    )


def _resolve_api_key() -> str | None:
    return (
        os.environ.get("SEMANTIK_SPECIALIST_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
    )


def _resolve_model() -> str:
    return (
        os.environ.get("SEMANTIK_SPECIALIST_MODEL")
        or os.environ.get("NVIDIA_LARGE_MODEL")
        or _DEFAULT_MODEL
    )


def _resolve_timeout() -> float:
    """Parse-with-fallback timeout (mirrors ED4ALL_ANSWER_TIMEOUT_SECONDS)."""
    raw = os.environ.get("SEMANTIK_SPECIALIST_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    if val <= 0 or val != val:  # non-positive / NaN
        return _DEFAULT_TIMEOUT_SECONDS
    return val


# ---------------------------------------------------------------------------
# Specialist-aware chat wrapping
# ---------------------------------------------------------------------------

# The Stage 6 prompt builders emit a single "SYSTEM: <role>\nUSER: <json>"
# string (see prompts.py::_format_prompt). The local adapter is trained to
# treat that whole string as the user turn (chat_format.wrap_for_qwen). For
# a hosted instruct model we split it into a real system + user turn AND
# reinforce the OUTPUT-ENVELOPE contract so the 70B emits a bare fragment
# the assembler can parse identically to the adapter's output.
_ENVELOPE_DIRECTIVE = (
    "You are a DART document-conversion specialist. Convert the single "
    "region described in the USER message into ONE accessible HTML5 "
    "fragment conforming to ARIA-in-HTML and WCAG 2.2 AA. For math emit "
    "MathML 4.0 (a single <math ...>...</math> with an alttext attribute); "
    "for tables emit a single <table> with <caption>, <thead>/<tbody>, and "
    "<th scope=...>; for prose emit clean semantic HTML. Output the "
    "fragment ONLY — no Markdown, no code fences, no commentary, no "
    "surrounding prose."
)


# System-turn directive for the BATCHED multi-region call. Instructs the
# model to emit one delimited block per region, in order, with NO commentary
# or code fences between blocks, so parse_batch_envelope can split the
# response back into per-region fragments.
_BATCH_ENVELOPE_DIRECTIVE = (
    "You will convert MULTIPLE regions in ONE response. The USER message "
    "lists each region, tagged with its id. For EACH region, emit EXACTLY "
    "this block and nothing else around it:\n"
    '<<<DART_REGION id="r{idx}">>>\n'
    "...single HTML5/MathML fragment for that region...\n"
    '<<<DART_REGION_END id="r{idx}">>>\n'
    "Emit the blocks IN THE SAME ORDER the regions appear, using each "
    "region's OWN id. Put NOTHING between blocks — no commentary, no "
    "Markdown, no code fences, no blank-line prose. Each block's body is the "
    "ONE accessible fragment that region's content converts to (the same "
    "single-fragment contract as a single-region conversion)."
)

# System-turn directive for the BATCHED REFINE call — stated ONCE (not per
# region). Each region's USER block carries its own DRAFT_FRAGMENT; this tells
# the model to improve+complete each draft into its delimited output block.
_BATCH_REFINE_DIRECTIVE = (
    "Each region below includes a DRAFT_FRAGMENT produced by a smaller local "
    "specialist. For each region, IMPROVE and COMPLETE its draft — fix "
    "malformed markup, fill gaps, raise quality — but keep it grounded in "
    "that region's described content. Emit ONLY the corrected fragment inside "
    "that region's delimited block."
)


def split_specialist_prompt(prompt: str) -> tuple[str, str]:
    """Split a ``SYSTEM: ...\\nUSER: ...`` prompt into (system, user).

    The Stage 6 builders always emit exactly that two-line shape. We pull
    the SYSTEM role line out (preserving the per-specialist guidance) and
    prepend the envelope directive so the hosted model honours the same
    output contract the local adapter is trained on. Anything that does
    not match the expected shape is passed through whole as the user turn
    under the generic envelope directive (defensive — never raise on an
    unexpected prompt shape; the model still gets the region text).
    """
    system_role = ""
    user_part = prompt
    if prompt.startswith("SYSTEM:"):
        # Split on the first "\nUSER:" boundary; the USER payload is a
        # single JSON line, so the first occurrence is authoritative.
        marker = "\nUSER:"
        idx = prompt.find(marker)
        if idx != -1:
            system_role = prompt[len("SYSTEM:") : idx].strip()
            user_part = prompt[idx + len(marker) :].strip()
        else:
            system_role = prompt[len("SYSTEM:") :].strip()
            user_part = ""
    system = _ENVELOPE_DIRECTIVE
    if system_role:
        system = f"{_ENVELOPE_DIRECTIVE}\n\nSpecialist role: {system_role}"
    return system, user_part


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class OpenAICompatibleRuntime:
    """Hosted-endpoint specialist runtime (OpenAI-compatible API).

    Construct via :func:`~.runtime.make_runtime` (mode ``"endpoint"``) or
    directly in tests. Config resolves from ENV at construction time.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Explicit args (tests) win; otherwise resolve from ENV.
        self._base_url = (base_url or _resolve_base_url() or "").rstrip("/")
        self._api_key = api_key or _resolve_api_key()
        self._model = model or _resolve_model()
        self._timeout = timeout if timeout is not None else _resolve_timeout()
        # Observable for tests, symmetric with MockRuntime.
        self.load_calls: list[Path | None] = []
        self.free_calls: int = 0

    # -- Protocol: load / free are no-ops (no local weights) ---------------

    def load(self, gguf_path: Path) -> None:  # noqa: D401 - no-op
        """No-op: the endpoint runtime holds no local model. Recorded for
        test observability; AdapterSwap may still call this when a path is
        configured, but it never binds weights."""
        self.load_calls.append(Path(gguf_path) if gguf_path is not None else None)

    def free(self) -> None:  # noqa: D401 - no-op
        """No-op: nothing local to release."""
        self.free_calls += 1

    # -- Protocol: generate ------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
    ) -> list[str]:
        """Return ``n`` completions via ``n`` POSTs to the endpoint.

        One HTTP call per candidate so K samples stay distinct under
        temperature (mirrors LlamaCppRuntime's K-loop). ``repeat_penalty``
        maps onto the OpenAI ``frequency_penalty`` knob only when != 1.0
        (the OpenAI default is 0.0 == no penalty; llama-cpp 1.0 == none).
        """
        if not self._api_key:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no API key. Set "
                "SEMANTIK_SPECIALIST_API_KEY or NVIDIA_API_KEY in the "
                "environment (never hardcode it). Fail-loud — no silent "
                "mock fallback (feedback_no_silent_fallbacks.md)."
            )
        if not self._base_url:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no base URL. Set "
                "SEMANTIK_SPECIALIST_BASE_URL or NVIDIA_BASE_URL in the "
                "environment."
            )
        system, user = split_specialist_prompt(prompt)
        outputs: list[str] = []
        for i in range(n):
            outputs.append(
                self._one_completion(
                    system=system,
                    user=user,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=(None if seed is None else int(seed + i)),
                    repeat_penalty=repeat_penalty,
                )
            )
        return outputs

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
        fail_soft: bool = False,
    ) -> list[str] | list[str | None]:
        """Return one completion per prompt, IN INPUT ORDER, CONCURRENTLY.

        Fires the N POSTs through a :class:`ThreadPoolExecutor` bounded by
        :func:`resolve_specialist_concurrency`
        (``SEMANTIK_SPECIALIST_CONCURRENCY``, default 8). Each prompt is
        one region's request (REUSES the single-call :meth:`_one_completion`
        path — same headers/body/error handling), so the whole Stage-6
        region set lands in roughly one call's wall-clock instead of N
        serial calls.

        Each item is attempted with a bounded retry on TRANSIENT errors
        (ReadTimeout / ConnectionError / 5xx / 429) — see
        :meth:`_one_completion_with_retry` /
        :func:`resolve_specialist_max_retries`. Permanent errors (missing
        key, 400/401/403, malformed request) are NOT retried.

        Failure mode is governed by ``fail_soft`` (default ``False`` keeps
        the BYTE-IDENTICAL legacy behaviour):

        * ``fail_soft=False`` (default): a slot whose POST fails (after
          retries) raises :class:`EndpointBatchItemError` carrying that
          slot's index — the whole batch is NOT silently failed and NO slot
          is fabricated or dropped. The FIRST failing index (by input
          order) is the one re-raised so the error is deterministic.
        * ``fail_soft=True``: a slot whose POST fails (after retries) is
          returned as the ``None`` SENTINEL in its input slot instead of
          raising; successful slots return their completion string. The
          CALLER (runner Phase-2) decides how to degrade that region (keep
          the local draft / emit a skip candidate). This is the per-region
          fail-soft path: one region's endpoint timeout degrades only THAT
          region, never the whole document.

        Pre-flight key/url checks mirror :meth:`generate` so a
        misconfiguration fails before any POST is fired."""
        if not self._api_key:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no API key. Set "
                "SEMANTIK_SPECIALIST_API_KEY or NVIDIA_API_KEY in the "
                "environment (never hardcode it). Fail-loud — no silent "
                "mock fallback (feedback_no_silent_fallbacks.md)."
            )
        if not self._base_url:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no base URL. Set "
                "SEMANTIK_SPECIALIST_BASE_URL or NVIDIA_BASE_URL in the "
                "environment."
            )
        if not prompts:
            return []

        from concurrent.futures import ThreadPoolExecutor  # noqa: WPS433

        max_workers = min(resolve_specialist_concurrency(), len(prompts))

        def _work(index: int, prompt: str) -> str:
            system, user = split_specialist_prompt(prompt)
            try:
                return self._one_completion_with_retry(
                    system=system,
                    user=user,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=(None if seed is None else int(seed + index)),
                    repeat_penalty=repeat_penalty,
                    index=index,
                )
            except EndpointRuntimeError as exc:
                # Re-wrap with the originating index so the caller can pin
                # the failure to a region. Fail-loud — no fabricated slot.
                raise EndpointBatchItemError(
                    index, str(exc), transient=getattr(exc, "transient", False)
                ) from exc

        # Pre-size the results list so completion order doesn't matter —
        # each future writes into its own input slot (order preserved).
        results: list[str | None] = [None] * len(prompts)
        errors: dict[int, BaseException] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_index = {
                pool.submit(_work, i, prompt): i
                for i, prompt in enumerate(prompts)
            }
            for future in future_to_index:
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except BaseException as exc:  # noqa: BLE001
                    errors[idx] = exc
        if errors:
            if fail_soft:
                # Per-region fail-soft: a failed slot stays the None sentinel
                # so the caller degrades only that region. Log each failure
                # so the degradation is visible in the run output.
                for idx in sorted(errors):
                    logger.warning(
                        "Stage6 endpoint item %d failed (fail_soft): %s",
                        idx,
                        errors[idx],
                    )
                return list(results)
            # Deterministic: re-raise the lowest-index failure.
            first = min(errors)
            raise errors[first]
        # All slots resolved; the list is now fully str-typed.
        return [r if r is not None else "" for r in results]

    def generate_multi(
        self,
        items: list[tuple[str, str]],
        *,
        system: str | None = None,
        max_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
        drafts: dict[str, str] | None = None,
        output_token_ceiling: int = _DEFAULT_OUTPUT_TOKEN_CEILING,
    ) -> dict[str, str | None]:
        """Convert MANY regions in ONE batched POST (the rate-limit defeat).

        ``items`` is a list of ``(region_id, user_payload)`` where
        ``region_id`` is the region's stable ``r{idx}`` id and
        ``user_payload`` is its per-region prompt (the same grounded JSON the
        per-region path sends — it travels VERBATIM into the batched user
        message). The whole batch is ONE ``/chat/completions`` POST, so the
        endpoint's request-per-minute cap counts it once.

        Returns ``{region_id: fragment | None}`` for EVERY item: a region
        whose delimited block is absent/malformed in the response maps to
        ``None`` (the same fail-soft sentinel as
        :meth:`generate_batch`'s ``None`` slot). The caller (runner Phase-2)
        degrades only those regions.

        ``max_tokens`` is the SUMMED per-region output budget for the batch,
        CAPPED at ``output_token_ceiling`` minus a safety margin (the
        truncation guard — a batch that asks for more than the model can emit
        would silently truncate the trailing regions, leaving them None).

        ``drafts`` (refine mode): ``{region_id: draft_fragment}``. When
        supplied, each region's draft is embedded in its delimited user
        block and the system turn carries the batched refine directive ONCE
        (not per region).

        ``system`` overrides the system turn; default is the batched
        envelope directive (+ the refine directive when ``drafts`` is set).
        REUSES :meth:`_one_completion_with_retry` UNCHANGED, so the
        429/5xx/timeout transient retry+backoff already applies to the
        batched call."""
        if not self._api_key:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no API key. Set "
                "SEMANTIK_SPECIALIST_API_KEY or NVIDIA_API_KEY in the "
                "environment (never hardcode it). Fail-loud — no silent "
                "mock fallback (feedback_no_silent_fallbacks.md)."
            )
        if not self._base_url:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: no base URL. Set "
                "SEMANTIK_SPECIALIST_BASE_URL or NVIDIA_BASE_URL in the "
                "environment."
            )
        if not items:
            return {}

        region_ids = [rid for rid, _ in items]
        refine = bool(drafts)
        if system is None:
            system_turn = _BATCH_ENVELOPE_DIRECTIVE
            if refine:
                system_turn = f"{_BATCH_ENVELOPE_DIRECTIVE}\n\n{_BATCH_REFINE_DIRECTIVE}"
        else:
            system_turn = system

        # Build ONE user message: each region's id-tagged payload block, in
        # order. The model is told (system turn) to mirror this structure in
        # its response. Each region's grounding payload travels verbatim.
        parts: list[str] = []
        for rid, payload in items:
            block = [
                _BATCH_REGION_OPEN.format(rid=rid),
                payload,
            ]
            if refine:
                draft = (drafts or {}).get(rid, "")
                block.append(f"DRAFT_FRAGMENT:\n{draft}")
            block.append(_BATCH_REGION_CLOSE.format(rid=rid))
            parts.append("\n".join(block))
        user_turn = "\n\n".join(parts)

        # Truncation guard: cap the summed budget so the response cannot ask
        # for more than the model can emit (which silently truncates trailing
        # regions to None). max_tokens is already the per-batch sum; clamp it.
        ceiling = max(1, int(output_token_ceiling) - _OUTPUT_TOKEN_SAFETY_MARGIN)
        capped_max_tokens = min(int(max_tokens), ceiling)

        try:
            response = self._one_completion_with_retry(
                system=system_turn,
                user=user_turn,
                temperature=temperature,
                top_p=top_p,
                max_tokens=capped_max_tokens,
                seed=seed,
                repeat_penalty=repeat_penalty,
            )
        except EndpointRuntimeError as exc:
            # The WHOLE batch POST failed (after retries) — every region in
            # this batch degrades to the None sentinel (fail-soft). The
            # caller decides how to degrade each region; a single batch
            # failure must NOT abort the document.
            logger.warning(
                "Stage6 batched POST (%d regions) failed after retries: %s "
                "— all regions in this batch -> None sentinel",
                len(items),
                exc,
            )
            return {rid: None for rid in region_ids}

        return parse_batch_envelope(response, region_ids)

    # -- internals ---------------------------------------------------------

    def _one_completion_with_retry(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
        repeat_penalty: float,
        index: int | None = None,
    ) -> str:
        """:meth:`_one_completion` with a bounded retry on TRANSIENT errors.

        Retries up to :func:`resolve_specialist_max_retries` times (default
        2) with a short linear back-off, but ONLY when the raised
        :class:`EndpointRuntimeError` is marked ``transient`` (ReadTimeout /
        ConnectionError / 5xx / 429). A permanent error (missing key,
        400/401/403, malformed response) is re-raised on the FIRST attempt
        — retrying it just burns the timeout budget. The final transient
        failure is re-raised so the caller's fail-soft / fail-loud policy
        applies."""
        import time  # noqa: WPS433 - local, mirrors lazy requests import

        max_retries = resolve_specialist_max_retries()
        attempt = 0
        last_exc: EndpointRuntimeError | None = None
        while True:
            try:
                return self._one_completion(
                    system=system,
                    user=user,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    repeat_penalty=repeat_penalty,
                )
            except EndpointRuntimeError as exc:
                last_exc = exc
                if not getattr(exc, "transient", False) or attempt >= max_retries:
                    # Permanent error, or out of retries -> propagate.
                    raise
                attempt += 1
                backoff = _RETRY_BACKOFF_BASE_SECONDS * attempt
                logger.warning(
                    "Stage6 endpoint%s transient error (attempt %d/%d): %s "
                    "— retrying in %.1fs",
                    f" item {index}" if index is not None else "",
                    attempt,
                    max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        # Unreachable (the loop either returns or raises), but keeps type
        # checkers happy.
        raise last_exc  # pragma: no cover

    def _one_completion(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
        repeat_penalty: float,
    ) -> str:
        # Lazy import: keeps the module importable in environments that
        # never construct this runtime (mirrors LlamaCppRuntime's lazy
        # llama_cpp import).
        import requests  # noqa: WPS433

        body: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        # llama-cpp repeat_penalty 1.0 == none; OpenAI frequency_penalty
        # 0.0 == none. Only forward a non-default penalty.
        if repeat_penalty and repeat_penalty != 1.0:
            body["frequency_penalty"] = float(repeat_penalty) - 1.0
        if resolve_disable_thinking():
            # Suppress chain-of-thought on REASONING models so the completion
            # budget is spent on the answer, not a <think> block that overruns
            # max_tokens (e.g. deepseek-ai/deepseek-v4-pro). Honored via the
            # endpoint's chat template; ignored by non-reasoning models.
            body["chat_template_kwargs"] = {"thinking": False}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        try:
            resp = requests.post(
                url, json=body, headers=headers, timeout=self._timeout
            )
        except requests.exceptions.RequestException as exc:  # timeout/conn
            # Timeout / connection errors are TRANSIENT — the endpoint may
            # succeed on a retry (this is the exact ReadTimeout that aborted
            # the real run). Mark transient so the retry/fail-soft policy
            # can recover the region.
            raise EndpointRuntimeError(
                f"OpenAICompatibleRuntime: request to endpoint failed "
                f"({type(exc).__name__}): {exc}",
                transient=True,
            ) from exc

        if resp.status_code != 200:
            # Never log the Authorization header; surface only the status
            # and a bounded body head for diagnosis.
            body_head = (resp.text or "")[:500]
            # 5xx (server) + 429 (rate limit) are TRANSIENT — worth a retry.
            # 4xx other than 429 (400/401/403/404/...) is a PERMANENT
            # client/config error: retrying just burns budget.
            status = resp.status_code
            transient = status >= 500 or status == 429
            raise EndpointRuntimeError(
                f"OpenAICompatibleRuntime: endpoint returned HTTP "
                f"{status} (model={self._model}). Body head: "
                f"{body_head!r}",
                transient=transient,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: endpoint returned non-JSON body"
            ) from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointRuntimeError(
                f"OpenAICompatibleRuntime: malformed response JSON "
                f"(no choices[0].message.content): keys={list(data) if isinstance(data, dict) else type(data)}"
            ) from exc
        if text is None:
            raise EndpointRuntimeError(
                "OpenAICompatibleRuntime: endpoint returned a null message "
                "content"
            )
        return str(text)


__all__ = [
    "EndpointBatchItemError",
    "EndpointRuntimeError",
    "OpenAICompatibleRuntime",
    "parse_batch_envelope",
    "resolve_specialist_batch_regions",
    "resolve_specialist_concurrency",
    "resolve_specialist_max_retries",
    "split_specialist_prompt",
]
