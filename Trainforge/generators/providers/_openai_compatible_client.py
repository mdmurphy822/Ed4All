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

import ast
import hashlib as _hashlib
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

from Trainforge.generators.providers._synthesis_common import (  # noqa: F401
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

# OP2 usage tap: when this env identifies a run, every chat-completion call
# appends one metering row to ``runtime/state/runs/<run_id>/llm_usage.jsonl`` (read
# by ``lib/aggregators/build_cost.py``). Best-effort + no-op when unset, so a
# bare library caller (no run id) is byte-identical to the pre-tap path.
ENV_RUN_ID: str = "ED4ALL_RUN_ID"

# Active-phase metering context. The workflow engine publishes the name of the
# phase currently executing (``core/executor.py::execute_phase`` sets this env
# at phase entry and restores it in a ``finally``, mirroring the instance-level
# ``self._active_phase_name`` it already tracks) so an in-process content-gen
# LLM call — which, unlike the SemantiK cascade taps, is dispatched from MANY
# phases (course_planning, concept_extraction, content_generation_*,
# assessment_synthesis, training_synthesis) and cannot know its phase from a
# static literal — can stamp the SPENDING phase on its usage row. Read at
# row-emit time, best-effort; unset / blank leaves ``phase`` off the row so a
# bare library caller stays byte-identical. Same wire contract as the executor's
# literal (kept in sync there by comment).
ENV_ACTIVE_PHASE: str = "ED4ALL_ACTIVE_PHASE"

# Nemotron "detailed thinking off" toggle. NVIDIA Nemotron-3 REASONING models
# emit ``<think>`` reasoning tokens by default plus a large system preamble;
# a bulk-authoring call whose preamble + reasoning + answer exceeds
# ``max_tokens`` trips the ``finish_reason == "length"`` anti-truncation guard
# in ``_extract_text`` and CRASHES the phase. When this env is truthy the
# client disables reasoning on every composed OpenAI-compatible call via a
# belt-and-suspenders dual-injection (mirroring the Wave-113 ``json_mode``
# rationale — servers ignore what they don't understand): (a) a ``detailed
# thinking off`` SYSTEM directive (the ollama-path mechanism) and (b)
# ``chat_template_kwargs={"enable_thinking": False}`` (the vLLM / HF-native
# mechanism the ``nemotron_v3`` reasoning parser honors). Default OFF →
# byte-identical to the legacy path (no messages change, no payload key added),
# so an existing Qwen / dev-box deployment is unaffected. Parse-with-fallback
# truthy semantics mirror the other ``ED4ALL_*`` boolean knobs.
ENV_REASONING_THINKING_OFF: str = "ED4ALL_REASONING_THINKING_OFF"

# The exact system-directive phrase the Nemotron ollama path keys off. Kept as
# a module constant so the injection site and the regression test agree on the
# literal.
_REASONING_THINKING_OFF_DIRECTIVE: str = "detailed thinking off"

# Strict-OpenAI servers (e.g. TRT-LLM ``trtllm-serve``) validate the request body
# with pydantic ``extra_forbidden`` and REJECT the Ollama-style top-level
# ``format`` field with HTTP 400. When this env is truthy the ``json_mode``
# dual-inject omits ``format: "json"`` and sends ONLY the OpenAI-spec
# ``response_format`` object (which trtllm-serve accepts). Default off →
# byte-identical (both fields sent, as Ollama / vLLM want). Parse-with-fallback
# truthy semantics mirror the other ``ED4ALL_*`` boolean knobs.
ENV_OMIT_OLLAMA_FORMAT: str = "ED4ALL_LLM_OMIT_OLLAMA_FORMAT"
RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT: str = "openai_json_schema_strict"


def _omit_ollama_format() -> bool:
    raw = os.environ.get(ENV_OMIT_OLLAMA_FORMAT)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"} if raw else False


def _rejects_ollama_format(response: httpx.Response) -> bool:
    """Return whether a strict OpenAI server rejected top-level ``format``.

    TRT-LLM reports this as a pydantic ``extra_forbidden`` error at
    ``["body", "format"]``.  Match that structured shape narrowly: an
    unrelated HTTP 400 must retain the normal fail-loud behavior.
    """
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    details = body.get("detail") if isinstance(body, dict) else None
    # Current ``trtllm-serve`` wraps this pydantic detail list in an ``error``
    # string using Python repr rather than returning the list as JSON. Parse
    # literals only (never evaluate code), then apply the same narrow match.
    if not isinstance(details, list) and isinstance(body, dict):
        encoded = body.get("error")
        if isinstance(encoded, str):
            try:
                decoded = ast.literal_eval(encoded)
            except (SyntaxError, ValueError):
                decoded = None
            if isinstance(decoded, list):
                details = decoded
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict) or detail.get("type") != "extra_forbidden":
            continue
        loc = detail.get("loc")
        if isinstance(loc, (list, tuple)) and tuple(loc) == ("body", "format"):
            return True
    return False

# NVIDIA's Nemotron-3 model card documents a third reasoning mode — LOW EFFORT —
# that keeps thinking ENABLED but emits "significantly fewer reasoning tokens
# than full thinking mode" via
# ``chat_template_kwargs={"enable_thinking": True, "low_effort": True}``. This
# env selects it. Precedence across the two reasoning knobs (high → low):
# ``ED4ALL_REASONING_LOW_EFFORT`` > ``ED4ALL_REASONING_THINKING_OFF`` > neither.
# Default OFF → byte-identical to the legacy path. Parse-with-fallback truthy
# semantics mirror the other ``ED4ALL_*`` boolean knobs.
ENV_REASONING_LOW_EFFORT: str = "ED4ALL_REASONING_LOW_EFFORT"

# Task #10 — TTFT (time-to-first-token) metering. When this env is truthy the
# chat-completion call switches from the non-streaming JSON POST to a streaming
# (``stream: true`` + ``stream_options.include_usage``) POST so the client can
# measure wall-clock from request-send to the FIRST content / reasoning delta
# (``ttft_ms``), then RECONSTRUCTS a response object byte-shape-identical to the
# non-streaming return (``message.content`` + ``finish_reason`` + ``usage``) so
# no downstream caller sees a behavioral difference. On ANY streaming-specific
# failure (server rejects stream, malformed SSE, transport error) the client
# falls back to the proven non-streaming ``_post_with_retry`` path with NO ttft
# recorded — metering never fails a real call. Default OFF → byte-identical to
# the legacy non-streaming path (no ``stream`` key on the wire, no new usage-row
# field). Parse-with-fallback truthy semantics mirror the other ``ED4ALL_*``
# boolean knobs (``1`` / ``true`` / ``yes`` / ``on``).
ENV_TTFT_METER: str = "ED4ALL_LLM_TTFT_METER"


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


def resolve_reasoning_thinking_off() -> bool:
    """Return whether the Nemotron "detailed thinking off" toggle is on.

    Reads :data:`ENV_REASONING_THINKING_OFF` and parses it as a boolean.
    Truthy values are ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive,
    whitespace-trimmed); anything else — including an unset / empty / garbage
    value — resolves to ``False``. Read at call time (not construction) so a
    test can monkeypatch the env per call, and so a live run can flip the
    behavior without rebuilding client instances. Mirrors the
    parse-with-fallback robustness of :func:`resolve_default_timeout`.
    """
    raw = os.environ.get(ENV_REASONING_THINKING_OFF)
    if not raw or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_reasoning_low_effort() -> bool:
    """Return whether the Nemotron "low effort" reasoning toggle is on.

    Reads :data:`ENV_REASONING_LOW_EFFORT` and parses it as a boolean. Truthy
    values are ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive,
    whitespace-trimmed); anything else — including an unset / empty / garbage
    value — resolves to ``False``. Read at call time (not construction) so a
    test can monkeypatch the env per call and a live run can flip the behavior
    without rebuilding client instances. Mirrors
    :func:`resolve_reasoning_thinking_off`.
    """
    raw = os.environ.get(ENV_REASONING_LOW_EFFORT)
    if not raw or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_ttft_meter() -> bool:
    """Return whether TTFT (time-to-first-token) metering is on.

    Reads :data:`ENV_TTFT_METER` and parses it as a boolean. Truthy values are
    ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive, whitespace-trimmed);
    anything else — including an unset / empty / garbage value — resolves to
    ``False``. Read at call time (not construction) so a test can monkeypatch
    the env per call and a live run can flip the behavior without rebuilding
    client instances. Mirrors :func:`resolve_reasoning_thinking_off`.
    """
    raw = os.environ.get(ENV_TTFT_METER)
    if not raw or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def maybe_append_usage_row(
    *,
    provider_label: str,
    model: str,
    usage: Dict[str, int],
    duration_ms: float,
    ttft_ms: Optional[float] = None,
    stream_usage_present: Optional[bool] = None,
    finish_reason: Optional[str] = None,
    task: Optional[str] = None,
    error: Optional[str] = None,
    include_run_identity: bool = False,
) -> None:
    """Append one OP2 metering row to ``runtime/state/runs/<run_id>/llm_usage.jsonl``.

    Roadmap OP2 + Task #10 + perf-doc P4 — the SHARED tap implementation.
    Module-level so every OpenAI-compatible call site meters through ONE
    seam: :meth:`OpenAICompatibleClient._maybe_append_usage_row` (the
    ``chat_completion`` path) delegates here, and call sites that POST via
    ``_post_with_retry`` directly — notably
    ``Courseforge/generators/_base.py::_dispatch_call_with_usage``, the
    outline/rewrite-tier dispatch that bypasses ``chat_completion`` — call
    this function themselves so their calls land in the same ledger.

    Fires only when ``ED4ALL_RUN_ID`` identifies a run; otherwise a strict
    no-op so bare library callers stay byte-identical. Base row shape:
    ``{ts, provider, model, prompt_tokens, completion_tokens, duration_ms}``.

    Task #10 additive fields (present ONLY when TTFT metering measured a
    streaming call, so the flag-off row is byte-identical to the legacy
    OP2 row): ``ttft_ms`` (float, time-to-first-token) and — when the
    streaming server omitted the ``usage`` block despite
    ``stream_options.include_usage`` — ``stream_usage_present: false`` so an
    auditor knows the token counts on that row are absent-defaulted to 0
    rather than server-reported.

    perf-doc P4 additive field: ``finish_reason`` (the first choice's
    server-reported terminator — ``"stop"`` / ``"length"`` /
    ``"tool_calls"`` / …). Present only when the server reported it; OMITTED
    when None so a body without the field keeps the legacy row shape. This
    is the signal that turns silent-truncation diagnosis (``finish_reason ==
    "length"``) from guesswork into a fact ``BuildCostAggregator`` consumers
    can read straight off the row. NB: the tap contract carries no per-call
    *call-site label* field (the tap sees only ``provider`` / ``model`` /
    ``usage`` — not the caller's ``decision_metadata``), so none is added;
    ``BuildCostAggregator`` reads only the fields it knows and ignores the
    rest, so this addition is additive and never breaks a reader.

    Fully best-effort — ANY failure (unresolvable run dir, unwritable path,
    serialization error) is swallowed so metering never perturbs a real LLM
    call.
    """
    run_id = os.environ.get(ENV_RUN_ID)
    if not run_id or not str(run_id).strip():
        return
    try:
        from datetime import datetime, timezone

        from lib.paths import get_state_runs_dir

        run_dir = get_state_runs_dir() / str(run_id).strip()
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": str(provider_label),
            "model": str(model),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "duration_ms": round(float(duration_ms), 3),
        }
        # Direct raw/structured dispatchers opt into the richer identity
        # fields.  The public ``chat_completion`` path leaves this off so its
        # established row shape remains byte-identical and, critically, a
        # caller that already uses that public path is never double-metered.
        if include_run_identity:
            row["run_id"] = str(run_id).strip()
            row["total_tokens"] = (
                row["prompt_tokens"] + row["completion_tokens"]
            )
            _run_contract = os.environ.get(
                "TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256"
            )
            if _run_contract and str(_run_contract).strip():
                row["synthesis_run_contract_sha256"] = str(
                    _run_contract
                ).strip()
            _component_sha = os.environ.get(
                "TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256"
            )
            if _component_sha and str(_component_sha).strip():
                row["synthesis_contract_components_sha256"] = str(
                    _component_sha
                ).strip()
        if task is not None and str(task).strip():
            row["task"] = str(task).strip()
        # Metering-correctness — stamp the SPENDING phase so the per-phase
        # token breakdown (GUI ``_usage_stats`` groups rows by ``phase``)
        # attributes local content-gen (e.g. the local-seat course_planning
        # TO/CO synthesis) to the phase that spent it, exactly as the SemantiK
        # cascade taps already do. Read from the executor-published active-phase
        # env; additive + best-effort — unset / blank leaves the field off so a
        # bare library caller (no active phase) stays byte-identical.
        _active_phase = os.environ.get(ENV_ACTIVE_PHASE)
        if _active_phase and str(_active_phase).strip():
            row["phase"] = str(_active_phase).strip()
        # perf-doc P4 — additive finish_reason, omitted when None so a body
        # without the field keeps the legacy row shape byte-identical.
        if finish_reason is not None:
            row["finish_reason"] = str(finish_reason)
        if error is not None and str(error).strip():
            row["error"] = str(error).strip()
        # Additive TTFT fields — omitted when None so a flag-off / non-
        # streaming row stays byte-identical to the legacy OP2 shape.
        if ttft_ms is not None:
            row["ttft_ms"] = round(float(ttft_ms), 3)
        if stream_usage_present is False:
            row["stream_usage_present"] = False
        with open(
            run_dir / "llm_usage.jsonl", "a", encoding="utf-8"
        ) as fh:
            fh.write(_json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — metering must never crash a call
        logger.debug("llm_usage tap failed (ignoring): %s", exc)


class _StreamingUnsupported(Exception):
    """Internal sentinel — the streaming attempt failed in a stream-specific
    way (server rejected the stream, malformed SSE, transport error). The
    caller catches this and falls back to the proven non-streaming
    ``_post_with_retry`` path (no ttft recorded), so metering never fails a
    real LLM call."""


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
        monotonic_fn: Optional[Callable[[], float]] = None,
        json_mode: bool = False,
        vision_capable: bool = False,
        response_dialect: Optional[str] = None,
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
        # ``None`` means the server dialect is not known yet. Strict OpenAI
        # implementations such as TRT-LLM identify the unsupported Ollama
        # extension with a structured 400; after that one negotiation request,
        # this client permanently omits it while retaining OpenAI
        # ``response_format`` JSON mode. The operator env remains an eager
        # opt-out that avoids even the negotiation request.
        self._ollama_format_supported: Optional[bool] = (
            False if _omit_ollama_format() else None
        )
        # ``sleep_fn`` lets composing providers route the retry-backoff
        # sleep through their own module so fixtures patching e.g.
        # ``Trainforge.generators.providers._together_provider.time.sleep`` keep
        # working post-refactor. Default is the stdlib ``time.sleep``.
        self._sleep_fn = sleep_fn or time.sleep
        # ``monotonic_fn`` lets a test inject a deterministic clock so the TTFT
        # measurement is assertable without real time passing. Default is the
        # stdlib ``time.monotonic``; used for BOTH the per-call wall-clock and
        # the time-to-first-token span so they share one clock source.
        self._monotonic = monotonic_fn or time.monotonic
        self._json_mode = bool(json_mode)
        if response_dialect not in {
            None, RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT,
        }:
            raise ValueError(
                f"unsupported response dialect: {response_dialect!r}"
            )
        self._response_dialect = response_dialect
        if response_dialect == RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT:
            self._ollama_format_supported = False
        self._vision_capable = bool(vision_capable)
        # Server-reported token usage of the most recent chat_completion
        # call (``{}`` until the first call). Read by the grounded-answer
        # truncation tripwire; harmless for every other caller.
        self.last_usage: Dict[str, int] = {}
        # Time-to-first-token (ms) of the most recent chat_completion call:
        # a float when TTFT metering was on AND the call streamed successfully,
        # otherwise ``None`` (metering off, or streaming fell back to the
        # non-streaming path). Task #10 observability surface.
        self.last_ttft_ms: Optional[float] = None

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
        # Nemotron "detailed thinking off": resolve the toggle once per call
        # (env-read so a test / live run can flip it without rebuilding the
        # client). When on, inject the system directive here — AFTER the vision
        # rewrite (which targets the LAST user message, unaffected by a
        # prepended system message) and via a fresh list (non-mutation
        # contract). The companion ``chat_template_kwargs`` setdefault runs
        # below, AFTER the ``extra_payload`` merge, so a caller override wins.
        thinking_off = resolve_reasoning_thinking_off()
        if thinking_off:
            messages = self._inject_thinking_off_system(messages)
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
        # vLLM / HF-native mechanism: the ``nemotron_v3`` reasoning parser
        # honors ``chat_template_kwargs.get("enable_thinking") is False``.
        # ``setdefault`` AFTER the merge so an explicit caller
        # ``extra_payload["chat_template_kwargs"]`` wins over our default.
        if thinking_off:
            payload.setdefault(
                "chat_template_kwargs", {"enable_thinking": False}
            )

        _call_start = self._monotonic()
        # Task #10 — TTFT metering. When on, stream the completion and measure
        # time-to-first-token; reconstruct a body identical in shape to the
        # non-streaming return. ANY streaming-specific failure falls back to the
        # proven non-streaming path with NO ttft (metering never fails a call).
        # Default off → byte-identical legacy non-streaming behavior.
        ttft_ms: Optional[float] = None
        stream_usage_present: Optional[bool] = None
        if resolve_ttft_meter():
            try:
                body, retry_count, ttft_ms, stream_usage_present = (
                    self._post_streaming(payload)
                )
            except _StreamingUnsupported as exc:
                logger.debug(
                    "TTFT streaming attempt failed (%s); falling back to "
                    "non-streaming (no ttft recorded)",
                    exc,
                )
                body, retry_count = self._post_with_retry(payload)
                ttft_ms = None
                stream_usage_present = None
        else:
            body, retry_count = self._post_with_retry(payload)
        _duration_ms = (self._monotonic() - _call_start) * 1000.0
        usage = self._extract_usage(body)
        # perf-doc P4: capture the first choice's ``finish_reason`` (``stop`` /
        # ``length`` / ``tool_calls`` / …) for the usage tap. Extracted from
        # the RAW body BEFORE ``_extract_text`` — which RAISES
        # ``output_truncated`` on ``finish_reason == "length"`` — so the
        # truncated call still records a metering row that names the truncation
        # (truncation diagnosis is exactly what this field exists for).
        finish_reason = self._extract_finish_reason(body)
        # OP2 usage tap — best-effort per-call metering row (no-op when
        # ``ED4ALL_RUN_ID`` is unset). Written BEFORE ``_extract_text`` so a
        # ``finish_reason == "length"`` response still records its row. The row
        # carries real server-reported token counts. ``ttft_ms`` (Task #10),
        # the stream-usage-present note, and ``finish_reason`` (perf-doc P4) are
        # OMITTED from the row when None so the flag-off / no-signal path stays
        # byte-identical to the legacy shape.
        self._maybe_append_usage_row(
            usage=usage,
            duration_ms=_duration_ms,
            ttft_ms=ttft_ms,
            stream_usage_present=stream_usage_present,
            finish_reason=finish_reason,
        )
        text = self._extract_text(body)
        self.last_ttft_ms = ttft_ms
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
    # Internals — OP2 usage tap
    # ------------------------------------------------------------------

    def _maybe_append_usage_row(
        self,
        *,
        usage: Dict[str, int],
        duration_ms: float,
        ttft_ms: Optional[float] = None,
        stream_usage_present: Optional[bool] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        """Append one metering row to ``runtime/state/runs/<run_id>/llm_usage.jsonl``.

        Thin delegator to the module-level :func:`maybe_append_usage_row`
        (the shared OP2 tap implementation) with this client's
        ``provider_label`` / ``model`` identity. Kept as a method so the
        existing ``chat_completion`` call site and its regression tests
        stay on the same seam name.
        """
        maybe_append_usage_row(
            provider_label=self._provider_label,
            model=self._model,
            usage=usage,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            stream_usage_present=stream_usage_present,
            finish_reason=finish_reason,
        )

    def post_with_usage(
        self,
        payload: Dict[str, Any],
        *,
        task: str,
    ) -> Tuple[Dict[str, Any], int]:
        """POST one direct raw request and meter that logical call exactly once.

        Provider adapters that need the raw response body cannot use
        :meth:`chat_completion`; historically they called
        :meth:`_post_with_retry` directly and silently bypassed OP2 metering.
        This is their shared seam.  Public ``chat_completion`` must not call
        this helper because it owns its existing streaming-aware meter.
        """
        payload_model = str(payload.get("model") or "").strip()
        if not payload_model or payload_model != self._model:
            error = SynthesisProviderError(
                "direct request model does not match configured client model",
                code="direct_model_identity_mismatch",
                details={
                    "configured_model": self._model,
                    "payload_model": payload_model or "<missing>",
                },
            )
            maybe_append_usage_row(
                provider_label=self._provider_label,
                model=payload_model or self._model,
                usage={},
                duration_ms=0.0,
                task=task,
                error=error.code,
                include_run_identity=True,
            )
            captured = self._emit_direct_failure_capture(
                task=task, model=payload_model or self._model, error=error.code,
                max_tokens=payload.get("max_tokens"),
            )
            if captured:
                setattr(error, "_direct_capture_emitted", True)
            raise error
        started = self._monotonic()
        try:
            body, retry_count = self._post_with_retry(payload)
        except Exception as exc:
            error_code = str(
                getattr(exc, "code", "") or type(exc).__name__
            )
            maybe_append_usage_row(
                provider_label=self._provider_label,
                model=payload_model,
                usage={},
                duration_ms=(self._monotonic() - started) * 1000.0,
                task=task,
                error=error_code,
                include_run_identity=True,
            )
            captured = self._emit_direct_failure_capture(
                task=task, model=payload_model, error=error_code,
                max_tokens=payload.get("max_tokens"),
            )
            if captured:
                try:
                    setattr(exc, "_direct_capture_emitted", True)
                except Exception:
                    pass
            raise
        usage = self._extract_usage(body)
        maybe_append_usage_row(
            provider_label=self._provider_label,
            model=self._model,
            usage=usage,
            duration_ms=(self._monotonic() - started) * 1000.0,
            finish_reason=self._extract_finish_reason(body),
            task=task,
            include_run_identity=True,
        )
        return body, retry_count

    def _emit_direct_failure_capture(
        self, *, task: str, model: str, error: str, max_tokens: Any,
    ) -> bool:
        """Capture a direct-call failure without serialising exception text."""
        if self._capture is None:
            return False
        self._capture.log_decision(
            decision_type="synthesis_provider_call",
            decision=f"Direct synthesis call failed for task={task}.",
            rationale=(
                f"model={model}, task={task}, max_tokens={max_tokens}, "
                f"error_code={error}; the request produced no accepted "
                "structured output and was recorded without secret-bearing "
                "headers or response bodies."
            ),
            alternatives_considered=[],
            context=_json.dumps({
                "provider": self._provider_label,
                "model": model,
                "task": task,
                "max_tokens": max_tokens,
                "error_code": error,
            }, sort_keys=True),
        )
        return True

    # ------------------------------------------------------------------
    # Internals — TTFT streaming (Task #10)
    # ------------------------------------------------------------------

    def _post_streaming(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int, Optional[float], bool]:
        """Stream one chat-completion and measure time-to-first-token.

        Returns ``(body, retry_count, ttft_ms, stream_usage_present)`` where
        ``body`` is reconstructed to be shape-identical to the non-streaming
        return (``choices[0].message.content`` + ``finish_reason`` + ``usage``)
        so ``_extract_text`` / ``_extract_usage`` and the truncation guard all
        behave identically. ``retry_count`` is always 0 (the streaming attempt
        is single-shot; retry lives in the non-streaming fallback).

        Raises :class:`_StreamingUnsupported` on ANY streaming-specific failure
        (non-200 status, malformed SSE, transport error) so the caller falls
        back to the proven non-streaming ``_post_with_retry`` path. Wrapped by
        the same admission gate as the non-streaming path so a hosted
        rate-limited seat is still governed.
        """
        from lib.llm.rate_limiter import get_admission_gate

        _admission = get_admission_gate()
        _est_tokens = self._estimate_payload_tokens(payload)
        _reservation = _admission.acquire(_est_tokens)
        try:
            return self._post_streaming_inner(payload, _reservation)
        finally:
            _reservation.release()

    def _post_streaming_inner(
        self, payload: Dict[str, Any], reservation: Any
    ) -> Tuple[Dict[str, Any], int, Optional[float], bool]:
        """Single streaming POST + SSE reconstruction (admission-gated body)."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Build a COPY carrying the streaming knobs — the caller's ``payload``
        # is left clean so the non-streaming fallback never sends ``stream``.
        stream_payload: Dict[str, Any] = dict(payload)
        stream_payload["stream"] = True
        # ``include_usage`` asks the server to emit a trailing usage-only chunk;
        # servers that don't honor it just omit usage (handled downstream).
        stream_payload["stream_options"] = {"include_usage": True}
        if self._json_mode:
            if self._ollama_format_supported is not False:
                stream_payload.setdefault("format", "json")
            stream_payload.setdefault(
                "response_format", {"type": "json_object"}
            )
        url = self.api_url
        req_start = self._monotonic()
        try:
            with self.client.stream(
                "POST", url, json=stream_payload, headers=headers
            ) as response:
                status = response.status_code
                if status != 200:
                    # Server rejected the streaming request — fall back to the
                    # proven non-streaming path (which will succeed if the
                    # rejection was stream-specific, or raise properly if the
                    # request is genuinely bad).
                    try:
                        response.read()
                    except Exception:  # noqa: BLE001 — best-effort drain
                        pass
                    raise _StreamingUnsupported(
                        f"stream returned HTTP {status}"
                    )
                body, ttft_ms, usage_present = self._consume_stream(
                    response, req_start
                )
        except _StreamingUnsupported:
            raise
        except httpx.HTTPError as exc:
            raise _StreamingUnsupported(
                f"stream transport error: {exc}"
            ) from exc

        # Reconcile the TPM bucket against server-reported usage when present
        # (no-op when rate-limiting is off) — mirrors the non-streaming inner.
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict) and usage:
            actual = usage.get("total_tokens")
            if actual is None:
                actual = (usage.get("prompt_tokens", 0) or 0) + (
                    usage.get("completion_tokens", 0) or 0
                )
            try:
                reservation.reconcile(float(actual))
            except (TypeError, ValueError):
                pass
        return body, 0, ttft_ms, usage_present

    def _consume_stream(
        self, response: "httpx.Response", req_start: float
    ) -> Tuple[Dict[str, Any], Optional[float], bool]:
        """Parse an OpenAI SSE stream into a non-streaming-shaped body.

        Accumulates ``delta.content`` into ``message.content`` and
        ``delta.reasoning_content`` (or ``delta.reasoning``) into
        ``message.reasoning_content``; records ``ttft_ms`` off the shared clock
        at the FIRST content-or-reasoning delta; captures the last non-null
        ``finish_reason`` and any trailing ``usage`` block. Returns
        ``(body, ttft_ms, usage_present)``. Raises :class:`_StreamingUnsupported`
        on malformed SSE JSON so the caller falls back to non-streaming.
        """
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage: Dict[str, Any] = {}
        usage_present = False
        ttft_ms: Optional[float] = None

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = _json.loads(data)
            except ValueError as exc:
                raise _StreamingUnsupported(
                    f"malformed SSE JSON: {exc}"
                ) from exc
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                first = choices[0]
                delta = first.get("delta") or {}
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    rpiece = delta.get("reasoning_content")
                    if rpiece is None:
                        rpiece = delta.get("reasoning")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (self._monotonic() - req_start) * 1000.0
                        content_parts.append(str(piece))
                    if rpiece:
                        if ttft_ms is None:
                            ttft_ms = (self._monotonic() - req_start) * 1000.0
                        reasoning_parts.append(str(rpiece))
                fr = first.get("finish_reason")
                if fr is not None:
                    finish_reason = fr
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict) and chunk_usage:
                usage = chunk_usage
                usage_present = True

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        body: Dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        if ttft_ms is not None:
            ttft_ms = round(ttft_ms, 3)
        return body, ttft_ms, usage_present

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
    # Internals — Nemotron "detailed thinking off" directive
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_thinking_off_system(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a NEW messages list carrying the reasoning-off directive.

        The ollama-path mechanism for disabling Nemotron reasoning: ensure a
        system directive containing the literal phrase
        :data:`_REASONING_THINKING_OFF_DIRECTIVE` (``"detailed thinking off"``)
        rides at the FRONT of the conversation.

        - When ``messages[0]`` is already a system message with string content,
          the directive is PREPENDED to that content (some servers only honor
          the first system message, so a second one is never added).
        - Otherwise a fresh ``{"role": "system", ...}`` message is INSERTED at
          index 0.

        Builds a fresh list (does NOT mutate the caller's list in place) — the
        same non-mutation contract as :meth:`_attach_vision_blocks`, so a
        caller retaining a reference to ``messages`` still sees its original.
        """
        directive = _REASONING_THINKING_OFF_DIRECTIVE
        rewritten = list(messages)
        first = rewritten[0] if rewritten else None
        if (
            isinstance(first, dict)
            and first.get("role") == "system"
            and isinstance(first.get("content"), str)
        ):
            rewritten[0] = {
                **first,
                "content": f"{directive}\n\n{first['content']}",
            }
        else:
            rewritten.insert(0, {"role": "system", "content": directive})
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

    def effective_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the sole payload admitted and sent for this client dialect."""
        effective = dict(payload)
        if self._json_mode:
            if (
                self._response_dialect
                != RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT
                and self._ollama_format_supported is not False
            ):
                effective.setdefault("format", "json")
            effective.setdefault("response_format", {"type": "json_object"})
        return effective

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
        caller_supplied_format = "format" in payload
        effective_payload = self.effective_payload(payload)
        payload.clear()
        payload.update(effective_payload)
        injected_ollama_format = (
            self._json_mode
            and self._response_dialect
            != RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT
            and self._ollama_format_supported is not False
            and "format" in payload
            and not caller_supplied_format
        )
        url = self.api_url
        provider_label = self._provider_label
        last_status: Optional[int] = None
        last_body: str = ""
        for attempt in range(1, self._max_retries + 1):
            # Hash assertion removed from hot path: prevents re-hashing 1.23 MiB
            # source on every retry and dialect attempt. Digest is still RECORDED
            # on journal row via register_contract_files() callsites, just not
            # gated inside the retry loop.
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
            if (
                injected_ollama_format
                and _rejects_ollama_format(response)
            ):
                # Dialect negotiation, not a transient retry: keep the full
                # configured retry budget for the real request. Recursion is
                # bounded because the instance flag prevents reinjection.
                self._ollama_format_supported = False
                payload.pop("format", None)
                logger.info(
                    "%s: server rejected Ollama format extension; retrying "
                    "with OpenAI response_format only",
                    provider_label,
                )
                return self._post_with_retry_inner(payload, reservation)

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
            response_ref = "sha256:" + _hashlib.sha256(
                _json.dumps(
                    body, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            raise SynthesisProviderError(
                "response finish_reason='length' — model hit the "
                "max_tokens cap and the output is truncated; raise "
                "max_tokens for this call",
                code="output_truncated",
                details={
                    "terminal_truncation": True,
                    "finish_reason": "length",
                    "response_ref": response_ref,
                },
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
    def _extract_finish_reason(body: Dict[str, Any]) -> Optional[str]:
        """Pull the first choice's ``finish_reason`` from an OpenAI body.

        Returns the string (``"stop"`` / ``"length"`` / ``"tool_calls"`` / …)
        or ``None`` when the field is absent / empty / not a string. Never
        raises — a malformed / choice-less body just yields ``None`` so the
        usage tap degrades to the legacy (no ``finish_reason``) row shape.
        Read from the RAW body so the value is available even when
        ``_extract_text`` would raise ``output_truncated`` on
        ``finish_reason == "length"``.
        """
        if not isinstance(body, dict):
            return None
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        fr = choices[0].get("finish_reason")
        return fr if isinstance(fr, str) and fr else None

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


def apply_reasoning_thinking_off_payload(
    payload: Dict[str, Any],
    *,
    force_thinking_off: bool = False,
) -> Dict[str, Any]:
    """Apply the Nemotron "detailed thinking off" transform to a built payload.

    For call sites that assemble the ``/chat/completions`` request body dict
    THEMSELVES and POST it via :meth:`OpenAICompatibleClient._post_with_retry`,
    thereby BYPASSING :meth:`OpenAICompatibleClient.chat_completion` — which is
    where the reasoning-off dual-injection normally lives. Two such bypass
    sites exist: ``Courseforge/generators/_base.py::_dispatch_call_with_usage``
    (every Courseforge / textbook-synthesis / outline / rewrite tier) and
    ``Trainforge/generators/providers/_synthesis_provider.py::_chat_completion_raw`` (the
    training-pair seat). Without this helper those paths silently ignore
    ``ED4ALL_REASONING_THINKING_OFF``, so a REASONING model (nemotron_v3) burns
    its ``max_tokens`` budget on ``<think>`` tokens and trips the
    ``finish_reason == "length"`` truncation guard in :meth:`_extract_text`.

    THREE-WAY reasoning mode, precedence (high → low): LOW_EFFORT
    (:func:`resolve_reasoning_low_effort`) > THINKING_OFF
    (:func:`resolve_reasoning_thinking_off`) > neither. Both env vars are read
    on EVERY call (never cached) so a test can toggle them inline and a live run
    can flip behavior without rebuilding client instances.

    - **LOW_EFFORT on** — thinking stays ENABLED but the Nemotron-3 low-effort
      mode is requested by MERGING ``{"enable_thinking": True, "low_effort":
      True}`` into any existing ``payload["chat_template_kwargs"]`` (sibling keys
      preserved, never clobbered). NO system directive is injected — thinking is
      on, just cheaper. Documented on NVIDIA's Nemotron-3 model card as
      "significantly fewer reasoning tokens than full thinking mode".
    - **THINKING_OFF on (LOW_EFFORT off)** — the EXISTING behavior: (a) rewrites
      ``payload["messages"]`` via
      :meth:`OpenAICompatibleClient._inject_thinking_off_system` (fresh list —
      the caller's message list is never mutated in place) so the ``detailed
      thinking off`` SYSTEM directive rides at the front, and (b) ``setdefault``s
      ``chat_template_kwargs={"enable_thinking": False}`` (the vLLM / HF-native
      mechanism the ``nemotron_v3`` reasoning parser honors) so an explicit value
      already present on the payload (a caller override / endpoint-row
      ``extra_body``) wins.
    - **neither** — no-op (byte-identical to today: no messages change, no
      payload key added).

    Mirrors the injection ``chat_completion`` performs; the actual message
    rewrite is the SAME shared static, so the two paths can't drift on the
    directive text. Call it AFTER any ``extra_payload`` / ``extra_body`` merge so
    a caller-supplied override takes precedence. Pure payload-mutator — never
    raises.
    """
    if not force_thinking_off and resolve_reasoning_low_effort():
        # Low-effort keeps thinking ENABLED (no "detailed thinking off" system
        # directive) but requests the cheaper reasoning budget. Merge into any
        # existing chat_template_kwargs so sibling keys survive.
        existing = payload.get("chat_template_kwargs")
        merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        merged["enable_thinking"] = True
        merged["low_effort"] = True
        payload["chat_template_kwargs"] = merged
        return payload
    if not force_thinking_off and not resolve_reasoning_thinking_off():
        return payload
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        payload["messages"] = OpenAICompatibleClient._inject_thinking_off_system(
            messages
        )
    payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return payload


__all__ = [
    "OpenAICompatibleClient",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_REQUEST_TIMEOUT",
    "ENV_RUN_ID",
    "ENV_REASONING_THINKING_OFF",
    "ENV_TTFT_METER",
    "ENV_REASONING_LOW_EFFORT",
    "resolve_default_timeout",
    "resolve_reasoning_thinking_off",
    "resolve_reasoning_low_effort",
    "resolve_ttft_meter",
    "apply_reasoning_thinking_off_payload",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_STATUS_CODES",
    "DEFAULT_INITIAL_BACKOFF_SECONDS",
    "DEFAULT_PROVIDER_LABEL",
    "SynthesisProviderError",
]
