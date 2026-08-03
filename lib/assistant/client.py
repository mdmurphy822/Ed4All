"""Loopback-only OpenAI-compatible chat client for the operator assistant.

The assistant talks to a LOCAL vLLM seat over the OpenAI chat-completions
API (tools/function-calling included) — by default the NVIDIA Nemotron
nano seat, but the code is strictly MODEL-AGNOSTIC: nano appears only in
env-knob defaults and docs, never in logic. Posture mirrors the
grounded-answer path (`lib/retrieval/answer_backend.py`):

* The resolved ``base_url`` host MUST be loopback. Anything else raises
  :class:`AssistantProviderNotLocal` — there is no cloud arm on the
  operator-assistant path, ever, and no escape-hatch env. The check reuses
  the answer path's ``_is_loopback_host`` so the two guards cannot drift.
* A connection failure at call time surfaces as
  :class:`AssistantSeatUnavailable` naming how to start the seat — NEVER a
  canned-answer fallback.

Reasoning-capable seats (e.g. the default Nemotron nano): the client
honors ``ED4ALL_REASONING_THINKING_OFF`` exactly the way the shared
OpenAI-compatible content-generation client does (CONDITIONAL dual
injection, only when that flag is truthy: the ``"detailed thinking off"``
system directive + ``chat_template_kwargs {"enable_thinking": false}`` —
templates that don't support the kwarg simply ignore it) by importing that
client's resolver + directive constant — one mechanism, two call sites.
No unconditional model-specific message surgery.

Env knobs (parse-with-fallback per repo convention; documented in
``docs/operations/behavior-flags.md`` + the root ``CLAUDE.md`` index):

* ``ED4ALL_ASSISTANT_BASE_URL`` — default ``http://localhost:8004/v1``.
* ``ED4ALL_ASSISTANT_MODEL`` — default ``nemotron-3-nano``.
* ``ED4ALL_ASSISTANT_AUTOSTART`` — default off; when truthy AND the seat is
  down, the CLI lazy-starts it via
  ``lib/vllm_container_lifecycle.start_seat_coherent`` (seat name from
  ``ED4ALL_ASSISTANT_SEAT``, default ``spark-nano``; standard liveness
  ceiling + coherence probes).
* ``ED4ALL_ASSISTANT_SEAT`` — default ``spark-nano``; logical registry seat
  name the autostart/hint path targets (the code is model-agnostic — nano
  is only the default deployment).
* ``ED4ALL_ASSISTANT_MAX_TOKENS`` — default ``1024``.
* ``ED4ALL_ASSISTANT_TIMEOUT_SECONDS`` — default ``120``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import lib.vllm_container_lifecycle as _lifecycle
from lib.retrieval.answer_backend import _is_loopback_host
from lib.vllm_container_lifecycle import _probe_ready, start_seat_coherent
from Trainforge.generators.providers._openai_compatible_client import (
    _REASONING_THINKING_OFF_DIRECTIVE,
    resolve_reasoning_thinking_off,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Env knobs
# --------------------------------------------------------------------------- #

ENV_ASSISTANT_BASE_URL = "ED4ALL_ASSISTANT_BASE_URL"
ENV_ASSISTANT_MODEL = "ED4ALL_ASSISTANT_MODEL"
ENV_ASSISTANT_AUTOSTART = "ED4ALL_ASSISTANT_AUTOSTART"
ENV_ASSISTANT_MAX_TOKENS = "ED4ALL_ASSISTANT_MAX_TOKENS"
ENV_ASSISTANT_TIMEOUT = "ED4ALL_ASSISTANT_TIMEOUT_SECONDS"
ENV_ASSISTANT_SEAT = "ED4ALL_ASSISTANT_SEAT"

#: Ordered priority list of LOGICAL seat names (resolved against
#: ``ED4ALL_SEAT_BASE_URLS`` via
#: :func:`lib.vllm_container_lifecycle.parse_seat_registry`) the dynamic
#: seat resolver walks: first LIVE seat wins. Default prefers the large
#: shared pipeline seat, then the small always-available seat. Names here
#: are DEPLOYMENT DEFAULTS only — the resolver is strictly registry-driven
#: and model-agnostic (no "super"/"nano"/"nemotron" logic anywhere).
ENV_ASSISTANT_SEAT_PRIORITY = "ED4ALL_ASSISTANT_SEAT_PRIORITY"

# The assistant is strictly MODEL-AGNOSTIC in code: the Nemotron nano seat
# is only the DEFAULT deployment. "nano"/"nemotron" appear exclusively in
# these default values and in docs/comments — never in logic.
DEFAULT_BASE_URL = "http://localhost:8004/v1"
DEFAULT_MODEL = "nemotron-3-nano"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 120.0

#: DEFAULT logical seat name in the ``ED4ALL_SEAT_BASE_URLS`` /
#: ``ED4ALL_VLLM_CONTAINERS`` / ``ED4ALL_SEAT_LAUNCH_SPECS`` registries the
#: autostart path targets; override with ``ED4ALL_ASSISTANT_SEAT``.
ASSISTANT_SEAT_NAME = "spark-nano"

#: DEFAULT dynamic seat-priority walk (see :data:`ENV_ASSISTANT_SEAT_PRIORITY`).
#: Deployment defaults only — never referenced in resolution LOGIC.
DEFAULT_SEAT_PRIORITY = ("spark-super", "spark-nano")

#: TTL (seconds) of the single-entry module-level active-seat cache. A
#: resolution younger than this returns the cached :class:`ResolvedSeat`
#: unless ``force=True``.
SEAT_RESOLUTION_TTL_SECONDS = 10.0

#: Per-seat liveness-probe timeout (seconds) the default probe passes to
#: :func:`lib.vllm_container_lifecycle._probe_ready` — a short window keeps
#: dynamic resolution cheap while the pipeline shares the seat.
SEAT_PROBE_TIMEOUT_SECONDS = 2.0


class AssistantProviderNotLocal(ValueError):
    """Resolved assistant base_url is non-loopback. No cloud assistant seat, ever."""


class AssistantSeatUnavailable(RuntimeError):
    """The nano seat is absent/refusing. NEVER triggers a canned-answer fallback."""


class AssistantClientError(RuntimeError):
    """The seat answered but the response body is not a usable chat completion."""


def resolve_assistant_base_url(value: Optional[str] = None) -> str:
    """Resolve + loopback-enforce the assistant seat base URL.

    Resolution: explicit ``value`` > ``ED4ALL_ASSISTANT_BASE_URL`` >
    ``http://localhost:8004/v1``. Trailing ``/`` stripped. A non-loopback
    host raises :class:`AssistantProviderNotLocal` (mirrors the
    grounded-answer path's Phase-IA loopback guard — checked on the
    RESOLVED url, so a reverse-proxied cloud URL is also refused).
    """
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_BASE_URL)
    base_url = (str(raw).strip() if raw and str(raw).strip() else DEFAULT_BASE_URL)
    base_url = base_url.rstrip("/")
    if not _is_loopback_host(base_url):
        raise AssistantProviderNotLocal(
            f"Assistant seat resolves to a non-loopback base_url ({base_url!r}). "
            f"The operator assistant permits no cloud seat — the resolved URL "
            f"host must be loopback (localhost / 127.0.0.1 / ::1). Fix "
            f"{ENV_ASSISTANT_BASE_URL} to point at the local assistant vLLM seat."
        )
    return base_url


def resolve_assistant_model(value: Optional[str] = None) -> str:
    """Explicit ``value`` > ``ED4ALL_ASSISTANT_MODEL`` > ``nemotron-3-nano``."""
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_MODEL)
    if raw and str(raw).strip():
        return str(raw).strip()
    return DEFAULT_MODEL


def resolve_assistant_autostart(value: Optional[str] = None) -> bool:
    """Truthy tokens ``1``/``true``/``yes``/``on``; anything else → off."""
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_AUTOSTART)
    if not raw or not str(raw).strip():
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_assistant_max_tokens(value: Optional[str] = None) -> int:
    """Positive int wins; unset / garbage / non-positive → 1024."""
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_MAX_TOKENS)
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_TOKENS
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS
    if parsed <= 0:
        return DEFAULT_MAX_TOKENS
    return parsed


def resolve_assistant_timeout(value: Optional[str] = None) -> float:
    """Positive float wins; unset / garbage / non-positive → 120.0."""
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_TIMEOUT)
    if raw is None or not str(raw).strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    if parsed <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return parsed


def resolve_assistant_seat(value: Optional[str] = None) -> str:
    """Logical seat name the autostart path targets (model-agnostic).

    Resolution: explicit ``value`` > ``ED4ALL_ASSISTANT_SEAT`` > the default
    (``spark-nano`` — a deployment default only). Blank/unset → default
    (parse-with-fallback).
    """
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_SEAT)
    if raw and str(raw).strip():
        return str(raw).strip()
    return ASSISTANT_SEAT_NAME


def resolve_assistant_seat_priority(value: Optional[str] = None) -> List[str]:
    """Resolve the dynamic seat-priority walk (logical seat names).

    Resolution: explicit ``value`` > ``ED4ALL_ASSISTANT_SEAT_PRIORITY`` >
    :data:`DEFAULT_SEAT_PRIORITY`. Comma-split, whitespace-stripped, empty
    tokens dropped; a blank / all-garbage value falls back to
    ``list(DEFAULT_SEAT_PRIORITY)`` (parse-with-fallback).
    """
    raw = value if value is not None else os.environ.get(ENV_ASSISTANT_SEAT_PRIORITY)
    if raw is None or not str(raw).strip():
        return list(DEFAULT_SEAT_PRIORITY)
    seats = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    return seats if seats else list(DEFAULT_SEAT_PRIORITY)


@dataclass(frozen=True)
class ResolvedSeat:
    """The active assistant seat the client should talk to right now.

    ``model`` is NEVER hardcoded: it is the served model id read from the
    seat's ``GET {base_url}/v1/models`` response when live, otherwise the
    :func:`resolve_assistant_model` env fallback. ``source`` is ``"priority"``
    when matched via the ``ED4ALL_SEAT_BASE_URLS`` priority walk, or
    ``"fallback"`` when derived from the ``ED4ALL_ASSISTANT_*`` env resolvers.
    """

    seat_name: Optional[str]
    base_url: str
    model: str
    live: bool
    source: str


def _default_probe(base_url: str) -> bool:
    """Default liveness probe — ``GET {base_url}/v1/models`` 2xx, never raises."""
    return _lifecycle._probe_ready(base_url, timeout=SEAT_PROBE_TIMEOUT_SECONDS)


def _default_model_reader(base_url: str) -> Optional[str]:
    """Default served-model-id reader — first ``/v1/models`` id, or None."""
    return _lifecycle._first_model_id(base_url)


# Single-entry module-level TTL cache: (time.monotonic stamp, ResolvedSeat).
_SEAT_CACHE: Optional[tuple] = None


def reset_seat_cache() -> None:
    """Clear the single-entry active-seat TTL cache (tests + post-autostart)."""
    global _SEAT_CACHE
    _SEAT_CACHE = None


def _non_loopback_seat_error(seat_name: str, base_url: str) -> AssistantProviderNotLocal:
    return AssistantProviderNotLocal(
        f"Assistant priority seat {seat_name!r} resolves to a non-loopback "
        f"base_url ({base_url!r}) via ED4ALL_SEAT_BASE_URLS. The operator "
        f"assistant permits no cloud seat — every resolved seat URL host must "
        f"be loopback (localhost / 127.0.0.1 / ::1). Fix the {seat_name!r} entry "
        f"in ED4ALL_SEAT_BASE_URLS to point at the local vLLM seat."
    )


def resolve_active_seat(
    *,
    force: bool = False,
    probe: Optional[Callable[[str], bool]] = None,
    model_reader: Optional[Callable[[str], Optional[str]]] = None,
) -> ResolvedSeat:
    """Resolve the active assistant seat by the normative priority walk.

    Algorithm (see the S1 shared contract):

    1. A TTL cache hit younger than :data:`SEAT_RESOLUTION_TTL_SECONDS`
       returns the cached :class:`ResolvedSeat` unless ``force=True``.
    2. Parse ``ED4ALL_SEAT_BASE_URLS`` into a ``{seat_name: base_url}``
       registry (fail-soft ``{}``).
    3. Walk :func:`resolve_assistant_seat_priority`; for each priority seat
       present in the registry, enforce the loopback guard (LOUD raise on a
       non-loopback entry — never a silent skip) and probe it. The first LIVE
       seat wins with ``source="priority"``; its served model id comes from
       ``model_reader`` (``/v1/models``) or the env fallback.
    4. No priority seat live → FALLBACK to :func:`resolve_assistant_seat` /
       :func:`resolve_assistant_base_url` with ``source="fallback"``.

    The served model id is NEVER hardcoded — it is read from ``/v1/models`` or
    the ``ED4ALL_ASSISTANT_MODEL`` resolver. ``probe`` / ``model_reader`` are
    injectable for tests (default = the lifecycle-lib helpers).
    """
    global _SEAT_CACHE
    now = time.monotonic()
    if not force and _SEAT_CACHE is not None:
        stamped, cached = _SEAT_CACHE
        if (now - stamped) < SEAT_RESOLUTION_TTL_SECONDS:
            return cached

    probe = probe or _default_probe
    model_reader = model_reader or _default_model_reader
    registry = _lifecycle.parse_seat_registry()

    for seat_name in resolve_assistant_seat_priority():
        if seat_name not in registry:
            continue
        base_url = registry[seat_name].rstrip("/")
        if not _is_loopback_host(base_url):
            raise _non_loopback_seat_error(seat_name, base_url)
        if probe(base_url):
            model = model_reader(base_url) or resolve_assistant_model()
            seat = ResolvedSeat(seat_name, base_url, model, True, "priority")
            _SEAT_CACHE = (now, seat)
            return seat

    # FALLBACK: no priority seat live.
    fb_seat = resolve_assistant_seat()
    reg_url = registry.get(fb_seat, "").rstrip("/")
    if reg_url:
        base_url = reg_url
        if not _is_loopback_host(base_url):
            raise _non_loopback_seat_error(fb_seat, base_url)
    else:
        base_url = resolve_assistant_base_url()
    live = bool(probe(base_url))
    if live:
        model = model_reader(base_url) or resolve_assistant_model()
    else:
        model = resolve_assistant_model()
    seat = ResolvedSeat(fb_seat, base_url, model, live, "fallback")
    _SEAT_CACHE = (now, seat)
    return seat


def seat_start_hint(base_url: str) -> str:
    """The operator-facing 'how do I start the seat' message."""
    return (
        f"The assistant seat at {base_url} is not serving. Start the "
        f"assistant vLLM seat (registry name {resolve_assistant_seat()!r} in "
        f"ED4ALL_SEAT_BASE_URLS / ED4ALL_VLLM_CONTAINERS / "
        f"ED4ALL_SEAT_LAUNCH_SPECS; override the seat name with "
        f"{ENV_ASSISTANT_SEAT}), or set {ENV_ASSISTANT_AUTOSTART}=1 to "
        f"let `ed4all assistant` lazy-start it."
    )


def seat_is_serving(base_url: str) -> bool:
    """Liveness probe (``GET {base_url}/v1/models`` 2xx). Never raises."""
    return _probe_ready(base_url)


def autostart_seat(run_dir: Optional[str] = None):
    """Lazy-start the assistant seat via the lifecycle lib's coherent-start
    path (seat name from :func:`resolve_assistant_seat`, model-agnostic).

    Delegates to :func:`lib.vllm_container_lifecycle.start_seat_coherent`
    (warm ``docker start`` + liveness poll to the standard
    ``ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS`` ceiling + the bounded coherence
    probes, cold-recreate self-heal when a launch spec exists). Returns the
    lifecycle lib's ``SeatStartResult``; the CLI turns a non-``ok`` result
    into a loud exit.
    """
    return start_seat_coherent(resolve_assistant_seat(), run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Transport + client
# --------------------------------------------------------------------------- #

Transport = Callable[[str, Dict[str, Any], float], Dict[str, Any]]


def _chat_completions_url(base_url: str) -> str:
    """Build the OpenAI chat-completions URL, tolerant of both base_url
    conventions in play.

    The dynamic seat resolver returns registry base URLs as HOST ROOTS without
    a ``/v1`` segment (``ED4ALL_SEAT_BASE_URLS`` entries like
    ``spark-super=http://localhost:8001`` — the lifecycle probes append
    ``/v1/models`` themselves), while the ``ED4ALL_ASSISTANT_BASE_URL`` env /
    :data:`DEFAULT_BASE_URL` fallback already carries ``/v1``. Both must produce
    ``{root}/v1/chat/completions`` — appending a bare ``/chat/completions`` to a
    ``/v1``-less registry root 404s (the seat serves under ``/v1``).
    """
    base = str(base_url).rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _urllib_transport(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST ``payload`` as JSON to ``url``; return the decoded JSON body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AssistantClient:
    """Minimal loopback-only chat-completions client with tools support.

    The shared ``OpenAICompatibleClient`` deliberately returns only the
    assistant ``content`` string, so the assistant needs its own thin client
    that surfaces ``tool_calls``. Transport is injectable for tests
    (``transport(url, payload, timeout) -> body dict``).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        # DYNAMIC when neither base_url nor model was pinned explicitly: the
        # seat/model are re-resolved per chat() via resolve_active_seat().
        # An explicit base_url OR model makes the client STATIC — byte-for-byte
        # today's behavior (no probing, last_seat stays None).
        self.dynamic = base_url is None and model is None
        self.base_url = resolve_assistant_base_url(base_url)
        self.model = resolve_assistant_model(model)
        self.timeout = float(timeout) if timeout else resolve_assistant_timeout()
        self.max_tokens = int(max_tokens) if max_tokens else resolve_assistant_max_tokens()
        self._transport = transport
        self.last_seat: Optional[ResolvedSeat] = None

    def _inject_thinking_off(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure the Nemotron ``detailed thinking off`` directive rides at the
        FRONT of the conversation.

        Fresh-list, non-mutating — mirrors
        ``OpenAICompatibleClient._inject_thinking_off_system`` exactly: when
        ``messages[0]`` is already a system message with string content the
        directive is PREPENDED into that content (some servers only honor the
        first system message, so a second system message is never added — which
        would silently drop the assistant's own system prompt / sandbox);
        otherwise a fresh system message is inserted at index 0.
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

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Issue one chat-completion call; return the raw response body.

        Raises :class:`AssistantSeatUnavailable` on transport failure and
        :class:`AssistantClientError` on a body without a usable
        ``choices[0].message``.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        if self.dynamic:
            seat = resolve_active_seat()
            self.last_seat = seat
            self.base_url = seat.base_url
            self.model = seat.model

        thinking_off = resolve_reasoning_thinking_off()
        wire_messages = (
            self._inject_thinking_off(messages) if thinking_off else messages
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "temperature": float(temperature),
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if thinking_off:
            # vLLM / HF-native mechanism — the nemotron_v3 reasoning parser
            # honors chat_template_kwargs.enable_thinking is False.
            payload.setdefault("chat_template_kwargs", {"enable_thinking": False})

        url = _chat_completions_url(self.base_url)
        transport = self._transport or _urllib_transport
        try:
            body = transport(url, payload, self.timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise AssistantSeatUnavailable(
                f"Assistant seat call to {url} failed ({exc}). "
                + seat_start_hint(self.base_url)
            ) from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise AssistantClientError(
                f"Assistant seat at {self.base_url} returned no choices "
                f"(body keys: {sorted(body.keys()) if isinstance(body, dict) else type(body).__name__})."
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise AssistantClientError(
                f"Assistant seat at {self.base_url} returned a choice without "
                f"a message object."
            )
        return body


__all__ = [
    "ENV_ASSISTANT_BASE_URL",
    "ENV_ASSISTANT_MODEL",
    "ENV_ASSISTANT_AUTOSTART",
    "ENV_ASSISTANT_MAX_TOKENS",
    "ENV_ASSISTANT_TIMEOUT",
    "ENV_ASSISTANT_SEAT",
    "ENV_ASSISTANT_SEAT_PRIORITY",
    "ASSISTANT_SEAT_NAME",
    "DEFAULT_SEAT_PRIORITY",
    "SEAT_RESOLUTION_TTL_SECONDS",
    "SEAT_PROBE_TIMEOUT_SECONDS",
    "AssistantClient",
    "AssistantClientError",
    "AssistantProviderNotLocal",
    "AssistantSeatUnavailable",
    "ResolvedSeat",
    "autostart_seat",
    "reset_seat_cache",
    "resolve_active_seat",
    "resolve_assistant_autostart",
    "resolve_assistant_base_url",
    "resolve_assistant_max_tokens",
    "resolve_assistant_model",
    "resolve_assistant_seat",
    "resolve_assistant_seat_priority",
    "resolve_assistant_timeout",
    "seat_is_serving",
    "seat_start_hint",
]
