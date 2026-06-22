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


class EndpointRuntimeError(RuntimeError):
    """Raised on any endpoint failure (missing key, non-200, timeout,
    malformed response). Fail-loud — there is no silent mock fallback."""


class EndpointBatchItemError(EndpointRuntimeError):
    """Raised when a SINGLE item in a ``generate_batch`` call fails.

    Carries the failing prompt's index so the caller can pin the error to
    the originating region rather than blaming the whole batch. The batch
    driver re-raises this (fail-loud) — one bad slot does NOT silently
    drop or fabricate that region's completion."""

    def __init__(self, index: int, message: str) -> None:
        self.index = index
        super().__init__(f"batch item {index}: {message}")


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
    ) -> list[str]:
        """Return one completion per prompt, IN INPUT ORDER, CONCURRENTLY.

        Fires the N POSTs through a :class:`ThreadPoolExecutor` bounded by
        :func:`resolve_specialist_concurrency`
        (``SEMANTIK_SPECIALIST_CONCURRENCY``, default 8). Each prompt is
        one region's request (REUSES the single-call :meth:`_one_completion`
        path — same headers/body/error handling), so the whole Stage-6
        region set lands in roughly one call's wall-clock instead of N
        serial calls.

        Fail-loud, per-item: a slot whose POST fails raises
        :class:`EndpointBatchItemError` carrying that slot's index — the
        whole batch is NOT silently failed and NO slot is fabricated or
        dropped. The FIRST failing index (by input order) is the one
        re-raised so the error is deterministic.

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
                return self._one_completion(
                    system=system,
                    user=user,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=(None if seed is None else int(seed + index)),
                    repeat_penalty=repeat_penalty,
                )
            except EndpointRuntimeError as exc:
                # Re-wrap with the originating index so the caller can pin
                # the failure to a region. Fail-loud — no fabricated slot.
                raise EndpointBatchItemError(index, str(exc)) from exc

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
            # Deterministic: re-raise the lowest-index failure.
            first = min(errors)
            raise errors[first]
        # All slots resolved; the list is now fully str-typed.
        return [r if r is not None else "" for r in results]

    # -- internals ---------------------------------------------------------

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
            raise EndpointRuntimeError(
                f"OpenAICompatibleRuntime: request to endpoint failed "
                f"({type(exc).__name__}): {exc}"
            ) from exc

        if resp.status_code != 200:
            # Never log the Authorization header; surface only the status
            # and a bounded body head for diagnosis.
            body_head = (resp.text or "")[:500]
            raise EndpointRuntimeError(
                f"OpenAICompatibleRuntime: endpoint returned HTTP "
                f"{resp.status_code} (model={self._model}). Body head: "
                f"{body_head!r}"
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
    "resolve_specialist_concurrency",
    "split_specialist_prompt",
]
