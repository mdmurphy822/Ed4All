#!/usr/bin/env python3
"""Curriculum alignment provider — LLM-agnostic teaching-role classifier.

Replaces the Anthropic-pinned LLM call inside
:func:`Trainforge.align_chunks._classify_with_llm` with a task-specific
provider that can route through Anthropic, Together AI, or a local
OpenAI-compatible server (Ollama / vLLM / llama.cpp / LM Studio /
Fireworks / Groq / etc.). The HTTP machinery for the OpenAI-compatible
backends is composed from
:class:`Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`
so this provider only owns the task semantics: the classification
prompt, the four-role validation, and the per-call decision-capture
emit.

Operator selects the backend via ``CURRICULUM_ALIGNMENT_PROVIDER`` env
or the ``provider`` constructor kwarg. Default is ``"anthropic"`` for
backward compatibility with the existing call site.

Default config:

- ``together`` / ``local`` reuse the same env vars the synthesis
  pipeline owns (``TOGETHER_API_KEY``, ``TOGETHER_SYNTHESIS_MODEL``,
  ``LOCAL_SYNTHESIS_BASE_URL``, ``LOCAL_SYNTHESIS_MODEL``) so the
  operator only needs one local server running for both task surfaces.
- ``max_tokens=64`` (single-word response — ``introduce`` /
  ``elaborate`` / ``reinforce`` / ``synthesize``).
- ``temperature=0.0`` (classification, not generation — deterministic).

Validation: the response must be exactly one of the four allowed
roles. Anything else raises ``SynthesisProviderError(code=
"invalid_role_response")`` so the caller's fallback path fires
instead of poisoning chunks with an out-of-vocabulary role.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from Trainforge.generators._anthropic_provider import (
    SynthesisProviderError,
)
from Trainforge.generators._anthropic_provider import (
    DEFAULT_SYNTHESIS_MODEL as ANTHROPIC_DEFAULT_MODEL,
)
from Trainforge.generators._anthropic_provider import (
    ENV_API_KEY as ANTHROPIC_ENV_API_KEY,
)
from Trainforge.generators._local_provider import (
    DEFAULT_BASE_URL as LOCAL_DEFAULT_BASE_URL,
)
from Trainforge.generators._local_provider import (
    DEFAULT_SYNTHESIS_MODEL as LOCAL_DEFAULT_MODEL,
)
from Trainforge.generators._local_provider import (
    ENV_API_KEY as LOCAL_ENV_API_KEY,
)
from Trainforge.generators._local_provider import (
    ENV_BASE_URL as LOCAL_ENV_BASE_URL,
)
from Trainforge.generators._local_provider import (
    ENV_MODEL as LOCAL_ENV_MODEL,
)
from Trainforge.generators._openai_compatible_client import (
    OpenAICompatibleClient,
)
from Trainforge.generators._together_provider import (
    DEFAULT_BASE_URL as TOGETHER_DEFAULT_BASE_URL,
)
from Trainforge.generators._together_provider import (
    DEFAULT_SYNTHESIS_MODEL as TOGETHER_DEFAULT_MODEL,
)
from Trainforge.generators._together_provider import (
    ENV_API_KEY as TOGETHER_ENV_API_KEY,
)
from Trainforge.generators._together_provider import (
    ENV_MODEL as TOGETHER_ENV_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_PROVIDER = "CURRICULUM_ALIGNMENT_PROVIDER"
DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("anthropic", "together", "local")

# Four allowed teaching roles for the curriculum-alignment classifier.
# Mirrored against ``Trainforge.align_chunks.VALID_ROLES`` — but the
# four LLM-routable roles are a subset (``assess`` / ``transfer`` are
# heuristic-only, never LLM-classified).
VALID_ROLES = ("introduce", "elaborate", "reinforce", "synthesize")

# Hints to the model. Each is short and pedagogically grounded so the
# four-class output stays well-calibrated regardless of backend.
_ROLE_DEFINITIONS = (
    "- introduce: First exposure to concepts in the course sequence.\n"
    "- elaborate: Adds depth to previously introduced concepts.\n"
    "- reinforce: Revisits concepts from earlier weeks in a new context.\n"
    "- synthesize: Connects multiple concepts or summarises."
)

_SYSTEM_PROMPT = (
    "You are a curriculum-alignment classifier. Given a chunk of course "
    "material plus its position and concept tags, output exactly one of "
    "the four allowed teaching roles: introduce, elaborate, reinforce, "
    "or synthesize. Output ONLY the role token — no preamble, no "
    "explanation, no markdown, no commentary. Do not output any role "
    "outside the four allowed values."
)

_DEFAULT_MAX_TOKENS = 64
_DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class CurriculumAlignmentProvider:
    """LLM-agnostic teaching-role classifier.

    Constructor selects the backend via the ``provider`` kwarg
    (``anthropic`` / ``together`` / ``local``); when none is passed,
    falls back to the ``CURRICULUM_ALIGNMENT_PROVIDER`` env var,
    defaulting to ``anthropic`` for backward compatibility with the
    pre-refactor align_chunks call site.

    Together + Local route through :class:`OpenAICompatibleClient`
    (composition, not inheritance). Anthropic routes through the
    Anthropic SDK directly via the same lazy-import pattern
    :class:`AnthropicSynthesisProvider` uses, so the curriculum
    surface doesn't grow a second SDK dependency surface.

    Public method:

    - ``classify_teaching_role(chunk_text, *, chunk_id, neighbors,
      course_outcomes=None) -> str`` — returns one of
      ``introduce`` / ``elaborate`` / ``reinforce`` / ``synthesize``.

    On any other response, raises
    ``SynthesisProviderError(code="invalid_role_response")`` so the
    caller's fallback path (mock heuristic) fires instead of writing
    an out-of-vocabulary role onto a chunk.

    Decision capture: every call emits one
    ``decision_type="curriculum_alignment_call"`` event whose
    rationale interpolates ``chunk_id``, ``chosen_role``, ``provider``,
    ``model``, and the underlying retry count. ≥20 chars per the
    project's LLM call-site instrumentation contract.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capture: Optional[Any] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
    ) -> None:
        resolved_provider = (
            provider
            or os.environ.get(ENV_PROVIDER)
            or DEFAULT_PROVIDER
        ).lower()
        if resolved_provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"CurriculumAlignmentProvider: unknown provider "
                f"{resolved_provider!r}; expected one of "
                f"{list(SUPPORTED_PROVIDERS)}"
            )
        self._provider = resolved_provider
        self._capture = capture
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

        # Each branch resolves model / base_url / api_key off the
        # synthesis-pipeline env vars so an operator running a single
        # local server (Ollama on :11434, say) doesn't have to set a
        # separate CURRICULUM_*_BASE_URL for the same endpoint.
        if resolved_provider == "anthropic":
            self._model = (
                model
                or os.environ.get("ANTHROPIC_SYNTHESIS_MODEL")
                or ANTHROPIC_DEFAULT_MODEL
            )
            resolved_key = api_key or os.environ.get(ANTHROPIC_ENV_API_KEY)
            if anthropic_client is None and not resolved_key:
                raise RuntimeError(
                    f"{ANTHROPIC_ENV_API_KEY} required for "
                    f"CurriculumAlignmentProvider(provider='anthropic'); "
                    "set the env var or inject an anthropic_client "
                    "(tests)."
                )
            self._api_key = resolved_key
            self._anthropic_client = anthropic_client
            self._oa_client: Optional[OpenAICompatibleClient] = None
            self._base_url: Optional[str] = None

        elif resolved_provider == "together":
            self._model = (
                model
                or os.environ.get(TOGETHER_ENV_MODEL)
                or TOGETHER_DEFAULT_MODEL
            )
            resolved_key = api_key or os.environ.get(TOGETHER_ENV_API_KEY)
            if client is None and not resolved_key:
                raise RuntimeError(
                    f"{TOGETHER_ENV_API_KEY} required for "
                    f"CurriculumAlignmentProvider(provider='together'); "
                    "set the env var or inject a client (tests)."
                )
            self._api_key = resolved_key
            self._base_url = (base_url or TOGETHER_DEFAULT_BASE_URL).rstrip("/")
            self._oa_client = OpenAICompatibleClient(
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
                capture=None,
                provider_label="together",
                client=client,
                # Tests inject ``sleep_fn=lambda _s: None`` via the
                # client kwarg; default keeps stdlib sleep.
            )
            self._anthropic_client = None

        else:  # local
            self._model = (
                model
                or os.environ.get(LOCAL_ENV_MODEL)
                or LOCAL_DEFAULT_MODEL
            )
            resolved_key = (
                api_key
                or os.environ.get(LOCAL_ENV_API_KEY)
                or "local"
            )
            self._api_key = resolved_key
            env_base_url = os.environ.get(LOCAL_ENV_BASE_URL)
            self._base_url = (
                base_url or env_base_url or LOCAL_DEFAULT_BASE_URL
            ).rstrip("/")
            self._oa_client = OpenAICompatibleClient(
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
                capture=None,
                provider_label="local",
                client=client,
            )
            self._anthropic_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_teaching_role(
        self,
        chunk_text: str,
        *,
        chunk_id: str,
        neighbors: List[Dict[str, Any]],
        course_outcomes: Optional[List[str]] = None,
    ) -> str:
        """Return the teaching role for ``chunk_text``.

        Args:
            chunk_text: Text body of the chunk being classified.
            chunk_id: Stable chunk identifier — used for decision-
                capture rationale + error correlation.
            neighbors: Adjacent chunks in the course sequence (caller-
                supplied window). Each entry is expected to have
                ``id``, ``concept_tags``, and a short ``text`` excerpt
                — but the field shape is intentionally permissive
                because different callers carry different chunk
                envelopes.
            course_outcomes: Optional list of outcome statements
                (TO/CO IDs or text) for additional pedagogical
                context. When omitted the prompt skips that section.

        Returns:
            One of ``introduce`` / ``elaborate`` / ``reinforce`` /
            ``synthesize``.

        Raises:
            SynthesisProviderError: When the model output is not one
                of the four allowed roles.
        """
        chunk_id = str(chunk_id)
        user_prompt = self._render_user_prompt(
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            neighbors=neighbors,
            course_outcomes=course_outcomes,
        )
        text, retry_count = self._dispatch_call(user_prompt)
        role = self._validate_role(text, chunk_id=chunk_id)

        self._emit_decision(
            chunk_id=chunk_id,
            role=role,
            retry_count=retry_count,
            raw_text=text,
        )
        return role

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        chunk_text: str,
        chunk_id: str,
        neighbors: List[Dict[str, Any]],
        course_outcomes: Optional[List[str]],
    ) -> str:
        neighbor_lines: List[str] = []
        for n in neighbors[:6]:  # bounded to keep prompt size sane
            nid = str(n.get("id") or "")
            tags = n.get("concept_tags") or []
            excerpt = str(n.get("text") or "")
            if len(excerpt) > 160:
                excerpt = excerpt[:160].rstrip() + "..."
            neighbor_lines.append(
                f"- {nid} (concepts={tags}): {excerpt}"
            )
        neighbors_block = (
            "\n".join(neighbor_lines) if neighbor_lines else "(none)"
        )
        outcomes_block = (
            "\n".join(f"- {o}" for o in course_outcomes)
            if course_outcomes
            else "(none)"
        )

        # Trim chunk text. Classification needs context, not the whole
        # chunk; ~1200 chars covers a typical paragraph.
        if len(chunk_text) > 1200:
            chunk_text = chunk_text[:1200].rstrip() + "..."

        return (
            f"Allowed roles:\n{_ROLE_DEFINITIONS}\n\n"
            f"Chunk ID: {chunk_id}\n\n"
            f"Adjacent chunks (pedagogical context):\n{neighbors_block}\n\n"
            f"Course outcomes:\n{outcomes_block}\n\n"
            f"Chunk text:\n{chunk_text}\n\n"
            "Respond with exactly one role token from the allowed list."
        )

    def _dispatch_call(self, user_prompt: str) -> tuple:
        """Route through the selected backend; return ``(text, retries)``."""
        if self._provider == "anthropic":
            return self._call_anthropic(user_prompt)
        # Together / Local both go through OpenAICompatibleClient via
        # the embedded ``self._oa_client``. The client's
        # ``chat_completion`` is the public surface; we drop down to
        # ``_post_with_retry`` here so we can surface the retry count
        # for the curriculum-alignment decision capture.
        assert self._oa_client is not None
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        body, retry_count = self._oa_client._post_with_retry(payload)
        text = self._oa_client._extract_text(body)
        return text, retry_count

    def _call_anthropic(self, user_prompt: str) -> tuple:
        """Run the call against the Anthropic SDK.

        Lazy-imports ``anthropic`` so callers using only Together /
        Local don't pay the import cost. Mirrors the
        :class:`AnthropicSynthesisProvider` pattern for consistency.
        Returns ``(assistant_text, retry_count=0)`` — the SDK has its
        own retry policy so we don't double-count here.
        """
        client = self._anthropic_client
        if client is None:
            try:
                import anthropic  # noqa: PLC0415 — lazy by design
            except ImportError as exc:  # pragma: no cover — covered via mocks
                raise RuntimeError(
                    "anthropic package required for "
                    "CurriculumAlignmentProvider(provider='anthropic'). "
                    "Install with: pip install anthropic"
                ) from exc
            client = anthropic.Anthropic(api_key=self._api_key)
            self._anthropic_client = client
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Pull text from a content list (mirrors the AnthropicSynthesisProvider
        # ``_extract_text``). Mocks may pass a dict; SDK passes typed objects.
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if not content:
            return "", 0
        parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "text":
                continue
            t = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if t:
                parts.append(str(t))
        return "".join(parts), 0

    def _validate_role(self, text: str, *, chunk_id: str) -> str:
        """Strip + lowercase + match against ``VALID_ROLES``.

        Tolerant: takes the first word of the response (so a model
        that returns ``"introduce."`` or ``"introduce — first ..."``
        still classifies). Anything outside the allowed set raises
        ``SynthesisProviderError(code="invalid_role_response")``.
        """
        if not isinstance(text, str):
            raise SynthesisProviderError(
                f"curriculum-alignment response was not a string for "
                f"chunk {chunk_id!r}",
                code="invalid_role_response",
                chunk_id=chunk_id,
            )
        candidate = text.strip().lower()
        if not candidate:
            raise SynthesisProviderError(
                f"curriculum-alignment response was empty for "
                f"chunk {chunk_id!r}",
                code="invalid_role_response",
                chunk_id=chunk_id,
            )
        # Strip surrounding punctuation; take the first word.
        first_token = candidate.split(None, 1)[0]
        first_token = "".join(
            ch for ch in first_token if ch.isalpha()
        )
        if first_token in VALID_ROLES:
            return first_token
        raise SynthesisProviderError(
            f"curriculum-alignment response {text!r} for chunk "
            f"{chunk_id!r} is not one of {list(VALID_ROLES)}",
            code="invalid_role_response",
            chunk_id=chunk_id,
        )

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------

    def _emit_decision(
        self,
        *,
        chunk_id: str,
        role: str,
        retry_count: int,
        raw_text: str,
    ) -> None:
        if self._capture is None:
            return
        try:
            self._capture.log_decision(
                decision_type="curriculum_alignment_call",
                decision=(
                    f"Curriculum-alignment classification chose "
                    f"role={role} for chunk {chunk_id} via "
                    f"provider={self._provider}, model={self._model}, "
                    f"retry_count={retry_count}."
                ),
                rationale=(
                    f"Routing teaching-role classification for chunk_id="
                    f"{chunk_id} through provider={self._provider}, "
                    f"model={self._model}"
                    + (
                        f", base_url={self._base_url}"
                        if self._base_url
                        else ""
                    )
                    + f". Chosen role: {role}; raw_response_chars="
                    f"{len(raw_text or '')}; retry_count={retry_count}. "
                    "Backend choice is operator-controlled via the "
                    "CURRICULUM_ALIGNMENT_PROVIDER env (anthropic / "
                    "together / local) so the curriculum surface stays "
                    "LLM-agnostic — the same align_chunks call site "
                    "works against Anthropic, ToS-clean Together, or "
                    "an offline local server."
                ),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("curriculum_alignment_call capture failed: %s", exc)


__all__ = [
    "CurriculumAlignmentProvider",
    "VALID_ROLES",
    "ENV_PROVIDER",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "SynthesisProviderError",
]
