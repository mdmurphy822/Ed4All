#!/usr/bin/env python3
"""Courseforge content-generator provider — LLM-agnostic page authoring.

Provides a Phase-1 in-process LLM seam for the Courseforge content-
generator surface. Mirrors :class:`Trainforge.generators._curriculum_provider.CurriculumAlignmentProvider`
line-for-line so the operator-facing env-var contract and decision-
capture posture match across the project's LLM call sites. The HTTP
machinery for the OpenAI-compatible backends is composed from
:class:`Trainforge.generators._openai_compatible_client.OpenAICompatibleClient`
so this provider only owns the task semantics: the page-authoring
prompt, the rendered-HTML return contract, and the per-call decision-
capture emit.

Operator selects the backend via ``COURSEFORGE_PROVIDER`` env or the
``provider`` constructor kwarg. Default is ``"anthropic"`` for backward
compatibility with the existing Wave-74 subagent path; the Phase-1 ToS
recommendation to flip operators to ``local`` lands in
``docs/LICENSING.md`` and the root ``CLAUDE.md``, not in code.

Default config:

- ``together`` / ``local`` reuse the same env vars the synthesis
  pipeline owns (``TOGETHER_API_KEY``, ``TOGETHER_SYNTHESIS_MODEL``,
  ``LOCAL_SYNTHESIS_BASE_URL``, ``LOCAL_SYNTHESIS_MODEL``) so an
  operator running a single local server (Ollama on :11434, say)
  doesn't have to set a separate ``COURSEFORGE_*_BASE_URL`` for the
  same endpoint.
- ``max_tokens=4096`` (long-form HTML body — Pattern-22 prevention
  requires substantial educational depth per page).
- ``temperature=0.4`` (light authorial variation while keeping
  determinism viable for cache-keyed reruns).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Phase 2 Subtask 35: ``blocks.py`` lives at
# ``Courseforge/scripts/blocks.py``; ensure the sibling-of-this-package
# directory is importable so ``from blocks import Block`` resolves the
# same regardless of how this provider module is loaded (CLI, MCP tool,
# pytest). Mirrors the pattern in ``Courseforge/scripts/generate_course.py``.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import (  # noqa: E402  (Phase 2 intermediate format)
    Block,
    Touch,
    _parse_provider_page_html,
    _slugify,
)

from Trainforge.generators._anthropic_provider import (  # noqa: E402
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

ENV_PROVIDER = "COURSEFORGE_PROVIDER"
DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("anthropic", "together", "local")

_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.4

# Condensation of ``Courseforge/agents/content-generator.md`` core
# directives. Kept terse on purpose — the model gets the per-page
# context (objectives, key terms, source attribution) through the
# user prompt; the system prompt only carries the always-on authoring
# constraints.
_SYSTEM_PROMPT = (
    "You are a Courseforge content-generator authoring a single page of "
    "accessible course HTML. Always follow Pattern 22 prevention: "
    "produce substantive educational depth (theoretical foundation "
    "before examples; progressive complexity; learning-objective "
    "alignment). Use only the official Courseforge color palette "
    "(#2c5aa0 primary blue, #1a3d6e secondary, #28a745 success, "
    "#ffc107 warning, #dc3545 danger, #f8f9fa light gray, #e0e0e0 "
    "border, #333333 text). Align content to the OSCQR rubric and the "
    "page's stated learning objectives. Ground every claim in the "
    "supplied source material when present. Emit ONLY the rendered "
    "HTML body for the page — no preamble, no markdown fences, no "
    "explanation, no commentary."
)


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class ContentGeneratorProvider:
    """LLM-agnostic Courseforge content-generator authoring provider.

    Constructor selects the backend via the ``provider`` kwarg
    (``anthropic`` / ``together`` / ``local``); when none is passed,
    falls back to the ``COURSEFORGE_PROVIDER`` env var, defaulting to
    ``anthropic`` for backward compatibility with the Wave-74
    subagent path.

    Together + Local route through :class:`OpenAICompatibleClient`
    (composition, not inheritance). Anthropic routes through the
    Anthropic SDK directly via the same lazy-import pattern
    :class:`Trainforge.generators._anthropic_provider.AnthropicSynthesisProvider`
    uses, so the Courseforge surface doesn't grow a second SDK
    dependency surface.

    Public method:

    - ``generate_page(*, course_code, week_number, page_id,
      page_template, page_context) -> str`` — returns rendered HTML as
      ``str``. (Phase 2: will return a ``Block`` dataclass; the env-var
      name does not change.)

    Decision capture: every call emits one
    ``decision_type="content_generator_call"`` event whose rationale
    interpolates ``page_id``, ``course_code``, ``week_number``,
    ``provider``, ``model``, and the underlying retry count. ≥20 chars
    per the project's LLM call-site instrumentation contract.
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
                f"ContentGeneratorProvider: unknown provider "
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
        # separate COURSEFORGE_*_BASE_URL for the same endpoint.
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
                    f"ContentGeneratorProvider(provider='anthropic'); "
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
                    f"ContentGeneratorProvider(provider='together'); "
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

    def generate_page(
        self,
        *,
        course_code: str,
        week_number: int,
        page_id: str,
        page_template: str,
        page_context: Dict[str, Any],
    ) -> Block:
        """Author a single course page.

        Returns a ``Block`` carrying the rendered prose, parsed
        structure, and a single Touch entry annotating the outline-tier
        provenance of this in-process LLM call.

        Args:
            course_code: Course slug (e.g. ``"DEMO_101"``). Used in the
                decision-capture rationale and propagated into the user
                prompt for the model's context.
            week_number: 1-indexed week number for the page.
            page_id: Stable page identifier (e.g.
                ``"week_03_content_01_topic"``). Used for decision-
                capture correlation.
            page_template: The slotted template the model should
                respect when authoring (raw HTML or a ``<!--SLOT-->``
                marker string). Embedded literally in the user prompt.
            page_context: Per-page authoring context — objectives, key
                terms, section headings, primary topic, source refs.
                JSON-serialised into the user prompt.

        Returns:
            A ``Block`` with ``block_type="explanation"``, the
            concatenated paragraphs as ``content``, the slugified
            ``key_terms`` from the page_context, and a single
            ``Touch(tier="outline", purpose="draft")`` entry on
            ``touched_by`` annotating the outline-tier provenance of
            this LLM call. ``block_type="explanation"`` is the sane
            default for the in-process LLM provider's outline draft;
            Phase 3's per-block-type router will dispatch alternative
            block types (e.g. ``self_check_question`` / ``activity``)
            to dedicated provider call sites.

        Raises:
            ValueError: When ``page_id`` or ``course_code`` is empty.
        """
        if not page_id or not str(page_id).strip():
            raise ValueError("page_id required")
        if not course_code or not str(course_code).strip():
            raise ValueError("course_code required")

        user_prompt = self._render_user_prompt(
            course_code=course_code,
            week_number=week_number,
            page_id=page_id,
            page_template=page_template,
            page_context=page_context,
        )
        text, retry_count = self._dispatch_call(user_prompt)

        self._emit_decision(
            course_code=course_code,
            week_number=week_number,
            page_id=page_id,
            retry_count=retry_count,
            raw_text=text,
        )

        # Phase 2 Subtask 35: parse the rendered HTML into the
        # canonical Block intermediate.
        heading, paragraphs = _parse_provider_page_html(text)

        # Slug for the Block ID derives from the heading (or the page_id
        # when the heading is missing). ``block_type="explanation"`` is
        # the outline-tier draft default.
        slug_seed = heading or page_id
        slug = _slugify(slug_seed) or "block"
        block_id = Block.stable_id(page_id, "explanation", slug, 0)

        # ``key_terms`` may arrive as a list of {"term": str, ...} dicts
        # or a list of bare strings; tolerate both shapes by extracting
        # ``term`` when the entry is a dict.
        raw_terms = page_context.get("key_terms") if isinstance(page_context, dict) else None
        key_term_slugs: List[str] = []
        if isinstance(raw_terms, (list, tuple)):
            for entry in raw_terms:
                if isinstance(entry, dict):
                    label = entry.get("term") or entry.get("slug") or ""
                else:
                    label = str(entry or "")
                slugged = _slugify(str(label))
                if slugged:
                    key_term_slugs.append(slugged)

        block = Block(
            block_id=block_id,
            block_type="explanation",
            page_id=page_id,
            sequence=0,
            content=" ".join(paragraphs),
            key_terms=tuple(key_term_slugs),
        )

        touch = Touch(
            model=self._model,
            provider=self._provider,
            tier="outline",
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision_capture_id=self._last_capture_id(),
            purpose="draft",
        )
        block = block.with_touch(touch)
        return block

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        course_code: str,
        week_number: int,
        page_id: str,
        page_template: str,
        page_context: Dict[str, Any],
    ) -> str:
        try:
            context_json = json.dumps(
                page_context or {},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        except (TypeError, ValueError):
            # Fall back to ``repr`` so a non-JSON-serialisable context
            # (e.g. a custom object) doesn't blow up authoring.
            context_json = repr(page_context)

        return (
            f"Course: {course_code}\n"
            f"Week: {week_number}\n"
            f"Page ID: {page_id}\n\n"
            "Page context (JSON):\n"
            f"{context_json}\n\n"
            "Page template (slotted HTML):\n"
            f"{page_template}\n\n"
            "Author the rendered HTML body for this page now. Emit "
            "ONLY the HTML — no preamble, no markdown fences, no "
            "commentary."
        )

    def _dispatch_call(self, user_prompt: str) -> tuple:
        """Route through the selected backend; return ``(text, retries)``."""
        if self._provider == "anthropic":
            return self._call_anthropic(user_prompt)
        # Together / Local both go through OpenAICompatibleClient via
        # the embedded ``self._oa_client``. We drop down to
        # ``_post_with_retry`` here so the retry count surfaces on the
        # decision-capture rationale.
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
        :class:`Trainforge.generators._anthropic_provider.AnthropicSynthesisProvider`
        pattern for consistency. Returns
        ``(assistant_text, retry_count=0)`` — the SDK has its own
        retry policy so we don't double-count here.
        """
        client = self._anthropic_client
        if client is None:
            try:
                import anthropic  # noqa: PLC0415 — lazy by design
            except ImportError as exc:  # pragma: no cover — covered via mocks
                raise RuntimeError(
                    "anthropic package required for "
                    "ContentGeneratorProvider(provider='anthropic'). "
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

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------

    def _last_capture_id(self) -> str:
        """Return ``{file_basename}:{event_index}`` for the most recent
        decision-capture event emitted via :meth:`_emit_decision`.

        Format mirrors the Wave 112 audit-trail convention so a
        ``Touch.decision_capture_id`` always resolves to the exact
        JSONL line that explained the LLM call. When the capture handle
        isn't a real :class:`DecisionCapture` (test injection of a
        ``_FakeCapture`` shape, ``capture=None``, or a streaming-disabled
        capture missing a stream path), falls back to
        ``in-memory:{id(self)}`` so the Wave 112 invariant
        (``decision_capture_id`` must be ≥1 char) is preserved without
        forcing tests to wire up a full capture surface.
        """
        capture = self._capture
        if capture is None:
            return f"in-memory:{id(self)}"

        # Resolve event index. ``DecisionCapture`` exposes ``decisions``;
        # the test fake exposes ``events``. Fall back to 0 when neither
        # surface is present.
        index: Optional[int] = None
        for attr in ("decisions", "events"):
            seq = getattr(capture, attr, None)
            if isinstance(seq, list):
                # ``log_decision`` was already called for the current
                # event, so the most recent entry sits at
                # ``len(seq) - 1``. Negative falls back to 0.
                index = max(len(seq) - 1, 0)
                break

        # Resolve file basename from the streaming-mode stream path
        # when present; otherwise tag the capture as in-memory.
        stream_path = getattr(capture, "_stream_path", None)
        if stream_path is not None:
            try:
                basename = Path(str(stream_path)).name
            except (TypeError, ValueError):  # pragma: no cover — defensive
                basename = None
            if basename:
                return f"{basename}:{index if index is not None else 0}"

        return f"in-memory:{id(self)}"

    def _emit_decision(
        self,
        *,
        course_code: str,
        week_number: int,
        page_id: str,
        retry_count: int,
        raw_text: str,
    ) -> None:
        if self._capture is None:
            return
        try:
            self._capture.log_decision(
                decision_type="content_generator_call",
                decision=(
                    f"Courseforge content-generator authored page "
                    f"page_id={page_id} for course_code={course_code}, "
                    f"week_number={week_number} via "
                    f"provider={self._provider}, model={self._model}, "
                    f"retry_count={retry_count}."
                ),
                rationale=(
                    f"Routing Courseforge content-generator call for "
                    f"page_id={page_id} (course_code={course_code}, "
                    f"week_number={week_number}) through "
                    f"provider={self._provider}, model={self._model}"
                    + (
                        f", base_url={self._base_url}"
                        if self._base_url
                        else ""
                    )
                    + f". Output chars={len(raw_text or '')}; "
                    f"retry_count={retry_count}. Backend choice is "
                    "operator-controlled via the COURSEFORGE_PROVIDER "
                    "env (anthropic / together / local) so the content-"
                    "generator surface stays LLM-agnostic — the same "
                    "page-authoring call site routes through Anthropic, "
                    "ToS-clean Together, or an offline local server "
                    "depending on the operator's licensing posture."
                ),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("content_generator_call capture failed: %s", exc)


__all__ = [
    "ContentGeneratorProvider",
    "ENV_PROVIDER",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "SynthesisProviderError",
]
