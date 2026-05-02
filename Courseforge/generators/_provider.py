#!/usr/bin/env python3
"""Courseforge content-generator provider — LLM-agnostic page authoring.

Provides a Phase-1 in-process LLM seam for the Courseforge content-
generator surface. Mirrors :class:`Trainforge.generators._curriculum_provider.CurriculumAlignmentProvider`
line-for-line so the operator-facing env-var contract and decision-
capture posture match across the project's LLM call sites.

Phase 3 Subtask 10: the HTTP / dispatch / decision-capture plumbing
moved into :class:`Courseforge.generators._base._BaseLLMProvider`;
this module now owns only the page-authoring task surface (the
``generate_page`` public entry, the page-context user prompt, and
the per-call ``content_generator_call`` decision-capture event).
The constructor signature, decision-capture rationale, and Block
return shape are byte-stable across the refactor — Phase 1 tests
(``Courseforge/tests/test_content_generator_provider.py``) pin the
contract.

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

from Courseforge.generators._base import (  # noqa: E402
    _BaseLLMProvider,
)
from Trainforge.generators._anthropic_provider import (  # noqa: E402
    SynthesisProviderError,
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


class ContentGeneratorProvider(_BaseLLMProvider):
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
      page_template, page_context) -> Block`` — returns a
      :class:`Block` (Phase 2 Subtask 35) carrying the rendered prose,
      parsed structure, and a single Touch entry annotating the
      outline-tier provenance of the in-process LLM call.

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
        super().__init__(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            capture=capture,
            max_tokens=max_tokens,
            temperature=temperature,
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            supported_providers=SUPPORTED_PROVIDERS,
            system_prompt=_SYSTEM_PROMPT,
        )

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

        self._emit_per_call_decision(
            raw_text=text,
            retry_count=retry_count,
            course_code=course_code,
            week_number=week_number,
            page_id=page_id,
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

    # ------------------------------------------------------------------
    # Decision capture (page-authoring specific)
    # ------------------------------------------------------------------

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Emit one ``content_generator_call`` decision-capture event
        per :meth:`generate_page` invocation.

        Rationale interpolates ``page_id``, ``course_code``,
        ``week_number``, ``provider``, ``model``, ``base_url`` (when
        present), output character count, and retry count per the
        project's LLM call-site instrumentation contract (≥20 chars,
        dynamic signals interpolated). Delegates to
        :meth:`_BaseLLMProvider._emit_decision` for the swallow-on-error
        capture-emit semantics.
        """
        course_code = call_context.get("course_code", "")
        week_number = call_context.get("week_number", 0)
        page_id = call_context.get("page_id", "")
        decision = (
            f"Courseforge content-generator authored page "
            f"page_id={page_id} for course_code={course_code}, "
            f"week_number={week_number} via "
            f"provider={self._provider}, model={self._model}, "
            f"retry_count={retry_count}."
        )
        rationale = (
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
        )
        self._emit_decision(
            decision_type="content_generator_call",
            decision=decision,
            rationale=rationale,
        )


__all__ = [
    "ContentGeneratorProvider",
    "ENV_PROVIDER",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "SynthesisProviderError",
]
