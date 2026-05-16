#!/usr/bin/env python3
"""Courseforge generators — three-stage textbook-synthesis provider.

Plan reference: ``plans/textbook-llm-synthesis-3stage-2026-05.md``.

The three-stage textbook-synthesis architecture maps a large LLM onto
the three post-DART phases of ``config/workflows.yaml::textbook_to_course``:

- **Stage 1** (``objective_extraction``): whole-textbook skeleton →
  course-level **semantic outline** + **DRAFT TO-NN** terminal
  objectives. One call (or one per chaptered split).
- **Stage 3** (``concept_extraction``): per-chapter chapter TEXT →
  **domain-concept vocabulary**. N per-chapter calls, batched.
- **Stage 2** (``course_planning``): per-chapter chapter TEXT → that
  chapter's **CO-NN objectives + sub-objectives**. N per-chapter calls,
  batched.
- **Reconciliation** (``course_planning`` post-Stage-2): adjust the
  Stage-1 draft TO-NN in light of the synthesized per-chapter CO-NN.
  One call.

Per plan §2.1 this is exactly ONE provider class exposing FOUR public
methods — NOT four provider classes. The W-D12 registry-dynamic
``__init__`` / ``_dispatch_call`` boilerplate is written once. The two
``_BaseLLMProvider`` abstract methods (``_render_user_prompt`` /
``_emit_per_call_decision``) are thin dispatchers keyed on a ``mode``
kwarg (``"outline" | "concepts" | "chapter_objectives" | "reconcile"``).

DESIGN INVARIANT (W-D12 contract): provider references stay dynamic.
The class consults the ``_OPENAI_COMPATIBLE_PROVIDERS`` registry at
``MCP/orchestrator/llm_backend.py`` for its supported providers, and
routes every non-Anthropic registry call through
:func:`MCP.orchestrator.llm_backend.resolve_openai_compatible_backend`.
Adding a new OpenAI-compatible provider is a registry-entry change,
NOT a new subclass.

Env vars:

- :data:`ENV_PROVIDER` — ``TEXTBOOK_SYNTHESIS_PROVIDER``. Selects the
  backend. Workflow default is "no in-process provider" — each phase
  handler checks this env var for a non-empty value before
  constructing the provider.
- :data:`ENV_MODEL` — ``TEXTBOOK_SYNTHESIS_MODEL``. Model-ID override.

Decision-capture: every dispatch emits exactly one typed decision
event whose ``decision_type`` is one of ``textbook_outline_call`` /
``textbook_concept_call`` / ``chapter_objective_call`` /
``terminal_objective_reconciliation`` (the four canonical enum members
added to ``schemas/events/decision_event.schema.json``). The
``provider`` field interpolates the runtime-resolved value, never a
static label, per the W-D12 dynamic-rationale contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from Courseforge.generators._base import (
    _BaseLLMProvider,
)
from MCP.orchestrator.llm_backend import (
    _OPENAI_COMPATIBLE_PROVIDERS,
    resolve_openai_compatible_backend,
)
from Trainforge.generators._openai_compatible_client import (
    SynthesisProviderError,
)
from lib.ontology.learning_objectives import mint_lo_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — env vars + defaults
# ---------------------------------------------------------------------------

ENV_PROVIDER = "TEXTBOOK_SYNTHESIS_PROVIDER"
ENV_MODEL = "TEXTBOOK_SYNTHESIS_MODEL"
# Per-call HTTP timeout (seconds) for the OpenAI-compatible backends.
# Synthesis calls are long-context (24-40k-char inputs, multi-k-token
# output) — a live needle-probe against Ollama 0.21.2 measured 7B
# calls already exceeding the OpenAICompatibleClient 60s default, and
# 14B + 4096-token output runs ~120-300s. 300s is the default; override
# via ``TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS`` or the ``timeout`` kwarg.
ENV_TIMEOUT = "TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS"

# DEFAULT_PROVIDER preserves legacy class-body behavior (parity with
# OutlinerProvider). The WORKFLOW default is "no in-process provider
# unless TEXTBOOK_SYNTHESIS_PROVIDER is explicitly set", enforced by
# each phase handler's short-circuit.
DEFAULT_PROVIDER = "anthropic"

_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.2

# Default per-call HTTP timeout (seconds) for OpenAI-compatible
# backends. Resolution chain: ``timeout`` kwarg > ``ENV_TIMEOUT`` env
# var > this default.
_DEFAULT_TIMEOUT_SECONDS = 300.0

# Defense-in-depth context-window pin for the ``local`` backend only.
# Ollama 0.21.2 auto-sizes context (a 40k-char input was NOT truncated
# at default), but pinning ``num_ctx`` makes the wire payload
# deterministic across Ollama versions/configs. qwen2.5 supports a
# 32768-token context; the live server accepts this with no error.
_LOCAL_NUM_CTX = 32768

# Stage-1 single-call rendered-skeleton char budget. When the rendered
# whole-textbook skeleton exceeds this, ``synthesize_outline`` splits
# into per-chapter-group calls (``call_mode="chaptered"``).
_SKELETON_CHAR_BUDGET = 40_000

# Per-chapter chapter-text char budget for Stages 2 / 3. A single
# OpenStax chapter's prose is typically 10-20 KB; the cap protects
# against an unusually long chapter. Over-budget chapters truncate the
# tail and the decision event records ``chapter_text_truncated=true``.
_CHAPTER_TEXT_BUDGET = 24_000

# Per-chapter batching: Stages 2 / 3 dispatch N per-chapter calls in
# groups of at most this many, honoring the orchestrator's
# "max 10 simultaneous" protocol (root ``CLAUDE.md`` § Phase 3).
_CHAPTER_BATCH_SIZE = 10


_BLOOM_LEVEL_ENUM: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


_TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a Courseforge textbook-synthesis author. You read a "
    "textbook's structure and prose and synthesize course design "
    "artifacts — a semantic outline, terminal/chapter learning "
    "objectives, and a domain-concept vocabulary. Emit ONLY a JSON "
    "object — no preamble, no markdown fences, no commentary. Every "
    "objective follows the ABCD framework (Audience, Behavior, "
    "Condition, Degree); behavior verbs MUST come from the canonical "
    "Bloom's taxonomy (remember/understand/apply/analyze/evaluate/"
    "create). Ground every artifact in the supplied textbook content; "
    "do NOT invent chapters, concepts, or facts not present."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TextbookSynthesisProviderError(RuntimeError):
    """Textbook-synthesis dispatch / parse / validation failure.

    Carries an opaque ``code`` field (plan §2.2) so callers can branch
    on the failure mode without parsing the message string.

    Canonical codes:

    - ``outline_exhausted`` — Stage-1 outline call exhausted its
      parse-retry budget.
    - ``concepts_exhausted`` — a Stage-3 per-chapter concept call
      exhausted its parse-retry budget.
    - ``chapter_objectives_exhausted`` — a Stage-2 per-chapter
      objective call exhausted its parse-retry budget.
    - ``reconcile_exhausted`` — the reconciliation call exhausted its
      parse-retry budget.
    """

    _CODES = frozenset({
        "outline_exhausted",
        "concepts_exhausted",
        "chapter_objectives_exhausted",
        "reconcile_exhausted",
    })

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


def _build_supported_providers() -> Tuple[str, ...]:
    """Compose the supported-providers tuple at call time.

    Read at construction time so a test that monkeypatches
    ``_OPENAI_COMPATIBLE_PROVIDERS`` sees the patched view; not cached
    at import. ``anthropic`` is added explicitly — it is NOT an
    OpenAI-compatible registry entry (it has its own native SDK path).
    """
    return ("anthropic",) + tuple(_OPENAI_COMPATIBLE_PROVIDERS.keys())


class TextbookSynthesisProvider(_BaseLLMProvider):
    """LLM-agnostic three-stage textbook-synthesis provider.

    Subclasses :class:`Courseforge.generators._base._BaseLLMProvider`
    for the shared decision-capture + Anthropic plumbing, then routes
    every non-Anthropic call through the W-D12
    :func:`resolve_openai_compatible_backend` factory so the runtime
    backend is dynamically built from the
    ``_OPENAI_COMPATIBLE_PROVIDERS`` registry.

    Public methods:

    - :meth:`synthesize_outline` — Stage 1.
    - :meth:`synthesize_concepts` — Stage 3, per-chapter.
    - :meth:`synthesize_chapter_objectives` — Stage 2, per-chapter.
    - :meth:`reconcile_terminal_objectives` — reconciliation.
    """

    # ------------------------------------------------------------------
    # Construction — W-D12 registry-dynamic resolution
    # ------------------------------------------------------------------

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
        # Per-call HTTP timeout (seconds). Resolution chain (high →
        # low): this kwarg > ``TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS``
        # env var > :data:`_DEFAULT_TIMEOUT_SECONDS` (300.0).
        timeout: Optional[float] = None,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        # Test seam: inject a pre-built W-D12 backend so unit tests
        # don't have to spin up an HTTP server.
        openai_compatible_backend: Optional[Any] = None,
    ) -> None:
        resolved_model = (
            model
            or os.environ.get(ENV_MODEL)
            or None  # let the base resolve from the synthesis env vars
        )
        supported = _build_supported_providers()
        resolved_provider = (
            provider
            or os.environ.get(ENV_PROVIDER)
            or DEFAULT_PROVIDER
        ).lower()
        if resolved_provider not in supported:
            raise ValueError(
                f"TextbookSynthesisProvider: unknown provider "
                f"{resolved_provider!r}; expected one of "
                f"{list(supported)}. Add a registry entry to "
                f"_OPENAI_COMPATIBLE_PROVIDERS in "
                f"MCP/orchestrator/llm_backend.py — no subclassing "
                f"required."
            )
        # Resolve the per-call HTTP timeout: kwarg > env > 300.0.
        resolved_timeout: float = _DEFAULT_TIMEOUT_SECONDS
        if timeout is not None:
            resolved_timeout = float(timeout)
        else:
            env_timeout = os.environ.get(ENV_TIMEOUT)
            if env_timeout is not None and str(env_timeout).strip():
                try:
                    resolved_timeout = float(env_timeout)
                except (TypeError, ValueError):
                    logger.warning(
                        "TextbookSynthesisProvider: ignoring "
                        "non-numeric %s=%r; using default %.1fs",
                        ENV_TIMEOUT, env_timeout, _DEFAULT_TIMEOUT_SECONDS,
                    )

        self._registry_provider: Optional[str] = None
        if (
            resolved_provider not in ("anthropic", "together", "local")
            and resolved_provider in _OPENAI_COMPATIBLE_PROVIDERS
        ):
            # Stash the real provider for the W-D12 dispatch override;
            # let the base init wire its plumbing under "local" so we
            # inherit the OpenAICompatibleClient + decision-capture seat.
            self._registry_provider = resolved_provider
            base_provider_arg = "local"
        else:
            base_provider_arg = resolved_provider

        super().__init__(
            provider=base_provider_arg,
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            capture=capture,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=resolved_timeout,
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            supported_providers=("anthropic", "together", "local"),
            system_prompt=_TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT,
        )
        # Restore the runtime provider label so decision-capture and
        # logging surface the registry name (e.g. "groq").
        if self._registry_provider is not None:
            self._provider = self._registry_provider

        self._injected_oa_backend = openai_compatible_backend

    # ------------------------------------------------------------------
    # Dispatch override — route OpenAI-compatible providers through W-D12
    # ------------------------------------------------------------------

    def _dispatch_call(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Route through the W-D12 backend for OpenAI-compatible
        providers; the base handles anthropic / together / local.

        M2 contract: ``OpenAICompatibleClient`` raises
        :class:`SynthesisProviderError` on a non-retryable 4xx, on
        max-retries exhaustion, and on a malformed / model-not-found
        response. The textbook-synthesis surface contract is that the
        four public methods raise only
        :class:`TextbookSynthesisProviderError` so the per-chapter
        fail-soft / Stage-1 fail-loud handlers in
        ``MCP/tools/pipeline_tools.py`` can branch on the typed ``code``.
        We therefore catch ``SynthesisProviderError`` at this provider
        boundary and re-raise it as ``TextbookSynthesisProviderError``.
        The caller's stage method has not yet emitted its per-call
        decision when the dispatch raises, so each public method's
        ``except TextbookSynthesisProviderError`` boundary owns
        re-classification into the stage-specific ``code``.
        """
        try:
            if self._provider == "anthropic":
                return self._call_anthropic(user_prompt)
            if self._provider in ("together", "local"):
                merged_extra = self._merge_local_num_ctx(extra_payload)
                return super()._dispatch_call(
                    user_prompt, extra_payload=merged_extra
                )
            backend = self._injected_oa_backend
            if backend is None:
                backend = resolve_openai_compatible_backend(
                    self._provider,
                    model_override=self._model,
                    api_key_override=self._api_key,
                    capture=None,  # capture fires at the provider seat
                )
            text = backend.complete_sync(
                self._system_prompt,
                user_prompt,
                model=self._model,
                max_tokens=int(self._max_tokens),
                temperature=float(self._temperature),
            )
            return text or "", 0
        except SynthesisProviderError as exc:
            # Re-raise at the provider boundary so the contract holds:
            # synthesis methods raise only TextbookSynthesisProviderError.
            # ``code`` carries the backend's opaque failure mode prefixed
            # with ``dispatch_`` so a post-hoc audit can tell a transport
            # failure apart from a parse-retry exhaustion.
            backend_code = getattr(exc, "code", None) or "unknown"
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider dispatch failed "
                f"(provider={self._provider}, model={self._model}, "
                f"backend_code={backend_code}): {exc}",
                code=f"dispatch_{backend_code}",
            ) from exc

    def _merge_local_num_ctx(
        self,
        extra_payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge the ``num_ctx`` context-window pin into the wire
        payload for the ``local`` backend only (Fix 5, defense-in-depth).

        Ollama reads a top-level ``options`` object; ``num_ctx`` pins
        the context window so the payload is deterministic across
        Ollama versions/configs regardless of the server's auto-sizing
        heuristics. Caller-supplied ``extra_payload`` (and any
        caller-supplied ``options`` sub-keys) take precedence — we only
        ``setdefault`` ``num_ctx``. ``together`` / ``anthropic`` are
        untouched: only the ``local`` backend speaks the Ollama
        ``options`` shape.
        """
        if self._provider != "local":
            return extra_payload
        merged: Dict[str, Any] = dict(extra_payload or {})
        options = dict(merged.get("options") or {})
        options.setdefault("num_ctx", _LOCAL_NUM_CTX)
        merged["options"] = options
        return merged

    # ==================================================================
    # Public API — Stage 1: outline synthesis
    # ==================================================================

    def synthesize_outline(
        self,
        textbook_structure: Dict[str, Any],
        *,
        course_name: str,
    ) -> Dict[str, Any]:
        """Stage 1 — synthesize the course-level semantic outline +
        DRAFT TO-NN terminal objectives.

        Args:
            textbook_structure: Parsed ``textbook_structure.json`` —
                at minimum ``{"chapters": [{"id", "title"/"headingText",
                "sections": [...], "chapter_text": str}]}``.
            course_name: Course slug for prompt header + capture
                rationale.

        Returns:
            ``{
                "semantic_outline": {"course_summary": str,
                    "themes": [{"title", "chapter_ids", "summary",
                        "prerequisite_theme_titles"}]},
                "draft_terminal_objectives": [{"id": "TO-NN",
                    "statement", "bloom_level", "bloom_verb", "abcd",
                    "source_refs", "draft": True}],
                "structure_enrichment": {"enriched": True, "provider":
                    str, "model": str, "call_mode": "single"|"chaptered",
                    "calls": int, "schema_version": "v1"},
              }``

        Raises:
            ValueError: When ``course_name`` is empty.
            TextbookSynthesisProviderError(code="outline_exhausted"):
                When the LLM tier fails to emit usable JSON across the
                parse-retry budget. Stage 1 is a single call → fail-loud
                (plan §2.2).
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")
        if not isinstance(textbook_structure, dict):
            raise ValueError("textbook_structure must be a dict")

        chapters = textbook_structure.get("chapters") or []
        chapter_groups = self._chunk_chapters_for_outline(chapters)
        call_mode = "single" if len(chapter_groups) <= 1 else "chaptered"

        merged_themes: List[Dict[str, Any]] = []
        merged_draft_tos: List[Dict[str, Any]] = []
        course_summary = ""
        calls = 0
        last_error: Optional[str] = None

        for group in chapter_groups:
            user_prompt = self._render_user_prompt(
                mode="outline",
                chapters=group,
                course_name=course_name,
            )
            text, retry_count = self._dispatch_call(user_prompt)
            parsed = self._extract_json_lenient(text)
            calls += 1

            themes_emit = 0
            draft_to_emit = 0
            normalised: Optional[Dict[str, Any]] = None
            if parsed is None:
                last_error = "lenient JSON parse returned None"
            else:
                normalised = self._normalise_outline_payload(parsed)
                if not course_summary:
                    course_summary = normalised["course_summary"]
                merged_themes.extend(normalised["themes"])
                merged_draft_tos.extend(normalised["draft_terminal_objectives"])
                themes_emit = len(normalised["themes"])
                draft_to_emit = len(normalised["draft_terminal_objectives"])

            self._emit_per_call_decision(
                mode="outline",
                raw_text=text,
                retry_count=retry_count,
                course_name=course_name,
                chapter_count_input=len(group),
                call_mode=call_mode,
                theme_count_emit=themes_emit,
                draft_to_count_emit=draft_to_emit,
                success=normalised is not None,
                last_error=last_error if normalised is None else None,
            )

        if not merged_themes and not merged_draft_tos:
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider.synthesize_outline exhausted "
                f"parse on course {course_name!r} "
                f"(last_error={last_error!r})",
                code="outline_exhausted",
            )

        # Re-mint draft TO IDs sequentially across the merged set.
        for idx, draft in enumerate(merged_draft_tos, start=1):
            draft["id"] = mint_lo_id("terminal", idx)
            draft["draft"] = True

        return {
            "semantic_outline": {
                "course_summary": course_summary,
                "themes": merged_themes,
            },
            "draft_terminal_objectives": merged_draft_tos,
            "structure_enrichment": {
                "enriched": True,
                "provider": self._provider,
                "model": self._model,
                "call_mode": call_mode,
                "calls": calls,
                "schema_version": "v1",
            },
        }

    # ==================================================================
    # Public API — Stage 3: per-chapter concept synthesis
    # ==================================================================

    def synthesize_concepts(
        self,
        chapter: Dict[str, Any],
        *,
        course_name: str,
    ) -> Dict[str, Any]:
        """Stage 3 — synthesize one chapter's domain-concept vocabulary.

        Args:
            chapter: One chapter dict from
                ``textbook_structure.json::chapters[]``, carrying at
                minimum ``id`` + ``chapter_text``.
            course_name: Course slug for the capture rationale.

        Returns:
            ``{
                "chapter_id": str,
                "concepts": [{"canonical": str, "aliases": [str],
                    "definition_hint": str, "chapter_ids": [str]}],
              }``

        Raises:
            ValueError: When ``course_name`` is empty.
            TextbookSynthesisProviderError(code="concepts_exhausted"):
                When the per-chapter LLM call exhausts the parse-retry
                budget. The caller (plan §5.4) catches this at the
                batch level and continues with the other chapters.
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")
        if not isinstance(chapter, dict):
            raise ValueError("chapter must be a dict")

        chapter_id = str(chapter.get("id") or "")
        chapter_text, truncated = self._chapter_text_bounded(chapter)

        user_prompt = self._render_user_prompt(
            mode="concepts",
            chapter=chapter,
            chapter_id=chapter_id,
            chapter_text=chapter_text,
            course_name=course_name,
        )
        text, retry_count = self._dispatch_call(user_prompt)
        parsed = self._extract_json_lenient(text)

        concepts: Optional[List[Dict[str, Any]]] = None
        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"
        else:
            concepts = self._normalise_concepts_payload(parsed, chapter_id)

        self._emit_per_call_decision(
            mode="concepts",
            raw_text=text,
            retry_count=retry_count,
            course_name=course_name,
            chapter_id=chapter_id,
            chapter_text_chars=len(chapter_text),
            concept_count_emit=len(concepts) if concepts is not None else 0,
            chapter_text_truncated=truncated,
            success=concepts is not None,
            last_error=last_error,
        )

        if concepts is None:
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider.synthesize_concepts exhausted "
                f"parse on chapter {chapter_id!r} of course "
                f"{course_name!r} (last_error={last_error!r})",
                code="concepts_exhausted",
            )
        return {"chapter_id": chapter_id, "concepts": concepts}

    # ==================================================================
    # Public API — Stage 2: per-chapter objective synthesis
    # ==================================================================

    def synthesize_chapter_objectives(
        self,
        chapter: Dict[str, Any],
        *,
        course_name: str,
        draft_terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 2 — synthesize one chapter's CO-NN objectives.

        Args:
            chapter: One chapter dict from
                ``textbook_structure.json::chapters[]``, carrying at
                minimum ``id`` + ``chapter_text``.
            course_name: Course slug for the capture rationale.
            draft_terminal_objectives: The Stage-1 draft TO-NN list,
                threaded so the model can ground each CO against the
                course's terminal goals. Optional.

        Returns:
            ``{
                "chapter_id": str,
                "chapter_objectives": [{"statement", "bloom_level",
                    "bloom_verb", "abcd", "source_refs",
                    "sub_objectives": [str], "chapter_id": str}],
              }``

            CO IDs are NOT minted here — the caller flattens every
            chapter's list course-ordered and mints globally sequential
            ``CO-01..CO-NN`` ids (plan §5.2).

        Raises:
            ValueError: When ``course_name`` is empty.
            TextbookSynthesisProviderError(
                code="chapter_objectives_exhausted"): When the
                per-chapter LLM call exhausts the parse-retry budget.
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")
        if not isinstance(chapter, dict):
            raise ValueError("chapter must be a dict")

        chapter_id = str(chapter.get("id") or "")
        chapter_text, truncated = self._chapter_text_bounded(chapter)

        user_prompt = self._render_user_prompt(
            mode="chapter_objectives",
            chapter=chapter,
            chapter_id=chapter_id,
            chapter_text=chapter_text,
            course_name=course_name,
            draft_terminal_objectives=draft_terminal_objectives or [],
        )
        text, retry_count = self._dispatch_call(user_prompt)
        parsed = self._extract_json_lenient(text)

        objectives: Optional[List[Dict[str, Any]]] = None
        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"
        else:
            objectives = self._normalise_chapter_objectives_payload(
                parsed, chapter_id
            )

        sub_obj_total = 0
        if objectives is not None:
            for obj in objectives:
                sub_obj_total += len(obj.get("sub_objectives") or [])

        self._emit_per_call_decision(
            mode="chapter_objectives",
            raw_text=text,
            retry_count=retry_count,
            course_name=course_name,
            chapter_id=chapter_id,
            chapter_text_chars=len(chapter_text),
            co_count_emit=len(objectives) if objectives is not None else 0,
            sub_objective_count_emit=sub_obj_total,
            chapter_text_truncated=truncated,
            success=objectives is not None,
            last_error=last_error,
        )

        if objectives is None:
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider.synthesize_chapter_objectives "
                f"exhausted parse on chapter {chapter_id!r} of course "
                f"{course_name!r} (last_error={last_error!r})",
                code="chapter_objectives_exhausted",
            )
        return {
            "chapter_id": chapter_id,
            "chapter_objectives": objectives,
        }

    # ==================================================================
    # Public API — reconciliation
    # ==================================================================

    def reconcile_terminal_objectives(
        self,
        draft_terminal_objectives: List[Dict[str, Any]],
        chapter_objectives: List[Dict[str, Any]],
        *,
        course_name: str,
    ) -> Dict[str, Any]:
        """Reconcile Stage-1 draft TO-NN against Stage-2 chapter CO-NN.

        Plan §6: "Stage 1 drafts, Stage 2 informs." The draft terminal
        objectives are kept / dropped / merged / re-worded so they
        accurately summarise the union of the synthesized CO-NN.

        Args:
            draft_terminal_objectives: Stage-1 ``draft_terminal_objectives``.
            chapter_objectives: The flattened Stage-2 per-chapter CO-NN
                list (statements consumed for the prompt).
            course_name: Course slug for the capture rationale.

        Returns:
            ``{"terminal_objectives": [{"id": "TO-NN", "statement",
                "bloom_level", "bloom_verb", "abcd", "source_refs"}]}``
            — TO IDs re-minted sequentially, ``draft`` flag dropped.

        Raises:
            ValueError: When ``course_name`` is empty.
            TextbookSynthesisProviderError(code="reconcile_exhausted"):
                When the reconciliation call exhausts the parse-retry
                budget. Reconciliation is one call → fail-loud (§6).
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")

        draft_tos = list(draft_terminal_objectives or [])
        chapter_cos = list(chapter_objectives or [])

        user_prompt = self._render_user_prompt(
            mode="reconcile",
            draft_terminal_objectives=draft_tos,
            chapter_objectives=chapter_cos,
            course_name=course_name,
        )
        text, retry_count = self._dispatch_call(user_prompt)
        parsed = self._extract_json_lenient(text)

        terminals: Optional[List[Dict[str, Any]]] = None
        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"
        else:
            terminals = self._normalise_reconcile_payload(parsed)

        self._emit_per_call_decision(
            mode="reconcile",
            raw_text=text,
            retry_count=retry_count,
            course_name=course_name,
            draft_to_count_input=len(draft_tos),
            co_count_input=len(chapter_cos),
            reconciled_to_count_emit=(
                len(terminals) if terminals is not None else 0
            ),
            success=terminals is not None,
            last_error=last_error,
        )

        if terminals is None:
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider.reconcile_terminal_objectives "
                f"exhausted parse on course {course_name!r} "
                f"(last_error={last_error!r})",
                code="reconcile_exhausted",
            )
        return {"terminal_objectives": terminals}

    # ==================================================================
    # Per-chapter batching helper (plan §5.3)
    # ==================================================================

    @staticmethod
    def batch_chapters(
        chapters: List[Any],
        batch_size: int = _CHAPTER_BATCH_SIZE,
    ) -> List[List[Any]]:
        """Split ``chapters`` into ordered batches of at most
        ``batch_size`` (default :data:`_CHAPTER_BATCH_SIZE` = 10).

        Stages 2 and 3 dispatch N per-chapter calls; the orchestrator's
        "max 10 simultaneous" protocol (root ``CLAUDE.md`` § Phase 3)
        applies at the handler level. The phase handler iterates the
        batches this helper returns, awaiting each batch fully before
        starting the next.
        """
        size = max(1, int(batch_size))
        return [
            list(chapters[i:i + size])
            for i in range(0, len(chapters), size)
        ]

    # ------------------------------------------------------------------
    # Internal — chapter-text bounding + outline chaptering
    # ------------------------------------------------------------------

    @staticmethod
    def _chapter_text_bounded(chapter: Dict[str, Any]) -> Tuple[str, bool]:
        """Return ``(bounded_chapter_text, truncated)``.

        Reads ``chapter_text`` (the new SemanticStructureExtractor
        field, plan §1); falls back to empty string when absent. Caps
        the tail at :data:`_CHAPTER_TEXT_BUDGET`.
        """
        raw = str(chapter.get("chapter_text") or "")
        if len(raw) <= _CHAPTER_TEXT_BUDGET:
            return raw, False
        return raw[:_CHAPTER_TEXT_BUDGET], True

    def _chunk_chapters_for_outline(
        self,
        chapters: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Split chapters into outline-call groups under
        :data:`_SKELETON_CHAR_BUDGET`.

        One group → ``call_mode="single"``. When the rendered skeleton
        of all chapters would exceed the budget (thick multi-volume
        textbook), splits into per-chapter-group calls
        (``call_mode="chaptered"``, plan §3.3).
        """
        if not chapters:
            return [[]]
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0
        for chapter in chapters:
            rendered = len(self._render_chapter_skeleton(chapter))
            if current and current_chars + rendered > _SKELETON_CHAR_BUDGET:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(chapter)
            current_chars += rendered
        if current:
            groups.append(current)
        return groups or [[]]

    @staticmethod
    def _render_chapter_skeleton(chapter: Dict[str, Any]) -> str:
        """Render one chapter's skeleton line for the Stage-1 prompt.

        Per plan §3.3: id + headingText/title + ordered section list +
        a bounded ~200-char sample drawn from ``chapter_text``. Reads
        both ``headingText`` (extractor camelCase) and ``title``
        defensively.
        """
        cid = str(chapter.get("id") or "")
        title = str(
            chapter.get("title")
            or chapter.get("headingText")
            or ""
        )
        sections = chapter.get("sections") or []
        section_titles: List[str] = []
        for sec in sections:
            stitle = str(
                sec.get("title")
                or sec.get("headingText")
                or sec.get("heading")
                or ""
            )
            if stitle:
                section_titles.append(stitle)
        sample = str(chapter.get("chapter_text") or "")[:200]
        lines = [f"  - [{cid}] {title}"]
        if section_titles:
            lines.append(
                f"    sections: {', '.join(section_titles[:12])}"
            )
        if sample:
            lines.append(f"    sample: {sample}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Abstract-method dispatchers (mode-keyed, plan §2.1)
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *args: Any,
        mode: str = "outline",
        **kwargs: Any,
    ) -> str:
        """Thin dispatcher for the four task-specific prompt builders.

        ``_BaseLLMProvider`` declares ``_render_user_prompt`` as a
        single abstract method; this subclass implements it as a
        ``mode``-keyed dispatcher delegating to four private builders.
        """
        if mode == "outline":
            return self._render_outline_prompt(**kwargs)
        if mode == "concepts":
            return self._render_concepts_prompt(**kwargs)
        if mode == "chapter_objectives":
            return self._render_chapter_objectives_prompt(**kwargs)
        if mode == "reconcile":
            return self._render_reconcile_prompt(**kwargs)
        raise ValueError(f"unknown _render_user_prompt mode {mode!r}")

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str = "",
        retry_count: int = 0,
        mode: str = "outline",
        **call_context: Any,
    ) -> None:
        """Thin dispatcher for the four task-specific decision emitters.

        ``_BaseLLMProvider`` declares ``_emit_per_call_decision`` as a
        single abstract method; this subclass implements it as a
        ``mode``-keyed dispatcher so each stage emits its canonical
        ``decision_type`` event.
        """
        if mode == "outline":
            self._emit_outline_decision(
                raw_text=raw_text, retry_count=retry_count, **call_context
            )
        elif mode == "concepts":
            self._emit_concepts_decision(
                raw_text=raw_text, retry_count=retry_count, **call_context
            )
        elif mode == "chapter_objectives":
            self._emit_chapter_objectives_decision(
                raw_text=raw_text, retry_count=retry_count, **call_context
            )
        elif mode == "reconcile":
            self._emit_reconcile_decision(
                raw_text=raw_text, retry_count=retry_count, **call_context
            )
        else:  # pragma: no cover — defensive
            raise ValueError(
                f"unknown _emit_per_call_decision mode {mode!r}"
            )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _render_outline_prompt(
        self,
        *,
        chapters: List[Dict[str, Any]],
        course_name: str,
        **_unused: Any,
    ) -> str:
        """Stage-1 outline prompt: whole-(group) skeleton + samples."""
        skeleton = "".join(
            self._render_chapter_skeleton(c) for c in chapters
        ) or "  (none)"
        return (
            f"Course: {course_name}\n"
            f"Chapter count: {len(chapters)}\n\n"
            "Textbook chapters (id, title, sections, prose sample):\n"
            f"{skeleton}\n"
            "Synthesize a course-level semantic outline and 1-4 DRAFT "
            "terminal objectives.\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level keys:\n"
            '  "course_summary": one-paragraph synthesized description,\n'
            '  "themes": [{"title", "chapter_ids" (verbatim chapter '
            'ids), "summary", "prerequisite_theme_titles" []}],\n'
            '  "draft_terminal_objectives": [{"statement", '
            '"bloom_level" (lowercase enum), "bloom_verb", "abcd" '
            '{audience, behavior {verb, action_object}, condition, '
            'degree}, "source_refs" [{"ref": chapter_id, '
            '"chunk_ids": []}]}].\n'
            "No preamble, no markdown fences, no commentary."
        )

    def _render_concepts_prompt(
        self,
        *,
        chapter_id: str,
        chapter_text: str,
        course_name: str,
        **_unused: Any,
    ) -> str:
        """Stage-3 per-chapter domain-concept prompt."""
        return (
            f"Course: {course_name}\n"
            f"Chapter id: {chapter_id}\n\n"
            "Chapter text:\n"
            f"{chapter_text or '(no chapter text supplied)'}\n\n"
            "Extract the domain concepts this chapter teaches — the "
            "real subject-matter terms (e.g. 'prime factorization', "
            "'slope-intercept form'), NOT instructional-design "
            "vocabulary.\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level key "
            '"concepts": [{"canonical": the canonical term, '
            '"aliases": [synonym strings], "definition_hint": a short '
            'gloss, "chapter_ids": ["' + chapter_id + '"]}].\n'
            "No preamble, no markdown fences, no commentary."
        )

    def _render_chapter_objectives_prompt(
        self,
        *,
        chapter_id: str,
        chapter_text: str,
        course_name: str,
        draft_terminal_objectives: List[Dict[str, Any]],
        **_unused: Any,
    ) -> str:
        """Stage-2 per-chapter CO-NN objective prompt."""
        draft_lines: List[str] = []
        for draft in draft_terminal_objectives or []:
            if isinstance(draft, dict):
                stmt = str(draft.get("statement") or "").strip()
                if stmt:
                    draft_lines.append(f"  - {stmt}")
        draft_block = (
            "\n".join(draft_lines) if draft_lines else "  (none)"
        )
        return (
            f"Course: {course_name}\n"
            f"Chapter id: {chapter_id}\n\n"
            "Course draft terminal objectives (for grounding):\n"
            f"{draft_block}\n\n"
            "Chapter text:\n"
            f"{chapter_text or '(no chapter text supplied)'}\n\n"
            "Synthesize this chapter's learning objectives.\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level key "
            '"chapter_objectives": [{"statement", "bloom_level" '
            '(lowercase enum), "bloom_verb", "abcd" {audience, '
            'behavior {verb, action_object}, condition, degree}, '
            '"source_refs" [{"ref": "' + chapter_id + '", '
            '"chunk_ids": []}], "sub_objectives": [short strings]}].\n'
            "No preamble, no markdown fences, no commentary."
        )

    def _render_reconcile_prompt(
        self,
        *,
        draft_terminal_objectives: List[Dict[str, Any]],
        chapter_objectives: List[Dict[str, Any]],
        course_name: str,
        **_unused: Any,
    ) -> str:
        """Reconciliation prompt: draft TO-NN vs synthesized CO-NN."""
        draft_lines: List[str] = []
        for draft in draft_terminal_objectives or []:
            if isinstance(draft, dict):
                stmt = str(draft.get("statement") or "").strip()
                if stmt:
                    draft_lines.append(f"  - {stmt}")
        co_lines: List[str] = []
        for co in chapter_objectives or []:
            if isinstance(co, dict):
                stmt = str(co.get("statement") or "").strip()
                if stmt:
                    co_lines.append(f"  - {stmt}")
        return (
            f"Course: {course_name}\n\n"
            "Draft terminal objectives (course-wide):\n"
            f"{chr(10).join(draft_lines) if draft_lines else '  (none)'}\n\n"
            "Synthesized chapter objectives:\n"
            f"{chr(10).join(co_lines) if co_lines else '  (none)'}\n\n"
            "Keep, drop, merge, or re-word each draft terminal "
            "objective so it accurately summarises the union of the "
            "chapter objectives. Do NOT invent terminal objectives "
            "unsupported by any chapter objective.\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level key "
            '"terminal_objectives": [{"statement", "bloom_level" '
            '(lowercase enum), "bloom_verb", "abcd" {audience, '
            'behavior {verb, action_object}, condition, degree}, '
            '"source_refs" []}].\n'
            "No preamble, no markdown fences, no commentary."
        )

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    def _normalise_outline_payload(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Project a Stage-1 LLM payload onto the canonical outline
        shape. Draft TO IDs are NOT minted here — the caller mints
        sequentially across the merged set."""
        themes: List[Dict[str, Any]] = []
        for raw in parsed.get("themes") or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            chapter_ids = [
                str(c) for c in (raw.get("chapter_ids") or [])
                if isinstance(c, (str, int))
            ]
            prereq = [
                str(p) for p in (
                    raw.get("prerequisite_theme_titles") or []
                )
                if isinstance(p, str)
            ]
            themes.append({
                "title": title,
                "chapter_ids": chapter_ids,
                "summary": str(raw.get("summary") or "").strip(),
                "prerequisite_theme_titles": prereq,
            })

        draft_tos: List[Dict[str, Any]] = []
        for raw in parsed.get("draft_terminal_objectives") or []:
            if not isinstance(raw, dict):
                continue
            obj = self._normalise_one_objective(raw)
            if obj is None:
                continue
            obj["draft"] = True
            draft_tos.append(obj)

        return {
            "course_summary": str(
                parsed.get("course_summary") or ""
            ).strip(),
            "themes": themes,
            "draft_terminal_objectives": draft_tos,
        }

    def _normalise_concepts_payload(
        self,
        parsed: Dict[str, Any],
        chapter_id: str,
    ) -> List[Dict[str, Any]]:
        """Project a Stage-3 LLM payload onto the concept-list shape."""
        concepts: List[Dict[str, Any]] = []
        for raw in parsed.get("concepts") or []:
            if not isinstance(raw, dict):
                continue
            canonical = str(
                raw.get("canonical") or raw.get("term") or ""
            ).strip()
            if not canonical:
                continue
            aliases = [
                str(a).strip() for a in (raw.get("aliases") or [])
                if isinstance(a, str) and str(a).strip()
            ]
            chapter_ids = [
                str(c) for c in (raw.get("chapter_ids") or [])
                if isinstance(c, (str, int))
            ] or ([chapter_id] if chapter_id else [])
            concepts.append({
                "canonical": canonical,
                "aliases": aliases,
                "definition_hint": str(
                    raw.get("definition_hint")
                    or raw.get("definition")
                    or ""
                ).strip(),
                "chapter_ids": chapter_ids,
            })
        return concepts

    def _normalise_chapter_objectives_payload(
        self,
        parsed: Dict[str, Any],
        chapter_id: str,
    ) -> List[Dict[str, Any]]:
        """Project a Stage-2 LLM payload onto the chapter-objective
        list shape. CO IDs are NOT minted here (plan §5.2)."""
        raw_list = parsed.get("chapter_objectives") or []
        # Tolerate the canonical group-of-groups shape too.
        flat: List[Dict[str, Any]] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            if "objectives" in entry and isinstance(
                entry["objectives"], list
            ):
                flat.extend(
                    o for o in entry["objectives"]
                    if isinstance(o, dict)
                )
            else:
                flat.append(entry)

        objectives: List[Dict[str, Any]] = []
        for raw in flat:
            obj = self._normalise_one_objective(raw)
            if obj is None:
                continue
            sub = [
                str(s).strip() for s in (raw.get("sub_objectives") or [])
                if isinstance(s, str) and str(s).strip()
            ]
            obj["sub_objectives"] = sub
            obj["chapter_id"] = chapter_id
            objectives.append(obj)
        return objectives

    def _normalise_reconcile_payload(
        self,
        parsed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Project a reconciliation LLM payload onto the canonical
        terminal-objective list. TO IDs re-minted sequentially; the
        ``draft`` flag is dropped."""
        terminals: List[Dict[str, Any]] = []
        for raw in parsed.get("terminal_objectives") or []:
            if not isinstance(raw, dict):
                continue
            obj = self._normalise_one_objective(raw)
            if obj is None:
                continue
            obj.pop("draft", None)
            terminals.append(obj)
        for idx, obj in enumerate(terminals, start=1):
            obj["id"] = mint_lo_id("terminal", idx)
        return terminals

    @staticmethod
    def _normalise_one_objective(
        raw: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Project one LLM-emitted objective onto the canonical shape.

        Returns ``None`` (rather than raising) when the objective has
        no statement, so the caller can filter — a single malformed
        objective never poisons a whole batch.
        """
        if not isinstance(raw, dict):
            return None
        statement = str(
            raw.get("statement") or raw.get("text") or ""
        ).strip()
        if not statement:
            return None
        entry: Dict[str, Any] = {"statement": statement}

        bloom_level = str(raw.get("bloom_level") or "").strip().lower()
        if bloom_level not in _BLOOM_LEVEL_ENUM:
            bloom_level = "understand"
        entry["bloom_level"] = bloom_level

        bloom_verb = str(raw.get("bloom_verb") or "").strip().lower()
        if bloom_verb:
            entry["bloom_verb"] = bloom_verb

        abcd = raw.get("abcd")
        if isinstance(abcd, dict):
            entry["abcd"] = dict(abcd)
            inner = abcd.get("behavior")
            if isinstance(inner, dict):
                entry["abcd"]["behavior"] = dict(inner)

        source_refs = raw.get("source_refs")
        entry["source_refs"] = (
            list(source_refs) if isinstance(source_refs, list) else []
        )
        return entry

    # ------------------------------------------------------------------
    # Lenient JSON parse (lifted verbatim from OutlinerProvider)
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_trailing_commas(blob: str) -> str:
        """Strip JSON-illegal trailing commas (``,}`` / ``,]``).

        Qwen-14B-q4 commonly drifts a trailing comma before a closing
        brace / bracket. ``json.loads`` rejects it; this regex pass
        deletes a comma followed only by whitespace and a closer.
        """
        return re.sub(r",(\s*[}\]])", r"\1", blob)

    @classmethod
    def _try_json_variants(cls, blob: str) -> Optional[Dict[str, Any]]:
        """Attempt ``json.loads`` on ``blob`` and on a
        trailing-comma-stripped variant. Returns the dict or ``None``.
        """
        for candidate in (blob, cls._strip_trailing_commas(blob)):
            try:
                obj = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
        return None

    @classmethod
    def _extract_json_lenient(cls, text: str) -> Optional[Dict[str, Any]]:
        """Lenient JSON object extractor.

        Handles common LLM response drift modes:

        1. Clean JSON object — ``json.loads`` succeeds verbatim.
        2. Markdown-fenced (```` ```json ... ``` ````) — strip fences
           then ``json.loads``. The fence regex is anchored to the
           FIRST fenced block so a response with two fenced blocks
           doesn't mis-slice across both.
        3. Trailing prose — find the outermost balanced ``{...}`` and
           parse the captured slice.

        Every parse attempt also retries against a
        trailing-comma-stripped variant (``,}`` / ``,]`` — a frequent
        Qwen-14B-q4 drift mode). Returns ``None`` on any parse failure.
        """
        if not text:
            return None
        candidate = text.strip()
        # 1. Clean parse (with trailing-comma fallback).
        obj = cls._try_json_variants(candidate)
        if obj is not None:
            return obj
        # 2. Strip markdown fences. The regex is non-greedy on the
        #    body AND requires the OPENING fence at a line boundary
        #    (start-of-string or after a newline) so a response with
        #    two fenced blocks slices the FIRST block cleanly rather
        #    than spanning from the first ``` to the last ```.
        fence_match = re.search(
            r"(?:^|\n)```(?:json)?[ \t]*\n?(.*?)\n?```",
            candidate,
            re.DOTALL,
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            obj = cls._try_json_variants(inner)
            if obj is not None:
                return obj
        # 3. Outermost balanced braces.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = cls._try_json_variants(candidate[start:end + 1])
            if obj is not None:
                return obj
        return None

    # ------------------------------------------------------------------
    # Decision-capture emitters (one per stage)
    # ------------------------------------------------------------------

    def _emit_outline_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **ctx: Any,
    ) -> None:
        """Emit one ``textbook_outline_call`` decision event."""
        course_name = ctx.get("course_name", "")
        success = bool(ctx.get("success", False))
        parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"chapter_count_input={int(ctx.get('chapter_count_input', 0))}",
            f"call_mode={ctx.get('call_mode', 'single')}",
            f"theme_count_emit={int(ctx.get('theme_count_emit', 0))}",
            f"draft_to_count_emit={int(ctx.get('draft_to_count_emit', 0))}",
            f"output_chars={len(raw_text or '')}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if ctx.get("last_error"):
            parts.append(f"last_error={str(ctx['last_error'])[:120]}")
        self._emit_decision(
            decision_type="textbook_outline_call",
            decision=(
                f"textbook_outline_call:{course_name}:"
                f"{'success' if success else 'failed'}"
            ),
            rationale="; ".join(parts),
        )

    def _emit_concepts_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **ctx: Any,
    ) -> None:
        """Emit one ``textbook_concept_call`` decision event."""
        course_name = ctx.get("course_name", "")
        chapter_id = ctx.get("chapter_id", "")
        success = bool(ctx.get("success", False))
        parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"chapter_id={chapter_id}",
            f"chapter_text_chars={int(ctx.get('chapter_text_chars', 0))}",
            f"concept_count_emit={int(ctx.get('concept_count_emit', 0))}",
            f"chapter_text_truncated="
            f"{bool(ctx.get('chapter_text_truncated', False))}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if ctx.get("last_error"):
            parts.append(f"last_error={str(ctx['last_error'])[:120]}")
        self._emit_decision(
            decision_type="textbook_concept_call",
            decision=(
                f"textbook_concept_call:{chapter_id}:"
                f"{'success' if success else 'failed'}"
            ),
            rationale="; ".join(parts),
        )

    def _emit_chapter_objectives_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **ctx: Any,
    ) -> None:
        """Emit one ``chapter_objective_call`` decision event."""
        course_name = ctx.get("course_name", "")
        chapter_id = ctx.get("chapter_id", "")
        success = bool(ctx.get("success", False))
        parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"chapter_id={chapter_id}",
            f"chapter_text_chars={int(ctx.get('chapter_text_chars', 0))}",
            f"co_count_emit={int(ctx.get('co_count_emit', 0))}",
            f"sub_objective_count_emit="
            f"{int(ctx.get('sub_objective_count_emit', 0))}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if ctx.get("last_error"):
            parts.append(f"last_error={str(ctx['last_error'])[:120]}")
        self._emit_decision(
            decision_type="chapter_objective_call",
            decision=(
                f"chapter_objective_call:{chapter_id}:"
                f"{'success' if success else 'failed'}"
            ),
            rationale="; ".join(parts),
        )

    def _emit_reconcile_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **ctx: Any,
    ) -> None:
        """Emit one ``terminal_objective_reconciliation`` decision
        event."""
        course_name = ctx.get("course_name", "")
        success = bool(ctx.get("success", False))
        parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"draft_to_count_input={int(ctx.get('draft_to_count_input', 0))}",
            f"co_count_input={int(ctx.get('co_count_input', 0))}",
            f"reconciled_to_count_emit="
            f"{int(ctx.get('reconciled_to_count_emit', 0))}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if ctx.get("last_error"):
            parts.append(f"last_error={str(ctx['last_error'])[:120]}")
        self._emit_decision(
            decision_type="terminal_objective_reconciliation",
            decision=(
                f"terminal_objective_reconciliation:{course_name}:"
                f"{'success' if success else 'failed'}"
            ),
            rationale="; ".join(parts),
        )


__all__ = [
    "TextbookSynthesisProvider",
    "TextbookSynthesisProviderError",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "DEFAULT_PROVIDER",
]
