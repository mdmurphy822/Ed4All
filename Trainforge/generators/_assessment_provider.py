#!/usr/bin/env python3
"""Trainforge generators — assessment-generator provider (Wave W-D15).

The assessment-generator surface authors assessment questions
(MCQ / T/F / fill-in-blank / short answer / essay) grounded in
course content chunks. Pre-W-D15 this surface ran exclusively as a
Claude Code subagent dispatched via ``ED4ALL_AGENT_DISPATCH=true`` —
a configuration that routes question text through Anthropic, which
the licensing posture in ``docs/LICENSING.md`` flags as training-data
exposure for derivative SLM corpora (the questions land in
``assessments.json`` and feed into the
``training_synthesis`` instruction-pair / preference-pair surface, so
they ARE training data for the resulting SLM adapter).

W-D15 ships an in-process LLM seam mirroring W-D11.A
``COURSEFORGE_PROVIDER`` and W-D14 ``COURSEPLANNER_PROVIDER``
short-circuit patterns:

- Operator sets ``TRAINFORGE_ASSESSMENT_PROVIDER=local`` (or
  ``together`` / any registered OpenAI-compatible provider) →
  ``MCP/core/executor.py`` bypasses the Claude Code subagent
  dispatch for the ``assessment-generator`` agent and falls
  through to the in-process registry tool ``generate_assessments``,
  which constructs an :class:`AssessmentGeneratorProvider` and uses
  it to author the questions.
- Default unset → the legacy subagent-or-deterministic path runs
  byte-stable for every existing run.

DESIGN INVARIANT (W-D12 contract): provider references stay dynamic.
The class consults the ``_OPENAI_COMPATIBLE_PROVIDERS`` registry at
``MCP/orchestrator/llm_backend.py`` for its supported providers (so a
new entry in that registry — ``groq`` / ``fireworks`` / ``deepseek`` /
any future addition — flows through here without a code edit) and
routes every non-Anthropic call through
:func:`MCP.orchestrator.llm_backend.resolve_openai_compatible_backend`
so the runtime backend is dynamically built from the registry entry.
**There is exactly ONE class for every OpenAI-compatible provider** —
adding a new one is a registry-entry change, NOT a new subclass.

For ``provider="anthropic"`` (legacy default for the class body — kept
for backward-compatibility when the provider is constructed directly),
the call routes through the Anthropic SDK via the base class's
inherited :meth:`_call_anthropic` plumbing.

BASE CLASS CHOICE: this provider subclasses
:class:`Courseforge.generators._base._BaseLLMProvider` rather than
the Trainforge-side :class:`_BaseSynthesisProvider`. Rationale: the
assessment-generator surface is structurally a one-shot
"text in → JSON out" call, parallel to the W-D14 ``OutlinerProvider``
shape — NOT the per-pair instruction/preference paraphrase loop the
``_BaseSynthesisProvider`` was shaped around. Reusing the
``_BaseLLMProvider`` skeleton keeps the constructor / dispatch /
decision-capture surface byte-identical with the W-D14 sibling, so
operators see the same env-var contract and audit shape across both
seams.

Public surface:

- :class:`AssessmentGeneratorProvider` — sub-class of
  :class:`Courseforge.generators._base._BaseLLMProvider`.
- :meth:`AssessmentGeneratorProvider.generate_assessments` — emits a
  list of canonical question dicts (matching
  :class:`Trainforge.generators.assessment_generator.QuestionData`
  ``to_dict()`` shape) consumed by
  ``MCP/tools/trainforge_tools.py::generate_assessments`` under the
  ``TRAINFORGE_ASSESSMENT_PROVIDER`` opt-in. Each question entry
  carries an ``evidence_quote`` field per the W-D11 T11.3 grounding
  contract.
- :data:`ENV_PROVIDER` — ``"TRAINFORGE_ASSESSMENT_PROVIDER"``.
- :data:`ENV_MODEL` — ``"TRAINFORGE_ASSESSMENT_MODEL"``.
- :data:`DEFAULT_PROVIDER` — ``"anthropic"`` (the class's hardcoded
  default; the WORKFLOW default is "no in-process provider — defer
  to the legacy subagent / deterministic path" and is enforced
  upstream by ``generate_assessments`` checking
  ``TRAINFORGE_ASSESSMENT_PROVIDER`` for a non-empty value).

Decision-capture: every dispatch emits exactly one
``decision_type="assessment_generator_call"`` event whose rationale
interpolates dynamic per-call signals (provider, model, course_code,
chunk_count, target_count, distinct_question_types,
emitted_questions, refusal_count, retry_count, output character
count, evidence_quote_emit_rate). The ``provider`` field is the
runtime value the operator selected — NOT a static label — so a
post-hoc audit can attribute every assessment-generation call to
the OSS / cloud / local backend that produced it.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — env vars + defaults
# ---------------------------------------------------------------------------

ENV_PROVIDER = "TRAINFORGE_ASSESSMENT_PROVIDER"
ENV_MODEL = "TRAINFORGE_ASSESSMENT_MODEL"

# DEFAULT_PROVIDER preserves legacy class-body behavior. The WORKFLOW
# default is "no in-process provider unless TRAINFORGE_ASSESSMENT_PROVIDER
# is explicitly set", enforced by the registry-tool short-circuit at
# ``MCP/tools/trainforge_tools.py::generate_assessments``.
DEFAULT_PROVIDER = "anthropic"


# Supported providers are derived dynamically from the W-D12 registry
# at construction time. ``anthropic`` is added explicitly because it is
# NOT an OpenAI-compatible registry entry (it has its own native SDK
# path). Any new entry in ``_OPENAI_COMPATIBLE_PROVIDERS`` flows through
# here without a code edit — that is the W-D12 dynamic-references
# contract this provider class is built around.
def _build_supported_providers() -> Tuple[str, ...]:
    """Compose the supported-providers tuple at call time.

    Read at construction time so a test that monkeypatches
    ``_OPENAI_COMPATIBLE_PROVIDERS`` sees the patched view; not cached
    at import.
    """
    return ("anthropic",) + tuple(_OPENAI_COMPATIBLE_PROVIDERS.keys())


_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.3


# Canonical Bloom enum mirrored from
# ``Courseforge/generators/outline/_outliner_provider.py::_BLOOM_LEVEL_ENUM``
# and ``schemas/knowledge/courseforge_jsonld_v1.schema.json::bloomLevel``.
_BLOOM_LEVEL_ENUM: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


# Canonical question types mirrored from the legacy
# ``AssessmentGenerator.BLOOM_LEVELS`` table; corresponds to the same
# 5-type universe consumed by W6.B / W7.A-C per-question-type
# segmentation across ``quality_report.json`` and ``eval_report.json``.
_QUESTION_TYPE_ENUM: Tuple[str, ...] = (
    "multiple_choice",
    "true_false",
    "short_answer",
    "essay",
    "fill_in_blank",
)


# Compact system prompt — the prompt body has to fit a 7B-class local
# model's instruction-following window. Mirrors the ≤80-word terseness
# of ``Courseforge/generators/outline/_outliner_provider.py::_OUTLINER_SYSTEM_PROMPT``.
# The ``evidence_quote`` directive lands in the user prompt rather
# than the system prompt so the per-question shape is in close
# proximity to the example payload (smaller models follow that more
# reliably).
_ASSESSMENT_SYSTEM_PROMPT = (
    "You are a Trainforge assessment-generator authoring rigorous, "
    "content-grounded assessment questions for a course. Emit ONLY a "
    "JSON object — no preamble, no markdown fences, no commentary. "
    "Every question MUST be answerable from the supplied source "
    "chunks. Bloom levels MUST come from the canonical enum "
    "(remember/understand/apply/analyze/evaluate/create). "
    "Question types MUST come from the canonical enum "
    "(multiple_choice/true_false/short_answer/essay/fill_in_blank). "
    "If you cannot author a content-grounded question for a slot, "
    "EMIT a refusal entry (kind='skip') rather than a generic "
    "placeholder — the consumer filters refusals downstream."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssessmentGeneratorProviderError(RuntimeError):
    """Assessment-generator dispatch / parse / validation failure.

    Carries an opaque ``code`` field so callers can branch on the
    failure mode without parsing the message string.

    Canonical codes:

    - ``assessment_exhausted`` — every parse retry returned non-JSON or
      structurally-invalid JSON; the LLM tier failed to produce a
      usable question payload.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AssessmentGeneratorProvider(_BaseLLMProvider):
    """LLM-agnostic assessment-generator provider.

    Subclasses :class:`Courseforge.generators._base._BaseLLMProvider`
    for the shared decision-capture + Anthropic plumbing, then routes
    every non-Anthropic call through the W-D12
    :func:`resolve_openai_compatible_backend` factory so the runtime
    backend is dynamically built from the
    ``_OPENAI_COMPATIBLE_PROVIDERS`` registry. Adding a new provider
    (Groq, Fireworks, DeepSeek, ...) is a registry-entry change in
    ``MCP/orchestrator/llm_backend.py`` — this class needs NO edit to
    pick it up.

    Public method:

    - :meth:`generate_assessments` — emits a list of canonical
      question dicts consumed by
      ``MCP/tools/trainforge_tools.py::generate_assessments``.

    Decision-capture: every successful or failed dispatch emits
    exactly one ``decision_type="assessment_generator_call"`` event
    whose rationale interpolates dynamic per-call signals (provider,
    model, course_code, chunk_count, target_count,
    distinct_question_types, emitted_questions, refusal_count,
    retry_count, output character count, evidence_quote_emit_rate).
    The ``provider`` field is the runtime value the operator selected
    so a post-hoc audit can attribute every assessment-generation
    call to the OSS / cloud / local backend that produced it.
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
        # Test seam: inject a pre-built W-D12 backend so unit tests
        # don't have to spin up an HTTP server. When provided, this
        # backend is used directly instead of resolving via the
        # registry. ``None`` is the production default — the registry
        # resolution chain in :meth:`_dispatch_call` runs.
        openai_compatible_backend: Optional[Any] = None,
    ) -> None:
        # Resolve env-var override BEFORE delegating to the base so the
        # base only sees a concrete model value. Resolution chain mirrors
        # the W-D12 / W-D14 contract: per-call kwarg > per-tier env var
        # (``TRAINFORGE_ASSESSMENT_MODEL``) > per-backend baseline
        # (resolved by the base from synthesis env vars).
        resolved_model = (
            model
            or os.environ.get(ENV_MODEL)
            or None  # let the base resolve from the synthesis env vars
        )
        # Compose supported_providers dynamically from the registry so a
        # new entry flows through without an edit here.
        supported = _build_supported_providers()

        # The base's __init__ has hardcoded if/elif branches for
        # anthropic / together / local. To keep its API contract intact
        # while admitting the W-D12 registry's other providers
        # (groq / fireworks / deepseek / ...), we resolve the desired
        # provider first; if it's an OpenAI-compatible registry entry
        # OTHER than the three the base understands natively, we fall
        # back to "local" inside the base init (so the base wires up
        # its OpenAICompatibleClient) and then override _dispatch_call
        # below to route through the W-D12 backend at call time.
        resolved_provider = (
            provider
            or os.environ.get(ENV_PROVIDER)
            or DEFAULT_PROVIDER
        ).lower()
        if resolved_provider not in supported:
            raise ValueError(
                f"AssessmentGeneratorProvider: unknown provider "
                f"{resolved_provider!r}; expected one of "
                f"{list(supported)}. Add a registry entry to "
                f"_OPENAI_COMPATIBLE_PROVIDERS in "
                f"MCP/orchestrator/llm_backend.py — no subclassing required."
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
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            # supported_providers passed to the base is the trio it
            # natively understands; the W-D12 surface is dispatched in
            # the override below. This keeps the base's init invariants
            # (api_key checks, model resolution) honored without forking
            # the base class.
            supported_providers=("anthropic", "together", "local"),
            system_prompt=_ASSESSMENT_SYSTEM_PROMPT,
        )
        # Restore the runtime provider label so decision-capture and
        # logging surface the registry name (e.g. "groq") and not the
        # base-init alias ("local").
        if self._registry_provider is not None:
            self._provider = self._registry_provider

        # Test seam — pre-built backend wins over registry resolution.
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
        """Route through the W-D12 backend for OpenAI-compatible providers.

        For ``provider == "anthropic"`` the base class's inherited
        :meth:`_call_anthropic` runs. For ``provider == "together"`` /
        ``"local"`` we keep the base's OpenAICompatibleClient path so
        the request retry / lenient JSON parse surface stays
        byte-stable with the W-D11 ``ContentGeneratorProvider``. For
        any OTHER registered OpenAI-compatible provider, route through
        :func:`resolve_openai_compatible_backend` so the runtime
        backend is built from the registry entry.

        The provider name lives in ``self._provider``; the W-D12
        backend handles base_url / api_key / default_model resolution
        from the registry per ``_OPENAI_COMPATIBLE_PROVIDERS[provider]``.
        Adding a new provider is a registry-entry change.
        """
        if self._provider == "anthropic":
            return self._call_anthropic(user_prompt)

        # together / local: defer to the base class so the existing
        # OpenAICompatibleClient retry / extract_text plumbing fires.
        if self._provider in ("together", "local"):
            return super()._dispatch_call(
                user_prompt, extra_payload=extra_payload
            )

        # All other OpenAI-compatible registry providers route through
        # the W-D12 backend dynamically. The provider_name is opaque —
        # it surfaces only as decision-capture metadata, NOT in any
        # branch inside the call body.
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
        # The W-D12 backend has its own internal retry policy and does
        # not surface a retry count — report 0 so the decision-capture
        # rationale stays well-typed; the per-call ``llm_chat_call``
        # event from the backend's own capture wiring carries finer
        # latency / dispatch detail.
        return text or "", 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_assessments(
        self,
        chunks: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
        target_count: int,
        *,
        course_code: str = "",
        bloom_levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Synthesize assessment questions grounded in the supplied chunks.

        Args:
            chunks: List of source chunk dicts (each carrying at
                minimum ``id`` / ``chunk_id`` and ``text``); the
                LLM grounds every question in one or more of these.
            objectives: List of learning-objective dicts (each
                carrying ``id`` and ``statement``); each question's
                ``objective_id`` MUST resolve to one of these.
            target_count: Number of questions to attempt to author.
                The LLM may emit fewer when content runs out (refusal
                entries land in the ``skipped_items`` list).
            course_code: Course slug used in the decision-capture
                rationale and the prompt header.
            bloom_levels: Optional list of Bloom levels to target.
                Defaults to ``["understand", "apply"]`` when unset.

        Returns:
            A dict carrying the canonical generation shape::

                {
                    "questions": [
                        {
                            "question_id": str,
                            "question_type": str,  # canonical enum
                            "stem": str,
                            "bloom_level": str,    # canonical enum
                            "objective_id": str,
                            "choices": [{"text": str, "is_correct": bool}],
                            "correct_answer": Optional[str],
                            "feedback": Optional[str],
                            "source_chunks": List[str],
                            "evidence_quote": Optional[str],
                            "generation_rationale": str,
                        },
                        ...
                    ],
                    "skipped_items": [
                        {
                            "objective_id": str,
                            "bloom_level": str,
                            "reason": str,
                        },
                        ...
                    ],
                    "mint_method": "trainforge_assessment_provider:<provider>",
                }

        Raises:
            ValueError: When ``chunks`` or ``objectives`` are empty.
            AssessmentGeneratorProviderError(code="assessment_exhausted"):
                When the LLM tier fails to emit usable JSON across
                the parse-retry budget.
        """
        if not chunks:
            raise ValueError("chunks required (non-empty list)")
        if not objectives:
            raise ValueError("objectives required (non-empty list)")
        if int(target_count) < 1:
            raise ValueError(f"target_count must be >= 1; got {target_count}")
        resolved_levels = list(bloom_levels) if bloom_levels else [
            "understand",
            "apply",
        ]

        user_prompt = self._render_user_prompt(
            chunks=chunks,
            objectives=objectives,
            target_count=int(target_count),
            course_code=course_code,
            bloom_levels=resolved_levels,
        )

        text, retry_count = self._dispatch_call(user_prompt)

        # Lenient parse — accept either a clean JSON object or a
        # markdown-fenced one. Mirrors the W-D14 OutlinerProvider
        # extractor semantics so the parse path is consistent across
        # both seams.
        parsed = self._extract_json_lenient(text)

        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"

        normalised: Optional[Dict[str, Any]] = None
        if parsed is not None:
            try:
                normalised = self._normalise_assessment_payload(
                    parsed,
                    objectives=objectives,
                )
            except ValueError as exc:
                last_error = f"normalisation failed: {exc}"

        # Always emit the decision-capture event — success OR failure —
        # so the audit trail records every dispatch.
        self._emit_per_call_decision(
            raw_text=text,
            retry_count=retry_count,
            course_code=course_code,
            chunk_count=len(chunks),
            target_count=int(target_count),
            success=normalised is not None,
            emitted_questions=(
                len(normalised.get("questions", []))
                if normalised is not None
                else 0
            ),
            refusal_count=(
                len(normalised.get("skipped_items", []))
                if normalised is not None
                else 0
            ),
            distinct_question_types=(
                len({
                    str(q.get("question_type") or "")
                    for q in (normalised.get("questions") or [])
                    if q.get("question_type")
                })
                if normalised is not None
                else 0
            ),
            evidence_quote_emit_rate=(
                self._compute_evidence_quote_emit_rate(normalised)
                if normalised is not None
                else 0.0
            ),
            last_error=last_error,
        )

        if normalised is None:
            raise AssessmentGeneratorProviderError(
                f"AssessmentGeneratorProvider exhausted parse on course "
                f"{course_code!r} (last_error={last_error!r})",
                code="assessment_exhausted",
            )
        return normalised

    # ------------------------------------------------------------------
    # User prompt rendering
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        chunks: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
        target_count: int,
        course_code: str,
        bloom_levels: List[str],
        # Required by the abstract base — unused for assessment generator.
        **_unused: Any,
    ) -> str:
        """Render the assessment-generator user prompt.

        The prompt embeds:

        - Course header (course_code, target_count, bloom_levels).
        - Objectives block: id + statement for each objective.
        - Chunks block: id + truncated text body so a thick corpus
          doesn't blow the model's context window.
        - One-shot example so smaller local models (Qwen-2.5-7B etc.)
          have a concrete shape to imitate.
        - ``evidence_quote`` directive per the W-D11 T11.3 grounding
          contract — every emitted question MUST carry a verbatim
          substring from one of the cited chunks supporting the
          correct answer.
        - Strict-JSON closing directive.
        """
        # Objectives block (always full-fidelity — typical course
        # carries 1-2 TOs + 8-30 COs, well within budget).
        obj_lines: List[str] = []
        for obj in objectives:
            if not isinstance(obj, dict):
                continue
            oid = str(obj.get("id") or "")
            stmt = str(obj.get("statement") or obj.get("text") or "")
            if oid and stmt:
                obj_lines.append(f"  - [{oid}] {stmt}")
        objectives_block = "\n".join(obj_lines) if obj_lines else "  (none)"

        # Chunks block — cap at 8 chunks and truncate each body at
        # 800 chars so a 7B-class model's context window stays
        # comfortable. The orchestrator caller is responsible for
        # selecting the most relevant chunks before dispatch; this
        # cap is a safety net, not a primary filter.
        chunk_lines: List[str] = []
        for chunk in chunks[:8]:
            cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
            text = str(chunk.get("text") or "").strip()
            if not cid or not text:
                continue
            if len(text) > 800:
                text = text[:797] + "..."
            chunk_lines.append(f"  - [{cid}] {text}")
        chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (none)"

        bloom_block = ", ".join(bloom_levels)

        # One-shot example pinned to the canonical shape. Helps a
        # 7B-class local model anchor on the JSON structure. The
        # ``evidence_quote`` field is in the example so the model
        # sees the verbatim-quote requirement inline.
        oneshot = (
            "Example output (illustrative — do NOT copy values):\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "question_type": "multiple_choice",\n'
            '      "stem": "Which of the following best describes X?",\n'
            '      "bloom_level": "understand",\n'
            '      "objective_id": "TO-01",\n'
            '      "choices": [\n'
            '        {"text": "Correct definition", "is_correct": true},\n'
            '        {"text": "Common misconception A", "is_correct": false},\n'
            '        {"text": "Plausible but wrong distractor", "is_correct": false},\n'
            '        {"text": "Another distractor", "is_correct": false}\n'
            "      ],\n"
            '      "feedback": "X is defined as ...",\n'
            '      "source_chunks": ["chunk_001"],\n'
            '      "evidence_quote": "X is defined as the canonical ..."\n'
            "    }\n"
            "  ],\n"
            '  "skipped_items": [\n'
            '    {"objective_id": "CO-05", "bloom_level": "evaluate", '
            '"reason": "no_extractable_content"}\n'
            "  ]\n"
            "}"
        )

        # ``evidence_quote`` directive — mirrors the canonical
        # ``EVIDENCE_QUOTE_PROMPT_DIRECTIVE`` from
        # ``Trainforge/generators/_base_synthesis_provider.py`` adapted
        # to the per-question shape (no per_claim_support array on
        # this surface — the quote attaches directly to the question
        # object). The directive lands in the user prompt so the
        # per-question shape is in close proximity to the example
        # payload (smaller models follow that more reliably).
        evidence_directive = (
            "For every emitted question, include an `evidence_quote` field "
            "whose value is a contiguous verbatim substring (≤240 chars) "
            "copied character-for-character from one of the cited "
            "`source_chunks` chunk text bodies that supports the correct "
            "answer. The quote MUST appear in that chunk's text exactly "
            "as written — do NOT paraphrase. If you cannot locate a "
            "contiguous span that supports the answer, emit "
            "`evidence_quote: null` rather than fabricating one."
        )

        return (
            f"Course: {course_code}\n"
            f"Target question count: {target_count}\n"
            f"Bloom levels to cover: {bloom_block}\n\n"
            "Learning objectives (every question's `objective_id` MUST "
            "match one of these IDs verbatim):\n"
            f"{objectives_block}\n\n"
            "Source chunks (every question's `source_chunks[]` MUST "
            "reference one or more of these IDs; ground every question "
            "in real chunk text):\n"
            f"{chunks_block}\n\n"
            f"{oneshot}\n\n"
            f"{evidence_directive}\n\n"
            "Synthesize up to the target_count questions. Each question "
            "MUST carry the canonical fields: question_type (canonical "
            "enum), stem, bloom_level (lowercase enum), objective_id, "
            "source_chunks[], evidence_quote (or null). MCQ questions "
            "MUST carry choices[] with exactly one is_correct=true; "
            "true_false MUST carry choices[] with True/False entries; "
            "fill_in_blank MUST carry correct_answer; short_answer + "
            "essay MAY carry correct_answer (model rubric). "
            "RESPOND ONLY WITH A JSON OBJECT containing top-level keys "
            "questions and skipped_items. No preamble, no markdown "
            "fences, no commentary."
        )

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    def _normalise_assessment_payload(
        self,
        parsed: Dict[str, Any],
        *,
        objectives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Project the LLM output onto the canonical generation shape.

        Required projection rules:

        - Every question gets a re-minted ``question_id``
          (``Q-LLM-NN``) so IDs are sequential and stable regardless
          of the LLM's emit. (The downstream consumer at
          ``MCP/tools/trainforge_tools.py`` further rewrites these
          when it materialises questions through the canonical
          ``AssessmentGenerator`` shape; the in-provider IDs are a
          stable seed.)
        - ``question_type`` is lowercased and validated against
          :data:`_QUESTION_TYPE_ENUM`; unknown values fall back to
          ``"short_answer"``.
        - ``bloom_level`` is lowercased and validated against
          :data:`_BLOOM_LEVEL_ENUM`; unknown values fall back to
          ``"understand"``.
        - ``objective_id`` is validated against the supplied
          ``objectives`` list; questions referencing a non-existent
          objective are dropped to skipped_items with reason
          ``"phantom_objective_id"``.
        - ``source_chunks`` defaults to ``[]`` when absent.
        - ``evidence_quote`` is preserved verbatim when emitted as a
          non-empty string; otherwise stamped ``None``.
        """
        if not isinstance(parsed, dict):
            raise ValueError(
                f"generate_assessments expected a JSON object, got "
                f"{type(parsed).__name__}"
            )
        questions_raw = parsed.get("questions") or []
        skipped_raw = parsed.get("skipped_items") or []

        # Build the set of valid objective IDs for the phantom check.
        valid_obj_ids: set = set()
        for obj in objectives:
            if isinstance(obj, dict):
                oid = str(obj.get("id") or "").strip()
                if oid:
                    valid_obj_ids.add(oid)

        questions_norm: List[Dict[str, Any]] = []
        skipped_from_phantom: List[Dict[str, Any]] = []
        for idx, raw in enumerate(questions_raw, start=1):
            if not isinstance(raw, dict):
                continue
            try:
                entry = self._normalise_one_question(raw, idx)
            except ValueError as exc:
                logger.debug(
                    "AssessmentGeneratorProvider: skipping malformed "
                    "question entry idx=%d: %s", idx, exc,
                )
                continue
            # Phantom-objective filter — track the slot as skipped
            # rather than admitting a question that points at a
            # non-existent LO.
            if entry["objective_id"] not in valid_obj_ids:
                skipped_from_phantom.append({
                    "objective_id": entry["objective_id"],
                    "bloom_level": entry["bloom_level"],
                    "reason": "phantom_objective_id",
                })
                continue
            questions_norm.append(entry)

        skipped_norm: List[Dict[str, Any]] = []
        for raw in skipped_raw:
            if not isinstance(raw, dict):
                continue
            entry = {
                "objective_id": str(raw.get("objective_id") or ""),
                "bloom_level": str(raw.get("bloom_level") or "").lower() or "understand",
                "reason": str(raw.get("reason") or "no_extractable_content"),
            }
            skipped_norm.append(entry)
        skipped_norm.extend(skipped_from_phantom)

        if not questions_norm and not skipped_norm:
            raise ValueError(
                "generate_assessments: LLM output carried no usable "
                "questions[] or skipped_items[]"
            )

        return {
            "questions": questions_norm,
            "skipped_items": skipped_norm,
            "mint_method": (
                f"trainforge_assessment_provider:{self._provider}"
            ),
        }

    @staticmethod
    def _normalise_one_question(
        raw: Dict[str, Any],
        idx: int,
    ) -> Dict[str, Any]:
        """Project a single LLM-emitted question onto the canonical shape."""
        entry: Dict[str, Any] = {}
        stem = str(raw.get("stem") or raw.get("text") or "").strip()
        if not stem:
            raise ValueError("question missing 'stem'")
        entry["stem"] = stem

        question_type = str(raw.get("question_type") or "").strip().lower()
        if question_type not in _QUESTION_TYPE_ENUM:
            question_type = "short_answer"
        entry["question_type"] = question_type

        bloom_level = str(raw.get("bloom_level") or "").strip().lower()
        if bloom_level not in _BLOOM_LEVEL_ENUM:
            bloom_level = "understand"
        entry["bloom_level"] = bloom_level

        entry["objective_id"] = str(raw.get("objective_id") or "").strip()

        # Mint a stable in-provider question_id.
        entry["question_id"] = f"Q-LLM-{idx:03d}"

        # Choices — pass through when emitted as a list of dicts.
        choices = raw.get("choices")
        if isinstance(choices, list):
            entry["choices"] = [
                {
                    "text": str(c.get("text") or ""),
                    "is_correct": bool(c.get("is_correct")),
                }
                for c in choices
                if isinstance(c, dict)
            ]
        else:
            entry["choices"] = []

        # correct_answer + feedback — best-effort pass-through.
        correct_answer = raw.get("correct_answer")
        if isinstance(correct_answer, str) and correct_answer.strip():
            entry["correct_answer"] = correct_answer.strip()
        else:
            entry["correct_answer"] = None
        feedback = raw.get("feedback")
        if isinstance(feedback, str) and feedback.strip():
            entry["feedback"] = feedback.strip()
        else:
            entry["feedback"] = None

        # source_chunks — best-effort list-of-strings projection.
        source_chunks = raw.get("source_chunks")
        if isinstance(source_chunks, list):
            entry["source_chunks"] = [
                str(s) for s in source_chunks if isinstance(s, str) and s.strip()
            ]
        else:
            entry["source_chunks"] = []

        # evidence_quote — preserve verbatim non-empty strings,
        # otherwise stamp None. Schema cap is 240 chars (mirrors the
        # canonical EVIDENCE_QUOTE_SCHEMA_FRAGMENT) but per the W-D11
        # graceful-degrade contract we don't reject longer quotes
        # here — the consumer-side validator handles that.
        evidence_quote = raw.get("evidence_quote")
        if isinstance(evidence_quote, str) and evidence_quote.strip():
            entry["evidence_quote"] = evidence_quote.strip()
        else:
            entry["evidence_quote"] = None

        # generation_rationale — pass through when emitted; fall
        # back to a structural default so downstream consumers
        # always see the field.
        rationale = raw.get("generation_rationale")
        if isinstance(rationale, str) and rationale.strip():
            entry["generation_rationale"] = rationale.strip()
        else:
            entry["generation_rationale"] = (
                f"LLM-authored {question_type} at {bloom_level} level "
                f"grounded in {len(entry['source_chunks'])} source "
                f"chunk(s)"
            )

        return entry

    @staticmethod
    def _compute_evidence_quote_emit_rate(
        normalised: Optional[Dict[str, Any]],
    ) -> float:
        """Return the fraction of emitted questions carrying an evidence_quote.

        Mirrors the W-D11 T11.3 emit-rate semantic (claims_with_quote /
        total) on the per-question surface. Returns 0.0 on an empty
        questions list so the rationale always stays well-typed.
        """
        if normalised is None:
            return 0.0
        questions = normalised.get("questions") or []
        total = len(questions)
        if total == 0:
            return 0.0
        with_quote = sum(
            1
            for q in questions
            if isinstance(q.get("evidence_quote"), str)
            and q["evidence_quote"].strip()
        )
        return with_quote / total

    @staticmethod
    def _extract_json_lenient(text: str) -> Optional[Dict[str, Any]]:
        """Lenient JSON object extractor.

        Mirrors :meth:`OutlinerProvider._extract_json_lenient` —
        handles three common LLM response shapes:

        1. Clean JSON object — ``json.loads`` succeeds verbatim.
        2. Markdown-fenced (``` ```json ... ``` ``` or ``` ``` ... ``` ```)
           — strip fences then ``json.loads``.
        3. Trailing prose — find the outermost balanced ``{...}`` and
           parse the captured slice.

        Returns ``None`` on any parse failure so the caller can branch
        on success without try/except.
        """
        if not text:
            return None
        candidate = text.strip()
        # 1. Clean parse
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
        # 2. Strip markdown fences
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)```",
            candidate,
            re.DOTALL,
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
        # 3. Outermost balanced braces
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(candidate[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Emit one ``assessment_generator_call`` decision-capture event.

        Rationale interpolates dynamic per-call signals so a post-hoc
        replay records the EXACT runtime values: which provider fired,
        which model produced the output, how many questions the LLM
        emitted, the parse / dispatch retry count, success/failure,
        the evidence_quote emit rate, and the last_error when the
        dispatch failed. Per the project's LLM call-site instrumentation
        contract: ≥20 chars, dynamic signals, provider value runtime
        not static.
        """
        course_code = str(call_context.get("course_code", ""))
        chunk_count = int(call_context.get("chunk_count", 0))
        target_count = int(call_context.get("target_count", 0))
        success = bool(call_context.get("success", False))
        emitted_questions = int(call_context.get("emitted_questions", 0))
        refusal_count = int(call_context.get("refusal_count", 0))
        distinct_question_types = int(
            call_context.get("distinct_question_types", 0)
        )
        evidence_quote_emit_rate = float(
            call_context.get("evidence_quote_emit_rate", 0.0)
        )
        last_error = call_context.get("last_error")

        decision = (
            f"assessment_generator_call:{course_code}:"
            f"{'success' if success else 'failed'}"
        )
        rationale_parts = [
            f"course_code={course_code}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"chunk_count={chunk_count}",
            f"target_count={target_count}",
            f"emitted_questions={emitted_questions}",
            f"distinct_question_types={distinct_question_types}",
            f"refusal_count={refusal_count}",
            f"evidence_quote_emit_rate={evidence_quote_emit_rate:.3f}",
            f"output_chars={len(raw_text or '')}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if last_error:
            rationale_parts.append(f"last_error={str(last_error)[:120]}")
        rationale = "; ".join(rationale_parts)

        self._emit_decision(
            decision_type="assessment_generator_call",
            decision=decision,
            rationale=rationale,
        )


__all__ = [
    "AssessmentGeneratorProvider",
    "AssessmentGeneratorProviderError",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "DEFAULT_PROVIDER",
    "_build_supported_providers",
]
