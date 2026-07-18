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
# Per-call generation cap (max_tokens) for the three-stage synthesis LLM
# calls. Whole-chapter objective / structure synthesis emits multiple TO/CO
# statements + sub-objectives, and a REASONING model (nemotron_v3) that leaks
# any residual thinking tokens can overrun a tight budget → the OpenAI-compat
# ``finish_reason='length'`` truncation guard fires and fails the phase. The
# provider sources a generous default from this env. Resolution (high → low):
# ``max_tokens`` kwarg > ``TEXTBOOK_SYNTHESIS_MAX_TOKENS`` env > the
# ``_DEFAULT_MAX_TOKENS`` provider default. Garbage / non-positive → default
# (parse-with-fallback, mirroring ``ENV_TIMEOUT`` above).
ENV_MAX_TOKENS = "TEXTBOOK_SYNTHESIS_MAX_TOKENS"
# Per-call HTTP timeout (seconds) for the OpenAI-compatible backends.
# Synthesis calls are long-context (24-40k-char inputs, multi-k-token
# output) — a live needle-probe against Ollama 0.21.2 measured 7B
# calls already exceeding the OpenAICompatibleClient 60s default, and
# 14B + 4096-token output runs ~120-300s. 300s is the default; override
# via ``TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS`` or the ``timeout`` kwarg.
ENV_TIMEOUT = "TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS"
# Per-attempt course_planning gate-retry re-roll salt (owner directive
# 2026-07-17). Set to ``attempt-N`` by
# ``MCP/core/workflow_runner.py::WorkflowRunner._retry_course_planning_gates``
# before each gate-failure re-dispatch of the ``course_planning`` phase;
# empty on every normal run. Read at CONSTRUCTION (the phase handler builds
# a fresh provider per attempt) and folded into the SYSTEM prompt so every
# dispatched synthesis call genuinely differs per retry attempt. NOTE: the
# Stage-2 per-window sidecar fingerprint in ``MCP/tools/pipeline_tools.py``
# hashes the UNSALTED module constant ``_TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT``,
# so the window/CO resume cache deliberately SURVIVES a salted retry — only
# the cluster (TO-derivation) stage re-rolls (its sidecar is evicted and its
# fingerprint carries the salt).
ENV_PLANNING_REROLL_SALT = "ED4ALL_PLANNING_REROLL_SALT"

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

# WS4 §1 — Stage-1 skeleton de-blinding. The legacy ``section_titles[:12]``
# head-slice and single 200-char ``chapter_text`` head sample front-loaded the
# Stage-1 prompt, leaving the back two-thirds of a many-section chapter unseen
# (the 141:1 collapse signature). ``_render_chapter_skeleton`` now even-stride
# samples up to ``_SKELETON_MAX_SECTION_TITLES`` titles across the FULL section
# range and draws head/mid/tail prose windows. Both feed the
# ``_SKELETON_CHAR_BUDGET`` measurement in ``_chunk_chapters_for_outline`` (the
# single overflow authority — a 141-section chapter renders ~3-6 KB titles +
# 3×200 prose < 7 KB, well under 40 KB → stays ``call_mode="single"`` and well
# under the 32k ``_LOCAL_NUM_CTX`` serving window). See WS4 §0.1 / §3.
_SKELETON_MAX_SECTION_TITLES = 48          # was implicit [:12]; ~3-6 KB short titles
_SKELETON_CHAPTER_SAMPLE_OFFSETS = 3       # head + mid + tail prose windows
_SKELETON_CHAPTER_SAMPLE_CHARS = 200       # per-window chars (unchanged per-window)

# WS4 §2 — scope-aware DRAFT terminal-objective band for the Stage-1 prompt.
# The hardcoded "1-4 DRAFT terminal objectives" literal was the SOLE TO-count
# lever (``_normalise_outline_payload`` does NOT clamp the count). The band is a
# prompt-band computation (the model self-limits): default ``(3,6)`` for normal
# multi-chapter input, bumped to ``(6,12)`` when the input is a single chapter
# carrying more than ``_COLLAPSE_SECTION_THRESHOLD`` sections (the collapse
# signature). Mirrors the extractor-side
# ``_STRUCTURE_COLLAPSE_SECTION_THRESHOLD`` (WS4 §4).
_DRAFT_TO_BAND_DEFAULT = (3, 6)            # normal multi-chapter input
_DRAFT_TO_BAND_COLLAPSE = (6, 12)          # 1-chapter, section_count >> normal
_COLLAPSE_SECTION_THRESHOLD = 40           # >40 sections in 1 chapter == collapse

# Per-chapter chapter-text char budget for Stages 2 / 3. A single
# textbook chapter's prose is typically 10-20 KB; the cap protects
# against an unusually long chapter. Over-budget chapters truncate the
# tail and the decision event records ``chapter_text_truncated=true``.
_CHAPTER_TEXT_BUDGET = 24_000

# Per-chapter batching: Stages 2 / 3 dispatch N per-chapter calls in
# groups of at most this many, honoring the orchestrator's
# "max 10 simultaneous" protocol (root ``CLAUDE.md`` § Phase 3).
_CHAPTER_BATCH_SIZE = 10

# W2 §1.5: Stage-2 window-map serving-window override. Default resolves via
# ``lib.retrieval._prompts.resolve_num_ctx()`` (which reads ``ED4ALL_ANSWER_
# NUM_CTX``, default 4096), so one knob drives both the answer path and the
# synthesis window budget — but synthesis gets its OWN override. Documented in
# ``Courseforge/CLAUDE.md`` § Opt-In Behavior Flags. No provider/model selected,
# so no ``docs/LICENSING.md`` row.
_SYNTHESIS_NUM_CTX_ENV = "TEXTBOOK_SYNTHESIS_NUM_CTX"

# W2 §3: grammar / constrained-decoding mode for the window-objectives surface.
# Mirrors ``COURSEFORGE_OUTLINE_GRAMMAR_MODE``; ``None`` autodetects from the
# resolved provider + base_url. No-op when ``TEXTBOOK_SYNTHESIS_PROVIDER`` unset.
ENV_GRAMMAR_MODE = "TEXTBOOK_SYNTHESIS_GRAMMAR_MODE"

# W2 §3: parse-retry budget for ``synthesize_window_objectives``. The payload is
# validated against ``_WINDOW_OBJECTIVES_SCHEMA`` per attempt; the schema dump is
# appended on retry (generic schema echo — the outline tier's per-pattern
# ``_RETRY_DIRECTIVE_PATTERNS`` table is outline-specific and NOT reused).
_MAX_SYNTHESIS_PARSE_RETRIES = 3

# Structure-aware synthesis — inject a compact HEADING SKELETON of the window's
# chapter into the per-window Stage-2 prompt so TO/CO derivation is
# structure-aware. Gated by ``ED4ALL_SYNTHESIS_SKELETON`` (default OFF →
# byte-identical prompt). The skeleton is CONTEXT ONLY; the prompt instructions
# still require every objective to cite chunk ids (never headings), and the
# citation-extraction path is unchanged. Budget: cap the rendered skeleton at
# ``_SKELETON_HEADING_TOKEN_BUDGET`` tokens per window, dropping the DEEPEST
# heading levels first when over. Token estimate reuses the repo-wide 2.5
# chars/token convention (``lib/retrieval/_prompts.py``).
ENV_SYNTHESIS_SKELETON = "ED4ALL_SYNTHESIS_SKELETON"
_SKELETON_HEADING_TOKEN_BUDGET = 1500
_SKELETON_HEADING_CHARS_PER_TOKEN = 2.5
_SYNTHESIS_SKELETON_TRUTHY = {"1", "true", "yes", "on"}


def _resolve_synthesis_skeleton() -> bool:
    """Resolve the ``ED4ALL_SYNTHESIS_SKELETON`` gate (parse-with-fallback).

    Truthy (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive) → on; unset /
    falsey / garbage → off (byte-identical prompt). Read once at construction so
    a test can ``monkeypatch.setenv`` before building the provider.
    """
    raw = os.environ.get(ENV_SYNTHESIS_SKELETON, "")
    return str(raw).strip().lower() in _SYNTHESIS_SKELETON_TRUTHY


_BLOOM_LEVEL_ENUM: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


# W2 §3: the single window-objectives JSON Schema (Draft 2020-12). NOT
# per-block-type — the synthesis surface emits one shape. ``source_chunk_ids``
# carries ``minItems: 1`` so the grammar itself forbids the empty citation list
# at sample time (the strongest lever); §2.3 is the post-parse ⊆ backstop for
# backends that ignore the schema. The grammar canNOT enforce ⊆ the supplied set
# (runtime data, not a fixed enum) — §2.3 enforces membership.
_WINDOW_OBJECTIVES_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_objectives"],
    "properties": {
        "candidate_objectives": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "bloom_level", "source_chunk_ids"],
                "properties": {
                    "statement": {"type": "string", "minLength": 1},
                    "bloom_level": {
                        "type": "string",
                        "enum": list(_BLOOM_LEVEL_ENUM),
                    },
                    "bloom_verb": {"type": "string"},
                    "abcd": {"type": "object"},
                    "source_chunk_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "sub_objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


def _bloom_verb_examples(*, per_level: int = 5) -> str:
    """Render a deterministic per-level sample of canonical Bloom *action*
    verbs, sourced from the single-source-of-truth taxonomy.

    Drives the synthesis prompt so the model picks a real action verb
    (``explain`` / ``apply`` / ``analyze`` / ``evaluate`` / ``design``)
    instead of reusing the level *name* (``understand``) as the verb — the
    measured root cause of both the ABCD_VERB_BLOOM_MISMATCH gate failures
    and the bloom-monotony of the "Understand the core concepts of X"
    template. Sorted output keeps the prompt byte-stable across runs.
    """
    try:
        from lib.ontology.learning_objectives import BLOOMS_VERBS
    except Exception:  # noqa: BLE001 — taxonomy load is the floor, never fatal
        return ""
    lines: List[str] = []
    for level in _BLOOM_LEVEL_ENUM:
        verbs = sorted(BLOOMS_VERBS.get(level, frozenset()))[:per_level]
        if verbs:
            lines.append(f"    {level}: {', '.join(verbs)}")
    return "\n".join(lines)


# W2 — Bloom-diversity + action-verb directive shared by every objective
# surface (window / chapter / outline / reconcile). Two coupled instructions,
# both grounding-preserving:
#   1. The behavior verb MUST be a concrete *action* verb, NEVER the Bloom
#      level name itself (the level name "understand"/"apply" is not a valid
#      Bloom verb — it fails AbcdObjectiveValidator's BLOOMS_VERBS check).
#   2. When a window/chapter teaches material at more than one cognitive
#      level, the emitted objectives SHOULD span that range (recall/explain
#      for foundational facts, apply/analyze/evaluate for the higher-order
#      skills the prose actually demonstrates) — diversity grounded in the
#      content, never fabricated.
_BLOOM_DIVERSITY_DIRECTIVE = (
    "BLOOM DIVERSITY + VERB RULE (grounded, do NOT fabricate):\n"
    "- The ABCD behavior verb MUST be a concrete ACTION verb chosen for the "
    "declared bloom_level — NEVER the level name itself. The bare words "
    '"understand"/"apply"/"analyze"/"remember"/"evaluate"/"create" are Bloom '
    "LEVELS, not verbs; do not use them as the verb.\n"
    "- Pick the verb from the level's canonical action verbs:\n"
    f"{_bloom_verb_examples()}\n"
    "- bloom_verb and abcd.behavior.verb MUST be the SAME action verb, and "
    "the statement MUST open with that verb (e.g. \"Explain ...\", "
    "\"Apply ...\", \"Evaluate ...\") — not \"Understand the core concepts "
    "of X\".\n"
    "- When these chunks teach material at more than one cognitive level, "
    "make the objectives span that range: some lower-order (remember/"
    "understand) for the foundational facts the prose states, and some "
    "higher-order (apply/analyze/evaluate) for the skills the prose actually "
    "demonstrates. Only assign a level the content genuinely supports."
)


_TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a Courseforge textbook-synthesis author. You read a "
    "textbook's structure and prose and synthesize course design "
    "artifacts — a semantic outline, terminal/chapter learning "
    "objectives, and a domain-concept vocabulary. Emit ONLY a JSON "
    "object — no preamble, no markdown fences, no commentary. Every "
    "objective follows the ABCD framework (Audience, Behavior, "
    "Condition, Degree); the behavior verb MUST be a concrete ACTION "
    "verb from the canonical Bloom's taxonomy (e.g. explain, apply, "
    "analyze, evaluate, design) chosen for the objective's cognitive "
    "level — never the bare level name (remember/understand/apply/"
    "analyze/evaluate/create are LEVELS, not verbs). Author objectives "
    "across a RANGE of Bloom levels appropriate to the content, not all "
    "at one level. Ground every artifact in the supplied textbook "
    "content; do NOT invent chapters, concepts, or facts not present."
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


def _resolve_synthesis_max_tokens(explicit: Optional[int]) -> int:
    """Resolve the per-call generation cap: kwarg > env > default.

    ``explicit`` (the ``max_tokens`` kwarg) wins when supplied. Otherwise read
    :data:`ENV_MAX_TOKENS` (``TEXTBOOK_SYNTHESIS_MAX_TOKENS``) and parse it as a
    positive int; a missing / unparseable / non-positive value falls back to
    :data:`_DEFAULT_MAX_TOKENS`. Parse-with-fallback mirrors the other
    ``TEXTBOOK_SYNTHESIS_*`` knobs so a garbage env value never shrinks the cap.
    """
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(ENV_MAX_TOKENS)
    if not raw or not str(raw).strip():
        return _DEFAULT_MAX_TOKENS
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "TextbookSynthesisProvider: ignoring non-integer %s=%r; "
            "using default %d",
            ENV_MAX_TOKENS, raw, _DEFAULT_MAX_TOKENS,
        )
        return _DEFAULT_MAX_TOKENS
    if parsed <= 0:
        logger.warning(
            "TextbookSynthesisProvider: ignoring non-positive %s=%r; "
            "using default %d",
            ENV_MAX_TOKENS, raw, _DEFAULT_MAX_TOKENS,
        )
        return _DEFAULT_MAX_TOKENS
    return parsed


def _coerce_heading_level(raw: Any, default: int) -> int:
    """Coerce a textbook_structure ``headingLevel`` to a positive int.

    Non-numeric / non-positive → ``default`` (parse-with-fallback), so a
    malformed structure node still slots into the deepest-first skeleton drop
    with a well-ordered level instead of raising.
    """
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


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
        # ``None`` sentinel → resolve from ``TEXTBOOK_SYNTHESIS_MAX_TOKENS``
        # env (falling back to ``_DEFAULT_MAX_TOKENS``). An explicit int wins.
        max_tokens: Optional[int] = None,
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
        # W2 §3: constrained-decoding mode for the window-objectives
        # surface. Resolution: kwarg > ``TEXTBOOK_SYNTHESIS_GRAMMAR_MODE``
        # env > autodetect-None. Mirrors the outline tier's knob.
        grammar_mode: Optional[str] = None,
        # Structure-aware synthesis: the parsed ``textbook_structure.json``
        # (``{"chapters": [...]}``) whose per-chapter heading tree feeds the
        # ``ED4ALL_SYNTHESIS_SKELETON`` per-window skeleton. ``None`` / no
        # chapters → the skeleton is empty and the window prompt is
        # byte-identical (the deterministic decoupling from SemantiK sidecars:
        # the skeleton is sourced from this already-in-path artifact only).
        textbook_structure: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_model = (
            model
            or os.environ.get(ENV_MODEL)
            or None  # let the base resolve from the synthesis env vars
        )
        # Resolve the per-call generation cap: kwarg > env > default. A
        # reasoning model doing whole-chapter objective synthesis needs a
        # generous cap (an 800-style default truncates the multi-TO/CO output).
        resolved_max_tokens = _resolve_synthesis_max_tokens(max_tokens)
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
            # ``nvidia`` is base-NATIVE (``_base.py`` has a direct nvidia
            # branch that resolves NVIDIA_API_KEY + the nvidia base_url),
            # so it must NOT route through the W-D12 registry override
            # (which wires the base under "local" and would send the
            # "local" placeholder api_key -> HTTP 401). Other registry
            # providers (groq/fireworks/deepseek) keep the override path.
            resolved_provider not in ("anthropic", "together", "local", "nvidia")
            and resolved_provider in _OPENAI_COMPATIBLE_PROVIDERS
        ):
            # Stash the real provider for the W-D12 dispatch override;
            # let the base init wire its plumbing under "local" so we
            # inherit the OpenAICompatibleClient + decision-capture seat.
            self._registry_provider = resolved_provider
            base_provider_arg = "local"
        else:
            base_provider_arg = resolved_provider

        # Gate-retry re-roll salt (see the ENV_PLANNING_REROLL_SALT comment
        # above): when the workflow runner set it, fold it into the system
        # prompt so a course_planning gate-failure retry attempt genuinely
        # re-rolls instead of reproducing the prior attempt's output.
        reroll_salt = os.environ.get(ENV_PLANNING_REROLL_SALT, "").strip()
        resolved_system_prompt = _TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT
        if reroll_salt:
            resolved_system_prompt = (
                f"{_TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT}\n\n"
                f"[Re-roll {reroll_salt}: a prior synthesis attempt failed "
                "validation. Author afresh — ground every statement strictly "
                "in the supplied source material and do not repeat the prior "
                "attempt's phrasing verbatim.]"
            )

        super().__init__(
            provider=base_provider_arg,
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            capture=capture,
            max_tokens=resolved_max_tokens,
            temperature=temperature,
            timeout=resolved_timeout,
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            supported_providers=("anthropic", "together", "local", "nvidia"),
            system_prompt=resolved_system_prompt,
        )
        # Restore the runtime provider label so decision-capture and
        # logging surface the registry name (e.g. "groq").
        if self._registry_provider is not None:
            self._provider = self._registry_provider

        self._injected_oa_backend = openai_compatible_backend

        # W2 §3: grammar mode is a pure string knob; ``None`` means
        # autodetect from ``provider`` + ``base_url`` at call time.
        self._grammar_mode: Optional[str] = (
            grammar_mode
            or os.environ.get(ENV_GRAMMAR_MODE)
            or None
        )

        # Structure-aware synthesis (``ED4ALL_SYNTHESIS_SKELETON``): read the
        # gate once and index the textbook_structure chapters by id so
        # ``synthesize_window_objectives`` can render the window's chapter
        # heading skeleton. Off / no chapters → empty map → byte-identical
        # window prompt.
        self._synthesis_skeleton_enabled: bool = _resolve_synthesis_skeleton()
        self._skeleton_chapters_by_id: Dict[str, Dict[str, Any]] = {}
        if isinstance(textbook_structure, dict):
            for _ch in textbook_structure.get("chapters") or []:
                if isinstance(_ch, dict):
                    _cid = str(_ch.get("id") or "")
                    if _cid:
                        self._skeleton_chapters_by_id[_cid] = _ch

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

            draft_band = self._draft_to_band(group)
            self._emit_per_call_decision(
                mode="outline",
                raw_text=text,
                retry_count=retry_count,
                course_name=course_name,
                chapter_count_input=len(group),
                section_count_input=sum(
                    len(c.get("sections") or []) for c in group
                ),
                draft_to_band=draft_band,
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
    # Public API — Stage 2 (W2): per-WINDOW candidate-objective synthesis
    # ==================================================================

    def synthesize_window_objectives(
        self,
        window: Dict[str, Any],
        *,
        course_name: str,
        draft_terminal_objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """W2 §1.5 — synthesize one chunk-window's CANDIDATE objectives.

        Sibling of :meth:`synthesize_chapter_objectives` (does NOT replace
        it — back-compat for the chapter-grained fallback). The map unit is a
        chunk-window (id + truncated bodies) instead of chapter prose, so the
        model cites ``source_chunk_ids`` it can COPY from the enumerated
        allowed-id set rather than infer. Grammar mode (§3) + ⊆-enforcement
        (§2.3) ride on this surface only; outline/concepts/reconcile keep their
        lenient parse.

        Args:
            window: a ``ChunkWindow.to_dict()`` (or the live dataclass exposing
                ``chapter_id`` / ``window_index`` / ``chunk_ids`` / ``chunks``).
            course_name: course slug for the prompt header + capture rationale.
            draft_terminal_objectives: Stage-1 draft TO-NN list (statements
                only, for grounding).

        Returns:
            ``{"chapter_id", "window_index", "candidate_objectives": [...]}``
            — each candidate carries a resolved ``source_chunk_ids`` ⊆ the
            window's allowed-id set (plus reconstructed ``source_refs`` for
            on-disk back-compat).

        Raises:
            ValueError: when ``course_name`` is empty.
            TextbookSynthesisProviderError(
                code="chapter_objectives_exhausted"): when the call exhausts
                the parse-retry budget (reuses the existing code so the caller's
                per-window fail-soft branch already handles it).
        """
        import jsonschema  # type: ignore[import-untyped]

        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")
        if not isinstance(window, dict):
            raise ValueError("window must be a dict (ChunkWindow.to_dict())")

        chapter_id = str(window.get("chapter_id") or "")
        window_index = int(window.get("window_index") or 0)
        allowed_chunk_ids = [
            str(c) for c in (window.get("chunk_ids") or []) if c
        ]
        allowed_set = set(allowed_chunk_ids)
        window_chunks = [
            c for c in (window.get("chunks") or []) if isinstance(c, dict)
        ]

        base_prompt = self._render_window_objectives_prompt(
            chapter_id=chapter_id,
            window_index=window_index,
            window_chunks=window_chunks,
            course_name=course_name,
            draft_terminal_objectives=draft_terminal_objectives or [],
            heading_skeleton=self._build_window_heading_skeleton(chapter_id),
        )
        extra_payload = self._build_synthesis_grammar_payload()

        last_error: Optional[str] = None
        last_raw: str = ""
        total_retries = 0
        parsed: Optional[Dict[str, Any]] = None
        attempt = 0
        while attempt < _MAX_SYNTHESIS_PARSE_RETRIES:
            user_prompt = base_prompt
            if attempt > 0 and last_error:
                schema_hint = json.dumps(
                    _WINDOW_OBJECTIVES_SCHEMA, sort_keys=True
                )
                user_prompt = (
                    f"{base_prompt}\n\n"
                    "Your previous output failed JSON Schema validation: "
                    f"{last_error}\n"
                    "Return ONLY a JSON object matching this schema:\n"
                    f"{schema_hint}"
                )
            raw_text, retry_count = self._dispatch_call(
                user_prompt, extra_payload=extra_payload or None
            )
            total_retries += int(retry_count)
            last_raw = raw_text
            candidate = self._extract_json_lenient(raw_text)
            if candidate is None:
                last_error = "lenient JSON parse returned None"
                attempt += 1
                continue
            try:
                jsonschema.Draft202012Validator(
                    _WINDOW_OBJECTIVES_SCHEMA
                ).validate(candidate)
            except jsonschema.ValidationError as exc:
                last_error = str(exc.message)[:300]
                attempt += 1
                continue
            parsed = candidate
            break

        objectives: List[Dict[str, Any]] = []
        out_of_set_dropped = 0
        empty_citation = 0
        if parsed is not None:
            (
                objectives,
                out_of_set_dropped,
                empty_citation,
            ) = self._normalise_window_objectives_payload(
                parsed,
                chapter_id=chapter_id,
                allowed_chunk_ids=allowed_set,
            )

        self._emit_per_call_decision(
            mode="chapter_objectives",
            raw_text=last_raw,
            retry_count=total_retries,
            course_name=course_name,
            chapter_id=chapter_id,
            chapter_text_chars=sum(
                len(str(c.get("text") or "")) for c in window_chunks
            ),
            co_count_emit=len(objectives),
            sub_objective_count_emit=sum(
                len(o.get("sub_objectives") or []) for o in objectives
            ),
            chapter_text_truncated=False,
            success=parsed is not None,
            last_error=last_error if parsed is None else None,
            # W2 §2.3 — the measured ⊆-enforcement signals.
            window_index=window_index,
            allowed_chunk_id_count=len(allowed_chunk_ids),
            cited_chunk_id_count=sum(
                len(o.get("source_chunk_ids") or []) for o in objectives
            ),
            citation_out_of_set_dropped=out_of_set_dropped,
            empty_citation_objectives=empty_citation,
        )

        if parsed is None:
            raise TextbookSynthesisProviderError(
                f"TextbookSynthesisProvider.synthesize_window_objectives "
                f"exhausted parse on window {chapter_id!r}#w{window_index} of "
                f"course {course_name!r} (last_error={last_error!r})",
                code="chapter_objectives_exhausted",
            )
        return {
            "chapter_id": chapter_id,
            "window_index": window_index,
            "candidate_objectives": objectives,
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

    def author_terminal_for_cluster(
        self,
        cluster_cos: List[Dict[str, Any]],
        *,
        course_name: str,
        cluster_index: int,
    ) -> Optional[Dict[str, Any]]:
        """WS1 — author ONE terminal objective summarizing one CO cluster.

        The bottom-up TO derivation (``MCP/tools/pipeline_tools.py::
        _derive_terminals_bottom_up``) clusters the grounded CO statement
        vectors and calls this once per cluster: author a single TO that
        accurately summarises that cluster's chapter objectives.

        Args:
            cluster_cos: the COs in one cluster (statements consumed for the
                prompt). Each is a dict carrying at least ``statement``.
            course_name: course slug for the capture rationale.
            cluster_index: 1-based cluster ordinal (capture rationale only).

        Returns:
            A single canonical TO dict ``{statement, bloom_level, bloom_verb,
            abcd, source_refs}`` — **NO id** (the caller mints ``TO-NN``) — or
            ``None`` on parse exhaustion.

        Fail-SOFT per cluster (deliberate divergence from
        :meth:`reconcile_terminal_objectives`'s fail-loud): one bad cluster
        degrades to the dedup-rep fallback at the call site, never sinks the
        run. Mirrors the Stage-2 per-chapter return-``None``-on-exhaustion
        posture.

        Raises:
            ValueError: When ``course_name`` is empty.
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")

        cos = list(cluster_cos or [])

        user_prompt = self._render_user_prompt(
            mode="author_terminal",
            cluster_cos=cos,
            course_name=course_name,
            cluster_index=cluster_index,
        )
        text, retry_count = self._dispatch_call(user_prompt)
        parsed = self._extract_json_lenient(text)

        terminal: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"
        else:
            terminal = self._normalise_author_terminal_payload(parsed)
            if terminal is None:
                last_error = "no terminal_objective in payload"

        self._emit_per_call_decision(
            mode="author_terminal",
            raw_text=text,
            retry_count=retry_count,
            course_name=course_name,
            cluster_index=cluster_index,
            co_count_input=len(cos),
            output_chars=len(text or ""),
            success=terminal is not None,
            last_error=last_error,
        )

        return terminal

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
    def _sample_section_titles(titles: List[str]) -> List[str]:
        """Even-stride sample up to ``_SKELETON_MAX_SECTION_TITLES`` titles
        across the FULL list (WS4 §1).

        Returns the titles unchanged when they fit the cap; otherwise picks
        even-stride indices spanning the whole range, ALWAYS including the
        first and last title (so the Stage-1 model sees the back of the book,
        not just its first sections — the 141:1 collapse de-blinder).
        """
        n = len(titles)
        cap = _SKELETON_MAX_SECTION_TITLES
        if n <= cap:
            return titles
        step = n / float(cap)
        idxs = sorted({int(i * step) for i in range(cap)} | {n - 1})
        return [titles[i] for i in idxs]

    @staticmethod
    def _sample_chapter_text(text: str) -> List[Tuple[str, str]]:
        """Head/mid/tail prose windows from ``chapter_text`` (WS4 §1.3).

        ``[]`` for empty text; a single ``("head", ...)`` window when the text
        is shorter than ~2 windows; otherwise head/mid/tail windows of
        ``_SKELETON_CHAPTER_SAMPLE_CHARS`` chars each, so a long chapter's back
        prose is no longer invisible to the Stage-1 model.
        """
        if not text:
            return []
        w = _SKELETON_CHAPTER_SAMPLE_CHARS
        if len(text) <= w * 2:
            return [("head", text[:w])]
        mid = (len(text) - w) // 2
        return [("head", text[:w]), ("mid", text[mid:mid + w]), ("tail", text[-w:])]

    @staticmethod
    def _render_chapter_skeleton(chapter: Dict[str, Any]) -> str:
        """Render one chapter's skeleton line for the Stage-1 prompt.

        Per plan §3.3: id + headingText/title + ordered section list +
        bounded samples drawn from ``chapter_text``. Reads both
        ``headingText`` (extractor camelCase) and ``title`` defensively.

        WS4 §1: the section list is now an even-stride sample across the FULL
        section range (cap ``_SKELETON_MAX_SECTION_TITLES``) instead of the
        legacy ``[:12]`` head-slice, and the prose sample is head/mid/tail
        windows instead of a single 200-char head — both budget-bounded so the
        ``_SKELETON_CHAR_BUDGET`` gate in ``_chunk_chapters_for_outline`` stays
        the single overflow authority.
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
        chapter_text = str(chapter.get("chapter_text") or "")
        samples = TextbookSynthesisProvider._sample_chapter_text(chapter_text)
        lines = [f"  - [{cid}] {title}"]
        if section_titles:
            sampled = TextbookSynthesisProvider._sample_section_titles(
                section_titles
            )
            lines.append(f"    sections: {', '.join(sampled)}")
        for label, snippet in samples:
            lines.append(f"    sample[{label}]: {snippet}")
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
        if mode == "author_terminal":
            return self._render_author_terminal_prompt(**kwargs)
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
        elif mode == "author_terminal":
            self._emit_author_terminal_decision(
                raw_text=raw_text, retry_count=retry_count, **call_context
            )
        else:  # pragma: no cover — defensive
            raise ValueError(
                f"unknown _emit_per_call_decision mode {mode!r}"
            )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _draft_to_band(chapters: List[Dict[str, Any]]) -> Tuple[int, int]:
        """(low, high) DRAFT-TO band for the Stage-1 prompt (WS4 §2).

        ~1 TO per major theme; the band is bumped to
        ``_DRAFT_TO_BAND_COLLAPSE`` when the input is a single chapter carrying
        more than ``_COLLAPSE_SECTION_THRESHOLD`` sections (the 141:1 collapse
        signature, where 1-4 drafts would grossly under-represent the book).
        """
        chapter_count = len(chapters)
        section_count = sum(
            len(c.get("sections") or []) for c in chapters
        )
        if (
            chapter_count == 1
            and section_count > _COLLAPSE_SECTION_THRESHOLD
        ):
            return _DRAFT_TO_BAND_COLLAPSE
        low, high = _DRAFT_TO_BAND_DEFAULT
        return (low, max(high, min(12, chapter_count)))

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
        lo, hi = self._draft_to_band(chapters)
        return (
            f"Course: {course_name}\n"
            f"Chapter count: {len(chapters)}\n\n"
            "Textbook chapters (id, title, sections, prose sample):\n"
            f"{skeleton}\n"
            f"Synthesize a course-level semantic outline and {lo}-{hi} "
            "DRAFT terminal objectives that together span the whole "
            "section range above (not only its first sections).\n"
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

    # ------------------------------------------------------------------
    # Structure-aware synthesis — per-window heading skeleton
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_skeleton_nodes(
        chapter: Dict[str, Any],
    ) -> List[Tuple[int, str]]:
        """Flatten one chapter's heading tree into ``(headingLevel, text)``.

        The chapter node itself leads (at its ``headingLevel``, default 1),
        followed by a depth-first walk of ``sections[]`` and their recursive
        ``subsections[]`` (the textbook_structure section shape). ``headingText``
        is preferred, falling back to ``title`` / ``heading``. Empty-text nodes
        are skipped. A section with no explicit ``headingLevel`` inherits one
        level below the chapter (so the deepest-first drop still has a
        well-ordered level to shed).
        """
        nodes: List[Tuple[int, str]] = []
        ch_level = _coerce_heading_level(chapter.get("headingLevel"), 1)
        ch_text = str(
            chapter.get("headingText")
            or chapter.get("title")
            or ""
        ).strip()
        if ch_text:
            nodes.append((ch_level, ch_text))

        def _walk(sections: Any, default_level: int) -> None:
            for sec in sections or []:
                if not isinstance(sec, dict):
                    continue
                lvl = _coerce_heading_level(sec.get("headingLevel"), default_level)
                txt = str(
                    sec.get("headingText")
                    or sec.get("title")
                    or sec.get("heading")
                    or ""
                ).strip()
                if txt:
                    nodes.append((lvl, txt))
                _walk(sec.get("subsections"), lvl + 1)

        _walk(chapter.get("sections"), ch_level + 1)
        return nodes

    @staticmethod
    def _render_heading_skeleton(chapter: Dict[str, Any]) -> str:
        """Render one chapter's compact, budget-capped heading skeleton.

        Deepest-heading-level-first truncation: while the rendered skeleton
        exceeds the ``_SKELETON_HEADING_TOKEN_BUDGET`` (≈ chars / 2.5), the nodes
        at the current MAXIMUM heading level are dropped, so the coarse chapter /
        section spine survives and only the fine-grained leaves are shed. The
        chapter-level heading is never dropped (a single surviving level breaks
        the loop). ``""`` when the chapter yields no headings.
        """
        nodes = TextbookSynthesisProvider._collect_skeleton_nodes(chapter)
        if not nodes:
            return ""
        budget_chars = int(
            _SKELETON_HEADING_TOKEN_BUDGET * _SKELETON_HEADING_CHARS_PER_TOKEN
        )

        def _rendered(ns: List[Tuple[int, str]]) -> str:
            if not ns:
                return ""
            min_level = min(lvl for lvl, _ in ns)
            lines: List[str] = []
            for lvl, text in ns:
                indent = "  " * max(0, lvl - min_level)
                lines.append(f"  {indent}- {text}")
            return "\n".join(lines)

        kept = list(nodes)
        while len(_rendered(kept)) > budget_chars:
            levels = {lvl for lvl, _ in kept}
            if len(levels) <= 1:
                break  # only the chapter spine remains — never drop it
            max_level = max(levels)
            kept = [(lvl, t) for lvl, t in kept if lvl != max_level]
        return _rendered(kept)

    def _build_window_heading_skeleton(self, chapter_id: str) -> str:
        """Heading skeleton for the window's chapter, or ``""`` when disabled.

        Empty (byte-identical prompt) when ``ED4ALL_SYNTHESIS_SKELETON`` is off,
        when no chapter matches ``chapter_id`` in the indexed textbook_structure,
        or when that chapter carries no headings. Deterministic — no LLM, no
        SemantiK sidecar read.
        """
        if not self._synthesis_skeleton_enabled:
            return ""
        chapter = self._skeleton_chapters_by_id.get(str(chapter_id or ""))
        if not isinstance(chapter, dict):
            return ""
        return self._render_heading_skeleton(chapter)

    def _render_window_objectives_prompt(
        self,
        *,
        chapter_id: str,
        window_index: int,
        window_chunks: List[Dict[str, Any]],
        course_name: str,
        draft_terminal_objectives: List[Dict[str, Any]],
        heading_skeleton: str = "",
    ) -> str:
        """W2 §2.1 — chunk-window candidate-objective prompt.

        ENUMERATES the window's allowed chunk ids + per-chunk bodies (the
        OUTLINE-tier discipline) and closes with the non-empty-subset
        ``source_chunk_ids`` contract so the model cites a COPY, not an
        inference. Drops the hardcoded ``source_refs:[{chunk_ids:[]}]`` sentinel
        of the legacy chapter prompt.

        ``heading_skeleton`` (``ED4ALL_SYNTHESIS_SKELETON``): when non-empty, a
        compact CONTEXT-ONLY heading-structure section for the window's chapter
        is inserted so TO/CO derivation is structure-aware. Empty → byte-identical
        legacy prompt.
        """
        draft_lines: List[str] = []
        for draft in draft_terminal_objectives or []:
            if isinstance(draft, dict):
                stmt = str(draft.get("statement") or "").strip()
                if stmt:
                    draft_lines.append(f"  - {stmt}")
        draft_block = "\n".join(draft_lines) if draft_lines else "  (none)"

        chunk_lines: List[str] = []
        for chunk in window_chunks or []:
            cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
            body = str(chunk.get("text") or chunk.get("body") or "")
            chunk_lines.append(f"  - [{cid}] {body}")
        chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (none)"

        skeleton_block = ""
        if heading_skeleton:
            skeleton_block = (
                "Document heading structure for this chapter (CONTEXT ONLY — "
                "to situate this window within the chapter's outline; do NOT "
                "cite headings — objectives still cite ONLY chunk ids):\n"
                f"{heading_skeleton}\n\n"
            )

        return (
            f"Course: {course_name}\n"
            f"Chapter id: {chapter_id}   (window {window_index})\n\n"
            "Course draft terminal objectives (for grounding):\n"
            f"{draft_block}\n\n"
            f"{skeleton_block}"
            "Source chunks (cite ONLY these ids in source_chunk_ids):\n"
            f"{chunks_block}\n\n"
            "Synthesize 1-3 candidate learning objectives this window's "
            "chunks teach.\n"
            f"{_BLOOM_DIVERSITY_DIRECTIVE}\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level key "
            '"candidate_objectives": [{"statement", "bloom_level" '
            '(lowercase enum), "bloom_verb", "abcd" {audience, behavior '
            '{verb, action_object}, condition, degree}, "source_chunk_ids": '
            '["<id>", ...], "sub_objectives": [short strings]}].\n'
            "For each objective, source_chunk_ids MUST be a non-empty subset "
            "of the chunk ids in the \"Source chunks\" section. An objective "
            "that synthesizes across N chunks carries N ids. NEVER cite an id "
            "not listed above. NEVER emit source_chunk_ids: [].\n"
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

    def _render_author_terminal_prompt(
        self,
        *,
        cluster_cos: List[Dict[str, Any]],
        course_name: str,
        **_unused: Any,
    ) -> str:
        """WS1 author prompt: ONE terminal objective over a CO cluster.

        Mirrors :meth:`_render_reconcile_prompt` but authors EXACTLY ONE TO and
        keeps the anti-invent guard. Top-level key ``terminal_objective``
        (singular).
        """
        co_lines: List[str] = []
        for co in cluster_cos or []:
            if isinstance(co, dict):
                stmt = str(co.get("statement") or "").strip()
                if stmt:
                    co_lines.append(f"  - {stmt}")
        return (
            f"Course: {course_name}\n\n"
            "Related chapter objectives (one cluster):\n"
            f"{chr(10).join(co_lines) if co_lines else '  (none)'}\n\n"
            "Author EXACTLY ONE terminal (course-wide) objective that "
            "captures the SINGLE higher-level competency the chapter "
            "objectives above share. Write ONE concise sentence with ONE "
            "main action verb. Do NOT produce a list — do NOT chain "
            "multiple distinct skills or topics with commas or the word "
            "'and'. When the objectives span several sub-topics, ABSTRACT "
            "upward to the unifying skill instead of enumerating them. "
            "Do NOT invent skills / topics / facts not present in the "
            "chapter objectives above.\n"
            "RESPOND ONLY WITH A JSON OBJECT with top-level key "
            '"terminal_objective": {"statement", "bloom_level" '
            '(lowercase enum), "bloom_verb", "abcd" {audience, '
            'behavior {verb, action_object}, condition, degree}, '
            '"source_refs" []}.\n'
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

    def _normalise_window_objectives_payload(
        self,
        parsed: Dict[str, Any],
        *,
        chapter_id: str,
        allowed_chunk_ids: set,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """W2 §2.2 — project + ENFORCE ⊆ on a window-objectives payload.

        Per emitted candidate:

        1. project statement/bloom/abcd via :meth:`_normalise_one_objective`
           (reused);
        2. read ``source_chunk_ids`` (tolerate ``chunk_ids`` /
           ``source_refs[].chunk_ids`` legacy key shapes defensively);
        3. intersect with ``allowed_chunk_ids`` (§2.3) — drop ids not in the
           set. Empty surviving set → ``grounded_citation: False`` +
           ``source_chunk_ids: []`` (NOT dropped here — Pass C owns drop vs
           re-ground). Non-empty → ``grounded_citation: True``;
        4. reconstruct ``source_refs: [{ref: chapter_id, chunk_ids: surviving}]``
           for on-disk back-compat AND keep the flat ``source_chunk_ids``.

        Returns ``(objectives, out_of_set_dropped, empty_citation_count)``.
        """
        raw_list = parsed.get("candidate_objectives") or []
        objectives: List[Dict[str, Any]] = []
        out_of_set_dropped = 0
        empty_citation = 0
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            obj = self._normalise_one_objective(raw)
            if obj is None:
                continue

            # Read the cited ids defensively across key shapes.
            cited: List[str] = []
            flat = raw.get("source_chunk_ids")
            if isinstance(flat, list):
                cited = [str(c) for c in flat if c]
            else:
                legacy = raw.get("chunk_ids")
                if isinstance(legacy, list):
                    cited = [str(c) for c in legacy if c]
                else:
                    refs = raw.get("source_refs")
                    if isinstance(refs, list):
                        for ref in refs:
                            if isinstance(ref, dict):
                                for cid in ref.get("chunk_ids") or []:
                                    if cid:
                                        cited.append(str(cid))

            # §2.3 — intersect with the allowed set; record hallucinated drops.
            surviving: List[str] = []
            for cid in cited:
                if cid in allowed_chunk_ids:
                    if cid not in surviving:
                        surviving.append(cid)
                else:
                    out_of_set_dropped += 1

            obj["source_chunk_ids"] = surviving
            obj["grounded_citation"] = bool(surviving)
            if not surviving:
                empty_citation += 1
            obj["source_refs"] = [
                {"ref": chapter_id, "chunk_ids": list(surviving)}
            ]

            sub = [
                str(s).strip()
                for s in (raw.get("sub_objectives") or [])
                if isinstance(s, str) and str(s).strip()
            ]
            obj["sub_objectives"] = sub
            obj["chapter_id"] = chapter_id
            objectives.append(obj)
        return objectives, out_of_set_dropped, empty_citation

    def _build_synthesis_grammar_payload(self) -> Dict[str, Any]:
        """W2 §3 — parameterless analogue of ``_build_grammar_payload``.

        ONE schema (``_WINDOW_OBJECTIVES_SCHEMA``) — the synthesis surface emits
        a single shape. Dispatch on ``(provider, base_url, grammar_mode)``
        mirrors the outline tier; the GBNF fallback REUSES the outline tier's
        ``_GENERIC_JSON_GBNF`` (imported, not duplicated). The grammar enforces
        shape + non-empty ``source_chunk_ids``; the ⊆ membership is enforced
        post-parse (§2.3).
        """
        # Lazy import to avoid pulling the outline-tier module at import time of
        # this provider; the GBNF string is the only symbol reused.
        try:
            from Courseforge.generators._outline_provider import (
                _GENERIC_JSON_GBNF,
            )
        except Exception:  # noqa: BLE001 — GBNF fallback is optional
            _GENERIC_JSON_GBNF = None  # type: ignore[assignment]

        schema = _WINDOW_OBJECTIVES_SCHEMA
        base_url = (self._base_url or "").lower()
        mode = (self._grammar_mode or "").lower() or None
        provider = self._provider

        # Explicit mode wins.
        if mode == "gbnf":
            return {"grammar": _GENERIC_JSON_GBNF} if _GENERIC_JSON_GBNF else {}
        if mode == "json_schema":
            return {"format": schema}
        if mode == "json_object":
            return {}
        if mode == "none":
            return {}

        # Auto-detect path (mirrors the outline tier).
        if provider == "together":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "WindowObjectives",
                        "schema": schema,
                        "strict": True,
                    },
                }
            }
        if provider in {"local", "openai_compatible"} or (
            self._registry_provider is not None
        ):
            if any(
                marker in base_url
                for marker in ("llama", "lmstudio", "lm-studio")
            ):
                return (
                    {"grammar": _GENERIC_JSON_GBNF}
                    if _GENERIC_JSON_GBNF
                    else {}
                )
            if "vllm" in base_url:
                return {"extra_body": {"guided_json": schema}}
            # Default for ``local`` (Ollama 0.5+): full-schema ``format``.
            return {"format": schema}

        # Anthropic / unrecognised — rely on JSON-mode + lenient parse.
        return {}

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

    def _normalise_author_terminal_payload(
        self,
        parsed: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """WS1 — project an author-terminal LLM payload onto ONE canonical TO.

        Reads ``parsed["terminal_objective"]`` (object) OR — leniently — the
        single element of a ``terminal_objectives`` list. Reuses
        :meth:`_normalise_one_objective`; does **NOT** mint an id (unlike
        :meth:`_normalise_reconcile_payload`; the caller mints ``TO-NN``).
        Drops any stray ``draft`` flag. Returns ``None`` when no usable
        objective is present.
        """
        raw = parsed.get("terminal_objective")
        if not isinstance(raw, dict):
            lst = parsed.get("terminal_objectives")
            if isinstance(lst, list) and lst and isinstance(lst[0], dict):
                raw = lst[0]
            else:
                return None
        obj = self._normalise_one_objective(raw)
        if obj is None:
            return None
        obj.pop("draft", None)
        return obj

    @staticmethod
    def _backfill_bloom_verb(statement: str, bloom_level: str) -> str:
        """Recover a missing ``bloom_verb`` from the objective's own statement.

        Used only when the model omitted the verb. Anti-fabrication: only a verb
        actually PRESENT in the statement is returned — never a copy of
        ``abcd.behavior.verb`` (which would silently force verb-triple agreement
        and mask a real misalignment the IB3 gate exists to catch). Prefers a
        concrete action verb over a bare Bloom *level name*, then one whose band
        matches the assigned level. When the statement carries no Bloom verb the
        field stays absent ("") and the gate legitimately flags it. Fail-soft:
        a taxonomy-load error returns "" (status quo).
        """
        try:
            from lib.ontology.bloom import detect_bloom_verbs  # noqa: PLC0415
        except Exception:  # noqa: BLE001 — taxonomy load is the floor
            return ""
        detected = detect_bloom_verbs(statement or "")
        if not detected:
            return ""
        non_level = [
            (lvl, v) for lvl, v in detected if v not in _BLOOM_LEVEL_ENUM
        ]
        pool = non_level or detected
        match = next((v for lvl, v in pool if lvl == bloom_level), None)
        return (match or pool[0][1]).strip().lower()

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
        if not bloom_verb:
            # The 7B sometimes omits bloom_verb entirely. Recover it from the
            # objective's OWN statement rather than leaving it absent (which
            # leaves the IB3 verb-triple gate nothing to compare against).
            bloom_verb = TextbookSynthesisProvider._backfill_bloom_verb(
                statement, bloom_level
            )
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
            f"section_count_input={int(ctx.get('section_count_input', 0))}",
            f"draft_to_band={tuple(ctx.get('draft_to_band', ()))}",
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
        # W2 §2.3 — surface the window-map ⊆-enforcement signals when the call
        # came from ``synthesize_window_objectives`` (absent on the legacy
        # chapter path, so only appended when present).
        if "window_index" in ctx:
            parts.extend([
                f"window_index={int(ctx.get('window_index', 0))}",
                f"allowed_chunk_id_count="
                f"{int(ctx.get('allowed_chunk_id_count', 0))}",
                f"cited_chunk_id_count="
                f"{int(ctx.get('cited_chunk_id_count', 0))}",
                f"citation_out_of_set_dropped="
                f"{int(ctx.get('citation_out_of_set_dropped', 0))}",
                f"empty_citation_objectives="
                f"{int(ctx.get('empty_citation_objectives', 0))}",
            ])
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

    def _emit_author_terminal_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **ctx: Any,
    ) -> None:
        """WS1 — emit one ``terminal_objective_authoring`` decision event.

        New LLM call site (mandatory decision-capture). Rationale interpolates
        ≥4 dynamic signals so the capture is replayable post-hoc.
        """
        course_name = ctx.get("course_name", "")
        success = bool(ctx.get("success", False))
        cluster_index = int(ctx.get("cluster_index", 0))
        parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"cluster_index={cluster_index}",
            f"co_count_input={int(ctx.get('co_count_input', 0))}",
            f"output_chars={int(ctx.get('output_chars', 0))}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if ctx.get("last_error"):
            parts.append(f"last_error={str(ctx['last_error'])[:120]}")
        self._emit_decision(
            decision_type="terminal_objective_authoring",
            decision=(
                f"terminal_objective_authoring:{course_name}:"
                f"cluster_{cluster_index}:"
                f"{'success' if success else 'failed'}"
            ),
            rationale="; ".join(parts),
        )


__all__ = [
    "TextbookSynthesisProvider",
    "TextbookSynthesisProviderError",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "ENV_MAX_TOKENS",
    "ENV_GRAMMAR_MODE",
    "DEFAULT_PROVIDER",
    "_DEFAULT_MAX_TOKENS",
    "_SYNTHESIS_NUM_CTX_ENV",
    "_MAX_SYNTHESIS_PARSE_RETRIES",
    "_WINDOW_OBJECTIVES_SCHEMA",
]
