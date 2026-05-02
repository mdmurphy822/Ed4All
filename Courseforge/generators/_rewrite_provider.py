#!/usr/bin/env python3
"""Courseforge rewrite-tier provider — pedagogical-depth HTML synthesis.

Phase 3 Subtasks 22-26. Sibling to
:class:`Courseforge.generators._provider.ContentGeneratorProvider`
(Phase 1) and :class:`Courseforge.generators._outline_provider.OutlineProvider`
(Phase 3 Subtasks 13-20). All three subclass
:class:`Courseforge.generators._base._BaseLLMProvider` so the HTTP
plumbing, decision-capture surface, and per-backend env-var resolution
stay in one place.

Tier responsibility:

- The outline tier (smaller, cheaper model — typically a 7B-class local
  Qwen) emits a structurally-correct outline dict per block (key claims,
  CURIEs to preserve, source refs, objective refs).
- The rewrite tier (larger, pedagogically-adept model — Anthropic
  Sonnet by default) consumes that outline dict and authors the rendered
  HTML body. The rewrite tier MUST preserve every CURIE the outline
  declared verbatim — drift would silently break the corpus's CURIE
  anchoring contract (root ``CLAUDE.md`` § Wave 135 + Wave 137 family
  completeness).

Operator selects the rewrite-tier backend via
``COURSEFORGE_REWRITE_PROVIDER`` (defaults to ``anthropic``) and the
rewrite-tier model via ``COURSEFORGE_REWRITE_MODEL`` (defaults to
``claude-sonnet-4-6``). The shared HTTP plumbing reuses the synthesis-
pipeline env vars (``ANTHROPIC_API_KEY`` / ``TOGETHER_API_KEY`` /
``LOCAL_SYNTHESIS_*``) so a single Ollama / Together / Anthropic
credentials surface serves both task surfaces.

Default config:

- ``max_tokens=2400`` (the rewrite tier authors a single block's HTML
  body, not a whole page; 2400 is the empirically-derived Pattern-22
  per-block budget).
- ``temperature=0.4`` (light authorial variation while keeping
  determinism viable for cache-keyed reruns; mirrors Phase 1's
  ContentGeneratorProvider default).

Public surface:

- :meth:`RewriteProvider.generate_rewrite` — the entry point the router
  calls; consumes a Block whose ``content`` is the outline-tier dict and
  returns a Block whose ``content`` is the rendered HTML body plus a
  cumulative ``Touch(tier="rewrite", purpose="pedagogical_depth", ...)``.

CURIE-preservation gate (Subtask 26): the rewrite tier asserts every
CURIE present in the input outline's ``content["curies"]`` survives into
the emitted HTML verbatim. On miss, the gate appends a remediation
turn naming the dropped CURIEs and retries up to ``MAX_PARSE_RETRIES``.
On exhaustion the call raises :class:`RewriteProviderError` with
``code="rewrite_curie_drop"`` so the router escalates upstream rather
than silently shipping CURIE-stripped HTML.

Direct port of the
:meth:`Trainforge.generators._local_provider.LocalSynthesisProvider._missing_preserve_tokens`
+ ``_append_preserve_remediation`` pattern (`Trainforge/generators/_local_provider.py:548-583`),
adapted to Block.content's outline-dict shape (the Trainforge precedent
operates on flat instruction / preference dicts).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from Courseforge.generators._base import _BaseLLMProvider

# Phase 2 Subtask 35: ``blocks.py`` lives at
# ``Courseforge/scripts/blocks.py``; mirror the import bridge from
# ``_provider.py`` so ``from blocks import Block`` resolves the same
# regardless of how this module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block, Touch  # noqa: E402  (Phase 2 intermediate format)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_PROVIDER = "COURSEFORGE_REWRITE_PROVIDER"
ENV_MODEL = "COURSEFORGE_REWRITE_MODEL"

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Per-backend defaults the rewrite tier passes through to the base. The
# rewrite tier prefers a larger / pedagogically-adept model than the
# outline tier even on the same backend, so the per-backend defaults
# differ from Phase 1's :class:`ContentGeneratorProvider`.
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_MODEL_TOGETHER = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_MODEL_LOCAL = "qwen2.5:14b-instruct-q4_K_M"

_DEFAULT_MAX_TOKENS = 2400
_DEFAULT_TEMPERATURE = 0.4

SUPPORTED_PROVIDERS = ("anthropic", "together", "local", "openai_compatible")

# Subtask 26: bounded remediation retries for the CURIE-preservation
# gate. Direct port of the Trainforge precedent
# (``_local_provider.py:540`` :: ``MAX_PARSE_RETRIES``).
MAX_PARSE_RETRIES = 2


# ---------------------------------------------------------------------------
# System prompt — Pattern-22 prevention contract + tier-specific contract.
# ---------------------------------------------------------------------------

# Pattern-22 prevention contract — verbatim port of Phase 1's
# ``Courseforge/generators/_provider.py::_SYSTEM_PROMPT`` (`:90-103`)
# so the rewrite-tier authoring constraints stay byte-stable with
# Phase 1's content-generator. Plus the rewrite-tier-specific paragraph
# the plan calls for (Subtask 22): preserve CURIEs / facts / refs;
# rewrite for pedagogical depth, scaffolding, examples, voice; never
# add facts not in the outline's ``key_claims`` or in the source
# chunks.
_REWRITE_SYSTEM_PROMPT = (
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
    "\n\n"
    "Outline is structurally correct but generated by a smaller model. "
    "PRESERVE: factual claims (verbatim), CURIEs (verbatim), objective "
    "refs, source refs. REWRITE: for pedagogical depth, scaffolding, "
    "examples, voice. DO NOT add facts not in the outline's key_claims "
    "or in the source chunks."
)


# ---------------------------------------------------------------------------
# Block-type → HTML output contract map (Subtask 24).
# ---------------------------------------------------------------------------
#
# Per-block-type HTML attribute contracts the rewrite tier must follow
# when authoring the rendered body. Mirrors ``Block.to_html_attrs``
# (`Courseforge/scripts/blocks.py:336-465`) so the rewrite output is
# downstream-extractable by the same priority chain Trainforge's
# ``process_course._extract_section_metadata`` walks.

_BLOCK_TYPE_OUTPUT_CONTRACTS: Dict[str, str] = {
    "objective": (
        "Emit a `<li>` carrying `data-cf-objective-id` (canonical "
        "TO-NN / CO-NN), `data-cf-bloom-level`, `data-cf-bloom-verb`, "
        "and `data-cf-cognitive-domain` attributes. The objective "
        "statement is the `<li>`'s text content."
    ),
    "concept": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` heading carrying `data-cf-content-type` + "
        "`data-cf-bloom-range` + `data-cf-key-terms`, followed by "
        "explanatory paragraphs."
    ),
    "example": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h3>` "
        "and `<p>` paragraphs presenting a worked example. Carry "
        "`data-cf-content-type=\"example\"` on the heading."
    ),
    "explanation": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` and explanatory paragraphs. Carry "
        "`data-cf-content-type=\"explanation\"` on the heading."
    ),
    "summary_takeaway": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h3>` "
        "and a short `<ul>` of takeaways. Carry "
        "`data-cf-content-type=\"summary\"` on the heading."
    ),
    "callout": (
        "Emit a `<div class=\"callout callout-{kind}\">` carrying "
        "`data-cf-component=\"callout\"` + `data-cf-purpose` + "
        "`data-cf-content-type=\"callout\"`."
    ),
    "flip_card_grid": (
        "Emit a `<div class=\"flip-card-grid\">` whose children are "
        "per-card `<div class=\"flip-card\">` elements carrying "
        "`data-cf-component=\"flip-card\"`, "
        "`data-cf-purpose=\"term-definition\"`, "
        "`data-cf-teaching-role`, and `data-cf-term`."
    ),
    "self_check_question": (
        "Emit a `<div class=\"self-check\">` carrying "
        "`data-cf-component=\"self-check\"`, "
        "`data-cf-purpose=\"formative-assessment\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids` / `data-cf-source-primary`."
    ),
    "activity": (
        "Emit a `<div class=\"activity-card\">` carrying "
        "`data-cf-component=\"activity\"`, `data-cf-purpose=\"practice\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids`."
    ),
    "misconception": (
        "Emit a `<section>` whose JSON-LD entry carries the "
        "misconception ID (mc_[0-9a-f]{16}) plus correction. The HTML "
        "body need not carry data-cf-* attributes — JSON-LD is the "
        "authoritative shape for misconception blocks."
    ),
    "assessment_item": (
        "Emit a `<div class=\"assessment-item\">` carrying the "
        "question stem, options, and correct-answer marker. "
        "Assessment items in IMSCC live in QTI XML downstream; the "
        "HTML emit here is the authoring fixture."
    ),
    "prereq_set": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` and an `<ol>` of prerequisite topic refs."
    ),
    "reflection_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` reflection prompts."
    ),
    "discussion_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` discussion prompts."
    ),
    "chrome": (
        "Emit page chrome (header / footer / nav). Carry "
        "`data-cf-role=\"template-chrome\"` on the wrapper."
    ),
    "recap": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h2>` or `<h3>` and a recap of the prior week's key terms."
    ),
}


def _block_type_output_contract(block_type: str) -> str:
    """Return the per-block-type HTML attribute contract paragraph.

    Falls back to a generic instruction when the block_type has no
    entry in the table — defensive only; ``Block.__post_init__``
    already validates the set.
    """
    return _BLOCK_TYPE_OUTPUT_CONTRACTS.get(
        block_type,
        (
            f"Emit the rendered HTML body for a block of type "
            f"{block_type!r}. Carry `data-cf-source-ids` on the top "
            f"wrapper to attribute the source chunks."
        ),
    )


def _safe_json_dumps(content: Any) -> str:
    """Serialize ``Block.content`` to a JSON string for the prompt.

    ``Block.content`` is ``Union[str, Dict[str, Any]]``. Strings pass
    through unchanged so the legacy Phase 1 path that emits a
    ``content=html`` Block still renders sensibly through the rewrite
    tier. Dicts are serialised with ``ensure_ascii=False`` so CURIEs
    (``sh:NodeShape``, ``rdfs:subClassOf``, …) survive verbatim —
    critical for the Subtask 26 CURIE-preservation gate.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(
            content, ensure_ascii=False, sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        # Defensive — a non-serialisable content payload would prevent
        # the rewrite tier from even seeing the outline. Surface a
        # readable repr instead so postmortem still has the data.
        logger.warning("Outline payload not JSON-serialisable: %s", exc)
        return repr(content)


def _format_source_chunks(chunks: Sequence[Any]) -> str:
    """Format the source-chunk list into a readable prompt block.

    Accepts either dict shape (``{"chunk_id": ..., "text": ...}``) or
    a chunk-like object exposing ``chunk_id`` / ``text`` attributes.
    Empty input renders ``"(none)"``.
    """
    if not chunks:
        return "(none)"
    parts: List[str] = []
    for c in chunks:
        if isinstance(c, dict):
            cid = c.get("chunk_id") or c.get("id") or "<unknown>"
            text = c.get("text") or c.get("content") or ""
        else:
            cid = (
                getattr(c, "chunk_id", None)
                or getattr(c, "id", None)
                or "<unknown>"
            )
            text = getattr(c, "text", "") or getattr(c, "content", "")
        parts.append(f"- [{cid}] {text}")
    return "\n".join(parts)


def _format_objectives(objectives: Sequence[Any]) -> str:
    """Format the objectives list into a readable prompt block.

    Accepts either dict shape (``{"id": ..., "statement": ...}``) or
    object with ``id`` + ``statement`` attributes. Empty input renders
    ``"(none)"``.
    """
    if not objectives:
        return "(none)"
    parts: List[str] = []
    for o in objectives:
        if isinstance(o, dict):
            oid = o.get("id") or "<unknown>"
            statement = o.get("statement") or ""
        else:
            oid = getattr(o, "id", "<unknown>")
            statement = getattr(o, "statement", "")
        parts.append(f"- {oid}: {statement}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RewriteProviderError(RuntimeError):
    """Raised when the rewrite tier cannot satisfy a structural / safety
    contract after exhausting its retry budget.

    The ``code`` discriminates the failure mode so the router and
    decision-capture rationale can branch on it without parsing the
    message string. Mirrors
    :class:`Trainforge.generators._anthropic_provider.SynthesisProviderError`.

    Codes:

    - ``rewrite_curie_drop`` — the rewrite output dropped one or more
      CURIEs declared in the input outline's ``content["curies"]`` and
      did not recover after ``MAX_PARSE_RETRIES`` remediation turns.
      ``missing_curies`` carries the dropped tokens for postmortem.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        missing_curies: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.missing_curies = list(missing_curies or [])


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class RewriteProvider(_BaseLLMProvider):
    """Rewrite-tier provider — turns an outline dict into rendered HTML.

    Subclass of :class:`_BaseLLMProvider`; reads tier-specific env vars
    (``COURSEFORGE_REWRITE_PROVIDER`` / ``COURSEFORGE_REWRITE_MODEL``)
    and forwards them through ``super().__init__(...)``. The base owns
    the dispatch / decision-capture plumbing.

    Public method:

    - :meth:`generate_rewrite` — consumes a Block whose ``content`` is
      the outline-tier dict (or a partial outline + ``escalation_marker``
      when the outline tier exhausted its budget) and returns a Block
      whose ``content`` is the rendered HTML body plus a cumulative
      ``Touch(tier="rewrite", purpose="pedagogical_depth", ...)``.

    Stub methods filled in by Subtasks 23-26:

    - :meth:`_render_user_prompt` (Subtask 24): standard rewrite prompt
      consuming Block.content as outline.
    - :meth:`_render_escalated_user_prompt` (Subtask 25): richer prompt
      template for blocks carrying a non-None ``escalation_marker``.
    - :meth:`generate_rewrite` (Subtask 26): the public entry point with
      the CURIE-preservation gate.
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
        # Tier-specific model resolution: ``COURSEFORGE_REWRITE_MODEL``
        # wins over the synthesis-pipeline ``ANTHROPIC_SYNTHESIS_MODEL``
        # / ``TOGETHER_SYNTHESIS_MODEL`` / ``LOCAL_SYNTHESIS_MODEL`` env
        # vars the base reads. We honor it here because the base only
        # reads the synthesis-pipeline vars by design (so a single Ollama
        # endpoint serves both task surfaces); the per-tier model knob
        # is the rewrite tier's own responsibility.
        import os
        resolved_model = model or os.environ.get(ENV_MODEL)

        # ``openai_compatible`` is reserved for a future plumbing pass
        # (the base currently routes ``local`` / ``together`` through
        # :class:`OpenAICompatibleClient` already, so the explicit
        # ``openai_compatible`` value would be redundant until the
        # base grows a separate branch). Until then, the rewrite tier
        # constructor accepts the same three the base accepts.
        super().__init__(
            provider=provider,
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
            default_model_anthropic=DEFAULT_MODEL_ANTHROPIC,
            default_model_together=DEFAULT_MODEL_TOGETHER,
            default_model_local=DEFAULT_MODEL_LOCAL,
            supported_providers=("anthropic", "together", "local"),
            system_prompt=_REWRITE_SYSTEM_PROMPT,
        )

    # _render_escalated_user_prompt filled in by Subtask 25.

    # generate_rewrite filled in by Subtask 26.

    # ------------------------------------------------------------------
    # User-prompt rendering (Subtask 24)
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        block: Block,
        source_chunks: Optional[Sequence[Any]] = None,
        objectives: Optional[Sequence[Any]] = None,
    ) -> str:
        """Render the rewrite-tier user prompt from an outline-dict Block.

        At this point the block's ``content`` field is the outline dict
        produced by the outline tier (typically containing
        ``key_claims`` / ``curies`` / ``source_refs`` / ``objective_refs``
        plus block-type-specific keys). The rewrite tier authors the
        rendered HTML body, preserving every CURIE the outline declared
        and citing the supplied source chunks.

        Sections of the prompt:

        - **Block context**: block_type / block_id / page_id.
        - **Outline**: ``json.dumps(block.content)`` so the model has
          the outline-tier dict verbatim as a single payload.
        - **Source chunks**: chunk text + chunk_id pairs (for
          ``data-cf-source-ids`` attribution).
        - **Objectives**: id + statement pairs (for
          ``data-cf-objective-id`` / ``data-cf-objective-ref``
          attribution).
        - **Output contract**: per-block-type HTML attribute contract
          (mirrors ``Block.to_html_attrs``).
        - **Final instruction**: emit ONLY rendered HTML, no markdown
          fences / commentary.
        """
        outline_payload = _safe_json_dumps(block.content)
        source_block = _format_source_chunks(source_chunks or [])
        objectives_block = _format_objectives(objectives or [])
        output_contract = _block_type_output_contract(block.block_type)

        return (
            f"Block type: {block.block_type}\n"
            f"Block id: {block.block_id}\n"
            f"Page id: {block.page_id}\n"
            "\n"
            "Outline (structurally correct, pedagogical-depth missing):\n"
            f"{outline_payload}\n"
            "\n"
            "Source chunks (cite via source_refs):\n"
            f"{source_block}\n"
            "\n"
            "Objectives:\n"
            f"{objectives_block}\n"
            "\n"
            "Output contract (HTML attributes for this block_type):\n"
            f"{output_contract}\n"
            "\n"
            "Author the rendered HTML body for this block now. Emit "
            "ONLY the HTML — no preamble, no markdown, no commentary."
        )

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        # Filled in by Subtask 26 (the rewrite-tier per-call decision
        # event lives alongside ``generate_rewrite``). For now emit a
        # generic capture so the abstract surface is satisfied — the
        # block_rewrite_call / content_generator_call enum is registered
        # in ``schemas/events/decision_event.schema.json`` already.
        self._emit_decision(
            decision_type="content_generator_call",
            decision=f"rewrite output chars={len(raw_text or '')}",
            rationale=(
                f"Rewrite tier dispatched provider={self._provider}, "
                f"model={self._model}, retry_count={retry_count}."
            ),
        )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_ANTHROPIC",
    "DEFAULT_MODEL_LOCAL",
    "DEFAULT_MODEL_TOGETHER",
    "DEFAULT_PROVIDER",
    "ENV_MODEL",
    "ENV_PROVIDER",
    "MAX_PARSE_RETRIES",
    "RewriteProvider",
    "RewriteProviderError",
    "SUPPORTED_PROVIDERS",
    "_REWRITE_SYSTEM_PROMPT",
]
