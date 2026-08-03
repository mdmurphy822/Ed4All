#!/usr/bin/env python3
"""Courseforge generators — course-outliner provider (Wave W-D14).

The course-outliner surface synthesizes canonical ``TO-NN`` / ``CO-NN``
learning objectives from a parsed ``textbook_structure.json``. Pre-W-D14
this surface ran exclusively as a Claude Code subagent dispatched via
``ED4ALL_AGENT_DISPATCH=true`` — a configuration that routes synthesised
objective text through Anthropic, which the licensing posture in
``docs/LICENSING.md`` flags as training-data exposure for derivative
SLM corpora.

W-D14 ships an in-process LLM seam mirroring the W-D11.A ``COURSEFORGE_PROVIDER``
short-circuit pattern (`Courseforge/generators/_provider.py`):

- Operator sets ``COURSEPLANNER_PROVIDER=local`` (or ``together`` / any
  registered OpenAI-compatible provider) → ``MCP/core/executor.py``
  bypasses the Claude Code subagent dispatch for the ``course-outliner``
  agent and falls through to the in-process registry tool
  ``plan_course_structure``, which constructs an :class:`OutlinerProvider`
  and uses it to author the LOs.
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

Public surface:

- :class:`OutlinerProvider` — sub-class of
  :class:`Courseforge.generators._base._BaseLLMProvider`.
- :meth:`OutlinerProvider.synthesize_objectives` — emits the canonical
  ``synthesized_objectives.json`` shape consumed by
  ``MCP/tools/pipeline_tools.py::_plan_course_structure`` (terminal +
  chapter objectives with ABCD verbs and source_refs) under the
  ``COURSEPLANNER_PROVIDER`` opt-in.
- :data:`ENV_PROVIDER` — ``"COURSEPLANNER_PROVIDER"``.
- :data:`DEFAULT_PROVIDER` — ``"anthropic"`` (the class's hardcoded
  default; the WORKFLOW default is "no in-process provider — defer to
  the deterministic synthesizer" and is enforced upstream by
  ``_plan_course_structure`` checking ``COURSEPLANNER_PROVIDER`` for a
  non-empty value).

Decision-capture: every dispatch emits exactly one
``decision_type="course_outline_call"`` event whose rationale
interpolates dynamic per-call signals (provider, model, course_name,
chapter_count, weeks, terminal_count, chapter_count_emit, retry_count,
output character count). The ``provider`` field is the runtime value
the operator selected — NOT a static label — so a post-hoc audit can
attribute every objective synthesis call to the OSS / cloud / local
backend that produced it.
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
from lib.ontology.learning_objectives import mint_lo_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — env vars + defaults
# ---------------------------------------------------------------------------

ENV_PROVIDER = "COURSEPLANNER_PROVIDER"
ENV_MODEL = "COURSEPLANNER_MODEL"

# DEFAULT_PROVIDER preserves legacy class-body behavior. The WORKFLOW
# default is "no in-process provider unless COURSEPLANNER_PROVIDER is
# explicitly set", enforced by the registry-tool short-circuit at
# ``_plan_course_structure``.
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


_DEFAULT_MAX_TOKENS = 3072
_DEFAULT_TEMPERATURE = 0.2

# Compact system prompt — the prompt body has to fit a 7B-class local
# model's instruction-following window. Mirrors the ≤80-word terseness
# of ``Courseforge/generators/outline/_outline_provider.py::_OUTLINE_SYSTEM_PROMPT``.
_OUTLINER_SYSTEM_PROMPT = (
    "You are a Courseforge course-outliner authoring canonical learning "
    "objectives for a single course from its textbook structure. Emit "
    "ONLY a JSON object — no preamble, no markdown fences, no commentary. "
    "Every objective must follow the ABCD framework (Audience, Behavior, "
    "Condition, Degree). Behavior verbs MUST come from the canonical "
    "Bloom's taxonomy (remember/understand/apply/analyze/evaluate/create). "
    "Terminal objectives (course-wide) carry ID prefix TO-; chapter "
    "objectives carry CO-. Number them sequentially starting at 01 within "
    "each prefix (TO-01, TO-02, ...; CO-01, CO-02, ...). Do NOT invent "
    "facts not in the supplied chapters[]; ground every objective in a "
    "real chapter or section heading."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutlinerProviderError(RuntimeError):
    """Course-outliner dispatch / parse / validation failure.

    Carries an opaque ``code`` field so callers can branch on the
    failure mode without parsing the message string.

    Canonical codes:

    - ``outliner_exhausted`` — every parse retry returned non-JSON or
      structurally-invalid JSON; the LLM tier failed to produce a
      usable objectives payload.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


_BLOOM_LEVEL_ENUM: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


_LO_ID_RE = re.compile(r"^[A-Z]{2,}-\d{2,}$")


class OutlinerProvider(_BaseLLMProvider):
    """LLM-agnostic course-outliner provider.

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

    - :meth:`synthesize_objectives` — emits the canonical
      ``synthesized_objectives.json`` shape consumed by
      ``MCP/tools/pipeline_tools.py::_plan_course_structure``.

    Decision-capture: every successful or failed dispatch emits
    exactly one ``decision_type="course_outline_call"`` event whose
    rationale interpolates dynamic per-call signals (provider, model,
    course_name, chapter_count, weeks, terminal_count,
    chapter_count_emit, retry_count, output character count). The
    ``provider`` field is the runtime value the operator selected so
    a post-hoc audit can attribute every synthesis call to the OSS /
    cloud / local backend that produced it.
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
        # the W-D12 / W-D11 contract: per-call kwarg > per-tier env var
        # (``COURSEPLANNER_MODEL``) > per-backend baseline (resolved by
        # the base from synthesis env vars).
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
                f"OutlinerProvider: unknown provider {resolved_provider!r}; "
                f"expected one of {list(supported)}. Add a registry entry "
                f"to _OPENAI_COMPATIBLE_PROVIDERS in "
                f"MCP/orchestrator/llm_backend.py — no subclassing required."
            )
        self._registry_provider: Optional[str] = None
        if (
            # ``nvidia`` is base-NATIVE (``_base.py`` resolves NVIDIA_API_KEY
            # + the nvidia base_url directly), so it must NOT route through
            # the W-D12 registry override (which wires the base under "local"
            # and would send the "local" placeholder api_key -> HTTP 401).
            # Other registry providers (groq/fireworks/deepseek) keep it.
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
            supported_providers=("anthropic", "together", "local", "nvidia"),
            system_prompt=_OUTLINER_SYSTEM_PROMPT,
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

    def synthesize_objectives(
        self,
        textbook_structure: Dict[str, Any],
        *,
        course_name: str,
        chapter_count: Optional[int] = None,
        weeks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synthesize canonical TO-NN / CO-NN objectives.

        Args:
            textbook_structure: Parsed ``textbook_structure.json`` shape
                — at minimum ``{"chapters": [{"id", "title", "sections":
                [{...}]}]}``. Other fields are passed verbatim into the
                user prompt so the model can ground every objective.
            course_name: Course slug used in the decision-capture
                rationale and the prompt header.
            chapter_count: Optional override for the chapter count
                published to the prompt; falls back to
                ``len(textbook_structure["chapters"])``.
            weeks: Optional pacing hint for the LLM. The
                ``_plan_course_structure`` re-scaling pass at
                ``WAVE18_COS_PER_WEEK`` is authoritative; this is just
                a prompt-time signal.

        Returns:
            A dict carrying the canonical synthesized_objectives shape:

            ``{
                "course_name": str,
                "terminal_objectives": [{"id": "TO-NN", "statement": ...,
                    "bloom_level": ..., "bloom_verb": ...,
                    "abcd": {...}, "source_refs": [...], ...}],
                "chapter_objectives": [{"chapter": "Week N",
                    "objectives": [{"id": "CO-NN", ...}]}],
                "mint_method": "courseplanner_provider:<provider>",
                "duration_weeks": int,
              }``

            ``_plan_course_structure`` reads this dict and folds it
            into the on-disk JSON at
            ``{project_path}/01_learning_objectives/synthesized_objectives.json``
            so downstream phases consume the exact same shape they
            consume from the deterministic synthesizer.

        Raises:
            ValueError: When ``course_name`` is empty.
            OutlinerProviderError(code="outliner_exhausted"): When the
                LLM tier fails to emit usable JSON across the parse-retry
                budget.
        """
        if not course_name or not str(course_name).strip():
            raise ValueError("course_name required")
        if textbook_structure is None or not isinstance(
            textbook_structure, dict
        ):
            raise ValueError("textbook_structure must be a dict")

        chapters = textbook_structure.get("chapters") or []
        resolved_chapter_count = (
            int(chapter_count)
            if chapter_count is not None
            else len(chapters)
        )
        resolved_weeks = int(weeks) if weeks is not None else 0

        user_prompt = self._render_user_prompt(
            textbook_structure=textbook_structure,
            course_name=course_name,
            chapter_count=resolved_chapter_count,
            weeks=resolved_weeks,
        )

        text, retry_count = self._dispatch_call(user_prompt)

        # Lenient parse — accept either a clean JSON object or a
        # markdown-fenced one. Mirrors
        # ``OpenAICompatibleClient._extract_json_lenient`` semantics
        # without importing it (works on Anthropic SDK output too).
        parsed = self._extract_json_lenient(text)

        last_error: Optional[str] = None
        if parsed is None:
            last_error = "lenient JSON parse returned None"

        normalised: Optional[Dict[str, Any]] = None
        if parsed is not None:
            try:
                normalised = self._normalise_objectives_payload(
                    parsed,
                    course_name=course_name,
                    weeks=resolved_weeks,
                )
            except ValueError as exc:
                last_error = f"normalisation failed: {exc}"

        # Always emit the decision-capture event — success OR failure —
        # so the audit trail records every dispatch.
        self._emit_per_call_decision(
            raw_text=text,
            retry_count=retry_count,
            course_name=course_name,
            chapter_count=resolved_chapter_count,
            weeks=resolved_weeks,
            success=normalised is not None,
            terminal_count=(
                len(normalised.get("terminal_objectives", []))
                if normalised is not None
                else 0
            ),
            chapter_count_emit=(
                self._count_chapter_objectives(normalised)
                if normalised is not None
                else 0
            ),
            last_error=last_error,
        )

        if normalised is None:
            raise OutlinerProviderError(
                f"OutlinerProvider exhausted parse on course "
                f"{course_name!r} (last_error={last_error!r})",
                code="outliner_exhausted",
            )
        return normalised

    # ------------------------------------------------------------------
    # User prompt rendering
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        textbook_structure: Dict[str, Any],
        course_name: str,
        chapter_count: int,
        weeks: int,
        # Required by the abstract base — unused for course-outliner.
        **_unused: Any,
    ) -> str:
        """Render the course-outliner user prompt.

        The prompt embeds:

        - Course header (course_name, chapter_count, weeks).
        - Chapters block: id + title + per-section heading list, with
          per-chapter bodies truncated at 600 chars so a thick textbook
          structure doesn't blow the model's context window.
        - One-shot example so smaller local models (Qwen-2.5-7B etc.)
          have a concrete shape to imitate.
        - Strict-JSON closing directive.
        """
        chapter_lines: List[str] = []
        for chapter in textbook_structure.get("chapters", []) or []:
            cid = str(chapter.get("id") or "")
            title = str(chapter.get("title") or "")
            sections = chapter.get("sections") or []
            section_titles: List[str] = []
            for sec in sections:
                stitle = str(sec.get("title") or sec.get("heading") or "")
                if stitle:
                    section_titles.append(stitle)
            section_blob = ", ".join(section_titles[:8])
            if len(section_blob) > 600:
                section_blob = section_blob[:597] + "..."
            chapter_lines.append(
                f"  - [{cid}] {title}"
                + (f"\n    sections: {section_blob}" if section_blob else "")
            )
        chapters_block = (
            "\n".join(chapter_lines) if chapter_lines else "  (none)"
        )

        # One-shot example pinned to the canonical shape. Helps a
        # 7B-class local model anchor on the JSON structure.
        oneshot = (
            "Example output (illustrative — do NOT copy values):\n"
            "{\n"
            '  "terminal_objectives": [\n'
            '    {"id": "TO-01", "statement": "Students will analyze ...",\n'
            '     "bloom_level": "analyze", "bloom_verb": "analyze",\n'
            '     "abcd": {"audience": "Students", "behavior": '
            '{"verb": "analyze", "action_object": "the structure"}, '
            '"condition": "given a textbook chapter", '
            '"degree": "with at least 80% accuracy"},\n'
            '     "source_refs": [{"ref": "ch1", "chunk_ids": []}]}\n'
            "  ],\n"
            '  "chapter_objectives": [\n'
            '    {"id": "CO-01", "statement": "...", "bloom_level": "apply",\n'
            '     "bloom_verb": "apply", "abcd": {...}, '
            '"source_refs": [{"ref": "ch1", "chunk_ids": []}]}\n'
            "  ]\n"
            "}"
        )

        return (
            f"Course: {course_name}\n"
            f"Chapter count: {chapter_count}\n"
            f"Target weeks: {weeks}\n\n"
            "Textbook chapters (preserve every chapter id verbatim in the "
            "objectives' source_refs[].ref):\n"
            f"{chapters_block}\n\n"
            f"{oneshot}\n\n"
            "Synthesize 1-2 terminal objectives covering the whole course "
            "and one chapter objective per chapter (CO-01..CO-NN). Every "
            "objective MUST carry the canonical fields: id, statement, "
            "bloom_level (lowercase enum), bloom_verb, abcd "
            "{audience, behavior {verb, action_object}, condition, "
            "degree}, source_refs []. "
            "RESPOND ONLY WITH A JSON OBJECT containing top-level "
            "keys terminal_objectives and chapter_objectives. No "
            "preamble, no markdown fences, no commentary."
        )

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    def _normalise_objectives_payload(
        self,
        parsed: Dict[str, Any],
        *,
        course_name: str,
        weeks: int,
    ) -> Dict[str, Any]:
        """Project the LLM output onto the canonical synthesized shape.

        The deterministic synthesizer at
        ``MCP/tools/_content_gen_helpers.py::synthesize_objectives_from_topics``
        emits the same shape; the registry-tool consumer
        ``_plan_course_structure`` writes the result to
        ``synthesized_objectives.json``. Mirroring that shape here lets
        the in-process LLM provider drop in as a byte-compatible
        replacement when ``COURSEPLANNER_PROVIDER`` is set.

        Required projection rules:

        - Every objective gets a re-minted ``id`` via
          :func:`mint_lo_id` (TO- for terminal, CO- for chapter) so
          IDs are sequential and 2+-digit zero-padded regardless of
          the LLM's emit.
        - ``bloom_level`` is lowercased and validated against
          :data:`_BLOOM_LEVEL_ENUM`; missing / unknown values fall back
          to ``"understand"`` (the most common default for chapter-LO
          authoring).
        - ``source_refs`` defaults to ``[]`` when absent.
        - ``chapter_objectives`` is wrapped in the
          ``{"chapter": "Week N", "objectives": [...]}`` group shape
          the canonical synthesizer emits.
        """
        if not isinstance(parsed, dict):
            raise ValueError(
                f"synthesize_objectives expected a JSON object, got "
                f"{type(parsed).__name__}"
            )
        terminal_raw = parsed.get("terminal_objectives") or []
        chapter_raw = parsed.get("chapter_objectives") or []

        # Some models emit chapter_objectives as the canonical
        # group-of-groups shape; tolerate both flat and grouped on input.
        chapter_flat: List[Dict[str, Any]] = []
        for entry in chapter_raw:
            if not isinstance(entry, dict):
                continue
            if "objectives" in entry and isinstance(entry["objectives"], list):
                chapter_flat.extend(
                    o for o in entry["objectives"] if isinstance(o, dict)
                )
            else:
                chapter_flat.append(entry)

        terminal_norm: List[Dict[str, Any]] = []
        for idx, raw in enumerate(terminal_raw, start=1):
            if not isinstance(raw, dict):
                continue
            entry = self._normalise_one_objective(raw)
            entry["id"] = mint_lo_id("terminal", idx)
            terminal_norm.append(entry)

        chapter_norm: List[Dict[str, Any]] = []
        for idx, raw in enumerate(chapter_flat, start=1):
            if not isinstance(raw, dict):
                continue
            entry = self._normalise_one_objective(raw)
            entry["id"] = mint_lo_id("chapter", idx)
            chapter_norm.append(entry)

        if not terminal_norm and not chapter_norm:
            raise ValueError(
                "synthesize_objectives: LLM output carried no usable "
                "terminal_objectives[] or chapter_objectives[]"
            )

        # Wrap chapter list in the canonical group-of-groups shape.
        chapter_groups: List[Dict[str, Any]] = []
        for idx, c in enumerate(chapter_norm, start=1):
            chapter_groups.append({
                "chapter": f"Week {idx}",
                "objectives": [dict(c)],
            })

        return {
            "course_name": course_name,
            "terminal_objectives": terminal_norm,
            "chapter_objectives": chapter_groups,
            "mint_method": f"courseplanner_provider:{self._provider}",
            "duration_weeks": int(weeks) if weeks else max(8, len(chapter_norm)),
        }

    @staticmethod
    def _normalise_one_objective(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Project a single LLM-emitted objective onto the canonical shape."""
        entry: Dict[str, Any] = {}
        statement = str(raw.get("statement") or raw.get("text") or "").strip()
        if not statement:
            # Skip silently downstream by raising — the caller filters.
            raise ValueError("objective missing 'statement'")
        entry["statement"] = statement

        bloom_level = str(raw.get("bloom_level") or "").strip().lower()
        if bloom_level not in _BLOOM_LEVEL_ENUM:
            bloom_level = "understand"
        entry["bloom_level"] = bloom_level

        bloom_verb = str(raw.get("bloom_verb") or "").strip().lower()
        if bloom_verb:
            entry["bloom_verb"] = bloom_verb

        abcd = raw.get("abcd")
        if isinstance(abcd, dict):
            # Pass through verbatim — the AbcdObjectiveValidator at
            # ``course_planning`` audits the verb-Bloom alignment, so
            # we don't second-guess shape here. Down-stream gates will
            # warn if the ABCD verb diverges from bloom_level.
            entry["abcd"] = dict(abcd)
            inner = abcd.get("behavior")
            if isinstance(inner, dict):
                entry["abcd"]["behavior"] = dict(inner)

        source_refs = raw.get("source_refs")
        if isinstance(source_refs, list):
            entry["source_refs"] = list(source_refs)
        else:
            entry["source_refs"] = []

        key_concepts = raw.get("key_concepts")
        if isinstance(key_concepts, list):
            entry["key_concepts"] = [
                str(k) for k in key_concepts if isinstance(k, str)
            ]

        return entry

    @staticmethod
    def _count_chapter_objectives(payload: Optional[Dict[str, Any]]) -> int:
        """Count flattened chapter objectives in a normalised payload."""
        if payload is None:
            return 0
        total = 0
        for grp in payload.get("chapter_objectives", []) or []:
            if isinstance(grp, dict):
                total += len(grp.get("objectives") or [])
        return total

    @staticmethod
    def _extract_json_lenient(text: str) -> Optional[Dict[str, Any]]:
        """Lenient JSON object extractor.

        Handles three common LLM response shapes:

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
        """Emit one ``course_outline_call`` decision-capture event.

        Rationale interpolates dynamic per-call signals so a post-hoc
        replay records the EXACT runtime values: which provider fired,
        which model produced the output, how many objectives the LLM
        emitted, the parse / dispatch retry count, success/failure, and
        the last_error when the dispatch failed. Per the project's LLM
        call-site instrumentation contract: ≥20 chars, dynamic signals,
        provider value runtime not static.
        """
        course_name = call_context.get("course_name", "")
        chapter_count = int(call_context.get("chapter_count", 0))
        weeks = int(call_context.get("weeks", 0))
        success = bool(call_context.get("success", False))
        terminal_count = int(call_context.get("terminal_count", 0))
        chapter_count_emit = int(call_context.get("chapter_count_emit", 0))
        last_error = call_context.get("last_error")

        decision = (
            f"course_outline_call:{course_name}:"
            f"{'success' if success else 'failed'}"
        )
        rationale_parts = [
            f"course_name={course_name}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"chapter_count_input={chapter_count}",
            f"weeks_input={weeks}",
            f"terminal_count_emit={terminal_count}",
            f"chapter_count_emit={chapter_count_emit}",
            f"output_chars={len(raw_text or '')}",
            f"retry_count={retry_count}",
            f"success={success}",
        ]
        if last_error:
            rationale_parts.append(f"last_error={str(last_error)[:120]}")
        rationale = "; ".join(rationale_parts)

        self._emit_decision(
            decision_type="course_outline_call",
            decision=decision,
            rationale=rationale,
        )


__all__ = [
    "OutlinerProvider",
    "OutlinerProviderError",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "DEFAULT_PROVIDER",
]
