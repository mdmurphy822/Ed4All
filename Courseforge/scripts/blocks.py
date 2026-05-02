"""Phase 2 intermediate block format.

Defines the canonical in-memory ``Block`` + ``Touch`` dataclasses that
``generate_course.py`` will project to (a) ``data-cf-*`` HTML attribute
strings on emit and (b) JSON-LD entries inside ``<script type=
"application/ld+json">`` blocks. Blocks are the stable intermediate
between the Phase 1 ``ContentGeneratorProvider`` (which currently
returns ``str`` HTML) and the legacy renderer surface; Phase 2 widens
the provider to return ``Block`` instances so the renderer composes
attribute strings from a typed object instead of regex-parsing back
out of the LLM's HTML.

The dataclass is intentionally frozen — Phase 2 mutations (touch chain
appends, validation-attempt increments, escalation marking) all return
new instances via ``dataclasses.replace``. Three feedback-driven fields
support Phase 3's per-block regeneration budget + escalation primitive:
``validation_attempts`` (incremented per failed validator pass) and
``escalation_marker`` (set when a block is escalated to the rewrite
tier after the outline-tier budget is exhausted).
"""

from __future__ import annotations

import dataclasses
import hashlib
import html as _html_mod
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ["Block", "Touch", "BLOCK_TYPES"]


# Phase-2 emit flag (mirror of ``generate_course._courseforge_emit_blocks_enabled``;
# the helper here lives at module level so :class:`Block` can append the new
# ``data-cf-block-id`` attribute without importing the larger renderer module).
_EMIT_BLOCKS_ENV = "COURSEFORGE_EMIT_BLOCKS"
_EMIT_BLOCKS_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _emit_blocks_enabled() -> bool:
    """Read ``COURSEFORGE_EMIT_BLOCKS`` each call so tests can toggle it.

    Default off — the new ``data-cf-block-id`` attribute is purely additive
    and must not break byte-stable emit until the Phase 2 migration window
    closes (per pre-resolved decision #8).
    """
    return os.environ.get(_EMIT_BLOCKS_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


def _esc(text: str) -> str:
    """HTML-escape mirroring ``html.escape`` (matches ``html_mod.escape`` in generate_course.py)."""
    return _html_mod.escape(text)


def _source_attr_string(
    source_ids: Tuple[str, ...],
    source_primary: Optional[str],
) -> str:
    """Wave 9 source attribute string — mirrors ``generate_course._source_attr_string``.

    Inlined here so :meth:`Block.to_html_attrs` does not need to import the
    renderer module (avoids a cyclic import once Round 3 lands the renderer
    migration).
    """
    if not source_ids:
        return ""
    joined = ",".join(_esc(sid) for sid in source_ids if sid)
    out = f' data-cf-source-ids="{joined}"'
    if source_primary:
        out += f' data-cf-source-primary="{_esc(source_primary)}"'
    return out


BLOCK_TYPES: frozenset = frozenset(
    {
        "objective",
        "concept",
        "example",
        "assessment_item",
        "explanation",
        "prereq_set",
        "activity",
        "misconception",
        "callout",
        "flip_card_grid",
        "self_check_question",
        "summary_takeaway",
        "reflection_prompt",
        "discussion_prompt",
        "chrome",
        "recap",
    }
)


_TOUCH_TIERS: frozenset = frozenset({"outline", "validation", "rewrite"})

_TOUCH_PROVIDERS: frozenset = frozenset(
    {"anthropic", "local", "together", "claude_session", "deterministic"}
)

_ESCALATION_MARKERS: frozenset = frozenset(
    {
        "outline_budget_exhausted",
        "structural_unfixable",
        "validator_consensus_fail",
    }
)


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lowercase + non-alnum-to-underscore + strip + max 40 chars.

    Mirrors Courseforge's existing slug helper (lowercase + space-to-
    underscore + non-alnum collapse) so block IDs concatenate cleanly
    with page IDs.
    """
    if not text:
        return ""
    lowered = text.strip().lower()
    collapsed = _SLUG_NON_ALNUM.sub("_", lowered).strip("_")
    return collapsed[:40]


@dataclass(frozen=True)
class Touch:
    """One revision attribution event in a Block's touch chain.

    Each tier (``outline`` / ``validation`` / ``rewrite``) emits a
    ``Touch`` when it modifies the block. The chain is cumulative — the
    audit value of the full chain is bounded by ~12k entries × ~80
    bytes ≈ ~1 MB JSON before gzip per course (well within IMSCC payload
    budgets), so retention is full per pre-resolved decision #2 in the
    Phase 2 plan.
    """

    model: str
    provider: str
    tier: str
    timestamp: str
    decision_capture_id: str
    purpose: str

    def __post_init__(self) -> None:
        if not self.decision_capture_id:
            raise ValueError(
                "Touch.decision_capture_id required (Wave 112 invariant)"
            )
        if self.tier not in _TOUCH_TIERS:
            raise ValueError(
                f"Touch.tier must be one of {sorted(_TOUCH_TIERS)}; got {self.tier!r}"
            )
        if self.provider not in _TOUCH_PROVIDERS:
            raise ValueError(
                f"Touch.provider must be one of {sorted(_TOUCH_PROVIDERS)}; "
                f"got {self.provider!r}"
            )

    def to_jsonld(self) -> Dict[str, Any]:
        """Wire shape — camelCase keys for JSON-LD emit."""
        return {
            "model": self.model,
            "provider": self.provider,
            "tier": self.tier,
            "timestamp": self.timestamp,
            "decisionCaptureId": self.decision_capture_id,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class Block:
    """Canonical intermediate block.

    Owns its identity (``block_id`` / ``page_id`` / ``sequence``) plus
    the pedagogical metadata the renderer + JSON-LD builder consume.
    Frozen — mutations return a new instance via ``dataclasses.replace``.

    Phase 3 feedback-driven fields:
        ``validation_attempts``: incremented by the outline-tier
            regeneration router on every failed validator pass.
        ``escalation_marker``: set to a non-empty marker (one of
            ``_ESCALATION_MARKERS``) when the block exhausts its
            outline-tier budget and is escalated to the rewrite tier.
    Both stay default (``0`` / ``None``) for blocks emitted by the
    deterministic / Phase-1-provider paths in Phase 2.
    """

    block_id: str
    block_type: str
    page_id: str
    sequence: int
    content: Union[str, Dict[str, Any]]
    template_type: Optional[str] = None
    key_terms: Tuple[str, ...] = ()
    objective_ids: Tuple[str, ...] = ()
    bloom_level: Optional[str] = None
    bloom_verb: Optional[str] = None
    bloom_range: Optional[str] = None
    bloom_levels: Tuple[str, ...] = ()
    bloom_verbs: Tuple[str, ...] = ()
    cognitive_domain: Optional[str] = None
    teaching_role: Optional[str] = None
    content_type_label: Optional[str] = None
    purpose: Optional[str] = None
    component: Optional[str] = None
    source_ids: Tuple[str, ...] = ()
    source_primary: Optional[str] = None
    source_references: Tuple[Dict[str, Any], ...] = ()
    touched_by: Tuple[Touch, ...] = ()
    content_hash: Optional[str] = None
    validation_attempts: int = 0
    escalation_marker: Optional[str] = None

    def __post_init__(self) -> None:
        if self.block_type not in BLOCK_TYPES:
            raise ValueError(
                f"Block.block_type must be one of {sorted(BLOCK_TYPES)}; "
                f"got {self.block_type!r}"
            )
        if self.sequence < 0:
            raise ValueError(
                f"Block.sequence must be >= 0; got {self.sequence}"
            )
        if not self.page_id:
            raise ValueError("Block.page_id must be non-empty")
        if self.validation_attempts < 0:
            raise ValueError(
                f"Block.validation_attempts must be >= 0; "
                f"got {self.validation_attempts}"
            )
        if (
            self.escalation_marker is not None
            and self.escalation_marker not in _ESCALATION_MARKERS
        ):
            raise ValueError(
                f"Block.escalation_marker must be None or one of "
                f"{sorted(_ESCALATION_MARKERS)}; got {self.escalation_marker!r}"
            )

    @classmethod
    def stable_id(cls, page_id: str, block_type: str, slug: str, idx: int) -> str:
        """Position-based block ID per pre-resolved decision #1.

        Format: ``{page_id}#{block_type}_{slug}_{idx}``. Hash-based IDs
        are deferred — bottom-up migration produces stable orderings
        per renderer; reorder churn is rare.
        """
        return f"{page_id}#{block_type}_{slug}_{idx}"

    def with_touch(self, touch: Touch) -> "Block":
        """Return a new Block with ``touch`` appended to ``touched_by``.

        The content hash is unchanged — touches are audit-only and
        excluded from the canonical hash payload.
        """
        return dataclasses.replace(self, touched_by=self.touched_by + (touch,))

    def compute_content_hash(self) -> str:
        """SHA-256 hex of the canonical Block payload.

        Excludes ``touched_by``, ``sequence``, ``validation_attempts``,
        and ``escalation_marker`` so a touch-only or budget-only
        revision keeps a stable hash. The hash exists for re-execution
        drift detection — same content → same hash regardless of
        which tier authored it or how many times it was retried.
        """
        payload = {
            "content": self.content,
            "block_type": self.block_type,
            "key_terms": list(self.key_terms),
            "bloom_level": self.bloom_level,
            "objective_ids": list(self.objective_ids),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    # ------------------------------------------------------------------
    # Subtask 6 — HTML attribute string emit
    # ------------------------------------------------------------------

    def to_html_attrs(self) -> str:
        """Render the ``data-cf-*`` attribute string for this block.

        Reproduces the exact format the legacy renderers in
        ``generate_course.py`` emit so the renderer migration in Round 3
        stays byte-stable when ``COURSEFORGE_EMIT_BLOCKS`` is off.

        When ``COURSEFORGE_EMIT_BLOCKS`` is set to a truthy value, the
        attribute string gains a trailing ``data-cf-block-id="..."``
        attribute — the only NEW HTML attribute Phase 2 introduces.
        Off by default so legacy snapshot tests stay green.
        """
        block_type = self.block_type
        if block_type == "objective":
            attrs = self._objective_attrs()
        elif block_type == "flip_card_grid":
            attrs = self._flip_card_grid_attrs()
        elif block_type == "self_check_question":
            attrs = self._self_check_question_attrs()
        elif block_type == "activity":
            attrs = self._activity_attrs()
        elif block_type in {
            "explanation",
            "example",
            "concept",
            "summary_takeaway",
        } or block_type in _CONTENT_SECTION_BLOCK_TYPES:
            # Heading content-section blocks share one attribute shape
            # (`_render_content_sections:1018-1035`). The block_type
            # itself is the resolved `content_type` — emit it directly.
            attrs = self._content_section_attrs()
        elif block_type == "callout":
            attrs = self._callout_attrs()
        elif block_type == "chrome":
            attrs = ' data-cf-role="template-chrome"'
        elif block_type in {
            "prereq_set",
            "reflection_prompt",
            "discussion_prompt",
            "recap",
        }:
            # Wrapper-only blocks (the inline `<section>` wrappers in
            # `generate_week`). Source-id attrs only.
            attrs = _source_attr_string(self.source_ids, self.source_primary)
        elif block_type == "misconception":
            # Misconceptions today emit only via JSON-LD (no data-cf-*
            # attribute on the rendered HTML). Emit empty so the only
            # change with the flag on is the appended block_id.
            attrs = ""
        elif block_type == "assessment_item":
            # Assessment items in IMSCC live in QTI XML, not HTML.
            # Reserved for Phase 4+; emit empty for now.
            attrs = ""
        else:  # pragma: no cover — defensive; __post_init__ already validates.
            attrs = ""

        if _emit_blocks_enabled() and self.block_id:
            attrs += f' data-cf-block-id="{_esc(self.block_id)}"'
        return attrs

    # --- per-block-type helpers (kept private to make dispatch readable) ---

    def _objective_attrs(self) -> str:
        """Match `_render_objectives:854-860`."""
        attrs = ""
        if self.objective_ids:
            attrs += f' data-cf-objective-id="{_esc(self.objective_ids[0])}"'
        if self.bloom_level:
            attrs += f' data-cf-bloom-level="{self.bloom_level}"'
        if self.bloom_verb:
            attrs += f' data-cf-bloom-verb="{self.bloom_verb}"'
        if self.cognitive_domain:
            attrs += f' data-cf-cognitive-domain="{self.cognitive_domain}"'
        return attrs

    def _flip_card_grid_attrs(self) -> str:
        """Match `_render_flip_cards:887-889`.

        Per-card emit. When ``key_terms`` carries a single term slug, it
        is emitted on the wrapper as ``data-cf-term``. Multi-term grids
        emit one Block per card upstream.
        """
        role_attr = (
            f' data-cf-teaching-role="{self.teaching_role}"' if self.teaching_role else ""
        )
        attrs = (
            ' data-cf-component="flip-card"'
            ' data-cf-purpose="term-definition"'
            f"{role_attr}"
        )
        if self.key_terms:
            # Single-term per-card emit; first slug wins. Matches the
            # legacy ``term_slug = _slugify(t["term"])`` per-card pattern.
            attrs += f' data-cf-term="{_esc(self.key_terms[0])}"'
        return attrs

    def _self_check_question_attrs(self) -> str:
        """Match `_render_self_check:929-944`."""
        role_attr = (
            f' data-cf-teaching-role="{self.teaching_role}"' if self.teaching_role else ""
        )
        bloom = self.bloom_level or "remember"
        attrs = (
            ' data-cf-component="self-check"'
            ' data-cf-purpose="formative-assessment"'
            f"{role_attr}"
            f' data-cf-bloom-level="{bloom}"'
        )
        if self.objective_ids and self.objective_ids[0]:
            attrs += f' data-cf-objective-ref="{_esc(self.objective_ids[0])}"'
        attrs += _source_attr_string(self.source_ids, self.source_primary)
        return attrs

    def _activity_attrs(self) -> str:
        """Match `_render_activities:1126-1140`."""
        role_attr = (
            f' data-cf-teaching-role="{self.teaching_role}"' if self.teaching_role else ""
        )
        bloom = self.bloom_level or "apply"
        attrs = (
            ' data-cf-component="activity"'
            ' data-cf-purpose="practice"'
            f"{role_attr}"
            f' data-cf-bloom-level="{bloom}"'
        )
        if self.objective_ids and self.objective_ids[0]:
            attrs += f' data-cf-objective-ref="{_esc(self.objective_ids[0])}"'
        attrs += _source_attr_string(self.source_ids, self.source_primary)
        return attrs

    def _content_section_attrs(self) -> str:
        """Match `_render_content_sections:1018-1035` (heading attrs).

        ``content_type_label`` carries the resolved content_type (or it
        falls back to ``block_type``); ``key_terms`` carries the term
        slugs already slugified by the renderer; ``bloom_range`` is the
        section span string.
        """
        content_type = self.content_type_label or self.block_type
        attrs = f' data-cf-content-type="{content_type}"'
        if self.key_terms:
            joined = ",".join(self.key_terms)
            attrs += f' data-cf-key-terms="{joined}"'
        if self.bloom_range:
            attrs += f' data-cf-bloom-range="{self.bloom_range}"'
        attrs += _source_attr_string(self.source_ids, self.source_primary)
        return attrs

    def _callout_attrs(self) -> str:
        """Match `_render_content_sections:1071-1073`."""
        content_type = self.content_type_label or "note"
        return f' data-cf-content-type="{content_type}"'

    # ------------------------------------------------------------------
    # Subtask 7 — JSON-LD entry emit
    # ------------------------------------------------------------------

    def to_jsonld_entry(self) -> Dict[str, Any]:
        """Render the JSON-LD entry dict for this block.

        Matches the camelCase shape the existing ``_build_*_metadata``
        helpers in ``generate_course.py`` emit (Subtask 7). The entry
        shape is dispatched on ``block_type``: legacy-shape entries
        (``objective`` / ``explanation`` / ``misconception`` etc.) carry
        the same keys the legacy builders emit so an inline migration
        of those builders in Round 3 keeps consumers unchanged. New
        block types (``flip_card_grid`` / ``self_check_question`` /
        ``activity`` / ``chrome`` / ``prereq_set`` / ``summary_takeaway`` /
        ``reflection_prompt`` / ``discussion_prompt`` / ``recap``) emit
        a minimal Phase-2-shaped entry carrying ``blockId`` /
        ``blockType`` / ``sequence`` plus ``touchedBy`` / ``contentHash``
        for the new top-level ``blocks[]`` array.
        """
        block_type = self.block_type
        if block_type == "objective":
            return self._objective_jsonld()
        if block_type == "misconception":
            return self._misconception_jsonld()
        if block_type in _CONTENT_SECTION_BLOCK_TYPES or block_type in {
            "explanation",
            "example",
            "concept",
            "summary_takeaway",
        }:
            # Legacy `_build_sections_metadata` shape — only fired when
            # the Block represents a section heading.
            return self._section_jsonld()
        # Default Phase-2 shape: small audit-only entry for the new
        # `blocks[]` array.
        return self._minimal_block_jsonld()

    def _objective_jsonld(self) -> Dict[str, Any]:
        """Match `_build_objectives_metadata:1364-1420`."""
        statement = self.content if isinstance(self.content, str) else ""
        entry: Dict[str, Any] = {
            "id": self.objective_ids[0] if self.objective_ids else "",
            "statement": statement,
            "bloomLevel": self.bloom_level,
            "bloomVerb": self.bloom_verb,
            "cognitiveDomain": self.cognitive_domain,
        }
        if self.bloom_levels:
            entry["bloomLevels"] = list(self.bloom_levels)
        if self.bloom_verbs:
            entry["bloomVerbs"] = list(self.bloom_verbs)
        if self.key_terms:
            entry["keyConcepts"] = list(self.key_terms)
            if self.bloom_level:
                entry["targetedConcepts"] = [
                    {"concept": slug, "bloomLevel": self.bloom_level}
                    for slug in self.key_terms
                ]
        return entry

    def _section_jsonld(self) -> Dict[str, Any]:
        """Match `_build_sections_metadata:1467-1490`."""
        heading = self.content if isinstance(self.content, str) else ""
        content_type = self.content_type_label or self.block_type
        entry: Dict[str, Any] = {
            "heading": heading,
            "contentType": content_type,
        }
        if self.key_terms:
            entry["keyTerms"] = list(self.key_terms)
        if self.teaching_role:
            entry["teachingRole"] = [self.teaching_role]
        if self.bloom_range:
            entry["bloomRange"] = (
                [self.bloom_range]
                if isinstance(self.bloom_range, str)
                else list(self.bloom_range)
            )
        if self.source_references:
            entry["sourceReferences"] = [dict(r) for r in self.source_references]
        return entry

    def _misconception_jsonld(self) -> Dict[str, Any]:
        """Match `_build_misconceptions_metadata:1571-1578`."""
        if isinstance(self.content, dict):
            mis_text = str(self.content.get("misconception", ""))
            cor_text = str(self.content.get("correction", ""))
        else:
            mis_text = ""
            cor_text = ""
        entry: Dict[str, Any] = {
            "misconception": mis_text,
            "correction": cor_text,
        }
        if self.bloom_level:
            entry["bloomLevel"] = self.bloom_level
            if self.cognitive_domain:
                entry["cognitiveDomain"] = self.cognitive_domain
        return entry

    def _minimal_block_jsonld(self) -> Dict[str, Any]:
        """Phase-2 default entry shape for blocks that don't have a
        legacy JSON-LD builder counterpart.

        Carries the audit fields (``blockId`` / ``blockType`` /
        ``sequence``) plus the new ``touchedBy`` / ``contentHash``
        fields so the new top-level ``blocks[]`` array keeps full
        attribution per pre-resolved decision #2.
        """
        entry: Dict[str, Any] = {
            "blockId": self.block_id,
            "blockType": self.block_type,
            "sequence": self.sequence,
        }
        if self.touched_by:
            entry["touchedBy"] = self._render_touched_by()
        if self.content_hash:
            entry["contentHash"] = self.content_hash
        return entry

    def _render_touched_by(self) -> List[Dict[str, Any]]:
        """Project the touch chain into the JSON-LD ``touchedBy`` array."""
        return [t.to_jsonld() for t in self.touched_by]


# Block types whose ``to_html_attrs`` / ``to_jsonld_entry`` should follow
# the legacy `_render_content_sections` / `_build_sections_metadata`
# shape. Section-heading content_types map onto these block_types
# directly (one block_type per resolved content_type label). Right now
# the canonical 16-type enum doesn't include ``procedure`` /
# ``comparison`` / ``definition`` / ``overview`` / ``summary`` /
# ``exercise`` — those resolve to ``content_type_label`` on the
# Block instead, while ``block_type`` stays in the canonical enum.
_CONTENT_SECTION_BLOCK_TYPES: frozenset = frozenset()
