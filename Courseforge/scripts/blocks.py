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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = ["Block", "Touch", "BLOCK_TYPES", "_parse_provider_page_html"]


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


# Phase 3.5 Subtask 14: extend the canonical Touch.tier enum with
# the post-validation tier labels ``outline_val`` and ``rewrite_val``.
# These mark a Touch emitted by the inter-tier validation seam (after
# the outline-tier draft) and the post-rewrite validation seam (after
# the rewrite-tier emit) respectively, distinguishing audit entries
# the validators append from the upstream authoring tier touches
# (``outline`` / ``rewrite``). The legacy ``validation`` value is
# retained for backwards compatibility with pre-Phase-3.5 captures.
#
# Plan refers to this constant as ``_TIER_VALUES``; the canonical
# in-tree name has always been ``_TOUCH_TIERS`` (Phase 2 introduction).
_TOUCH_TIERS: frozenset = frozenset(
    {"outline", "validation", "rewrite", "outline_val", "rewrite_val"}
)

_TOUCH_PROVIDERS: frozenset = frozenset(
    {"anthropic", "local", "together", "claude_session", "deterministic"}
)

_ESCALATION_MARKERS: frozenset = frozenset(
    {
        "outline_budget_exhausted",
        "structural_unfixable",
        "validator_consensus_fail",
        # C2 silent-degradation fix: dedicated markers for per-block
        # dispatch errors (network failure / provider raise / unhandled
        # exception inside ``CourseforgeRouter.route_all``). Previously
        # such errors were silently logged and the block was DROPPED
        # from the output, so the W5 packager-side filter
        # (``escalation_marker is not None``) never saw the block and
        # the IMSCC shipped without it. Stamping with one of these two
        # markers keeps the block in the return list AND triggers the
        # W5 filter / ESCALATED_BLOCK_IN_IMSCC gate.
        "outline_dispatch_error",
        "rewrite_dispatch_error",
        # Wave 1.5 W1.5.C: per-claim attribution unfixable. Fires when
        # the outline-tier regen budget is exhausted PURELY on per-
        # claim source-attribution misses (the ``BlockSourceRefValidator``
        # ``OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS`` warning code) with
        # no block-level structural miss across the regen chain. The
        # rewrite-tier prompt-builder reads this marker via
        # ``_ESCALATION_MARKER_CONTEXT`` and treats the per-claim
        # citation map as advisory rather than authoritative.
        "per_claim_attribution_unfixable",
        # Wave 1.7 W1.7.D: block-objective delivery unfixable. Fires
        # when the rewrite-tier regen budget is exhausted PURELY on
        # Wave-1.7 block-objective delivery misses (the
        # ``BlockObjectiveDeliveryValidator`` warning codes
        # ``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` /
        # ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` /
        # ``BLOCK_OBJECTIVE_VERB_ABSENT``) with no upstream structural
        # miss across the regen chain. The rewrite-tier prompt-builder
        # reads this marker via ``_ESCALATION_MARKER_CONTEXT`` so a
        # postmortem reader sees the gate failure even though the
        # rendered block ships; the surviving block also carries
        # ``objective_alignment[*].status="unverifiable"`` so the
        # JSON-LD audit trail records the unverifiable delivery state.
        "block_objective_undelivered",
        # W5 best-of-N: no candidate cleared BOTH objective-coverage AND
        # zero-contradiction among the validator-passing samples, so the
        # entailment-argmax selector fell back to the highest-entailment
        # passing candidate and stamped this marker (never fabricates a clean
        # winner). The block still ships (it passed the validator chain); the
        # marker tells a postmortem the NLI-selection had no clean pick.
        "best_of_n_no_clean_candidate",
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


# ---------------------------------------------------------------------------
# Provider HTML parser (Phase 2 Subtask 35: moved from
# ``MCP/tools/_content_gen_helpers.py`` so :class:`ContentGeneratorProvider`
# can build a Block from the LLM's rendered HTML without importing the
# MCP-side helper module).
# ---------------------------------------------------------------------------

_PROV_HEADING_RE = re.compile(r"(?is)<h(?:1|2|3)[^>]*>(.*?)</h(?:1|2|3)>")
_PROV_PARAGRAPH_RE = re.compile(r"(?is)<p[^>]*>(.*?)</p>")
_PROV_TAG_STRIP_RE = re.compile(r"(?is)<[^>]+>")


def _parse_provider_page_html(
    html: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    """Extract ``(heading, paragraphs[])`` from provider-rendered HTML.

    Returns ``(None, [])`` when input is empty / unparseable. Strips
    inner tags from extracted text so callers don't double-escape when
    they wrap the paragraphs in ``html.escape``. Empty paragraphs (after
    tag strip + whitespace collapse) are dropped.

    Phase 1 ToS unblock: minimal regex-based HTML parser for the
    in-process content-provider's rendered HTML body. BeautifulSoup
    intentionally NOT used so this stays dependency-light.
    """
    if not html or not isinstance(html, str):
        return None, []
    heading_match = _PROV_HEADING_RE.search(html)
    heading: Optional[str] = None
    if heading_match:
        raw_heading = heading_match.group(1) or ""
        heading = _PROV_TAG_STRIP_RE.sub("", raw_heading).strip()
        if not heading:
            heading = None

    paragraphs: List[str] = []
    for m in _PROV_PARAGRAPH_RE.finditer(html):
        raw = m.group(1) or ""
        text = _PROV_TAG_STRIP_RE.sub("", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
    return heading, paragraphs


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
    # GPT Feedback v2 Wave 1 / W1.A — observed Bloom level as classified
    # by the BERT ensemble at validation time, plus the boolean alignment
    # signal (``observed_bloom_level == bloom_level``). Both stay default
    # ``None`` for blocks emitted before the BERT classifier wires in
    # (Wave 2 router work). Audit-only — INTENTIONALLY excluded from
    # ``compute_content_hash()`` so a classifier retro-fit doesn't drift
    # every existing block hash on rebuild.
    observed_bloom_level: Optional[str] = None
    bloom_alignment: Optional[bool] = None
    # GPT Feedback v2 Wave 1.7 / W1.7.A — per-objective-ref delivery
    # alignment, populated by the BlockObjectiveDeliveryValidator at the
    # inter_tier_validation + post_rewrite_validation seams. Stays default
    # empty tuple for blocks emitted before Wave 1.7 wires in. Audit-only
    # — INTENTIONALLY excluded from ``compute_content_hash()`` so a
    # validator retro-fit doesn't drift every existing block hash.
    objective_alignment: Tuple[Dict[str, Any], ...] = ()
    # Bloom-diversity fix — DETERMINISTIC per-template target Bloom level
    # carried from the page block plan (``_PAGE_BLOCK_PLAN`` in
    # ``MCP/tools/pipeline_tools.py``) onto each outline-tier Block stub.
    # The outline provider surfaces this as a per-block "author at
    # bloom_level=<target>" directive AND enforces it as a FLOOR after the
    # LLM emits (the target LIFTS a lazy "understand" but NEVER lowers below
    # the existing ≥-objective rule). Routing/authoring metadata only —
    # INTENTIONALLY excluded from ``compute_content_hash()`` (a template
    # re-target must not drift every existing block hash) and from the
    # JSON-LD / HTML projections (the resolved ``bloom_level`` is what the
    # renderer stamps, not this declared target). Stays default ``None``
    # for blocks emitted before the diversity fix wires in / by callers
    # that don't carry a target.
    target_bloom: Optional[str] = None

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
        ``escalation_marker``, ``observed_bloom_level``,
        ``bloom_alignment``, and ``objective_alignment`` so a
        touch-only / budget-only / classifier-retrofit / objective-
        delivery-retrofit revision keeps a stable hash. The hash exists
        for re-execution drift detection — same content → same hash
        regardless of which tier authored it or how many times it was
        retried, and regardless of which audit-only signals were
        attached after the fact.
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
        # GPT Feedback v2 Wave 1 / W1.A — emit observed Bloom + alignment
        # signals only when non-None so legacy emits stay byte-stable.
        # camelCase keys mirror schemas/knowledge/courseforge_jsonld_v1
        # .schema.json::$defs.Block.properties.{observedBloomLevel,
        # bloomAlignment}.
        if self.observed_bloom_level is not None:
            entry["observedBloomLevel"] = self.observed_bloom_level
        if self.bloom_alignment is not None:
            entry["bloomAlignment"] = self.bloom_alignment
        # GPT Feedback v2 Wave 1.7 / W1.7.A — emit per-objective-ref
        # delivery alignment only when non-empty so legacy emits stay
        # byte-stable. camelCase keys mirror schemas/knowledge/
        # courseforge_jsonld_v1.schema.json::$defs.Block.properties.
        # objectiveAlignment + $defs.ObjectiveAlignment.
        if self.objective_alignment:
            entry["objectiveAlignment"] = [dict(a) for a in self.objective_alignment]
        # W3 — optional pointer to the sidecar block_synthesis_manifest.jsonl
        # line keyed on this block_id. We do NOT inline the whole manifest into
        # the JSON-LD (it would bloat the IMSCC payload + duplicate the
        # sidecar); the pointer just lets a JSON-LD consumer join to the
        # canonical manifest artifact. Gated behind COURSEFORGE_EMIT_BLOCKS (the
        # gate for the whole blocks[] projection) so legacy flag-off emit stays
        # byte-stable.
        if self.block_id and _emit_blocks_enabled():
            entry["synthesisManifestRef"] = self.block_id
        return entry

    def _render_touched_by(self) -> List[Dict[str, Any]]:
        """Project the touch chain into the JSON-LD ``touchedBy`` array."""
        return [t.to_jsonld() for t in self.touched_by]

    # ------------------------------------------------------------------
    # W3 — Per-block synthesis-manifest projection
    # ------------------------------------------------------------------

    def to_synthesis_manifest(
        self,
        resolver: Optional["Callable[[str], Optional[Union[str, Dict[str, Any]]]]"] = None,
        concept_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Project this Block into a block-synthesis-manifest dict (W3).

        Pure projection off the Block — mirrors :meth:`to_jsonld_entry` /
        :meth:`to_html_attrs`. The manifest ASSEMBLES the provenance facts
        already captured elsewhere (the outline ``source_refs`` / ``key_claims``,
        the template identity, the ``Touch`` chain's model/provider/timestamp/
        ``decision_capture_id``, the concept tags) into one canonical per-block
        record. It does NOT re-log model/provider (read from ``Touch``) or
        re-derive claims (read from the outline ``content`` dict).

        Schema: ``schemas/knowledge/block_synthesis_manifest.schema.json``.

        Args:
            resolver: Optional callable mapping an outline ``sourceId`` (the
                ``dart:{slug}#{block_id}`` shape) to a DART chunk identity. It
                may return either:

                * a chunk-id ``str`` (resolved, no span info available), or
                * a dict ``{"id": <chunk_id>, "char_span": [...],
                  "html_xpath": ...}`` (resolved + span info, lifted into the
                  ``char_spans[]`` field), or
                * ``None`` when the sourceId does not resolve — the id is kept
                  in ``source_refs[]`` verbatim but omitted from
                  ``source_chunk_ids[]`` (the completeness validator catches the
                  resulting gap on a substantive block).

                When ``resolver`` is ``None`` every outline ``sourceId`` is
                passed through verbatim as a ``source_chunk_id`` (test / no-DART
                path).

        Returns:
            A manifest dict that validates against the W3 schema.
        """
        content = self.content if isinstance(self.content, dict) else {}

        # --- source_refs[] (verbatim copy of the outline dict) ---
        raw_refs = content.get("source_refs")
        source_refs: List[Dict[str, Any]] = []
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if isinstance(ref, dict):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid.strip():
                        source_refs.append(dict(ref))
        # Fall back to the Block's own source_references tuple (rewrite-tier
        # blocks carry the refs on the dataclass rather than the content dict).
        if not source_refs and self.source_references:
            for ref in self.source_references:
                if isinstance(ref, dict):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid.strip():
                        source_refs.append(dict(ref))

        # --- key_claims[] (W2 output recorded verbatim) ---
        raw_claims = content.get("key_claims")
        key_claims: List[Dict[str, Any]] = []
        if isinstance(raw_claims, list):
            for claim in raw_claims:
                if isinstance(claim, dict) and claim.get("claim"):
                    key_claims.append(dict(claim))

        # --- source_chunk_ids[] + char_spans[] (resolution) ---
        # Union of: every source_refs[].sourceId resolved to a chunk id, plus
        # every key_claims[].source_chunk_ids[] (already chunk ids per W2).
        chunk_ids: List[str] = []
        char_spans: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()
        seen_span_ids: set = set()

        def _record_chunk_id(cid: str) -> None:
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                chunk_ids.append(cid)

        def _record_span(record: Dict[str, Any]) -> None:
            cid = record.get("id") or record.get("chunk_id")
            if not isinstance(cid, str) or not cid or cid in seen_span_ids:
                return
            span_entry: Dict[str, Any] = {"chunk_id": cid}
            cspan = record.get("char_span")
            if (
                isinstance(cspan, (list, tuple))
                and len(cspan) == 2
                and all(isinstance(x, int) for x in cspan)
            ):
                span_entry["char_span"] = [int(cspan[0]), int(cspan[1])]
            xpath = record.get("html_xpath")
            if isinstance(xpath, str) and xpath:
                span_entry["html_xpath"] = xpath
            # Only record a span entry when it carries span/xpath info beyond
            # the bare chunk id (an id-only resolver yields no char_spans line).
            if len(span_entry) > 1:
                seen_span_ids.add(cid)
                char_spans.append(span_entry)

        for ref in source_refs:
            sid = ref.get("sourceId")
            if not isinstance(sid, str) or not sid.strip():
                continue
            if resolver is None:
                _record_chunk_id(sid)
                continue
            resolved = resolver(sid)
            if resolved is None:
                # Unresolved — kept in source_refs[] only.
                continue
            if isinstance(resolved, dict):
                cid = resolved.get("id") or resolved.get("chunk_id")
                if isinstance(cid, str) and cid:
                    _record_chunk_id(cid)
                    _record_span(resolved)
            elif isinstance(resolved, str) and resolved:
                _record_chunk_id(resolved)

        # Per-claim chunk ids (already chunk ids per the W2 contract).
        for claim in key_claims:
            for cid in claim.get("source_chunk_ids", []) or []:
                if isinstance(cid, str) and cid:
                    _record_chunk_id(cid)

        # --- synthesis.tiers[] (read from the Touch chain — NOT re-logged) ---
        tiers: List[Dict[str, Any]] = []
        for touch in self.touched_by:
            tiers.append(
                {
                    "tier": touch.tier,
                    "provider": touch.provider,
                    "model": touch.model,
                    "timestamp": touch.timestamp,
                    "decision_capture_id": touch.decision_capture_id,
                    "purpose": touch.purpose,
                }
            )

        # --- template_id (deterministic template that shaped the block) ---
        template_id = (
            self.template_type
            or self.content_type_label
            or self.block_type
        )

        # --- concept_tags (chunk-grounded; passed explicitly by the emit caller
        # — preferred, because rewrite-tier blocks carry self.content as an HTML
        # STRING so the content-dict path can't thread them — else read from the
        # content dict / key_terms fallback) ---
        if concept_tags is not None:
            manifest_concept_tags = [t for t in concept_tags if isinstance(t, str) and t]
        else:
            raw_tags = content.get("concept_tags")
            manifest_concept_tags = (
                [t for t in raw_tags if isinstance(t, str) and t]
                if isinstance(raw_tags, list)
                else []
            )

        manifest: Dict[str, Any] = {
            "schema_version": "1.0",
            "block_id": self.block_id,
            "block_type": self.block_type,
            "page_id": self.page_id,
            "sequence": self.sequence,
            "source_chunk_ids": chunk_ids,
            "source_refs": source_refs,
            "span_granularity": "chunk",
            "template_id": template_id,
            "synthesis": {"tiers": tiers},
            "concept_tags": manifest_concept_tags,
        }
        if char_spans:
            manifest["char_spans"] = char_spans
        if key_claims:
            manifest["key_claims"] = key_claims
        # slot_ids — present when the block was produced by slot-fill.
        raw_slots = content.get("slot_ids")
        if isinstance(raw_slots, list):
            slot_ids = [s for s in raw_slots if isinstance(s, str) and s]
            if slot_ids:
                manifest["slot_ids"] = slot_ids
        if self.content_hash:
            manifest["content_hash"] = self.content_hash
        return manifest


# Block types whose ``to_html_attrs`` / ``to_jsonld_entry`` should follow
# the legacy `_render_content_sections` / `_build_sections_metadata`
# shape. Section-heading content_types map onto these block_types
# directly (one block_type per resolved content_type label). Right now
# the canonical 16-type enum doesn't include ``procedure`` /
# ``comparison`` / ``definition`` / ``overview`` / ``summary`` /
# ``exercise`` — those resolve to ``content_type_label`` on the
# Block instead, while ``block_type`` stays in the canonical enum.
_CONTENT_SECTION_BLOCK_TYPES: frozenset = frozenset()
