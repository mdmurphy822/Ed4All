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

__all__ = [
    "Block",
    "Touch",
    "QualityScore",
    "BLOCK_TYPES",
    "QUALITY_DIMENSIONS",
    "CORE_QUALITY_DIMENSIONS",
    "_parse_provider_page_html",
]


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


# IB1 — six-slot anatomy contract emit flag (framework pp.13-17, QA-4). Default
# OFF: with this unset the JSON-LD ``blocks[]`` entry carries NO ``anatomy``
# key, so every existing snapshot / ``contentHash`` stays byte-identical. The
# five new ``Block`` slot fields, the hash exclusion, the derivation helper, and
# the lifecycle helpers are all flag-INDEPENDENT (pure additive API surface) —
# only the JSON-LD *emit* is gated, mirroring the ``COURSEFORGE_EMIT_BLOCKS``
# posture. Reuses ``_EMIT_BLOCKS_TRUTHY`` for the truthy parse.
_ANATOMY_EMIT_ENV = "ED4ALL_BLOCK_ANATOMY"


def _anatomy_emit_enabled() -> bool:
    """Read ``ED4ALL_BLOCK_ANATOMY`` each call so tests can toggle it.

    Default off — the IB1 ``anatomy`` JSON-LD sub-object is purely additive
    and must not break byte-stable emit. Falsey / garbage values → off
    (parse-with-fallback, mirroring :func:`_emit_blocks_enabled`).
    """
    return os.environ.get(_ANATOMY_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# IB4 — per-block WCAG 2.2 AA + UDL emit flag (ED4ALL_BLOCK_A11Y). Default OFF:
# with this unset the UDL coverage fields are NOT emitted to HTML / JSON-LD
# (byte-stable) and the per-block a11y sub-check in RewriteHtmlShapeValidator is
# a no-op. The canonical resolver lives in lib/generation/block_a11y.py; this
# module-level reader keeps blocks.py dependency-light (the emit gate needs no
# import of the larger generation package at module load).
_BLOCK_A11Y_EMIT_ENV = "ED4ALL_BLOCK_A11Y"


def _block_a11y_emit_enabled() -> bool:
    """Read ``ED4ALL_BLOCK_A11Y`` each call so tests can toggle it.

    Default off — the IB4 UDL fields (``n_representations`` /
    ``response_formats`` / ``engagement_affordance``) are purely additive and
    must not break byte-stable emit. Falsey / garbage values → off
    (parse-with-fallback, mirroring :func:`_anatomy_emit_enabled`).
    """
    return os.environ.get(_BLOCK_A11Y_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# IB5 — new-block-types emit flag (ED4ALL_NEW_BLOCK_TYPES). Default OFF: with
# this unset the IB5 type-specific fields (``fade_state`` / ``long_description``
# / ``media_a11y``) are NOT projected to HTML / JSON-LD (byte-stable). The four
# new tokens are unconditionally valid BLOCK_TYPES members (the dataclass stays
# permissive — gating lives in the planner / renderer / emit), mirroring the I6
# table/acronym/key_idea posture. Canonical resolver: lib/generation/
# new_block_types.py; this module-level reader keeps blocks.py dependency-light.
_NEW_BLOCK_TYPES_EMIT_ENV = "ED4ALL_NEW_BLOCK_TYPES"


def _new_block_types_emit_enabled() -> bool:
    """Read ``ED4ALL_NEW_BLOCK_TYPES`` each call so tests can toggle it.

    Default off — the IB5 type-specific fields (``fade_state`` /
    ``long_description`` / ``media_a11y``) are purely additive and must not break
    byte-stable emit. Falsey / garbage values → off (parse-with-fallback,
    mirroring :func:`_block_a11y_emit_enabled`).
    """
    return os.environ.get(_NEW_BLOCK_TYPES_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# FR-INT-03 — B11 reflection predict-then-reveal calibration emit flag
# (ED4ALL_REFLECTION_CALIBRATION). Default OFF: with this unset the three B11
# calibration fields (``prediction_prompt`` / ``reveal_content`` /
# ``calibration_feedback``) are NOT projected to the <details> render scaffold
# (byte-stable). The fields are unconditionally valid additive Optional Block
# fields (hash-excluded); only the renderer emit + the two validator arms are
# flag-gated. Canonical resolver: lib/generation/reflection_calibration.py; this
# module-level reader keeps blocks.py dependency-light.
_REFLECTION_CALIBRATION_EMIT_ENV = "ED4ALL_REFLECTION_CALIBRATION"


def _reflection_calibration_emit_enabled() -> bool:
    """Read ``ED4ALL_REFLECTION_CALIBRATION`` each call so tests can toggle it.

    Default off — the three FR-INT-03 calibration fields are purely additive and
    must not break byte-stable emit. Falsey / garbage values → off
    (parse-with-fallback, mirroring :func:`_new_block_types_emit_enabled`).
    """
    return os.environ.get(_REFLECTION_CALIBRATION_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# recall_self_check — free-recall / cloze self-check variant emit flag
# (ED4ALL_RECALL_SELF_CHECK). Default OFF: with this unset the new
# ``recall_format`` Block field (hash-excluded) is not projected and
# ``_UDL_RESPONSE_BY_BLOCK_TYPE`` still returns ``select`` for
# self_check_question. Canonical resolver: lib/generation/recall_self_check.py.
_RECALL_SELF_CHECK_EMIT_ENV = "ED4ALL_RECALL_SELF_CHECK"


def _recall_self_check_emit_enabled() -> bool:
    """Read ``ED4ALL_RECALL_SELF_CHECK`` each call so tests can toggle it.

    Default off — the new ``recall_format`` field + the corrected UDL
    response-format are purely additive and must not break byte-stable emit.
    Falsey / garbage values → off (parse-with-fallback, mirroring
    :func:`_reflection_calibration_emit_enabled`).
    """
    return os.environ.get(_RECALL_SELF_CHECK_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# misconception_rich — named subject-specific misconception + productive-failure
# emit flag (ED4ALL_MISCONCEPTION_RICH). Default OFF: with this unset the three
# new Block fields (``mc_named_concept`` / ``mc_predict_prompt`` /
# ``mc_reconcile``, hash-excluded) are not projected to HTML/JSON-LD, so every
# existing snapshot / contentHash stays byte-identical. Canonical resolver:
# lib/generation/misconception_rich.py.
_MISCONCEPTION_RICH_EMIT_ENV = "ED4ALL_MISCONCEPTION_RICH"


def _misconception_rich_emit_enabled() -> bool:
    """Read ``ED4ALL_MISCONCEPTION_RICH`` each call so tests can toggle it.

    Default off — the three productive-failure fields are purely additive and
    must not break byte-stable emit. Falsey / garbage values → off
    (parse-with-fallback, mirroring :func:`_reflection_calibration_emit_enabled`).
    """
    return os.environ.get(_MISCONCEPTION_RICH_EMIT_ENV, "").strip().lower() in _EMIT_BLOCKS_TRUTHY


# IB6.5 — universal (verb · level · knowledge-type) triple chip emit flag
# (ED4ALL_BLOCK_QUALITY_RUBRIC, the IB6 keystone). Default OFF: with this unset
# the cognitive-domain chip stays objectives-only (``_objective_attrs``) and no
# ``data-cf-bloom-triple`` attr is emitted, so every existing snapshot stays
# byte-identical. When on, every pedagogical block that carries a resolvable
# (verb, level, knowledge-type) triple emits a universal ``data-cf-cognitive-
# domain`` + a combined ``data-cf-bloom-triple`` chip (additive — the existing
# three separate objective attrs are unchanged). Reuses ``_EMIT_BLOCKS_TRUTHY``.
_BLOCK_QUALITY_RUBRIC_EMIT_ENV = "ED4ALL_BLOCK_QUALITY_RUBRIC"


def _block_quality_rubric_emit_enabled() -> bool:
    """Read ``ED4ALL_BLOCK_QUALITY_RUBRIC`` each call so tests can toggle it.

    Default off — the IB6.5 universal cognitive-domain + bloom-triple chips are
    purely additive and must not break byte-stable emit. Falsey / garbage →
    off (parse-with-fallback, mirroring :func:`_new_block_types_emit_enabled`).
    """
    return (
        os.environ.get(_BLOCK_QUALITY_RUBRIC_EMIT_ENV, "").strip().lower()
        in _EMIT_BLOCKS_TRUTHY
    )


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
        # Wave-2 block-variety additions (snake_case canonical tokens).
        "scenario",
        "problem",
        "vocab_card",
        "formula",
        "checklist",
        # Issue I6 instruction-palette-v2 additions (snake_case canonical
        # tokens). WCAG-correct structural block types: ``table`` (a real
        # ``<table>`` with ``<caption>`` + scoped ``<th>``), ``acronym`` (a
        # ``<dl>`` mapping each letter to its expansion term), and
        # ``key_idea`` (an ``<aside>`` promoted from the generic callout).
        # Emitted ONLY via the dynamic block planner path
        # (``ED4ALL_DYNAMIC_BLOCK_PLAN``); the fixed-plan / legacy paths
        # never select them, so every existing snapshot stays byte-stable.
        "table",
        "acronym",
        "key_idea",
        # IB5 framework-aligned pedagogical block types (snake_case canonical
        # tokens). The four genuine catalog gaps the framework reserves:
        # ``hook`` (B02 activation — gain attention / surface prior knowledge),
        # ``multimedia`` (B04 — the ONLY time-based block carrying the full
        # caption / audio-description / transcript / controls a11y stack),
        # ``worked_example`` (B05 — subgoal labels + per-step Why + fade-state;
        # DISTINCT from ``example``, a single un-labeled instance, which is NOT
        # renamed or removed), and ``diagram`` (B06 — a dual-coded spatial
        # artifact with a structured long-description + a data-table
        # equivalent). Emitted ONLY via the dynamic block planner path behind
        # ``ED4ALL_NEW_BLOCK_TYPES`` (default OFF); the fixed-plan / legacy
        # paths never select them, so every existing snapshot stays
        # byte-stable (mirrors the I6 table/acronym/key_idea posture).
        "hook",
        "multimedia",
        "worked_example",
        "diagram",
        # B15 framework-aligned addition (snake_case canonical token). The last
        # remaining catalog gap: ``resources`` (B15 Resources / Further Reading
        # — an accessible list of curated external links / references each with
        # DESCRIPTIVE link text, never a bare URL or "click here"). DISTINCT
        # from ``callout`` (a single inline highlight) and ``summary_takeaway``
        # (the section's own key points, not outbound links). Emitted ONLY via
        # the dynamic block planner path behind ``ED4ALL_NEW_BLOCK_TYPES``
        # (default OFF); the fixed-plan / legacy paths never select it, so every
        # existing snapshot stays byte-stable (mirrors the IB5 hook/multimedia/
        # worked_example/diagram posture). The 2.4.4 link-purpose contract is
        # gated by ``lib/validators/resource_link_purpose.py``.
        "resources",
        # FR-INT-02 framework-aligned addition (snake_case canonical token).
        # ``guided_practice`` (B08 Guided Practice) — a FIRST-CLASS practice
        # scaffold DISTINCT from ``self_check_question`` (B07, a single low-stakes
        # knowledge check) and ``problem``/``activity`` (also B08 siblings but
        # generic). A guided_practice is the framework's faded-scaffold practice
        # block: it FOLLOWS a worked_example and carries a ``fade_state`` (worked
        # | completion | independent) marking the gradual-release stage. Emitted
        # ONLY via the dynamic block planner path behind ``ED4ALL_NEW_BLOCK_TYPES``
        # (default OFF); the fixed-plan / legacy paths never select it, so every
        # existing snapshot stays byte-stable (mirrors the IB5 hook/multimedia/
        # worked_example/diagram + B15 resources posture). REUSES the existing
        # ``fade_state`` field (no new fading field). The follow-a-worked-example
        # + carry-a-fade-state contract is gated by
        # ``lib/validators/b08_sequence.py``.
        "guided_practice",
    }
)


# IB2.3 — 8-dimension quality-rubric DATA MODEL (empty scaffold). The eight
# orthogonal quality dimensions the framework scores per block on an anchored
# 0–3 scale (framework §6.2-6.3). ORDERED tuple — verbatim dimension names.
# Dims 1,2,7,8 (alignment / cognitive_load / accessibility / coherence) are the
# "load-bearing core applying to every block" per the framework. This wave
# stands up the SHAPE only — NO scoring, NO mean/rollup, NO gate (IB6 fills it).
QUALITY_DIMENSIONS: Tuple[str, ...] = (
    "alignment",
    "cognitive_load",
    "multimedia",
    "retrieval",
    "feedback",
    "engagement",
    "accessibility",
    "coherence",
)

# Framework dims 1,2,7,8 — the load-bearing core that applies to EVERY block
# (the other four are applicable-when-relevant). Used by IB6's mean = applicable
# dims; declared here so the constant has one home alongside QUALITY_DIMENSIONS.
CORE_QUALITY_DIMENSIONS: frozenset = frozenset(
    {"alignment", "cognitive_load", "accessibility", "coherence"}
)

# The valid anchored 0–3 scores (plus None = not-yet-scored / not-applicable).
_QUALITY_SCORE_VALUES: frozenset = frozenset({None, 0, 1, 2, 3})


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

# DERIVED from the unified endpoint registry (config/endpoints.yaml) via
# lib.llm.endpoints.provenance_provider_names() — the single authority for
# the closed Touch.provider set (union of every endpoint row's
# provenance_provider + the "deterministic" sentinel). The two static
# provenance sites (the JSON-LD schema enum + the SHACL sh:in list) are
# kept byte-equal to this set by scripts/codegen/sync_provenance_enum.py
# (CI-enforced by schemas/tests/test_touch_provider_enum_sync.py). lib.llm.
# endpoints imports only stdlib + yaml + jsonschema + lib.paths, so this
# import is cycle-safe (one direction: blocks.py -> endpoints).
from lib.llm.endpoints import provenance_provider_names as _provenance_provider_names

_TOUCH_PROVIDERS: frozenset = frozenset(_provenance_provider_names())

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
        # rewrite-overflow-fix-2026-06: the rewrite-tier input-truncation
        # tripwire detected the served context window silently truncated the
        # prompt HEAD (the system-prompt authoring CONTRACT was dropped so
        # the model authored with source but no rules). HARD, NON-RETRYABLE —
        # the block is stamped + short-circuited rather than re-dispatched
        # (re-dispatching the same over-window prompt re-truncates).
        # Surfaces as ``escalated`` and routes through the W5 packager-side
        # escalation filter.
        "input_prompt_truncated",
        # rewrite-overflow-fix-2026-07: the rewrite-tier whole-prompt budget
        # (ED4ALL_REWRITE_FIT_WINDOW) found the NON-CHUNK scaffold alone (the
        # trimmed system prompt + the outline dict + per-claim + objectives +
        # contract) already exceeds the served window, so NO grounding chunk
        # could fit and the prompt cannot be authored without silent head-
        # truncation. HARD, NON-RETRYABLE — the block is stamped + never
        # dispatched (dispatching a prompt that cannot fit would truncate the
        # authoring CONTRACT). Surfaces as ``escalated`` and routes through the
        # W5 packager-side escalation filter. Operator fix: raise the served
        # window / ED4ALL_REWRITE_NUM_CTX, or shrink the upstream outline.
        "rewrite_scaffold_overflow",
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


# ---------------------------------------------------------------------------
# IB1 — six-slot anatomy contract + five-stage micro-lifecycle (framework
# pp.13-17, QA-4, p.138). Representation only — addressable + validatable; NO
# behavior change here. The malformed-block (slot-presence) JUDGEMENT is IB6's
# validator; these helpers expose presence DATA only.
# ---------------------------------------------------------------------------

# The six anatomy slots. BODY is the existing ``Block.content`` (NOT a new
# field — IB1.2); the other five are new Optional ``Block`` fields (IB1.1).
_BODY_SLOT = "content"  # explicit anchor for IB1.2 / tests
_ANATOMY_SLOTS: Tuple[str, ...] = (
    "heading",
    "purpose_tag",
    _BODY_SLOT,
    "interaction",
    "feedback",
    "transition",
)

# Five-stage micro-lifecycle (activate→present→apply→check→consolidate;
# Gagné's Nine Events onto Merrill's First Principles, framework pp.13-17,
# p.138). DISTINCT from the FIVE PAGE TYPES (overview/content/application/
# self_check/summary) which are page-FILE groupings (``CANONICAL_PAGE_TYPES``
# in ``lib/generation/block_planner.py``) — the page types decide which .html
# file a block lands in; these stages are block-INTERNAL — every block runs
# the whole cycle on its own slots regardless of which page it sits on. They
# are NOT the same five and MUST NOT be unified.
LIFECYCLE_STAGES: Tuple[str, ...] = (
    "activate",
    "present",
    "apply",
    "check",
    "consolidate",
)

# Slot↔stage mapping (gap text: heading+purpose→Activate, body→Present,
# interaction→Apply, feedback→Check, transition→Consolidate).
_STAGE_SLOTS: Dict[str, Tuple[str, ...]] = {
    "activate": ("heading", "purpose_tag"),
    "present": (_BODY_SLOT,),  # content == body (IB1.2)
    "apply": ("interaction",),
    "check": ("feedback",),
    "consolidate": ("transition",),
}


def _slot_value(block: "Block", slot: str) -> Optional[Any]:
    """Read an anatomy slot off a Block (``content``/heading/…)."""
    return getattr(block, slot, None)


def _slot_present(block: "Block", slot: str) -> bool:
    """True iff the slot carries a non-empty value.

    ``content`` (the BODY slot) is present when truthy; the other five
    string slots are present when non-None and non-empty after strip.
    """
    value = _slot_value(block, slot)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def slots_to_lifecycle(block: "Block") -> Dict[str, List[str]]:
    """Stage -> [present slot names] for this block (representation only).

    Reports, per lifecycle stage, which of its mapped anatomy slots are
    present (non-empty). Neither asserts nor gates — IB6 owns the
    malformed-block verdict. Distinct from the five PAGE TYPES (see
    :data:`LIFECYCLE_STAGES`).
    """
    return {
        stage: [slot for slot in slots if _slot_present(block, slot)]
        for stage, slots in _STAGE_SLOTS.items()
    }


def lifecycle_stage_coverage(block: "Block") -> Dict[str, bool]:
    """Stage -> bool(any mapped slot is non-empty).

    The "malformed if a slot is absent" QA-4 JUDGEMENT is IB6's validator —
    this is presence DATA only.
    """
    return {
        stage: bool(present)
        for stage, present in slots_to_lifecycle(block).items()
    }


# Per-block-type interaction presence markers — REUSE of the existing
# ``data-cf-component`` tokens the attr helpers already compute
# (``_self_check_question_attrs`` / ``_activity_attrs`` /
# ``_flip_card_grid_attrs``). The marker signals "this block carries an
# interaction"; it is NOT the interaction HTML (lossless, deterministic). For
# block types with no dedicated component token the block_type itself is the
# marker. Non-interactive types are absent from this map → ``interaction``
# stays None (IB1.5).
_INTERACTION_COMPONENT_MARKERS: Dict[str, str] = {
    "self_check_question": "self-check",
    "activity": "activity",
    "flip_card_grid": "flip-card",
    "discussion_prompt": "discussion_prompt",
    "reflection_prompt": "reflection_prompt",
    "problem": "problem",
    "scenario": "scenario",
}


def derive_anatomy_slots(block: "Block") -> "Block":
    """Return a NEW block with anatomy slots filled ONLY where deterministic.

    Pure, deterministic, NO-LLM back-derivation (the gap JSON's explicit
    "derive only where deterministic, leave None otherwise" guardrail):

    * ``heading``: parsed from ``content`` HTML via the existing
      :func:`_parse_provider_page_html`; or a ``heading``/``title`` key when
      ``content`` is a dict. Else None.
    * ``interaction``: a SHORT machine MARKER (the existing ``data-cf-component``
      token) for the interaction-bearing block types — presence signal, not the
      interaction HTML. None for non-interactive types.
    * ``purpose_tag``: ``self.purpose or self.teaching_role`` (the loose-mapping
      consolidation the gap text calls out). Else None.
    * ``feedback`` / ``transition``: NEVER back-derived (parsing LLM prose is
      lossy) — always None from this helper; reserved for an authoring wave.

    NOT called anywhere in the emit path by default (no caller wired in IB1 →
    zero behavior change). It exists as the API a later wave / the IB1.4 emit
    test invokes. Always returns a NEW frozen instance; never mutates the input.
    """
    updates: Dict[str, Any] = {}

    # heading
    if block.heading is None:
        content = block.content
        derived_heading: Optional[str] = None
        if isinstance(content, str):
            derived_heading, _ = _parse_provider_page_html(content)
        elif isinstance(content, dict):
            raw = content.get("heading") or content.get("title")
            if isinstance(raw, str) and raw.strip():
                derived_heading = raw.strip()
        if derived_heading:
            updates["heading"] = derived_heading

    # interaction (presence marker)
    if block.interaction is None:
        marker = _INTERACTION_COMPONENT_MARKERS.get(block.block_type)
        if marker:
            updates["interaction"] = marker

    # purpose_tag
    if block.purpose_tag is None:
        purpose_tag = block.purpose or block.teaching_role
        if purpose_tag:
            updates["purpose_tag"] = purpose_tag

    if not updates:
        return block
    return dataclasses.replace(block, **updates)


# ---------------------------------------------------------------------------
# IB4 — UDL multiple-means coverage detector (QA-13 / D7). DETERMINISTIC,
# pure string/HTML inspection — NO LLM, NO embeddings. Anti-fabrication: derive
# only from what is present; leave empty / None when nothing resolves. Feeds
# IB4.5's UdlCoverageValidator (which derives on read when these fields are
# empty) and IB6's Engagement + Accessibility/UDL quality dimensions.
# ---------------------------------------------------------------------------

# Representation-mode HTML markers (count of DISTINCT modes present). REUSE of
# the existing body-tag vocabulary — prose <p>, tabular <table>, figure
# <img>/<figure>, formula <math> (or a data-cf-content-type="formula"
# wrapper / a <span class="math"> token), list <ul>/<ol>, reveal <details>.
# NOTE: the "prose" mode is detected by :func:`_udl_has_prose` (a block-level
# prose container with visible text), NOT by a single regex — a <div>/<section>
# wrapped paragraph is genuinely prose, but a <table>-only block must NOT
# register prose just because its cells carry text. The other modes stay
# regex-driven. The "prose" mode is decided by _udl_has_prose, not by this tuple.
#
# A literal <p> is an explicit prose container and ALWAYS counts (preserves the
# original byte-stable behaviour where any <p> registered prose). A
# <div>/<section> is a generic structural container, so it counts as prose only
# when it carries non-trivial visible text (the IB4 detection-artifact fix:
# <div>-wrapped prose with no literal <p>) — this keeps a <div>-wrapping a
# table/empty container from over-counting prose.
_UDL_PROSE_P_RE = re.compile(r"(?is)<p[\s>]")
_UDL_PROSE_BLOCK_CONTAINER_RE = re.compile(r"(?is)<(?:div|section)[\s>]")

_UDL_REPRESENTATION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("table", re.compile(r"(?is)<table[\s>]")),
    ("image", re.compile(r"(?is)<(?:img|figure)[\s>]")),
    ("formula", re.compile(r"(?is)<math[\s>]|class=\"[^\"]*\bmath\b|data-cf-content-type=\"formula\"")),
    ("list", re.compile(r"(?is)<(?:ul|ol)[\s>]")),
    ("reveal", re.compile(r"(?is)<details[\s>]")),
)

# Minimum visible-text length (chars, tags stripped) for a generic
# <div>/<section> container to count as the "prose" representation mode. A
# <div>-wrapped paragraph clears it; an empty <div></div> wrapper does not.
# (Only consulted for <div>/<section>; a literal <p> always counts.)
_UDL_PROSE_MIN_TEXT_CHARS: int = 20

# Tag-stripper used to recover visible text for the prose-presence check.
_UDL_TAG_STRIP_RE = re.compile(r"(?is)<[^>]+>")


def _udl_has_prose(text: str) -> bool:
    """Detect the "prose" representation mode.

    Prose is present when EITHER a literal ``<p>`` is present (always counts —
    byte-stable with the original behaviour), OR a generic block-level
    container (``<div>``/``<section>``) is present AND the visible text (tags
    stripped) is non-trivial (>= ``_UDL_PROSE_MIN_TEXT_CHARS``). The latter
    counts ``<div>``/``<section>``-wrapped prose that carries no literal
    ``<p>`` (the IB4 detection-artifact fix) WITHOUT over-counting a
    ``<table>``-only block whose cells happen to carry text — a ``<table>``
    with no block-level prose container registers no prose.
    """
    if not text:
        return False
    if _UDL_PROSE_P_RE.search(text):
        return True
    if not _UDL_PROSE_BLOCK_CONTAINER_RE.search(text):
        return False
    visible = _UDL_TAG_STRIP_RE.sub(" ", text)
    return len(visible.strip()) >= _UDL_PROSE_MIN_TEXT_CHARS


def _derive_udl_representation_modes(block: "Block") -> "frozenset[str]":
    """Return the SET of distinct representation-mode names present in a block.

    Single source of truth for both the per-block ``n_representations`` count
    (``len(...)`` of this set) and the page-level union the
    ``UdlCoverageValidator`` aggregates. Pure string/HTML inspection — NO LLM,
    NO embeddings; anti-fabrication (only modes whose markers are literally
    present are returned).
    """
    text = _udl_content_text(block)
    modes: set = set()
    if _udl_has_prose(text):
        modes.add("prose")
    for mode, pattern in _UDL_REPRESENTATION_PATTERNS:
        if pattern.search(text):
            modes.add(mode)
    return frozenset(modes)


# block_type -> learner response/expression mode (Action/Expression network).
_UDL_RESPONSE_BY_BLOCK_TYPE: Dict[str, str] = {
    "self_check_question": "select",
    "assessment_item": "select",
    "activity": "construct",
    "problem": "construct",
    "reflection_prompt": "reflect",
    "discussion_prompt": "discuss",
    "flip_card_grid": "recall",
}

# Deterministic real-world / scenario phrase markers for the engagement /
# autonomy affordance (Engagement/Affective network). Lower-cased substring
# match over the rendered content text.
_UDL_REAL_WORLD_MARKERS: Tuple[str, ...] = (
    "real-world", "real world", "in practice", "everyday", "scenario",
    "imagine", "suppose you", "consider a situation",
)


def _udl_content_text(block: "Block") -> str:
    """Return the inspectable HTML/text body of a block for UDL detection."""
    content = block.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts: List[str] = []
        for value in content.values():
            if isinstance(value, str):
                parts.append(value)
        return " ".join(parts)
    return ""


def _derive_udl_coverage(
    block: "Block",
) -> "Tuple[int, Tuple[str, ...], Optional[str]]":
    """Deterministically derive ``(n_representations, response_formats, engagement_affordance)``.

    Pure string/HTML inspection — NO LLM, NO embeddings. Anti-fabrication:
    derive only from what is present; leave 0 / () / None when nothing
    resolves.

    * ``n_representations`` = count of DISTINCT present representation modes
      (prose / table / image / formula / list / reveal).
    * ``response_formats`` = the block_type's learner action/expression mode
      (single-element tuple) when one is mapped; () otherwise.
    * ``engagement_affordance`` = a deterministic autonomy hook:
      ``reflection`` for reflection_prompt; ``tiered_resource`` for a
      prereq_set / tiered block; ``real_world`` when a real-world/scenario
      phrase is present; None otherwise.
    """
    text = _udl_content_text(block)

    # Single source of truth: the count is the size of the mode SET (so the
    # per-block n_representations and the validator's page-level union stay
    # consistent — including the <div>/<section> prose detection fix).
    n_representations = len(_derive_udl_representation_modes(block))

    response_formats: Tuple[str, ...] = ()
    rf = _UDL_RESPONSE_BY_BLOCK_TYPE.get(block.block_type)
    # recall_self_check — a recall-marked self_check_question is NO LONGER
    # recognition ("select"); its learner action is retrieval-practice. cloze →
    # "recall" (produce a single term); free_recall → "construct" (produce the
    # answer). Double-gated behind ED4ALL_RECALL_SELF_CHECK AND only-when-set, so
    # default-off resolution is byte-identical ("select").
    if (
        block.block_type == "self_check_question"
        and _recall_self_check_emit_enabled()
        and isinstance(block.recall_format, str)
        and block.recall_format.strip()
    ):
        rf = "recall" if block.recall_format.strip() == "cloze" else "construct"
    if rf:
        response_formats = (rf,)

    engagement_affordance: Optional[str] = None
    bt = block.block_type
    if bt == "reflection_prompt":
        engagement_affordance = "reflection"
    elif bt in ("prereq_set",):
        engagement_affordance = "tiered_resource"
    elif bt == "scenario":
        engagement_affordance = "real_world"
    else:
        lowered = text.lower()
        if any(marker in lowered for marker in _UDL_REAL_WORLD_MARKERS):
            engagement_affordance = "real_world"

    return n_representations, response_formats, engagement_affordance


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
class QualityScore:
    """One per-block quality-dimension score (IB2.3 — empty scaffold).

    The typed cell of the 8-dimension anchored 0–3 quality rubric (framework
    §6.2-6.3). One :class:`QualityScore` per applicable
    :data:`QUALITY_DIMENSIONS` member; a :class:`Block` carries a tuple of them
    in ``quality_rubric``. This wave defines the SHAPE only — no code populates
    it (IB6 scores). ``score is None`` means not-yet-scored / not-applicable;
    ``applicable=False`` records that the dimension does not apply to the block
    (so IB6's "mean of applicable dims" excludes it without confusing a
    None-because-unscored with a None-because-N/A).

    Frozen sibling of :class:`Touch`. Audit/scoring metadata only — the
    enclosing ``Block.quality_rubric`` field is EXCLUDED from
    ``compute_content_hash()`` (mirrors ``observed_bloom_level``).
    """

    dimension: str
    score: Optional[int] = None
    applicable: bool = True
    rationale: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dimension not in QUALITY_DIMENSIONS:
            raise ValueError(
                f"QualityScore.dimension must be one of {QUALITY_DIMENSIONS}; "
                f"got {self.dimension!r}"
            )
        if self.score not in _QUALITY_SCORE_VALUES:
            raise ValueError(
                f"QualityScore.score must be None or an anchored 0–3 int; "
                f"got {self.score!r}"
            )

    def to_jsonld(self) -> Dict[str, Any]:
        """Wire shape — keys for the JSON-LD ``qualityRubric`` array entry.

        Omits ``rationale`` when None so a populated-but-rationale-less score
        stays compact; ``score``/``applicable`` always present.
        """
        entry: Dict[str, Any] = {
            "dimension": self.dimension,
            "score": self.score,
            "applicable": self.applicable,
        }
        if self.rationale is not None:
            entry["rationale"] = self.rationale
        return entry


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

    IB1 six-slot anatomy (framework pp.13-17, QA-4): a well-formed block
    instantiates ``heading + purpose_tag + content(BODY) + interaction +
    feedback + transition``. ``content`` IS the canonical BODY slot — the
    other five are the new Optional ``heading``/``purpose_tag``/
    ``interaction``/``feedback``/``transition`` fields below. No redundant
    ``body`` field is minted (IB1.2). All five new slots are hash-excluded
    (IB1.3) and emitted only-when-set behind ``ED4ALL_BLOCK_ANATOMY``.
    """

    block_id: str
    block_type: str
    page_id: str
    sequence: int
    # IB1 — the canonical BODY slot of the six-slot anatomy (framework
    # pp.13-17). NO separate ``body`` field is minted: ``content`` IS the body.
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
    # IB1 — six-slot anatomy contract (framework pp.13-17, QA-4). FIVE new
    # Optional slots; the sixth slot (BODY) is already ``content`` (above) — do
    # NOT duplicate it. heading/purpose_tag are deterministically back-derivable
    # (derive_anatomy_slots); interaction/feedback/transition default None and
    # are populated only where deterministically inferrable or by a later
    # authoring wave. Representation only — addressable + validatable; NO
    # behavior change here. INTENTIONALLY excluded from compute_content_hash()
    # and emitted only-when-set + only-when-flag-on (ED4ALL_BLOCK_ANATOMY) so
    # existing hashes / snapshots stay byte-identical (mirrors
    # observed_bloom_level / objective_alignment).
    heading: Optional[str] = None
    purpose_tag: Optional[str] = None
    interaction: Optional[str] = None
    feedback: Optional[str] = None
    transition: Optional[str] = None
    # IB2.3 — 8-dimension quality-rubric DATA MODEL (empty scaffold). A tuple of
    # per-dimension QualityScore cells (alignment / cognitive_load / multimedia /
    # retrieval / feedback / engagement / accessibility / coherence on an
    # anchored 0–3 scale; framework §6.2-6.3). ALWAYS ``()`` after this wave — no
    # code populates it (IB6's scoring pass does). Audit/scoring metadata only —
    # INTENTIONALLY excluded from compute_content_hash() (a scoring retro-fit
    # must not drift every existing block hash) and JSON-LD-projected
    # only-when-non-empty (so emit stays byte-identical until IB6 populates),
    # mirroring observed_bloom_level / objective_alignment.
    quality_rubric: Tuple[QualityScore, ...] = ()
    # IB3.4 — anchored-rubric for Evaluate/Create scored blocks (framework
    # pp.26-33, 5.2, B14, B11). An exemplar-anchored rubric (criteria bands +
    # exemplar anchors) published BEFORE the task — required for valid scoring
    # of the highest-Bloom work. Shape:
    #   {"criteria": [{"name": str,
    #                  "bands": [{"level": ..., "descriptor": str,
    #                             "exemplar": str}, ...]}],
    #    "published_before_task": bool}
    # Default ``None`` and INTENTIONALLY excluded from compute_content_hash()
    # (mirrors objective_alignment / quality_rubric) so a rubric retro-fit does
    # not drift any existing block hash. JSON-LD-projected as ``anchoredRubric``
    # only-when-non-None (additive; emit stays byte-identical until a block sets
    # it under ED4ALL_ALIGNMENT_VERB_TRIPLE). Populated only by an authoring
    # wave / operator; nothing in IB3 emits it (IB3.4's validator only READS it).
    anchored_rubric: Optional[Dict[str, Any]] = None
    # IB4 — UDL multiple-means coverage (QA-13 / D7). Audit-only, populated by
    # the deterministic UDL detector (see :func:`_derive_udl_coverage`);
    # EXCLUDED from compute_content_hash() (mirrors target_bloom / quality_rubric)
    # so a retro-fit never drifts an existing block hash. Empty defaults =>
    # legacy / flag-off blocks are byte-identical. Emitted to HTML/JSON-LD only
    # when ED4ALL_BLOCK_A11Y is set (default OFF). Feeds IB6's Engagement +
    # Accessibility/UDL quality dimensions.
    n_representations: int = 0          # count of distinct representation modes (prose, table, image, formula, list, ...)
    response_formats: Tuple[str, ...] = ()   # learner action/expression modes the block affords (recall, construct, select, reflect, discuss, ...)
    engagement_affordance: Optional[str] = None  # autonomy/engagement hook (choice, real_world, self_pace, reflection, tiered_resource)
    # IB5 — type-specific fields for the four framework-aligned pedagogical
    # block types (hook / multimedia / worked_example / diagram). All three are
    # additive Optional/empty defaults and INTENTIONALLY excluded from
    # compute_content_hash() (mirrors target_bloom / n_representations) so a
    # retro-fit never drifts an existing block hash; empty defaults => legacy /
    # flag-off blocks are byte-identical. Emitted to HTML/JSON-LD only when
    # ED4ALL_NEW_BLOCK_TYPES is set (default OFF). Reused (NOT redeclared): the
    # IB4 UDL fields above carry the multiple-means coverage for the new types.
    fade_state: Optional[str] = None     # B05 worked_example fade stage: worked | completion | independent
    long_description: Optional[str] = None  # B06 diagram structured long-description text
    media_a11y: Tuple[str, ...] = ()     # B04 present time-based-media track tokens (captions, audio_description, transcript, controls)
    # IB7.4 — slot-edit escalation annotation (framework p.139 Step 4: differentiate
    # by DEMAND — change slot weights / verb — NOT by ornament / type-swap). When the
    # planner's lifecycle pass faces an over-escalated block it FIRST stamps a heavier
    # interaction/feedback slot weight here (e.g. {"interaction": "heavy"}) before any
    # IB7.6 type re-route. Additive Optional/None default and INTENTIONALLY excluded
    # from compute_content_hash() (mirrors target_bloom / quality_rubric / the IB5
    # fields) so a slot-weight retro-fit never drifts an existing block hash; the
    # default-None state keeps legacy / flag-off blocks byte-identical. Read by the
    # planner reasoning only; not projected to HTML/JSON-LD.
    anatomy_slot_weights: Optional[Dict[str, Any]] = None
    # FR-INT-05 — per-distractor misconception feedback (B07 knowledge-check /
    # B14 graded assessment). A mapping of option key/text -> the misconception-
    # targeted "why this is wrong" feedback string the framework's Feedback
    # dimension (D5 / QA-8) requires for an interaction's distractors. Additive
    # Optional/None default and INTENTIONALLY excluded from compute_content_hash()
    # (the hash payload is an explicit 5-key allowlist; mirrors target_bloom /
    # quality_rubric / anatomy_slot_weights) so an option-feedback retro-fit
    # never drifts an existing block hash; the default-None state keeps legacy /
    # flag-off blocks byte-identical. Read by the IB6.3 InteractionFeedback-
    # Validator's _check_misconception_targeting arm (gated by
    # ED4ALL_BLOCK_QUALITY_RUBRIC). Not projected to HTML/JSON-LD here.
    option_feedback: Optional[Dict[str, str]] = None
    # FR-A11Y-03 — typed B12 callout kind (note / tip / warning / example /
    # key-idea). Drives the redundant non-color coding (visible LABEL + icon +
    # border) in the generate_course callout renderer behind ED4ALL_CALLOUT_TYPED
    # (default OFF; byte-stable when unset). Additive Optional/None default and
    # INTENTIONALLY excluded from compute_content_hash() (the hash payload is an
    # explicit 5-key allowlist; mirrors target_bloom / anatomy_slot_weights) so a
    # kind retro-fit never drifts an existing block hash; the default-None state
    # keeps legacy / flag-off blocks byte-identical. Read by the
    # CalloutStructureValidator (CALLOUT_NO_KIND) + the renderer.
    callout_kind: Optional[str] = None
    # FR-PLAN-01 — the Ch.5 14-type activity catalog → planner-selected
    # interaction type for this block (e.g. ``multiple_choice`` / ``drag_drop`` /
    # ``fill_in_blank`` / ``short_answer`` / ``matching`` / ``hotspot`` / …). A
    # DETERMINISTIC resolver (``lib/generation/block_planner.py::
    # _resolve_interaction_types``) selects it per block from its framework
    # B-code × catalog ``bloom_fit`` × the catalog ``default_activity_types``,
    # gated behind ``ED4ALL_DYNAMIC_BLOCK_PLAN`` (identity-no-op when off).
    # Threaded into the rewrite prompt so the rewrite tier authors to the
    # selected interaction type. Additive Optional/None default and
    # INTENTIONALLY excluded from compute_content_hash() (the hash payload is an
    # explicit 5-key allowlist; mirrors target_bloom / callout_kind / the IB5
    # fields) so an interaction-type retro-fit never drifts an existing block
    # hash; the default-None state keeps legacy / flag-off blocks byte-identical.
    interaction_type: Optional[str] = None
    # FR-INT-04 — B10 three-move discussion protocol (post -> respond ->
    # synthesize). A real graded discussion is not a single prompt; the
    # framework's B10 contract is a three-move protocol: an initial POST, a
    # required RESPOND to peers, and a SYNTHESIZE move that consolidates the
    # thread. ``discussion_protocol`` carries the structured move map
    # ({"post": str, "respond": str, "synthesize": str}); ``discussion_bloom_verb``
    # records the target Bloom verb the synthesize move drives to (a real
    # discussion climbs to Create). Both additive Optional/None defaults and
    # INTENTIONALLY excluded from compute_content_hash() (the hash payload is an
    # explicit 5-key allowlist; mirrors target_bloom / callout_kind /
    # interaction_type) so a protocol retro-fit never drifts an existing block
    # hash; the default-None state keeps legacy / flag-off blocks byte-identical.
    # Read by the B10ProtocolValidator (DISCUSSION_PROTOCOL_COLLAPSED) + the
    # discussion renderer (the three-move scaffold), both gated by
    # ED4ALL_NEW_BLOCK_TYPES (the same flag the B10 type rides). Not projected to
    # HTML/JSON-LD here.
    discussion_protocol: Optional[Dict[str, Any]] = None
    discussion_bloom_verb: Optional[str] = None
    # FR-INT-03 — B11 reflection predict-then-reveal calibration. A real
    # reflection does not merely ASK; it captures the learner's PREDICTION /
    # judgment, then REVEALS the benchmark / expert answer, then gives
    # CALIBRATION feedback comparing the two (the predict-then-reveal pattern
    # that makes self-assessment honest). ``prediction_prompt`` = the captured-
    # judgment prompt; ``reveal_content`` = the benchmark/expert answer revealed
    # after the prediction; ``calibration_feedback`` = the compare-your-judgment
    # feedback. All three additive Optional/None defaults and INTENTIONALLY
    # excluded from compute_content_hash() (the hash payload is an explicit 5-key
    # allowlist; mirrors discussion_protocol / target_bloom) so a calibration
    # retro-fit never drifts an existing block hash; the default-None state keeps
    # legacy / flag-off blocks byte-identical. Read by the interaction_feedback
    # REFLECTION_NO_CAPTURE arm + the anatomy benchmark-in-feedback-slot check
    # (both ride existing gates) + the reflection renderer's <details> predict-
    # then-reveal scaffold, all gated by ED4ALL_REFLECTION_CALIBRATION.
    prediction_prompt: Optional[str] = None
    reveal_content: Optional[str] = None
    calibration_feedback: Optional[str] = None
    # C3-4 — B07 self-check CONFIDENCE capture (metacognition / calibration).
    # The framework wants a knowledge-check to capture the learner's CONFIDENCE /
    # certainty ALONGSIDE their answer, so the learner can later compare their
    # confidence against correctness (the calibration loop that makes
    # self-assessment honest; the same theme as the B11 predict-then-reveal
    # work). ``confidence_prompt`` carries the confidence-capture prompt text
    # (e.g. "How sure are you of this answer?"); the renderer projects it into a
    # radio / scale control on the B07 self-check. Additive Optional/None default
    # and INTENTIONALLY excluded from compute_content_hash() (the hash payload is
    # an explicit 5-key allowlist; mirrors prediction_prompt / self_rating_prompt
    # / target_bloom) so a confidence-capture retro-fit never drifts an existing
    # block hash; the default-None state keeps legacy / flag-off blocks
    # byte-identical. Read by the interaction_feedback
    # SELF_CHECK_NO_CONFIDENCE_CAPTURE arm (rides the existing gate) + the
    # self-check renderer's confidence-capture control, both gated by
    # ED4ALL_REFLECTION_CALIBRATION (reused — confidence capture is the same
    # calibration theme). Not projected to HTML/JSON-LD here.
    confidence_prompt: Optional[str] = None
    # C3-3 — B01 objective-block constructive-alignment ENRICHMENT (framework
    # B01, constructive alignment). A well-authored learning-objective block
    # carries the full alignment surface: the Bloom (verb · level · knowledge-
    # type) chip (already emitted as a block-level attr via _bloom_triple_attrs —
    # NOT re-added here), a learner confidence / self-rating slot, and the
    # objective→assessment thread (which assessment(s) evidence this objective).
    # ``self_rating_prompt`` carries the confidence-capture prompt text;
    # ``objective_assessment_thread`` carries the structured map
    # ({"prompt": str, "assessment_refs": [str, ...]}) linking this objective to
    # the blocks/items that assess it. Both additive Optional/None defaults and
    # INTENTIONALLY excluded from compute_content_hash() (the hash payload is an
    # explicit 5-key allowlist; mirrors target_bloom / discussion_protocol / the
    # IB5 fields) so an enrichment retro-fit never drifts an existing block hash;
    # the default-None state keeps legacy / flag-off blocks byte-identical. Read
    # by the generate_course objective renderer (the confidence slot + the
    # objective↔assessment <details> thread), gated by ED4ALL_BLOCK_ANATOMY (the
    # same flag the six-slot anatomy rides). Not projected to HTML/JSON-LD here.
    self_rating_prompt: Optional[str] = None
    objective_assessment_thread: Optional[Dict[str, Any]] = None
    # FR-INT-06 — B09 case/scenario authoring MODE. A real B09 block is one of
    # three escalating forms: ``case`` (a static worked situation the learner
    # analyzes), ``scenario`` (a situated decision the learner makes), or
    # ``branching`` (a multi-step decision tree whose paths diverge by choice).
    # The mode is selected by-Bloom in the dynamic block planner
    # (``lib/generation/block_planner.py``, gated ED4ALL_DYNAMIC_BLOCK_PLAN —
    # identity-no-op when off) and drives the renderer's debrief scaffold (a B09
    # block MUST end with a debrief in its transition/consolidate slot). Additive
    # Optional/None default and INTENTIONALLY excluded from
    # compute_content_hash() (the hash payload is an explicit 5-key allowlist;
    # mirrors target_bloom / callout_kind / the IB5 fields) so a mode retro-fit
    # never drifts an existing block hash; the default-None state keeps legacy /
    # flag-off blocks byte-identical. Read by the B09DebriefValidator
    # (SCENARIO_DEBRIEF_MISSING) + the scenario renderer's debrief scaffold, both
    # gated by ED4ALL_NEW_BLOCK_TYPES. Not projected to HTML/JSON-LD here.
    scenario_mode: Optional[str] = None
    # recall_self_check — B07 self-check RECALL-format variant. A self-check can
    # be recognition (MCQ, the legacy radio render) OR retrieval-practice:
    # ``free_recall`` (produce the answer, reveal behind a ``<details>``) or
    # ``cloze`` (a ``____`` fill-in-the-blank). The deterministic planner pass
    # ``lib/generation/recall_self_check.py::apply_recall_self_check`` stamps this
    # on already-taught-CO self_check_question blocks (spaced retrieval). Additive
    # Optional/None default and INTENTIONALLY excluded from compute_content_hash()
    # (the hash payload is an explicit 5-key allowlist; mirrors scenario_mode /
    # the IB5 fields) so a recall retro-fit never drifts an existing block hash;
    # the default-None state keeps legacy / flag-off blocks byte-identical. Read
    # by the structured renderer's recall branch + the rewrite-tier cloze contract
    # + the corrected UDL response-format map + the recall_self_check_format gate,
    # all gated by ED4ALL_RECALL_SELF_CHECK.
    recall_format: Optional[str] = None     # B07 recall variant: free_recall | cloze
    # misconception_rich — B03/B12 named subject-specific misconception +
    # productive-failure (predict → reveal → reconcile). ``mc_named_concept`` =
    # the grounded faulty-model concept slug (named ONLY from the block's own
    # concept_tags / domain vocabulary — None when unresolved, never invented);
    # ``mc_predict_prompt`` = the predict-then-reveal prompt before the
    # correction; ``mc_reconcile`` = the reconcile step (why the named model
    # fails). All three additive Optional/None defaults and INTENTIONALLY excluded
    # from compute_content_hash() (the hash payload is an explicit 5-key
    # allowlist; mirrors scenario_mode / the FR-INT-03 calibration fields) so a
    # misconception retro-fit never drifts an existing block hash; the default-None
    # state keeps legacy / flag-off blocks byte-identical. Read by the misconception
    # render scaffold + the rewrite-tier productive-failure contract + the
    # _misconception_jsonld conceptId/productiveFailure emit + the
    # misconception_productive_failure gate, all gated by ED4ALL_MISCONCEPTION_RICH.
    mc_named_concept: Optional[str] = None
    mc_predict_prompt: Optional[str] = None
    mc_reconcile: Optional[str] = None

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

    def quality_score_for(self, dimension: str) -> Optional[QualityScore]:
        """Return the :class:`QualityScore` for ``dimension`` if present (IB2.3).

        Reader-only convenience over the ``quality_rubric`` tuple — no scoring
        (IB6 populates the tuple). Returns ``None`` when no cell for that
        dimension exists (the always-empty state after this wave).
        """
        for qs in self.quality_rubric:
            if qs.dimension == dimension:
                return qs
        return None

    def compute_content_hash(self) -> str:
        """SHA-256 hex of the canonical Block payload.

        Excludes ``touched_by``, ``sequence``, ``validation_attempts``,
        ``escalation_marker``, ``observed_bloom_level``,
        ``bloom_alignment``, ``objective_alignment``, ``target_bloom``,
        the IB1 six-slot anatomy metadata slots ``heading`` / ``purpose_tag``
        / ``interaction`` / ``feedback`` / ``transition``, the IB2.3
        ``quality_rubric`` audit/scoring tuple, the IB3.4
        ``anchored_rubric`` Evaluate/Create scoring rubric, the IB4 UDL
        coverage fields ``n_representations`` / ``response_formats`` /
        ``engagement_affordance``, and the IB5 type-specific fields
        ``fade_state`` / ``long_description`` / ``media_a11y``, the FR-INT-05
        ``option_feedback`` per-distractor feedback map, the FR-A11Y-03
        ``callout_kind`` typed-callout enum, the FR-PLAN-01
        ``interaction_type`` planner-selected activity type, the FR-INT-04
        ``discussion_protocol`` / ``discussion_bloom_verb`` B10 three-move
        protocol fields, and the FR-INT-03 ``prediction_prompt`` /
        ``reveal_content`` / ``calibration_feedback`` B11 predict-then-reveal
        calibration fields, the C3-3 ``self_rating_prompt`` /
        ``objective_assessment_thread`` B01 objective-enrichment fields, the
        C3-4 ``confidence_prompt`` B07 self-check confidence-capture field, the
        recall_self_check ``recall_format`` B07 recall variant, and the
        misconception_rich ``mc_named_concept`` / ``mc_predict_prompt`` /
        ``mc_reconcile`` B03/B12 productive-failure fields so a
        touch-only / budget-only / classifier-retrofit / objective-
        delivery-retrofit / anatomy-slot-back-derivation / rubric-scoring /
        anchored-rubric-attach / udl-coverage-retrofit / ib5-field-attach /
        option-feedback-attach / callout-kind-attach
        revision keeps a stable hash. The payload below is an explicit 5-key
        allowlist, so any new field is excluded by construction. ``content`` (the BODY slot) IS in the payload; the other
        five anatomy slots are derived-or-authored metadata ABOUT the same
        content, so hashing them would drift every existing block hash on a
        back-derivation retrofit — exactly the failure the
        ``observed_bloom_level`` exclusion exists to prevent. The hash exists
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
            # Wave-2 block-variety additions: component / section-ish
            # wrappers (.scenario-card / .problem-card / .vocab-card /
            # .formula-card / .checklist). Source-id attrs + the gated
            # block-id are the only metadata they carry.
            "scenario",
            "problem",
            "vocab_card",
            "formula",
            "checklist",
            # Issue I6 instruction-palette-v2: ``table`` (<table> wrapper),
            # ``acronym`` (<dl> wrapper), ``key_idea`` (<aside> wrapper).
            # Same wrapper-only metadata posture as the Wave-2 additions
            # above — source-id attrs + the gated block-id. The
            # content-type backstop (_REWRITE_BLOCK_TYPE_CONTENT_TYPE in
            # pipeline_tools.py) stamps data-cf-content-type on the rewrite
            # path so BlockContentTypeValidator passes.
            "table",
            "acronym",
            "key_idea",
            # IB5 framework-aligned types: hook (B02) / worked_example (B05) are
            # heading content-section-ish wrappers; multimedia (B04) / diagram
            # (B06) carry their a11y richness in the rendered HTML body
            # (validated by rewrite_html_shape's IB5 arms), not in data-cf-*
            # attrs. All four are wrapper-only at the attr surface — source-id
            # attrs + the gated block-id (mirrors the I6 posture). Only
            # selectable / constructed on the dynamic path under
            # ED4ALL_NEW_BLOCK_TYPES, so these arms are dead on legacy runs.
            "hook",
            "multimedia",
            "worked_example",
            "diagram",
        }:
            # Wrapper-only blocks (the inline `<section>` wrappers in
            # `generate_week`). Source-id attrs only.
            attrs = _source_attr_string(self.source_ids, self.source_primary)
            # IB5 — worked_example carries an additional data-cf-fade-state attr
            # when the fade stage is set. DOUBLE-gated behind
            # ED4ALL_NEW_BLOCK_TYPES (default OFF) AND only-when-set, so default-
            # off emit is byte-identical (mirrors the IB4 UDL attr posture).
            if (
                block_type == "worked_example"
                and self.fade_state
                and _new_block_types_emit_enabled()
            ):
                attrs += f' data-cf-fade-state="{_esc(self.fade_state)}"'
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
        # IB4 — UDL multiple-means coverage attrs. DOUBLE-gated: behind
        # ED4ALL_BLOCK_A11Y (default OFF) AND only-when-set, so default-off emit
        # is byte-identical. Mirrors the anatomy / observed-bloom emit posture.
        if _block_a11y_emit_enabled():
            if self.n_representations:
                attrs += f' data-cf-udl-representations="{int(self.n_representations)}"'
            if self.response_formats:
                joined = ",".join(_esc(rf) for rf in self.response_formats if rf)
                if joined:
                    attrs += f' data-cf-udl-response-formats="{joined}"'
            if self.engagement_affordance:
                attrs += (
                    f' data-cf-udl-engagement="{_esc(self.engagement_affordance)}"'
                )
        # IB6.5 — universal (verb · level · knowledge-type) triple chip. Behind
        # ED4ALL_BLOCK_QUALITY_RUBRIC (default OFF) so legacy snapshots stay
        # byte-identical. Only emitted on pedagogical blocks (chrome has no
        # framework B-code) and only-when the triple resolves. The existing
        # objective-only ``data-cf-cognitive-domain`` (``_objective_attrs``) is
        # unchanged; this promotes it to every block + adds the combined chip.
        if _block_quality_rubric_emit_enabled():
            attrs += self._bloom_triple_attrs()
        return attrs

    def _knowledge_type(self) -> Optional[str]:
        """Resolve the block's knowledge-type axis (the framework's 2nd axis).

        Prefers the authored ``cognitive_domain`` field (factual / conceptual /
        procedural / metacognitive). Falls back to deriving it from
        ``bloom_level`` via :func:`lib.ontology.bloom.bloom_to_cognitive_domain`
        so a block carrying only a Bloom level still resolves a triple. Returns
        ``None`` when neither resolves.
        """
        if isinstance(self.cognitive_domain, str) and self.cognitive_domain.strip():
            return self.cognitive_domain.strip().lower()
        if isinstance(self.bloom_level, str) and self.bloom_level.strip():
            try:
                from lib.ontology.bloom import bloom_to_cognitive_domain

                derived = bloom_to_cognitive_domain(self.bloom_level.strip().lower())
                if isinstance(derived, str) and derived.strip():
                    return derived.strip().lower()
            except Exception:  # noqa: BLE001 — fall through to None
                return None
        return None

    def _bloom_triple_attrs(self) -> str:
        """IB6.5 universal chip attrs: cognitive-domain + the combined triple.

        ``chrome`` (no framework B-code) and blocks with no resolvable
        knowledge-type emit nothing. The combined ``data-cf-bloom-triple`` only
        fires when ALL three axes (verb, level, knowledge-type) resolve — a
        machine-checkable single chip. The standalone ``data-cf-cognitive-
        domain`` fires whenever the knowledge-type resolves (so a level-only
        block still carries the 2nd axis universally).
        """
        from lib.ontology.framework_blocks import framework_block_for

        if framework_block_for(self.block_type) is None:
            return ""
        ktype = self._knowledge_type()
        if not ktype:
            return ""
        out = ""
        # Promote cognitive-domain to a universal chip (objectives already
        # carry it via _objective_attrs; avoid a duplicate attr on objectives).
        if self.block_type != "objective":
            out += f' data-cf-cognitive-domain="{_esc(ktype)}"'
        verb = self.bloom_verb.strip().lower() if isinstance(self.bloom_verb, str) and self.bloom_verb.strip() else None
        level = self.bloom_level.strip().lower() if isinstance(self.bloom_level, str) and self.bloom_level.strip() else None
        if verb and level:
            out += f' data-cf-bloom-triple="{_esc(verb)}.{_esc(level)}.{_esc(ktype)}"'
        return out

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

    def _resolved_content_type(self) -> str:
        """Return an enum-valid content_type for this block's HTML attrs.

        Prefers a VALID ``content_type_label``; repairs an invalid one to the
        block_type's canonical ChunkType default (``lib/ontology/content_types``);
        falls back to the raw ``block_type`` only when no default exists (e.g.
        scaffolding types the content_type gate skips anyway).
        """
        from lib.ontology.content_types import (
            is_valid_content_type,
            resolve_block_content_type,
        )

        label = self.content_type_label
        if isinstance(label, str) and label.strip() and is_valid_content_type(
            label.strip()
        ):
            return label.strip()
        return resolve_block_content_type(self.block_type) or self.block_type

    def _content_section_attrs(self) -> str:
        """Match `_render_content_sections:1018-1035` (heading attrs).

        ``content_type_label`` carries the resolved content_type (or it
        falls back to a canonical ChunkType default for ``block_type``);
        ``key_terms`` carries the term slugs already slugified by the renderer;
        ``bloom_range`` is the section span string.

        An invalid ``content_type_label`` (e.g. a domain term like
        ``expression`` the planner / LLM wrote into the content-type slot) is
        REPAIRED to the block_type's canonical default rather than emitted
        verbatim — emitting it would fail BlockContentTypeValidator's enum
        check (OUTLINE_BLOCK_INVALID_CONTENT_TYPE). The raw ``block_type`` (not
        itself a ChunkType member, e.g. ``concept``) is used only as the last
        resort when no canonical default exists.
        """
        content_type = self._resolved_content_type()
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
        # misconception_rich — emit the named-concept link + productive-failure
        # object ONLY when ED4ALL_MISCONCEPTION_RICH is on AND the fields are set
        # (double-gated, mirroring the worked_example fade_state attr posture), so
        # flag-off / legacy emit is byte-identical.
        if _misconception_rich_emit_enabled():
            if isinstance(self.mc_named_concept, str) and self.mc_named_concept.strip():
                entry["conceptId"] = self.mc_named_concept.strip()
            pf: Dict[str, Any] = {}
            if isinstance(self.mc_predict_prompt, str) and self.mc_predict_prompt.strip():
                pf["predict"] = self.mc_predict_prompt.strip()
            if isinstance(self.mc_reconcile, str) and self.mc_reconcile.strip():
                pf["reconcile"] = self.mc_reconcile.strip()
            if pf:
                entry["productiveFailure"] = pf
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
        # IB2.4 — 8-dimension quality rubric (framework §6.2-6.3). Emit a
        # ``qualityRubric`` array ONLY when non-empty (mirrors the
        # objectiveAlignment / anatomy only-when-set guards). The field is
        # ALWAYS ``()`` this wave (nothing populates it — IB6 does), so this
        # projection NEVER fires on a real run and emit stays byte-identical.
        # camelCase keys mirror schemas/knowledge/courseforge_jsonld_v1
        # .schema.json::$defs.QualityScore.
        if self.quality_rubric:
            entry["qualityRubric"] = [qs.to_jsonld() for qs in self.quality_rubric]
        # IB3.4 — anchored rubric (framework pp.26-33, 5.2, B14, B11). Emit the
        # ``anchoredRubric`` object ONLY when non-None (mirrors the
        # qualityRubric / objectiveAlignment only-when-set guards). Nothing in
        # IB3 sets the field (the validator only reads it), so this projection
        # NEVER fires on a default run and emit stays byte-identical. camelCase
        # key mirrors schemas/knowledge/courseforge_jsonld_v1.schema.json::
        # $defs.Block.properties.anchoredRubric.
        if self.anchored_rubric is not None:
            entry["anchoredRubric"] = dict(self.anchored_rubric)
        # IB1 — six-slot anatomy contract (framework pp.13-17, QA-4). Emit a
        # nested ``anatomy`` sub-object carrying ONLY the non-None slots,
        # DOUBLE-gated: behind BOTH ``_anatomy_emit_enabled()`` (the new
        # ED4ALL_BLOCK_ANATOMY flag, default OFF) AND only-when-set. Default-OFF
        # flag ⇒ no ``anatomy`` key ⇒ byte-identical JSON-LD. The BODY slot is
        # the block's ``content`` (NOT repeated here). camelCase keys mirror
        # schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.BlockAnatomy.
        if _anatomy_emit_enabled():
            anatomy: Dict[str, Any] = {}
            if self.heading is not None:
                anatomy["heading"] = self.heading
            if self.purpose_tag is not None:
                anatomy["purposeTag"] = self.purpose_tag
            if self.interaction is not None:
                anatomy["interaction"] = self.interaction
            if self.feedback is not None:
                anatomy["feedback"] = self.feedback
            if self.transition is not None:
                anatomy["transition"] = self.transition
            if anatomy:
                anatomy["lifecycle"] = slots_to_lifecycle(self)
                entry["anatomy"] = anatomy
        # C3-7 — IB2 canonical framework B-code (B01–B15). Emit the
        # ``frameworkBlock`` field carrying this block_type's primary framework
        # parent, resolved from the SoT catalog loader
        # (lib/ontology/framework_blocks.py::framework_block_for, backed by
        # Courseforge/config/block_catalog.yaml::framework_block). Gated behind
        # ``_anatomy_emit_enabled()`` (ED4ALL_BLOCK_ANATOMY, default OFF) so
        # legacy emit stays byte-identical / contentHash is unchanged. Emitted
        # only when the type resolves a B-code (None for the non-pedagogical
        # ``chrome`` scaffolding / any unknown type → no key). camelCase key
        # mirrors schemas/knowledge/courseforge_jsonld_v1.schema.json::
        # $defs.Block.properties.frameworkBlock. Excluded from contentHash
        # (resolved deterministically from block_type — never a content field).
        if _anatomy_emit_enabled():
            from lib.ontology.framework_blocks import framework_block_for

            fb_code = framework_block_for(self.block_type)
            if fb_code is not None:
                entry["frameworkBlock"] = fb_code
        # IB4 — UDL multiple-means coverage (QA-13 / D7). Emit a ``udlCoverage``
        # sub-object DOUBLE-gated: behind BOTH ``_block_a11y_emit_enabled()``
        # (the new ED4ALL_BLOCK_A11Y flag, default OFF) AND only-when-set.
        # Default-OFF flag ⇒ no ``udlCoverage`` key ⇒ byte-identical JSON-LD.
        # camelCase keys; feeds IB6's Engagement + Accessibility/UDL dimensions.
        if _block_a11y_emit_enabled():
            udl: Dict[str, Any] = {}
            if self.n_representations:
                udl["nRepresentations"] = int(self.n_representations)
            if self.response_formats:
                udl["responseFormats"] = list(self.response_formats)
            if self.engagement_affordance is not None:
                udl["engagementAffordance"] = self.engagement_affordance
            if udl:
                entry["udlCoverage"] = udl
        # IB5 — type-specific fields for the four framework-aligned pedagogical
        # block types (worked_example fadeState / diagram longDescription).
        # DOUBLE-gated: behind BOTH ``_new_block_types_emit_enabled()`` (the new
        # ED4ALL_NEW_BLOCK_TYPES flag, default OFF) AND only-when-set. Default-
        # OFF flag ⇒ no key ⇒ byte-identical JSON-LD (the four types are never
        # even constructed on a legacy run). camelCase keys; additive.
        if _new_block_types_emit_enabled():
            if self.block_type == "worked_example" and self.fade_state:
                entry["fadeState"] = self.fade_state
            if self.block_type == "diagram" and self.long_description:
                entry["longDescription"] = self.long_description
            if self.block_type == "multimedia" and self.media_a11y:
                entry["mediaA11y"] = list(self.media_a11y)
        # recall_self_check — emit the recall variant on a self_check_question.
        # DOUBLE-gated behind ED4ALL_RECALL_SELF_CHECK AND only-when-set, so
        # default-OFF emit is byte-identical (additive camelCase key).
        if (
            self.block_type == "self_check_question"
            and _recall_self_check_emit_enabled()
            and isinstance(self.recall_format, str)
            and self.recall_format.strip()
        ):
            entry["recallFormat"] = self.recall_format.strip()
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
# the canonical 21-type enum doesn't include ``procedure`` /
# ``comparison`` / ``definition`` / ``overview`` / ``summary`` /
# ``exercise`` — those resolve to ``content_type_label`` on the
# Block instead, while ``block_type`` stays in the canonical enum.
_CONTENT_SECTION_BLOCK_TYPES: frozenset = frozenset()
