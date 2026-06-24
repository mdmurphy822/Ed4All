"""Content-aware per-TO block planner (Wave-2 block-variety redesign, Part 3).

The keystone of the block-variety redesign. Replaces the FIXED per-week
block template (``MCP/tools/pipeline_tools.py::_PAGE_TYPE_BLOCK_PLAN``) with
a 70B-driven planner that, PER TERMINAL OBJECTIVE (week), chooses the block
sequence that best conveys THAT TO's content — so each week is
content-shaped, not template-filled.

Surface
-------

    plan_week_blocks(
        *,
        terminal_objective: Dict,        # {id, statement}
        chapter_objectives: List[Dict],  # [{id, statement, bloom_level}, ...]
        source_chunks: List[Dict],       # [{id, text, heading}, ...] (digest)
        catalog: List[Dict] | None = None,  # load_block_catalog() output
        provider: Optional[_BaseLLMProvider-like] = None,
        budget: Tuple[int, int] = (5, 12),
        capture: Optional[DecisionCapture] = None,
        course_code: str = "",
    ) -> WeekBlockPlan

``WeekBlockPlan`` carries:

- ``page_plan``: ``Dict[str, List[Tuple[block_type, target_bloom]]]`` keyed
  by the five canonical page types (``overview`` / ``content`` /
  ``application`` / ``self_check`` / ``summary``). This is the EXACT shape
  of ``_PAGE_TYPE_BLOCK_PLAN``, so the planner output drops into the
  existing page-descriptor assembly with no change to the
  ``_page_id_for`` page-file grouping mechanics.
- ``selected``: the validated, ordered list of
  ``{block_type, target_co_ids[], page_type, content_focus, target_bloom}``
  dicts the 70B chose (post-guardrail).
- ``fallback_used``: ``True`` when the deterministic fixed-plan fallback
  fired (LLM error / unparseable / empty), ``False`` on a real LLM plan.

How it prompts the 70B
----------------------

The planner builds ONE prompt per TO containing: the TO statement; its
child COs (id + statement + bloom); a bounded digest of the TO's source
chunk text (truncated to keep the prompt small); and the block catalog's
``use_when`` / ``bloom_fit`` menu. It asks the model to select an ORDERED
sequence of instruction blocks — picking the RIGHT block type for each
piece of content (a procedure→worked example/practice problem; a key
term→vocab_card; a common error→misconception; a real situation→scenario;
a checkpoint→self_check_question/reflection_prompt; a recap→
summary_takeaway/checklist) — and to return JSON: an ordered list of
``{block_type, target_co_ids, page_type, content_focus}`` within the
budget.

Guardrails (applied to the raw LLM output)
------------------------------------------

1. Every ``block_type`` MUST be in ``BLOCK_TYPES`` — unknown types are
   DROPPED.
2. ``page_type`` MUST be one of the five canonical page types — an
   out-of-set / missing value is repaired to a content-shape default for
   the chosen block type.
3. The block count is CLAMPED to ``[budget_min, budget_max]`` (excess
   blocks past the max are dropped; too-few triggers a default top-up to
   the min).
4. COVERAGE: every child CO MUST be covered by ≥1 block. A CO the 70B
   dropped gets a default ``concept`` block appended targeting it.
5. ``target_bloom`` is resolved per block (the LLM-declared bloom if valid,
   else the catalog ``bloom_fit`` floor, else the covered CO's bloom).

Fail-safe
---------

ANY LLM error, unparseable response, or empty selection → a DETERMINISTIC
fallback to the fixed per-page-type plan (``_default_page_plan``, a verbatim
copy of ``_PAGE_TYPE_BLOCK_PLAN``). The planner NEVER breaks the build.

Decision capture
----------------

One ``block_plan`` decision event per TO (chosen types, budget, coverage,
model, fallback-used) so the selection is replayable post-hoc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from lib.generation.block_catalog import load_block_catalog
from lib.ontology.bloom import BLOOM_LEVELS

logger = logging.getLogger(__name__)

__all__ = [
    "plan_week_blocks",
    "WeekBlockPlan",
    "BlockPlannerProvider",
    "CANONICAL_PAGE_TYPES",
    "DEFAULT_BLOCK_BUDGET",
    "detect_acronyms",
    "detect_canonical_mnemonics",
    "detect_tabular_content",
    "detect_procedure",
    "detect_media_reference",
    "detect_diagram_reference",
]

# System prompt for the dedicated planner provider — frames the 70B as an
# instructional designer choosing block types, NOT authoring HTML (so the
# rewrite tier's HTML-authoring system prompt does not bleed into planning).
_PLANNER_SYSTEM_PROMPT = (
    "You are an expert instructional designer. Your only job is to PLAN "
    "which instruction-block types best teach a given learning objective "
    "and its source content — you do NOT author the block bodies. You "
    "always respond with a single JSON object and nothing else."
)

# The five canonical page TYPES the outline assembly groups blocks into
# (mirrors ``MCP/tools/pipeline_tools.py::_WEEK_PAGE_TYPES``). The planner
# must assign every selected block to one of these so the existing
# ``_page_id_for`` page-file grouping is preserved.
CANONICAL_PAGE_TYPES: Tuple[str, ...] = (
    "overview",
    "content",
    "application",
    "self_check",
    "summary",
)

# Min / max blocks the 70B may select per week (per TO). The max is raised
# (was 12) to FUND the per-page-type floors below: a balanced week needs
# overview (≥3) + content (open) + application (≥4) + self_check (≥4) +
# summary (≥4) blocks, so a 12-block ceiling starved application / self_check
# / summary (findings 2/7/8 of the Sonnet-vs-7B review). 24 leaves room for
# the floors plus a content tier without clamping a content-rich week.
DEFAULT_BLOCK_BUDGET: Tuple[int, int] = (5, 24)

# Per-page-type minimum block counts (findings 2/7/8/18 of the Sonnet-vs-7B
# review). The planner pours budget into content pages and STARVES
# overview / application / self_check / summary (2-3 blocks each vs Sonnet's
# richer pages). These floors guarantee no page TYPE is threadbare: after
# coverage + budget the planner TOPS UP any short page with the
# page-appropriate filler in ``_PAGE_TYPE_FLOOR_FILLERS`` below.
#
# - self_check ≥ 4 self_check_question blocks (finding 7: 7B always emitted
#   2 blocks / 1 question vs Sonnet's 5).
# - application ≥ 4 blocks (finding 2).
# - summary ≥ 4 blocks (finding 8: 7B emitted 2 thin sections vs Sonnet's 5).
# - overview ≥ 3 blocks AND must carry the objective ENUMERATION (finding 18:
#   the overview must list the week's TO + all its COs — the leading
#   ``objective`` block is the enumeration anchor the consumer renders).
# - content has no floor (it is the page the planner naturally fills).
_PAGE_TYPE_FLOORS: Dict[str, int] = {
    "overview": 3,
    "content": 0,
    "application": 4,
    # self_check floor is 5 so that, after the typed floor places >= 4
    # self_check_question blocks (finding 7 — Sonnet ships 5 questions), the
    # variety top-up still adds a 5th block — a reflection_prompt (finding 6)
    # — to close the page the way Sonnet does (questions + a reflection).
    "self_check": 5,
    "summary": 4,
}

# Per-page-type minimum count of a SPECIFIC block type, enforced BEFORE the
# generic variety top-up. ``self_check`` finding 7 is a count of QUESTIONS,
# not of blocks: Sonnet's self-check page carries 5 questions + reveals while
# the 7B always emitted 1. So self_check guarantees >= 4 ``self_check_question``
# blocks first; the variety top-up (reflection_prompt, …) then runs only if
# the total still falls below ``_PAGE_TYPE_FLOORS`` (it won't, since 4
# questions already meets the 4-block floor — the reflection_prompt rides in
# as a 5th block via the +1 enrichment below).
_PAGE_TYPE_MIN_TYPED: Dict[str, Tuple[str, int]] = {
    "self_check": ("self_check_question", 4),
}

# Ordered filler specs used to TOP UP a short page to its floor. Each filler
# is a ``(block_type, target_bloom)`` pair. Deliberately DEPLOY the
# contracted-but-never-authored block types (findings 6/16: reflection_prompt,
# callout, misconception, discussion_prompt, prereq_set, flip_card_grid were
# authored 0 times) by giving them slots here, so a floor top-up makes them
# REACHABLE on a real run rather than dead catalog entries. Fillers cycle in
# order; the first filler for a page is the page's pedagogical anchor
# (overview→objective enumeration, self_check→a 4th question, …).
_PAGE_TYPE_FLOOR_FILLERS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "overview": (
        # finding 18 — the objective enumeration anchor (TO + every CO) must
        # lead the overview; prereq_set + explanation give it depth.
        ("objective", "understand"),
        ("prereq_set", "remember"),
        ("explanation", "understand"),
        ("callout", "understand"),
    ),
    "content": (
        # content has no floor, but if a floor is ever configured these
        # deploy concept + the I6 palette-v2 variety types (table / key_idea /
        # acronym, findings 5/7/16 — tabular content, emphasized principles,
        # mnemonics the 7B never authored) + misconception + flip_card_grid.
        ("concept", "understand"),
        ("key_idea", "understand"),
        ("table", "understand"),
        ("acronym", "remember"),
        ("callout", "understand"),
        ("misconception", "analyze"),
        ("flip_card_grid", "understand"),
    ),
    "application": (
        # finding 2 — a real application page is activity + scenario +
        # problem + a peer discussion_prompt, not 2 thin blocks.
        ("activity", "apply"),
        ("scenario", "apply"),
        ("problem", "apply"),
        ("discussion_prompt", "evaluate"),
    ),
    "self_check": (
        # finding 7 — at least 4 self_check_question blocks; reflection_prompt
        # (finding 6) and a flip_card_grid reveal round out the page.
        ("self_check_question", "apply"),
        ("self_check_question", "analyze"),
        ("self_check_question", "understand"),
        ("reflection_prompt", "evaluate"),
        ("flip_card_grid", "understand"),
    ),
    "summary": (
        # finding 8 — 4+ blocks: distilled takeaways + a checklist + a recap
        # + a reflection_prompt close.
        ("summary_takeaway", "understand"),
        ("checklist", "evaluate"),
        ("recap", "remember"),
        ("reflection_prompt", "evaluate"),
    ),
}

# Deterministic fixed-plan fallback — a VERBATIM copy of
# ``MCP/tools/pipeline_tools.py::_PAGE_TYPE_BLOCK_PLAN``. Kept here (not
# imported) so the planner module has no dependency on the pipeline module
# (avoids an import cycle: pipeline_tools imports this planner). The
# off-path / fallback path is byte-identical to the fixed plan because the
# CONSUMER falls back to its own ``_PAGE_TYPE_BLOCK_PLAN`` — this copy is
# only consulted for the planner-internal coverage/top-up math.
# The fixed fallback now satisfies the same per-page-type floors
# (``_PAGE_TYPE_FLOORS``) so a planner FAILURE still yields a balanced week
# (overview enumeration + ≥4 application / self_check / summary blocks) and
# deploys the previously-unused block types (reflection_prompt, callout,
# misconception, discussion_prompt, prereq_set, flip_card_grid — findings
# 6/16). The consumer ultimately re-derives its own page plan, but the
# planner's coverage / top-up math and the standalone fallback ``WeekBlockPlan``
# both read this table, so it must be floor-compliant.
_DEFAULT_PAGE_PLAN: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "overview": (
        ("objective", "understand"),
        ("prereq_set", "remember"),
        ("explanation", "understand"),
    ),
    "content": (
        ("concept", "understand"),
        ("explanation", "understand"),
        ("callout", "understand"),
        ("misconception", "analyze"),
        ("example", "apply"),
        # finding 7 (PLAN side) — Sonnet uses TABLES for comparative content;
        # the 7B planned 0. There is no dedicated table BLOCK_TYPE (the <table>
        # HTML rides on a content section), so the comparative/tabular block
        # the planner CAN select is flip_card_grid — a grid of paired cards.
        # Seat it in the fixed-plan content page so even a planner FAILURE
        # deploys the grid block (mirrors how the fallback deploys callout /
        # misconception), and the prompt's SELECTION GUIDANCE steers
        # comparative content here on the live path.
        ("flip_card_grid", "understand"),
    ),
    "application": (
        ("activity", "apply"),
        ("scenario", "apply"),
        ("problem", "apply"),
        ("discussion_prompt", "evaluate"),
    ),
    "self_check": (
        ("self_check_question", "analyze"),
        ("self_check_question", "apply"),
        ("self_check_question", "understand"),
        ("self_check_question", "remember"),
        ("reflection_prompt", "evaluate"),
    ),
    "summary": (
        ("summary_takeaway", "understand"),
        ("checklist", "evaluate"),
        ("recap", "remember"),
        ("reflection_prompt", "evaluate"),
    ),
}

# When the planner needs to assign a block whose ``page_type`` is missing /
# invalid, map the block TYPE to the page type it most naturally lives on.
# Used to repair an out-of-set page_type rather than dropping the block.
_BLOCK_TYPE_DEFAULT_PAGE: Dict[str, str] = {
    "objective": "overview",
    "explanation": "overview",
    "concept": "content",
    "example": "content",
    "formula": "content",
    "vocab_card": "content",
    "callout": "content",
    "discussion_prompt": "application",
    "activity": "application",
    "scenario": "application",
    "problem": "application",
    "misconception": "application",
    "prereq_set": "overview",
    "self_check_question": "self_check",
    "reflection_prompt": "self_check",
    "assessment_item": "self_check",
    "flip_card_grid": "self_check",
    "summary_takeaway": "summary",
    "recap": "summary",
    "checklist": "summary",
    "chrome": "overview",
    # Issue I6 instruction-palette-v2 default page placements. A comparison
    # table and an emphasized key idea live on the content page (they teach
    # the concept); an acronym / mnemonic also lives on content (it is a
    # learning aid presented alongside the material).
    "table": "content",
    "acronym": "content",
    "key_idea": "content",
}

# Bounded prompt sizing knobs (keep the per-TO prompt small enough for the
# 70B context without summarising the model away from the real content).
_MAX_SOURCE_CHUNKS_IN_PROMPT = 8
_MAX_CHARS_PER_CHUNK = 600
_MAX_COS_IN_PROMPT = 30


# ---------------------------------------------------------------------------
# Issue I6 instruction-palette-v2 content detectors.
# ---------------------------------------------------------------------------
#
# Lightweight, deterministic, source-grounded detectors that flag WHEN a TO's
# source content is shaped for one of the three palette-v2 block types
# (``acronym`` / ``table``). They feed the planner two signals:
#   1. the per-TO selection-guidance prompt (so the 70B is nudged toward the
#      right block type when the shape is present), and
#   2. a deterministic floor-filler seat (so the block type is REACHABLE on a
#      real run even if the planner under-uses it).
# The detectors are precision-first: they fire ONLY when the source genuinely
# carries the shape, so a false-positive block is never seeded.

# An all-caps acronym candidate: 3-8 contiguous capital letters (PEMDAS,
# FOIL, SOHCAHTOA), optionally with internal digits. Bounded so a stray
# two-letter abbreviation or a 9+ run of capitals (likely a shout, not an
# acronym) never matches.
_ACRONYM_CANDIDATE_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,7})\b")

# A token that LOOKS like tabular / comparison framing in the source.
_TABULAR_HINT_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|contrast|"
    r"proper|improper|mixed|"
    r"the following table|as shown in the table|each (type|kind|category)|"
    r"types of|kinds of|categories of)\b",
    re.IGNORECASE,
)


# ``ED4ALL_DYNAMIC_BLOCK_PLAN`` truthy set (mirrors the consumer's gate in
# ``MCP/tools/pipeline_tools.py``). The deterministic palette-v2 injection
# (Part A) is a strict NO-OP unless this is truthy — so the legacy / off path
# (where the consumer never even calls the planner) and every existing
# snapshot stay byte-stable.
_DYNAMIC_BLOCK_PLAN_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _dynamic_block_plan_on() -> bool:
    """True iff ``ED4ALL_DYNAMIC_BLOCK_PLAN`` is truthy (read each call)."""
    import os  # noqa: PLC0415

    return (
        os.environ.get("ED4ALL_DYNAMIC_BLOCK_PLAN", "").strip().lower()
        in _DYNAMIC_BLOCK_PLAN_TRUTHY
    )


def _new_block_types_on() -> bool:
    """True iff ``ED4ALL_NEW_BLOCK_TYPES`` is truthy (IB5; read each call).

    Gates the IB5 prompt nudges + deterministic injection of the four
    framework-aligned types so the planner prompt is byte-identical when off
    (mirrors :func:`_dynamic_block_plan_on`). Delegates to the canonical
    resolver so there is one truth for the flag."""
    from lib.generation.new_block_types import resolve_new_block_types  # noqa: PLC0415

    return resolve_new_block_types()


# P4 pedagogical-depth floors — both DEFAULT OFF (parse-with-fallback) so the
# planner's output is BYTE-STABLE unless an operator opts in.
#
# ``ED4ALL_WORKED_EXAMPLE_FLOOR`` — when truthy, guarantee every PROCEDURAL CO
#   (a CO whose Bloom level maps to the procedural cognitive domain, i.e.
#   apply / create) is taught by >= 1 worked ``example`` or ``problem`` block.
#   Closes the low-worked-example-DENSITY gap (66/495 blocks = 13% on the
#   reviewed run) without raising the budget.
# ``ED4ALL_BLOOM_SPREAD_FLOOR`` — when truthy, guarantee at least one
#   analyze-or-higher block per week (deploying scenario / misconception /
#   discussion_prompt / table whose catalog ``bloom_fit`` reaches analyze+),
#   widening the low Bloom spread (apply 64% / analyze 4% / evaluate 4% /
#   create 0% on the reviewed run).
_WORKED_EXAMPLE_FLOOR_ENV = "ED4ALL_WORKED_EXAMPLE_FLOOR"
_BLOOM_SPREAD_FLOOR_ENV = "ED4ALL_BLOOM_SPREAD_FLOOR"

# IB7 planner-pedagogy flags — ALL default OFF (parse-with-fallback) so the
# planner's output is BYTE-STABLE unless an operator opts in. Each gates a pure
# post-pass that returns identity when off (mirrors the P4 floors above):
#   ED4ALL_PLANNER_BLOOM_CLIMB   — IB7.3 programmatic Bloom-climb re-sort onto
#       the canonical activation→exposition→worked-example→case→check→summary
#       template (pretraining/vocab before exposition; within-tier Bloom
#       monotonic).
#   ED4ALL_PLANNER_LIFECYCLE     — IB7.4 lifecycle open/close guarantee (an
#       Activate-stage opener + a Consolidate-stage closer per TO) + slot-edit
#       escalation (stamp anatomy_slot_weights before a type swap).
#   ED4ALL_PLANNER_SPACING       — IB7.5a within-module temporal spacing
#       (separate a check/reflection from the exposition that taught its CO).
#   ED4ALL_PLANNER_BLOOM_CEILING — IB7.6b per-type Bloom-range ceiling re-route
#       (an over-escalated block routes to a higher-order type after a slot
#       edit; catalog ``bloom_ceiling`` becomes a planner gate).
_BLOOM_CLIMB_ENV = "ED4ALL_PLANNER_BLOOM_CLIMB"
_LIFECYCLE_ENV = "ED4ALL_PLANNER_LIFECYCLE"
_SPACING_ENV = "ED4ALL_PLANNER_SPACING"
_BLOOM_CEILING_ENV = "ED4ALL_PLANNER_BLOOM_CEILING"
# FR-INT-01 — B08 guided-practice FADING-sequence planner pass. Default OFF
# (parse-with-fallback) so the planner output is BYTE-STABLE unless an operator
# opts in; the pass is a strict identity no-op when off (mirrors the IB7 passes
# above). When on, an explicit faded-practice (B08) block is injected after a
# worked_example (B05) and stamped with a ``fade_state`` (the field already
# exists from IB5 — reused, not re-added).
_FADING_ENV = "ED4ALL_PLANNER_FADING"
_IB7_FLAG_ENVS: Tuple[str, ...] = (
    _BLOOM_CLIMB_ENV, _LIFECYCLE_ENV, _SPACING_ENV, _BLOOM_CEILING_ENV,
    _FADING_ENV,
)


def _any_ib7_flag_on() -> bool:
    """True iff ANY IB7 planner-pedagogy flag is truthy (read each call)."""
    return any(_env_floor_on(env) for env in _IB7_FLAG_ENVS)

# IB7.3 canonical lifecycle-phase TEMPLATE order (the 100%-frequency
# objectives→activation→exposition→worked-example→case→check→summary template,
# framework pp.13-17 / p.138 Step 2). The Bloom-climb re-sort buckets every
# block into one of these tiers and emits the tiers in this order. NOTE: this
# is the cross-block TEACHING-SEQUENCE order — DISTINCT from both the five PAGE
# TYPES (page-file grouping) and the five block-INTERNAL lifecycle STAGES
# (Courseforge.scripts.blocks.LIFECYCLE_STAGES). Index 0 sorts first.
_CLIMB_TIERS: Tuple[str, ...] = (
    "activation",
    "exposition",
    "worked_example",
    "case",
    "check",
    "summary",
)

# Block TYPE -> canonical climb tier. A type absent here falls back to a tier
# derived from its default page (``_BLOCK_TYPE_DEFAULT_PAGE``) via
# ``_PAGE_TO_CLIMB_TIER``. Vocabulary / pretraining types sort FIRST within the
# exposition tier (see ``_VOCAB_PRETRAINING_TYPES``).
_BLOCK_TYPE_CLIMB_TIER: Dict[str, str] = {
    # activation — open the objective (hook / activation / objective / prereq).
    "hook": "activation",
    "objective": "activation",
    "prereq_set": "activation",
    "recap": "activation",
    # exposition — teach the concept (vocab/pretraining sort first within tier).
    "vocab_card": "exposition",
    "acronym": "exposition",
    "flip_card_grid": "exposition",
    "concept": "exposition",
    "explanation": "exposition",
    "key_idea": "exposition",
    "callout": "exposition",
    "misconception": "exposition",
    "table": "exposition",
    "formula": "exposition",
    "multimedia": "exposition",
    "diagram": "exposition",
    # worked_example — show the procedure applied.
    "example": "worked_example",
    "worked_example": "worked_example",
    "problem": "worked_example",
    # case — apply to a realistic situation / collaborate.
    "scenario": "case",
    "activity": "case",
    "discussion_prompt": "case",
    # check — formative checkpoints.
    "self_check_question": "check",
    "reflection_prompt": "check",
    "assessment_item": "check",
    # summary — consolidate.
    "summary_takeaway": "summary",
    "checklist": "summary",
    "chrome": "summary",
}

# Fallback: page TYPE -> climb tier (used when a block type is not in
# ``_BLOCK_TYPE_CLIMB_TIER``, e.g. a catalog-drift new type).
_PAGE_TO_CLIMB_TIER: Dict[str, str] = {
    "overview": "activation",
    "content": "exposition",
    "application": "case",
    "self_check": "check",
    "summary": "summary",
}

# Vocabulary / pretraining types that must sort AHEAD of technical exposition
# within the exposition tier (framework: pretraining/vocabulary precedes
# technical exposition).
_VOCAB_PRETRAINING_TYPES: frozenset = frozenset(
    {"vocab_card", "acronym", "flip_card_grid"}
)

# IB7.4 — Activate-stage opener candidate types (the first present one opens a
# TO that lacks an opener). Consolidate-stage closer candidates likewise.
_ACTIVATION_OPENER_TYPES: Tuple[str, ...] = (
    "hook", "objective", "prereq_set",
)
_CONSOLIDATE_CLOSER_TYPES: Tuple[str, ...] = (
    "summary_takeaway", "recap", "checklist", "reflection_prompt",
)

# IB7.6 — ordered higher-order re-route targets for an over-ceiling block. The
# first whose catalog bloom_ceiling admits the demanded level (and whose type is
# in BLOCK_TYPES) wins. No "everything block": exposition Analyze+ routes OUT.
_BLOOM_REROUTE_TARGETS: Tuple[str, ...] = (
    "scenario", "problem", "assessment_item",
)

# Block types that count as a worked example for the per-procedural-CO floor.
_WORKED_EXAMPLE_BLOCK_TYPES: Tuple[str, ...] = ("example", "problem")
# Analyze-or-higher levels for the Bloom-spread floor.
_ANALYZE_PLUS_LEVELS: frozenset = frozenset({"analyze", "evaluate", "create"})
# Ordered candidates for the per-week analyze-or-higher top-up: deploy the
# block types whose catalog bloom_fit reaches analyze+ (mirrors the
# _PAGE_TYPE_FLOOR_FILLERS analyze entries). First valid one is used.
_BLOOM_SPREAD_FILLERS: Tuple[Tuple[str, str, str], ...] = (
    ("misconception", "analyze", "content"),
    ("scenario", "analyze", "application"),
    ("discussion_prompt", "evaluate", "application"),
    ("table", "analyze", "content"),
)


def _env_floor_on(env_var: str) -> bool:
    """True iff ``env_var`` is truthy (parse-with-fallback; default off)."""
    import os  # noqa: PLC0415

    return (
        os.environ.get(env_var, "").strip().lower()
        in _DYNAMIC_BLOCK_PLAN_TRUTHY
    )


def _is_procedural_co(co_bloom_level: str) -> bool:
    """True iff a CO's Bloom level maps to the procedural cognitive domain.

    Reuses the single source of truth (``lib.ontology.bloom``) — apply /
    create map to the procedural domain. A procedural CO is the one that
    teaches a step-by-step skill and therefore wants a worked example.
    """
    from lib.ontology.bloom import bloom_to_cognitive_domain  # noqa: PLC0415

    level = str(co_bloom_level or "").strip().lower()
    if level not in BLOOM_LEVELS:
        return False
    return bloom_to_cognitive_domain(level) == "procedural"


def _source_text_blob(source_chunks: Sequence[Dict[str, Any]]) -> str:
    """Concatenate the source-chunk text + headings into one search blob."""
    parts: List[str] = []
    for ch in source_chunks or []:
        if not isinstance(ch, dict):
            continue
        text = str(ch.get("text") or "")
        heading = str(ch.get("heading") or "")
        if heading:
            parts.append(heading)
        if text:
            parts.append(text)
    return "\n".join(parts)


def detect_acronyms(source_text: str) -> List[str]:
    """Return all-caps acronym candidates whose expansion is in the source.

    PRECISION over recall (false positives are bad — an unexpanded all-caps
    word like an ID, a heading, or a unit must NOT fire). A candidate
    ``XYZ`` qualifies only when, for every letter, a word STARTING with that
    letter (case-insensitive) appears in the source text — i.e. the source
    actually spells the acronym out. PEMDAS fires iff the source contains
    words starting P, E, M, D, A, S (e.g. "Parentheses Exponents
    Multiplication Division Addition Subtraction" or "Please Excuse My Dear
    Aunt Sally"). A bare ``PEMDAS`` with no expansion does NOT fire.

    Returns the de-duplicated list of qualifying acronym tokens (may be
    empty). Deterministic; no model call.
    """
    if not source_text:
        return []
    # Word-initial letters present in the source (lowercased).
    initials: Set[str] = {
        w[0].lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", source_text)
    }
    out: List[str] = []
    seen: Set[str] = set()
    for cand in _ACRONYM_CANDIDATE_RE.findall(source_text):
        if cand in seen:
            continue
        letters = [c.lower() for c in cand if c.isalpha()]
        if not letters:
            continue
        # Every letter of the acronym must be spelled out by some source word.
        # Require the FULL expansion present (precision): a single missing
        # letter drops the candidate. The acronym token itself supplies one
        # initial per letter, so exclude self-match by requiring at least one
        # OTHER word per letter — approximated by demanding the count of
        # distinct source words starting with each letter ≥ 1 beyond the
        # acronym; in practice the acronym contributes only one capital token,
        # so a genuine expansion adds the spelled-out words.
        if all(ltr in initials for ltr in letters):
            # Guard against the trivial self-match: the acronym alone provides
            # only its own initial letters via its single token, so require
            # that the source has MORE word-initial coverage than just the
            # acronym — i.e. at least len(letters) expansion words exist.
            expansion_words = [
                w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", source_text)
                if w != cand and w and w[0].lower() in letters
            ]
            if len(expansion_words) >= len(letters):
                seen.add(cand)
                out.append(cand)
    return out


# A small CURATED table of well-known mnemonics whose canonical acronym token
# is often ABSENT from a source that nonetheless teaches the underlying concept
# by spelling out the expansion. Order-of-operations material (OpenStax) writes
# "Parentheses Exponents Multiplication Division Addition Subtraction" and/or
# "Please Excuse My Dear Aunt Sally" but frequently NEVER writes the literal
# token "PEMDAS" — so the precision-first literal-token detector above stays
# silent. This table re-introduces the canonical acronym as a GROUNDED learning
# aid: it fires ONLY when the full expansion (every per-letter term, or the
# verbatim mnemonic phrase) is present in the source, so no fabrication occurs
# (every expansion term is in the source). Each entry maps the canonical token
# to (a) the per-letter expansion TERMS and (b) optional verbatim mnemonic
# PHRASES (any one phrase present also fires it).
#
# Precision-first: the canonical mnemonic fires iff EVERY expansion term (or one
# full mnemonic phrase) is present in the source — a partial match never fires.
_CANONICAL_MNEMONICS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "PEMDAS": {
        "terms": (
            "parentheses", "exponents", "multiplication", "division",
            "addition", "subtraction",
        ),
        "phrases": ("please excuse my dear aunt sally",),
    },
    "BODMAS": {
        "terms": (
            "brackets", "orders", "division", "multiplication",
            "addition", "subtraction",
        ),
        "phrases": (),
    },
}


def detect_canonical_mnemonics(source_text: str) -> List[str]:
    """Return canonical mnemonic tokens whose full expansion is in the source.

    The curated-set companion to :func:`detect_acronyms`. Where the general
    detector requires the LITERAL all-caps token to be present (precision over
    recall for arbitrary acronyms), this fires a SMALL curated set of canonical
    mnemonics (``PEMDAS`` / ``BODMAS``) even when the literal token is ABSENT —
    as long as the source spells out the FULL expansion (every per-letter term)
    OR a verbatim mnemonic phrase (e.g. "Please Excuse My Dear Aunt Sally").

    This is NOT fabrication: every expansion term is present in the source, so
    introducing the canonical acronym is a grounded learning aid, not invented
    content. Precision-first — a PARTIAL expansion never fires (every term, or
    one full phrase, must be present).

    Returns the de-duplicated list of qualifying canonical tokens (may be
    empty). Deterministic; no model call.
    """
    if not source_text:
        return []
    lowered = source_text.lower()
    out: List[str] = []
    for token, spec in _CANONICAL_MNEMONICS.items():
        terms = spec.get("terms") or ()
        phrases = spec.get("phrases") or ()
        # Fire when EITHER every per-letter expansion term is present (whole-
        # word) OR any one verbatim mnemonic phrase is present.
        terms_present = bool(terms) and all(
            re.search(r"\b" + re.escape(t) + r"\b", lowered) for t in terms
        )
        phrase_present = any(p in lowered for p in phrases)
        if terms_present or phrase_present:
            out.append(token)
    return out


def detect_tabular_content(source_text: str) -> bool:
    """True when the source content is shaped for a comparison ``table``.

    Fires on an explicit HTML ``<table>`` in the source, OR on comparison /
    multi-category framing (``compare``, ``versus``, ``types of``,
    ``proper / improper / mixed``, ``the following table``). Deterministic;
    precision-leaning (a single hit suffices because the planner only uses
    this as a NUDGE, not a hard placement)."""
    if not source_text:
        return False
    if "<table" in source_text.lower():
        return True
    return bool(_TABULAR_HINT_RE.search(source_text))


# IB5 content-shape detectors (deterministic, precision-leaning — a single hit
# suffices because the planner only uses these as NUDGES, never a hard
# placement). Mirror the detect_tabular_content / detect_acronyms posture; each
# is GATED behind ED4ALL_NEW_BLOCK_TYPES at the call site so the prompt is
# byte-identical when off.
_PROCEDURE_HINT_RE = re.compile(
    r"(?is)\b("
    r"step\s*\d|step[\s-]by[\s-]step|first,?\s+(then|next)|"
    r"follow\s+these\s+steps|the\s+following\s+steps|to\s+solve|"
    r"worked\s+example|procedure\s+(for|to)|algorithm|"
    r"\d+\.\s+\w+.{0,80}?\n\s*\d+\.\s+\w+"  # an enumerated 1. … 2. … list
    r")\b"
)
_MEDIA_HINT_RE = re.compile(
    r"(?is)\b("
    r"watch\s+(the\s+)?video|video|audio|podcast|"
    r"listen\s+to|play\s+the|recording|webcast|lecture\s+capture"
    r")\b|<(?:video|audio|iframe)[\s>]|https?://[^\s\"']*\.(?:mp4|mov|webm|mp3|wav|m4a)"
)
_DIAGRAM_HINT_RE = re.compile(
    r"(?is)\b("
    r"figure\s*\d|diagram|flowchart|flow\s+chart|"
    r"schematic|the\s+graph\s+(below|above|shows)|"
    r"as\s+shown\s+in\s+the\s+figure|see\s+figure|"
    r"tree\s+diagram|venn\s+diagram|concept\s+map|process\s+flow"
    r")\b|<svg[\s>]"
)
_RESOURCES_HINT_RE = re.compile(
    r"(?is)\b("
    r"further\s+reading|for\s+further\s+study|"
    r"additional\s+resources|external\s+resources|"
    r"recommended\s+(reading|resources)|"
    r"references|bibliography|works\s+cited|"
    r"see\s+also|learn\s+more\s+at|read\s+more\s+about"
    r")\b"
)


def detect_procedure(source_text: str) -> bool:
    """True when the source describes a step-by-step PROCEDURE (IB5 B05 nudge).

    Fires on enumerated steps / step-by-step framing / "to solve" / an explicit
    worked-example reference. Deterministic; precision-leaning."""
    if not source_text:
        return False
    return bool(_PROCEDURE_HINT_RE.search(source_text))


def detect_media_reference(source_text: str) -> bool:
    """True when the source references a time-based audio/video artifact (B04).

    Fires on watch/video/audio framing, an embedded <video>/<audio>/<iframe>,
    or a media-file URL. Deterministic; precision-leaning."""
    if not source_text:
        return False
    return bool(_MEDIA_HINT_RE.search(source_text))


def detect_diagram_reference(source_text: str) -> bool:
    """True when the source references a spatial/diagram artifact (B06).

    Fires on figure/diagram/flowchart/schematic framing or an inline <svg>.
    Deterministic; precision-leaning."""
    if not source_text:
        return False
    return bool(_DIAGRAM_HINT_RE.search(source_text))


def detect_resources_reference(source_text: str) -> bool:
    """True when the source points OUTWARD to further resources (B15 nudge).

    Fires on "further reading" / "additional resources" / "references" /
    "bibliography" / "see also" framing. Deterministic; precision-leaning."""
    if not source_text:
        return False
    return bool(_RESOURCES_HINT_RE.search(source_text))


def build_planner_provider(
    *,
    provider: str = "nvidia",
    model: Optional[str] = None,
    capture: Optional[Any] = None,
    client: Optional[Any] = None,
    max_tokens: int = 2048,
):
    """Construct the dedicated planner provider (70B, planning system prompt).

    Reuses the SAME ``_BaseLLMProvider`` dispatch plumbing as the rewrite
    tier + objective review (so the ``nvidia`` seat resolves
    ``NVIDIA_API_KEY`` / base_url identically), but injects the planning
    system prompt instead of the HTML-authoring one. Returns ``None`` on any
    construction error (missing key, import failure) so the caller falls
    back to the fixed plan rather than breaking the build.
    """
    try:
        cls = _resolve_provider_cls()
        resolved_model = model or _resolve_planner_model(provider)
        return cls(
            provider=provider,
            model=resolved_model,
            capture=capture,
            client=client,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — fail-safe to fixed plan
        logger.warning(
            "block_planner: provider construction failed (%s); fixed fallback",
            exc,
        )
        return None


def _planner_provider_base():
    """Lazy-import the shared LLM base (avoids a module-load dependency)."""
    from Courseforge.generators._base import _BaseLLMProvider  # noqa: PLC0415

    return _BaseLLMProvider


# The 70B planner seat. Mirrors the NVIDIA rewrite-tier model
# (``Courseforge/config/block_routing.nvidia_large.yaml`` →
# ``meta/llama-3.3-70b-instruct``). The planner is a content-aware authoring
# decision, so it pins the same large model as the rewrite tier rather than
# the lighter nemotron base default.
_PLANNER_70B_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"


def _resolve_planner_model(provider: Optional[str]) -> Optional[str]:
    """Resolve the planner model (env override > 70B default for nvidia).

    Resolution (high → low):
        ``ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL`` > ``NVIDIA_LARGE_MODEL`` >
        the 70B literal (nvidia seat) > ``None`` (base default for other
        providers).
    """
    import os  # noqa: PLC0415

    explicit = os.environ.get("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL")
    if explicit and explicit.strip():
        return explicit.strip()
    prov = (provider or "").lower()
    if prov == "nvidia":
        return os.environ.get("NVIDIA_LARGE_MODEL") or _PLANNER_70B_NVIDIA_MODEL
    if prov == "local":
        # IB7.2 — license-clean local Qwen seat. Reuse the shared base's
        # ``local`` registry default (LOCAL_SYNTHESIS_MODEL → the Apache-2.0
        # Qwen2.5 registry default) when no env override is set; returning the
        # resolved id keeps the seat reachable WITHOUT an NVIDIA key.
        env_local = os.environ.get("LOCAL_SYNTHESIS_MODEL")
        if env_local and env_local.strip():
            return env_local.strip()
        try:  # registry default (e.g. qwen2.5:7b-instruct-q4_K_M)
            from Courseforge.generators._base import LOCAL_DEFAULT_MODEL
            return LOCAL_DEFAULT_MODEL
        except Exception:  # noqa: BLE001 — let the base resolve its own default
            return None
    return None


def _make_block_planner_provider_cls():
    base = _planner_provider_base()

    class BlockPlannerProvider(base):  # type: ignore[misc, valid-type]
        """Planner-tier provider: dispatches a block-plan prompt to the 70B.

        Thin ``_BaseLLMProvider`` subclass. Overrides only the two abstract
        hooks (no task-specific authoring) and exposes ``plan_blocks`` so
        :func:`plan_week_blocks` can call it without touching the base's
        protected ``_dispatch_call`` directly.
        """

        def __init__(
            self,
            *,
            provider: Optional[str] = "nvidia",
            model: Optional[str] = None,
            capture: Optional[Any] = None,
            client: Optional[Any] = None,
            max_tokens: int = 2048,
        ) -> None:
            super().__init__(
                provider=provider,
                model=model,
                capture=capture,
                client=client,
                max_tokens=max_tokens,
                temperature=0.3,
                env_provider_var="ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER",
                default_provider=provider or "nvidia",
                system_prompt=_PLANNER_SYSTEM_PROMPT,
                json_mode=True,
            )

        def _render_user_prompt(self, *args: Any, **kwargs: Any) -> str:
            # The planner builds its prompt in plan_week_blocks; this hook is
            # required by the ABC but unused (plan_blocks passes the prompt
            # straight through).
            return args[0] if args else ""

        def _emit_per_call_decision(
            self, *, raw_text: str, retry_count: int, **call_context: Any
        ) -> None:
            # The block_plan decision event is emitted by plan_week_blocks
            # (it has the validated selection + coverage); the per-call hook
            # is a no-op here to avoid a duplicate event.
            return None

        def plan_blocks(self, prompt: str) -> str:
            """Dispatch the planner prompt; return the raw model text."""
            text, _retries = self._dispatch_call(prompt)
            return text

    return BlockPlannerProvider


# Public name; resolved lazily on first construction so the import stays light.
BlockPlannerProvider: Any = None  # populated by build_planner_provider


_PROVIDER_CLS_CACHE: Dict[str, Any] = {}


def _resolve_provider_cls():
    cls = _PROVIDER_CLS_CACHE.get("cls")
    if cls is None:
        cls = _make_block_planner_provider_cls()
        _PROVIDER_CLS_CACHE["cls"] = cls
        # Backfill the module-level name so callers can isinstance-check.
        globals()["BlockPlannerProvider"] = cls
    return cls


@dataclass
class WeekBlockPlan:
    """The validated per-TO block plan returned by :func:`plan_week_blocks`."""

    page_plan: Dict[str, List[Tuple[str, str, List[str]]]]
    selected: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    terminal_objective_id: str = ""
    model: str = ""

    def page_block_plan_for(
        self, page_type: str
    ) -> List[Tuple[str, str, List[str]]]:
        """Return the ordered ``(block_type, target_bloom, target_co_ids)``
        list for a page.

        Mirrors ``MCP/tools/pipeline_tools.py::_page_block_plan_for`` so the
        outline-phase loop can consult this object identically — EXCEPT each
        entry now carries the per-block ``target_co_ids`` (the specific CO ids
        this block teaches; an empty list means "no specific CO — fall back to
        the week TO"). Unknown page types fall back to the ``content`` plan
        (defensive)."""
        return self.page_plan.get(
            page_type, self.page_plan.get("content", [])
        )


def _resolve_block_types() -> frozenset:
    """Load the canonical block-type set (lazy, no import cycle risk)."""
    from Courseforge.scripts.blocks import BLOCK_TYPES  # noqa: PLC0415

    return BLOCK_TYPES


def _default_page_plan() -> Dict[str, List[Tuple[str, str, List[str]]]]:
    """Return a fresh mutable copy of the fixed fallback plan.

    Entries are 3-tuples ``(block_type, target_bloom, target_co_ids)``; the
    fixed plan has no per-block CO target, so ``target_co_ids`` is always an
    empty list — the consumer's "use the week-TO fallback" signal (this keeps
    the fixed-plan / fallback path byte-identical in objective_ids terms)."""
    return {
        ptype: [(spec[0], spec[1], []) for spec in specs]
        for ptype, specs in _DEFAULT_PAGE_PLAN.items()
    }


def _clamp_budget(budget: Tuple[int, int]) -> Tuple[int, int]:
    """Sanitise the (min, max) budget; fall back to the default on garbage."""
    try:
        lo, hi = int(budget[0]), int(budget[1])
    except (TypeError, ValueError, IndexError):
        return DEFAULT_BLOCK_BUDGET
    if lo < 1:
        lo = DEFAULT_BLOCK_BUDGET[0]
    if hi < lo:
        hi = max(lo, DEFAULT_BLOCK_BUDGET[1])
    return lo, hi


def _bloom_floor_for_catalog_entry(entry: Dict[str, Any]) -> Optional[str]:
    """Return the lowest Bloom level in a catalog entry's ``bloom_fit``."""
    fit = entry.get("bloom_fit") or []
    valid = [b for b in fit if b in BLOOM_LEVELS]
    if not valid:
        return None
    # Lowest level by canonical ordering.
    return min(valid, key=lambda b: BLOOM_LEVELS.index(b))


def _build_prompt(
    *,
    terminal_objective: Dict[str, Any],
    chapter_objectives: Sequence[Dict[str, Any]],
    source_chunks: Sequence[Dict[str, Any]],
    catalog: Sequence[Dict[str, Any]],
    budget: Tuple[int, int],
) -> str:
    """Render the per-TO planner user prompt for the 70B."""
    lo, hi = budget
    to_id = str(terminal_objective.get("id") or "TO-01")
    to_stmt = str(terminal_objective.get("statement") or "").strip()

    co_lines: List[str] = []
    for co in list(chapter_objectives)[:_MAX_COS_IN_PROMPT]:
        cid = str(co.get("id") or "").strip()
        cstmt = str(co.get("statement") or "").strip()
        cbloom = str(co.get("bloom_level") or "").strip()
        if not cid:
            continue
        suffix = f" [bloom: {cbloom}]" if cbloom else ""
        co_lines.append(f"  - {cid}: {cstmt}{suffix}")
    co_block = "\n".join(co_lines) if co_lines else "  (none)"

    chunk_lines: List[str] = []
    for ch in list(source_chunks)[:_MAX_SOURCE_CHUNKS_IN_PROMPT]:
        text = str(ch.get("text") or "").strip()
        if not text:
            continue
        if len(text) > _MAX_CHARS_PER_CHUNK:
            text = text[:_MAX_CHARS_PER_CHUNK].rstrip() + " …"
        heading = str(ch.get("heading") or "").strip()
        prefix = f"[{heading}] " if heading else ""
        chunk_lines.append(f"  - {prefix}{text}")
    source_block = "\n".join(chunk_lines) if chunk_lines else "  (no source digest available)"

    catalog_lines: List[str] = []
    for entry in catalog:
        bt = str(entry.get("block_type") or "").strip()
        if not bt:
            continue
        use_when = " ".join(str(entry.get("use_when") or "").split())
        bloom_fit = entry.get("bloom_fit") or []
        bloom_str = ", ".join(b for b in bloom_fit if isinstance(b, str))
        catalog_lines.append(
            f"  - {bt}: {use_when} (bloom_fit: {bloom_str})"
        )
    catalog_block = "\n".join(catalog_lines)

    page_types_csv = ", ".join(CANONICAL_PAGE_TYPES)

    # Issue I6: content-shape detector nudges. When the source genuinely
    # carries a comparison/tabular shape or an explained acronym, steer the
    # planner toward the right palette-v2 block type (precision-first — the
    # detectors only fire on real shape, so the nudge never fabricates one).
    source_blob = _source_text_blob(source_chunks)
    detector_lines: List[str] = []
    if detect_tabular_content(source_blob):
        detector_lines.append(
            "  - The source COMPARES items across shared dimensions — use a "
            "`table` (caption + scoped header cells, one row per item) for it."
        )
    # Union the literal-token detector with the curated canonical-mnemonic
    # detector so a source that spells out PEMDAS/BODMAS without the literal
    # token still nudges the 70B toward the `acronym` block.
    detected_acronyms = list(
        dict.fromkeys(
            detect_acronyms(source_blob) + detect_canonical_mnemonics(source_blob)
        )
    )
    if detected_acronyms:
        names = ", ".join(detected_acronyms[:4])
        detector_lines.append(
            f"  - The source spells out the acronym(s) {names} — use an "
            "`acronym` block (a <dl> mapping each letter to its term) for it."
        )
    # IB5 content-shape nudges (GATED behind ED4ALL_NEW_BLOCK_TYPES so the
    # prompt is byte-identical when off). procedure -> worked_example;
    # media reference -> multimedia; spatial/figure -> diagram. The hook nudge
    # (open every TO with an activation block) is unconditional once the flag is
    # on — full lifecycle-aware opener placement is IB7's scope; IB5 only makes
    # `hook` SELECTABLE.
    if _new_block_types_on():
        if detect_procedure(source_blob):
            detector_lines.append(
                "  - The source describes a step-by-step PROCEDURE — use a "
                "`worked_example` (subgoal-labeled steps, a per-step Why, and a "
                "fade-state) so the learner can fade toward independent practice."
            )
        if detect_media_reference(source_blob):
            detector_lines.append(
                "  - The source references a time-based audio/video artifact — "
                "use a `multimedia` block (it carries the mandatory captions / "
                "audio-description / transcript / controls a11y stack)."
            )
        if detect_diagram_reference(source_blob):
            detector_lines.append(
                "  - The source references a spatial/diagram artifact "
                "(figure / flowchart / schematic) — use a `diagram` block "
                "(a structured long-description + a data-table equivalent)."
            )
        if detect_resources_reference(source_blob):
            detector_lines.append(
                "  - The source points OUTWARD to further reading / references "
                "/ external resources — CLOSE this objective with a `resources` "
                "block: an accessible list of links each with DESCRIPTIVE text "
                "(a title, never a bare URL or 'click here')."
            )
        detector_lines.append(
            "  - OPEN this objective with a `hook` block — a short activation / "
            "predict prompt that surfaces the learner's prior knowledge BEFORE "
            "any new content (no new teaching content in the hook itself)."
        )
    detector_block = (
        "\nDETECTED CONTENT SHAPES (the source supports these — prefer the "
        "named block type):\n" + "\n".join(detector_lines) + "\n"
        if detector_lines
        else ""
    )

    return (
        "You are an expert instructional designer planning ONE week of a "
        "course built around a single terminal objective. Choose the "
        "ORDERED sequence of instruction blocks that teaches this objective "
        "WELL — pick the RIGHT block type for each piece of content rather "
        "than defaulting to the same blocks every week.\n\n"
        f"TERMINAL OBJECTIVE ({to_id}):\n  {to_stmt}\n\n"
        f"CHILD OBJECTIVES (each MUST be covered by at least one block):\n"
        f"{co_block}\n\n"
        f"SOURCE CONTENT TO TEACH (digest):\n{source_block}\n\n"
        f"AVAILABLE BLOCK TYPES (use_when guidance):\n{catalog_block}\n\n"
        "SELECTION GUIDANCE:\n"
        "  - a procedure  -> worked example / problem\n"
        "  - a key term   -> vocab_card\n"
        "  - a cluster of terms / a comparison across items -> flip_card_grid\n"
        "  - a formula / equation -> formula\n"
        "  - a common error-> misconception\n"
        "  - a real situation -> scenario\n"
        "  - a checkpoint -> self_check_question / reflection_prompt\n"
        "  - a recap      -> summary_takeaway / checklist\n"
        # Issue I6 instruction-palette-v2 selection guidance.
        "  - a comparison across shared columns -> table (an accessible "
        "<table> with a caption + scoped headers)\n"
        "  - an acronym / mnemonic the source spells out -> acronym\n"
        "  - the single most important principle / takeaway -> key_idea\n\n"
        "When content is COMPARATIVE or TABULAR (several items contrasted "
        "across the same dimensions, a criteria-by-option matrix, or "
        "proper/improper/mixed-style categories), prefer a `table` (a real "
        "accessible <table>) when the relationship is across shared COLUMNS, "
        "or flip_card_grid for term/definition pairs — both beat a flat prose "
        "concept. When the source presents a memory aid whose letters spell "
        "out terms (e.g. PEMDAS), use an `acronym` block. When one principle "
        "is THE point to remember, set it apart in a `key_idea` aside.\n"
        f"{detector_block}\n"
        f"BUDGET: select between {lo} and {hi} blocks total.\n"
        "Assign each block to exactly one page_type from: "
        f"{page_types_csv}.\n\n"
        "Return ONLY a JSON object of the form:\n"
        '{"blocks": [{"block_type": "<one of the available types>", '
        '"target_co_ids": ["CO-01", ...], "page_type": "<one page type>", '
        '"content_focus": "<short phrase naming the content this block '
        'teaches>"}, ...]}\n'
        "Order the array in teaching order (overview first, summary last). "
        "Every child objective id must appear in at least one block's "
        "target_co_ids."
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_blocks(raw_text: str) -> List[Dict[str, Any]]:
    """Lenient parse of the 70B response into a list of block dicts.

    Tolerates code fences / leading prose by extracting the first balanced
    JSON object. Returns ``[]`` on any parse failure (the caller treats an
    empty list as a fallback trigger)."""
    if not raw_text or not raw_text.strip():
        return []
    text = raw_text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    candidates = [text]
    m = _JSON_OBJ_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            doc = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(doc, dict):
            blocks = doc.get("blocks")
            if isinstance(blocks, list):
                return [b for b in blocks if isinstance(b, dict)]
        if isinstance(doc, list):
            return [b for b in doc if isinstance(b, dict)]
    return []


def _validate_and_repair(
    *,
    raw_blocks: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
    block_types: frozenset,
    budget: Tuple[int, int],
    co_bloom: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Apply the guardrails to the raw LLM block list.

    Drops unknown ``block_type``; repairs invalid ``page_type``; clamps to
    the budget; enforces CO coverage; resolves ``target_bloom`` per block.
    Returns the validated, ordered block list (may be empty → caller falls
    back).
    """
    lo, hi = budget
    valid_co_ids = {
        str(co.get("id")) for co in chapter_objectives if co.get("id")
    }

    cleaned: List[Dict[str, Any]] = []
    for blk in raw_blocks:
        bt = str(blk.get("block_type") or "").strip()
        if bt not in block_types:
            # Guardrail 1: drop unknown block_type.
            continue
        page_type = str(blk.get("page_type") or "").strip()
        if page_type not in CANONICAL_PAGE_TYPES:
            # Guardrail 2: repair invalid/missing page_type.
            page_type = _BLOCK_TYPE_DEFAULT_PAGE.get(bt, "content")
        # Filter target_co_ids to real CO ids (drop hallucinated ids).
        raw_targets = blk.get("target_co_ids") or []
        targets = [
            str(t) for t in raw_targets
            if isinstance(t, str) and str(t) in valid_co_ids
        ]
        # Resolve target_bloom.
        target_bloom = _resolve_target_bloom(
            declared=blk.get("target_bloom"),
            block_type=bt,
            catalog_by_type=catalog_by_type,
            targets=targets,
            co_bloom=co_bloom,
        )
        cleaned.append({
            "block_type": bt,
            "page_type": page_type,
            "target_co_ids": targets,
            "content_focus": str(blk.get("content_focus") or "").strip(),
            "target_bloom": target_bloom,
        })

    # Guardrail 3a: clamp the MAX (drop excess past hi).
    if len(cleaned) > hi:
        cleaned = cleaned[:hi]

    # Guardrail 4: CO coverage — every CO covered by >= 1 block.
    covered: set = set()
    for blk in cleaned:
        covered.update(blk["target_co_ids"])
    uncovered = [
        str(co.get("id")) for co in chapter_objectives
        if co.get("id") and str(co.get("id")) not in covered
    ]
    for cid in uncovered:
        bloom = co_bloom.get(cid) or _bloom_floor_for_catalog_entry(
            catalog_by_type.get("concept", {})
        ) or "understand"
        cleaned.append({
            "block_type": "concept",
            "page_type": "content",
            "target_co_ids": [cid],
            "content_focus": "default coverage block (CO dropped by planner)",
            "target_bloom": bloom,
        })

    # Guardrail 3b: top-up the MIN. If still below the floor (a sparse plan
    # with few COs), append default content blocks so a week is never
    # threadbare. Re-clamp to hi afterward in case coverage+top-up overshot.
    while len(cleaned) < lo:
        cleaned.append({
            "block_type": "explanation",
            "page_type": "content",
            "target_co_ids": [],
            "content_focus": "default top-up block (budget floor)",
            "target_bloom": "understand",
        })
    if len(cleaned) > hi and uncovered:
        # Coverage blocks must never be sacrificed; only trim non-coverage
        # filler beyond hi.
        coverage_ids = set(uncovered)
        kept: List[Dict[str, Any]] = []
        filler: List[Dict[str, Any]] = []
        for blk in cleaned:
            if set(blk["target_co_ids"]) & coverage_ids:
                kept.append(blk)
            else:
                filler.append(blk)
        room = max(hi - len(kept), 0)
        cleaned = (filler[: room] + kept) if room < len(filler) else cleaned

    return cleaned


def _next_floor_filler(
    fillers: Sequence[Tuple[str, str]],
    present_types: set,
    block_types: frozenset,
) -> Optional[Tuple[str, str]]:
    """Pick the next floor filler — prefer a NEW (variety) type, then repeat.

    Returns the first valid filler whose ``block_type`` is not yet on the
    page (so floor top-ups deploy the contracted-but-unused types rather than
    re-stacking one type), or — once every filler type is already present —
    the first valid filler to repeat. ``None`` only if no filler type is a
    valid ``BLOCK_TYPES`` token (a catalog-drift guard)."""
    valid = [(bt, bloom) for bt, bloom in fillers if bt in block_types]
    if not valid:
        return None
    for bt, bloom in valid:
        if bt not in present_types:
            return bt, bloom
    return valid[0]


def _apply_worked_example_floor(
    *,
    selected: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Guarantee >= 1 worked ``example``/``problem`` block per PROCEDURAL CO.

    P4 worked-example DENSITY floor (gated on ``ED4ALL_WORKED_EXAMPLE_FLOOR``;
    NO-OP / byte-stable when off). For every procedural CO (Bloom →
    procedural domain) not already taught by an ``example`` or ``problem``
    block that targets it, APPEND one ``example`` block targeting that CO on
    the ``content`` page (the worked-example seat). NEVER invents a CO id —
    the appended block targets the procedural CO it is for. Block type
    presence is guarded so a catalog drift can never inject an invalid type.
    """
    if not _env_floor_on(_WORKED_EXAMPLE_FLOOR_ENV):
        return selected
    we_type = next(
        (bt for bt in _WORKED_EXAMPLE_BLOCK_TYPES if bt in block_types), None
    )
    if we_type is None:
        return selected
    # CO ids already covered by a worked-example block (per-CO).
    covered_by_we: set = set()
    for blk in selected:
        if blk.get("block_type") in _WORKED_EXAMPLE_BLOCK_TYPES:
            covered_by_we.update(
                str(c) for c in (blk.get("target_co_ids") or [])
            )
    appended = list(selected)
    for co in chapter_objectives or ():
        if not isinstance(co, dict):
            continue
        cid = str(co.get("id") or "")
        if not cid or cid in covered_by_we:
            continue
        if not _is_procedural_co(co.get("bloom_level") or ""):
            continue
        appended.append({
            "block_type": we_type,
            "page_type": "content",
            "target_co_ids": [cid],
            "content_focus": f"P4 worked-example floor (procedural CO {cid})",
            "target_bloom": "apply",
        })
        covered_by_we.add(cid)
    return appended


def _apply_bloom_spread_floor(
    *,
    selected: List[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Guarantee >= 1 analyze-or-higher block per week.

    P4 Bloom-spread floor (gated on ``ED4ALL_BLOOM_SPREAD_FLOOR``; NO-OP /
    byte-stable when off). If no selected block already carries an
    analyze-or-higher ``target_bloom``, APPEND one analyze+ block from
    ``_BLOOM_SPREAD_FILLERS`` (the first whose type is in ``block_types``
    AND whose catalog ``bloom_fit`` reaches analyze+ — widening the apply-heavy
    Bloom distribution). NEVER injects an invalid block type.
    """
    if not _env_floor_on(_BLOOM_SPREAD_FLOOR_ENV):
        return selected
    has_analyze_plus = any(
        str(b.get("target_bloom") or "") in _ANALYZE_PLUS_LEVELS
        for b in selected
    )
    if has_analyze_plus:
        return selected
    for bt, bloom, ptype in _BLOOM_SPREAD_FILLERS:
        if bt not in block_types:
            continue
        # Only deploy a type the catalog agrees can reach analyze+ (a content
        # shape the source supports), guarding against forcing analyze onto a
        # block type the catalog caps lower.
        fit = catalog_by_type.get(bt, {}).get("bloom_fit") or []
        if not any(lv in _ANALYZE_PLUS_LEVELS for lv in fit):
            continue
        return list(selected) + [{
            "block_type": bt,
            "page_type": ptype,
            "target_co_ids": [],
            "content_focus": "P4 Bloom-spread floor (analyze-or-higher)",
            "target_bloom": bloom,
        }]
    return selected


def _apply_page_floors(
    *,
    selected: List[Dict[str, Any]],
    block_types: frozenset,
    budget: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Top up any starved page TYPE to its ``_PAGE_TYPE_FLOORS`` minimum.

    The planner naturally pours blocks onto ``content`` and starves
    ``overview`` / ``application`` / ``self_check`` / ``summary`` (findings
    2/7/8/18 of the Sonnet-vs-7B review). This pass counts the blocks the
    planner assigned to each page TYPE and, for any page below its floor,
    APPENDS page-appropriate filler from ``_PAGE_TYPE_FLOOR_FILLERS`` until
    the floor is met — deliberately deploying the contracted-but-unused
    block types (reflection_prompt, callout, misconception,
    discussion_prompt, prereq_set, flip_card_grid — findings 6/16) that the
    7B never authored on its own.

    Floor top-up takes PRECEDENCE over the budget max: a balanced week is
    worth more than a tight ceiling (the default max is raised to fund the
    floors). Filler block types not in ``BLOCK_TYPES`` are skipped (a
    catalog drift can never inject an invalid type). The relative order of
    the planner-selected blocks is preserved; floor fillers are appended in
    page-emit order so each page's fillers stay contiguous.
    """
    by_page: Dict[str, List[Dict[str, Any]]] = {
        ptype: [] for ptype in CANONICAL_PAGE_TYPES
    }
    extra_page = "content"
    for blk in selected:
        ptype = blk.get("page_type")
        if ptype not in by_page:
            ptype = extra_page
        by_page[ptype].append(blk)

    for ptype in CANONICAL_PAGE_TYPES:
        floor = _PAGE_TYPE_FLOORS.get(ptype, 0)
        # Typed floor (finding 7): guarantee >= N blocks of a SPECIFIC type
        # (self_check → >= 4 self_check_question) BEFORE the variety top-up.
        typed = _PAGE_TYPE_MIN_TYPED.get(ptype)
        if typed:
            typed_bt, typed_min = typed
            if typed_bt in block_types:
                have = sum(
                    1 for b in by_page[ptype]
                    if b.get("block_type") == typed_bt
                )
                typed_bloom = "apply"
                for bt, bl in _PAGE_TYPE_FLOOR_FILLERS.get(ptype, ()):
                    if bt == typed_bt:
                        typed_bloom = bl
                        break
                for _ in range(max(typed_min - have, 0)):
                    by_page[ptype].append({
                        "block_type": typed_bt,
                        "page_type": ptype,
                        "target_co_ids": [],
                        "content_focus": f"typed floor ({ptype}/{typed_bt})",
                        "target_bloom": typed_bloom,
                    })
        if floor <= 0:
            continue
        fillers = _PAGE_TYPE_FLOOR_FILLERS.get(ptype, ())
        if not fillers:
            continue
        # First, for the page enumeration anchor (finding 18: the overview's
        # objective enumeration) ensure the leading filler type is present
        # even if the page already meets count — the anchor is load-bearing.
        present_types = {b["block_type"] for b in by_page[ptype]}
        anchor_bt, anchor_bloom = fillers[0]
        if anchor_bt in block_types and anchor_bt not in present_types:
            by_page[ptype].insert(0, {
                "block_type": anchor_bt,
                "page_type": ptype,
                "target_co_ids": [],
                "content_focus": f"page floor anchor ({ptype})",
                "target_bloom": anchor_bloom,
            })
            present_types.add(anchor_bt)
        # Top up to the floor. Prefer filler types NOT already on the page so
        # a starved page gains VARIETY (findings 6/16 — deploy the
        # contracted-but-unused block types) rather than re-stacking the same
        # type. Only once every filler type is present do we cycle to repeat.
        while len(by_page[ptype]) < floor:
            spec = _next_floor_filler(fillers, present_types, block_types)
            if spec is None:
                break
            bt, bloom = spec
            present_types.add(bt)
            by_page[ptype].append({
                "block_type": bt,
                "page_type": ptype,
                "target_co_ids": [],
                "content_focus": f"page floor top-up ({ptype})",
                "target_bloom": bloom,
            })

    # Re-flatten in canonical page-emit order so each page's blocks (incl.
    # fillers) stay contiguous and the teaching order (overview → summary)
    # holds. Pages with no floor keep their planner-assigned blocks verbatim.
    rebuilt: List[Dict[str, Any]] = []
    for ptype in CANONICAL_PAGE_TYPES:
        rebuilt.extend(by_page[ptype])
    return rebuilt


# ---------------------------------------------------------------------------
# Issue I6 — deterministic palette-v2 injection (Part A).
# ---------------------------------------------------------------------------
#
# The I6 block types (``table`` / ``acronym`` / ``key_idea``) are seated as
# CONTENT-page floor fillers (``_PAGE_TYPE_FLOOR_FILLERS["content"]``), but the
# content page has NO floor (``_PAGE_TYPE_FLOORS["content"] == 0``), so the
# top-up never deploys them — deployment relied entirely on the 70B CHOOSING
# them (it chose them 0× across 7 weeks in the live run). This pass makes the
# three types deterministically deploy whenever their content SHAPE is present
# in the TO's source, independent of 70B judgment:
#
#   * tabular source shape present + no ``table`` block        → inject one
#   * acronym/mnemonic spelled out + no ``acronym`` block      → inject one
#   * always exactly one ``key_idea`` on content (the emphasized-principle
#     framing is near-universal): PROMOTE an existing generic ``callout`` /
#     ``concept`` to ``key_idea`` if one is on the content page, else inject one
#
# Bounded: at most ONE of each type per TO (the content page is never flooded).
# Each injected entry is a DESCRIPTOR only (block_type + target_co_ids +
# page_type=content + a target bloom); the rewrite tier authors the HTML per the
# I6 per-type output contracts. ANTI-FABRICATION: an injected block targets the
# TO's first real CO id (or none → the consumer's week-TO fallback); no CO id is
# invented.
_PALETTE_V2_KEY_IDEA_PROMOTABLE: Tuple[str, ...] = ("callout", "concept")


def _inject_palette_v2(
    *,
    selected: List[Dict[str, Any]],
    source_blob: str,
    chapter_objectives: Sequence[Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Deterministically deploy the I6 palette-v2 types by CONTENT SHAPE.

    Mutates / extends ``selected`` in place (returns it) so the three I6 block
    types appear on the content page when their shape is present — without
    relying on the 70B selecting them. NO-OP for any I6 type already present.
    Caller gates this on ``ED4ALL_DYNAMIC_BLOCK_PLAN`` being on, so the legacy
    / off path is byte-stable.
    """
    present_content_types = {
        b.get("block_type")
        for b in selected
        if b.get("page_type") == "content"
    }
    # The CO id an injected block teaches: the TO's first real CO (or none →
    # week-TO fallback in the consumer). Never invents an id.
    first_co_id = next(
        (str(co.get("id")) for co in chapter_objectives if co.get("id")), ""
    )
    target_co_ids = [first_co_id] if first_co_id else []

    def _mk(block_type: str, bloom: str, focus: str) -> Dict[str, Any]:
        return {
            "block_type": block_type,
            "page_type": "content",
            "target_co_ids": list(target_co_ids),
            "content_focus": focus,
            "target_bloom": bloom,
        }

    # (1) table — inject one when the source is tabular/comparative and no
    # table block is on the content page yet.
    if (
        "table" in block_types
        and "table" not in present_content_types
        and detect_tabular_content(source_blob)
    ):
        selected.append(_mk(
            "table", "understand",
            "I6 deterministic injection: source compares items across shared "
            "dimensions",
        ))
        present_content_types.add("table")

    # (2) acronym — inject one when an acronym/mnemonic is detected (literal
    # token OR curated canonical mnemonic) and none is present.
    detected_acronyms = (
        detect_acronyms(source_blob) + detect_canonical_mnemonics(source_blob)
    )
    if (
        "acronym" in block_types
        and "acronym" not in present_content_types
        and detected_acronyms
    ):
        selected.append(_mk(
            "acronym", "remember",
            "I6 deterministic injection: source spells out the acronym/"
            f"mnemonic {detected_acronyms[0]}",
        ))
        present_content_types.add("acronym")

    # (3) key_idea — guarantee exactly one on the content page. Prefer
    # PROMOTING an existing generic callout/concept to key_idea; else inject.
    if "key_idea" in block_types and "key_idea" not in present_content_types:
        promoted = False
        for blk in selected:
            if (
                blk.get("page_type") == "content"
                and blk.get("block_type") in _PALETTE_V2_KEY_IDEA_PROMOTABLE
            ):
                blk["block_type"] = "key_idea"
                blk["content_focus"] = (
                    "I6 deterministic injection: promoted "
                    f"{blk.get('content_focus') or 'generic content block'} to "
                    "an emphasized key idea"
                )
                promoted = True
                break
        if not promoted:
            selected.append(_mk(
                "key_idea", "understand",
                "I6 deterministic injection: the single most important "
                "principle of this objective set apart as a key idea",
            ))

    return selected


def _inject_ib5_types(
    *,
    selected: List[Dict[str, Any]],
    source_blob: str,
    chapter_objectives: Sequence[Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Deterministically deploy the IB5 framework types by CONTENT SHAPE.

    Mirrors :func:`_inject_palette_v2`. Mutates / extends ``selected`` in place
    (returns it) so the four IB5 types appear when their shape is present —
    without relying on the 70B selecting them. NO-OP for any IB5 type already
    present. Caller gates this on ``ED4ALL_NEW_BLOCK_TYPES`` being on, so the
    legacy / off path is byte-stable.

    Bounded: at most ONE of each type per TO. Each injected entry is a
    DESCRIPTOR only (block_type + target_co_ids + page_type + target bloom); the
    rewrite tier authors the HTML per the IB5 per-type output contracts.
    ANTI-FABRICATION: an injected block targets the TO's first real CO id (or
    none → the consumer's week-TO fallback); no CO id is invented.
    """
    present_types = {b.get("block_type") for b in selected}
    first_co_id = next(
        (str(co.get("id")) for co in chapter_objectives if co.get("id")), ""
    )
    target_co_ids = [first_co_id] if first_co_id else []

    def _mk(block_type: str, bloom: str, page_type: str, focus: str) -> Dict[str, Any]:
        return {
            "block_type": block_type,
            "page_type": page_type,
            "target_co_ids": list(target_co_ids),
            "content_focus": focus,
            "target_bloom": bloom,
        }

    # hook — always open the TO with an activation block (an Activate-stage
    # opener; full lifecycle placement is IB7). Inserted FIRST.
    if "hook" in block_types and "hook" not in present_types:
        selected.insert(0, _mk(
            "hook", "understand", "overview",
            "IB5 deterministic injection: open the objective with an "
            "activation/predict prompt surfacing prior knowledge",
        ))
        present_types.add("hook")

    # worked_example — inject one when the source describes a procedure.
    if (
        "worked_example" in block_types
        and "worked_example" not in present_types
        and detect_procedure(source_blob)
    ):
        selected.append(_mk(
            "worked_example", "apply", "application",
            "IB5 deterministic injection: source describes a step-by-step "
            "procedure (subgoal-labeled, faded worked example)",
        ))
        present_types.add("worked_example")

    # multimedia — inject one when the source references a time-based artifact.
    if (
        "multimedia" in block_types
        and "multimedia" not in present_types
        and detect_media_reference(source_blob)
    ):
        selected.append(_mk(
            "multimedia", "understand", "content",
            "IB5 deterministic injection: source references a time-based "
            "audio/video artifact",
        ))
        present_types.add("multimedia")

    # diagram — inject one when the source references a spatial artifact.
    if (
        "diagram" in block_types
        and "diagram" not in present_types
        and detect_diagram_reference(source_blob)
    ):
        selected.append(_mk(
            "diagram", "understand", "content",
            "IB5 deterministic injection: source references a spatial/diagram "
            "artifact (figure / flowchart / schematic)",
        ))
        present_types.add("diagram")

    return selected


# ---------------------------------------------------------------------------
# IB7 — planner-pedagogy passes (Bloom-climb / lifecycle / spacing / ceiling).
# Each is a PURE function gated by its own env flag; default-off ⇒ identity, so
# the planner's output is byte-stable unless an operator opts in. They run AFTER
# the page/P4 floors + palette-v2 + IB5 injection (they see the FINAL block set)
# and BEFORE ``_to_page_plan``, in the order climb → lifecycle → spacing →
# ceiling (the plan's internal dependency order).
# ---------------------------------------------------------------------------


def _block_climb_tier(blk: Dict[str, Any]) -> str:
    """Map a selected block to its canonical climb tier (IB7.3)."""
    bt = str(blk.get("block_type") or "")
    tier = _BLOCK_TYPE_CLIMB_TIER.get(bt)
    if tier:
        return tier
    page_type = str(blk.get("page_type") or "")
    return _PAGE_TO_CLIMB_TIER.get(page_type, "exposition")


def _apply_bloom_climb(
    *, selected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Re-sort ``selected`` onto the canonical Bloom-climb template (IB7.3).

    Gated on ``ED4ALL_PLANNER_BLOOM_CLIMB``; NO-OP / byte-stable when off.
    Two-level STABLE sort (preserves planner relative order within an equal
    key):
      1. by canonical lifecycle-phase tier (activation → exposition →
         worked_example → case → check → summary — the 100%-frequency
         template), and WITHIN the exposition tier vocabulary / pretraining
         types sort ahead of technical exposition;
      2. within a tier, ascending ``target_bloom`` so lower-order scaffolds
         precede higher-order blocks (the Bloom climb UP the pyramid).
    The climb is the FINAL ordering authority — it runs after the page-emit
    re-flatten in ``_apply_page_floors`` so there is no double-reorder
    conflict. Returns a NEW list; never mutates the input blocks.
    """
    if not _env_floor_on(_BLOOM_CLIMB_ENV):
        return selected

    def _bloom_index(blk: Dict[str, Any]) -> int:
        b = str(blk.get("target_bloom") or "")
        return BLOOM_LEVELS.index(b) if b in BLOOM_LEVELS else 0

    def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, int]:
        idx, blk = item
        tier = _block_climb_tier(blk)
        tier_rank = (
            _CLIMB_TIERS.index(tier) if tier in _CLIMB_TIERS else len(_CLIMB_TIERS)
        )
        # Within exposition, vocabulary / pretraining sort FIRST (sub-rank 0).
        vocab_rank = (
            0
            if (
                tier == "exposition"
                and str(blk.get("block_type") or "") in _VOCAB_PRETRAINING_TYPES
            )
            else 1
        )
        return (tier_rank, vocab_rank, _bloom_index(blk))

    # Python's sort is stable, so equal-key blocks keep their planner order.
    indexed = list(enumerate(selected))
    indexed.sort(key=_sort_key)
    return [blk for _idx, blk in indexed]


def _ensure_lifecycle_endpoints(
    *,
    selected: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Guarantee an Activate-stage opener + Consolidate-stage closer (IB7.4).

    Gated on ``ED4ALL_PLANNER_LIFECYCLE``; NO-OP / byte-stable when off.

    1. If no Activate-stage block (hook / objective / prereq_set) is present,
       PREPEND a deterministic activation opener on ``overview`` (the first
       opener type available in ``BLOCK_TYPES``).
    2. If no Consolidate-stage block (summary_takeaway / recap / checklist /
       reflection_prompt) is present, APPEND one on ``summary``.
    3. Slot-edit escalation: for any block whose ``target_bloom`` exceeds its
       catalog ``bloom_ceiling``, FIRST stamp a heavier interaction/feedback
       weight on its ``anatomy_slot_weights`` annotation (additive,
       hash-excluded) rather than swapping the type — only IB7.6's ceiling
       re-route swaps the type when the slot edit cannot carry the demand.

    ANTI-FABRICATION: an injected opener/closer targets the TO's first real CO
    id (mirrors ``_inject_palette_v2``); no CO id is invented. Returns a NEW
    list.
    """
    if not _env_floor_on(_LIFECYCLE_ENV):
        return selected

    out = list(selected)
    first_co_id = next(
        (str(co.get("id")) for co in chapter_objectives if co.get("id")), ""
    )
    target_co_ids = [first_co_id] if first_co_id else []

    # (1) activation opener.
    has_opener = any(
        str(b.get("block_type") or "") in _ACTIVATION_OPENER_TYPES for b in out
    )
    if not has_opener:
        opener_bt = next(
            (bt for bt in _ACTIVATION_OPENER_TYPES if bt in block_types), None
        )
        if opener_bt is not None:
            out.insert(0, {
                "block_type": opener_bt,
                "page_type": "overview",
                "target_co_ids": list(target_co_ids),
                "content_focus": (
                    "IB7.4 lifecycle: Activate-stage opener (no activation "
                    "block opened this objective)"
                ),
                "target_bloom": "understand",
            })

    # (2) consolidate closer.
    has_closer = any(
        str(b.get("block_type") or "") in _CONSOLIDATE_CLOSER_TYPES for b in out
    )
    if not has_closer:
        closer_bt = next(
            (bt for bt in _CONSOLIDATE_CLOSER_TYPES if bt in block_types), None
        )
        if closer_bt is not None:
            out.append({
                "block_type": closer_bt,
                "page_type": "summary",
                "target_co_ids": list(target_co_ids),
                "content_focus": (
                    "IB7.4 lifecycle: Consolidate-stage closer (no summary "
                    "block closed this objective)"
                ),
                "target_bloom": "understand",
            })

    # (3) slot-edit escalation BEFORE any type re-route (IB7.6 swaps the type).
    for blk in out:
        if _bloom_over_ceiling(blk, catalog_by_type):
            weights = blk.get("anatomy_slot_weights")
            if not isinstance(weights, dict):
                weights = {}
            # Heavier interaction + feedback demand carries the higher Bloom
            # work without changing the block type (framework p.139 Step 4).
            weights.setdefault("interaction", "heavy")
            weights.setdefault("feedback", "heavy")
            blk["anatomy_slot_weights"] = weights

    return out


def _apply_spacing(
    *, selected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Separate a check/reflection from the exposition that taught its CO (IB7.5a).

    Gated on ``ED4ALL_PLANNER_SPACING``; NO-OP / byte-stable when off.
    Deterministic, bounded, ADDS NO blocks. For each CO, if its first
    check/reflection block sits IMMEDIATELY AFTER an exposition block teaching
    the SAME CO (massed, back-to-back), move the check one position LATER past
    the next intervening non-same-CO block so a checkpoint follows intervening
    material rather than the exposition it checks (spacing axis 3 / QA-15 /
    p.140). A single bounded pass; returns a NEW list.
    """
    if not _env_floor_on(_SPACING_ENV):
        return selected

    _CHECK_TYPES = {"self_check_question", "reflection_prompt", "assessment_item"}
    _EXPO_TIER = "exposition"
    out = list(selected)
    n = len(out)
    i = 1
    while i < n:
        blk = out[i]
        if str(blk.get("block_type") or "") not in _CHECK_TYPES:
            i += 1
            continue
        co_ids = set(str(c) for c in (blk.get("target_co_ids") or []))
        if not co_ids:
            i += 1
            continue
        prev = out[i - 1]
        prev_is_same_co_expo = (
            _block_climb_tier(prev) == _EXPO_TIER
            and bool(set(str(c) for c in (prev.get("target_co_ids") or [])) & co_ids)
        )
        if not prev_is_same_co_expo:
            i += 1
            continue
        # Find the next intervening block that does NOT teach the same CO and
        # is NOT itself a check — move the check to sit AFTER it.
        j = i + 1
        target = -1
        while j < n:
            cand = out[j]
            cand_cos = set(str(c) for c in (cand.get("target_co_ids") or []))
            if (
                str(cand.get("block_type") or "") not in _CHECK_TYPES
                and not (cand_cos & co_ids)
            ):
                target = j
                break
            j += 1
        if target == -1:
            i += 1
            continue
        moved = out.pop(i)
        out.insert(target, moved)  # target index shifts left by 1 after pop
        i += 1
    return out


def _catalog_bloom_ceiling(
    block_type: str, catalog_by_type: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Return a type's catalog ``bloom_ceiling`` (None ⇒ uncapped/advisory)."""
    entry = catalog_by_type.get(block_type) or {}
    ceiling = entry.get("bloom_ceiling")
    if isinstance(ceiling, str) and ceiling in BLOOM_LEVELS:
        return ceiling
    return None


def _bloom_over_ceiling(
    blk: Dict[str, Any], catalog_by_type: Dict[str, Dict[str, Any]],
) -> bool:
    """True iff a block's ``target_bloom`` exceeds its type's catalog ceiling."""
    bt = str(blk.get("block_type") or "")
    ceiling = _catalog_bloom_ceiling(bt, catalog_by_type)
    if ceiling is None:
        return False
    bloom = str(blk.get("target_bloom") or "")
    if bloom not in BLOOM_LEVELS:
        return False
    return BLOOM_LEVELS.index(bloom) > BLOOM_LEVELS.index(ceiling)


def _apply_bloom_ceilings(
    *,
    selected: List[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """Re-route an over-ceiling block to a higher-order type (IB7.6b).

    Gated on ``ED4ALL_PLANNER_BLOOM_CEILING``; NO-OP / byte-stable when off.
    For any block whose ``target_bloom`` exceeds its type's catalog
    ``bloom_ceiling`` (the advisory ``bloom_fit`` becomes a gate): IB7.4's
    slot-edit has already fired (heavier slot weights stamped); if the demand
    STILL exceeds the type, RE-ROUTE the block to the first higher-order target
    (``scenario`` / ``problem`` / ``assessment_item``) whose own ceiling admits
    the demanded level — no "everything block". Re-route changes
    ``block_type`` / ``page_type`` only; ``target_co_ids`` are PRESERVED
    (anti-fabrication: never invents a CO id). Returns a NEW list.
    """
    if not _env_floor_on(_BLOOM_CEILING_ENV):
        return selected

    out = list(selected)
    for blk in out:
        if not _bloom_over_ceiling(blk, catalog_by_type):
            continue
        bloom = str(blk.get("target_bloom") or "")
        demand = BLOOM_LEVELS.index(bloom) if bloom in BLOOM_LEVELS else 0
        reroute_to = None
        for cand in _BLOOM_REROUTE_TARGETS:
            if cand not in block_types:
                continue
            cand_ceiling = _catalog_bloom_ceiling(cand, catalog_by_type)
            # The target must admit the demanded level (ceiling None ⇒ uncapped,
            # admits anything; else ceiling index >= demand).
            if cand_ceiling is None or BLOOM_LEVELS.index(cand_ceiling) >= demand:
                reroute_to = cand
                break
        if reroute_to is None:
            continue
        original_type = str(blk.get("block_type") or "")
        blk["block_type"] = reroute_to
        blk["page_type"] = _BLOCK_TYPE_DEFAULT_PAGE.get(reroute_to, "application")
        focus = str(blk.get("content_focus") or "")
        blk["content_focus"] = (
            f"IB7.6 ceiling re-route: {original_type}@{bloom} exceeded its "
            f"type ceiling — routed to {reroute_to}"
            + (f" ({focus})" if focus else "")
        )
    return out


# FR-PLAN-01 — the Chapter-5 14-type activity catalog. The canonical interaction
# (activity) types the deterministic resolver may stamp onto an interaction-
# bearing block, drawn from the catalog's per-type ``default_activity_types``.
# The resolver picks ONE per block from that list × the block's target Bloom.
_ACTIVITY_TYPES: Tuple[str, ...] = (
    "multiple_choice",
    "multiple_response",
    "true_false",
    "fill_in_blank",
    "matching",
    "ordering",
    "drag_drop",
    "hotspot",
    "short_answer",
    "essay",
    "numeric",
    "categorization",
    "labeling",
    "branching_scenario",
)

# A coarse Bloom-band → activity-type AFFINITY ordering. Higher-order Bloom
# levels prefer constructed-response / scenario activities; lower-order levels
# prefer recognition / recall activities. The resolver intersects this affinity
# order with the block's catalog ``default_activity_types`` and picks the first
# match — a DETERMINISTIC choice driven by B-code × bloom_fit × catalog list.
_BLOOM_ACTIVITY_AFFINITY: Dict[str, Tuple[str, ...]] = {
    "remember": ("true_false", "multiple_choice", "matching", "labeling", "fill_in_blank"),
    "understand": ("multiple_choice", "matching", "categorization", "fill_in_blank", "true_false"),
    "apply": ("numeric", "fill_in_blank", "drag_drop", "ordering", "short_answer", "multiple_choice"),
    "analyze": ("categorization", "hotspot", "matching", "short_answer", "multiple_response"),
    "evaluate": ("short_answer", "essay", "branching_scenario", "multiple_response"),
    "create": ("essay", "branching_scenario", "short_answer"),
}


def _catalog_activity_types(
    block_type: str, catalog_by_type: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return a type's catalog ``default_activity_types`` (valid tokens only)."""
    entry = catalog_by_type.get(block_type) or {}
    raw = entry.get("default_activity_types") or []
    return [a for a in raw if isinstance(a, str) and a in _ACTIVITY_TYPES]


def _resolve_one_interaction_type(
    *,
    block_type: str,
    target_bloom: str,
    catalog_by_type: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Deterministically pick ONE interaction type for a single block.

    Returns ``None`` for a non-interaction-bearing type (no
    ``default_activity_types`` in the catalog). Otherwise intersects the block's
    target Bloom affinity order with the catalog list and returns the first
    match; falls back to the catalog list's FIRST entry when no affinity token
    is offered (so an interaction-bearing block always gets a type).
    """
    candidates = _catalog_activity_types(block_type, catalog_by_type)
    if not candidates:
        return None
    candidate_set = set(candidates)
    bloom = target_bloom if target_bloom in _BLOOM_ACTIVITY_AFFINITY else "understand"
    for activity in _BLOOM_ACTIVITY_AFFINITY[bloom]:
        if activity in candidate_set:
            return activity
    # No affinity token offered — keep the catalog's authored first choice.
    return candidates[0]


def _resolve_interaction_types(
    *,
    selected: List[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """FR-PLAN-01 — stamp an ``interaction_type`` on each interaction-bearing block.

    Gated on ``ED4ALL_DYNAMIC_BLOCK_PLAN`` (the existing planner flag); a strict
    identity NO-OP when off so a default-off / legacy run is byte-stable. For
    every block whose type carries catalog ``default_activity_types``, picks ONE
    interaction type from its framework B-code × ``bloom_fit`` × the catalog list
    (see :func:`_resolve_one_interaction_type`); non-interaction-bearing blocks
    are left untouched. Returns a NEW list (the input dicts are mutated in place,
    matching the IB7 passes' posture).
    """
    if not _dynamic_block_plan_on():
        return selected
    out = list(selected)
    for blk in out:
        bt = str(blk.get("block_type") or "")
        bloom = str(blk.get("target_bloom") or "")
        itype = _resolve_one_interaction_type(
            block_type=bt, target_bloom=bloom, catalog_by_type=catalog_by_type,
        )
        if itype is not None:
            blk["interaction_type"] = itype
    return out


# FR-INT-06 — B09 case/scenario authoring MODE-by-Bloom. A real B09 block is one
# of three escalating forms; the demanded cognitive level selects the form:
#   apply            -> ``case``      (analyze a static worked situation)
#   analyze/evaluate -> ``scenario``  (make a situated decision)
#   create           -> ``branching`` (navigate a multi-step decision tree)
# Lower/unknown levels fall back to ``case`` (the least-escalated form).
_SCENARIO_MODE_BY_BLOOM: Dict[str, str] = {
    "remember": "case",
    "understand": "case",
    "apply": "case",
    "analyze": "scenario",
    "evaluate": "scenario",
    "create": "branching",
}


def _resolve_one_scenario_mode(target_bloom: str) -> str:
    """Pick the B09 authoring mode for one scenario block from its target Bloom.

    Returns ``case`` | ``scenario`` | ``branching``; defaults to ``case`` (the
    least-escalated form) for a lower-order or unknown Bloom level.
    """
    return _SCENARIO_MODE_BY_BLOOM.get(
        target_bloom if target_bloom in _SCENARIO_MODE_BY_BLOOM else "", "case"
    )


def _resolve_scenario_modes(
    *,
    selected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """FR-INT-06 — stamp a ``scenario_mode`` on each B09 ``scenario`` block.

    Gated on ``ED4ALL_DYNAMIC_BLOCK_PLAN`` (the existing planner flag); a strict
    identity NO-OP when off so a default-off / legacy run is byte-stable (mirrors
    :func:`_resolve_interaction_types`). For every ``scenario`` block, picks the
    case/scenario/branching mode from its target Bloom level (see
    :func:`_resolve_one_scenario_mode`); a block that already carries an explicit
    ``scenario_mode`` is left untouched. Returns a NEW list (the input dicts are
    mutated in place, matching the other passes' posture).
    """
    if not _dynamic_block_plan_on():
        return selected
    out = list(selected)
    for blk in out:
        if str(blk.get("block_type") or "") != "scenario":
            continue
        if blk.get("scenario_mode"):
            continue
        blk["scenario_mode"] = _resolve_one_scenario_mode(
            str(blk.get("target_bloom") or "")
        )
    return out


# FR-INT-01 — the canonical B08 guided-practice fade ladder. After a fully-
# worked example the learner moves to a completion problem (partial scaffold)
# then to independent practice. The fading pass injects a single B08 completion
# block carrying the ``completion`` fade_state after each worked_example.
_FADE_LADDER: Tuple[str, str, str] = ("worked", "completion", "independent")

# The B08 (guided-practice) block type the fading pass injects as the faded
# step after a worked_example, in preference order — the first one present in
# the resolved block-type palette is used.
_FADING_B08_TYPES: Tuple[str, ...] = ("problem", "activity", "checklist")


def _apply_fading_sequence(
    *,
    selected: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    block_types: frozenset,
) -> List[Dict[str, Any]]:
    """FR-INT-01 — inject a B08 faded-practice block after a worked_example (B05).

    Gated on ``ED4ALL_PLANNER_FADING``; a strict identity NO-OP when off so a
    default-off run is byte-stable (mirrors the IB7 planner passes' posture). The
    framework's guided-practice fading sequence (worked → completion →
    independent, p.139) is realized by: for each ``worked_example`` block that is
    not already FOLLOWED by a faded-practice (B08) block, inject ONE B08
    completion block (``problem`` / ``activity`` / ``checklist`` — the first in
    the palette) right after it and stamp its ``fade_state="completion"`` so the
    practice is partially-scaffolded (the reused IB5 ``fade_state`` field). The
    worked_example itself is stamped ``fade_state="worked"`` when it carries
    none. Anti-fabrication: the injected block inherits the worked_example's
    ``target_co_ids`` (or the TO's first real CO id) — no CO id is invented.
    Returns a NEW list.
    """
    if not _env_floor_on(_FADING_ENV):
        return selected

    # The B08 type to use (first present in the palette).
    b08_type: Optional[str] = None
    for cand in _FADING_B08_TYPES:
        if cand in block_types:
            b08_type = cand
            break
    if b08_type is None:
        return selected

    # Fallback CO id (first real CO) for anti-fabrication grounding.
    fallback_co: List[str] = []
    for co in chapter_objectives or ():
        cid = str(co.get("id") or co.get("co_id") or "").strip()
        if cid:
            fallback_co = [cid]
            break

    b08_page = _BLOCK_TYPE_DEFAULT_PAGE.get(b08_type, "application")

    out: List[Dict[str, Any]] = []
    for idx, blk in enumerate(selected):
        out.append(blk)
        if str(blk.get("block_type") or "") != "worked_example":
            continue
        # Stamp the worked stage on the worked_example when it has none.
        if not blk.get("fade_state"):
            blk["fade_state"] = "worked"
        # Skip injection when the NEXT block is already a faded-practice B08.
        nxt = selected[idx + 1] if idx + 1 < len(selected) else None
        if nxt is not None and str(nxt.get("block_type") or "") in _FADING_B08_TYPES:
            # Ensure the existing follow-on carries a completion fade_state.
            if not nxt.get("fade_state"):
                nxt["fade_state"] = "completion"
            continue
        target_co_ids = [str(c) for c in (blk.get("target_co_ids") or [])] or list(fallback_co)
        out.append({
            "block_type": b08_type,
            "page_type": b08_page,
            "target_co_ids": target_co_ids,
            "content_focus": (
                "FR-INT-01 faded practice: completion step after the worked "
                "example (guided-practice fade ladder worked→completion→"
                "independent)"
            ),
            "target_bloom": str(blk.get("target_bloom") or "apply"),
            "fade_state": "completion",
        })
    return out


def _apply_ib7_passes(
    *,
    selected: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    catalog_by_type: Dict[str, Dict[str, Any]],
    block_types: frozenset,
    signals: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run the four IB7 planner-pedagogy passes in dependency order.

    climb → lifecycle (open/close + slot-edit escalation) → spacing → ceiling
    re-route. Each pass is a strict identity no-op when its env flag is off, so
    a default-off run returns ``selected`` byte-identical. Records per-pass
    SIGNALS into ``signals`` (consumed by the ``block_plan`` decision capture):
    ``bloom_climb_applied`` / ``lifecycle_opened`` / ``lifecycle_closed`` /
    ``spacing_moves`` / ``bloom_ceiling_reroutes`` / ``slot_weight_edits``.
    """
    # Climb.
    climb_on = _env_floor_on(_BLOOM_CLIMB_ENV)
    selected = _apply_bloom_climb(selected=selected)
    signals["bloom_climb_applied"] = climb_on

    # Lifecycle open/close + slot-edit escalation.
    lifecycle_on = _env_floor_on(_LIFECYCLE_ENV)
    n_before = len(selected)
    opener_types_before = any(
        str(b.get("block_type") or "") in _ACTIVATION_OPENER_TYPES
        for b in selected
    )
    closer_types_before = any(
        str(b.get("block_type") or "") in _CONSOLIDATE_CLOSER_TYPES
        for b in selected
    )
    selected = _ensure_lifecycle_endpoints(
        selected=selected,
        chapter_objectives=chapter_objectives,
        catalog_by_type=catalog_by_type,
        block_types=block_types,
    )
    signals["lifecycle_opened"] = bool(
        lifecycle_on and not opener_types_before
        and any(
            str(b.get("block_type") or "") in _ACTIVATION_OPENER_TYPES
            for b in selected
        )
    )
    signals["lifecycle_closed"] = bool(
        lifecycle_on and not closer_types_before
        and any(
            str(b.get("block_type") or "") in _CONSOLIDATE_CLOSER_TYPES
            for b in selected
        )
    )
    signals["slot_weight_edits"] = sum(
        1 for b in selected if isinstance(b.get("anatomy_slot_weights"), dict)
    ) if lifecycle_on else 0

    # Spacing.
    spacing_on = _env_floor_on(_SPACING_ENV)
    order_before = [id(b) for b in selected]
    selected = _apply_spacing(selected=selected)
    signals["spacing_moves"] = (
        sum(1 for a, b in zip(order_before, [id(x) for x in selected]) if a != b)
        if spacing_on else 0
    )

    # Ceiling re-route.
    ceiling_on = _env_floor_on(_BLOOM_CEILING_ENV)
    types_before = [b.get("block_type") for b in selected]
    selected = _apply_bloom_ceilings(
        selected=selected,
        catalog_by_type=catalog_by_type,
        block_types=block_types,
    )
    signals["bloom_ceiling_reroutes"] = (
        sum(
            1 for a, b in zip(types_before, [x.get("block_type") for x in selected])
            if a != b
        )
        if ceiling_on else 0
    )

    # FR-INT-01 fading sequence (inject a B08 completion block after each
    # worked_example). Identity no-op when ED4ALL_PLANNER_FADING is off.
    fading_on = _env_floor_on(_FADING_ENV)
    n_before_fade = len(selected)
    selected = _apply_fading_sequence(
        selected=selected,
        chapter_objectives=chapter_objectives,
        block_types=block_types,
    )
    signals["fading_blocks_injected"] = (
        len(selected) - n_before_fade if fading_on else 0
    )

    # FR-PLAN-01 interaction-type resolution (stamp an interaction_type on each
    # interaction-bearing block). Runs LAST so every final block — including any
    # injected by the fading pass — gets a type. Identity no-op when
    # ED4ALL_DYNAMIC_BLOCK_PLAN is off.
    selected = _resolve_interaction_types(
        selected=selected, catalog_by_type=catalog_by_type,
    )
    signals["interaction_types_assigned"] = sum(
        1 for b in selected if b.get("interaction_type")
    )
    # FR-INT-06 B09 scenario-mode resolution (stamp a scenario_mode on each
    # scenario block by its target Bloom). Identity no-op when
    # ED4ALL_DYNAMIC_BLOCK_PLAN is off.
    selected = _resolve_scenario_modes(selected=selected)
    signals["scenario_modes_assigned"] = sum(
        1 for b in selected if b.get("scenario_mode")
    )
    return selected


def _resolve_target_bloom(
    *,
    declared: Any,
    block_type: str,
    catalog_by_type: Dict[str, Dict[str, Any]],
    targets: Sequence[str],
    co_bloom: Dict[str, str],
) -> str:
    """Resolve a block's target Bloom level (declared > catalog > CO > floor)."""
    if isinstance(declared, str) and declared.strip() in BLOOM_LEVELS:
        return declared.strip()
    entry = catalog_by_type.get(block_type, {})
    floor = _bloom_floor_for_catalog_entry(entry)
    if floor:
        return floor
    for cid in targets:
        if cid in co_bloom and co_bloom[cid] in BLOOM_LEVELS:
            return co_bloom[cid]
    return "understand"


def _to_page_plan(
    selected: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, str, List[str]]]]:
    """Project the validated block list into the ``_PAGE_TYPE_BLOCK_PLAN``
    shape: ``page_type -> ordered [(block_type, target_bloom, target_co_ids)]``.

    Each entry is a 3-tuple; ``target_co_ids`` is the (possibly empty) list of
    specific CO ids this block teaches, carried through from the planner so the
    outline-phase consumer can stamp the block's ``objective_ids`` with the
    SPECIFIC CO (not the week TO). An empty list is the signal "no specific CO —
    use the week-TO fallback".

    Every canonical page type is present as a key (empty list when no block
    landed there) so the consumer's per-page loop never KeyErrors."""
    plan: Dict[str, List[Tuple]] = {
        ptype: [] for ptype in CANONICAL_PAGE_TYPES
    }
    for blk in selected:
        ptype = blk["page_type"]
        if ptype not in plan:
            ptype = "content"
        target_co_ids = [str(c) for c in (blk.get("target_co_ids") or [])]
        # FR-PLAN-01 / FR-INT-01: emit a 4th tuple element (the planner-selected
        # interaction_type) and a 5th (the FR-INT-01 fade_state) ONLY when set,
        # so a flag-off run stays byte-stable at the legacy 3-tuple shape. The
        # consumer's unpack is arity-tolerant.
        itype = blk.get("interaction_type")
        fade = blk.get("fade_state")
        if fade:
            plan[ptype].append(
                (
                    blk["block_type"], blk["target_bloom"], target_co_ids,
                    str(itype) if itype else None, str(fade),
                )
            )
        elif itype:
            plan[ptype].append(
                (blk["block_type"], blk["target_bloom"], target_co_ids, str(itype))
            )
        else:
            plan[ptype].append(
                (blk["block_type"], blk["target_bloom"], target_co_ids)
            )
    return plan


def plan_week_blocks(
    *,
    terminal_objective: Dict[str, Any],
    chapter_objectives: Optional[Sequence[Dict[str, Any]]] = None,
    source_chunks: Optional[Sequence[Dict[str, Any]]] = None,
    catalog: Optional[Sequence[Dict[str, Any]]] = None,
    provider: Optional[Any] = None,
    budget: Tuple[int, int] = DEFAULT_BLOCK_BUDGET,
    capture: Optional[Any] = None,
    course_code: str = "",
) -> WeekBlockPlan:
    """Plan a week's block sequence for one terminal objective via the 70B.

    Args:
        terminal_objective: ``{id, statement}`` for the week's TO.
        chapter_objectives: child COs (``{id, statement, bloom_level}``).
        source_chunks: digest of the TO's source chunk text
            (``{id, text, heading}``); truncated for the prompt.
        catalog: block catalog list (defaults to ``load_block_catalog()``).
        provider: an object exposing ``plan_blocks(prompt) -> str`` OR a
            ``_BaseLLMProvider``-like with ``_dispatch_call(prompt) ->
            (text, retries)``. ``None`` → deterministic fixed-plan fallback.
        budget: ``(min, max)`` blocks per week.
        capture: optional :class:`DecisionCapture`.
        course_code: course slug for the decision event.

    Returns:
        A :class:`WeekBlockPlan`. NEVER raises — any failure degrades to the
        fixed fallback plan.
    """
    chapter_objectives = list(chapter_objectives or [])
    source_chunks = list(source_chunks or [])
    budget = _clamp_budget(budget)
    to_id = str(terminal_objective.get("id") or "")
    # IB7.1 signal: did ANY source chunk carry a non-empty structural heading?
    source_headings_present = any(
        isinstance(ch, dict) and str(ch.get("heading") or "").strip()
        for ch in source_chunks
    )
    # IB7.2 signal: which planner seat is in use (nvidia | local | …)?
    planner_seat = str(
        getattr(provider, "_provider", "")
        or getattr(provider, "provider", "")
        or ""
    )
    # The TO's source blob — fed to the deterministic palette-v2 injection on
    # both the LLM-success and every fallback path (gated on
    # ED4ALL_DYNAMIC_BLOCK_PLAN inside ``_inject_palette_v2`` / ``_fallback_plan``).
    source_blob = _source_text_blob(source_chunks)

    try:
        catalog = list(catalog) if catalog is not None else load_block_catalog()
    except Exception as exc:  # noqa: BLE001 — never break the build
        logger.warning("block_planner: catalog load failed (%s); fallback", exc)
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason=f"catalog load error: {exc}",
            source_blob=source_blob, chapter_objectives=chapter_objectives,
        )

    block_types = _resolve_block_types()
    catalog_by_type = {
        str(e.get("block_type")): e for e in catalog if e.get("block_type")
    }
    co_bloom = {
        str(co.get("id")): str(co.get("bloom_level") or "")
        for co in chapter_objectives if co.get("id")
    }

    # No provider → deterministic fallback (the OFF path / unit-test default).
    if provider is None:
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason="no provider supplied",
            source_blob=source_blob, chapter_objectives=chapter_objectives,
        )

    prompt = _build_prompt(
        terminal_objective=terminal_objective,
        chapter_objectives=chapter_objectives,
        source_chunks=source_chunks,
        catalog=catalog,
        budget=budget,
    )

    model = getattr(provider, "_model", "") or getattr(provider, "model", "")
    raw_text = ""
    try:
        if hasattr(provider, "plan_blocks"):
            raw_text = provider.plan_blocks(prompt)
        else:
            result = provider._dispatch_call(prompt)
            raw_text = result[0] if isinstance(result, tuple) else result
    except Exception as exc:  # noqa: BLE001 — fail-safe
        logger.warning(
            "block_planner: LLM dispatch failed for %s (%s); fixed fallback",
            to_id, exc,
        )
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason=f"dispatch error: {exc}", model=str(model),
            source_blob=source_blob, chapter_objectives=chapter_objectives,
        )

    raw_blocks = _parse_llm_blocks(raw_text or "")
    if not raw_blocks:
        logger.warning(
            "block_planner: unparseable/empty plan for %s; fixed fallback",
            to_id,
        )
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason="unparseable/empty LLM response",
            model=str(model),
            source_blob=source_blob, chapter_objectives=chapter_objectives,
        )

    selected = _validate_and_repair(
        raw_blocks=raw_blocks,
        chapter_objectives=chapter_objectives,
        catalog_by_type=catalog_by_type,
        block_types=block_types,
        budget=budget,
        co_bloom=co_bloom,
    )
    if not selected:
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason="all blocks dropped by guardrails",
            model=str(model),
            source_blob=source_blob, chapter_objectives=chapter_objectives,
        )

    # Per-page-type FLOORS (findings 2/7/8/18): top up any starved page TYPE
    # so no page is threadbare, deploying the contracted-but-unused block
    # types (findings 6/16). Applied after coverage + budget; floors take
    # precedence over the budget max (the default max is raised to fund them).
    selected = _apply_page_floors(
        selected=selected,
        block_types=block_types,
        budget=budget,
    )

    # P4 pedagogical-depth floors (both default OFF → byte-stable). The
    # worked-example floor guarantees every procedural CO gets >= 1
    # example/problem block (DENSITY); the Bloom-spread floor guarantees >= 1
    # analyze-or-higher block per week (Bloom VARIETY). Applied after the
    # per-page floors so they see the post-top-up block set.
    selected = _apply_worked_example_floor(
        selected=selected,
        chapter_objectives=chapter_objectives,
        block_types=block_types,
    )
    selected = _apply_bloom_spread_floor(
        selected=selected,
        catalog_by_type=catalog_by_type,
        block_types=block_types,
    )

    # Issue I6 (Part A): deterministically deploy the palette-v2 types
    # (table / acronym / key_idea) by CONTENT SHAPE so they no longer depend on
    # the 70B choosing them. NO-OP when ED4ALL_DYNAMIC_BLOCK_PLAN is off (the
    # legacy path never reaches here anyway — the consumer skips the planner —
    # but the env gate keeps unit-test byte-stability for callers that pass a
    # provider without the flag set).
    if _dynamic_block_plan_on():
        selected = _inject_palette_v2(
            selected=selected,
            source_blob=source_blob,
            chapter_objectives=chapter_objectives,
            block_types=block_types,
        )

    # IB5 (mirrors the palette-v2 injection): deterministically deploy the four
    # framework-aligned types (hook / multimedia / worked_example / diagram) by
    # CONTENT SHAPE so they no longer depend on the 70B selecting them. Gated on
    # ED4ALL_NEW_BLOCK_TYPES — a strict NO-OP when off so the byte-stability
    # guard holds (the four types are never selected on a legacy run).
    if _new_block_types_on():
        selected = _inject_ib5_types(
            selected=selected,
            source_blob=source_blob,
            chapter_objectives=chapter_objectives,
            block_types=block_types,
        )

    # IB7 planner-pedagogy passes (each gated by its own flag; default-off ⇒
    # identity, so the planner output is byte-stable). Order: climb → lifecycle
    # → spacing → ceiling re-route (the plan's internal dependency order).
    ib7_signals: Dict[str, Any] = {}
    selected = _apply_ib7_passes(
        selected=selected,
        chapter_objectives=chapter_objectives,
        catalog_by_type=catalog_by_type,
        block_types=block_types,
        signals=ib7_signals,
    )

    page_plan = _to_page_plan(selected)
    _emit_block_plan_decision(
        capture=capture,
        course_code=course_code,
        to_id=to_id,
        selected=selected,
        chapter_objectives=chapter_objectives,
        budget=budget,
        model=str(model),
        fallback_used=False,
        ib7_signals=ib7_signals,
        planner_seat=planner_seat,
        source_headings_present=source_headings_present,
    )
    return WeekBlockPlan(
        page_plan=page_plan,
        selected=selected,
        fallback_used=False,
        terminal_objective_id=to_id,
        model=str(model),
    )


def _fallback_plan(
    *,
    to_id: str,
    capture: Optional[Any],
    course_code: str,
    budget: Tuple[int, int],
    reason: str,
    model: str = "",
    source_blob: str = "",
    chapter_objectives: Optional[Sequence[Dict[str, Any]]] = None,
) -> WeekBlockPlan:
    """Build the deterministic fixed-plan ``WeekBlockPlan`` + capture."""
    page_plan = _default_page_plan()
    selected = [
        {
            "block_type": bt,
            "page_type": ptype,
            "target_co_ids": [],
            "content_focus": "fixed-plan fallback",
            "target_bloom": bloom,
        }
        for ptype, specs in page_plan.items()
        for bt, bloom, _co_ids in specs
    ]
    # Issue I6 (Part A): the palette-v2 injection ALSO runs on the fallback
    # path (an LLM error must not strand the I6 types), gated on
    # ED4ALL_DYNAMIC_BLOCK_PLAN. When off (the legacy path + the provider=None
    # unit-test path) this is a strict NO-OP, so the fixed-plan fallback stays
    # byte-identical (every entry keeps its empty target_co_ids). When on, the
    # injected entries extend ``selected`` and ``page_plan`` is RE-DERIVED from
    # it so the new blocks surface through ``page_block_plan_for``.
    if _dynamic_block_plan_on():
        selected = _inject_palette_v2(
            selected=selected,
            source_blob=source_blob,
            chapter_objectives=chapter_objectives or [],
            block_types=_resolve_block_types(),
        )
        page_plan = _to_page_plan(selected)
    # IB5 injection ALSO runs on the fallback path (gated on
    # ED4ALL_NEW_BLOCK_TYPES) so an LLM error does not strand the four types;
    # strict NO-OP when off so the fixed-plan fallback stays byte-identical.
    if _new_block_types_on():
        selected = _inject_ib5_types(
            selected=selected,
            source_blob=source_blob,
            chapter_objectives=chapter_objectives or [],
            block_types=_resolve_block_types(),
        )
        page_plan = _to_page_plan(selected)
    # IB7 passes ALSO run on the fallback path (each gated by its own flag; a
    # strict NO-OP when off so the fixed-plan fallback stays byte-identical).
    # When any IB7 flag is on the injected/reordered entries re-derive page_plan.
    ib7_signals: Dict[str, Any] = {}
    if _any_ib7_flag_on():
        try:
            _catalog_by_type = {
                str(e.get("block_type")): e
                for e in load_block_catalog() if e.get("block_type")
            }
        except Exception:  # noqa: BLE001 — never break the fallback
            _catalog_by_type = {}
        selected = _apply_ib7_passes(
            selected=selected,
            chapter_objectives=chapter_objectives or [],
            catalog_by_type=_catalog_by_type,
            block_types=_resolve_block_types(),
            signals=ib7_signals,
        )
        page_plan = _to_page_plan(selected)
    _emit_block_plan_decision(
        capture=capture,
        course_code=course_code,
        to_id=to_id,
        selected=selected,
        chapter_objectives=chapter_objectives or [],
        budget=budget,
        model=model,
        fallback_used=True,
        reason=reason,
        ib7_signals=ib7_signals,
    )
    return WeekBlockPlan(
        page_plan=page_plan,
        selected=selected,
        fallback_used=True,
        terminal_objective_id=to_id,
        model=model,
    )


def _emit_block_plan_decision(
    *,
    capture: Optional[Any],
    course_code: str,
    to_id: str,
    selected: List[Dict[str, Any]],
    chapter_objectives: Sequence[Dict[str, Any]],
    budget: Tuple[int, int],
    model: str,
    fallback_used: bool,
    reason: str = "",
    ib7_signals: Optional[Dict[str, Any]] = None,
    planner_seat: str = "",
    source_headings_present: Optional[bool] = None,
) -> None:
    """Emit one ``block_plan`` decision event per TO (replayable rationale).

    IB7.7: EXTENDS the single ``block_plan`` event (no second event) with the
    IB7 planner-pedagogy signals (``bloom_climb_applied`` / ``lifecycle_opened``
    / ``lifecycle_closed`` / ``spacing_moves`` / ``bloom_ceiling_reroutes`` /
    ``slot_weight_edits`` / ``planner_seat`` / ``source_headings_present``); the
    rationale interpolates them so a replay reconstructs the ordering/lifecycle
    decisions without re-running the planner.
    """
    if capture is None:
        return
    ib7 = ib7_signals or {}
    chosen_types = [b["block_type"] for b in selected]
    type_counts: Dict[str, int] = {}
    for bt in chosen_types:
        type_counts[bt] = type_counts.get(bt, 0) + 1
    co_ids = {str(co.get("id")) for co in chapter_objectives if co.get("id")}
    covered: set = set()
    for b in selected:
        covered.update(b.get("target_co_ids") or [])
    coverage = (
        f"{len(covered & co_ids)}/{len(co_ids)}" if co_ids else "n/a (no COs)"
    )
    decision = (
        f"Planned {len(selected)} block(s) for {to_id or 'TO'}: "
        + ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))
    )
    # IB7 signal summary string interpolated into BOTH rationale branches.
    ib7_summary = (
        f"IB7 [seat={planner_seat or 'n/a'}, "
        f"climb={bool(ib7.get('bloom_climb_applied'))}, "
        f"lifecycle_open={bool(ib7.get('lifecycle_opened'))}, "
        f"lifecycle_close={bool(ib7.get('lifecycle_closed'))}, "
        f"spacing_moves={int(ib7.get('spacing_moves') or 0)}, "
        f"ceiling_reroutes={int(ib7.get('bloom_ceiling_reroutes') or 0)}, "
        f"slot_edits={int(ib7.get('slot_weight_edits') or 0)}, "
        f"headings={source_headings_present}]"
    )
    if fallback_used:
        rationale = (
            f"Fixed-plan FALLBACK fired for {to_id or 'TO'} ({reason}); used "
            f"the deterministic _PAGE_TYPE_BLOCK_PLAN template "
            f"({len(selected)} blocks across {len(_DEFAULT_PAGE_PLAN)} page "
            f"types). Budget was {budget[0]}-{budget[1]}; model='{model or 'n/a'}'. "
            f"The build is never broken by a planner failure. {ib7_summary}"
        )
    else:
        rationale = (
            f"{planner_seat or '70B'} content-aware planner selected "
            f"{len(selected)} blocks for {to_id or 'TO'} within budget "
            f"{budget[0]}-{budget[1]} (model='{model or 'n/a'}'); block-type mix "
            f"{dict(sorted(type_counts.items()))}; CO coverage {coverage}. "
            f"Each block was chosen for its content shape rather than a fixed "
            f"per-week template, so the week is content-shaped. {ib7_summary}"
        )
    try:
        capture.log_decision(
            decision_type="block_plan",
            decision=decision,
            rationale=rationale,
            ml_features={
                "terminal_objective_id": to_id,
                "n_blocks": len(selected),
                "block_type_counts": type_counts,
                "budget_min": budget[0],
                "budget_max": budget[1],
                "co_coverage": coverage,
                "fallback_used": fallback_used,
                "model": model,
                # IB7.7 planner-pedagogy signals (single event extension).
                "bloom_climb_applied": bool(ib7.get("bloom_climb_applied")),
                "lifecycle_opened": bool(ib7.get("lifecycle_opened")),
                "lifecycle_closed": bool(ib7.get("lifecycle_closed")),
                "spacing_moves": int(ib7.get("spacing_moves") or 0),
                "bloom_ceiling_reroutes": int(
                    ib7.get("bloom_ceiling_reroutes") or 0
                ),
                "slot_weight_edits": int(ib7.get("slot_weight_edits") or 0),
                "planner_seat": planner_seat,
                "source_headings_present": source_headings_present,
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break a run
        logger.debug("block_planner: decision capture failed: %s", exc)
