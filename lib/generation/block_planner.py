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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.generation.block_catalog import load_block_catalog
from lib.ontology.bloom import BLOOM_LEVELS

logger = logging.getLogger(__name__)

__all__ = [
    "plan_week_blocks",
    "WeekBlockPlan",
    "BlockPlannerProvider",
    "CANONICAL_PAGE_TYPES",
    "DEFAULT_BLOCK_BUDGET",
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
        # deploy callout ("Key Idea" framing, finding 16) + misconception.
        ("concept", "understand"),
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
}

# Bounded prompt sizing knobs (keep the per-TO prompt small enough for the
# 70B context without summarising the model away from the real content).
_MAX_SOURCE_CHUNKS_IN_PROMPT = 8
_MAX_CHARS_PER_CHUNK = 600
_MAX_COS_IN_PROMPT = 30


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
    if (provider or "").lower() == "nvidia":
        return os.environ.get("NVIDIA_LARGE_MODEL") or _PLANNER_70B_NVIDIA_MODEL
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
        "  - a recap      -> summary_takeaway / checklist\n\n"
        "When content is COMPARATIVE or TABULAR (several items contrasted "
        "across the same dimensions, term/definition pairs, or a "
        "criteria-by-option matrix), prefer flip_card_grid (a grid of paired "
        "cards) or a checklist over a flat prose concept — tabular content is "
        "clearer as a grid than as a paragraph.\n\n"
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
    plan: Dict[str, List[Tuple[str, str, List[str]]]] = {
        ptype: [] for ptype in CANONICAL_PAGE_TYPES
    }
    for blk in selected:
        ptype = blk["page_type"]
        if ptype not in plan:
            ptype = "content"
        target_co_ids = [str(c) for c in (blk.get("target_co_ids") or [])]
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

    try:
        catalog = list(catalog) if catalog is not None else load_block_catalog()
    except Exception as exc:  # noqa: BLE001 — never break the build
        logger.warning("block_planner: catalog load failed (%s); fallback", exc)
        return _fallback_plan(
            to_id=to_id, capture=capture, course_code=course_code,
            budget=budget, reason=f"catalog load error: {exc}",
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
    _emit_block_plan_decision(
        capture=capture,
        course_code=course_code,
        to_id=to_id,
        selected=selected,
        chapter_objectives=[],
        budget=budget,
        model=model,
        fallback_used=True,
        reason=reason,
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
) -> None:
    """Emit one ``block_plan`` decision event per TO (replayable rationale)."""
    if capture is None:
        return
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
    if fallback_used:
        rationale = (
            f"Fixed-plan FALLBACK fired for {to_id or 'TO'} ({reason}); used "
            f"the deterministic _PAGE_TYPE_BLOCK_PLAN template "
            f"({len(selected)} blocks across {len(_DEFAULT_PAGE_PLAN)} page "
            f"types). Budget was {budget[0]}-{budget[1]}; model='{model or 'n/a'}'. "
            f"The build is never broken by a planner failure."
        )
    else:
        rationale = (
            f"70B content-aware planner selected {len(selected)} blocks for "
            f"{to_id or 'TO'} within budget {budget[0]}-{budget[1]} "
            f"(model='{model or 'n/a'}'); block-type mix "
            f"{dict(sorted(type_counts.items()))}; CO coverage {coverage}. "
            f"Each block was chosen for its content shape rather than a fixed "
            f"per-week template, so the week is content-shaped."
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
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break a run
        logger.debug("block_planner: decision capture failed: %s", exc)
