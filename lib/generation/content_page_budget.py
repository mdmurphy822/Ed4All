"""Token-aware page-per-CO content-emit helpers (deterministic, no model calls).

Phase 0 of the page-per-CO fix
(``plans/finegrain/page-per-co-token-aware-emit-2026-06.md``). Pure-library
resolvers + a token-budget helper + a per-page union-then-top-K chunk cap. All
deterministic — NO model dispatch, NO GPU required for the resolvers or the
budget math; the optional top-K cap reuses the shared embedding client ONLY to
cosine-rank chunks, and degrades to citation-order when it is unavailable.

Gated entirely at the call site by ``ED4ALL_CONTENT_PAGE_PER_CO`` (default OFF,
no-op unless ``COURSEFORGE_TWO_PASS``). This module reads its companion env
knobs through parse-with-fallback resolvers and is otherwise side-effect-free,
so a default-off run never imports the embedding stack.

Companion knobs (all parse-with-fallback):

- ``ED4ALL_CONTENT_PAGE_PER_CO`` — master flag (default OFF).
- ``ED4ALL_CONTENT_PAGE_NUM_CTX`` — the authoring serving window; default falls
  back to ``ED4ALL_ANSWER_NUM_CTX`` then 4096.
- ``ED4ALL_CONTENT_PAGE_MAX_CHUNKS`` — hard top-K ceiling per page (default 5,
  mirroring ``ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE``).

Anti-fabrication contract for :func:`cap_page_chunks`: the kept id set is always
a SUBSET of the supplied union (only ever DROPS, never invents an id), and at
least one chunk is always kept (always-keep-≥1) when the union is non-empty.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var names + defaults
# ---------------------------------------------------------------------------
ENV_CONTENT_PAGE_PER_CO = "ED4ALL_CONTENT_PAGE_PER_CO"
ENV_CONTENT_PAGE_NUM_CTX = "ED4ALL_CONTENT_PAGE_NUM_CTX"
ENV_CONTENT_PAGE_MAX_CHUNKS = "ED4ALL_CONTENT_PAGE_MAX_CHUNKS"

#: Opt-in satellite of ``ED4ALL_CONTENT_PAGE_PER_CO``: when truthy, the
#: per-week content-page count is EXACTLY the week's CO count — the
#: NEVER-INCREASE topic-count cap is lifted (true one-HTML-page-per-CO).
#: Default unset → the guard stays intact (byte-identical). See
#: :func:`content_page_per_co_uncapped`.
ENV_CONTENT_PAGE_PER_CO_UNCAPPED = "ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED"

#: Gap #11 — per-block-role chunk-order diversification. Default OFF →
#: byte-identical (every co-located block keeps the identical page-ranked
#: chunk head, so one anchor example recurs across a whole page's blocks). See
#: :func:`diversify_block_chunk_order`.
ENV_CHUNK_ROLE_DIVERSIFY = "ED4ALL_CHUNK_ROLE_DIVERSIFY"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Default serving-window fallback when neither this feature's knob nor the
#: shared answer knob is set. Mirrors ``lib.retrieval._prompts.DEFAULT_NUM_CTX``.
_DEFAULT_NUM_CTX = 4096

#: Default hard top-K ceiling. Mirrors
#: ``ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE`` (5).
_DEFAULT_MAX_CHUNKS = 5

#: Conservative chars/token divisor, reused from the answer/window math so the
#: budget never re-derives its own divisor. 2.5 over-estimates (errs toward
#: dropping a trailing chunk rather than overflowing the window).
_CHARS_PER_TOKEN = 2.5

#: Per-chunk body truncation (characters) applied when estimating a chunk's
#: token weight for the budget. Mirrors the outline tier's per-chunk cap so a
#: single fat chunk cannot blow the estimate.
_MAX_CHUNK_BODY_CHARS = 1200

#: Constant headroom subtracted from the window for the rendered-output reserve
#: (citation block, required-attrs scaffold, remediation directive). Deliberate
#: over-reserve so the budget stays comfortably inside the wire window.
_RESERVE_TOKENS = 256


# ---------------------------------------------------------------------------
# Resolvers (parse-with-fallback)
# ---------------------------------------------------------------------------
def content_page_per_co_enabled() -> bool:
    """Read ``ED4ALL_CONTENT_PAGE_PER_CO`` each call (tests toggle inline).

    Default OFF — only the canonical truthy tokens (``1``/``true``/``yes``/
    ``on``) enable. Falsey / garbage → off (parse-with-fallback). The call site
    additionally no-ops this unless ``COURSEFORGE_TWO_PASS`` is on.
    """
    return os.environ.get(ENV_CONTENT_PAGE_PER_CO, "").strip().lower() in _TRUTHY


def content_page_per_co_uncapped() -> bool:
    """Read ``ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED`` each call (tests toggle
    inline).

    Opt-in satellite of the page-per-CO gate: when truthy AND page-per-CO is
    on, the per-week content-page count is ``max(co_count, 1)`` with NO
    topic-count cap — the operator explicitly chose ONE HTML page per CO, so
    the NEVER-INCREASE guard must not fold a CO-rich week onto shared slice
    pages. Default unset → the guard stays intact (byte-identical). Only the
    canonical truthy tokens (``1``/``true``/``yes``/``on``) enable; falsey /
    garbage → off (parse-with-fallback). Inert when
    ``ED4ALL_CONTENT_PAGE_PER_CO`` is off (the call site only consults it on
    the page-per-CO path).
    """
    return (
        os.environ.get(ENV_CONTENT_PAGE_PER_CO_UNCAPPED, "").strip().lower()
        in _TRUTHY
    )


def resolve_content_page_num_ctx(value: Optional[int] = None) -> int:
    """Resolve the authoring serving window: arg → this knob → answer knob → 4096.

    ``ED4ALL_CONTENT_PAGE_NUM_CTX`` wins; absent, fall back to the shared
    ``ED4ALL_ANSWER_NUM_CTX`` (so a box configured once for the answer path is
    honored here too); absent both, the 4096 default. Garbage / non-positive
    values fall back (a misconfigured window must never disable the budget).
    """
    if value is not None:
        try:
            ival = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_NUM_CTX
        return ival if ival > 0 else _DEFAULT_NUM_CTX
    for env_name in (ENV_CONTENT_PAGE_NUM_CTX, "ED4ALL_ANSWER_NUM_CTX"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            ival = int(raw)
        except (TypeError, ValueError):
            continue
        if ival > 0:
            return ival
    return _DEFAULT_NUM_CTX


def resolve_content_page_max_chunks(value: Optional[int] = None) -> int:
    """Resolve the per-page hard top-K ceiling: arg → env → default (5).

    Out-of-range / garbage env values fall back to the default (a misconfigured
    cap must never silently disable the cap or drop to zero). Mirrors
    :func:`lib.objectives.objective_dedup.resolve_max_chunks_per_objective`.
    """
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_CONTENT_PAGE_MAX_CHUNKS)
    if raw:
        try:
            ival = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_CHUNKS
        if ival >= 1:
            return ival
    return _DEFAULT_MAX_CHUNKS


# ---------------------------------------------------------------------------
# Gap #11 — per-block-role chunk-order diversification
# ---------------------------------------------------------------------------
def chunk_role_diversify_enabled() -> bool:
    """Read ``ED4ALL_CHUNK_ROLE_DIVERSIFY`` each call (tests toggle inline).

    Default OFF — only the canonical truthy tokens (``1``/``true``/``yes``/
    ``on``) enable. Falsey / garbage → off (parse-with-fallback).
    """
    return os.environ.get(ENV_CHUNK_ROLE_DIVERSIFY, "").strip().lower() in _TRUTHY


def _stable_offset(key: str, modulo: int) -> int:
    """Deterministic, ``PYTHONHASHSEED``-independent offset in ``[0, modulo)``.

    Uses sha1 (NOT the builtin ``hash()``, which is salted per-process) so the
    rotation is byte-stable across processes and hash seeds. ``modulo <= 0`` →
    ``0`` (caller guards the empty case).
    """
    if modulo <= 0:
        return 0
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def diversify_block_chunk_order(
    ordered_ids: Sequence[str],
    *,
    role_key: str,
    enabled: Optional[bool] = None,
) -> List[str]:
    """Role-keyed deterministic rotation of a page's ranked chunk id list.

    Root cause of the within-week anchor-example duplication (plan §6 item 11):
    every block on a page is handed the SAME page-ranked chunk universe in the
    SAME order, so the authoring model draws its anchor example from the same
    head chunk in block after block (one worked example recurring across 15/65
    week-1 blocks). This rotates the offered order by a stable per-role offset so
    co-located blocks of different roles lead with a DIFFERENT chunk — a
    chunk-universe remap that beats the re-roll treadmill.

    Contract:

    * **Permutation, never a drop/invent** — the returned list is exactly the
      input id set (de-duplicated, preserving first-seen order) rotated by an
      offset; the SAME chunks are offered, only their priority order changes. No
      anti-fabrication concern (no id added, none removed).
    * **Deterministic** — the offset is ``sha1(role_key) % len`` (see
      :func:`_stable_offset`), so it is stable across processes and
      ``PYTHONHASHSEED`` values. No randomness.
    * **Default OFF / byte-identical** — when ``enabled`` resolves falsey (env
      ``ED4ALL_CHUNK_ROLE_DIVERSIFY`` unset), a fewer-than-2 id list, or an
      empty ``role_key``, the input order is returned verbatim (as a fresh
      list). The zero-offset case also returns the input order.

    ``role_key`` is the diversification seed — the caller keys it by the block's
    teaching role (and, to spread same-role siblings, its per-page sequence
    index). Distinct roles land on distinct offsets; identical role_keys land on
    the identical rotation (deterministic).
    """
    ids: List[str] = []
    seen: set = set()
    for cid in (ordered_ids or []):
        if isinstance(cid, str) and cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)

    if enabled is None:
        enabled = chunk_role_diversify_enabled()
    if not enabled or len(ids) < 2 or not role_key:
        return ids

    offset = _stable_offset(str(role_key), len(ids))
    if offset == 0:
        return ids
    return ids[offset:] + ids[:offset]


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Upper-bound token estimate (math-safe, conservative).

    Mirrors :func:`lib.retrieval._prompts.estimate_tokens` exactly (ceil of
    char-length / 2.5) so the budget shares one divisor with the answer path.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / _CHARS_PER_TOKEN))


def page_chunk_token_budget(
    *,
    num_ctx: int,
    system_prompt_tokens: int,
    user_fixed_tokens: int = 0,
    max_tokens: int = 0,
    reserve_tokens: int = _RESERVE_TOKENS,
) -> int:
    """Tokens left for per-page grounding chunks after the fixed costs.

    ``budget = num_ctx − system_prompt − user_fixed − reserve − max_tokens``.

    The fixed terms are MEASURED by the caller (the system-prompt length is read
    DYNAMICALLY at import so a prompt edit cannot silently invalidate the budget;
    the user-fixed term is the rendered user prompt with an EMPTY chunk list).
    A negative result is returned VERBATIM (not clamped to 0) so the caller can
    detect "fixed cost alone overflows the window" and fail/escalate the page
    rather than ship a chunk that guarantees head-truncation.
    """
    return int(num_ctx) - int(system_prompt_tokens) - int(user_fixed_tokens) - int(
        reserve_tokens
    ) - int(max_tokens)


def chunk_token_weight(text: str) -> int:
    """Estimated token weight of one chunk body (truncated to the per-chunk cap).

    Truncating to ``_MAX_CHUNK_BODY_CHARS`` before estimating mirrors the
    outline tier's per-chunk render cap, so a single fat chunk cannot blow the
    budget estimate beyond what the authoring prompt would actually render.
    """
    if not text:
        return 0
    return _estimate_tokens(text[:_MAX_CHUNK_BODY_CHARS])


# ---------------------------------------------------------------------------
# Per-page union-then-top-K cap
# ---------------------------------------------------------------------------
def cap_page_chunks(
    *,
    union_ids: Sequence[str],
    cited_ids: Optional[Sequence[str]] = None,
    chunk_text_map: Optional[Dict[str, str]] = None,
    statement: str = "",
    token_budget: Optional[int] = None,
    max_chunks: Optional[int] = None,
    embed_client: Any = None,
    embed_builder: Optional[Callable[[], Any]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Union-then-top-K cap a page's grounding chunks to fit the authoring window.

    Bounds a CO page's consolidated chunk universe so a multi-section CO cannot
    overflow the per-call serving window (silent head-truncation → fabricated
    citations). The contract mirrors
    :func:`lib.objectives.objective_dedup._prune_union_chunks`:

    * **Ranking** — cited chunks FIRST (they back a real claim and must survive),
      then the remaining chunks by cosine(``statement``, chunk_text) descending.
      When no embedding client resolves (or embedding the chunk texts fails), the
      rank degrades to CITATION ORDER (cited-first, then union order) — a
      deterministic, GPU-free fallback.
    * **Cap** — keep at most ``min(max_chunks, token_budget_allows)``. The token
      budget greedily admits chunks (in rank order) while their cumulative
      :func:`chunk_token_weight` fits ``token_budget``; ``max_chunks`` is the
      hard ceiling on top of that.
    * **Always-keep-≥1** — when the union is non-empty, at least the single
      strongest chunk is kept even if it alone exceeds the token budget (the
      caller's tripwire / fail-closed handling owns the genuine-overflow case;
      this helper never returns an empty kept set for a non-empty union).
    * **Anti-fabrication** — the kept set is a strict SUBSET of ``union_ids``
      (re-projected onto union order for a stable on-disk shape); no id is ever
      invented.

    Returns ``(kept_ids, dropped_cited_ids, stats)`` where ``dropped_cited_ids``
    is the set of CITED chunks that did not survive the cap (the caller records
    them — a dropped citation is the one thing worth a warning), and ``stats``
    carries replayable counters for the decision capture.
    """
    union_list = [c for c in (union_ids or []) if isinstance(c, str) and c]
    # De-dup preserving union order.
    seen: set = set()
    union: List[str] = []
    for cid in union_list:
        if cid not in seen:
            seen.add(cid)
            union.append(cid)

    cited_set = {c for c in (cited_ids or []) if isinstance(c, str) and c}
    resolved_cap = resolve_content_page_max_chunks(max_chunks)
    text_map = chunk_text_map or {}

    stats: Dict[str, Any] = {
        "union_count": len(union),
        "cited_count": len([c for c in union if c in cited_set]),
        "cap": resolved_cap,
        "token_budget": token_budget,
        "rank_method": "citation_order",
        "kept_count": 0,
        "dropped_count": 0,
        "dropped_cited_count": 0,
    }

    if not union:
        return [], [], stats

    # --------------------------------------------------------------- ranking
    # Cited chunks always rank ahead of uncited ones (they back a real claim).
    cited_in_union = [c for c in union if c in cited_set]
    uncited_in_union = [c for c in union if c not in cited_set]

    ranked_uncited = uncited_in_union
    rank_method = "citation_order"
    if statement and uncited_in_union:
        client = embed_client
        if client is None and embed_builder is not None:
            try:
                client = embed_builder()
            except Exception:  # noqa: BLE001 — any backend failure → fallback
                client = None
        if client is not None:
            ranked_try = _cosine_rank(
                statement=statement,
                ids=uncited_in_union,
                text_map=text_map,
                client=client,
            )
            if ranked_try is not None:
                ranked_uncited = ranked_try
                rank_method = "cosine"
    stats["rank_method"] = rank_method

    ranked = cited_in_union + ranked_uncited

    # ----------------------------------------------------------------- cap
    kept: List[str] = []
    used_tokens = 0
    for cid in ranked:
        if len(kept) >= resolved_cap:
            break
        weight = chunk_token_weight(text_map.get(cid, ""))
        if (
            token_budget is not None
            and kept  # always admit the first (keep-≥1)
            and used_tokens + weight > token_budget
        ):
            continue
        kept.append(cid)
        used_tokens += weight

    # Always-keep-≥1: a non-empty union must keep at least the strongest chunk.
    if not kept and ranked:
        kept = [ranked[0]]

    # Re-project onto union order for a stable on-disk shape.
    kept_set = set(kept)
    kept_ordered = [c for c in union if c in kept_set]

    dropped_cited = [c for c in union if c in cited_set and c not in kept_set]
    stats["kept_count"] = len(kept_ordered)
    stats["dropped_count"] = len(union) - len(kept_ordered)
    stats["dropped_cited_count"] = len(dropped_cited)
    return kept_ordered, dropped_cited, stats


# ---------------------------------------------------------------------------
# W1.6 — per-page estimated learning-time (ED4ALL_PAGE_EST_MINUTES)
# ---------------------------------------------------------------------------
#
# A deterministic, GPU-free per-page reading-time estimate:
#
#     minutes = ceil(visible_words / WPM + interactions * per_interaction)
#
# Gated entirely at the emit call site (``Courseforge/scripts/generate_course.py``
# ``_wrap_page``) by ``ED4ALL_PAGE_EST_MINUTES`` (default OFF → byte-identical).
# All resolvers are parse-with-fallback so a misconfigured knob never disables
# the estimate (it falls back to the calibrated default). No model, no env
# side-effects at import.
ENV_PAGE_EST_MINUTES = "ED4ALL_PAGE_EST_MINUTES"
ENV_PAGE_WPM = "ED4ALL_PAGE_WPM"
ENV_PAGE_INTERACTION_MINUTES = "ED4ALL_PAGE_INTERACTION_MINUTES"

#: Default adult silent-reading rate (words / minute). 200 is the widely-cited
#: instructional-design median for prose learning material.
_DEFAULT_PAGE_WPM = 200

#: Default per-interaction time cost (minutes) added on top of the reading time
#: for each interactive component (self-check, activity, assessment item, …).
_DEFAULT_PER_INTERACTION_MINUTES = 1.0

_PAGE_TAG_RE = re.compile(r"<[^>]+>")
_PAGE_WS_RE = re.compile(r"\s+")

#: Interactive components that carry a ``data-cf-component`` attr (flip-card /
#: self-check / activity-card) — one interaction each.
_INTERACTION_COMPONENT_RE = re.compile(r"data-cf-component\s*=", re.IGNORECASE)
#: Interactive block markers that do NOT carry ``data-cf-component`` (so they are
#: counted separately without double-counting the component-attr elements).
_INTERACTION_CLASS_RE = re.compile(
    r'class="[^"]*\b(?:assessment-item|guided-practice|reflection-calibration'
    r"|inline-quiz)\b",
    re.IGNORECASE,
)


def page_est_minutes_enabled() -> bool:
    """Read ``ED4ALL_PAGE_EST_MINUTES`` each call (tests toggle inline).

    Default OFF — only the canonical truthy tokens enable. Falsey / garbage →
    off (parse-with-fallback).
    """
    return os.environ.get(ENV_PAGE_EST_MINUTES, "").strip().lower() in _TRUTHY


def resolve_page_wpm(value: Optional[int] = None) -> int:
    """Resolve the words-per-minute rate: arg → ``ED4ALL_PAGE_WPM`` → 200.

    Non-positive / garbage values fall back to the default (a misconfigured WPM
    must never zero-divide or disable the estimate).
    """
    if value is not None:
        try:
            ival = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_PAGE_WPM
        return ival if ival > 0 else _DEFAULT_PAGE_WPM
    raw = os.environ.get(ENV_PAGE_WPM, "").strip()
    if raw:
        try:
            ival = int(raw)
            if ival > 0:
                return ival
        except (TypeError, ValueError):
            pass
    return _DEFAULT_PAGE_WPM


def resolve_per_interaction_minutes(value: Optional[float] = None) -> float:
    """Resolve the per-interaction minute cost: arg → env → 1.0.

    Negative / garbage values fall back to the default. Zero is honored (an
    operator may explicitly disable the interaction term).
    """
    if value is not None:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return _DEFAULT_PER_INTERACTION_MINUTES
        return fval if fval >= 0 else _DEFAULT_PER_INTERACTION_MINUTES
    raw = os.environ.get(ENV_PAGE_INTERACTION_MINUTES, "").strip()
    if raw:
        try:
            fval = float(raw)
            if fval >= 0:
                return fval
        except (TypeError, ValueError):
            pass
    return _DEFAULT_PER_INTERACTION_MINUTES


def count_visible_words(html: str) -> int:
    """Count whitespace-delimited words in the visible (tag-stripped) text."""
    if not html:
        return 0
    text = _PAGE_WS_RE.sub(" ", _PAGE_TAG_RE.sub(" ", html)).strip()
    if not text:
        return 0
    return len([w for w in text.split(" ") if w])


def count_interactions(html: str) -> int:
    """Count interactive components on the page (deterministic, markup-based).

    ``data-cf-component``-bearing elements (flip-card / self-check /
    activity-card) plus the non-component interactive markers (assessment-item /
    guided-practice / reflection-calibration / inline-quiz). The two marker sets
    are disjoint so no interaction is double-counted.
    """
    if not html:
        return 0
    return len(_INTERACTION_COMPONENT_RE.findall(html)) + len(
        _INTERACTION_CLASS_RE.findall(html)
    )


def estimate_page_minutes(
    html: str,
    *,
    wpm: Optional[int] = None,
    per_interaction_minutes: Optional[float] = None,
) -> int:
    """Estimate a page's learning time in whole minutes (ceil).

    ``ceil(visible_words / WPM + interactions * per_interaction_minutes)``.
    Returns ``0`` for an empty page (no words, no interactions) so a chrome-only
    page carries no misleading estimate; any content rounds up to ≥1 minute.
    """
    resolved_wpm = resolve_page_wpm(wpm)
    resolved_per = resolve_per_interaction_minutes(per_interaction_minutes)
    words = count_visible_words(html)
    interactions = count_interactions(html)
    if words == 0 and interactions == 0:
        return 0
    total = (words / resolved_wpm) + (interactions * resolved_per)
    return max(0, int(math.ceil(total)))


def _cosine_rank(
    *,
    statement: str,
    ids: List[str],
    text_map: Dict[str, str],
    client: Any,
) -> Optional[List[str]]:
    """Rank ``ids`` by cosine(``statement``, chunk_text) descending.

    Returns ``None`` (so the caller degrades to citation order) on ANY embedding
    failure. A chunk with no resolvable text ranks last. Stable: ties keep the
    input order. Mirrors
    :func:`lib.objectives.objective_dedup._prune_union_chunks`'s ranking.
    """
    try:
        from lib.embedding._math import cosine_similarity
    except Exception:  # noqa: BLE001
        return None

    texts: List[str] = []
    has_text: List[bool] = []
    for cid in ids:
        t = text_map.get(cid) or ""
        texts.append(t if t else " ")
        has_text.append(bool(t))

    try:
        stmt_vec = client.encode_batch([statement])[0]
        chunk_vecs = client.encode_batch(texts)
    except Exception:  # noqa: BLE001 — any embed failure → citation-order
        logger.warning(
            "content_page_budget: embedding page chunk texts failed; ranking "
            "by citation order (no cosine prune)."
        )
        return None

    scored: List[Tuple[str, float]] = []
    for cid, vec, real in zip(ids, chunk_vecs, has_text):
        cos = float(cosine_similarity(stmt_vec, vec)) if real else -1.0
        scored.append((cid, cos))

    order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    return [scored[i][0] for i in order]


__all__ = [
    "ENV_CONTENT_PAGE_PER_CO",
    "ENV_CONTENT_PAGE_NUM_CTX",
    "ENV_CONTENT_PAGE_MAX_CHUNKS",
    "ENV_CHUNK_ROLE_DIVERSIFY",
    "content_page_per_co_enabled",
    "resolve_content_page_num_ctx",
    "resolve_content_page_max_chunks",
    "chunk_role_diversify_enabled",
    "diversify_block_chunk_order",
    "page_chunk_token_budget",
    "chunk_token_weight",
    "cap_page_chunks",
    "ENV_PAGE_EST_MINUTES",
    "ENV_PAGE_WPM",
    "ENV_PAGE_INTERACTION_MINUTES",
    "page_est_minutes_enabled",
    "resolve_page_wpm",
    "resolve_per_interaction_minutes",
    "count_visible_words",
    "count_interactions",
    "estimate_page_minutes",
]
