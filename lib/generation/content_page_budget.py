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

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var names + defaults
# ---------------------------------------------------------------------------
ENV_CONTENT_PAGE_PER_CO = "ED4ALL_CONTENT_PAGE_PER_CO"
ENV_CONTENT_PAGE_NUM_CTX = "ED4ALL_CONTENT_PAGE_NUM_CTX"
ENV_CONTENT_PAGE_MAX_CHUNKS = "ED4ALL_CONTENT_PAGE_MAX_CHUNKS"

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
    "content_page_per_co_enabled",
    "resolve_content_page_num_ctx",
    "resolve_content_page_max_chunks",
    "page_chunk_token_budget",
    "chunk_token_weight",
    "cap_page_chunks",
]
