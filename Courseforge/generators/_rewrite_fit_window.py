"""Rewrite-tier fit-window resolvers + a num_ctx-aware chunk selector.

The Courseforge REWRITE tier authors a single block's HTML from a system
prompt (the authoring CONTRACT) + a user scaffold + the block's grounding
chunks. On an 8k-window local 7B the untrimmed system prompt alone (≈7,800
tok) plus the scaffold + output budget already overflows the window BEFORE
a single grounding chunk — so Ollama silently head-truncates, dropping the
contract, and the 7B authors with source but no rules (the 7B→Sonnet parity
gap). See ``plans/finegrain/rewrite-overflow-fix-2026-06.md``.

This module carries the OFF-by-default fit-window plumbing:

- :func:`resolve_fit_window` — ``ED4ALL_REWRITE_FIT_WINDOW`` (default OFF;
  gates the system-prompt trim AND the chunk-window budget jointly).
- :func:`resolve_rewrite_num_ctx` — ``ED4ALL_REWRITE_NUM_CTX`` (default
  8192; the served-window number BOTH the budget and the tripwire read).
- :func:`resolve_truncation_tripwire` — ``ED4ALL_REWRITE_TRUNCATION_TRIPWIRE``
  (default ON; escape hatch ``=off``).
- :func:`select_chunks_under_budget` — the num_ctx-aware grounding selector
  (cited-first, then cosine rank, always-keep-≥1, drop-trailing-whole,
  per-chunk cap symmetric with the outline tier). Anti-fabrication: kept ⊆
  input universe; never invents a chunk.

Every resolver is PARSE-WITH-FALLBACK (truthy ``1/true/yes/on`` enables a
boolean; garbage/unset → the documented default), mirroring the existing
``ED4ALL_ANSWER_NUM_CTX`` / ``COURSEFORGE_REWRITE_BATCH`` resolvers.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval._prompts import estimate_tokens

# ---------------------------------------------------------------------------
# Env-var names + defaults
# ---------------------------------------------------------------------------

ENV_FIT_WINDOW = "ED4ALL_REWRITE_FIT_WINDOW"
ENV_REWRITE_NUM_CTX = "ED4ALL_REWRITE_NUM_CTX"
ENV_TRUNCATION_TRIPWIRE = "ED4ALL_REWRITE_TRUNCATION_TRIPWIRE"

#: Default served window assumed for the rewrite tier when
#: ``ED4ALL_REWRITE_NUM_CTX`` is unset. 8192 matches the qwen2.5-7b-8k
#: Modelfile the local lane ships; the design is window-parametric so a
#: 16384 Modelfile is a one-line env flip.
DEFAULT_REWRITE_NUM_CTX = 8192

#: Truthy tokens (case-insensitive) that enable a boolean flag. Mirrors
#: ``Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY``.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Falsey tokens (case-insensitive) that DISABLE a default-ON flag.
#: Mirrors the ``COURSEFORGE_REWRITE_BATCH`` falsey-frozenset pattern.
_FALSEY = frozenset({"0", "false", "no", "off"})

#: Per-chunk body truncation (characters). Symmetric with the outline
#: tier's per-chunk cap (``lib/objectives/chunk_window.py`` ships 1500 for
#: LO synthesis, the outline/rewrite tier ships 1200) — the plan pins 1200
#: for the rewrite grounding window.
MAX_CHUNK_BODY_CHARS = 1200

#: Constant reserve on top of the fixed prompt cost, mirroring the outline
#: tier's ``GRAMMAR_RESERVE_TOKENS`` + ``SAFETY_MARGIN_TOKENS`` posture: the
#: token estimate is an upper bound but rough, and the head-truncation
#: failure mode is severe, so keep a comfortable cushion.
RESERVE_TOKENS = 248


# ---------------------------------------------------------------------------
# Resolvers (parse-with-fallback)
# ---------------------------------------------------------------------------
def resolve_fit_window(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_REWRITE_FIT_WINDOW`` (default OFF).

    Truthy ``1/true/yes/on`` (case-insensitive) → on; everything else
    (unset / garbage / explicit falsey) → off. Gates BOTH the system-
    prompt trim and the chunk-window budget jointly, so OFF is byte-
    identical to today.
    """
    raw = (env or os.environ).get(ENV_FIT_WINDOW)
    if not raw:
        return False
    return str(raw).strip().lower() in _TRUTHY


def resolve_rewrite_num_ctx(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve ``ED4ALL_REWRITE_NUM_CTX`` (default 8192).

    The served-window token budget BOTH the chunk-window budget and the
    truncation tripwire read. Garbage / non-positive values fall back to
    :data:`DEFAULT_REWRITE_NUM_CTX` (a misconfigured window must never
    silently disable the budget — mirrors ``resolve_num_ctx``).
    """
    raw = (env or os.environ).get(ENV_REWRITE_NUM_CTX)
    if not raw:
        return DEFAULT_REWRITE_NUM_CTX
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_REWRITE_NUM_CTX
    return val if val > 0 else DEFAULT_REWRITE_NUM_CTX


def resolve_truncation_tripwire(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_REWRITE_TRUNCATION_TRIPWIRE`` (default ON).

    The tripwire is output-neutral (it only fail-closes on a detected
    silent head-truncation), so it ships ON by default; an explicit
    falsey ``0/false/no/off`` (case-insensitive) is the escape hatch.
    Unset / garbage / truthy → on (parse-with-fallback, mirroring the
    default-ON ``COURSEFORGE_REWRITE_BATCH``).
    """
    raw = (env or os.environ).get(ENV_TRUNCATION_TRIPWIRE)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


# ---------------------------------------------------------------------------
# Chunk-id / body helpers (duck-typed dict OR chunk-like object)
# ---------------------------------------------------------------------------
def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("chunk_id") or chunk.get("id") or "")
    return str(getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or "")


def _chunk_body(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or chunk.get("body") or "")
    return str(getattr(chunk, "text", "") or getattr(chunk, "content", "") or "")


def _render_chunk_line(chunk_id: str, body: str) -> str:
    """Mirror ``_format_source_chunks`` per-line shape (``- [id] body``)."""
    return f"- [{chunk_id}] {body}"


# ---------------------------------------------------------------------------
# Cited-chunk extraction (from the block's outline ``key_claims``)
# ---------------------------------------------------------------------------
def cited_chunk_ids_from_content(content: Any) -> List[str]:
    """Return the ordered, de-duplicated cited-chunk ids from a block.

    Reads ``content['key_claims'][i]['source_chunk_ids']`` — the per-claim
    attribution the outline tier emits. A non-dict content (rewrite-tier
    str, or a degraded payload) yields no cited ids (the selector then
    falls back to cosine / citation order).
    """
    if not isinstance(content, dict):
        return []
    claims = content.get("key_claims")
    if not isinstance(claims, list):
        return []
    ordered: List[str] = []
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        cids = claim.get("source_chunk_ids")
        if not isinstance(cids, list):
            continue
        for cid in cids:
            sid = str(cid or "")
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    return ordered


# ---------------------------------------------------------------------------
# The num_ctx-aware grounding selector
# ---------------------------------------------------------------------------
def select_chunks_under_budget(
    chunks: Sequence[Any],
    *,
    num_ctx: int,
    sys_tokens: int,
    scaffold_tokens: int,
    max_tokens: int,
    cited_chunk_ids: Optional[Sequence[str]] = None,
    rank_query: Optional[str] = None,
    embedder: Optional[Any] = None,
    reserve_tokens: int = RESERVE_TOKENS,
    per_chunk_char_cap: int = MAX_CHUNK_BODY_CHARS,
) -> Tuple[List[Any], List[str]]:
    """Select the grounding chunks that fit the rewrite tier's window.

    Returns ``(selected_chunks, dropped_cited_chunk_ids)``:

    - ``selected_chunks`` — a sublist of ``chunks`` (kept ⊆ input universe,
      anti-fabrication) re-ordered cited-first then by descending cosine
      relevance, packed greedily until the next chunk's rendered line would
      exceed the available budget. The per-chunk body is truncated to
      ``per_chunk_char_cap`` chars BEFORE the token estimate.
    - ``dropped_cited_chunk_ids`` — the CITED ids that did NOT make the cut
      (for the ``dropped_cited_chunk_ids`` audit field + a decision capture).

    Budget (mirrors ``lib/objectives/chunk_window.py``)::

        available = num_ctx - sys_tokens - scaffold_tokens
                  - reserve_tokens - max_tokens

    Always-keep-≥1: a single over-budget chunk is still emitted (the per-
    chunk cap bounds it; an empty grounding is worse than one truncated
    chunk). Drop-trailing-whole: lower-ranked chunks are dropped whole,
    never head-truncated.

    Ranking: cited chunks (in their cited order) FIRST; then the remaining
    chunks by ``cosine(chunk_body, rank_query)`` descending when an
    ``embedder`` + ``rank_query`` are supplied; embedding-absent (or a
    missing query) → the remaining chunks keep their input order (citation-
    order fallback — NOT a new fail-closed dep).
    """
    chunk_list = [c for c in chunks if _chunk_id(c) or _chunk_body(c)]
    if not chunk_list:
        return [], []

    cited_set = {str(c) for c in (cited_chunk_ids or []) if str(c)}
    cited_order = {str(c): i for i, c in enumerate(cited_chunk_ids or []) if str(c)}

    # --- Rank the non-cited remainder by cosine when possible. ----------
    non_cited = [c for c in chunk_list if _chunk_id(c) not in cited_set]
    cited = [c for c in chunk_list if _chunk_id(c) in cited_set]
    # Stable cited order: follow the cited_chunk_ids order, then any cited
    # chunk not enumerated (defensive) by input order.
    cited.sort(key=lambda c: cited_order.get(_chunk_id(c), len(cited_order)))

    ranked_non_cited = _rank_by_cosine(non_cited, rank_query, embedder)

    ordered = cited + ranked_non_cited

    # --- Compute the available token budget. ----------------------------
    available = (
        int(num_ctx)
        - max(0, int(sys_tokens))
        - max(0, int(scaffold_tokens))
        - max(0, int(reserve_tokens))
        - max(0, int(max_tokens))
    )
    # Floor at a small positive value so a tiny/negative budget still keeps
    # the always-keep-≥1 first chunk (the per-chunk cap bounds it).
    available = max(1, available)

    # --- Greedy pack, drop-trailing-whole. ------------------------------
    selected: List[Any] = []
    running = 0
    for chunk in ordered:
        cid = _chunk_id(chunk)
        body = _chunk_body(chunk)
        if len(body) > per_chunk_char_cap:
            body = body[:per_chunk_char_cap]
        line_tokens = estimate_tokens(_render_chunk_line(cid, body))
        if selected and (running + line_tokens) > available:
            # Over budget — drop this (and every later) lower-ranked chunk.
            continue
        selected.append(chunk)
        running += line_tokens

    selected_ids = {_chunk_id(c) for c in selected}
    dropped_cited = [
        cid for cid in (cited_chunk_ids or [])
        if str(cid) and str(cid) in cited_set and str(cid) not in selected_ids
    ]
    # De-dup preserving order.
    seen: set[str] = set()
    dropped_cited_unique: List[str] = []
    for cid in dropped_cited:
        s = str(cid)
        if s not in seen:
            seen.add(s)
            dropped_cited_unique.append(s)
    return selected, dropped_cited_unique


def _rank_by_cosine(
    chunks: List[Any],
    rank_query: Optional[str],
    embedder: Optional[Any],
) -> List[Any]:
    """Rank ``chunks`` by descending cosine to ``rank_query``.

    Embedding-absent (no embedder / no query) → input order (citation-
    order fallback). Any embedder error degrades to input order too — the
    selector must never fail closed on a missing/broken embedding dep.
    """
    if embedder is None or not rank_query or not rank_query.strip():
        return list(chunks)
    if not chunks:
        return []
    try:
        import numpy as np  # noqa: PLC0415

        q_vec = embedder.encode(rank_query, normalize=True)
        scored: List[Tuple[float, int, Any]] = []
        for i, chunk in enumerate(chunks):
            body = _chunk_body(chunk)
            if not body.strip():
                scored.append((-1.0, i, chunk))
                continue
            c_vec = embedder.encode(body, normalize=True)
            sim = float(np.dot(q_vec, c_vec))
            scored.append((sim, i, chunk))
        # Sort by descending similarity, stable on the original index.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored]
    except Exception:  # noqa: BLE001 — never fail closed on embedding
        return list(chunks)


__all__ = [
    "ENV_FIT_WINDOW",
    "ENV_REWRITE_NUM_CTX",
    "ENV_TRUNCATION_TRIPWIRE",
    "DEFAULT_REWRITE_NUM_CTX",
    "MAX_CHUNK_BODY_CHARS",
    "RESERVE_TOKENS",
    "resolve_fit_window",
    "resolve_rewrite_num_ctx",
    "resolve_truncation_tripwire",
    "cited_chunk_ids_from_content",
    "select_chunks_under_budget",
]
