"""Deterministic, offline BOOK PROFILER + same-model seat-tier RIGHT-SIZER for
the GLM-OCR heading judge.

The heading judge (:mod:`semantik_structure.glmocr.heading_judge`) runs in
chapter-mode: ONE judge call per book chapter, each carrying that chapter's full
content + heading skeleton + a document-schema preamble. On a single GPU the
seat's VRAM ceiling is roughly ``context × seqs ≈ constant`` (validated
operating points: 128k×8 seqs ≈ 250k×4 seqs). Most books have small chapters, so
running a big-context / low-concurrency seat wastes VRAM that could be
concurrency.

This module measures each book's TRUE per-chapter judge-window sizes — reusing
the judge's OWN segmentation + token estimator + digest-budget derivation so
"the profiler says it fits" == "the judge's digest budget actually holds it" —
then selects the SMALLEST seat CONTEXT tier whose derived digest budget holds
the book's largest chapter (freeing VRAM for more concurrent seqs). The SAME
model (e.g. Super-120B) serves every tier; only the launch args (max-model-len,
max-num-seqs) differ. A bucketer then groups a SET of books by their selected
tier.

CPU-only, deterministic, offline: no GPU, no network, no LLM call. Profiling
forces an UNBOUNDED digest budget and disables the seat-context probe (see
:func:`profile_book`) so it measures the FULL, un-truncated chapter window sizes
regardless of any live seat, and leaves ``os.environ`` byte-for-byte unchanged.

NO LLM call site → NO ``DecisionCapture`` (a pure deterministic measurement).

⚠️ EXTRAPOLATION WARNING (seat concurrency): the default tier table's ``seqs``
values are anchored on TWO empirically validated single-GPU operating points
(128k×8, 250k×4). The 32k×32 point — and ANY ``seqs`` a ``KV_FRONTIER`` knob
derives — is an EXTRAPOLATED operating point the operator MUST validate
empirically on the target card before trusting it. This code provides the knob;
it does NOT assert the card sustains a given ``context × seqs`` product.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Env-flag names (all parse-with-fallback, mirroring the root ED4ALL_* posture)
ENV_SEAT_TIERS = "SEMANTIK_HEADING_JUDGE_SEAT_TIERS"
ENV_SEAT_KV_FRONTIER = "SEMANTIK_HEADING_JUDGE_SEAT_KV_FRONTIER"
ENV_SEAT_MAX_SEQS = "SEMANTIK_HEADING_JUDGE_SEAT_MAX_SEQS"
ENV_SEAT_SELECT_SAFETY = "SEMANTIK_HEADING_JUDGE_SEAT_SELECT_SAFETY"

#: Default seat-tier table: ``name:context:seqs`` (context ascending). The two
#: validated single-GPU operating points are 128k×8 and 250k×4; 32k×32 is an
#: EXTRAPOLATED point (see the module warning) the operator must validate.
_DEFAULT_SEAT_TIERS: Tuple[Tuple[str, int, int], ...] = (
    ("super-32k", 32768, 32),
    ("super-128k", 131072, 8),
    ("super-250k", 262144, 4),
)

_DEFAULT_SEAT_MAX_SEQS = 64
_DEFAULT_SELECT_SAFETY = 1.3

# One-time malformed-token warning latches (mirror parse_seat_registry style).
_TIERS_WARNED = False


# ── Book profile ─────────────────────────────────────────────────────────────
@dataclass
class BookProfile:
    """Per-book measurement of the judge's TRUE per-chapter window sizes.

    ``source`` is an OPAQUE identifier (layout-path basename) — no book CONTENT
    is ever stored on the profile.
    """

    n_chapters: int  # judge windows that carry ≥1 pending heading
    n_pending: int
    window_tokens: List[int]  # per-window ``_estimate_tokens(win_digest)``
    max_window_tokens: int
    p50_window_tokens: int
    p95_window_tokens: int
    total_window_tokens: int
    largest_window_index: int  # index into ``window_tokens`` (−1 when empty)
    source: str = ""


def _percentile(values: List[int], pct: float) -> int:
    """Deterministic nearest-rank percentile over an int list (0 on empty)."""
    if not values:
        return 0
    s = sorted(values)
    if pct <= 0:
        return s[0]
    k = math.ceil(pct / 100.0 * len(s))
    k = min(max(k, 1), len(s))
    return s[k - 1]


@contextmanager
def _profiling_env(chapter_mode: bool, doc_schema: bool):
    """Set the measurement env then RESTORE every prior value (incl.
    unset→unset), so profiling has ZERO lasting ``os.environ`` side effects.

    - ``SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET`` → a huge value so a big chapter is
      NEVER pre-clipped by ``_chapter_content_budgeted`` (explicit budget wins
      over any seat-derived one in ``resolve_digest_budget_tokens``), capturing
      the TRUE full-chapter window size.
    - ``SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT=off`` → no ``/models`` seat probe, so
      measurement is fully offline / deterministic.
    - ``CHAPTER_MODE`` / ``DOC_SCHEMA`` → measure under the INTENDED run flags
      (the schema preamble adds tokens).
    """
    overrides = {
        "SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET": "10000000",
        "SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT": "off",
        "SEMANTIK_HEADING_JUDGE_CHAPTER_MODE": "1" if chapter_mode else "0",
        "SEMANTIK_HEADING_JUDGE_DOC_SCHEMA": "1" if doc_schema else "0",
    }
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def profile_book(
    layout_path,
    *,
    chapter_mode: bool = True,
    doc_schema: bool = True,
) -> BookProfile:
    """Measure a book's TRUE per-chapter judge-window sizes, fully offline.

    Reuses the judge's EXACT functions so the measurement matches what the judge
    will feed: ``heading_judge_standalone._load_layout_pages`` →
    ``transform.transform_document`` → ``heading_judge.build_heading_skeleton``
    (under chapter-mode, one window per chapter with pendings) →
    ``heading_judge._estimate_tokens`` over each window's exact ``win_digest``.

    Runs inside :func:`_profiling_env` so the digest budget is unbounded (no
    pre-clipping), the seat probe is disabled, and ``os.environ`` is restored.
    """
    # Imported lazily so the module stays import-light and the judge's env
    # resolvers are read fresh under the profiling overrides.
    from . import heading_judge
    from .heading_judge_standalone import _load_layout_pages
    from .transform import transform_document

    layout_path = Path(layout_path)
    with _profiling_env(chapter_mode, doc_schema):
        pages = _load_layout_pages(layout_path)
        tr = transform_document(pages)
        # Deep-copy the transform output so the profiler never mutates a shared
        # object (mirrors run_standalone's deep-copy posture).
        prov = copy.deepcopy(tr.region_provenance)
        plan = heading_judge.build_heading_skeleton(prov)
        window_tokens = [
            heading_judge._estimate_tokens(win_digest)
            for win_digest, _pending in plan.windows
        ]
        n_pending = len(plan.pending_ids)

    n_chapters = len(window_tokens)
    max_window = max(window_tokens) if window_tokens else 0
    largest_index = window_tokens.index(max_window) if window_tokens else -1
    return BookProfile(
        n_chapters=n_chapters,
        n_pending=n_pending,
        window_tokens=window_tokens,
        max_window_tokens=max_window,
        p50_window_tokens=_percentile(window_tokens, 50),
        p95_window_tokens=_percentile(window_tokens, 95),
        total_window_tokens=sum(window_tokens),
        largest_window_index=largest_index,
        source=layout_path.name,
    )


# ── Seat tiers ───────────────────────────────────────────────────────────────
@dataclass
class SeatTier:
    name: str
    context: int
    seqs: int


def _resolve_max_seqs() -> int:
    raw = (os.environ.get(ENV_SEAT_MAX_SEQS) or "").strip()
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return _DEFAULT_SEAT_MAX_SEQS


def _resolve_kv_frontier() -> Optional[int]:
    """Optional positive-int KV frontier — a single knob that RE-DERIVES every
    tier's ``seqs`` as ``max(1, min(max_seqs, frontier // context))``. Unset /
    garbage / non-positive → ``None`` (use each tier's explicit ``seqs``)."""
    raw = (os.environ.get(ENV_SEAT_KV_FRONTIER) or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def resolve_seat_tiers() -> List[SeatTier]:
    """Data-driven seat tiers from ``SEMANTIK_HEADING_JUDGE_SEAT_TIERS``
    (``name:context:seqs`` comma-separated), sorted by context ascending.

    Parse-with-fallback (mirrors ``vllm_container_lifecycle.parse_seat_registry``):
    a malformed token is SKIPPED with a one-time warning — a partly-garbage
    override still yields its valid tiers. Unset / blank / all-malformed → the
    default table. When ``SEMANTIK_HEADING_JUDGE_SEAT_KV_FRONTIER`` is a positive
    int, each tier's ``seqs`` is DERIVED as ``max(1, min(max_seqs, frontier //
    context))`` instead of the explicit ``seqs`` (``max_seqs`` from
    ``SEMANTIK_HEADING_JUDGE_SEAT_MAX_SEQS``, default 64).
    """
    global _TIERS_WARNED
    raw = os.environ.get(ENV_SEAT_TIERS)
    parsed: List[Tuple[str, int, int]] = []
    if raw and str(raw).strip():
        bad_tokens: List[str] = []
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            parts = token.split(":")
            if len(parts) != 3:
                bad_tokens.append(token)
                continue
            name, ctx_s, seqs_s = (p.strip() for p in parts)
            try:
                ctx = int(ctx_s)
                seqs = int(seqs_s)
            except (TypeError, ValueError):
                bad_tokens.append(token)
                continue
            if not name or ctx <= 0 or seqs <= 0:
                bad_tokens.append(token)
                continue
            parsed.append((name, ctx, seqs))
        if bad_tokens and not _TIERS_WARNED:
            logger.warning(
                "seat_profile: ignored %d malformed %s token(s) %r "
                "(expected 'name:context:seqs'); using %d valid tier(s).",
                len(bad_tokens), ENV_SEAT_TIERS, bad_tokens, len(parsed),
            )
            _TIERS_WARNED = True

    table = parsed if parsed else list(_DEFAULT_SEAT_TIERS)

    frontier = _resolve_kv_frontier()
    max_seqs = _resolve_max_seqs()
    tiers: List[SeatTier] = []
    for name, ctx, seqs in table:
        if frontier is not None:
            seqs = max(1, min(max_seqs, frontier // ctx))
        tiers.append(SeatTier(name=name, context=ctx, seqs=seqs))
    tiers.sort(key=lambda t: t.context)
    return tiers


def resolve_select_safety() -> float:
    """Safety multiplier on the largest-chapter window before the digest-budget
    fit check (``SEMANTIK_HEADING_JUDGE_SEAT_SELECT_SAFETY``, default 1.3).
    Parse-with-fallback: blank / non-float / NaN / ±Inf / non-positive → 1.3."""
    raw = (os.environ.get(ENV_SEAT_SELECT_SAFETY) or "").strip()
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_SELECT_SAFETY
    if f != f or f in (float("inf"), float("-inf")) or f <= 0:
        return _DEFAULT_SELECT_SAFETY
    return f


def select_seat_tier(
    profile: BookProfile,
    tiers: Optional[List[SeatTier]] = None,
    *,
    safety: Optional[float] = None,
) -> Tuple[SeatTier, bool]:
    """Pick the SMALLEST context tier whose judge digest budget holds the book's
    largest chapter (× safety); return ``(tier, overflow)``.

    Uses ``heading_judge._derive_budgets(tier.context)[0]`` — the judge's OWN
    digest-budget derivation — so "the profiler says it fits" == "the judge's
    digest budget actually holds it" (no separate formula). Ascending by
    context: the first tier whose ``digest_budget >= ceil(max_window ×
    safety)`` wins with ``overflow=False``. If NONE fit (a monster chapter
    exceeds even the biggest tier's digest budget) → ``(largest tier, True)``:
    NEVER silently pick a too-small tier — ``overflow=True`` signals the judge
    will truncate that one chapter (bounded + logged behavior, acceptable but
    surfaced).
    """
    from . import heading_judge

    if tiers is None:
        tiers = resolve_seat_tiers()
    if safety is None:
        safety = resolve_select_safety()

    ordered = sorted(tiers, key=lambda t: t.context)
    if not ordered:  # defensive: resolve_seat_tiers never returns empty
        ordered = [SeatTier(*t) for t in _DEFAULT_SEAT_TIERS]

    need = math.ceil(profile.max_window_tokens * safety)
    for tier in ordered:
        digest_budget = heading_judge._derive_budgets(tier.context)[0]
        if digest_budget >= need:
            return tier, False
    largest = ordered[-1]
    logger.warning(
        "seat_profile: book %r largest chapter (%d tok × %.2f = %d) exceeds "
        "the biggest tier %r digest budget (%d) — selecting it with OVERFLOW "
        "(the judge will truncate that one chapter).",
        profile.source, profile.max_window_tokens, safety, need, largest.name,
        heading_judge._derive_budgets(largest.context)[0],
    )
    return largest, True


def bucket_profiles(
    items: Iterable[Tuple[str, BookProfile]],
) -> "OrderedDict[str, List[str]]":
    """Group a SET of ``(book_id, BookProfile)`` by their selected seat tier.

    Returns ``tier_name -> [book_id, ...]`` as an ``OrderedDict`` ordered
    smallest→largest context, with EMPTY buckets omitted. This is the
    "bucket a set of books by required seat tier" deliverable.
    """
    tiers = resolve_seat_tiers()
    safety = resolve_select_safety()
    ordered = sorted(tiers, key=lambda t: t.context)

    grouped: Dict[str, List[str]] = {t.name: [] for t in ordered}
    for book_id, profile in items:
        tier, _overflow = select_seat_tier(profile, tiers, safety=safety)
        grouped.setdefault(tier.name, []).append(book_id)

    result: "OrderedDict[str, List[str]]" = OrderedDict()
    seen = set()
    for t in ordered:
        if grouped.get(t.name):
            result[t.name] = grouped[t.name]
            seen.add(t.name)
    # Any tier a book selected that is not in the resolved order (defensive)
    # is appended last, deterministically by name.
    for name in sorted(grouped):
        if name not in seen and grouped[name]:
            result[name] = grouped[name]
    return result
