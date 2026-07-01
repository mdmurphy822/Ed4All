"""Column-aware reading-order helper (Fix A).

Deterministic, CPU-only, model-free. The whole pipeline historically ordered
text blocks by a raw raster ``(y0, x0)`` sort with ZERO column awareness, so a
two-column page (e.g. arXiv papers) got line-interleaved across the gutter
before any classifier saw it. This module lifts the gutter / column-clustering
math out of :func:`dart_semantic.council.merge_or_split._column_ids_for_page`
(where it was used ONLY as a per-pair BERT feature, never to reorder) into a
small dependency-free core so the FB-ordering sites can sort
**column-major** (``page, column_index, y0, x0``) instead of raster.

Gated behind ``SEMANTIK_COLUMN_ORDER`` (default OFF). When off, every caller
keeps the byte-identical raster key. On a single-column page the clustering
yields one column for every block, so the column-major key collapses to the
raster key -> byte-identical on/off.

Imports stdlib only (no SemantiK internals) so it can be imported from both
``council/`` and the extract/feature layers with no import cycle.
"""
from __future__ import annotations

import math
import os

# Mirror the gutter constant baked into the trained merge_or_split feature
# (``0.06 * page_width``) so the lifted core stays byte-identical to the
# original ``_column_ids_for_page``.
_GUTTER_GAP_FRAC = 0.06
_DEFAULT_PAGE_WIDTH = 612.0

_COLUMN_ORDER_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Single-column guard (W7.2): a genuine multi-column layout has each column
# well-populated. A spurious gutter from paragraph indents / a centered figure
# / a right-aligned page number leaves a TINY minority column — those must NOT
# spawn a column that reorders the page. A candidate split is honoured only
# when EVERY resulting column carries at least ``_MIN_COLUMN_SEEDS`` seed
# left-edges AND at least ``_MIN_COLUMN_SEED_SHARE`` of the total seeds;
# otherwise the page is treated as single-column (identity / no-op reorder).
_MIN_COLUMN_SEEDS = 2
_MIN_COLUMN_SEED_SHARE = 0.15


def resolve_deploy_profile() -> bool:
    """Return True when the bundled deploy profile (``SEMANTIK_DEPLOY_PROFILE``)
    is on.

    The deploy/render seam ships figure detection AND column reading-order OFF
    by default, so a deploy-like run silently drops every image (WCAG 1.1.1
    content loss) and mis-orders multi-column reading order. This single opt-in
    flag turns BOTH on together (W7.1 figure detection + W7.2 column order) — a
    high-value accessibility recovery that stays opt-in so current deploy
    output is byte-identical by default.

    Reads ``SEMANTIK_DEPLOY_PROFILE``. **Default OFF** (unset / blank / falsey /
    garbage -> off). Truthy (``1``/``true``/``yes``/``on``, case-insensitive) ->
    on. Parse-with-fallback, mirroring :func:`resolve_column_order_mode`. Lives
    here (stdlib-only, import-cycle-free) so both the figure-detection resolvers
    (``cascade.resolve_detect_figures`` / ``extract_shared._detect_figures_enabled``)
    and the column-order resolver can compose it.
    """
    raw = (os.environ.get("SEMANTIK_DEPLOY_PROFILE") or "").strip().lower()
    return raw in _COLUMN_ORDER_TRUTHY


def resolve_column_order_mode() -> bool:
    """Return True when the column-major reading-order re-sort runs.

    Reads ``SEMANTIK_COLUMN_ORDER`` OR the bundled ``SEMANTIK_DEPLOY_PROFILE``.
    **Default OFF** (unset / blank / falsey / garbage -> off; byte-identical to
    the legacy raster ``(y0, x0)`` sort). A truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) on EITHER flag -> on.
    Parse-with-fallback, mirroring
    :func:`dart_semantic.structure_graph.resolve_reading_order_fix` (with the
    inverted default).
    """
    raw = (os.environ.get("SEMANTIK_COLUMN_ORDER") or "").strip().lower()
    return raw in _COLUMN_ORDER_TRUTHY or resolve_deploy_profile()


def _is_genuine_multicolumn(seeds: list[float], splits: list[float]) -> bool:
    """True only when EVERY column carved by ``splits`` is well-populated.

    Buckets the gutter-detection ``seeds`` (the sorted seeding left-edges) into
    the columns the splits define and requires each bucket to clear both the
    absolute (``_MIN_COLUMN_SEEDS``) and relative (``_MIN_COLUMN_SEED_SHARE``)
    floors. A single-column page whose indents / floats produced a stray wide
    gap fails here → the caller collapses to one column (no invented gutter).
    """
    n = len(seeds)
    if n == 0 or not splits:
        return False
    counts = [0] * (len(splits) + 1)
    for x in seeds:
        c = 0
        for s in splits:
            if x > s:
                c += 1
            else:
                break
        counts[c] += 1
    min_needed = max(_MIN_COLUMN_SEEDS, math.ceil(_MIN_COLUMN_SEED_SHARE * n))
    return all(cnt >= min_needed for cnt in counts)


def _column_splits(seed_x0s: list[float], gap: float) -> list[float]:
    """Single-linkage gutter detection over SORTED left-edge x0s.

    Returns the midpoint x of every gap wider than ``gap`` (one split per
    column boundary). Mirrors the bin-walk in the original
    ``_column_ids_for_page`` exactly: a column boundary is created wherever two
    consecutive sorted x0s differ by more than the gutter threshold.
    """
    splits: list[float] = []
    prev: float | None = None
    for x in seed_x0s:
        if prev is not None and x - prev > gap:
            splits.append((prev + x) / 2.0)
        prev = x
    return splits


def column_ids_for_x0s(
    x0s: list[float],
    page_w: float,
    *,
    seed_mask: list[bool] | None = None,
    single_column_guard: bool = False,
) -> list[int]:
    """Cluster blocks into columns by their left-edge x0. Deterministic.

    Returns a ``column_index`` per input position (0 = leftmost column),
    preserving input order. Produces byte-identical output to the original
    ``merge_or_split._column_ids_for_page`` when ``seed_mask`` is None AND
    ``single_column_guard`` is False: both use single-linkage clustering on the
    ``0.06 * page_width`` gutter gap, and a block's column index equals the
    number of detected gutters strictly to its left.

    ``seed_mask`` (optional, float-exclusion arm): only positions flagged True
    seed the gutter detection, so text trapped INSIDE a figure/table float does
    not spawn a spurious column; every position is still ASSIGNED to a column.
    Degrades gracefully — an all-False (or empty) mask falls back to
    every-position-seeds (the byte-identical default).

    ``single_column_guard`` (optional, W7.2): when True, a detected gutter is
    honoured ONLY when it carves GENUINELY balanced columns
    (:func:`_is_genuine_multicolumn`) — otherwise the input is treated as
    single-column and every position gets column 0 (identity / no-op reorder).
    This is what the reading-order re-sort callers pass so a single-column doc
    with paragraph indents / a centered figure never gets an invented gutter.
    **Default False** keeps the council BERT-feature path (the ``merge_or_split``
    caller) and every direct caller byte-identical.
    """
    n = len(x0s)
    if n == 0:
        return []
    if page_w is None or page_w <= 0:
        page_w = _DEFAULT_PAGE_WIDTH
    gap = _GUTTER_GAP_FRAC * page_w

    if seed_mask is not None and len(seed_mask) == n and any(seed_mask):
        seeds = sorted(float(x0s[i]) for i in range(n) if seed_mask[i])
    else:
        seeds = sorted(float(x) for x in x0s)

    splits = _column_splits(seeds, gap)
    if not splits:
        return [0] * n
    if single_column_guard and not _is_genuine_multicolumn(seeds, splits):
        # Genuinely single-column input — the wide gap came from indents /
        # a float / a stray outlier, not a real gutter. Do not invent one.
        return [0] * n

    out: list[int] = []
    for x in x0s:
        xf = float(x)
        c = 0
        for s in splits:
            if xf > s:
                c += 1
            else:
                break
        out.append(c)
    return out


def _centroid_in_any(bbox, floats: list) -> bool:
    """True when the bbox centroid lies inside any float (figure/table) bbox.

    Defensive against short / malformed bboxes (never raises) so the
    float-exclusion arm degrades gracefully when bboxes are unavailable.
    """
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError, IndexError):
        return False
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    for fb in floats or ():
        try:
            fx0, fy0, fx1, fy1 = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
        except (TypeError, ValueError, IndexError):
            continue
        if fx0 <= cx <= fx1 and fy0 <= cy <= fy1:
            return True
    return False


def column_ids_for_bboxes(
    bboxes: list,
    page_w: float,
    *,
    float_bboxes: list | None = None,
    single_column_guard: bool = False,
) -> list[int]:
    """Convenience wrapper: column index per (x0, y0, x1, y1) bbox.

    When ``float_bboxes`` is supplied (figure / table rects), blocks whose
    centroid falls inside a float are excluded from SEEDING the gutter
    detection (best-effort float-exclusion) but are still assigned a column.
    Missing / empty ``float_bboxes`` -> the plain histogram reorder (the
    default behaviour, works with no figure detection).

    ``single_column_guard`` (W7.2) threads through to
    :func:`column_ids_for_x0s`; **default False** keeps every caller
    byte-identical.
    """
    x0s = [float(b[0]) if b is not None and len(b) >= 1 else 0.0 for b in bboxes]
    seed_mask: list[bool] | None = None
    if float_bboxes:
        seed_mask = [not _centroid_in_any(b, float_bboxes) for b in bboxes]
    return column_ids_for_x0s(
        x0s, page_w, seed_mask=seed_mask, single_column_guard=single_column_guard,
    )


__all__ = [
    "resolve_column_order_mode",
    "resolve_deploy_profile",
    "column_ids_for_x0s",
    "column_ids_for_bboxes",
]
