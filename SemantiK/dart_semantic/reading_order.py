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

import os

# Mirror the gutter constant baked into the trained merge_or_split feature
# (``0.06 * page_width``) so the lifted core stays byte-identical to the
# original ``_column_ids_for_page``.
_GUTTER_GAP_FRAC = 0.06
_DEFAULT_PAGE_WIDTH = 612.0

_COLUMN_ORDER_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_column_order_mode() -> bool:
    """Return True when the column-major reading-order re-sort runs.

    Reads ``SEMANTIK_COLUMN_ORDER``. **Default OFF** (unset / blank / falsey /
    garbage -> off; byte-identical to the legacy raster ``(y0, x0)`` sort). A
    truthy value (``1``/``true``/``yes``/``on``, case-insensitive) -> on.
    Parse-with-fallback, mirroring
    :func:`dart_semantic.structure_graph.resolve_reading_order_fix` (with the
    inverted default).
    """
    raw = (os.environ.get("SEMANTIK_COLUMN_ORDER") or "").strip().lower()
    return raw in _COLUMN_ORDER_TRUTHY


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
) -> list[int]:
    """Cluster blocks into columns by their left-edge x0. Deterministic.

    Returns a ``column_index`` per input position (0 = leftmost column),
    preserving input order. Produces byte-identical output to the original
    ``merge_or_split._column_ids_for_page`` when ``seed_mask`` is None: both use
    single-linkage clustering on the ``0.06 * page_width`` gutter gap, and a
    block's column index equals the number of detected gutters strictly to its
    left.

    ``seed_mask`` (optional, float-exclusion arm): only positions flagged True
    seed the gutter detection, so text trapped INSIDE a figure/table float does
    not spawn a spurious column; every position is still ASSIGNED to a column.
    Degrades gracefully — an all-False (or empty) mask falls back to
    every-position-seeds (the byte-identical default).
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
) -> list[int]:
    """Convenience wrapper: column index per (x0, y0, x1, y1) bbox.

    When ``float_bboxes`` is supplied (figure / table rects), blocks whose
    centroid falls inside a float are excluded from SEEDING the gutter
    detection (best-effort float-exclusion) but are still assigned a column.
    Missing / empty ``float_bboxes`` -> the plain histogram reorder (the
    default behaviour, works with no figure detection).
    """
    x0s = [float(b[0]) if b is not None and len(b) >= 1 else 0.0 for b in bboxes]
    seed_mask: list[bool] | None = None
    if float_bboxes:
        seed_mask = [not _centroid_in_any(b, float_bboxes) for b in bboxes]
    return column_ids_for_x0s(x0s, page_w, seed_mask=seed_mask)


__all__ = [
    "resolve_column_order_mode",
    "column_ids_for_x0s",
    "column_ids_for_bboxes",
]
