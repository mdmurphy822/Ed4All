"""Stage-5 geometric region-order derivation (ITEM5 prong 2).

Deterministic, CPU-only, model-free. The single ordering owner at the exit of
:func:`dart_semantic.structure_graph.build_structure_graph`. Replaces blind
inheritance of the frozen Stage-2 FB-index order with a geometry-grounded
per-page order ``(page -> column -> y0 -> x0 -> min_fb)`` derived from ITEM2's
per-page Region bboxes (with an FB-derived fallback), plus a normalized
inversion-distance divergence tripwire (geometric order vs FB-index order) that
ships in SHADOW mode first.

One flag, three modes (``SEMANTIK_REGION_ORDER``):
  * ``fb``  — the legacy min-FB stable sort, byte-identical to the historic
    default-ON ``SEMANTIK_READING_ORDER_FIX`` behavior (SUBSUMES it as one mode).
    The divergence is still MEASURED (shadow) but the order is not changed.
  * ``off`` — the legacy segregated pass-order list (byte-identical to the
    explicit ``SEMANTIK_READING_ORDER_FIX=0`` revert lever).
  * ``geom`` — the geometric derivation (the target default, owner-gated flip).

Imports only :mod:`reading_order` helpers (stdlib-backed) → no import cycle
(``structure_graph`` imports ``region_order``, never the reverse). NO LLM call
site → NO DecisionCapture obligation (deterministic pass, flagged loudly per
the campaign rules).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .reading_order import _DEFAULT_PAGE_WIDTH, column_ids_for_bboxes

logger = logging.getLogger(__name__)

_REGION_ORDER_VALUES = frozenset({"geom", "fb", "off"})
_READING_ORDER_FALSEY = frozenset({"0", "false", "no", "off"})

# Phase-B (shadow) default. Phase C flips this literal to "geom" once the
# calibration evidence gate (D5a) passes + owner sign-off.
_DEFAULT_MODE = "fb"

# Divergence constants — calibrated in Phase C (D6.7).
_GEOM_ORDER_PAGE_CEILING = 0.45  # TODO(calibration Phase C): page fb-fallback ceiling
_WARN_PAGE_DIVERGENCE = 0.10     # TODO(calibration Phase C): per-page WARN floor
_WARN_DOC_DIVERGENCE = 0.02      # TODO(calibration Phase C): doc WARN floor

# Region kinds excluded from SEEDING the gutter detection (a figure/table/math
# float must not seed a phantom column) — same mechanism as _merge_page.
_FLOAT_KINDS = frozenset({"figure", "table", "math"})

_legacy_deprecation_logged = False


def resolve_region_order_mode() -> str:
    """Resolve the region-order mode (``geom`` | ``fb`` | ``off``).

    ``SEMANTIK_REGION_ORDER`` (lower/strip) wins when it names a mode. When it
    is UNSET / blank, the legacy ``SEMANTIK_READING_ORDER_FIX`` explicitly-falsey
    revert lever maps to ``off`` (one-shot deprecation log). Any other value —
    a garbage ``SEMANTIK_REGION_ORDER`` OR an unset/truthy legacy env — resolves
    to the module default (parse-with-fallback). Phase B default: ``fb``.
    """
    raw = (os.environ.get("SEMANTIK_REGION_ORDER") or "").strip().lower()
    if raw in _REGION_ORDER_VALUES:
        return raw
    if raw == "":
        # New flag unset/blank → honor the legacy alias's explicit-falsey revert.
        legacy = (os.environ.get("SEMANTIK_READING_ORDER_FIX") or "").strip().lower()
        if legacy in _READING_ORDER_FALSEY:
            global _legacy_deprecation_logged
            if not _legacy_deprecation_logged:
                logger.warning(
                    "SEMANTIK_READING_ORDER_FIX is deprecated and folded into "
                    "SEMANTIK_REGION_ORDER; its explicit-falsey value maps to "
                    "SEMANTIK_REGION_ORDER=off."
                )
                _legacy_deprecation_logged = True
            return "off"
    return _DEFAULT_MODE


def _kind_name(region: Any) -> str:
    """Best-effort lowercase kind string (RegionKind is a Literal str)."""
    kind = getattr(region, "kind", None)
    val = getattr(kind, "value", kind)
    return str(val).lower() if val is not None else ""


def _region_geom(
    region: Any, feature_blocks: list
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Return ``(page, (x0, y0, x1, y1))`` for the region's FIRST page, or None.

    Prefers ITEM2's materialized ``page_bboxes`` first-page entry (already
    sorted ascending by page); falls back to deriving from member FBs (min page
    among ``feature_blocks[i].raw.page``, bbox = union of member FB bboxes on
    that page — mirroring ``structure_graph.compute_region_page_bboxes``).
    Returns ``None`` only when neither source yields geometry.
    """
    pb = getattr(region, "page_bboxes", None)
    if pb:
        try:
            page, bbox = pb[0]
            x0, y0, x1, y1 = (float(v) for v in bbox)
            return (int(page), (x0, y0, x1, y1))
        except (TypeError, ValueError, IndexError):
            pass  # fall through to FB derivation

    by_page: dict[int, tuple[float, float, float, float]] = {}
    for idx in getattr(region, "feature_block_indices", ()) or ():
        try:
            fb = feature_blocks[idx]
        except (IndexError, TypeError):
            continue
        raw = getattr(fb, "raw", None)
        page = getattr(raw, "page", None)
        bbox = getattr(raw, "bbox", None)
        if page is None or bbox is None:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        cur = by_page.get(int(page))
        if cur is None:
            by_page[int(page)] = (x0, y0, x1, y1)
        else:
            by_page[int(page)] = (
                min(cur[0], x0), min(cur[1], y0), max(cur[2], x1), max(cur[3], y1),
            )
    if not by_page:
        return None
    p = min(by_page)
    return (p, by_page[p])


def _page_width_for(page: int, feature_blocks: list) -> float:
    """Page width of any member FB on ``page`` (raw.page_width), else default.

    ``raw.page_width`` is correct in both point and pixel spaces (blocks_from_
    shared normalizes), so it matches the bbox coordinate space.
    """
    for fb in feature_blocks:
        raw = getattr(fb, "raw", None)
        if raw is None:
            continue
        if getattr(raw, "page", None) == page:
            pw = getattr(raw, "page_width", None)
            if pw:
                try:
                    return float(pw)
                except (TypeError, ValueError):
                    continue
    return _DEFAULT_PAGE_WIDTH


def _inversion_distance(perm: list[int]) -> float:
    """Normalized Kendall-tau distance (inversions / C(n,2)); 0.0 for n<2."""
    n = len(perm)
    if n < 2:
        return 0.0
    inv = 0
    for i in range(n):
        pi = perm[i]
        for j in range(i + 1, n):
            if pi > perm[j]:
                inv += 1
    return inv / (n * (n - 1) / 2.0)


def _column_for_regions(
    page_regions: list, geoms: dict, page_w: float
) -> dict:
    """Column index per region on a page, with the full-width straddle → col 0.

    Reuses ``reading_order.column_ids_for_bboxes`` (single_column_guard=True,
    figure/table/math floats excluded from seeding). A region whose x-span
    STRADDLES an interior column left-edge (a genuine full-width heading /
    masthead) is pinned to column 0 (mirroring the extraction rule).
    """
    bboxes = [geoms[id(r)][1] for r in page_regions]
    float_bboxes = [
        geoms[id(r)][1] for r in page_regions if _kind_name(r) in _FLOAT_KINDS
    ]
    cols = column_ids_for_bboxes(
        bboxes,
        page_w,
        float_bboxes=float_bboxes or None,
        single_column_guard=True,
    )
    # Derive per-column left edges (min x0 in each detected column) so we can
    # detect a region that straddles an interior gutter.
    edges: dict[int, float] = {}
    for r, c in zip(page_regions, cols):
        x0 = geoms[id(r)][1][0]
        if c not in edges or x0 < edges[c]:
            edges[c] = x0
    edge_list = [edges[c] for c in sorted(edges)]

    out: dict = {}
    for r, c in zip(page_regions, cols):
        x0 = geoms[id(r)][1][0]
        x1 = geoms[id(r)][1][2]
        straddles = any(
            x0 < edge_list[j] < x1 for j in range(1, len(edge_list))
        )
        out[id(r)] = 0 if straddles else c
    return out


def apply_region_order(
    regions: list, feature_blocks: list, *, mode: str, audit: dict | None = None
) -> list:
    """Order the region list geometrically (``geom``) or by min-FB (``fb``).

    ``mode`` is one of ``geom`` / ``fb`` (``off`` is handled by the caller,
    which never calls this). ``fb`` is byte-identical to the historic
    ``sorted(regions, key=min(feature_block_indices))`` stable sort. ``geom``
    derives per-page column order and applies it, with a per-page divergence
    ceiling falling back to fb order. The divergence tripwire is ALWAYS measured
    (shadow in ``fb`` mode). Conserves the region multiset (permutation-only);
    ``audit`` (mutated in place) records the D6.6 shape.
    """
    n_fb = len(feature_blocks)

    def min_fb(r: Any) -> int:
        fbi = getattr(r, "feature_block_indices", None)
        return min(fbi) if fbi else n_fb

    # fb order = the byte-identical stable min-FB sort (the baseline / fallback).
    fb_ordered = sorted(regions, key=min_fb)

    # Partition geometry-bearing vs bbox-less.
    geoms: dict = {}
    for r in regions:
        g = _region_geom(r, feature_blocks)
        if g is not None:
            geoms[id(r)] = g
    geom_bearing = [r for r in fb_ordered if id(r) in geoms]
    n_bboxless = len(regions) - len(geom_bearing)

    # Group geometry-bearing regions by page (fb order preserved within a page).
    pages: dict[int, list] = {}
    for r in geom_bearing:
        pages.setdefault(geoms[id(r)][0], []).append(r)

    audit_pages: list[dict] = []
    warnings: list[str] = []
    weighted_div_sum = 0.0
    total_n = 0
    geom_order: list = []  # geometry-bearing regions in final (geom) order

    for page in sorted(pages):
        page_regions = pages[page]  # fb order
        page_w = _page_width_for(page, feature_blocks)
        region_col = _column_for_regions(page_regions, geoms, page_w)
        n_columns = len(set(region_col.values())) if region_col else 1

        geom_sorted = sorted(
            page_regions,
            key=lambda r: (
                region_col[id(r)],
                geoms[id(r)][1][1],  # y0
                geoms[id(r)][1][0],  # x0
                min_fb(r),
            ),
        )
        fb_rank = {id(r): i for i, r in enumerate(page_regions)}
        perm = [fb_rank[id(r)] for r in geom_sorted]
        divergence = _inversion_distance(perm)

        fallback = False
        chosen = page_regions  # default: keep fb order (fb mode / ceiling fall)
        if mode == "geom":
            if divergence > _GEOM_ORDER_PAGE_CEILING:
                fallback = True  # geometry-pathological page → keep fb order
            else:
                chosen = geom_sorted
        geom_order.extend(chosen)

        n = len(page_regions)
        weighted_div_sum += divergence * n
        total_n += n
        if divergence > _WARN_PAGE_DIVERGENCE:
            warnings.append(f"page {page}: divergence {divergence:.3f}")
        audit_pages.append(
            {
                "page": page,
                "n_regions": n,
                "n_columns": n_columns,
                "divergence": round(divergence, 4),
                "fallback": fallback,
            }
        )

    doc_divergence = (weighted_div_sum / total_n) if total_n else 0.0

    # Re-insert bbox-less regions immediately after the geometry-bearing region
    # that precedes them in FB-index order (front if none) — preserving their
    # relative FB-index order exactly (D6.4).
    after: dict[int, list] = {}
    front: list = []
    last_geom: Any = None
    for r in fb_ordered:
        if id(r) in geoms:
            last_geom = r
        elif last_geom is None:
            front.append(r)
        else:
            after.setdefault(id(last_geom), []).append(r)

    result: list = list(front)
    for r in geom_order:
        result.append(r)
        for b in after.get(id(r), ()):  # noqa: B007
            result.append(b)

    # D6.5 conservation: permute-only (never drop/add). Reuse the identity-set
    # assert pattern from the legacy reading-order sort.
    if {id(r) for r in result} != {id(r) for r in regions} or len(result) != len(regions):
        raise AssertionError(
            "region-order pass dropped/added a region (conservation violated)"
        )

    if mode == "geom" and (
        doc_divergence > _WARN_DOC_DIVERGENCE or warnings
    ):
        logger.warning(
            "SEMANTIK_REGION_ORDER=geom divergence: doc=%.3f pages=[%s]",
            doc_divergence,
            "; ".join(warnings) or "-",
        )

    if audit is not None:
        audit.clear()
        audit.update(
            {
                "mode": mode,
                "applied": mode == "geom",
                "doc_divergence": round(doc_divergence, 4),
                "pages": audit_pages,
                "bboxless_regions": n_bboxless,
                "warnings": warnings,
            }
        )

    return result


__all__ = [
    "resolve_region_order_mode",
    "apply_region_order",
    "_region_geom",
    "_inversion_distance",
]
