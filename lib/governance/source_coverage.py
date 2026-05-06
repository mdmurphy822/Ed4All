"""Canonical ``source_coverage`` block helpers.

W3.H — per-stage source-coverage emit. The five sub-tasks (H1-H5)
each emit a canonical block on a different phase output:

* H1 — DART → CourseForge blocks: ``dart_chunks/manifest.json``
* H2 — CourseForge blocks → rewritten HTML: two-pass
  ``02_validation_report/report.json``
* H3 — CourseForge HTML → IMSCC: ``05_final_package/packaging_report.json``
* H4 — IMSCC → assessment items: ``assessment_data.json``
* H5 — assessment items → training pairs:
  ``training_specs/synthesis_summary.json``

W3.G consumes ALL 5 emits to build the master promotion chain, so a
single shape is load-bearing across the whole wave.

Canonical shape::

    {
      "source_coverage": {
        "consumed_count": <int>,
        "emitted_count": <int>,
        "dropped_count": <int>,
        "drop_reasons": {"<code>": <int>, ...},
        "coverage_pct": <float>
      }
    }

Invariants:

* ``coverage_pct == emitted_count / consumed_count`` (or ``0.0``
  when ``consumed_count == 0``).
* ``dropped_count == sum(drop_reasons.values())`` — when the math
  doesn't balance, ``INTERNAL_DROP_REASON_MISSING`` fires (warning)
  and the histogram is augmented with the missing-count bucket so the
  emit shape stays internally consistent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# Canonical reason key surfaced when ``drop_reasons`` total is less
# than the dropped count. Surfaces the silent-gaming class: a drop
# without a reason is ALWAYS a bug because the caller's instrumentation
# missed a code path. Mirrors the wave's anti-silent-degradation
# contract per plan §W3.H.
INTERNAL_DROP_REASON_MISSING = "internal_drop_reason_missing"


def build_source_coverage(
    *,
    consumed_count: int,
    emitted_count: int,
    drop_reasons: Optional[Mapping[str, int]] = None,
    dropped_count: Optional[int] = None,
    label: str = "source_coverage",
) -> Dict[str, Any]:
    """Build a canonical ``source_coverage`` block.

    Parameters
    ----------
    consumed_count
        Number of upstream artifacts (chunks / blocks / pages /
        items / pairs) the stage consumed.
    emitted_count
        Number successfully emitted downstream.
    drop_reasons
        Histogram of ``{reason_code: count}`` for the artifacts that
        were dropped. Empty / None is treated as "no drops". When the
        sum of values is less than the inferred ``dropped_count``, the
        ``INTERNAL_DROP_REASON_MISSING`` bucket is added so the
        invariant ``dropped_count == sum(drop_reasons.values())``
        holds in the emitted shape (and a warning is logged).
    dropped_count
        Optional explicit dropped count. When omitted, derived as
        ``max(0, consumed_count - emitted_count)``. Pass an explicit
        value when consumed/emitted are not in 1:1 correspondence
        (e.g. H4 where one assessment fans out to many items).
    label
        Logging label so an operator scanning the warning stream can
        tell which emit site fired the silent-gaming check.

    Returns
    -------
    dict
        ``{"consumed_count": ..., "emitted_count": ..., ...}`` —
        ready to wrap under a ``"source_coverage"`` key in the
        caller's emit shape.
    """
    consumed = max(0, int(consumed_count))
    emitted = max(0, int(emitted_count))
    if dropped_count is None:
        dropped = max(0, consumed - emitted)
    else:
        dropped = max(0, int(dropped_count))

    histogram: Dict[str, int] = {}
    if drop_reasons:
        for code, count in drop_reasons.items():
            try:
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if count_int <= 0:
                continue
            key = str(code)
            histogram[key] = histogram.get(key, 0) + count_int

    histogram_total = sum(histogram.values())
    if dropped > histogram_total:
        # Silent-gaming check: more drops happened than the caller
        # could attribute. Surface the missing count via the canonical
        # internal bucket so the invariant holds AND the operator gets
        # a warning to investigate the unattributed code path.
        missing = dropped - histogram_total
        histogram[INTERNAL_DROP_REASON_MISSING] = (
            histogram.get(INTERNAL_DROP_REASON_MISSING, 0) + missing
        )
        logger.warning(
            "%s: %d dropped artifact(s) lacked a drop_reasons code; "
            "augmented histogram with %s=%d so the dropped_count == "
            "sum(drop_reasons.values()) invariant holds. "
            "Operator action: locate the unattributed drop path and "
            "instrument it with a reason code.",
            label,
            missing,
            INTERNAL_DROP_REASON_MISSING,
            missing,
        )
    elif dropped < histogram_total:
        # Histogram over-attributed. Round the dropped count up so the
        # invariant holds; this is a bookkeeping bug on the caller's
        # side (probably double-counting), but we'd rather emit a
        # consistent shape than crash the phase.
        logger.warning(
            "%s: drop_reasons total (%d) exceeds dropped_count (%d); "
            "raising dropped_count to match. Operator action: locate "
            "the double-counted drop path.",
            label,
            histogram_total,
            dropped,
        )
        dropped = histogram_total

    coverage_pct = (emitted / consumed) if consumed > 0 else 0.0

    # Canonical key order — emit-stable across all 5 sub-tasks so a
    # downstream byte-diff doesn't trip on insertion-order variance.
    return {
        "consumed_count": consumed,
        "emitted_count": emitted,
        "dropped_count": dropped,
        "drop_reasons": dict(sorted(histogram.items())),
        "coverage_pct": round(coverage_pct, 6),
    }


__all__ = [
    "INTERNAL_DROP_REASON_MISSING",
    "build_source_coverage",
]
