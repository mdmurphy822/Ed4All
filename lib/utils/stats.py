"""Pure-stdlib bootstrap CI estimators.

Replaces 2 inline implementations:

- :mod:`Trainforge.eval.baseline_compare._bootstrap_ci` -- paired-bootstrap
  over (base, trained) outcome tuples for the procurement-claim
  ``baseline_delta`` headline.
- :mod:`Trainforge.scripts.harness.calibrate_pair_validation._bootstrap_percentile_ci` --
  univariate-percentile bootstrap for the pair-validation calibration
  proposal.

Both were already pure-stdlib (the consolidation roadmap Section 4.2
misread Site A as numpy-based; verification confirms it uses
``random.Random``). The canonical helper unifies the percentile
bootstrap shape; Site A keeps its paired-deltas wrapper to project
into the univariate input.

See plan ``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 3.4 for
the numpy-vs-stdlib decision (resolved to stdlib for consistency with the
calibration script's no-numpy posture).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

__all__ = ["bootstrap_percentile_ci", "percentile"]


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (0-100). Returns ``None`` on empty input.

    Lifted verbatim from
    :func:`Trainforge.scripts.harness.calibrate_pair_validation._percentile`.
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo]) * (1 - frac) + float(s[hi]) * frac


def bootstrap_percentile_ci(
    values: Sequence[float],
    pct: float,
    *,
    iterations: int = 100,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Pure-stdlib bootstrap CI for the percentile estimator at ``pct``.

    Args:
        values: Scalar observations.
        pct: Percentile to estimate (0-100).
        iterations: Bootstrap resample count (default 100, matching the
            calibration script).
        seed: RNG seed for reproducibility.
        confidence: CI width (default 0.95 -> 2.5/97.5 percentile of the
            bootstrap distribution).

    Returns:
        ``(ci_low, ci_high)`` or ``(None, None)`` on empty input or when
        every resample produces ``None`` (degenerate input).
    """
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    estimates: List[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        est = percentile(sample, pct)
        if est is not None:
            estimates.append(est)
    if not estimates:
        return (None, None)
    estimates.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2.0 * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)))
    return (float(estimates[lo_idx]), float(estimates[hi_idx]))
