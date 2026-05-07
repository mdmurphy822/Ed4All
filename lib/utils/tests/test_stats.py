"""Unit tests for :mod:`lib.utils.stats`."""
from __future__ import annotations

import random

from lib.utils.stats import bootstrap_percentile_ci, percentile


def test_percentile_empty() -> None:
    assert percentile([], 50) is None


def test_percentile_single() -> None:
    assert percentile([3.0], 50) == 3.0
    assert percentile([3.0], 99) == 3.0


def test_percentile_50th_known() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_percentile_25th_interpolated() -> None:
    assert percentile([1, 2, 3, 4, 5], 25) == 2.0


def test_percentile_extrema() -> None:
    assert percentile([1, 2, 3, 4, 5], 0) == 1.0
    assert percentile([1, 2, 3, 4, 5], 100) == 5.0


def test_bootstrap_empty_returns_none_pair() -> None:
    assert bootstrap_percentile_ci([], 50) == (None, None)


def test_bootstrap_seeded_deterministic() -> None:
    values = [random.Random(1).random() for _ in range(50)]
    a = bootstrap_percentile_ci(values, 50, seed=42)
    b = bootstrap_percentile_ci(values, 50, seed=42)
    assert a == b


def test_bootstrap_ci_brackets_median() -> None:
    rng = random.Random(0)
    values = [rng.random() for _ in range(200)]
    lo, hi = bootstrap_percentile_ci(values, 50, iterations=200, seed=42)
    assert lo is not None
    assert hi is not None
    # 50th-percentile of uniform [0,1] is ~0.5
    assert lo < 0.5 < hi
    assert (hi - lo) < 0.25  # tight enough for 200-sample uniform


def test_bootstrap_iteration_count_honored(monkeypatch) -> None:
    """Verify ``iterations`` controls the resample count via Random.randrange."""
    counter = {"calls": 0}

    real_randrange = random.Random.randrange

    def counting_randrange(self, *args, **kwargs):
        counter["calls"] += 1
        return real_randrange(self, *args, **kwargs)

    monkeypatch.setattr(random.Random, "randrange", counting_randrange)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    bootstrap_percentile_ci(values, 50, iterations=10, seed=0)
    # Each iteration calls randrange(n) `n` times -> 10 * 5 = 50 calls.
    assert counter["calls"] == 50
