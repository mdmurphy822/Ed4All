"""Wave 92 — BaselineComparator tests.

Synthesise a base callable that always scores 0 and a trained
callable that always scores 1; assert mean_delta = 1.0 with a
tight bootstrap CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.baseline_compare import BaselineComparator, _bootstrap_ci  # noqa: E402


def _score_yes(resp: str) -> float:
    return 1.0 if "yes" in resp.lower() else 0.0


def test_perfect_improvement_delta_one():
    base = lambda p: "no"
    trained = lambda p: "yes"
    prompts = [(f"q_{i}", _score_yes) for i in range(20)]
    cmp = BaselineComparator(base, trained, prompts, bootstrap_iterations=200, seed=42).evaluate()
    assert cmp["mean_delta"] == 1.0
    assert cmp["base_mean"] == 0.0
    assert cmp["trained_mean"] == 1.0
    # CI tight around 1.0 — both bounds equal 1.0 because every
    # delta is exactly 1.0.
    assert cmp["ci_low"] == 1.0
    assert cmp["ci_high"] == 1.0


def test_no_improvement_delta_zero():
    base = lambda p: "yes"
    trained = lambda p: "yes"
    prompts = [(f"q_{i}", _score_yes) for i in range(20)]
    cmp = BaselineComparator(base, trained, prompts, bootstrap_iterations=200).evaluate()
    assert cmp["mean_delta"] == 0.0


def test_regression_negative_delta():
    """Trained worse than base → negative mean_delta."""
    base = lambda p: "yes"
    trained = lambda p: "no"
    prompts = [(f"q_{i}", _score_yes) for i in range(15)]
    cmp = BaselineComparator(base, trained, prompts, bootstrap_iterations=200).evaluate()
    assert cmp["mean_delta"] == -1.0


def test_bootstrap_ci_zero_pairs_returns_zeros():
    mean, lo, hi = _bootstrap_ci([], iterations=100)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_bootstrap_ci_contains_mean():
    """For a non-degenerate sample, the CI should bracket the mean."""
    pairs = [(0.1, 0.7), (0.2, 0.6), (0.3, 0.8), (0.0, 0.5), (0.5, 0.9)]
    mean, lo, hi = _bootstrap_ci(pairs, iterations=500, seed=42)
    assert lo <= mean <= hi


def test_callable_exception_logged_per_prompt():
    def boom(prompt: str) -> str:
        raise RuntimeError("oops")
    prompts = [("q1", _score_yes), ("q2", _score_yes)]
    cmp = BaselineComparator(boom, boom, prompts, bootstrap_iterations=50).evaluate()
    assert cmp["n"] == 0
    assert all("error" in p for p in cmp["per_prompt"])
