"""Wave 92 — Layer 4 base-vs-trained comparative delta.

Same prompt set through (a) the base model and (b) the trained
adapter; report per-metric deltas with paired-bootstrap confidence
intervals. This is the procurement-claim surface — the ``baseline_delta``
that lands in ``model_card.json::eval_scores``.

The eval is INTENTIONALLY abstract over what each "metric" measures:
the harness composes per-question outcomes (0/1 from faithfulness,
pass/fail from invariants, etc.) and the bootstrap operates on the
aggregate. This keeps the comparative-delta module reusable across
all the upstream evaluators.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _bootstrap_ci(
    paired_outcomes: Sequence[Tuple[float, float]],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap over (base_score, trained_score) pairs.

    Returns ``(mean_delta, ci_low, ci_high)`` where delta = trained - base.
    """
    if not paired_outcomes:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(paired_outcomes)
    deltas = [t - b for (b, t) in paired_outcomes]
    mean_delta = sum(deltas) / n

    samples: List[float] = []
    for _ in range(iterations):
        idxs = [rng.randrange(n) for _ in range(n)]
        s = sum(deltas[i] for i in idxs) / n
        samples.append(s)
    samples.sort()
    alpha = 1.0 - confidence
    lo_idx = int(alpha / 2.0 * iterations)
    hi_idx = int((1.0 - alpha / 2.0) * iterations) - 1
    lo_idx = max(0, min(lo_idx, iterations - 1))
    hi_idx = max(0, min(hi_idx, iterations - 1))
    return mean_delta, samples[lo_idx], samples[hi_idx]


class BaselineComparator:
    """Run the same probes through two models and compute the delta.

    Args:
        base_callable: Callable for the base/untrained model.
        trained_callable: Callable for the trained model.
        prompts: List of (prompt, score_fn) tuples. ``score_fn`` is
            a ``Callable[[str], float]`` returning a 0..1 score.
        bootstrap_iterations: Bootstrap resample count.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        base_callable: Callable[[str], str],
        trained_callable: Callable[[str], str],
        prompts: List[Tuple[str, Callable[[str], float]]],
        bootstrap_iterations: int = 1000,
        seed: int = 42,
    ) -> None:
        self.base_callable = base_callable
        self.trained_callable = trained_callable
        self.prompts = prompts
        self.bootstrap_iterations = bootstrap_iterations
        self.seed = seed

    def evaluate(self) -> Dict[str, Any]:
        per_prompt: List[Dict[str, Any]] = []
        paired: List[Tuple[float, float]] = []
        for prompt, score_fn in self.prompts:
            try:
                br = str(self.base_callable(prompt))
                tr = str(self.trained_callable(prompt))
            except Exception as exc:  # noqa: BLE001
                per_prompt.append({
                    "prompt": prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            bs = float(score_fn(br))
            ts = float(score_fn(tr))
            paired.append((bs, ts))
            per_prompt.append({
                "prompt": prompt,
                "base_score": bs,
                "trained_score": ts,
                "delta": ts - bs,
                "base_response": br,
                "trained_response": tr,
            })

        mean_delta, ci_lo, ci_hi = _bootstrap_ci(
            paired,
            iterations=self.bootstrap_iterations,
            seed=self.seed,
        )

        base_mean = (
            sum(b for b, _ in paired) / len(paired) if paired else 0.0
        )
        trained_mean = (
            sum(t for _, t in paired) / len(paired) if paired else 0.0
        )

        return {
            "n": len(paired),
            "base_mean": base_mean,
            "trained_mean": trained_mean,
            "mean_delta": mean_delta,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "per_prompt": per_prompt,
        }


__all__ = ["BaselineComparator"]
