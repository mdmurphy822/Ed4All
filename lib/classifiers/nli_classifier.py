"""Singleton-load NLI classifier wrapping
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``.

Wave 2 W2.F authors the real implementation; Wave 1.7 W1.7.C ships
this stub so :class:`lib.validators.block_objective_delivery.
BlockObjectiveDeliveryValidator`'s import path is stable across the
W2.F swap.

Surface contract (frozen for W2.F):

* :meth:`NliClassifier.get_or_load` is the singleton accessor — returns
  the process-wide instance, lazy-loaded on first access. Until W2.F
  lands the real loader, returns ``None`` so the validator's
  graceful-degrade path (``BLOCK_OBJECTIVE_NLI_DEPS_MISSING`` warning,
  ``passed=True, action=None``) is exercised.
* :meth:`NliClassifier.score_pair` returns one :class:`NliScore` for a
  ``(premise, hypothesis)`` pair.
* :meth:`NliClassifier.score_batch` returns a list of scores for a list
  of pairs (for fan-out batching across many block / objective pairs).

Callers MUST handle the ``None`` return from ``get_or_load`` via the
graceful-degrade contract — the validator emits a warning-severity
GateIssue with ``passed=True, action=None`` so CPU-only dev boxes don't
fail closed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class NliScore:
    """Three-way NLI score.

    The MNLI / FEVER / ANLI heads emit logits over the
    ``(entailment, neutral, contradiction)`` triple; this dataclass-like
    container exposes the post-softmax probabilities so the
    Wave-1.7 validator can apply per-axis thresholds without coupling
    to the underlying transformer's logits API.
    """

    def __init__(
        self,
        *,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> None:
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


class NliClassifier:
    """Process-singleton NLI classifier.

    Wave 1.7 ships this STUB; Wave 2 W2.F wires the real loader against
    ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (the same model
    used as the third member of the BERT ensemble in
    :mod:`lib.classifiers.bloom_bert_ensemble`).
    """

    @classmethod
    def get_or_load(cls) -> Optional["NliClassifier"]:
        """Return the process-singleton instance.

        STUB: returns ``None`` until W2.F lands the real loader.

        Callers MUST handle the ``None`` return via the graceful-degrade
        contract: emit a warning-severity GateIssue with ``passed=True,
        action=None``. Strict-mode opt-in (``TRAINFORGE_REQUIRE_NLI=true``)
        is the W2.F follow-up — Wave 1.7 always graceful-degrades.
        """
        return None

    def score_pair(
        self,
        *,
        premise: str,
        hypothesis: str,
    ) -> NliScore:
        raise NotImplementedError(
            "W2.F lands the real implementation; the W1.7 stub returns "
            "None from get_or_load() so callers never reach this path."
        )

    def score_batch(
        self,
        *,
        pairs: List[Tuple[str, str]],
    ) -> List[NliScore]:
        raise NotImplementedError(
            "W2.F lands the real implementation; the W1.7 stub returns "
            "None from get_or_load() so callers never reach this path."
        )


__all__ = ["NliClassifier", "NliScore"]
