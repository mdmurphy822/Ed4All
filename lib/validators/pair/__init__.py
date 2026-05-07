"""Pair-shape validators.

Cluster of four validators that operate on Trainforge instruction /
preference pair emit:
* :class:`PairClaimSupportValidator` — per-claim NLI entailment vs cited chunks.
* :class:`PairLearningOutcomeRefsValidator` — per-pair LO subset check.
* :class:`PairObjectiveDeliveryValidator` — tri-axis NLI / Bloom-gap / verb cooccurrence.
* :class:`TrainingPairPromotionValidator` — promotion-ledger gates.

All four are wired into ``textbook_to_course::training_synthesis``.

Wave W-D10 T10.1: this is the canonical import path. The legacy flat
modules (``pair_claim_support.py``, ``pair_lo_refs.py``,
``pair_objective_delivery.py``, ``training_pair_promotion.py``)
re-export from here via shim files for one minor-version cycle.
"""

from lib.validators.pair.claim_support import PairClaimSupportValidator
from lib.validators.pair.lo_refs import PairLearningOutcomeRefsValidator
from lib.validators.pair.objective_delivery import (
    PairObjectiveDeliveryValidator,
    _PER_PAIR_KIND_ENTAILMENT_FLOOR,
)
from lib.validators.pair.promotion import TrainingPairPromotionValidator

__all__ = [
    "PairClaimSupportValidator",
    "PairLearningOutcomeRefsValidator",
    "PairObjectiveDeliveryValidator",
    "TrainingPairPromotionValidator",
    "_PER_PAIR_KIND_ENTAILMENT_FLOOR",
]
