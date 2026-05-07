"""DEPRECATED: re-exports from :mod:`lib.validators.pair.promotion`.

Wave W-D10 T10.1: the validator now lives in the ``pair/`` subpackage as
``promotion.py``. Re-exported here for back-compat with
``config/workflows.yaml``, external MCP clients, and existing test
imports. Will be removed in a future minor version. Use the new path
directly.
"""
import warnings

from lib.validators.pair.promotion import *  # noqa: F401,F403
from lib.validators.pair.promotion import (  # noqa: F401
    TrainingPairPromotionValidator,
    DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
    DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
    DEFAULT_MIN_RATIONALE_RICHNESS_SCORE,
)

warnings.warn(
    "lib.validators.training_pair_promotion is deprecated; "
    "use lib.validators.pair.promotion",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TrainingPairPromotionValidator",
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_MIN_RATIONALE_RICHNESS_SCORE",
]
