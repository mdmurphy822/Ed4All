"""W-D7 T7.6 — extracted helpers for the
:class:`lib.validators.pair.promotion.TrainingPairPromotionValidator`.

Day-1 surface holds the threshold + strict-mode helper module. Per-criterion
helper extraction (placeholder / grounding / distractor / bloom / rationale)
is queued as follow-up work; the validator class body stays intact in the
canonical ``promotion.py`` module to keep the W-D7 diff bounded.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.6.
"""
from __future__ import annotations

from lib.validators._pair_promotion_stages.thresholds import (
    DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
    DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
    DEFAULT_MIN_RATIONALE_RICHNESS_SCORE,
    _FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    _FALLBACK_MIN_ANSWER_SUPPORT_SCORE,
    _FALLBACK_MIN_PROMPT_CHUNK_JACCARD,
    _RATIONALE_MIN_TOKENS_FOR_AUDIT,
    _RATIONALE_TOKEN_RE,
    _STRICT_EMBEDDINGS_ENV_VAR,
    _TRUTHY_VALUES,
    _is_strict_embeddings_mode,
)

__all__ = [
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_MIN_RATIONALE_RICHNESS_SCORE",
    "_FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "_FALLBACK_MIN_ANSWER_SUPPORT_SCORE",
    "_FALLBACK_MIN_PROMPT_CHUNK_JACCARD",
    "_RATIONALE_MIN_TOKENS_FOR_AUDIT",
    "_RATIONALE_TOKEN_RE",
    "_STRICT_EMBEDDINGS_ENV_VAR",
    "_TRUTHY_VALUES",
    "_is_strict_embeddings_mode",
]
