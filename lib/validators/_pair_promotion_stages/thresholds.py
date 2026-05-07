"""Per-criterion threshold + fallback constants for the
:class:`lib.validators.pair.promotion.TrainingPairPromotionValidator`.

W-D7 T7.6: extracted from :mod:`lib.validators.pair.promotion` so the
calibration knobs + the strict-mode env-var helper live in one
auditable file. The validator and the legacy module re-export every
name so existing imports keep resolving.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.6.
"""
from __future__ import annotations

import os
import re

__all__ = [
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_MIN_RATIONALE_RICHNESS_SCORE",
    "_FALLBACK_MIN_ANSWER_SUPPORT_SCORE",
    "_FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "_FALLBACK_MIN_PROMPT_CHUNK_JACCARD",
    "_STRICT_EMBEDDINGS_ENV_VAR",
    "_TRUTHY_VALUES",
    "_RATIONALE_TOKEN_RE",
    "_RATIONALE_MIN_TOKENS_FOR_AUDIT",
    "_is_strict_embeddings_mode",
]


#: Default cosine-similarity floor between answer text and cited chunk.
DEFAULT_MIN_ANSWER_SUPPORT_SCORE: float = 0.40

#: Default cosine semantic-distinctness floor between ``chosen`` and
#: ``rejected`` for preference pairs.
DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.40

#: Jaccard-fallback floor for answer-support when the [embedding]
#: extras are absent.
_FALLBACK_MIN_ANSWER_SUPPORT_SCORE: float = -1.0

#: Jaccard-fallback floor for distractor distinctness.
_FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.40

#: Jaccard-fallback floor for prompt↔chunk overlap.
_FALLBACK_MIN_PROMPT_CHUNK_JACCARD: float = -1.0

#: Default Jaccard floor between the prompt and the cited source chunk.
DEFAULT_MIN_PROMPT_CHUNK_JACCARD: float = 0.05

#: Default heuristic-richness floor on the rationale string.
DEFAULT_MIN_RATIONALE_RICHNESS_SCORE: float = 0.30

#: Strict-mode env var consumed by the embedding loader. Mirrors the
#: pattern in :mod:`lib.embedding.sentence_embedder.is_strict_mode`.
_STRICT_EMBEDDINGS_ENV_VAR = "TRAINFORGE_REQUIRE_EMBEDDINGS"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


def _is_strict_embeddings_mode() -> bool:
    """True when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is truthy. Read once
    per :meth:`validate_pair` invocation so test-time env mutation is
    honoured."""
    raw = os.environ.get(_STRICT_EMBEDDINGS_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


# Word-token regex for the rationale-richness heuristic. Lowercase
# alphabetic-only tokens; numbers and punctuation are dropped because
# they don't carry semantic richness in a free-form rationale string.
_RATIONALE_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)

# Cap at which we call a rationale "long enough to be evaluated". A
# rationale below this length (in content tokens) is structurally too
# short to discriminate richness; treat it as passing the gate so we
# don't false-positive on legitimately terse rationales.
_RATIONALE_MIN_TOKENS_FOR_AUDIT: int = 5
