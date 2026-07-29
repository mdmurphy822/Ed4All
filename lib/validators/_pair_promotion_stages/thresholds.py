"""Per-criterion threshold + fallback constants for the
:class:`lib.validators.pair.promotion.TrainingPairPromotionValidator`.

The calibration knobs and the strict-mode env-var helper live here in one
auditable file; the validator and the legacy ``lib.validators.pair.promotion``
module re-export every name, so existing imports keep resolving.
"""
from __future__ import annotations

import os
import re

__all__ = [
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_BLOOM_ALIGNMENT_MIN_CONFIDENCE",
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
#: Calibrated: real on-topic answers sit around p1=0.104, so an
#: intuitive-looking 0.40 floor rejects the overwhelming majority of them.
#: Re-measure via ``Trainforge/scripts/calibrate_pair_validation`` before
#: tightening.
DEFAULT_MIN_ANSWER_SUPPORT_SCORE: float = 0.10

#: Default cosine semantic-distinctness floor between ``chosen`` and
#: ``rejected`` for preference pairs.
#: Calibrated low on purpose: good distractors are deliberately plausible
#: and semantically CLOSE to the chosen answer, so real pairs measure
#: p5≈0.051 / p1≈0.02 and a 0.40 floor rejects nearly all of them.
#: Re-measure via ``Trainforge/scripts/calibrate_pair_validation`` before
#: tightening.
DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.05

#: Jaccard-fallback floor for answer-support when the [embedding]
#: extras are absent.
_FALLBACK_MIN_ANSWER_SUPPORT_SCORE: float = -1.0

#: Jaccard-fallback floor for distractor distinctness.
_FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS: float = 0.40

#: Jaccard-fallback floor for prompt↔chunk overlap.
_FALLBACK_MIN_PROMPT_CHUNK_JACCARD: float = -1.0

#: Default Jaccard floor between the prompt and the cited source chunk.
#: Measured non-separating: paraphrase prompts are INSTRUCTED to use
#: different wording from the source, so genuinely on-topic prompts can
#: score exactly 0. At 0.0 the strict ``<`` comparison in criterion 4 can
#: never fire — the reject arm is retired while ``prompt_chunk_jaccard``
#: is still stamped on every pair for audit. Re-measure via
#: ``Trainforge/scripts/calibrate_pair_validation`` before tightening.
DEFAULT_MIN_PROMPT_CHUNK_JACCARD: float = 0.0

#: Confidence floor the Bloom classifier must EXCEED for criterion 6
#: (``low_bloom_alignment``) to reject a pair.
#:
#: Set above 1.0 on purpose. Classifier scores are softmax probabilities
#: bounded by 1.0, so the strict ``>`` comparison in criterion 6 can never
#: fire and the reject arm is RETIRED — exactly the treatment criterion 4
#: got above. ``observed_bloom`` / ``bloom_alignment`` are still stamped on
#: every pair, so the signal stays available for audit and calibration.
#:
#: Why retired: the Bloom head backing this criterion is not trustworthy
#: enough to discard training data. The three-member BERT ensemble degrades
#: to the ``unknown`` sentinel (per-member dispatch is an unfinished
#: placeholder), so classification falls through to the DeBERTa zero-shot
#: head — which is itself pending replacement by a purpose-trained
#: classifier. An unreliable judge that can silently drop good pairs is
#: strictly worse than no judge, and it interacts badly with the now-
#: blocking ``min_dpo_pairs`` floor: false rejections there do not merely
#: shrink the corpus, they can fail the run outright.
#:
#: Re-arm by passing an explicit value in [0.0, 1.0] once a trustworthy
#: Bloom classifier lands.
DEFAULT_BLOOM_ALIGNMENT_MIN_CONFIDENCE: float = 1.01

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
