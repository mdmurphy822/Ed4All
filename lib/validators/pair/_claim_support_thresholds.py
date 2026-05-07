"""W-D7 T7.2 — Threshold + code constants for the pair-claim-support
validator.

Extracted from :mod:`lib.validators.pair.claim_support` so the
calibration knobs (entailment / contradiction / unsupported-rate / DART
dual-source ceilings) + the canonical GateIssue codes live in one
auditable file. Re-exported through the canonical module so existing
imports keep resolving without change.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.2.
"""
from __future__ import annotations

import re

__all__ = [
    "_CODE_DART_DISAGREEMENT_RATE_HIGH",
    "_CODE_MISSING_CLAIM_SUPPORT_RATE",
    "_CODE_MISSING_INPUTS",
    "_CODE_MISSING_PER_CLAIM_SUPPORT",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_PAIRS_FILE_READ_ERROR",
    "_CONTENT_TOKEN_RE",
    "_DART_DISAGREEMENT_RATE_WARN_CEILING",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_DART_CONTRADICTION_FLOOR",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_DEFAULT_MAX_UNSUPPORTED_RATE",
    "_GATE_ISSUE_CAP",
    "_ISSUE_LIST_CAP",
    "_MIN_SENTENCE_TOKENS",
    "_OUTCOME_DART_DISAGREEMENT",
    "_PER_CLAIM_MATCH_COSINE_FLOOR",
    "_REASON_CONTRADICTED_CLAIM",
    "_REASON_UNSUPPORTED_CLAIM",
    "_SENTENCE_SPLIT_RE",
]


#: Per-claim entailment floor — sentences at or above this entailment
#: score are considered "entailed" by the cited chunk.
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70

#: Per-claim contradiction floor — sentences at or above this
#: contradiction score are considered "contradicted" by the cited chunk.
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50

#: Per-pair unsupported_claim_rate ceiling.
_DEFAULT_MAX_UNSUPPORTED_RATE: float = 0.20

#: Per-pair contradicted-claim rate ceiling.
_DEFAULT_MAX_CONTRADICTED_RATE: float = 0.05

#: Minimum content-token count for a sentence to be considered a
#: pedagogical "claim" worth scoring.
_MIN_SENTENCE_TOKENS: int = 4

#: Cap per-pair issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Cap on number of audit-trail issues for the gate-runner walk path.
_GATE_ISSUE_CAP: int = 50


# --------------------------------------------------------------------- #
# Canonical rejection reasons + GateIssue codes
# --------------------------------------------------------------------- #

_REASON_UNSUPPORTED_CLAIM: str = "unsupported_claim"
_REASON_CONTRADICTED_CLAIM: str = "contradicted_claim"

_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"
_CODE_MISSING_PER_CLAIM_SUPPORT: str = "MISSING_PER_CLAIM_SUPPORT"
_CODE_MISSING_CLAIM_SUPPORT_RATE: str = "MISSING_CLAIM_SUPPORT_RATE"
_CODE_PAIRS_FILE_READ_ERROR: str = "PAIRS_FILE_READ_ERROR"
_CODE_MISSING_INPUTS: str = "MISSING_INPUTS"


#: Sentence-split regex.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Word-token regex for sentence-length filtering.
_CONTENT_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}", re.UNICODE)


# --------------------------------------------------------------------- #
# Wave 5 W5.D — per-claim attribution helpers
# --------------------------------------------------------------------- #


#: Cosine-similarity floor for declaring that a pair-side sentence
#: "corresponds to" one of the cited chunk's structured ``key_claims``
#: entries.
_PER_CLAIM_MATCH_COSINE_FLOOR: float = 0.50


# --------------------------------------------------------------------- #
# Wave 9 TIGHT — dual-source DART cross-check
# --------------------------------------------------------------------- #


#: DART-side per-claim contradiction floor for the dual-source check.
_DEFAULT_DART_CONTRADICTION_FLOOR: float = 0.50

#: Aggregate-rate ceiling for the gate-runner walk's
#: ``DART_DISAGREEMENT_RATE_HIGH`` warning.
_DART_DISAGREEMENT_RATE_WARN_CEILING: float = 0.05

#: GateIssue code for the gate-runner walk's aggregate warning.
_CODE_DART_DISAGREEMENT_RATE_HIGH: str = "DART_DISAGREEMENT_RATE_HIGH"

#: New per-claim outcome value added by Wave 9. Used when the IMSCC NLI
#: pass scored ``"entailed"`` but the dual-source DART NLI re-score
#: contradicted.
_OUTCOME_DART_DISAGREEMENT: str = "dart_disagreement"
