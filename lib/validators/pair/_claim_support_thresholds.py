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
    "_CODE_EVIDENCE_CHAR_SPAN_MISMATCH",
    "_CODE_EVIDENCE_QUOTE_MISSING",
    "_CODE_EVIDENCE_QUOTE_NOT_SUBSTRING",
    "_CODE_MISSING_CLAIM_SUPPORT_RATE",
    "_CODE_MISSING_INPUTS",
    "_CODE_MISSING_PER_CLAIM_SUPPORT",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_PAIRS_FILE_READ_ERROR",
    "_CONTENT_TOKEN_RE",
    "_DART_DISAGREEMENT_RATE_WARN_CEILING",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_CONTRADICTION_MAX_ENTAILMENT",
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

#: Per-claim entailment CEILING for the ``contradicted`` bucket: a sentence is
#: only contradicted when contradiction clears the floor above AND entailment
#: is below this. The contradiction head fires spuriously on long textbook
#: premises — measured on real accepted pairs, entailment 0.989 coexisted with
#: contradiction 0.825, 0.984 with 0.720, 0.975 with 0.516, 0.823 with 0.502.
#: Those survived only because ``entailed`` is bucketed before ``contradicted``,
#: which made the bucketing a trapdoor: a sentence carrying the same spurious
#: contradiction while dipping just under the 0.70 entailment floor fell
#: straight into ``contradicted``, whose 0.05 rate ceiling rejects a whole pair
#: on one sentence. The adjudicated casualty scored entailment 0.406 /
#: contradiction 0.549 while restating two sentences the same premise entailed
#: at 0.998 each.
#:
#: 0.25 is chosen as follows. The casualty bounds the value from ABOVE at
#: 0.406 — any higher and the evidenced error is not fixed — and 0.25 leaves
#: ~0.15 of absolute margin there, comfortably beyond the low-order variation
#: batched NLI scoring introduces. From BELOW it is bounded by wanting a
#: genuine contradiction to still register: a decisive contradiction puts
#: near-zero mass on entailment, so 0.25 (a quarter of the 3-way softmax
#: assigned to the exact OPPOSITE relation) is already well past "the model is
#: torn", not merely past numerical noise.
#:
#: This only ever moves a sentence from ``contradicted`` to ``unsupported``.
#: It cannot make anything ``entailed`` — it sits far below the untouched 0.70
#: entailment floor — and the moved sentence is still counted against the
#: unsupported ceiling, so nothing is made permissible.
_DEFAULT_CONTRADICTION_MAX_ENTAILMENT: float = 0.25

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


# --------------------------------------------------------------------- #
# Wave W-D11 T11.2: evidence-quote enforcement codes
# --------------------------------------------------------------------- #
#
# Day-1 severity: WARNING. Mirror of T11.1 on the pair surface — the
# validator's overall ``passed`` field MUST NOT flip to False solely
# because of evidence-quote findings; only the existing NLI-outcome
# (UNSUPPORTED_CLAIM / CONTRADICTED_CLAIM rates) gate the per-pair
# reject as before. These codes are emitted alongside aggregate-rate
# counters on ``GateResult.metadata`` so the T11.5 promotion-chain
# aggregator can roll them up without re-walking the per-claim shape.
# Code names match T11.1 (block surface) verbatim so T11.4
# (decision-capture rationale) + T11.5 (aggregator) can match on a
# single set across both seams.

#: Quote is set on the matched per-claim entry but is not a substring
#: of any of the cited chunks' text fields.
_CODE_EVIDENCE_QUOTE_NOT_SUBSTRING: str = "EVIDENCE_QUOTE_NOT_SUBSTRING"

#: ``evidence_char_span`` is set but ``chunk_text[start:end]`` does
#: not equal ``evidence_quote``. Indicates either a stale offset or a
#: wrong-chunk mismatch.
_CODE_EVIDENCE_CHAR_SPAN_MISMATCH: str = "EVIDENCE_CHAR_SPAN_MISMATCH"

#: Reserved for the threshold-gated "claims missing a quote" warning
#: path. Currently never emitted by the validator (day-1 the
#: ``claims_without_quote`` counter is reported via
#: ``GateResult.metadata`` only). T11.5 / T11.6 wires the threshold.
_CODE_EVIDENCE_QUOTE_MISSING: str = "EVIDENCE_QUOTE_MISSING"


#: Atomic-claim split regex. Semicolons commonly join independent factual
#: clauses in locally synthesized completions; treating the compound as one
#: hypothesis can dilute two individually entailed claims below the NLI floor.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

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
