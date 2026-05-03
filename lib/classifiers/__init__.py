"""Statistical-tier classifiers for Phase 4 Category D validators.

Currently exposes :class:`BloomBertEnsemble` — a 3-member ensemble of
HuggingFace BERT-family models that votes on the Bloom's-taxonomy level
of a candidate text. The ensemble is consumed by
:class:`lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator`
to flag outline-tier ``objective`` / ``assessment_item`` blocks whose
declared ``bloom_level`` disagrees with the ensemble winner OR whose
ensemble dispersion (entropy of normalised votes) exceeds the
configured threshold.

Phase 4 plan reference: ``plans/phase4_statistical_tier_detailed.md``
Subtasks 24-31.

Public surface:
- :class:`BloomBertEnsemble` — model wrapper with
  ``classify(text) -> {winner_level, winner_score, dispersion, per_member}``.
- :class:`BertEnsembleDepsMissing` — raised in strict mode when the
  ``transformers`` extras are unavailable.
"""
from __future__ import annotations

from lib.classifiers.bloom_bert_ensemble import (
    BertEnsembleDepsMissing,
    BloomBertEnsemble,
)

__all__ = [
    "BertEnsembleDepsMissing",
    "BloomBertEnsemble",
]
