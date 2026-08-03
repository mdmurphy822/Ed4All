"""Statistical-tier classifier compatibility surfaces.

Currently exposes :class:`BloomBertEnsemble`, an abstaining compatibility
wrapper retained for consumers of the historical ensemble API. No reliable
Bloom classifier is provisioned, and its model-specific dispatch is
unimplemented, so the default wrapper returns ``winner_level="unknown"``
without loading registry weights. It is consumed by
:class:`lib.validators.bloom.classifier_disagreement.BloomClassifierDisagreementValidator`
which records the unavailable signal. The separate opt-in DeBERTa zero-shot
path is an NLI heuristic, not a trained Bloom classifier. The configured
MultiBERT training path is staged but remains unproven and unprovisioned.

Public surface:
- :class:`BloomBertEnsemble` — model wrapper with
  ``classify(text) -> {winner_level, winner_score, dispersion, per_member}``.
- :class:`BertEnsembleDepsMissing` — raised in strict mode when the
  compatibility surface cannot provide a usable classifier.
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
