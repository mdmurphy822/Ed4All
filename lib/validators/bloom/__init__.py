"""Bloom-axis validators.

Cluster of three validators that operate on the Bloom-level dimension:
* :class:`BloomAlignmentValidator` — surface-form Bloom verb detection.
* :class:`BloomClassifierDisagreementValidator` — BERT-ensemble vs declared.
* :class:`BloomStructuralEnforcementValidator` — DOM-shape enforcement.

Single source of truth for all three is the canonical helpers at
``lib/ontology/bloom.py`` (verbs, levels, cognitive domain).

Wave W-D10 T10.1: this is the canonical import path. The legacy flat
modules (``lib/validators/bloom_classifier_disagreement.py`` and
``lib/validators/bloom_structural_enforcement.py``) re-export from here
via shim files for one minor-version cycle. The legacy
``lib/validators/bloom.py`` does not exist as a flat shim because it
collides with this subpackage; ``from lib.validators.bloom import
BloomAlignmentValidator`` resolves through this ``__init__.py``.
"""

from lib.validators.bloom.alignment import (
    BLOOM_VERBS,
    BloomAlignmentValidator,
    detect_bloom_level,
)
from lib.validators.bloom.classifier_disagreement import (
    BloomClassifierDisagreementValidator,
    _DISAGREEMENT_CONFIDENCE_FLOOR,
    _DISPERSION_THRESHOLD,
)
from lib.validators.bloom.structural_enforcement import (
    BloomStructuralEnforcementValidator,
)

__all__ = [
    "BLOOM_VERBS",
    "BloomAlignmentValidator",
    "BloomClassifierDisagreementValidator",
    "BloomStructuralEnforcementValidator",
    "detect_bloom_level",
    "_DISAGREEMENT_CONFIDENCE_FLOOR",
    "_DISPERSION_THRESHOLD",
]
