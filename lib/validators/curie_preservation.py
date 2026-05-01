"""Wave 135d — DEPRECATED.

Wave 130b's mean-retention metric was superseded in Wave 135c by
``lib.validators.curie_anchoring.CurieAnchoringValidator`` (binary
per-pair anchoring rate). This module remains as a backward-compat
shim — importing ``CuriePreservationValidator`` emits a
DeprecationWarning and re-exports the new validator class.

Operators using ``curie_preservation`` in custom workflows: rename
to ``curie_anchoring`` and update the threshold key from
``min_mean_retention`` to ``min_pair_anchoring_rate`` (default
0.95 vs the old 0.40).

Removal target: Wave 137. The shim survives one wave (135d→136)
and is then deleted.
"""
import warnings

from lib.validators.curie_anchoring import CurieAnchoringValidator


class CuriePreservationValidator(CurieAnchoringValidator):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CuriePreservationValidator is deprecated since Wave "
            "135c; use CurieAnchoringValidator. Removal target: "
            "Wave 137.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["CuriePreservationValidator"]
