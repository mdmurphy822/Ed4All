"""DEPRECATED: re-exports from :mod:`lib.validators.pair.objective_delivery`.

Wave W-D10 T10.1: the validator now lives in the ``pair/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.pair.objective_delivery import *  # noqa: F401,F403
from lib.validators.pair.objective_delivery import (  # noqa: F401
    PairObjectiveDeliveryValidator,
    _load_synthesized_objectives_for_w4c,
    _PER_PAIR_KIND_ENTAILMENT_FLOOR,
    _DEFAULT_ENTAILMENT_FLOOR,
    _CONTRADICTION_FLOOR,
    _BLOOM_GAP_THRESHOLD,
    _DETERMINISTIC_TEMPLATE_PREFIXES,
    _CODE_STATEMENT_UNDERSUPPORTED,
    _CODE_BLOOM_UNDERMET,
    _CODE_VERB_ABSENT,
    _CODE_NLI_DEPS_MISSING,
    _CODE_OBSERVED_BLOOM_UNAVAILABLE,
    _CODE_OBJECTIVE_NOT_RESOLVED,
    _CODE_VERB_AXIS_UNAVAILABLE,
    _REASON_OBJECTIVE_STATEMENT_UNDERSUPPORTED,
    _REASON_OBJECTIVE_BLOOM_UNDERMET,
    _REASON_OBJECTIVE_VERB_ABSENT,
)

warnings.warn(
    "lib.validators.pair_objective_delivery is deprecated; "
    "use lib.validators.pair.objective_delivery",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PairObjectiveDeliveryValidator",
    "_load_synthesized_objectives_for_w4c",
    "_PER_PAIR_KIND_ENTAILMENT_FLOOR",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_CONTRADICTION_FLOOR",
    "_BLOOM_GAP_THRESHOLD",
    "_DETERMINISTIC_TEMPLATE_PREFIXES",
    "_CODE_STATEMENT_UNDERSUPPORTED",
    "_CODE_BLOOM_UNDERMET",
    "_CODE_VERB_ABSENT",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_OBSERVED_BLOOM_UNAVAILABLE",
    "_CODE_OBJECTIVE_NOT_RESOLVED",
    "_CODE_VERB_AXIS_UNAVAILABLE",
    "_REASON_OBJECTIVE_STATEMENT_UNDERSUPPORTED",
    "_REASON_OBJECTIVE_BLOOM_UNDERMET",
    "_REASON_OBJECTIVE_VERB_ABSENT",
]
