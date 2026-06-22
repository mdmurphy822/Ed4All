"""Two-tier gate package (Stages 7, 8, 10, 11 of the v2 pipeline).

Replaces v1's single end-of-pipeline axe pass with:

    Stage 7  : per-region hard gate     (axe + html5validator + text-preserve)
    Stage 8  : per-region soft reranker (cross-encoder over surviving K)
    Stage 10 : per-document hard gate   (full-doc axe + heading tree + landmarks)
    Stage 11 : per-document soft reranker (composite over assembled candidates)

Phase 0 lands type contracts and stub modules. Implementations land in
Phase 7 / 8 / 10.
"""

from .hard_document import check_document, gate_document
from .hard_region import check_region, gate_per_region
from .types import CheckOutcome, GateCheck, GateResult


def __getattr__(name: str):
    """Lazy re-export of Stage 8 symbols from the ``soft_reranker`` package.

    We avoid eagerly importing :mod:`.soft_region` at gates package init
    time because :mod:`dart_semantic.soft_reranker.region` imports
    :mod:`dart_semantic.gates.text_preserve` (a sibling submodule) —
    which in turn engages this same ``__init__``. Forming the cycle
    eagerly would deadlock the partial import. Lazy re-export breaks
    the cycle: by the time a caller asks for these symbols, the rest
    of the package is fully initialised.
    """
    if name in {"DEFAULT_WEIGHTS", "RankedCandidate", "rerank_per_region"}:
        from .soft_region import (
            DEFAULT_WEIGHTS,
            RankedCandidate,
            rerank_per_region,
        )
        return {
            "DEFAULT_WEIGHTS": DEFAULT_WEIGHTS,
            "RankedCandidate": RankedCandidate,
            "rerank_per_region": rerank_per_region,
        }[name]
    raise AttributeError(f"module 'dart_semantic.gates' has no attribute {name!r}")


__all__ = [
    "DEFAULT_WEIGHTS",
    "CheckOutcome",
    "GateCheck",
    "GateResult",
    "RankedCandidate",
    "check_document",
    "check_region",
    "gate_document",
    "gate_per_region",
    "rerank_per_region",
]
