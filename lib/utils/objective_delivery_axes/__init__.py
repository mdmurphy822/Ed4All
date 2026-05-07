"""Shared per-axis scoring helpers for objective-delivery validators.

W-D7 T7.3 mints this package as the single source of truth for the
three axes consumed by both
:mod:`lib.validators.block_objective_delivery` (W1.7.C) and
:mod:`lib.validators.pair.objective_delivery` (W4.C MEDIUM).

Public API:

* :func:`score_nli_axis` — single-(premise, hypothesis) NLI scoring.
* :func:`score_bloom_gap_axis` — Bloom-level gap calculation.
* :func:`score_verb_cooccurrence_axis` — action-verb cooccurrence check.
* :func:`resolve_status` — 3-valued status enum mapper.
* :data:`BLOOM_INDEX` — canonical level → index lookup.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.3.
"""
from __future__ import annotations

from lib.utils.objective_delivery_axes.bloom_gap import (
    BLOOM_INDEX,
    score_bloom_gap_axis,
)
from lib.utils.objective_delivery_axes.nli import score_nli_axis
from lib.utils.objective_delivery_axes.status import (
    STATUS_DELIVERED,
    STATUS_UNDERDELIVERED,
    STATUS_UNVERIFIABLE,
    STATUS_VERB_ONLY,
    resolve_status,
)
from lib.utils.objective_delivery_axes.verb_cooccurrence import (
    score_verb_cooccurrence_axis,
)

__all__ = [
    "BLOOM_INDEX",
    "STATUS_DELIVERED",
    "STATUS_UNDERDELIVERED",
    "STATUS_UNVERIFIABLE",
    "STATUS_VERB_ONLY",
    "resolve_status",
    "score_bloom_gap_axis",
    "score_nli_axis",
    "score_verb_cooccurrence_axis",
]
