"""Shared status resolver — exact body lifted from
:mod:`lib.validators.block_objective_delivery._resolve_status` so neither
validator owns the dependency.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.3.2.
"""
from __future__ import annotations

from typing import Optional

__all__ = [
    "STATUS_DELIVERED",
    "STATUS_UNDERDELIVERED",
    "STATUS_VERB_ONLY",
    "STATUS_UNVERIFIABLE",
    "resolve_status",
]


STATUS_DELIVERED: str = "delivered"
STATUS_UNDERDELIVERED: str = "underdelivered"
STATUS_VERB_ONLY: str = "verb_only"
STATUS_UNVERIFIABLE: str = "unverifiable"


def resolve_status(
    *,
    entailment_passed: Optional[bool],
    bloom_passed: Optional[bool],
    verb_passed: Optional[bool],
) -> str:
    """Map per-axis pass/skip flags to the canonical status enum.

    Three-valued logic per axis: ``True`` (pass), ``False`` (fail),
    ``None`` (skip — axis unverifiable due to missing inputs).

    Status semantics:

    * ``delivered`` — every running axis passed; no axis was unverifiable.
    * ``underdelivered`` — entailment OR Bloom axis missed.
    * ``verb_only`` — verb axis is the only one that passed
      (entailment / Bloom both missed or were unverifiable while verb
      passed).
    * ``unverifiable`` — any required signal was unavailable AND
      no axis fired a real miss; postmortem reader sees the gate ran
      but couldn't draw a conclusion.
    """
    axes = [entailment_passed, bloom_passed, verb_passed]
    real_misses = [a is False for a in axes]
    has_skip = any(a is None for a in axes)

    if any(real_misses):
        if (
            verb_passed is True
            and entailment_passed is not True
            and bloom_passed is not True
        ):
            return STATUS_VERB_ONLY
        return STATUS_UNDERDELIVERED

    if has_skip:
        return STATUS_UNVERIFIABLE
    return STATUS_DELIVERED
