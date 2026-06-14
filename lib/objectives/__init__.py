"""Objective-synthesis support library (W2 — claims-tier granularity + grounding).

This package holds the deterministic, hermetically-testable helpers that the
three-stage textbook synthesis surface uses to turn chapter-grained LO drafts
into GROUNDED, deduplicated chapter objectives that cite the DART chunks they
are derived from:

- :mod:`lib.objectives.chunk_window` — chapter → chunk-window partitioning sized
  to fit a 7B serving window (Pass B input prep).
- :mod:`lib.objectives.objective_grounding` — Pass C: the in-synthesis
  NLI-grounding FILTER (drops/flags candidates whose statement isn't entailed by
  the chunks they cite). This is NOT a workflow validation gate — W4 owns the
  gate-as-validator. The two share :func:`lib.retrieval.groundedness.score_groundedness`
  + the :data:`_DEFAULT_ENTAILMENT_FLOOR` re-exported here so "grounded" means the
  same thing in both places and the floor can never drift.
- :mod:`lib.objectives.objective_dedup` — Pass D: embedding-dedup of near-duplicate
  candidates → a canonical CO set.
- :mod:`lib.objectives.block_alignment` — §6: deterministic post-outline
  block↔objective alignment (populates the field the ``block_objective_delivery``
  gate checks; never fabricates a ref to dodge the gate).

The single shared seam between W2's in-synthesis filter and W4's gate-as-validator
is :data:`_DEFAULT_ENTAILMENT_FLOOR`. It is re-exported here (sourced from
:mod:`lib.validators.pair._claim_support_thresholds` via
:mod:`lib.retrieval.groundedness`) so both workstreams import the floor from ONE
place — they cannot drift to two different definitions of "grounded".
"""
from __future__ import annotations

from lib.retrieval.groundedness import _DEFAULT_ENTAILMENT_FLOOR

__all__ = [
    "_DEFAULT_ENTAILMENT_FLOOR",
]
