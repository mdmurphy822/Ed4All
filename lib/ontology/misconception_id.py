"""Canonical misconception ID helper (Wave 99).

Single source of truth for the ``mc_<16-hex>`` content-hash misconception ID
used across Trainforge:

* ``Trainforge/process_course.py::_build_misconceptions_for_graph``
* ``Trainforge/generators/preference_factory.py::_misconception_id``
* ``Trainforge/pedagogy_graph_builder.py::_mc_id``

Wave 99 unifies the three previously-redundant inline implementations behind
this helper to prevent the drift class that motivated Wave 95
(``preference_factory`` rebuild) and Wave 97 (``pedagogy_graph.json``
one-shot rebuild for ``rdf-shacl-551-2``).

Algorithm (Wave 69 / 72 lineage)::

    statement  = (statement or "").strip()
    correction = (correction or "").strip()
    bloom      = (bloom_level or "").strip().lower()
    seed = f"{statement}|{correction}|{bloom}"  if bloom else f"{statement}|{correction}"
    return "mc_" + sha256(seed).hexdigest()[:16]

The two-segment seed for bloom-less misconceptions keeps pre-Wave-60 /
legacy-corpus IDs stable with the pre-Wave-69 hash. Outer whitespace is
normalised but inner whitespace is preserved, so cosmetic edits do not
churn IDs but real text edits do.

Schema: ``schemas/knowledge/misconception.schema.json``
       ``id`` pattern: ``^mc_[0-9a-f]{16}$``
"""

from __future__ import annotations

import hashlib
from typing import Optional


def canonical_mc_id(
    statement: str,
    correction: str,
    bloom_level: Optional[str] = None,
) -> str:
    """Compute the canonical misconception content-hash ID.

    Parameters
    ----------
    statement
        The misconception text (chunk-level ``misconception`` field). Outer
        whitespace stripped; inner whitespace preserved.
    correction
        The correction text (chunk-level ``correction`` field). Outer
        whitespace stripped. Empty string is the canonical fallback when a
        chunk omits a correction.
    bloom_level
        Optional Bloom level (chunk-level ``bloom_level``). When falsy
        (``None`` / empty / whitespace-only), the seed degrades to the
        2-segment form to preserve pre-Wave-60 corpus IDs.

    Returns
    -------
    str
        Misconception ID in the form ``mc_<16 lowercase hex>``.
    """

    s = (statement or "").strip()
    c = (correction or "").strip()
    b = (bloom_level or "").strip().lower()
    if b:
        seed = f"{s}|{c}|{b}"
    else:
        seed = f"{s}|{c}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"mc_{digest}"
