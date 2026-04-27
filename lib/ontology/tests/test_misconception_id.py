"""Unit tests for ``lib.ontology.misconception_id.canonical_mc_id``.

Wave 99 extracted the canonical mc-ID hash to one helper; this module
locks the algorithm + fallback semantics so future call sites stay
byte-equivalent. The three current call sites are:

* ``Trainforge/pedagogy_graph_builder.py::_mc_id``
* ``Trainforge/process_course.py::_build_misconceptions_for_graph``
* ``Trainforge/generators/preference_factory.py::_misconception_id``

Schema: ``schemas/knowledge/misconception.schema.json``
       (``id`` pattern ``^mc_[0-9a-f]{16}$``).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.misconception_id import canonical_mc_id  # noqa: E402

_ID_PATTERN = re.compile(r"^mc_[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# Format conformance
# ---------------------------------------------------------------------------


def test_id_matches_schema_pattern():
    mc_id = canonical_mc_id("Wrong belief.", "Right belief.", "understand")
    assert _ID_PATTERN.match(mc_id), mc_id


def test_id_matches_schema_pattern_no_bloom():
    mc_id = canonical_mc_id("Wrong belief.", "Right belief.", "")
    assert _ID_PATTERN.match(mc_id), mc_id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_repeat():
    a = canonical_mc_id("statement", "correction", "remember")
    b = canonical_mc_id("statement", "correction", "remember")
    assert a == b


def test_deterministic_known_hash():
    """Pin the canonical algorithm against a hand-computed reference value.

    If this fails it means the hash recipe changed. Bumping it requires a
    cross-corpus rehash story (see Wave 95 / Wave 97 commits).
    """

    seed = "wrong belief|right correction|understand"
    expected = "mc_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert canonical_mc_id("wrong belief", "right correction", "understand") == expected


def test_deterministic_known_hash_no_bloom():
    seed = "wrong belief|right correction"
    expected = "mc_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert canonical_mc_id("wrong belief", "right correction", "") == expected


# ---------------------------------------------------------------------------
# Fallback semantics for missing fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bloom_input",
    [None, "", "  ", "\n\t"],
)
def test_falsy_bloom_takes_two_segment_path(bloom_input):
    """None/empty/whitespace-only bloom falls back to 2-segment seed."""

    with_falsy = canonical_mc_id("S", "C", bloom_input)
    no_arg = canonical_mc_id("S", "C")
    explicit_empty = canonical_mc_id("S", "C", "")
    assert with_falsy == no_arg == explicit_empty


def test_none_correction_treated_as_empty():
    a = canonical_mc_id("S", None, "remember")  # type: ignore[arg-type]
    b = canonical_mc_id("S", "", "remember")
    assert a == b


def test_none_statement_treated_as_empty():
    a = canonical_mc_id(None, "C", "remember")  # type: ignore[arg-type]
    b = canonical_mc_id("", "C", "remember")
    assert a == b


# ---------------------------------------------------------------------------
# Whitespace + casing semantics
# ---------------------------------------------------------------------------


def test_outer_whitespace_normalised():
    a = canonical_mc_id("  S  ", "  C  ", "  understand  ")
    b = canonical_mc_id("S", "C", "understand")
    assert a == b


def test_inner_whitespace_preserved():
    """Inner whitespace IS hashed — cosmetic edits cause new IDs."""

    a = canonical_mc_id("S one", "C", "understand")
    b = canonical_mc_id("S  one", "C", "understand")  # double-space inside
    assert a != b


def test_bloom_case_normalised():
    a = canonical_mc_id("S", "C", "Understand")
    b = canonical_mc_id("S", "C", "UNDERSTAND")
    c = canonical_mc_id("S", "C", "understand")
    assert a == b == c


def test_statement_case_preserved():
    """Statement casing is part of the hash — capitalization is meaningful."""

    a = canonical_mc_id("Triples", "C", "understand")
    b = canonical_mc_id("triples", "C", "understand")
    assert a != b


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def test_distinct_statement_produces_distinct_ids():
    a = canonical_mc_id("statement A", "C", "")
    b = canonical_mc_id("statement B", "C", "")
    assert a != b


def test_distinct_correction_produces_distinct_ids():
    a = canonical_mc_id("S", "correction A", "")
    b = canonical_mc_id("S", "correction B", "")
    assert a != b


def test_distinct_bloom_produces_distinct_ids():
    a = canonical_mc_id("S", "C", "remember")
    b = canonical_mc_id("S", "C", "understand")
    assert a != b
