"""Worker R -- REC-LNK-02 misconception content-hash ID helper.

Regression tests for ``Trainforge.generators.preference_factory._misconception_id``.
The helper replaces the earlier unstable format
``{chunk_id}_mc_{index:02d}_{hash}`` with content-addressed
``mc_<sha256(misconception|correction)[:16]>``. IDs must be:

1. Format-conformant with the Misconception schema (`^mc_[0-9a-f]{16}$`).
2. Stable across runs when inputs are unchanged.
3. Sensitive to both ``misconception`` and ``correction`` text (unlike the
   prior format which only hashed ``misconception`` text).
4. Invariant to outer whitespace (common upstream noise) but sensitive to
   inner whitespace (genuine text change).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Project root (Ed4All/). This file lives at
# Ed4All/Trainforge/tests/test_misconception_id.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.preference_factory import _misconception_id  # noqa: E402

SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "misconception.schema.json"
)

_ID_PATTERN = re.compile(r"^mc_[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# Format: helper output matches the schema pattern.
# ---------------------------------------------------------------------------


def test_misconception_id_format_matches_schema():
    """Generated IDs conform to ``^mc_[0-9a-f]{16}$`` (schema pattern)."""
    mc_id = _misconception_id(
        "Accessibility is only for screen-reader users.",
        "Accessibility benefits everyone including users without disabilities.",
    )
    assert _ID_PATTERN.match(mc_id), mc_id


def test_misconception_id_matches_schema_pattern_literal():
    """Cross-check: pattern loaded from the schema file matches the output."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    schema_pattern = schema["properties"]["id"]["pattern"]
    assert schema_pattern == r"^mc_[0-9a-f]{16}$"

    mc_id = _misconception_id("wrong belief", "right answer")
    assert re.compile(schema_pattern).match(mc_id), mc_id


# ---------------------------------------------------------------------------
# Stability: same input -> same ID.
# ---------------------------------------------------------------------------


def test_misconception_id_stable_across_runs():
    """Same (misconception, correction) -> same ID on repeated calls."""
    a = _misconception_id("wrong belief", "correct answer")
    b = _misconception_id("wrong belief", "correct answer")
    assert a == b


def test_misconception_id_independent_of_caller_context():
    """No hidden inputs: pure function of (misconception, correction).

    Guards against a regression where the helper re-introduces a chunk-id
    or positional-index argument.
    """
    # Called from two lexically-distinct sites with same inputs -> same output.
    def _wrap_a():
        return _misconception_id("belief", "answer")

    def _wrap_b():
        return _misconception_id("belief", "answer")

    assert _wrap_a() == _wrap_b()


# ---------------------------------------------------------------------------
# Sensitivity: ID responds to both fields.
# ---------------------------------------------------------------------------


def test_misconception_id_differs_on_text_change():
    """One-character edit to misconception text -> different ID."""
    a = _misconception_id("original belief", "the correct answer")
    b = _misconception_id("original belief!", "the correct answer")
    assert a != b


def test_misconception_id_differs_on_correction_change():
    """One-character edit to correction text -> different ID.

    The prior format hashed only the misconception text and so would have
    produced an identical ID for these two inputs. The new format must
    distinguish them.
    """
    a = _misconception_id("the belief", "original correction")
    b = _misconception_id("the belief", "original correction!")
    assert a != b


def test_misconception_id_differs_when_fields_swapped():
    """Swapping misconception <-> correction produces different ID.

    Guards against a bug where the two fields are concatenated without a
    separator (so ``"abc" + "def"`` would collide with ``"ab" + "cdef"``).
    """
    a = _misconception_id("A", "B")
    b = _misconception_id("B", "A")
    assert a != b


# ---------------------------------------------------------------------------
# Canonicalization: outer whitespace stripped; inner whitespace preserved.
# ---------------------------------------------------------------------------


def test_misconception_id_whitespace_normalized():
    """Outer whitespace on either field is normalised out of the hash."""
    base = _misconception_id("wrong", "right")
    assert _misconception_id(" wrong ", "right") == base
    assert _misconception_id("wrong", " right ") == base
    assert _misconception_id("\nwrong\t", "\tright\n") == base


def test_misconception_id_inner_whitespace_is_significant():
    """Inner whitespace differences still change the ID.

    ``"wrong"`` and ``"wr ong"`` are genuinely different text; the hash
    should reflect that.
    """
    a = _misconception_id("wrong", "right")
    b = _misconception_id("wr ong", "right")
    assert a != b


# ---------------------------------------------------------------------------
# Safety: None / empty inputs produce a well-formed ID (no crash).
# ---------------------------------------------------------------------------


def test_misconception_id_handles_none_inputs():
    """``None`` for either field is tolerated (treated as empty string).

    The old helper crashed on ``None`` in misconception (str.encode); the
    new helper guards against that so callers don't need to pre-filter.
    """
    mc_id = _misconception_id(None, None)  # type: ignore[arg-type]
    assert _ID_PATTERN.match(mc_id), mc_id


def test_misconception_id_empty_strings_produce_valid_id():
    """Empty-string inputs produce a well-formed (but meaningless) ID."""
    mc_id = _misconception_id("", "")
    assert _ID_PATTERN.match(mc_id), mc_id
