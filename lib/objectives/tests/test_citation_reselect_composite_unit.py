"""Wave #22 quick-wins — the citation-reselect exercise-demote arm uses the
chunk's pedagogical metadata (``composite_unit`` / ``unit_roles``) as
the PRIMARY signal, falling back to the text heuristic on legacy chunks.

Both paths covered; legacy behavior (no metadata) is byte-identical to the
text heuristic.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.citation_reselect import (  # noqa: E402
    _is_exercise_like,
    _is_exercise_like_chunk,
    _pedagogical_exercise_signal,
)

# A text the conservative heuristic flags as exercise-like (leads with the
# textbook "In the following exercises" instruction line).
_EXERCISE_TEXT = "In the following exercises, find the place value of the digit."
# Plain instructional prose the heuristic does NOT flag.
_PROSE_TEXT = "A radical expression is simplified by factoring perfect squares."


# --------------------------------------------------------------------------
# Tri-state metadata signal
# --------------------------------------------------------------------------


def test_signal_exercise_set_unit_demotes():
    assert _pedagogical_exercise_signal({"composite_unit": "exercise_set"}) is True


def test_signal_worked_example_unit_is_instructional():
    assert (
        _pedagogical_exercise_signal({"composite_unit": "worked_example"}) is False
    )


def test_signal_practice_role_demotes():
    assert _pedagogical_exercise_signal({"unit_roles": ["try_it"]}) is True
    assert _pedagogical_exercise_signal({"unit_roles": ["practice"]}) is True


def test_signal_statement_role_is_instructional():
    assert _pedagogical_exercise_signal({"unit_roles": ["statement"]}) is False


def test_signal_instructional_wins_over_practice():
    chunk = {"unit_roles": ["worked_example", "try_it"]}
    assert _pedagogical_exercise_signal(chunk) is False


def test_signal_absent_metadata_is_none():
    assert _pedagogical_exercise_signal({"text": _EXERCISE_TEXT}) is None
    assert _pedagogical_exercise_signal({}) is None
    assert _pedagogical_exercise_signal("not a dict") is None


# --------------------------------------------------------------------------
# _is_exercise_like_chunk: metadata PRIMARY, text heuristic FALLBACK
# --------------------------------------------------------------------------


def test_metadata_overrides_exercise_looking_text():
    # A worked example whose prose happens to read like an exercise line is
    # NOT demoted — the metadata is authoritative.
    chunk = {"composite_unit": "worked_example", "text": _EXERCISE_TEXT}
    assert _is_exercise_like(_EXERCISE_TEXT) is True  # heuristic would flag it
    assert _is_exercise_like_chunk(chunk) is False  # metadata wins → instructional


def test_metadata_demotes_prose_looking_exercise_set():
    # An exercise_set whose prose reads instructional is still demoted.
    chunk = {"composite_unit": "exercise_set", "text": _PROSE_TEXT}
    assert _is_exercise_like(_PROSE_TEXT) is False
    assert _is_exercise_like_chunk(chunk) is True


def test_fallback_to_heuristic_when_no_metadata():
    # Legacy / non-SemantiK chunk (no pedagogical fields) → text heuristic,
    # byte-identical to _is_exercise_like.
    ex_chunk = {"text": _EXERCISE_TEXT}
    prose_chunk = {"text": _PROSE_TEXT}
    assert _is_exercise_like_chunk(ex_chunk) is True
    assert _is_exercise_like_chunk(prose_chunk) is False
    assert _is_exercise_like_chunk(ex_chunk) == _is_exercise_like(_EXERCISE_TEXT)
    assert _is_exercise_like_chunk(prose_chunk) == _is_exercise_like(_PROSE_TEXT)


def test_non_dict_chunk_falls_back_safely():
    assert _is_exercise_like_chunk(None) is False
    assert _is_exercise_like_chunk("raw string") is False
