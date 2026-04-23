"""Wave 55: every Bloom-level detector routes through the canonical matcher.

Pre-Wave-55 there were four divergent detectors across the repo:

  1. ``Courseforge/scripts/generate_course.py::detect_bloom_level``
  2. ``Trainforge/parsers/html_content_parser.py::HTMLContentParser._detect_bloom_level``
  3. ``lib/validators/bloom.py::detect_bloom_level`` (re-implemented higher-level
     tie-breaking locally)
  4. ``LibV2/tools/libv2/query_decomposer.py::QueryDecomposer._detect_bloom_level``
     (used whole-word set intersection; missed longest-verb-first ties)

Sites 1-3 used ``text_lower.startswith(verb) or f" {verb} " in text_lower``
which silently missed verbs at end-of-text or followed by punctuation. Wave 55
has every site delegate to ``lib.ontology.bloom.detect_bloom_level`` (for
packages inside Ed4All) or to a byte-identical vendored implementation at
``LibV2/tools/libv2/_bloom_verbs.py`` (for LibV2, which is sandboxed from
``lib/`` per ``LibV2/CLAUDE.md``).

These tests assert two properties:

  * Behavioral: the canonical matcher catches the end-of-text / punctuation-
    adjacent / tie-breaking cases the old matchers missed.
  * Unification: each site produces the same ``(level, verb)`` pair as the
    canonical matcher on a shared input corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.ontology.bloom import detect_bloom_level as canonical_detect  # noqa: E402


# ---------------------------------------------------------------------------
# Inputs that exercise the specific bugs Wave 55 closes
# ---------------------------------------------------------------------------

# Each entry: (text, expected_level). Verb is derived at runtime from the
# canonical matcher so the table doesn't duplicate the ontology.
REGRESSION_INPUTS = [
    # End-of-text verb — pre-Wave-55 matchers required a trailing space and
    # fell through to (None, None) here.
    ("Students will apply", "apply"),
    # Verb immediately followed by a period — same failure mode.
    ("The student should apply.", "apply"),
    # Verb followed by a comma — pre-Wave-55 matchers required " verb "
    # (space-delimited).
    ("First, analyze the data", "analyze"),
    # Longest-verb-first tie-breaking — ``evaluate`` (8 chars) should beat
    # ``analyze`` (7 chars) when both appear. Pre-Wave-55 Courseforge/
    # Trainforge iterated in level order and returned the lower-level
    # ``analyze`` first.
    ("Analyze and evaluate the problem", "evaluate"),
    # Start-of-text still works (this was one of the two paths the old
    # matchers did handle — confirms we haven't regressed it).
    ("Design a scalable system", "create"),
    # Free text with no Bloom verb — must still return None, not a
    # false-positive from substring matching.
    ("No applicable taxonomy reference here", None),
]


def test_canonical_matcher_regression_corpus():
    """The canonical matcher handles every Wave 55 regression input."""
    for text, expected_level in REGRESSION_INPUTS:
        level, verb = canonical_detect(text)
        assert level == expected_level, (
            f"canonical_detect({text!r}): expected level={expected_level!r}, "
            f"got level={level!r} verb={verb!r}"
        )


# ---------------------------------------------------------------------------
# Site parity — each site returns what the canonical matcher returns
# ---------------------------------------------------------------------------


def test_lib_validators_bloom_delegates_to_canonical():
    """``lib.validators.bloom.detect_bloom_level`` returns the canonical level.

    The wrapper's signature is ``Optional[str]`` (level only, no verb) — it
    discards the canonical's verb tuple element.
    """
    from lib.validators.bloom import detect_bloom_level as site_detect

    for text, expected_level in REGRESSION_INPUTS:
        got = site_detect(text)
        assert got == expected_level, (
            f"lib.validators.bloom.detect_bloom_level({text!r}): "
            f"expected {expected_level!r}, got {got!r}"
        )


def test_courseforge_generate_course_delegates_to_canonical():
    """``Courseforge/scripts/generate_course.py`` re-exports the canonical."""
    from Courseforge.scripts.generate_course import detect_bloom_level as site_detect

    for text, expected_level in REGRESSION_INPUTS:
        level, _verb = site_detect(text)
        assert level == expected_level, (
            f"Courseforge.generate_course.detect_bloom_level({text!r}): "
            f"expected level={expected_level!r}, got level={level!r}"
        )


def test_trainforge_html_parser_delegates_to_canonical():
    """``HTMLContentParser._detect_bloom_level`` returns the canonical tuple."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    for text, expected_level in REGRESSION_INPUTS:
        level, _verb = parser._detect_bloom_level(text)
        assert level == expected_level, (
            f"HTMLContentParser._detect_bloom_level({text!r}): "
            f"expected level={expected_level!r}, got level={level!r}"
        )


def test_libv2_query_decomposer_delegates_to_vendored_canonical():
    """``QueryDecomposer._detect_bloom_level`` uses the vendored matcher.

    LibV2 is sandboxed from ``lib/`` (``LibV2/CLAUDE.md``) so it vendors
    ``detect_bloom_level`` in ``LibV2/tools/libv2/_bloom_verbs.py``. The
    vendored copy and the canonical must agree byte-for-byte on behavior,
    which this test enforces.
    """
    from LibV2.tools.libv2.query_decomposer import QueryDecomposer

    decomposer = QueryDecomposer()
    for text, expected_level in REGRESSION_INPUTS:
        got = decomposer._detect_bloom_level(text)
        assert got == expected_level, (
            f"QueryDecomposer._detect_bloom_level({text!r}): "
            f"expected {expected_level!r}, got {got!r}"
        )


def test_libv2_vendored_detector_matches_canonical():
    """Direct test of the vendored LibV2 detector against the canonical.

    Covers the (level, verb) tuple — not just the level — so any divergence
    in verb selection surfaces here rather than upstream of the vendored
    copy being re-used.
    """
    from LibV2.tools.libv2._bloom_verbs import detect_bloom_level as vendored

    for text, _expected_level in REGRESSION_INPUTS:
        assert vendored(text) == canonical_detect(text), (
            f"vendored detect_bloom_level({text!r}) diverged from canonical: "
            f"vendored={vendored(text)!r} canonical={canonical_detect(text)!r}"
        )
