"""Tests for the shared exercise-/apparatus-detection lexicon (fix W1, Defect C).

Covers:
  * the taxonomy file loads through ``load_taxonomy`` (generic name mode);
  * structural detection (composite_unit / unit_roles tri-state, instructional
    wins over apparatus);
  * lexical marker detection over rendered text;
  * profile union vs single-profile selection (default = union of all profiles);
  * the learning-objective boilerplate prefix-strip helper (Defect-D consumer);
  * the legacy constants are byte-identical re-exports at the old call sites.

Synthetic fixtures only — no course slugs / paths.
"""

from __future__ import annotations

import re

from lib.objectives.apparatus_lexicon import (
    EXERCISE_BANNER_RE,
    EXTRA_EXERCISE_BANNER_RE,
    FOLLOWING_EXERCISES_RE,
    JUNK_MARKERS,
    TAXONOMY_NAME,
    compile_profile,
)


# ---------------------------------------------------------------------------
# Taxonomy loads through the generic loader
# ---------------------------------------------------------------------------


def test_taxonomy_loads_via_load_taxonomy():
    from lib.ontology.taxonomy import load_taxonomy

    lex = load_taxonomy(TAXONOMY_NAME)
    assert isinstance(lex, dict)
    assert "profiles" in lex and isinstance(lex["profiles"], dict)
    assert "generic" in lex["profiles"]
    # Structural signal enums are present.
    assert "structural" in lex
    assert "exercise_set" in lex["structural"]["apparatus_units"]


def test_load_taxonomy_default_mode_unchanged():
    # The name-less call still returns the subject taxonomy (backward-compat).
    from lib.ontology.taxonomy import load_taxonomy

    data = load_taxonomy()
    assert "divisions" in data


def test_load_taxonomy_rejects_path_traversal():
    from lib.ontology.taxonomy import load_taxonomy

    for bad in ("../secrets", "a/b", "a\\b"):
        try:
            load_taxonomy(bad)
        except ValueError:
            continue
        except FileNotFoundError:
            continue
        raise AssertionError(f"expected {bad!r} to be rejected")


# ---------------------------------------------------------------------------
# Structural detection
# ---------------------------------------------------------------------------


def test_structural_signal_tristate():
    p = compile_profile()
    assert p.structural_signal({"composite_unit": "exercise_set"}) is True
    assert p.structural_signal({"unit_roles": ["practice"]}) is True
    assert p.structural_signal({"unit_roles": ["answer_key"]}) is True
    # Instructional wins over apparatus when both present.
    assert (
        p.structural_signal(
            {"composite_unit": "exercise_set", "unit_roles": ["statement"]}
        )
        is False
    )
    assert p.structural_signal({"composite_unit": "worked_example"}) is False
    # No metadata → None (defer to lexical).
    assert p.structural_signal({}) is None
    assert p.structural_signal({"text": "prose"}) is None
    assert p.structural_signal("not a dict") is None


def test_is_apparatus_chunk_falls_back_to_lexical():
    p = compile_profile()
    # No metadata but the body carries a marker → apparatus.
    assert p.is_apparatus_chunk(
        {"text": "In the following exercises, solve each equation."}
    )
    # No metadata, clean prose → not apparatus.
    assert not p.is_apparatus_chunk(
        {"text": "The distributive property multiplies across a sum."}
    )


# ---------------------------------------------------------------------------
# Lexical detection
# ---------------------------------------------------------------------------


def test_text_has_marker():
    p = compile_profile()
    assert p.text_has_marker("BE PREPARED to solve this")
    assert p.text_has_marker("Practice   Makes   Perfect")  # whitespace-flex
    assert p.text_has_marker("try it now")  # case-insensitive
    assert p.text_has_marker("answer ⓐ then ⓑ")  # glyph markers
    assert not p.text_has_marker("A clean instructional sentence about slopes.")
    assert not p.text_has_marker("")


# ---------------------------------------------------------------------------
# Profile union vs single profile
# ---------------------------------------------------------------------------


def test_default_is_union_of_all_profiles():
    default = compile_profile()
    generic = compile_profile("generic")
    # The union pulls in textbook-shaped markers the generic profile lacks
    # (e.g. the fused "EXERCISES Practice Makes Perfect" banner).
    assert len(default.marker_regexes) >= len(generic.marker_regexes)
    assert default.text_has_marker("EXERCISES Practice Makes Perfect")


def test_unknown_profile_falls_back_to_union():
    p = compile_profile("does-not-exist")
    assert p.marker_regexes  # non-empty (fell back to the union)


def test_legacy_profile_key_alias_resolves_same_lexicon():
    # Pre-rename compat: the old publisher-named key spelling resolves to the
    # SAME compiled lexicon as the neutral "textbook-shaped" key (silently,
    # via lib.ontology.taxonomy.LEGACY_PROFILE_ALIASES).
    legacy = compile_profile("generic+openstax-shaped")
    renamed = compile_profile("generic+textbook-shaped")
    assert [r.pattern for r in legacy.marker_regexes] == [
        r.pattern for r in renamed.marker_regexes
    ]
    # The bare legacy key aliases too — and carries the fused banner marker
    # the generic profile lacks.
    bare = compile_profile("openstax-shaped")
    assert bare.text_has_marker("EXERCISES Practice Makes Perfect")


def test_compile_profile_is_cached():
    assert compile_profile() is compile_profile()
    assert compile_profile("generic") is compile_profile("generic")


# ---------------------------------------------------------------------------
# LO-boilerplate prefix strip (Defect-D consumer)
# ---------------------------------------------------------------------------


def test_strip_lo_boilerplate_recovers_tail_after_colon():
    p = compile_profile()
    out = p.strip_lo_boilerplate_prefix(
        "By the end of this section, you will be able to: Round whole numbers."
    )
    assert out == "Round whole numbers."


def test_strip_lo_boilerplate_pure_preamble_is_empty():
    p = compile_profile()
    assert p.strip_lo_boilerplate_prefix("By the end of this section:") == ""
    # Leads with boilerplate but no colon → pure preamble → empty.
    assert p.strip_lo_boilerplate_prefix("In the following exercises solve.") == ""


def test_strip_lo_boilerplate_leaves_non_boilerplate_untouched():
    p = compile_profile()
    text = "The slope of a line measures its steepness."
    assert p.strip_lo_boilerplate_prefix(text) == text


# ---------------------------------------------------------------------------
# Legacy re-export byte-identity
# ---------------------------------------------------------------------------


def test_legacy_constants_are_reexported_identically():
    import lib.chunk_heading_sanity as chs
    import lib.objectives.citation_reselect as cr
    import lib.objectives.sub_objectives as so

    assert chs._EXERCISE_BANNER_RE is EXERCISE_BANNER_RE
    assert cr._EXERCISE_BANNER_RE is EXERCISE_BANNER_RE
    assert cr._FOLLOWING_EXERCISES_RE is FOLLOWING_EXERCISES_RE
    assert cr._EXTRA_EXERCISE_BANNER_RE is EXTRA_EXERCISE_BANNER_RE
    assert so._JUNK_MARKERS is JUNK_MARKERS


def test_legacy_patterns_unchanged():
    # Exact pattern strings pinned so a future lexicon edit can't silently
    # drift the byte-identical legacy behavior.
    assert EXERCISE_BANNER_RE.pattern == (
        r"\b(?:EXERCISES?\s+Practice\s+Makes\s+Perfect"
        r"|In\s+the\s+following\s+exercises)\b"
    )
    assert EXERCISE_BANNER_RE.flags & re.IGNORECASE
    assert FOLLOWING_EXERCISES_RE.pattern == r"In\s+the\s+following\s+exercises"
    assert EXTRA_EXERCISE_BANNER_RE.pattern == (
        r"\b(?:Practice\s+Makes\s+Perfect"
        r"|Section\s+Exercises"
        r"|Review\s+Exercises)\b"
    )
    assert JUNK_MARKERS == (
        "::", "ⓐ", "ⓑ", "ⓒ", "ⓓ", "ⓔ",
        "BE PREPARED", "TRY IT", "HOW TO", "LEARNING OBJECTIVES",
    )


def test_statement_matcher_excludes_callout_labels():
    """Statement-scoped matching must not flag ordinary statement English.

    ``callout_labels`` phrases ("how to", "learning objectives") are common
    inside legitimate CO statements; only banners/boilerplate/glyphs flag.
    """
    from lib.objectives.apparatus_lexicon import compile_profile

    profile = compile_profile()
    # Callout tokens: full-text matcher flags, statement matcher must not.
    for stmt in (
        "Explain how to solve a two-step linear equation.",
        "Identify the learning objectives addressed by each module.",
    ):
        assert profile.text_has_marker(stmt)
        assert not profile.statement_has_apparatus_marker(stmt)
    # Banners / boilerplate / glyphs flag in BOTH scopes.
    for stmt in (
        "In the following exercises, solve each system.",
        "By the end of this section you can factor trinomials.",
        "Simplify ⓐ the first expression.",
    ):
        assert profile.text_has_marker(stmt)
        assert profile.statement_has_apparatus_marker(stmt)
