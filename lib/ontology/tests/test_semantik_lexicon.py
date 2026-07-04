"""Wave #22 — SemantiK pedagogical-lexicon taxonomy loader.

The opener / apparatus / confusable vocabularies moved out of hardcoded Python
constants into ``schemas/taxonomies/semantik_lexicon.json`` (owner directive 3:
lexicon profiles, not corpus-specific code). These tests lock the loader's
profile-merge semantics AND assert the default ``generic-academic+openstax``
profile reproduces the historical classifier vocabulary exactly (behavior-
preserving refactor).
"""
from __future__ import annotations

from lib.ontology import taxonomy as T


# ---------------------------------------------------------------------------
# Schema shape + caching.
# ---------------------------------------------------------------------------


def test_lexicon_loads_and_has_profiles():
    lex = T.load_semantik_lexicon()
    assert "profiles" in lex and isinstance(lex["profiles"], dict)
    assert "generic-academic" in lex["profiles"]
    assert "openstax" in lex["profiles"]


def test_lexicon_cached_identity():
    # lru_cache — the same dict object is returned each call.
    assert T.load_semantik_lexicon() is T.load_semantik_lexicon()


# ---------------------------------------------------------------------------
# Profile resolution.
# ---------------------------------------------------------------------------


def test_default_profile_spec():
    assert T.resolve_lexicon_profile({}) == "generic-academic+openstax"


def test_env_override_profile_spec():
    assert (
        T.resolve_lexicon_profile({"SEMANTIK_LEXICON_PROFILE": "generic-academic"})
        == "generic-academic"
    )


def test_unknown_profile_falls_back_to_default_keys():
    # An all-unknown spec yields the default profiles' vocab, never empty.
    openers = T.get_lexicon_openers("does-not-exist")
    roles = {o["role"] for o in openers}
    assert "try_it" in roles and "objectives" in roles


# ---------------------------------------------------------------------------
# Behavior-preserving vocabulary (default profile == historical constants).
# ---------------------------------------------------------------------------


def test_default_openers_reproduce_historical_order():
    openers = T.get_lexicon_openers("generic-academic+openstax")
    got = [(o["role"], o["display"], o["numbered"]) for o in openers]
    assert got == [
        ("objectives", "Learning Objectives", False),
        ("readiness_check", "Be Prepared", True),
        ("try_it", "Try It", True),
        ("worked_example", "Example", True),
        ("how_to", "How To", False),
        ("solution", "Solution", False),
    ]


def test_interior_split_subset():
    openers = T.get_lexicon_openers()
    interior = {o["role"] for o in openers if o["interior_split"]}
    assert interior == {"readiness_check", "try_it", "worked_example"}


def test_association_roles():
    by_role = {o["role"]: o["association_role"] for o in T.get_lexicon_openers()}
    assert by_role["worked_example"] == "example"
    assert by_role["try_it"] == "practice"
    assert by_role["how_to"] == "procedure"
    assert by_role["readiness_check"] == "readiness"


def test_apparatus_names_reproduce_historical():
    assert T.get_lexicon_apparatus_names("generic-academic+openstax") == (
        "Key Terms",
        "Key Concepts",
        "Chapter Review",
        "Review Exercises",
        "Practice Test",
    )


def test_interior_apparatus_names_are_allcaps_subset():
    names = set(T.get_lexicon_interior_apparatus_names())
    assert names == {"KEY TERMS", "KEY CONCEPTS", "REVIEW EXERCISES", "PRACTICE TEST"}
    # "Chapter Review" is NOT an interior banner.
    assert "CHAPTER REVIEW" not in names


def test_apparatus_whitelist_reproduces_historical():
    wl = T.get_lexicon_apparatus_whitelist()
    for expected in (
        "solution",
        "key terms",
        "key concepts",
        "practice test",
        "review exercises",
        "chapter outline",
        "introduction",
        "learning objectives",
        "self check",
    ):
        assert expected in wl


def test_confusables_default_has_trvit():
    conf = T.get_lexicon_confusables()
    patterns = {c["pattern"]: c["canonical"] for c in conf}
    assert patterns.get("tr[vy]it") == "TRY IT"


# ---------------------------------------------------------------------------
# Overlay semantics: openstax vocab is additive over generic-academic.
# ---------------------------------------------------------------------------


def test_generic_profile_excludes_textbook_openers():
    roles = {o["role"] for o in T.get_lexicon_openers("generic-academic")}
    assert "try_it" not in roles and "readiness_check" not in roles
    assert {"objectives", "worked_example", "how_to", "solution"} <= roles


def test_generic_profile_has_no_confusables():
    assert T.get_lexicon_confusables("generic-academic") == ()
