"""Wave #22 — SemantiK pedagogical-lexicon taxonomy loader.

The opener / apparatus / confusable vocabularies moved out of hardcoded Python
constants into ``schemas/taxonomies/semantik_lexicon.json`` (owner directive 3:
lexicon profiles, not corpus-specific code). These tests lock the loader's
profile-merge semantics AND assert the default ``generic-academic+open-textbook``
profile reproduces the historical classifier vocabulary exactly (behavior-
preserving refactor). The legacy publisher-named profile-key spellings must
keep resolving via ``LEGACY_PROFILE_ALIASES`` (silent back-compat).
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
    assert "open-textbook" in lex["profiles"]


def test_lexicon_cached_identity():
    # lru_cache — the same dict object is returned each call.
    assert T.load_semantik_lexicon() is T.load_semantik_lexicon()


# ---------------------------------------------------------------------------
# Profile resolution.
# ---------------------------------------------------------------------------


def test_default_profile_spec():
    assert T.resolve_lexicon_profile({}) == "generic-academic+open-textbook"


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
    openers = T.get_lexicon_openers("generic-academic+open-textbook")
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
    assert T.get_lexicon_apparatus_names("generic-academic+open-textbook") == (
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
# Overlay semantics: open-textbook vocab is additive over generic-academic.
# ---------------------------------------------------------------------------


def test_generic_profile_excludes_open_textbook_openers():
    roles = {o["role"] for o in T.get_lexicon_openers("generic-academic")}
    assert "try_it" not in roles and "readiness_check" not in roles
    assert {"objectives", "worked_example", "how_to", "solution"} <= roles


def test_generic_profile_has_no_confusables():
    assert T.get_lexicon_confusables("generic-academic") == ()


# ---------------------------------------------------------------------------
# Legacy profile-key aliases (pre-rename compat): the old publisher-named
# spelling resolves to the SAME lexicon as the neutral key — silently.
# ---------------------------------------------------------------------------


def test_legacy_alias_spec_resolves_to_same_lexicon():
    legacy = "generic-academic+openstax"
    renamed = "generic-academic+open-textbook"
    assert T.get_lexicon_openers(legacy) == T.get_lexicon_openers(renamed)
    assert T.get_lexicon_apparatus_names(legacy) == T.get_lexicon_apparatus_names(renamed)
    assert T.get_lexicon_confusables(legacy) == T.get_lexicon_confusables(renamed)
    assert T.get_lexicon_subclass_glosses(legacy) == T.get_lexicon_subclass_glosses(renamed)


def test_legacy_alias_env_value_resolves_to_same_lexicon():
    # An operator env still set to the LEGACY value keeps working: the raw
    # spec passes through resolve_lexicon_profile and aliases at key-split.
    env = {"SEMANTIK_LEXICON_PROFILE": "generic-academic+openstax"}
    spec = T.resolve_lexicon_profile(env)
    assert spec == "generic-academic+openstax"  # raw spec, aliased downstream
    assert T.get_lexicon_openers(spec) == T.get_lexicon_openers(
        T.resolve_lexicon_profile({})
    )
    # Bare legacy key aliases too (not just the composite default).
    conf = T.get_lexicon_confusables("openstax")
    assert {c["pattern"]: c["canonical"] for c in conf}.get("tr[vy]it") == "TRY IT"


# ---------------------------------------------------------------------------
# Task #24 — composite-unit ontology expansion: scholarly + admonition profiles.
# ---------------------------------------------------------------------------


def test_scholarly_and_admonition_profiles_present():
    profiles = T.load_semantik_lexicon()["profiles"]
    assert "scholarly" in profiles and "admonition" in profiles


def test_scholarly_openers_map_theorem_apparatus_roles():
    by_role = {
        o["role"]: o["association_role"]
        for o in T.get_lexicon_openers("generic-academic+scholarly")
    }
    # Statements coagulate a theorem_block; proof / remark / corollary are its
    # satellites; definition + notation drive the existing definition_group shape.
    assert by_role["theorem"] == "theorem"
    assert by_role["lemma"] == "theorem"
    assert by_role["proposition"] == "theorem"
    assert by_role["proof"] == "proof"
    assert by_role["remark"] == "remark"
    assert by_role["corollary"] == "remark"  # trailing satellite
    assert by_role["definition"] == "definition"
    assert by_role["notation"] == "definition"


def test_admonition_openers_all_map_to_admonition_role():
    openers = T.get_lexicon_openers("generic-academic+admonition")
    admon = {o["role"] for o in openers if o["association_role"] == "admonition"}
    assert {
        "note", "warning", "tip", "caution",
        "important", "danger", "attention", "hint",
    } <= admon


def test_new_profiles_do_not_perturb_default_openers():
    # The default profile spec is unchanged by the new opt-in overlays.
    default_roles = [o["role"] for o in T.get_lexicon_openers("generic-academic+open-textbook")]
    assert default_roles == [
        "objectives", "readiness_check", "try_it", "worked_example", "how_to", "solution",
    ]
    # No scholarly / admonition vocabulary leaks into the default resolution.
    assert "theorem" not in default_roles and "note" not in default_roles


def test_scholarly_subclass_seed_vocab():
    sub = T.get_lexicon_subclasses("scholarly")
    assert "theorem-proof" in sub["theorem_block"]
    assert "notation-intro" in sub["definition_group"]


def test_admonition_subclass_seed_vocab():
    sub = T.get_lexicon_subclasses("admonition")
    assert "note" in sub["admonition"] and "warning" in sub["admonition"]


def test_subclass_glosses_dict_form():
    # generic-academic ships {label: gloss} — every seed carries a non-empty
    # one-line definition, and label ORDER matches get_lexicon_subclasses.
    glosses = T.get_lexicon_subclass_glosses("generic-academic")
    seeds = T.get_lexicon_subclasses("generic-academic")
    assert set(glosses) == set(seeds)
    for unit_type, labels in seeds.items():
        assert tuple(glosses[unit_type].keys()) == labels
        assert all(glosses[unit_type][lab] for lab in labels)
    assert "real-world" in glosses["worked_example"]["application-problem"]


def test_subclass_glosses_list_form_fallback():
    # A legacy list-form profile yields the labels with EMPTY glosses (the
    # prompt then falls back to the bare label list). open-textbook declares
    # no subclasses of its own, so the merged default spec inherits the
    # generic-academic glosses unchanged.
    merged = T.get_lexicon_subclass_glosses("generic-academic+open-textbook")
    base = T.get_lexicon_subclass_glosses("generic-academic")
    assert merged == base
