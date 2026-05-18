"""Tests for the dynamic CURIE-minting helpers in ``curie_discovery``.

The minting helpers (``mint_curie_prefix`` / ``curie_for_concept`` /
``build_minted_curie_map`` / ``minted_curie_by_canonical``) derive
stable, grammar-valid CURIEs from a per-course domain-concept
vocabulary so prose corpora (zero RDF CURIEs in the text) can satisfy
``BlockCurieAnchoringValidator``.

The load-bearing invariant is the DUAL-GRAMMAR ROUND-TRIP: every minted
CURIE must satisfy BOTH the outline JSON schema's ``_CURIE_PATTERN``
(``^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$``) AND survive a clean round-trip
through ``lib.ontology.curie_extraction.extract_curies`` (which runs
``CURIE_REGEX`` — local name ``[A-Za-z][A-Za-z0-9_]*``, NO hyphens). A
minted CURIE with a hyphen would be silently truncated by
``extract_curies`` to the part before the hyphen.
"""

from __future__ import annotations

import re

from lib.ontology.curie_discovery import (
    build_minted_curie_map,
    curie_for_concept,
    mint_curie_prefix,
    minted_curie_by_canonical,
)
from lib.ontology.curie_extraction import extract_curies

# The outline JSON-schema CURIE pattern, mirrored verbatim from
# ``Courseforge/generators/_outline_provider.py::_CURIE_PATTERN``.
_OUTLINE_CURIE_PATTERN = re.compile(r"^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$")


def _assert_dual_grammar_valid(curie: str) -> None:
    """Assert ``curie`` satisfies BOTH CURIE grammars."""
    assert _OUTLINE_CURIE_PATTERN.match(curie), (
        f"{curie!r} fails the outline schema _CURIE_PATTERN"
    )
    # extract_curies must return the CURIE intact — no truncation.
    extracted = extract_curies(curie)
    assert curie in extracted, (
        f"{curie!r} did not round-trip through extract_curies "
        f"(got {extracted!r}) — likely a hyphen in the local name"
    )


# --------------------------------------------------------------------------- #
# mint_curie_prefix
# --------------------------------------------------------------------------- #


class TestMintCuriePrefix:
    def test_deterministic(self):
        assert mint_curie_prefix("OPENSTAX_ALG_9") == mint_curie_prefix(
            "OPENSTAX_ALG_9"
        )

    def test_lowercases_and_strips_punctuation(self):
        # Underscores / hyphens / dots all fuse out.
        assert mint_curie_prefix("OPENSTAX_ALG_9") == "samplecoursea"
        assert mint_curie_prefix("phys-101.intro") == "phys101intro"

    def test_digit_led_gets_letter_prepended(self):
        prefix = mint_curie_prefix("9to5-physics")
        assert prefix[0].isalpha()
        assert prefix == "c9to5physics"

    def test_length_capped(self):
        prefix = mint_curie_prefix("a" * 100)
        assert len(prefix) <= 16

    def test_empty_falls_back_to_course(self):
        assert mint_curie_prefix("") == "course"
        assert mint_curie_prefix("!!!") == "course"

    def test_prefix_matches_grammar(self):
        for course_id in ("OPENSTAX_ALG_9", "9to5", "Bio 201", "x"):
            prefix = mint_curie_prefix(course_id)
            assert re.match(r"^[a-z][a-z0-9]*$", prefix), prefix


# --------------------------------------------------------------------------- #
# curie_for_concept
# --------------------------------------------------------------------------- #


class TestCurieForConcept:
    def test_basic(self):
        assert curie_for_concept("slope", "alg") == "alg:slope"

    def test_hyphen_converted_to_underscore(self):
        # canonical_slug kebab-cases multi-word concepts; the hyphen
        # MUST become an underscore so extract_curies doesn't truncate.
        curie = curie_for_concept("Slope-Intercept Form", "alg")
        assert curie == "alg:slope_intercept_form"
        assert "-" not in curie.split(":", 1)[1]

    def test_digit_led_localname_gets_letter_prepended(self):
        curie = curie_for_concept("2D vectors", "phys")
        local = curie.split(":", 1)[1]
        assert local[0].isalpha()
        assert curie == "phys:c2d_vectors"

    def test_empty_canonical_still_valid(self):
        curie = curie_for_concept("", "alg")
        _assert_dual_grammar_valid(curie)

    def test_dual_grammar_round_trip(self):
        # The keystone assertion: every minted CURIE matches BOTH
        # grammars and round-trips through extract_curies intact.
        cases = [
            ("slope", "alg"),
            ("Slope-Intercept Form", "alg"),
            ("2D vectors", "phys"),
            ("the quadratic formula", "samplecoursea"),
            ("RDF/SHACL validation rules", "rdf"),
            ("a-b-c-d-e", "x"),
        ]
        for canonical, prefix in cases:
            curie = curie_for_concept(canonical, prefix)
            _assert_dual_grammar_valid(curie)


# --------------------------------------------------------------------------- #
# build_minted_curie_map
# --------------------------------------------------------------------------- #


def _vocab(concepts, course_id="OPENSTAX_ALG_9"):
    return {
        "schema_version": "v1",
        "course_id": course_id,
        "concept_count": len(concepts),
        "concepts": concepts,
    }


class TestBuildMintedCurieMap:
    def test_basic_map(self):
        vocab = _vocab([
            {"canonical": "slope", "aliases": ["gradient", "rise over run"]},
            {"canonical": "y-intercept"},
        ])
        result = build_minted_curie_map(vocab)
        assert len(result) == 2
        # Every minted CURIE is dual-grammar valid.
        for curie, entry in result.items():
            _assert_dual_grammar_valid(curie)
            assert "canonical" in entry
            assert "surface_forms" in entry
        # surface_forms carries canonical first, then aliases.
        slope_curie = "samplecoursea:slope"
        assert slope_curie in result
        assert result[slope_curie]["surface_forms"][0] == "slope"
        assert "gradient" in result[slope_curie]["surface_forms"]

    def test_empty_concepts_returns_empty(self):
        assert build_minted_curie_map(_vocab([])) == {}
        assert build_minted_curie_map({}) == {}
        assert build_minted_curie_map({"concepts": None}) == {}
        assert build_minted_curie_map(None) == {}  # type: ignore[arg-type]

    def test_course_id_arg_overrides_vocab(self):
        vocab = _vocab([{"canonical": "slope"}], course_id="IGNORED")
        result = build_minted_curie_map(vocab, course_id="PHYS_101")
        assert "phys101:slope" in result

    def test_course_slug_fallback(self):
        vocab = {
            "schema_version": "v1",
            "course_slug": "bio-201",
            "concept_count": 1,
            "concepts": [{"canonical": "mitosis"}],
        }
        result = build_minted_curie_map(vocab)
        assert "bio201:mitosis" in result

    def test_canonical_collision_merges_surface_forms(self):
        # Two concepts whose canonical slugs collide merge rather than
        # clobber — the second's aliases land on the first's entry.
        vocab = _vocab([
            {"canonical": "slope", "aliases": ["gradient"]},
            {"canonical": "slope", "aliases": ["steepness"]},
        ])
        result = build_minted_curie_map(vocab)
        assert len(result) == 1
        entry = result["samplecoursea:slope"]
        assert "gradient" in entry["surface_forms"]
        assert "steepness" in entry["surface_forms"]

    def test_malformed_concept_skipped(self):
        vocab = _vocab([
            {"canonical": "slope"},
            {"no_canonical": "x"},      # missing canonical
            "not a dict",               # not a dict
            {"canonical": ""},          # empty canonical
        ])
        result = build_minted_curie_map(vocab)
        assert len(result) == 1


# --------------------------------------------------------------------------- #
# minted_curie_by_canonical
# --------------------------------------------------------------------------- #


class TestMintedCurieByCanonical:
    def test_inverse_lookup(self):
        vocab = _vocab([
            {"canonical": "slope"},
            {"canonical": "y-intercept"},
        ])
        minted = build_minted_curie_map(vocab)
        inverse = minted_curie_by_canonical(minted)
        assert inverse["slope"] == "samplecoursea:slope"
        assert inverse["y-intercept"] == "samplecoursea:y_intercept"

    def test_empty_input(self):
        assert minted_curie_by_canonical({}) == {}
        assert minted_curie_by_canonical(None) == {}  # type: ignore[arg-type]
