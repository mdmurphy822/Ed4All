"""Wave 82 tests for the acronym-preserving label helper."""

from __future__ import annotations

from lib.ontology.labels import (
    KNOWN_ACRONYMS,
    slug_to_label,
    titlecase_with_acronyms,
)


# ---------------------------------------------------------------------------
# Audit-case reproductions
# ---------------------------------------------------------------------------


class TestAuditReproductions:
    def test_owl_2_rl_renders_uppercase(self):
        # Audit reported "Owl 2 Rl" — must now be "OWL 2 RL".
        assert slug_to_label("owl-2-rl") == "OWL 2 RL"

    def test_owl_2_dl_renders_uppercase(self):
        assert slug_to_label("owl-2-dl") == "OWL 2 DL"

    def test_owl_2_el_ql_render_uppercase(self):
        assert slug_to_label("owl-2-el") == "OWL 2 EL"
        assert slug_to_label("owl-2-ql") == "OWL 2 QL"

    def test_rdfs_renders_uppercase(self):
        assert slug_to_label("rdfs") == "RDFS"
        assert slug_to_label("rdfs-class") == "RDFS Class"

    def test_sparql_renders_uppercase(self):
        assert slug_to_label("sparql") == "SPARQL"
        assert slug_to_label("sparql-query") == "SPARQL Query"

    def test_shacl_renders_uppercase(self):
        assert slug_to_label("shacl") == "SHACL"
        assert slug_to_label("shacl-shape") == "SHACL Shape"


# ---------------------------------------------------------------------------
# Hyphenated tokens (json-ld is the canonical example)
# ---------------------------------------------------------------------------


class TestHyphenatedTokens:
    def test_json_ld_both_segments_uppercase(self):
        assert titlecase_with_acronyms("json-ld") == "JSON-LD"

    def test_n_triples_n_lowercased_n_not_acronym_in_set(self):
        # "n" alone isn't in KNOWN_ACRONYMS — render lowercase title.
        # "Triples" is title-cased.
        assert titlecase_with_acronyms("n-triples") == "N-Triples"

    def test_mixed_hyphen_acronym_segments(self):
        # url-shortener: URL is acronym, shortener is title.
        assert titlecase_with_acronyms("url-shortener") == "URL-Shortener"


# ---------------------------------------------------------------------------
# Plain title-case (non-acronym tokens)
# ---------------------------------------------------------------------------


class TestPlainTitleCase:
    def test_no_acronyms_normal_titlecase(self):
        assert titlecase_with_acronyms("blank node") == "Blank Node"

    def test_slug_to_label_replaces_hyphens(self):
        assert slug_to_label("blank-node") == "Blank Node"

    def test_acronym_followed_by_word(self):
        assert slug_to_label("rdf-graph") == "RDF Graph"

    def test_word_followed_by_acronym(self):
        assert slug_to_label("named-graph") == "Named Graph"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCompoundAcronymOverrides:
    """Slug-level overrides for compound acronyms whose canonical form keeps the hyphen."""

    def test_json_ld_preserves_hyphen(self):
        assert slug_to_label("json-ld") == "JSON-LD"

    def test_n_triples_preserves_hyphen(self):
        assert slug_to_label("n-triples") == "N-Triples"

    def test_n_quads_preserves_hyphen(self):
        assert slug_to_label("n-quads") == "N-Quads"

    def test_rdf_xml_preserves_slash(self):
        # RDF/XML canonical form uses a slash, not a hyphen — round-trip
        # the slug to its W3C-spec rendering.
        assert slug_to_label("rdf-xml") == "RDF/XML"

    def test_compound_prefix_does_not_match_override(self):
        # "json-ld-context" is NOT in the override table, so it falls
        # through to the standard hyphen-stripping → "JSON LD Context".
        # Exact-slug matching prevents over-broad rewrites.
        assert slug_to_label("json-ld-context") == "JSON LD Context"


class TestEdgeCases:
    def test_empty_input(self):
        assert slug_to_label("") == ""
        assert titlecase_with_acronyms("") == ""

    def test_non_string_input_returns_empty(self):
        assert titlecase_with_acronyms(None) == ""  # type: ignore[arg-type]
        assert slug_to_label(None) == ""  # type: ignore[arg-type]

    def test_already_uppercase_acronym_passes_through(self):
        assert titlecase_with_acronyms("OWL") == "OWL"

    def test_already_correctly_cased_input_idempotent(self):
        assert titlecase_with_acronyms("OWL 2 RL") == "OWL 2 RL"
        assert slug_to_label("OWL 2 RL") == "OWL 2 RL"  # no hyphens, just space-cased

    def test_multi_word_with_no_acronyms(self):
        assert slug_to_label("first-class-citizen") == "First Class Citizen"


# ---------------------------------------------------------------------------
# Acronym set integrity
# ---------------------------------------------------------------------------


class TestAcronymSet:
    def test_w3c_standards_present(self):
        for a in ["rdf", "rdfs", "owl", "shacl", "sparql"]:
            assert a in KNOWN_ACRONYMS, f"missing W3C acronym {a}"

    def test_owl_2_profiles_present(self):
        for a in ["rl", "el", "ql", "dl"]:
            assert a in KNOWN_ACRONYMS, f"missing OWL 2 profile {a}"

    def test_uri_iri_present(self):
        for a in ["uri", "iri", "url"]:
            assert a in KNOWN_ACRONYMS, f"missing identifier acronym {a}"

    def test_acronyms_are_lowercased_in_storage(self):
        # Storage form is lowercase so lookup matches case-insensitively.
        for a in KNOWN_ACRONYMS:
            assert a == a.lower(), f"non-lowercase entry: {a!r}"
