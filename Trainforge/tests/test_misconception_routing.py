"""Wave 82 regression tests for misconception → concept_tag routing.

The rdf-shacl-551-2 audit found `interferes_with` edges landing on the
wrong target concept. Example: a misconception about "blank node vs IRI"
routed to ``concept:one-line-rule`` because the legacy heuristic always
picked the chunk's first concept_tag, regardless of the misconception's
subject.

Pre-Wave-82, ``_build_misconceptions_for_graph`` set
``concept_id = explicit_cid or _make_concept_id(first_tag, course_id)``.
Verified empirically against the rdf-shacl-551-2 chunks: 0% of
authored misconceptions carry an explicit ``concept_id``, so the
first-tag fallback fired 100% of the time.

Wave 82 inserts a token-overlap match between the legacy ``explicit_cid``
path and the legacy ``first_tag`` fallback. Tag tokens (slug split on
``-``) are matched against statement tokens (extracted by regex), with
:func:`lib.ontology.concept_classifier.singular_form` applied to both
sides so plural mismatches (``triple`` vs ``triples``) don't lose
signal.
"""

from __future__ import annotations

from Trainforge.process_course import _route_misconception_to_tag


# ---------------------------------------------------------------------------
# Audit-case reproductions — the actual misroutings the audit named.
# ---------------------------------------------------------------------------


class TestAuditReproductions:
    def test_triple_misconception_routes_to_triples_not_statement(self):
        # rdf_shacl_551_chunk_00001 actual concept_tags:
        # ['statement', 'triples', 'directional', 'subject', 'predicate'].
        # Pre-Wave-82 picked 'statement' (first tag); the misconception is
        # actually about triples.
        tags = ["statement", "triples", "directional", "subject", "predicate"]
        statement = "An RDF triple is like a row in a relational table."
        assert _route_misconception_to_tag(statement, tags) == "triples"

    def test_predicate_misconception_routes_to_predicate_not_statement(self):
        tags = ["statement", "triples", "directional", "subject", "predicate"]
        statement = "The predicate is just a string label like a column name."
        assert _route_misconception_to_tag(statement, tags) == "predicate"

    def test_blank_node_misconception_routes_to_blank_node_not_one_line_rule(self):
        # The exact audit case (mc_17f542e32b29d766 in the wave76.bak run).
        tags = ["one-line-rule", "blank-node", "iri"]
        statement = "A blank node is just an anonymous URI you can dereference in a browser."
        assert _route_misconception_to_tag(statement, tags) == "blank-node"


# ---------------------------------------------------------------------------
# Plural / singular handling — the audit chunks have ``triples`` (plural)
# but misconception statements use ``triple`` (singular). Both sides go
# through ``singular_form`` so the match still fires.
# ---------------------------------------------------------------------------


class TestPluralFolding:
    def test_singular_statement_matches_plural_tag(self):
        tags = ["foo", "triples"]
        assert _route_misconception_to_tag("a triple is X", tags) == "triples"

    def test_plural_statement_matches_singular_tag(self):
        tags = ["foo", "triple"]
        assert _route_misconception_to_tag("triples are X", tags) == "triple"


# ---------------------------------------------------------------------------
# Tie-breaking, fallbacks, edge cases.
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_no_overlap_returns_first_tag(self):
        tags = ["alpha", "beta", "gamma"]
        # "quokka" has zero overlap with any tag → fallback to alpha.
        assert (
            _route_misconception_to_tag("Some unrelated text about quokkas", tags)
            == "alpha"
        )

    def test_equal_score_breaks_to_first_in_list(self):
        # Both tags have one overlapping token → first wins.
        tags = ["screen-reader", "aria-label"]
        statement = "A screen reader can read aria labels automatically."
        # Both match 2 tokens. screen-reader is first → wins.
        assert _route_misconception_to_tag(statement, tags) == "screen-reader"

    def test_higher_score_beats_first_position(self):
        # alpha is first but overlaps zero; gamma overlaps two.
        tags = ["alpha", "beta", "gamma-delta"]
        statement = "the gamma delta term wins"
        assert _route_misconception_to_tag(statement, tags) == "gamma-delta"


class TestEdgeCases:
    def test_empty_tags_returns_none(self):
        assert _route_misconception_to_tag("any text", []) is None

    def test_empty_statement_returns_first_tag(self):
        assert _route_misconception_to_tag("", ["x", "y"]) == "x"

    def test_statement_only_stopwords_returns_first_tag(self):
        # "is the of and" → all stopwords → no signal → fallback.
        assert (
            _route_misconception_to_tag("is the of and", ["alpha", "beta"])
            == "alpha"
        )

    def test_tag_only_stopwords_skipped(self):
        # The "to" tag is pure stopword — must not pollute scoring.
        # The "owl" tag matches 'owl' in the statement → wins.
        tags = ["to", "owl"]
        assert (
            _route_misconception_to_tag("OWL is a description-logic vocab", tags)
            == "owl"
        )

    def test_multi_word_statement_picks_strongest_overlap(self):
        # gamma-related has 1 hit (gamma); alpha-beta-gamma has 1 hit
        # (gamma); single-letter alpha has 0. So gamma-related and
        # alpha-beta-gamma both score 1 — tie breaks to first in list.
        tags = ["alpha", "gamma-related", "alpha-beta-gamma"]
        # But statement also mentions alpha → gives alpha-beta-gamma 2 hits.
        statement = "alpha gamma considerations"
        assert (
            _route_misconception_to_tag(statement, tags) == "alpha-beta-gamma"
        )
