"""Wave 84 tests: IDF-weighted tag overlap, chunk-type intent prior, and
retrieval-method preset resolution.

These signals are designed to be A/B'd against BM25 alone via the
``method=`` parameter on ``retrieve_chunks`` and the ``libv2
retrieval-compare`` CLI. The audit's empirical finding was that
chunk-type intent gives material lift on rdf-shacl-551-2 (+16% MRR over
BM25) while tag-IDF overlap is near-zero net positive — these tests pin
the *primitives* so future ranker work can build on a stable foundation.
"""

from __future__ import annotations

import math

import pytest

from LibV2.tools.libv2.retrieval_scoring import (
    BoostContributions,
    RETRIEVAL_METHOD_PRESETS,
    chunk_type_intent_prior,
    combine_bm25_with_boosts,
    compute_tag_idf,
    detect_query_intent,
    resolve_method_preset,
    tag_idf_overlap_score,
)


# ---------------------------------------------------------------------------
# compute_tag_idf
# ---------------------------------------------------------------------------


class TestComputeTagIdf:
    def test_rare_tag_gets_higher_idf_than_common(self):
        # 4 chunks: rdf in all 4, subclassof in 1.
        corpus = [
            ["rdf"],
            ["rdf", "shacl"],
            ["rdf", "owl"],
            ["rdf", "subclassof"],
        ]
        weights = compute_tag_idf(corpus)
        assert weights["subclassof"] > weights["rdf"]
        # rdf appears in every chunk → very small IDF (close to 0).
        assert weights["rdf"] < 0.5

    def test_empty_corpus_returns_empty(self):
        assert compute_tag_idf([]) == {}

    def test_corpus_with_empty_chunks(self):
        # Only one chunk has tags; the other two are empty. df=1, N=3.
        weights = compute_tag_idf([[], ["rdf"], []])
        assert "rdf" in weights
        # IDF formula: log((N+1)/(df+0.5)) = log(4/1.5) ≈ 0.98
        assert weights["rdf"] == pytest.approx(math.log(4 / 1.5), rel=1e-3)

    def test_duplicate_tags_within_chunk_dont_inflate_df(self):
        # df is per-chunk, so duplicate tags in the same chunk count once.
        weights = compute_tag_idf([["rdf", "rdf", "rdf"], ["rdf", "shacl"]])
        # rdf appears in 2 chunks of 2 → very low IDF.
        # shacl appears in 1 chunk of 2 → higher IDF.
        assert weights["shacl"] > weights["rdf"]

    def test_case_normalized_to_lower(self):
        weights = compute_tag_idf([["RDF"], ["rdf"]])
        # Both should resolve to the same key.
        assert "rdf" in weights
        assert "RDF" not in weights

    def test_non_string_tags_ignored(self):
        weights = compute_tag_idf([[None, "rdf", 42], ["rdf"]])
        assert "rdf" in weights
        assert None not in weights
        assert 42 not in weights


# ---------------------------------------------------------------------------
# tag_idf_overlap_score
# ---------------------------------------------------------------------------


class TestTagIdfOverlapScore:
    def test_overlap_of_rare_tag_scores_high(self):
        # In a corpus where 'subclassof' is rare, a chunk tagged with
        # subclassof + a query containing subclassof should score high.
        corpus = [["rdf"], ["rdf", "shacl"], ["rdf", "subclassof"]]
        weights = compute_tag_idf(corpus)
        chunk = {"concept_tags": ["rdf", "subclassof"]}
        score = tag_idf_overlap_score(chunk, ["subclassof"], weights)
        assert score > 0.5  # rare match should dominate the chunk's IDF mass

    def test_overlap_of_common_tag_scores_low(self):
        corpus = [["rdf"], ["rdf", "shacl"], ["rdf", "subclassof"]]
        weights = compute_tag_idf(corpus)
        chunk = {"concept_tags": ["rdf", "subclassof"]}
        # Querying with the common tag should give a much smaller score.
        score = tag_idf_overlap_score(chunk, ["rdf"], weights)
        assert 0 < score < 0.5

    def test_no_overlap_zero(self):
        weights = compute_tag_idf([["a"], ["b"]])
        chunk = {"concept_tags": ["a"]}
        assert tag_idf_overlap_score(chunk, ["c"], weights) == 0.0

    def test_empty_query_zero(self):
        weights = compute_tag_idf([["a"]])
        assert tag_idf_overlap_score({"concept_tags": ["a"]}, [], weights) == 0.0

    def test_empty_chunk_tags_zero(self):
        weights = compute_tag_idf([["a"]])
        assert tag_idf_overlap_score({}, ["a"], weights) == 0.0

    def test_empty_idf_weights_zero(self):
        # No corpus → no IDF map → score is 0.0 regardless of overlap.
        chunk = {"concept_tags": ["a"]}
        assert tag_idf_overlap_score(chunk, ["a"], {}) == 0.0

    def test_score_bounded_above_at_one(self):
        # Extreme case: a chunk's only tag matches the query.
        # Score is overlap_idf / chunk_idf_mass = 1.0 (capped).
        weights = {"rare": 5.0}
        chunk = {"concept_tags": ["rare"]}
        score = tag_idf_overlap_score(chunk, ["rare"], weights)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# detect_query_intent
# ---------------------------------------------------------------------------


class TestDetectQueryIntent:
    def test_what_is_detects_definition(self):
        intents = detect_query_intent("What is RDF?")
        assert "definition" in intents
        assert "explanation" in intents

    def test_explain_detects_explanation(self):
        assert "explanation" in detect_query_intent("Explain SPARQL queries")

    def test_example_detects_example(self):
        assert "example" in detect_query_intent("Show me an example of SHACL")

    def test_how_to_detects_procedure(self):
        intents = detect_query_intent("How to author a SHACL shape")
        assert "procedure" in intents

    def test_no_intent_returns_empty(self):
        # A noun-only query has no intent verb.
        assert detect_query_intent("subClassOf entailment") == set()

    def test_word_boundary_avoids_false_positive(self):
        # 'test' in 'context' must NOT trigger the test/quiz intent.
        # No verb / no phrase match at all.
        assert "assessment" not in detect_query_intent("context-aware retrieval")

    def test_empty_query_empty_intent(self):
        assert detect_query_intent("") == set()
        assert detect_query_intent(None) == set()  # type: ignore[arg-type]

    def test_multiple_intents_all_returned(self):
        # Query with both 'define' and 'example' verbs.
        intents = detect_query_intent("define and give an example of an IRI")
        assert "definition" in intents
        assert "example" in intents


# ---------------------------------------------------------------------------
# chunk_type_intent_prior
# ---------------------------------------------------------------------------


class TestChunkTypeIntentPrior:
    def test_chunk_type_matches_intent(self):
        chunk = {"chunk_type": "definition"}
        assert chunk_type_intent_prior(chunk, ["definition"]) == 1.0

    def test_chunk_type_does_not_match(self):
        chunk = {"chunk_type": "example"}
        assert chunk_type_intent_prior(chunk, ["definition"]) == 0.0

    def test_no_intents_returns_zero(self):
        # When the query carries no intent, the prior must NOT penalize
        # any chunk — it should just return 0 across the board.
        chunk = {"chunk_type": "definition"}
        assert chunk_type_intent_prior(chunk, set()) == 0.0

    def test_missing_chunk_type_zero(self):
        assert chunk_type_intent_prior({}, ["definition"]) == 0.0

    def test_case_insensitive_match(self):
        chunk = {"chunk_type": "Explanation"}
        assert chunk_type_intent_prior(chunk, ["explanation"]) == 1.0


# ---------------------------------------------------------------------------
# resolve_method_preset
# ---------------------------------------------------------------------------


class TestResolveMethodPreset:
    def test_bm25_preset_disables_metadata(self):
        preset = resolve_method_preset("bm25")
        assert preset["metadata_scoring"] is False

    def test_hybrid_preset_enables_all_signals(self):
        preset = resolve_method_preset("hybrid")
        assert preset["metadata_scoring"] is True
        assert preset["use_concept_graph_boost"] is True
        assert preset["use_lo_match_boost"] is True
        assert preset["use_targets_concept_boost"] is True
        assert preset["use_tag_idf_boost"] is True
        assert preset["use_chunk_type_intent_boost"] is True

    def test_bm25_intent_only_enables_intent(self):
        preset = resolve_method_preset("bm25+intent")
        assert preset["use_chunk_type_intent_boost"] is True
        assert preset["use_concept_graph_boost"] is False
        assert preset["use_tag_idf_boost"] is False

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown retrieval method preset"):
            resolve_method_preset("does-not-exist")

    def test_none_returns_empty_dict(self):
        # None means "no override", so callers preserve their defaults.
        assert resolve_method_preset(None) == {}

    def test_case_insensitive_lookup(self):
        # Method names are user-typed; should normalize.
        assert resolve_method_preset("HYBRID") == resolve_method_preset("hybrid")

    def test_all_documented_presets_resolve(self):
        # Pin the public method-name set so removal is intentional.
        for name in ["bm25", "bm25+graph", "bm25+intent", "bm25+tag", "hybrid"]:
            assert name in RETRIEVAL_METHOD_PRESETS
            assert resolve_method_preset(name)  # non-empty


# ---------------------------------------------------------------------------
# combine_bm25_with_boosts — Wave 84 boosts integrate with cap
# ---------------------------------------------------------------------------


class TestCombineWithWave84Boosts:
    def test_tag_idf_contribution_lifts_score(self):
        # Default tag_idf_overlap weight is 0.15.
        contrib = BoostContributions(tag_idf_overlap=1.0)
        final, capped = combine_bm25_with_boosts(10.0, contrib)
        assert final == pytest.approx(10.0 * (1 + 0.15))
        assert capped == pytest.approx(0.15)

    def test_chunk_type_intent_contribution_lifts_score(self):
        # Default chunk_type_intent weight is 0.25.
        contrib = BoostContributions(chunk_type_intent=1.0)
        final, capped = combine_bm25_with_boosts(10.0, contrib)
        assert final == pytest.approx(10.0 * (1 + 0.25))

    def test_all_wave84_boosts_respect_overall_cap(self):
        # If every signal saturates, MAX_TOTAL_BOOST=0.5 caps the lift.
        contrib = BoostContributions(
            concept_graph_overlap=1.0,
            lo_match=1.0,
            prereq_coverage=1.0,
            targets_concept=1.0,
            tag_idf_overlap=1.0,
            chunk_type_intent=1.0,
        )
        final, capped = combine_bm25_with_boosts(10.0, contrib)
        # Sum of all default weights >> 0.5; cap kicks in at 0.5.
        assert capped == pytest.approx(0.5)
        assert final == pytest.approx(15.0)
