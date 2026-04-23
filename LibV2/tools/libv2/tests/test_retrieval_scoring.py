"""Worker J tests: the three metadata-aware boost functions in isolation,
the combine helper, and the loader fallbacks."""

from __future__ import annotations

import pytest

from LibV2.tools.libv2.retrieval_scoring import (
    MAX_TOTAL_BOOST,
    BoostContributions,
    combine_bm25_with_boosts,
    concept_graph_overlap_boost,
    extract_query_concepts,
    lo_match_boost,
    load_concept_graph_node_ids,
    load_course_outcomes,
    load_pedagogy_model,
    prereq_coverage_boost,
)

# ---------------------------------------------------------------------------
# concept_graph_overlap_boost
# ---------------------------------------------------------------------------

class TestConceptGraphOverlap:
    def test_jaccard_on_overlap(self):
        chunk = {"concept_tags": ["aria", "pour", "landmark"]}
        q = {"aria", "pour"}
        # Intersection=2, Union=3 → 2/3
        s = concept_graph_overlap_boost(chunk, q)
        assert s == pytest.approx(2 / 3)

    def test_no_overlap_zero(self):
        assert concept_graph_overlap_boost({"concept_tags": ["x"]}, ["y"]) == 0.0

    def test_empty_query_zero(self):
        assert concept_graph_overlap_boost({"concept_tags": ["x"]}, []) == 0.0

    def test_empty_chunk_tags_zero(self):
        assert concept_graph_overlap_boost({}, ["x"]) == 0.0


# ---------------------------------------------------------------------------
# extract_query_concepts (bigram expansion)
# ---------------------------------------------------------------------------

class TestExtractQueryConcepts:
    def test_single_token_match(self):
        nodes = {"aria", "pour"}
        assert extract_query_concepts("aria", nodes) == {"aria"}

    def test_bigram_expansion_finds_hyphenated_node(self):
        nodes = {"color-contrast", "focus-indicator"}
        concepts = extract_query_concepts("body color contrast matters", nodes)
        assert "color-contrast" in concepts

    def test_hyphenated_query_token_direct_match(self):
        nodes = {"aria-labelledby"}
        concepts = extract_query_concepts("use aria-labelledby correctly", nodes)
        assert "aria-labelledby" in concepts

    def test_empty_graph_empty_return(self):
        assert extract_query_concepts("anything", set()) == set()


# ---------------------------------------------------------------------------
# lo_match_boost
# ---------------------------------------------------------------------------

class TestLoMatchBoost:
    def test_explicit_filter_returns_1(self):
        chunk = {"learning_outcome_refs": ["co-03", "co-05"]}
        assert lo_match_boost(chunk, "", [], explicit_lo_filter=["co-03"]) == 1.0

    def test_id_in_query_text_returns_1(self):
        chunk = {"learning_outcome_refs": ["co-03"]}
        assert lo_match_boost(chunk, "see co-03 for details", []) == 1.0

    def test_statement_overlap_returns_07(self):
        chunk = {"learning_outcome_refs": ["co-01"]}
        outcomes = [
            {"id": "co-01", "statement": "accessibility of color contrast in web design"}
        ]
        score = lo_match_boost(chunk, "color contrast accessibility", outcomes)
        assert score == 0.7

    def test_low_statement_overlap_zero(self):
        chunk = {"learning_outcome_refs": ["co-01"]}
        outcomes = [{"id": "co-01", "statement": "completely unrelated topic"}]
        assert lo_match_boost(chunk, "color contrast", outcomes) == 0.0

    def test_no_refs_zero(self):
        assert lo_match_boost({}, "query", []) == 0.0


# ---------------------------------------------------------------------------
# prereq_coverage_boost
# ---------------------------------------------------------------------------

class TestPrereqCoverageBoost:
    def test_all_covered_positive(self):
        chunk = {"prereq_concepts": ["aria", "pour"]}
        model = {"prerequisite_chain": [
            {"concept": "aria"}, {"concept": "pour"}, {"concept": "other"},
        ]}
        assert prereq_coverage_boost(chunk, model) == 0.7

    def test_violation_negative(self):
        chunk = {"prereq_concepts": ["aria"]}
        model = {
            "prerequisite_chain": [],
            "prerequisite_violations": [{"concept": "aria"}],
        }
        assert prereq_coverage_boost(chunk, model) == -0.5

    def test_partial_coverage_zero(self):
        chunk = {"prereq_concepts": ["aria", "unknown"]}
        model = {"prerequisite_chain": [{"concept": "aria"}]}
        assert prereq_coverage_boost(chunk, model) == 0.0

    def test_no_prereqs_zero(self):
        assert prereq_coverage_boost({}, {"prerequisite_chain": []}) == 0.0


# ---------------------------------------------------------------------------
# combine_bm25_with_boosts
# ---------------------------------------------------------------------------

class TestCombineBoosts:
    def test_positive_boosts_lift_score(self):
        contrib = BoostContributions(concept_graph_overlap=1.0, lo_match=1.0)
        final, capped = combine_bm25_with_boosts(10.0, contrib)
        # weight sum = 0.3 + 0.3 = 0.6, capped to 0.5 → final = 15.0
        assert final == pytest.approx(15.0)
        assert capped == pytest.approx(0.5)

    def test_cap_enforced_at_max_total_boost(self):
        contrib = BoostContributions(
            concept_graph_overlap=1.0, lo_match=1.0, prereq_coverage=1.0,
        )
        _, capped = combine_bm25_with_boosts(1.0, contrib)
        assert abs(capped) <= MAX_TOTAL_BOOST + 1e-9

    def test_prereq_violation_reduces_score(self):
        contrib = BoostContributions(prereq_coverage=-1.0)
        final, capped = combine_bm25_with_boosts(10.0, contrib)
        # -1.0 * 0.2 = -0.2 → final = 10 * 0.8 = 8.0
        assert final == pytest.approx(8.0)
        assert capped == pytest.approx(-0.2)

    def test_final_never_negative(self):
        contrib = BoostContributions(prereq_coverage=-10.0)  # pathological
        final, _ = combine_bm25_with_boosts(1.0, contrib)
        assert final >= 0.0

    def test_custom_weights_override(self):
        contrib = BoostContributions(concept_graph_overlap=1.0)
        _, capped = combine_bm25_with_boosts(
            1.0, contrib, weights={"concept_graph_overlap": 0.1},
        )
        assert capped == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Loader graceful degradation
# ---------------------------------------------------------------------------

class TestLoaderFallbacks:
    def test_missing_graph_returns_empty(self, tmp_path):
        assert load_concept_graph_node_ids(tmp_path) == set()

    def test_missing_outcomes_returns_empty(self, tmp_path):
        assert load_course_outcomes(tmp_path) == []

    def test_missing_pedagogy_returns_empty_dict(self, tmp_path):
        assert load_pedagogy_model(tmp_path) == {}

    def test_valid_graph_loads_node_ids(self, tmp_path):
        (tmp_path / "graph").mkdir()
        import json
        (tmp_path / "graph" / "concept_graph.json").write_text(json.dumps({
            "kind": "concept",
            "nodes": [
                {"id": "a", "label": "A", "frequency": 3},
                {"id": "b", "label": "B", "frequency": 2},
            ],
            "edges": [],
        }))
        node_ids = load_concept_graph_node_ids(tmp_path)
        assert node_ids == {"a", "b"}
