"""Phase 6 Subtask 15 tests for the concept-objective linker.

Targets :func:`lib.ontology.concept_objective_linker.link_concepts_to_objectives`.

Coverage contract per the plan
(``plans/phase6_abcd_concept_extractor.md`` Subtask 15): "Tests:
happy-path matches, no-match emits empty, empty graph no-op, multi-
match capture, preserves user-supplied keyConcepts." Verification
target: >=5 PASSED.
"""

from __future__ import annotations

import pytest

from lib.ontology.concept_objective_linker import link_concepts_to_objectives


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_concept_node(slug, label=None, cls="Concept"):
    """Build a concept-graph node mirroring ``Trainforge.pedagogy_graph_builder``.

    Pedagogy-graph emit format (line 786 of ``pedagogy_graph_builder.py``):
    ``id="concept:{slug}", class="Concept", label=...``.
    """
    return {
        "id": f"concept:{slug}",
        "class": cls,
        "label": label or slug.replace("-", " ").title(),
    }


def _make_graph(nodes, edges=None):
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-05-03T00:00:00",
        "nodes": list(nodes),
        "edges": list(edges or []),
    }


# ---------------------------------------------------------------------------
# Plan-cited test cases (>=5)
# ---------------------------------------------------------------------------


class TestHappyPath:
    """LOs whose statement text contains a concept slug get the concept linked."""

    def test_three_los_each_match_a_concept_node_via_statement_text(self):
        """Plan-cited #1: 3 LOs + 5 concept nodes -> all LOs get keyConcepts populated."""
        los = [
            {"id": "TO-01", "statement": "Identify cell parts in a microscope view."},
            {"id": "TO-02", "statement": "Compare property paths across SPARQL queries."},
            {"id": "CO-01", "statement": "Apply the framework to a new dataset."},
        ]
        graph = _make_graph([
            _make_concept_node("cell-parts", "Cell Parts"),
            _make_concept_node("property-paths", "Property Paths"),
            _make_concept_node("framework", "Framework"),
            _make_concept_node("microscope", "Microscope"),
            _make_concept_node("dataset", "Dataset"),
        ])

        out = link_concepts_to_objectives(los, graph)

        assert len(out) == 3
        # Pure transform: original list unchanged.
        assert "key_concepts" not in los[0]

        # TO-01 statement contains "cell parts" + "microscope" — both link.
        assert "cell-parts" in out[0]["key_concepts"]
        assert "microscope" in out[0]["key_concepts"]
        # TO-02 statement contains "property paths" — links.
        assert "property-paths" in out[1]["key_concepts"]
        # CO-01 statement contains "framework" + "dataset" — both link.
        assert "framework" in out[2]["key_concepts"]
        assert "dataset" in out[2]["key_concepts"]


class TestNoMatchProducesEmptyKeyConcepts:
    """When no concept-graph node matches, the LO carries an empty list."""

    def test_lo_with_no_matching_concept_emits_empty_key_concepts(self):
        """Plan-cited #2: LO with no matching concept -> empty keyConcepts list."""
        los = [{"id": "TO-01", "statement": "Discuss the history of the Gregorian calendar."}]
        graph = _make_graph([
            _make_concept_node("photosynthesis"),
            _make_concept_node("mitochondria"),
            _make_concept_node("ribosome"),
        ])

        out = link_concepts_to_objectives(los, graph)

        # No concept slug appears verbatim in the LO statement, so the
        # field is present but empty.
        assert out[0]["key_concepts"] == []


class TestEmptyGraphNoOp:
    """An empty / missing concept graph leaves LOs unchanged."""

    def test_empty_graph_no_op(self):
        """Plan-cited #3: concept-graph empty -> LOs unchanged."""
        los = [
            {"id": "TO-01", "statement": "Identify cell parts.", "key_concepts": ["foo"]},
            {"id": "TO-02", "statement": "Compare paths."},
        ]

        for empty in (None, {}, {"nodes": []}, {"nodes": "not-a-list"}):
            out = link_concepts_to_objectives(los, empty)
            assert len(out) == 2
            # User-supplied key_concepts on TO-01 preserved verbatim.
            assert out[0]["key_concepts"] == ["foo"]
            # TO-02 had no key_concepts and an empty graph — should NOT
            # gain a key_concepts field, so the runtime LO shape stays
            # identical to the input.
            assert "key_concepts" not in out[1]
            assert "keyConcepts" not in out[1]

    def test_non_list_objectives_returns_empty_list(self):
        """Defensive: non-list objectives input returns empty list."""
        assert link_concepts_to_objectives(None, _make_graph([])) == []
        assert link_concepts_to_objectives("not-a-list", _make_graph([])) == []


class TestMultipleMatches:
    """LOs whose statement carries multiple concepts capture all of them."""

    def test_multiple_concept_matches_in_one_lo(self):
        """Plan-cited #4: multiple matches -> all matches captured."""
        los = [{
            "id": "TO-01",
            "statement": (
                "Apply the framework to a property paths analysis using the dataset."
            ),
        }]
        graph = _make_graph([
            _make_concept_node("framework", "Framework"),
            _make_concept_node("property-paths", "Property Paths"),
            _make_concept_node("dataset", "Dataset"),
            _make_concept_node("microscope", "Microscope"),
        ])

        out = link_concepts_to_objectives(los, graph)

        # All three concepts whose labels appear in the statement are
        # linked. "microscope" is NOT linked (not in statement).
        assert set(out[0]["key_concepts"]) == {"framework", "property-paths", "dataset"}
        # Deterministic sorted ordering.
        assert out[0]["key_concepts"] == sorted(out[0]["key_concepts"])


class TestPreservesUserSuppliedConcepts:
    """User-supplied keyConcepts are merged, not overwritten."""

    def test_preserves_user_supplied_key_concepts(self):
        """Plan-cited #5: preserves user-supplied keyConcepts (merge not overwrite)."""
        los = [{
            "id": "TO-01",
            "statement": "Discuss property paths in SPARQL.",
            "key_concepts": ["sparql", "user-tag"],
        }]
        graph = _make_graph([
            _make_concept_node("property-paths", "Property Paths"),
            _make_concept_node("sparql-queries", "SPARQL queries"),
        ])

        out = link_concepts_to_objectives(los, graph)
        kc = out[0]["key_concepts"]

        # User-supplied entries preserved at the front in original order.
        assert kc[0] == "sparql"
        assert kc[1] == "user-tag"
        # property-paths added by Pass 2 (statement-text contains
        # "property paths"). sparql-queries added by Pass 1
        # (substring-relates "sparql" to "sparql-queries").
        assert "property-paths" in kc
        assert "sparql-queries" in kc

    def test_preserves_camelcase_keyconcepts_field_name(self):
        """JSON-LD form (camelCase) is preserved verbatim on emit."""
        los = [{
            "id": "TO-01",
            "statement": "Identify cell parts.",
            "keyConcepts": ["seed-tag"],
        }]
        graph = _make_graph([_make_concept_node("cell-parts", "Cell Parts")])

        out = link_concepts_to_objectives(los, graph)

        # camelCase preserved — runtime form NOT introduced.
        assert "keyConcepts" in out[0]
        assert "key_concepts" not in out[0]
        assert "seed-tag" in out[0]["keyConcepts"]
        assert "cell-parts" in out[0]["keyConcepts"]


# ---------------------------------------------------------------------------
# Symmetry / edge case (>5 floor)
# ---------------------------------------------------------------------------


class TestPureTransform:
    """The function is a pure transform — input dicts must not be mutated."""

    def test_input_objectives_dict_is_not_mutated(self):
        los = [{"id": "TO-01", "statement": "Identify framework usage."}]
        graph = _make_graph([_make_concept_node("framework", "Framework")])

        out = link_concepts_to_objectives(los, graph)

        # Output enriched...
        assert out[0]["key_concepts"] == ["framework"]
        # ...input untouched.
        assert "key_concepts" not in los[0]
        assert los[0] == {"id": "TO-01", "statement": "Identify framework usage."}


class TestSlugSubstringPass1:
    """Pass 1 substring-match enriches existing slugs from the graph."""

    def test_pass1_lo_ref_suffixed_concept_links_to_bare_slug(self):
        """Common case: LO has 'property-paths', graph has 'property-paths-co-15'."""
        los = [{
            "id": "TO-01",
            "statement": "(no statement-level matches here)",
            "key_concepts": ["property-paths"],
        }]
        graph = _make_graph([
            _make_concept_node("property-paths-co-15", "Property Paths (CO-15)"),
            _make_concept_node("property-paths-to-03", "Property Paths (TO-03)"),
        ])

        out = link_concepts_to_objectives(los, graph)
        kc = out[0]["key_concepts"]

        # User-supplied first.
        assert kc[0] == "property-paths"
        # Both LO-ref-suffixed variants linked via Pass 1.
        assert "property-paths-co-15" in kc
        assert "property-paths-to-03" in kc


class TestNonConceptNodesFiltered:
    """BloomLevel / Outcome / etc. nodes are NOT treated as concepts."""

    def test_bloomlevel_nodes_skipped(self):
        los = [{"id": "TO-01", "statement": "Apply the analyze framework to data."}]
        graph = _make_graph([
            # Concept node — should match.
            _make_concept_node("framework", "Framework"),
            # BloomLevel node — should NOT be linked even though
            # "analyze" appears in the statement text.
            {"id": "bloom:analyze", "class": "BloomLevel", "label": "Analyze"},
            # Outcome node — should NOT be linked.
            {"id": "TO-01", "class": "Outcome", "label": "TO-01"},
            # ComponentObjective — should NOT.
            {"id": "CO-04", "class": "ComponentObjective", "label": "CO-04"},
        ])

        out = link_concepts_to_objectives(los, graph)

        assert out[0]["key_concepts"] == ["framework"]


class TestDuplicateSlugsCollapse:
    """Duplicate concept references collapse cleanly."""

    def test_duplicates_collapse(self):
        los = [{
            "id": "TO-01",
            "statement": "Discuss property paths and property paths again.",
            "key_concepts": ["property-paths", "property-paths"],
        }]
        graph = _make_graph([
            _make_concept_node("property-paths", "Property Paths"),
        ])

        out = link_concepts_to_objectives(los, graph)
        # Even though the user-supplied list had a duplicate, AND the
        # statement matches multiple times, the result has a single
        # entry.
        assert out[0]["key_concepts"].count("property-paths") == 1


class TestDeterministicOrdering:
    """Two runs over identical inputs produce byte-identical output."""

    def test_same_inputs_produce_byte_identical_output(self):
        los = [{
            "id": "TO-01",
            "statement": (
                "Apply framework analysis to property paths in the dataset."
            ),
        }]
        graph = _make_graph([
            _make_concept_node("dataset"),
            _make_concept_node("framework"),
            _make_concept_node("property-paths"),
            _make_concept_node("microscope"),  # not in statement
        ])

        out_a = link_concepts_to_objectives(los, graph)
        out_b = link_concepts_to_objectives(los, graph)

        assert out_a == out_b
        # Sorted order — independent of graph-emit order.
        assert out_a[0]["key_concepts"] == ["dataset", "framework", "property-paths"]
