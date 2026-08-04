"""Bounded ontology fallback regressions for synthesis objective focus."""

from Trainforge.synthesis.synthesis_eligibility import (
    focus_chunk_on_canonical_objective,
    pair_eligibility,
)


def _objective(statement: str, **extra: object) -> dict:
    return {
        "statement": statement,
        "bloom_level": "apply",
        **extra,
    }


def test_explicit_related_objective_can_focus_and_preserves_provenance() -> None:
    relation_provenance = {
        "artifact": "canonical-objectives",
        "assertion_id": "relation-7",
    }
    objectives = {
        "co-01": _objective(
            "Demonstrate fluency across the complete arithmetic curriculum.",
            ontology_relations=[{
                "type": "descendant",
                "target_id": "co-02",
                "provenance": relation_provenance,
            }],
        ),
        "co-02": _objective(
            "Multiply rational numbers using the sign rules.",
            provenance={"artifact": "canonical-objectives", "record": "co-02"},
            source_chunk_ids=["chunk-7"],
        ),
    }
    chunk = {
        "id": "chunk-7",
        "text": (
            "To multiply rational numbers, apply the sign rules: equal signs "
            "produce a positive product and different signs produce a "
            "negative product."
        ),
        "learning_outcome_refs": ["co-01"],
    }

    focused = focus_chunk_on_canonical_objective(
        chunk, seed=2, objectives=objectives,
    )

    assert focused["learning_outcome_refs"] == ["co-02"]
    focus = focused["synthesis_focus_objective"]
    assert focus["provenance"] == objectives["co-02"]["provenance"]
    assert focus["source_chunk_ids"] == ["chunk-7"]
    assert focus["ontology_relation"]["provenance"] == relation_provenance


def test_unrelated_vocabulary_collision_cannot_supply_objective() -> None:
    objectives = {
        "co-01": _objective(
            "Demonstrate fluency across the complete arithmetic curriculum.",
        ),
        # Strong lexical alignment is insufficient without an explicit
        # provenance-bearing ontology relation to the declared co-01.
        "co-99": _objective(
            "Multiply rational numbers using the sign rules.",
        ),
    }
    chunk = {
        "id": "collision",
        "text": (
            "Multiply rational numbers using the sign rules. Equal signs "
            "produce a positive result and different signs produce a "
            "negative result."
        ),
        "learning_outcome_refs": ["co-01"],
    }

    focused = focus_chunk_on_canonical_objective(
        chunk, seed=2, objectives=objectives,
    )

    assert focused["learning_outcome_refs"] == []
    assert focused["synthesis_focus_skip_reason"] == (
        "objective_content_obligations_not_evidenced"
    )
    assert not pair_eligibility(focused, kind="instruction").eligible


def test_relation_without_authoritative_provenance_is_not_a_fallback_edge() -> None:
    objectives = {
        "co-01": _objective(
            "Demonstrate fluency across the complete arithmetic curriculum.",
            ontology_relations=[{
                "type": "narrower",
                "target_id": "co-02",
            }],
        ),
        "co-02": _objective(
            "Multiply rational numbers using the sign rules.",
        ),
    }
    chunk = {
        "id": "unprovenanced-edge",
        "text": (
            "Multiply rational numbers using the sign rules. Equal signs "
            "produce a positive result and different signs produce a "
            "negative result."
        ),
        "learning_outcome_refs": ["co-01"],
    }

    focused = focus_chunk_on_canonical_objective(
        chunk, seed=2, objectives=objectives,
    )

    assert focused["learning_outcome_refs"] == []
