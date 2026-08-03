"""Arbitrary-course coverage tests for objective-derived alignment."""

from __future__ import annotations

from Trainforge.alignment.outcomes import (
    build_auto_vocabularies,
    build_parent_map,
    merged_vocabularies,
    retag_chunk_outcomes,
)


def _objectives():
    return {
        "terminal_outcomes": [
            {"id": "goal-systems", "statement": "Integrate OmegaSystem."},
        ],
        "component_objectives": [
            {
                "id": "objective-alpha",
                "parent_terminal": "goal-systems",
                "statement": "Apply AlphaSignal calibration.",
            },
            {
                "id": "objective-beta",
                "parent_terminal": "goal-systems",
                "statement": "Analyze BetaVector telemetry.",
            },
            {
                "id": "objective-gamma",
                "parent_terminal": "goal-systems",
                "statement": "Evaluate GammaMatrix stability.",
            },
        ],
    }


def test_merged_map_covers_only_supplied_matchable_objectives():
    objectives = _objectives()
    expected = {
        "goal-systems",
        "objective-alpha",
        "objective-beta",
        "objective-gamma",
    }
    assert set(merged_vocabularies(objectives)) == expected


def test_each_supplied_objective_has_a_matchable_derived_term():
    objectives = _objectives()
    vocabularies = merged_vocabularies(objectives)
    for objective in (
        objectives["terminal_outcomes"] + objectives["component_objectives"]
    ):
        assert vocabularies[objective["id"]]


def test_unrelated_empty_input_never_inherits_prior_objectives():
    assert merged_vocabularies(_objectives())
    assert merged_vocabularies({}) == {}
    assert merged_vocabularies(None) == {}


def test_unique_terms_retag_only_the_matching_objective():
    vocabularies = merged_vocabularies(_objectives())
    parent_map = build_parent_map(_objectives())
    cases = [
        ("AlphaSignal", "objective-alpha"),
        ("BetaVector", "objective-beta"),
        ("GammaMatrix", "objective-gamma"),
    ]
    for term, expected_id in cases:
        chunk = {"id": f"chunk-{expected_id}", "text": term}
        retag_chunk_outcomes(
            chunk,
            parent_map=parent_map,
            vocabularies=vocabularies,
        )
        assert chunk["learning_outcome_refs"] == [expected_id, "goal-systems"]


def test_specific_vocabulary_terms_do_not_mass_collide():
    vocabularies = build_auto_vocabularies(_objectives())
    component_ids = sorted(
        key for key in vocabularies if key.startswith("objective-")
    )
    for index, left in enumerate(component_ids):
        left_terms = {term.casefold() for term in vocabularies[left]}
        for right in component_ids[index + 1 :]:
            right_terms = {term.casefold() for term in vocabularies[right]}
            assert left_terms.isdisjoint(right_terms)


def test_existing_references_survive_alignment_and_repeat_runs():
    objectives = _objectives()
    kwargs = {
        "parent_map": build_parent_map(objectives),
        "vocabularies": merged_vocabularies(objectives),
    }
    chunk = {
        "id": "chunk-alpha",
        "text": "AlphaSignal",
        "learning_outcome_refs": ["objective-existing"],
    }
    retag_chunk_outcomes(chunk, **kwargs)
    first = list(chunk["learning_outcome_refs"])
    retag_chunk_outcomes(chunk, **kwargs)
    assert chunk["learning_outcome_refs"] == first
    assert first == ["objective-existing", "objective-alpha", "goal-systems"]
