"""Outcome-alignment tests over arbitrary synthetic objectives."""

from __future__ import annotations

import copy

from Trainforge.alignment.outcomes import (
    build_parent_map,
    retag_chunk_outcomes,
)

OBJECTIVES = {
    "terminal_outcomes": [
        {"id": "goal-main", "statement": "Integrate AlphaSignal systems."},
    ],
    "component_objectives": [
        {
            "id": "objective-alpha",
            "parent_terminal": "goal-main",
            "statement": "Apply AlphaSignal calibration.",
        },
        {
            "id": "objective-beta",
            "parent_terminal": "goal-main",
            "statement": "Analyze BetaVector telemetry.",
        },
    ],
}


def _chunk(text: str, refs=None):
    return {
        "id": "synthetic-chunk",
        "text": text,
        "learning_outcome_refs": list(refs or []),
    }


def test_supplied_vocabulary_adds_only_its_objective():
    chunk = _chunk("The AlphaSignal reading is stable.")
    retag_chunk_outcomes(
        chunk,
        vocabularies={
            "objective-alpha": ["AlphaSignal"],
            "objective-beta": ["BetaVector"],
        },
    )
    assert chunk["learning_outcome_refs"] == ["objective-alpha"]


def test_no_supplied_vocabulary_adds_no_hidden_objective_ids():
    chunk = _chunk("AlphaSignal and BetaVector are both present.")
    retag_chunk_outcomes(chunk)
    assert chunk["learning_outcome_refs"] == []


def test_parent_rollup_is_additive():
    chunk = _chunk("Neutral text.", refs=["objective-alpha"])
    retag_chunk_outcomes(chunk, parent_map=build_parent_map(OBJECTIVES))
    assert chunk["learning_outcome_refs"] == ["objective-alpha", "goal-main"]


def test_vocabulary_match_and_parent_rollup_compose():
    chunk = _chunk("AlphaSignal is measured here.")
    retag_chunk_outcomes(
        chunk,
        parent_map=build_parent_map(OBJECTIVES),
        vocabularies={"objective-alpha": ["AlphaSignal"]},
    )
    assert chunk["learning_outcome_refs"] == ["objective-alpha", "goal-main"]


def test_existing_unrelated_references_are_preserved():
    chunk = _chunk("AlphaSignal is measured here.", refs=["objective-existing"])
    retag_chunk_outcomes(
        chunk,
        vocabularies={"objective-alpha": ["AlphaSignal"]},
    )
    assert chunk["learning_outcome_refs"] == [
        "objective-existing",
        "objective-alpha",
    ]


def test_retag_is_idempotent():
    chunk = _chunk("AlphaSignal is measured here.")
    kwargs = {
        "parent_map": build_parent_map(OBJECTIVES),
        "vocabularies": {"objective-alpha": ["AlphaSignal"]},
    }
    retag_chunk_outcomes(chunk, **kwargs)
    first = copy.deepcopy(chunk)
    retag_chunk_outcomes(chunk, **kwargs)
    assert chunk == first


def test_case_insensitive_dedup_preserves_first_casing():
    chunk = _chunk("AlphaSignal", refs=["OBJECTIVE-ALPHA"])
    retag_chunk_outcomes(
        chunk,
        vocabularies={"objective-alpha": ["AlphaSignal"]},
    )
    assert chunk["learning_outcome_refs"] == ["OBJECTIVE-ALPHA"]


def test_unmatched_vocabulary_does_not_collide():
    chunk = _chunk("AlphaSignal is measured here.")
    retag_chunk_outcomes(
        chunk,
        vocabularies={
            "objective-alpha": ["AlphaSignal"],
            "objective-beta": ["BetaVector"],
        },
    )
    assert "objective-beta" not in chunk["learning_outcome_refs"]


def test_no_change_preserves_the_entire_chunk():
    chunk = _chunk("No supplied term matches.", refs=["objective-existing"])
    snapshot = copy.deepcopy(chunk)
    assert retag_chunk_outcomes(chunk, parent_map={}, vocabularies={}) is chunk
    assert chunk == snapshot


def test_missing_reference_list_is_initialized_without_hidden_tags():
    chunk = {"id": "synthetic-chunk", "text": "AlphaSignal"}
    retag_chunk_outcomes(chunk)
    assert chunk["learning_outcome_refs"] == []


def test_build_parent_map_supports_canonical_shape():
    assert build_parent_map(OBJECTIVES) == {
        "objective-alpha": "goal-main",
        "objective-beta": "goal-main",
    }


def test_build_parent_map_supports_loader_shape():
    objectives = {
        "chapter_objectives": [
            {"id": "objective-one", "parent_to": "goal-one"},
            {
                "objectives": [
                    {"id": "objective-two", "parent_terminal": "goal-two"}
                ]
            },
        ]
    }
    assert build_parent_map(objectives) == {
        "objective-one": "goal-one",
        "objective-two": "goal-two",
    }


def test_build_parent_map_empty_inputs_are_empty():
    assert build_parent_map(None) == {}
    assert build_parent_map({}) == {}
