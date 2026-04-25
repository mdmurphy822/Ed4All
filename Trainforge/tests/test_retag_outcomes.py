"""Wave 76: vocabulary retag + parent-outcome rollup tests.

Locks in the contract that:

* Vocabulary terms in chunk text trigger additive CO retags
  (co-18, co-19, co-22 are the live coverage gaps the rule targets).
* Parent rollup adds the terminal id for every component id present.
* Chunks with no vocabulary matches and no parent-mapped CO ids are
  unchanged.
* The retag pass is idempotent — running it twice is a no-op.

The tests intentionally stay at the pure-data layer
(``Trainforge.retag_outcomes``) rather than spinning up a full
``CourseProcessor``; the emit-time wiring is exercised separately by
the integration test path that runs the retroactive script and
checks coverage.
"""

from __future__ import annotations

import copy

from Trainforge.retag_outcomes import (
    RETAG_VOCABULARIES,
    build_parent_map,
    retag_chunk_outcomes,
)


# ---- Fixtures -------------------------------------------------------

# Mirror the rdf-shacl-550-rdf-shacl-550 objectives.json shape so the
# parent-rollup contract is tested against the real mapping.
OBJECTIVES = {
    "schema_version": "v1",
    "course_code": "RDF_SHACL_550",
    "terminal_outcomes": [
        {"id": "to-04", "statement": "..."},
        {"id": "to-05", "statement": "..."},
        {"id": "to-06", "statement": "..."},
        {"id": "to-07", "statement": "..."},
    ],
    "component_objectives": [
        {"id": "co-18", "parent_terminal": "to-04", "statement": "..."},
        {"id": "co-19", "parent_terminal": "to-04", "statement": "..."},
        {"id": "co-22", "parent_terminal": "to-05", "statement": "..."},
        {"id": "co-25", "parent_terminal": "to-06", "statement": "..."},
        {"id": "co-26", "parent_terminal": "to-06", "statement": "..."},
        {"id": "co-27", "parent_terminal": "to-07", "statement": "..."},
        {"id": "co-28", "parent_terminal": "to-07", "statement": "..."},
        {"id": "co-29", "parent_terminal": "to-07", "statement": "..."},
    ],
}


def _chunk(text: str, refs=None):
    return {
        "id": "test_chunk",
        "text": text,
        "learning_outcome_refs": list(refs or []),
    }


# ---- Vocabulary retag rule -----------------------------------------

def test_co18_vocabulary_minCount_triggers_retag():
    chunk = _chunk(
        "The shape declares sh:minCount 1 to require at least one value."
    )
    retag_chunk_outcomes(chunk)
    assert "co-18" in chunk["learning_outcome_refs"]


def test_co18_vocabulary_multiple_terms_only_added_once():
    chunk = _chunk(
        "The constraint uses sh:minCount, sh:maxCount, and sh:datatype."
    )
    retag_chunk_outcomes(chunk)
    # co-18 appears exactly once even though three vocabulary terms hit.
    assert chunk["learning_outcome_refs"].count("co-18") == 1


def test_co19_validation_report_triggers_retag():
    chunk = _chunk(
        "The validation report carries sh:result entries for each violation."
    )
    retag_chunk_outcomes(chunk)
    assert "co-19" in chunk["learning_outcome_refs"]


def test_co22_shacl_sparql_triggers_retag():
    chunk = _chunk(
        "SHACL-SPARQL constraints extend SHACL Core via sh:sparql clauses."
    )
    retag_chunk_outcomes(chunk)
    assert "co-22" in chunk["learning_outcome_refs"]


def test_vocabulary_retag_is_additive_does_not_remove_existing():
    chunk = _chunk(
        "Discusses sh:minCount.",
        refs=["co-16", "co-17"],
    )
    retag_chunk_outcomes(chunk)
    # Existing tags preserved; co-18 added.
    assert "co-16" in chunk["learning_outcome_refs"]
    assert "co-17" in chunk["learning_outcome_refs"]
    assert "co-18" in chunk["learning_outcome_refs"]


def test_vocabulary_keys_match_real_co_ids():
    # Guard against typos creeping into the vocabulary table.
    for co_id in RETAG_VOCABULARIES:
        assert co_id.startswith("co-")
        assert RETAG_VOCABULARIES[co_id], f"empty vocabulary for {co_id}"


# ---- Parent rollup rule --------------------------------------------

def test_parent_rollup_adds_terminal_for_co25():
    parent_map = build_parent_map(OBJECTIVES)
    chunk = _chunk("Capstone integration material.", refs=["co-25"])
    retag_chunk_outcomes(chunk, parent_map=parent_map)
    assert "to-06" in chunk["learning_outcome_refs"]


def test_parent_rollup_adds_to07_for_co27_co28_co29():
    parent_map = build_parent_map(OBJECTIVES)
    for co in ("co-27", "co-28", "co-29"):
        chunk = _chunk("Capstone work.", refs=[co])
        retag_chunk_outcomes(chunk, parent_map=parent_map)
        assert "to-07" in chunk["learning_outcome_refs"], (
            f"{co} did not roll up to to-07"
        )


def test_parent_rollup_runs_after_vocabulary_retag():
    # Chunk text triggers co-18 (vocabulary), which then rolls up to to-04.
    parent_map = build_parent_map(OBJECTIVES)
    chunk = _chunk("Defines sh:minCount on the property shape.")
    retag_chunk_outcomes(chunk, parent_map=parent_map)
    assert "co-18" in chunk["learning_outcome_refs"]
    assert "to-04" in chunk["learning_outcome_refs"]


def test_parent_rollup_without_parent_map_is_noop():
    # Vocabulary retag still runs; parent rollup silently skipped.
    chunk = _chunk("Defines sh:minCount.", refs=["co-25"])
    retag_chunk_outcomes(chunk, parent_map=None)
    assert "co-18" in chunk["learning_outcome_refs"]
    assert "co-25" in chunk["learning_outcome_refs"]
    assert "to-06" not in chunk["learning_outcome_refs"]


# ---- No-match and idempotency --------------------------------------

def test_chunk_with_no_vocabulary_matches_no_parent_unchanged():
    parent_map = build_parent_map(OBJECTIVES)
    chunk = _chunk(
        "An RDF graph is a set of triples comprising subject, predicate, object.",
        refs=["co-01"],
    )
    expected = list(chunk["learning_outcome_refs"])
    # co-01 isn't in OBJECTIVES so parent map can't roll it up either.
    retag_chunk_outcomes(chunk, parent_map=parent_map)
    assert chunk["learning_outcome_refs"] == expected


def test_idempotent_running_twice_does_not_double_add():
    parent_map = build_parent_map(OBJECTIVES)
    chunk = _chunk(
        "sh:minCount appears alongside sh:maxCount.",
        refs=["co-25"],
    )
    retag_chunk_outcomes(chunk, parent_map=parent_map)
    snapshot = list(chunk["learning_outcome_refs"])
    retag_chunk_outcomes(chunk, parent_map=parent_map)
    assert chunk["learning_outcome_refs"] == snapshot


def test_dedup_case_insensitive_preserves_first_seen_casing():
    chunk = _chunk("sh:minCount", refs=["CO-18"])
    retag_chunk_outcomes(chunk)
    # The vocabulary rule wants to add "co-18" but case-insensitive
    # dedup keeps the existing "CO-18" entry only.
    assert chunk["learning_outcome_refs"] == ["CO-18"]


def test_returns_chunk_for_chaining():
    chunk = _chunk("plain text with no vocabulary terms")
    out = retag_chunk_outcomes(chunk)
    assert out is chunk


def test_handles_missing_learning_outcome_refs_field():
    chunk = {"id": "x", "text": "sh:minCount appears here."}
    retag_chunk_outcomes(chunk)
    assert chunk["learning_outcome_refs"] == ["co-18"]


def test_build_parent_map_handles_chapter_objectives_shape():
    # Loader-shape input with chapter_objectives[].
    legacy = {
        "chapter_objectives": [
            {"id": "co-A", "parent_to": "to-X"},
            {"objectives": [{"id": "co-B", "parent_terminal": "to-Y"}]},
        ]
    }
    pm = build_parent_map(legacy)
    assert pm.get("co-a") == "to-x"
    assert pm.get("co-b") == "to-y"


def test_build_parent_map_empty_when_no_objectives():
    assert build_parent_map(None) == {}
    assert build_parent_map({}) == {}


def test_retag_does_not_mutate_input_when_no_changes():
    chunk = _chunk("nothing to match", refs=["co-99"])
    snapshot = copy.deepcopy(chunk)
    retag_chunk_outcomes(chunk, parent_map={})
    assert chunk == snapshot
