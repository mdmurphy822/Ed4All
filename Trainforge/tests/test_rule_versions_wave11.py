"""Wave 11 — rule_versions map exposes bumped versions for 5 touched rules.

Each of the 5 chunk-anchored rule modules bumps ``RULE_VERSION`` from 1 to 2
unconditionally (regardless of ``TRAINFORGE_SOURCE_PROVENANCE`` state — the
version reflects the schema-generation shift, not runtime emit state).

The orchestrator (`Trainforge/rag/typed_edge_inference.py::build_semantic_graph`)
collects per-rule versions into ``artifact["rule_versions"]``. This suite
locks in:

- Each touched rule's module-level ``RULE_VERSION == 2``.
- Each untouched rule's module-level ``RULE_VERSION == 1`` (P4 abstract-arm
  deferral).
- The orchestrator's ``rule_versions`` map exposes the bumped values.
- Flag state (on/off) does NOT change the emitted version (the version is
  unconditional; only the payload is gated).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.rag.inference_rules import (
    assesses_from_question_lo,
    defined_by_from_first_mention,
    derived_from_lo_ref,
    exemplifies_from_example_chunks,
    is_a_from_key_terms,
    misconception_of_from_misconception_ref,
    prerequisite_from_lo_order,
    related_from_cooccurrence,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph

WAVE_11_TOUCHED = {
    "is_a_from_key_terms": is_a_from_key_terms,
    "exemplifies_from_example_chunks": exemplifies_from_example_chunks,
    "derived_from_lo_ref": derived_from_lo_ref,
    "defined_by_from_first_mention": defined_by_from_first_mention,
    "assesses_from_question_lo": assesses_from_question_lo,
}

WAVE_11_UNTOUCHED = {
    "prerequisite_from_lo_order": prerequisite_from_lo_order,
    "related_from_cooccurrence": related_from_cooccurrence,
    "misconception_of_from_misconception_ref": misconception_of_from_misconception_ref,
}


# --------------------------------------------------------------------- #
# Module-level RULE_VERSION constants
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rule_name,rule_mod", sorted(WAVE_11_TOUCHED.items())
)
def test_wave11_touched_rule_version_bumped_to_2(rule_name, rule_mod):
    assert rule_mod.RULE_VERSION == 2, (
        f"{rule_name} RULE_VERSION must be bumped to 2 for Wave 11"
    )


@pytest.mark.parametrize(
    "rule_name,rule_mod", sorted(WAVE_11_UNTOUCHED.items())
)
def test_wave11_untouched_rule_version_still_1(rule_name, rule_mod):
    assert rule_mod.RULE_VERSION == 1, (
        f"{rule_name} RULE_VERSION must remain 1 (Wave 11 P4 deferral)"
    )


def test_rule_name_matches_module():
    """Sanity: each module's RULE_NAME matches the expected identifier."""
    for name, mod in WAVE_11_TOUCHED.items():
        assert mod.RULE_NAME == name
    for name, mod in WAVE_11_UNTOUCHED.items():
        assert mod.RULE_NAME == name


# --------------------------------------------------------------------- #
# Orchestrator rule_versions map
# --------------------------------------------------------------------- #


def _minimal_semantic_graph_inputs():
    """Build chunks + concept_graph + questions that exercise all 8 rules."""
    chunks = [
        {
            "id": "chunk_01",
            "source": {
                "course_id": "SAMPLE_101",
                "module_id": "m",
                "lesson_id": "l",
            },
            "key_terms": [
                {
                    "term": "cognitive load",
                    "definition": "Cognitive load is a type of mental effort.",
                }
            ],
            "learning_outcome_refs": ["to-01"],
            "concept_tags": ["cognitive-load"],
            "chunk_type": "example",
        },
    ]
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "cognitive-load",
                "label": "cognitive-load",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
            {
                "id": "mental-effort",
                "label": "mental-effort",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
        ],
    }
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_01"}
    ]
    return chunks, concept_graph, questions


def test_orchestrator_rule_versions_includes_wave11_bumps():
    chunks, concept_graph, questions = _minimal_semantic_graph_inputs()
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )
    versions = artifact["rule_versions"]
    # All 5 Wave-11 rules should appear at version 2.
    for rule_name in WAVE_11_TOUCHED:
        assert versions.get(rule_name) == 2, (
            f"orchestrator did not surface Wave 11 bump for {rule_name}: {versions}"
        )


def test_orchestrator_rule_versions_keeps_untouched_at_1():
    chunks, concept_graph, questions = _minimal_semantic_graph_inputs()
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )
    versions = artifact["rule_versions"]
    for rule_name in WAVE_11_UNTOUCHED:
        assert versions.get(rule_name) == 1, (
            f"{rule_name} version drifted — expected 1, got {versions.get(rule_name)}"
        )


def test_orchestrator_rule_versions_sorted_keys():
    """Map preserves sorted-keys invariant for deterministic output."""
    chunks, concept_graph, questions = _minimal_semantic_graph_inputs()
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )
    keys = list(artifact["rule_versions"].keys())
    assert keys == sorted(keys), (
        f"rule_versions keys must be sorted for determinism; got {keys}"
    )


def test_flag_state_does_not_change_rule_version_emission(monkeypatch):
    """Flag on / off must NOT change the version emission — the bump is
    unconditional."""
    for flag_value in (True, False):
        for mod in WAVE_11_TOUCHED.values():
            monkeypatch.setattr(mod, "SOURCE_PROVENANCE", flag_value)
        for mod in WAVE_11_TOUCHED.values():
            assert mod.RULE_VERSION == 2


def test_rule_version_is_integer_greater_than_zero():
    """Schema contract: RULE_VERSION must be a positive int."""
    for mod in WAVE_11_TOUCHED.values():
        assert isinstance(mod.RULE_VERSION, int)
        assert mod.RULE_VERSION > 0
    for mod in WAVE_11_UNTOUCHED.values():
        assert isinstance(mod.RULE_VERSION, int)
        assert mod.RULE_VERSION > 0
