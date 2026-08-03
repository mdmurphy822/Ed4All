"""Tests for deterministic objective-derived vocabulary extraction."""

from __future__ import annotations

from Trainforge.alignment.outcomes import (
    auto_extract_vocabulary,
    build_auto_vocabularies,
    merged_vocabularies,
)


def test_extract_preserves_prefixed_and_camel_case_terms():
    terms = auto_extract_vocabulary(
        "Apply astro:OrbitNode with ThermalFlux telemetry."
    )
    assert "astro:OrbitNode" in terms
    assert "ThermalFlux" in terms


def test_extract_preserves_uppercase_and_hyphenated_terms():
    terms = auto_extract_vocabulary(
        "Analyze QUASAR readings in a sensor-specific workflow."
    )
    assert "QUASAR" in terms
    assert "sensor-specific" in terms


def test_leading_cognitive_verb_is_not_a_candidate():
    terms = auto_extract_vocabulary("Apply AlphaSignal calibration.")
    assert "Apply" not in terms
    assert "apply" not in terms


def test_empty_and_stopword_only_statements_are_empty():
    assert auto_extract_vocabulary("") == []
    assert auto_extract_vocabulary("the and or but") == []
    assert auto_extract_vocabulary(None) == []  # type: ignore[arg-type]


def test_extraction_is_deterministic_and_bounded():
    statement = " ".join(f"TOKEN{letter}" for letter in "ABCDEFGHIJKLMNO")
    first = auto_extract_vocabulary(statement)
    assert first == auto_extract_vocabulary(statement)
    assert len(first) <= 10


def test_technical_bigram_requires_two_technical_terms():
    terms = auto_extract_vocabulary("Apply astro:OrbitNode ThermalFlux safely.")
    assert "astro:OrbitNode ThermalFlux" in terms
    assert "ThermalFlux safely" not in terms


def test_builder_uses_only_supplied_component_objectives():
    objectives = {
        "component_objectives": [
            {"id": "objective-alpha", "statement": "Apply AlphaSignal."},
            {"id": "objective-beta", "statement": "Analyze BetaVector."},
        ]
    }
    vocabularies = build_auto_vocabularies(objectives)
    assert set(vocabularies) == {"objective-alpha", "objective-beta"}
    assert "AlphaSignal" in vocabularies["objective-alpha"]
    assert "BetaVector" in vocabularies["objective-beta"]


def test_builder_supports_nested_loader_shape():
    objectives = {
        "chapter_objectives": [
            {"id": "objective-one", "statement": "Apply GammaMatrix."},
            {
                "objectives": [
                    {"id": "objective-two", "statement": "Analyze DeltaFrame."}
                ]
            },
        ]
    }
    vocabularies = build_auto_vocabularies(objectives)
    assert set(vocabularies) == {"objective-one", "objective-two"}


def test_builder_includes_supplied_terminal_outcomes():
    objectives = {
        "terminal_outcomes": [
            {"id": "goal-main", "statement": "Integrate OmegaSystem."}
        ]
    }
    assert build_auto_vocabularies(objectives) == {
        "goal-main": ["OmegaSystem"]
    }


def test_empty_objectives_produce_no_vocabulary():
    assert build_auto_vocabularies(None) == {}
    assert build_auto_vocabularies({}) == {}
    assert merged_vocabularies(None) == {}
    assert merged_vocabularies({}) == {}


def test_merged_vocabulary_has_no_hidden_objective_ids():
    objectives = {
        "component_objectives": [
            {"id": "objective-only", "statement": "Apply UniqueSignal."}
        ]
    }
    assert merged_vocabularies(objectives) == {
        "objective-only": ["UniqueSignal"]
    }


def test_objective_without_matchable_terms_is_omitted():
    objectives = {
        "component_objectives": [
            {"id": "objective-empty", "statement": "Apply the correct result."}
        ]
    }
    assert build_auto_vocabularies(objectives) == {}
