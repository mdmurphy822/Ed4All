"""Tests for the Wave 137 followup semantic profile layer.

Five surfaces under test:

1. The shipped ``rdf_type_instanceof`` profile loads from the canonical
   YAML path with the expected target_curie + min_good_signals.
2. ``evaluate_profile`` flags BAD signals and missing GOOD signals on a
   semantically-wrong rdf:type entry (the failure mode the auto-redraft
   loop on rdf:type was producing).
3. ``evaluate_profile`` returns no violations on a semantically-correct
   rdf:type entry.
4. ``validate_form_data_contract`` propagates SEMANTIC_* violations into
   ``content_violations`` when invoked with ``semantic_profile=...``.
5. ``draft_form_data_entry._build_drafting_prompt`` prepends the
   profile's ``prompt_directive`` above the structural rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from lib.ontology.semantic_profiles import (  # noqa: E402
    SemanticProfile,
    evaluate_profile,
    load_semantic_profile,
    load_semantic_profiles,
)
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SurfaceFormData,
    validate_form_data_contract,
)


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------


def test_rdf_type_profile_loads_from_canonical_path():
    profiles = load_semantic_profiles()
    assert "rdf_type_instanceof" in profiles, (
        f"shipped profile is missing; got {sorted(profiles.keys())}"
    )
    p = profiles["rdf_type_instanceof"]
    assert isinstance(p, SemanticProfile)
    assert p.target_curie == "rdf:type"
    assert p.min_good_signals >= 2
    assert any("instance of" in g for g in p.good_signals)
    assert any("rdfs:domain" in b or "domain of rdf:type" in b
               for b in p.bad_signals)
    assert "instance" in p.prompt_directive.lower()


def test_load_semantic_profile_unknown_name_raises():
    with pytest.raises(KeyError):
        load_semantic_profile("nope_does_not_exist")


# ----------------------------------------------------------------------
# evaluate_profile
# ----------------------------------------------------------------------


def test_evaluate_profile_flags_bad_signals_on_rdfs_domain_confusion():
    """The exact failure mode the auto-redraft loop on rdf:type was
    producing: the model rewrote rdf:type as if it defined the domain /
    range of a property."""
    p = load_semantic_profile("rdf_type_instanceof")
    defs = [
        "rdf:type defines the domain of a property in RDFS contexts.",
        "rdf:type defines the range of a property when constraints apply.",
    ]
    usage = [
        ("How is rdf:type used?",
         "rdf:type targets a class to constrain values."),
    ]
    violations = evaluate_profile(p, definitions=defs, usage_examples=usage)
    codes = {v["code"] for v in violations}
    assert "SEMANTIC_BAD_SIGNAL" in codes
    # And the minimum good-signals check should also fire because the
    # entry contains zero "instance of" / "member of" references.
    assert "SEMANTIC_INSUFFICIENT_GOOD_SIGNALS" in codes


def test_evaluate_profile_passes_on_instance_of_semantics():
    p = load_semantic_profile("rdf_type_instanceof")
    defs = [
        "rdf:type asserts that a subject is an instance of a class.",
        "In a triple ?s rdf:type ?C, the subject becomes a member of class ?C.",
        "rdf:type is a class membership predicate connecting an instance to its class.",
    ]
    usage = [
        ("What does rdf:type mean?",
         "?resource rdf:type ?Class . means the resource is a member of the class."),
    ]
    violations = evaluate_profile(p, definitions=defs, usage_examples=usage)
    assert violations == [], (
        f"clean entry should pass; got {violations}"
    )


def test_evaluate_profile_flags_missing_required_term_in_definitions():
    p = load_semantic_profile("rdf_type_instanceof")
    # No "instance" anywhere in defs (despite using "member of" + good
    # signals).
    defs = [
        "rdf:type marks a member of a class.",
        "?subject rdf:type ?Class identifies the subject as a class member.",
    ]
    usage = [
        ("Q?", "rdf:type ?Class . — the subject is a class member."),
    ]
    violations = evaluate_profile(p, definitions=defs, usage_examples=usage)
    codes = {v["code"] for v in violations}
    assert "SEMANTIC_MISSING_REQUIRED_TERM_DEFINITIONS" in codes


# ----------------------------------------------------------------------
# Contract-validator integration
# ----------------------------------------------------------------------


def _build_complete_rdf_type_entry(*, semantically_wrong: bool) -> SurfaceFormData:
    """Build a SurfaceFormData that passes Wave 136b structural rules
    but is semantically wrong (or right) per the rdf_type profile."""
    if semantically_wrong:
        defs = [
            "rdf:type defines the domain of a property in RDFS specifications.",
            "rdf:type defines the range of a property when constraints apply.",
            "rdf:type is a property shape used to constrain values declaratively.",
            "rdf:type targets a class for instance validation purposes here.",
            "rdf:type is similar to sh:class in modern SHACL validation contexts.",
            "rdf:type is used to constrain values during property shape evaluation.",
            "rdf:type defines the domain when applied to a property predicate.",
        ]
        usage = [
            (
                "Question prompt about how rdf:type is used in RDFS-flavored "
                "constraint declarations across the W3C ecosystem in practice.",
                "Answer body where rdf:type defines the domain of a predicate "
                "and constrains the range to a specific class hierarchy in "
                "the way SHACL property shapes operate over typed values.",
            ),
        ] * 7
    else:
        defs = [
            "rdf:type asserts that a subject is an instance of a class in a triple.",
            "In ?s rdf:type ?C, the subject ?s becomes a member of class ?C explicitly.",
            "rdf:type is the class membership predicate, an instance of typing in RDF.",
            "When ?resource rdf:type ?Class appears, the resource is a class instance.",
            "rdf:type asserts class membership for any instance of a typed concept.",
            "Instance assertions with rdf:type place the subject as a member of a class.",
            "rdf:type is a class predicate; the subject is a class instance under it.",
        ]
        usage = [
            (
                "How does rdf:type assert class membership for an instance "
                "of a class within a typical RDF triple expression in TTL?",
                "An instance assertion ?resource rdf:type ?Class . places the "
                "resource as a member of the class; the subject is an instance "
                "of the class ?Class in the asserted triple.",
            ),
        ] * 7
    return SurfaceFormData(
        curie="rdf:type",
        short_name="type",
        anchored_status="complete",
        definitions=defs,
        usage_examples=usage,
        provenance={
            "provider": "test",
            "generated_by": "test",
            "reviewed_by": "@test-handle",
            "prompt_version": "v1",
            "timestamp": "2026-05-01T00:00:00Z",
        },
    )


def test_validate_form_data_contract_emits_semantic_violations_when_profile_passed():
    profile = load_semantic_profile("rdf_type_instanceof")
    form_data = {
        "rdf:type": _build_complete_rdf_type_entry(semantically_wrong=True),
    }
    report = validate_form_data_contract(
        form_data,
        ["rdf:type"],
        semantic_profile=profile,
    )
    codes = {
        v.get("code") for v in report.get("content_violations", []) or []
    }
    assert "SEMANTIC_BAD_SIGNAL" in codes, (
        f"profile rules did not propagate; got codes={codes}"
    )


def test_validate_form_data_contract_no_semantic_violations_on_clean_entry():
    profile = load_semantic_profile("rdf_type_instanceof")
    form_data = {
        "rdf:type": _build_complete_rdf_type_entry(semantically_wrong=False),
    }
    report = validate_form_data_contract(
        form_data,
        ["rdf:type"],
        semantic_profile=profile,
    )
    semantic_codes = {
        v.get("code") for v in report.get("content_violations", []) or []
        if str(v.get("code", "")).startswith("SEMANTIC_")
    }
    assert semantic_codes == set(), (
        f"clean entry should not trip SEMANTIC_* rules; got {semantic_codes}"
    )


def test_validate_form_data_contract_skips_profile_on_degraded_entry():
    """When the target_curie's entry is degraded_placeholder, profile
    rules MUST NOT fire (degraded entries are intentionally stubbed
    and the profile contract only applies once the operator has
    drafted real content)."""
    profile = load_semantic_profile("rdf_type_instanceof")
    form_data = {
        "rdf:type": SurfaceFormData(
            curie="rdf:type",
            short_name="type",
            anchored_status="degraded_placeholder",
            definitions=["[degraded: stub]"],
            usage_examples=[("[degraded: prompt]", "[degraded: answer]")],
        ),
    }
    report = validate_form_data_contract(
        form_data,
        ["rdf:type"],
        semantic_profile=profile,
    )
    semantic_codes = {
        v.get("code") for v in report.get("content_violations", []) or []
        if str(v.get("code", "")).startswith("SEMANTIC_")
    }
    assert semantic_codes == set(), (
        f"profile rules should skip degraded entry; got {semantic_codes}"
    )


# ----------------------------------------------------------------------
# Drafting CLI prompt injection
# ----------------------------------------------------------------------


def test_drafting_prompt_prepends_profile_directive():
    from Trainforge.scripts.draft_form_data_entry import _build_drafting_prompt

    profile = load_semantic_profile("rdf_type_instanceof")

    class _FakeEntry:
        curie = "rdf:type"
        label = "rdf:type label"
        surface_forms = ["rdf:type"]

    prompt_with = _build_drafting_prompt(
        _FakeEntry(),
        prior_violations=None,
        semantic_profile=profile,
    )
    prompt_without = _build_drafting_prompt(
        _FakeEntry(),
        prior_violations=None,
        semantic_profile=None,
    )
    # Profile directive must be present and ordered BEFORE the
    # structural rules.
    assert "SEMANTIC CONTRACT" in prompt_with
    assert prompt_with.index("SEMANTIC CONTRACT") < prompt_with.index(
        "Definitions MUST"
    )
    # Without a profile, the directive must not leak into the prompt.
    assert "SEMANTIC CONTRACT" not in prompt_without
