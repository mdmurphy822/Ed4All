"""Phase 6 Subtask 9 tests for ``compose_abcd_prose``.

Targets the helper added in Phase 6 Subtask 3 (commit ``b46e433``):
:func:`lib.ontology.learning_objectives.compose_abcd_prose`.

Coverage contract per the plan (``plans/phase6_abcd_concept_extractor.md``
Subtask 9): "Round-trip tests across 6 Bloom levels x 3 fixture verbs each.
Asserts terminal period; capitalization; no double-spaces; no
double-periods." Verification target: >=18 PASSED.

The 6 x 3 grid of round-trip cases lives in :class:`TestRoundTripBloomGrid`
below; the remainder of the file pins edge cases that are loadbearing for
downstream consumers (the upcoming
``lib.validators.abcd_objective.AbcdObjectiveValidator`` and the
course-outliner agent's ABCD authorship path) so a future regression in
the helper trips the suite rather than corrupting emitted JSON-LD.
"""

from __future__ import annotations

import re

import pytest

from lib.ontology.learning_objectives import (
    BLOOMS_VERBS,
    compose_abcd_prose,
)


# ---------------------------------------------------------------------------
# Plan-cited contract
# ---------------------------------------------------------------------------


class TestPlanCitedContract:
    """Pin the exact byte-string from the plan's verification snippet.

    Plan source: ``plans/phase6_abcd_concept_extractor.md`` Subtask 3
    verification line 65. A drift here is a Wave 6-A2 regression, not a
    test issue.
    """

    def test_plan_cited_example_byte_identical(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {
                    "verb": "identify",
                    "action_object": "cell parts",
                },
                "condition": "from a labeled diagram",
                "degree": "with 90% accuracy",
            }
        )
        assert (
            out
            == "Students will identify cell parts from a labeled diagram, with 90% accuracy."
        )


# ---------------------------------------------------------------------------
# 6 Bloom levels x 3 fixture verbs each = 18 round-trip cases
# ---------------------------------------------------------------------------


# A canonical fixture verb selection per Bloom level. Every verb here MUST
# be a member of ``BLOOMS_VERBS[level]`` (asserted in
# ``test_fixture_verbs_are_canonical_per_level``) so the round-trip
# implicitly stress-tests the canonical Bloom verb set without taking a
# direct dependency on the JSON taxonomy ordering.
_BLOOM_VERB_FIXTURES = {
    "remember": ("identify", "list", "define"),
    "understand": ("describe", "explain", "summarize"),
    "apply": ("apply", "calculate", "demonstrate"),
    "analyze": ("analyze", "differentiate", "contrast"),
    "evaluate": ("evaluate", "critique", "justify"),
    "create": ("create", "design", "construct"),
}


def _round_trip_cases():
    """Yield (level, verb) tuples for the 6x3 round-trip grid."""

    for level, verbs in _BLOOM_VERB_FIXTURES.items():
        for verb in verbs:
            yield level, verb


class TestRoundTripBloomGrid:
    """6 Bloom levels x 3 verbs each = 18 PASSED rows.

    For every (level, verb) combination, compose a sentence and assert
    the four mechanical invariants the plan calls out:

    1. exactly one terminal period;
    2. audience capitalised;
    3. no double-spaces;
    4. no double-periods.
    """

    @pytest.mark.parametrize("level, verb", list(_round_trip_cases()))
    def test_round_trip_invariants(self, level, verb):
        # Sanity: the fixture verb must be canonical for its level.
        assert verb in BLOOMS_VERBS[level], (
            f"Fixture drift: verb {verb!r} not in BLOOMS_VERBS[{level!r}]; "
            f"update _BLOOM_VERB_FIXTURES to track schemas/taxonomies/bloom_verbs.json."
        )

        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {
                    "verb": verb,
                    "action_object": "the relevant content",
                },
                "condition": "in a structured exercise",
                "degree": "with measurable proficiency",
            }
        )

        # Mechanical invariant 1: exactly one terminal period.
        assert out.endswith(".")
        assert out.count(".") == 1, (
            f"Expected exactly one period in composed sentence, got "
            f"{out.count('.')}: {out!r}"
        )

        # Mechanical invariant 2: audience capitalised (first character
        # is uppercase).
        assert out[0].isupper(), f"First char not uppercase: {out!r}"

        # Mechanical invariant 3: no double-spaces.
        assert "  " not in out, f"Double-space found in {out!r}"

        # Mechanical invariant 4: no double-periods.
        assert ".." not in out, f"Double-period found in {out!r}"

        # The verb must appear verbatim in the output (helper does no
        # Bloom-level remapping; that is the validator's job).
        assert f" {verb} " in out, f"Verb {verb!r} not found in {out!r}"


# ---------------------------------------------------------------------------
# Fixture sanity guard
# ---------------------------------------------------------------------------


class TestFixtureSanity:
    """Guards that the test fixtures track the canonical taxonomy."""

    def test_fixture_covers_all_six_bloom_levels(self):
        assert set(_BLOOM_VERB_FIXTURES) == set(BLOOMS_VERBS), (
            "_BLOOM_VERB_FIXTURES drifted from canonical Bloom levels; "
            "update the fixture to match BLOOMS_VERBS keys."
        )

    def test_fixture_three_verbs_per_level(self):
        for level, verbs in _BLOOM_VERB_FIXTURES.items():
            assert len(verbs) == 3, (
                f"Fixture for {level!r} has {len(verbs)} verbs; "
                f"plan calls for 3 verbs per level."
            )

    def test_fixture_verbs_are_canonical_per_level(self):
        for level, verbs in _BLOOM_VERB_FIXTURES.items():
            for verb in verbs:
                assert verb in BLOOMS_VERBS[level], (
                    f"Fixture verb {verb!r} not in BLOOMS_VERBS[{level!r}]"
                )


# ---------------------------------------------------------------------------
# Capitalisation
# ---------------------------------------------------------------------------


class TestCapitalisation:
    """The helper must capitalise the audience even when the input is
    lowercase, so JSON-LD emit doesn't depend on the upstream LLM's
    capitalisation discipline."""

    def test_lowercase_audience_is_capitalised(self):
        out = compose_abcd_prose(
            {
                "audience": "students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a diagram",
                "degree": "with 90% accuracy",
            }
        )
        assert out.startswith("Students will ")

    def test_already_capitalised_audience_passes_through(self):
        out = compose_abcd_prose(
            {
                "audience": "Nursing students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a diagram",
                "degree": "with 90% accuracy",
            }
        )
        assert out.startswith("Nursing students will ")

    def test_multiword_audience_only_capitalises_first_letter(self):
        # Helper logic capitalises index 0 only; downstream emitters
        # depend on this exact behavior to avoid mangling proper nouns
        # like 'iOS engineers'.
        out = compose_abcd_prose(
            {
                "audience": "nursing students",
                "behavior": {"verb": "identify", "action_object": "anatomy"},
                "condition": "in lab",
                "degree": "accurately",
            }
        )
        assert out.startswith("Nursing students will ")


# ---------------------------------------------------------------------------
# Empty / optional condition + degree
# ---------------------------------------------------------------------------


class TestEmptyConditionDegree:
    """Per the helper docstring, ``condition`` and ``degree`` may be
    empty strings (later passes fill them in). The composed sentence
    must omit empty fields gracefully — no dangling spaces, no stranded
    commas."""

    def test_empty_condition_drops_clause(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "",
                "degree": "with 90% accuracy",
            }
        )
        assert out == "Students will identify cell parts, with 90% accuracy."
        assert "  " not in out

    def test_empty_degree_drops_clause(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a diagram",
                "degree": "",
            }
        )
        assert out == "Students will identify cell parts from a diagram."
        # No dangling comma before the period.
        assert ",." not in out
        assert ", ." not in out

    def test_both_empty_yields_minimal_sentence(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "",
                "degree": "",
            }
        )
        assert out == "Students will identify cell parts."
        assert ",." not in out

    def test_whitespace_only_condition_treated_as_empty(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "   ",
                "degree": "with 90% accuracy",
            }
        )
        # Stripped condition collapses to '' so the clause is omitted.
        assert out == "Students will identify cell parts, with 90% accuracy."


# ---------------------------------------------------------------------------
# Trailing-punctuation hygiene
# ---------------------------------------------------------------------------


class TestTrailingPunctuationHygiene:
    """Helper strips trailing punctuation on each field so the assembled
    sentence has exactly one terminal period and one comma (between
    condition and degree)."""

    def test_trailing_period_on_degree_does_not_double(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a diagram",
                "degree": "with 90% accuracy.",
            }
        )
        assert out == "Students will identify cell parts from a diagram, with 90% accuracy."
        assert ".." not in out
        assert out.count(".") == 1

    def test_trailing_period_on_condition_does_not_strand(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a diagram.",
                "degree": "with 90% accuracy",
            }
        )
        # Stripped condition is 'from a diagram' so the comma fires
        # cleanly between condition and degree.
        assert out == "Students will identify cell parts from a diagram, with 90% accuracy."

    def test_trailing_comma_on_action_object_stripped(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts,"},
                "condition": "from a diagram",
                "degree": "with 90% accuracy",
            }
        )
        # No double-comma artefact.
        assert ",," not in out
        assert out == "Students will identify cell parts from a diagram, with 90% accuracy."

    def test_input_whitespace_around_fields_stripped(self):
        out = compose_abcd_prose(
            {
                "audience": "  Students  ",
                "behavior": {"verb": "  identify  ", "action_object": "  cell parts  "},
                "condition": "  from a diagram  ",
                "degree": "  with 90% accuracy  ",
            }
        )
        assert out == "Students will identify cell parts from a diagram, with 90% accuracy."
        assert "  " not in out


# ---------------------------------------------------------------------------
# Required-field error handling
# ---------------------------------------------------------------------------


class TestRequiredFieldErrors:
    """All four ABCD fields are required at the function boundary;
    behavior must carry both ``verb`` and ``action_object``. Missing
    keys raise ``ValueError``; non-mapping input raises ``TypeError``."""

    def test_non_mapping_input_raises_type_error(self):
        with pytest.raises(TypeError, match="expected a mapping"):
            compose_abcd_prose("not a mapping")  # type: ignore[arg-type]

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="expected a mapping"):
            compose_abcd_prose(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "missing",
        ["audience", "behavior", "condition", "degree"],
    )
    def test_missing_top_level_field_raises_value_error(self, missing):
        full = {
            "audience": "Students",
            "behavior": {"verb": "identify", "action_object": "cell parts"},
            "condition": "from a diagram",
            "degree": "with 90% accuracy",
        }
        del full[missing]
        with pytest.raises(ValueError, match=f"missing required ABCD field {missing!r}"):
            compose_abcd_prose(full)

    def test_non_mapping_behavior_raises_value_error(self):
        with pytest.raises(ValueError, match="behavior.*must be a mapping"):
            compose_abcd_prose(
                {
                    "audience": "Students",
                    "behavior": "identify cell parts",  # not a mapping
                    "condition": "from a diagram",
                    "degree": "with 90% accuracy",
                }
            )

    @pytest.mark.parametrize("missing_sub", ["verb", "action_object"])
    def test_missing_behavior_subfield_raises_value_error(self, missing_sub):
        behavior = {"verb": "identify", "action_object": "cell parts"}
        del behavior[missing_sub]
        with pytest.raises(ValueError, match=f"missing required sub-field {missing_sub!r}"):
            compose_abcd_prose(
                {
                    "audience": "Students",
                    "behavior": behavior,
                    "condition": "from a diagram",
                    "degree": "with 90% accuracy",
                }
            )

    def test_empty_audience_after_strip_raises(self):
        with pytest.raises(ValueError, match="'audience' must be non-empty"):
            compose_abcd_prose(
                {
                    "audience": "   ",
                    "behavior": {"verb": "identify", "action_object": "cell parts"},
                    "condition": "from a diagram",
                    "degree": "with 90% accuracy",
                }
            )

    def test_empty_verb_after_strip_raises(self):
        with pytest.raises(ValueError, match="behavior.verb must be non-empty"):
            compose_abcd_prose(
                {
                    "audience": "Students",
                    "behavior": {"verb": "   ", "action_object": "cell parts"},
                    "condition": "from a diagram",
                    "degree": "with 90% accuracy",
                }
            )

    def test_empty_action_object_after_strip_raises(self):
        with pytest.raises(
            ValueError, match="behavior.action_object must be non-empty"
        ):
            compose_abcd_prose(
                {
                    "audience": "Students",
                    "behavior": {"verb": "identify", "action_object": "  "},
                    "condition": "from a diagram",
                    "degree": "with 90% accuracy",
                }
            )


# ---------------------------------------------------------------------------
# Cross-Bloom-level verb portability (not gated by validator)
# ---------------------------------------------------------------------------


class TestVerbBloomLevelDecoupling:
    """The helper does NOT gate on verb-Bloom alignment. That is the
    contract of the upstream ``AbcdObjectiveValidator`` (Phase 6 ST 4),
    which calls this helper post-validation. Composing a sentence with
    a mismatched verb must succeed at the helper level so the validator
    sees the actual emitted prose when it issues a ``regenerate``
    action."""

    def test_mismatched_verb_still_composes(self):
        # 'create' is in the 'create' Bloom level, not 'remember' — but
        # the helper composes the sentence regardless. Validation
        # responsibility lives in lib.validators.abcd_objective.
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "create", "action_object": "an outline"},
                "condition": "given a topic",
                "degree": "in 30 minutes",
            }
        )
        assert out == "Students will create an outline given a topic, in 30 minutes."


# ---------------------------------------------------------------------------
# Output shape regression guard
# ---------------------------------------------------------------------------


class TestOutputShapeRegression:
    """A regex pin against the helper's deterministic output shape so a
    silent format drift trips this test rather than corrupting LO emit
    sites that re-parse the prose downstream."""

    _FULL_SHAPE = re.compile(
        r"^[A-Z][^.]* will [a-z][a-zA-Z'-]* [^,.]+, [^,.]+\.$"
    )

    def test_full_shape_matches_regex(self):
        out = compose_abcd_prose(
            {
                "audience": "Students",
                "behavior": {"verb": "identify", "action_object": "cell parts"},
                "condition": "from a labeled diagram",
                "degree": "with 90% accuracy",
            }
        )
        assert self._FULL_SHAPE.match(out), (
            f"Output shape regressed: {out!r}"
        )
