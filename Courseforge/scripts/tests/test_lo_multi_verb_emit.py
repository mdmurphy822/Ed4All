"""Wave 58 — multi-verb / multi-level LOs emit ``bloomLevels[]`` / ``bloomVerbs[]``.

A learning objective like "Analyze and evaluate the market trends" targets
two cognitive demands at once. Pre-Wave-58 only the primary verb survived
(via ``detect_bloom_level`` returning the longest-verb-first match); the
secondary verb was silently discarded, losing KG signal.

Wave 58 emits every canonical verb / level as ``bloomVerbs[]`` / ``bloomLevels[]``
arrays on each ``LearningObjective`` alongside the existing singular fields.
The schema-level invariant is ``bloomLevels[0] == bloomLevel`` and
``bloomVerbs[0] == bloomVerb`` so single-verb consumers see no change and
multi-verb consumers get the full set.

Covers:

* Schema: ``bloomLevels[]`` and ``bloomVerbs[]`` validate; rejection of
  non-enum level values; backward-compat without the plural fields.
* Emit: multi-verb statements populate both arrays in canonical order;
  single-verb statements emit singleton arrays; verb-less statements
  elide both plural fields.
* Invariant: ``bloomLevels[0]`` always equals ``bloomLevel`` when both
  are present (singular-plural alignment).
* Pre-set singular alignment: when an LO carries an upstream ``bloom_level``
  that disagrees with detection, the plural fields are elided (the pre-set
  singular stays authoritative rather than emit misleading plurals).
* Pre-set rotation: when an LO's pre-set ``bloom_level`` appears in the
  detected list but not at index 0, the plural arrays rotate to place
  the authoritative match first.
* End-to-end: a ``generate_week`` round trip writes HTML whose JSON-LD
  block carries ``bloomVerbs`` / ``bloomLevels`` on each LO.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    _align_bloom_matches,
    _build_objectives_metadata,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JSONLD_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "knowledge" / "courseforge_jsonld_v1.schema.json"
)
_BLOOM_VERBS_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "bloom_verbs.json"
)
_COGNITIVE_DOMAIN_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "cognitive_domain.json"
)
_QUESTION_TYPE_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "question_type.json"
)


def _lo_validator() -> Draft202012Validator:
    """Validator for a single ``LearningObjective`` payload with all refs resolved."""
    with open(_JSONLD_SCHEMA_PATH, encoding="utf-8") as f:
        root = json.load(f)
    with open(_BLOOM_VERBS_SCHEMA_PATH, encoding="utf-8") as f:
        bloom = json.load(f)
    with open(_COGNITIVE_DOMAIN_SCHEMA_PATH, encoding="utf-8") as f:
        cog = json.load(f)
    with open(_QUESTION_TYPE_SCHEMA_PATH, encoding="utf-8") as f:
        qtype = json.load(f)
    store = {
        root["$id"]: root,
        bloom["$id"]: bloom,
        cog["$id"]: cog,
        qtype["$id"]: qtype,
    }
    resolver = RefResolver.from_schema(root, store=store)
    subschema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"{root['$id']}#/$defs/LearningObjective",
    }
    return Draft202012Validator(subschema, resolver=resolver)


# ---------------------------------------------------------------------- #
# 1. Schema — plural fields validate with canonical bloom levels
# ---------------------------------------------------------------------- #


def test_schema_accepts_bloom_levels_and_verbs_arrays():
    validator = _lo_validator()
    lo = {
        "id": "CO-01",
        "statement": "Analyze and evaluate the market trends.",
        "bloomLevel": "evaluate",
        "bloomVerb": "evaluate",
        "bloomLevels": ["evaluate", "analyze"],
        "bloomVerbs": ["evaluate", "analyze"],
        "cognitiveDomain": "conceptual",
    }
    errors = sorted(validator.iter_errors(lo), key=lambda e: list(e.absolute_path))
    assert not errors, f"Unexpected schema errors: {[e.message for e in errors]}"


def test_schema_rejects_non_enum_bloom_level_in_plural_array():
    validator = _lo_validator()
    lo = {
        "id": "CO-02",
        "statement": "Apply the framework.",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "bloomLevels": ["apply", "bogus-level"],
        "bloomVerbs": ["apply", "bogus"],
        "cognitiveDomain": "procedural",
    }
    errors = list(validator.iter_errors(lo))
    assert errors, "Non-enum bloom level in plural array should fail validation"


def test_schema_backward_compatible_without_plural_fields():
    """Legacy LO payloads without plural fields still validate."""
    validator = _lo_validator()
    lo = {
        "id": "CO-03",
        "statement": "Explain the water cycle.",
        "bloomLevel": "understand",
        "bloomVerb": "explain",
        "cognitiveDomain": "conceptual",
    }
    errors = list(validator.iter_errors(lo))
    assert not errors, f"Legacy LO should validate; got {[e.message for e in errors]}"


# ---------------------------------------------------------------------- #
# 2. Emit — multi-verb statement produces full plural arrays
# ---------------------------------------------------------------------- #


def test_emit_multi_verb_statement_produces_both_arrays():
    objectives = [
        {
            "id": "CO-10",
            "statement": "Analyze and evaluate the market trends in Q4.",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevel"] == "evaluate"
    assert lo["bloomVerb"] == "evaluate"
    assert lo["bloomLevels"] == ["evaluate", "analyze"]
    assert lo["bloomVerbs"] == ["evaluate", "analyze"]


def test_emit_single_verb_statement_produces_singleton_arrays():
    objectives = [
        {
            "id": "CO-11",
            "statement": "Apply the framework to the data.",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevel"] == "apply"
    assert lo["bloomLevels"] == ["apply"]
    assert lo["bloomVerbs"] == ["apply"]


def test_emit_no_verb_statement_elides_plural_fields():
    objectives = [
        {
            "id": "CO-12",
            "statement": "Mystery statement with no taxonomy verb here.",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevel"] is None
    assert "bloomLevels" not in lo
    assert "bloomVerbs" not in lo


# ---------------------------------------------------------------------- #
# 3. Invariant — bloomLevels[0] == bloomLevel, bloomVerbs[0] == bloomVerb
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "statement",
    [
        "Analyze and evaluate the market trends.",
        "Solve the equation and explain your reasoning.",
        "Apply the framework.",
        "Design a scalable system from scratch.",
        "Recall and apply the relevant principles.",
    ],
)
def test_emit_preserves_singular_plural_alignment(statement):
    objectives = [{"id": "CO-20", "statement": statement}]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    if lo.get("bloomLevels"):
        assert lo["bloomLevels"][0] == lo["bloomLevel"], (
            f"Invariant broken: bloomLevels[0]={lo['bloomLevels'][0]!r} vs "
            f"bloomLevel={lo['bloomLevel']!r} for {statement!r}"
        )
        assert lo["bloomVerbs"][0] == lo["bloomVerb"], (
            f"Invariant broken: bloomVerbs[0]={lo['bloomVerbs'][0]!r} vs "
            f"bloomVerb={lo['bloomVerb']!r} for {statement!r}"
        )


# ---------------------------------------------------------------------- #
# 4. Pre-set singular — alignment / rotation / elision
# ---------------------------------------------------------------------- #


def test_emit_preset_level_matches_detection_keeps_canonical_order():
    """When pre-set singular equals the natural detect[0], no rotation needed."""
    objectives = [
        {
            "id": "CO-30",
            "statement": "Analyze and evaluate the market trends.",
            "bloom_level": "evaluate",
            "bloom_verb": "evaluate",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevels"] == ["evaluate", "analyze"]
    assert lo["bloomVerbs"] == ["evaluate", "analyze"]


def test_emit_preset_level_rotated_to_index_zero():
    """Pre-set singular appears in detection but not at index 0 — rotate."""
    objectives = [
        {
            "id": "CO-31",
            "statement": "Analyze and evaluate the market trends.",
            "bloom_level": "analyze",  # upstream authority overrides the "evaluate" natural primary
            "bloom_verb": "analyze",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevel"] == "analyze"
    assert lo["bloomLevels"] == ["analyze", "evaluate"], (
        f"Rotation failed: pre-set 'analyze' should sit at index 0; "
        f"got {lo['bloomLevels']!r}"
    )
    assert lo["bloomVerbs"] == ["analyze", "evaluate"]


def test_emit_preset_level_not_in_detection_elides_plurals():
    """Pre-set singular disagrees with detection entirely — elide plurals."""
    objectives = [
        {
            "id": "CO-32",
            "statement": "Explain the process thoroughly.",
            "bloom_level": "apply",  # pre-set; statement only has "explain" (understand)
            "bloom_verb": "apply",
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo["bloomLevel"] == "apply"  # singular stays authoritative
    assert "bloomLevels" not in lo, (
        "Plurals must be elided when singular disagrees with detection"
    )
    assert "bloomVerbs" not in lo


# ---------------------------------------------------------------------- #
# 5. _align_bloom_matches unit behavior
# ---------------------------------------------------------------------- #


def test_align_no_matches_returns_empty():
    assert _align_bloom_matches([], "apply", "apply") == []


def test_align_no_authoritative_returns_input_unchanged():
    matches = [("evaluate", "evaluate"), ("analyze", "analyze")]
    assert _align_bloom_matches(matches, None, None) == matches


def test_align_prefers_exact_level_verb_match():
    """When multiple entries share a level, exact verb match rotates first."""
    matches = [
        ("understand", "explain"),
        ("understand", "describe"),
        ("apply", "solve"),
    ]
    # Authoritative (understand, describe) — must pick the describe entry,
    # not the first understand entry.
    out = _align_bloom_matches(matches, "understand", "describe")
    assert out[0] == ("understand", "describe")
    assert set(out) == set(matches)


def test_align_falls_back_to_level_when_verb_not_in_list():
    matches = [("understand", "explain"), ("apply", "solve")]
    out = _align_bloom_matches(matches, "understand", "clarify")  # clarify not in list
    assert out[0] == ("understand", "explain")  # level-only fallback


def test_align_returns_empty_when_level_not_in_list():
    matches = [("understand", "explain"), ("apply", "solve")]
    out = _align_bloom_matches(matches, "create", "design")
    assert out == []


# ---------------------------------------------------------------------- #
# 6. End-to-end — JSON-LD block in generated HTML carries plurals
# ---------------------------------------------------------------------- #


def test_generated_page_jsonld_carries_bloom_plural_fields(tmp_path):
    """Full ``generate_week`` round trip writes HTML whose JSON-LD block
    carries ``bloomVerbs`` / ``bloomLevels`` on each multi-verb LO."""
    week_data = {
        "week_number": 1,
        "title": "Multi-verb smoke",
        "objectives": [
            {
                "id": "CO-01",
                "statement": "Analyze and evaluate the market trends.",
                "bloom_level": "evaluate",
            },
            {
                "id": "CO-02",
                "statement": "Apply the framework.",
                "bloom_level": "apply",
            },
        ],
        "overview_text": ["Intro paragraph."],
        "readings": ["Ch. 1"],
        "content_modules": [
            {
                "title": "Basics",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["Some text."],
                    }
                ],
            }
        ],
        "activities": [],
        "key_takeaways": ["Takeaway."],
        "reflection_questions": ["Question?"],
    }
    out = tmp_path / "out"
    generate_course.generate_week(week_data, out, "TEST_101", source_module_map=None)

    overview_html = (out / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    jsonld_blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        overview_html,
        flags=re.DOTALL,
    )
    parsed = [json.loads(b) for b in jsonld_blocks]
    with_los = [p for p in parsed if p.get("learningObjectives")]
    assert with_los
    los = {lo["id"]: lo for lo in with_los[0]["learningObjectives"]}

    multi = los["CO-01"]
    assert multi["bloomLevels"] == ["evaluate", "analyze"]
    assert multi["bloomVerbs"] == ["evaluate", "analyze"]

    single = los["CO-02"]
    assert single["bloomLevels"] == ["apply"]
    assert single["bloomVerbs"] == ["apply"]
