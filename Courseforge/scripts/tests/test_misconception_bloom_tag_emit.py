"""Wave 60 — Bloom-tagged misconceptions in Courseforge JSON-LD.

Pre-Wave-60 misconceptions shipped as free ``{misconception, correction}``
pairs. The KG couldn't tell an "apply"-level mistake (wrong procedure
sequence) from an "analyze"-level one (misread evidence), so diagnostic
question generation couldn't target the right cognitive demand.

Wave 60 attaches optional ``bloomLevel`` and ``cognitiveDomain`` to every
Misconception, inferred from the correction statement at emit time with
a fallback to the misconception statement. Upstream-supplied values
override detection.

Covers:

* Schema: Misconception accepts optional ``bloomLevel`` / ``cognitiveDomain``;
  rejects non-enum values; still validates legacy payloads without them.
* Helper behavior: inference from correction text; fallback to
  misconception text; upstream override; elision when nothing matches.
* cognitiveDomain always derives from the canonical bloom_to_cognitive_domain
  map when bloomLevel is populated (never shipped in isolation).
* End-to-end: generate_week round trip emits tagged misconceptions in
  the JSON-LD block of a generated HTML page.
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
from generate_course import _build_misconceptions_metadata  # noqa: E402

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


def _misc_validator() -> Draft202012Validator:
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
        "$ref": f"{root['$id']}#/$defs/Misconception",
    }
    return Draft202012Validator(subschema, resolver=resolver)


# ---------------------------------------------------------------------- #
# 1. Schema
# ---------------------------------------------------------------------- #


def test_schema_accepts_bloom_tagged_misconception():
    v = _misc_validator()
    m = {
        "misconception": "Photosynthesis happens only in leaves.",
        "correction": "Apply photosynthesis to any chloroplast-bearing cell.",
        "bloomLevel": "apply",
        "cognitiveDomain": "procedural",
    }
    errors = sorted(v.iter_errors(m), key=lambda e: list(e.absolute_path))
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"


def test_schema_rejects_non_enum_bloom_level_on_misconception():
    v = _misc_validator()
    m = {
        "misconception": "x",
        "correction": "y",
        "bloomLevel": "bogus",
    }
    errors = list(v.iter_errors(m))
    assert errors, "Non-enum bloomLevel must fail validation"


def test_schema_backward_compatible_without_bloom_fields():
    v = _misc_validator()
    m = {"misconception": "x", "correction": "y"}
    errors = list(v.iter_errors(m))
    assert not errors, (
        f"Legacy misconception should validate; got {[e.message for e in errors]}"
    )


# ---------------------------------------------------------------------- #
# 2. Helper — inference from correction statement
# ---------------------------------------------------------------------- #


def test_helper_infers_bloom_from_correction():
    misconceptions = [
        {
            "misconception": "Students often forget Ohm's law applies only to linear resistors.",
            "correction": "Apply Ohm's law only within its linearity assumption.",
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert result[0]["bloomLevel"] == "apply"
    assert result[0]["cognitiveDomain"] == "procedural"
    # Original fields preserved
    assert (
        result[0]["misconception"]
        == "Students often forget Ohm's law applies only to linear resistors."
    )
    assert (
        result[0]["correction"]
        == "Apply Ohm's law only within its linearity assumption."
    )


def test_helper_falls_back_to_misconception_text_when_correction_has_no_verb():
    misconceptions = [
        {
            "misconception": "Learners mistakenly analyze the distribution without checking for skew.",
            "correction": "Always check first.",  # no canonical Bloom verb
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert result[0]["bloomLevel"] == "analyze"
    assert result[0]["cognitiveDomain"] == "conceptual"


def test_helper_elides_fields_when_neither_text_has_canonical_verb():
    misconceptions = [
        {
            "misconception": "A common error in this area.",
            "correction": "Not the right way.",
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert "bloomLevel" not in result[0]
    assert "cognitiveDomain" not in result[0]


def test_helper_upstream_bloom_level_overrides_detection():
    """Upstream-supplied bloomLevel takes precedence over inference."""
    misconceptions = [
        {
            "misconception": "Students often analyze the trends incorrectly.",
            "correction": "Apply the correction.",
            "bloomLevel": "evaluate",  # upstream authority
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert result[0]["bloomLevel"] == "evaluate"
    # Domain matches the upstream level, not the detected level — the
    # detected correction text "Apply the correction" would yield
    # bloomLevel='apply' → cognitiveDomain='procedural', but the upstream
    # 'evaluate' wins and produces 'metacognitive' via the canonical map.
    assert result[0]["cognitiveDomain"] == "metacognitive"


def test_helper_accepts_snake_case_upstream_key():
    misconceptions = [
        {
            "misconception": "A common slip.",
            "correction": "Explain the correct mechanism.",
            "bloom_level": "apply",  # snake_case upstream key
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert result[0]["bloomLevel"] == "apply"


def test_helper_preserves_additional_properties_false_invariant():
    """Helper must not smuggle undeclared keys through to the emit.

    Schema has ``additionalProperties: false`` on Misconception. Any
    extra keys on the input dict should be dropped (not passed through).
    """
    misconceptions = [
        {
            "misconception": "A common slip.",
            "correction": "Apply the correct logic.",
            "undeclared_key": "leakage",
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert "undeclared_key" not in result[0]
    # Emit still schema-valid
    v = _misc_validator()
    errors = list(v.iter_errors(result[0]))
    assert not errors, f"Emit violated schema: {[e.message for e in errors]}"


# ---------------------------------------------------------------------- #
# 3. cognitiveDomain derivation is canonical
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bloom,expected_domain",
    [
        ("remember", "factual"),
        ("understand", "conceptual"),
        ("apply", "procedural"),
        ("analyze", "conceptual"),
        ("evaluate", "metacognitive"),
        ("create", "procedural"),
    ],
)
def test_cognitive_domain_matches_canonical_map(bloom, expected_domain):
    misconceptions = [
        {
            "misconception": "placeholder",
            "correction": "placeholder",
            "bloomLevel": bloom,
        }
    ]
    result = _build_misconceptions_metadata(misconceptions)
    assert result[0]["cognitiveDomain"] == expected_domain


# ---------------------------------------------------------------------- #
# 4. End-to-end — generated HTML JSON-LD carries Bloom-tagged misconceptions
# ---------------------------------------------------------------------- #


def test_generated_page_jsonld_carries_bloom_tagged_misconceptions(tmp_path):
    week_data = {
        "week_number": 1,
        "title": "Misconception smoke",
        "objectives": [
            {
                "id": "CO-01",
                "statement": "Apply the framework correctly.",
                "bloom_level": "apply",
            },
        ],
        "overview_text": ["Intro."],
        "readings": ["Ch. 1"],
        "content_modules": [
            {
                "title": "M",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["body."],
                    }
                ],
                "misconceptions": [
                    {
                        "misconception": "Learners often skip the normalization step.",
                        "correction": "Apply normalization before aggregation.",
                    }
                ],
            }
        ],
        "activities": [],
        "key_takeaways": ["k"],
        "reflection_questions": ["q"],
    }
    out = tmp_path / "out"
    generate_course.generate_week(week_data, out, "TEST_101", source_module_map=None)

    # Misconceptions land on the content page, not the overview.
    content_html_path = next((out / "week_01").glob("week_01_content_*.html"))
    html = content_html_path.read_text(encoding="utf-8")
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html,
        flags=re.DOTALL,
    )
    parsed = [json.loads(b) for b in blocks]
    with_miscs = [p for p in parsed if p.get("misconceptions")]
    assert with_miscs, "No JSON-LD block carried misconceptions"
    m = with_miscs[0]["misconceptions"][0]
    assert m["misconception"] == "Learners often skip the normalization step."
    assert m["correction"] == "Apply normalization before aggregation."
    assert m["bloomLevel"] == "apply"
    assert m["cognitiveDomain"] == "procedural"
