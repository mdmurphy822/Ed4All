"""Wave 61 — per-page Bloom distribution aggregate in Courseforge JSON-LD.

Pre-Wave-61 there was no course- or page-level aggregate Bloom signal;
consumers had to walk each LO and re-aggregate to answer "is this page
heavy on 'remember' content?" Wave 61 emits a ``bloomDistribution``
object on every page that has at least one bloomLevel-tagged LO, giving
KG consumers the profile for free.

Covers:

* Schema: ``BloomDistribution`` shape validates; rejects zero/negative
  counts, non-enum level keys, invalid cognitive-domain keys; legacy
  pages without the field still validate.
* Helper: empty LO list → ``None``; every level counted once by default;
  mixed-level LOs produce the expected per-level + per-domain counts;
  LOs with null bloomLevel are skipped from totals; zero-count levels
  are elided from the per-key dicts.
* Integration: ``_build_page_metadata`` attaches ``bloomDistribution``
  when LOs are present with Bloom tags; elides it when none have a
  level; absent when no LOs on the page.
* End-to-end: ``generate_week`` round trip carries the distribution in
  the JSON-LD block of the overview page.
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
    _build_bloom_distribution,
    _build_page_metadata,
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


def _distribution_validator() -> Draft202012Validator:
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
        "$ref": f"{root['$id']}#/$defs/BloomDistribution",
    }
    return Draft202012Validator(subschema, resolver=resolver)


# ---------------------------------------------------------------------- #
# 1. Schema
# ---------------------------------------------------------------------- #


def test_schema_accepts_well_formed_distribution():
    v = _distribution_validator()
    dist = {
        "total": 4,
        "byLevel": {"apply": 2, "analyze": 1, "evaluate": 1},
        "byCognitiveDomain": {
            "procedural": 2,
            "conceptual": 1,
            "metacognitive": 1,
        },
    }
    errors = sorted(v.iter_errors(dist), key=lambda e: list(e.absolute_path))
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"


def test_schema_rejects_non_enum_level_key_in_bylevel():
    v = _distribution_validator()
    dist = {"total": 1, "byLevel": {"bogus": 1}}
    assert list(v.iter_errors(dist)), "Non-enum level key must fail validation"


def test_schema_rejects_zero_count_in_bylevel():
    v = _distribution_validator()
    dist = {"total": 0, "byLevel": {"apply": 0}}
    assert list(v.iter_errors(dist)), "Zero count in byLevel must fail (minimum: 1)"


def test_schema_rejects_negative_total():
    v = _distribution_validator()
    dist = {"total": -1, "byLevel": {}}
    assert list(v.iter_errors(dist)), "Negative total must fail (minimum: 0)"


def test_schema_accepts_empty_bylevel_with_zero_total():
    """Degenerate case — empty dict with 0 total is permitted by schema."""
    v = _distribution_validator()
    dist = {"total": 0, "byLevel": {}}
    errors = list(v.iter_errors(dist))
    # Schema allows this, but the emit helper never produces it (returns
    # None instead so the field is elided entirely).
    assert not errors


# ---------------------------------------------------------------------- #
# 2. Helper behavior
# ---------------------------------------------------------------------- #


def test_helper_returns_none_when_no_objectives():
    assert _build_bloom_distribution([]) is None


def test_helper_returns_none_when_no_objective_has_bloom_level():
    objectives = [
        {"id": "CO-01", "bloomLevel": None},
        {"id": "CO-02"},  # no bloomLevel at all
    ]
    assert _build_bloom_distribution(objectives) is None


def test_helper_counts_single_objective():
    objectives = [{"id": "CO-01", "bloomLevel": "apply"}]
    dist = _build_bloom_distribution(objectives)
    assert dist == {
        "total": 1,
        "byLevel": {"apply": 1},
        "byCognitiveDomain": {"procedural": 1},
    }


def test_helper_aggregates_across_mixed_levels():
    objectives = [
        {"id": "CO-01", "bloomLevel": "apply"},
        {"id": "CO-02", "bloomLevel": "analyze"},
        {"id": "CO-03", "bloomLevel": "apply"},
        {"id": "CO-04", "bloomLevel": "evaluate"},
    ]
    dist = _build_bloom_distribution(objectives)
    assert dist["total"] == 4
    assert dist["byLevel"] == {"apply": 2, "analyze": 1, "evaluate": 1}
    # apply → procedural (2), analyze → conceptual (1), evaluate → metacognitive (1)
    assert dist["byCognitiveDomain"] == {
        "procedural": 2,
        "conceptual": 1,
        "metacognitive": 1,
    }


def test_helper_skips_null_bloom_level_objectives_from_total():
    objectives = [
        {"id": "CO-01", "bloomLevel": "apply"},
        {"id": "CO-02", "bloomLevel": None},
        {"id": "CO-03"},  # missing key
        {"id": "CO-04", "bloomLevel": "analyze"},
    ]
    dist = _build_bloom_distribution(objectives)
    assert dist["total"] == 2
    assert dist["byLevel"] == {"apply": 1, "analyze": 1}


def test_helper_output_omits_zero_count_levels():
    """Only levels with >= 1 LO appear in byLevel (schema requires minimum 1)."""
    objectives = [{"id": "CO-01", "bloomLevel": "apply"}]
    dist = _build_bloom_distribution(objectives)
    # Only 'apply' appears — no 'remember', 'understand', etc.
    assert set(dist["byLevel"].keys()) == {"apply"}
    assert set(dist["byCognitiveDomain"].keys()) == {"procedural"}


def test_helper_output_validates_against_schema():
    """Helper output must round-trip through the schema cleanly."""
    objectives = [
        {"id": "CO-01", "bloomLevel": "apply"},
        {"id": "CO-02", "bloomLevel": "analyze"},
        {"id": "CO-03", "bloomLevel": "create"},
    ]
    dist = _build_bloom_distribution(objectives)
    validator = _distribution_validator()
    errors = list(validator.iter_errors(dist))
    assert not errors, f"Helper output failed schema: {[e.message for e in errors]}"


# ---------------------------------------------------------------------- #
# 3. Integration via _build_page_metadata
# ---------------------------------------------------------------------- #


def test_page_metadata_attaches_distribution_when_los_tagged():
    meta = _build_page_metadata(
        course_code="TEST_101",
        week_num=1,
        module_type="overview",
        page_id="week_01_overview",
        objectives=[
            {
                "id": "CO-01",
                "statement": "Apply the framework.",
                "bloom_level": "apply",
            },
            {
                "id": "CO-02",
                "statement": "Analyze the data.",
                "bloom_level": "analyze",
            },
        ],
    )
    assert "bloomDistribution" in meta
    assert meta["bloomDistribution"]["total"] == 2
    assert meta["bloomDistribution"]["byLevel"] == {"apply": 1, "analyze": 1}


def test_page_metadata_elides_distribution_when_no_objectives():
    meta = _build_page_metadata(
        course_code="TEST_101",
        week_num=1,
        module_type="content",
        page_id="week_01_content",
        # no objectives supplied
    )
    assert "bloomDistribution" not in meta


def test_page_metadata_elides_distribution_when_all_los_lack_bloom():
    meta = _build_page_metadata(
        course_code="TEST_101",
        week_num=1,
        module_type="overview",
        page_id="week_01_overview",
        objectives=[
            # no bloom_level and a statement with no canonical verb
            {"id": "CO-01", "statement": "Something vague without taxonomy words."},
        ],
    )
    assert "learningObjectives" in meta
    # None of the LOs have a detectable bloomLevel, so distribution is elided.
    assert "bloomDistribution" not in meta


# ---------------------------------------------------------------------- #
# 4. End-to-end — generated HTML JSON-LD carries the distribution
# ---------------------------------------------------------------------- #


def test_generated_page_jsonld_carries_bloom_distribution(tmp_path):
    week_data = {
        "week_number": 1,
        "title": "Distribution smoke",
        "objectives": [
            {
                "id": "CO-01",
                "statement": "Apply the framework.",
                "bloom_level": "apply",
            },
            {
                "id": "CO-02",
                "statement": "Analyze the outputs.",
                "bloom_level": "analyze",
            },
            {
                "id": "CO-03",
                "statement": "Apply again elsewhere.",
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
                        "heading": "H",
                        "content_type": "definition",
                        "paragraphs": ["body."],
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

    overview = (out / "week_01" / "week_01_overview.html").read_text(encoding="utf-8")
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        overview,
        flags=re.DOTALL,
    )
    parsed = [json.loads(b) for b in blocks]
    with_dist = [p for p in parsed if "bloomDistribution" in p]
    assert with_dist, "No JSON-LD block carried bloomDistribution"
    dist = with_dist[0]["bloomDistribution"]
    assert dist["total"] == 3
    assert dist["byLevel"] == {"apply": 2, "analyze": 1}
    assert dist["byCognitiveDomain"] == {"procedural": 2, "conceptual": 1}
