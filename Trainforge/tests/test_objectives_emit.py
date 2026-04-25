"""Wave 75 — Trainforge emits objectives.json + normalizes comma refs.

Two tightly coupled fixes covered here:

1. ``CourseProcessor._build_objectives_json`` produces the canonical
   Wave-75 objectives shape from a synthesized_objectives.json.
   Includes BOTH terminal_outcomes[] and component_objectives[] with
   parent_terminal back-pointers and the v1 schema_version flag.

2. ``normalize_outcome_refs`` (module-level helper) splits malformed
   comma-delimited strings (``"co-01,co-02,co-03"``) into separate
   refs so chunk ``learning_outcome_refs`` can resolve against
   objectives.json.

3. End-to-end: a chunk that originally carried a comma-delimited
   ref lands in the corpus with split refs, all of which resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import (  # noqa: E402
    CourseProcessor,
    normalize_outcome_refs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_synthesized_objectives() -> dict:
    """Realistic flat-list shape — matches Wave-24 plan_course_structure emit.

    This is the shape that broke pre-Wave-75 ``_build_course_json``
    (which expected nested ``[{chapter, objectives}]``). RDF_SHACL_550's
    synthesized_objectives.json carries this shape for all 29 COs.
    """
    return {
        "course_name": "WAVE75_FLAT",
        "duration_weeks": 4,
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Demonstrate RDF basics.",
                "bloom_level": "understand",
                "bloom_verb": "demonstrate",
                "cognitive_domain": "conceptual",
                "weeks": [1, 2],
            },
            {
                "id": "TO-02",
                "statement": "Construct SPARQL queries.",
                "bloom_level": "create",
                "bloom_verb": "construct",
                "cognitive_domain": "procedural",
                "weeks": [3, 4],
            },
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "parent_to": "TO-01",
                "statement": "Identify RDF triple components.",
                "bloom_level": "remember",
                "bloom_verb": "identify",
                "cognitive_domain": "factual",
                "week": 1,
                "source_refs": ["dart:rdf_primer#s1"],
            },
            {
                "id": "CO-02",
                "parent_to": "TO-01",
                "statement": "Describe IRIs and literals.",
                "bloom_level": "understand",
                "bloom_verb": "describe",
                "cognitive_domain": "conceptual",
                "week": 2,
                "source_refs": ["dart:rdf_primer#s2"],
            },
            {
                "id": "CO-03",
                "parent_to": "TO-02",
                "statement": "Write basic SELECT queries.",
                "bloom_level": "apply",
                "bloom_verb": "write",
                "cognitive_domain": "procedural",
                "week": 3,
            },
        ],
    }


def _processor_with_objectives(tmp_path, synthesized: dict) -> CourseProcessor:
    """Build a CourseProcessor whose self.objectives is populated.

    We bypass ``__init__``'s IMSCC path — it needs a real zip. Instead we
    construct via ``object.__new__`` and set the minimum attributes the
    builder methods read.
    """
    proc = object.__new__(CourseProcessor)
    proc.course_code = synthesized.get("course_name", "WAVE75_FLAT")
    proc.objectives = {
        "terminal_objectives": synthesized.get("terminal_objectives", []),
        "chapter_objectives": synthesized.get("chapter_objectives", []),
        "week_bloom_map": {},
        "bloom_distribution": {},
        "description": "",
        "course_title": "",
        "domain_concepts": [],
    }
    proc.output_dir = tmp_path / "out"
    proc.output_dir.mkdir(parents=True, exist_ok=True)
    return proc


# ---------------------------------------------------------------------------
# Part 1 — _build_objectives_json
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_objectives_json_from_flat_chapter_list(tmp_path):
    """Flat chapter_objectives shape (Wave-24 emit) produces full CO list."""
    proc = _processor_with_objectives(tmp_path, _flat_synthesized_objectives())
    doc = proc._build_objectives_json()

    assert doc is not None
    assert doc["schema_version"] == "v1"
    assert doc["course_code"] == "WAVE75_FLAT"
    assert doc["objective_count"] == {"terminal": 2, "component": 3}

    # Default (no TRAINFORGE_PRESERVE_LO_CASE) is lowercase emit.
    to_ids = [to["id"] for to in doc["terminal_outcomes"]]
    co_ids = [co["id"] for co in doc["component_objectives"]]
    assert to_ids == ["to-01", "to-02"]
    assert co_ids == ["co-01", "co-02", "co-03"]

    # parent_terminal back-pointers are preserved (lowercased).
    parents = {co["id"]: co.get("parent_terminal") for co in doc["component_objectives"]}
    assert parents == {"co-01": "to-01", "co-02": "to-01", "co-03": "to-02"}

    # source_refs propagate when present.
    co1 = next(co for co in doc["component_objectives"] if co["id"] == "co-01")
    assert co1["source_refs"] == ["dart:rdf_primer#s1"]
    assert co1["week"] == 1


@pytest.mark.unit
def test_build_objectives_json_returns_none_without_objectives(tmp_path):
    """No self.objectives → None (so _write_metadata skips the file)."""
    proc = object.__new__(CourseProcessor)
    proc.course_code = "EMPTY"
    proc.objectives = None
    proc.output_dir = tmp_path
    assert proc._build_objectives_json() is None


@pytest.mark.unit
def test_objectives_json_validates_against_schema(tmp_path):
    """Emitted shape passes the canonical objectives_v1 schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "objectives_v1.schema.json"
    )
    with schema_path.open() as fh:
        schema = json.load(fh)

    proc = _processor_with_objectives(tmp_path, _flat_synthesized_objectives())
    doc = proc._build_objectives_json()
    jsonschema.validate(doc, schema)


# ---------------------------------------------------------------------------
# Part 2 — _build_course_json now includes COs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_course_json_includes_components_from_flat_list(tmp_path):
    """Pre-Wave-75 only emitted TOs from a flat chapter_objectives list.

    With Wave-75 the flat shape is recognized too — all 5 LOs (2 TO +
    3 CO) land in course.json's learning_outcomes[]. This is the
    canonical bug fix.
    """
    proc = _processor_with_objectives(tmp_path, _flat_synthesized_objectives())
    course_data = proc._build_course_json({"title": "T"})

    los = course_data["learning_outcomes"]
    assert len(los) == 5  # 2 terminal + 3 component
    terminal_ids = [lo["id"] for lo in los if lo["hierarchy_level"] == "terminal"]
    chapter_ids = [lo["id"] for lo in los if lo["hierarchy_level"] == "chapter"]
    assert terminal_ids == ["to-01", "to-02"]
    assert sorted(chapter_ids) == ["co-01", "co-02", "co-03"]
    # Wave-75 type discriminator on chapter LOs.
    chapter = [lo for lo in los if lo["hierarchy_level"] == "chapter"]
    assert all(lo.get("type") == "component" for lo in chapter)


@pytest.mark.unit
def test_build_course_json_handles_legacy_nested_shape(tmp_path):
    """Pre-Wave-24 fixtures use nested ``[{chapter, objectives:[...]}]``.

    That shape must keep working — Wave-75 only ADDS support for the
    flat list, doesn't drop the nested form.
    """
    nested = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Recall.", "bloomLevel": "remember"}
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "statement": "List terms.", "bloomLevel": "remember"}
                ],
            }
        ],
    }
    proc = _processor_with_objectives(tmp_path, nested)
    course_data = proc._build_course_json({"title": "T"})
    los = course_data["learning_outcomes"]
    assert len(los) == 2
    assert los[0]["id"] == "to-01"
    assert los[1]["id"] == "co-01"


# ---------------------------------------------------------------------------
# Part 3 — comma-ref normalization
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_normalize_outcome_refs_splits_comma_string():
    """Single comma-delimited string → multiple refs."""
    assert normalize_outcome_refs(["co-01,co-02,co-03"]) == [
        "co-01", "co-02", "co-03"
    ]


@pytest.mark.unit
def test_normalize_outcome_refs_mixed_input():
    """Mixed input: clean refs + comma string → all separated."""
    assert normalize_outcome_refs(["co-01,co-02", "co-03"]) == [
        "co-01", "co-02", "co-03"
    ]


@pytest.mark.unit
def test_normalize_outcome_refs_preserves_order_and_dedupes():
    """Order preserved + duplicates collapsed to first occurrence."""
    assert normalize_outcome_refs(
        ["to-01", "co-01,co-02", "co-01", "co-03"]
    ) == ["to-01", "co-01", "co-02", "co-03"]


@pytest.mark.unit
def test_normalize_outcome_refs_handles_none_and_empty():
    """Robust to None / empty input — never raises."""
    assert normalize_outcome_refs(None) == []
    assert normalize_outcome_refs([]) == []
    assert normalize_outcome_refs([""]) == []
    assert normalize_outcome_refs([",,,"]) == []


@pytest.mark.unit
def test_normalize_outcome_refs_strips_whitespace():
    """Whitespace around comma-split parts is stripped."""
    assert normalize_outcome_refs(["co-01 , co-02 ,co-03"]) == [
        "co-01", "co-02", "co-03"
    ]


@pytest.mark.unit
def test_normalize_outcome_refs_accepts_single_string():
    """Single string (not a list) is also accepted."""
    assert normalize_outcome_refs("co-01,co-02") == ["co-01", "co-02"]


# ---------------------------------------------------------------------------
# Part 4 — End-to-end resolution after normalization
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_normalized_chunk_refs_resolve_against_objectives(tmp_path):
    """A chunk with the bug-shape ref ``co-01,co-02,co-03`` resolves
    against objectives.json after normalization. Real RDF_SHACL_550
    chunk pattern (chunk_00001 carried this exact ref string)."""
    proc = _processor_with_objectives(tmp_path, _flat_synthesized_objectives())
    objectives = proc._build_objectives_json()

    # Build the resolution set the same way LibV2 retrieval does.
    valid_ids = {to["id"] for to in objectives["terminal_outcomes"]}
    valid_ids |= {co["id"] for co in objectives["component_objectives"]}

    # Real bug-shape chunk refs from rdf_shacl_550_chunk_00001.
    raw_refs = ["co-01", "co-02", "co-03", "co-01,co-02,co-03"]
    normed = normalize_outcome_refs(raw_refs)

    # All normalized refs must resolve.
    unresolved = [r for r in normed if r not in valid_ids]
    assert unresolved == [], f"unresolved after normalization: {unresolved}"
    # And the result is the deduplicated 3-ref list.
    assert sorted(normed) == ["co-01", "co-02", "co-03"]
