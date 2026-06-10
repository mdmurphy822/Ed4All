"""Wave 1.6 W1.6.A — back-compat ``oneOf`` on ``source_refs``.

Pinned to plan ``plans/gpt-feedback-2-wave1.6-per-objective-attribution-2026-05.md``
§2 Fix 1.6.A. Bumps ``schemas/knowledge/objectives_v1.schema.json`` so
that BOTH ``definitions.TerminalOutcome`` and
``definitions.ComponentObjective`` admit a ``oneOf`` on ``source_refs``:

* legacy ``List[str]`` shape (back-compat with every existing LibV2
  archive — see plan §6.1)
* structured ``List[{ref, chunk_ids: [str], minItems: 1}]`` shape
  (the W1.6.B / W1.6.C target shape)

The companion patches in this wave preserve the new shape across the
load and emit roundtrips:

* ``MCP/tools/_content_gen_helpers.py::_normalize_objective_entry``
  passes ``source_refs`` through verbatim (pre-Wave-1.6 it was dropped).
* ``Trainforge/process_course.py::_build_objectives_json`` per-entry
  shape-preserving copy (pre-Wave-1.6 a flat ``list(obj["source_refs"])``
  call destroyed structured-shape entries on emit).

The plan refers to ``$defs.TerminalOutcome`` / ``$defs.ComponentObjective``;
the actual schema uses Draft-07 ``definitions/`` (the semantic intent is
identical — ``$defs`` is the Draft 2019-09+ name for the same construct).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "objectives_v1.schema.json"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _terminal_outcome_schema() -> Dict[str, Any]:
    """Return the standalone ``TerminalOutcome`` def for direct validation."""
    schema = _load_schema()
    return schema["definitions"]["TerminalOutcome"]


def _component_objective_schema() -> Dict[str, Any]:
    """Return the standalone ``ComponentObjective`` def for direct validation."""
    schema = _load_schema()
    return schema["definitions"]["ComponentObjective"]


def _legacy_list_str_refs() -> List[str]:
    return ["dart:owl2_primer#s1", "dart:owl2_primer#s2"]


def _structured_refs() -> List[Dict[str, Any]]:
    return [
        {
            "ref": "ch7",
            "chunk_ids": ["dart:shacl-spec_accessible#sec3-1-targets"],
        },
        {
            "ref": "ch8",
            "chunk_ids": [
                "dart:shacl-spec_accessible#sec4-1-core-constraints",
                "dart:shacl-spec_accessible#sec4-2-property-shapes",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Schema-level oneOf admission tests — TerminalOutcome
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_list_str_source_refs_validates_on_terminal_outcome():
    """Legacy ``List[str]`` source_refs validates against TerminalOutcome.

    Back-compat with every existing LibV2 archive on disk (the RDF/SHACL
    calibration corpus plus 60+ siblings) where COs carry ``source_refs: ["dart:foo#s1", ...]``
    and TOs carry no ``source_refs`` at all. The bump ADDs ``source_refs``
    to TO; this test asserts the legacy shape is admitted there too.
    """
    jsonschema = pytest.importorskip("jsonschema")
    to = {
        "id": "to-04",
        "statement": "Develop SHACL shape graphs that validate RDF data.",
        "source_refs": _legacy_list_str_refs(),
    }
    jsonschema.validate(to, _terminal_outcome_schema())


@pytest.mark.unit
def test_legacy_list_str_source_refs_validates_on_component_objective():
    """Legacy ``List[str]`` source_refs validates against ComponentObjective.

    The PRE-bump shape — every existing LibV2 archive emits this. Plan
    §6.1: forcing the structured shape immediately would break every
    archive on disk; ``oneOf`` admits both per-emit so back-compat holds.
    """
    jsonschema = pytest.importorskip("jsonschema")
    co = {
        "id": "co-01",
        "parent_terminal": "to-01",
        "statement": "Identify foundational terms.",
        "source_refs": _legacy_list_str_refs(),
    }
    jsonschema.validate(co, _component_objective_schema())


@pytest.mark.unit
def test_structured_list_objects_validates_on_terminal_outcome():
    """Structured ``List[{ref, chunk_ids: [str]}]`` validates against TO.

    The Wave-1.6 target shape. Once W1.6.B's prompt + synthesizer patch
    lands, every newly authored TO with attribution emits this shape.
    """
    jsonschema = pytest.importorskip("jsonschema")
    to = {
        "id": "to-04",
        "statement": "Develop SHACL shape graphs that validate RDF data.",
        "source_refs": _structured_refs(),
    }
    jsonschema.validate(to, _terminal_outcome_schema())


@pytest.mark.unit
def test_structured_list_objects_validates_on_component_objective():
    """Structured shape validates against ComponentObjective."""
    jsonschema = pytest.importorskip("jsonschema")
    co = {
        "id": "co-04",
        "parent_terminal": "to-04",
        "statement": "Author SHACL property shapes for cardinality.",
        "source_refs": _structured_refs(),
    }
    jsonschema.validate(co, _component_objective_schema())


@pytest.mark.unit
def test_mixed_shape_fails_on_terminal_outcome():
    """Mixed string+object source_refs FAILS via oneOf semantics on TO.

    Plan §2 Fix 1.6.A: mixed-shape rejection is ``oneOf``-native. A
    single ``source_refs`` array containing both string entries and
    object entries fails validation — the clean-contract rule (no
    partial migration within a single objective's array).
    """
    jsonschema = pytest.importorskip("jsonschema")
    to = {
        "id": "to-04",
        "statement": "Develop SHACL shape graphs.",
        "source_refs": [
            "dart:owl2_primer#s1",
            {
                "ref": "ch7",
                "chunk_ids": ["dart:shacl-spec#sec3"],
            },
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(to, _terminal_outcome_schema())


@pytest.mark.unit
def test_mixed_shape_fails_on_component_objective():
    """Mixed string+object source_refs FAILS on CO via oneOf semantics."""
    jsonschema = pytest.importorskip("jsonschema")
    co = {
        "id": "co-01",
        "statement": "Identify foundational terms.",
        "source_refs": [
            {
                "ref": "ch7",
                "chunk_ids": ["dart:shacl-spec#sec3"],
            },
            "dart:owl2_primer#s2",
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(co, _component_objective_schema())


@pytest.mark.unit
def test_empty_chunk_ids_fails_on_terminal_outcome():
    """``chunk_ids: []`` FAILS the inner ``minItems: 1`` constraint on TO.

    A structured-shape entry with no chunk IDs is meaningless — it
    declares "this objective is grounded in nothing in particular." The
    schema rejects it.
    """
    jsonschema = pytest.importorskip("jsonschema")
    to = {
        "id": "to-04",
        "statement": "Develop SHACL shape graphs.",
        "source_refs": [
            {"ref": "ch7", "chunk_ids": []},
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(to, _terminal_outcome_schema())


@pytest.mark.unit
def test_empty_chunk_ids_fails_on_component_objective():
    """``chunk_ids: []`` FAILS ``minItems: 1`` on CO."""
    jsonschema = pytest.importorskip("jsonschema")
    co = {
        "id": "co-01",
        "statement": "Identify foundational terms.",
        "source_refs": [
            {"ref": "ch7", "chunk_ids": []},
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(co, _component_objective_schema())


# ---------------------------------------------------------------------------
# _build_objectives_json round-trip tests
# ---------------------------------------------------------------------------


def _processor_with_objectives(tmp_path, synthesized: Dict[str, Any]):
    """Build a CourseProcessor whose ``self.objectives`` is populated.

    Mirrors ``Trainforge/tests/test_objectives_emit.py``'s pattern of
    bypassing ``__init__``'s IMSCC-zip path via ``object.__new__``.
    """
    from Trainforge.process_course import CourseProcessor  # type: ignore

    proc = object.__new__(CourseProcessor)
    proc.course_code = synthesized.get("course_name", "TEST_W1_6_A")
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


@pytest.mark.unit
def test_build_objectives_json_round_trips_structured_shape(tmp_path):
    """Structured-shape input survives _build_objectives_json verbatim.

    Pre-Wave-1.6 the line ``entry["source_refs"] = list(obj["source_refs"])``
    shallow-copied the outer list but kept the inner dict references; on
    a future mutation that would have cross-contaminated two CO entries
    sharing a ref by reference. Patched to per-entry dispatch:
    strings pass through unchanged; dicts are deep-copied with only the
    canonical ``{ref, chunk_ids[]}`` keys.
    """
    structured_input = {
        "course_name": "TEST_STRUCTURED",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Recall basics."}
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "parent_to": "TO-01",
                "statement": "Identify SHACL constraints.",
                "source_refs": _structured_refs(),
            }
        ],
    }
    proc = _processor_with_objectives(tmp_path, structured_input)
    doc = proc._build_objectives_json()

    assert doc is not None
    cos = doc["component_objectives"]
    assert len(cos) == 1
    refs = cos[0]["source_refs"]

    # Shape preserved: list of dicts, not coerced to list of strings.
    assert isinstance(refs, list)
    assert all(isinstance(r, dict) for r in refs)

    # Content preserved verbatim.
    assert refs[0]["ref"] == "ch7"
    assert refs[0]["chunk_ids"] == [
        "dart:shacl-spec_accessible#sec3-1-targets"
    ]
    assert refs[1]["ref"] == "ch8"
    assert refs[1]["chunk_ids"] == [
        "dart:shacl-spec_accessible#sec4-1-core-constraints",
        "dart:shacl-spec_accessible#sec4-2-property-shapes",
    ]

    # The emit must validate against the bumped schema.
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.validate(doc, schema)

    # Defensive: deep-copy isolation. Mutating the input dict's chunk_ids
    # MUST NOT mutate the emitted entry — proves the per-entry dispatch
    # is making a fresh dict + fresh inner list rather than aliasing.
    structured_input["chapter_objectives"][0]["source_refs"][0][
        "chunk_ids"
    ].append("dart:shacl-spec#sec3-mutated")
    assert refs[0]["chunk_ids"] == [
        "dart:shacl-spec_accessible#sec3-1-targets"
    ]


@pytest.mark.unit
def test_build_objectives_json_round_trips_legacy_shape(tmp_path):
    """Legacy ``List[str]`` source_refs round-trips through emit cleanly.

    Back-compat invariant: the RDF/SHACL calibration corpus archive's 29 COs all
    carry ``source_refs: List[str]``. The patched emit handles this
    arm of the oneOf identically to the pre-Wave-1.6 behavior.
    """
    legacy_input = {
        "course_name": "TEST_LEGACY",
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Recall basics."}
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "parent_to": "TO-01",
                "statement": "Identify SHACL constraints.",
                "source_refs": _legacy_list_str_refs(),
            }
        ],
    }
    proc = _processor_with_objectives(tmp_path, legacy_input)
    doc = proc._build_objectives_json()

    assert doc is not None
    cos = doc["component_objectives"]
    refs = cos[0]["source_refs"]
    assert refs == ["dart:owl2_primer#s1", "dart:owl2_primer#s2"]
    assert all(isinstance(r, str) for r in refs)

    # The emit must validate against the bumped schema.
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.validate(doc, schema)


# ---------------------------------------------------------------------------
# _normalize_objective_entry preserve tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_objective_entry_preserves_legacy_source_refs():
    """``_normalize_objective_entry`` preserves legacy List[str] source_refs.

    Pre-Wave-1.6 ``--reuse-objectives`` round-trips silently dropped the
    field. The patched helper preserves the legacy arm as a fresh list
    of strings.
    """
    from MCP.tools._content_gen_helpers import (  # type: ignore
        _normalize_objective_entry,
    )

    raw = {
        "id": "TO-04",
        "statement": "Develop SHACL shape graphs.",
        "source_refs": _legacy_list_str_refs(),
    }
    entry = _normalize_objective_entry(raw)
    assert entry is not None
    assert "source_refs" in entry
    assert entry["source_refs"] == [
        "dart:owl2_primer#s1",
        "dart:owl2_primer#s2",
    ]
    assert all(isinstance(r, str) for r in entry["source_refs"])


@pytest.mark.unit
def test_normalize_objective_entry_preserves_structured_source_refs():
    """``_normalize_objective_entry`` preserves structured source_refs.

    The Wave-1.6 target shape survives the load roundtrip verbatim.
    """
    from MCP.tools._content_gen_helpers import (  # type: ignore
        _normalize_objective_entry,
    )

    raw = {
        "id": "TO-04",
        "statement": "Develop SHACL shape graphs.",
        "source_refs": copy.deepcopy(_structured_refs()),
    }
    entry = _normalize_objective_entry(raw)
    assert entry is not None
    refs = entry["source_refs"]
    assert isinstance(refs, list)
    assert all(isinstance(r, dict) for r in refs)
    assert refs[0]["ref"] == "ch7"
    assert refs[0]["chunk_ids"] == [
        "dart:shacl-spec_accessible#sec3-1-targets"
    ]
    assert refs[1]["ref"] == "ch8"
    assert refs[1]["chunk_ids"] == [
        "dart:shacl-spec_accessible#sec4-1-core-constraints",
        "dart:shacl-spec_accessible#sec4-2-property-shapes",
    ]
