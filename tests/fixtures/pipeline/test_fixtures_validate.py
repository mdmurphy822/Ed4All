"""Validate the pipeline reference fixtures against their schemas.

Runs in the default test suite (fast, no external deps beyond ``jsonschema``
+ ``referencing``). Workers α / β / γ rely on these fixtures as their
target-shape source of truth — so the fixtures must stay conformant or the
block assumption breaks.

Covers:
- ``reference_week_01/*.html`` → extract each ``<script type="application/ld+json">``
  body, validate against ``courseforge_jsonld_v1.schema.json``.
- ``reference_libv2/corpus/chunks.jsonl`` → validate each line against
  ``chunk_v4.schema.json`` (strict, with ``additionalProperties`` enforced
  on ``source``).
- ``reference_libv2/graph/concept_graph_semantic.json`` → validate against
  ``concept_graph_semantic.schema.json``; strict mode (FallbackProvenance
  arm stripped) mirrors Wave 6 Worker W's opt-in.
- ``reference_libv2/graph/misconceptions.json`` → each entry validates
  against ``misconception.schema.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent
REFERENCE_WEEK = FIXTURE_DIR / "reference_week_01"
REFERENCE_LIBV2 = FIXTURE_DIR / "reference_libv2"

SCHEMAS = PROJECT_ROOT / "schemas"
COURSEFORGE_SCHEMA = SCHEMAS / "knowledge" / "courseforge_jsonld_v1.schema.json"
CHUNK_SCHEMA = SCHEMAS / "knowledge" / "chunk_v4.schema.json"
GRAPH_SCHEMA = SCHEMAS / "knowledge" / "concept_graph_semantic.schema.json"
MISCONCEPTION_SCHEMA = SCHEMAS / "knowledge" / "misconception.schema.json"
SOURCE_REFERENCE_SCHEMA = SCHEMAS / "knowledge" / "source_reference.schema.json"

_TAXONOMY_FILES = [
    "bloom_verbs.json",
    "module_type.json",
    "content_type.json",
    "cognitive_domain.json",
    "question_type.json",
]


def _require_jsonschema():
    jsonschema = pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    return jsonschema


def _registry_with(*schema_paths: Path):
    """Return a ``referencing.Registry`` populated with the named schemas
    plus every taxonomy under ``schemas/taxonomies/``."""
    _require_jsonschema()
    from referencing import Registry, Resource

    resources = []
    for path in schema_paths:
        doc = json.loads(path.read_text())
        resources.append((doc["$id"], Resource.from_contents(doc)))

    tax_dir = SCHEMAS / "taxonomies"
    for name in _TAXONOMY_FILES:
        tax = json.loads((tax_dir / name).read_text())
        resources.append((tax["$id"], Resource.from_contents(tax)))

    return Registry().with_resources(resources)


# ---------------------------------------------------------------------- #
# Courseforge JSON-LD — reference week 01
# ---------------------------------------------------------------------- #


_JSON_LD_RE = re.compile(
    r'<script type="application/ld\+json">(.*?)</script>',
    re.DOTALL,
)


def _extract_jsonld(html: str) -> dict:
    m = _JSON_LD_RE.search(html)
    assert m, "No <script type=application/ld+json> block found"
    return json.loads(m.group(1))


@pytest.mark.parametrize("name", [
    "week_01_overview.html",
    "week_01_content_01_two_stages.html",
    "week_01_application.html",
    "week_01_self_check.html",
    "week_01_summary.html",
])
def test_reference_week_page_jsonld_validates(name):
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    html = (REFERENCE_WEEK / name).read_text(encoding="utf-8")
    meta = _extract_jsonld(html)

    schema = json.loads(COURSEFORGE_SCHEMA.read_text())
    registry = _registry_with(COURSEFORGE_SCHEMA, SOURCE_REFERENCE_SCHEMA)
    validator = Draft202012Validator(schema, registry=registry)
    errors = sorted(validator.iter_errors(meta), key=lambda e: e.path)
    assert not errors, (
        f"{name} JSON-LD has schema errors:\n" +
        "\n".join(f"  - {e.message} at /{'/'.join(str(p) for p in e.path)}" for e in errors)
    )


def test_reference_week_page_has_data_cf_markers():
    """Sanity: every reference page carries the essential data-cf-* surface."""
    for name in REFERENCE_WEEK.glob("*.html"):
        html = name.read_text()
        assert 'data-cf-role="template-chrome"' in html, name
        assert "data-cf-objective-id=" in html, name
        assert "data-cf-bloom-level=" in html, name
        assert "data-cf-content-type=" in html, name


# ---------------------------------------------------------------------- #
# Chunks (reference_libv2/corpus/chunks.jsonl)
# ---------------------------------------------------------------------- #


def test_reference_chunks_validate_strict():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    chunks_path = REFERENCE_LIBV2 / "corpus" / "chunks.jsonl"
    schema = json.loads(CHUNK_SCHEMA.read_text())
    registry = _registry_with(CHUNK_SCHEMA, SOURCE_REFERENCE_SCHEMA)
    validator = Draft202012Validator(schema, registry=registry)

    lines = [
        line for line in chunks_path.read_text().splitlines() if line.strip()
    ]
    assert len(lines) >= 3
    for i, line in enumerate(lines):
        chunk = json.loads(line)
        errors = sorted(validator.iter_errors(chunk), key=lambda e: e.path)
        assert not errors, (
            f"chunk {i} has schema errors:\n" +
            "\n".join(f"  - {e.message}" for e in errors)
        )


# ---------------------------------------------------------------------- #
# Concept graph (reference_libv2/graph/concept_graph_semantic.json)
# ---------------------------------------------------------------------- #


def test_reference_graph_validates_strict():
    """Concept graph validates under strict evidence mode.

    Strict = FallbackProvenance arm stripped from the oneOf (see
    ``lib/validators/evidence.py::get_schema``). Our reference graph
    carries only rule-specific evidence so it must pass under strict.
    """
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    graph_path = REFERENCE_LIBV2 / "graph" / "concept_graph_semantic.json"
    schema_dict = json.loads(GRAPH_SCHEMA.read_text())

    # Strip FallbackProvenance from the oneOf to enforce strict-mode shape.
    # We keep all rule-specific arms so each of our edges gets validated
    # against its matching evidence $def.
    edges_props = (
        schema_dict["properties"]["edges"]["items"]["properties"]
    )
    provenance = edges_props["provenance"]
    provenance["oneOf"] = [
        arm for arm in provenance["oneOf"]
        if arm.get("$ref", "").split("/")[-1] != "FallbackProvenance"
    ]

    registry = _registry_with(GRAPH_SCHEMA, SOURCE_REFERENCE_SCHEMA)
    validator = Draft202012Validator(schema_dict, registry=registry)

    graph = json.loads(graph_path.read_text())
    errors = sorted(validator.iter_errors(graph), key=lambda e: e.path)
    assert not errors, (
        "Reference graph has schema errors:\n" +
        "\n".join(f"  - {e.message} at /{'/'.join(str(p) for p in e.path)}" for e in errors)
    )

    # Structural checks complement the schema validation.
    assert graph["kind"] == "concept_semantic"
    assert len(graph["nodes"]) >= 5
    assert len(graph["edges"]) >= 3
    edge_types = {edge["type"] for edge in graph["edges"]}
    assert len(edge_types) >= 2, f"need ≥2 edge types, got {edge_types}"


# ---------------------------------------------------------------------- #
# Misconceptions (reference_libv2/graph/misconceptions.json)
# ---------------------------------------------------------------------- #


_MC_ID_RE = re.compile(r"^mc_[0-9a-f]{16}$")


def test_reference_misconceptions_validate():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    mc_path = REFERENCE_LIBV2 / "graph" / "misconceptions.json"
    doc = json.loads(mc_path.read_text())
    assert "misconceptions" in doc
    assert len(doc["misconceptions"]) >= 1

    schema = json.loads(MISCONCEPTION_SCHEMA.read_text())
    validator = Draft202012Validator(schema)
    for entry in doc["misconceptions"]:
        errors = sorted(validator.iter_errors(entry), key=lambda e: e.path)
        assert not errors, (
            f"Misconception {entry.get('id')} errors: " +
            "; ".join(e.message for e in errors)
        )
        assert _MC_ID_RE.match(entry["id"]), (
            f"Misconception ID {entry['id']!r} doesn't match mc_[0-9a-f]{{16}}"
        )
