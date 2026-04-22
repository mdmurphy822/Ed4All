"""Worker P (Wave 4.1) — REC-PRV-01: run_id + created_at provenance fields.

Every newly-emitted chunk, concept-graph node, and concept-graph edge must
carry both ``run_id`` (sourced from the active DecisionCapture ledger) and
``created_at`` (ISO 8601 UTC). Schemas make both fields OPTIONAL — legacy
artifacts without them continue to validate.

Tests:

1. ``test_chunks_carry_run_id_and_created_at`` — _create_chunk stamps both
   fields when ``self.capture`` is present with a ``run_id``.
2. ``test_concept_nodes_carry_run_id_and_created_at`` — build_semantic_graph
   stamps both fields on every node.
3. ``test_concept_edges_carry_run_id_and_created_at`` — build_semantic_graph
   stamps both fields on every edge.
4. ``test_legacy_chunks_without_fields_validate`` — a chunk dict without
   run_id / created_at still validates against chunk_v4.schema.json.
5. ``test_run_id_source_from_decision_capture`` — when ``run_id`` kwarg is
   omitted, build_semantic_graph reads run_id off ``decision_capture``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

# Project root (Ed4All/). This file lives at Trainforge/tests/ → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.rag.typed_edge_inference import build_semantic_graph  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mini_course_typed_graph"
CHUNK_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "knowledge" / "chunk_v4.schema.json"
FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# ISO 8601 matcher: permissive on offset form (+00:00 or Z).
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover — test-env bootstrap
        pytest.skip("jsonschema not installed")


def _load_fixture():
    with open(FIXTURE_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(FIXTURE_DIR / "course.json", encoding="utf-8") as f:
        course = json.load(f)
    with open(FIXTURE_DIR / "concept_graph.json", encoding="utf-8") as f:
        concept_graph = json.load(f)
    return chunks, course, concept_graph


def _make_valid_legacy_chunk() -> Dict[str, Any]:
    """Construct a minimal chunk_v4-compliant record WITHOUT run_id/created_at.

    Mirrors ``_make_valid_chunk`` in test_chunk_validation.py. Represents a
    chunk produced before Worker P's provenance fields landed — it must
    still validate against the updated schema (fields are optional).
    """
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "m1",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "understand",
    }


def _build_chunk_validator():
    """Build a Draft202012Validator for chunk_v4.schema.json with $ref resolver.

    Mirrors the helper in test_chunk_validation.py so our positive case for
    legacy-chunk validation exercises the same machinery as the existing
    chunk-validation suite.
    """
    jsonschema = _require_jsonschema()
    from jsonschema import Draft202012Validator, RefResolver

    schemas_root = PROJECT_ROOT / "schemas"
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in schemas_root.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return Draft202012Validator(schema, resolver=resolver)


# ---------------------------------------------------------------------------
# 1. _create_chunk stamps run_id + created_at
# ---------------------------------------------------------------------------


def test_chunks_carry_run_id_and_created_at():
    """Invoke ``CourseProcessor._create_chunk`` with a stub ``self.capture``
    carrying a known ``run_id``; assert both fields land on the output.

    Uses ``object.__new__`` to bypass ``__init__`` — this mirrors the
    pattern used in unit tests for other Trainforge internals (no IMSCC
    on disk required). We wire only the attributes ``_create_chunk`` reads:

      - self.capture (with .run_id)
      - self.course_code
      - self.stats (a defaultdict-like accumulator)
      - self._all_concept_tags (a set)
      - self.domain_concept_seeds (empty list — no concept seeding)
      - self.OBJECTIVE_CODE_RE / WEEK_PREFIX_RE / NON_CONCEPT_TAGS
        (filter helpers used by the JSON-LD keyTerm merge path)
    """
    from collections import defaultdict

    from Trainforge.process_course import CourseProcessor

    processor = object.__new__(CourseProcessor)
    # Stub DecisionCapture: exposes run_id (the source of truth for this
    # test) and a no-op ``log_decision`` because ``_create_chunk`` may log
    # a bloom-level-source decision for low-confidence resolution paths.
    processor.capture = SimpleNamespace(
        run_id="test_run_42",
        log_decision=lambda **kwargs: None,
    )
    processor.course_code = "TEST_101"
    processor.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    processor._all_concept_tags = set()
    processor.domain_concept_seeds = []
    processor.objectives = None
    # Class-level attributes are inherited automatically once __class__ is
    # set by ``object.__new__``; the explicit assignments below are defensive
    # belt-and-suspenders for attributes touched along the code path.
    processor.OBJECTIVE_CODE_RE = CourseProcessor.OBJECTIVE_CODE_RE
    processor.WEEK_PREFIX_RE = CourseProcessor.WEEK_PREFIX_RE
    processor.NON_CONCEPT_TAGS = CourseProcessor.NON_CONCEPT_TAGS
    processor.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    processor.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE

    item = {
        "module_id": "m1",
        "module_title": "Module One",
        "item_id": "l1",
        "title": "Lesson One",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {},
        "sections": [],
        "misconceptions": [],
        "item_path": "m1/l1.html",
    }

    chunk = processor._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Accessibility is a core requirement of WCAG 2.2 conformance.",
        html="<p>Accessibility is a core requirement of WCAG 2.2 conformance.</p>",
        item=item,
        section_heading="Overview",
        chunk_type="explanation",
    )

    assert chunk["run_id"] == "test_run_42", (
        f"chunk missing expected run_id; got {chunk.get('run_id')!r}"
    )
    created_at = chunk.get("created_at")
    assert isinstance(created_at, str) and created_at, "chunk missing created_at"
    assert _ISO_RE.match(created_at), (
        f"created_at is not ISO 8601 UTC: {created_at!r}"
    )
    # Timezone-aware parse round-trip — datetime.fromisoformat rejects pre-py3.11
    # forms with trailing 'Z'. Ours is '+00:00' because timezone.utc is UTC.
    parsed = datetime.fromisoformat(created_at)
    assert parsed.tzinfo is not None, "created_at must be timezone-aware"


# ---------------------------------------------------------------------------
# 2. All concept-graph nodes carry run_id + created_at
# ---------------------------------------------------------------------------


def test_concept_nodes_carry_run_id_and_created_at():
    chunks, course, concept_graph = _load_fixture()
    artifact = build_semantic_graph(
        chunks,
        course,
        concept_graph,
        run_id="test_run_42",
        now=FIXED_NOW,
    )
    assert artifact["nodes"], "fixture should produce at least one node"
    expected_created_at = FIXED_NOW.isoformat()
    for node in artifact["nodes"]:
        assert node.get("run_id") == "test_run_42", (
            f"node {node.get('id')!r} missing run_id: {node}"
        )
        assert node.get("created_at") == expected_created_at, (
            f"node {node.get('id')!r} created_at mismatch: {node.get('created_at')!r}"
        )


# ---------------------------------------------------------------------------
# 3. All concept-graph edges carry run_id + created_at
# ---------------------------------------------------------------------------


def test_concept_edges_carry_run_id_and_created_at():
    chunks, course, concept_graph = _load_fixture()
    artifact = build_semantic_graph(
        chunks,
        course,
        concept_graph,
        run_id="test_run_42",
        now=FIXED_NOW,
    )
    assert artifact["edges"], "fixture should produce at least one edge"
    expected_created_at = FIXED_NOW.isoformat()
    for edge in artifact["edges"]:
        assert edge.get("run_id") == "test_run_42", (
            f"edge {edge.get('source')}->{edge.get('target')} missing run_id: {edge}"
        )
        assert edge.get("created_at") == expected_created_at, (
            f"edge {edge.get('source')}->{edge.get('target')} created_at mismatch"
        )


# ---------------------------------------------------------------------------
# 4. Legacy chunks without the fields still validate
# ---------------------------------------------------------------------------


def test_legacy_chunks_without_fields_validate():
    """A chunk_v4 record emitted before Worker P landed has no run_id and no
    created_at. The updated schema must accept it — both fields are OPTIONAL.
    """
    jsonschema = _require_jsonschema()
    validator = _build_chunk_validator()
    legacy_chunk = _make_valid_legacy_chunk()
    # Defense: make sure we're actually testing the legacy case.
    assert "run_id" not in legacy_chunk
    assert "created_at" not in legacy_chunk

    errors = sorted(validator.iter_errors(legacy_chunk), key=lambda e: list(e.absolute_path))
    if errors:  # pragma: no cover — diagnostic on failure
        first = errors[0]
        pytest.fail(
            f"Legacy chunk without run_id/created_at failed schema validation: "
            f"{'.'.join(str(p) for p in first.absolute_path) or 'root'}: {first.message}"
        )
    # Silence jsonschema unused-import warning when the assert above short-circuits.
    del jsonschema


# ---------------------------------------------------------------------------
# 5. run_id is sourced from decision_capture when no kwarg is passed
# ---------------------------------------------------------------------------


def test_run_id_source_from_decision_capture():
    """When ``run_id`` is not passed explicitly to build_semantic_graph, the
    orchestrator reads ``decision_capture.run_id`` as the source of truth.

    This locks in the "DecisionCapture ledger is the canonical source"
    contract documented in the Worker P sub-plan.
    """
    chunks, course, concept_graph = _load_fixture()
    capture = SimpleNamespace(run_id="capture_sourced_run_99")
    artifact = build_semantic_graph(
        chunks,
        course,
        concept_graph,
        decision_capture=capture,
        now=FIXED_NOW,
    )
    assert artifact["nodes"], "fixture should produce at least one node"
    assert artifact["edges"], "fixture should produce at least one edge"
    for node in artifact["nodes"]:
        assert node.get("run_id") == "capture_sourced_run_99", (
            f"node {node.get('id')!r} did not inherit run_id from decision_capture"
        )
    for edge in artifact["edges"]:
        assert edge.get("run_id") == "capture_sourced_run_99", (
            f"edge {edge.get('source')}->{edge.get('target')} did not inherit "
            f"run_id from decision_capture"
        )
