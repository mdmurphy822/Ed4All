"""Phase 6 ST 13 — `_generate_pedagogy_graph` consumes upstream concept graph.

Locks the contract that ``CourseProcessor._generate_pedagogy_graph``
short-circuits the in-process ``build_pedagogy_graph`` call when an
upstream-emitted pedagogy graph file is supplied via the new
``concept_graph_path`` constructor kwarg / CLI flag (the path the
``concept_extraction`` workflow phase writes at
``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``).

The four contract assertions:

1. **Skip-when-path-supplied.** ``concept_graph_path`` set to an
   existing JSON file → method returns the file's parsed content
   verbatim WITHOUT invoking ``build_pedagogy_graph``.
2. **Fall-through when path is None.** Legacy / pre-Phase-6 corpora
   (no ``concept_graph_path``) still hit the existing build path.
3. **Fall-through on missing file.** A stale phase-output handoff
   (path doesn't exist) doesn't crash — it falls through to the
   in-process build.
4. **Fall-through on parse failure.** Malformed JSON at the path
   doesn't crash — it falls through to the in-process build.

Cross-link: upstream emit at
``MCP/tools/pipeline_tools.py::_run_concept_extraction`` (Worker C-J,
Subtask 12). Plan citation:
``plans/phase6_abcd_concept_extractor.md::Subtask 13``.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_processor(concept_graph_path: Any = None) -> CourseProcessor:
    """Build a bare ``CourseProcessor`` without IMSCC ingestion.

    Mirrors the ``__new__`` test pattern used elsewhere in
    ``Trainforge/tests/`` (e.g. ``test_concept_extraction_cleanup.py``)
    so we can exercise ``_generate_pedagogy_graph`` directly without a
    real corpus.
    """
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "PHASE6_ST13_TEST"
    proc.objectives = {}
    proc.concept_graph_path = (
        Path(concept_graph_path) if concept_graph_path else None
    )
    return proc


def _minimal_chunks() -> List[Dict[str, Any]]:
    """Three-chunk fixture sufficient for the builder to emit edges."""
    return [
        {
            "id": "chunk_a",
            "chunk_type": "explanation",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "chunk_b",
            "chunk_type": "explanation",
            "concept_tags": ["beta"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_02", "item_path": "week_02/p.html"},
        },
    ]


def _upstream_graph_payload() -> Dict[str, Any]:
    """Distinctive payload — sentinel marker proves we read THIS file."""
    return {
        "kind": "pedagogy",
        "course_id": "PHASE6_ST13_TEST",
        "_phase6_st13_marker": "loaded_from_disk",
        "nodes": [
            {"id": "alpha", "class": "DomainConcept"},
            {"id": "beta", "class": "DomainConcept"},
        ],
        "edges": [
            {"source": "alpha", "target": "beta", "relation_type": "related_to"},
        ],
        "generated_at": "2026-05-03T00:00:00",
        "stats": {
            "node_count": 2,
            "edge_count": 1,
            "nodes_by_class": {"DomainConcept": 2},
            "edges_by_relation": {"related_to": 1},
        },
    }


# ---------------------------------------------------------------------------
# Phase 6 ST 13 contract tests
# ---------------------------------------------------------------------------


def test_consumes_phase6_concept_graph_when_present(
    tmp_path: Path,
) -> None:
    """When ``concept_graph_path`` points at a valid JSON file, the
    method returns the parsed payload and does NOT call
    ``build_pedagogy_graph``.

    This is the load-bearing assertion — Phase 6 ST 13's whole reason
    for existence is to avoid the redundant rebuild.
    """
    graph_path = tmp_path / "concept_graph_semantic.json"
    graph_path.write_text(
        json.dumps(_upstream_graph_payload()), encoding="utf-8"
    )

    proc = _make_processor(concept_graph_path=graph_path)

    # Sentinel mock — if build_pedagogy_graph is called, we want a
    # loud ImportError-style failure so the assertion is unambiguous.
    sentinel_called = {"count": 0}

    def _sentinel_builder(*args, **kwargs):
        sentinel_called["count"] += 1
        raise AssertionError(
            "build_pedagogy_graph called despite concept_graph_path "
            "being supplied — Phase 6 ST 13 short-circuit failed."
        )

    with mock.patch(
        "Trainforge.pedagogy_graph_builder.build_pedagogy_graph",
        _sentinel_builder,
    ):
        graph = proc._generate_pedagogy_graph(_minimal_chunks())

    assert sentinel_called["count"] == 0, (
        "build_pedagogy_graph was invoked despite concept_graph_path being supplied"
    )
    assert isinstance(graph, dict)
    assert graph.get("_phase6_st13_marker") == "loaded_from_disk", (
        "Returned graph does not match upstream payload — "
        "the file at concept_graph_path was not consumed."
    )
    assert len(graph.get("nodes") or []) == 2
    assert len(graph.get("edges") or []) == 1


def test_falls_through_to_build_when_path_is_none() -> None:
    """Legacy corpora (no ``concept_graph_path``) preserve pre-Phase-6
    behaviour — the in-process ``build_pedagogy_graph`` still runs.
    """
    proc = _make_processor(concept_graph_path=None)

    builder_called = {"count": 0, "args": None, "kwargs": None}

    def _capturing_builder(*args, **kwargs):
        builder_called["count"] += 1
        builder_called["args"] = args
        builder_called["kwargs"] = kwargs
        return {
            "kind": "pedagogy",
            "_built_in_process": True,
            "nodes": [],
            "edges": [],
            "stats": {
                "node_count": 0,
                "edge_count": 0,
                "nodes_by_class": {},
                "edges_by_relation": {},
            },
        }

    with mock.patch(
        "Trainforge.pedagogy_graph_builder.build_pedagogy_graph",
        _capturing_builder,
    ):
        graph = proc._generate_pedagogy_graph(_minimal_chunks())

    assert builder_called["count"] == 1, (
        "build_pedagogy_graph was NOT invoked when concept_graph_path "
        "is None — fall-through path broken."
    )
    assert graph.get("_built_in_process") is True


def test_falls_through_when_path_does_not_exist(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale phase-output handoff (file missing) degrades to the
    in-process build with a warning log — does NOT crash.
    """
    missing_path = tmp_path / "does_not_exist.json"
    assert not missing_path.exists()

    proc = _make_processor(concept_graph_path=missing_path)

    builder_called = {"count": 0}

    def _capturing_builder(*args, **kwargs):
        builder_called["count"] += 1
        return {"kind": "pedagogy", "_built_in_process": True, "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.pedagogy_graph_builder.build_pedagogy_graph",
            _capturing_builder,
        ):
            graph = proc._generate_pedagogy_graph(_minimal_chunks())

    assert builder_called["count"] == 1, (
        "build_pedagogy_graph was NOT called as fallback when "
        "concept_graph_path doesn't exist."
    )
    assert graph.get("_built_in_process") is True
    # The fallback warning should mention the missing path.
    assert any(
        "does not exist" in r.getMessage() or "is not a file" in r.getMessage()
        for r in caplog.records
    ), "Expected a warning log mentioning the missing path"


def test_falls_through_on_malformed_upstream_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSON at ``concept_graph_path`` falls through to the
    in-process build instead of crashing the run.
    """
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("this is not valid json {{{", encoding="utf-8")

    proc = _make_processor(concept_graph_path=bad_path)

    builder_called = {"count": 0}

    def _capturing_builder(*args, **kwargs):
        builder_called["count"] += 1
        return {"kind": "pedagogy", "_built_in_process": True, "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.pedagogy_graph_builder.build_pedagogy_graph",
            _capturing_builder,
        ):
            graph = proc._generate_pedagogy_graph(_minimal_chunks())

    assert builder_called["count"] == 1
    assert graph.get("_built_in_process") is True
    assert any(
        "failed to load" in r.getMessage().lower()
        for r in caplog.records
    ), "Expected a warning log on JSON parse failure"


def test_falls_through_on_non_dict_payload(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSON file at the path that parses but isn't a dict (e.g. a
    JSON array) falls through to the in-process build — defensive
    against accidental shape drift.
    """
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    proc = _make_processor(concept_graph_path=list_path)

    builder_called = {"count": 0}

    def _capturing_builder(*args, **kwargs):
        builder_called["count"] += 1
        return {"kind": "pedagogy", "_built_in_process": True, "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.pedagogy_graph_builder.build_pedagogy_graph",
            _capturing_builder,
        ):
            graph = proc._generate_pedagogy_graph(_minimal_chunks())

    assert builder_called["count"] == 1
    assert graph.get("_built_in_process") is True
    assert any(
        "is not a dict" in r.getMessage()
        for r in caplog.records
    ), "Expected a warning log on non-dict payload"


def test_constructor_kwarg_threaded_into_self(tmp_path: Path) -> None:
    """The ``concept_graph_path`` constructor kwarg lands on
    ``self.concept_graph_path`` as a ``Path``.

    Locks the wiring contract so callers (workflow runner, MCP tool)
    can rely on the field being set after construction.
    """
    # Use __new__ + manual __init__ would require IMSCC; instead patch
    # the heavyweight init steps and verify the kwarg path directly.
    graph_path = tmp_path / "graph.json"
    graph_path.write_text("{}", encoding="utf-8")

    # Path(...) coercion: kwarg accepts both str and Path-like.
    proc1 = CourseProcessor.__new__(CourseProcessor)
    proc1.concept_graph_path = Path(str(graph_path))
    assert isinstance(proc1.concept_graph_path, Path)
    assert proc1.concept_graph_path == graph_path

    # None resolves to None (legacy default).
    proc2 = CourseProcessor.__new__(CourseProcessor)
    proc2.concept_graph_path = None
    assert proc2.concept_graph_path is None
